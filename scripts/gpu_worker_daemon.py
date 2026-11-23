#!/usr/bin/env python3
"""Serve GPU timing and open Frontier jobs over authenticated HTTP.

The daemon owns signing credentials, durable receipts and the physical GPU
lease. Workload code executes in the isolated worker phase.
FLASHRSI_GPU_AUTH_TOKEN, FLASHRSI_GPU_IMAGE, FLASHRSI_GPU_UUID and
FLASHRSI_CUDA_VERSION configure the daemon's defaults.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hmac
import json
import os
import re
import secrets
import shutil
import sqlite3
import sys
import tempfile
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socket import socket
from threading import BoundedSemaphore
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashrsi._records import _redact, _secrets  # noqa: E402
from flashrsi.benchmarks import _suite  # noqa: E402
from flashrsi.benchmarks.design_suite import FrontierEng, KernelBench  # noqa: E402
from flashrsi.integrations import design_suite, gpu_worker, kernelbench  # noqa: E402
from flashrsi.integrations.frontier_eng import execute_frontier  # noqa: E402
from flashrsi.integrations.gpu_worker import ProtocolError  # noqa: E402

MAX_BODY = 768 * 1024
MAX_SOURCE = 512 * 1024


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dry = args.dry_run_cpu
        self.backend = "dry_run_cpu" if self.dry else "cuda"
        if self.dry:
            self.device = args.device if args.device.startswith("GPU-") else "GPU-dry-run"
        else:
            self.device = gpu_worker.hardware_info(args.device)["device_uuid"]
        self.state = Path(args.state_dir).expanduser().resolve()
        self.state.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock = self.state / (self.device + ".lock")
        # All daemon instances use the same per-device lease regardless of their receipt store.
        if not self.dry:
            self.lock = Path("/tmp") / ("flashrsi-exclusive-" + self.device + ".lock")
        self.db = self.state / "receipts.sqlite"
        with sqlite3.connect(self.db) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, digest TEXT NOT NULL, receipt TEXT
            )""")
        self.kernel = KernelBench({"name": "kernelbench", "root": args.asset_root})
        self.kernel.manifest()
        self.frontier = None
        if args.frontier_manifest:
            self.frontier = FrontierEng({
                "name": "frontier_eng", "mode": "real_tasks", "root": args.asset_root,
                "frontier_manifest": args.frontier_manifest,
                "frontier_images": {"gpu": args.cuda_image} if args.cuda_image else {},
            })
            self.frontier.load()

    def run_phase(self, phase: str, folder: Path, cfg: dict[str, Any], deadline: float) -> tuple[dict[str, Any] | None, str, str | None]:
        script = Path(gpu_worker.__file__).resolve()
        config = json.dumps(cfg, allow_nan=False)
        env = {**gpu_worker.SAFE_ENV, "HOME": str(folder), "TMPDIR": str(folder)}
        if self.dry:
            command = [sys.executable, "-I", "-B", str(script), phase, str(folder / "reference.py"), str(folder / "candidate.py"), config]
            return gpu_worker.run_phase(command, env, deadline, phase, kernelbench.COMPILE_STDERR_BYTES)
        name = "flashrsi-gpu-" + uuid.uuid4().hex[:14]
        cache = folder / "cache"
        (cache / "tmp").mkdir(parents=True, exist_ok=True)
        command = design_suite.docker_args(
            self.args.cuda_image, name,
            ((script, "/worker.py"), (folder / "reference.py", "/reference.py"), (folder / "candidate.py", "/candidate.py")),
            resources={**design_suite.ENG_RESOURCES, "memory": "16g"}, writable=((cache, "/worker-cache"),),
            gpu_device=self.device,
            env=("TRITON_CACHE_DIR=/worker-cache/triton", "CUDA_CACHE_PATH=/worker-cache/cuda",
                 "TORCH_EXTENSIONS_DIR=/worker-cache/extensions", "TMPDIR=/worker-cache/tmp"),
        )
        try:
            command.extend(("/worker.py", phase, "/reference.py", "/candidate.py", config))
            return gpu_worker.run_phase(command, env, deadline, phase, kernelbench.COMPILE_STDERR_BYTES)
        finally:
            design_suite.cleanup(name)

    def timing(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("device_uuid") != self.device or payload.get("lease_mode") != "exclusive":
            raise ProtocolError(400, "GPU job targets a different lease")
        dtype = payload.get("dtype")
        if (
            not isinstance(dtype, str)
            or dtype not in kernelbench.GPU_TOLERANCES
            or payload.get("timing_protocol") != "locked-clock-flush-l2"
            or not isinstance(payload.get("snapshot_id"), str)
        ):
            raise ProtocolError(400, "invalid GPU timing protocol")
        try:
            reference = self.kernel.reference(payload["task_id"])
        except PermissionError:
            raise ProtocolError(403, "kernel task is outside the worker's open manifest") from None
        source = payload["source"]
        try:
            errors = kernelbench.kernel_guard(source)
            tree = ast.parse(source)
        except SyntaxError:
            raise ProtocolError(422, "candidate failed static AST checks") from None
        if errors:
            raise ProtocolError(422, "candidate failed static AST checks")
        if not self.dry and not kernelbench.kernel_has_triton_launch(tree):
            raise ProtocolError(422, "candidate must launch a custom Triton kernel")
        config = {
            "backend": self.backend,
            "dtype": dtype,
            "seeds": secrets.SystemRandom().sample(range(2**32), kernelbench.GPU_CORRECT_TRIALS),
            "tolerance": kernelbench.GPU_TOLERANCES[dtype],
            "l2_flush_bytes": kernelbench.GPU_L2_BYTES,
            "warmup": kernelbench.GPU_WARMUP,
            "perf_trials": kernelbench.GPU_PERF_TRIALS,
        }
        hardware = {"device_uuid": self.device, "clocks_locked": False}
        if not self.dry:
            before = gpu_worker.hardware_info(self.device)
            hardware["initial_graphics_clock_mhz"] = before["graphics_clock_mhz"]
            hardware["initial_memory_clock_mhz"] = before["memory_clock_mhz"]
        locks = self.args.lock_clocks
        if locks:
            mhz = locks if locks != "max" else self.max_graphics_clock()
            gpu_worker.smi(self.device, "-lgc", f"{mhz},{mhz}")
            hardware.update(clocks_locked=True, configured_graphics_clock_mhz=mhz)
        try:
            with tempfile.TemporaryDirectory(prefix="flashrsi-gpu-job-") as tmp:
                folder = Path(tmp)
                shutil.copyfile(reference, folder / "reference.py")
                (folder / "candidate.py").write_text(source, encoding="utf-8")
                compiled, compile_stderr, failure = self.run_phase("compile", folder, config, kernelbench.GPU_COMPILE_TIMEOUT)
                result, runtime_stderr = None, ""
                if compiled:
                    if compiled != {"compiled": True}:
                        raise ProtocolError(503, "invalid GPU compile status")
                    result, runtime_stderr, failure = self.run_phase("run", folder, config, kernelbench.GPU_RUN_TIMEOUT)
                if not self.dry:
                    hardware.update(gpu_worker.hardware_info(self.device))
                    hardware["cuda_version"] = result.get("cuda_version") if result else self.args.cuda_version
                receipt = kernelbench.timing_receipt(payload, config, hardware, compiled, result, failure, compile_stderr, runtime_stderr)
                return kernelbench.validate_kernel_receipt(receipt, payload, require_real_cuda=not self.dry)
        finally:
            if locks:
                gpu_worker.smi(self.device, "-rgc")

    def max_graphics_clock(self) -> str:
        values = gpu_worker.smi(self.device, "--query-supported-clocks=graphics", "--format=csv,noheader,nounits")
        choices = [int(line) for line in values.splitlines() if line.strip().isdigit()]
        if not choices:
            raise ProtocolError(503, "GPU did not report supported graphics clocks")
        return str(max(choices))

    def frontier_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.dry or self.frontier is None:
            raise ProtocolError(503, "this worker does not serve open Frontier CUDA tasks")
        task = self.frontier._tasks.get(payload["task_id"])
        if task is None or task.metadata["queue"] != "gpu":
            raise ProtocolError(403, "task is outside the worker's open Frontier GPU manifest")
        if payload.get("timeout_s") != task.metadata["timeout_s"]:
            raise ProtocolError(400, "Frontier job timeout differs from the worker manifest")
        result = execute_frontier(self.frontier, task, payload["source"], gpu_device=self.device)
        return {
            **{key: payload[key] for key in ("job_id", "task_id", "candidate_sha256", "timeout_s")},
            "backend": self.backend, "device_uuid": self.device, "result": result,
        }

    def submit(self, path: str, payload: Any, request_id: str | None) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("source"), str):
            raise ProtocolError(400, "expected a JSON job with source")
        source = payload["source"]
        job_id, task_id = payload.get("job_id"), payload.get("task_id")
        if (
            not isinstance(job_id, str)
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", job_id)
            or job_id != request_id
            or not isinstance(task_id, str)
            or not 1 <= len(task_id) <= 512
            or not 1 <= len(source.encode()) <= MAX_SOURCE
            or payload.get("candidate_sha256") != design_suite.sha(source.encode())
        ):
            raise ProtocolError(400, "invalid GPU job binding")
        digest = design_suite.sha(json.dumps([path, self.backend, self.device, payload], sort_keys=True).encode())
        timing = path == "/v1/timing-jobs"
        with gpu_worker.gpu_lease(self.lock, 240 if timing else 15):
            with sqlite3.connect(self.db) as db:
                row = db.execute("SELECT digest, receipt FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                if row:
                    if row[0] != digest:
                        raise ProtocolError(409, "job ID already binds a different request")
                    if row[1] is None:
                        raise ProtocolError(503, "worker stopped before committing this job receipt")
                    return json.loads(row[1])
                db.execute("INSERT INTO jobs VALUES (?, ?, NULL)", (job_id, digest))
            try:
                result = self.timing(payload) if timing else self.frontier_job(payload)
                result = _redact(result, _secrets(os.environ))
            except ProtocolError as exc:
                if exc.status < 500:
                    with sqlite3.connect(self.db) as db:
                        db.execute("DELETE FROM jobs WHERE job_id=? AND receipt IS NULL", (job_id,))
                raise
            with sqlite3.connect(self.db) as db:
                db.execute("UPDATE jobs SET receipt=? WHERE job_id=?", (json.dumps(result, allow_nan=False), job_id))
            return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *args: Any) -> None:
        pass

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(20)

    def send_json(self, status: int, body: dict[str, Any]) -> None:
        value = json.dumps(body, ensure_ascii=True, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(value)))
        self.end_headers()
        self.wfile.write(value)

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self.send_json(404, {"error": "unknown route"})
        else:
            self.send_json(200, {"backend": self.server.worker.backend})

    def do_POST(self) -> None:
        token = self.server.worker.args.auth_token
        try:
            if self.path not in ("/v1/timing-jobs", "/v1/frontier-jobs"):
                raise ProtocolError(404, "unknown route")
            if token and not hmac.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + token).encode()):
                raise ProtocolError(401, "worker authentication required")
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ProtocolError(400, "invalid Content-Length") from None
            if not 0 < size <= MAX_BODY:
                raise ProtocolError(413, "invalid job body size")
            try:
                payload = json.loads(self.rfile.read(size))
            except (ValueError, TimeoutError):
                raise ProtocolError(400, "invalid JSON job body") from None
            receipt = self.server.worker.submit(self.path, payload, self.headers.get("Idempotency-Key"))
            if token:
                receipt["signature"] = design_suite.receipt_signature(receipt, token)
            self.send_json(200, receipt)
        except ProtocolError as exc:
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.send_json(exc.status, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            # Never log submitted source, environment variables, or exception arguments.
            print(f"GPU worker request failed ({type(exc).__name__})", file=sys.stderr)
            self.send_json(503, {"error": "GPU worker infrastructure error"})


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, address: tuple[str, int], worker: Worker) -> None:
        self.worker = worker
        self.inflight = BoundedSemaphore(8)
        super().__init__(address, Handler)

    def process_request(self, request: socket, address: tuple[str, int]) -> None:
        if not self.inflight.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.inflight.release()
            raise

    def process_request_thread(self, request: socket, address: tuple[str, int]) -> None:
        try:
            super().process_request_thread(request, address)
        finally:
            self.inflight.release()

    def handle_error(self, _request: socket, _address: tuple[str, int]) -> None:
        error = sys.exc_info()[1]
        print(f"GPU worker connection failed ({type(error).__name__})", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8047)
    parser.add_argument("--device", default=os.getenv("FLASHRSI_GPU_UUID", "0"), help="NVIDIA index or physical UUID")
    parser.add_argument("--lock-clocks", nargs="?", const="max", metavar="MHZ")
    parser.add_argument("--auth-token", default=os.getenv("FLASHRSI_GPU_AUTH_TOKEN"))
    parser.add_argument("--dry-run-cpu", action="store_true")
    parser.add_argument("--asset-root", default=_suite.resolve_root({}, "eng_and_mol"))
    parser.add_argument("--frontier-manifest", help="Reviewed open Frontier manifest to serve")
    parser.add_argument("--cuda-image", default=os.getenv("FLASHRSI_GPU_IMAGE"))
    parser.add_argument("--cuda-version", default=os.getenv("FLASHRSI_CUDA_VERSION"),
                        help="CUDA runtime recorded when a candidate fails before timing")
    parser.add_argument("--state-dir", default=f"/tmp/flashrsi-gpu-worker-{os.getuid()}")
    args = parser.parse_args(argv)
    if args.dry_run_cpu and (args.host != "127.0.0.1" or args.lock_clocks):
        parser.error("dry-run mode is loopback-only and cannot lock clocks")
    if not args.dry_run_cpu and not all((args.auth_token, args.cuda_image, args.cuda_version)):
        parser.error("CUDA mode requires an auth token, offline CUDA image, and CUDA runtime version")
    if args.lock_clocks not in (None, "max") and not re.fullmatch(r"[1-9]\d*", args.lock_clocks):
        parser.error("clock target must be a positive MHz value")
    if not re.fullmatch(r"(?:\d+|GPU-[A-Za-z0-9-]+)", args.device):
        parser.error("device must be an NVIDIA index or physical UUID")
    try:
        worker = Worker(args)
        with WorkerServer((args.host, args.port), worker) as server:
            print(f"GPU worker listening on {args.host}:{server.server_port} backend={worker.backend}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except (ProtocolError, OSError, ValueError) as exc:
        parser.exit(1, f"GPU worker startup failed ({type(exc).__name__})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
