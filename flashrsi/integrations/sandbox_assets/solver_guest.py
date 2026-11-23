"""Execute the staged harness inside the solver container.

The guest reads its redacted request and snapshot from the mounted bundle.
It connects to the host only through the mounted Unix relay socket.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import socketserver
import sys
from contextlib import redirect_stdout
from pathlib import Path
from threading import Thread
from typing import Any

from flashrsi._records import _trajectory_record
from flashrsi.core import Snapshot, Trajectory
from flashrsi.core.infra import infra_reason
from flashrsi.integrations._container import _agent_class
from flashrsi.integrations._container_payload import _plain
from flashrsi.integrations.sandbox_runtime import CLAUDE_GUEST_CLI


def _copy_stream(source: socket.socket, destination: socket.socket) -> None:
    try:
        while block := source.recv(65536):
            destination.sendall(block)
    except OSError:
        pass
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class _Bridge(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
            upstream.connect("/native/relay.sock")
            outbound = Thread(
                target=_copy_stream, args=(self.request, upstream), daemon=True
            )
            outbound.start()
            _copy_stream(upstream, self.request)
            outbound.join(timeout=1)


async def run() -> Trajectory:
    request = json.loads(Path("/native/request.json").read_text())
    cls = _agent_class(request["agent"])
    request["config"].pop("source_dir", None)
    agent = cls(request["config"])
    if agent.name == "claude_code":
        agent.config["cli_path"] = CLAUDE_GUEST_CLI
    agent._snapshot = Snapshot.load(Path("/native/snapshot.json"))
    os.environ["DISABLE_TELEMETRY"] = "1"
    return await agent.run(request["prompt"], Path.cwd())


_STREAM_COPIES = {"message_start", "message_update", "tool_execution_update"}


def transport_record(trajectory: Trajectory) -> dict[str, Any]:
    """Keep completed tool/turn events and usage, omitting streaming copies."""
    value = _plain(_trajectory_record(trajectory))
    messages = []
    for event in value["messages"]:
        if event.get("type") in _STREAM_COPIES:
            continue
        event = dict(event)
        if event.get("type") == "agent_end":
            event.pop("messages", None)
        encoded = json.dumps(event, ensure_ascii=True)
        if len(encoded) > 64 * 1024:
            event = {
                "type": event.get("type"),
                "trace_truncated": True,
                **{
                    key: event[key]
                    for key in ("toolCallId", "toolName", "usage")
                    if key in event
                },
            }
        messages.append(event)
    value["messages"] = messages
    value["output"] = value["output"][:65536]
    return value


def main() -> None:
    with socketserver.ThreadingTCPServer(("127.0.0.1", 28451), _Bridge) as server:
        server.daemon_threads = True
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with redirect_stdout(sys.stderr):
                result = asyncio.run(run())
        except Exception as exc:
            reason = infra_reason(exc)
            events = ({"type": "InfraError", "reason": reason},) if reason else ()
            result = Trajectory(
                status="error",
                messages=events,
                error_message=reason or type(exc).__name__,
            )
        finally:
            server.shutdown()
            thread.join()
    print(json.dumps(transport_record(result), ensure_ascii=True))


if __name__ == "__main__":
    main()
