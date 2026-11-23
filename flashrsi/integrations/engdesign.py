"""Run EngDesign solvers and grade responses with the pinned simulator.

Public input assets enter the solver workspace. Evaluators, schemas and
private dependencies stay in the offline grader container. The host resolves
the shared FLASHRSI_ENG_IMAGE override for both containers.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
import subprocess
import sysconfig
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flashrsi.agents.program_search import ProgramSearchTarget
from flashrsi.benchmarks.design_suite import EngDesignOpen, EngDesignSettings
from flashrsi.core import Agent, InfraError, Task, Trajectory
from flashrsi.integrations.design_suite import (
    ENG_RESOURCES,
    Resources,
    Verdict,
    artifact_proposal,
    docker_json,
    engineering_image,
    scratch,
    sha,
)
from flashrsi.integrations.workspace_suite import episode_config

ENG_GRADER = """
import ast, contextlib, importlib, io, json, math, sys, types
from pathlib import Path
sys.path.insert(0, '/grader')
sys.path.append('/grader-dependencies')

def nested(value):
    if isinstance(value, dict):
        return types.SimpleNamespace(**{key: nested(item) for key, item in value.items()})
    if isinstance(value, list):
        return [nested(item) for item in value]
    return value

value=json.load(sys.stdin)
with contextlib.redirect_stdout(io.StringIO()):
    schema=Path('/grader/output_structure.py')
    structure=None
    if sys.argv[1]!='XG_13' and schema.is_file():
        tree=ast.parse(schema.read_text(),filename=str(schema))
        names={node.id for node in ast.walk(tree) if isinstance(node,ast.Name)}
        for node in tree.body:
            if isinstance(node,ast.Import):
                node.names=[alias for alias in node.names
                            if alias.name not in {'instructor','openai'}
                            or (alias.asname or alias.name) in names]
        tree.body=[node for node in tree.body if not isinstance(node,ast.Import) or node.names]
        structure=types.ModuleType('output_structure')
        structure.__file__=str(schema)
        sys.modules[structure.__name__]=structure
        exec(compile(tree,str(schema),'exec'),structure.__dict__)
    evaluate=importlib.import_module('evaluate')
    if sys.argv[1]=='XG_13':
        response=types.SimpleNamespace(config=types.SimpleNamespace(**value))
    else:
        candidates=[getattr(structure,n,None) for n in ('Response_structure','LLMResponse','Response','Output','OutputStructure')]
        cls=next((c for c in candidates if isinstance(c,type) and hasattr(c,'model_validate')),None)
        try:
            response=cls.model_validate(value) if cls else nested(value)
        except ValueError:
            print(json.dumps({'score':0.0,'raw_score':0.0}),file=sys.__stdout__)
            raise SystemExit(0)
    passed, details, score, confidence=evaluate.evaluate_llm_response(response)
if (score is None and sys.argv[1]=='XG_13' and
    details.get('error')=='index 0 is out of bounds for axis 0 with size 0'):
    print(json.dumps({'score':0.0,'raw_score':0.0,'passed':False,'candidate_error':'undefined step response'}))
    raise SystemExit(0)
if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=100:
    raise ValueError('grader did not return a finite 0-100 score')
print(json.dumps({'score':float(score)/100,'raw_score':float(score),'passed':bool(passed)}))
"""


async def _solve_engdesign(
    agent: Agent, task: Task, public: Path, resources: Resources, settings: EngDesignSettings, image: str
) -> tuple[str, Trajectory, str]:
    kwargs = {"image": image, "filename": "candidate.json", "resources": resources}
    deadline = asyncio.timeout(settings.solver_timeout)
    try:
        async with deadline:
            if agent is None:
                return await artifact_proposal(agent, task, public, **kwargs)
            with episode_config(agent, deadline=settings.solver_timeout, max_turns=settings.max_turns):
                return await artifact_proposal(agent, task, public, **kwargs)
    except TimeoutError:
        if not deadline.expired():
            raise
        raise InfraError("EngDesign solver transport deadline exceeded") from None


def _stage_grader(benchmark: EngDesignOpen, task: Task, private: Path) -> None:
    # The pinned archive excludes mutable caches and logs from other evaluations.
    private.mkdir()
    revision = benchmark.manifest_data["commit"]
    archived = subprocess.check_output(["git", "-C", benchmark.root / "EngDesign", "archive", revision, "EngDesign-Open/" + task.task_id])
    subprocess.run(["tar", "-xf", "-", "-C", private, "--strip-components=2"], input=archived, check=True)
    for filename, expected in benchmark.manifest_data["files"][task.task_id].items():
        if sha((private / filename).read_bytes()) != expected:
            raise InfraError("pinned EngDesign grader payload changed")


def _stage_public(official: Path, public: Path) -> None:
    public.mkdir()
    for path in official.iterdir():
        if path.is_file() and not path.is_symlink() and re.fullmatch(r"input\.(npy|csv|json)", path.name):
            shutil.copy2(path, public / path.name)


def _validate_pid(candidate: Any) -> None:
    if (
        not isinstance(candidate, dict)
        or set(candidate) != {"Kp", "Ki", "Kd"}
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in candidate.values())
    ):
        raise ValueError("invalid PID proposal")


async def engdesign(agent: Agent, benchmark: EngDesignOpen, task: Task) -> Verdict:
    configured = benchmark.config.get("solver_resources", {})
    if not isinstance(configured, Mapping) or set(configured) - ENG_RESOURCES.keys():
        raise ValueError("EngDesign solver_resources must contain supported resource names")
    limits = {key: benchmark.config.get(key, value) for key, value in ENG_RESOURCES.items()}
    limits.update(configured)
    settings = getattr(benchmark, "settings", None) or EngDesignSettings.from_config(benchmark.config)
    grader_timeout = settings.grader_timeout
    with scratch() as tmp:
        folder = Path(tmp)
        official = benchmark.directory(task.task_id)
        private, public = folder / "private", folder / "public"
        _stage_grader(benchmark, task, private)
        _stage_public(official, public)
        image = engineering_image()
        if isinstance(agent, ProgramSearchTarget):
            candidate, usage = await agent.propose(task.question)
            trace = Trajectory(usage=usage)
            name = None
        else:
            raw, trace, name = await _solve_engdesign(agent, task, public, limits, settings, image)
            candidate = json.loads(raw)
        if task.task_id == "XG_13":
            _validate_pid(candidate)
        packages = Path(sysconfig.get_path("purelib"))
        try:
            verdict, grader = await asyncio.to_thread(
                docker_json,
                image,
                ["-c", ENG_GRADER, task.task_id],
                ((private, "/grader"), (packages, "/grader-dependencies")),
                grader_timeout,
                stdin=json.dumps(candidate, allow_nan=False),
                resources=ENG_RESOURCES,
            )
        except subprocess.TimeoutExpired:
            return trace, {
                "score": 0.0,
                "candidate_valid": False,
                "solver_container": name,
                "candidate_error": f"EngDesign grader exceeded {grader_timeout}s",
            }
        return trace, {**verdict, "verifier_container": grader, "solver_container": name}
