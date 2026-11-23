"""Verify solver isolation, infrastructure retries, and trace redaction."""

from __future__ import annotations

import asyncio
import importlib
import json
from functools import partial
from types import SimpleNamespace

import httpx
import pytest

from flashrsi.agents.claude_code import _authentication_env
from flashrsi.agents.codex.python_core import client as codex_client
from flashrsi.agents.codex.python_core.models import Model
from flashrsi.agents.miniagent.runtime import agent as mini_loop
from flashrsi.agents.miniagent.runtime import llm as mini_llm
from flashrsi.agents.pi_agent import _thinking
from flashrsi.agents.pi_agent.runtime.pi_native import compaction
from flashrsi.agents.pi_agent.runtime.pi_native.anthropic_messages import (
    anthropic_response,
    build_anthropic_payload,
)
from flashrsi.agents.pi_agent.runtime.pi_native.openai_provider import build_payload
from flashrsi.benchmarks._harbor import _grade
from flashrsi.core import Agent, Algorithm, InfraError, Task, Trajectory, infra


class MockAgent(Agent):
    name = "sandbox_mock"

    async def _run(self, prompt, working_dir):
        return Trajectory(output=prompt)


async def _track_concurrent_rollout(concurrency, _agent, task):
    concurrency.active += 1
    concurrency.peak = max(concurrency.peak, concurrency.active)
    await asyncio.sleep(0.002 * (8 - int(task.task_id) % 8))
    concurrency.active -= 1
    return Trajectory(output=task.question)


def test_bounded_rollout_overlaps_and_keeps_input_order(monkeypatch):
    monkeypatch.setenv("FLASHRSI_MAX_CONCURRENCY", "8")
    agent = MockAgent()
    concurrency = SimpleNamespace(active=0, peak=0)
    tasks = [Task(str(i), f"q{i}", env="mock") for i in range(24)]

    solve = partial(_track_concurrent_rollout, concurrency)

    agent._env_runner = solve
    rows = asyncio.run(agent.rollout(tasks))
    assert [row.task for row in rows] == tasks
    assert [row.output for row in rows] == [task.question for task in tasks]
    assert concurrency.peak == 8


async def _record_delay(delays, delay):
    delays.append(delay)


async def _retrying_rollout(attempts, failure, _agent, task):
    attempts.append(task)
    if len(attempts) <= 2:
        raise failure
    return Trajectory(output="recovered")


async def _infra_failed_rollout(_agent, _task):
    return Trajectory(
        status="error",
        messages=({"type": "InfraError", "reason": "incomplete_stream"},),
        error_message="transport interrupted",
    )


@pytest.mark.parametrize(
    "failure",
    [
        httpx.RemoteProtocolError("synthetic disconnect"),
        InfraError("incomplete_stream"),
        RuntimeError("Cannot connect to the Docker daemon"),
        *[
            httpx.HTTPStatusError(
                "synthetic",
                request=httpx.Request("POST", "https://example.test"),
                response=httpx.Response(status),
            )
            for status in (429, 502, 503, 504)
        ],
    ],
)
def test_retry_and_exhaustion_do_not_score(failure, monkeypatch):
    monkeypatch.setenv("FLASHRSI_INFRA_RETRIES", "2")
    monkeypatch.setenv("FLASHRSI_MAX_CONCURRENCY", "1")
    delays = []

    sleep = partial(_record_delay, delays)

    monkeypatch.setattr(infra.asyncio, "sleep", sleep)
    monkeypatch.setattr(infra.random, "uniform", lambda *_args: 0.75)
    agent = MockAgent()
    task = Task("a", "q", env="mock")
    attempts = []

    solve = partial(_retrying_rollout, attempts, failure)

    agent._env_runner = solve
    assert asyncio.run(agent.rollout([task]))[0].output == "recovered"
    assert len(attempts) == 3 and delays == [0.75, 1.5]

    cannot_solve = _infra_failed_rollout

    agent._env_runner = cannot_solve
    with pytest.raises(InfraError, match="incomplete_stream"):
        asyncio.run(agent.rollout([task]))
    with pytest.raises(InfraError, match="http_503"):
        _grade(
            [Trajectory(task, status="error", error_message="HTTP 503")],
            {"a": task},
            "mock",
        )
    assert (
        _grade(
            [Trajectory(task, status="error", error_message="model gave up")],
            {"a": task},
            "mock",
        )[0].metrics["reward"]
        == 0
    )


async def _forbid_host_rollout(_agent, _task):
    pytest.fail("a missing environment may not be dispatched")


def test_environment_guard_and_mock_override(monkeypatch):
    agent = MockAgent()

    never = _forbid_host_rollout

    agent._env_runner = never
    task = Task("a", "text")
    monkeypatch.delenv("FLASHRSI_ALLOW_HOST_FALLBACK", raising=False)
    with pytest.raises(ValueError, match="task.env"):
        asyncio.run(agent.rollout([task]))
    monkeypatch.setenv("FLASHRSI_ALLOW_HOST_FALLBACK", "1")
    assert asyncio.run(agent.rollout([task]))[0].output == "text"


def test_harbor_denies_missing_task_environment():
    pytest.importorskip("flashrsi.integrations.harbor_suite")
    from flashrsi.integrations.harbor import _run

    with pytest.raises(ValueError, match="task.env"):
        asyncio.run(_run(MockAgent(), Task("a", "text")))


def test_worker_evidence_strips_fields_and_verifier_logs():
    task = Task(
        "a",
        "text",
        ground_truth="private reference",
        metadata={"nested": {"reference_answers": ["private alternate"], "public": 3}},
    )
    original = Trajectory(
        task=task,
        messages=(
            {"type": "RunConfig", "config": {"model": "private configuration"}},
            {
                "type": "HarborVerifier",
                "stdout": "private grader transcript",
                "rewards": {"reward": 1},
            },
            {
                "type": "Trace",
                "items": [{"reference_solution": "private program", "public": 4}],
            },
        ),
    )
    evidence = Algorithm._trajectory_evidence(SimpleNamespace(), original)
    assert "private" not in json.dumps(evidence)
    assert evidence["task"]["metadata"]["nested"] == {"public": 3}
    assert evidence["messages"][0]["rewards"] == {"reward": 1}
    assert original.task.ground_truth == "private reference"


def _sse(events, *, done=False):
    lines = [": keep-alive", ""]
    for event in events:
        lines.extend(("data: " + json.dumps(event), ""))
    if done:
        lines.extend(("data: [DONE]", ""))
    return "\n".join(lines) + "\n"


def _anthropic_fixture():
    return [
        {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "  keep héllo\n"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "exact"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "+signature"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "tool_1",
                "name": "demo",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"x":2}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 8},
        },
        {"type": "message_stop"},
    ]


_SIGNED_BLOCK = {
    "type": "thinking",
    "thinking": "  keep héllo\n",
    "signature": "exact+signature",
}
_TOOL_BLOCK = {"type": "tool_use", "id": "tool_1", "name": "demo", "input": {"x": 2}}


def _replay_provider_response(calls, api, request):
    calls.append(json.loads(request.content))
    if api == "anthropic":
        wire = _sse(_anthropic_fixture())
    else:
        wire = _sse(
            [
                {"choices": [{"delta": {"reasoning_content": "  keep héllo\n"}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": "second",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "tool_1",
                                        "function": {
                                            "name": "demo",
                                            "arguments": '{"x":2}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ],
            done=True,
        )
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=wire)


async def _run_embedded_replay(api, server, provider):
    config = {
        "llm": {
            "base_url": "https://example.test",
            "model_name": "deepseek-flash",
            "model_api": api,
            "api_key": "fake",
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        history = [{"role": "user", "content": "start"}]
        result = await provider.complete(config, history, [], client=client)
        history.append(
            {
                "role": "assistant",
                "content": result.text,
                "tool_calls": list(result.tool_calls),
                **result.history,
            }
        )
        history.append({"role": "tool", "content": "value", "tool_call_id": "tool_1"})
        await provider.complete(config, history, [], client=client)


@pytest.mark.parametrize("adapter", ["openclaw", "opencode"])
@pytest.mark.parametrize("api", ["chat_completions", "anthropic"])
def test_embedded_two_turn_http_replay(adapter, api):
    provider = importlib.import_module(
        f"flashrsi.agents.{adapter}.runtime.{adapter}_native.providers"
    )
    calls = []

    server = partial(_replay_provider_response, calls, api)

    run = partial(_run_embedded_replay, api, server, provider)

    asyncio.run(run())
    assert calls[0]["thinking"] == {"type": "enabled"}
    if api == "anthropic":
        assert calls[0]["output_config"] == {"effort": "high"}
        assert calls[1]["messages"][1]["content"] == [_SIGNED_BLOCK, _TOOL_BLOCK]
    else:
        assert calls[0]["reasoning_effort"] == "high"
        assert calls[1]["messages"][1]["reasoning_content"] == "  keep héllo\nsecond"


async def _read_heartbeat_only(api, provider):
    config = {
        "llm": {
            "base_url": "https://example.test",
            "model_name": "deepseek-flash",
            "model_api": api,
            "api_key": "fake",
        }
    }
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, text=": keep-alive\n\n")
    )
    async with httpx.AsyncClient(transport=transport) as client:
        await provider.complete(
            config, [{"role": "user", "content": "start"}], [], client=client
        )


@pytest.mark.parametrize("adapter", ["openclaw", "opencode"])
@pytest.mark.parametrize("api", ["chat_completions", "anthropic"])
def test_embedded_heartbeat_only_is_retryable(adapter, api):
    provider = importlib.import_module(
        f"flashrsi.agents.{adapter}.runtime.{adapter}_native.providers"
    )

    run = partial(_read_heartbeat_only, api, provider)

    with pytest.raises(InfraError, match="incomplete_stream"):
        asyncio.run(run())


def test_pi_replays_signed_thinking_and_claude_timeouts():
    llm = {
        "model_name": "deepseek-flash",
        "provider_name": "deepseek",
        "model_api": "anthropic-messages",
        "reasoning_effort": "high",
        "thinking": {"type": "enabled"},
        "api_key": "fake",
    }
    config = {"llm": llm}
    assert _thinking(config) == "high"
    reply = anthropic_response(
        config, {"content": [_SIGNED_BLOCK, _TOOL_BLOCK], "stop_reason": "tool_use"}
    )
    second, _, _ = build_anthropic_payload(
        config, [{"role": "user", "content": "start"}, reply], "high"
    )
    assert second["thinking"] == {"type": "enabled"}
    assert second["output_config"] == {"effort": "high"}
    assert second["messages"][1]["content"] == [_SIGNED_BLOCK, _TOOL_BLOCK]
    env = _authentication_env({"api_key": "fake"})
    assert env["API_TIMEOUT_MS"] == "900000"
    assert env["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] == "900000"


def _capture_minimal_response(records, request):
    records.append(json.loads(request.content))
    assert request.url.path == "/anthropic/v1/messages"
    return httpx.Response(200, text=_sse(_anthropic_fixture()))


def test_miniagent_replays_native_anthropic_and_chat_fields(monkeypatch):
    records = []
    real_client = httpx.AsyncClient

    server = partial(_capture_minimal_response, records)

    old = {
        "model_name": mini_llm.DEFAULT_MODEL,
        "side_model": mini_llm.SIDE_MODEL,
        "base_url": mini_llm.BASE_URL,
        "side_base_url": mini_llm.SIDE_BASE_URL,
        "max_tokens": mini_llm.MAX_TOKENS,
        "llm_max_retries": mini_llm.LLM_MAX_RETRIES,
        "api_key": mini_llm.API_KEY,
        "reasoning_effort": mini_llm.REASONING_EFFORT,
        "extra_body": mini_llm.EXTRA_BODY,
        "request_timeout": mini_llm.REQUEST_TIMEOUT,
        "model_api": mini_llm.MODEL_API,
        "thinking": mini_llm.THINKING,
        "stream": mini_llm.STREAM,
    }
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(server), **kw),
    )
    try:
        mini_llm.configure(
            {
                "model_name": "deepseek-flash",
                "api_key": "fake",
                "base_url": "https://example.test/anthropic",
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            }
        )
        assert mini_llm.REQUEST_TIMEOUT == 900
        history = [{"role": "user", "content": "start"}]
        result = asyncio.run(mini_llm.chat(history))
        history.append(mini_loop._serialize_assistant(result.choices[0].message))
        history.append({"role": "tool", "tool_call_id": "tool_1", "content": "ok"})
        asyncio.run(mini_llm.chat(history))
    finally:
        mini_llm.configure(old)
    assert records[1]["messages"][1]["content"] == [_SIGNED_BLOCK, _TOOL_BLOCK]
    exact = "  slightly odd reasoning\n\t"
    assert (
        mini_loop._serialize_assistant(
            SimpleNamespace(role="assistant", content="answer", reasoning_content=exact)
        )["reasoning_content"]
        == exact
    )


async def _unavailable_docker_trial(*_):
    raise RuntimeError("Cannot connect to the Docker daemon")


def test_harbor_marks_transient_trial_failure_before_grading(tmp_path, monkeypatch):
    pytest.importorskip("flashrsi.integrations.harbor_suite")
    from flashrsi.integrations import harbor

    task = Task("a", "test", env=str(tmp_path))
    native = SimpleNamespace(
        instruction=task.question,
        checksum="checkpoint",
        config=SimpleNamespace(
            steps=None,
            environment=SimpleNamespace(os=SimpleNamespace(value="linux")),
            agent=SimpleNamespace(timeout_sec=300),
        ),
    )
    agent = SimpleNamespace(
        config={"llm": {"model_name": "fixture"}},
        snapshot=SimpleNamespace(id="version"),
        _check_container=lambda: None,
    )
    monkeypatch.setattr(harbor.shutil, "which", lambda _: "/fixture/docker")
    monkeypatch.setattr(harbor, "HarborTask", lambda _: native)
    monkeypatch.setattr(harbor, "resolve_task_verifier_mode", lambda _: None)
    monkeypatch.setattr(harbor, "_docker_environment_import", lambda *_: None)
    monkeypatch.setattr(harbor, "TaskConfig", lambda **_: None)
    monkeypatch.setattr(harbor, "TrialConfig", lambda **_: None)
    monkeypatch.setattr(harbor, "AgentConfig", lambda **_: None)
    monkeypatch.setattr(harbor, "EnvironmentConfig", lambda **_: None)
    monkeypatch.setattr(harbor, "VerifierConfig", lambda **_: None)
    monkeypatch.setattr(harbor, "_stage", lambda *_: None)

    monkeypatch.setattr(harbor.Trial, "create", _unavailable_docker_trial)
    row = asyncio.run(harbor._run(agent, task))
    assert row.status == "error"
    assert row.messages[-1]["infra_error"] == "docker_daemon"


def _failed_codex_reply(requests, failure, request):
    requests.append(request)
    return httpx.Response(failure, text="")


async def _request_with_failed_codex_stream(settings, failure, reason):
    llm = {
        "model_name": "deepseek-flash",
        "api_key": "fixture-provider-credential",
        "base_url": "https://api.deepseek.com",
        "_deepseek_responses": True,
    }
    async with codex_client.ResponsesClient(
        llm, Model("deepseek-flash", {}), settings, "thread", lambda _: None
    ) as client:
        with pytest.raises(codex_client.ResponseError) as caught:
            await client.response([], [], retry_stream=(failure == 200))
        assert caught.value.infra_reason == reason


@pytest.mark.parametrize(
    "failure,reason", [(429, "http_429"), (503, "http_503"), (200, "incomplete_stream")]
)
def test_codex_retries_and_classifies_upstream_failures(monkeypatch, failure, reason):
    requests = []
    backoff = []

    reply = partial(_failed_codex_reply, requests, failure)

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(reply), **kwargs),
    )
    monkeypatch.setattr(
        codex_client, "_backoff", lambda attempt: backoff.append(attempt) or 0
    )
    settings = {
        "model_provider": "fixture",
        "model_providers": {
            "fixture": {
                "request_max_retries": 2,
                "stream_max_retries": 2,
            }
        },
    }

    exercise = partial(_request_with_failed_codex_stream, settings, failure, reason)

    asyncio.run(exercise())
    assert backoff == [1, 2]
    assert len(requests) == 3


@pytest.mark.parametrize("override", [None, 32768])
def test_pi_and_mini_use_deepseek_output_budget(override, tmp_path):
    values = {"llm": {"model_name": "deepseek-flash", "max_tokens": override}}
    payload, _ = build_payload(
        values, [{"role": "user", "content": "test"}], "openai-completions", "high"
    )
    assert payload["max_tokens"] == (override or 65536)
    assert compaction.model_limits(values) == (1_000_000, 65536)
    assert compaction.settings(values, tmp_path) == compaction.DEFAULT_SETTINGS
    try:
        mini_llm.configure(values["llm"])
        assert mini_llm.MAX_TOKENS == (override or 65536)
    finally:
        mini_llm.configure({})
