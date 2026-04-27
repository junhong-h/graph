from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from graphmemory.llm_client import OpenAIClient


def _response(content: str = "{}"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


def test_complete_json_mode_adds_json_instruction_and_sampling_params():
    client = OpenAIClient(
        model="qwen3-4b",
        api_key="test",
        base_url="http://example.test/v1",
        temperature=0.0,
        top_p=0.7,
        seed=42,
    )
    create = MagicMock(return_value=_response('{"ok": true}'))
    client.client.chat.completions.create = create

    result = client.complete(
        [{"role": "user", "content": "Return an object."}],
        json_mode=True,
    )

    assert result == '{"ok": true}'
    request = create.call_args.kwargs
    assert request["response_format"] == {"type": "json_object"}
    assert request["top_p"] == 0.7
    assert request["seed"] == 42
    assert any("json" in m["content"].lower() for m in request["messages"])


def test_complete_uses_dashscope_extra_body_to_disable_thinking():
    client = OpenAIClient(
        model="qwen3-4b",
        api_key="test",
        base_url="http://example.test/v1",
        disable_thinking=True,
        use_extra_body_thinking=True,
    )
    create = MagicMock(return_value=_response("ok"))
    client.client.chat.completions.create = create

    assert client.complete([{"role": "user", "content": "hi"}]) == "ok"
    assert create.call_args.kwargs["extra_body"] == {"enable_thinking": False}


def test_complete_retries_without_seed_when_provider_rejects_it():
    client = OpenAIClient(
        model="qwen3-4b",
        api_key="test",
        base_url="http://example.test/v1",
        seed=42,
        max_retries=1,
    )
    create = MagicMock(side_effect=[Exception("unsupported parameter: seed"), _response("ok")])
    client.client.chat.completions.create = create

    assert client.complete([{"role": "user", "content": "hi"}]) == "ok"
    assert create.call_args_list[0].kwargs["seed"] == 42
    assert "seed" not in create.call_args_list[1].kwargs
