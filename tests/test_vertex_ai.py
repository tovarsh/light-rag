import types

import pytest

np = pytest.importorskip("numpy")

from lightrag.llm import vertex_ai
from lightrag.llm.binding_options import (
    VertexAIEmbeddingOptions,
    VertexAILLMOptions,
)


def test_vertex_binding_options_parsing():
    args = types.SimpleNamespace(
        vertex_ai_llm_project_id="demo-project",
        vertex_ai_llm_location="europe-west1",
        vertex_ai_llm_temperature=0.3,
        vertex_ai_llm_stop_sequences=["END"],
        vertex_ai_llm_safety_settings={"HARASSMENT": "BLOCK_MEDIUM"},
        vertex_ai_embedding_project_id="demo-project",
        vertex_ai_embedding_location="europe-west1",
        vertex_ai_embedding_output_dimensionality=256,
        vertex_ai_embedding_task_type="RETRIEVAL_QUERY",
    )

    llm_options = VertexAILLMOptions.options_dict(args)
    embedding_options = VertexAIEmbeddingOptions.options_dict(args)

    assert llm_options["project_id"] == "demo-project"
    assert llm_options["location"] == "europe-west1"
    assert llm_options["temperature"] == 0.3
    assert llm_options["stop_sequences"] == ["END"]
    assert llm_options["safety_settings"] == {"HARASSMENT": "BLOCK_MEDIUM"}

    assert embedding_options["project_id"] == "demo-project"
    assert embedding_options["output_dimensionality"] == 256
    assert embedding_options["task_type"] == "RETRIEVAL_QUERY"


@pytest.mark.asyncio
async def test_vertex_llm_builds_generation_config(monkeypatch):
    capture: dict = {}

    class DummyResponse:
        def __init__(self):
            content = types.SimpleNamespace(parts=[types.SimpleNamespace(text="hi")])
            self.candidates = [types.SimpleNamespace(content=content)]
            self.usage_metadata = types.SimpleNamespace(
                prompt_token_count=1, candidates_token_count=2, total_token_count=3
            )

    class DummyChat:
        def __init__(self, capture_dict):
            self.capture = capture_dict

        def send_message(self, prompt, stream=False, **kwargs):
            self.capture["prompt"] = prompt
            self.capture["stream"] = stream
            self.capture["kwargs"] = kwargs
            return DummyResponse()

    class DummyModel:
        def __init__(self, name, **kwargs):
            capture["model"] = name
            capture["model_kwargs"] = kwargs

        def start_chat(self, history=None):
            capture["history"] = history
            return DummyChat(capture)

    monkeypatch.setattr(vertex_ai, "GenerativeModel", DummyModel)
    monkeypatch.setattr(vertex_ai, "_init_vertex_ai", lambda *_, **__: True)

    result = await vertex_ai.vertex_ai_complete_if_cache(
        model="gemini-1.5-flash-001",
        prompt="Hello",
        system_prompt="You are a test bot",
        history_messages=[{"role": "user", "content": "prior"}],
        top_p=0.9,
        max_output_tokens=128,
        stop_sequences=["STOP"],
    )

    assert result == "hi"
    assert capture["model"] == "gemini-1.5-flash-001"
    assert capture["prompt"] == "Hello"
    generation_config = capture["kwargs"]["generation_config"]
    assert generation_config.max_output_tokens == 128
    assert generation_config.top_p == 0.9
    assert generation_config.stop_sequences == ["STOP"]


@pytest.mark.asyncio
async def test_vertex_embedding_formats_request(monkeypatch):
    class DummyEmbeddingModel:
        def __init__(self):
            self.captured = None

        def get_embeddings(self, texts, **kwargs):
            self.captured = {"texts": texts, **kwargs}
            return [types.SimpleNamespace(values=[1.0, 2.0], token_count=3)]

    dummy_model = DummyEmbeddingModel()

    monkeypatch.setattr(vertex_ai, "_init_vertex_ai", lambda *_, **__: True)
    monkeypatch.setattr(
        vertex_ai.TextEmbeddingModel,
        "from_pretrained",
        classmethod(lambda cls, name: dummy_model),
    )

    embeddings = await vertex_ai.vertex_ai_embed(
        ["hello", "world"],
        model="text-embedding-004",
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=128,
        title="Doc",
        timeout=30,
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 2)
    assert dummy_model.captured["task_type"] == "RETRIEVAL_QUERY"
    assert dummy_model.captured["output_dimensionality"] == 128
    assert dummy_model.captured["title"] == "Doc"
