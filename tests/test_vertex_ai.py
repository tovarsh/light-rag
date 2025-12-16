import sys
import importlib.util
import pathlib
import sys
import types

import pytest


if "httpx" not in sys.modules:
    class _DummyResponse:
        def __init__(self):
            self.request = None
            self.status_code = 200
            self.headers = {}

    class _DummyRequest:
        pass

    sys.modules["httpx"] = types.SimpleNamespace(
        Response=_DummyResponse, Request=_DummyRequest
    )

if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *_, **__: None)

if "pydantic" not in sys.modules:
    class _DummyBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    sys.modules["pydantic"] = types.SimpleNamespace(BaseModel=_DummyBaseModel)

if "numpy" not in sys.modules:
    class _DummyNDArray(list):
        @property
        def shape(self):
            if not self:
                return (0, 0)
            first = self[0]
            if isinstance(first, (list, tuple)):
                return (len(self), len(first))
            return (len(self),)

    def _dummy_array(data, dtype=None):
        return _DummyNDArray(data)

    sys.modules["numpy"] = types.SimpleNamespace(
        ndarray=_DummyNDArray,
        array=_dummy_array,
        asarray=_dummy_array,
        float32=float,
    )


fake_lightrag = types.ModuleType("lightrag")
fake_utils = types.ModuleType("lightrag.utils")


def _dummy_get_env_value(name, default=None, value_type=None):
    return default


fake_utils.get_env_value = _dummy_get_env_value
fake_constants = types.ModuleType("lightrag.constants")
fake_constants.DEFAULT_TEMPERATURE = 0.7
sys.modules.setdefault("lightrag", fake_lightrag)
sys.modules.setdefault("lightrag.utils", fake_utils)
sys.modules.setdefault("lightrag.constants", fake_constants)
sys.modules.setdefault("lightrag.llm", types.ModuleType("lightrag.llm"))


binding_options_path = (
    pathlib.Path(__file__).resolve().parents[1] / "lightrag" / "llm" / "binding_options.py"
)
_spec = importlib.util.spec_from_file_location(
    "vertex_binding_options", binding_options_path
)
_module = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_module)

VertexAIEmbeddingOptions = _module.VertexAIEmbeddingOptions
VertexAILLMOptions = _module.VertexAILLMOptions


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


def test_vertex_ai_args_do_not_conflict(monkeypatch, tmp_path):
    pytest.skip("Requires full lightrag package to load config parser")
    working_dir = tmp_path / "work"
    input_dir = tmp_path / "inputs"
    working_dir.mkdir()
    input_dir.mkdir()

    argv = [
        "lightrag-server",
        "--llm-binding",
        "vertex_ai",
        "--embedding-binding",
        "vertex_ai",
        "--working-dir",
        str(working_dir),
        "--input-dir",
        str(input_dir),
    ]
    monkeypatch.setenv("LLM_BINDING", "")
    monkeypatch.setenv("EMBEDDING_BINDING", "")
    monkeypatch.setattr("sys.argv", argv)

    # Should not raise argparse conflicts when registering Vertex AI options twice
    args = __import__("lightrag.api.config", fromlist=["parse_args"]).parse_args()

    assert args.llm_binding == "vertex_ai"
    assert args.embedding_binding == "vertex_ai"


@pytest.mark.skip(reason="requires pytest-asyncio and Vertex AI SDK")
async def test_vertex_llm_builds_generation_config(monkeypatch):
    pytest.importorskip("pytest_asyncio")
    pytest.importorskip("vertexai")
    pytest.importorskip("google")
    from lightrag.llm import vertex_ai

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


@pytest.mark.skip(reason="requires pytest-asyncio and Vertex AI SDK")
async def test_vertex_embedding_formats_request(monkeypatch):
    pytest.importorskip("pytest_asyncio")
    pytest.importorskip("vertexai")
    pytest.importorskip("google")
    from lightrag.llm import vertex_ai

    np = pytest.importorskip("numpy")
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
