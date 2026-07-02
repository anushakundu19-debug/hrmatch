import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("app", Path(__file__).resolve().parents[1] / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def test_parse_json_response_handles_code_fences_and_non_json_text():
    raw = '```json\n{"match_score": 82}\n```'
    assert app.parse_json_response(raw) == {"match_score": 82}


def test_parse_json_response_falls_back_when_model_returns_plain_text():
    raw = 'Match score: 74'
    assert app.parse_json_response(raw) == {"match_score": 74}


def test_get_client_falls_back_to_environment_when_streamlit_secrets_are_missing(monkeypatch):
    class BrokenSecrets:
        def get(self, *_args, **_kwargs):
            raise RuntimeError("No secrets found")

    monkeypatch.setattr(app.st, "secrets", BrokenSecrets())
    monkeypatch.setattr(app.st, "session_state", {})
    monkeypatch.setenv("GROQ_API_KEY", "env-key")

    client = app.get_client()

    assert client.api_key == "env-key"


def test_normalize_match_score_converts_decimal_to_percentage():
    assert app.normalize_match_score(0.4) == 40
    assert app.normalize_match_score(0.2) == 20
    assert app.normalize_match_score(0.0) == 0
    assert app.normalize_match_score(1.0) == 100


def test_normalize_match_score_keeps_0_to_100_values():
    assert app.normalize_match_score(40) == 40
    assert app.normalize_match_score(20) == 20
    assert app.normalize_match_score(75) == 75


def test_normalize_match_score_handles_string_values():
    assert app.normalize_match_score("0.5") == 50
    assert app.normalize_match_score("75") == 75
