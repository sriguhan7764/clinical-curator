"""Integration tests for the FastAPI backend using TestClient.

These tests run against the full ASGI app with fallback (demo) models,
so no checkpoint files or GPU are required.

Run with:
    cd clinical-curator
    pytest tests/test_api.py -v
"""

import io
import json
import sys
import os

import pytest

# Allow importing from backend/ without installing it as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a TestClient that runs the full lifespan (startup + shutdown)."""
    from main import app
    with TestClient(app) as c:
        yield c


def _dummy_png() -> bytes:
    """Return a minimal valid 1×1 white PNG image as bytes."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_response_has_required_keys(self, client):
        data = client.get("/api/health").json()
        assert "status" in data
        assert "models_loaded" in data

    def test_status_is_ok(self, client):
        assert client.get("/api/health").json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------

class TestModelsEndpoint:
    def test_returns_200(self, client):
        assert client.get("/api/models").status_code == 200

    def test_returns_list(self, client):
        data = client.get("/api/models").json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_model_has_required_fields(self, client):
        model = client.get("/api/models").json()[0]
        for field in ("id", "name", "auc_score", "inference_ms"):
            assert field in model, f"Field '{field}' missing from model object"


# ---------------------------------------------------------------------------
# Single predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_returns_200(self, client):
        r = client.post(
            "/api/predict",
            files={"image": ("test.png", _dummy_png(), "image/png")},
            data={"model_name": "cnn_r1"},
        )
        assert r.status_code == 200

    def test_response_structure(self, client):
        r = client.post(
            "/api/predict",
            files={"image": ("test.png", _dummy_png(), "image/png")},
            data={"model_name": "cnn_r1"},
        ).json()
        assert "predictions" in r
        assert "model_name" in r
        assert "inference_ms" in r
        assert "grad_cam_base64" in r

    def test_predictions_count(self, client):
        """Should return exactly 14 predictions."""
        preds = client.post(
            "/api/predict",
            files={"image": ("test.png", _dummy_png(), "image/png")},
            data={"model_name": "cnn_r1"},
        ).json()["predictions"]
        assert len(preds) == 14

    def test_predictions_have_label_and_confidence(self, client):
        preds = client.post(
            "/api/predict",
            files={"image": ("test.png", _dummy_png(), "image/png")},
            data={"model_name": "cnn_r1"},
        ).json()["predictions"]
        for p in preds:
            assert "label" in p
            assert "confidence_pct" in p
            assert 0 <= p["confidence_pct"] <= 100

    def test_invalid_model_returns_404(self, client):
        r = client.post(
            "/api/predict",
            files={"image": ("test.png", _dummy_png(), "image/png")},
            data={"model_name": "nonexistent_model_xyz"},
        )
        assert r.status_code == 404

    def test_all_registered_models_respond(self, client):
        models = client.get("/api/models").json()
        for m in models:
            r = client.post(
                "/api/predict",
                files={"image": ("test.png", _dummy_png(), "image/png")},
                data={"model_name": m["id"]},
            )
            assert r.status_code == 200, f"Model {m['id']} returned {r.status_code}"


# ---------------------------------------------------------------------------
# Batch predict
# ---------------------------------------------------------------------------

class TestBatchPredictEndpoint:
    def test_returns_200(self, client):
        r = client.post(
            "/api/predict/batch",
            files=[("image", ("test.png", _dummy_png(), "image/png"))],
            data=[("model_names", "cnn_r1"), ("model_names", "mlp_r1")],
        )
        assert r.status_code == 200

    def test_returns_one_result_per_model(self, client):
        model_names = ["cnn_r1", "mlp_r1"]
        results = client.post(
            "/api/predict/batch",
            files=[("image", ("test.png", _dummy_png(), "image/png"))],
            data=[("model_names", n) for n in model_names],
        ).json()
        assert isinstance(results, list)
        assert len(results) == len(model_names)

    def test_each_result_has_model_name(self, client):
        model_names = ["cnn_r1", "mlp_r1"]
        results = client.post(
            "/api/predict/batch",
            files=[("image", ("test.png", _dummy_png(), "image/png"))],
            data=[("model_names", n) for n in model_names],
        ).json()
        returned_names = {r["model_name"] for r in results}
        assert returned_names == set(model_names)


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

class TestPdfEndpoint:
    def _sample_prediction(self):
        return {
            "model_name": "cnn_r1",
            "model_version": "V4.2",
            "study_id": "STUDY-TEST-001",
            "inference_ms": 78,
            "viz_type": "GRAD-CAM",
            "grad_cam_base64": None,
            "predictions": [
                {"label": "Atelectasis", "confidence_pct": 82.5, "rank": 1},
                {"label": "Effusion", "confidence_pct": 45.1, "rank": 2},
            ],
        }

    def test_returns_200(self, client):
        r = client.post(
            "/api/report/pdf",
            json={"prediction_result": self._sample_prediction(), "patient_id": "PAT-001"},
        )
        assert r.status_code == 200

    def test_content_type_is_pdf(self, client):
        r = client.post(
            "/api/report/pdf",
            json={"prediction_result": self._sample_prediction(), "patient_id": "PAT-001"},
        )
        assert "pdf" in r.headers.get("content-type", "").lower()

    def test_pdf_has_nonzero_content(self, client):
        r = client.post(
            "/api/report/pdf",
            json={"prediction_result": self._sample_prediction(), "patient_id": "PAT-001"},
        )
        assert len(r.content) > 1024  # a real PDF is at least 1 KB

    def test_missing_body_returns_422(self, client):
        r = client.post("/api/report/pdf", json={})
        assert r.status_code == 422
