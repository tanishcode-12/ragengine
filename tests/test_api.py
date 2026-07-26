import io


def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["main_index_count"] == 0


def test_root(api_client):
    resp = api_client.get("/")
    assert resp.status_code == 200
    assert resp.json()["name"] == "ragengine"


def test_ingest_text_then_query(api_client):
    resp = api_client.post(
        "/ingest/text",
        json={
            "text": "Full time employees receive 20 days of paid vacation leave per year.",
            "source_name": "handbook.txt",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["chunks_indexed"] == 1

    resp = api_client.post("/query", json={"question": "How many vacation days?", "k": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sources"]) == 1
    assert body["sources"][0]["metadata"]["source"] == "handbook.txt"
    assert isinstance(body["answer"], str) and body["answer"]


def test_ingest_file_upload(api_client, fixtures_dir):
    with open(fixtures_dir / "sample.txt", "rb") as f:
        resp = api_client.post("/ingest", files={"file": ("sample.txt", f, "text/plain")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["documents_loaded"] == 1
    assert body["chunks_indexed"] > 0


def test_ingest_file_upload_unknown_extension_returns_400(api_client):
    resp = api_client.post("/ingest", files={"file": ("data.zip", io.BytesIO(b"not a real archive"), "application/zip")})
    assert resp.status_code == 400


def test_query_with_unknown_retriever_returns_400(api_client):
    api_client.post("/ingest/text", json={"text": "some content to index"})
    resp = api_client.post("/query", json={"question": "anything", "retriever_name": "not-a-real-retriever"})
    assert resp.status_code == 400


def test_validate_question_rejects_empty(api_client):
    resp = api_client.post("/query", json={"question": ""})
    assert resp.status_code == 422  # pydantic min_length=1 validation


def test_agent_query_end_to_end(api_client):
    api_client.post("/ingest/text", json={"text": "Full time employees receive 20 days of paid vacation leave per year."})
    resp = api_client.post("/agent/query", json={"question": "How many vacation days?"})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["answer"], str) and body["answer"]
    assert isinstance(body["trace"], list) and len(body["trace"]) >= 1


def test_ingest_json_via_upload_uses_default_jq_schema(api_client, fixtures_dir):
    # No jq_schema is exposed over the API (it's loader-specific and the
    # whole-file default is a reasonable zero-config behaviour), so a
    # plain JSON upload is ingested as one document containing the raw
    # JSON structure rendered as text.
    with open(fixtures_dir / "sample.json", "rb") as f:
        resp = api_client.post("/ingest", files={"file": ("sample.json", f, "application/json")})
    assert resp.status_code == 200
    assert resp.json()["documents_loaded"] == 1
