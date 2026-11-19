import io
from pathlib import Path
from app import app

def test_flask_endpoints():
    client = app.test_client()

    print("1. Testing GET / (Redirects to /login)...")
    res = client.get("/")
    assert res.status_code in (200, 302)
    print("   Status:", res.status_code)

    print("2. Testing GET /login...")
    res = client.get("/login")
    assert res.status_code == 200
    assert b"System Authentication" in res.data
    print("   Login page rendered successfully.")

    print("3. Testing POST /login with Rohit credentials...")
    res = client.post("/login", data={"username": "@Rohit", "password": "rohit123"}, follow_redirects=True)
    assert res.status_code == 200
    assert b"Document Upload & Analysis Form" in res.data
    print("   Authenticated session established for @Rohit.")

    print("4. Testing GET /api/meta...")
    res = client.get("/api/meta")
    assert res.status_code == 200
    meta = res.get_json()
    assert "ensemble_formula" in meta
    print("   Metadata returned:", meta["ensemble_formula"])

    print("5. Testing POST /api/text/analyse...")
    res = client.post("/api/text/analyse", json={
        "text": "Furthermore, it is crucial to delve into the intricate tapestry of modern large language model architectures."
    })
    assert res.status_code == 200
    t_res = res.get_json()
    assert t_res["success"] is True
    assert len(t_res["sentences"]) > 0
    print("   Direct text analysis probability:", t_res["overall_ai_prob"], "Band:", t_res["sentences"][0]["band"])

    print("6. Testing POST /api/presentation/analyse with sample_presentation.pptx...")
    with open("sample_presentation.pptx", "rb") as f:
        file_bytes = f.read()

    data = {
        "file": (io.BytesIO(file_bytes), "sample_presentation.pptx")
    }
    res = client.post("/api/presentation/analyse", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    p_res = res.get_json()
    assert p_res["success"] is True
    print(f"   PPTX analysis parsed {len(p_res['slides'])} slides and {len(p_res['sentences'])} sentences.")
    print("   Total warnings triggered:", p_res["doc_summary"]["total_warnings"])

    print("7. Testing POST /api/presentation/chat (Grounded RAG)...")
    res = client.post("/api/presentation/chat", json={"query": "pure text"})
    assert res.status_code == 200
    chat_res = res.get_json()
    assert len(chat_res["citations"]) > 0
    print("   RAG Grounded Response:", chat_res["answer"])

    print("\n[✓] All Flask endpoints and application workflows verified successfully!")

if __name__ == "__main__":
    test_flask_endpoints()
