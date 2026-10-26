import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app

@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "ev-retrieval" in str(data) or len(data["active_mcp_tools"]) > 0

@pytest.mark.asyncio
async def test_mcp_tools_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/mcp/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "mcp_tools" in data
        assert len(data["mcp_tools"]) >= 3

@pytest.mark.asyncio
async def test_diagnose_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "query": "Battery voltage drops from 40% to 5% instantly under acceleration",
            "vehicle_model": "Ather 450X",
            "subsystem": "Battery Management System (BMS)"
        }
        resp = await client.post("/api/v1/diagnose", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "root_cause_analysis" in data
        assert "evidence_sources" in data
        assert data["grounding_score"] > 0
