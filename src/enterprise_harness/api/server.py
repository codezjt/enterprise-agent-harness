from fastapi import FastAPI


app = FastAPI(
    title="Enterprise Agent Harness",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "enterprise-agent-harness",
    }