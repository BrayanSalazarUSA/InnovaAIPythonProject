from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("INNOVA_API_HOST", "0.0.0.0")
    port = int(os.getenv("INNOVA_API_PORT", "8512"))
    uvicorn.run("innova_investigation.api_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
