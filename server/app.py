import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uvicorn

def main():
    from app import app as fastapi_app
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
