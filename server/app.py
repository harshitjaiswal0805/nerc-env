import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uvicorn

def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "server_module",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server.py")
    )
    module = importlib.util.load_from_spec(spec)
    spec.loader.exec_module(module)
    uvicorn.run(module.app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
