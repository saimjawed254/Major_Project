import os
import importlib.util
from http.server import HTTPServer

# Load the fetch-image.py module dynamically since it has a dash in the name
spec = importlib.util.spec_from_file_location("fetch_image", "api/fetch-image.py")
fetch_image = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_image)

HandlerClass = fetch_image.handler

if __name__ == '__main__':
    # Load .env.local manually for local testing
    if os.path.exists(".env.local"):
        with open(".env.local", "r") as f:
            content = f.read().strip()
            if content.startswith("GEE_SERVICE_ACCOUNT="):
                # Extract the JSON string and remove outer quotes
                env_val = content.replace("GEE_SERVICE_ACCOUNT=", "").strip().strip("'").strip('"')
                os.environ["GEE_SERVICE_ACCOUNT"] = env_val

    server_address = ('', 8000)
    print("Starting Local Python Server on port 8000...")
    print("This server manually emulates the Vercel Serverless environment for Windows.")
    
    httpd = HTTPServer(server_address, HandlerClass)
    httpd.serve_forever()
