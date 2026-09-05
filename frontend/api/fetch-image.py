import os
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler
import ee
from google.oauth2 import service_account
import urllib.request
import base64

# Global initialization so it is cached between serverless invocations
_GEE_INITIALIZED = False

def init_gee():
    global _GEE_INITIALIZED
    if _GEE_INITIALIZED:
        return
        
    try:
        service_account_str = os.environ.get('GEE_SERVICE_ACCOUNT')
        if not service_account_str:
            raise ValueError("GEE_SERVICE_ACCOUNT environment variable is not set")
            
        credentials_dict = json.loads(service_account_str)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        
        ee.Initialize(credentials, project=credentials_dict.get('project_id'))
        _GEE_INITIALIZED = True
    except Exception as e:
        print(f"Failed to initialize GEE: {e}")
        raise

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Parse query parameters
            parsed_url = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            # Extract coordinates and dimensions
            minLat = float(query_params.get('minLat', [0])[0])
            maxLat = float(query_params.get('maxLat', [0])[0])
            minLon = float(query_params.get('minLon', [0])[0])
            maxLon = float(query_params.get('maxLon', [0])[0])
            dim = query_params.get('dim', ['224x224'])[0]
            
            if minLat == 0 and maxLat == 0:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing coordinates")
                return

            init_gee()

            # Create bounding box geometry
            roi = ee.Geometry.Rectangle([minLon, minLat, maxLon, maxLat])

            # Query Sentinel-2
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                          .filterBounds(roi)
                          .filterDate('2023-01-01', '2024-01-01')
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                          .sort('CLOUDY_PIXEL_PERCENTAGE'))

            # Get the best image
            image = collection.first()
            
            # Select RGB bands (B4=Red, B3=Green, B2=Blue)
            rgb_image = image.select(['B4', 'B3', 'B2'])
            
            # Visualize / Normalize (Sentinel-2 SR is usually 0-10000, we scale to 0-255 for JPEG)
            visualized = rgb_image.visualize(min=0, max=3000, gamma=1.4)

            # Generate thumbnail URL
            thumb_url = visualized.getThumbURL({
                'dimensions': dim,
                'region': roi,
                'format': 'jpg'
            })

            # Fetch the image bytes from Google's CDN
            req = urllib.request.Request(thumb_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                image_bytes = response.read()

            # Convert to base64
            base64_img = base64.b64encode(image_bytes).decode('utf-8')
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response_data = {
                'image_base64': f"data:image/jpeg;base64,{base64_img}",
                'bounds': {
                    'minLat': minLat,
                    'maxLat': maxLat,
                    'minLon': minLon,
                    'maxLon': maxLon
                }
            }
            
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            error_response = {'error': str(e)}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
