# GeoScan: Browser-Based Active Learning for Satellite Land-Cover Classification

GeoScan is a distributed geospatial intelligence platform designed to perform real-time land-use and land-cover (LULC) classification over Sentinel-2 satellite imagery of the Indian subcontinent. It addresses the **domain shift** problem—where models trained on European data (EuroSAT) systematically fail on Indian landscapes—by employing **Human-in-the-Loop Active Learning** directly in the browser.

The system runs a quantized EfficientNetV2-Small model entirely client-side via WebAssembly (ONNX Runtime Web). It identifies high-uncertainty geographic regions using Shannon Entropy and routes them to an expert annotator. Annotations are stored as coordinate bounding boxes (zero-image-storage), using Google Earth Engine as a serverless renderer.

Read the full research paper: [`GeoScan_Research_Paper.md`](GeoScan_Research_Paper.md).

## Live Testing (No Installation Required)

You can test the trained model directly in your browser without downloading the repository or setting up a local environment. The platform fetches live Sentinel-2 imagery from Google Earth Engine and performs inference entirely on your local device via WebAssembly.

> [!WARNING]
> **Availability Disclaimer:** The live application is hosted on a free-tier domain that may be discontinued or rate-limited in the future. If the live site is down, please refer to the Local Setup instructions below.

[Insert Live App Link Here]

## Key Features
- **Zero-Cost Edge Inference**: Runs CNN inference in-browser (WASM) without requiring server-side GPUs.
- **Targeted Domain Adaptation**: Entropy-guided active learning corrects specific cross-regional spectral biases (e.g., turbid Indian rivers misclassified as cropland).
- **Zero-Storage Dataset (IndiaSat)**: Ground-truth data is stored as coordinate shards in Firestore, rather than physical pixels.
- **Serverless Data Pipeline**: Fetches Sentinel-2 composites dynamically via a Python microservice integrated with Google Earth Engine.

## Local Setup Instructions

If you want to run the platform locally or contribute to the project, follow these steps.

### Prerequisites
- Node.js (v18+)
- Python (3.10+)
- A Google Earth Engine account and service account credentials.

### 1. Clone the Repository
```bash
git clone https://github.com/saimjawed254/Major_Project.git
cd Major_Project
```

### 2. Setup the Python Data Backend (Serverless Microservice)
The frontend relies on a Python script to fetch Sentinel-2 imagery from Google Earth Engine.

```bash
# Navigate to the frontend directory
cd frontend

# Install Python dependencies
pip install -r api/requirements.txt

# Start the local Python server
python local_server.py
```
*The local image server will run on `http://127.0.0.1:5000`.*

### 3. Setup the Frontend (Vite + React)
In a new terminal window, navigate back to the `frontend` directory:

```bash
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```
*The frontend will be available at `http://localhost:5173`.*

### 4. Environment Variables
You will need to provide your own Firebase configuration and Google Earth Engine service account credentials in a `.env.local` file inside the `frontend/` directory. (Note: These files are ignored by git to prevent accidental credential leaks).

```env
VITE_FIREBASE_API_KEY="your_api_key"
VITE_FIREBASE_AUTH_DOMAIN="your_auth_domain"
VITE_FIREBASE_PROJECT_ID="your_project_id"
VITE_FIREBASE_STORAGE_BUCKET="your_storage_bucket"
VITE_FIREBASE_MESSAGING_SENDER_ID="your_sender_id"
VITE_FIREBASE_APP_ID="your_app_id"
```

## Repository Structure

- `frontend/`: React/Vite web application, ONNX models, and Python GEE serverless script.
- `GeoScan_Research_Paper.md`: The complete academic paper detailing the methodology and results.
- `Phase1_Colab_Notebook.ipynb` & `phase_6_finetuning.ipynb`: Jupyter notebooks containing the original model training and domain adaptation fine-tuning pipelines.

## License

This project is licensed under the MIT License.
