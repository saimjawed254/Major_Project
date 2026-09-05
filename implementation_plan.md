# Master Architecture Document: Distributed Land Cover Classification Platform

## 1. Executive Summary & Project Objectives
This project addresses the **domain shift** problem in satellite land cover classification. Deep learning models trained on source domains (e.g., European landscapes in EuroSAT) suffer significant accuracy degradation when applied to morphologically distinct target domains (e.g., Indian landscapes). 

To overcome this without expensive pre-labeling, we are engineering a **Distributed Crowdsourcing Platform**. The architecture leverages:
1. **Serverless Microservices** for on-demand geospatial data extraction.
2. **Heterogeneous Edge Computing** (Volunteer Computing) to perform heavy neural network inference directly in client web browsers.
3. **Distributed Consensus Mechanisms** to allow public crowdsourcing of error corrections safely.

These human-verified corrections form the dataset used to continuously fine-tune the baseline model. The entire architecture is designed to be highly scalable, fault-tolerant, and hosted entirely on **100% Free Tier** resources, serving as a master-class demonstration of Advanced Distributed Systems applied to Deep Learning.

---

## 2. Phase 1: Robust Baseline Model Engineering (The "Brain")
*Compute Resource: Google Colab (Free Tier) | Framework: PyTorch*

Before distributing the workload to the edge, we must engineer a highly robust baseline model that resists domain shift as much as possible out-of-the-box.

### 2.1 The Model Architecture: EfficientNetV2-Small
We utilize `EfficientNetV2-Small` rather than older ResNet architectures.
- **Theoretical Justification:** EfficientNetV2 utilizes Fused-MBConv blocks (which replace the depthwise convs in early layers) and progressive learning (dynamically adjusting image size and regularization during training). 
- **Edge Computing Necessity:** It achieves state-of-the-art accuracy with a parameter count of only ~24M (compared to ResNet-50's ~25M or ViT's ~86M), making the compiled weights small enough (~90MB) to be downloaded and executed in a standard client web browser via WebGL.

### 2.2 Taxonomy Integration (Early Merging)
EuroSAT provides 10 classes, but our application requires a 6-class canonical taxonomy. We enforce this mapping at the `torch.utils.data.Dataset` level:
- `AnnualCrop`, `PermanentCrop` $\rightarrow$ `Crop` (Index 0)
- `Forest` $\rightarrow$ `Forest` (Index 1)
- `HerbaceousVegetation` $\rightarrow$ `HerbaceousVegetation` (Index 2)
- `Highway` $\rightarrow$ `Highway` (Index 3)
- `Industrial`, `Residential` $\rightarrow$ `Urban` (Index 4)
- `River`, `SeaLake` $\rightarrow$ `WaterBodies` (Index 5)
By mapping targets before the loss calculation, the model outputs exactly 6 logits from Epoch 1, preventing wasted representational capacity on irrelevant sub-class boundaries.

### 2.3 Robust Training Pipeline
To maximize out-of-the-box generalization to the target Indian domain, we implement three advanced techniques:
1. **Stratified K-Fold Cross-Validation (K=5):** The dataset is split into 5 distinct folds. Training 5 models ensures the architecture is not overfitting to a statistical anomaly in a single train/validation split.
2. **Advanced Augmentation (CutMix & Mixup):** 
   - *Mixup* blends two images and their one-hot labels linearly. 
   - *CutMix* cuts a spatial patch of image A and pastes it onto image B, proportionally mixing the labels based on patch area. 
   - *Why?* This forces the model to learn structural context (e.g., houses mixed with trees) rather than relying on pure pixel colors (e.g., green = forest), directly combating domain shift.
3. **Focal Loss:** Replaces standard Cross-Entropy Loss. 
   - Formula: $FL(p_t) = -\alpha_t(1 - p_t)^\gamma \log(p_t)$
   - *Why?* By setting the focusing parameter $\gamma > 0$, Focal Loss automatically down-weights the gradient contribution of "easy" examples (where $p_t$ is high) and forces the optimizer to focus heavily on "hard", ambiguous boundary examples (where $p_t$ is low).

### 2.4 Edge Export Pipeline
The final PyTorch weights (`.pth`) are exported to the ONNX graph format. The ONNX model is then converted to `TensorFlow.js` format (`model.json` + binary `.bin` weight shards) using the `tfjs-converter`. This is the exact artifact that will be cached in the user's browser.

---

## 3. Phase 2: The Virtual Quadtree (The "Map")
*Resource: React Frontend Math Logic (Vite/TypeScript)*

We entirely replace heavy, monolithic mapping libraries (like Leaflet or Mapbox) with a lightweight, mathematically generated hierarchical "File Explorer" UI based on a Quadtree data structure.

### 3.1 Mathematical Folder Structure
There are no physical folders stored on a server. The React application calculates the Bounding Box (BBox) dynamically as the user navigates deeper into the tree.
- **Root Node (Level 0):** Defined by a massive BBox covering the target region (e.g., `MinLat: 24.0, MinLon: 83.0, MaxLat: 28.0, MaxLon: 88.0` for Bihar).
- **Navigation Logic:** Clicking a "sub-folder" mathematically divides the Lat/Lon span by 2. For example, the top-left sub-folder spans from `MinLat` to `MidLat`, and `MinLon` to `MidLon`.
- **Leaf Node Resolution:** We calculate the maximum depth (tree height) such that the final leaf node spans exactly **2.24 km x 2.24 km** on the ground (matching the 224x224 pixel resolution at Sentinel-2's 10m/pixel spatial resolution). For a 500x500km state, this requires a tree height of exactly **8 levels**.

---

## 4. Phase 3: Serverless GEE Fetcher (Microservice)
*Resource: Vercel Serverless Functions (Python)*

We cannot fetch imagery from Google Earth Engine (GEE) directly from the React frontend because it would expose our private Service Account JSON keys to the public internet.

### 4.1 On-Demand Extraction API Workflow
- **Frontend Request:** When a user reaches Level 8 (Leaf Node), the React app sends an HTTP GET request to our Vercel Serverless Function containing the calculated BBox: `?minLat=X&minLon=Y&maxLat=Z&maxLon=W`.
- **Backend Authentication:** The Vercel function securely loads the GEE Service Account Key from environment variables and initializes the `ee` Python API.
- **Data Extraction:** The function queries the Sentinel-2 ImageCollection, filters for the least cloudy image in the specified date range, and clips it to the requested BBox.
- **Processing:** The raw tensor is resized to exactly `224x224x3`, normalized, and converted to a Base64 JPEG string.
- **Response:** The Base64 string is returned to the React frontend.
- **Distributed Advantage:** This microservice is stateless, scales infinitely to zero or thousands of concurrent requests, and incurs $0 in static storage costs.

---

## 5. Phase 4: Distributed Edge Computing (Volunteer Computing)
*Resource: Client Web Browsers (TensorFlow.js WebGL Backend)*

This phase represents the core of the Heterogeneous Distributed Computing architecture. We offload heavy neural network inference from a centralized cloud GPU to the distributed CPUs/GPUs of the user base.

### 5.1 The Edge Workflow
1. **Image Load:** User clicks a Leaf Node. The React app fetches the Base64 image via the Phase 3 Microservice.
2. **State Check:** The React app queries the Firebase database: *"Do predictions exist for this specific Leaf Node ID (e.g., `node_L8_X_Y`)?"*
3. **Execution Prompt:** If **NO**, the UI prompts the user: *"Compute predictions locally using your device?"*
4. **Model Caching:** If accepted, the browser downloads the `model.json` and weight shards (Phase 1). TF.js caches these in the browser's IndexedDB so subsequent computations are instant.
5. **Hardware Acceleration:** TF.js utilizes the user's local GPU via the WebGL or WebGPU backend to execute the EfficientNetV2 inference on the 224x224 image patch.
6. **Data Payload Generation:** The browser computes the predicted class, confidence scores, raw Softmax probabilities, and Shannon Entropy (uncertainty).
7. **Synchronization:** The browser silently POSTs this JSON payload to Firebase. The Leaf Node is now "solved", and all future global users visiting this coordinate will instantly fetch these cached predictions without running the model.

---

## 6. Phase 5: Distributed Consensus & Crowdsourcing
*Resource: Google Firebase Firestore (NoSQL)*

To achieve Domain Adaptation, we rely on human-in-the-loop corrections to generate a target-domain dataset.

### 6.1 Database Schema (Firestore)
Each Leaf Node has a document in the `cells` collection:
```json
{
  "cell_id": "L8_lat24.5_lon83.2",
  "ai_prediction": "Forest",
  "ai_confidence": 0.88,
  "canonical_label": "Forest", 
  "votes": {
    "Urban": 2,
    "Crop": 0,
    "WaterBodies": 0
  },
  "consensus_reached": false
}
```

### 6.2 Optimistic Concurrency Control (OCC)
- Users can click any cell to dispute the AI prediction (e.g., voting for `Urban`).
- **The Race Condition:** If User A in Delhi and User B in Mumbai vote for `Urban` on the exact same cell at the exact same millisecond, a naive `votes + 1` database write will overwrite one vote, resulting in a count of 1 instead of 2.
- **The Solution:** We use Firebase Firestore **Transactions**. A transaction reads the current document state, applies the increment locally, and attempts to write. If the underlying document changed during this process (e.g., User B's vote landed first), the transaction safely aborts, re-reads the new state, and applies User A's vote again. This guarantees strict mathematical consistency across the distributed network.

### 6.3 Majority Rules (Eventual Consistency)
- Once a specific class in the `votes` map reaches a predetermined threshold (e.g., 3 votes), the `canonical_label` is permanently flipped to that class, and `consensus_reached` is set to `true`.
- Firebase's real-time listeners push this state change to all active React clients globally, updating the UI color for that cell instantly.

---

## 7. Phase 6: The Iterative Loop (Domain Adaptation Fine-Tuning)
- As the platform operates, the Firestore database passively accumulates a high-quality, human-verified dataset of errors (where `ai_prediction` != `canonical_label`).
- Periodically, a Colab script pulls all documents where `consensus_reached == true`.
- The baseline EfficientNetV2 model is fine-tuned (using a lower learning rate and freezing early layers) exclusively on this newly verified target-domain data.
- The updated model is converted to TF.js and deployed to the frontend, completing the Machine Learning lifecycle.

---

## 8. Extensive Theoretical Q&A (Design Decisions & Justifications)

**Q: Why use EfficientNetV2 instead of Wide ResNet-50-2 or a Vision Transformer (ViT)?**
> **A:** Deploying a deep learning model to the edge (web browsers via TF.js) creates extreme constraints on memory, download size, and computational complexity. Wide ResNet-50 is an older architecture that is highly parameter-heavy and slow. Vision Transformers (ViTs) require massive amounts of data to train, lack inductive biases for local spatial structures (making them poor at small datasets), and are incredibly difficult to run smoothly inside a standard web browser without crashing WebGL. EfficientNetV2 was specifically designed by Google using Neural Architecture Search (NAS) to optimize the Pareto frontier of parameter size, FLOPs (inference speed), and accuracy. It is the undisputed best choice for "Edge Computing".

**Q: Why are we generating imagery On-Demand via GEE instead of pre-computing the entire Quadtree?**
> **A:** This is a problem of geometric scaling. An 8-level quadtree for a single Indian state contains over $4^8 = 65,536$ leaf nodes. Pre-computing and storing millions of image patches would obliterate free-tier storage limits (like IPFS, AWS S3 Free Tier, or Firebase Storage) and take weeks of continuous Colab compute time. By shifting to a Serverless Microservice architecture, we generate images dynamically *only when requested*. Storage costs become exactly $0, and the system can theoretically scale to cover the entire globe instantly.

**Q: Why use Firebase Transactions (Optimistic Concurrency Control)?**
> **A:** In a distributed system, you must account for race conditions and concurrent state mutations. If two nodes attempt to mutate the same state simultaneously, a standard database write will result in a "Lost Update" anomaly. Optimistic Concurrency Control (OCC) assumes that conflicts are rare. It proceeds with the transaction but checks for conflicts before committing. Firestore transactions implement OCC perfectly, ensuring that the global vote tally is strictly atomic, consistent, isolated, and durable (ACID) across the distributed network.

**Q: Why are we using a Virtual Folder structure instead of a mapping library like Leaflet or Mapbox?**
> **A:** Map libraries are incredibly heavy dependencies that require complex tile-server backend setups (serving XYZ/TMS tiles) and often rely on paid APIs if usage spikes. The hierarchical folder (Quadtree) UI is mathematically identical to how map tiles work under the hood, but it offloads the navigation logic entirely to simple React state variables. It is lighter, faster to build, immune to third-party mapping API rate limits, and natively supports the "drill-down" UX needed for leaf-node validation.

**Q: Why enforce exactly a 224x224 pixel resolution for the leaf nodes?**
> **A:** $224 \times 224$ is the gold standard input dimension for almost all ImageNet-pretrained Convolutional Neural Networks (including EfficientNet). Changing this dimension requires interpolating positional embeddings or adjusting strides, which degrades pre-trained weights. Furthermore, at Sentinel-2's 10-meter-per-pixel spatial resolution, a $224 \times 224$ grid covers exactly $2.24 \text{ km} \times 2.24 \text{ km}$ on the ground. This specific size is perfect for land cover classification: it is large enough to provide meaningful neighborhood context to the model (capturing whole farms or urban blocks), but small enough that a human annotator can quickly visually verify the contents without needing to zoom or pan.

**Q: Why are CutMix and Mixup augmentations necessary for the baseline model?**
> **A:** The core problem of this paper is Domain Shift. The model trains on European data and fails on Indian data. A major reason for this failure is that CNNs are "lazy"; they learn to associate pure, solid green textures with "Forest" and pure grey textures with "Urban." Indian landscapes are often highly intermixed (e.g., dense villages with houses surrounded by heavy tree canopy). CutMix mathematically cuts a patch of trees and pastes it onto a house image during training, mixing the target labels proportionally. This forces the neural network to stop relying on single-pixel color distributions and start learning structural context and shape boundaries, drastically improving its robustness when it eventually sees intermixed target domains in production.

**Q: In academic terms, what makes this a "Distributed Systems" project?**
> **A:** This architecture is not a standard full-stack app; it demonstrates four major, distinct paradigms of distributed systems:
> 1. **Heterogeneous Edge Computing (Volunteer Computing):** Offloading deep learning inference from centralized cloud GPUs to distributed, untrusted client web browsers, similar to SETI@home or BOINC architectures.
> 2. **Microservice Architecture:** Decoupling the heavy authentication and extraction logic into a stateless, independently scaling serverless function (the GEE Fetcher).
> 3. **Distributed Consensus:** Utilizing Optimistic Concurrency Control to safely aggregate concurrent crowdsourced state mutations across an asynchronous global network of users.
> 4. **Eventual Consistency:** The React frontend updates optimistically for the local user, while the backend ensures the global state eventually synchronizes and propagates across all active clients when consensus thresholds are mathematically met.
