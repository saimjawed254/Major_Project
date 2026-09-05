import * as ort from 'onnxruntime-web';

// 6-class Taxonomy from Phase 1
export const CLASS_NAMES = [
  'Crop',
  'Forest',
  'HerbaceousVegetation',
  'Highway',
  'Urban',
  'WaterBodies'
];

export const CLASS_COLORS = [
  'rgba(234, 179, 8, 0.4)',    // Crop (Yellow)
  'rgba(34, 197, 94, 0.4)',    // Forest (Green)
  'rgba(163, 230, 53, 0.4)',   // Herbaceous (Light Green)
  'rgba(156, 163, 175, 0.4)',  // Highway (Gray)
  'rgba(239, 68, 68, 0.4)',    // Urban (Red)
  'rgba(59, 130, 246, 0.4)'    // Water (Blue)
];

let session: ort.InferenceSession | null = null;
let isLoading = false;

export async function initModel() {
  if (session) return true;
  if (isLoading) {
    while (isLoading) {
      await new Promise(r => setTimeout(r, 100));
    }
    return true;
  }
  
  isLoading = true;
  try {
    if (!session) {
      console.log('Loading ONNX Model...');
      
      // Load the quantized EuroSAT EfficientNetV2 model
      const modelResponse = await fetch('/model/efficientnet_quantized.onnx');
      const modelBuffer = await modelResponse.arrayBuffer();
      
      session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['wasm']
      });
      
      console.log('Quantized EfficientNet ONNX Model Loaded Successfully!');
    }
    return true;
  } catch (error) {
    console.error('Failed to load ONNX model:', error);
    return false;
  } finally {
    isLoading = false;
  }
}

// Prepare the image data for EfficientNet (1x3x224x224 tensor, normalized)
function preprocess(imageData: ImageData): ort.Tensor {
  const { data, width, height } = imageData;
  
  // Create a Float32Array for the tensor (3 channels, 224 height, 224 width)
  const float32Data = new Float32Array(3 * width * height);
  
  // ImageNet normalization constants used during training
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  
  // Loop through pixels and convert to CHW format while normalizing
  for (let i = 0; i < width * height; i++) {
    // R, G, B are interleaved in the Canvas ImageData
    const r = data[i * 4 + 0] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;
    
    // Normalize and assign to channels (R=0, G=1, B=2)
    float32Data[0 * width * height + i] = (r - mean[0]) / std[0];
    float32Data[1 * width * height + i] = (g - mean[1]) / std[1];
    float32Data[2 * width * height + i] = (b - mean[2]) / std[2];
  }
  
  return new ort.Tensor('float32', float32Data, [1, 3, height, width]);
}

export interface PredictionResult {
  classIndex: number;
  className: string;
  confidence: number;
  probabilities: number[];
  rawProbabilities?: number[];
  uncertainty: number;
}

export async function predictCell(imageData: ImageData): Promise<PredictionResult> {
  if (!session) {
    await initModel();
  }
  if (!session) {
    throw new Error("Model failed to load");
  }

  // 1. Preprocess
  const tensor = preprocess(imageData);
  
  // 2. Run inference
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = tensor;
  
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data as Float32Array;
  
  // 3. Softmax & Argmax
  let maxLogit = -Infinity;
  let maxIndex = 0;
  
  for (let i = 0; i < output.length; i++) {
    if (output[i] > maxLogit) {
      maxLogit = output[i];
      maxIndex = i;
    }
  }
  
  let sumExp = 0;
  const exps = new Float32Array(output.length);
  for (let i = 0; i < output.length; i++) {
    const expVal = Math.exp(output[i] - maxLogit);
    exps[i] = expVal;
    sumExp += expVal;
  }
  
  const probabilities = Array.from(exps).map(val => val / sumExp);
  const rawProbabilities = [...probabilities]; // Save original unpenalized array
  
  // Re-calculate the true maxIndex and confidence
  let maxIndexPost = 0;
  let maxProb = -1;
  for (let i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIndexPost = i;
    }
  }
  const confidence = probabilities[maxIndexPost];

  // 4. Calculate Entropy (Uncertainty)
  // H = -sum(p * log(p))
  let entropy = 0;
  for (let i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > 0) {
      entropy -= probabilities[i] * Math.log(probabilities[i]);
    }
  }
  
  // Normalize entropy to [0, 1] using max possible entropy log(N)
  const maxEntropy = Math.log(CLASS_NAMES.length);
  const uncertainty = entropy / maxEntropy;

  return {
    classIndex: maxIndexPost,
    className: CLASS_NAMES[maxIndexPost],
    confidence,
    probabilities,
    rawProbabilities,
    uncertainty
  };
}

