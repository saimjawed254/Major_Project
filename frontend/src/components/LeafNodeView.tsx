import React, { useState, useEffect, useRef } from 'react';
import { type Quadrant } from '../utils/quadtree';
import { predictCell, initModel, type PredictionResult, CLASS_COLORS, CLASS_NAMES } from '../utils/inference';
import { db } from '../firebase';
import { collection, addDoc } from 'firebase/firestore';

interface Props {
  node: Quadrant;
  customImageUrl?: string;
}

type ViewMode = 'original' | 'overlay';

const CANVAS_COLORS = [
  [234, 179, 8, 120],   // Crop
  [34, 197, 94, 120],   // Forest
  [163, 230, 53, 120],  // HerbaceousVegetation
  [156, 163, 175, 120], // Highway
  [239, 68, 68, 120],   // Urban
  [59, 130, 246, 120]   // WaterBodies
];

interface DashboardProps extends Props {
  imageUrl: string;
  modelType: 'base' | 'finetuned';
}

const ModelDashboard: React.FC<DashboardProps> = ({ node, customImageUrl, imageUrl, modelType }) => {
  const [viewMode, setViewMode] = useState<ViewMode>('overlay');
  const [isInferencing, setIsInferencing] = useState<boolean>(false);
  const [scanProgress, setScanProgress] = useState<number>(0);
  
  const [activePrediction, setActivePrediction] = useState<PredictionResult | null>(null);
  const [activeThumbnail, setActiveThumbnail] = useState<string>('');
  const [activeBBox, setActiveBBox] = useState<any>(null);
  
  const [selectedLabel, setSelectedLabel] = useState<string>('');
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [firebaseError, setFirebaseError] = useState<string | null>(null);
  
  const rawImageCanvasRef = useRef<HTMLCanvasElement>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!imageUrl || !rawImageCanvasRef.current || !heatmapCanvasRef.current) return;
    
    const img = new Image();
    img.src = imageUrl;
    img.onload = async () => {
      setIsInferencing(true);
      setScanProgress(0);
      
      const rawCtx = rawImageCanvasRef.current!.getContext('2d', { willReadFrequently: true });
      if (!rawCtx) return;
      rawCtx.drawImage(img, 0, 0, 1792, 896);
      
      const heatCtx = heatmapCanvasRef.current!.getContext('2d');
      if (!heatCtx) return;
      
      const sumProbs = new Float32Array(1792 * 896 * 6);
      
      const stride = 168;
      const xSteps = [];
      const ySteps = [];
      
      for (let x = 0; x <= 1792 - 224; x += stride) xSteps.push(x);
      if (xSteps[xSteps.length - 1] !== 1792 - 224) xSteps.push(1792 - 224);
      
      for (let y = 0; y <= 896 - 224; y += stride) ySteps.push(y);
      if (ySteps[ySteps.length - 1] !== 896 - 224) ySteps.push(896 - 224);
      
      const totalPatches = xSteps.length * ySteps.length;
      let patchesDone = 0;

      const heatImgData = new ImageData(1792, 896);
      
      for (let y of ySteps) {
        for (let x of xSteps) {
          const imageData = rawCtx.getImageData(x, y, 224, 224);
          
          try {
            const result = await predictCell(imageData, modelType);
            
            for (let py = y; py < y + 224; py++) {
              for (let px = x; px < x + 224; px++) {
                const idx = (py * 1792 + px) * 6;
                for (let c = 0; c < 6; c++) {
                  sumProbs[idx + c] += result.probabilities[c];
                }
              }
            }
            
            patchesDone++;
            setScanProgress((patchesDone / totalPatches) * 100);
            await new Promise(r => setTimeout(r, 10));
          } catch (e) {
            console.error(`Prediction failed for patch ${x},${y}`, e);
          }
        }
      }
      
      for (let py = 0; py < 896; py++) {
        for (let px = 0; px < 1792; px++) {
          const idx = (py * 1792 + px) * 6;
          
          let maxProb = -1;
          let maxClass = 0;
          for (let c = 0; c < 6; c++) {
            if (sumProbs[idx + c] > maxProb) {
              maxProb = sumProbs[idx + c];
              maxClass = c;
            }
          }
          
          const pxIdx = (py * 1792 + px) * 4;
          heatImgData.data[pxIdx] = CANVAS_COLORS[maxClass][0];
          heatImgData.data[pxIdx + 1] = CANVAS_COLORS[maxClass][1];
          heatImgData.data[pxIdx + 2] = CANVAS_COLORS[maxClass][2];
          heatImgData.data[pxIdx + 3] = CANVAS_COLORS[maxClass][3];
        }
      }
      
      heatCtx.putImageData(heatImgData, 0, 0);
      setIsInferencing(false);
    };
  }, [imageUrl, modelType]);

  const handleMapClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isInferencing || !rawImageCanvasRef.current) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = ((e.clientX - rect.left) / rect.width) * 1792;
    const clickY = ((e.clientY - rect.top) / rect.height) * 896;
    
    const patchX = Math.max(0, Math.min(1792 - 224, Math.floor(clickX - 112)));
    const patchY = Math.max(0, Math.min(896 - 224, Math.floor(clickY - 112)));
    
    const { minLat, maxLat, minLon, maxLon } = node.bbox;
    const centerLat = (minLat + maxLat) / 2;
    const centerLon = (minLon + maxLon) / 2;
    const widthDeg = maxLon - minLon;
    const heightDeg = maxLat - minLat;
    
    const fetchMinLat = centerLat - (heightDeg * 2);
    const fetchMaxLat = centerLat + (heightDeg * 2);
    const fetchMinLon = centerLon - (widthDeg * 4);
    const fetchMaxLon = centerLon + (widthDeg * 4);
    
    const patchMinLon = fetchMinLon + (patchX / 1792) * (fetchMaxLon - fetchMinLon);
    const patchMaxLon = fetchMinLon + ((patchX + 224) / 1792) * (fetchMaxLon - fetchMinLon);
    const patchMaxLat = fetchMaxLat - (patchY / 896) * (fetchMaxLat - fetchMinLat);
    const patchMinLat = fetchMaxLat - ((patchY + 224) / 896) * (fetchMaxLat - fetchMinLat);
    
    setActiveBBox({
      minLat: patchMinLat,
      maxLat: patchMaxLat,
      minLon: patchMinLon,
      maxLon: patchMaxLon
    });
    
    const rawCtx = rawImageCanvasRef.current.getContext('2d');
    if (!rawCtx) return;
    
    const imageData = rawCtx.getImageData(patchX, patchY, 224, 224);
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 224;
    tempCanvas.height = 224;
    tempCanvas.getContext('2d')?.putImageData(imageData, 0, 0);
    setActiveThumbnail(tempCanvas.toDataURL('image/jpeg', 0.9));
    
    setSaveSuccess(false);
    
    try {
      setActivePrediction(null);
      const result = await predictCell(imageData, modelType);
      setActivePrediction(result);
      setSelectedLabel(result.className);
    } catch (err) {
      console.error("Live inference failed", err);
    }
  };

  const handleSaveAnnotation = async () => {
    if (!activePrediction || !activeBBox || !selectedLabel) return;
    
    setIsSaving(true);
    try {
      await addDoc(collection(db, "annotations"), {
        timestamp: new Date().toISOString(),
        model_version: modelType === 'finetuned' ? 'finetuned_v3' : modelType,
        ai_prediction: activePrediction.className,
        ai_confidence: activePrediction.confidence,
        human_label: selectedLabel,
        bbox: activeBBox,
        centerLat: (activeBBox.minLat + activeBBox.maxLat) / 2,
        centerLon: (activeBBox.minLon + activeBBox.maxLon) / 2
      });
      setSaveSuccess(true);
    } catch (error: any) {
      console.error("Error saving annotation to Firestore:", error);
      setFirebaseError(error.message || "Unknown Firebase Error");
      alert(`Firebase Error: ${error.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="dashboard-top-row">
      {/* LEFT PANEL: MAP VIEWER */}
      <div className="map-panel">
        <div className="panel-header">
          <h2>
            <span className="icon">🗺️</span> Sentinel-2 Panoramic Map
            <span style={{ marginLeft: '10px', fontSize: '1rem', color: '#10b981' }}>
              ({modelType === 'base' ? 'Base Model' : 'Fine-Tuned Model'})
            </span>
          </h2>
          
          <div className="view-toggles">
            <button 
              className={`toggle-btn ${viewMode === 'original' ? 'active' : ''}`}
              onClick={() => setViewMode('original')}
            >Raw Image</button>
            <button 
              className={`toggle-btn ${viewMode === 'overlay' ? 'active' : ''}`}
              onClick={() => setViewMode('overlay')}
            >Heatmap Overlay</button>
          </div>
        </div>
        
        <div className="satellite-data-section">
          <div className="canvas-wrapper">
            <img src={imageUrl} alt="Background" className="background-img" />
            
            <canvas ref={rawImageCanvasRef} width={1792} height={896} style={{ display: 'none' }} />
            
            <canvas 
              ref={heatmapCanvasRef} 
              width={1792} 
              height={896} 
              className={`seamless-heatmap ${viewMode === 'original' ? 'hidden' : ''}`}
              onClick={handleMapClick}
            />
            
            {isInferencing && (
              <div className="scanning-overlay">
                <div className="progress-bar-container">
                  <div className="progress-bar" style={{ width: `${scanProgress}%` }}></div>
                </div>
                <p>Running Sliding Window Inference ({Math.round(scanProgress)}%)...</p>
              </div>
            )}
          </div>
        </div>
        
        <div className="class-legend-panel glass-panel-inner" style={{ marginTop: '1.5rem', padding: '1rem 1.5rem' }}>
          <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.1rem', textAlign: 'center' }}>Class Legend</h3>
          <div className="legend-grid-horizontal" style={{ marginTop: '0.5rem' }}>
            {CLASS_NAMES.map((name, i) => (
              <div key={name} className="legend-item">
                <div className="legend-color" style={{ backgroundColor: CLASS_COLORS[i].replace('0.4', '1.0') }}></div>
                <span>{name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* RIGHT PANEL: CELL INSPECTOR */}
      <div className="inspector-panel glass-panel-inner">
        <h3>Cell Inspector</h3>
        {!activeThumbnail ? (
          <div className="empty-inspector">
            <p>Click anywhere on the seamless heatmap to extract a 224x224 patch, run a live inference, and analyze its uncertainty.</p>
          </div>
        ) : (
          <div className="inspector-content">
            <img src={activeThumbnail} alt="Cell" className="inspector-img" />
            
            {activePrediction ? (
              <div className="prediction-stats">
                <div className="stat-row">
                  <span>Live Prediction:</span>
                  <strong style={{ color: CLASS_COLORS[activePrediction.classIndex].replace('0.4', '1.0') }}>
                    {activePrediction.className}
                  </strong>
                </div>
                
                <div className="stat-row">
                  <span>Confidence:</span>
                  <strong>{(activePrediction.confidence * 100).toFixed(2)}%</strong>
                </div>

                <div className="stat-row">
                  <span>Uncertainty Score:</span>
                  <div className="uncertainty-bar-container">
                    <div 
                      className="uncertainty-bar" 
                      style={{ 
                        width: `${activePrediction.uncertainty * 100}%`,
                        backgroundColor: activePrediction.uncertainty > 0.5 ? 'var(--neon-accent)' : '#10b981'
                      }}
                    ></div>
                  </div>
                  <span>{(activePrediction.uncertainty).toFixed(3)}</span>
                </div>
                
                {activePrediction.uncertainty > 0.6 && (
                  <p className="uncertainty-warning">⚠️ High uncertainty detected. Human review recommended.</p>
                )}

                {activePrediction.rawProbabilities && (
                  <div className="raw-probabilities">
                    <h4>Raw Probabilities:</h4>
                    <ul>
                      {activePrediction.rawProbabilities.map((prob, idx) => (
                        <li key={idx}>
                          <span style={{ color: CLASS_COLORS[idx].replace('0.4', '1.0') }}>{CLASS_NAMES[idx]}</span>
                          <span>{(prob * 100).toFixed(1)}%</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="annotation-section">
                  <h4>Verify / Annotate</h4>
                  {customImageUrl ? (
                    <p style={{ color: '#fca5a5', fontSize: '0.85rem' }}>
                      Firestore saving is disabled for custom uploads.
                    </p>
                  ) : (
                    <>
                      <select 
                        className="annotation-select" 
                        value={selectedLabel}
                        onChange={(e) => setSelectedLabel(e.target.value)}
                      >
                        {CLASS_NAMES.map(name => (
                          <option key={name} value={name}>{name}</option>
                        ))}
                      </select>
                      <button 
                        className="save-annotation-btn" 
                        onClick={handleSaveAnnotation}
                        disabled={isSaving || saveSuccess}
                        style={{ backgroundColor: saveSuccess ? '#10b981' : '' }}
                      >
                        {isSaving ? 'Saving...' : (saveSuccess ? 'Saved! ✅' : 'Save to Firestore')}
                      </button>
                      {!saveSuccess && !firebaseError && (
                        <p className="annotation-hint">Submits a precise geospatial bounding box to the database for Phase 6 training.</p>
                      )}
                      {firebaseError && (
                        <p className="annotation-hint" style={{ color: '#ef4444' }}>
                          Firebase Error: {firebaseError}
                        </p>
                      )}
                    </>
                  )}
                </div>
              </div>
            ) : (
              <p>Running local AI inference...</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export const LeafNodeView: React.FC<Props> = ({ node, customImageUrl }) => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loadingImage, setLoadingImage] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchSatelliteImage = async () => {
      setLoadingImage(true);
      setError(null);
      
      try {
        initModel();
        
        if (customImageUrl) {
          setImageUrl(customImageUrl);
          setLoadingImage(false);
          return;
        }
        
        const { minLat, maxLat, minLon, maxLon } = node.bbox;
        const centerLat = (minLat + maxLat) / 2;
        const centerLon = (minLon + maxLon) / 2;
        
        const widthDeg = maxLon - minLon;
        const heightDeg = maxLat - minLat;
        
        const fetchMinLat = centerLat - (heightDeg * 2);
        const fetchMaxLat = centerLat + (heightDeg * 2);
        const fetchMinLon = centerLon - (widthDeg * 4);
        const fetchMaxLon = centerLon + (widthDeg * 4);
        
        const response = await fetch(`/api/fetch-image?minLat=${fetchMinLat}&maxLat=${fetchMaxLat}&minLon=${fetchMinLon}&maxLon=${fetchMaxLon}&dim=1792x896`);
        
        if (!response.ok) throw new Error(`Failed to fetch imagery: ${response.statusText}`);
        
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        setImageUrl(data.image_base64);
      } catch (err: any) {
        setError(err.message || 'An unknown error occurred');
      } finally {
        setLoadingImage(false);
      }
    };

    fetchSatelliteImage();
  }, [node.id, customImageUrl]);

  return (
    <div className="leaf-node-container glass-panel dashboard-layout">
      {loadingImage && (
        <div className="loading-state" style={{ padding: '2rem' }}>
          <div className="spinner"></div>
          <p>Fetching 1792x896 composite from Google Earth Engine...</p>
        </div>
      )}

      {error && <div className="error-state" style={{ padding: '2rem' }}><p>Error: {error}</p></div>}

      {imageUrl && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <ModelDashboard node={node} customImageUrl={customImageUrl} imageUrl={imageUrl} modelType="finetuned" />
          <ModelDashboard node={node} customImageUrl={customImageUrl} imageUrl={imageUrl} modelType="base" />
        </div>
      )}
    </div>
  );
};
