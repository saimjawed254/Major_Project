import { useState } from 'react';
import { type Quadrant, ROOT_BBOX, getQuadrants, findPathToCoordinate } from './utils/quadtree';
import { QuadtreeNavigator } from './components/QuadtreeNavigator';
import { LeafNodeView } from './components/LeafNodeView';

function App() {
  const [history, setHistory] = useState<Quadrant[]>([
    { id: 'ROOT', label: '25.000°N, 85.500°E', bbox: ROOT_BBOX }
  ]);
  const [searchQuery, setSearchQuery] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const [customImageUrl, setCustomImageUrl] = useState<string | null>(null);



  const clearCustomImage = () => {
    setCustomImageUrl(null);
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg('');
    try {
      const parts = searchQuery.split(',').map(s => parseFloat(s.trim()));
      if (parts.length !== 2 || isNaN(parts[0]) || isNaN(parts[1])) {
        throw new Error("Invalid format. Use: 26.500, 85.500");
      }
      const newHistory = findPathToCoordinate(parts[0], parts[1]);
      setHistory(newHistory);
    } catch (err: any) {
      setErrorMsg(err.message);
    }
  };
  
  const currentDepth = history.length - 1;
  const currentNode = history[currentDepth];
  
  // Calculate child quadrants for the current node
  const childQuadrants = currentDepth < 13 
    ? getQuadrants(currentNode.bbox, currentNode.id)
    : [];

  const handleSelectQuadrant = (quadrant: Quadrant) => {
    if (currentDepth < 13) {
      setHistory([...history, quadrant]);
    }
  };

  const handleNavigateToBreadcrumb = (index: number) => {
    setHistory(history.slice(0, index + 1));
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>Virtual Quadtree Navigator</h1>
        <p>Level {currentDepth} / 13 • Distributed AI Earth Observation</p>
      </div>

      <div className="search-container glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          <form onSubmit={handleSearch} className="search-form" style={{ flex: 1 }}>
            <input 
              type="text" 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={`Paste coordinates to jump (e.g., ${((ROOT_BBOX.minLat + ROOT_BBOX.maxLat) / 2).toFixed(3)}, ${((ROOT_BBOX.minLon + ROOT_BBOX.maxLon) / 2).toFixed(3)})`} 
              className="search-input"
            />
            <button type="submit" className="search-button">Jump to Node</button>
          </form>

        </div>
        {errorMsg && <div className="search-error">{errorMsg}</div>}
      </div>

      {customImageUrl ? (
        <div className="custom-image-mode">
          <div style={{ padding: '10px 20px', background: 'rgba(239, 68, 68, 0.2)', color: '#fca5a5', borderRadius: '8px', marginBottom: '15px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span><strong>Custom Image Mode:</strong> Geographic features are disabled. Firestore saving is disabled.</span>
            <button onClick={clearCustomImage} style={{ padding: '5px 10px', background: 'rgba(255,255,255,0.1)', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>Exit Custom Mode</button>
          </div>
          {/* Render LeafNodeView with dummy root node but passing customImageUrl */}
          <LeafNodeView node={{ id: 'CUSTOM', label: 'Custom Upload', bbox: ROOT_BBOX }} customImageUrl={customImageUrl} />
        </div>
      ) : (
        <>
          <div className="breadcrumbs glass-panel">
            {history.map((node, index) => (
              <div key={node.id} className="breadcrumb-item">
                <button 
                  className={`breadcrumb-btn ${index === currentDepth ? 'breadcrumb-active' : ''}`}
                  onClick={() => handleNavigateToBreadcrumb(index)}
                >
                  {index === 0 ? 'Home' : node.label}
                </button>
                {index < history.length - 1 && (
                  <span className="breadcrumb-separator">/</span>
                )}
              </div>
            ))}
          </div>

          {currentDepth < 12 ? (
            <QuadtreeNavigator 
              quadrants={childQuadrants} 
              onSelect={handleSelectQuadrant}
              currentDepth={currentDepth}
            />
          ) : (
            <LeafNodeView node={currentNode} />
          )}
        </>
      )}
    </div>
  );
}

export default App;
