import React from 'react';
import { type Quadrant } from '../utils/quadtree';

interface Props {
  quadrants: Quadrant[];
  onSelect: (quadrant: Quadrant) => void;
  currentDepth: number;
}

export const QuadtreeNavigator: React.FC<Props> = ({ quadrants, onSelect }) => {
  return (
    <div className="quadtree-grid">
      {quadrants.map((quad) => (
        <div 
          key={quad.id} 
          className="quadrant glass-panel glass-panel-interactive"
          onClick={() => onSelect(quad)}
        >
          <div className="quadrant-label">{quad.label}</div>
          <div className="quadrant-coords">
            Lat: {quad.bbox.minLat.toFixed(4)} to {quad.bbox.maxLat.toFixed(4)}
            <br />
            Lon: {quad.bbox.minLon.toFixed(4)} to {quad.bbox.maxLon.toFixed(4)}
          </div>
        </div>
      ))}
    </div>
  );
};
