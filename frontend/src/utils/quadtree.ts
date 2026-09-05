export interface BoundingBox {
    minLat: number;
    maxLat: number;
    minLon: number;
    maxLon: number;
}

export interface Quadrant {
    id: string;
    label: string;
    bbox: BoundingBox;
}

/**
 * Divides a parent bounding box into 4 equal child quadrants.
 */
export function getQuadrants(parent: BoundingBox, parentId: string): Quadrant[] {
    const midLat = (parent.minLat + parent.maxLat) / 2;
    const midLon = (parent.minLon + parent.maxLon) / 2;

    const getCenter = (minLat: number, maxLat: number, minLon: number, maxLon: number) => {
        return `${((minLat + maxLat) / 2).toFixed(3)}°N, ${((minLon + maxLon) / 2).toFixed(3)}°E`;
    };

    return [
        {
            id: `${parentId}_NW`,
            label: getCenter(midLat, parent.maxLat, parent.minLon, midLon),
            bbox: { minLat: midLat, maxLat: parent.maxLat, minLon: parent.minLon, maxLon: midLon }
        },
        {
            id: `${parentId}_NE`,
            label: getCenter(midLat, parent.maxLat, midLon, parent.maxLon),
            bbox: { minLat: midLat, maxLat: parent.maxLat, minLon: midLon, maxLon: parent.maxLon }
        },
        {
            id: `${parentId}_SW`,
            label: getCenter(parent.minLat, midLat, parent.minLon, midLon),
            bbox: { minLat: parent.minLat, maxLat: midLat, minLon: parent.minLon, maxLon: midLon }
        },
        {
            id: `${parentId}_SE`,
            label: getCenter(parent.minLat, midLat, midLon, parent.maxLon),
            bbox: { minLat: parent.minLat, maxLat: midLat, minLon: midLon, maxLon: parent.maxLon }
        }
    ];
}

/**
 * Mathematically calculates the quadtree path from the Root down to Level 8 for a given coordinate.
 */
export function findPathToCoordinate(lat: number, lon: number): Quadrant[] {
    // Check if within bounds
    if (lat < ROOT_BBOX.minLat || lat > ROOT_BBOX.maxLat || lon < ROOT_BBOX.minLon || lon > ROOT_BBOX.maxLon) {
        throw new Error("Coordinate is outside the bounds of the Root Region.");
    }

    const history: Quadrant[] = [
        { 
            id: 'ROOT', 
            label: `${((ROOT_BBOX.minLat + ROOT_BBOX.maxLat) / 2).toFixed(3)}°N, ${((ROOT_BBOX.minLon + ROOT_BBOX.maxLon) / 2).toFixed(3)}°E`, 
            bbox: ROOT_BBOX 
        }
    ];

    let currentBbox = ROOT_BBOX;
    let currentId = 'ROOT';

    // Depth 13 maintains the exact same leaf node resolution (0.00585 deg) as the old Depth 10
    for (let depth = 0; depth < 13; depth++) {
        const quadrants = getQuadrants(currentBbox, currentId);
        
        let found = false;
        for (const q of quadrants) {
            if (lat >= q.bbox.minLat && lat <= q.bbox.maxLat && lon >= q.bbox.minLon && lon <= q.bbox.maxLon) {
                history.push(q);
                currentBbox = q.bbox;
                currentId = q.id;
                found = true;
                break;
            }
        }
        if (!found) throw new Error("Could not trace coordinate into Quadtree.");
    }
    return history;
}

// Root BBox expanded to 48x48 degrees to cover the entire Indian subcontinent and beyond
// (Depth 13 is exactly ~640m x 640m EuroSAT native scale)
export const ROOT_BBOX: BoundingBox = {
    minLat: 0.0,
    maxLat: 48.0,
    minLon: 60.0,
    maxLon: 108.0
};
