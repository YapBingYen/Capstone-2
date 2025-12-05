import json
import os
import random
from math import cos, sin, pi

CACHE_DIR = r"D:\Cursor AI projects\Capstone2.1"
METADATA_CACHE = os.path.join(CACHE_DIR, 'cat_metadata_cache.json')
FOUND_CATS_METADATA = os.path.join(CACHE_DIR, 'found_cats_metadata.json')

# Neighborhoods across Kuala Lumpur & Klang Valley with approximate centers and radius (km)
AREAS = [
    {"name": "Kuala Lumpur City Centre", "lat": 3.1579, "lng": 101.7123, "radius_km": 3.0},
    {"name": "Bangsar", "lat": 3.1285, "lng": 101.6780, "radius_km": 3.0},
    {"name": "Mont Kiara", "lat": 3.1673, "lng": 101.6554, "radius_km": 2.5},
    {"name": "Cheras", "lat": 3.0803, "lng": 101.7446, "radius_km": 4.0},
    {"name": "Setapak", "lat": 3.2113, "lng": 101.7180, "radius_km": 3.0},
    {"name": "TTDI", "lat": 3.1398, "lng": 101.6246, "radius_km": 2.5},
    {"name": "Kepong", "lat": 3.2183, "lng": 101.6444, "radius_km": 4.0},
    {"name": "Wangsa Maju", "lat": 3.2055, "lng": 101.7372, "radius_km": 3.0},
    {"name": "Petaling Jaya", "lat": 3.1073, "lng": 101.6067, "radius_km": 5.0},
    {"name": "Subang Jaya", "lat": 3.0649, "lng": 101.5851, "radius_km": 5.0},
    {"name": "Shah Alam", "lat": 3.0733, "lng": 101.5185, "radius_km": 6.0},
    {"name": "Setia Alam", "lat": 3.1000, "lng": 101.4500, "radius_km": 4.0},
    {"name": "Bukit Jelutong", "lat": 3.1064, "lng": 101.5325, "radius_km": 3.0},
    {"name": "Klang", "lat": 3.0433, "lng": 101.4455, "radius_km": 6.0},
    {"name": "Kota Kemuning", "lat": 3.0038, "lng": 101.5357, "radius_km": 4.0},
    {"name": "Puchong", "lat": 3.0123, "lng": 101.6197, "radius_km": 5.0},
]

def random_point(lat, lng, radius_km):
    # Uniform random point within circle of given radius
    r = radius_km / 111.0  # degrees per km (~111 km per degree)
    u = random.random()
    v = random.random()
    w = r * (u ** 0.5)
    t = 2 * pi * v
    dlat = w * cos(t)
    dlng = w * sin(t) / max(0.001, cos(lat * pi / 180.0))
    return lat + dlat, lng + dlng

def main():
    if not os.path.exists(METADATA_CACHE):
        print('❌ cat_metadata_cache.json not found')
        return

    with open(METADATA_CACHE, 'r') as f:
        metadata = json.load(f)

    # Randomize found cats only
    changed = 0
    for cid, data in metadata.items():
        if data.get('status') == 'found':
            area = random.choice(AREAS)
            lat, lng = random_point(area['lat'], area['lng'], area['radius_km'])
            data['lat'] = round(lat, 6)
            data['lng'] = round(lng, 6)
            data['location'] = f"{area['name']}, Malaysia"
            changed += 1

    with open(METADATA_CACHE, 'w') as f:
        json.dump(metadata, f)

    # Update found subset file
    found_subset = {cid: data for cid, data in metadata.items() if data.get('status') == 'found'}
    with open(FOUND_CATS_METADATA, 'w') as f:
        json.dump(found_subset, f, indent=2)

    # Simple distribution summary
    counts = {}
    for v in found_subset.values():
        loc = v.get('location', '')
        counts[loc] = counts.get(loc, 0) + 1
    print(f"✅ Randomized {changed} found cats across {len(counts)} areas")
    for k, c in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f" - {k}: {c}")

if __name__ == '__main__':
    main()
