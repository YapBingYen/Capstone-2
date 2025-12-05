import json
import os
from datetime import datetime

import numpy as np

CACHE_DIR = r"D:\Cursor AI projects\Capstone2.1"
METADATA_CACHE = os.path.join(CACHE_DIR, 'cat_metadata_cache.json')
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, 'cat_embeddings_cache.npy')
FOUND_CATS_METADATA = os.path.join(CACHE_DIR, 'found_cats_metadata.json')

DEFAULT_LAT = 3.1390
DEFAULT_LNG = 101.6869
DEFAULT_LOCATION = 'Kuala Lumpur, Malaysia'

def main():
    # Load metadata
    if not os.path.exists(METADATA_CACHE):
        print('❌ cat_metadata_cache.json not found')
        return

    with open(METADATA_CACHE, 'r') as f:
        metadata = json.load(f)

    # Ensure embeddings exist for consistency (not strictly required to mark found)
    if os.path.exists(EMBEDDINGS_CACHE):
        try:
            embeddings = np.load(EMBEDDINGS_CACHE, allow_pickle=True).item()
        except Exception:
            embeddings = {}
    else:
        embeddings = {}

    count_before = sum(1 for v in metadata.values() if v.get('status') == 'found')
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_date = datetime.now().strftime("%Y-%m-%d")

    # Mark all cats as found, add defaults for map
    updated = 0
    for cat_id, info in metadata.items():
        # Skip already marked found cats to avoid overwriting user-provided details
        if info.get('status') == 'found':
            continue

        info['status'] = 'found'
        info.setdefault('location', DEFAULT_LOCATION)
        info.setdefault('date_found', today_date)
        info.setdefault('upload_time', now_time)
        info.setdefault('description', '')
        info.setdefault('contact_info', '')
        # Add coordinates for map display
        info.setdefault('lat', DEFAULT_LAT)
        info.setdefault('lng', DEFAULT_LNG)
        updated += 1

    # Write back main metadata
    with open(METADATA_CACHE, 'w') as f:
        json.dump(metadata, f)

    # Write found cats subset metadata
    found_subset = {cid: data for cid, data in metadata.items() if data.get('status') == 'found'}
    with open(FOUND_CATS_METADATA, 'w') as f:
        json.dump(found_subset, f, indent=2)

    count_after = sum(1 for v in metadata.values() if v.get('status') == 'found')
    print(f"✅ Marked {updated} cats as found. Found cats total: {count_before} → {count_after}")

if __name__ == '__main__':
    main()
