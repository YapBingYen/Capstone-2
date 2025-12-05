import json
import os

CACHE_DIR = r"D:\Cursor AI projects\Capstone2.1"
METADATA_CACHE = os.path.join(CACHE_DIR, 'cat_metadata_cache.json')
FOUND_CATS_METADATA = os.path.join(CACHE_DIR, 'found_cats_metadata.json')

def main():
    if not os.path.exists(METADATA_CACHE):
        print('❌ cat_metadata_cache.json not found')
        return

    with open(METADATA_CACHE, 'r') as f:
        metadata = json.load(f)

    changed = 0
    for cid, data in metadata.items():
        if data.get('status') == 'found':
            data['status'] = 'dataset'
            data.pop('lat', None)
            data.pop('lng', None)
            changed += 1

    with open(METADATA_CACHE, 'w') as f:
        json.dump(metadata, f)

    # Clear found subset
    with open(FOUND_CATS_METADATA, 'w') as f:
        json.dump({}, f, indent=2)

    print(f"✅ Hidden {changed} found cats (set status to 'dataset' and cleared coordinates)")

if __name__ == '__main__':
    main()
