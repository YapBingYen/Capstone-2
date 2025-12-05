import json
import os
import numpy as np

CACHE_DIR = r"D:\Cursor AI projects\Capstone2.1"
METADATA_CACHE = os.path.join(CACHE_DIR, 'cat_metadata_cache.json')
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, 'cat_embeddings_cache.npy')
FOUND_CATS_METADATA = os.path.join(CACHE_DIR, 'found_cats_metadata.json')

def main():
    if not os.path.exists(FOUND_CATS_METADATA):
        print('No found_cats_metadata.json present')
        return

    with open(FOUND_CATS_METADATA, 'r') as f:
        found = json.load(f)

    if not found:
        print('No found cats to remove')
        return

    found_ids = list(found.keys())
    print(f'Removing found cats: {found_ids}')

    # Delete image files
    for cid, data in found.items():
        img_path = data.get('image_path')
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
                print(f'Deleted image: {img_path}')
            except Exception as e:
                print(f'Failed to delete {img_path}: {e}')

    # Update metadata cache
    if os.path.exists(METADATA_CACHE):
        with open(METADATA_CACHE, 'r') as f:
            meta = json.load(f)
        for cid in found_ids:
            if cid in meta:
                del meta[cid]
                print(f'Removed {cid} from metadata cache')
        with open(METADATA_CACHE, 'w') as f:
            json.dump(meta, f)

    # Update embeddings cache
    if os.path.exists(EMBEDDINGS_CACHE):
        try:
            emb = np.load(EMBEDDINGS_CACHE, allow_pickle=True).item()
            for cid in found_ids:
                if cid in emb:
                    del emb[cid]
                    print(f'Removed {cid} from embeddings cache')
            np.save(EMBEDDINGS_CACHE, np.array(emb, dtype=object))
        except Exception as e:
            print(f'Failed to update embeddings cache: {e}')

    # Clear found cats metadata
    with open(FOUND_CATS_METADATA, 'w') as f:
        json.dump({}, f, indent=2)
    print('Cleared found_cats_metadata.json')

if __name__ == '__main__':
    main()
