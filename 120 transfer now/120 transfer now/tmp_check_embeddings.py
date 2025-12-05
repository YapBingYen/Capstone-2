import numpy as np
cache = np.load(" cat_embeddings_cache.npy\, allow_pickle=True).item()
print('cats', len(cache))
first_key = next(iter(cache))
emb = cache[first_key]
print('key', first_key)
print('type', type(emb))
arr = np.asarray(emb)
print('shape', arr.shape)
print('dtype', arr.dtype)
print('norm', np.linalg.norm(arr))
