import os
import argparse
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from sklearn.cluster import KMeans

IMG_SIZE = 224
MAX_IMAGES_PER_CAT = 20
PROTOTYPE_COUNT = 3
USE_TTA = True
FUSION_ALPHA = 0.7
FUSION_BETA = 0.15
FUSION_GAMMA = 0.15

def get_haar():
    local = os.path.join(os.path.dirname(__file__), '..', 'static', 'models', 'haarcascade_frontalcatface.xml')
    if os.path.exists(local):
        return cv2.CascadeClassifier(local)
    try:
        cat_xml = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalcatface.xml')
        if os.path.exists(cat_xml):
            return cv2.CascadeClassifier(cat_xml)
    except Exception:
        pass
    return cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))

def center_crop(arr):
    h, w = arr.shape[:2]
    m = min(h, w)
    sh = (h - m) // 2
    sw = (w - m) // 2
    return arr[sh:sh+m, sw:sw+m]

def preprocess_image(path, haar=None):
    img = cv2.imread(path)
    if img is None:
        return None, None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cropped = None
    if haar is not None and not haar.empty():
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                p = 20
                xs = max(0, x - p)
                ys = max(0, y - p)
                xe = min(rgb.shape[1], x + w + p)
                ye = min(rgb.shape[0], y + h + p)
                cropped = rgb[ys:ye, xs:xe]
        except Exception:
            pass
    if cropped is None:
        cropped = center_crop(rgb)
    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
    pil = Image.fromarray(resized)
    arr = tf.keras.utils.img_to_array(pil)
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    batch = np.expand_dims(arr, axis=0)
    return batch, pil

def build_model(weights):
    base = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
        include_top=False, weights=weights, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg'
    )
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inp, training=False)
    x = tf.keras.layers.Lambda(lambda t: t / (tf.norm(t, axis=-1, keepdims=True) + 1e-8))(x)
    return tf.keras.Model(inp, x)

def embed_batch(model, batch):
    e = model(batch)
    if hasattr(e, 'numpy'):
        e = e.numpy()
    e = e.flatten()
    e = e / (np.linalg.norm(e) + 1e-8)
    return e

def extract_embedding(model, path, haar):
    batch, pil = preprocess_image(path, haar)
    if batch is None or pil is None:
        return None, None
    if not USE_TTA:
        return embed_batch(model, batch), pil
    base = np.squeeze(batch, axis=0)
    variants = [base]
    variants.append(np.flip(base, axis=1))
    variants.append(np.clip(base * 1.10, -1.0, 1.0))
    variants.append(np.clip((base - np.mean(base)) * 1.10 + np.mean(base), -1.0, 1.0))
    embs = []
    for v in variants:
        b = np.expand_dims(v, axis=0)
        embs.append(embed_batch(model, b))
    emb = np.mean(np.stack(embs, axis=0), axis=0)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb, pil

def compute_color_feature(pil_img):
    try:
        arr = np.array(pil_img)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        hist_h = np.histogram(h, bins=16, range=(0, 180))[0].astype(np.float32)
        hist_s = np.histogram(s, bins=16, range=(0, 255))[0].astype(np.float32)
        hist_v = np.histogram(v, bins=16, range=(0, 255))[0].astype(np.float32)
        feat = np.concatenate([hist_h, hist_s, hist_v])
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        return feat
    except Exception:
        return None

def orb_similarity(pil1, pil2):
    try:
        a1 = np.array(pil1)
        a2 = np.array(pil2)
        g1 = cv2.cvtColor(a1, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(a2, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=500)
        _, d1 = orb.detectAndCompute(g1, None)
        _, d2 = orb.detectAndCompute(g2, None)
        if d1 is None or d2 is None:
            return 0.0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(d1, d2, k=2)
        good = 0
        for m in matches:
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                good += 1
        denom = max(1, min(len(d1), len(d2)))
        return float(good / denom)
    except Exception:
        return 0.0

def build_prototypes(model, dataset_path, haar):
    cat_to_imgs = {}
    for root, _, files in os.walk(dataset_path):
        imgs = [os.path.join(root, f) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            cat_id = os.path.basename(root)
            cat_to_imgs[cat_id] = imgs[:MAX_IMAGES_PER_CAT]
    cat_prototypes = {}
    for cid, paths in cat_to_imgs.items():
        embs = []
        pimgs = []
        for p in paths:
            e, pil = extract_embedding(model, p, haar)
            if e is not None and pil is not None:
                embs.append(e)
                pimgs.append((p, pil))
        if not embs:
            continue
        E = np.stack(embs, axis=0)
        n = min(PROTOTYPE_COUNT, max(1, len(E)))
        plist = []
        if len(E) >= max(2, PROTOTYPE_COUNT):
            try:
                km = KMeans(n_clusters=n, n_init='auto', random_state=42)
                labels = km.fit_predict(E)
                centers = km.cluster_centers_
                for j in range(n):
                    c = centers[j]
                    c = c / (np.linalg.norm(c) + 1e-8)
                    idxs = [i for i, lb in enumerate(labels) if lb == j]
                    rp, rpil = pimgs[idxs[0]] if idxs else pimgs[0]
                    color = compute_color_feature(rpil)
                    plist.append({'id': f'{cid}#p{j+1}', 'emb': c, 'color': color, 'rep_path': rp, 'rep_pil': rpil, 'base_id': cid})
            except Exception:
                rp, rpil = pimgs[0]
                c = np.mean(E, axis=0)
                c = c / (np.linalg.norm(c) + 1e-8)
                color = compute_color_feature(rpil)
                plist.append({'id': f'{cid}#p1', 'emb': c, 'color': color, 'rep_path': rp, 'rep_pil': rpil, 'base_id': cid})
        else:
            rp, rpil = pimgs[0]
            c = np.mean(E, axis=0)
            c = c / (np.linalg.norm(c) + 1e-8)
            color = compute_color_feature(rpil)
            plist.append({'id': f'{cid}#p1', 'emb': c, 'color': color, 'rep_path': rp, 'rep_pil': rpil, 'base_id': cid})
        cat_prototypes[cid] = plist
    ids = []
    vecs = []
    for cid, plist in cat_prototypes.items():
        for it in plist:
            ids.append(it['id'])
            vecs.append(it['emb'])
    proto_matrix = np.stack(vecs, axis=0) if vecs else None
    return cat_prototypes, proto_matrix, ids

def match_query(model, query_path, cat_prototypes, proto_matrix, proto_ids, haar, top_k=10):
    q_emb, q_pil = extract_embedding(model, query_path, haar)
    if q_emb is None:
        return []
    q_color = compute_color_feature(q_pil)
    fused = []
    for i, pid in enumerate(proto_ids):
        it = None
        base_id = pid.split('#')[0]
        for item in cat_prototypes.get(base_id, []):
            if item['id'] == pid:
                it = item
                break
        cos = float(np.dot(proto_matrix[i], q_emb))
        color_sim = 0.0 if it is None else float(np.dot(q_color, it['color'])) if q_color is not None and it['color'] is not None else 0.0
        orb_sim = 0.0 if it is None else orb_similarity(q_pil, it['rep_pil'])
        score = FUSION_ALPHA * cos + FUSION_BETA * color_sim + FUSION_GAMMA * orb_sim
        l2 = float(np.sum((proto_matrix[i] - q_emb) ** 2))
        fused.append((score, base_id, cos, l2))
    best_by_cat = {}
    for score, cid, cos, l2 in fused:
        if cid not in best_by_cat or score > best_by_cat[cid][0]:
            best_by_cat[cid] = (score, cos, l2)
    ordered = [(cid, v[0], v[1], v[2]) for cid, v in best_by_cat.items()]
    ordered.sort(key=lambda x: -x[1])
    return ordered[:top_k]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', choices=['none', 'imagenet'], default='imagenet')
    default_dataset = os.environ.get('DATASET_PATH', r"D:\Cursor AI projects\Capstone2.1\dataset_individuals")
    parser.add_argument('--dataset', required=False, default=default_dataset)
    parser.add_argument('--query', nargs='+', required=True)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--limit_cats', type=int, default=150)
    parser.add_argument('--max_per_cat', type=int, default=MAX_IMAGES_PER_CAT)
    args = parser.parse_args()
    w = None if args.weights == 'none' else 'imagenet'
    model = build_model(w)
    haar = get_haar()
    if not args.dataset or not os.path.exists(args.dataset):
        print('Dataset not found')
        return
    # Build prototypes with limits
    max_per_cat_local = int(args.max_per_cat)
    # Limit cats by lexicographic order
    cat_dirs = []
    for root, dirs, files in os.walk(args.dataset):
        # Only consider immediate subfolders that contain images
        imgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            cat_dirs.append(root)
    cat_dirs = sorted(cat_dirs)[:int(args.limit_cats)]
    # Temporarily create a filtered view
    def build_filtered_prototypes():
        cat_to_imgs = {}
        for root in cat_dirs:
            files = [f for f in os.listdir(root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            imgs = [os.path.join(root, f) for f in files]
            if imgs:
                cid = os.path.basename(root)
                cat_to_imgs[cid] = imgs[:max_per_cat_local]
        # Build like build_prototypes but with known set
        cat_prototypes = {}
        processed = 0
        total = len(cat_to_imgs)
        for cid, paths in cat_to_imgs.items():
            embs = []
            pimgs = []
            for p in paths:
                e, pil = extract_embedding(model, p, haar)
                if e is not None and pil is not None:
                    embs.append(e)
                    pimgs.append((p, pil))
            if not embs:
                continue
            E = np.stack(embs, axis=0)
            n = min(PROTOTYPE_COUNT, max(1, len(E)))
            plist = []
            if len(E) >= max(2, PROTOTYPE_COUNT):
                try:
                    km = KMeans(n_clusters=n, n_init='auto', random_state=42)
                    labels = km.fit_predict(E)
                    centers = km.cluster_centers_
                    for j in range(n):
                        c = centers[j]
                        c = c / (np.linalg.norm(c) + 1e-8)
                        idxs = [i for i, lb in enumerate(labels) if lb == j]
                        rp, rpil = pimgs[idxs[0]] if idxs else pimgs[0]
                        color = compute_color_feature(rpil)
                        plist.append({'id': f'{cid}#p{j+1}', 'emb': c, 'color': color, 'rep_path': rp, 'rep_pil': rpil, 'base_id': cid})
                except Exception:
                    rp, rpil = pimgs[0]
                    c = np.mean(E, axis=0)
                    c = c / (np.linalg.norm(c) + 1e-8)
                    color = compute_color_feature(rpil)
                    plist.append({'id': f'{cid}#p1', 'emb': c, 'color': color, 'rep_path': rp, 'rep_pil': rpil, 'base_id': cid})
            else:
                rp, rpil = pimgs[0]
                c = np.mean(E, axis=0)
                c = c / (np.linalg.norm(c) + 1e-8)
                color = compute_color_feature(rpil)
                plist.append({'id': f'{cid}#p1', 'emb': c, 'color': color, 'rep_path': rp, 'rep_pil': rpil, 'base_id': cid})
            cat_prototypes[cid] = plist
            processed += 1
            if processed % 25 == 0 or processed == total:
                print(f'Built prototypes for {processed}/{total} cats...')
        ids = []
        vecs = []
        for cid, plist in cat_prototypes.items():
            for it in plist:
                ids.append(it['id'])
                vecs.append(it['emb'])
        proto_matrix = np.stack(vecs, axis=0) if vecs else None
        return cat_prototypes, proto_matrix, ids
    cat_prototypes, proto_matrix, proto_ids = build_filtered_prototypes()
    if proto_matrix is None or not proto_ids:
        print('No prototypes built')
        return
    for q in args.query:
        if not os.path.exists(q):
            print(f'{q} not found')
            continue
        res = match_query(model, q, cat_prototypes, proto_matrix, proto_ids, haar, top_k=args.topk)
        print(f'Query: {q}')
        for i, (cid, score, cos, l2) in enumerate(res, start=1):
            print(f'{i}. {cid}  score={score*100:.1f}%  cosine={cos:.4f}  l2={l2:.4f}')

if __name__ == '__main__':
    main()
