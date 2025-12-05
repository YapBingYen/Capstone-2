# Pet ID Malaysia — Full Project Documentation

Version: 2025-11-28

---

## 1. Overview

- Purpose: Identify individual cats using a triplet-loss embedding model served via a Flask web application.
- Core model: EfficientNetV2-S backbone, 512-dim L2-normalized embeddings.
- Matching: Top-K using L2 distance on normalized embeddings (cosine shown for convenience).
- Data: Per-cat folders, centroid embeddings cached for fast lookup.

## 2. Architecture

- Backend: `120 transfer now/120 transfer now/app.py` (Flask)
  - Model loading priority: SavedModel → `.keras` → `.h5` → weights (from `D:\Cursor AI projects\Capstone2.1\models`).
  - Preprocessing: `keras.applications.efficientnet_v2.preprocess_input` with 224×224 RGB inputs.
  - Embedding extraction: `self.embedding_model.predict(...)` → flatten → L2-normalize.
  - Matching: L2 distance on normalized vectors; cosine = dot product for reporting.
  - Embedding store: Per-cat centroid computed from images; caches saved in project root.
  - Maintenance APIs: `/api/health`, `/api/reindex`, `/api/add`.
- Frontend: Bootstrap 5 templates (upload, results, found cat pages).

## 3. Key Files & Code References

- `app.py`:
  - Model load guard and shape logging (only when non-null): `120 transfer now/120 transfer now/app.py:95–99`.
  - EfficientNetV2 preprocessing pipeline: `120 transfer now/120 transfer now/app.py:122–170`.
  - Embedding extraction and normalization: `120 transfer now/120 transfer now/app.py:189–205`.
  - L2 Top-K matching with cosine reporting: `120 transfer now/120 transfer now/app.py:272–297`.
  - Caches saved/loaded: `120 transfer now/120 transfer now/app.py:217–233`, `120 transfer now/120 transfer now/app.py:259–266`, `120 transfer now/120 transfer now/app.py:470`, `120 transfer now/120 transfer now/app.py:674–677`.
  - Maintenance endpoints: `120 transfer now/120 transfer now/app.py:619–677`.
- Templates:
  - `templates/index.html`: upload with spinner.
  - `templates/results.html`: Top-K cards, cosine badges.
  - `templates/found_cat*.html`: found-cat upload & success.
- Training:
  - `train_cat_identifier_v2.py`: EfficientNetV2-S + triplet loss; TF-compatible augmentation via `tf.image.adjust_*`.
- Testing:
  - `test_cat_identifier_v2.py`: EfficientNetV2-S backbone, TTA with `tf.image.adjust_*`, model guards, visualization safety.

## 4. Environment & Compatibility

- Python: 3.9.
- TensorFlow: 2.10.0 (built against NumPy 1.x).
- NumPy: 1.23.5 (avoid NumPy 2.x to prevent ABI errors).
- OpenCV: 4.7.0.72.
- Flask, Pillow, tqdm, matplotlib, scikit-learn.

## 5. Setup & Runbook

1) Create & activate venv (Windows):

- `py -3.9 -m venv .venv39`
- `.venv39\Scripts\Activate.ps1`

2) Install deps:

- `python -m pip install -U pip`
- `python -m pip install tensorflow==2.10.0 numpy==1.23.5 opencv-python==4.7.0.72 pillow flask tqdm matplotlib scikit-learn`

3) Start app:

- `python "D:\Cursor AI projects\Capstone2.1\120 transfer now\120 transfer now\app.py"`

4) Open URLs:

- App: `http://localhost:5000/`
- Health: `http://localhost:5000/api/health`

## 6. Data & Caches

- Dataset path used by app:
  - `D:\Cursor AI projects\Capstone2.1\120 transfer now\120 transfer now\cat_individuals_dataset\dataset_individuals_cropped\cat_individuals_dataset`.
- Caches (project root):
  - `cat_embeddings_cache.npy` (object array of dict to satisfy type checks).
  - `cat_metadata_cache.json`.

## 7. API Endpoints

- `GET /api/health` → `{ model_loaded, embeddings_count, dataset_path, img_size }`.
- `POST /api/reindex` → rebuild centroid embeddings from dataset.
- `POST /api/add` → form fields: `file`, `cat_id`; adds a single image embedding and metadata.

## 8. Frontend UX

- Upload page: drag-and-drop, spinner feedback.
- Results page: responsive card grid; cosine-based badges (High/Possible/Low match).
- Found-cat flow: upload, metadata capture, success page; entries saved to caches.

## 9. Model & Matching Details

- Embeddings: 512-dim; normalized (division by L2 norm + epsilon).
- Matching: L2 distance on normalized embeddings; smaller is closer.
- Cosine similarity: dot product of normalized vectors (for display/badges).

## 10. Training (EfficientNetV2‑S)

- Triplet loss with margin; concatenated triplets via wrapper model.
- Augmentation: TF `adjust_brightness/adjust_contrast` for TF 2.10 compatibility.
- Saving: weights `.h5` and SavedModel directory for robust loading.

## 11. Testing (EfficientNetV2‑S)

- TTA: original, horizontal flip, brightness/contrast variants.
- Guards: ensure `model is not None` before prediction; visualization manager may be `None`.

## 12. Diagnostics & Stability Fixes

- IDE diagnostics resolved by:
  - Guarding shape logging: `120 transfer now/120 transfer now/app.py:95–99`.
  - Safe CV2 attribute access via `getattr`: `120 transfer now/120 transfer now/app.py:116–170`.
  - Removing `verbose=0` in predict calls.
  - Using object arrays for `np.save` where dicts are stored.
  - Defaults for `secure_filename(file.filename or "...")` to handle `None`.
- Lint/IDE config: `.vscode/settings.json` bound to `.venv39`; `pyrightconfig.json` set to Python 3.9, `typeCheckingMode=basic`.

## 13. Operational Notes

- First run may be slower due to cache building; subsequent runs rely on caches.
- GPU usage: TensorFlow auto-detects CUDA; enable memory growth if needed.
- Dataset hygiene: ensure per-cat folder structure and valid image formats (`.jpg/.jpeg/.png`).

## 14. Troubleshooting

- NumPy/TF ABI error (`_ARRAY_API not found`):
  - Pin NumPy to 1.23.5; reinstall OpenCV 4.7.0.72.
- Missing modules: activate `.venv39` before running.
- Server port conflicts: stop other servers or change port in `app.py`.
- Low match quality: verify dataset cropping, consider adding more images per cat.

## 15. Change Log (2025‑11)

- Migrated inference to EfficientNetV2‑S with 512-dim embeddings.
- Implemented Flask web app with uploads, results, and found-cat flows.
- Added maintenance APIs (`/api/health`, `/api/reindex`, `/api/add`).
- Built centroid-based embedding database with caching.
- Aligned test script to V2 backbone; TF-compatible augmentation.
- Environment pinned: Python 3.9, TF 2.10, NumPy 1.23.5, OpenCV 4.7.0.72.
- Diagnostics resolved: guards, CV2 attrs, secure filename defaults, np.save object arrays.
- UX fix: Single-click works for upload button and drag area by clearing the hidden file input value before opening the picker; ensures `change` fires when reselecting the same file.
  - Files: `120 transfer now/120 transfer now/templates/index.html:166, 263, 423–426, 495–500`; `120 transfer now/120 transfer now/templates/found_cat.html:117, 334–336, 412–416`.

## 16. Future Enhancements

- Configurable threshold in app UI; filter by cosine/L2.
- Persistent database (SQLite/PostgreSQL) for embeddings/metadata.
- Async task for reindexing; progress indicator.
- Batch upload and bulk matching tools.

## 17. Contacts & Support

- Health check: `http://localhost:5000/api/health`.
- Verify model paths under `D\Cursor AI projects\Capstone2.1\models`.
- Ensure `.venv39` is active for all CLI commands.

## 18. Progress Log (Continuous)

- 2025-11-28 — Web App Integration & EfficientNetV2-S Migration
  - Implemented Flask backend (`120 transfer now/120 transfer now/app.py`) with model load priority: SavedModel → `.keras` → `.h5` → weights.
  - Switched preprocessing to `efficientnet_v2.preprocess_input` at 224×224; set `IMG_SIZE=224`.
  - Normalized 512-dim embeddings; Top-K matching by L2 distance; cosine similarity for reporting.
  - Built per-cat centroid embedding store from dataset folders; added caching at repo root (`cat_embeddings_cache.npy`, `cat_metadata_cache.json`).
  - Added maintenance endpoints: `/api/health`, `/api/reindex`, `/api/add`.
  - Frontend templates updated for upload spinner and responsive results cards with confidence badges.
  - Training script (`train_cat_identifier_v2.py`) aligned to EfficientNetV2-S; replaced Keras 3-only augmentations with `tf.image.adjust_*`.
  - Test script (`test_cat_identifier_v2.py`) aligned to EfficientNetV2-S; added model guards; cleaned visualization manager access.
  - Environment pinned for compatibility: Python 3.9, TensorFlow 2.10.0, NumPy 1.23.5, OpenCV 4.7.0.72.
  - Diagnostics resolved by guarding shape logging (e.g., `120 transfer now/120 transfer now/app.py:95–99`), safe CV2 attribute access (`getattr` usage around `CascadeClassifier`, `cvtColor`, constants), secure filename defaults, and type‑safe `np.save` object arrays.
- 2025-11-28 — UX Fix: Upload button single-click
  - Root cause: browsers avoid firing `change` when the same file is reselected.
  - Implementation: clear `fileInput.value` before `fileInput.click()` and on `click` listener.
  - Files: `120 transfer now/120 transfer now/templates/index.html:166, 263, 423–426, 495–500`; `120 transfer now/120 transfer now/templates/found_cat.html:117, 334–336, 412–416`.

## 19. Detailed UX Fix — Double Dialog Prevention

- Problem: The “Choose File” button opened two file dialogs because both the inline button handler and the drag-drop area click listener invoked `fileInput.click()`.
- Fix:
  - Removed inline `onclick` from the button and assigned `id="chooseFileBtn"`.
  - Added a JS click handler on `chooseFileBtn` that calls `e.stopPropagation()` to prevent the parent area’s click listener from firing, then clears and opens the picker.
  - Ensured input clearing on `fileInput` click and on drag area click to guarantee the `change` event when reselecting the same file.
- Code references:
  - `120 transfer now/120 transfer now/templates/index.html:166` — Button uses `id="chooseFileBtn"` (inline handler removed).
  - `120 transfer now/120 transfer now/templates/index.html:378` — `const chooseFileBtn = document.getElementById('chooseFileBtn');`.
  - `120 transfer now/120 transfer now/templates/index.html:505` — `chooseFileBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.value=''; fileInput.click(); });`.
  - `120 transfer now/120 transfer now/templates/index.html:423–426` — Clear input on `fileInput` click.
  - `120 transfer now/120 transfer now/templates/index.html:495–500` — Drag area click clears input then opens picker.
  - `120 transfer now/120 transfer now/templates/found_cat.html:117` — Button uses `id="chooseFileBtn"`.
  - `120 transfer now/120 transfer now/templates/found_cat.html:288` — `const chooseFileBtn = document.getElementById('chooseFileBtn');`.
  - `120 transfer now/120 transfer now/templates/found_cat.html:423` — `chooseFileBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.value=''; fileInput.click(); });`.
  - `120 transfer now/120 transfer now/templates/found_cat.html:334–336` — Clear input on `fileInput` click.
  - `120 transfer now/120 transfer now/templates/found_cat.html:412–416` — Drag area click clears input then opens picker.
- Validation steps:
  - Click “Choose File” once; verify one dialog and single-image selection yields preview.
  - Reselect the same image; confirm preview updates on first try.
  - Drag & drop continues to show preview and enables submission where applicable.

## 20. Beginner-Friendly Explanation — What Changed and Why

- Think of the upload area like a box with a button inside it. Before, both the box and the button were wired to open the file chooser. So when you clicked the button, both wires fired, and Windows showed you two file chooser windows.
- We fixed it by:
  - Disconnecting the button’s old inline click wire (removed the inline `onclick`).
  - Giving the button its own simple wire (a JavaScript handler) that says: “Stop the box from reacting to this click” (`stopPropagation`), then open the file chooser once.
  - Clearing the hidden file input’s previous value before opening the chooser. This matters because computers often ignore picking the same file twice unless you clear the old selection first; clearing it makes the next pick look “new” and triggers the preview.
- In plain steps, when you click the button now:
  - The button stops the box (drag area) from also handling the click.
  - The button clears the old selection so the system will notice your pick.
  - The button opens the file chooser once.
  - When you pick an image, the page shows a preview and gets the file ready to upload.
- Drag & drop already worked fine because it only uses one path (dropping a file directly), and our change makes the button behave the same way: one action, one chooser, one image.
- Where this was changed in code:
  - Home page button and script: `templates/index.html:166, 378, 423–426, 495–500, 505`.
  - Found Cat page button and script: `templates/found_cat.html:117, 288, 334–336, 412–416, 423`.

## 21. UI/UX Contrast Improvements

- Goal: Improve readability of muted text on dark and gradient backgrounds to enhance user experience and accessibility.
- Changes applied:
  - Introduced color variables for dark contexts: `--muted-on-dark`, `--muted-on-gradient`.
  - Overrode muted text color within dark sections: `footer .text-muted`, `.bg-dark .text-muted`.
  - Overrode muted text color within gradient sections: `.bg-gradient .text-muted`, and restricted hero drag-drop muted text to the gradient context: `.hero-section .drag-drop-content .text-muted`.
  - Added focus-visible outlines for interactive elements (`.btn`, `.nav-link`, `.drag-drop-area`, `.form-control`, `.form-select`) to improve keyboard navigation and accessibility.
- Code references:
  - `120 transfer now/120 transfer now/static/css/style.css:13–33` — Added `--muted-on-dark`, `--muted-on-gradient` variables.
  - `120 transfer now/120 transfer now/static/css/style.css:562–570` — Footer hover remains unchanged for brand consistency.
  - `120 transfer now/120 transfer now/static/css/style.css:571–586` — Dark-context muted text and hero gradient muted text overrides.
  - `120 transfer now/120 transfer now/static/css/style.css:587–595` — Focus-visible accessibility outlines.
- Supported formats text color updated to `text-info` for clearer visibility:
  - Home page: `templates/index.html:193` now uses `<small class="text-info">...`
  - Found Cat page: `templates/found_cat.html:135–139` replaced `.form-text` block with centered `<small class="text-info">...`
- Outcome:
  - Muted text now maintains sufficient contrast on dark/gradient backgrounds while preserving original appearance on light cards.
  - Improved accessibility and clearer focus states across interactive UI components.

## 22. Visibility Fix — Drag & Drop Content Colors

- Problem: Drag & drop content text was forced to white, but the upload area sits on a white card background; white-on-white made text hard to read.
- Fix:
  - Set `.drag-drop-content` text color to `var(--dark-color)` for high contrast on light cards.
  - Set `.upload-icon` color to `var(--primary-color)` so the icon stands out against light backgrounds.
- Code references:
  - `120 transfer now/120 transfer now/static/css/style.css:207–209` → `.drag-drop-content { color: var(--dark-color); }`.
  - `120 transfer now/120 transfer now/static/css/style.css:211–215` → `.upload-icon { color: var(--primary-color); }`.
- Outcome:
  - Upload area text is readable on light cards and consistent across pages; icons retain brand color emphasis.

## 23. Feature Addition — Found Cats Map

- Goal: visualize the location of found cats on an interactive map, allowing users to see where pets have been spotted.
- Implementation:
  - Backend: Installed `geopy` and configured `Nominatim` geocoder. Updated `upload_found_cat` in `app.py` to geocode the location string into latitude/longitude coordinates. Created `/api/found-cats-map` endpoint to serve geolocated cat data.
  - Frontend: Integrated Leaflet.js in `view_found_cats.html`. Added a map container and custom CSS for circular photo markers. Implemented JS to fetch map data and render markers with popups.
- Code references:
  - `app.py`: Imports, `geolocator` init, geocoding logic in upload route, and `/api/found-cats-map` route.
  - `view_found_cats.html`: Added Leaflet CSS/JS, map container div, custom marker CSS, and map initialization script.
- Usage: When a user uploads a found cat with a location (e.g., "Bangsar, Kuala Lumpur"), the system automatically converts it to coordinates and displays it on the map in the "View Found Cats" page.

## 24. Bug Fix — Map Visibility & API Route

- Symptom: The map on "View Found Cats" page was empty, and the API endpoint `/api/found-cats-map` returned 404 Not Found.
- Cause: The API route definition was accidentally omitted during a previous `app.py` update.
- Fix:
  - Restored the `/api/found-cats-map` route in `app.py`.
  - Ensured `geopy` dependency is listed in `requirements.txt`.
  - Ran a coordinate update script to backfill latitude/longitude for existing found cats.
- Verification: Validated that the API now returns JSON data and the map correctly renders markers for found cats.

## 25. Feature Addition — Home Page Map

- Goal: Show recent found cats on the homepage with the same photo markers.
- Implementation:
  - Added Leaflet CSS/JS and custom marker styles to `templates/index.html:11–33, 19–26`.
  - Inserted a new “Recent Found Cats” map section `templates/index.html:270–294`.
  - Initialized Leaflet and fetched `/api/found-cats-map` `templates/index.html:419–490`.
- CSP: Added permissive meta tag to allow map scripts: `templates/index.html:11–19`.
- Verification: Map renders and clusters markers; popups show photo, name, location, date.

## 26. Bug Fix — CSP Blocking Map Scripts

- Symptom: Browser console showed CSP errors blocking `eval`/inline scripts; map did not load.
- Cause: Default policy too strict for Leaflet and inline initialization.
- Fix:
  - Added `<meta http-equiv="Content-Security-Policy" content="default-src 'self' https: data: 'unsafe-inline' 'unsafe-eval'; img-src 'self' https: data: blob:;">` in head. Files: `templates/index.html:11–19`; `templates/view_found_cats.html:4–6`.
  - Note: Wallet extensions (e.g., MetaMask) may log provider warnings; unrelated to map.

## 27. Data Operations — Populate & Randomize Found Cats

- Scripts added:
  - `add_all_to_found.py`: mark all dataset cats as `status: 'found'`, set defaults (KL coordinates, date, location).
  - `randomize_found_cat_locations.py`: spread found cats across KL/Klang neighborhoods with realistic radii; updates `lat/lng/location`.
  - `remove_found_cats.py`: purge found cats (image files, cache entries, subset metadata).
- Caches updated: `cat_metadata_cache.json`, `found_cats_metadata.json`; embeddings cache pruned where applicable.
- Verification: Home/Found pages list all cats; maps display distributed markers.

## 28. Bug Fix — Images Not Showing (Found/Reunited)

- Symptom: Card images blank for dataset-backed entries.
- Cause: Templates used `serve_found_cat_image(filename)` which only resolves images in `found_cats_dataset/`; dataset cats use full `image_path`.
- Fix:
  - Switch to `serve_cat_image(cat_id)` so backend reads `image_path` from metadata.
  - Files: `templates/view_found_cats.html:171–176`; `templates/reunited_cats.html:145–149`.
- Verification: Cards now render images for both dataset and found-cats entries.

## 29. Model Loading & Inference Compatibility (Keras 3)

- Problem: `.savedmodel` not supported by `load_model()` in Keras 3; `.h5` deserialization error on `DepthwiseConv2D(groups=1)`.
- Approach:
  - Load order: `.keras` → `.h5` → SavedModel with `keras.layers.TFSMLayer` fallback. Files: `120 transfer now/120 transfer now/app.py:79–141`.
  - Warm-up handles both Model and TFSMLayer: `120 transfer now/120 transfer now/app.py:392–405`.
  - Embedding extraction supports `predict()` or callable layer; converts tensors to NumPy; normalizes output. Files: `120 transfer now/120 transfer now/app.py:229–262`.
- Outcome: Model loads reliably; inference works across backends; system initializes successfully.

## 30. Bug Fix — NumPy Cache Incompatibility ("No module named numpy._core")

- Symptom: Startup failed in `.venv39` with `No module named 'numpy._core'` when loading `cat_embeddings_cache.npy`.
- Root cause: Pickled cache created under a different NumPy/Python version; unpickling in another env attempted to import internal modules that don’t exist.
- Fix:
  - Added cache-load guard: on failure, rebuild from dataset and re-save caches. Files: `120 transfer now/120 transfer now/app.py:269–321` (cache path and guarded load in `load_database`).
  - Saved embeddings as object arrays: `np.save(..., np.array(self.cat_embeddings, dtype=object))`; loaded with `allow_pickle=True`.
  - Ensured `.venv39` uses NumPy 1.23.5 for TF 2.10 compatibility.
- Outcome: First run rebuilds caches; subsequent runs start cleanly in the selected venv.

## 31. Environment Compatibility — geopy & Virtual Envs

- Issue: `ModuleNotFoundError: geopy` and IDE red underlines due to mismatched interpreters.
- Approach:
  - Installed `geopy` in the active venv; added `requirements.txt`.
  - Added `pyrightconfig.json` with `typeCheckingMode: basic`, venv mapping to `.venv39`, and reduced dynamic-library warnings; diagnostics clear for `app.py`.
  - Verified secure filename defaults and analyzer-friendly predict calls. Files: `120 transfer now/120 transfer now/app.py:241–243`, `396–399`.
- Outcome: Consistent environment and clean IDE diagnostics without changing runtime behavior.

## 32. Feature Addition — Sorting in View Found Cats

- Goal: Let users sort found cats by common fields and order.
- Implementation:
  - Backend supports `?sort=<field>&dir=<asc|desc>`. Fields: `upload_time`, `date_found`, `name`, `location`, `id`. Files: `120 transfer now/120 transfer now/app.py:591–609` (route `view_found_cats`).
  - Frontend adds a sort control card with dropdowns and Apply button above the grid. Files: `templates/view_found_cats.html:164–196` (stats end) and additional sort form.
- Outcome: Users can reorder cards by report time, date found, name, location, or ID in ascending/descending order.

## 33. Data Tools — Mass Add, Randomize, and Purge Found Cats

- Scripts:
  - `add_all_to_found.py`: marks all dataset cats as found, sets defaults (KL location/coords) and writes `found_cats_metadata.json`.
  - `randomize_found_cat_locations.py`: spreads found cats across KL/Klang neighborhoods with realistic radii; updates `lat/lng/location`.
  - `remove_found_cats.py`: deletes found-cat images, removes entries from caches, clears `found_cats_metadata.json`.
- Usage: Run each script with the active venv’s Python; changes are reflected in both list views and maps.

## 34. UX Improvement — Cat Name Field on Found Cat Upload

- Feature: Added optional "Cat Name" input with friendly note: (Don't know the name? Give it one).
- Implementation:
  - Frontend: `templates/found_cat.html` new input `name="cat_name"` under the upload form.
  - Backend: `upload_found_cat` reads `cat_name`; if empty, defaults to `Found Cat <timestamp>`. Files: `120 transfer now/120 transfer now/app.py:510–519`, `548–560`.
- Outcome: Submitted found cats show user-provided names; otherwise a sensible default is used.

## 35. Cat Profile — Structured Fields for Found Cats

- Feature: Added structured profile fields (Gender, Age, Vaccinated, Dewormed, Spayed/Neutered, Condition, Body Size, Fur Length, Color).
- Implementation:
  - Frontend: Inputs added to `templates/found_cat.html` within the upload form.
  - Backend: `upload_found_cat` processes new fields and stores them in metadata; also builds a `profile_summary` for quick display. Files: `120 transfer now/120 transfer now/app.py:514–525`, `548–569`.
  - Display: Cards in `templates/view_found_cats.html` now show profile details beneath the description.
- Outcome: Each found cat has a consistent profile that helps owners identify pets quickly.

## 36. Breed Field — Classification for Found Cats

- Feature: Added a breed selector with common options and Unknown/Other choices.
- Implementation:
  - Frontend: `templates/found_cat.html` adds `name="breed"` select near Color(s).
  - Backend: `upload_found_cat` captures `breed` and stores it in metadata; cards show "Cat: <breed>". Files: `120 transfer now/120 transfer now/app.py:514–525`, `560–571`; `templates/view_found_cats.html` profile block.
- Outcome: Listings include breed information helping owners filter and identify pets more quickly.

## 37. New Section — Lost Cat Reporting

- Feature: Added a complete workflow for owners to post lost cats.
- Implementation:
  - New page: `templates/lost_cat.html` with photo upload, name, last seen location, date lost, profile fields (gender, age, vaccinated, dewormed, spayed/neutered, condition, body size, fur length, color, breed), and contact info.
  - Backend: `upload_lost_cat` route saves image to `LOST_CATS_PATH`, extracts embedding, stores metadata with `status: 'lost'`, and geocodes location. Files: `app.py` (routes and constants, adds `LOST_CATS_PATH`, `lost_cat`, `upload_lost_cat`, `view_lost_cats`).
  - Listing page: `templates/view_lost_cats.html` to browse lost cat reports with sorting.
  - Navigation: Added “Report Lost Cat” link in navbar (`templates/index.html`).
- Outcome: Owners can report lost cats; community can view and contact owners; embeddings integrate into matching pipeline.

---

## Change Logging Policy (Ongoing)

- From 2025-12-01, all changes and major bug fixes are recorded in this Markdown file.
- Each entry includes: Summary, Files affected, Approach, Verification, Impact, Next steps.
- Entries are appended in reverse chronological order.

### Entry Template

```
## [YYYY-MM-DD] Summary Title

### Files Affected
- path/to/file1
- path/to/file2

### Approach
- What was changed and why
- Key decisions and trade-offs

### Verification
- How it was tested (manual, unit tests, preview)
- Results

### Impact
- User-facing effects
- Performance, security, compatibility notes

### Next Steps
- Follow-ups, monitoring, or TODOs
```

## [2025-12-01] Establish Continuous Change Logging

### Files Affected
- `docs/PetIDMalaysia_Project_Documentation.md`

### Approach
- Converted existing RTF documentation to Markdown for easier versioning and diffs.
- Established this file as the canonical location for all future change records.
- Adopted a structured template for consistency and traceability.

### Verification
- Reviewed RTF content and ensured key sections were ported with correct code references.
- Validated Markdown rendering for headings, lists, and code blocks.

### Impact
- Centralized, diff-friendly documentation; simpler collaboration and audits.
- Faster navigation to code via `file_path:line_number` references.

### Next Steps
- Append a new entry here for every change/bug fix going forward.

## [2025-12-01] Pagination, Map Clustering, CSP Tightening, and Form Validations

### Files Affected
- `120 transfer now/120 transfer now/app.py`
- `120 transfer now/120 transfer now/templates/view_found_cats.html`
- `120 transfer now/120 transfer now/templates/view_lost_cats.html`
- `120 transfer now/120 transfer now/templates/index.html`

### Approach
- Added server-side pagination for found and lost cats with default `page_size=24` and query params `?page=&page_size=`.
- Preserved existing sorting (`sort`, `dir`) and route paths; passed pagination metadata to templates.
- Integrated Leaflet MarkerCluster with chunked marker addition to improve map performance.
- Tightened CSP by removing `unsafe-eval` while retaining `unsafe-inline` for current inline init.
- Optimized images with `loading="lazy"` and explicit dimensions for consistent thumbnails.
- Added basic HTML validations (`maxlength`) to found/lost forms without changing optional fields.

### Verification
- Manual testing across listing pages: sorting retained; pagination links navigate correctly; counts match total.
- Maps load without CSP errors; clusters render and popups show required details.
- `/api/health` unchanged; server starts cleanly.

### Impact
- Listing pages render faster with fewer DOM nodes per page.
- Maps handle larger datasets smoothly; better UX with clustering.
- Stricter CSP eliminates `unsafe-eval` warnings in console.
- Improved form robustness; prevents excessively long inputs.

### Code References
- Pagination logic: `120 transfer now/120 transfer now/app.py:639–666` (found) and `120 transfer now/120 transfer now/app.py:869–889` (lost) updated to include `page`, `page_size`, and slicing.
- Found cats template: sort form hidden fields and pagination UI `templates/view_found_cats.html:176–204, 205–221`; image lazy loading `templates/view_found_cats.html:213–217`.
- Lost cats template: sort form hidden fields and pagination UI `templates/view_lost_cats.html:80–105, 106–151`; image lazy loading `templates/view_lost_cats.html:113`.
- Map clustering and CSP: `templates/index.html:13–15, 431–496` and `templates/view_found_cats.html:6–8, 448–510`.

### Next Steps
- Consider moving inline scripts to static files to allow stricter CSP without `unsafe-inline`.
- Optional: add page number list for direct jumps when total pages > 5.

## [2025-12-01] Breed Options Standardized to Dataset Categories

### Files Affected
- `120 transfer now/120 transfer now/templates/found_cat.html`
- `120 transfer now/120 transfer now/templates/lost_cat.html`
- `120 transfer now/120 transfer now/templates/view_found_cats.html`
- `120 transfer now/120 transfer now/templates/view_lost_cats.html`

### Approach
- Updated breed selects to include 13 dataset breeds: Abyssinian, Bengal, Birman, Bombay, British_Shorthair, Domestic_Shorthair, Egyptian_Mau, Maine_Coon, Persian, Ragdoll, Russian_Blue, Siamese, Sphynx.
- Kept `Unknown` and `Other` for flexibility; option values match dataset folder names (underscores) while UI displays friendly names.
- Displayed breed on listing cards using a replace filter to show spaces instead of underscores.

### Verification
- Forms render updated options; submitted metadata stores selected breed.
- Listing pages show user-friendly breed names; no impact on sorting or pagination.

### Code References
- Found form options: `templates/found_cat.html:263–278`.
- Lost form options: `templates/lost_cat.html:199–213`.
- Found list display: `templates/view_found_cats.html:266`.
- Lost list display: `templates/view_lost_cats.html:118`.

### Next Steps
- If backend mapping is needed later, use the stored values to align with dataset labels.

## [2025-12-02] Reunited Feature Added to View Lost Cats

### Files Affected
- `120 transfer now/120 transfer now/templates/view_lost_cats.html`

### Approach
- Added a "Mark as Reunited" button to each lost-cat card, consistent with the Found Cats page.
- Implemented a Bootstrap modal with optional `owner_name` and `reunion_story` fields.
- Modal form posts to existing route `POST /mark-reunited/<cat_id>` which updates metadata and saves caches.
- After marking, the cat is excluded from lost listings and appears in Reunited.

### Verification
- Clicking the button opens the modal; form posts to the correct cat ID.
- Reunited entries show on `/reunited-cats` and disappear from `/view-lost-cats`.

### Code References
- Lost list actions and modal: `templates/view_lost_cats.html:129–136, 155–162` (actions) and new modal/JS at end of file.

### Impact
- Consistent UX across Found and Lost listings; quick workflow to confirm reunions.

## [2025-12-02] Lost Cat Form Redesigned to Match Found Cat Drag‑Drop Style

### Files Affected
- `120 transfer now/120 transfer now/templates/lost_cat.html`

### Approach
- Replaced basic file input with the same drag‑and‑drop upload UI used on the Found Cat page, including hidden file input, preview block, and “Choose File” button.
- Added client‑side validation for type and size, preview rendering, and enabling/disabling the submit button.
- Included a loading overlay while posting the lost cat.
- Kept all existing fields and route `POST /upload-lost-cat` unchanged.

### Verification
- File select and drag‑drop both produce a preview and enable the submit button.
- On submit, overlay appears; backend receives `file` as before.

### Code References
- Drag‑drop and preview: `templates/lost_cat.html` upload section and JS block at end of file.

### Impact
- Consistent, modern upload UX; fewer user errors; improved clarity with preview.

## [2025-12-02] Lost Cat Form Title Added

### Files Affected
- `120 transfer now/120 transfer now/templates/lost_cat.html`

### Approach
- Inserted a centered card title with camera icon: “Upload Lost Cat Details” above the upload form, matching Found page style.

### Verification
- Title renders above the drag‑drop area on the Lost Cat page.

## [2025-12-02] Search Bar and Filters on Found/Lost Listings

### Files Affected
- `120 transfer now/120 transfer now/app.py`
- `120 transfer now/120 transfer now/templates/view_found_cats.html`
- `120 transfer now/120 transfer now/templates/view_lost_cats.html`

### Approach
- Added search/filter parameters: `q` (keyword across name/id/location/description), `breed1`, `breed2`, `age` (All, Kitten, Young, Adult, Senior), and `loc` (substring match).
- Implemented filtering in both listing routes before sorting and pagination.
- UI: Inserted a search card with inputs and “Search Cats” button; preserves sort/dir and resets page to 1.

### Verification
- Filters combine correctly; pagination and sorting preserved.
- Keyword and location perform case-insensitive substring search.
- Age buckets computed from `age_years` and `age_months`.

### Code References
- Filtering logic: `app.py:646–680` (found) and `app.py:895–925` (lost).
- Search UI: `templates/view_found_cats.html` search card above sort; same for `templates/view_lost_cats.html`.

## [2025-12-02] Multi-Image Search (3–5 Photos, Aggregated Scores)

### Files Affected
- `120 transfer now/120 transfer now/app.py`
- `120 transfer now/120 transfer now/templates/index.html`
- `120 transfer now/120 transfer now/templates/results.html`

### Approach
- Backend: Added `find_matches_multi(img_paths, top_k)` that extracts embeddings per photo and aggregates per-cat scores by mean L2 distance and mean cosine.
- Upload route: Accepts up to 5 images via `getlist('file')`, saves them, and calls single or multi matching accordingly; passes `uploaded_images` for results.
- Frontend: Homepage upload now supports selecting multiple files; preview shows the first photo and count; preserves drag‑drop flow.
- Results: Displays a grid of uploaded photos when multiple files are used and updates analysis footer.

### Verification
- Selecting 2–5 photos produces aggregated results; single photo path unchanged.
- Results render with percent bars and badges without template errors.

### Code References
- Matching: `app.py` new method in `PetRecognitionSystem` and updated `/upload` route.
- Multi-select input and preview: `templates/index.html` (`multiple` input, preview count, JS updates).
- Results grid: `templates/results.html` shows uploaded photos and analysis count.

## [2025-12-02] Fix Inconsistent Success Flash on Home Page

### Files Affected
- `120 transfer now/120 transfer now/app.py`
- `120 transfer now/120 transfer now/templates/found_cat_success.html`

### Problem
- After uploading a found cat, a success flash was set and the success page did not consume the flash. Returning to the Home page caused the same success message to appear incorrectly in the search upload card.

### Approach
- Removed success `flash(...)` in `upload_found_cat` and relied on the success page content.
- Added a flash consumption block to `found_cat_success.html` to ensure any residual messages are consumed on that page.

### Verification
- Uploading a found cat shows the success page without causing unrelated flashes on Home afterward.
- Home page now only shows messages related to its own actions (search uploads/errors).

## [2025-12-02] Results Template Percent Rendering Fix

### Files Affected
- `120 transfer now/120 transfer now/templates/results.html`

### Problem
- Jinja raised "expected token 'end of print statement', got '%'" when printing a percent sign immediately after a `{{ ... }}` expression on some environments.

### Approach
- Used Jinja string concatenation with `~ '%'` to include the percent sign inside the expression for both the numeric label and the progress bar width style.

### Code References
- Updated lines: `results.html:217` and `results.html:222`.

### Verification
- Matches page renders without template errors; width and label display correctly with percent.

## [2025-12-02] Results Similarity Block Refactor

### Files Affected
- `120 transfer now/120 transfer now/templates/results.html`

### Problem
- Linter/templating red underline persisted due to complex inline expressions for class selection and percent concatenation.

### Approach
- Introduced local Jinja variables `cosine_percent` and `bar_class` using `{% set %}` to simplify expressions.
- Reused these variables across the progress bar width, `aria-valuenow`, the badge class, and the icon selection, reducing nested conditionals.

### Code References
- Refactor area: `results.html:215–241`.

### Impact
- Cleaner, more readable template; IDE diagnostics cleared in the similarity section.
