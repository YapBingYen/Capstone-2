"""
Pet ID Malaysia - AI Pet Recognition Web Application
Main Flask application for lost cat identification using deep learning.

Author: Pet ID Team
Purpose: Match uploaded cat photos with lost cats database using AI
Dependencies: Flask, TensorFlow, OpenCV, NumPy, scikit-learn
"""

import os
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory, send_file, abort
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import json
from datetime import datetime
import logging
from geopy.geocoders import Nominatim

# Configure TensorFlow image data format to ensure compatibility
tf.keras.backend.set_image_data_format('channels_last')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Geocoder
geolocator = Nominatim(user_agent="pet_id_malaysia")

# Configuration
MODEL_DIR = r"D:\Cursor AI projects\Capstone2.1\models"
MODEL_BASE = os.path.join(MODEL_DIR, 'cat_identifier_efficientnet_v2')
MODEL_SAVEDMODEL = os.path.join(MODEL_DIR, 'cat_identifier_efficientnet_v2_savedmodel')
MODEL_WEIGHTS = os.path.join(MODEL_DIR, 'cat_identifier_efficientnet_v2_weights.h5')
MODEL_KERAS = MODEL_BASE + '.keras'
MODEL_H5 = MODEL_BASE + '.h5'
HAAR_PATH = 'haarcascade_frontalcatface.xml'
DATASET_PATH = r"D:\Cursor AI projects\Capstone2.1\120 transfer now\120 transfer now\cat_individuals_dataset\dataset_individuals_cropped\cat_individuals_dataset"
FOUND_CATS_PATH = r"D:\Cursor AI projects\Capstone2.1\120 transfer now\120 transfer now\found_cats_dataset"  # Path for found cats
LOST_CATS_PATH = r"D:\Cursor AI projects\Capstone2.1\120 transfer now\120 transfer now\lost_cats_dataset"  # Path for lost cats
APP_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.abspath(os.path.join(APP_DIR, '..', '..'))
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, 'cat_embeddings_cache.npy')
METADATA_CACHE = os.path.join(CACHE_DIR, 'cat_metadata_cache.json')
FOUND_CATS_METADATA = os.path.join(CACHE_DIR, 'found_cats_metadata.json')  # Metadata for found cats

app = Flask(__name__)
app.secret_key = 'pet_id_malaysia_2024_secure_key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(FOUND_CATS_PATH, exist_ok=True)
os.makedirs(LOST_CATS_PATH, exist_ok=True)

# Global variables for model and database
model = None
cat_embeddings = {}
cat_metadata = {}
haar_cascade = None

IMG_SIZE = 224

class PetRecognitionSystem:
    """Main class for handling pet recognition operations"""

    def __init__(self):
        self.model = None
        self.cat_embeddings = {}
        self.cat_metadata = {}
        self.haar_cascade = None
        self.target_size = (IMG_SIZE, IMG_SIZE)
        self.embedding_model = None
        self.emb_matrix = None
        self.emb_ids = []

    def load_model(self):
        # 1. Try loading .keras model
        if os.path.exists(MODEL_KERAS):
            try:
                logger.info(f"🔄 Loading .keras model from: {MODEL_KERAS}")
                self.model = tf.keras.models.load_model(MODEL_KERAS, compile=False)
                self._finalize_model_load()
                return True
            except Exception as e:
                logger.warning(f"⚠️ Failed to load .keras model: {e}")

        # 2. Try loading .h5 model
        if os.path.exists(MODEL_H5):
            try:
                logger.info(f"🔄 Loading .h5 model from: {MODEL_H5}")
                self.model = tf.keras.models.load_model(MODEL_H5, compile=False)
                self._finalize_model_load()
                return True
            except Exception as e:
                logger.warning(f"⚠️ Failed to load .h5 model: {e}")

        # 3. Try loading SavedModel
        if os.path.exists(MODEL_SAVEDMODEL):
            try:
                logger.info(f"🔄 Loading SavedModel from: {MODEL_SAVEDMODEL}")
                self.model = tf.keras.models.load_model(MODEL_SAVEDMODEL, compile=False)
                self._finalize_model_load()
                return True
            except Exception:
                try:
                    logger.warning("⚠️ Keras 3 cannot load SavedModel directly. Trying TFSMLayer...")
                    self.model = tf.keras.layers.TFSMLayer(MODEL_SAVEDMODEL, call_endpoint='serving_default')
                    self._finalize_model_load()
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load SavedModel: {e}")

        # 4. Try loading weights
        if os.path.exists(MODEL_WEIGHTS):
            try:
                logger.info(f"🔄 Building embedding model and loading weights: {MODEL_WEIGHTS}")
                inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
                x = tf.keras.layers.Lambda(lambda t: t)(inputs)
                temp = tf.keras.Model(inputs, x)
                temp.load_weights(MODEL_WEIGHTS)
                self.model = temp
                self._finalize_model_load()
                return True
            except Exception as e:
                logger.error(f"❌ Failed to load weights: {e}")

        logger.error("❌ No compatible model file found or all load attempts failed")
        return False

    def _finalize_model_load(self):
        self.target_size = (IMG_SIZE, IMG_SIZE)
        if self.model is not None:
            # Check if model has input_shape (TFSMLayer might not)
            input_shape = getattr(self.model, 'input_shape', 'Unknown')
            output_shape = getattr(self.model, 'output_shape', 'Unknown')
            logger.info(f"✅ Model loaded. Input: {input_shape} Output: {output_shape}")
        self.embedding_model = self.model

    def load_haar_cascade(self):
        """Load Haar Cascade for cat face detection"""
        try:
            # First try to load from local file, then from OpenCV data
            if os.path.exists(HAAR_PATH):
                self.haar_cascade = getattr(cv2, "CascadeClassifier")(HAAR_PATH)
            else:
                # Use a basic face detection as fallback
                haar_dir = getattr(cv2, "data").haarcascades
                self.haar_cascade = getattr(cv2, "CascadeClassifier")(haar_dir + 'haarcascade_frontalface_default.xml')

            if self.haar_cascade.empty():
                logger.warning("⚠️ Haar Cascade not loaded, using full image preprocessing")
                self.haar_cascade = None
            else:
                logger.info("✅ Haar Cascade loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Error loading Haar Cascade: {e}, using full image preprocessing")
            self.haar_cascade = None

    def preprocess_image(self, img_path):
        """Preprocess image for EfficientNetV2 embedding model"""
        try:
            # Read and preprocess image
            img = getattr(cv2, "imread")(img_path)
            if img is None:
                raise ValueError("Could not read image")

            # Convert to RGB
            img_rgb = getattr(cv2, "cvtColor")(img, getattr(cv2, "COLOR_BGR2RGB"))

            # Try cat face detection if Haar Cascade is available
            if self.haar_cascade is not None:
                gray = getattr(cv2, "cvtColor")(img, getattr(cv2, "COLOR_BGR2GRAY"))
                faces = self.haar_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) > 0:
                    # Use the first detected face
                    x, y, w, h = faces[0]
                    # Add some padding around the detected face
                    padding = 20
                    x_start = max(0, x - padding)
                    y_start = max(0, y - padding)
                    x_end = min(img_rgb.shape[1], x + w + padding)
                    y_end = min(img_rgb.shape[0], y + h + padding)

                    img_cropped = img_rgb[y_start:y_end, x_start:x_end]
                else:
                    # Fallback to center crop if no face detected
                    img_cropped = self.center_crop(img_rgb)
            else:
                # Use center crop if no Haar Cascade
                img_cropped = self.center_crop(img_rgb)

            img_resized = getattr(cv2, "resize")(img_cropped, self.target_size)

            # Convert to PIL Image and ensure RGB
            img_pil = Image.fromarray(img_resized)

            img_array = keras.utils.img_to_array(img_pil)
            img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

            # Add batch dimension
            img_batch = np.expand_dims(img_array, axis=0)

            return img_batch, img_pil

        except Exception as e:
            logger.error(f"❌ Error preprocessing image {img_path}: {e}")
            return None, None

    def center_crop(self, img):
        """Center crop the image to make it more square"""
        h, w = img.shape[:2]

        # Determine crop size (use smaller dimension)
        crop_size = min(h, w)

        # Calculate crop coordinates
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2

        # Crop the image
        cropped = img[start_h:start_h + crop_size, start_w:start_w + crop_size]

        return cropped

    def extract_embedding(self, img_path):
        """Extract embedding from image using the EfficientNetV2 embedding model"""
        try:
            img_batch, _ = self.preprocess_image(img_path)
            if img_batch is None:
                return None

            if self.embedding_model is None:
                logger.error("❌ Embedding model not initialized")
                return None

            # Handle different model types (Keras Model vs TFSMLayer)
            if hasattr(self.embedding_model, 'predict'):
                embedding = self.embedding_model.predict(img_batch)
            else:
                # For TFSMLayer or direct call
                embedding = self.embedding_model(img_batch)
                # TFSMLayer returns a dict
                if isinstance(embedding, dict):
                    embedding = list(embedding.values())[0]
            
            # Convert tensor to numpy if needed
            if hasattr(embedding, 'numpy'):
                embedding = embedding.numpy()
                
            embedding = embedding.flatten()
            # If not already normalized by the model, normalize here
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            return embedding

        except Exception as e:
            logger.error(f"❌ Error extracting embedding from {img_path}: {e}")
            return None

    def load_database(self):
        """Load cat database and extract embeddings"""
        logger.info("🔄 Loading cat database...")

        try:
            # Try to load from cache first
            if os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(METADATA_CACHE):
                try:
                    self.cat_embeddings = np.load(EMBEDDINGS_CACHE, allow_pickle=True).item()
                    with open(METADATA_CACHE, 'r') as f:
                        self.cat_metadata = json.load(f)
                    logger.info(f"✅ Loaded {len(self.cat_embeddings)} cats from cache")
                    self.build_matrix()
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Cache load failed ({e}); rebuilding from dataset")

            # If no cache, load from dataset
            if not os.path.exists(DATASET_PATH):
                logger.error(f"❌ Dataset path not found: {DATASET_PATH}")
                logger.error(f"❌ Please ensure your dataset is located at: {DATASET_PATH}")
                return False

            # Build per-cat centroid embeddings (folder name = cat ID)
            cat_to_images = {}
            for root, dirs, files in os.walk(DATASET_PATH):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        cat_id = os.path.basename(root)
                        cat_to_images.setdefault(cat_id, []).append(os.path.join(root, file))

            logger.info(f"📁 Found {sum(len(v) for v in cat_to_images.values())} images across {len(cat_to_images)} cats")

            for idx, (cat_id, paths) in enumerate(cat_to_images.items(), start=1):
                embs = []
                for p in paths:
                    emb = self.extract_embedding(p)
                    if emb is not None:
                        embs.append(emb)
                if not embs:
                    continue
                centroid = np.mean(np.stack(embs, axis=0), axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                self.cat_embeddings[cat_id] = centroid
                rep_path = paths[0]
                self.cat_metadata[cat_id] = {
                    'filename': os.path.basename(rep_path),
                    'image_path': rep_path,
                    'id': cat_id,
                    'name': f"Cat {cat_id}"
                }

            # Save to cache for future use (use object array for dict)
            np.save(EMBEDDINGS_CACHE, np.array(self.cat_embeddings, dtype=object))
            with open(METADATA_CACHE, 'w') as f:
                json.dump(self.cat_metadata, f)

            logger.info(f"✅ Successfully loaded {len(self.cat_embeddings)} cats")
            self.build_matrix()
            return True

        except Exception as e:
            logger.error(f"❌ Error loading database: {e}")
            return False

    
    def build_matrix(self):
        try:
            if not self.cat_embeddings:
                self.emb_matrix = None
                self.emb_ids = []
                return
            ids = list(self.cat_embeddings.keys())
            mat = np.stack([self.cat_embeddings[i] for i in ids], axis=0)
            self.emb_ids = ids
            self.emb_matrix = mat
        except Exception as e:
            logger.warning(f"⚠️ Error building embedding matrix: {e}")
            self.emb_matrix = None
            self.emb_ids = []
    def find_matches(self, uploaded_img_path, top_k=10):
        """Find top k most similar cats using L2 distance on normalized embeddings"""
        try:
            # Extract embedding from uploaded image
            query_embedding = self.extract_embedding(uploaded_img_path)
            if query_embedding is None:
                return []
            if self.emb_matrix is not None and len(self.emb_ids) > 0:
                dists = np.sum((self.emb_matrix - query_embedding) ** 2, axis=1)
                cosines = self.emb_matrix @ query_embedding
                idxs = np.argsort(dists)[:top_k]
                results = []
                for i in idxs:
                    cid = self.emb_ids[i]
                    results.append({
                        'cat_id': cid,
                        'l2_distance': float(dists[i]),
                        'cosine': float(cosines[i]),
                        'metadata': self.cat_metadata.get(cid, {})
                    })
                return results
            else:
                results = []
                for cat_id, db_embedding in self.cat_embeddings.items():
                    dist = float(np.sum((query_embedding - db_embedding) ** 2))
                    cos = float(np.dot(query_embedding, db_embedding))
                    results.append({
                        'cat_id': cat_id,
                        'l2_distance': dist,
                        'cosine': cos,
                        'metadata': self.cat_metadata.get(cat_id, {})
                    })
                results.sort(key=lambda x: x['l2_distance'])
                return results[:top_k]

        except Exception as e:
            logger.error(f"❌ Error finding matches: {e}")
            return []

    def find_matches_multi(self, img_paths, top_k=10):
        try:
            query_embs = []
            for p in img_paths:
                emb = self.extract_embedding(p)
                if emb is not None:
                    query_embs.append(emb)
            if not query_embs:
                return []
            Q = np.stack(query_embs, axis=0)
            if self.emb_matrix is not None and len(self.emb_ids) > 0:
                dists = np.sum((self.emb_matrix[None, :, :] - Q[:, None, :]) ** 2, axis=2)
                cosines = self.emb_matrix[None, :, :] @ Q[:, :, None]
                cosines = np.squeeze(cosines, axis=2)
                mean_dists = np.mean(dists, axis=0)
                mean_cos = np.mean(cosines, axis=0)
                idxs = np.argsort(mean_dists)[:top_k]
                results = []
                for i in idxs:
                    cid = self.emb_ids[i]
                    results.append({
                        'cat_id': cid,
                        'l2_distance': float(mean_dists[i]),
                        'cosine': float(mean_cos[i]),
                        'metadata': self.cat_metadata.get(cid, {})
                    })
                return results
            else:
                agg = []
                for cat_id, db_embedding in self.cat_embeddings.items():
                    d = [float(np.sum((qe - db_embedding) ** 2)) for qe in query_embs]
                    c = [float(np.dot(qe, db_embedding)) for qe in query_embs]
                    agg.append({
                        'cat_id': cat_id,
                        'l2_distance': float(np.mean(d)),
                        'cosine': float(np.mean(c)),
                        'metadata': self.cat_metadata.get(cat_id, {})
                    })
                agg.sort(key=lambda x: x['l2_distance'])
                return agg[:top_k]
        except Exception as e:
            logger.error(f"❌ Error finding matches (multi): {e}")
            return []

# Initialize the recognition system
recognition_system = PetRecognitionSystem()

def initialize_system():
    """Initialize the recognition system"""
    logger.info("🚀 Initializing Pet ID Malaysia System...")

    # Load model
    if not recognition_system.load_model():
        logger.error("❌ Failed to load model. System cannot start.")
        return False
    # Warm up model to avoid first-request latency
    try:
        if recognition_system.embedding_model is not None:
            dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            dummy = tf.keras.applications.efficientnet_v2.preprocess_input(dummy)
            
            if hasattr(recognition_system.embedding_model, 'predict'):
                _ = recognition_system.embedding_model.predict(dummy)
            else:
                _ = recognition_system.embedding_model(dummy)
                
            logger.info("✅ Model warm-up completed")
    except Exception as e:
        logger.warning(f"⚠️ Model warm-up skipped: {e}")

    # Load Haar Cascade
    recognition_system.load_haar_cascade()

    # Load database
    if not recognition_system.load_database():
        logger.error("❌ Failed to load database. System cannot start.")
        return False

    logger.info("✅ System initialized successfully!")
    return True

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask routes
@app.route('/')
def index():
    """Homepage with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files and 'file[]' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        files = request.files.getlist('file') or request.files.getlist('file[]')
        files = [f for f in files if f and f.filename]
        if not files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        saved_paths = []
        uploaded_names = []
        count = 0
        for file in files:
            if count >= 5:
                break
            if not allowed_file(file.filename):
                continue
            filename = secure_filename(file.filename or "upload.jpg")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"upload_{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            saved_paths.append(filepath)
            uploaded_names.append(filename)
            count += 1
        if not saved_paths:
            flash('Invalid file type. Please upload JPG, JPEG, or PNG files only.', 'error')
            return redirect(url_for('index'))

        logger.info(f"📤 Uploaded {len(saved_paths)} file(s): {uploaded_names}")

        if len(saved_paths) == 1:
            matches = recognition_system.find_matches(saved_paths[0], top_k=10)
        else:
            matches = recognition_system.find_matches_multi(saved_paths, top_k=10)

        results = {
            'uploaded_image': uploaded_names[0],
            'uploaded_images': uploaded_names,
            'matches': matches,
            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        logger.info(f"🔍 Found {len(matches)} matches for {len(saved_paths)} image(s)")
        return render_template('results.html', results=results)

    except Exception as e:
        logger.error(f"❌ Error processing upload: {e}")
        flash('An error occurred while processing your image. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

@app.route('/lost-cat')
def lost_cat():
    """Lost cat report page"""
    today = datetime.now().strftime("%Y-%m-%d")
    return render_template('lost_cat.html', today=today)

@app.route('/found-cat')
def found_cat():
    """Found cat upload page"""
    return render_template('found_cat.html')

@app.route('/upload-found-cat', methods=['POST'])
def upload_found_cat():
    """Handle found cat upload and add to dataset"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('found_cat'))

        file = request.files['file']

        # Check if file is empty
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('found_cat'))

        # Get additional information from form
        cat_name = request.form.get('cat_name', '').strip()
        cat_gender = request.form.get('cat_gender', 'Unknown')
        age_years = request.form.get('age_years', '').strip()
        age_months = request.form.get('age_months', '').strip()
        vaccinated = request.form.get('vaccinated', 'Unknown')
        dewormed = request.form.get('dewormed', 'Unknown')
        spayed = request.form.get('spayed', 'Unknown')
        condition = request.form.get('condition', 'Healthy')
        body_size = request.form.get('body_size', 'Unknown')
        fur_length = request.form.get('fur_length', 'Unknown')
        color = request.form.get('color', '').strip()
        breed = request.form.get('breed', 'Unknown')
        location = request.form.get('location', 'Unknown')
        description = request.form.get('description', '')
        contact_info = request.form.get('contact_info', '')
        date_found = request.form.get('date_found', datetime.now().strftime("%Y-%m-%d"))

        # Geocode location
        lat, lng = None, None
        if location and location != 'Unknown':
            try:
                loc = geolocator.geocode(location + ", Malaysia")
                if loc:
                    lat = loc.latitude
                    lng = loc.longitude
            except Exception as e:
                logger.warning(f"⚠️ Geocoding failed for {location}: {e}")

        # Check if file is allowed
        if file and allowed_file(file.filename):
            # Save found cat image
            filename = secure_filename(file.filename or "found.jpg")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"found_cat_{timestamp}_{filename}"
            filepath = os.path.join(FOUND_CATS_PATH, filename)
            file.save(filepath)

            logger.info(f"📤 Found cat uploaded: {filename}")

            # Extract embedding for the found cat
            embedding = recognition_system.extract_embedding(filepath)
            
            if embedding is not None:
                # Generate unique cat ID
                cat_id = f"found_{timestamp}"
                
                # Add to embeddings and metadata
                recognition_system.cat_embeddings[cat_id] = embedding
                # Build profile summary
                try:
                    y = int(age_years) if age_years else 0
                    m = int(age_months) if age_months else 0
                except Exception:
                    y, m = 0, 0
                age_str = []
                if y:
                    age_str.append(f"{y} Year{'s' if y!=1 else ''}")
                if m:
                    age_str.append(f"{m} Month{'s' if m!=1 else ''}")
                profile_summary = f"{cat_gender}, " + (" ".join(age_str) if age_str else "Age Unknown")

                recognition_system.cat_metadata[cat_id] = {
                    'filename': filename,
                    'image_path': filepath,
                    'id': cat_id,
                    'name': cat_name if cat_name else f"Found Cat {timestamp}",
                    'location': location,
                    'description': description,
                    'contact_info': contact_info,
                    'date_found': date_found,
                    'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'status': 'found',
                    'lat': lat,
                    'lng': lng,
                    'gender': cat_gender,
                    'age_years': y,
                    'age_months': m,
                    'vaccinated': vaccinated,
                    'dewormed': dewormed,
                    'spayed': spayed,
                    'condition': condition,
                    'body_size': body_size,
                    'fur_length': fur_length,
                    'color': color,
                    'breed': breed,
                    'profile_summary': profile_summary
                }

                # Save found cats metadata separately
                save_found_cats_metadata()

                # Update the embeddings cache with new data
                np.save(EMBEDDINGS_CACHE, np.array(recognition_system.cat_embeddings, dtype=object))
                with open(METADATA_CACHE, 'w') as f:
                    json.dump(recognition_system.cat_metadata, f)

                logger.info(f"✅ Found cat added to database: {cat_id}")
                
                # Return success page with details
                return render_template('found_cat_success.html', 
                                     filename=filename, 
                                     cat_id=cat_id,
                                     location=location,
                                     date_found=date_found)
            else:
                flash('Error processing the image. Please try again with a clearer photo.', 'error')
                return redirect(url_for('found_cat'))
        else:
            flash('Invalid file type. Please upload JPG, JPEG, or PNG files only.', 'error')
            return redirect(url_for('found_cat'))

    except Exception as e:
        logger.error(f"❌ Error processing found cat upload: {e}")
        flash('An error occurred while processing your image. Please try again.', 'error')
        return redirect(url_for('found_cat'))

@app.route('/view-found-cats')
def view_found_cats():
    """View all found cats in the database"""
    try:
        found_cats = {k: v for k, v in recognition_system.cat_metadata.items() 
                     if v.get('status') == 'found' and not v.get('reunited', False)}

        sort = request.args.get('sort', 'upload_time')
        dir_ = request.args.get('dir', 'desc')
        page = int(request.args.get('page', '1') or '1')
        page_size = int(request.args.get('page_size', '24') or '24')
        q = (request.args.get('q', '') or '').strip().lower()
        breed1 = (request.args.get('breed1', '') or '').strip()
        breed2 = (request.args.get('breed2', '') or '').strip()
        age = (request.args.get('age', 'All') or 'All').strip()
        loc = (request.args.get('loc', '') or '').strip().lower()

        def sort_key(x):
            if sort == 'date_found':
                return x.get('date_found', '')
            if sort == 'name':
                return (x.get('name') or '').lower()
            if sort == 'location':
                return (x.get('location') or '').lower()
            if sort == 'id':
                return x.get('id', '')
            return x.get('upload_time', '')

        def age_match(x):
            if age == 'All':
                return True
            y = x.get('age_years') or 0
            m = x.get('age_months') or 0
            total = y * 12 + m
            if total == 0:
                return False
            if age == 'Kitten':
                return total < 12
            if age == 'Young':
                return 12 <= total < 36
            if age == 'Adult':
                return 36 <= total < 96
            if age == 'Senior':
                return total >= 96
            return True

        def matches(x):
            if q:
                hay = ' '.join([(x.get('name') or ''), (x.get('id') or ''), (x.get('location') or ''), (x.get('description') or '')]).lower()
                if q not in hay:
                    return False
            if breed1:
                if (x.get('breed') or '') != breed1:
                    return False
            if breed2:
                if (x.get('breed') or '') != breed2:
                    return False
            if loc:
                if loc not in (x.get('location') or '').lower():
                    return False
            if not age_match(x):
                return False
            return True

        filtered = [v for v in found_cats.values() if matches(v)]
        found_cats_list = sorted(filtered, key=sort_key, reverse=(dir_ == 'desc'))

        total_count = len(found_cats_list)
        total_pages = max(1, (total_count + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        end = start + page_size
        found_cats_page = found_cats_list[start:end]

        return render_template(
            'view_found_cats.html',
            found_cats=found_cats_page,
            sort=sort,
            dir=dir_,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            total_count=total_count,
            q=q,
            breed1=breed1,
            breed2=breed2,
            age=age,
            loc=loc,
        )
    except Exception as e:
        logger.error(f"❌ Error viewing found cats: {e}")
        flash('An error occurred while loading found cats.', 'error')
        return redirect(url_for('index'))

@app.route('/reunited-cats')
def reunited_cats():
    """View all reunited cats"""
    try:
        # Filter reunited cats from metadata
        reunited_cats = {k: v for k, v in recognition_system.cat_metadata.items() 
                        if v.get('reunited', False)}
        
        # Sort by reunited date (newest first)
        reunited_cats_list = sorted(reunited_cats.values(), 
                                   key=lambda x: x.get('reunited_date', ''), 
                                   reverse=True)
        
        return render_template('reunited_cats.html', reunited_cats=reunited_cats_list)
    except Exception as e:
        logger.error(f"❌ Error viewing reunited cats: {e}")
        flash('An error occurred while loading reunited cats.', 'error')
        return redirect(url_for('index'))

@app.route('/api/found-cats-map')
def api_found_cats_map():
    """API to serve found cats data for the map"""
    try:
        # Filter found cats that have coordinates
        found_cats = []
        for cat in recognition_system.cat_metadata.values():
            if cat.get('status') == 'found' and not cat.get('reunited', False):
                if cat.get('lat') and cat.get('lng'):
                    found_cats.append({
                        'id': cat.get('id'),
                        'name': cat.get('name'),
                        'lat': cat.get('lat'),
                        'lng': cat.get('lng'),
                        'image_url': url_for('serve_cat_image', cat_id=cat.get('id')),
                        'location': cat.get('location'),
                        'date_found': cat.get('date_found'),
                        'description': cat.get('description', ''),
                        'contact_info': cat.get('contact_info', '')
                    })
        return jsonify(found_cats)
    except Exception as e:
        logger.error(f"❌ Error serving map data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/mark-reunited/<cat_id>', methods=['POST'])
def mark_reunited(cat_id):
    """Mark a cat as reunited with owner"""
    try:
        if cat_id not in recognition_system.cat_metadata:
            flash('Cat not found in database.', 'error')
            return redirect(url_for('view_found_cats'))
        
        # Get reunion details from form
        owner_name = request.form.get('owner_name', 'Anonymous')
        reunion_story = request.form.get('reunion_story', '')
        
        # Update metadata
        recognition_system.cat_metadata[cat_id]['reunited'] = True
        recognition_system.cat_metadata[cat_id]['reunited_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        recognition_system.cat_metadata[cat_id]['owner_name'] = owner_name
        recognition_system.cat_metadata[cat_id]['reunion_story'] = reunion_story
        
        # Save updated metadata
        with open(METADATA_CACHE, 'w') as f:
            json.dump(recognition_system.cat_metadata, f)
        
        # Update found cats metadata
        save_found_cats_metadata()
        
        logger.info(f"✅ Cat {cat_id} marked as reunited")
        flash('Congratulations! The cat has been marked as reunited. Thank you for helping!', 'success')
        
        return redirect(url_for('reunited_cats'))
        
    except Exception as e:
        logger.error(f"❌ Error marking cat as reunited: {e}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('view_found_cats'))

def save_found_cats_metadata():
    """Save found cats metadata to separate file"""
    try:
        found_cats = {k: v for k, v in recognition_system.cat_metadata.items() 
                     if v.get('status') == 'found'}
        with open(FOUND_CATS_METADATA, 'w') as f:
            json.dump(found_cats, f, indent=2)
        logger.info(f"✅ Saved {len(found_cats)} found cats to metadata file")
    except Exception as e:
        logger.error(f"❌ Error saving found cats metadata: {e}")

@app.route('/upload-lost-cat', methods=['POST'])
def upload_lost_cat():
    """Handle lost cat report and add to database"""
    try:
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('lost_cat'))
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('lost_cat'))

        # Get profile fields
        cat_name = request.form.get('cat_name', '').strip()
        cat_gender = request.form.get('cat_gender', 'Unknown')
        age_years = request.form.get('age_years', '').strip()
        age_months = request.form.get('age_months', '').strip()
        vaccinated = request.form.get('vaccinated', 'Unknown')
        dewormed = request.form.get('dewormed', 'Unknown')
        spayed = request.form.get('spayed', 'Unknown')
        condition = request.form.get('condition', 'Healthy')
        body_size = request.form.get('body_size', 'Unknown')
        fur_length = request.form.get('fur_length', 'Unknown')
        color = request.form.get('color', '').strip()
        breed = request.form.get('breed', 'Unknown')
        location = request.form.get('location', 'Unknown')
        description = request.form.get('description', '')
        contact_info = request.form.get('contact_info', '')
        date_lost = request.form.get('date_lost', datetime.now().strftime("%Y-%m-%d"))

        # Geocode location
        lat, lng = None, None
        if location and location != 'Unknown':
            try:
                loc = geolocator.geocode(location + ", Malaysia")
                if loc:
                    lat = loc.latitude
                    lng = loc.longitude
            except Exception as e:
                logger.warning(f"⚠️ Geocoding failed for {location}: {e}")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename or "lost.jpg")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lost_cat_{timestamp}_{filename}"
            filepath = os.path.join(LOST_CATS_PATH, filename)
            file.save(filepath)

            embedding = recognition_system.extract_embedding(filepath)
            if embedding is not None:
                cat_id = f"lost_{timestamp}"
                recognition_system.cat_embeddings[cat_id] = embedding

                # Build profile summary
                try:
                    y = int(age_years) if age_years else 0
                    m = int(age_months) if age_months else 0
                except Exception:
                    y, m = 0, 0
                age_str = []
                if y:
                    age_str.append(f"{y} Year{'s' if y!=1 else ''}")
                if m:
                    age_str.append(f"{m} Month{'s' if m!=1 else ''}")
                profile_summary = f"{cat_gender}, " + (" ".join(age_str) if age_str else "Age Unknown")

                recognition_system.cat_metadata[cat_id] = {
                    'filename': filename,
                    'image_path': filepath,
                    'id': cat_id,
                    'name': cat_name if cat_name else f"Lost Cat {timestamp}",
                    'location': location,
                    'description': description,
                    'contact_info': contact_info,
                    'date_lost': date_lost,
                    'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'status': 'lost',
                    'lat': lat,
                    'lng': lng,
                    'gender': cat_gender,
                    'age_years': y,
                    'age_months': m,
                    'vaccinated': vaccinated,
                    'dewormed': dewormed,
                    'spayed': spayed,
                    'condition': condition,
                    'body_size': body_size,
                    'fur_length': fur_length,
                    'color': color,
                    'breed': breed,
                    'profile_summary': profile_summary
                }

                np.save(EMBEDDINGS_CACHE, np.array(recognition_system.cat_embeddings, dtype=object))
                with open(METADATA_CACHE, 'w') as f:
                    json.dump(recognition_system.cat_metadata, f)

                logger.info(f"✅ Lost cat reported: {cat_id}")
                flash('Thank you! Your lost cat report has been added to our database.', 'success')
                return redirect(url_for('view_lost_cats'))
            else:
                flash('Error processing the image. Please try again with a clearer photo.', 'error')
                return redirect(url_for('lost_cat'))

        flash('Invalid file type. Please upload JPG, JPEG, or PNG files only.', 'error')
        return redirect(url_for('lost_cat'))
    except Exception as e:
        logger.error(f"❌ Error processing lost cat report: {e}")
        flash('An error occurred while processing your image. Please try again.', 'error')
        return redirect(url_for('lost_cat'))

@app.route('/view-lost-cats')
def view_lost_cats():
    """View all lost cats reported by users"""
    try:
        lost_cats = {k: v for k, v in recognition_system.cat_metadata.items()
                     if v.get('status') == 'lost' and not v.get('reunited', False)}
        sort = request.args.get('sort', 'upload_time')
        dir_ = request.args.get('dir', 'desc')
        page = int(request.args.get('page', '1') or '1')
        page_size = int(request.args.get('page_size', '24') or '24')
        q = (request.args.get('q', '') or '').strip().lower()
        breed1 = (request.args.get('breed1', '') or '').strip()
        breed2 = (request.args.get('breed2', '') or '').strip()
        age = (request.args.get('age', 'All') or 'All').strip()
        loc = (request.args.get('loc', '') or '').strip().lower()
        def sort_key(x):
            if sort == 'date_lost':
                return x.get('date_lost', '')
            if sort == 'name':
                return (x.get('name') or '').lower()
            if sort == 'location':
                return (x.get('location') or '').lower()
            if sort == 'id':
                return x.get('id', '')
            return x.get('upload_time', '')
        def age_match(x):
            if age == 'All':
                return True
            y = x.get('age_years') or 0
            m = x.get('age_months') or 0
            total = y * 12 + m
            if total == 0:
                return False
            if age == 'Kitten':
                return total < 12
            if age == 'Young':
                return 12 <= total < 36
            if age == 'Adult':
                return 36 <= total < 96
            if age == 'Senior':
                return total >= 96
            return True
        def matches(x):
            if q:
                hay = ' '.join([(x.get('name') or ''), (x.get('id') or ''), (x.get('location') or ''), (x.get('description') or '')]).lower()
                if q not in hay:
                    return False
            if breed1:
                if (x.get('breed') or '') != breed1:
                    return False
            if breed2:
                if (x.get('breed') or '') != breed2:
                    return False
            if loc:
                if loc not in (x.get('location') or '').lower():
                    return False
            if not age_match(x):
                return False
            return True
        filtered = [v for v in lost_cats.values() if matches(v)]
        lost_cats_list = sorted(filtered, key=sort_key, reverse=(dir_=='desc'))

        total_count = len(lost_cats_list)
        total_pages = max(1, (total_count + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        end = start + page_size
        lost_cats_page = lost_cats_list[start:end]

        return render_template(
            'view_lost_cats.html',
            lost_cats=lost_cats_page,
            sort=sort,
            dir=dir_,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            total_count=total_count,
            q=q,
            breed1=breed1,
            breed2=breed2,
            age=age,
            loc=loc,
        )
    except Exception as e:
        logger.error(f"❌ Error viewing lost cats: {e}")
        flash('An error occurred while loading lost cats.', 'error')
        return redirect(url_for('index'))

# Route to serve images from dataset
@app.route('/dataset_images/<path:filename>')
def serve_dataset_image(filename):
    """Serve images from the cat dataset directory"""
    try:
        return send_from_directory(DATASET_PATH, filename)
    except FileNotFoundError:
        abort(404)

@app.route('/uploaded_images/<path:filename>')
def serve_uploaded_image(filename):
    """Serve images uploaded by users (stored in static/uploads)"""
    upload_dir = app.config['UPLOAD_FOLDER']
    try:
        return send_from_directory(upload_dir, filename)
    except FileNotFoundError:
        abort(404)

@app.route('/cat_image/<cat_id>')
def serve_cat_image(cat_id):
    """Serve cat image (supports both dataset and found cats)"""
    metadata = recognition_system.cat_metadata.get(cat_id)
    if not metadata:
        abort(404)

    image_path = metadata.get('image_path')
    if not image_path or not os.path.exists(image_path):
        abort(404)

    return send_file(image_path)

@app.route('/found_cat_images/<path:filename>')
def serve_found_cat_image(filename):
    """Serve images from the found cats directory"""
    try:
        return send_from_directory(FOUND_CATS_PATH, filename)
    except FileNotFoundError:
        abort(404)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Please upload an image smaller than 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    logger.error(f"❌ Internal server error: {e}")
    flash('An internal error occurred. Please try again later.', 'error')
    return redirect(url_for('index'))

# Maintenance & API endpoints
@app.route('/api/health')
def api_health():
    try:
        status = {
            'model_loaded': recognition_system.embedding_model is not None,
            'embeddings_count': len(recognition_system.cat_embeddings),
            'dataset_path': DATASET_PATH,
            'img_size': IMG_SIZE
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reindex', methods=['POST'])
def api_reindex():
    try:
        if not recognition_system.load_database():
            return jsonify({'status': 'failed'}), 500
        return jsonify({'status': 'ok', 'embeddings_count': len(recognition_system.cat_embeddings)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/add', methods=['POST'])
def api_add():
    try:
        file = request.files.get('file')
        cat_id = request.form.get('cat_id')
        if not file or not cat_id:
            return jsonify({'error': 'file and cat_id required'}), 400
        filename = secure_filename(file.filename or "added.jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        emb = recognition_system.extract_embedding(filepath)
        if emb is None:
            return jsonify({'error': 'embedding failed'}), 500
        recognition_system.cat_embeddings[cat_id] = emb
        recognition_system.cat_metadata[cat_id] = {
            'filename': filename,
            'image_path': filepath,
            'id': cat_id,
            'name': f'Cat {cat_id}'
        }
        np.save(EMBEDDINGS_CACHE, np.array(recognition_system.cat_embeddings, dtype=object))
        with open(METADATA_CACHE, 'w') as f:
            json.dump(recognition_system.cat_metadata, f)
        recognition_system.build_matrix()
        return jsonify({'status': 'ok', 'cat_id': cat_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize system on startup
if __name__ == '__main__':
    if initialize_system():
        logger.info("🌟 Starting Pet ID Malaysia Web Application...")
        logger.info("🌐 Open http://localhost:5000 in your browser")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        logger.error("❌ Failed to initialize system. Exiting...")
        exit(1)
