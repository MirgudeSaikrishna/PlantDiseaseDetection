"""
Utility functions for AgriDetect AI
Author: [Your Name]
Date: [Current Date]
"""

import os
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, current_app
from PIL import Image
import cv2
import numpy as np

def setup_logging():
    """Setup application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def generate_unique_filename(original_filename):
    """Generate unique filename to avoid conflicts"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    extension = original_filename.rsplit('.', 1)[1].lower()
    return f"{timestamp}_{unique_id}.{extension}"

def validate_image_file(file):
    """Validate uploaded image file"""
    if not file or file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, "Invalid file type. Only images are allowed."
    
    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > current_app.config['MAX_CONTENT_LENGTH']:
        return False, "File too large. Maximum size is 16MB."
    
    return True, "Valid file"

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input"""
    try:
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array, True
    
    except Exception as e:
        logging.error(f"Error preprocessing image: {str(e)}")
        return None, False

def calculate_image_quality(image_path):
    """Calculate image quality score"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 scale
        quality_score = min(laplacian_var / 1000.0, 1.0)
        
        return quality_score
    
    except Exception as e:
        logging.error(f"Error calculating image quality: {str(e)}")
        return 0.0

def rate_limit(max_requests=100, window_seconds=3600):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Simple in-memory rate limiting (use Redis in production)
            client_ip = request.remote_addr
            current_time = datetime.now()
            
            # This is a simplified implementation
            # In production, use Redis or similar for distributed rate limiting
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def validate_api_key(api_key):
    """Validate API key (simplified implementation)"""
    # In production, implement proper API key validation
    valid_keys = current_app.config.get('API_KEYS', [])
    return api_key in valid_keys

def require_api_key(f):
    """Decorator to require API key for certain endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or not validate_api_key(api_key):
            return jsonify({'error': 'Invalid or missing API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

def cache_result(expiry_seconds=300):
    """Cache function result decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Simple in-memory caching (use Redis in production)
            cache_key = f"{f.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Check cache
            if hasattr(current_app, 'cache'):
                cached_result = current_app.cache.get(cache_key)
                if cached_result:
                    return cached_result
            
            # Execute function
            result = f(*args, **kwargs)
            
            # Store in cache
            if hasattr(current_app, 'cache'):
                current_app.cache.set(cache_key, result, timeout=expiry_seconds)
            
            return result
        return decorated_function
    return decorator

def format_disease_name(disease_name):
    """Format disease name for display"""
    if not disease_name:
        return "Unknown"
    
    # Replace underscores with spaces
    formatted = disease_name.replace('_', ' ')
    
    # Capitalize words
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    
    return formatted

def calculate_confidence_level(confidence_score):
    """Calculate confidence level based on score"""
    if confidence_score >= 0.9:
        return "Very High"
    elif confidence_score >= 0.8:
        return "High"
    elif confidence_score >= 0.7:
        return "Medium"
    elif confidence_score >= 0.6:
        return "Low"
    else:
        return "Very Low"

def generate_detection_id():
    """Generate unique detection ID"""
    return str(uuid.uuid4())

def sanitize_filename(filename):
    """Sanitize filename for safe storage"""
    # Remove or replace unsafe characters
    safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
    return safe_filename

def get_file_hash(file_path):
    """Calculate file hash for integrity checking"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def cleanup_old_files(directory, max_age_days=7):
    """Clean up old files from directory"""
    try:
        current_time = datetime.now()
        max_age = timedelta(days=max_age_days)
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_age = current_time - datetime.fromtimestamp(os.path.getctime(file_path))
                if file_age > max_age:
                    os.remove(file_path)
                    logging.info(f"Cleaned up old file: {filename}")
    
    except Exception as e:
        logging.error(f"Error cleaning up files: {str(e)}")

def validate_coordinates(lat, lon):
    """Validate latitude and longitude coordinates"""
    try:
        lat = float(lat)
        lon = float(lon)
        return -90 <= lat <= 90 and -180 <= lon <= 180
    except (ValueError, TypeError):
        return False

def format_timestamp(timestamp):
    """Format timestamp for display"""
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
    
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')

def get_system_info():
    """Get system information for monitoring"""
    import psutil
    import platform
    
    return {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent
    }

def create_response(data, status_code=200, message=None):
    """Create standardized API response"""
    response = {
        'success': status_code < 400,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    
    if message:
        response['message'] = message
    
    return jsonify(response), status_code

def handle_exception(e):
    """Handle exceptions and return appropriate response"""
    logging.error(f"Exception occurred: {str(e)}")
    
    if current_app.debug:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500
    else:
        return jsonify({
            'error': 'Internal server error'
        }), 500
