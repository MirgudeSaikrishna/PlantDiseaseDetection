"""
Configuration settings for AgriDetect AI
Author: [Your Name]
Date: [Current Date]
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # Database settings
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///plant_detection.db'
    
    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH') or 'plant_disease_model_1_latest.pt'
    MODEL_CONFIDENCE_THRESHOLD = float(os.environ.get('MODEL_CONFIDENCE_THRESHOLD', '0.7'))
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    
    # API settings
    API_RATE_LIMIT = os.environ.get('API_RATE_LIMIT') or '100 per hour'
    BATCH_SIZE_LIMIT = int(os.environ.get('BATCH_SIZE_LIMIT', '10'))
    
    # Weather API settings
    WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')
    WEATHER_API_URL = 'http://api.openweathermap.org/data/2.5/weather'
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE = os.environ.get('LOG_FILE') or 'app.log'
    
    # Cache settings
    CACHE_TYPE = os.environ.get('CACHE_TYPE') or 'simple'
    CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_DEFAULT_TIMEOUT', '300'))
    
    # Security settings
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        pass

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # Development-specific settings
    MODEL_CONFIDENCE_THRESHOLD = 0.6  # Lower threshold for development
    LOG_LEVEL = 'DEBUG'
    
    @staticmethod
    def init_app(app):
        Config.init_app(app)
        import logging
        logging.basicConfig(level=logging.DEBUG)

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    
    # Testing-specific settings
    DATABASE_URL = 'sqlite:///:memory:'
    MODEL_PATH = 'test_model.pt'
    UPLOAD_FOLDER = 'test_uploads'
    
    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Production-specific settings
    SESSION_COOKIE_SECURE = True
    LOG_LEVEL = 'WARNING'
    
    # Use environment variables for sensitive data
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set in production")
    
    @staticmethod
    def init_app(app):
        Config.init_app(app)
        
        # Log to file in production
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not app.debug and not app.testing:
            if not os.path.exists('logs'):
                os.mkdir('logs')
            
            file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            app.logger.setLevel(logging.INFO)
            app.logger.info('AgriDetect AI startup')

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    return config.get(os.environ.get('FLASK_ENV', 'development'), DevelopmentConfig)
