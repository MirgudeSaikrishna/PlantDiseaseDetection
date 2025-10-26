"""
API Tests for AgriDetect AI
Author: [Your Name]
Date: [Current Date]
"""

import pytest
import json
import os
import tempfile
from PIL import Image
import numpy as np
from app import app, model, analyzer

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='green')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    img.save(temp_file.name)
    temp_file.close()
    yield temp_file.name
    os.unlink(temp_file.name)

@pytest.fixture
def sample_image_file():
    """Create a sample image file for upload"""
    img = Image.new('RGB', (224, 224), color='red')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    img.save(temp_file.name)
    temp_file.close()
    
    with open(temp_file.name, 'rb') as f:
        yield f
    
    os.unlink(temp_file.name)

class TestDetectionAPI:
    """Test disease detection API endpoints"""
    
    def test_detect_endpoint_success(self, client, sample_image_file):
        """Test successful disease detection"""
        response = client.post('/api/detect', 
                             data={'image': sample_image_file},
                             content_type='multipart/form-data')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'disease_name' in data
        assert 'confidence' in data
        assert 'severity_score' in data
        assert 'health_score' in data
        assert 'treatment_plan' in data
        assert 'recommendations' in data
        
        # Validate data types
        assert isinstance(data['disease_name'], str)
        assert isinstance(data['confidence'], (int, float))
        assert isinstance(data['severity_score'], (int, float))
        assert isinstance(data['health_score'], (int, float))
        assert isinstance(data['treatment_plan'], dict)
        assert isinstance(data['recommendations'], list)
    
    def test_detect_endpoint_no_image(self, client):
        """Test detection endpoint without image"""
        response = client.post('/api/detect')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No image provided' in data['error']
    
    def test_detect_endpoint_empty_filename(self, client):
        """Test detection endpoint with empty filename"""
        response = client.post('/api/detect',
                             data={'image': (io.BytesIO(b''), '')},
                             content_type='multipart/form-data')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No image selected' in data['error']
    
    def test_detect_with_location(self, client, sample_image_file):
        """Test detection with location parameter"""
        response = client.post('/api/detect',
                             data={
                                 'image': sample_image_file,
                                 'location': 'Test City, Test Country'
                             },
                             content_type='multipart/form-data')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'weather_data' in data or data.get('weather_data') is None
    
    def test_batch_detect_endpoint(self, client):
        """Test batch detection endpoint"""
        # Create multiple test images
        images = []
        for i in range(3):
            img = Image.new('RGB', (224, 224), color=(i*50, 100, 150))
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            img.save(temp_file.name)
            temp_file.close()
            images.append(open(temp_file.name, 'rb'))
        
        try:
            response = client.post('/api/batch-detect',
                                 data={'images': images},
                                 content_type='multipart/form-data')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'results' in data
            assert len(data['results']) == 3
            
            for result in data['results']:
                assert 'disease_name' in result
                assert 'confidence' in result
        
        finally:
            for img_file in images:
                img_file.close()
                os.unlink(img_file.name)
    
    def test_batch_detect_too_many_images(self, client):
        """Test batch detection with too many images"""
        images = []
        for i in range(15):  # More than the 10 image limit
            img = Image.new('RGB', (224, 224), color=(i*10, 100, 150))
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            img.save(temp_file.name)
            temp_file.close()
            images.append(open(temp_file.name, 'rb'))
        
        try:
            response = client.post('/api/batch-detect',
                                 data={'images': images},
                                 content_type='multipart/form-data')
            
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert 'Maximum 10 images' in data['error']
        
        finally:
            for img_file in images:
                img_file.close()
                os.unlink(img_file.name)

class TestAnalyticsAPI:
    """Test analytics API endpoints"""
    
    def test_health_report_endpoint(self, client):
        """Test health report endpoint"""
        response = client.get('/api/health-report')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Should return a report or message about no data
        assert isinstance(data, dict)
    
    def test_health_report_with_days(self, client):
        """Test health report with custom days parameter"""
        response = client.get('/api/health-report?days=7')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, dict)
    
    def test_history_endpoint(self, client):
        """Test detection history endpoint"""
        response = client.get('/api/history')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
    
    def test_history_with_days(self, client):
        """Test history with custom days parameter"""
        response = client.get('/api/history?days=14')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)

class TestLegacyEndpoints:
    """Test legacy endpoints for backward compatibility"""
    
    def test_home_page(self, client):
        """Test home page"""
        response = client.get('/')
        assert response.status_code == 200
    
    def test_contact_page(self, client):
        """Test contact page"""
        response = client.get('/contact')
        assert response.status_code == 200
    
    def test_ai_engine_page(self, client):
        """Test AI engine page"""
        response = client.get('/index')
        assert response.status_code == 200
    
    def test_market_page(self, client):
        """Test market page"""
        response = client.get('/market')
        assert response.status_code == 200
    
    def test_submit_endpoint_post(self, client, sample_image_file):
        """Test submit endpoint with POST"""
        response = client.post('/submit',
                             data={'image': sample_image_file},
                             content_type='multipart/form-data')
        
        assert response.status_code == 200
    
    def test_submit_endpoint_get(self, client):
        """Test submit endpoint with GET"""
        response = client.get('/submit')
        assert response.status_code == 405  # Method not allowed

class TestModelIntegration:
    """Test model integration and prediction functions"""
    
    def test_prediction_function(self, sample_image):
        """Test basic prediction function"""
        from app import prediction
        
        result = prediction(sample_image)
        assert isinstance(result, int)
        assert 0 <= result < 39  # Should be a valid class index
    
    def test_advanced_prediction_function(self, sample_image):
        """Test advanced prediction function"""
        from app import advanced_prediction
        
        result = advanced_prediction(sample_image)
        assert isinstance(result, dict)
        assert 'disease_name' in result
        assert 'confidence' in result
        assert 'severity_score' in result
        assert 'health_score' in result
    
    def test_model_loading(self):
        """Test that model loads correctly"""
        assert model is not None
        assert hasattr(model, 'eval')
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze_plant_health')

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_image_format(self, client):
        """Test with invalid image format"""
        # Create a text file instead of image
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        temp_file.write(b'This is not an image')
        temp_file.close()
        
        try:
            with open(temp_file.name, 'rb') as f:
                response = client.post('/api/detect',
                                     data={'image': f},
                                     content_type='multipart/form-data')
            
            # Should handle gracefully
            assert response.status_code in [200, 400, 500]
        
        finally:
            os.unlink(temp_file.name)
    
    def test_large_image_file(self, client):
        """Test with very large image file"""
        # Create a large image
        large_img = Image.new('RGB', (4000, 4000), color='blue')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        large_img.save(temp_file.name, quality=95)
        temp_file.close()
        
        try:
            with open(temp_file.name, 'rb') as f:
                response = client.post('/api/detect',
                                     data={'image': f},
                                     content_type='multipart/form-data')
            
            # Should handle large files
            assert response.status_code in [200, 400, 413]  # 413 = Payload Too Large
        
        finally:
            os.unlink(temp_file.name)

if __name__ == '__main__':
    pytest.main([__file__])
