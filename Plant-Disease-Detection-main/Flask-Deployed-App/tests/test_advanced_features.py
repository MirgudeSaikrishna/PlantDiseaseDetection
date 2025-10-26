"""
Advanced Features Tests for AgriDetect AI
Author: [Your Name]
Date: [Current Date]
"""

import pytest
import tempfile
import os
import numpy as np
from PIL import Image
import torch
from advanced_features import AdvancedPlantAnalyzer, MobileOptimizer

class TestAdvancedPlantAnalyzer:
    """Test the AdvancedPlantAnalyzer class"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing"""
        class MockModel:
            def eval(self):
                pass
            
            def __call__(self, x):
                # Return mock predictions
                batch_size = x.shape[0]
                return torch.randn(batch_size, 39)
        
        return MockModel()
    
    @pytest.fixture
    def analyzer(self, mock_model):
        """Create analyzer instance for testing"""
        class_names = {i: f"disease_{i}" for i in range(39)}
        return AdvancedPlantAnalyzer(mock_model, class_names)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        img = Image.new('RGB', (224, 224), color='green')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        img.save(temp_file.name)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert hasattr(analyzer, 'model')
        assert hasattr(analyzer, 'class_names')
        assert hasattr(analyzer, 'confidence_threshold')
    
    def test_database_initialization(self, analyzer):
        """Test database initialization"""
        assert os.path.exists(analyzer.db_path)
    
    def test_analyze_plant_health(self, analyzer, sample_image):
        """Test plant health analysis"""
        result = analyzer.analyze_plant_health(sample_image)
        
        assert isinstance(result, dict)
        assert 'disease_name' in result
        assert 'confidence' in result
        assert 'severity_score' in result
        assert 'health_score' in result
        assert 'treatment_plan' in result
        assert 'recommendations' in result
        
        # Validate data types
        assert isinstance(result['disease_name'], str)
        assert isinstance(result['confidence'], (int, float))
        assert isinstance(result['severity_score'], (int, float))
        assert isinstance(result['health_score'], (int, float))
        assert isinstance(result['treatment_plan'], dict)
        assert isinstance(result['recommendations'], list)
    
    def test_batch_analyze(self, analyzer, sample_image):
        """Test batch analysis"""
        image_paths = [sample_image, sample_image, sample_image]
        results = analyzer.batch_analyze(image_paths)
        
        assert isinstance(results, list)
        assert len(results) == 3
        
        for result in results:
            assert isinstance(result, dict)
            assert 'disease_name' in result
    
    def test_severity_score_calculation(self, analyzer):
        """Test severity score calculation"""
        # Test with healthy image (green)
        healthy_img = Image.new('RGB', (224, 224), color='green')
        severity = analyzer._calculate_severity_score(healthy_img, 'healthy')
        assert 0 <= severity <= 1
        
        # Test with diseased image (brown spots)
        diseased_img = Image.new('RGB', (224, 224), color='brown')
        severity = analyzer._calculate_severity_score(diseased_img, 'spot_disease')
        assert 0 <= severity <= 1
    
    def test_health_score_calculation(self, analyzer):
        """Test health score calculation"""
        # Test with high confidence and low severity
        health_score = analyzer._calculate_health_score(0.9, 0.1)
        assert health_score > 80
        
        # Test with low confidence
        health_score = analyzer._calculate_health_score(0.5, 0.1)
        assert health_score == 50.0
        
        # Test with high severity
        health_score = analyzer._calculate_health_score(0.9, 0.8)
        assert health_score < 30
    
    def test_treatment_plan_generation(self, analyzer):
        """Test treatment plan generation"""
        # Test healthy plant
        plan = analyzer._generate_treatment_plan('healthy_plant', 0.1)
        assert 'immediate_actions' in plan
        assert 'preventive_measures' in plan
        assert 'expected_recovery_time' in plan
        
        # Test diseased plant
        plan = analyzer._generate_treatment_plan('scab_disease', 0.5)
        assert len(plan['immediate_actions']) > 0
        assert len(plan['preventive_measures']) > 0
    
    def test_recommendations_generation(self, analyzer):
        """Test recommendations generation"""
        recommendations = analyzer._generate_recommendations('fungal_disease', 0.6)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check for severity indicator
        severity_indicators = [r for r in recommendations if '🔴' in r or '🟡' in r or '🟢' in r]
        assert len(severity_indicators) > 0
    
    def test_detection_storage(self, analyzer, sample_image):
        """Test detection storage in database"""
        # Clear any existing data
        import sqlite3
        conn = sqlite3.connect(analyzer.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM detection_history')
        conn.commit()
        conn.close()
        
        # Analyze and store
        result = analyzer.analyze_plant_health(sample_image, 'Test Location')
        
        # Check if stored in database
        conn = sqlite3.connect(analyzer.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM detection_history')
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count > 0
    
    def test_health_report_generation(self, analyzer):
        """Test health report generation"""
        report = analyzer.generate_health_report(30)
        
        assert isinstance(report, dict)
        assert 'total_detections' in report
        assert 'disease_distribution' in report
        assert 'average_confidence' in report
        assert 'average_severity' in report
    
    def test_detection_history_retrieval(self, analyzer):
        """Test detection history retrieval"""
        history_df = analyzer.get_detection_history(30)
        
        assert hasattr(history_df, 'columns')
        expected_columns = ['timestamp', 'image_path', 'disease_name', 'confidence', 'severity_score']
        for col in expected_columns:
            assert col in history_df.columns

class TestMobileOptimizer:
    """Test the MobileOptimizer class"""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        img = Image.new('RGB', (2000, 2000), color='red')  # Large image
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        img.save(temp_file.name, quality=95)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    def test_image_compression(self, sample_image):
        """Test image compression"""
        compressed_path = MobileOptimizer.compress_image(sample_image, max_size=1024, quality=85)
        
        assert os.path.exists(compressed_path)
        
        # Check file size reduction
        original_size = os.path.getsize(sample_image)
        compressed_size = os.path.getsize(compressed_path)
        
        assert compressed_size < original_size
        
        # Check image dimensions
        with Image.open(compressed_path) as img:
            assert max(img.size) <= 1024
        
        # Cleanup
        os.unlink(compressed_path)
    
    def test_mobile_response_generation(self):
        """Test mobile response generation"""
        analysis_result = {
            'disease_name': 'test_disease',
            'confidence': 0.85,
            'severity_score': 0.3,
            'health_score': 75.0,
            'treatment_plan': {
                'immediate_actions': ['action1', 'action2', 'action3'],
                'preventive_measures': ['prevent1', 'prevent2', 'prevent3']
            },
            'recommendations': ['rec1', 'rec2', 'rec3', 'rec4']
        }
        
        mobile_response = MobileOptimizer.generate_mobile_response(analysis_result)
        
        assert isinstance(mobile_response, dict)
        assert 'disease' in mobile_response
        assert 'confidence' in mobile_response
        assert 'severity' in mobile_response
        assert 'health_score' in mobile_response
        assert 'urgent' in mobile_response
        assert 'recommendations' in mobile_response
        assert 'treatment_plan' in mobile_response
        
        # Check data types and limits
        assert isinstance(mobile_response['confidence'], (int, float))
        assert isinstance(mobile_response['severity'], (int, float))
        assert isinstance(mobile_response['urgent'], bool)
        assert len(mobile_response['recommendations']) <= 3
        assert len(mobile_response['treatment_plan']['immediate']) <= 2
        assert len(mobile_response['treatment_plan']['prevention']) <= 2

class TestErrorHandling:
    """Test error handling in advanced features"""
    
    def test_analyzer_with_invalid_image(self, analyzer):
        """Test analyzer with invalid image"""
        # Create invalid image file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        temp_file.write(b'This is not an image')
        temp_file.close()
        
        try:
            result = analyzer.analyze_plant_health(temp_file.name)
            # Should handle gracefully or raise appropriate exception
            assert isinstance(result, dict) or isinstance(result, Exception)
        except Exception as e:
            assert isinstance(e, (FileNotFoundError, OSError, ValueError))
        finally:
            os.unlink(temp_file.name)
    
    def test_analyzer_with_nonexistent_file(self, analyzer):
        """Test analyzer with nonexistent file"""
        result = analyzer.analyze_plant_health('nonexistent_file.jpg')
        # Should handle gracefully
        assert isinstance(result, dict) or isinstance(result, Exception)
    
    def test_batch_analyze_with_errors(self, analyzer):
        """Test batch analyze with some errors"""
        # Mix of valid and invalid files
        valid_img = Image.new('RGB', (224, 224), color='green')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        valid_img.save(temp_file.name)
        temp_file.close()
        
        try:
            results = analyzer.batch_analyze([
                temp_file.name,
                'nonexistent_file.jpg',
                temp_file.name
            ])
            
            assert isinstance(results, list)
            assert len(results) == 3
            
            # Some results should have errors
            error_results = [r for r in results if 'error' in r]
            assert len(error_results) > 0
            
        finally:
            os.unlink(temp_file.name)

class TestPerformance:
    """Test performance of advanced features"""
    
    def test_analysis_performance(self, analyzer):
        """Test analysis performance"""
        import time
        
        # Create test image
        img = Image.new('RGB', (224, 224), color='green')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        img.save(temp_file.name)
        temp_file.close()
        
        try:
            start_time = time.time()
            result = analyzer.analyze_plant_health(temp_file.name)
            end_time = time.time()
            
            # Should complete within reasonable time (5 seconds)
            assert (end_time - start_time) < 5.0
            assert isinstance(result, dict)
        
        finally:
            os.unlink(temp_file.name)
    
    def test_batch_analysis_performance(self, analyzer):
        """Test batch analysis performance"""
        import time
        
        # Create multiple test images
        image_paths = []
        for i in range(5):
            img = Image.new('RGB', (224, 224), color=(i*50, 100, 150))
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            img.save(temp_file.name)
            temp_file.close()
            image_paths.append(temp_file.name)
        
        try:
            start_time = time.time()
            results = analyzer.batch_analyze(image_paths)
            end_time = time.time()
            
            # Should complete within reasonable time (10 seconds for 5 images)
            assert (end_time - start_time) < 10.0
            assert len(results) == 5
        
        finally:
            for path in image_paths:
                os.unlink(path)

if __name__ == '__main__':
    pytest.main([__file__])
