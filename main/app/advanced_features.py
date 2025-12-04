"""
Advanced Features for Plant Disease Detection System
Author: [Your Name]
Date: [Current Date]

This module contains advanced features that enhance the plant disease detection system:
- Batch processing
- Confidence scoring
- Disease severity assessment
- Treatment recommendations
- Historical tracking
- Weather integration
- Mobile optimization
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import requests
import json
from datetime import datetime, timedelta
import sqlite3
from typing import List, Dict, Tuple, Optional
import pandas as pd

class AdvancedPlantAnalyzer:
    """Advanced plant disease analyzer with additional features"""
    
    def __init__(self, model, class_names, confidence_threshold=0.7):
        self.model = model
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.db_path = "plant_detection_history.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for tracking detection history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                disease_name TEXT,
                confidence REAL,
                severity_score REAL,
                location TEXT,
                weather_condition TEXT,
                treatment_applied TEXT,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_plant_health(self, image_path: str, location: str = None) -> Dict:
        """
        Comprehensive plant health analysis with multiple metrics
        """
        # Load and preprocess image
        image = Image.open(image_path)
        processed_image = self._preprocess_image(image)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(processed_image)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        disease_name = self.class_names[predicted_class.item()]
        confidence_score = confidence.item()
        
        # Calculate additional metrics
        severity_score = self._calculate_severity_score(image, disease_name)
        health_score = self._calculate_health_score(confidence_score, severity_score)
        
        # Get weather data if location provided
        weather_data = self._get_weather_data(location) if location else None
        
        # Generate treatment recommendations
        treatment_plan = self._generate_treatment_plan(disease_name, severity_score, weather_data)
        
        # Store in database
        self._store_detection(
            image_path, disease_name, confidence_score, 
            severity_score, location, weather_data
        )
        
        return {
            'disease_name': disease_name,
            'confidence': confidence_score,
            'severity_score': severity_score,
            'health_score': health_score,
            'treatment_plan': treatment_plan,
            'weather_data': weather_data,
            'recommendations': self._generate_recommendations(disease_name, severity_score)
        }
    
    def batch_analyze(self, image_paths: List[str], location: str = None) -> List[Dict]:
        """Analyze multiple images in batch"""
        results = []
        for image_path in image_paths:
            try:
                result = self.analyze_plant_health(image_path, location)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'image_path': image_path
                })
        return results
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Enhanced image preprocessing"""
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.unsqueeze(0)
    
    def _calculate_severity_score(self, image: Image.Image, disease_name: str) -> float:
        """Calculate disease severity based on visual analysis"""
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for different disease symptoms
        if 'healthy' in disease_name.lower():
            return 0.0
        
        # Calculate affected area percentage
        if 'spot' in disease_name.lower() or 'blight' in disease_name.lower():
            # Look for dark spots
            lower_brown = np.array([10, 50, 20])
            upper_brown = np.array([20, 255, 200])
            mask = cv2.inRange(hsv, lower_brown, upper_brown)
        elif 'rust' in disease_name.lower():
            # Look for orange/rust colored areas
            lower_rust = np.array([5, 50, 50])
            upper_rust = np.array([15, 255, 255])
            mask = cv2.inRange(hsv, lower_rust, upper_rust)
        else:
            # General disease detection
            lower_disease = np.array([0, 30, 30])
            upper_disease = np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower_disease, upper_disease)
        
        # Calculate percentage of affected area
        total_pixels = cv_image.shape[0] * cv_image.shape[1]
        affected_pixels = cv2.countNonZero(mask)
        severity = min(affected_pixels / total_pixels * 100, 100) / 100
        
        return severity
    
    def _calculate_health_score(self, confidence: float, severity: float) -> float:
        """Calculate overall plant health score (0-100)"""
        if confidence < self.confidence_threshold:
            return 50.0  # Uncertain diagnosis
        
        if severity < 0.1:
            return 90.0  # Very healthy
        elif severity < 0.3:
            return 70.0  # Mild issues
        elif severity < 0.6:
            return 50.0  # Moderate issues
        else:
            return 20.0  # Severe issues
    
    def _get_weather_data(self, location: str) -> Optional[Dict]:
        """Get current weather data for the location"""
        try:
            # Using OpenWeatherMap API (you'll need to get an API key)
            api_key = "YOUR_WEATHER_API_KEY"  # Replace with actual API key
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['main']['temp'] - 273.15,  # Convert to Celsius
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'description': data['weather'][0]['description'],
                    'wind_speed': data['wind']['speed']
                }
        except Exception as e:
            print(f"Weather data unavailable: {e}")
        
        return None
    
    def _generate_treatment_plan(self, disease_name: str, severity: float, weather_data: Dict = None) -> Dict:
        """Generate comprehensive treatment plan"""
        treatment_plan = {
            'immediate_actions': [],
            'preventive_measures': [],
            'monitoring_schedule': [],
            'expected_recovery_time': 'Unknown'
        }
        
        if 'healthy' in disease_name.lower():
            treatment_plan['immediate_actions'] = ['Continue current care routine']
            treatment_plan['preventive_measures'] = [
                'Maintain proper watering schedule',
                'Ensure adequate sunlight',
                'Monitor for early signs of disease'
            ]
            treatment_plan['expected_recovery_time'] = 'N/A - Plant is healthy'
        
        elif 'scab' in disease_name.lower():
            treatment_plan['immediate_actions'] = [
                'Remove and destroy infected leaves',
                'Apply fungicide spray',
                'Improve air circulation around plant'
            ]
            treatment_plan['preventive_measures'] = [
                'Plant resistant varieties',
                'Avoid overhead watering',
                'Apply preventive fungicide in early spring'
            ]
            treatment_plan['monitoring_schedule'] = ['Check weekly for new infections']
            treatment_plan['expected_recovery_time'] = '2-4 weeks with proper treatment'
        
        elif 'blight' in disease_name.lower():
            treatment_plan['immediate_actions'] = [
                'Remove infected plant parts immediately',
                'Apply copper-based fungicide',
                'Improve drainage around plant'
            ]
            treatment_plan['preventive_measures'] = [
                'Crop rotation',
                'Proper spacing between plants',
                'Avoid working with wet plants'
            ]
            treatment_plan['monitoring_schedule'] = ['Check every 2-3 days during wet weather']
            treatment_plan['expected_recovery_time'] = '3-6 weeks depending on severity'
        
        # Adjust based on severity
        if severity > 0.7:
            treatment_plan['immediate_actions'].insert(0, 'URGENT: Consider removing severely infected plants')
            treatment_plan['expected_recovery_time'] = 'May require plant replacement'
        
        # Weather-based adjustments
        if weather_data and weather_data.get('humidity', 0) > 80:
            treatment_plan['immediate_actions'].append('Reduce humidity around plant')
            treatment_plan['preventive_measures'].append('Improve ventilation')
        
        return treatment_plan
    
    def _generate_recommendations(self, disease_name: str, severity: float) -> List[str]:
        """Generate specific recommendations based on disease and severity"""
        recommendations = []
        
        if severity > 0.5:
            recommendations.append("🔴 High severity detected - immediate action required")
        elif severity > 0.2:
            recommendations.append("🟡 Moderate severity - monitor closely")
        else:
            recommendations.append("🟢 Low severity - preventive measures recommended")
        
        if 'fungal' in disease_name.lower() or 'mildew' in disease_name.lower():
            recommendations.extend([
                "💧 Avoid overhead watering",
                "🌬️ Improve air circulation",
                "🧪 Apply appropriate fungicide"
            ])
        
        if 'bacterial' in disease_name.lower():
            recommendations.extend([
                "🧼 Sterilize tools between uses",
                "✂️ Remove infected plant parts",
                "🦠 Consider copper-based treatments"
            ])
        
        if 'viral' in disease_name.lower():
            recommendations.extend([
                "🦟 Control insect vectors",
                "🧹 Remove infected plants",
                "🛡️ Plant resistant varieties"
            ])
        
        return recommendations
    
    def _store_detection(self, image_path: str, disease_name: str, confidence: float, 
                        severity: float, location: str = None, weather_data: Dict = None):
        """Store detection in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        weather_condition = weather_data.get('description', 'Unknown') if weather_data else 'Unknown'
        
        cursor.execute('''
            INSERT INTO detection_history 
            (image_path, disease_name, confidence, severity_score, location, weather_condition)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (image_path, disease_name, confidence, severity, location, weather_condition))
        
        conn.commit()
        conn.close()
    
    def get_detection_history(self, days: int = 30) -> pd.DataFrame:
        """Get detection history for the last N days"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM detection_history 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def generate_health_report(self, days: int = 30) -> Dict:
        """Generate comprehensive health report"""
        history_df = self.get_detection_history(days)
        
        if history_df.empty:
            return {"message": "No detection history found"}
        
        report = {
            'total_detections': len(history_df),
            'disease_distribution': history_df['disease_name'].value_counts().to_dict(),
            'average_confidence': history_df['confidence'].mean(),
            'average_severity': history_df['severity_score'].mean(),
            'most_common_disease': history_df['disease_name'].mode().iloc[0] if not history_df.empty else None,
            'health_trend': self._calculate_health_trend(history_df),
            'recommendations': self._generate_historical_recommendations(history_df)
        }
        
        return report
    
    def _calculate_health_trend(self, df: pd.DataFrame) -> str:
        """Calculate health trend over time"""
        if len(df) < 2:
            return "Insufficient data"
        
        # Calculate weekly averages
        df['week'] = pd.to_datetime(df['timestamp']).dt.isocalendar().week
        weekly_health = df.groupby('week')['severity_score'].mean()
        
        if len(weekly_health) < 2:
            return "Insufficient data"
        
        # Simple trend calculation
        recent_avg = weekly_health.tail(2).mean()
        older_avg = weekly_health.head(2).mean()
        
        if recent_avg < older_avg * 0.9:
            return "Improving"
        elif recent_avg > older_avg * 1.1:
            return "Deteriorating"
        else:
            return "Stable"
    
    def _generate_historical_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on historical data"""
        recommendations = []
        
        # Check for recurring diseases
        disease_counts = df['disease_name'].value_counts()
        if len(disease_counts) > 0 and disease_counts.iloc[0] > len(df) * 0.3:
            recommendations.append(f"⚠️ {disease_counts.index[0]} appears frequently - consider preventive measures")
        
        # Check severity trends
        if df['severity_score'].mean() > 0.5:
            recommendations.append("🔴 High average severity - review treatment protocols")
        
        # Check confidence levels
        if df['confidence'].mean() < 0.8:
            recommendations.append("❓ Low average confidence - consider improving image quality")
        
        return recommendations

class MobileOptimizer:
    """Optimize the system for mobile devices"""
    
    @staticmethod
    def compress_image(image_path: str, max_size: int = 1024, quality: int = 85) -> str:
        """Compress image for mobile upload"""
        image = Image.open(image_path)
        
        # Resize if too large
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Save with compression
        compressed_path = image_path.replace('.', '_compressed.')
        image.save(compressed_path, 'JPEG', quality=quality, optimize=True)
        
        return compressed_path
    
    @staticmethod
    def generate_mobile_response(analysis_result: Dict) -> Dict:
        """Generate mobile-optimized response"""
        return {
            'disease': analysis_result['disease_name'],
            'confidence': round(analysis_result['confidence'] * 100, 1),
            'severity': round(analysis_result['severity_score'] * 100, 1),
            'health_score': round(analysis_result['health_score'], 1),
            'urgent': analysis_result['severity_score'] > 0.7,
            'recommendations': analysis_result['recommendations'][:3],  # Limit for mobile
            'treatment_plan': {
                'immediate': analysis_result['treatment_plan']['immediate_actions'][:2],
                'prevention': analysis_result['treatment_plan']['preventive_measures'][:2]
            }
        }
