import os
from flask import Flask, redirect, render_template, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import json
from datetime import datetime
from advanced_features import AdvancedPlantAnalyzer, MobileOptimizer
import io
import base64

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load data
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Initialize model
model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

# Initialize advanced analyzer
class_names = {v: k for k, v in CNN.idx_to_classes.items()}
analyzer = AdvancedPlantAnalyzer(model, class_names)

def prediction(image_path):
    """Original prediction function for backward compatibility"""
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

def advanced_prediction(image_path, location=None):
    """Enhanced prediction with advanced features"""
    try:
        result = analyzer.analyze_plant_health(image_path, location)
        return result
    except Exception as e:
        print(f"Advanced prediction error: {e}")
        # Fallback to original prediction
        index = prediction(image_path)
        return {
            'disease_name': disease_info['disease_name'][index],
            'confidence': 0.8,  # Default confidence
            'severity_score': 0.5,  # Default severity
            'health_score': 50.0,
            'treatment_plan': {'immediate_actions': ['Consult with agricultural expert']},
            'recommendations': ['Get professional diagnosis']
        }


# API Routes
@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for disease detection"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image = request.files['image']
        location = request.form.get('location', '')
        
        if image.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Save uploaded image
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image.filename}"
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        
        # Get advanced prediction
        result = advanced_prediction(file_path, location)
        
        # Get additional info from CSV
        disease_name = result['disease_name']
        pred_index = None
        for idx, name in disease_info['disease_name'].items():
            if name == disease_name:
                pred_index = idx
                break
        
        if pred_index is not None:
            result['description'] = disease_info['description'][pred_index]
            result['prevention'] = disease_info['Possible Steps'][pred_index]
            result['image_url'] = disease_info['image_url'][pred_index]
            result['supplement_name'] = supplement_info['supplement name'][pred_index]
            result['supplement_image'] = supplement_info['supplement image'][pred_index]
            result['supplement_link'] = supplement_info['buy link'][pred_index]
        
        print(result)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-detect', methods=['POST'])
def api_batch_detect():
    """API endpoint for batch disease detection"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        images = request.files.getlist('images')
        location = request.form.get('location', '')
        
        if len(images) > 10:  # Limit batch size
            return jsonify({'error': 'Maximum 10 images allowed per batch'}), 400
        
        results = []
        for image in images:
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image.filename}"
            file_path = os.path.join('static/uploads', filename)
            image.save(file_path)
            
            result = advanced_prediction(file_path, location)
            results.append(result)
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health-report', methods=['GET'])
def api_health_report():
    """API endpoint for health report"""
    try:
        days = request.args.get('days', 30, type=int)
        report = analyzer.generate_health_report(days)
        return jsonify(report)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def api_history():
    """API endpoint for detection history"""
    try:
        days = request.args.get('days', 30, type=int)
        history_df = analyzer.get_detection_history(days)
        return jsonify(history_df.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Original routes for backward compatibility
@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
