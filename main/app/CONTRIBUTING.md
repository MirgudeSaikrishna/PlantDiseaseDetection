# Contributing to AgriDetect AI

Thank you for your interest in contributing to AgriDetect AI! This document provides guidelines and information for contributors.

## 🤝 **How to Contribute**

### **1. Fork and Clone**
```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/yourusername/agridetect-ai.git
cd agridetect-ai
```

### **2. Set Up Development Environment**
```bash
# Backend setup
cd Plant-Disease-Detection-main/Flask-Deployed-App
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../../UI-of-Plant-Disease-Detection
npm install
```

### **3. Create a Branch**
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number
```

## 📋 **Contribution Guidelines**

### **Code Style**
- Follow PEP 8 for Python code
- Use TypeScript for React components
- Write meaningful commit messages
- Add docstrings to functions and classes

### **Testing**
- Write unit tests for new features
- Ensure all tests pass before submitting
- Add integration tests for API endpoints

### **Documentation**
- Update README.md for significant changes
- Add docstrings to new functions
- Update API documentation

## 🐛 **Reporting Issues**

### **Bug Reports**
When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- System information (OS, Python version, etc.)

### **Feature Requests**
For feature requests, please include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (if you have ideas)
- Any additional context

## 🔄 **Pull Request Process**

### **Before Submitting**
1. Ensure your code follows the style guidelines
2. Run all tests and ensure they pass
3. Update documentation if needed
4. Rebase your branch on the latest main branch

### **Pull Request Template**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## 🏗️ **Development Areas**

### **Backend (Flask)**
- API endpoint development
- Model improvements
- Database optimization
- Performance enhancements

### **Frontend (React)**
- UI component development
- User experience improvements
- Mobile optimization
- Accessibility enhancements

### **AI/ML**
- Model architecture improvements
- Training pipeline optimization
- New disease detection capabilities
- Performance benchmarking

### **DevOps**
- Deployment automation
- CI/CD pipeline improvements
- Monitoring and logging
- Security enhancements

## 🧪 **Testing Guidelines**

### **Unit Tests**
```python
# Example test structure
def test_disease_detection():
    # Arrange
    test_image = "test_images/healthy_leaf.jpg"
    
    # Act
    result = predict_disease(test_image)
    
    # Assert
    assert result['confidence'] > 0.8
    assert 'healthy' in result['disease_name'].lower()
```

### **Integration Tests**
```python
def test_api_endpoint():
    # Test API endpoint
    response = client.post('/api/detect', data={'image': test_file})
    assert response.status_code == 200
    assert 'disease_name' in response.json
```

## 📚 **Documentation Standards**

### **Code Documentation**
```python
def predict_disease(image_path: str, location: str = None) -> Dict:
    """
    Predict plant disease from image.
    
    Args:
        image_path (str): Path to the input image
        location (str, optional): Geographic location for weather context
        
    Returns:
        Dict: Prediction results including disease name, confidence, and treatment plan
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is not supported
    """
```

### **API Documentation**
```yaml
/api/detect:
  post:
    summary: Detect plant disease
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: Plant leaf image
    responses:
      200:
        description: Successful detection
        schema:
          type: object
          properties:
            disease_name:
              type: string
            confidence:
              type: number
```

## 🎯 **Project Roadmap**

### **Current Priorities**
1. Mobile app development
2. Real-time video analysis
3. Multi-language support
4. Advanced analytics dashboard

### **Future Features**
1. 3D plant modeling
2. AR disease visualization
3. IoT sensor integration
4. Blockchain treatment tracking

## 💬 **Communication**

### **Discussions**
- Use GitHub Discussions for general questions
- Tag maintainers for urgent issues
- Be respectful and constructive

### **Code Reviews**
- Provide constructive feedback
- Focus on code quality and functionality
- Be open to suggestions and improvements

## 📄 **License**

By contributing to AgriDetect AI, you agree that your contributions will be licensed under the MIT License.

## 🙏 **Recognition**

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Social media acknowledgments

## 📞 **Getting Help**

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: dev@agridetect.ai for direct contact
- **Discord**: [Join our community](https://discord.gg/agridetect)

---

Thank you for contributing to AgriDetect AI! Together, we can help farmers worldwide protect their crops and improve food security. 🌱
