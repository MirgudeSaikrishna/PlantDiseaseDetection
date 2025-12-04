"""
Model Tests for AgriDetect AI
Author: [Your Name]
Date: [Current Date]
"""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os
from CNN import CNN, ImprovedCNN

class TestCNNModel:
    """Test the original CNN model"""
    
    def test_cnn_initialization(self):
        """Test CNN model initialization"""
        model = CNN(39)
        assert model is not None
        assert hasattr(model, 'conv_layers')
        assert hasattr(model, 'dense_layers')
    
    def test_cnn_forward_pass(self):
        """Test CNN forward pass"""
        model = CNN(39)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (1, 39)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_cnn_parameters(self):
        """Test CNN model parameters"""
        model = CNN(39)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable
        assert total_params > 1000000  # Should have reasonable number of parameters

class TestImprovedCNNModel:
    """Test the improved CNN model"""
    
    def test_improved_cnn_initialization(self):
        """Test ImprovedCNN model initialization"""
        model = ImprovedCNN(39)
        assert model is not None
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert hasattr(model, 'conv3')
        assert hasattr(model, 'conv4')
        assert hasattr(model, 'conv5')
        assert hasattr(model, 'attention')
        assert hasattr(model, 'classifier')
    
    def test_improved_cnn_forward_pass(self):
        """Test ImprovedCNN forward pass"""
        model = ImprovedCNN(39)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (1, 39)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_improved_cnn_attention_mechanism(self):
        """Test attention mechanism in ImprovedCNN"""
        model = ImprovedCNN(39)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            # Get features before attention
            x = model.conv1(dummy_input)
            x = model.conv2(x)
            x = model.conv3(x)
            x = model.conv4(x)
            x = model.conv5(x)
            x = model.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            
            # Get attention weights
            attention_weights = model.attention(x)
            
            # Apply attention
            attended_features = x * attention_weights
            
            # Check attention weights are in valid range
            assert torch.all(attention_weights >= 0)
            assert torch.all(attention_weights <= 1)
            
            # Check attended features are different from original
            assert not torch.allclose(x, attended_features)
    
    def test_improved_cnn_parameters(self):
        """Test ImprovedCNN model parameters"""
        model = ImprovedCNN(39)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params
        assert total_params > 2000000  # Should have more parameters than original
    
    def test_improved_cnn_dropout(self):
        """Test dropout behavior in ImprovedCNN"""
        model = ImprovedCNN(39, dropout_rate=0.5)
        
        # In training mode, dropout should be active
        model.train()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        outputs = []
        for _ in range(10):
            with torch.no_grad():
                output = model(dummy_input)
                outputs.append(output)
        
        # Outputs should vary due to dropout
        outputs = torch.stack(outputs)
        assert not torch.allclose(outputs[0], outputs[1])
        
        # In eval mode, dropout should be inactive
        model.eval()
        with torch.no_grad():
            output1 = model(dummy_input)
            output2 = model(dummy_input)
        
        assert torch.allclose(output1, output2)

class TestModelComparison:
    """Compare original and improved models"""
    
    def test_model_output_shapes(self):
        """Test that both models have same output shape"""
        original_model = CNN(39)
        improved_model = ImprovedCNN(39)
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            original_output = original_model(dummy_input)
            improved_output = improved_model(dummy_input)
        
        assert original_output.shape == improved_output.shape
        assert original_output.shape == (1, 39)
    
    def test_model_parameter_count(self):
        """Test parameter count comparison"""
        original_model = CNN(39)
        improved_model = ImprovedCNN(39)
        
        original_params = sum(p.numel() for p in original_model.parameters())
        improved_params = sum(p.numel() for p in improved_model.parameters())
        
        # Improved model should have more parameters
        assert improved_params > original_params
        print(f"Original model parameters: {original_params:,}")
        print(f"Improved model parameters: {improved_params:,}")

class TestModelRobustness:
    """Test model robustness and edge cases"""
    
    def test_model_with_different_input_sizes(self):
        """Test model with different input sizes"""
        model = ImprovedCNN(39)
        model.eval()
        
        # Test with different sizes (should be resized internally)
        sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
        
        for size in sizes:
            dummy_input = torch.randn(1, 3, *size)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            assert output.shape == (1, 39)
            assert not torch.isnan(output).any()
    
    def test_model_with_edge_case_inputs(self):
        """Test model with edge case inputs"""
        model = ImprovedCNN(39)
        model.eval()
        
        # Test with all zeros
        zero_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            output = model(zero_input)
        assert output.shape == (1, 39)
        
        # Test with all ones
        ones_input = torch.ones(1, 3, 224, 224)
        with torch.no_grad():
            output = model(ones_input)
        assert output.shape == (1, 39)
        
        # Test with random noise
        noise_input = torch.randn(1, 3, 224, 224) * 10  # High variance
        with torch.no_grad():
            output = model(noise_input)
        assert output.shape == (1, 39)
    
    def test_model_memory_usage(self):
        """Test model memory usage"""
        model = ImprovedCNN(39)
        
        # Test forward pass memory usage
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Should not cause memory issues
        assert output is not None
        del output
        del dummy_input

class TestModelTraining:
    """Test model training capabilities"""
    
    def test_model_training_mode(self):
        """Test model in training mode"""
        model = ImprovedCNN(39)
        model.train()
        
        # Create dummy data
        dummy_input = torch.randn(2, 3, 224, 224)
        dummy_target = torch.randint(0, 39, (2,))
        
        # Forward pass
        output = model(dummy_input)
        assert output.shape == (2, 39)
        
        # Loss calculation
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, dummy_target)
        assert loss.item() > 0
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()
    
    def test_model_optimization(self):
        """Test model optimization"""
        model = ImprovedCNN(39)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(5):
            model.train()
            dummy_input = torch.randn(4, 3, 224, 224)
            dummy_target = torch.randint(0, 39, (4,))
            
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
            
            assert not torch.isnan(loss).any()

if __name__ == '__main__':
    pytest.main([__file__])
