# test_model.py
import torch
import pytest
from model import SimpleCNN, count_parameters

def test_model_parameters():
    model = SimpleCNN()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 100000"

def test_input_shape():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        assert True, "Model accepts 28x28 input"
    except:
        assert False, "Model failed to process 28x28 input"

def test_output_shape():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape[1] == 10, f"Model output should have 10 classes, got {output.shape[1]}"


if __name__ == "__main__":
    pytest.main([__file__])