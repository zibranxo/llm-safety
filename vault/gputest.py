import torch

# Test 1: Basic CUDA
print(f"CUDA available: {torch.cuda.is_available()}")

# Test 2: GPU performance
if torch.cuda.is_available():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x)
    end.record()
    torch.cuda.synchronize()
    
    print(f"GPU matrix multiply time: {start.elapsed_time(end):.2f} ms")
    
    if start.elapsed_time(end) > 100:  # More than 100ms is very slow
        print("❌ GPU is extremely slow or not working")
    else:
        print("✅ GPU working normally")

# Test 3: Check your model device
from load_openchat import load_openchat_model
model, tokenizer = load_openchat_model()

# THIS IS THE CRITICAL CHECK:
model_device = next(model.parameters()).device
print(f"🎯 MODEL IS ACTUALLY ON: {model_device}")

if str(model_device) == 'cpu':
    print("❌ FOUND THE PROBLEM: Model is on CPU!")
else:
    print("✅ Model appears to be on GPU")