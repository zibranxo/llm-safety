import torch
import time
import gc
import psutil
import subprocess

def comprehensive_gpu_diagnostic():
    """Complete GPU diagnostic to find the real problem"""
    print("=== COMPREHENSIVE GPU DIAGNOSTIC ===")
    
    # 1. Basic CUDA check
    print(f"1. CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("❌ CUDA not available - this is your problem!")
        return False
    
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   PyTorch Version: {torch.__version__}")
    
    # 2. GPU Hardware check
    print(f"2. GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name}")
        print(f"   Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"   Compute Capability: {props.major}.{props.minor}")
    
    # 3. Current GPU status
    current_device = torch.cuda.current_device()
    print(f"3. Current Device: {current_device}")
    print(f"   Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # 4. GPU Performance Test
    print("4. GPU Performance Test...")
    try:
        # Simple GPU computation test
        start_time = time.time()
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)  # Matrix multiplication
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_test_time = time.time() - start_time
        
        print(f"   ✅ GPU Matrix Multiply (1000x1000): {gpu_test_time:.3f} seconds")
        
        # Compare with CPU
        start_time = time.time()
        x_cpu = torch.randn(1000, 1000)
        y_cpu = torch.randn(1000, 1000)
        z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_test_time = time.time() - start_time
        
        print(f"   CPU Matrix Multiply (1000x1000): {cpu_test_time:.3f} seconds")
        print(f"   GPU Speedup: {cpu_test_time/gpu_test_time:.1f}x")
        
        if gpu_test_time > 0.1:  # Should be much faster
            print("   ⚠️ GPU computation is unusually slow!")
        
        # Cleanup
        del x, y, z, x_cpu, y_cpu, z_cpu
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
        return False
    
    # 5. Check nvidia-smi
    print("5. NVIDIA-SMI Check...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("   ✅ nvidia-smi accessible")
            # Look for Python processes
            if 'python' in result.stdout.lower():
                print("   ⚠️ Other Python processes detected on GPU")
        else:
            print("   ❌ nvidia-smi failed")
    except Exception as e:
        print(f"   ❌ nvidia-smi error: {e}")
    
    return True

def force_model_to_gpu(model, tokenizer):
    """Aggressively force model to GPU and verify"""
    print("\n=== FORCING MODEL TO GPU ===")
    
    if not torch.cuda.is_available():
        print("❌ No CUDA - cannot use GPU")
        return model, tokenizer
    
    device = torch.device("cuda:0")
    print(f"Target device: {device}")
    
    # Clear GPU memory first
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # Method 1: Direct .to() call
        print("1. Attempting .to(device)...")
        model = model.to(device)
        print("   ✅ Model moved to GPU")
        
        # Method 2: Verify each parameter
        print("2. Verifying parameters...")
        gpu_params = 0
        cpu_params = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            if param.device.type == 'cuda':
                gpu_params += 1
            else:
                cpu_params += 1
                print(f"   ⚠️ Parameter on CPU: {name} -> {param.device}")
        
        print(f"   Parameters: {gpu_params} on GPU, {cpu_params} on CPU, {total_params} total")
        
        if cpu_params > 0:
            print(f"   ❌ {cpu_params} parameters still on CPU!")
            # Try to move them
            for name, param in model.named_parameters():
                if param.device.type != 'cuda':
                    param.data = param.data.to(device)
                    if hasattr(param, 'grad') and param.grad is not None:
                        param.grad = param.grad.to(device)
        
        # Method 3: Check model device
        model_device = next(model.parameters()).device
        print(f"3. Model device check: {model_device}")
        
        if model_device.type != 'cuda':
            print("   ❌ Model still not on GPU after force move!")
            return None, None
        
        # Method 4: Set model to eval and check GPU memory
        model.eval()
        gpu_memory_after = torch.cuda.memory_allocated() / 1024**3
        print(f"4. GPU memory after model load: {gpu_memory_after:.2f} GB")
        
        if gpu_memory_after < 1.0:  # Model should use at least 1GB
            print("   ⚠️ Very low GPU memory usage - model might not be fully loaded")
        
        print("   ✅ Model successfully on GPU")
        return model, tokenizer
        
    except Exception as e:
        print(f"   ❌ Failed to move model to GPU: {e}")
        return None, None

def test_model_inference_speed(model, tokenizer):
    """Test actual model inference speed on GPU vs CPU"""
    print("\n=== MODEL INFERENCE SPEED TEST ===")
    
    if model is None:
        print("❌ No model to test")
        return
    
    test_prompt = "Hello, how are you?"
    
    # Check where model actually is
    model_device = next(model.parameters()).device
    print(f"Model is on: {model_device}")
    
    # Test 1: Current inference speed
    print("1. Testing current inference...")
    torch.cuda.empty_cache()
    
    # Prepare input
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model_device)
    
    # Time the inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            do_sample=False,  # Greedy for consistency
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    if model_device.type == 'cuda':
        torch.cuda.synchronize()  # Wait for GPU
    
    inference_time = time.time() - start_time
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
    tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
    
    print(f"   Time: {inference_time:.2f} seconds")
    print(f"   Tokens generated: {tokens_generated}")
    print(f"   Speed: {tokens_per_second:.1f} tokens/second")
    print(f"   Response: {response}")
    
    # Performance assessment
    if model_device.type == 'cuda':
        if tokens_per_second < 5:
            print("   ❌ EXTREMELY SLOW for GPU! Something is very wrong.")
        elif tokens_per_second < 20:
            print("   ⚠️ Slow for GPU - should be faster")
        else:
            print("   ✅ Reasonable GPU speed")
    else:
        if tokens_per_second < 1:
            print("   ❌ Extremely slow even for CPU")
        else:
            print("   ℹ️ CPU speed (expected to be slow)")
    
    return tokens_per_second

def emergency_gpu_fix():
    """Last resort GPU fixes"""
    print("\n=== EMERGENCY GPU FIXES ===")
    
    print("1. Clearing all GPU memory...")
    torch.cuda.empty_cache()
    gc.collect()
    
    print("2. Resetting GPU state...")
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print("   ✅ GPU state reset")
    except Exception as e:
        print(f"   ❌ GPU reset failed: {e}")
    
    print("3. Checking for memory leaks...")
    print(f"   Current allocation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    print("4. Environment variables...")
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Force synchronous execution
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    print("   ✅ Set CUDA environment variables")

def main_diagnostic():
    """Run complete diagnostic"""
    print("🔍 DIAGNOSING GPU PERFORMANCE ISSUE...\n")
    
    # Step 1: Basic diagnostics
    if not comprehensive_gpu_diagnostic():
        print("\n❌ Basic GPU diagnostics failed. Check CUDA installation.")
        return
    
    # Step 2: Emergency fixes
    emergency_gpu_fix()
    
    # Step 3: Try to load and fix model
    print("\n📥 TESTING MODEL LOADING...")
    try:
        from load_openchat import load_openchat_model
        model, tokenizer = load_openchat_model()
        
        if model is None:
            print("❌ Model loading failed")
            return
        
        # Force to GPU
        model, tokenizer = force_model_to_gpu(model, tokenizer)
        
        if model is None:
            print("❌ Could not force model to GPU")
            return
        
        # Test inference speed
        speed = test_model_inference_speed(model, tokenizer)
        
        print(f"\n🎯 FINAL DIAGNOSIS:")
        if speed > 20:
            print("✅ GPU is working properly now!")
        elif speed > 5:
            print("⚠️ GPU working but slower than expected")
        else:
            print("❌ GPU still not working properly")
            print("\nPOSSIBLE CAUSES:")
            print("- Model is actually running on CPU despite messages")
            print("- CUDA installation is broken")
            print("- GPU drivers are outdated")
            print("- Another process is hogging the GPU")
            print("- Hardware issue with your GPU")
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")

if __name__ == "__main__":
    main_diagnostic()