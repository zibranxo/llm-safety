import torch
import gc
import os
import subprocess
import sys

def emergency_gpu_cleanup():
    """Emergency GPU memory cleanup"""
    print("🚨 EMERGENCY GPU MEMORY CLEANUP")
    
    # Check initial state
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Before cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved of {total:.1f}GB total")
    
    # Step 1: Clear Python objects
    print("1. Clearing Python garbage...")
    gc.collect()
    
    if torch.cuda.is_available():
        # Step 2: Clear PyTorch cache
        print("2. Clearing PyTorch cache...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Step 3: Reset memory stats
        print("3. Resetting memory stats...")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # Step 4: Set memory management
        print("4. Setting memory management...")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,roundup_power2_divisions:8"
        
        # Step 5: Force synchronization
        print("5. Synchronizing GPU...")
        torch.cuda.synchronize()
        
        # Check final state
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = total - reserved
        
        print(f"After cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")
        
        if free < 3.5:
            print("\n⚠️ WARNING: Still low on GPU memory!")
            print("Recommendations:")
            print("1. Close other applications using GPU")
            print("2. Restart your Python session/Jupyter kernel")
            print("3. Restart your computer if needed")
            
            # Check for other processes
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    python_processes = [line for line in lines if 'python' in line.lower()]
                    if python_processes:
                        print(f"\n🔍 Found {len(python_processes)} Python processes on GPU:")
                        for proc in python_processes[:3]:  # Show first 3
                            print(f"   {proc.strip()}")
            except:
                pass
                
            return False
        else:
            print("✅ GPU memory cleanup successful!")
            return True
    else:
        print("❌ CUDA not available")
        return False

def test_gpu_basic():
    """Test basic GPU functionality"""
    print("\n🧪 TESTING GPU FUNCTIONALITY...")
    
    try:
        # Simple GPU test
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        x = torch.randn(1000, 1000).cuda()
        y = torch.mm(x, x)
        end_time.record()
        torch.cuda.synchronize()
        
        elapsed = start_time.elapsed_time(end_time)
        print(f"GPU matrix multiply: {elapsed:.1f}ms")
        
        # Cleanup test tensors
        del x, y
        torch.cuda.empty_cache()
        
        if elapsed > 100:
            print("⚠️ GPU seems slow")
            return False
        else:
            print("✅ GPU working normally")
            return True
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def main():
    print("🚀 GPU MEMORY CLEANUP & TEST")
    print("="*50)
    
    # Step 1: Cleanup
    cleanup_ok = emergency_gpu_cleanup()
    
    # Step 2: Test
    if cleanup_ok:
        test_ok = test_gpu_basic()
        
        if test_ok:
            print("\n✅ READY TO LOAD MODEL!")
            print("Now run: python load_openchat.py")
        else:
            print("\n❌ GPU issues detected. Check drivers/hardware.")
    else:
        print("\n❌ Memory cleanup failed. Consider restarting Python.")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    main()