import pandas as pd
import torch
import gc
from tqdm import tqdm
from load_prompts import load_prompts
from load_openchat import load_openchat_model
from define_query import query_openchat
import time
import psutil
import os

def check_system_resources():
    """Check available system resources"""
    print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total VRAM: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            print(f"  Free VRAM: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1024**3:.1f} GB")

def clear_memory():
    """Aggressive memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def test_single_generation(model, tokenizer):
    """Test with a simple prompt first"""
    print("Testing model generation speed...")
    test_prompt = "Hello, how are you? Please tell me about your day."
    
    # Clear memory before test
    clear_memory()
    
    start_time = time.time()
    try:
        # Monitor memory during generation
        if torch.cuda.is_available():
            initial_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"Initial GPU memory: {initial_mem:.2f} GB")
        
        response = query_openchat(model, tokenizer, test_prompt)
        elapsed = time.time() - start_time
        
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            current_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"Peak GPU memory: {peak_mem:.2f} GB")
            print(f"Current GPU memory: {current_mem:.2f} GB")
        
        # Show more of the response and analyze performance
        response_length = len(response.split())
        tokens_per_second = response_length / elapsed if elapsed > 0 else 0
        
        print(f"✓ Test successful in {elapsed:.1f}s!")
        print(f"Response length: {response_length} words")
        print(f"Speed: {tokens_per_second:.1f} words/second")
        print(f"Full response: {response}")
        print("-" * 50)
        
        # Performance warning
        if elapsed > 60:  # More than 1 minute for a simple prompt
            print("⚠️  WARNING: Very slow generation detected!")
            print("This suggests an issue with your query_openchat function.")
            
        if tokens_per_second < 5:
            print("⚠️  WARNING: Low throughput detected!")
            print("GPU utilization might be poor.")
        
        # Clear memory after test
        clear_memory()
        
        # Estimate total time
        estimated_total = (elapsed * 142) / 60  # minutes
        print(f"Estimated total time for 142 prompts: {estimated_total:.1f} minutes")
        
        if estimated_total > 60:  # More than 1 hour
            print("⚠️  This will take a very long time. Consider optimizing your query function.")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        clear_memory()
        return False

def safe_model_inference(model, tokenizer, prompt, max_retries=3):
    """Safely run model inference with retries and memory management"""
    for attempt in range(max_retries):
        try:
            # Clear memory before each inference
            if attempt > 0:
                clear_memory()
                time.sleep(1)  # Brief pause between retries
            
            # Check memory before inference
            if torch.cuda.is_available():
                free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
                if free_mem < 1.0:  # Less than 1GB free
                    print(f"Warning: Low GPU memory ({free_mem:.2f} GB), clearing cache...")
                    clear_memory()
            
            response = query_openchat(model, tokenizer, prompt)
            return response
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM error on attempt {attempt + 1}, clearing memory...")
                clear_memory()
                if attempt == max_retries - 1:
                    return f"ERROR: Out of memory after {max_retries} attempts"
            else:
                return f"ERROR: {str(e)}"
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    return "ERROR: Max retries exceeded"

def main():
    print("=== OpenChat Response Generator (GPU Fixed) ===")
    
    # Check system resources
    check_system_resources()
    
    # Set memory management settings
    if torch.cuda.is_available():
        # Prevent memory fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        torch.cuda.empty_cache()
        
        # Set memory fraction to prevent system freeze
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
        
        print(f"CUDA available: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print("GPU memory fraction set to 80%")
    else:
        print("CUDA not available - using CPU")
    
    # Load prompts
    print("\n1. Loading prompts...")
    prompts = load_prompts()
    if not prompts:
        print("No prompts found!")
        return
    print(f"Loaded {len(prompts)} prompts")
    
    # Load model
    print("\n2. Loading model...")
    try:
        model, tokenizer = load_openchat_model()
        print("Model loaded successfully")
        
        # Check memory after model loading
        if torch.cuda.is_available():
            model_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"Model using {model_mem:.2f} GB GPU memory")
            
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Test generation
    print("\n3. Testing generation...")
    if not test_single_generation(model, tokenizer):
        print("Model test failed - stopping")
        return
    
    # Additional speed test
    '''print("\n3.5. Running comprehensive speed test...")
    from define_query import test_speed_comparison, check_gpu_utilization
    check_gpu_utilization()
    test_speed_comparison(model, tokenizer)
    
    print("\nProceed with full processing? (y/n):")
    response = input().lower()
    if response != 'y':
        print("Stopping as requested")
        return'''
    
    # Process all prompts - BATCH VERSION for speed
    print(f"\n4. Processing {len(prompts)} prompts with BATCH processing...")
    
    try:
        from define_query import query_openchat_batch_ultra_fast
        print("Using batch processing for maximum speed...")
        
        start_time = time.time()
        responses = query_openchat_batch_ultra_fast(model, tokenizer, prompts, batch_size=6)
        total_time = time.time() - start_time
        
        print(f"Batch processing completed in {total_time/60:.1f} minutes")
        errors = 0  # Batch processing handles errors internally
        
    except ImportError:
        print("Batch processing not available, using single processing...")
        # Fallback to single processing
        responses = []
        errors = 0
        start_time = time.time()
        
        for i, prompt in enumerate(tqdm(prompts, desc="Processing")):
            response = safe_model_inference(model, tokenizer, prompt)
            responses.append(response)
            
            if response.startswith("ERROR"):
                errors += 1
            
            # Progress update and memory monitoring
            if (i + 1) % 3 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60  # per minute
                remaining_time = (len(prompts) - i - 1) / (rate/60) if rate > 0 else 0
                
                # Memory status
                ram_usage = psutil.virtual_memory().percent
                
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_percent = (gpu_mem / gpu_total) * 100
                    print(f"Done: {i+1}/{len(prompts)} | Rate: {rate:.1f}/min | ETA: {remaining_time/60:.1f}min")
                    print(f"  RAM: {ram_usage:.1f}% | GPU: {gpu_mem:.1f}/{gpu_total:.1f}GB ({gpu_percent:.1f}%)")
                else:
                    print(f"Done: {i+1}/{len(prompts)} | Rate: {rate:.1f}/min | ETA: {remaining_time/60:.1f}min | RAM: {ram_usage:.1f}%")
            
            # Aggressive memory clearing every 5 prompts
            if (i + 1) % 5 == 0:
                clear_memory()
            
            # Emergency stop if too many errors
            if errors > 10:
                print("Too many errors - stopping")
                break
            
            # Emergency stop if RAM usage too high
            if psutil.virtual_memory().percent > 90:
                print("RAM usage too high - stopping to prevent system crash")
                break
    
    # Final memory cleanup
    clear_memory()
    
    # Save results
    print(f"\n5. Saving results...")
    df = pd.DataFrame({
        'prompt': prompts[:len(responses)],
        'response': responses
    })
    
    output_file = "openchat_responses.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Summary
    total_time = time.time() - start_time
    success_count = len([r for r in responses if not r.startswith("ERROR")])
    
    print(f"\n=== COMPLETED ===")
    print(f"Processed: {len(responses)}/{len(prompts)} prompts")
    print(f"Success: {success_count}")
    print(f"Errors: {errors}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average: {total_time/len(responses):.1f} seconds per prompt")
    print(f"Saved to: {output_file}")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()