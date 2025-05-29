import torch
import time
from typing import List, Tuple, Union

def setup_inference_optimizations():
    """Enable all possible GPU optimizations"""
    # Enable TensorFloat-32 for RTX 4050
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set optimal threading
    torch.set_num_threads(4)
    
    # Enable JIT optimizations
    torch.jit.set_fusion_strategy([('STATIC', 2), ('DYNAMIC', 2)])

def query_openchat_ultra_optimized(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    """
    Ultra-optimized single query with maximum GPU utilization
    """
    setup_inference_optimizations()
    
    # OpenChat 3.5 format
    formatted_prompt = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
    
    # Optimized tokenization
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=1800,
        add_special_tokens=False
    ).to(model.device, non_blocking=True)
    
    # Ultra-fast generation
    with torch.no_grad(), torch.cuda.amp.autocast():  # Use mixed precision
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.02,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            # Speed optimizations
            synced_gpus=False,
            output_scores=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict_in_generate=False,
            # Prevent excessive generation
            min_new_tokens=10,
            max_time=30.0  # Max 30 seconds per prompt
        )
    
    # Fast decode
    new_tokens = outputs[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return response.strip()

def query_openchat_batch_maximum_speed(model, tokenizer, prompts: List[str], 
                                     max_new_tokens: int = 150, batch_size: int = 4) -> List[str]:
    """
    Maximum speed batch processing optimized for RTX 4050 6GB
    """
    setup_inference_optimizations()
    
    all_responses = []
    total_start = time.time()
    
    print(f"🚀 Processing {len(prompts)} prompts in batches of {batch_size}")
    
    # Process in batches
    for batch_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_idx:batch_idx + batch_size]
        batch_start = time.time()
        
        # Format prompts
        formatted_prompts = [f"GPT4 Correct User: {p}<|end_of_turn|>GPT4 Correct Assistant:" for p in batch_prompts]
        
        # Batch tokenization with dynamic padding
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding="longest",  # Dynamic padding saves memory
            truncation=True,
            max_length=1800,
            add_special_tokens=False
        ).to(model.device, non_blocking=True)
        
        # Batch generation with mixed precision
        with torch.no_grad(), torch.cuda.amp.autocast():
            try:
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=40,
                    repetition_penalty=1.02,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    synced_gpus=False,
                    output_scores=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict_in_generate=False,
                    min_new_tokens=10,
                    max_time=120.0  # Max 2 minutes per batch
                )
                
                # Decode responses
                batch_responses = []
                for i, output in enumerate(outputs):
                    input_length = len(inputs.input_ids[i])
                    # Find actual end (remove padding tokens)
                    response_tokens = output[input_length:]
                    response = tokenizer.decode(response_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    batch_responses.append(response.strip())
                
                all_responses.extend(batch_responses)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ OOM in batch {batch_idx//batch_size + 1}, processing individually...")
                    torch.cuda.empty_cache()
                    
                    # Fallback to individual processing for this batch
                    for prompt in batch_prompts:
                        try:
                            response = query_openchat_ultra_optimized(model, tokenizer, prompt, max_new_tokens)
                            all_responses.append(response)
                        except Exception as e2:
                            print(f"Failed on individual prompt: {e2}")
                            all_responses.append(f"ERROR: {str(e2)}")
                else:
                    # Non-memory error, add error responses
                    error_msg = f"ERROR: {str(e)}"
                    all_responses.extend([error_msg] * len(batch_prompts))
        
        # Progress and performance monitoring
        batch_time = time.time() - batch_start
        completed = len(all_responses)
        total_elapsed = time.time() - total_start
        
        if completed > 0:
            avg_time_per_prompt = total_elapsed / completed
            remaining_prompts = len(prompts) - completed
            eta_minutes = (remaining_prompts * avg_time_per_prompt) / 60
            prompts_per_minute = completed / (total_elapsed / 60) if total_elapsed > 0 else 0
            
            # GPU monitoring
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                gpu_util = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
                
                print(f"Batch {batch_idx//batch_size + 1}: {completed}/{len(prompts)} | "
                      f"{prompts_per_minute:.1f} prompts/min | ETA: {eta_minutes:.1f}min | "
                      f"GPU: {gpu_util:.1f}% ({gpu_mem:.1f}GB)")
            else:
                print(f"Batch {batch_idx//batch_size + 1}: {completed}/{len(prompts)} | "
                      f"{prompts_per_minute:.1f} prompts/min | ETA: {eta_minutes:.1f}min")
        
        # Memory cleanup every few batches
        if (batch_idx // batch_size + 1) % 3 == 0:
            torch.cuda.empty_cache()
    
    total_time = time.time() - total_start
    success_count = len([r for r in all_responses if not r.startswith("ERROR")])
    avg_time = total_time / len(all_responses) if all_responses else 0
    
    print(f"\n🎉 Batch processing complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Success rate: {success_count}/{len(all_responses)} ({success_count/len(all_responses)*100:.1f}%)")
    print(f"Average: {avg_time:.1f}s per prompt")
    print(f"Throughput: {len(all_responses)/(total_time/60):.1f} prompts/minute")
    
    return all_responses

def query_openchat_streaming(model, tokenizer, prompt: str, max_new_tokens: int = 150):
    """
    Streaming generation for real-time feedback (optional)
    """
    setup_inference_optimizations()
    
    formatted_prompt = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Streaming generation
    generated_tokens = []
    with torch.no_grad():
        past_key_values = None
        
        for _ in range(max_new_tokens):
            if past_key_values is None:
                outputs = model(inputs.input_ids, use_cache=True)
            else:
                outputs = model(inputs.input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs.past_key_values
            
            # Sample next token
            logits = outputs.logits[:, -1, :] / 0.8  # temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_tokens.append(next_token.item())
            
            # Check for end token
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Prepare for next iteration
            inputs.input_ids = torch.cat([inputs.input_ids, next_token], dim=-1)
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()

# Drop-in replacement for your existing function
def query_openchat(model, tokenizer, prompt: str) -> str:
    """
    Drop-in replacement with maximum optimization
    """
    return query_openchat_ultra_optimized(model, tokenizer, prompt)

def benchmark_performance(model, tokenizer, num_test_prompts: int = 5):
    """
    Comprehensive performance benchmark
    """
    test_prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot.",
        "How does machine learning work?",
        "What are the benefits of renewable energy?"
    ][:num_test_prompts]
    
    print("=== PERFORMANCE BENCHMARK ===")
    
    # Single prompt test
    print(f"\n1. Single prompt test ({num_test_prompts} prompts):")
    single_start = time.time()
    single_responses = []
    
    for prompt in test_prompts:
        response = query_openchat_ultra_optimized(model, tokenizer, prompt)
        single_responses.append(response)
    
    single_time = time.time() - single_start
    single_avg = single_time / len(test_prompts)
    
    print(f"Total time: {single_time:.2f}s | Average: {single_avg:.2f}s per prompt")
    
    # Batch test
    print(f"\n2. Batch test ({num_test_prompts} prompts):")
    batch_start = time.time()
    batch_responses = query_openchat_batch_maximum_speed(
        model, tokenizer, test_prompts, batch_size=min(3, len(test_prompts))
    )
    batch_time = time.time() - batch_start
    batch_avg = batch_time / len(test_prompts)
    
    print(f"Total time: {batch_time:.2f}s | Average: {batch_avg:.2f}s per prompt")
    
    # Performance analysis
    speedup = single_time / batch_time if batch_time > 0 else 1
    print(f"\n📊 Batch speedup: {speedup:.2f}x")
    
    # Estimate for 142 prompts
    single_estimate = (single_avg * 142) / 60
    batch_estimate = (batch_avg * 142) / 60
    
    print(f"\n🔮 Estimates for 142 prompts:")
    print(f"Single processing: {single_estimate:.1f} minutes")
    print(f"Batch processing: {batch_estimate:.1f} minutes")
    print(f"Recommended: {'Batch' if speedup > 1.2 else 'Single'} processing")
    
    return batch_avg if speedup > 1.2 else single_avg

def check_gpu_utilization():
    """Enhanced GPU utilization checker"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    print("=== GPU UTILIZATION CHECK ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Memory info
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"Memory: {allocated:.2f}GB allocated | {reserved:.2f}GB reserved | {total:.2f}GB total")
    print(f"Utilization: {(allocated/total)*100:.1f}%")
    
    # Performance test
    print("\n🔥 GPU Performance Test:")
    try:
        # Matrix multiplication test
        size = 2048
        a = torch.randn(size, size, device='cuda', dtype=torch.float16)
        b = torch.randn(size, size, device='cuda', dtype=torch.float16)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(10):
            c = torch.mm(a, b)
        
        torch.cuda.synchronize()
        compute_time = time.time() - start
        
        operations = 10 * size * size * size * 2  # 10 iterations of matrix mult
        tflops = (operations / compute_time) / 1e12
        
        print(f"Compute performance: {tflops:.2f} TFLOPS")
        
        if tflops > 5:
            print("✅ Excellent GPU performance!")
        elif tflops > 2:
            print("✅ Good GPU performance")
        else:
            print("⚠️ Limited GPU performance")
        
        del a, b, c
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Performance test failed: {e}")

if __name__ == "__main__":
    # This would normally be called from your main script
    print("Ultra-fast query functions ready!")
    print("Use query_openchat_batch_maximum_speed() for 142 prompts")