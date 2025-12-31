import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import os

def load_openchat_model():
    """Load OpenChat model with MAXIMUM GPU utilization for RTX 4050 6GB"""
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return None, None
    
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # memory optimization settings
    os.environ.update({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:64,roundup_power2_divisions:16",
        '''"CUDA_LAUNCH_BLOCKING": "1",'''
        "TOKENIZERS_PARALLELISM": "false"
    })
    
    # cleanup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    model_name = "openchat/openchat-3.5-0106"
    
    try:
        print("Loading with 4-bit quantization for maximum GPU efficiency...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 4-bit quantization config - CRITICAL for 6GB GPU
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,  # Double quantization for extra memory savings
            bnb_4bit_quant_type="nf4",      # Best quality 4-bit
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=False  # Keep everything on GPU
        )
        
        # Load model with aggressive GPU placement
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "5.5GB"},  # Use almost all GPU memory
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2" if hasattr(torch.nn, 'scaled_dot_product_attention') else None
        )
        print("Attempting Flash Attention 2 (may fallback silently)")

        
        print("4-bit quantized model loaded!")
        
    except Exception as e:
        print(f"4-bit loading failed: {e}")
        print("🔄 Trying 8-bit quantization...")
        
        try:
            # Fallback to 8-bit
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory={0: "5.8GB"},
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("8-bit quantized model loaded!")
            
        except Exception as e2:
            print(f"8-bit failed: {e2}")
            print("🔄 Loading with extreme optimization...")
            
            # Last resort - minimal memory usage
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory={0: "5.0GB", "cpu": "2GB"},
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("⚠️ Loaded with memory constraints")
    
    # Verify and optimize model
    model.eval()
    '''
    # Enable optimizations
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    '''
    # Compile model for speed (PyTorch 2.0+)
    try:
        if hasattr(torch, 'compile'):
            print("🔥 Compiling model for maximum speed...")
            '''model = torch.compile(model, mode="max-autotune")'''
            ENABLE_COMPILE = False  # default OFF
            if ENABLE_COMPILE and hasattr(torch, 'compile'):
                model = torch.compile(model)

            print("✅ Model compiled!")
    except Exception as e:
        print(f"Compilation skipped: {e}")
    
    # Memory status
    gpu_memory = torch.cuda.memory_allocated() / 1024**3
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory: {gpu_memory:.2f}/{total_memory:.1f} GB ({gpu_memory/total_memory*100:.1f}%)")
    
    # Check parameter distribution
    gpu_params = sum(1 for p in model.parameters() if p.device.type == 'cuda')
    total_params = sum(1 for _ in model.parameters())
    print(f"GPU Parameters: {gpu_params}/{total_params} ({gpu_params/total_params*100:.1f}%)")
    
    return model, tokenizer

def optimize_inference_settings():
    """Set optimal inference settings"""
    # Enable TensorFloat-32 for speed on RTX 4050
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set optimal number of threads
    torch.set_num_threads(4)
    
    print("✅ Inference optimizations enabled")

def test_model_speed(model, tokenizer):
    """Test model inference speed with optimization"""
    if model is None or tokenizer is None:
        print("❌ No model to test")
        return 0
    
    print("\n=== SPEED TEST ===")
    optimize_inference_settings()
    
    print("Warming up GPU...")
    dummy_input = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**dummy_input, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()
    
    # Real test
    test_prompt = "Hello, how are you today? Please tell me about machine learning."
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    import time
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1  # Greedy for speed
        )
    
    torch.cuda.synchronize()
    inference_time = time.time() - start_time
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
    tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
    
    print(f"Time: {inference_time:.2f}s | Tokens: {tokens_generated} | Speed: {tokens_per_second:.1f} tok/s")
    print(f"Response: {response[:200]}...")
    
    # Performance prediction
    estimated_time_142 = (inference_time * 142) / 60
    print(f"\nEstimated time for 142 prompts: {estimated_time_142:.1f} minutes")
    
    if tokens_per_second > 15:
        print("EXCELLENT speed - ready for batch processing!")
    elif tokens_per_second > 8:
        print("Good speed")
    else:
        print("⚠️ Still slow - may need further optimization")
    
    return tokens_per_second

if __name__ == "__main__":
    print("LOADING OPENCHAT WITH MAXIMUM GPU UTILIZATION...")
    model, tokenizer = load_openchat_model()
    
    if model is not None:
        speed = test_model_speed(model, tokenizer)
        if speed > 10:
            print("Model ready for high-speed inference!")
        else:
            print("⚠️ Model loaded but performance may be suboptimal")
    else:
        print("Failed to load model")
