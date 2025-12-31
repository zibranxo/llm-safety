import os
import gc
import time
import psutil
import torch
from typing import List

#global env steup (once)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

#memory utilities
def clear_memory(only_if_cuda=True):
    gc.collect()
    if torch.cuda.is_available() and only_if_cuda:
        torch.cuda.empty_cache()

def format_openchat_prompt(prompt: str) -> str:
    return f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"

@torch.inference_mode()
def query_openchat_single(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8
) -> str:
    inputs = tokenizer(
        format_openchat_prompt(prompt),
        return_tensors="pt",
        truncation=True,
        max_length=1800,
        add_special_tokens=False
    ).to(model.device)

    output = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.02,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        max_time=30.0
    )

    gen_tokens = output[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

@torch.inference_mode()
def query_openchat_batch(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 4,
    max_new_tokens: int = 150
) -> List[str]:
    results = []
    total_start = time.time()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        formatted = [format_openchat_prompt(p) for p in batch]

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=1800,
            add_special_tokens=False
        ).to(model.device)

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
                max_time=120.0
            )

            for j, out in enumerate(outputs):
                start = inputs.input_ids[j].shape[0]
                text = tokenizer.decode(out[start:], skip_special_tokens=True).strip()
                results.append(text)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                clear_memory()
                for p in batch:
                    try:
                        results.append(
                            query_openchat_single(
                                model, tokenizer, p, max_new_tokens
                            )
                        )
                    except Exception as err:
                        results.append(f"ERROR: {err}")
            else:
                results.extend([f"ERROR: {e}"] * len(batch))

        # light monitoring
        if (i + batch_size) % (batch_size * 10) == 0:
            ram = psutil.virtual_memory().percent
            print(f"[Progress] {i+len(batch)}/{len(prompts)} | RAM {ram:.1f}%")

    elapsed = time.time() - total_start
    print(
        f"Completed {len(prompts)} prompts "
        f"in {elapsed/60:.2f} min "
        f"({len(prompts)/(elapsed/60):.1f} prompts/min)"
    )

    return results

#warmup
def warmup_model(model, tokenizer):
    print("Running warmup...")
    _ = query_openchat_single(
        model,
        tokenizer,
        "Say hello in one sentence.",
        max_new_tokens=20
    )
    clear_memory()
    print("Warmup done.")


#pipeline entrypoint
def run_inference_pipeline(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 4
) -> List[str]:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    warmup_model(model, tokenizer)
    return query_openchat_batch(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size
    )
