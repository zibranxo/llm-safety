import torch
import time
from typing import List

# global initilization once only

def setup_inference_optimizations():
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    torch.set_num_threads(4)

def format_openchat_prompt(prompt: str) -> str:
    """
    Centralized OpenChat prompt format.
    Keeps pipeline consistent.
    """
    return f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
    
def query_openchat_ultra(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    use_amp: bool = True
) -> str:

    formatted_prompt = format_openchat_prompt(prompt)

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1800,
        add_special_tokens=False
    ).to(model.device)

    autocast_ctx = (
        torch.cuda.amp.autocast() if use_amp and torch.cuda.is_available()
        else torch.no_grad()
    )

    with torch.no_grad(), autocast_ctx:
        output = model.generate(
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
            return_dict_in_generate=False,
            min_new_tokens=10,
            max_time=30.0
        )

    new_tokens = output[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

#batch processing
def query_openchat_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 150,
    batch_size: int = 4,
    use_amp: bool = True
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

        autocast_ctx = (
            torch.cuda.amp.autocast() if use_amp and torch.cuda.is_available()
            else torch.no_grad()
        )

        try:
            with torch.no_grad(), autocast_ctx:
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
                    return_dict_in_generate=False,
                    min_new_tokens=10,
                    max_time=120.0
                )

            for j, out in enumerate(outputs):
                start = inputs.input_ids[j].shape[0]
                text = tokenizer.decode(
                    out[start:], skip_special_tokens=True
                ).strip()
                results.append(text)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                # fallback: single prompt
                for p in batch:
                    try:
                        results.append(
                            query_openchat_ultra(
                                model, tokenizer, p, max_new_tokens, use_amp
                            )
                        )
                    except Exception as err:
                        results.append(f"ERROR: {err}")
            else:
                results.extend([f"ERROR: {e}"] * len(batch))

    elapsed = time.time() - total_start
    print(
        f"Processed {len(prompts)} prompts "
        f"in {elapsed/60:.2f} min "
        f"({len(prompts)/(elapsed/60):.1f} prompts/min)"
    )

    return results

def query_openchat_streaming(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 150
) -> str:

    formatted_prompt = format_openchat_prompt(prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    generated = []
    past_key_values = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                inputs.input_ids[:, -1:] if past_key_values else inputs.input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :] / 0.8
            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)

            if token.item() == tokenizer.eos_token_id:
                break

            generated.append(token.item())
            inputs.input_ids = torch.cat([inputs.input_ids, token], dim=-1)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()

#entrypoint
def query_openchat(model, tokenizer, prompt: str) -> str:
    return query_openchat_ultra(model, tokenizer, prompt)
