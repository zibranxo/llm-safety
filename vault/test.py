import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "openchat/openchat-3.5-0106"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Read prompts from CSV
with open("red_teaming_prompts.csv", "r", encoding="utf-8") as infile, open("outputs.csv", "w", newline="", encoding="utf-8") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    writer.writerow(["Prompt", "Response"])  # Output header

    for row in reader:
        prompt = row[0]
        result = generator(prompt, max_new_tokens=100)[0]['generated_text']
        writer.writerow([prompt, result])
