import pandas as pd
import requests
import time

API_URL = "https://api-inference.huggingface.co/models/openchat/openchat-3.5-1210"
HF_TOKEN = "LoL"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(prompt, max_tokens=200):
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens}
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result[0]["generated_text"]
    else:
        print("Error:", response.status_code, response.text)
        return "ERROR"

df = pd.read_csv("red_teaming_prompts.csv")

results = []
for index, row in df.iterrows():
    prompt = row['prompt']
    print(f"Processing prompt {index + 1}/{len(df)}")
    completion = query(prompt)
    results.append({"id": row['id'], "prompt": prompt, "response": completion})
    time.sleep(2)  # avoid rate limiting

output_df = pd.DataFrame(results)
output_df.to_csv("output.csv", index=False)

print("Done! Responses saved to output.csv")
