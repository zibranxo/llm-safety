import pandas as pd

def load_prompts(filepath="red_teaming_prompts.csv"):
    """Load prompts from CSV file with error handling"""
    try:
        df = pd.read_csv(filepath, sep=',', quotechar='"', on_bad_lines='skip')
        prompts = df["prompt"].dropna().tolist()  # Remove any NaN values
        print(f"Loaded {len(prompts)} prompts from {filepath}")
        return prompts
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return []
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return []