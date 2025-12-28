# 🛡️ LLM Safety: Evaluation, Classification & Mitigation Framework

This repository contains a **practical LLM safety experimentation framework** focused on **classification, red-teaming, and mitigation of unsafe model behavior**.  
It explores multiple safety strategies using **API-based classifiers, open-source models, rule-based logic, and fine-tuned transformers**, along with pipelines for querying both hosted and local LLMs.

The project emphasizes **hands-on safety evaluation**, experimentation, and honest reporting of limitations.

---

## 🎯 Project Objective

Large Language Models can generate unsafe, biased, or policy-violating responses when exposed to adversarial or red-teaming prompts.  
This project aims to:

- Classify LLM responses as **safe / unsafe**
- Compare different **safety detection approaches**
- Apply and test **mitigation techniques**
- Analyze model behavior under red-teaming prompts
- Experiment with both **hosted APIs** and **local LLM instances**

---

## 🧠 System Overview

```
Red-Teaming Prompts
        ↓
LLM Query Pipeline
(ChatGPT / OpenChat)
        ↓
Generated Responses
        ↓
Safety Classification
(ChatGPT / Detoxify / DistilBERT / Hybrid)
        ↓
Mitigation Logic
        ↓
Post-Mitigation Analysis
```

---

## 🧩 Core Components

### 🔹 1. ChatGPT-based Classifier
**`classfier_chatgpt.py`**

- Uses the **GPT-4o-mini API** as a zero-shot classifier
- Prompts the model to label responses as safe or unsafe
- Acts as a strong semantic baseline for classification

---

### 🔹 2. Detoxify-based Classifier
**`classifier_detoxify.py`**

- Uses the **Detoxify** model for toxicity detection
- Outputs:
  - Toxicity scores
  - Binary safety labels
- Provides quantitative safety metrics

---

### 🔹 3. Hybrid Classifier
**`classifier_hybrid.py`**

- Combines **Detoxify predictions** with **rule-based overrides**
- Improves detection quality by:
  - Catching edge cases missed by Detoxify
  - Enforcing hard safety rules

---

### 🔹 4. DistilBERT Training & Evaluation
**`distilbert.py`**

- Fine-tunes a **DistilBERT** model on labeled response data
- Produces:
  - Confusion matrix
  - Accuracy / precision / recall metrics
  - Training and evaluation plots

This component explores **learned safety classifiers** over heuristic ones.

---

### 🔹 5. OpenChat Query Pipelines (Local & Hosted)

**Local pipeline**
- `load_openchat.py`
- `load_prompts.py`
- `define_query.py`
- `query.py`

Used to interact with a **locally hosted OpenChat model**.

**Hosted pipeline**
- `response_openchat.py`

Sends red-teaming prompts to **OpenChat / OpenChat-3.5-1210 hosted on Hugging Face** and collects responses for analysis.

---

### 🔹 6. Mitigation Engine
**`mitigation.py`**

- Applies mitigation strategies to unsafe prompts or responses
- Designed to reduce harmful output while preserving task intent
- Demonstrates prompt-level and response-level mitigation logic

---

## 📂 Repository Structure

```
llm-safety/
├── classfier_chatgpt.py
├── classifier_detoxify.py
├── classifier_hybrid.py
├── distilbert.py
├── mitigation.py
├── load_openchat.py
├── load_prompts.py
├── define_query.py
├── query.py
├── response_openchat.py
├── labeled_responses.csv
├── red_teaming_prompts.csv
├── outputs.csv
├── requirements.txt
├── documentation.docx
└── vault/
```

---

## 🚀 How to Use

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run safety classification
```bash
python classifier_hybrid.py
```

### 3️⃣ Train DistilBERT classifier
```bash
python distilbert.py
```

### 4️⃣ Query OpenChat (hosted)
```bash
python response_openchat.py
```

### 5️⃣ Apply mitigation
```bash
python mitigation.py
```

---

## ⚠️ Honest Limitation & Explanation

During development, responses for mitigation testing were **generated using Gemini** instead of reusing the existing labeled prompt-response dataset.

As a result:
- A **direct before/after quantitative comparison** of mitigation effectiveness is missing
- Corresponding **comparison visualizations** could not be produced

However:
- The mitigation logic itself is correctly implemented
- **Run examples demonstrating mitigation effectiveness** are included in the documentation
- This limitation is **non-fundamental** and can be easily rectified by re-running mitigation on the existing dataset

This is an acknowledged design mistake made under **time constraints**, not a conceptual flaw.

---

## 🧠 What This Project Demonstrates

- Practical understanding of **LLM safety challenges**
- Experience with **multiple safety classifiers**
- Comparison of rule-based vs learned approaches
- Red-teaming and response analysis workflows
- Awareness of experimental limitations and honest reporting
- End-to-end thinking: prompts → responses → detection → mitigation

---

## ⚠️ Disclaimer

This project is an **experimental exploration of LLM safety techniques**.  
It is **not a production-ready safety system**.

---

## 📄 License

MIT License
