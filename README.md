
# Personality and Bias Detection from User Data

This project bridges psychology and machine learning to extract personality traits and detect financial decision-making biases from user-generated text. It leverages cutting-edge NLP models, fine-tuning techniques, and behavioral science principles to analyze linguistic patterns and infer deeper psychological insights.

## 📌 Table of Contents

- [Introduction](#introduction)
- [Problem Statements](#problem-statements)
- [Methodology](#methodology)
- [Experiments and Results](#experiments-and-results)
- [Datasets](#datasets)
- [Modeling Techniques](#modeling-techniques)
- [Bias Detection](#bias-detection)
- [Visual Insights](#visual-insights)
- [Conclusion and Future Work](#conclusion-and-future-work)

---

## 🧠 Introduction

This project consists of two core objectives:

1. **Personality Trait Extraction:**  
   Predict the Big Five (OCEAN) personality traits from user-generated content using transformer-based models and prompting strategies.

2. **Financial Bias Detection:**  
   Identify cognitive biases in financial decision-making by analyzing user posts, primarily from social media and forums like Reddit.

---

## 🔍 Problem Statements

### Problem 1: Personality Trait Extraction
Classify user posts into OCEAN traits:
- **Openness:** Imagination, abstract thinking.
- **Conscientiousness:** Discipline, planning.
- **Extraversion:** Sociability, energy.
- **Agreeableness:** Empathy, politeness.
- **Neuroticism:** Anxiety, emotional instability.

### Problem 2: Financial Bias Detection
Detect 16 well-documented financial biases such as:
- Loss Aversion
- Overconfidence
- Herd Behavior
- Status Quo Bias
- Sunk Cost Fallacy

---

## ⚙️ Methodology

### Data Sources

| Attribute         | Essay Corpus                       | Pandora Corpus                    |
|------------------|-------------------------------------|------------------------------------|
| Source           | Public psychology dataset           | Internal dataset with 1.9M entries |
| Content Type     | Long essays (~600 words)            | Short user posts (~50–100 words)   |
| Volume           | ~2,400 essays                       | ~1.9 million posts                 |

### Preprocessing
- Essay corpus: Lemmatization, stopword removal, tokenization.
- Pandora: Emoji/slang filtering, context-aware tokenization, normalization (0–1 scale).

---

## 🧪 Experiments and Results

### Personality Trait Detection (Pandora)

| Model              | O | C | E | A | N | Overall Accuracy |
|-------------------|---|---|---|---|---|------------------|
| PersLLM (SOTA)     | 68.9 | 66.7 | 73.7 | 68.6 | 69.5 | 69.5 |
| LLaMA-3.3 (Ours)   | 75 | 72 | 74 | 71 | 73 | 73 |
| LLaMA-4 (Ours)     | 74 | 72 | 73 | 73 | 73 | 73 |
| DistilBERT (Ours)  | 65 | 64 | 63 | 65 | 65.5 | 64.5 |
| RoBERTa-FT (Ours)  | 64 | 63 | 62.1 | 64 | 63 | 63.2 |

---

## 📊 Modeling Techniques

### Prompt-Based Zero-Shot
- Models: `LLaMA`, `Mistral`, `meta-llama`
- Direct inference using well-crafted personality psychology prompts.

### Fine-Tuned Transformers
- **DistilBERT**
  - LoRA: Rank 8, Alpha 16
  - Binary classification with BCE loss
- **RoBERTa**
  - Trait-specific binary classifiers
  - Efficient parameter tuning

---

## 🧭 Bias Detection

### Financial Biases Detected
16 financial biases including:
- **Loss Aversion**
- **Status Quo Bias**
- **Sunk Cost Fallacy**
- **Mental Accounting**
- **Framing Effect**

### Prompt Strategy
```text
You are a behavioral finance expert. Analyze the following statement and determine if any of the financial decision-making biases from the list below are present...
```

### Bias Detection Experiments
1. **No Keywords** — Trait-based scoring.
2. **With Keywords** — Direct keyword match.
3. **Semantic Expansion** — Fuzzy matching using similar terms.
4. **Filtered Analysis** — Only consider decision-relevant posts.

---

## 📈 Visual Insights

- **Reddit Analysis:** 6708 posts from 334 users
- **Conscientiousness** was dominant (~98% users).
- **Loss Aversion** and **Status Quo** were the most frequent biases.
- **Bias Clusters:** Present Bias co-occurs strongly with Status Quo and Sunk Cost.

### Sample Plots
- Trait correlation matrices
- Heatmaps of co-occurring biases
- Boxplots and distribution curves for bias intensity

---

## 🔮 Conclusion and Future Work

### Key Takeaways
- Prompting and fine-tuning LLMs yields reliable detection of both personality and bias.
- OCEAN traits are inferable from informal short posts with high accuracy.
- Behavioral biases tend to cluster, calling for multi-bias intervention strategies.

### Future Scope
- Develop a conversational assessment bot.
- Integrate insights for **personalized financial advice**.
- Extend to real-time decision support systems.

---

## 🛠️ Tech Stack

- Python, PyTorch, HuggingFace Transformers
- PEFT (LoRA)
- Datasets: Essay Corpus, Pandora Corpus, Reddit
- Visualizations: Matplotlib, Seaborn

---

## 📄 References

- McCrae & Costa (1987), Goldberg (1990) — OCEAN Personality Framework
- PsyAttention: Psychological Attention Model for Personality Detection
- Behavioral Finance Review Literature

---

## 📬 Contact

**Author:** Madhav Singh  
Feel free to reach out via [LinkedIn](https://www.linkedin.com/) or GitHub Issues for collaboration or queries.

---

## 🧠 Academic References

This project is inspired and supported by recent research in computational psychology and behavioral finance. Key references include:

- **Brown et al. (2020)**: "Language Models are Few-Shot Learners" – Introduction of GPT-3 and its zero-shot reasoning capabilities.
- **Hu et al. (2021)**: "LoRA: Low-Rank Adaptation of Large Language Models" – Parameter-efficient fine-tuning method used in RoBERTa-based classifiers.
- **Campbell et al. (2023)**: Relationship between Big Five traits and financial behavior.
- **Pennebaker et al. (2001)**: LIWC tool and early personality prediction using psycholinguistic features.
- **Golbeck et al. (2011)**: Predicting personality from Twitter using regression techniques.
- **Majumder et al. (2017)**: Deep learning architectures for personality classification.
- **Vinit & Singh (2021)**: Using BERT embeddings for personality classification.
- **Liu et al. (2023)**: Survey of the pre-train prompting paradigm for LLMs.

These foundational works support the techniques and theoretical direction of this project.

---
