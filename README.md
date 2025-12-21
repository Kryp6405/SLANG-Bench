# SLANG-Bench

A benchmark and training pipeline for evaluating and **improving LLM robustness to modern slang**, using supervised fine-tuning, retrieval-augmented generation (RAG), and reinforcement learning (RL).

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://tinyurl.com/slang-bench-paper)

---

## 🌐 Motivation

Modern LLMs are usually trained on large, relatively formal text. They often struggle with:

- Rapidly evolving **internet slang**
- Platform-specific abbreviations (e.g., Snapchat, Discord, texting)
- Short, context-dependent expressions

**SLANG-Bench** provides:
- A **curated slang dictionary** (terms + canonical definitions + examples)
- A set of **forward and reverse tasks** that probe understanding
- A **training / evaluation pipeline** to test:
  - Pretrained base models (baseline)
  - RAG variants
  - Supervised fine-tuning (SFT)
  - RL with task-aligned rewards (F1, semantic similarity, hybrid)

---

## 📁 Repository Structure

```text
SLANG-Bench/
├── forward/             # Forward task: sentence with blank → choose correct slang (MCQ)
├── reverse/             # Reverse task: sentence with slang → generate meaning (SFT + RL, F1 reward)
├── reverse_semantic/    # Reverse task variants with semantic / hybrid rewards
├── dataset.csv          # Core slang dictionary + curated examples for training
├── eval.csv             # Evaluation set (forward + reverse variants)
└── misc.ipynb           # Scratch experiments / analysis
```

High-level:

- **`dataset.csv`**  
  Curated slang-term dataset with:
  - `slang` term
  - Canonical **definition / gloss**
  - 2+ example sentences per term

- **`eval.csv`**  
  Evaluation data derived from `dataset.csv`, with:
  - Forward-task examples (cloze MCQ: choose the slang)
  - Reverse-task examples (cloze MCQ: choose the meaning)
  - Reverse-Semantic-task examples (sentence with slang → meaning)
  - Labels/columns formatted to match the Python evaluation code (e.g., `sentence`, options and correct label for MCQ; `prompt` + `completion` for reverse).

- **`forward/`**  
  Code/notebooks for:
  - Building MCQ prompts: sentence with the slang word blank + 4 candidate slang words (A–D)
  - Baseline evaluation on the forward MCQ task
  - SFT training loop (LoRA via Tinker) for predicting the correct **letter** (A/B/C/D)
  - Re-evaluating the finetuned model on the same MCQ benchmark

- **`reverse/`**  
  Code/notebooks for:
  - Building MCQ prompts: sentence including a slang word + 4 candidate meanings for the slang word (A–D)
  - Baseline evaluation on the forward MCQ task
  - SFT training loop (LoRA via Tinker) for predicting the correct **letter** (A/B/C/D)
  - Re-evaluating the finetuned model on the same MCQ benchmark

- **`reverse_semantic/`**  
  Code/notebooks for:
  - Reverse **explanation** task:  
    > *“You will be given a sentence that contains a modern slang term. Explain what the sentence means in standard English, focusing on the slang.”*
  - SFT on `{"prompt": ..., "completion": <canonical meaning>}`
  - RL fine-tuning (GRPO-style) with an **F1-based reward** over tokens:
    - sample generations from the current policy
    - compute reward = generated tokens vs gold definition
      - **Token Similarity**: mean token F1 similarity reward
      - **Semantic Similarity**: cosine similarity in embedding space reward
      - **Hybrid**: weighted combination of token F1 + semantic similarity (+ optional penalties for verbosity / repetition)
    - compute group-relative advantages and update with `importance_sampling` loss

- **`misc.ipynb`**  
  Ad-hoc analysis, plotting loss curves, debugging generations, etc.

---

## ⚙️ Setup

Recommended:

- Python 3.10+
- Access to a compatible base model, e.g. `meta-llama/Llama-3.1-8B-Instruct`, via [Tinker](https://tinker-docs.thinkingmachines.ai/)

Install core dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

---

## 🧪 Tasks & Pipelines

### 1. Forward Task: Slang Cloze MCQ

**Goal:**  
Given a sentence with a blank, choose the correct slang word from 4 options (A–D).

**Example:**

> Sentence: `"That ____ is so good on you."`  
> Options: A) lowkey  B) drip  C) ratio  D) nrn

**Pipeline:**

1. Build evaluation prompts from `eval.csv`:
   - `row_to_prompt(row)` → MCQ-style instruction + sentence + options
2. **Baseline evaluation**:
   - Wrap base model in a Tinker sampling client
   - Generate the next few tokens
   - Parse the first letter A/B/C/D
   - Compute **accuracy**, **confusion matrix**, **classification report**
3. **SFT**:
   - Convert each training example into `{"prompt": ..., "completion": "<letter>"}` JSONL
   - Use a LoRA SFT loop where:
     - Loss is only applied on the answer letter tokens
   - Save sampler weights and re-run eval

---

### 2. Reverse Task (SFT): Slang In-Context Definition
**Goal:**  
Given a sentence with the slang word, choose the correct in-context definition of the slang word from 4 options (A–D).

**Example:**

> Sentence: `"That drip is so good on you."`  
> Options: A) Drip means clothing  B) ...  C) ...  D) Drip means water

**Pipeline:**

1. Build evaluation prompts from `eval.csv`:
   - `row_to_prompt(row)` → MCQ-style instruction + sentence + options
2. **Baseline evaluation**:
   - Wrap base model in a Tinker sampling client
   - Generate the next few tokens
   - Parse the first letter A/B/C/D
   - Compute **accuracy**, **confusion matrix**, **classification report**
3. **SFT**:
   - Convert each training example into `{"prompt": ..., "completion": "<letter>"}` JSONL
   - Use a LoRA SFT loop where:
     - Loss is only applied on the answer letter tokens
   - Save sampler weights and re-run eval

---

### 3. Reverse Semantic Task: Slang Meaning
**Goal:**  
Given a sentence containing a slang term, generate a short **standard-English explanation** of the slang.

**Example prompt:**

```text
You will be given a sentence that contains a modern slang term.
Explain what the sentence means in standard English, focusing on the slang.

Sentence: "The movie was fine, but honestly kind of meh."
Meaning:
```

**Target completion:**

```text
Expresses indifference or lack of enthusiasm about something.
```

**Pipeline:**

1. Build `finetune_reverse.jsonl` from `dataset.csv`:
   - `{"prompt": <instruction + sentence>, "completion": <gold meaning>}`
2. Run SFT with Tinker:
   - Same `make_datums` pattern as forward task, but completion is **free-form text**, not a letter.
3. Evaluate on held-out `eval` split:
   - Generate an explanation for each prompt.
   - Compute:
     - **Token-level F1** vs target meaning
     - **BERTScore F1**
     - **Exact match rate**
     - **Average predicted length** (words)

4. Improve explanation quality beyond SFT by optimizing a task-aligned reward.
    - **F1 reward** (main RL run)

      ```python
      reward = explanation_f1(pred_text, target_text)   # token-level F1 in [0,1]
      ```

      - **Semantic reward** (ablation)

      ```python
      reward = cosine_sim(emb(pred), emb(target))       # embedding similarity in [0,1]
      ```

      - **Hybrid reward** (ablation)

      ```python
      reward = 0.4 * F1 + 0.6 * semantic_similarity
      # + optional penalties for length/repetition
      ```

**GRPO-style training loop** (high level)

For each RL step:

1. Sample a mini-batch of prompts from `finetune_reverse_rl.jsonl`.
2. For each prompt, generate `GROUP_SIZE` completions from the current policy.
3. For each completion:
   - Clean it with `parse_explanation(...)`
   - Compute reward with one of the reward functions above.
4. For each prompt group:
   - Compute group-wise baseline `mean_reward`
   - Compute advantages `A_i = R_i - mean_reward`
5. Build `types.Datum` objects with:
   - `target_tokens`, `logprobs`, and `advantages`
6. Call:

```python
training_client.forward_backward(datums, loss_fn="importance_sampling")
training_client.optim_step(adam_params)
```

7. Log metrics per step:
   - mean reward, number of datums, time per step
8. Periodically save sampler weights; at the end, save final RL checkpoint.

Evaluation uses the same **reverse explanation metrics** as SFT.

---

## 🧪 Reproducing Experiments (High Level)

1. **Prepare data**
   - Ensure `dataset.csv` and `eval.csv` are in place.
   - Use the provided notebooks/scripts in each subfolder to:
     - Build JSONL for SFT / RL (`finetune_*.jsonl`, `eval_*.jsonl`).

2. **Forward task (MCQ)**
   - Run the forward notebook/script to:
     - Evaluate the **baseline** model on MCQ.
     - Train an SFT model and re-evaluate.

3. **Reverse task (SFT)**
   - Run the reverse notebook/script to:
     - Train SFT on `finetune_reverse.jsonl`.
     - Evaluate on the reverse eval set with F1/BERTScore/EM.

4. **Reverse task (RL)**
   - Start from the saved SFT state.
   - Run the RL loop (F1 / semantic / hybrid) from `reverse/` and `reverse_semantic/`.
   - Evaluate each RL variant with the same metrics for direct comparison.

