---
base_model: unsloth/llama-3-8b
language:
- en
license: llama3
library_name: transformers
pipeline_tag: text-generation
datasets:
- theatticusproject/cuad-qa
tags:
- legal
- contracts
- contract-review
- cuad
- extraction
- llama-3
- unsloth
- gguf
---

<div align="center">

# ⚖️ Llama-3-Legal-8B

**Finds and quotes clauses in commercial contracts, or tells you plainly that they aren't there.**

[![Built with Meta Llama 3](https://img.shields.io/badge/Built%20with-Meta%20Llama%203-0467DF)](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
[![License: Llama 3](https://img.shields.io/badge/License-Llama%203%20Community-blue)](./LICENSE)
[![Dataset: CUAD](https://img.shields.io/badge/Dataset-CUAD-orange)](https://huggingface.co/datasets/theatticusproject/cuad-qa)
[![GGUF](https://img.shields.io/badge/GGUF-Q8__0%20%7C%20Q4__K__M-green)](#run-locally-gguf)

</div>

| **0.735** | **99.6%** | **95.8%** |
|:---:|:---:|:---:|
| word overlap with the correct clause<br>(previous version: 0.185) | of answers are real text from the contract<br>(previous version: 12.3%) | correct "Not found." when the clause is absent<br>(previous version: 0%) |

<sub>Measured on 1,000 pages from 102 contracts the model never saw during training.</sub>

---

## What it does

Give the model **a clause type** and **a piece of a contract**. It returns **the exact sentences from the contract** about that clause, or `Not found.`.

> **Question:** Highlight the parts (if any) of this contract related to "Governing Law" that should be reviewed by a lawyer.
>
> **Contract excerpt:** *…This Agreement shall be interpreted in accordance with the laws of the State of Indiana. Neither party may assign…*
>
> **Model:** `This Agreement shall be interpreted in accordance with the laws of the State of Indiana.`

**Who it's for:** lawyers, paralegals, and legal-tech developers who need to **locate** specific clauses in long contracts quickly, the way a highlighter would.

**Not for:** legal advice, judging whether a clause is good or bad, or general legal chat. It points to the text; a person makes the call.

It recognises the **41 clause types** of the [CUAD](https://www.atticusprojectai.org/cuad) dataset:

<details>
<summary><b>Show all 41 clause types</b></summary>

Document Name · Parties · Agreement Date · Effective Date · Expiration Date · Renewal Term · Notice Period To Terminate Renewal · Governing Law · Most Favored Nation · Non-Compete · Exclusivity · No-Solicit Of Customers · Competitive Restriction Exception · No-Solicit Of Employees · Non-Disparagement · Termination For Convenience · Rofr/Rofo/Rofn · Change Of Control · Anti-Assignment · Revenue/Profit Sharing · Price Restrictions · Minimum Commitment · Volume Restriction · Ip Ownership Assignment · Joint Ip Ownership · License Grant · Non-Transferable License · Affiliate License-Licensor · Affiliate License-Licensee · Unlimited/All-You-Can-Eat-License · Irrevocable Or Perpetual License · Source Code Escrow · Post-Termination Services · Audit Rights · Uncapped Liability · Cap On Liability · Liquidated Damages · Warranty Duration · Insurance · Covenant Not To Sue · Third Party Beneficiary

</details>

---

## Quick start

### 1. One page of a contract

Write the message as **the question, then a line containing only `---`, then the contract text**. The prompt format is stored in the tokenizer, so `apply_chat_template` builds the correct prompt for you.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "CSroseX/Llama-3-Legal-8B"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16, device_map="auto")

question = 'Highlight the parts (if any) of this contract related to "Governing Law" that should be reviewed by a lawyer.'
excerpt = "...paste up to about one page (~500 tokens) of the contract here..."

messages = [{"role": "user", "content": f"{question}\n---\n{excerpt}"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
print(tok.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

> 💡 **Small GPU?** Load in 4-bit (about 6 GB of VRAM) by adding `quantization_config=BitsAndBytesConfig(load_in_4bit=True)` to `from_pretrained`.

### 2. A whole contract

The model was trained on **pages of about 500 tokens**. For a full contract, split it into overlapping pages, ask each page, and merge the answers:

```python
def find_clause(contract, question, window=512, overlap=64):
    offsets = tok(contract, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    n = len(offsets)
    starts = [s for s in range(0, n, window - overlap) if s == 0 or s < n - overlap]
    found = []
    for s in starts:
        page = contract[offsets[s][0] : offsets[min(s + window, n) - 1][1]]
        messages = [{"role": "user", "content": f"{question}\n---\n{page}"}]
        inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
        output = model.generate(**inputs, max_new_tokens=256)
        answer = tok.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if not answer.lower().startswith("not found"):
            found += [line.strip() for line in answer.split("\n") if line.strip()]
    unique = list(dict.fromkeys(found))
    return [a for a in unique if not any(a != b and a in b for b in unique)] or ["Not found."]

print(find_clause(open("contract.txt").read(), question))
```

The last line removes duplicates. When a clause sits where two pages overlap, the model quotes it from both pages, and only the complete version is kept.

### Run locally (GGUF)

| File | Size | Use when |
|---|---|---|
| `unsloth.Q8_0.gguf` | 8.5 GB | Best quality, needs ~10 GB RAM |
| `unsloth.Q4_K_M.gguf` | 4.9 GB | Ordinary laptops, slightly lower quality |

Both files contain the prompt format, for apps such as LM Studio and llama.cpp. If your app does **not** apply it, send this exact prompt yourself:

```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{question}
Quote the exact text from the contract excerpt that answers this, one passage per line. If there is none, answer "Not found."

### Input:
{contract excerpt}

### Response:
```

---

## How well it works

All three models took the **same exam**: 500 pages that contain the answer and 500 that don't, cut from CUAD's **102 test contracts**. None of these contracts were used in training. A program does the marking.

| Score | Base Llama 3 8B | Previous version | **Llama-3-Legal-8B** |
|---|:---:|:---:|:---:|
| Word overlap with the correct clause ↑ | 0.003 | 0.185 | **0.735** |
| Answers found (≥ half the words right) ↑ | 0.2% | 14.6% | **73.8%** |
| Missed: said "Not found." but the answer was there ↓ | 86.8% | 0.0% | **8.2%** |
| Correct "Not found." when nothing was there ↑ | 83.6% | 0.0% | **95.8%** |
| Answer is **not** real text from the contract ↓ | 93.2% | 87.7% | **0.4%** |

**By clause type** (word overlap, new model):

| Strongest | Score | | Weakest (≥ 10 test pages) | Score |
|---|:---:|---|---|:---:|
| Renewal Term | 0.97 | | Change Of Control | 0.28 |
| Governing Law | 0.97 | | Parties | 0.25 |
| Expiration Date | 0.96 | | | |
| Document Name | 0.95 | | | |
| Agreement Date | 0.93 | | | |
| Effective Date | 0.91 | | | |
| Anti-Assignment | 0.86 | | | |

**Whole-contract test:** 3 complete test contracts × 6 clause types, run through the page-splitting code above. Average score **0.77**.

---

## How it was trained

| | |
|---|---|
| Base model | Llama 3 8B (`unsloth/llama-3-8b-bnb-4bit` for training, merged into `unsloth/llama-3-8b` in float16) |
| Method | QLoRA with [Unsloth](https://github.com/unslothai/unsloth): rank 16, alpha 16, all attention and MLP projections |
| Data | CUAD: 13,705 examples from 388 contracts. 20 contracts held out for validation, 102 for testing. |
| Examples | 512-token contract pages (up to 1,024 for long clauses). The clause sits at a random position. 21.7% are "Not found." pages. |
| Loss | Only the answer is graded; the question and contract text are masked |
| Schedule | 1 epoch, 1,713 steps, effective batch 8, learning rate 2e-4 with linear decay |
| Hardware | 1 × NVIDIA T4 (Kaggle free tier), 7.7 hours |
| Validation loss | 0.723 → 0.076 |

Every step is reproducible from the notebooks in **[github.com/CSroseX/Legal-Llama](https://github.com/CSroseX/Legal-Llama/tree/main/notebooks)**: diagnosis, feasibility, training, evaluation and publishing.

---

## License and credits

- **Built with Meta Llama 3.** Licensed under the [Meta Llama 3 Community License](./LICENSE), and use must follow Meta's [Acceptable Use Policy](https://llama.meta.com/llama3/use-policy). See [NOTICE](./NOTICE).
- **Data:** [CUAD](https://www.atticusprojectai.org/cuad) by The Atticus Project, CC BY 4.0.
- **Training:** [Unsloth](https://github.com/unslothai/unsloth).

```bibtex
@article{hendrycks2021cuad,
  title   = {CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author  = {Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
  journal = {NeurIPS},
  year    = {2021}
}
```

---

## What changed from the previous version

The first version of this model (2024) rambled and invented answers. It has been rebuilt from scratch.

| | Before | Now |
|---|---|---|
| **Answers** | Reworded or invented. 87.7% weren't text from the contract. | Quoted word for word. 99.6% are real contract text. |
| **Absent clauses** | Never said "Not found." and always made something up | Says "Not found." correctly 95.8% of the time |
| **Accuracy** (word overlap) | 0.185 | **0.735** |
| **Prompt format** | Not saved, so apps sent the wrong format and the model rambled | Saved in the tokenizer and in both GGUF files |
| **Training data** | Whole contracts, mostly longer than the model could read, so answers were often cut off | One page per example, containing the answer or teaching "Not found." |
| **Name and license** | Called "Llama 3.1" (it's Llama 3) and labelled Apache-2.0 | Correct name and the Llama 3 Community License |
| **Files** | Old weights stored twice (`.bin` + `.safetensors`, 40.7 GB) | One set of weights plus Q8_0 and Q4_K_M GGUF |
| **Evidence** | None | Measured on unseen contracts, with reproducible notebooks |

> 🔁 **Downloaded before 20 November 2026?** That copy is the old version. Python code that loads this repo updates automatically. **GGUF files must be downloaded again.** The old version remains in this repo's [history](https://huggingface.co/CSroseX/Llama-3-Legal-8B/tree/1b6a47d9321c73464466131d1d7eb594c1e6c847).

---

## Limitations

- **Uneven across clause types.** It's strongest on dates, governing law, renewal and assignment (0.86–0.97). It's weak on **Parties** (0.25: it finds the main company names but misses the nicknames each contract uses) and on long, complex clauses such as **Change of Control** (0.28). Because it reads one page at a time, it can also return extra candidates on full contracts, such as a signature date alongside the agreement date.
- **A review aid, not legal advice.** It was trained only on CUAD's 510 English-language US commercial contracts. Always check its output against the contract.
