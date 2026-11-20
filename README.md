<div align="center">

<img src="https://huggingface.co/CSroseX/Llama-3-Legal-8B/resolve/main/banner.png" alt="Llama-3-Legal-8B" width="100%">

# Legal-Llama

**How a broken legal fine-tune was diagnosed, rebuilt and measured, on free GPUs.**

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Llama--3--Legal--8B-yellow)](https://huggingface.co/CSroseX/Llama-3-Legal-8B)
[![Built with Meta Llama 3](https://img.shields.io/badge/Built%20with-Meta%20Llama%203-0467DF)](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
[![Dataset: CUAD](https://img.shields.io/badge/Dataset-CUAD-orange)](https://huggingface.co/datasets/theatticusproject/cuad-qa)
[![Trained on: Kaggle T4](https://img.shields.io/badge/Trained%20on-Kaggle%20T4%20(free)-20BEFF)](https://www.kaggle.com)

</div>

---

## The model

[**Llama-3-Legal-8B**](https://huggingface.co/CSroseX/Llama-3-Legal-8B) finds and quotes clauses in commercial contracts, or says `Not found.` when the clause isn't there. Give it a clause type (for example "Governing Law") and a page of a contract. It returns the exact sentences from that page.

| | Previous version | **Now** |
|---|:---:|:---:|
| Word overlap with the correct clause | 0.185 | **0.735** |
| Answers that are real text from the contract | 12.3% | **99.6%** |
| Correct "Not found." when the clause is absent | 0% | **95.8%** |

<sub>Measured on 1,000 pages from 102 CUAD contracts that weren't used in training. Full details: <a href="results/evaluation.md">results/evaluation.md</a>.</sub>

Usage code, GGUF files and the prompt format are on the **[model card](https://huggingface.co/CSroseX/Llama-3-Legal-8B)**.

---

## From broken to working

The first version (2024) rambled and invented answers, and its training notebook had been lost. This repo rebuilds it step by step.

**1. Diagnose.** The old model was tested with three prompt formats, and the training data was measured:
- The prompt format had **never been saved** with the model, so apps sent the wrong format and the model fell back to rambling autocomplete.
- CUAD contracts are long: a median of 7,200 tokens. **89.8% of training examples were longer than 2,048 tokens**, and **60.5% of answers start after token 2,048**. If the original run used Unsloth's default 2,048-token limit (the notebook is lost, so this can't be confirmed), the model was mostly trained without seeing the answers. That would explain why it learned to guess.

**2. Check it's feasible.** Training the 8B model with QLoRA on a free T4 peaked at **7.4 GB of 15.6 GB** of GPU memory, at about 350 tokens/second.

**3. Rebuild the training data.**
- Each example is one **512-token page**, with the clause at a random position.
- 21.7% of the pages contain no answer and teach the model to say **`Not found.`**.
- Only the answer is graded; the prompt is masked.
- 20 contracts are held out for validation.

**4. Train safely.** 13,705 examples, 1 epoch, **7.7 hours** on Kaggle. Progress is uploaded to Hugging Face every 200 steps, and a restart resumes automatically. That safeguard was added after the first run's output was lost with a closed session.

**5. Measure honestly.** The same 1,000-page exam was given to the base model, the previous version and the new model, plus a whole-contract test.

**6. Publish safely.**
- The adapter is merged into the 16-bit model.
- The prompt format is saved, so apps apply it automatically.
- Q8_0 and Q4_K_M GGUF files are built.
- The merged model is re-tested **before** anything is uploaded.
- The public repo is replaced in **one** update.

---

## Notebooks

Each notebook runs on a free T4. Notebooks 03–05 need a Hugging Face write token stored as a Kaggle secret called `HF_TOKEN`.

| # | Notebook | What it does | Time |
|:---:|---|---|:---:|
| 01 | [Diagnose rambling](notebooks/01_diagnose_rambling.ipynb) | Tests the old model with three prompt formats, checks whether it learned to stop, measures contract lengths | ~20 min |
| 02 | [Training feasibility](notebooks/02_training_feasibility.ipynb) | Measures GPU memory and speed at 1,024 / 2,048 / 4,096 tokens and estimates training time | ~20 min |
| 03 | [Train](notebooks/03_train.ipynb) | Builds the page-level dataset and trains with QLoRA (Unsloth); auto-saves and auto-resumes through Hugging Face | ~8 h |
| 04 | [Evaluate](notebooks/04_evaluate.ipynb) | Runs the 1,000-page exam on three models plus the whole-contract test | ~2.5 h |
| 05 | [Publish](notebooks/05_publish.ipynb) | Merge → safety test → GGUF → single-commit upload | ~1–1.5 h |

**To run one on Kaggle:**
1. **File → Import Notebook → GitHub**, then paste the notebook's GitHub link.
2. **Settings → Accelerator → GPU T4 x2**, and **Internet → On**.
3. **Save Version → Save & Run All (Commit)**, so it keeps running in the background.

---

## Repository layout

```text
notebooks/      the five notebooks above, in order
results/        evaluation results behind the model card's numbers
model_card/     the published Hugging Face model card, LICENSE and NOTICE
```

---

## Limitations

- **Uneven across clause types.** It's strong on dates, governing law, renewal and assignment (0.86–0.97), and weak on **Parties** (0.25) and long clauses like **Change of Control** (0.28).
- **A review aid, not legal advice.** It was trained only on CUAD's 510 English-language US commercial contracts.

## Credits

Built with **Meta Llama 3**. The model is licensed under the [Meta Llama 3 Community License](model_card/LICENSE).
Data: [CUAD](https://www.atticusprojectai.org/cuad) by The Atticus Project (CC BY 4.0). Training: [Unsloth](https://github.com/unslothai/unsloth).
