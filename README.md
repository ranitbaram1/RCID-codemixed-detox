# RCID: A Parallel Corpus and Evaluation Framework for Roman-Script Code-Mixed Indic Text Detoxification

*St. Thomas' College of Engineering and Technology, Kolkata, India*

---

## Overview

This repository contains the code and experiments for the RCID project — a parallel corpus, formal annotation protocol, domain-specific classifier, and systematic evaluation framework for detoxifying Roman-script code-mixed Indic social media text (Hinglish and Bengali-Roman).

**This is not a generative detoxification model.** The primary contributions are the RCID corpus, the 8-rule annotation protocol, and the XLM-RoBERTa-based toxicity classifier and evaluation pipeline. Generative approaches (mBART-50 + LoRA) are planned future work.

**Dataset:** [ranitbaram/RCID-codemixed-detox](https://huggingface.co/datasets/ranitbaram/RCID-codemixed-detox)

---

## The Problem

General-purpose toxicity tools such as Detoxify are blind to this domain: **89.7% of toxic inputs score LOW toxicity** (mean score 0.1266) because Roman-script Hinglish and Bengali-Roman slurs are entirely absent from their training data. RCID addresses this gap with a domain-specific parallel corpus and classifier.

---

## Key Results

| System | XLM-R Reduction | SBERT Similarity | Detoxify Reduction |
|---|---|---|---|
| Rule-Based Baseline | 50.54% | 0.9576 | −3.16% |
| LLM Zero-Shot (API-based) | 74.00% | 0.9197 | 26.35% |
| **XLM-R Fine-tuned (Ours)** | **94.86%** | **0.8592** | **18.07%** |

**Classifier:** 92.61% accuracy, Macro F1 0.9260 (xlm-roberta-base, held-out 20% independent evaluation)

**Human Evaluation** (50 pairs, 3 annotators, Fleiss κ):

| Criterion | Mean (1–5) | Fleiss κ |
|---|---|---|
| Toxicity Removal | 4.45 | 0.405 |
| Meaning Preservation | 4.41 | 0.309 |
| Fluency | 4.51 | 0.474 |
| Overall | 4.46 | 0.396 |

---

## Repository Structure

```
RCID/
├── Final_RCID_Main_Pipeline.ipynb          # Classifier training + full evaluation
├── Final_RCID_Heldout_Evaluation.ipynb     # Independent held-out evaluation (circularity fix)
├── Final_RCID_Seq2Seq_Experiments.ipynb    # Seq2seq failure analysis (5 architectures)
├── RCID_enrich_dataset.py                  # Adds language, toxicity_level, sbert_similarity to CSV
└── README.md
```

---

## Notebooks

### `Final_RCID_Main_Pipeline.ipynb`

End-to-end pipeline:

- Dataset load and statistics
- Detoxify external scoring (blindness finding — 89.7% LOW)
- XLM-R toxicity classifier training (10,392 sentences)
- Full-dataset evaluation (XLM-R reduction + SBERT similarity)
- Rule-based baseline (8-rule annotation protocol applied deterministically)
- Ablation: zero-shot XLM-R (random head baseline)
- Paper-ready results table + metrics saved to Drive

### `Final_RCID_Heldout_Evaluation.ipynb`

Retrains classifier on 80% of pairs (4,156 pairs) and evaluates on unseen 20% (1,040 pairs). Gap between full-dataset (94.67%) and held-out (94.86%) is 0.19pp — confirms results are not an overfitting artefact.

### `Final_RCID_Seq2Seq_Experiments.ipynb`

Documents why all five seq2seq architectures fail on Roman-script Hinglish under standard fine-tuning conditions. Each experiment is self-contained with the expected failure mode documented inline.

| Model | Observed Failure | Root Cause |
|---|---|---|
| T5-base | Copy bias; loss 2.02→1.47; no detoxification | Span denoising pretraining incompatible with register transformation |
| mT5-base | High initial loss (15.48, 7.6× T5); no convergence | SentencePiece fragments Hinglish tokens; near-zero domain vocabulary |
| FLAN-T5-large | OOV ratio 15–23%; incoherent generation | 32K vocabulary has no Hinglish entries; 6-word → 17 sub-word tokens |
| mBART-50 | Identity collapse; 4/5 outputs exact copies (100% overlap) | Translation pretraining prior collapses to copying |
| IndicBART | Devanagari tokens injected; repetitive loops ('in in in') | Pretrained on Devanagari-only languages; Roman-script fully OOV |

---

## Environment

All notebooks run on **Google Colab Pro, Tesla T4 GPU**.

**Locked versions — do not upgrade:**

```
transformers == 5.0.0
peft         == 0.18.1
torch        == 2.10.0
```

Additional installs handled inside notebooks:

```
detoxify
sentence-transformers
sentencepiece
```

---

## Dataset

[**ranitbaram/RCID-codemixed-detox**](https://huggingface.co/datasets/ranitbaram/RCID-codemixed-detox)

| Property | Value |
|---|---|
| Total pairs | 5,196 |
| Hinglish / Hindi-Roman | 4,943 (95.1%) |
| Bengali-Roman | 229 (4.4%) |
| Mean SBERT similarity | 0.8621 |

To enrich a raw `input/output` CSV with `language`, `toxicity_level`, and `sbert_similarity` columns, run `RCID_enrich_dataset.py` in Colab.

---

## Annotation Protocol (8 Rules)

| Rule | Description |
|---|---|
| R1 | Slur removal — remove slur, keep remainder |
| R2 | Pronoun normalisation — tu/tum → aap |
| R3 | Group accusation hedging — prefix with *mujhe lagta hai* |
| R4 | Noise removal — @mentions, #hashtags, URLs, emojis |
| R5 | Imperative softening — karo → karein |
| R6 | Semantic constraint — transformation beyond pronoun swap alone |
| R7 | Similarity filter — SBERT ≥ 0.88 excluded |
| R8 | Register filter — no *yaar* in outputs |

---

## Contributors

| Name | Role |
|---|---|
| Ranit Baram, Ritesh Prasad, Sayani Bose, Silpi Ghosh, Nilanjan Mandal | Corpus design, annotation protocol, classifier training, evaluation |
| Soham Ghosh | Contributed Hinglish parallel pairs (1,067 pairs, 20.5% of dataset) |

---

## Citation

**Dataset:**

```bibtex
@dataset{rcid2026,
  title  = {RCID: Roman-Script Code-Mixed Indic Detoxification Corpus},
  author = {Ranit Baram and Soham Ghosh},
  year   = {2026},
  url    = {https://huggingface.co/datasets/ranitbaram/RCID-codemixed-detox}
}
```

*Paper citation will be added upon publication.*

---

## License

Code: [MIT License](LICENSE)  
Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
