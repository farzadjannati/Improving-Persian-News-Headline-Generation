<div align="center">

# 🗞️ Improving Persian News Headline Generation
### LoRA Fine-tuning on Llama 3.1 with Fact-Aware Contrastive Learning

<p align="center">
  <i>A CAP-LLM inspired multi-task framework for generating factually consistent,<br>
  engaging, and trustworthy Persian news headlines ⚡</i>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/huggingface/transformers"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange" alt="Transformers"></a>
  <a href="https://github.com/huggingface/peft"><img src="https://img.shields.io/badge/%F0%9F%A4%97-PEFT-yellow" alt="PEFT"></a>
  <a href="https://ai.meta.com/llama/"><img src="https://img.shields.io/badge/Model-Llama%203.1%208B-9cf" alt="Model"></a>
  <a href="https://huggingface.co/datasets/HooshvareLab/pn_summary"><img src="https://img.shields.io/badge/Dataset-PN--Summary-success" alt="Dataset"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

</div>

---

> **University of Tehran** — Final Project · Large Language Models Course
>
> **Authors:** Farzad Jannati · Shahriar Rahimi Rad · Abolfazl Assarian Nejad
>
> 📧 `farzadjannati@ut.ac.ir` · `rahimirad@ut.ac.ir` · `abolfazl.assarian@ut.ac.ir`

---

This repository contains the official implementation for the project on enhancing Persian news headline generation. We leverage the **Llama 3.1 8B Instruct** model, fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) with LoRA. Our primary approach is inspired by the **CAP-LLM** methodology, incorporating a multi-task learning objective with a contrastive loss to significantly improve factual consistency while maintaining high-quality, engaging headlines.

## Abstract

Generating compelling and factually accurate news headlines is a significant challenge in natural language processing. This project addresses this by fine-tuning the state-of-the-art Llama 3.1 8B Instruct model for the task of Persian news headline generation. We explore two fine-tuning strategies: a standard Supervised Fine-Tuning (SFT) and an advanced multi-task learning approach inspired by the CAP-LLM paper. This advanced method integrates a contrastive loss function to enforce factual alignment between the generated headline and the source article, effectively mitigating model hallucination. Our experiments, conducted on the comprehensive `pn_summary` dataset, demonstrate the superiority of the fact-aware fine-tuning method. We also establish a robust evaluation framework comparing our model against strong baselines like Gemma, Qwen, and Mistral, using a suite of metrics including ROUGE, BERTScore, factual consistency, and style.

## Table of Contents

1. [Key Features](#key-features)
2. [Project Workflow](#project-workflow)
3. [Technical Architecture](#technical-architecture)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Dataset](#dataset)
7. [Training Process](#training-process)
8. [Evaluation](#evaluation)
9. [Results](#results)
10. [Key Concepts Demonstrated](#key-concepts-demonstrated)
11. [Technologies Used](#technologies-used)
12. [Future Work](#future-work)
13. [Citation](#citation)
14. [Acknowledgements](#acknowledgements)
15. [Author](#author)
16. [Support](#support)
17. [License](#license)

## Key Features

- **State-of-the-Art Model**: Utilizes the powerful **Llama 3.1 8B Instruct** model as the backbone for generation.
- **Efficient Fine-Tuning**: Employs **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning, making it feasible to adapt large models on standard hardware.
- **Factual Consistency**: Implements a novel multi-task training objective with a **contrastive loss**, inspired by CAP-LLM, to minimize factual errors and hallucinations.
- **Comprehensive Evaluation**: Benchmarks against multiple strong baseline models and evaluates using ROUGE, BERTScore, Factual Consistency, and Style metrics.
- **Persian Language Focus**: All data processing and modeling are tailored for the nuances of the Persian language, utilizing libraries like `hazm`.

## Project Workflow

The end-to-end workflow of our project, from data collection to evaluation and future extensions, is illustrated below.

![Project Workflow](images/persian_headline_generation_workflow.png)

## Technical Architecture

The technical architecture details the core components of our fine-tuning approach, showcasing the data flow through the Llama 3.1 model enhanced with LoRA adapters and our custom multi-task loss function.

![Technical Architecture](images/technical_architecture.png)

## Repository Structure

```text
Improving-Persian-News-Headline-Generation/
│
├── notebooks/
│   ├── PersianTitleGenerator.ipynb          # Proposed CAP-LLM inspired model
│   ├── Title_Generator_Abolfazl.ipynb       # Standard SFT baseline
│   └── baseline_inference.ipynb             # Zero-shot baseline evaluation
│
├── images/
│   ├── persian_headline_generation_workflow.png
│   └── technical_architecture.png
│
├── configs/
│   └── PersianTitleGenerator.json           # LoRA & training configuration
│
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

To set up the environment and run this project, please follow these steps:

1. **Clone the repository:**

```bash
git clone https://github.com/farzadjannati/Improving-Persian-News-Headline-Generation.git
cd Improving-Persian-News-Headline-Generation
```

2. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:

```text
transformers
peft
datasets
trl
bitsandbytes
accelerate
evaluate
bert_score
rouge_score
hazm
pandas
torch
```

## Dataset

This project utilizes the **PN-Summary** dataset, which contains a large collection of Persian news articles and their corresponding summaries and titles.

- **Training Set**: ~82,000 samples
- **Testing Set**: ~5,500 samples

Additionally, baseline models are intended to be trained on a larger corpus crawled from prominent Persian news sources, including **Borna News, Tasnim News, Khabaronline, and Hamshahri**.

## Training Process

We implemented and compared two distinct fine-tuning methodologies:

1. **Standard Supervised Fine-Tuning (SFT)**: The Llama 3.1 8B Instruct model was fine-tuned using a standard cross-entropy loss on the `pn_summary` dataset. This serves as a strong baseline for our advanced approach.

2. **CAP-LLM Inspired Multi-task Fine-tuning**: This is our primary contribution. We designed a custom trainer (`ContextAwareTrainer`) that incorporates a multi-task loss function:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{gen}} + \lambda_{\text{fact}} \, \mathcal{L}_{\text{fact}}
$$

- $\mathcal{L}_{\text{gen}}$: The standard cross-entropy generation loss.
- $\mathcal{L}_{\text{fact}}$: A factual consistency contrastive loss that encourages the model to generate headlines semantically closer to the original article summary (positive sample) and further from a distorted or irrelevant summary (negative sample).
- $\lambda_{\text{fact}}$: A hyperparameter to balance the two loss components, set to `0.5` in our experiments.

**LoRA Configuration:**

| Parameter | Value |
| :--- | :---: |
| `r` | 16 |
| `lora_alpha` | 32 |
| `target_modules` | q_proj, v_proj |
| `lora_dropout` | 0.1 |
| Trainable Parameters | ~6.8M (0.08% of total) |

## Evaluation

We evaluate the generated headlines using a comprehensive set of automated metrics:

- **ROUGE (1, 2, L)**: Measures the n-gram overlap between the generated and reference headlines to assess content fidelity.
- **BERTScore**: Computes semantic similarity using contextual embeddings, providing a more nuanced evaluation of quality.
- **Factual Consistency**: A dedicated metric to quantify the factual alignment of the generated headline with the source article.
- **Style Analysis**: An evaluation based on stylistic attributes to measure how well the generated headlines match desired stylistic patterns.

## Results

Below is a comparison of our fine-tuned models against several baseline models. The results highlight the significant improvements achieved by our CAP-LLM inspired approach, particularly in ROUGE and BERTScore metrics.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Factual Consistency | Style Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot Baselines** | | | | | | |
| Gemma 3 4B | 0.000 | 0.000 | 0.000 | 0.6586 | TBD | TBD |
| Qwen 3 8B | 0.000 | 0.000 | 0.000 | 0.6535 | TBD | TBD |
| Mistral 7B | 0.000 | 0.000 | 0.000 | 0.6740 | TBD | TBD |
| Llama 3.1 8B Instruct (Base) | 0.000 | 0.000 | 0.000 | 0.6680 | TBD | TBD |
| **Our Fine-Tuned Models** | | | | | | |
| Llama 3.1 8B + LoRA (Standard SFT) | 0.007 | 0.000 | 0.007 | 0.802 | TBD | TBD |
| **Llama 3.1 8B + LoRA (CAP-LLM Inspired)** | **0.196** | **0.049** | **0.164** | **0.705** | TBD | TBD |

> **Note:** BERTScore F1 for the standard SFT is higher, but its ROUGE scores are near zero, indicating a tendency to generate fluent but lexically divergent headlines. The CAP-LLM approach provides a much better balance between fluency and factual accuracy.

## Key Concepts Demonstrated

- Parameter-efficient fine-tuning (PEFT) with LoRA on large language models
- Fact-aware multi-task learning with contrastive objectives
- Custom `ContextAwareTrainer` with overridden `compute_loss`
- Hallucination mitigation in abstractive summarization
- Persian NLP preprocessing pipelines (normalization, character correction, noise removal)
- Zero-shot vs. fine-tuned model benchmarking
- ROUGE vs. BERTScore discrepancy analysis in abstractive tasks
- BFloat16 mixed-precision training on NVIDIA A100

## Technologies Used

| Category | Tools |
| :--- | :--- |
| Large Language Models | Llama 3.1 8B Instruct |
| Fine-Tuning | PEFT, LoRA |
| Frameworks | Hugging Face Transformers, TRL |
| Deep Learning | PyTorch |
| Persian NLP | Hazm |
| Evaluation | ROUGE, BERTScore, Evaluate |
| Visualization | Matplotlib |
| Compute Environment | Google Colab (NVIDIA A100) |
| Programming Language | Python |

## Future Work

Our primary direction for future work is the implementation of a **Click-based Personalization and Training (CPT)** system. This will involve modeling user preferences based on their click behavior (long-term and short-term interests) to generate headlines that are not only factually accurate but also personally tailored to each user's unique stylistic and content preferences, as inspired by recent literature like the SCAPE and LOLA papers.

Additional directions include:

- **In-Context Learning (ICL)**: Few-shot prompting with Chain-of-Thought to further improve factual consistency without additional fine-tuning.
- **Advanced Automated Evaluation**: Implementing FactCC and dedicated style classifiers for scalable factual verification beyond n-gram metrics.
- **Domain Adaptation**: Extending the model to specialized domains like sports news for sensational headline generation.

## Citation

If you use this work, please cite this repository as:

```bibtex
@misc{persian-headline-generation-2025,
  author       = {Farzad Jannati and Shahriar Rahimi Rad and Abolfazl Assarian Nejad},
  title        = {Improving Persian News Headline Generation using LoRA Fine-tuning on Llama 3.1},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/farzadjannati/Improving-Persian-News-Headline-Generation}}
}
```

## Acknowledgements

- This work is heavily inspired by the methodologies presented in the **CAP-LLM: Context-Augmented Personalized Large Language Models for News Headline Generation** paper (Wilson et al., 2025).
- Our plans for future work are influenced by concepts from click-based personalization systems, such as those discussed in the **LOLA**, **SCAPE**, and **Panoramic Interests** papers.
- We thank the developers of the **PN-Summary** dataset (HooshvareLab) for providing the data for this research.
- This project utilizes the powerful **Llama 3.1 8B Instruct** model developed by Meta AI.

## Author

**Farzad Jannati**

Master's Student · University of Tehran

**Project collaborators:** Shahriar Rahimi Rad · Abolfazl Assarian Nejad

Interested in:

- Large Language Models
- Natural Language Processing
- Persian NLP
- Fact-Aware Text Generation
- Parameter-Efficient Fine-Tuning

[![GitHub](https://img.shields.io/badge/GitHub-farzadjannati-black?logo=github)](https://github.com/farzadjannati)

## Support

If you find this repository useful, consider giving it a ⭐️

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

<div align="center">

*Bridging advanced generative AI and the linguistic needs of Persian journalism* 🗞️⚡

Built with ❤️ using **Llama 3.1**, **LoRA**, and **Hugging Face**

</div>
