# NeoBabel: An Inclusive Multilingual Open Tower for Visual Generation

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Text-to-image generation advancements have been predominantly English-centric, creating barriers for non-English speakers and perpetuating digital inequities. While existing systems rely on translation pipelines, these introduce semantic drift, computational overhead, and cultural misalignment. We introduce NoeBabel, a novel multilingual image generation framework that sets a new Pareto frontier in performance, efficiency and inclusivity, supporting six languages: English, Chinese, Dutch, French, Hindi, and Persian. The model is trained using a combination of large-scale multilingual pretraining and high-resolution instruction tuning. To evaluate its capabilities, we expand two English-only benchmarks to multilingual equivalents: m-GenEval and m-DPG. NoeBabel achieves state-of-the-art multilingual performance while retaining strong English capability. Notably, NoeBabel matches or exceeds English-only models while being 2–4× smaller. We release an open toolkit, including all code, model checkpoints, a curated dataset of 124M multilingual text-image pairs, and standardized multilingual evaluation protocols, to advance inclusive AI research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This manuscript presents NEOBABEL, a multilingual text‑to‑image generator that natively supports six languages (English, Chinese, Dutch, French, Hindi, Persian) without relying on translate‑then‑generate pipelines. The core is a decoder‑only transformer built on a multilingual Gemma‑2 tokenizer, augmented with MAGVIT‑v2 visual codebook (8,192 tokens) and a modality‑aware attention scheme (causal for text; full attention for image tokens). Training proceeds via progressive pretraining (class‑conditional ImageNet → large web/synthetic corpora → refined multilingual data) and two instruction‑tuning stages at 512×512. The authors curate 124M multilingual image–text pairs by (i) English recaptioning and (ii) translation into the five non‑English targets, plus filtering for length, language, vision‑text match, and NSFW. For evaluation, they extend GenEval and DPG‑Bench to m‑GenEval/m‑DPG, reporting state‑of‑the‑art multilingual alignment—particularly in mid/low‑resource languages—at 2B parameters, and claim faster/leaner inference than translation pipelines.

### Strengths
* The paper articulates why translation‑first pipelines hurt inclusivity (semantic drift, latency, cultural misalignment), and builds a direct multilingual generator that avoids those issues
* A single decoder‑only stack with shared embeddings, image token full attention, and special prompt tokens simplifies training and avoids frozen components/adapters—well aligned with recent unified T2I trends.

### Weaknesses
* While image‑grounded translation reduces drift compared to direct prompt translation, linguistic/cultural authenticity (idioms, register, morphology) still risks being Anglocentric because captions originate in English. This may bias both training and evaluation, including m‑GenEval/m‑DPG if translated from English seeds.
* The abstract claims 2.8× faster and 59% less memory than translation pipelines, but I did not find a dedicated latency/memory section with wall‑clock numbers, batch sizes, context lengths, or hardware parity; only high‑level statements are present.

### Questions
* Evaluate latency/quality vs. multilingual prompt length (short/medium/long) to show robustness to languages with longer tokenizations (e.g., Hindi/Persian)
* Why MAGVIT‑v2 over alternative tokenizers (e.g., hierarchical/token‑semantics‑aware)? Any ablations on codebook size or retraining on multilingual‑centric images?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces NEOBABEL, a 2B parameter text-to-image generation model designed to address the English-centric bias of existing systems. The model directly supports six languages (English, Chinese, Dutch, French, Hindi, and Persian) without relying on an input translation pipeline. The authors propose a unified architecture based on the Gemma-2 LLM, trained with a progressive, five-stage strategy on a new, curated dataset of 124M multilingual image-text pairs. Key contributions include the model itself, two new multilingual benchmarks (m-GenEval and m-DPG) created by extending existing English benchmarks, and the public release of the model, data, and evaluation tools to foster more inclusive AI research.

### Strengths
The paper addresses the important and challenging problem of non-English text-to-image generation. 

Curated dataset & multilingual Evaluation benchmarks, could help community.

The proposed model, NEOBABEL, shows impressive results, achieving strong performance in six different languages while being significantly smaller than competitors like BLIP3-0.

### Weaknesses
The authors rightly criticize translation-first approaches for causing "semantic drift" and losing cultural nuance. However, their own data curation process (Section 3) relies on this exact method: generating detailed captions in English and then machine-translating them into other languages. As well as the evaluation benchmarks.

The architecture is not particularly novel. The model is a well-engineered combination of existing components (Gemma-2 backbone, MAGVIT-v2 tokenizer).

### Questions
Have you tried different translation models of varying quality to see how they affect the results?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
NEOBABEL addresses digital inequality and semantic drift from over-reliance on English in text-to-image generation by proposing a unified multilingual framework directly supporting six languages (English, Chinese, Dutch, French, Hindi, Persian).
The model employs a unified multimodal architecture trained on 124 million multilingual image-text pairs with large-scale multilingual pretraining and high-resolution instruction tuning, achieving higher efficiency and 2-4× smaller model size compared to translate-then-generate pipelines. NEOBABEL achieves SOTA performance on multilingual benchmarks (m-GenEval, m-DPG) while releasing code, model checkpoints, datasets, and evaluation protocols to advance inclusive AI research.

### Strengths
1. This work is well-presented and well-written. 
2. NEOBABEL employs a unified Transformer backbone with two critical innovations: (1) Extended embedding space incorporating 8,192 learnable discrete image token embeddings alongside text tokens, enabling native joint processing in a shared space; (2) Modality-aware attention mechanism with causal attention for text and full bidirectional attention for images, dynamically configured when both modalities coexist. This principled design elegantly unifies text-to-image generation as autoregressive sequence prediction without modality-specific branching.
3. The paper addresses multilingual data scarcity by scaling 39M English pairs to 124M multilingual pairs through quality-filtered translation using NLLB and Gemini models. Progressive training across three pretraining stages (pixel dependencies → large-scale alignment → refined synthesis) and two instruction-tuning stages, combined with model merging via Simple Moving Average, systematically improves generalization and multilingual instruction-following capabilities.
4. The work extends GenEval and DPG-Bench to multilingual versions (m-GenEval, m-DPG) across six languages with human verification, introduces Code-Switching Similarity (CSS) metrics to measure cross-lingual consistency, and achieves SOTA performance while maintaining 2-4× smaller model size than monolingual baselines. Full release of code, checkpoints, datasets, and evaluation protocols significantly advances inclusive AI research reproducibility.

### Weaknesses
1. The paper lacks rigorous analysis of whether multilingual training provides mutual benefits or if language-specific models would suffice. How do text embeddings from different languages interact in the unified space? Does representation confusion occur across languages? No evidence demonstrates that joint training outperforms separate language models.

2. The paper evaluates semantic-level consistency but ignores critical fine-grained capabilities: Can NEOBABEL generate readable text in different scripts (Chinese, Devanagari, Persian)? Fine-grained details (typography, numerics, spatial text) cannot be solved by LLM prompt expansion and are completely absent from evaluation.

3. No evaluation of realistic multilingual scenarios mixing multiple languages in single images (e.g., "Chinese title + English subtitle"). Without demonstrating generation quality when multiple languages coexist, multilingual support claims remain partially unvalidated.

### Questions
1. Without presentation, how do the multiple languages ​​facilitate each other? Are there similarities between text embeddings? Won't the representations of different languages ​​cause confusion?

2. Can this model generate text?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
NEOBABEL is a novel multilingual text-to-image generation framework that supports six languages without relying on intermediate translation. Through large-scale multilingual pretraining and high-resolution instruction tuning, it achieves state-of-the-art generation performance across all supported languages, matching or exceeding the capability of larger English-only models. The model is also highly efficient, running 2.8× faster than translation-based pipelines while using ~59% less memory. The paper introduces new multilingual evaluation benchmarks and shows NEOBABEL outperforms other approaches on these, and it releases an open-source toolkit with code, model checkpoints, a curated dataset, and evaluation protocols to encourage inclusive AI research.

### Strengths
1. The model demonstrates strong empirical performance, achieving the highest overall score on benchmarks while using only 2B parameters. NEOBABEL matches or surpasses larger state-of-the-art models in English generation and delivers consistently high results in other languages, with especially large gains in medium- and low-resource languages (e.g. significantly outperforming baselines in Hindi and Persian).
2. The authors bolster reproducibility by open-sourcing the entire project, providing model checkpoints, a 124M multilingual image-text dataset, all code, and standardized evaluation scripts.

### Weaknesses
1. NEOBABEL currently supports only six languages, which, while diverse, leaves out many globally important languages. Scaling to additional languages would require non-trivial effort (e.g. adapting the tokenizer and retraining with new data).
2. The training data largely comes from translated or AI-recaptioned English captions, which might introduce biases or unnatural phrasing. Despite quality controls, relying on machine translations can risk subtle semantic shifts, and any biases in the curated dataset (or the translation models used) may be reflected in NEOBABEL’s outputs, potentially affecting fairness or cultural authenticity.
3. The evaluation focuses on the new automated benchmarks (m-GenEval and m-DPG) and examples, without reported human studies of image quality or fidelity. Important aspects like subjective image quality, user preference across languages, or the preservation of nuanced cultural concepts are not deeply evaluated, so it remains unclear how the model’s outputs are perceived by native speakers or in real-world creative applications.

### Questions
1. In the progressive training pipeline, Stage 2 involves fine-tuning on a large English-only subset of the data. Why did the authors choose to include an English-centric stage during pretraining, and how do they ensure this does not bias the model towards English at the expense of other languages?
2. NEOBABEL currently supports six languages with a fixed multilingual tokenizer. How easily can the approach be scaled to additional languages or scripts (e.g. adding Spanish or Arabic) – would this require retraining a new model or tokenizer, and have the authors considered the potential impact on performance when extending to a broader set of languages?

### Soundness
3

### Presentation
3

### Contribution
3
