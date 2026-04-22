# SciGram: a large-scale dataset for scientific diagram understanding

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Multimodal large language models (MLLMs) have achieved strong results in visual question answering with natural images, yet their performance on diagram-based reasoning remains limited, largely due to the scarcity of high-quality training data. We present SciGram, a dataset of 200,000 scientific diagrams paired with synthetic instruction-following data grounded in middle-school science terminology. SciGram is built through a cost-efficient pipeline for generating multimodal alignment and instruction data tailored to vision-language understanding with scientific diagrams. Fine-tuning LLaVA on SciGram (LLaVA-SciGram) consistently improves diagram-based question answering on TQA, ScienceQA, and AI2D, surpassing LLaVA-OneVision with substantially fewer instructions. Furthermore, incorporating SciGram as an additional instruction-tuning stage for LLaVA-OneVision establishes new state-of-the-art results across all three benchmarks, underscoring the robustness and effectiveness of our dataset. To foster progress in diagram understanding, we release both the SciGram dataset and the LLaVA-SciGram model.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SciGram, a large-scale dataset (200k+ scientific diagrams; ~1.4M multimodal instructions) aimed at scientific diagram understanding. It is built via a six-stage pipeline: terminology extraction from TQA textbooks, LLM‑generated scientific claims, web diagram retrieval, caption synthesis, MCQ synthesis, and a curated mixture with existing QA sets (TQA/AI2D/ScienceQA/ARC/OpenBookQA). Fine‑tuning LLaVA with SciGram yields strong gains and new SOTA on diagram‑centric benchmarks: TQA diagram MC, ScienceQA image‑supported subsets, and AI2D with opaque labels (Tables 3–5, pp. 8–9). Ablations show each subset contributes, with the best results from using SciGram to further tune LLaVA‑OneVision (Table 6, p. 9). Figures 5–6 (p. 23) provide concrete examples; Figure 4 (p. 7) contrasts data scale vs. OneVision.

### Strengths
- Targeted dataset for an underserved modality (scientific diagrams), assembled with a transparent six‑step pipeline
- Strong empirical results: SOTA or on‑par on TQA (diagram MC), ScienceQA (image‑supported), and AI2D (opaque labels) with compact domain‑focused data (Tables 3–5).
- Design choices like balanced answer letters and shuffling in curated sets to reduce superficial cues.
- Ablations and training details clarify how each subset (Align, VIT, M³) contributes (Table 6, p. 9; Table 7, p. 22).

### Weaknesses
- Label quality not audited: no human evaluation of caption/MCQ correctness or of whether MCQs are answerable purely from the image (Sec. 3.4–3.5).
- Potential data leakage/overlap: no deduplication analysis between web‑retrieved diagrams and evaluation sets (AI2D/ScienceQA/TQA). Shuffling answer options does not address visual content overlap.
- Licensing & reproducibility risks: images are referenced via URLs only (Appendix A.2), which can suffer link rot and variable availability; unclear long‑term reproducibility.
- Methodological gaps: heuristic thresholds (e.g., retaining images linked to ≥5 claims) lack sensitivity analysis; no error analysis by diagram type (process, topology, etc.).
- The evaluated models may need to be updated. For example, gemini 2.0 or 2.5 is necessary to be used for evaluation.

### Questions
- Can you provide a human audit (e.g., ~500 samples) measuring: (a) caption factuality, (b) MCQ correctness, and (c) proportion of MCQs truly answerable from the image alone?
- For baselines run with custom prompts, please share the exact prompts and any temperature/seeds; report variance across runs.
- Could you add an error analysis by diagram category (process flows, part–whole, graphs) to identify where SciGram helps most/least?

### Soundness
3

### Presentation
3

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
The paper presents SciGram, a 200k‑diagram, ~1.37M‑instruction dataset and a six‑stage pipeline (terminology → claim generation → web retrieval → captioning → MCQ synthesis → curated mixture) targeted at scientific diagram understanding. Fine‑tuning LLaVA with SciGram yields strong gains and new SOTA on diagram‑centric portions of TQA, ScienceQA, and AI2D; ablations show each subset (Align, VIT, M³) contributes, and tuning atop OneVision performs best (Tables 3–6; Fig. 1). Training details and hyperparameters are provided; examples appear in Figs. 5–6.

### Strengths
- Strong empirical gains and new SOTA on diagram‑centric benchmarks (TQA diagram MC, ScienceQA with IMG support, AI2D opaque), with ablations that clarify contributions.
- Practical dataset design details (balanced answer letters; instruction formatting) that reduce superficial biases.
- Targets an underserved modality (scientific diagrams) with a transparent, reproducible six‑step pipeline; clear subset roles (Align, VIT, M³).

### Weaknesses
- There is a lack of human validation study of caption factuality, MCQ correctness, or image‑only answerability.
- Heuristic filters lack sensitivity analysis. Retaining only images linked to ≥5 claims may bias the dataset toward very common/templated diagrams (and possible benchmark overlap) and discard rare but educationally valuable diagrams. Without sensitivity studies, it’s unclear if results hinge on this threshold.
- Many baseline results are re‑produced with “custom prompts,” yet exact prompts, temperatures, and decoding settings are not provided, making comparisons difficult to reproduce or audit.
- The paper asserts a cost‑efficient pipeline, but does not report dollar‑costs, wall‑clock time, or GPU‑hours for data generation
- Understanding systematic errors (e.g., process diagrams vs. part‑whole, graphs with axes, occluded labels) would reveal where SciGram helps least and guide future data generation; none is provided.
- Table 6 studies subsets across stages but does not isolate: (i) the ≥5‑claims filter, (ii) number of captions/MCQs per diagram, (iii) balanced answer‑letter shuffling, or (iv) contribution of text‑only sets (ARC/OpenBookQA) within M³ to image tasks. Hence, causality of gains remains partly opaque.

### Questions
Please see the weaknesses above.

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
3

### Summary
This paper introduces **SciGram**, a new large-scale dataset specifically designed to improve **multimodal reasoning over scientific diagrams**, a long-standing challenge in vision-language research.

The authors develop a **six-stage pipeline** to construct SciGram:

1. **Terminology Extraction** – Extracts 4,820 key domain terms from middle-school science textbooks (based on the TQA dataset).  
2. **Scientific Claim Generation** – Uses *LLaMA-3-8B* to synthesize 5.5M concise factual statements grounded in these domain terms.  
3. **Diagram Retrieval** – Collects approximately 255K unique scientific diagrams from DuckDuckGo that align with the generated claims.  
4. **Caption Synthesis** – Employs *Qwen2-VL-7B* to produce detailed textual descriptions, supporting image–text alignment.  
5. **Multiple-Choice Question Generation** – Creates five reasoning-focused MCQs per diagram to assess visual comprehension.  
6. **Curated Integration** – Incorporates expert datasets (TQA, ScienceQA, AI2D, ARC, OpenBookQA) to form the **M3 subset**, balancing diagram- and text-based reasoning tasks.

The final **SciGram dataset** contains approximately **1.4 million multimodal instruction samples** and over **200,000 diagrams**.  

Fine-tuning **LLaVA-based architectures** on SciGram yields **state-of-the-art results** across multiple diagram reasoning benchmarks, including **TQA**, **ScienceQA**, and the **“opaque-label” split of AI2D**, demonstrating the dataset’s effectiveness in enhancing multimodal scientific understanding.

### Strengths
1. **Large-Scale, Domain-Specific Dataset for Scientific Diagrams**  
   The paper introduces **SciGram**, the largest open-source dataset dedicated to **scientific diagram understanding**, containing over **1.4 million multimodal samples** and approximately **200,000 diagrams**.  
   Unlike prior datasets (e.g., *ScienceQA*, *AI2D*, *TQA*) that combine text and natural images, SciGram focuses exclusively on **diagrammatic visual reasoning**, addressing a crucial gap in multimodal research.

2. **Transparent Data Source and Clear Licensing Claims**  
   The authors provide a **well-documented public repository**, along with **explicit statements about image sources, usage rights, and permissions**.  
   Such transparency is particularly important for a dataset paper, ensuring **ethical integrity**, **reproducibility**, and **legal clarity** for future research built on SciGram.

### Weaknesses
1. **Writing and Presentation Issues**  
   -  The *Introduction* section is too brief and lacks sufficient context or motivation. Moreover, the final paragraph is redundant, as each subsequent section title already clearly indicates its content.  
   -  *Figure 4* conveys very limited information but occupies a disproportionately large portion of the paper. Its inclusion could be condensed or supplemented with more meaningful analysis.  
   - For a paper targeting **ICLR 2026** and discussing a relatively common task, the related work section cites only **two papers from 2025**, which is not sufficient. The survey of recent advances in multimodal and diagram reasoning is noticeably incomplete.

2. **Lack of Error and Bias Analysis**  
   The paper reports overall accuracy improvements but provides little to no discussion of **failure cases**, **bias sources**, or **error typologies** (e.g., how models misinterpret symbolic diagrams or legends).  
   For a dataset or benchmark paper, this type of analysis is **fundamental and expected**, as it helps the community understand where current models fail and how future work might address such weaknesses.

### Questions
Some general-purpose VLMs, such as QwenVL 2.5, have already achieved strong performance on ScienceQA.
Why didn’t the authors include such models for training or fine-tuning on SciGram?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The research presents a scalable approach for gathering multimodal data by utilizing domain-specific scientific terminology to extract diagrams from online sources. The authors introduce SciGram, a dataset that combines visual and linguistic elements, featuring diagrams alongside synthetically created instructions within the natural sciences field. Building upon the LLaVA architecture, they develop the LLaVA-SciGram models, which represent a series of vision-language models specifically trained on the SciGram dataset.

### Strengths
1. The method of extracting terminology from the documents and reorganizing them into a sentence to retrieve scientific diagrams is interesting.

2. The paper is well-written and easy to follow.

### Weaknesses
1. Lack of scientific benchmark results. The authors should test on more recent scientific benchmarks like Mmmu[1] and Sciverse [2].

2. The paper uses Qwen2-VL-7B to generate the captions for the collected images. Therefore, it appears that the knowledge in the curated dataset originates from the scientific knowledge of Qwen2-VL-7B. So the improvement of the finetuned LLaVA may stem from the distillation of the Qwen model. Besides, this also limits the usability of the dataset. If this curated dataset is used to finetune a model more powerful than Qwen2-VL-7B, e.g., Qwen2-VL-72B or Qwen2.5-VL-7B, could it still bring improvement?

3. The paper mainly involves a dataset curation phase and an SFT phase. This paradigm has been widely explored in recent works. The novelty of this paper is limited, both in terms of data synthesis and training method.

[1] Yue, Xiang, et al. "Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[2] Guo, Ziyu, et al. "Sciverse: Unveiling the knowledge comprehension and visual reasoning of lmms on multi-modal scientific problems." arXiv preprint arXiv:2503.10627 (2025).

### Questions
1. Can this dataset be used to improve models more powerful than Qwen2-VL-7B?

Please also refer to the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2
