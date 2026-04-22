# CP-Agent: Context‑Aware Multimodal Reasoning for Cellular Morphological Profiling under Chemical Perturbations

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Cell Painting combines multiplexed fluorescent staining, high‑content imaging, and quantitative analysis to generate high-dimensional phenotypic readouts to support diverse downstream tasks such as mechanism-of-action (MoA) inference, toxicity prediction, and construction of drug–disease atlases. However, existing workflows are slow, costly and difficult to interpret. Approaches for drug screening modeling predominantly focus on molecular representation learning, while neglecting actual experimental context (e.g., cell line, dosing schedule, etc.), limiting generalization and MoA resolution. We introduce CP-Agent, an agentic multimodal large language model (MLLM) capable of generating mechanism-relevant, human-interpretable rationales for cell morphological changes under drug perturbations. At its core, CP-Agent leverages a context-aware alignment module, CP-CLIP, that jointly embeds high-content images and experimental metadata to enable robust treatment and MoA discrimination (achieving a maximum F1-score of 0.896). By integrating CP-CLIP outputs with agentic tool usage and reasoning, CP‑Agent compiles rationales into a structured report to guide experimental design and hypothesis refinement. These capabilities highlight CP-Agent’s potential to accelerate drug discovery by enabling more interpretable, scalable, and context-aware phenotypic screening---streamlining iterative cycles of hypothesis generation in drug discovery.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CP-Agent, a multimodal system for analyzing Cell Painting drug perturbation experiments. The core technical idea is inserting learned embeddings for compound, dose, and time as placeholder tokens into the text encoder to align images with experimental context. The system achieves F1=0.896 on compound classification using 1.9M training pairs across three datasets which were normalized and fused in a scalable way. CP-Clip is turned into CP-Agent via subtool definitions, scaffolding and integration with GPT-5 to generate reports.

### Strengths
* Clear reframing that experimental context (compound, concentration, time) is signal and should be fused into the text branch via learned token projections.
* Principled multi-dataset curation with MoA harmonization. This is scalable and onboarding additional datasets should be cheap
* The dataset integration and cross-normalization seems sensible and principled and onboarding new Cell Painting datasets should be cheap.
* Clarity: I found the paper reasonably easy to read.

### Weaknesses
* “Agentic” framing: The system is a single-pass orchestrated pipeline with predefined tool calls, there is little evidence of learned action selection or closed-loop planning beyond routing, so the agentic claim feels a little overstated.
* I found the MLLM comparisons somewhat unfair and uninformative. CP-CLIP receives >1M domain-specific training pairs while GPT-5/Gemini/Claude receive (presumably) zero Cell Painting training and only minimal 2-stage prompting. MLLMs with extensive prompt engineering or few-shot learning would be a better comparison
* Weak expert evaluation. No inter-rater reliability metrics or comparison to baseline methods (human-written reports, template-based summaries) are provided. This makes the high scores uninterpretable without reference points.

### Questions
* For the expert evaluation, what is inter-rater reliability? Are differences between models statistically significant? How do the reports compare to human-written reports or simpler template-based baselines? Without these comparisons, the scores lack context.
* Can you clarify what makes this system "agentic" vs being a fixed pipeline?
* Can you provide image-free baselines using only molecular fingerprints and metadata (no microscopy images) for the classification tasks? This would quantify how much the images actually contribute to performance.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes an agent-based framework comprising two main components: CP-CLIP, a contrastive learning model for chemical perturbation–image alignment, and CP-Agent, a agentic system built atop CP-CLIP.

### Strengths
The main novelty of this work lies in the introduction of an agentic framework specifically designed for Cell Painting images. While the underlying components (e.g., CLIP-based contrastive learning and vision encoders) are not new, applying an agent-based paradigm to organize multimodal reasoning, analysis, and interpretation within this biological context is conceptually original. This represents an interesting step toward structured and interpretable automation in cellular image understanding and drug discovery workflows.

### Weaknesses
While incorporating agentic system for Cell Painting is interesting, there are several weakness of the proposed work,


**Insufficient survey of related work in cross-modal contrastive learning for Cell Painting**
* The related work section lacks sufficient discussion of recent multimodal and contrastive learning methods for Cell Painting. While the paper introduces CP-CLIP as a novel contribution, several recent works within this context are not adequately discussed [1, 2, 3]. 
* In addition, the manuscript lacks quantitative comparisons with prior frameworks. Although CP-CLIP is presented as a key innovation, the evaluation primarily contrasts against generic CLIP variants and omits stronger baselines from recent multimodal or contrastive approaches. This makes it difficult to disentangle whether the reported improvements stem from the proposed context-aware token injection mechanism or from other architectural or training differences.

**Method**
* While interpretability is highlighted as a major advantage of CP-Agent, the paper lacks quantitative evaluation of this aspect (e.g., faithfulness, consistency, or attribution accuracy).
* The authors pretrained CP-CLIP on roughly 500 distinct compounds, which is small relative to the nearly two million image–context pairs. Since standard CLIP loss assumes each image–text pair is unique, simply aligning the same perturbation would potentially lead to overfitting or biased representations. 

**Limited evaluation and non-standard evaluation metrics for cross-modal contrastive learning**
* The number of chemical perturbations explored in this work is relatively limited. Prior studies in cross-modal contrastive learning for Cell Painting [1,2,3] have leveraged datasets with over 10k compounds, whereas this study appears to utilize a smaller subset (approximately 500 compounds across 1.9M image–context pairs). This limitation constrains the generalizability of CP-CLIP and raises concerns regarding its robustness across broader chemical spaces.
* The evaluation of CP-CLIP relies primarily on cosine similarity scores--the optimization objective of CLIP loss itself--rather than standard retrieval metrics such as Recall@K. This metric choice makes the reported improvements less convincing, as higher cosine similarity naturally aligns with the training objective.
* The paper benchmarks CP-Agent primarily against general-purpose MLLMs (e.g., GPT-5, Gemini-2.5-Pro, Claude-4-Sonnet), which are not pretrained on Cell Painting data. A more appropriate baseline would involve fine-tuning a VLM on the same Cell Painting datasets for a fairer comparison.
* How classification is done for standard CLIP, e.g.,whether a separate classifier is trained or retrieval is done by ranking cosine similarity, is not clearly explained.

**Missing Ablation Studies and Component Analysis**
 * The agentic system (CP-Agent) includes several modules (e.g., CPContext, FeatRank, StatSynth, ReportGen), yet the manuscript provides no ablation to assess their individual contributions. Without such analysis, it remains unclear which components are necessary or most influential for final performance.

**Unclear dataset and evaluation setup**
* The description of training, validation, and test splits is ambiguous. It is unclear which datasets are used for each stage and how compounds are selected for evaluation. Given the paper’s focus on chemical perturbations, omitting comprehensive datasets such as [4] limits the scope. 

**Presentation and clarity issues**
* Several figures (e.g., Figures 1–4) lack sufficient captions and details. For example, the use of a GPT-2 tokenizer in Figure 1 is not clear, what constitutes “raw text” and how it is tokenized remain unclear. Similarly, later figures provide only high-level summaries without specifying details.

[1] CLOOME: contrastive learning unlocks bioimaging databases for queries with chemical structures Sanchez-Fernandez et al Nature com 2023

[2] How Molecules Impact Cells: Unlocking Contrastive PhenoMolecular Retrieval Fradkin et al, NeurIPS 2024

[3] CellCLIP – Learning Perturbation Effects in Cell Painting via Text-Guided Contrastive Learning Lu et al NeurIPS 2025

[4] Cell Painting, a high-content image-based assay for morphological profiling using multiplexed fluorescent dyes, Bray et al Nature protocal 2016

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces CP-Agent, an agentic multimodal large language model (MLLM) system designed to analyze Cell Painting assay data to study how chemical perturbations affect cell morphology. The core of CP-Agent is CP-CLIP, pretrained with cell paint images and experimental metadata pairs by contrastive learning. Then CP-Agent leverages the context inferred from CP-CLIP to help generate structured report including experimental design and hypothesis refinement.

### Strengths
1. The paper is original in explicitly treating experimental context as signal and injecting it into the text encoder for semantically image–text alignment with CLIP style training. The pretraining corpus including 1.9M image–context pairs, which is reasonable scale for Cell Painting experiments and well-suited to learning robust, context-aware representations.

2. The system is presented as a clean, step-wise workflow with structured JSON outputs and rich case studies that connect reasoning steps, making the reasoning traceable for practitioners.

3. CP-CLIP achieves robust MoA/treatment discrimination. The framework is portable to other imaging modalities and broader phenotypic screening use cases.

### Weaknesses
1. More ablation study is needed for the proposed method. The ablation study could include: 1) remove each context field (<CMPD>, <CONC>, <TIME>) separately, and then retrain the model, to show how the model performance change 2) remove control images from the image embedding, use only the perturbation tile, to verify the contribution of control embedding 

2. Table 2 results with high performance on compound recovery, may be results from the fact that the CP-CLIP is training on the meta data text, and the model can recover compound from the text context. Additional experiment to test performance change: image-only, text-only, and shuffled-context controls (time and dosage).

### Questions
1. Can you report  image-only, text-only baselines and counterfactual context (swap dose/time ) to test the performance change?

2. Are agent outputs fully deterministic or do responses vary with randomness? How reproducible are the reasoning outputs across runs?

### Soundness
3

### Presentation
3

### Contribution
4
