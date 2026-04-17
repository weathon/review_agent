# ScaleCap: Scalable Image Captioning via Dual-Modality Debiasing

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
This paper presents ScaleCap, a scalable image captioning strategy that generates
comprehensive and detailed image captions. The key challenges of high-quality
image captioning lie in the inherent biases of LVLMs: multimodal bias resulting in
imbalanced descriptive granularity, offering detailed accounts of some elements
while merely skimming over others; linguistic bias leading to hallucinated de-
scriptions of non-existent objects. To address these issues, we propose a scalable
debiased captioning strategy, which continuously enriches and calibrates the caption
with increased inference budget. Specifically, we propose two novel components:
heuristic question answering and contrastive sentence rating. The former generates
content-specific questions based on the image and answers them to progressively
inject relevant information into the caption. The latter employs sentence-level
offline contrastive decoding to effectively identify and eliminate hallucinations
caused by linguistic biases. With increased inference cost, more heuristic questions
are raised by ScaleCap to progressively capture additional visual details, generating
captions that are more accurate, balanced, and informative. Extensive modality
alignment experiments demonstrate the effectiveness of ScaleCap. Annotating
450K images with ScaleCap and using them for LVLM pretraining leads to consis-
tent performance gains across 11 widely used benchmarks. Furthermore, ScaleCap
showcases superb richness and fidelity of generated captions with two additional
tasks: replacing images with captions in VQA task, and reconstructing images
from captions to assess semantic coverage.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes ScaleCap, a detailed image captioning pipeline that uses multiple models, including an LLM and an LVLM. It first generates an initial caption and identifies golden sentences, highly likely non-hallucinatory sentences within the generated caption, based on contrastive sentence ratings. Contrastive sentence rating is a strategy for evaluating the factual precision of a sentence by comparing output token probabilities between multimodal decoding and language-only decoding. Then, ScaleCap further obtains visual information by iteratively requesting detailed descriptions for each object mentioned in the golden sentences (referred to as heuristic question-answering module). By alternating between contrastive sentence rating and heuristic question answering, ScaleCap collects reliable sentences and finally summarizes them using an LLM. The authors demonstrate the effectiveness of ScaleCap through reconstruction and pretraining experiments.

### Strengths
1. This paper is clear and well-organized overall.
2. The problem addressed in the paper is interesting and timely.

### Weaknesses
1. The authors use image reconstruction and pretraining experiments to demonstrate the effectiveness of the proposed method. However, there are still many evaluation approaches remaining for assessing detailed image captions. For example, GPT-based methods [1,2] and QA-based methods [3] could be leveraged. I strongly encourage the authors to incorporate these additional evaluation metrics.
2. Several previous studies have proposed detailed image captioning systems that involve multiple deep learning models [3, 4]. Comparisons and discussions with these works are needed to better position the proposed method within the existing literature.

[1] Petryck et al., "ALOHa: A New Measure for Hallucination in Captioning Models"  
[2] Chan et al., "CLAIR: Evaluating Image Captions with Large Language Models"  
[3] Lee et al., "Toward Robust Hyper-Detailed Image Captioning: A Multiagent Approach and Dual Evaluation Metrics for Factuality and Coverage"  
[4] Get et al., "Visual Fact Checker: Enabling High-Fidelity Detailed Caption Generation"

### Questions
oes the proposed method outperform existing methods across multiple evaluation metrics?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ScaleCap, a pipeline designed to generate comprehensive and detailed image captions for pretraining VLM. The authors identify multimodal bias and linguistic bias as key challenges. ScaleCap addresses these through Heuristic Question Answering and Contrastive Sentence Rating. Using ScaleCap, the authors create the ScaleCap-450K dataset  and show that pretraining LVLMs on this dataset consistently improves performance compared to using datasets like ShareGPT4V and DenseFusion.

### Strengths
1.The paper is generally well-written and clearly structured, making the proposed pipeline and experiments easy to follow.

2.The work proposes a complete pipeline aimed at improving the quality, detail, and factuality of LVLM-generated captions.

3.Experiments effectively demonstrate that the ScaleCap-450K dataset, generated by the proposed pipeline, leads to superior LVLM pretraining outcomes compared to existing large-scale caption datasets like ShareGPT4V-450k and DenseFusion-450k.

### Weaknesses
1.The core components of ScaleCap, namely Heuristic Question Answering and Contrastive Sentence Rating, appear to have limited novelty, primarily combining or refining existing techniques rather than introducing fundamentally new concepts.

2.The proposed ScaleCap pipeline suggests a high computational cost for generating each caption, involving multiple model inference stages including initial captioning, filtering, question generation, iterative question answering, answer filtering, and final integration using a large LLM. This multi-step process appears resource-intensive, potentially limiting its practical scalability.

### Questions
1.Could the authors quantify the computational cost—such as average time per caption or total GPU hours required to generate a single caption using the full ScaleCap pipeline, and how does this compare to baseline caption generation costs?

2.While the paper compares pretraining benefits against other datasets, how does the quality of ScaleCap captions or the performance of models pretrained on them compare against models directly improved using training-based hallucination mitigation techniques like RLAIF-V or LLaVA-RLHF?

3.Considering the introduction's potentially restrictive view on tool-based captioning , could the ScaleCap pipeline itself benefit from integrating specific tools, perhaps for grounding question generation or verifying answers, rather than solely relying on the VLM and CSR

### Soundness
3

### Presentation
3

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
The paper introduces *ScaleCap, a scalable image-text paired data distillation pipeline designed to produce long, instance-complete, and visually grounded captions from LVLM without external detectors/tools. ScaleCap has two core components:
(1) Heuristic Question Answering (HQA): an LLM generates targeted, object/attribute/position questions from an initial caption; a LVLM answers them to surface missing details under a controllable budget.
(2) Contrastive Sentence Rating (CSR): an offline filter that compares token probabilities with vs. without the image and retains sentences whose “critical tokens” are better supported by the image than by language priors, aiming to suppress hallucinations.
Using this pipeline, the authors build ScaleCap-450K, a 450k-image caption dataset sourced mainly from LAION and ShareGPT4V images with resolution/complexity filtering.

### Strengths
1. The two identified blockers map neatly to HQA (add missing visual facts) and CSR (filter unsupported text). The mechanism is easy to reason about and implement with standard LVLM/LLM primitives.
2. The observation that 7B-class LVLMs are often sufficient for perception while larger LLMs help during long-context integration provides concrete guidance for cost-constrained systems.
3. Improvements appear across multiple LVLM backbones/settings, suggesting the dataset’s benefits aren’t model-specific.

### Weaknesses
1. The core claim that balanced, instance-complete detail drives the gains isn’t cleanly isolated from caption length or sheer verbosity. Controlled studies (equal length across methods; fixed token budgets redistributed among object/attribute/position details) are missing.
2. CSR accepts sentences based on a max-over-critical-tokens Δ probability threshold. This may be unstable (single-token spikes; POS-tagging noise). Robustness analyses (pooling variants, τ-sweeps, cross-dataset calibration) are not provided.

### Questions
1. For frontier models, does the way ScaleCap introduces more prior information help the LVLM itself in the pipeline, such as fine-tuning the LVLM with generated data?
2. If CSR is computed with a different LVLM than the one used to answer HQA (or than the pretraining backbone), do conclusions hold?
3. A clear pseudocode algorithm will be more helpful for reading and understanding.

### Soundness
2

### Presentation
3

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
The paper proposes ScaleCap, a captioning pipeline that iteratively enriches captions while reducing hallucinations by combining (i) heuristic question answering and (ii) contrastive sentence rating. A tunable scale budget controls how many questions are asked, trading cost for detail. Using ScaleCap, the authors build ScaleCap-450K and show consistent gains across 11 benchmarks, plus benefits in Prism-style perception tests and an image-reconstruction study. The author argues that the approach improves informativeness and alignment even with smaller LVLMs.

### Strengths
- Clear, modular method that targets two real failure modes. The offline contrastive, sentence-level rating is a neat way to down-weight language priors without destabilizing decoding.
- Strong empirical coverage and breadth. Pretraining with ScaleCap-450K beats ShareGPT4V-450K and DenseFusion-450K on most of the 11 benchmarks.
- Practical efficiency levers. The pipeline uses a small LVLM for perception (object/position Q&A) and a budget N to scale detail, giving users cost–quality control.

### Weaknesses
- Practical efficiency levers. The pipeline uses a small LVLM for perception (object/position Q&A) and a budget N to scale detail, giving users cost–quality control.
- Prompt dependence in question generation. The method hinges on a “powerful LLM” to craft good object/position prompts; robustness across domains or weaker LLMs isn’t deeply probed.

### Questions
Refer to the weaknesses,

### Soundness
3

### Presentation
3

### Contribution
2
