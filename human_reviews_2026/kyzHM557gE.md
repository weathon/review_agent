# Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Any entity in the visual world can be hierarchically grouped based on shared characteristics and mapped to fine-grained sub-categories. While Multi-modal Large Language Models (MLLMs) achieve strong performance on coarse-grained visual tasks, they often struggle with Fine-Grained Visual Recognition (FGVR). Adapting general-purpose MLLMs to FGVR typically requires large amounts of annotated data, which is costly to obtain, leaving a substantial performance gap compared to contrastive CLIP models dedicated for discriminative tasks. Moreover, MLLMs tend to overfit to seen sub-categories and generalize poorly to unseen ones. To address these challenges, we propose Fine-R1, an MLLM tailored for FGVR through an R1-style training framework: (1) Chain-of-Thought Supervised Fine-tuning, where we construct a high-quality FGVR CoT dataset with rationales of "visual analysis, candidate sub-categories, comparison, and  prediction”, transition the model into a strong open-world classifier; and (2) Triplet Augmented Policy Optimization, where Intra-class Augmentation mixes trajectories from anchor and positive images within the same category to improve robustness to intra-class variance, while Inter-class Augmentation  maximizes the response distinction conditioned on images across sub-categories to enhance discriminative ability. With only 4-shot training, Fine-R1 outperforms existing general MLLMs, reasoning MLLMs, and even contrastive CLIP models in identifying both seen and unseen sub-categories, showing promise in working in knowledge-intensive domains where gathering expert annotations for all sub-categories is arduous. Code is available at [https://github.com/PKU-ICST-MIPL/FineR1_ICLR2026](https://github.com/PKU-ICST-MIPL/FineR1\_ICLR2026).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces Fine-R1, a multi-modal large model (MLLM) designed to excel at fine-grained visual recognition (FGVR). The authors propose a two-stage framework, starting with Chain-of-Thought Supervised Fine-tuning (CoT SFT) to teach the model structured reasoning, followed by Triplet Augmented Policy Optimization (TAPO). TAPO is an RL algorithm based on DAPO that uses anchor, positive, and negative image triplets to handle the FGVR challenges of high intra-class and low inter-class variance.

### Strengths
The paper's primary strength lies in its impressive empirical results. Fine-R1 achieves state-of-the-art performance on six FGVR datasets, outperforming general MLLMs, reasoning-focused MLLMs, and even strong contrastive CLIP models. The model shows particularly strong generalization to unseen categories, which is a key challenge in FGVR. The analysis (hypotheses H1-H3) provides an insightful conclusion that the gains come from an improved ability to deploy existing knowledge, rather than learning new features or knowledge.

### Weaknesses
Weakness
* Limited Novelty of TAPO: The core algorithmic contribution, TAPO, does not appear to be a novel RL algorithm. It feels like a forced "splicing" of positive ($x^{pos}$) and negative ($x^{neg}$) sampling techniques onto an existing baseline (DAPO/GRPO). The use of a $D_{\text{KL}}$ loss for the $x^{neg}$ sample is a common regularization technique in standard (non-RL) fine-grained classification, which calls into question its novelty as a policy optimization method.
* Unclear Ablation: The paper fails to clearly disentangle the individual contributions of the $x^{pos}$ and $x^{neg}$ components. It is unclear if the $x^{pos}$ (hybrid rollouts) provides any significant benefit on its own. The ablation study is missing crucial comparisons (e.g., Baseline + $x^{pos}$ only, Baseline + $x^{neg}$ only) and also lacks an analysis of the $n_1:n_2$ ratio (anchor vs. positive rollouts).

### Questions
If the authors can clearly address the following points during the rebuttal, I am open to reconsidering my score:
1. Can you further justify the novelty of TAPO as an RL algorithm, distinguishing it from simply applying a known FGC regularizer ($x^{neg}$) and a data augmentation strategy ($x^{pos}$) to a DAPO baseline?
2. Can you provide a decoupled ablation study that shows the individual performance contributions of $x^{pos}$ (Intra-class Augmentation) and $x^{neg}$ (Inter-class Augmentation)? I am particularly interested in the (Baseline + $x^{pos}$ only) result.
3. Please provide an ablation study on the ratio of anchor-to-positive rollouts ($n_1$ vs $n_2$).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Fine-R1, a multimodal large language model (MLLM) framework for fine-grained visual recognition (FGVR). It introduces a two-stage R1-style training pipeline: (1) Chain-of-Thought Supervised Fine-Tuning (CoT-SFT), where the model learns structured reasoning steps from synthesized CoT data; and (2) Triplet Augmented Policy Optimization (TAPO), a reinforcement-learning method that augments intra- and inter-class examples (anchor, positive, negative) to improve robustness and discrimination. Fine-R1 achieves superior accuracy on six FGVR benchmarks, outperforming both general and reasoning MLLMs (e.g., Qwen2.5-VL, DeepPerception) and even contrastive CLIP models, especially in few-shot and unseen-category settings.

### Strengths
**Clear motivation**: The paper clearly articulates the limitations of existing MLLMs in fine-grained visual recognition and motivates the need for improved reasoning and generalization capabilities.

**Methodological novelty**: The proposed TAPO combines reinforcement learning with triplet-based augmentation, conceptually bridging contrastive and policy-optimization paradigms.

**Strong empirical results**: Fine-R1 consistently surpasses prior MLLMs and CLIP baselines across multiple FGVR datasets and evaluation settings (closed/open-world, seen/unseen).

**Well-structured analyses**: The ablation studies and hypothesis testing (H1–H3) are thoughtful, showing that Fine-R1’s gains stem from better knowledge deployment rather than merely improved visual features or memorization.

### Weaknesses
**Limited data scale for CoT-SFT**: The CoT dataset reportedly contains only 404 samples, raising doubts about generalization and potential overfitting to synthetic patterns.

**Evaluation bias toward Qwen-based baselines**: All base models are Qwen-VL variants; cross-model validation (e.g., on LLaVA or InternVL foundations) is missing, which might limit claims of generality.

**Complexity vs. gain**: TAPO adds considerable training and sampling overhead (triplet construction, multi-rollout reward computation), but the improvement over DAPO (+1.6%) is relatively modest.

**Conceptual overlap**: While well-positioned as an R1-style method, the framework’s connection to previous RL-based reasoning systems (e.g., Visual-RFT, VLM-R1) could be made more precise to clarify incremental novelty.

**Interpretability of CoT generation**: The reasoning chains are auto-synthesized by another MLLM (Qwen2.5-VL-32B), but no human verification or quality metrics are provided, leaving uncertainty about rationale faithfulness.

### Questions
**1. Data efficiency and generalization:**
The CoT-SFT dataset contains only 404 samples. Could the authors elaborate on how they ensure generalization beyond this limited synthetic set? For instance, were any experiments conducted to test scaling behavior when using larger or more diverse CoT data?

**2. On cross-model validation:**
Since all base models are Qwen-VL variants, have the authors attempted to reproduce the results on alternative architectures (e.g., LLaVA or InternVL) to confirm that the proposed TAPO framework generalizes across backbones?

**3. On training efficiency and computational cost:**
TAPO introduces triplet sampling and additional rollouts. Could the authors quantify the training-time or GPU-hour overhead compared to DAPO or standard GRPO?

**4. On incremental novelty and relation to prior work:**
The paper positions Fine-R1 as an R1-style framework. Could the authors clarify the specific conceptual or algorithmic distinctions from prior RL-based reasoning methods such as Visual-RFT, Vision-R1, or VLM-R1? What unique design choices make TAPO fundamentally different rather than a variant?

**5. On CoT faithfulness and quality control:**
The CoT rationales are generated automatically by Qwen2.5-VL-32B. Did the authors evaluate their correctness or consistency? How sensitive is model performance to potential noise in these synthesized CoTs?

I will adjust my score based on the authors’ response.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper targets FGVR by proposing a two-stage approach to improve MLLMs. The first stage, CoT SFT, uses supervised fine-tuning with a structured Chain-of-Thought to teach the model an interpretable, fine-grained reasoning process. The second stage, TAPO, employs a triplet-augmented policy optimization to sharpen the model's ability to distinguish between highly similar classes. In a 4-shot, base-to-new setting across six FGVR datasets, the method reports significant gains over both general-purpose MLLMs and strong contrastive models like SigLIP.

### Strengths
1、t's widely acknowledged that general MLLMs underperform contrastive models like CLIP/SigLIP on fine-grained tasks. The paper's attempt to close this gap using structured CoT and reinforcement learning, rather than relying solely on massive labeled datasets, is a practical and appealing direction.

2、The two-stage approach is well-motivated. The CoT SFT stage provides an interpretable reasoning framework ("visual analysis → candidate subclasses → comparison → prediction"), while the triplet-based policy optimization (TAPO) directly targets the core challenge of FGVR: maximizing inter-class variance while minimizing intra-class variance.

3、The experiments are extensive, covering both closed-set and open-set scenarios. The use of multiple evaluation metrics (including semantic similarity) and thorough ablations helps to clearly identify the sources of performance improvement.
Weaknesses & Suggestions

### Weaknesses
1、The method feels like an application of existing CoT and RL techniques, not a new paradigm for FGVR. A direct comparison against a generic CoT prompt is needed to prove the proposed reasoning structure is truly beneficial. 

2、Using a SigLIP encoder to calculate a key metric while also comparing against SigLIP is a potential conflict. The results should be cross-verified with another encoder (like CLIP's) to ensure fairness. 

3、The reported gains are marginal and lack error bars, making them unconvincing given the high variance of RL and CoT methods. 

4、It's unclear if the gains come from the reasoning structure or just from generating longer text, a known confounder. The paper needs length-controlled experiments to prove its central claim. Weak Baselines. 

5、The CLIP/SigLIP baselines seem under-tuned, as they lack standard optimizations like prompt ensembling.

### Questions
1、Novelty of CoT Application Needs Clarification. The core idea is to combine structured CoT with policy optimization. However, using CoT to enhance reasoning is already a well-explored area in LLMs. The paper needs to more clearly articulate the fundamental difference between its "CoT SFT" and existing work on few-shot CoT prompting or standard CoT-based supervised fine-tuning.

2、Details on Semantic Similarity Metric are Lacking. The paper relies on a SigLIP text encoder to calculate semantic similarity, which is a key metric. However, it needs to provide more details on the threshold selection, the metric's sensitivity to different class granularities, and the potential impact of text normalization or synonyms.

3、Clarity on Acronyms. Several new acronyms ("CoT SFT," "No-Thinking-RL," "TAPO," etc.) should be defined with their full names upon first use to improve readability.

### Soundness
2

### Presentation
2

### Contribution
2
