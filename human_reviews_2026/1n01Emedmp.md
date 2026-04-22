# Exploration for Building Next-Generation Foundation MLLMs via Self-Learning

- Avg Score: 4.40
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 4, 6

## Abstract
While inference-time computation and post-training optimization have significantly advanced multimodal large language models (MLLMs), these advancements remain constrained by the capabilities of foundation models. We argue that effective model advancement requires strong synergy among pre-training, inference-time computation, and post-training optimization. In this paper, we introduce Self-Improving cognition (SIcog), a self-learning framework for building next-generation foundation MLLMs by imparting multimodal knowledge and enhancing systematic cognitive capabilities through multimodal pre-training with self-generated data. Specifically, we propose Chain-of-Description for step-by-step visual understanding and integrate structured Chain-of-Thought (CoT) reasoning to support in-depth multimodal reasoning. SIcog first equips a base model with systematic perception and reasoning using minimal external supervision. The enhanced models then generate candidate image captions and CoT reasoning responses for unlabeled images and image-question pairs across diverse tasks, which are filtered through a semantic-similarity-guided self-consistency mechanism. These high-quality, self-generated samples enable large-scale multimodal pre-training, creating a self-improvement loop. Experiments demonstrate SIcog's effectiveness in developing MLLMs with enhanced multimodal cognition. Using only 213K self-generated pre-training samples, SIcog achieves significant improvements, including +3.6\% on MMStar and +3.5\% on AI2D, outperforming previous pre-training approaches. When combined with post-training techniques for CoT reasoning, SIcog yields +9\% gains on MMVet and +8.5\% on ScienceQA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SICOG, a self-improving framework for Multimodal Large Language Models (MLLMs) that enhances multimodal cognition (perception & reasoning) through self-generated data. SICOG's core innovations are the Chain-of-Description (CoD), which structures visual analysis into systematic steps for comprehensive perception, and the integration of structured Chain-of-Thought (CoT) reasoning into pre-training. The framework operates in a loop: 1) Fine-tune a base MLLM with minimal annotations using CoD and CoT data to develop systematic cognition; 2) Use the enhanced model to generate candidate captions and responses for unlabeled images; 3) Curate high-quality self-generated data via self-consistency checks; 4) Perform multimodal pre-training on the curated data to build a stronger next-generation MLLM.

Experiments show SICOG significantly outperforms previous pre-training methods (e.g., strong-to-weak distillation, multi-agent collaboration) across diverse benchmarks (e.g., +3.6% on MMStar, +3.5% on AI2D) using only 213K self-generated samples. It also provides a stronger foundation for post-training CoT reasoning, achieving further gains (e.g., +9% on MMVet, +8.5% on ScienceQA). SICOG demonstrates that effective model enhancement requires synergistic collaboration between pre-training (providing foundational cognition) and downstream mechanisms.

### Strengths
Originality: SICOG introduces a novel self-improving framework that uniquely integrates structured perception (Chain-of-Description) and reasoning (Chain-of-Thought) into multimodal pre-training, diverging from prior works that treat these as separate post-training enhancements. 

Significance: SICOG addresses a critical bottleneck in MLLM development: the dependency on large-scale, externally annotated data for pre-training, which often lacks coherence and reasoning focus. By enabling models to self-generate high-quality, cognitively aligned data, it reduces reliance on costly proprietary tools or expert ensembles.

### Weaknesses
1. Why was GRPO not investigated in Section 4.4?

2. Limited Exploration of Perception, Reasoning and multi-turn conversation Capabilities in Pre-Training: While SICOG innovatively introduces structured mechanisms for perception (Chain-of-Description) and reasoning (Chain-of-Thought), the evaluation of the pre-trained base MLLM does not fully or directly probe the nuanced improvements in these specific capabilities. The transformation of high-consistency CoD captions into a multi-turn format is a clever idea for building conversational data. However, the paper provides no evaluation of this capability. Evaluate these capabilities on such benchmark, like ConvBench,VisIT-Bench, or similar benchmarks.

3. Potential Overfitting and Generalization Concerns: The self-improving loop risks creating a "feedback bubble", where the model reinforces its own styles and potential biases. The data used for generation and for evaluation (standard benchmarks) may share underlying distributions. Test generalization on more challenging and diverse benchmarks that require complex, compositional reasoning, such as VCR (Visual Commonsense Reasoning).

### Questions
1. "(i) fine-tuning with 35 detailed descriptions" in caption of Table 14, 35 or 35K?

### Soundness
2

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
3

### Summary
This paper proposes a self-learning framework that trains a multimodal model using self-generated data. The authors adapt a base model with minimal external supervision. This adapted model is then used to generate image captions, chain-of-thought responses, and image–question pairs. The generated data is filtered based on semantic similarity and subsequently used to further pretrain the multimodal model.

### Strengths
1. The core idea of leveraging self-generated multimodal data for training is clear and holds promise for advancing multimodal models.
2. The method is described in sufficient detail.
3. The experiments are comprehensive and cover a variety of benchmarks.

### Weaknesses
1. The technical novelty is limited, as the proposed method primarily combines existing techniques like self-training, data filtering, and multimodal pretraining.
2. The experimental results are not particularly impressive and appear comparable to those of existing methods on several benchmarks.
3. The approach is largely heuristic and lacks theoretical analysis or justification.
4. While there are extensive evaluations of performance, there is limited analysis of computational or data efficiency.
5. [Minor organizational issue] There is significant redundancy between the main paper and the appendix, which makes the paper harder to follow.

### Questions
The authors argue that there is synergy between pre-training and downstream mechanisms and introduce an intermediate stage (Stage 1.5). Could you provide more in-depth analysis of this aspect?

Could the authors share additional statistics regarding the quality of the self-generated data? For example, metrics on diversity, as well as the average length of generated captions and CoT responses?

### Soundness
3

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
4

### Summary
injecting multimodal knowledge and enhancing systematic cognition. The approach first instills structured visual perception (Chain-of-Description, CoD) and structured reasoning (CoT) with minimal supervision, then adds a Stage-1.5 multimodal pre-training step that uses self-generated data filtered by semantic self-consistency. The principal contributions of this paper can be organized into three layers:
1) Methodological: propose SICOG, a self-learning framework that enhances MLLMs’ systematic cognition for constructing next-generation foundation MLLMs through multimodal pre-training with self-generated data;
2) Framework: introduce Chain-of-Description, a structured visual understanding mechanism that enables step-by-step interpretation of visual content to improve perceptual quality.;
3) Empirical: demonstrate SICOG’s effectiveness across various benchmarks on both low- and high-resolution MLLMs, significantly surpassing previous approaches.

### Strengths
1）Methodological Innovation: Proposes a closed-loop self-learning pipeline in which the model generates CoD and CoT data, filters them via semantic self-consistency, reducing dependency on external annotations；

2）Framework Innovation: Introduces a Stage-1.5 multimodal pre-training step to the standard two-stage pipeline, injecting systematic cognition into the foundation model before instruction tuning, and co-designing pre-training, post-training, and inference-time computation as a single self-improving paradigm；

3）Comprehensive Validation: With approximately 213K self-generated samples, the method yields consistent gains across diverse benchmarks (e.g., +3–4 on MMStar), and combining with CoT post-training brings further lifts (e.g. +8–9 on MMVet/ScienceQA); ablations under matched compute outperform recaption-style pre-training.

### Weaknesses
1）This paper positions the innovation as the combination of CoD, CoT, and the self-learning process, but these components largely build on established methods; the independent contributions and motivations should be articulated more clearly；

2）The demonstration of performance improvement is adequate, but lacks the analysis of computational overhead in the self-learning framework. Is the extra cost of generation and filtering justified versus conventional external model pretraining;

### Questions
1）In Step 3 (self-consistency filtering), how was the semantic-similarity threshold selected? The paper provides the value but no rationale or sensitivity analysis.

2）Since the self-generated pre-training data were derived from existing datasets, what steps did the paper take to ensure that there was no overlap between these data and the benchmarks (e.g., MMBench, ScienceQA)?

3）The performance improvement is attributed to both the self-learning framework and the CoD/CoT paradigm. How was the contribution of each component isolated？

4）what is the dependency between the CoD and CoT stages? Would swapping their execution order significantly affect the final performance?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the limitation that current MLLMs’ capabilities are constrained by foundational pre-training (overreliant on external annotations and lacking systematic cognition). It proposes **SICOG**, a self-learning framework for building next-generation foundation MLLMs by enhancing multimodal perception and reasoning via self-generated data. Key components include: (1) **Chain-of-Description (CoD)** (step-by-step visual analysis: salient content → fine-grained details → relational attributes → peripheral context → synthesis) for perception; (2) **structured Chain-of-Thought (CoT)** (task clarification → visual extraction → logical reasoning → conclusion) for reasoning. The 4-step pipeline: 1) Fine-tune a base model with minimal annotations to develop basic cognition; 2) Generate candidate captions/responses via the enhanced model; 3) Curate high-quality data using semantic-similarity self-consistency; 4) Pre-train the model on curated data. Experiments on LLaVA-Qwen2-7B (low-res) and LLaVA-Qwen2-7B-UHD (high-res) show SICOG outperforms baselines (e.g., +3.6% on MMStar, +9% on MMVet with post-training CoT).

### Strengths
* The SICOG framework presents a novel and well-motivated self-learning paradigm for MLLM pre-training. Moving beyond static datasets, this self-improvement loop  (finetune -> generate -> curate -> pre-train) is a promising and logical direction for scaling model capabilities without relying on continuous, large-scale human annotation.
- The paper thoroughly evaluates its framework against relevant and strong baselines, including "Strong-to-Weak Distillation" and "Multi-Agent Collaboration". The method is tested across a wide array of 11+ benchmarks covering comprehensive understanding, hallucination, chart/table understanding, and knowledge-oriented tasks, demonstrating broad improvements.
- The paper demonstrates significant performance gains (e.g., +3.6% on MMStar, +3.5% on AI2D) using a relatively small number (213K) of *self-generated* pre-training samples. This highlights an efficient path to model improvement that does not rely solely on massive, externally-generated (and often proprietary) datasets.

### Weaknesses
* The entire self-improvement loop is bootstrapped from a base MLLM (Step 1). The quality of the self-generated data in Step 2 is therefore fundamentally capped by the capabilities of this "improved" model. If the base model's perception or reasoning is weak, it may generate poor-quality candidates that the self-consistency filter in Step 3 fails to catch, potentially leading to error propagation or "performance saturation" (as acknowledged in Appendix M ).
* The curation (Step 3) relies on a "semantic-similarity-guided self-consistency" mechanism. The effectiveness of this filter is crucial. However, the paper only provides a high-level formula (Eq. 5) and mentions using NV-Embed-v2. It's unclear how well this semantic similarity score correlates with *factual accuracy* or *reasoning correctness*. A model could "consistently" generate plausible but incorrect reasoning steps, which might still be selected by this filter.
*  The SICOG framework is a complex, multi-stage process (Finetune -> Generate -> Curate -> Align -> Pre-train -> Instruction-Tune). This complexity, while shown to be effective, may pose a high barrier to reproduction and implementation for the wider community compared to more straightforward, single-stage pre-training approaches.
* The data and code not be publicly released.

### Questions
* Regarding Step 3 (Curation), how did you validate that the self-consistency filter (semantic similarity) is a reliable proxy for data quality, especially for CoT reasoning?
* The Chain-of-Description (CoD) is a 5-step process. Did you experiment with different steps or a different order (e.g., peripheral content before relational attributes)? How sensitive is the caption quality to the specific structure of the CoD?

I look forward to an active discussion with the authors during the rebuttal phase and will revise my score accordingly.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SICOG (Self-Improving cognition), a self-learning framework designed to build next-generation foundation MLLMs by enhancing their cognitive abilities through pre-training with self-generated data. First, it proposes a self-improvement loop where a base model is fine-tuned with minimal supervision, then generates candidate captions and reasoning responses, which are filtered and used for its own large-scale pre-training. Second, it introduces "Chain-of-Description" (CoD), a novel structured mechanism that enables step-by-step visual understanding to improve perceptual quality , and integrates this with structured Chain-of-Thought (CoT) to enhance in-depth multimodal reasoning. Third, the framework uses a semantic-similarity-guided self-consistency mechanism to curate the high-quality, self-generated data, reducing reliance on external models. Finally, experiments demonstrate that this approach is effective, with SICOG achieving significant performance improvements on diverse multimodal benchmarks like MMStar and AI2D compared to previous pre-training methods.

### Strengths
- This paper is well written and easy to follow
- The proposed SICOG method is both simple and effective, enabling the model to bootstrap its own high-quality multimodal data without heavy external supervision.
- By injecting Chain-of-Description and structured Chain-of-Thought, the method produces rich, step-by-step image captions and reasoning chains, along with accurate VQA pairs.
- Comprehensive experiments on both low- and high-resolution backbones across 10+ standard benchmarks (MMStar, ScienceQA, AI2D, POPE, etc.) show consistent gains over leading pre-training approaches.

### Weaknesses
- A significant weakness of the SICOG paper is its naive curation filter, which risks systemic error propagation. The framework employs a "Self-Consistency-Guided Quality Evaluation" that selects the self-generated data candidate with the highest average semantic similarity to its peers, a method that fundamentally mistakes consensus for correctness. This creates a critical vulnerability: if the model has a consistent bias, such as a recurring hallucination , the filter will incorrectly reward this error as a "high-quality," high-consistency sample . This flawed data is then fed back into the pre-training loop (Stage 1.5) , reinforcing the very bias the framework is supposed to fix and leading to a potential "modal collapse" where the error becomes more entrenched over subsequent self-learning iterations.
- The framework's final training stage (Stage 1.5) uses a mix of self-generated perception data (CoD), reasoning data (CoT), and text-only data. The paper provides no analysis on how the ratio of these different data types impacts performance.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
