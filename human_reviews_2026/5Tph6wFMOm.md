# SCoT: Teaching 3D-LLMs to Think Spatially with Million-scale CoT Annotations

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Recent advances in 3D Large Language Models (3D-LLMs) show strong potential in understanding and interacting with 3D environments, yet their training data typically lack explicit reasoning processes, limiting complex spatial reasoning and task planning.
To address this, we annotate SCoT, a million-scale Chain-of-Thought dataset spanning three levels: a) Spatial Perception (what is there), recognizing object properties, relations, and scene attributes; b) Spatial Analysis (what does it mean), inferring rationality, functionalities, and physical implications; c) Spatial Planning (what should I do), integrating perception and reasoning for actionable strategies. Unlike prior datasets supervising only answers, SCoT annotates intermediate reasoning grounded in scene cues, specifically for analysis and planning tasks. Results show that CoT supervision greatly benefits complex analysis and planning but induces hallucinations and accuracy drops in simple perception. These findings highlight both the necessity and the nuanced challenges of scene-grounded reasoning for advancing 3D intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SCOT, a new million-scale dataset designed to teach 3D-LLMs to reason spatially. To address the limitations of existing datasets that lack explicit reasoning steps, SCOT provides structured Chain-of-Thought annotations across a three-level taxonomy of tasks: Spatial Perception, Spatial Analysis, and Spatial Planning. A key contribution is the use of scene-grounded annotations, which force the model to base its reasoning on verifiable 3D evidence, thus improving the transparency and accuracy of complex analysis and planning tasks. Experiments show that this method significantly enhances the model ability to perform complex spatial reasoning, while also demonstrating that overuse of CoT for simple perception can be detrimental.

### Strengths
This paper introduces SCOT, a new million-scale dataset designed to teach 3D-LLMs to reason spatially. To address the limitations of existing datasets that lack explicit reasoning steps, SCOT provides structured Chain-of-Thought annotations across a three-level taxonomy of tasks: Spatial Perception, Spatial Analysis, and Spatial Planning. A key contribution is the use of scene-grounded annotations, which force the model to base its reasoning on verifiable 3D evidence, thus improving the transparency and accuracy of complex analysis and planning tasks. Experiments show that this method significantly enhances the model ability to perform complex spatial reasoning, while also demonstrating that overuse of CoT for simple perception can be detrimental.

### Weaknesses
The scene-grounded reasoning lacks strong evidence, which relies on LLM judge evaluations rather than direct verification against ground-truth data, which may lead some problems on robustness and objectivity.

### Questions
What is the impact of generating detailed CoT reasoning on inference efficiency, and how does this affect the suitability for real-time applications such as robotics?

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
This work introduces a million-scale chain-of-thought dataset aimed at improving the spatial reasoning capabilities of existing 3D LLMs, along with a baseline model, SCoT-Reasoner. Experiments show that models fine-tuned on the SCoT dataset exhibit performance improvements across various 3D visual question answering benchmarks. Additionally, these fine-tuned models exhibit reasoning processes that are transparent, faithful to the scene, and inherently more trustworthy.

### Strengths
1. The proposed SCoT dataset covers a diverse set of spatial tasks and is large in scale. The paper provides sufficient details to enable the research community to reproduce the data generation pipeline.
2. The paper is well-presented and easy to follow.

### Weaknesses
1. SCoT is based on only 800 scenes from ScanNet, which is a relatively small scale. While the number of CoT examples is large, I’m concerned whether such a limited range of scene samples can genuinely enhance models' spatial reasoning ability on unseen real-world 3D environments.
2. In Section 3, the author describes the SCoT format as “Query–CoT–Answer.” However, the spatial perception data seems to include only QA pairs without explicit reasoning. Clarification is needed on how these are treated as CoT examples.
3. Most experiments are conducted on the SCoT test set. It would strengthen the paper if the authors included additional evaluations on out-of-domain datasets such as MSR3D [1] and Hypo3D [2] to assess generalization.
4. The authors are encouraged to provide more discussion, ideally supported by results, demonstrating that fine-tuning on SCoT offers greater benefits for spatial reasoning compared to fine-tuning on previous 3D SCoT datasets.

[1] Linghu, Xiongkun, et al. "Multi-modal situated reasoning in 3d scenes." NeurIPS 2024.

[2] Mao, Ye, et al. "Hypo3D: Exploring Hypothetical Reasoning in 3D." ICML 2025.

### Questions
1. Throughout the paper, the three main tasks are introduced in the order of SCoT-Perception, SCoT-Analysis, and SCoT-Planning. Why is a different order used in the results section (Section 5.2)? Maintaining consistency would improve readability.

2. It is unclear which model is evaluated in Table 5. The authors should clarify the model configuration or variant used for these results.

3. SCoT-Reasoner appears to be a key model in the evaluation, yet most of its technical details are relegated to the appendix. I recommend moving more of these details into the main Method section to help readers better understand the approach.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper propose SCoT, a large-scale CoT datasets for spatial reasoning. It spanning three levels including spatial perception, spatial analysis and spatial planning. SCoT annotates intermediate reasoning grounded in scene cues. It introduces<SI> token let CoT grounded in scenes context, and reduce hallucinations. Results shows model trained on their proposed datasets benefits complex analysis and planning.

### Strengths
1. This paper constructed a valuable large-scale datasets which including diverse scenes with their classified three-tier taxonomy of 3D task.

2. This paper propose a detailed data construction pipeline which provides empirical insights for spatial reasoning data construction.

### Weaknesses
1. The novelty is not very clear in this paper, could the author emphasize the key differences between this work and previous spatial reasoning CoT studies, as listed in Table 1 such as 3D-R1 and SpaceR-151k, they are also including multi tasks and reasoning, so what's the biggest advantage of your dataset compared with them? Have you compared with these datasets in controlled settings?

2. The base model used to train is too weak to verify the usefulness of the proposed method. I don't understand why using Vicuna-7B as the pretrained model even in 2025 today. 

3. I doubt the validity of the evaluation. If I understand correctly, part of the evaluation data are self-constructed by yourself? (In the table 2, and table 3.) The public evaluation data only appear in Table 7, but over half of results in Table 7 your proposed method can't beat other methods. I think it's extremely unfair to compare with other methods if using your self-built data in the main results, considering we don't know whether the performance gain comes from in distribution benefits. Are training and test scenes completely disjoint?

4. Using same models (GPT-4.1, DeepSeek, Qwen) generating training data as evaluator, which will further introduce bias in evaluation.

### Questions
Refer to weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents SCoT, a million-scale Chain-of-Thought (CoT) dataset designed to teach 3D Large Language Models (3D-LLMs) to reason spatially.
SCoT organizes tasks into three tiers — Spatial Perception (“what is there”), Spatial Analysis (“what does it mean”), and Spatial Planning (“what should I do”) — and annotates CoT reasoning only where necessary.
The dataset introduces scene-grounded reasoning with explicit \<SI> tags to ensure factual alignment with 3D context.
Experiments on multiple baselines (Chat3D, ChatScene, Video3D-LLM) and the proposed SCoT-Reasoner show significant improvements in reasoning explainability, faithfulness, and planning accuracy.

### Strengths
High-quality dataset: 1.1M diverse samples covering perception, reasoning, and planning with strong annotation rigor and cross-checking.

Innovative design: The three-tier CoT taxonomy (perception–analysis–planning) effectively balances reasoning depth and hallucination control.

Grounded CoT methodology: The \<SI> tag mechanism enforces transparent, scene-based reasoning rather than textual hallucination.

Strong empirical validation: Extensive quantitative and qualitative analyses demonstrate consistent gains in complex reasoning tasks.

Practical impact: Provides a scalable framework for training reliable, interpretable 3D-LLMs relevant to embodied AI and robotics.

Solid writing and clarity: The paper is well-organized, and figures (e.g., Fig. 1 & 3) effectively illustrate the framework and dataset pipeline.

### Weaknesses
The use of LLM-based evaluators (ChatGPT, Qwen, DeepSeek) for “Explainability,” “Faithfulness,” and “Trustworthiness” is well-motivated but inherently subjective.
It would strengthen credibility if the authors validated inter-evaluator consistency (e.g., correlation scores between evaluators).

The paper does not evaluate how models trained on SCoT generalize to unseen 3D environments or other datasets (e.g., ARKitScenes or Omni3D).

The paper could provide more detailed ablations: Comparing models trained with different CoT lengths or varying levels of \<SI> grounding.；Analyzing which task types (object vs. scene vs. planning) benefit most from CoT supervision.；Evaluating performance trade-offs when CoT is partially removed during inference.

Although Table 5 highlights hallucination in perceptual CoT, the causes (e.g., linguistic priors vs. visual overfitting) are not deeply analyzed.
This discussion could offer more insight into why CoT sometimes harms visual fidelity.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
