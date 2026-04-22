# Think Twice to See More: Iterative Visual Reasoning in Medical VLMs

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 4, 6

## Abstract
Medical vision-language models (VLMs) excel at image-text understanding but typically rely on a single-pass reasoning that neglects localized visual cues. In clinical practice, however, human experts iteratively scan, focus, and refine the regions of interest before reaching a final diagnosis. To narrow this machine-human perception gap, we introduce ViTAR, a novel VLM framework that emulates the iterative reasoning process of human experts through a cognitive chain of "think-act-rethink-answer''. ViTAR treats medical images as interactive cognitive objects, enabling models to engage multi-step visual reasoning. To support this approach, we curate a high-quality instruction dataset comprising 1K interactive examples that encode expert-like diagnostic behaviors. In addition, a 16K visual question answering training data has been curated towards fine-grained visual diagnosis. We introduce a two-stage training strategy that begins with supervised fine-tuning to guide cognitive trajectories, followed by the reinforcement learning to optimize decision-making. Extensive evaluations demonstrate that ViTAR outperforms strong state-of-the-art models. Visual attention analysis reveals that from the "think" to "rethink" rounds, ViTAR increasingly anchors visual grounding to clinically critical regions and maintains high attention allocation to visual tokens during reasoning, providing mechanistic insight into its improved performance. These findings demonstrate that embedding expert-style iterative thinking chains into VLMs enhances both performance and trustworthiness of medical AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents the ViTAR (Visual Thinking and Action-centric Reasoning) framework, designed for iterative visual reasoning in the medical domain with explicit focus refinement.  It tackles the challenge of single-pass reasoning in traditional medical VLMs, introducing the "think-act-rethink-answer" cognitive chain that treats medical images as interactive cognitive objects for multi-step diagnosis. The framework is supported by two new, high-quality datasets—a 1K interactive instruction dataset encoding expert diagnostic trajectories for supervised fine-tuning (SFT), and a 16K VQA training dataset geared toward fine-grained visual diagnosis utilized for reinforcement learning (RL).  The model integrates its intermediate act (e.g., placing an ROI bounding box) directly into the reasoning process and is trained using a two-stage guidance and optimization strategy that leverages format and accuracy rewards. ViTAR achieves competitive performance across seven medical VQA benchmarks, demonstrating its efficacy on reasoning-intensive tasks.

### Strengths
1. The paper makes a significant leap by introducing the explicit "think-act-rethink-answer" cognitive chain, which systematically integrates the iterative reasoning behavior of human experts into the medical VLM framework, thereby addressing the limitations of single-pass inference in capturing subtle, localized visual cues.
2. A robust methodology is employed through the two-stage training strategy, utilizing both Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), ensuring the model acquires both the foundational cognitive structure and optimized decision-making skills.
3. The quality of the work is bolstered by the creation of specialized, high-fidelity datasets, including the 1,000-example interactive instruction dataset, which is essential for training the model's unique action and refinement capabilities.

### Weaknesses
1. The experimental validation lacks direct human evaluation of the model's interpretability. The paper does not provide evidence from clinicians on whether the "think-act-rethink" output is genuinely more helpful or trustworthy in a clinical context compared to simpler chain-of-thought models.
2. The reliance on bounding boxes for the "act" step may introduce a limitation in localization precision. Future work should explore whether using more fine-grained action representations, such as segmentation masks, could lead to a more accurate and beneficial "rethink" step.
3. The paper would benefit from a more detailed analysis, such as an ablation study, specifically detailing potential failure modes of the Reinforcement Learning (RL) stage. Understanding scenarios where the current format and accuracy rewards might drive clinically sub-optimal reasoning is necessary for future refinement of the reward function.

### Questions
1. How does the model's performance on the crucial "act" step—the placing of the bounding box ROI—generalize to out-of-distribution medical images (e.g., rare diseases, novel image modalities, or images with poor resolution) that are not represented in the small 1,000-example interactive instruction dataset?
2. The paper focuses solely on bounding boxes for the "act" step.  Would the model achieve superior performance and more clinically relevant interpretability if it utilized a more fine-grained action representation, such such as segmentation masks or heatmaps, to guide the rethink process?
3. The core claim is that the iterative output closes the "machine-human perception gap."  Therefore, a critical question is: Does the final output, including the intermediate "think-act-rethink" steps, lead to a statistically higher agreement or trust score when evaluated by human clinicians compared to a high-performing single-pass VLM?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ViTAR, a medical vision–language framework that mimics medical expert workflows via "think-act-rethink-answer", trying to enable 2-step visual reasoning on images. The training strategy consists of SFT and RL. Some datasets are curated. The authors further do some attention analyses showing tighter grounding on clinically salient regions across reasoning rounds.

### Strengths
- The curated sets could be potentially valuable if released with full documentation.
- The comparisons (vanilla VLM vs. ViTAR; action-centric reasoning) are informative.
- The SFT --> RL two-stage recipe is classical.

### Weaknesses
- Using LLMs to both generate and validate VQA datasets, which might have a risk of compounding hallucinations and confirmation bias.
    - A general LLM (e.g., Qwen2.5-72B-Instruct) may lack medical grounding.
- Question templates might introduce leakage shortcuts.
- More detailed analyses about training and datasets are required.
- Inferring "trustworthiness" from attention visualizations is tenuous; attention is not a validated explanation and remains debated [1].

[1] "Attention is not explanation." arXiv preprint arXiv:1902.10186 (2019).

### Questions
### Datasets

> VQA Generation, VQA Validation

1. It seems that both VQA generation and validation are done by LLMs, which might introduce compounding hallucinations. The authors should consider to add radiologist adjudication on some samples; report inter-annotator / human agreement or any empirical error rate.

> We utilize Qwen2.5-72B-Instruct (Yang et al., 2024) to generate and verify VQA samples in our constructed training dataset.

2. Qwen2.5-72B-Instruct may lack medical grounding. It might be better if the authors could consider an ensemble way (e.g., general + medically fine-tuned validators) , retrieval-augmented checks, and cross-model agreement before accepting items. 

3. Question templates might leak shortcuts (yes/no imbalance, answerable from labels without image). The authors should add paraphrases/natural phrasing, balance labels, and include “unanswerable” cases with precise criteria.

4. In practice, real clinical VQA might need multi-view/series context and report-grounded reasoning. How does the proposed dataset take such scenaiors into consideration?

### Training (SFT, RL)

5. The authors should consider to ablate SFT -> RL vs. RL-from-scratch (R1-Zero–style). A curriculum learning way (format-only reward → add accuracy) on actions could enable emergent thinking without full SFT; try to report stability, sample efficiency, and quality trade-offs.

> the model receives 0.2 points if the “thought” and “action” fields can be successfully parsed according to the prescribed schema

> an additional 0.2 points if the final answer adheres to the expected format

6. **Reward hacking**. Current reward shaping (up to +0.4 for format, +1.0 for exact match) invites potential risks of rewarding hacking (well-formed but vacuous thoughts/actions, or premature terminate guessing). Is there any case that the agent learns to generate responses with good formats but wrong reasoning or wrong answers? The authors should conduct some ablation studies for the reward design.

> Figure 7: Evolution of RL training over three phases (P1–P3). P1: Initial and sub-optimal convergence. P2: Exploration and enhancement. P3: Pruning and final convergence. Shading indicates confidence interval.

7.1 Figure 7. Show total reward curves with loss and format reward. How about the total reward / accuracy reward vs. steps? 

7.2 Besides, in RL training, it is common that while reward is increasing, the downstream validation accuracy is not very good. The authors should consider add a figure of downstream validation accuracy vs. steps. 

> L462-472

7.3 Is there any proof / detailed validation for your discussions about P1, P2 and P3?

### General Questions

8. With "Re-thinking after action", it seems that in the current setting, LLM plays like an agent and only has two rounds: think1, act1, think2, answer. What if some medical issues require multiple rounds of reasoning? e.g., think1, act1, think2, act2, think3, answer?

8.1 Should an agent behave better if the agent could do thinking in multiple turns? Why only two turns are considered here?

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
5

### Summary
This paper proposes ViTAR, a two-stage vision–language reasoning framework for visual question answering. The model is first trained with supervised fine-tuning (SFT) and then refined with reinforcement fine-tuning (RFT) to promote step-by-step reasoning and self-correction. At the first turn, ViTAR produces a first-glance answer along with a bounding box that localizes the visual evidence. This box is overlaid on the image to guide a brief reflective dialogue, allowing the model to reassess and potentially revise its initial prediction. ViTAR achieves competitive performance across several standard VQA datasets compared to non-reasoning baselines.

### Strengths
1).  The idea of mimicking the iterative diagnostic workflow of human radiologists is a sensible and interesting direction. While this concept has been mentioned in prior literature, its explicit modeling in vision-language systems is still relatively underexplored.
2). The paper is clearly written and generally easy to follow. The structure of the method, training strategy, and experiments are all presented in a logical flow.
3). I find the use of bounding boxes as intermediate visual grounding signals to be a meaningful design choice. This adds interpretability to the reasoning process, which is especially valuable in high-stakes medical settings.
4). The authors built a reasonably comprehensive data pipeline, combining instruction-based supervision and VQA sample generation from object detection datasets. While not perfect, the attempt to enrich training data with structured reasoning is appreciated.
5). It's good to see that the authors go beyond standard answer accuracy by also evaluating the success rate of action executions (e.g., correct parsing of bounding boxes). This adds an extra layer of analysis on the model's functional reliability.

### Weaknesses
1). While the paper motivates ViTAR by appealing to the iterative workflow of radiologists, the instantiated “action” space in the method and data is essentially limited to drawing a bounding box and then rethinking. In clinical reading, radiologists typically zoom/pan, adjust window/level, scroll through slices or rotate views, and use cine/MPR rather than marking boxes on the image. As a result, the claimed alignment with expert workflows feels overstated: the current action set resembles annotation tooling more than real diagnostic interaction. I would encourage the authors to (i) tone down the claim of workflow alignment, or (ii) extend the action space and evaluate tasks that require realistic interactions (e.g., window/level control, slice scrolling on CT, rotation, or view changes). The paper’s own formulation and data construction emphasize box marking as the core interaction (e.g., “mark” actions and bbox-driven VQA/instructions), which reinforces this concern.

2). While the paper adopts a two-turn reasoning routine, the think–answer–rethink–answer pattern has already been explored since 2023 [1]. Given this prior art, the claimed novelty should squarely focus on what is new in the reward design and how the scheme shapes the training signal beyond plain GRPO-style reinforcement fine-tuning that simply optimizes group-wise answers. However, the reward is not sufficiently analyzed: relying on R = R_format + R_acc is common in prior work, and the paper does not examine alternative/shaped rewards or their sensitivity. In particular, it is unclear why there is no explicit reward for the first-turn bounding box quality (e.g., IoU/coverage/precision), despite the second-turn refinement depending on that localization. The paper would benefit from a more in-depth, head-to-head comparison and summarization to clarify what is truly different from prior iterative-reasoning + RL methods.

3). The comparison in Table 1 is potentially not FAIR. The proposed method is trained on a self-curated corpus (interactive instructions plus ~16K VQA samples derived from detection data), whereas many baselines are off-the-shelf models whose training data, scale, and overlap with the target benchmarks are not matched to ViTAR. Moreover, the paper itself notes that some reasoning baselines (e.g., Chiron-o1 / MedCCO) may have train–test overlap on certain benchmarks (numbers shown in gray), underscoring that different systems rely on different data regimes. As a result, the current leaderboard-style comparison risks over- or under-estimating the true effect of the proposed training recipe. 

4). The paper adopts a two-stage pipeline (SFT warm-up followed by GRPO), but it is unclear why the method does not directly cold-start RL on the curated dataset or whether such a setting fails in practice. A convincing case would require head-to-head comparisons among (i) SFT-only, (ii) RL-from-scratch (cold start), and (iii) SFT then RL, ideally under matched compute and token budgets. Beyond final accuracy, please report learning curves and stability metrics (e.g., convergence speed, variance across seeds, action execution success rate, and format reward trajectories). If the claim is that SFT is necessary to learn the output schema or stabilize action calls, show where cold-start RL breaks (e.g., persistent parsing errors, sparse-reward plateaus, or degenerate actions) and how SFT alleviates these failure modes. Ablations on SFT data size/quality, KL regularization to the SFT policy, and curriculum or behavior-cloning pretraining would further clarify whether the two-stage design is an essential ingredient or merely a convenient choice. 

5). In addition to 4), the two-stage (SFT and GRPO) pipeline is harder to reproduce and more compute-intensive than one-stage RFT baselines (e.g., Med-R1).

6). The paper does not commit to a public code release and an anonymous repo link is insufficient for reproducibility.

7). The curated instruction set and the 16K VQA corpus are not stated to be publicly released. 

[1] REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS

### Questions
Refer to weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ViTAR, a vision–language framework that emulates expert iterative reasoning via a “think–act–rethink–answer” cognitive chain. ViTAR treats medical images as interactive cognitive objects to enable multi-step visual reasoning. The authors curate a high-quality instruction set of 1,000 interactive examples encoding expert-like diagnostic behaviors and assemble 16,000 VQA training instances for fine-grained visual diagnosis. Extensive evaluations show that ViTAR outperforms strong state-of-the-art baselines.

### Strengths
1. This paper introduce ViTAR, a VLM grounded in the “think–act–rethink–answer” paradigm for multi-step visual reasoning aligned with expert diagnostic workflows;
2.  ViTAR dynamically sharpens visual grounding on clinically critical regions while sustaining strong attention to visual tokens, thereby enhancing multimodal reasoning.

### Weaknesses
1. Both the two-stage training regimen and the “think–act–rethink–answer” paradigm are already well established in prior MLLM and medical VLM literature. This weakens the paper’s claimed contribution.
2. The paper does not clearly distinguish itself from existing “think-with-images” approaches in medical MLLMs. In lines 143–145, the authors discuss only tool-augmented variants, omitting other “think-with-images” methods (e.g., [1,2]).
3. In the reinforcement learning stage, are there more effective optimization objectives or reward designs that could further improve performance?

[1] Think Twice: Perspective-Taking Improves Large Language Models’ Theory-of-Mind Capabilities

[2]  VGR: Visual Grounded Reasoning

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ViTAR (Visual Thinking and Action-centric Reasoning), a medical vision-language model that mimics clinicians’ iterative diagnostic process through a “think-act-rethink-answer” reasoning framework. Unlike conventional single-pass models, it employs a two-stage training strategy: supervised fine-tuning to guide expert-style cognitive trajectories, followed by reinforcement learning with accuracy and format rewards to optimize autonomous decision-making. To support this paradigm, the authors construct a 1K interactive instruction dataset encoding expert-like diagnostic behaviors and a 16K fine-grained VQA dataset derived from medical detection corpora. Experiments across seven medical VQA benchmarks show that ViTAR achieves state-of-the-art performance among open-source models.

### Strengths
1. The paper introduces a “think-act-rethink-answer” process inspired by real diagnostic workflows, offering a more interpretable and human-aligned reasoning paradigm for medical vision-language models.
2. ViTAR achieves state-of-the-art or competitive results on multiple medical VQA datasets, demonstrating that iterative reasoning and reinforcement learning can effectively enhance both diagnostic accuracy.

### Weaknesses
1. The core framework (think-act-rethink-answer) seems an extension of existing iterative reasoning paradigms used in general vision-language models, e.g., [1] [2]. While the paper adapts this to the medical domain, it lacks a clear claim for the fundamental difference from prior multi-step reasoning or self-reflection frameworks.
2. The paper claims that ViTAR follows a multi-step reasoning process, but there is no quantitative evidence showing that the model truly reasons step by step during inference. It is unclear whether correct answers result from genuine reasoning or accidental alignment despite incorrect intermediate steps. Metrics such as step-wise consistency or attention-ROI overlap would help verify the faithfulness of the claimed reasoning process.

[1] Perception Before Reasoning: Two-Stage Reinforcement Learning for Visual Reasoning in Vision-Language Models. 2025.

[2] Grounded Reinforcement Learning for Visual Reasoning. 2025.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
