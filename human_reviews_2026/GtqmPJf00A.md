# Synergizing Understanding and Generation with Interleaved Analyzing-Drafting Thinking

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 8, 4, 6

## Abstract
Unified Vision–Language Models (UVLMs) aim to advance multimodal learning by supporting both understanding and generation within a single framework. However, existing approaches largely focus on architectural unification while overlooking the need for explicit interaction between the two capabilities during task solving. As a result, current models treat understanding and generation as parallel skills rather than synergistic processes. To achieve real synergy, we introduce the interleaved Analyzing–Drafting problem-solving loop (AD-Loop), a new think paradigm that dynamically alternates between analytic and drafting operations. By interleaving textual thoughts with visual thoughts, AD-Loop enables models to iteratively refine both comprehension and outputs, fostering genuine synergy. To train this mechanism, we design a two-stage strategy: supervised learning on interleaved thought data to initialize alternation, followed by reinforcement learning to promote adaptive and autonomous control. Extensive experiments demonstrate that AD-Loop consistently improves performance across standard benchmarks for both understanding and generation, with strong transferability to various UVLMs architectures. Visual analyses further validate the effectiveness of implicit visual thoughts. These results highlight AD-Loop as a principled and broadly applicable strategy for synergizing comprehension and creation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a new thinking paradigm called the Alternating Analysis–Drafting Problem-Solving Loop (AD-Loop), designed to achieve genuine synergy between the understanding and generation capabilities in Unified Vision-Language Models (UVLMs). The AD-Loop enables the model to iteratively refine its comprehension and output by dynamically alternating between textual thoughts (for analysis and reasoning) and visual thoughts (for sketching and spatial layout).

### Strengths
The AD‑Loop paradigm is proposed, emphasizing explicit, dynamic, and reciprocal interaction between understanding and generation. This concept goes beyond mere architectural unification, offering a new approach for achieving deeper levels of general multimodal intelligence.

### Weaknesses
Existing MLLM research has begun to explore the alternation or integration of textual and visual information. This paper needs to more clearly define the fundamental differences between AD-Loop and such works — specifically, how AD-Loop achieves genuine synergy rather than merely alternation.

### Questions
Please elaborate on the specific composition and weight distribution of the internal reward (Intra-reward) and external reward (Inter-reward) in S2. In particular, how is the internal reward designed to encourage “intelligent and autonomous” alternating decision-making?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors propose a new method to improve the capabilities of unified vision-language models. They propose a new approach that combines analytic processes and drafting thoughts (textual and visual thoughts).  They design a two stage strategy for training consisting of supervised learning and RL.  They show performance improvements for benchmark tests for understanding and generation.

### Strengths
The authors address an important issue with what seems to be an innovative approach.

The comprehensive evaluation for understanding and generation is also a strength.

Ablation of the thinking types section was also a strength.

Good discussion addressing interesting questions spanning extensions into other MLLMs,  whether visual thoughts should be derived from understanding vs. the generation encoder, and the visualization of implicit visual thoughts.

Very thorough methods section with detailed descriptions.

### Weaknesses
I was unclear about how novel the work is. There are many publications using visual representations (imagination) to augment language reasoning. Please expand on how this is different.

Please show some examples in Figure 7 where the proposed method did not generate better results. This would also be interesting. 

There were parts where I had a difficult time following the methods and even the evaluation.  The paper is dense, but other reviewers more expert in the field might have an easier time following the paper.

### Questions
see weaknesses

### Soundness
4

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
3

### Summary
The paper introduces the Interleaved Analyzing–Drafting Loop (AD-Loop), a framework for unified vision-language models that alternates between textual analysis and visual drafting during reasoning. Through supervised fine-tuning on interleaved reasoning traces and reinforcement learning using Group-Relative Preference Optimization (GRPO), the model learns to decide when to analyze or draft. Applied to UVLMs (BAGEL-7B), the method improves both visual understanding and image generation, setting new state-of-the-art results on several benchmarks.

### Strengths
1. The interleaved analyzing–drafting mechanism establishes a tighter synergy between vision understanding and generation than prior “unified” models. This new paradigm addresses a clear gap by turning generation and analysis into mutually reinforcing steps rather than independent skills.

2. The two-stage training strategy (SFT of interleaved reasoning followed by RL) is well-motivated. This pipeline enables the model to learn the complex analyze-then-draft procedure in a guided way, then optimize adaptively via reinforcement learning. The use of preference-based rewards and an autonomous switching policy shows a way to balance the dual objectives.

3. AD-Loop yields consistent improvements across diverse benchmarks for both modalities. The approach outperforms baselines on multiple datasets and even when applied to different underlying model architectures, indicating broad applicability.

### Weaknesses
1. The framework introduces significant complexity in both training and inference. It requires a specialized two-phase training (including an RL stage), and at runtime the model must perform multiple analyze–draft iterations per query. This likely incurs substantial computational cost and latency compared to standard one-pass models. The paper does not discuss inference speed or resource requirements, which raises practical concerns for real-world deployment.

2. The method relies on a curated interleaved reasoning corpus for supervised training. It is not fully clear how generalizable these textual–visual “thought” sequences are. For example, if the SFT dataset is biased or lacks certain complex reasoning patterns, the RL stage will be unable to discover them.

### Questions
1. The qualitative examples focus on successes. Could you provide and analyze a few failure cases? Specifically, are there types of problems where the AD-Loop consistently fails or even degrades performance compared to simpler reasoning strategies?
2. Could you provide more detail on the heuristics used to "reorganize" and "synthesize" the AD-Loop traces from existing datasets (Sec 3.2)? How did you ensure the quality and logical coherence of these generated traces?
3. What is the typical computational overhead (e.g., latency, FLOPs) of invoking the AD-Loop for a complex query compared to a standard generation? How does the number of interleaved steps affect this cost?

### Soundness
3

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
Existing approaches primarily focus on architectural unification while overlooking the importance of explicit interaction between the two capabilities during task solving.
This paper introduces the interleaved Analyzing–Drafting problem-solving loop (AD-Loop), a new think paradigm that dynamically alternates between analytic and drafting operations.

### Strengths
- The paper introduces a two-stage training strategy: (1) supervised learning on interleaved thought data to initialize the alternation, followed by (2) reinforcement learning to promote adaptive and autonomous control.
- The paper conducts extensive experiments and ablation studies to validate the effectiveness of the proposed method.
- AD-Loop consistently improves performance across standard benchmarks for both understanding and generation, and further showed the adaptability of the proposed method on other unified VLM.

### Weaknesses
- The definition of the inter-group reward in Equation (6) is unclear. What does $m$ represent? Does it indicate whether the trajectory is AD-Loop-enabled or not? Additionally, in Equation (7), the intra- and inter-reward terms on the right-hand side seem to be missing the superscript $m$?
- In the ablation study (Table 3), it is unclear whether the paper trained three additional variants corresponding to different thinking strategies and evaluated them separately, or whether they used the final model but selectively disabled certain thinking capabilities.
- In Figure 4, what does unified-R1 refer to? Is it AD-Loop?
- Although Figure 5 presents examples of latent visual thoughts, it would be beneficial if the paper also included qualitative examples illustrating the complete thinking trajectories along with visualizations of the generated latent visual thoughts.

### Questions
- When constructing the dataset, did you control the number of visual drafts generated during the thinking process? What were your design choices and constraints here?
- In the final training loss (Equation 2), it appears that all loss components are simply summed together without weighting. Could you clarify whether any weighting scheme was considered or tested?

### Soundness
3

### Presentation
3

### Contribution
3
