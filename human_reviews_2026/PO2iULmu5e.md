# RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format

- Decision: Accept (Oral)
- Scores: 8, 4, 8, 6

## Abstract
Large reasoning models (LRMs) excel at a long chain of reasoning but often fail to faithfully follow instructions regarding output format, constraints, or specific requirements. We investigate whether this gap can be closed by integrating an instruction-tuned model (ITM) into an LRM. Analyzing their differences in parameter space, namely task vectors, we find that their principal subspaces are nearly orthogonal across key modules, suggesting a lightweight merging with minimal interference. However, we also demonstrate that naïve merges are fragile because they overlook the output format mismatch between LRMs (with explicit *thinking* and *response* segments) and ITMs (answers-only). We introduce **RAIN-Merging** (Reasoning-Aware Instruction-attention guided Null-space projection Merging), a gradient-free method that integrates instruction following while preserving thinking format and reasoning performance. First, with a small reasoning calibration set, we project the ITM task vector onto the null space of forward features at thinking special tokens, which preserves the LRM's structured reasoning mechanisms. Second, using a small instruction calibration set, we estimate instruction attention to derive module-specific scaling that amplifies instruction-relevant components and suppresses leakage. Across four instruction-following benchmarks and nine reasoning & general capability benchmarks, RAIN-Merging substantially improves instruction adherence while maintaining reasoning quality. The gains are consistent across model scales and architectures, translating to improved performance in agent settings.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper uses task vectors to alleviate the issue of reasoning models where they are struggling with instruction following. Authors proposed a gradient-free method to address this problem, called Reasoning-Aware Instruction-attention guided Null-space projection Merging (RAIN-Merging). At the first stage, ITM task vector is projected on the null space of forward features at thinking tokens, and at the next stage, an instruction attention is estimated to further amplify instruction-relevant components.

### Strengths
1. The paper is well-written, and figures and tables help better understand the complicated method that authors proposed.  
2. Reasoning models are bad at instruction-following is a well-known phenomenon, and the proposed method uses simple task vectors to boost the instruction-following ability while maintaining the original reasoning performance.  
3. The results are reported in multi-dimensions according to the research questions in Section 4. Specifically, reporting a performance in agentic scenarios is a very good experimental evidence that the proposed method works well in the real world.  
4. The proposed method is theoretically well-grounded.

### Weaknesses
1. In Table 1, there are some cases where RAIN-Merging outperforms the performance of the original LRM. Authors hypothesize that stronger instruction adherence improves CoT quality. -- I suggest to prove this hypothesis (by manually checking randomly sampled predictions). This phenomenon seems very interesting to me since my intuition says the opposite. Specifically, the performance is increased by ~10% in GPQA. How is it possible?  
2. In the ablation study, only stage 2 is ablated. Therefore, the paragraph's name should be scoped down. Also, could you ablate stage 1 as well to prove its effectiveness?

Despite some of these questions, I believe this paper would contribute to the community significantly.

### Questions
See Weaknesses.

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
This paper introduces RAIN-Merging, a gradient-free method for enhancing instruction following in Large Reasoning Models while preserving their structured thinking format. The approach projects instruction task vectors onto the null space of thinking token features, then applies instruction-attention guided scaling coefficients. The method demonstrates improvements across instruction-following and reasoning benchmarks without requiring gradient-based training.

### Strengths
1. Merging ITM and LRM is an interesting and practical problem.
2. The finding that task vectors' principal subspaces are nearly orthogonal across key modules provides an interesting understanding about the parameter space structure of these capabilities.
3. The evaluation spans multiple model families and sizes.
4. The gradient-free nature makes this a practical, accessible alternative to SFT.
5. Using four instruction-following benchmarks and multiple reasoning datasets provides reasonable empirical breadth.

### Weaknesses
### 1. Data Contamination / Generalization Concerns
For example, Qwen2.5-7B-Instruct is trained on IFEval, InfoBench, and ComplexBench as calibration data, and this paper evaluates RAIN-Merging on the same benchmarks. Results may not generalize to unseen instruction-following or reasoning scenarios. Maybe the null-space projection and coefficients are optimized on the same distribution they're tested on.

### 2. Data
The paper evaluates instruction-following and reasoning on separate benchmark datasets. While the paper attempts to evaluate integrated capabilities using agentic scenarios in Table 3, these tasks may not simultaneously stress complex reasoning and strict, arbitrary instruction-following to the same degree as the benchmarks. The main evaluation in Table 1 still separates these two skills, leaving a gap in the core claim.

### 3. Metric
The paper relies primarily on accuracy metrics across all benchmarks. More metrics would be valuable to answer such questions: Is the thinking process coherent, or just structurally preserved (only thinking tokens)? Which types of instructions improve most/least? Are outputs semantically following instructions, or just superficially (evaluation on phrased instructions)?
 
### 4. Method
In stage 1, how do we know null-space projection preserves reasoning ability rather than just token usage? Does the model still perform meaningful reasoning in <think> blocks, or just maintain the format while reasoning quality degrades? If the reasoning benchmark results show preservation, but is this because the reasoning content is truly preserved, or the model learned to use thinking tokens without meaningful reasoning?

In stage 2, what happens when reasoning and instruction-following are highly entangled? How are attention-guided coefficients computed when both reasoning and instruction-following are active in the same tokens?

### Questions
* What is your hypothesis for why reasoning ability and instruction-following have low coupling in parameter space? Is this due to structural differences (thinking tokens vs. output format) rather than semantic content differences?
* How can you verify that Stage 1 preserves actual reasoning ability (content) and not just the habit of using thinking tokens? What metrics or analyses distinguish these scenarios?
* In Stage 2, how do attention-guided coefficients handle tokens where reasoning and instruction-following are deeply entangled (e.g., instructions about reasoning strategy, logical constraints, or structured argumentation)? Can you provide examples and analysis?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents RAIN-Merging, a novel two-stage, gradient-free method for merging Large Reasoning Models (LRMs) with Instruction-Tuned Models (ITMs) to enhance instruction-following capability while preserving structured reasoning outputs. The approach leverages task-vector orthogonality and introduces a reasoning-aware null-space projection to maintain thinking formats, combined with instruction-attention guided coefficients to improve instruction adherence. Extensive experiments across multiple benchmarks and model scales demonstrate that RAIN-Merging significantly improves instruction-following performance without compromising reasoning or general capabilities. The method offers a computationally efficient and interpretable alternative to supervised fine-tuning, making it highly relevant for real-world applications of LRMs.

### Strengths
1.   Novel Research Problem: The work addresses an important and underexplored challenge—balancing instruction-following and reasoning capabilities in LRMs—through model merging, a lightweight and training-free approach.
2.  Effective Methodology: The two-stage RAIN-Merging framework is well-motivated, combining null-space projection to preserve reasoning structure with attention-guided scaling to enhance instruction alignment, all without gradient updates.
3.   Comprehensive Experiments: The paper provides extensive evaluations across multiple instruction-following, reasoning, and agentic benchmarks, with consistent improvements shown across different model sizes and architectures.
4.  Clear and Well-Structured Writing: The paper is clearly written, with a logical flow, detailed derivations, and accessible explanations of both the motivation and technical contributions.

### Weaknesses
While the method is evaluated on several model families (Qwen, Llama), further validation on a wider range of architectures and modalities (e.g., multimodal or multilingual models) would strengthen the generalizability claims.

### Questions
1.  Overall, I think you have done very meaningful work. My question is: Will your work be open-sourced as a toolkit in the future? I am very much looking forward to using the methods proposed in your paper.
2.  Have you considered validating your method on larger model sizes (e.g., above 30B parameters) to further verify its effectiveness?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes RAIN-Merging, a training-free (gradient-free) technique to improve instruction following in Large Reasoning Models (LRMs) without degrading their explicit “thinking→answer” format. The method has two stages: (1) Reasoning-aware null-space projection that projects the instruction-tuned model’s task vector into the null space of forward features at special thinking tokens (e.g., <think> … </think>), thereby preserving the LRM’s thinking distribution (formalized via a KL constraint). (2) Instruction-attention guided scaling, which uses forward attention statistics from a small instruction calibration set to compute per-module coefficients that increase attention alignment to instruction spans while penalizing leakage to unrelated spans. Experiments across Qwen/Llama backbones and sizes show instruction-following gains with maintained or improved reasoning and lower compute than SFT; ablations support both stages’ roles and show the method keeps the <think> format intact (no missing terminator).

### Strengths
Clear problem & neat insight. The paper pinpoints a real pain point: LRMs reason well, but violate format/constraints. The idea to protect the thinking segment explicitly while injecting instruction-following behavior is crisp and well-motivated.

Strong empirical results. On the headline 7B setting, RAIN-Merging improves instruction-following average (48.11 vs. 44.12 LRM; +4 points absolute) while also improving reasoning/general (55.59 vs. 51.03) and beating task-arithmetic, SLERP, Karcher, TIES, DARE-TIES, and activation-based methods (AIM/ACM/LEWIS combined with TIES) 

Efficiency. Minutes to merge (20.96 min reported) vs. SFT’s 120+ min; GPU memory also far below SFT (22.1 GB vs. 112.6 GB in their config)

### Weaknesses
Calibration-set specificity. The instruction calibration set is distilled from IFEval-style instructions (365 samples). This may bias the proxy to rule-verifiable patterns and possibly underrepresent open-ended or tool-use instructions.

Reliance on explicit thinking markers. Stage 1 presumes accessible special tokens and feature extraction around them. It is unclear how well this transfers to LRMs with different templates (or hidden/implicit thinking) or to models without consistent <think> tags, using ReAct format for thinking tool use.

### Questions
How does Stage 1 handle LRMs whose “thinking” is not demarcated by explicit tokens, or that interleave tool calls and thoughts (e.g., ReAct-style)? 

Generalization of the proxy. If the calibration set used open-ended instructions (no machine-checkable rules), would Stage 2 still pick effective heads? Any results using other instruction corpora?

### Soundness
3

### Presentation
2

### Contribution
3
