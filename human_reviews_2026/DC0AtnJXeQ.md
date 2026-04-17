# Plan-and-Paint: Collaborating Semantic and Noise Reasoning for Text-to-Image Generation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Despite the transformative success of chain-of-thought (CoT) and reinforcement learning (RL) in large language models, their application to visual generation—where reasoning is a critical challenge—remains largely unexplored. In this paper, we present \textbf{Plan-and-Paint}, a novel framework that integrates a dual-level reasoning hierarchy for text-to-image generation. Our framework operates at two critical stages: (1) at the semantic level, an adaptive planner first decomposes the input prompt into a structured generation plan, and (2) at the foundational level, a reinforcement learning agent optimizes the initial noise prior to align with this plan. To seamlessly coordinate these two stages, we introduce a unified reinforcement learning paradigm GRPO to jointly optimizes both the planning coherence and the execution fidelity through a composite reward function. Extensive experiments demonstrate the superiority of our approach: Plan-and-Paint achieves significant improvements on both GenEval (0.87→0.90) and WISE benchmarks. Most importantly, on GenEval benchmark, our method secures the top rank, outperforming a wide range of top-tier open-source and closed-source competitors, including GPT-Image-1 High, Janus-Pro-7B, Qwen-Image, BAGEL, and Seedream 3.0 by a significant margin. Our work advances the state-of-the-art in text-to-image generation, proving that an explicit reasoning hierarchy is key to unlocking controllable and compositional text-to-image generation. To facilitate future research, we will make our code and pre-trained models publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Plan-and-Paint, a text-to-image framework that addresses the drawbacks of blind initial noise, rigid semantic reasoning, and difficult RL adaptation in conventional models. Built upon the GRPO reinforcement-learning algorithm, it adopts a two-level reasoning design: the high-level stage employs an ALP-CoT two-step self-query mechanism to dynamically adjust the length of semantic chains for accurate prompt decomposition, while the low-level stage refines the initial noise prior via noise reasoning to align it with semantic demands. A multi-dimensional reward ensemble—integrating human preference, compositional correctness, and attribute–subject fidelity—prevents reward hacking.

### Strengths
* To address the core issue of "disconnection between semantic planning and noise generation" in text-to-image generation, a two-level architecture of "high-level semantic adaptive planning (ALP-CoT) + low-level noise enhancement optimization" was proposed. This architecture simulates the human cognitive logic of "planning first, then creating" and achieves the coordinated optimization of semantic reasoning and noise generation.

* During RL training, it integrated three types of expert models: human preference, combination correctness, and attribute themes, to construct a comprehensive evaluation system. It also implemented "group relative advantage estimation" based on the GRPO algorithm, eliminating the impact of absolute reward fluctuations while balancing training stability and generation quality. This effectively addressed the difficulty in balancing multi-dimensional requirements in visual generation.

### Weaknesses
* This method relies on the accuracy of MLLM's prompt decomposition. For highly abstract or vague instructions (such as "artistic expression of futuristic cities" and "emotional visualization in abstract style"), semantic decomposition deviations are prone to occur, resulting in generated results that do not meet user expectations.

* The ablation experiment only compares the Qwen-Image and does not consider the effects of methods that have their own reasoning capabilities, such as BAGEL.

* The paper only shows the final experimental results, and does not provide performance fluctuation curves during training (such as loss function changes, subtask score trends), making it impossible to judge the impact of the GRPO algorithm in long-term training. In addition, there is a lack of visualization results during the inference process, making it impossible to see the changes in the generated results during the COT process.

### Questions
* Although the method proposes to "optimize the initial noise to fit the semantics", it fails to clarify the "association mechanism between noise and semantics" through visualization or quantitative analysis. For example, it is impossible to intuitively demonstrate how the noise prior of "Roman Café" contains "Roman architectural features", nor is it possible to quantify the improvement in the efficiency of subsequent denoising steps by noise optimization (such as how much the number of denoising steps is reduced). As a result, the logical chain of underlying noise reasoning is incomplete.

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
The paper introduces Plan-and-Paint, a dual-level reasoning framework for text-to-image generation that jointly optimizes semantic planning and noise-level reasoning. It aims to bridge the gap between structured reasoning (as in LLMs) and visual synthesis (as in diffusion or flow models).

### Strengths
1. The dual-level “semantic plan + noise execution” setup is somehow intuitively mimics human creative reasoning.
2. The ALP-CoT self-querying design is lightweight—no external regressors or heuristic tuning. It addresses real weaknesses of fixed-length CoT in image generation.
3. Comprehensive benchmarks (GenEval, WISE) with both quantitative and qualitative analyses.

### Weaknesses
1. Relies heavily on existing components (GRPO, Qwen-Image, HPSv2, GroundingDINO, GIT). The core novelty lies in their integration rather than algorithmic innovation.
2. The composite reward depends on third-party pretrained models, introducing possible biases and fragility. Moreover, they are commonly used in existing methods.
3. Benchmarks focus on alignment and compositionality; however, perceptual diversity, artistic control, or realism metrics are less explored. No user study to validate human preference improvements claimed via HPSv2 reward.
4. The claim that it “mirrors human creative planning and execution” is more rhetorical than empirically supported.

### Questions
see the weaknesses.

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
4

### Summary
The paper proposes a training-free, inference-time pipeline for compositional T2I. It first uses an LLM to produce a 2.5D semantic layout (object captions, boxes, relative depth), then generates each object independently, segments it with a referring-segmentation model, and composites a composite object prior (foreground layers + background). Then, it injects this prior via object-prior reinforcement and spatial-controlled self-attention for the first few steps, and reverts to standard diffusion. The method improves compositional metrics on T2I-CompBench and NSR-1k and shows human-preference wins over compared inference-time baselines.

### Strengths
+Interpretable control: The object-prior + early-step spatial control design is clear and editable, and it fits into multiple backbones (SDXL/SD3/FLUX) with consistent gains, as shown in Table 3.

+Training free: the method is training free and does not require additional modules for compositional generation.

### Weaknesses
-Efficiency scaling with object count K:
The pipeline generates and segments each object independently (Sec. 3.2)—cost scales with K (the number of objects). Table 6 reports averaged runtimes (prompts with 3/5/7/10 objects averaged together) and mentions multi-GPU parallelization, but does not plot runtime vs. K nor compare scaling curves against baselines. A time-vs-K analysis (with/without parallelism) and per-backbone breakdown would be more informative. 

-Per-object synthesis can introduce unintended instances.
Because each object image is synthesized once per class caption and then segmented via Hyperseg (referring segmentation), it’s plausible that extra same-class instances or background leakage in the per-object generation could be included by the mask—The risk  grows with K. The paper doesn’t analyze such failure modes or report instance-purity metrics. 

-SDEdit-style re-init may degrade fine-grained alignment.
The method initializes from a noisy prior at $t_p$ (SDEdit idea) and applies guidance only in early steps before switching to standard diffusion. Table 1 shows 3D-Spatial drops from OP→FI (86.71→77.16), suggesting that reverting to base denoising can weaken certain compositional cues even as realism improves elsewhere. In addition, the paper does not quantify prompt alignment retention of the full, fine-grained compositional caption across this transition.

### Questions
Please see the weakness above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Plan-and-Paint, a dual-level reasoning framework for text-to-image generation that combines high-level semantic planning with low-level noise-space optimization. The method uses an Adaptive Length Prediction Chain-of-Thought (ALP-CoT) mechanism to dynamically adjust textual reasoning depth and applies reinforcement learning (GRPO) to refine the initial noise prior.

### Strengths
1. The idea of combining semantic-level planning with noise-level reasoning is well-motivated, drawing inspiration from human “plan-and-execute” creativity.

2. The paper includes both quantitative and qualitative analyses.

3. The authors perform systematic ablations to validate the contributions of ALP-CoT and noise-level reasoning.

4. The use of multiple vision-language experts for reward design (HPS, detection, VQA) is thorough and helps prevent reward hacking.

### Weaknesses
1. The process where the SemanticLengthPredictor module instructs the MLLM to analyze the task and predict a reasoning length appears tedious. Intuitively, for a powerful large language model, the length of reasoning should not significantly impact reasoning correctness. Could you provide more examples demonstrating how randomly assigned reasoning lengths negatively affect reasoning accuracy?

2. The experimental setup may overstate the model's core advancement. Reinforcement learning is applied to a powerful base model (Qwen-Image) using a training set (from T2I-R1) rich in compositional prompts, which are highly aligned with the object-centric, compositional tasks in the GenEval benchmark. Consequently, the observed improvement on GenEval is somewhat expected and may not sufficiently demonstrate a significant improvement. This concern is compounded by the minimal performance gain on the WISE benchmark (+0.01). The saturated nature of GenEval scores further obscures whether the marginal improvement signifies a fundamental advance or merely an incremental optimization tailored to a specific benchmark.  It would be helpful if the author can demonstrate that the model can generalize to more diverse prompts not seen during training.

### Questions
please see weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
