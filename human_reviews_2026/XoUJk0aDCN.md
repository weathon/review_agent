# GRPO-CARE: Consistency-Aware Reinforcement Learning for Multimodal Reasoning

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2

## Abstract
Recent reinforcement learning (RL) approaches, such as outcome-supervised GRPO, have advanced Chain-of-Thought reasoning in large language models (LLMs), yet their adaptation to multimodal LLMs (MLLMs) remains underexplored. 
Progress has been further limited by the lack of evaluation settings that jointly test perception and reasoning under controlled generalization challenges. 
To enable such analysis, we reorganize prior benchmarks featuring complex real-world videos that demand intricate visual understanding and commonsense planning into **SEED-Bench-R1**, a structured testbed with large-scale training data and hierarchical evaluation across in-distribution, cross-environment, and cross-environment-task scenarios.
Using this setting, we conduct a systematic experimental analysis of post-training methods, which reveals a key limitation of outcome-supervised GRPO: while it improves answer accuracy, it often compromises the logical coherence between reasoning and final answers, yielding only a 57.9\% consistency rate. This stems from optimizing exclusively for final-answer rewards, which encourages shortcuts, and from rigid KL divergence penalties, which overly constrain adaptive reasoning.
To address these issues, we propose **GRPO-CARE**, a novel consistency-aware RL framework that jointly optimizes correctness and coherence without requiring explicit process supervision. GRPO-CARE introduces a two-tiered reward: (1) a base reward for accuracy, and (2) an adaptive consistency bonus derived from a slowly evolving reference model that calibrates reasoning-to-answer likelihoods within peer groups. This mechanism rewards reasoning paths that are both correct and logically consistent, while removing the constraints of KL penalties.
Experiments on SEED-Bench-R1 show that GRPO-CARE consistently outperforms standard GRPO, achieving a 6.7\% gain on the hardest evaluation level and a 24.5\% increase in reasoning consistency. Moreover, models trained with GRPO-CARE transfer effectively to diverse video understanding and even language-only reasoning benchmarks, highlighting its robustness and generality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes GRPO-CARE, a novel RL algorithm that improves MLLM's multimodal reasoning capabilities by enforcing consistency between the final answer and the generated reasoning trace. In addition, the paper introduces a new validation benchmark, SEED-Bench-R1, that focuses on video understanding that requires intricate visual perception and reasoning/planning capabilities. Using such a benchmark for validation, the paper thoroughly evaluates existing methods and their answer-reasoning trace consistency. Among the experiments, the GRPO-CARE exhibits improvement over other baseline methods on both general benchmark and SEED-Bench-R1.

### Strengths
- The paper approaches an important problem: how to improve the model's multimodal reasoning capability by designing better RL algorithms.
- The paper proposes a novel algorithm, GRPO-care, and a new validation benchmark, Seed-Bench-R1, which is beneficial to the general multimodal learning community. 
- The motivation is clear, and the considered approach is reasonable, with decent performance improvement over other baseline methods over the proposed Seed-Bench-R1 benchmark. 
- The listed and compared baselines are comprehensive, and the ablation study designs were also thoughtful.

### Weaknesses
- The performance improvement of GRPO-CARE on the general video understanding benchmark is not significant. Also, the other baselines, such as Video-R1-7B, are not evaluated on Seed-Bench-R1, which makes the reported results slightly less convincing.
- The idea of using a slowly updated EMA reference model for calibrating rollout reasoning trace log probability is interesting, but the paper lacks an in-depth investigation of why this approach works better than other options. A more systematic study on why this works would make the paper more impactful.
- GRPO CARE filtered out low-accuracy trajectories, which degrades its sample learning efficiency and prevents it from effectively learning using negative samples.

### Questions
Besides the points in the weakness section, I have the following additional questions:
- How necessary is the trajectory filtering based on the accuracy, and how does this operation affect the final accuracy? For challenging tasks where base models often achieve low accuracy, this could be detrimental. 
- What are the accuracy metric and consistency metric used in the experiments (their definition, how to compute them, etc)? Why is the accuracy metric not 0-1?
- How much success of GRPO-CARE can be attributed to the introduced consistency bonus rather than the filtered, high-quality rollouts used in the process?
- How does GRPO-CARE compare with other methods in terms of compute used for training a 7B model, e.g., Video-R1?

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
3

### Summary
This paper introduces **SEED-Bench-R1**, a hierarchical benchmark for evaluating post-training methods in multimodal large language models (MLLMs), and proposes **GRPO-CARE**, a consistency-aware reinforcement learning (RL) framework. SEED-Bench-R1 organizes real-world video understanding tasks into three generalization levels (in-distribution, cross-environment, cross-environment-task) to rigorously assess MLLM performance. GRPO-CARE addresses the limitations of outcome-supervised RL (e.g., logical inconsistency between reasoning and answers) by introducing a two-tiered reward system: a base reward for answer correctness and an adaptive consistency bonus derived from an EMA-updated reference model. Experiments demonstrate GRPO-CARE’s superiority over baseline methods, achieving improved accuracy and consistency across SEED-Bench-R1 and transfer tasks.  

Key contributions include:  

- A structured benchmark (SEED-Bench-R1) for evaluating MLLM post-training methods.  
- A novel RL framework (GRPO-CARE) that jointly optimizes answer correctness and reasoning consistency.  
- Empirical validation showing GRPO-CARE’s robustness in video understanding and language-only reasoning tasks.

### Strengths
**Originality**:  

- The hierarchical design of SEED-Bench-R1 addresses a critical gap in evaluating MLLM generalization across controlled OOD scenarios.  
- GRPO-CARE’s use of an EMA reference model for likelihood calibration and group-relative sparse rewards is a creative adaptation of existing RL principles to enforce logical consistency.  

**Quality**:  

- The ablation studies (Tables 3–5) rigorously validate GRPO-CARE’s components (e.g., EMA reference, two-stage filtering).  
- Transfer experiments (Tables 6–7) demonstrate the framework’s generalizability to diverse tasks, including language-only reasoning.  

**Clarity**:  

- The paper is well-structured, with clear explanations of SEED-Bench-R1’s design (Appendix A) and GRPO-CARE’s algorithmic details (Algorithm 1).  
- Visualizations (Figures 1–3) effectively illustrate the benchmark’s hierarchy and the framework’s reward mechanism.  

**Significance**:  

- The work provides actionable insights into the limitations of outcome-supervised RL for MLLMs (e.g., reasoning-answer inconsistency) and offers a practical solution.  
- SEED-Bench-R1 fills a need for standardized evaluation in multimodal reasoning, which could benefit future research.

### Weaknesses
**1. Limited Benchmark Scope**:  

- SEED-Bench-R1 focuses on video understanding but does not cover other critical multimodal domains (e.g., audio-visual tasks, interactive environments). While the authors mention future expansion, the current narrow scope limits the benchmark’s utility for broader MLLM research.  

**2. Reliance on GPT-4 for Consistency Evaluation**:  

- The consistency metric (Figure 5) depends on GPT-4.1 judgments, which may inherit biases or errors from the LLM. The paper does not validate whether GPT-4.1’s criteria align with human notions of logical coherence, raising concerns about circularity (using LLMs to evaluate LLM-based systems).  

**3. Computational Overhead**:  

- GRPO-CARE’s two-stage filtering and EMA reference model introduce significant computational costs (e.g., maintaining reference models, group-based calibration). The paper does not quantify these overheads or discuss scalability to larger models (e.g., 70B+ parameters).  

**4. Incomplete Comparison to Prior Work**:  

- While KL-oriented and reward-based baselines are compared (Table 3), the paper omits recent RL methods like WARP (Ramé et al., 2024) or process-supervised approaches (Luo et al., 2024). A broader comparison would better contextualize GRPO-CARE’s advancements.

### Questions
**Key Questions**:  

1. **Benchmark Generalizability**: Could SEED-Bench-R1 incorporate additional modalities (e.g., audio) or tasks (e.g., dialogue-based reasoning) to better reflect real-world multimodal challenges?  
2. **Consistency Evaluation**: Have you explored alternative consistency metrics (e.g., human evaluation, rule-based checks) to reduce reliance on GPT-4?  
3. **Computational Efficiency**: What steps could be taken to reduce GRPO-CARE’s computational overhead (e.g., parameter sharing, dynamic group sizing)?  
4. **Broader Baselines**: How does GRPO-CARE compare to WARP or process-supervised RL in terms of consistency and training stability?  

**Suggestions**:  

- Expand SEED-Bench-R1 to include non-video tasks (e.g., image-text reasoning) to enhance its utility.  
- Conduct a sensitivity analysis of GPT-4.1’s consistency judgments against human annotations.  
- Provide computational cost metrics (e.g., FLOPs, training time) for GRPO-CARE versus baselines.  
- Compare GRPO-CARE to process-supervised methods (e.g., AlphaMath, Lightman et al., 2023) to highlight trade-offs between annotation cost and performance.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper 
1. introduces SEED-Bench-R1, a structured benchmark for evaluating post-training
methods for MLLM 
2. proposes a RL framework: GRPO-CARE for MLLMs to jointly optimizes for answer's correctness and logical consistency. 
3.  demonstrates  that GRPO-CARE consistently outperforms standard GRPO for experiments on SEED-Bench-R1

### Strengths
The paper identifies an important direction of improving multimodal reasoning capabilities through consistency reward. However the paper could be strengthened by more experiments and analysis as mentioned below.

### Weaknesses
1. The paper could be strengthened by providing analysis on how the weighting between correctness reward and consistency reward could affect model performance
2. The vast majority of the paper's experiments, analyses, and conclusions are based on a single model architecture : Qwen2.5-VL. The effectiveness of the proposed method could be further tested on other model architectures and scales.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
2
