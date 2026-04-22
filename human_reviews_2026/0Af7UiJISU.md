# THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Large Language Models (LLMs) have made remarkable progress in mathematical reasoning, but still continue to struggle with high-precision tasks like numerical computation and formal symbolic manipulation. Integrating external tools has emerged as a promising approach to bridge this gap. Despite recent advances, existing methods struggle with three key challenges: constructing tool-integrated reasoning data, performing fine-grained optimization, and enhancing inference. To overcome these limitations, we propose THOR (Tool-Integrated Hierarchical Optimization via RL). First, we introduce TIRGen, a multi-agent based pipeline for constructing high-quality datasets of tool-integrated reasoning paths, aligning with the policy and generalizing well across diverse models. Second, to perform fine-grained hierarchical optimization, we introduce an RL strategy that jointly optimizes for both episode-level problem solving and step-level code generation. This is motivated by our key insight that the success of an intermediate tool call is a strong predictor of the final answer's correctness. Finally, THOR incorporates a self-correction mechanism that leverages immediate tool feedback to dynamically revise erroneous reasoning paths during inference. Our approach demonstrates strong generalization across diverse models, performing effectively in both reasoning and non-reasoning models. It further achieves state-of-the-art performance for models of a similar scale on multiple mathematical benchmarks, while also delivering consistent improvements on code benchmarks. Our code will be publicly available at https://github.com/JingMog/THOR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
THOR is a technically competent and empirically thorough paper that explores hierarchical optimization for tool-integrated reasoning—a topic of rising importance in post-RLHF LLM training. The paper’s main contributions, especially the combination of TIR data generation (TIRGen) and dual-level GRPO optimization, are sound and moderately novel. The experiments are comprehensive, covering both reasoning and code tasks, and the observed gains are consistent.

However, while the framework is methodically interesting, it does not radically depart from prior lines such as ToRL (Li et al., 2025b) or (Lin et al., 2025). The hierarchical decomposition is intuitively motivated but somewhat incremental relative to the existing literature, e.g., GiGPO (Feng et al., 2025) , and several details are under-specified.

### Strengths
• Clear Problem Motivation and Scope: The paper is grounded in well-known issues of reasoning—error propagation in long CoT traces and failures in symbolic computation—and formulates these in the RL framework (Sec. 2.1). The empirical correlation between code success and final correctness offers solid motivation for dual-level credit assignment.

• Hierarchical RL Formulation: The two-stage optimization (Eq. 4–9) represents a practical way to address sparse reward signals. Step-level RL on failed tool invocations is technically sensible and extends GRPO with fine-grained local feedback, offering a pragmatic, if not groundbreaking, advance.

### Weaknesses
• Data Generation Pipeline (TIRGen): The actor–critic construction (Alg. 1, Fig. 2) potentially yields a more noisy dataset than rule-based data due to the uncertainty introduced by LLMs. Althought multi-stage filtering avoids trivial tool use errors, I think human-in-the-loop verification and double-check is needed.

• Reproducibility and Transparency: Appendix D only offers implementation details, including batch sizes, optimizer settings, and prompt templates. But data and code are critical to support reproducibility and checking.

• Missing Analysis of Failure Modes: The paper does not report how often self-correction succeeds or fails, nor how many inference steps are affected by code errors or sandbox limitations (Sec. 2.4).

• Restricted Generalization Domain: All evaluations focus on mathematical or code generation. No evidence is given that THOR generalizes to broader tool environments (e.g., multi-turn search and agentic tasks).

### Questions
How to compute advantages at the step-level in Eq. 8? I think each sample contains a single think-act-observe loop, and the reward is computed from execution correctness. Do you mean the advantage is estimated by sampling a single response?

Moreover, does this method cause efficiency issue due the correctness evaluation and regeneration?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the poor computational and symbolic reasoning of LLMs in math problems. It introduces THOR, a framework built on three key parts. 1) TIRGen: A data pipeline described as a multi-agent "actor-critic" system, for generating high-quality examples of tool-integrated reasoning. 2) Hierarchical RL: A two-level optimization strategy. It combines a sparse, "episode-level" reward for the final answer's correctness with a denser, "step-level" reward for the success of intermediate code generation. 3) Self-Correction: An inference-time mechanism that lets the model backtrack and revise its reasoning path when a tool call fails.

### Strengths
1. Novel and Effective RL Framework: The paper's core strength lies in its hierarchical reinforcement learning strategy, which decisively tackles the sparse-reward problem in complex reasoning by introducing a denser, intermediate reward signal (code execution success), all while maintaining the focus on the primary objective (final answer correctness). This approach is both elegant and purposeful.
2.  Strong Empirical Performance: The paper achieves state-of-the-art results on several challenging math benchmarks (Tab 1). These gains, particularly against the non-reasoning model baseline, underscore the system's high efficacy.
3. Comprehensive Ablation Study: The ablation analysis in Table 3 systematically exposes each component's significant and additive impact—TIRGen dataset (T3 vs T1), hierarchical RL (T5 vs T3), and self-correction (T6 vs T5). This compellingly supports the paper's claims.

### Weaknesses
1.  Could the authors please clarify the choice of the term "Actor-Critic" for TIRGen? Since the "Critic" model converts reasoning steps into code, I am curious whether there are reasons to favor this RL-inspired terminology over alternatives such as "Generator-Refiner" or "Proposer-Converter."

2.   In the context of model evaluation, the Appendix indicates that the "Critic" is a 32B model, whereas the "Actor" is an 8B or 1.5B model. Could the authors comment on the importance of this size difference? Additionally, how might the quality of the TIRGen dataset ($\mathcal{D}_{SFT}$) be affected if the Critic model were the same size as (or smaller than) the Actor model?

3.  The self-correction mechanism (Section 2.4) is triggered when code execution fails. I am interested in how the framework deals with logically incorrect code that still executes successfully (e.g., producing a number from the wrong equation). Would an episode-level RL objective suffice to address this, or is an additional mechanism required?

### Questions
See weakness.

### Soundness
2

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
This paper proposes a tool-integrated reasoning framework with three components: (i) a multi-agent pipeline that builds tool-integrated reasoning (TIR) trajectories for cold-start SFT; (ii) hierarchical RL that jointly optimizes episode-level answer accuracy and step-level code-pass rewards; and (iii) an inference-time self-correction mechanism that backtracks to the last failed tool-use step.

### Strengths
1. The writing is clear.
2. The method shows gains over recent TIR/CoT baselines.

### Weaknesses
1. The method assumes a failed execution implies both the action and its preceding reasoning step are wrong, and always backtracks/regenerates the suffix. This coupling can be brittle when failures arise from syntax/runtime issues; action-only repair may suffice.
2. Optimizing only code-pass is not a promising step for fine-grained TIR reward shaping; analyses of code-trigger timing, necessity/helpfulness, and over/under-calling would be more informative.
3. The self-correction trigger is hand-coded, whereas prior work such as ReTool [2] reports emergent self-correction behavior during training.
4. TIR data curation and RL for tool use have seen substantial prior exploration[1,2,3], while the experiments and design here are solid, the technical novelty appears incremental relative to recent work.

[1] ToRL: Scaling Tool-Integrated RL

[2] ReTool: Reinforcement Learning for Strategic Tool Use in LLMs

[3] ToolRL: Reward is All Tool Learning Needs

### Questions
1. How often do failures stem from action errors versus reasoning errors? Please report an ablation: (a) action-only repair, (b) step-suffix repair, (c) full re-plan.
2. In Table 2, how does your self-rewarded test-time selection compare with BoN/self-consistency and PRM/verifier-based reranking under the same (N), temperature, and compute?
3. Does optimizing code-pass reward improve code-generation ability itself, or only downstream answer accuracy?
4. Is the training data comparable to baselines (tokens, steps, RL updates)?
5. Could you add more TIR baselines on the same backbone and training setup to isolate method effects?

### Soundness
2

### Presentation
2

### Contribution
2
