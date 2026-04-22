# Plan Then Action: High-Level Planning Guidance Reinforcement Learning  for LLM Reasoning

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 8

## Abstract
Large language models (LLMs) have demonstrated remarkable reasoning abilities in complex tasks, often relying on Chain-of-Thought (CoT) reasoning. However, due to their autoregressive token-level generation, the reasoning process is largely constrained to local decision-making and lacks global planning. This limitation frequently results in redundant, incoherent, or inaccurate reasoning, which significantly degrades overall performance. Existing approaches such as tree-based algorithms and reinforcement learning (RL) attempt to address this issue but suffer from high computational costs and often fail to produce optimal reasoning trajectories.
To tackle this challenge, we propose Plan-Then-Action Enhanced Reasoning withGroup Relative Policy Optimization (PTA-GRPO), a two-stage framework designed to improve both high-level planning and fine-grained CoT reasoning. In the first stage, we leverage advanced LLMs to distill CoT into compact high-level guidance, which is then used for supervised fine-tuning (SFT). In the second stage, we introduce a guidance-aware RL method that jointly optimizes the final output and the quality of high-level guidance, thereby enhancing reasoning effectiveness.
We conduct extensive experiments on multiple mathematical reasoning benchmarks, including MATH, AIME2024, AIME2025, and AMC, across diverse base models such as Qwen2.5-7B-Instruct, Qwen3-8B, Qwen3-14B, and LLaMA3.2-3B. Experimental results demonstrate that PTA-GRPO consistently achieves stable and significant improvements across different models and tasks, validating its effectiveness and generalization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
PTA-GRPO is a two-stage “plan-then-act” recipe for LLM reasoning. First, a strong teacher LLM distills each solution’s chain-of-thought into a short, high-level plan and the student is SFT’d to emit <plan><think><answer>; second, GRPO is modified to reward (i) final correctness, (ii) plan quality (measured by how well CoT sampled under a plan yields correct answers), and (iii) output format/conciseness. On math benchmarks (AIME-24/25, MATH500, AMC) and across Qwen/LLaMA backbones, this yields consistent but marginal gains over plain GRPO/DAPO while siginificantly require more compute for the plan reward calculation

### Strengths
1- Well written, and clear practical recipe: add an explicit planning phase (teacher-distilled plans, <plan><think><answer>) then optimize with plan-aware RL—easy to port to other backbones/tasks.

2- Consistent even though marginal gains across multiple math benchmarks
3- this theme aligns with evidence from many recent works showing that adding a planning stage boosts reasoning, whether in SFT (e.g., planning-distilled data, step/plan-level preference learning like Step-DPO/Full-Step-DPO) or in RL.

### Weaknesses
1- Limited baseline comparisons. The paper only compares against GRPO and DAPO. By 2024–2025 several other RL methods offer improved credit assignment or planning (CPL, TS‑LLM, Step‑/Full‑Step‑DPO, ...). Omitting these makes it difficult to gauge PTA‑GRPO’s relative advantage. For instance, CPL uses MCTS to explore plan steps and Step‑APO to learn plan step preferences, which seems directly comparable.  


2- Lack of analysis on generalization. Experiments focus on mathematical reasoning; there is no evidence that high‑level guidance improves coding or general knowledge tasks. while math datasets are good proxy for assessing general reasoning capabilities of LLMs, overfitting on these datasets while incuring significant cost, needs better justification

3. Computational cost. The analytic plan reward requires sampling multiple CoTs per plan to estimate plan quality. Although this may be cheaper than tree search, the paper does not quantify the overhead relative to GRPO or DAPO. RL with plan reward might still be expensive for large models.


references:
CPL - https://arxiv.org/pdf/2409.08642
TS-LLM : Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training.
multi-step dpo: https://arxiv.org/pdf/2502.14356v1

### Questions
1- Computational efficiency. How does the runtime and sample complexity of PTA‑GRPO compare with GRPO, DAPO, and MCTS‑based methods like CPL? What is the overhead of resampling CoTs for each plan?

2- Does figure 3. support  argument in 4.5 ? it seems to show the opposite

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
This paper is another in the line of "how do I teach LLMs to plan" research. The good news is that the authors acknowledge that simple next-token prediction does not do planning so they encourage the LLM to first sketch a broad plan and then aim to flesh it out with detailed steps using CoT and supported by global context.

### Strengths
- The proposed architecture is interesting: an initial supervised fine-tuning phase where an advanced LLM extracts core reasoning paths from CoT traces and distills these into concise, high-level plans, and a subsequent reinforcement learning stage using a suitably designed reward system.
- The reward system is also interesting, giving greater weightage to high quality plan guidance.
- The ablation results are good and are aimed at teasing out the importance of the two stages relative to each other.

### Weaknesses
- While the paper is good and has sound ideas, the experimental results do not show any major improvement over the strong models (and in fact the paper does not use any of the latest generation of models)
- The quality of analytic plans will be heavily dependent on the strength of the LLM used for plan generation; weaker models may actually degrade performance when generating their own plans. In one sense this is to be expected but then the main contribution then remains unproven (i.e., whether you can solve an open-ended task that does not have an analytical plan known to the stronger LLM).

### Questions
- Please address the questions raised in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the lack of global planning in LLM chain-of-thought (CoT) reasoning, which often leads to redundant or incoherent reasoning chains. The authors propose **PTA-GRPO (Plan-Then-Action Enhanced Reasoning with Group Relative Policy Optimization)**, a two-stage framework that teaches LLMs to generate high-level analytical plans before detailed reasoning.

The approach consists of two stages: (1) **Planning Structured Reasoning Cold-Start (PSR-CS)**, where a strong teacher model (Qwen3-235B) distills detailed CoT into concise analytical plans, which are used for supervised fine-tuning alongside the original CoT; and (2) **Plan Structure-Guided RL (PSG-RL)**, which extends GRPO with a composite reward function that evaluates not only final answer correctness ($r_{outcome}$) but also plan quality ($r_{analytic}$) and output format ($r_{format}$). Plan quality is measured by sampling multiple CoT trajectories per plan and computing empirical accuracy, thereby rewarding plans that consistently guide correct reasoning.

**Main contributions** include: (1) a two-stage training framework combining plan distillation with plan-aware reinforcement learning, (2) a novel reward mechanism that evaluates analytical plan quality through multi-sampling (generating $m$ plans, then $z$ CoTs per plan)

### Strengths
**Originality:** The paper addresses a legitimate gap—that standard RLVR methods reward only final answers while ignoring reasoning quality. Evaluating plan quality through multi-sampling (generating multiple CoTs per plan and measuring empirical accuracy) is a reasonable idea, and the structured format aids interpretability. 

**Quality:** The experimental design spans multiple model scales (3B-14B) and benchmarks, with ablations and data scaling analysis. Training procedures are documented with hyperparameters in the appendix. 

**Clarity:** The paper is well-written with clear motivation. Figure 1 and 2 effectively illustrates the problem and solution. The two-stage framework is logically presented. 

**Significance:** The problem is relevant. Explicit plans improve interpretability and the method shows consistent improvements across model scales.

### Weaknesses
**Absence of Statistical Validation:** This is the most critical flaw. The paper reports point estimates without standard deviations, confidence intervals, or significance tests, despite claiming "16 independent runs." With improvements of only 1-3 points in most cases, **these differences could easily be random noise**. For example, Table 1 shows PTA-GRPO achieving 93.31 vs DAPO's 91.27 on MATH500 (Qwen3-8B)—a 2-point difference—but without error bars, this could be statistically insignificant. For Qwen3-14B, the average improvement is merely 0.71 points (83.09 vs 82.38), essentially within noise. Standard practice in RL and LLM research requires reporting mean +/- std and conducting significance tests (t-tests, bootstrap). 

**Unjustified Computational Cost:** The method samples m=3 plans, then z=3 CoTs per plan, requiring **9 forward passes per question** compared to ~3-4 for standard GRPO—roughly **3× computational overhead**. Yet improvements are marginal (1-3 points), and for strong models like Qwen3-14B, gains are negligible. The paper never analyzes this cost-benefit tradeoff or compares wall-clock training time. A fairer baseline would give GRPO the same computational budget (9 samples or 9x prompts) and compare performance. It's possible GRPO with more samples would match or exceed PTA-GRPO's performance. 

**Limited Novelty and Weak Theoretical Contribution:** The core method is GRPO + auxiliary reward for plan quality, which is a straightforward extension. Plan-then-execute frameworks are well-established (Wei et al., 2022 on CoT; Yao et al., 2023 on Tree-of-Thoughts; similar ideas in ReAct, least-to-most prompting). The multi-sampling evaluation strategy is computationally expensive but not conceptually novel. The theoretical analysis (Theorem 3.1) establishes that error probability is bounded by $\frac{1}{2}[H(y) - I(\hat{y}; y | t, q)]$, which is generic and tautological—it says "better plans reduce error" without providing actionable insights, convergence rates, or guidance on reward design. The proof adapts standard information-theoretic inequalities and doesn't illuminate why this specific multi-sampling reward structure is optimal. 

**Narrow Experimental Scope and Missing Baselines:** Evaluation is limited to mathematical reasoning (MATH, AIME, AMC)—a domain where structured planning intuitively helps. Generalization to other reasoning types (commonsense, open-ended QA, coding, scientific reasoning) is completely unexplored. The paper cites extensive related work on planning and reflection (Gandhi et al., 2025; Tree-of-Thoughts; ReAct) but **includes no comparisons to these methods**.

### Questions
See weakness above

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes Plan-Then-Action enhancement to GRPO (PTA-GRPO), a two stage training framework for enhanced LLM reasoning tasks. It adds an additional "plan" part before the standard "think" part in deep reasoning. It uses an SFT stage for the "plan", and a RL stage using GRPO for the "plan" and "think" jointly. Experiments show strong performance over GRPO and similar methods.

### Strengths
+ The method is intuitive, focusing on a major disadvantage in LLM deep reasoning -- lack of global guidance, which causes many additional token consumption of redundancy and off-topic reasoning.
+ The two-stage training ensures a smooth learning of the additional "plan" part. The second stage features the newly proposed Planning Structure-Guided variant of GRPO. The technical method is novel.
+ The additional analytical plan reward fits the new RL training.
+ The framework, dataset, and algorithms are clearly explained in the paper.
+ Experiments show consistent performance better than GRPO and other similar methods across popular mathematical reasoning methods across different LLM architecture and sizes.

### Weaknesses
+ Theoretically, the standard GRPO can trace back and refine its action steps if it makes a mistake. However, the proposed PTA-GRPO has a fixed plan, which may be not accurate before execution but lacks a mechanism to correct itself ***during inference***
+ It lacks possible experiments with larger models, which is known better at reasoning, while small models are generally considered not very suitable for GRPO fine-tuning and reasoning. Though such training requires extensive resources, I still recommend include some as baselines for better demonstration.

### Questions
+ If possible, I suggest include prompting and SFT-based (including stage 1 results) methods in Table 1, further confirming the importance of the high level plan. It's also good to include baseline numbers from top model (e.g. GPT-5) experiments / technical reports.

### Soundness
3

### Presentation
4

### Contribution
3
