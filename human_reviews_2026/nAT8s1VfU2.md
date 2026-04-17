# XRPO: Pushing the Limits of GRPO with Targeted Exploration and Exploitation

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Reinforcement learning algorithms such as GRPO have driven recent advances in large language model (LLM) reasoning. While scaling the number of rollouts stabilizes training, existing approaches suffer from limited exploration on challenging prompts and leave informative feedback signals underexploited, due to context-independent rollout allocation across prompts (e.g., generating 16 rollouts per prompt) and relying heavily on sparse rewards. 
This paper presents XRPO (eXplore–eXploit GRPO), a unified framework that recasts policy optimization through the principled lens of rollout exploration–exploitation. To enhance exploration, XRPO introduces a mathematically grounded rollout allocator that adaptively prioritizes prompts with higher potential for uncertainty reduction. It further addresses stagnation on zero-reward prompts through an in-context seeding strategy that injects curated exemplars, steering the model into more difficult reasoning trajectories. To strengthen exploitation, XRPO develops a group-relative, novelty-aware advantage sharpening mechanism that leverages sequence likelihoods to amplify low-probability yet correct responses, thereby extending the policy’s reach beyond sparse rewards.
Experiments across diverse math and coding benchmarks on both reasoning and non-reasoning models demonstrate that XRPO outperforms existing advances (e.g., GRPO and GSPO) up to 4% pass@1 and 6% cons@32, while accelerating training convergence by up to 2.7x.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
XRPO (eXplore–eXploit GRPO) is a unified framework designed to **recast policy optimization through the principled lens of rollout exploration–exploitation** within the context of Reinforcement Learning (RL) for Large Language Models (LLMs). Existing methods, such such as GRPO and GSPO, suffer from limited exploration on challenging prompts due to **static rollout allocation** and under-exploitation of informative feedback due to heavy reliance on **sparse rewards**.

XRPO addresses these bottlenecks through three novel mechanisms:

1.  **Hierarchical Rollout Exploration:** A mathematically grounded rollout allocator that **adaptively prioritizes prompts** with higher potential for uncertainty reduction, focusing resources near the decision boundary.
2.  **ICL Seeding:** An in-context seeding strategy that **injects curated exemplars** into zero-reward prompts to break symmetry and steer the model toward more difficult reasoning trajectories.
3.  **Novelty-Guided Advantage Sharpening:** A group-relative mechanism that uses **sequence likelihoods** to amplify correct, low-probability (atypical) responses, extending the policy’s reach beyond simple sparse rewards.

Experiments show that XRPO consistently **outperforms existing advances** (GRPO and GSPO) by up to 4% `pass@1` and 6% `cons@32`, while simultaneously **accelerating training convergence by up to 2.7×**.

### Strengths
### Originality

The originality of XRPO stems from its systematic approach to integrate sophisticated exploration and exploitation mechanisms directly into the GRPO framework:

*   **Principled Rollout Allocation:** XRPO introduces a **novel hierarchical rollout planner** that dynamically allocates resources based on two mathematically defined criteria: the expected reduction in statistical uncertainty ($\Delta\hat{q}(n_q)$) and an exploration bonus ($\phi_q(T, n_q)$). This approach is differentiated from existing dynamic sampling methods that simply over-sample or discard prompts.
*   **ICL in RL Framework:** The paper presents **the first demonstration** of using In-Context Learning (ICL) seeding, drawing from an evolving corpus of verified successes, to refine performance on hard prompts with near-zero accuracy within an RL framework. This is a unique method for breaking the "zero-reward symmetry" bottleneck.
*   **Sequence-Level Novelty Measure:** XRPO develops a **sequence-level novelty measure** ($\eta_i = e^{s(y_i)-\bar{s}}$) to guide advantage sharpening, effectively extending the classical entropy bonus from the token level to the full trajectory level. This mechanism allows for fine-grained differentiation among rollouts that receive identical sparse rewards.

### Quality

The work demonstrates high quality through rigorous mechanism design and empirical validation:

*   **Comprehensive Mechanism Integration:** XRPO cohesively integrates three complementary components (Hierarchical Rollout Planning, ICL Seeding, and Advantage Sharpening) to systematically address the limitations of GRPO. Ablation studies confirm that **all three components are necessary** to fully realize the benefits of XRPO, with removal of any module causing consistent performance degradation.
*   **Superior Performance and Efficiency:** XRPO delivers substantial relative improvements, such as **+9.2% in `pass@4` and +20% in `cons@32`** for Qwen2.5-7B-Instruct. Crucially, it achieves **substantially faster training convergence**—up to 2.7× faster on MATH-500 compared to GRPO.
*   **Improved Reasoning Efficiency:** XRPO not only enhances accuracy but also leads to **better inference efficiency**, resulting in substantially shorter response lengths (e.g., a 13.6% reduction on AIME’24). This suggests the model learns to reason in a more targeted and precise fashion.
*   **Demonstrated Precision in Reasoning:** Case studies (Example 1, AIME 2025; Example 2, HMMT 2025) demonstrate XRPO’s superior reasoning rigor and methodological precision compared to GSPO, showing its ability to correctly handle complex boundary conditions and intricate sum splitting.

### Clarity

The paper is well-structured and clearly explains complex concepts:

*   **Detailed Methodology:** The mathematical criteria for rollout prioritization ($\Pi_q$) are clearly defined, and the derivation of the sequence-level novelty measure ($\eta_i$) is explicitly formulated.
*   **Cohesive Algorithm Presentation:** **Algorithm 1** clearly summarizes how the exploration (rollout planning and ICL seeding) and exploitation (advantage sharpening) components are integrated into a cohesive loop.
*   **Reproducibility:** The paper provides a clear reproducibility statement, detailing the models, publicly available datasets, experimental settings, and key hyperparameters ($\lambda_{novelty}=2.5$, $\kappa_{clip}=0.5$).

### Significance

XRPO provides highly significant contributions toward practical and efficient LLM reasoning via RL:

*   **Addressing Fundamental Bottlenecks:** It directly tackles the persistent challenges of **slow training and sparse feedback** inherent in RLVR.
*   **Optimizing Resource Allocation:** The hierarchical rollout planning ensures computational resources are **focused on high-variance prompts near the decision boundary**, where additional rollouts are most informative, thus improving sample efficiency.
*   **Generality and Compatibility:** The exploration–exploitation mechanisms are shown to be **complementary and compatible** with other state-of-the-art methods like GSPO, yielding improved performance when integrated.

### Weaknesses
While XRPO offers significant improvements, the following areas could be strengthened with more detailed analysis or discussion of inherent limitations:

#### 1. Limitations Imposed by ICL Seeding Constraints

The effectiveness of ICL seeding, which is critical for breaking zero-reward symmetry, relies on successful retrieval and prompt constraints:

*   **Impact of Truncation on Hard Problems:** The ICL prompt template limits the number of retrieved examples to $K=2$ to conserve context length, and notes that **overly long example solutions are truncated as needed**. For complex, multi-step reasoning questions that consistently fail, truncating the verified successful solution might eliminate critical reasoning steps, potentially reducing the quality of the policy guidance provided by the ICL prompt.
*   **Cold-Start and Retrieval Failure:** The ICL strategy relies on an **evolving corpus** of verified successes. If a prompt is far beyond the model's current capability ("zero-accuracy prompts underexplored," as noted in Section 3.2), there may be no similar solved examples in the corpus. The reliance on similarity search using Qwen3-Embedding-8B means that if retrieval fails, the prompt falls back to its zero-shot form, losing the necessary contextual guidance.

#### 2. Lack of Quantified Runtime Overhead for Dynamic Planning

The hierarchical rollout planning strategy requires periodic re-estimation of prompt statistics ($\bar{r}_q, s_q$) and dynamic allocation across planning rounds.

*   **Unquantified Per-Step Cost:** Although the implementation aims to be "lightweight" and the paper demonstrates overall training *convergence* is accelerated up to 2.7×, the sources **do not explicitly quantify the actual computational overhead** (e.g., latency or GPU time) introduced by the dynamic allocation calculations and the statistical gathering process *per training step*, compared to the fixed-allocation overhead of GRPO/GSPO. A detailed runtime analysis would more fully validate the efficiency claim.

#### 3. Interpretation and Generalization of Novelty-Guided Advantage Sharpening

The novelty mechanism amplifies "low-probability yet correct responses" ($\eta_i < 1$), under the hypothesis that these rare successes are critical learning opportunities that expand the model’s solution repertoire.

*   **Robustness of "Atypical" Solutions:** While the sources show that XRPO learns to produce **shorter, more efficient responses**, it is not deeply analyzed whether these statistically atypical, novelty-boosted trajectories consistently represent logically **more robust, concise, or generally applicable** reasoning paths compared to higher-likelihood correct trajectories. Further analysis is needed to confirm that the novelty measure is reliably capturing high-quality generalization opportunities, rather than merely rewarding successful but statistically rare deviations caused by sampling variance.

### Questions
**1. Sensitivity Analysis for the Rollout Planning Trade-off Parameter ($\lambda$):**

*   **Question:** The hierarchical rollout planning priority score $\Pi_q = \Delta\hat{q}(n_q) + \phi_q(T, n_q)$ requires the hyperparameter $\lambda > 0$ to trade off uncertainty-driven exploitation against exploration. Given that $\lambda$ dictates how rollout budgets are distributed across ambiguous vs. sparsely sampled prompts, could the authors provide a **sensitivity analysis** showing how varying the value of $\lambda$ (e.g., $\lambda=0.5, 1.0, 2.0$) affects overall performance metrics (`pass@k`, `cons@32`), similar to the analysis provided for $\lambda_{novelty}$ and $\kappa_{clip}$ in Table 2?

**2. Follow-up Strategy for Persistent Zero-Reward Groups:**

*   **Question:** ICL seeding is activated for prompts where **all rollouts have failed** ("zero-accuracy prompts"). Figures 1b and 4b demonstrate that ICL successfully flips a portion of these hard questions. However, for the persistent group of hard prompts that *still* yield zero successful rollouts even after being augmented by ICL examples in the current training step, how does XRPO manage these prompts in **subsequent training steps**? Are they subjected to repeated ICL seeding, or are they filtered out? Clarification on the long-term management of persistently unsolved groups would enhance understanding of the exploration strategy.

**3. Clarification on Inference Efficiency and Context Management:**

*   **Question:** The paper notes that ICL seeding truncates overly long example solutions to fit the context window. Simultaneously, Figure 4a demonstrates that XRPO achieves better **inference efficiency** by producing shorter average response lengths.
    *   **a) Is there a feedback loop:** Does the preference for shorter (more efficient) successful rollouts, amplified by novelty sharpening, indirectly influence the ICL corpus selection or the successful examples stored, thereby promoting shorter reasoning chains for future ICL seeding?
    *   **b) Impact of Truncation:** Can the authors quantify the fraction of ICL examples that were truncated during training, and whether there was an observable performance difference between ICL-seeded rollouts using truncated vs. untruncated successful examples?

**4. Mathematical Justification for Example 1 Exclusion (Case B):**

*   **Question:** In Example 1 (AIME 2025), XRPO correctly determines $t=10$ by excluding Case B, where $f'(x)=0$ if $\cos(7\pi \sin(5x)) = 0$. This exclusion relies on showing that the condition $\sin(5x) = k/7$ (from $f(x)=0$) cannot simultaneously satisfy $\sin(5x) = (m+1/2)/7$ (from $f'(x)=0$) for integers $k$ and $m$. Could the authors explicitly provide the algebraic proof that this simultaneous satisfaction is impossible, as this strict mathematical filtering is highlighted as key to XRPO's superior accuracy compared to GSPO?

### Soundness
3

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
4

### Summary
The paper proposes XRPO, a method to improve exploration and exploitation in RL-based reasoning. It combines three components: (1) uncertainty-based sampling budget allocation, (2) ICL seeding for zero-reward prompts, and (3) novelty-aware advantage sharpening that boosts rare but correct trajectories. The goal is to make GRPO training more efficient and effective on reasoning tasks.

### Strengths
1. The paper tackles a classic and important problem: the EE balance problem in RLVR. The problem is important and practically relevant for efficient RL training in reasoning-heavy domains.

2. The components address meaningful aspects of this problem: sampling allocation for more informative prompts, ICL seeding to break zero-signal cases, and sharpening advantages for rare but correct responses.

### Weaknesses
1. Limited novelty: while each of the three components addresses a meaningful aspect of the problem, similar strategies have been studied in related work. The contribution mostly lies in combining them, which makes the overall novelty limited.

2. Ablation results raise concerns: In Figure 3b, removing any single technique leads to performance close to or below GSPO (pass@1=35.52 / cons@32=45.84 per Table 1 if my understanding is correct). This raises concerns that the gains might not be robust and could come from tuning or noise, rather than stable improvements of each technique.

3. Insufficient experiments: Table 1 only compares XRPO with GSPO on Qwen3-1.7B, while Figure 3a compares GRPO on Qwen2.5-7B-Instruct, but reports different metrics (pass@4 and cons@32) from Table 1 (pass@1 / cons@32). The selective choice of evaluation metrics raises concerns regarding the generality and robustness of the reported improvements. In addition, the settings are not clearly described in the paper, and the reader has to infer them from the context.

4. Unclear training details and insufficient clarity in writing: the paper only briefly mentions using MATH data at Line401, but it’s unclear whether all experiments are trained on MATH or multiple datasets. The data setup needs to be stated explicitly.

5. Inconsistent baselines and metrics: The paper compares XRPO mainly to GSPO and GRPO, with few other strong baselines. There are existing works on prompt selection and external-guidance exploration that should be discussed or compared against.

6. XRPO introduces more complexity compared to GRPO/GSPO, but the benefits are not convincingly demonstrated.

### Questions
Please see above weaknesses.

### Soundness
2

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
This paper proposes XRPO to improve RLVR-style training for reasoning LLMs by explicitly balancing exploration (adaptive rollout allocation & ICL seeding for zero-reward prompts) and exploitation (novelty-guided advantage sharpening).
This paper through uncertainty-driven hierarchical allocation + UCB-style exploration, finite rollouts are intelligently allocated to prompts that maximize reduction of statistical error; when encountering completely incorrect groups, ICL seeding (small-sample prompts) breaks the zero variance dilemma under 0/1 rewards;
For sequences already judged correct, this paper employ advantage sharpening based on sequence novelty to expand the policy boundary. The experiments demonstrate higher accuracy and faster convergence across multiple math and coding benchmarks, alongside shorter and more decisive inference lengths.

### Strengths
This paper tackles a clear weakness of GRPO and does so with a simple yet principled exploration–exploitation design. The rollout allocator and novelty-based advantage adjustment both make intuitive sense.

The results are strong across both math and coding tasks, with faster convergence and better reasoning efficiency.

The ablation and sensitivity studies are thorough enough to show that each part of XRPO matters.

The way of how ICL seeding is used to 'revive' dead prompts is insightful — it’s a practical fix for zero-reward cases that actually shows measurable impact.

### Weaknesses
I think the baseline comparisons are a bit narrow, mostly GRPO and GSPO. I’d like to see results against other dynamic rollout or sampling-based RL methods for a fairer picture.

The reported speedup is in training steps, not wall-clock time. It’s unclear how much overhead the rollout allocator and ICL retrieval add in real runtime and I don't find more informations about this profilling work.

The novelty metric could bias toward short or lucky outputs. Some length-controlled analysis or ablations would help clarify that. I strongly suggest the authors analyze length sensitivity and show per-length bins or alternate normalization.

The priority $ \hat{\Delta}_q $ relies on sample std $ s_q $ and t-critical values; when $ s_q \to 0 $ early, the exploration bonus $ \phi_q $ rescues, but practical stability under heavy class imbalance or reward mis-calibration is not deeply analyzed; a toy-simulation study could support the allocator’s robustness.

The ICL corpus reuse raises my mild concerns about data leakage, more explicit filtering or similarity checks would make the claim cleaner.

All experiments in this paper are on Qwen series models; broader model coverage (e.g., Open sourced models like Llama, deepseek series) would strengthen generality.

### Questions
1. Can you share wall-clock or GPU-hour comparisons or detailed profill results to confirm the claimed speedup? Including overhead from allocation, novelty scoring, and ICL retrieval, to substantiate the 2.4–2.7× speedup.
2. How do you deduplicate ICL examples against eval? Any semantic similarity thresholds to prevent near-leakage? 
3. Could you add controlled comparisons to DAPO / selective rollouts / tree-search RL under the same total rollout budget and same decoding?
4. Have you ever done the fine-gained ablation studies on per-dataset ablations instead of the final average multiple datasets result?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes XRPO (eXplore–eXploit GRPO), a reinforcement learning method designed to improve the reasoning ability of LLMs. XRPO extends Group Relative Policy Optimization (GRPO) by introducing three main operations: (1) Hierarchical Rollout Planning, which allocates rollouts based on statistical uncertainty and exploration bonuses; (2) In-Context Learning Seeding, which helps overcome zero-reward prompts by providing relevent solved examples; and (3) Novelty-Guided Advantage Sharpening, which amplifies rare but correct responses to enhance exploitation. Experiments across math reasoning and code generation benchmarks demonstrate improvements over GRPO and GSPO.

### Strengths
- The paper targets two fundamental limitations of existing RLVR methods—under-exploration and under-exploitation.
- The experimental evaluation is comprehensive, covering diverse benchmarks across reasoning and code generation tasks.

### Weaknesses
- The method appears to be a combination of specific techniques without substantial novelty, rather than a broadly generalizable exploration–exploitation framework. The three designs presented in the paper are one of many means to achieve exploration (design 1,2) and exploitation (design 3), and naming it XRPO (eXplore–eXploit GRPO) seems somewhat overstated, as many methods already incorporate or implicitly address exploration and exploitation. Similarly, describing it in Line 17 as "a unified framework that recasts policy optimization through the principled lens of rollout exploration–exploitation" feels like an overclaim.
- The method's practicality for deployment is hindered by its many hyperparameters and specific design choices, such as $\lambda$, confidence $\alpha$, auxiliary model-dependent ICL seeding, and Eq. (6) with  $\lambda_{novelty}$, $\kappa_{clip}$.
- A key hypothesis behind the Novelty-Guided Advantage Sharpening is that "correct solutions with relatively low likelihood compared to other rollouts can drive the most effective learning in LLM reasoning" (Line 173). However, this claim is not sufficiently supported in the paper.
- The BACKGROUND AND MOTIVATION section (Section 3.2) already includes parts of the method and results, which is structurally inappropriate.
- The final priority score for allocating rollouts is defined based on the sample mean and standard deviation of rewards for each prompt. This requires re-estimation in each phase and fails to function in the first phase, despite there being only three phases in total.
- The ICL seeding corpus evolves during training, but its scalability and retrieval efficiency are not adequately analyzed.
- Typo: Figure 4(c) — "Incomplete Response."

### Questions
- Figure 4(a): Why does XRPO produce shorter response lengths than GRPO? The paper introduces no explicit mechanism that reduces generation length.
- How does the computational overhead of dynamic rollout allocation compare to static allocation?

### Soundness
2

### Presentation
2

### Contribution
2
