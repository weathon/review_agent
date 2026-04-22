# Master Skill Learning with Policy-Grounded Synergy of LLM-based Reward Shaping and Exploring

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 2, 8, 6, 6

## Abstract
The acquisition of robotic skills via reinforcement learning (RL) is crucial for advancing embodied intelligence, but designing effective reward functions for complex tasks remains challenging. Recent methods using large language models (LLMs) can generate reward functions from language instructions, but they often produce overly goal-oriented rewards that neglect state exploration, causing robots to get stuck in local optima. Traditional RL addresses this by adding exploration bonuses, but these are typically generic and inefficient, wasting resources on exploring task-irrelevant areas. To address these limitations, we propose Policy-grounded Synergy of Reward Shaping and Exploration (PoRSE), a novel and unified framework that guides LLMs to generate task-aware reward functions while constructing an abstract affordance space for efficient exploration bonuses. Given the vast number of possible reward-bonus combinations, it is impractical to exhaustively train a policy from scratch for each configuration to identify the best one. Instead, PoRSE employs an in-policy-improvement grounding process, dynamically and continuously generating and filtering out reward-bonus pairs along the policy improvement process. This approach accelerates skill acquisition and fosters a mutually reinforcing relationship between reward shaping, exploration and policy enhancement through close feedback. Experiments show that PoRSE is highly effective, achieving significant improvement in average returns across all robotic tasks compared to previous state-of-the-art methods. It also achieves initial success in two highly challenging manipulation tasks, marking a significant breakthrough.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces PoRSE (Policy-grounded Synergy of Reward Shaping and Exploration), a framework for robotic skill learning that integrates LLM-generated reward design with efficient, task-relevant exploration. Existing LLM-based methods like Eureka and ROSKA focus too narrowly on goal rewards or reuse past policies but neglect structured exploration. PoRSE overcomes this by constructing an Affordance State Space (AFS) which is a low-dimensional, task-aware abstraction of the environment where exploration bonuses are assigned based on visitation frequency.

### Strengths
1.1 The paper’s idea of guiding exploration with a task-aware state mapping instead of generic novelty is clear and makes sense. The ablations confirm that this mapping improves learning stability and performance.

1.2 The experiments span a wide range of tasks and are supported by detailed ablations and additional appendix results, which make the findings more convincing and strengthen the overall paper.

### Weaknesses
2.1 The results are incomplete. The paper lacks comparison and positioning against state-of-the-art frameworks, such as the REvolve, which likewise uses LLMs in reward design, and should therefore have been included.

2.2 Section 5.1 tournament selection is unclear. You start K=6; at 500/1000 you prune to 4; at 1500 you add J=5 mutants; then you say “remove 6” at 1500. It is not stated whether the five new candidates receive any training before ranking, nor whether J=5 is total or “per survivor.” As written, untrained mutants seem pruned alongside half-trained survivors. This needs a further clarification. 

2.3 In Table 3 (Appendix A.2.1), MTS is reported as $\text{mean} \pm \text{std}$ of the per-seed maximum. For manipulation tasks this “maximum” represents a success rate $\in [0,1]$, while for locomotion it is a return (unbounded). The caption incorrectly labels all rows as “success rates,” which conflates metrics and makes cross-row comparisons invalid. Moreover, the table selectively bolds PoRSE even when other methods achieve identical or better results (e.g., 1.000 ties), which is misleading.

2.4 β mixing is inconsistent. In the main document at Section 4.2 the authors use β with (2−β) but in Appendix A.1 uses β with (1−β).

2.5 Pruning every 500 epochs assumes early rank predicts who will still be bad at 3000. The paper does not support this claim with empirical results.

2.6 Several interesting results exist in the Appendix which are not referenced in the main paper.

2.7 Abstract reads like an extended introduction; it mixes motivation, method and broad claims. This could be tightened.

### Questions
3.1 Why absent comparison with state-of-the-art frameworks? Can you add comparisons with further baselines?

3.2 At 1500 epochs after adding J=5, how many epochs do the mutants train before any ranking; is J=5 total or five per survivor?

3.3 Which β formulation produced for the results based on point 2.4?

3.4 Can you provide evidence that pruning the lowest-performing candidate every 500 epochs truly leads to worse final performance?

3.5 Could you provide further clarification on Table 3, as mentioned in point 2.3?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Building on LLM-based reward with evolutionary optimization over reward code (i.e., EUREKA), this method improves exploration by having an LLM compress the state space into a low-dimensional representation and add a count-based novelty bonus. The LLM then balances exploration and goal-oriented rewards by producing a set of weights based on the policy performance. To avoid a combinatorial sweep over weight–reward pairs, weak candidates are pruned early while promising ones are lightly refined. Finally, the policy is retrained with the refined reward (warm-started from the previous policy), and the LLM decides how to mix the new and prior rewards based on observed results.

### Strengths
- This paper tackles a core RL challenge—exploration—by combining LLM with a classical novelty bonus. The approach is clear and leverages both strengths: LLMs can keep improving, while novelty rewards remain general and broadly applicable.
- Presents rich analyses that identify which and how introduced components contribute.

### Weaknesses
- **Intuition for LLM-chosen $\alpha$**
  + The rationale for how the LLM predicts  $\alpha$ isn’t clear. Could authors elaborate on the intuition, and if applicable, include the related prompt, as well as show how  $\alpha$ evolves during training?

- **Trends for $\beta$**
  + L302–304 describe how $\beta$ would evolve with policy performance, but no experimental evidence is shown. Could authors support this? Having a plot or table showing performance vs. $\beta$ over training would give clarification.

- **Clarification on metric**
  + The main paper reports only the best reward, which raises a concern for me that the variance may be high, given that the method introduces multiple coefficients. However, later I found that Table 3 shows reasonable standard deviations, so I suggest either including a reference in the main results to guide readers.
  + As a minor suggestion, adding performance drop percentages to Table 1 would make it easier to see which components are most critical. Ultimately, up to the authors.

### Questions
- Q1. L200 says it maps the robot’s state to a low-dimensional space, but the example (“distance to the door handle”) depends on the environment state. Is the mapping’s input the robot state only, or the full environment state?
- Q2. Results are shown up to 5 iterations. How does performance evolve with more iterations—does it converge or diverge? With more iterations, can it solve harder tasks such as “ReOrientation” or “Switch” in Fig. 4?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
PoRSE is a framework for robotic skill learning that synergizes LLM-generated goal-oriented rewards with task-aware exploration bonuses via an affordance state space (AFS), using in-policy grounding (IPG) for dynamic refinement and policy fusion. It iteratively filters reward-bonus pairs and adjusts trade-offs based on policy feedback, evaluated on 24 simulated tasks (IsaacGym, Bi-DexHands) with PPO, achieving superior MTS/HNS over baselines like Eureka, ROSKA, and LLMCount, including breakthroughs in hard manipulation tasks.

### Strengths
1.Effectively addresses exploration gaps in LLM rewards by integrating task-relevant AFS bonuses, with IPG enabling adaptive synergy without exhaustive retraining.
2.Strong results across 24 diverse tasks, outperforming SOTA on 23/24 vs. LLMCount and achieving first successes in sparse-reward challenges like TwoCatch.
3.Comprehensive ablations validate components (rewards, bonuses, fusion), robustness to LLM variants (e.g., GPT-4o-mini), and parameters (bins, normalization), with fair compute parity.
4.Practical insights into policy-reward co-evolution, supported by visualizations and multi-seed stats.

### Weaknesses
1.Limited to simulation and no real-robot validation.
2.High LLM dependence (GPT-4o) without cost analysis—multiple iterations with 6 rewards each could be token-expensive.
3.Baselines like Eureka/ROSKA use same LLM but may not be optimally adapted; sparse/human rewards are weak strawmen in some tasks.
4.AFS discretization assumes low-dim abstractions work universally, but ablations show sensitivity to randomness or prompts.
5.Claims of "breakthroughs" in hard tasks lack comparison to non-LLM exploration methods like RND or ICM.

### Questions
1.What are the total LLM token costs per task/iteration, and how does efficiency scale to longer horizons?
2.Are there evaluations on open-source LLMs (e.g., Llama) or weaker models beyond GPT-4o-mini?
3.How robust is AFS to multi-modal inputs like vision; could it integrate VLMs?
4.In cross-task extensions (A.4.4), what specific adaptations are needed for real environments?

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
This paper proposes PoRSE, a framework that unifies LLM-based reward shaping with task-relevant exploration for robot skill learning. An LLM generates (1) goal-oriented rewards and (2) an affordance state space (AFS) that maps high-dimensional states to discrete, task-meaningful features; a count-based bonus over AFS promotes exploration. PoRSE dynamically balances rewards and bonuses via a weighting β and accelerates training with an in-policy-improvement grounding (IPG) process. Across 24 tasks (Bi-DexHands, Isaac Gym), PoRSE outperforms sparse, human, Eureka, and ROSKA baselines, with ablations validating each component and robustness to different LLMs.

### Strengths
1. PoRSE provides a unified and generalizable structure that integrates reward shaping and exploration within an LLM-guided loop.

2. Extensive Experiments: Evaluated across 24 robotic skill tasks (Bi-DexHands and Isaac Gym) with strong quantitative results — outperforming baselines in 23/24 tasks.

3. Clear Practical Value: Avoids costly full retraining via policy inheritance, making it more computationally efficient.

4. Comprehensive Ablations: Studies on β, α, mapping functions, and LLM robustness (e.g., GPT-4o-mini variant) show the framework’s adaptability and resilience.

### Weaknesses
1. The reliance on specific LLM outputs (e.g., DeepSeek-V3) introduces reproducibility and bias issues, not fully explored in the paper.

2. Limited Real-World Evaluation: While the appendix outlines a feasible sim-to-real adaptation (fine-tuning with small datasets, RealSense/tactile sensors, and lightweight VLMs for affordance cues), no empirical real-robot results are yet presented. Thus, deployment remains conceptually described rather than experimentally validated.

3. The paper asserts “nearly identical” cost to Eureka/ROSKA, but PoRSE trains K candidates per round with elimination/expansion and mid-round mutations (J new variants at 1500 epochs), plus alternating searches over β and α. This likely alters wall-clock and sample budgets vs. baselines. Provide exact environment steps, PPO updates, and GPU hours for each method.

### Questions
1. I am wondering about the training efficiency. E.g., for one epoch, how much time is needed for the proposed PoRSE and other baselines?

2. How sensitive is PoRSE to the quality of the LLM outputs? Does fine-tuning or prompt engineering significantly alter results?

3. How do the LLMs perform with increasing task complexity or policy dimensionality? E.g., for long-horizon tasks or when envs get more complex.

### Soundness
3

### Presentation
3

### Contribution
2
