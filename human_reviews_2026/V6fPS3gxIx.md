# STORI: A Benchmark and Taxonomy for Stochastic Environments

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Reinforcement learning (RL) techniques have achieved impressive performance on simulated benchmarks such as Atari100k, yet recent advances remain largely confined to simulation and show limited transfer to real-world domains. A central obstacle is environmental stochasticity, as real systems involve noisy observations, unpredictable dynamics, and non-stationary conditions that undermine the stability of current methods. Existing benchmarks rarely capture these uncertainties and favor simplified settings where algorithms can be tuned to succeed. The absence of a well-defined taxonomy of stochasticity further complicates evaluation, as robustness to one type of stochastic perturbation, such as sticky actions, does not guarantee robustness to other forms of uncertainty. To address this critical gap, we introduce STORI (STOchastic-ataRI), a benchmark that systematically incorporates diverse stochastic effects and enables rigorous evaluation of RL techniques under different forms of uncertainty. We propose a comprehensive five-type taxonomy of environmental stochasticity and demonstrate systematic vulnerabilities in state-of-the-art model-based RL algorithms through targeted evaluation of DreamerV3 and STORM. Our findings reveal that world models dramatically underestimate environmental variance, struggle with action corruption, and exhibit unreliable dynamics under partial observability. We release the code and benchmark publicly at https://anonymous.4open.science/r/stori-353D, providing a unified framework for developing more robust RL systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces STORI, a new benchmark for testing how robust RL agents are in stochastic environments that contain uncertainty. The authors argue that current benchmarks, like the Atari 100k, are too deterministic, which hides how brittle RL agents are and doesn't help when applied in the real world. Their contribution consists of: (1) A new five-type taxonomy of stochasticity, and (2) The STORI framework, which adds these types of stochasticity as wrappers on top of four Atari games. They test two modern model based RL agents, DreamerV3 and STORM, and find they can't handle these new kinds of uncertainty. The paper's most compelling contribution is the targeted probing of the world models. These probes reveal several fundamental flaws, for example, that the learned models underestimate the true environmental variance by a very large factor (~300x).

### Strengths
1. The work tackles a big, known gap between how RL agents do in clean simulators and how they perform in the real world. Trying to build a systematic benchmark for this is a good, well motivated, goal.
2. The proposed five-type taxonomy of stochasticity is a clear and helpful way to organize the problem, which seems like it covers the important aspects of the problem. 
3. The paper's best part is the analysis in probe analysis showing that SOTA world models miss the environment's built-in randomness (the ~300x variance underestimation claim). The "clear-start" vs. "obscured-start" test is also a clever way to test how robust the agent's internal beliefs are.

### Weaknesses
1. The biggest problem with this paper is that its "benchmark for RL" only tests two agents from one single family (model-based RL). The conclusions are all about "world models," but they're written up as if they apply to all of RL. We have no idea how model-free agents, the other main approach, would perform. Do they also break, or is this a specific problem for MBRL? 
The paper abstract claims that STORI “...is a benchmark that systematically incorporates diverse stochastic effects and enables rigorous evaluation of RL techniques under different forms of uncertainty.”
This is misleading, as the results in the paper completely ignore an entire family of algorithms. The authors need to fix this by including SOTA MF baselines that perform well on the Atari 100k standard benchmark. What is the reason for the omission?
2. The paper only used 3 seeds for all experiments. This could be enough for a general RL paper, but for a benchmark that's about stochasticity, this might not be enough. For example, re table 5, the authors state in the text “DreamerV3 shows positive NLL values”. But the value in the table is  1.15±2.46, the mean is within the std of the result! This heavily questions the conclusions drawn from these experiments.

### Questions
Please address the points in the Weaknesses section. In addition:

1. The probe in Section 5.2.1 (variance underestimation) is your most interesting finding. Can you give more detail on how the ~300x variance difference was calculated? Is this an average over the 1000-step probe, a final-step measurement, or just one example?

### Soundness
1

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
This paper introduces STORI, a benchmark suite for studying stochasticity in reinforcement learning environments under different forms of uncertainty. The authors formalize a taxonomy composed of five types of uncertainty (action-dependent
noise, action-independent randomness, concept drift, representation learning challenges,
and missing state information) and provide wrapper implementations on top of existing RL environments including these stochastic events. To evaluate the framework, authors conduct experiments with two model-based RL algorithms (DreamerV3, STORM) to illustrate how standard world models behave under each type of stochasticity. They also provide an open-source repository to test the framework.

### Strengths
- **Timely and practically relevant problem:** The framework explicitly targets a crucial aspect of world models and RL agents which are influenced by real-world uncertainty.
- **Sound conceptual formalization:** The study of the different types of uncertainty as a unified integration of existing perspectives is well-motivated and tied to concrete environment mechanisms (e.g., sticky actions, time-varying transition kernels, observation aliasing).
- **Accessible and modular benchmark:** The wrappers appear easy to integrate, allowing systematic testing of RL algorithms under controlled perturbations.
- **Interesting evaluations:** The analysis of aleatoric vs. epistemic/aliasing effects via the law of total variance is conceptually clean and useful.

### Weaknesses
- **Mathematical underspecification:** Overall the mathematical notation lacks formality. Several formulas use an undefined divergence D(·||·) (e.g., for measuring concept drift) and do not specify how this is aggregated over states and actions. The notation for the corruption channel alternates between C(ã|a) and C(ã|s,a) with no explanation. Equation 4 shows the Bayes filter update as a proportional relation but does not include or mention the normalization term. The observation aliasing statement (“O(o|s_i)=O(o|s_j)=1”) is mathematically incorrect: it should read O(·|s_i) = O(·|s_j) to indicate identical observation distributions.

- **Relativity of results:** It is difficult to assess the relative performance of the evaluated algorithms since results are not compared against a clearly defined baseline (e.g., the non-perturbed environment). 

- **Learning curves:** Some learning curves in the appendix do not reach stability, raising concerns about premature training. Listing algorithm hyperparameters in the appendix would improve reproducibility.

- **Editorial and referencing issues:** Equations 12/14 and Table 3 are inconsistently referenced or missing from the text, suggesting incomplete editing. Several cross-references (e.g., between definitions in section 3.2 and results) are broken or misnumbered.

- **Difficult to follow:** While until section 4 the paper is easy to follow and clearly written, it becomes a bit cumbersome in the experiment section. The evaluation is inconsistent across uncertainty types (for example the analysis on type 2.1 is not done on type 2.2). Also, to understand that even unmentioned types have been evaluated (like type 1) is necessary to read the appendix. At the same time, tables and plots hinder the understanding since they are weakly commented and sometimes misplaced, i.e., they are sometimes placed far from their discussion, making navigation cumbersome (e.g., Figure 1b is at page 1 but described at page 7).

### Questions
- At lines 336 and 353, authors mention a baseline and comment the performance drop of Breakout with uncertanty type 2.1 of 15% wrt a baseline. Though is not really clear what this baseline actually is. Please, describe it a little bit.
- What is the intuition behind the analysis of the error caused by type 2.1 whose relative variance is compared with the one of type 3.1 and only with that? Why an equivalent analysis is not provided for type 2.2, for example? 
- Does the framework allow to combine multiple uncertainty types at the same time?

### Soundness
3

### Presentation
2

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
The authors introduce STORI, a benchmark designed to simulate various types of environmental stochasticity, and propose a five-type taxonomy to categorize such uncertainties. They conduct experiments on two state-of-the-art model-based RL algorithms, DreamerV3 and STORM, and show that these methods suffer significant performance degradation when stochasticity is introduced. The results highlight that existing world models tend to underestimate environmental variance, are sensitive to action corruption, and struggle under partial observability.

### Strengths
The main strengths of this paper lie in its clear motivation and systematic approach to a well-recognized problem. The proposed taxonomy provides a useful conceptual framework for discussing stochasticity, and the benchmark implementation can help the community evaluate algorithms under diverse types of uncertainty. Moreover, the empirical analysis is thorough and provides valuable insights into the weaknesses of existing model-based RL methods when facing stochastic dynamics. The public release of the benchmark and code is also a strong practical contribution that may encourage further research in this direction.

### Weaknesses
The experiments primarily evaluate DreamerV3 and STORM, which are not explicitly designed for stochastic environments. While the performance drop observed in these methods is interesting, it is also expected, and readers might be more curious about how adversarial training, domain randomization, or uncertainty-aware approaches perform in comparison. Including such baselines would strengthen the empirical findings and make the results more informative for practitioners aiming to improve robustness.

In terms of novelty, although the proposed STORI benchmark is more systematic and comprehensive than previous attempts, similar studies have explored stochastic perturbations in RL environments. The authors should better position their work relative to these efforts, clarifying what aspects of coverage, taxonomy, or reproducibility truly distinguish STORI from prior benchmarks.

The definition of the first type of stochasticity, in which the environment repeats the previous action with some probability, captures one form of action-dependent noise. However, it might not fully represent the broader class of real-world cases. For example, gradual drift in actuator response, changes in friction, or variations in torque output due to power fluctuations could also be categorized under this type. Expanding or at least discussing such scenarios would make the taxonomy more realistic and convincing.

Finally, the presentation of results can be improved. In the current tables, the authors mainly report the relative performance difference between stochastic and non-stochastic settings. It would be clearer to also include the absolute performance scores for each method, as this would help readers interpret the significance of the degradation. For instance, if DreamerV3 consistently outperforms STORM in the base environment, a larger performance drop under stochasticity may be less surprising. Presenting all results in well-organized summary tables would also make the paper easier to follow.

### Questions
Would it be more informative to include comparisons with methods that explicitly address uncertainty, such as adversarial or domain randomization approaches?

Have the authors analyzed whether different forms of stochasticity interact, i.e., whether combining multiple types leads to compounding effects on performance?

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
This paper presents STORI (STOchastic-ataRI), a new benchmark and taxonomy developed to systematically assess the robustness of Reinforcement Learning (RL) algorithms under stochastic environmental conditions. The central contribution lies in the STORI benchmark, which introduces a diverse range of stochastic effects across four classic Atari environments: Breakout, Boxing, Gopher, and BankHeist.
The authors further propose a comprehensive five-category taxonomy to characterize different types of environmental stochasticity.
Through experiments evaluating two state-of-the-art MBRL algorithms, DreamerV3 and STORM, the study uncovers systematic weaknesses in how these models handle uncertainty and non-determinism.

### Strengths
1. STORI offers a systematic framework that enables fine-grained introduction and control of diverse stochastic effects in standard Atari environments.

2. The evaluation extends beyond aggregate performance metrics by analyzing concrete failure modes, such as variance underestimation and state aliasing, to diagnose the root causes of algorithmic weaknesses.

### Weaknesses
1. The evaluation focuses solely on two model-based algorithms, DreamerV3 and STORM, which constrains the generalizability of the conclusions regarding robustness across the broader RL landscape.
2. The benchmark focuses on four selected Atari environments, but the paper does not discuss how these tasks were chosen. As a result, potential researcher bias in task selection, and its impact on benchmark representativeness, remains unaddressed.
3. A key limitation lies in cross-type comparability, the varying intensity and nature of stochastic effects make it difficult to draw direct conclusions about algorithm robustness across all categories.

### Questions
1. Are current model-free approaches more or less resilient to stochastic perturbations compared to model-based methods?
2. The paper indicates that DreamerV3’s world model exhibits brittleness under Type 3.2 conditions (Missing State Variables). Would increasing the model’s history-tracking capacity mitigate this limitation, or is the weakness more fundamentally rooted in its belief state update mechanism?

### Soundness
3

### Presentation
3

### Contribution
2
