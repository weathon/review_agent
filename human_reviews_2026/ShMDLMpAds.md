# Buzz, Choose, Forget: A Meta-Bandit Framework for Bee-Like Decision Making

- Decision: Reject
- Scores: 2, 6, 2

## Abstract
We introduce a sequential reinforcement learning framework for imitation learning designed to model heterogeneous cognitive strategies in pollinators. Focusing on honeybees, our approach leverages trajectory similarity to capture and forecast behavior across individuals that rely on distinct strategies: some exploiting numerical cues, others drawing on memory, or being influenced by environmental factors such as weather. Through empirical evaluation, we show that state-of-the-art imitation learning methods often fail in this setting: when expert policies shift across memory windows or deviate from optimality, these models overlook both fast and slow learning behaviors and cannot faithfully reproduce key decision patterns. Moreover, they offer limited interpretability, hindering biological insight. Our contribution addresses these challenges by (i) introducing a model that minimizes predictive loss while identifying the effective memory horizon most consistent with behavioral data, and (ii) ensuring full interpretability to enable biologists to analyze underlying decision-making strategies and finally (iii)  providing a mathematical framework linking bee policy search with bandit formulations under varying exploration–exploitation dynamics, and releasing a novel dataset of 80 tracked bees observed under diverse weather conditions. This benchmark facilitates research on pollinator cognition and supports ecological governance by improving simulations of insect behavior in agroecosystems. Our findings shed new light on the learning strategies and memory interplay shaping pollinator decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes an approach to modelling empirically observed behavior of bees. A mathematical model can help understand the nature of the behavior and can also serve as a tool in simulated experiments. The authors use the contextual bandit framework to explain decisions made by the bees in Y-maze experiments. The authors observe different behavioral patterns and propose to capture these using different policies. The resulting policy switching model adequately captures the behavior of bees in the sense of reproducing regret trajectories.

### Strengths
1. The proposed model captures observed switching in behavioral patterns
2. Some parameters of the proposed model have physiological interpretation (e.g., durability of memory)
3. The authors compare several behavioral policies
4. The paper introduces a new dataset with experimental observations

### Weaknesses
1. The utility of the proposed model is not clear.

1a) If the goal is to facilitate understanding of the underlying natural phenomenon, then the methodology is not sufficiently rigorous. For example, how reliable is the observed effect of climate on memory? If one used a different set of policies in modelling, or different parameters of those policies, would the conclusion remain? Also, statistical significance of this trend is not analysed.

1b) If the goal is to use the model for simulation, then what is the process being simulated? The model appears to be specific to the Y-maze setup, and not directly suitable for simulating behavior in a more complex environment. Thus simulation capacity appears to be rather restricted.

2. Technical choices made by the authors are not fully explained. First, it is not clear why to use RL-based approaches to begin with. Next, within the proposed method, the particular collection of the policies is arbitrary to some extent. For example, would the modelling be more effective if one only used greedy and CMAB policies? Would it be a better model with three policies? Or would the model improve if one added more policies?

3. There are a number of ways presentation can be improved. First, citations are not formatted properly. Second, there is no need in introducing state transitions, if the proposed framework is based on bandits.

### Questions
1. Has the proposed model helped uncover an underlying natural phenomenon? If yes, has this discovery been assessed from the point of view of statistical significance?
2. How can the proposed model be used to test ecological decisions in simulations? Would such simulations be constrained to Y-maze experiments?
3. Have any non-RL based approaches considered for modelling?
4. Would the framework be more effective if only a subset of policies was used (e.g., greedy and CMAB)? Would the framework be more effective if more policies are used (e.g., different variations of CMAB are added)?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes MAYA (Multi-Agent Y-maze Allocation), a meta-bandit imitation learning framework designed to model heterogeneous decision-making strategies in pollinators, particularly honeybees. Unlike standard imitation learning (IL) or inverse reinforcement learning (IRL) approaches that assume an optimal and stationary expert, MAYA explicitly accounts for non-optimal, memory-limited, and strategy-switching behaviors observed in biological agents. MAYA treats each bee as dynamically aligning with one of several candidate multi-armed bandit (MAB) policies (Uniform, ε-greedy, UCB, LinUCB), while optimizing a sliding memory window τ that defines how much historical information influences current decisions. The framework uses three trajectory similarity measures—KL, Wasserstein, and DTW distances—to match observed regret sequences with simulated bandit policies, thereby identifying both the likely strategy and effective memory horizon per individual. However, the study remains primarily exploratory and descriptive, lacking deeper theoretical analysis or quantitative generalization beyond the examined Y-maze datasets.

### Strengths
1. Interdisciplinary contribution. The paper successfully bridges reinforcement learning, imitation learning, and animal cognition. The idea of representing animal behavior as a dynamic mixture over bandit policies is biologically meaningful.

2. Addresses a real limitation in imitation learning. Unlike conventional IL methods assuming near-optimal experts, MAYA explicitly handles heterogeneous and non-optimal behaviors with limited memory—precisely the challenge in modeling real animals.

3. Interpretable and biologically relevant framework. The framework allows scientists to extract interpretable parameters—dominant strategy type, memory span τ, and environmental dependence—directly from data. This makes the model genuinely useful for behavioral biology.

4. Comprehensive empirical validation. The authors evaluate across five geographically and climatically diverse datasets plus cross-species validation (mice). The discovery that τ stabilizes around 7 trials and shortens in cold weather adds ecological credibility.

### Weaknesses
1. Limited theoretical novelty. The algorithm mostly reinterprets known bandit strategies within an imitation-learning context; the “meta-bandit” framing is elegant but not deeply theoretical. Formal analysis (e.g., convergence, identifiability) remains light.

2. Interpretation depends heavily on heuristic similarity metrics. The KL, Wasserstein, and DTW distances are reasonable, but their selection and weighting are somewhat ad hoc. It’s unclear how sensitive results are to these choices.

3. Potential overfitting to small-scale behavioral data. The datasets, though diverse, are limited in trajectory length (22–40 trials). The generalization to other tasks or to continuous-action animals remains untested.

4. Exploratory rather than predictive. The study focuses on reproducing observed behavior but does not evaluate predictive accuracy on unseen trials or individuals, which limits its use.

### Questions
1. Robustness of τ selection: How stable is the discovered τ = 7 across random seeds or subsamples? Could τ be treated as a learnable latent variable rather than manually sweeping it?


2. Extension beyond two-armed tasks: The Y-maze is effectively binary. Would the framework scale to multi-armed or continuous-action scenarios?


3. Practical use by biologists: How easy is it for a behavioral scientist (without ML background) to use the open-source MAYA toolkit for analyzing new data?

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
3

### Summary
The paper proposes MAYA, an imitation-learning framework that models individual animals (bees and mice) as dynamically switching among a small library of two-armed bandit strategies (Uniform, Epsilon-Greedy, UCB, LinUCB). At each trial MAYA selects the candidate policy whose recent regret trajectory (within a sliding window tau) best matches the observed animal trajectory under a chosen similarity metric (Wasserstein, KL, or DTW), updates an imitator accordingly, and outputs interpretable per-trial alignments. Experiments use a new dataset of 80 bees across five conditions and comparisons to IRL/IL baselines and GLMs; authors report MAYA-Wass with tau=7 performs best.

### Strengths
The paper addresses an applied, interpretable modeling problem of interest to behavioral ecologists.

Simple, intuitive approach producing per-trial interpretability that domain scientists can inspect.

Evaluates multiple similarity metrics (distributional and temporal), allowing comparison of approaches.

The authors intend to release data and code, which could be valuable if fully reproducible and documented.

### Weaknesses
The empirical results contain implausible anomalies (e.g., AIRL reported as 0), indicating likely bugs, data leakage, or incorrect evaluation scripts; this undermines the credibility of comparisons.

Key methodological details are ambiguous or missing: exact form of the sequences compared (instantaneous vs cumulative or normalized regret), normalization, and whether candidate bandits are updated on observed rewards (which would leak future information).

Baseline implementations, hyperparameter search, and evaluation protocols are insufficiently specified. The fairness of comparisons is unclear.

The considered dataset is small (80 bees, ~16 per dataset) with short per-subject sequences; the paper lacks rigorous statistical analyses (paired tests, bootstrap CIs, per-subject variability) to substantiate broad claims like a universal tau=7.

Presentation appears partly noisy: typographical errors, low-resolution figures, bad legends, and important details scattered across a long appendix, harming reproducibility.

No explicit statement of animal ethics approvals or dataset-release documentation is provided.

### Questions
Can you explain baseline anomalies (e.g., AIRL = 0 everywhere). What are the implementation details, hyperparameters, training logs, and demonstrate that no test data leaked into training or candidate simulations.

Formally define what sequence is compared by similarity metrics (instantaneous/cumulative/normalized regret or actions/rewards) and include a worked numeric example.

Clarify how candidate policies are simulated/updated during evaluation: do they consume observed rewards (risking leakage) or are they evaluated in a simulator? If offline, explain how rewards for hypothetical candidate actions are handled.

Provide full hyperparameter tables and tuning procedures for MAYA (tau grid, epsilon/UCB/LinUCB params) and for every baseline; specify whether tuning used held-out subjects and whether the same budgets where used across baselines.

Report statistical significance for key comparisons (paired tests, bootstrap CIs) and include per-bee variability analyses.

Provide ablations: sensitivity to the candidate policy library (add/remove policies like Thompson Sampling), sensitivity to tau over broader ranges, and controlled comparisons among similarity metrics.

Include an explicit animal ethics statement and data-release documentation (approvals, welfare protocols).

### Soundness
2

### Presentation
2

### Contribution
2
