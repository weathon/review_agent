# Benchmarks for Reinforcement Learning with Biased Offline Data and Imperfect Simulators

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
In many reinforcement learning (RL) applications one cannot easily let the agent act in the world; this is true for autonomous vehicles, healthcare applications, and even some recommender systems, to name a few examples. Offline RL provides a way to train agents without real-world exploration, but is often faced with biases due to data distribution shifts, limited coverage, and incomplete representation of the environment. To address these issues, practical applications have tried to combine simulators with grounded offline data, using so-called hybrid methods. However, constructing a reliable simulator is in itself often challenging due to intricate system complexities as well as missing or incomplete information. In this work, we outline four principal challenges for combining offline data with imperfect simulators in RL: simulator modeling error, partial observability, state and action discrepancies, and hidden confounding. To help drive the RL community to pursue these problems, we construct ``Benchmarks for Mechanistic Offline Reinforcement Learning'' (B4MRL), which provide dataset-simulator benchmarks for the aforementioned challenges. Our results show that current algorithms fail to synergize these sources, often performing worse than using one source alone, especially when faced with hidden confounding.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work presents B4MRL, a set of offline datasets and code to add errors to MuJoCo and Highway simulators. The aim is to provide a benchmark for evaluating challenges in hybrid simulator-augmented offline RL, including modeling error, partial observability, state/action discrepancies, and hidden confounding.

### Strengths
The problem addressed by the paper is a relevant practical problem and the paper includes some interesting analysis.

### Weaknesses
**W1.** This work releases some datasets and code for modifying two simulators and presents results with some prior methods. I am not sure this is enough for a paper at this venue. Other reviewers may have a different opinion, but, since the aim of the paper is to aid future algorithm development, I think a paper of this kind should additionally include:

- (A) Specification of a small collection of experiments that serves as a standard testbed for developing future new methods (ie. similar to how D4RL specifies the -random, -mixed, -medium, etc.). Otherwise, there are so many degrees of freedom that future papers will all pick different settings, making fair evaluation of methods extremely difficult.

- (B) A more complete codebase benchmarking current methods on each of these experiments (e.g. potentially based on Unifloral or CORL). At the moment the baseline implementations are drawn from seven different codebases, which likely introduces confounding variables due to differences in implementation details beyond the core algorithms, which makes comparisons and conclusions difficult. I think for a paper of this kind, rigorous evaluation of prior methods should be part of the contribution.

**W2.** Since the main contribution of this paper is the code and datasets for the benchmark, rather than algorithm insights or results, the authors should release the code in a way that can be reviewed easily. It is easy to release anonymised repos with https://anonymous.4open.science/. The benchmark would also benefit from web-hosted docs.

**W3.** The paper's own novel baseline, HyMOPO, is noted to be unsuitable for the Walker2D and Hopper environments due to observation clipping issues, limiting its generality.

### Questions
**Q1.** Do you have an intuition as to why hybrid-RL algorithms scored worse than algorithms without the offline dataset? Do you think they could perform better if tuned properly, or do you think there is a fundamental problem?

**Q2.** Since baselines were drawn from different repositories, is it possible that the poor performance of some hybrid methods is due to suboptimal tuning for this specific hybrid task rather than a fundamental algorithmic flaw?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents B4MRL (Benchmarks for Mechanistic Offline Reinforcement Learning), a comprehensive benchmark suite designed to evaluate offline-to-online RL algorithms when combining offline datasets with imperfect simulators in hybrid RL settings. The paper identifies four principal challenges in hybrid RL: (1) simulator modeling error (sim2real gap), (2) partial observability and state discrepancy, (3) action discrepancy, and (4) hidden confounding (offline2real bias). Unlike existing benchmarks (D4RL, VD4RL, ODRL, CARL, etc.), B4MRL uniquely addresses all four challenges simultaneously, providing a systematic framework for evaluation. Through extensive empirical evaluation on MuJoCo and Highway environments using online (TD3, SAC), offline (TD3-BC, IQL, MOPO), and hybrid (H2O, PAR-BC, HyMOPO) algorithms, the paper demonstrates a critical finding: current hybrid RL methods frequently fail to leverage both data sources synergistically, often performing worse than using either source alone, particularly when hidden confounding is present.

### Strengths
**Comprehensive and well-motivated problem characterization.** The paper articulates a compelling and timely problem: while offline RL and simulator-based RL are well-studied, hybrid methods that combine both remain poorly understood and underspecified. The four challenges (modeling error, partial observability, action discrepancy, confounding) are systematically presented with concrete real-world examples (healthcare, autonomous driving, recommender systems), making the motivation clear and accessible. The distinction between partial observability and hidden confounding (Section 2.4) is particularly valuable and often overlooked.

**Rigorous and modular benchmark design.** B4MRL provides a principled, composable benchmark architecture where challenges can be independently controlled and combined. The design choices are justified through ablation studies (Figure 4 demonstrates how to select which variables to hide), and the implementation details (Section B, Appendix) are thorough and reproducible. The use of parametric modifications to MuJoCo environments (gravity, friction, action noise) provides clean, interpretable ways to introduce discrepancies.

**Diverse algorithm coverage.** Evaluating eight different RL algorithms (online, offline, and hybrid) across multiple environments and challenges provides broad empirical coverage.​

### Weaknesses
**Hidden confounding implementation is somewhat simplistic.** The confounding benchmarks (Section 3, Challenge 4) introduce confounding by either adding noise or removing variables from observations, with the assumption that the data-generating agent saw the missing variables. This is a stylized form of confounding that may not capture the full complexity of real-world confounding scenarios (e.g., time-dependent confounding, continuous latent confounders). More sophisticated confounding mechanisms could strengthen the benchmarks.

**Narrow scope of environments and tasks.** All experiments use MuJoCo continuous control and one highway driving environment. The scalability and applicability to image-based observations, discrete action spaces, or more complex domains remain unclear. D4RL benchmarks are well-studied but represent a narrow slice of RL problems.

### Questions
**Q1:** Why is three seeds used instead of five or more? Was this a computational constraint? Can results be re-run with additional seeds for higher statistical confidence?

**Q2:** How sensitive are confounding results to the specific variables chosen to hide? Figure 4 shows variable importance varies across algorithms. Is there a principled way to select impactful confounders?

**Q3:** Can you provide theoretical characterization of when hybrid methods have fundamental limitations? Is there a theoretical explanation for why some combinations of challenges make hybrid methods perform worse than individual methods?

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
The authors introduce a new benchmark for offline RL with biased offline data and imperfect simulators. Unlike standard benchmarks in offline RL (D4RL, etc.), they focus on realistic challenges such as modeling errors, causal confusion, and partial observability. Across four high-level challenges, they propose diverse variants of MuJoCo tasks (halfcheetah, hopper, and walker2d) and the Highway task. They benchmark several representative offline and online RL algorithms on this benchmark, showing that there is still room for improvement in addressing these challenges.

### Strengths
* The problem posed in this paper is sensible. I believe the community can benefit from offline RL benchmarks that incorporate more realistic challenges encountered in the real world.
* The paper is well-organized and easy to understand.

### Weaknesses
* The main weakness of this benchmark is its relevance. The authors motivate this benchmark from diverse real-world challenges in offline RL, such as dynamics discrepancy, causal confusion, and partial observability. However, the quality of the environments provided in this benchmark is limited in realizing these challenges. Specifically, the environments are limited to simple 2-D MuJoCo tasks (halfcheetah, hopper, and walker2d) and the (highly simplified) Highway environment. I'm unsure how impactful and useful these tasks are for today's offline RL research. From the motivation in the Introduction, I expected much more realistic benchmarks, such as datasets collected from actual human demonstrators or at least more realistic scripted or non-Markovian policies on more relevant tasks (e.g., complex and realistic robotic manipulation, long-horizon navigation, computer games, etc.).
* Moreover, while the individual challenges listed in Table 1 are sensible, they are implemented in a highly contrived manner. For example, the authors simply change the gravity or friction parameter to simulate modeling errors, and add Gaussian noise to simulate state/action discrepancies or causal confusion. In the real world, these challenges are often much more subtle -- such errors or noises are typically temporally correlated, non-Markovian, and biased. I'm not sure how representative and realistic the challenges implemented in this benchmark are. I also believe these implementations are too simplistic to be impactful enough as a standalone benchmark. I'd be fine with such simplifications in methodology papers, but as a benchmark paper, I think the bar should be higher than that of typical experiments in such papers.

### Questions
* Why is the benchmark called "mechanistic" offline RL?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces B4MRL, a new benchmark suite for evaluating reinforcement learning algorithms that combine offline data and simulators. The benchmarks address four real-world challenges: modeling error, state and action discrepancies, partial observability, and hidden confounding. Experiments show that current hybrid RL methods often fail, especially when offline data contains hidden confounders. The results highlight the need for more robust algorithms that can reliably integrate both data sources. The paper proposes the benchmark to test these algorithms.

### Strengths
- Originality: Proposes (to my knowledge) the first RL benchmark that systematically combines all four sources of sim2real and offline2real error.
  
- Quality: Has educational value in the taxonomy of challenges. Implements them in simulated benchmark settings.
- Quality: Provides careful experimental evaluation with many baselines across offline, online, and hybrid RL methods

- Clarity: Clearly motivated and systematic presentation

- Significance: Addresses an important issue in the RL community, that is important for real-world applications.

### Weaknesses
- W1: partial observability calls for methods for POMDPs. The methods used, such as TC3-BC, SAC etc are all using TD learning and rely on the Markov property. Methods that use Monte-Carlo returns (such as PPO) or methods that have recurrent architectures (policy and value functions) would be the natural algorithms to be tested.

- W2: The hidden confounder problem is present in modern sim2real pipelines with teach to student distillation and privileged information used for the teacher. The students are always recurrent networks, to perform latent state estimation. I think you benchmark is great, but I also believe the problems have been solved already in practice.

- W3: Limited diversity of tasks/environments: While MuJoCo and Highway are used, the benchmarks are centered on classic continuous control tasks and may not generalize to more complex domains such as vision-based control.

### Questions
- Q1: I am mostly concerned about the non-Markovianity. I suspect that the results will change quite drastically, if methods are used that are designed for partial observability. Would be very interesting to understand if the confounding is really such a strong problem then. Now, confounding is somewhat convolved with partial observability. Can you provide evidence that the observed phenomena persist?

- Q2: what happens if you use recurrent architectures? (or as a first approximation provide a history of 4 observations?

- Q3: Fig 5: what do you mean with "moving from simple modeling error to high-impact hidden confounding"? Is the drop when setting 1+4 vs the setting 1? 



Comments:
- line 204: "and P (r = 1|z = 1, a = a0) = 1/6" should that not be z=0?
- line 294/295: "...Gaussian noise into the action implemented by the agent to the simulator’s present state...". What means action into simulator's state?
- line 377: $o'_{\text{sim}}$ sim should prob. be a subscript. 
- Fig 5b: I think the labels are confusing and redundant. The x-labels already contain the challenge. (Maybe use the descriptive names instead of numbers). Reduce the number of markers to the actually different runs (so one line should have only one marker). Also, a clearer description of sigma, h and g would be good.

### Soundness
3

### Presentation
3

### Contribution
3
