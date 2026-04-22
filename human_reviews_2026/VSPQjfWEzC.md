# EvoSyn: Generalizable Evolutionary Data Synthesis for Verifiable Learning

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2

## Abstract
Reliable verifiable data has become a key driver of capability gains in modern language models, enabling stable reinforcement learning with verifiable rewards and effective distillation that transfers competence across math, coding, and agentic tasks. Yet constructing generalizable synthetic verifiable data remains difficult due to hallucination-prone generation, and weak or trivial verification artifacts that fail to separate strong from weak solutions. Existing approaches often rely on task-specific heuristics or post-hoc filters that do not transfer across domains and lack a principled, universal evaluator of verifiability. In this work, we introduce an evolutionary, task-agnostic, strategy-guided, executably-checkable data synthesis framework that, from minimal seed supervision, jointly synthesizes problems, diverse candidate solutions, and verification artifacts, and iteratively discovers strategies via a consistency-based evaluator that enforces agreement between human-annotated and strategy-induced checks. This pipeline upgrades filtering into principled synthesis: it reliably assembles coherent, verifiable training instances and generalizes without domain-specific rules. Our experiments demonstrate the effectiveness of the proposed approach under both RLVR and model distillation training paradigms. The results show that training with our synthesized data yields significant improvements on both the LiveCodeBench and AgentBench-OS tasks, highlighting the robust generalization of our framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents EvoSyn, an evolutionary data synthesis framework for constructing verifiable training datasets in tasks where correctness can be determined by executable tests. The method starts from a simple heuristic and then evolves data-filtering strategies, guided by a consistency-based evaluator that checks reliability on a small human-verified seed set. EvoSyn is evaluated on two executably-checkable domains — LiveCodeBench (for RL with verifiable rewards, RLVR) and AgentBench-OS (for model distillation). In both settings, EvoSyn-filtered data improve downstream model performance, leading to stronger reward learning and enabling smaller models to surpass their teachers.

### Strengths
- Treating reliable synthetic instance selection as a search over filtering strategies rather than fixed heuristics offers a clean, general abstraction applicable across verification-based learning setups.
- The two consistency-based criteria (ensuring solvability and discriminative tests) address the main causes of unreliable verifiable data, and the Zero-Variance Pruning step provides an efficient quality control mechanism.
- Applying the same pipeline to RLVR and distillation demonstrates strong generality.
- The paper clearly describes the limitations of handcrafted, task-specific test-synthesis heuristics and positions EvoSyn as a more automated alternative, though execution-cost limitations still constrain scale.

### Weaknesses
- Data scale is modest (231 RLVR; 673 distillation) from small seeds (51/129), in part due to the $O(MN)$ execution cost. The authors do not report variance across multiple evolutionary runs, so generality/reproducibility is hard to judge.

- Baselines are mostly intra-method (random/relaxed). Adding strong hand-designed verification baselines would clarify the benefit of evolution.

- The method selects for solvability and discriminativeness but does not report problem-level diversity/coverage/difficulty metrics. Quality is only inferred via downstream gains (and test-count statistics).

- Positioning vs prior evolutionary program/data-search work could be sharpened.

### Questions
- The authors report 231 (RLVR) and 673 (distillation) retained instances. Are these from a single evolutionary run/seed, or averaged over multiple runs?

- Given the $O(MN)$ execution cost, can the authors quantify the wall-clock/compute cost for the reported configuration and describe how much parallelism was used in practice?

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
The paper proposes EvoSyn, a task-agnostic evolutionary data synthesis framework designed to generate verifiable synthetic data for LLM training. Verifiable data (i.e., data with executable correctness checks) is crucial for reinforcement learning with verifiable rewards (RLVR) and distillation, yet remains expensive to curate manually and hard to generalize across domains.

EvoSyn tackles this by evolving data filtering strategies that identify reliable problems, solutions, and verification artifacts (tests) through an evolutionary optimization process guided by consistency with a small human-verified seed dataset. The method iteratively refines filtering strategies based on two strict criteria ensuring alignment between human-annotated and model-inferred correctness.

Once a high-quality filtering strategy is obtained, EvoSyn synthesizes new tasks, candidate solutions, and tests, filters them using the evolved strategy, and trains models using this curated data. Experiments on LiveCodeBench (coding tasks, RLVR setting) and AgentBench-OS (agentic reasoning tasks, model distillation setting) demonstrate gains. EvoSyn-filtered data substantially improves model performance, enabling smaller distilled student models to outperform teacher models.

### Strengths
- The approach of synthesizing verifiable data is domain-agnostic, contrasting prior heuristic or task-specific filtering methods. Its evolutionary optimization of filtering strategies is broadly applicable.

- The framework is evaluated on two different benchmarks (LiveCodeBench and AgentBench-OS) under both RLVR and distillation paradigms, showing performance gains.

- The paper decomposes the pipeline (strategy evolution, synthesis, filtering, training), provides detailed ablations (e.g., effect of M, N, criteria sufficiency, pruning), and articulates trade-offs between data reliability, diversity, and computational cost.

- The paper includes prompts, strategy variants, and explicit evaluation criteria in the appendix, thus focusing on reproducibility.

### Weaknesses
- The paper is dense but not well-written and well-structured. For instance, throughout the introduction, the authors repeatedly emphasize developing a general framework for synthesizing verifiable data, yet the exact task formulation and problem statement remain vague. The objective is presented at a very high level without clearly defining the input-output structure of the task. Only by examining the experimental setup and the prompts in the appendix does it become apparent that the core task is test-case generation from NL problem descriptions. These descriptions, similar to those in competitive programming problems, may contain a few example test cases while the exhaustive test suite remains hidden. The framework also asks the LLM to generate several candidate solutions. Subsequently, EvoSyn performs cross-execution of the generated solutions and tests, for example, using TF-IDF-like, coverage-based, inverse filtering, or exclusion-based scoring approaches.
However, since both the candidate solutions and test cases are generated by LLMs, they may both be unreliable or semantically inconsistent with the original NL description. How do the authors ensure that the generated test cases are meaningful and semantically faithful to the input description, rather than reflecting coincidental or spurious correlations?

- The paper introduces a TF-IDF-like scoring mechanism where solutions that pass "difficult" tests receive higher scores, with difficulty defined as tests that are passed by only a few solutions. However, the underlying task—NL description to test-case generation, makes this assumption problematic. If a test case is passed by only a few solutions, it does not necessarily indicate that it is difficult; rather, it could simply be faulty or semantically misaligned with the problem description. In the absence of ground-truth verification or semantic alignment checks, there is no clear justification for treating such cases as valuable or discriminative. This undermines the reliability of the evolved scoring strategies and calls into question whether the "difficulty" metric genuinely correlates with test-case quality.

- "For example, RLVR-style training methods...": please define any abbreviation before using it for the first time

- There are typos in the paper e.g., "synthsizing" in line 146

- For the prompt provided in Figure 6, what are the "Problem 1", "Problem 2", "Problem 3" being referred to? These are not defined anywhere in the prompt provided in Figure 6.

- Instead of providing code-snippets in the paper and appendix, I would suggest providing algorithms that are typically more reader-friendly.

- The metrics in the experimental section are not clearly defined. How is "accuracy" in Table 3 computed?

- The evaluation covers only two benchmarks and model families. Given that the paper mentions that it focuses on developing a "general framework" for synthesizing verifiable data, broader tests on other verifiable domains (math reasoning, data-to-text, scientific QA) would better demonstrate true generality.

- EvoSyn relies on a small set of human-verified seed data to guide consistency-based evaluation. The paper does not deeply explore how biases or poor coverage in this seed data affect the evolved strategies.

### Questions
See weaknesses

### Soundness
3

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
The paper describes a method to generate verifiable synthetic data along with an automatic evolution based filtering technique to select useful datapoints from it. The evolution based filtering technique works by initializing a simple strategy and evolving it while trying to 'fit' to a small human-annotated dataset of synthetic datapoints. The overall approach works in 3 stages: (a) evolve a strategy iteratively (b) ask a strong model to generate problems, solutions and tests/oracles and filter them using the evolved strategy (c) train the model on this data. The paper evaluates this approach on LiveCodeBench using RLVR and on AgentBench-OS using distillation, and shows that the filtering helps train better models compared to randomly sampling synthetic data.

### Strengths
The paper proposes an approach to filter synthetic data without relying on task-specific heuristics, and evaluates the approach on two common approaches of post-training (RLVR and model distillation), and on two benchmarks providing some evidence of the generality of this approach.

### Weaknesses
**Baseline**: The paper does not compare the approach with real baselines. The baselines used in the paper are simple/artificial. What would be great is if the paper can compare against other filtering approaches (heuristics or other automatic approaches) so as to compare the efficacy of this filtering approach over other filtering/synthetic-data-generation approaches. Being better than random baseline is not very meaningful as it is expected that randomly generated data without any kind of filtering will be very noisy, what would be a meaningful claim is if you can show that you get similar benefits as SOTA task-specific-heuristics without having to actually manually define them yourself.

Relatedly, the "related works" section is also very sparse and the paper could benefit from broadening it significantly and actually comparing it with the approach in this paper.

**Method clarity**: The other major concern I have is that the methodology is not entirely clearly described. E.g., how exactly is the strategy evolution happening? Perhaps you could describe the algorithm in more details/using pseudocode. An example of something that's not clear from the writing: how can a strategy that outputs a *ranked* list of (solutions, tests) for every problem be used for *filtering* the data (filtering is a boolean function) -- are you filtering based on a cutoff? How are you picking problems that go in the training set based on this ranked list? Just formally writing down the process would clarify the method significantly, and would greatly improve the paper.

### Questions
Apart from the above major concerns I raised, I have a question about the criteria -- doesn't satisfying criterion 2 imply that criterion 1 is automatically satisfied?

### Soundness
2

### Presentation
2

### Contribution
2
