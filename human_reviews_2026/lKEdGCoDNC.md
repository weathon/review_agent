# ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution

- Avg Score: 4.40
- Decision: Accept (Poster)
- Scores: 6, 0, 6, 6, 4

## Abstract
We introduce ShinkaEvolve: a new framework leveraging large language models (LLMs) to advance scientific discovery with state-of-the-art performance and efficiency. The field of LLM-driven scientific discovery has seen significant progress, but has yet to overcome a critical limitation: sample inefficiency, requiring thousands of samples to identify effective solutions. ShinkaEvolve takes a concrete step towards addressing this critical limitation by introducing three key innovations: a parent sampling technique balancing exploration and exploitation, code novelty rejection-sampling for efficient search space exploration, and a bandit-based LLM ensemble selection strategy. When applied to the canonical circle-packing optimization task, ShinkaEvolve discovers a new state-of-the-art circle packing solution using only 150 samples, orders of magnitude fewer than prior frameworks. Furthermore, applied to a broader set of engineering problems, ShinkaEvolve designs robust agentic harnesses for AIME mathematical reasoning tasks, identifies improvements to ALE-Bench competitive programming solutions, and discovers novel mixture-of-expert load balancing loss functions to stabilize LLM training itself. We provide ShinkaEvolve's full code together with this submission, which will be open-sourced to accelerate open advancements to open-ended automated discovery across diverse computational problems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
SHINKAEVOLVE is a framework that uses LLMs to evolve programs efficiently for scientific and engineering discovery. It addresses a major limitation of prior systems—sample inefficiency—by introducing three innovations: novelty-aware parent selection, code novelty rejection sampling, and a bandit-based strategy to choose the most effective LLMs during evolution. These components work together to dramatically reduce the number of evaluations needed to find high-quality solutions. On the classic circle packing optimization task, SHINKAEVOLVE achieves state-of-the-art results in only 150 program evaluations, orders of magnitude fewer than previous methods. It also generalizes across domains, improving math reasoning agents (AIME), competitive programming solutions (ALE-Bench), and even discovering a new load-balancing loss for training mixture-of-experts models. The system demonstrates that LLM-driven evolutionary search can be both powerful and sample-efficient, advancing progress toward open-ended automated discovery. However, it still relies on human-defined fitness functions and structured problem objectives, leaving full autonomy as future work.

### Strengths
SHINKAEVOLVE’s key strength is its high sample efficiency, achieving state-of-the-art results in as few as 150 iterations, which is orders of magnitude faster than prior evolutionary LLM systems. It also demonstrates generalization across domains, improving solutions in optimization, mathematical reasoning, competitive programming, and even neural network training. Additionally, it introduces reasonable mechanisms—like novelty-aware sampling and adaptive LLM selection—that make the search process both effective and scalable, setting a foundation for more open-ended AI discovery systems.

### Weaknesses
1. #1 claim (sample efficiency) is shown on one canonical task with incomplete fairness controls. They highlight “SOTA in <150 evaluations” on circle packing and show an evolution tree and cost curve (Fig. 4), but I don’t see: multiple seeds, wall-clock comparisons, or matched compute/budget against the strongest recent baselines. The paper states 150 generations and that it “outperforms AlphaEvolve in less than 150 evaluations,” but it seems not to report variance or rigorous ablation under equalized budgets. That makes the “orders-of-magnitude fewer iterations” headline hard to audit.

2. Every domain here is driven by an explicit numeric fitness (packing objective; CE+load-balance proxy for MoE; AIME accuracy under a fixed query budget). That’s fine for optimization, but the paper’s “towards open-ended discovery” framing isn’t backed by tasks where the objective is underspecified or emergent. 

3. For the MoE study, SHINKAEVOLVE runs only 30 iterations during pretraining and optimizes a fitness that is the sum of CE and a load-imbalance term (L1 distance to uniform), with downstream generalization only implied. They later “scale up” to 2.7B parameters, but again emphasize CE/LBL trade-offs rather than end-task metrics; this makes it unclear whether the new regularizer robustly improves model quality beyond balancing tokens.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper claims to introduce a new method that leverages LLMs to perform evolutionary algorithm search for the benefit of scientific discovery. Empirical results across a handful of domains are presented.

### Strengths
The setting is promising and timely. Generally, the text was typo-free.

### Weaknesses
The paper is not coherently written, and I suspect it is entirely generated by an LLM. A (very non-exhaustive) list of suspicious features include the following.
- Introduction is mostly the abstract repeated verbatim.
- Paragraph starting line 110 is suspicious. Many terms are imprecisely used or not defined, e.g. “elite size,” “mutation context,” “primary parent,” “archive samples.” I can’t imagine a human writing “diverse exemplars for creative recombination.” The rest of the paragraph is similarly flowery language while not being quite precise enough for an ML audience.
-The overall problem is not defined anywhere
- Line 129: F, P_i are not defined before use. p_i is defined twice, differently. 
- r_i is both used for the rank (line 124) and for fitness value (line 210)
- The name ShinkaEvolve is never explained. 

Even if the paper was not LLM generated, it does a poor job of explaining the concepts with enough details that I would be able to reconstruct the algorithm or convince me that the authors understand the novelty and components.

### Questions
Unfortunately, even if the authors can prove there are no ethical issues, I cannot understand the novelty of the contributions from the writing.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces ShinkaEvolve, a method that uses LLMs in a sample-efficient way to generate code and algorithms for scientific discovery. The system is built to improve sample inefficiency by guiding the LLMs to only pursue promising ideas through three key steps: adaptive parent sampling, code novelty rejection-sampling and a bandit-based LLM selection strategy. The authors demonstrate their results on four scientific and engineering tasks.

### Strengths
- The paper aims to establish the building blocks for efficient agentic scientific discovery, which is a potentially high impact problem across multiple fields.
- The paper is well-structured, demonstrating clear explanations and principled ablations supporting its key contributions.
- The experiments cover four diverse and relevant domains, including an MoE architectural design challenge which is more open-ended. The results seem convincing and the discovered LBL regularization term found through this method is interesting (at the very least).
- The authors have shared their code and scripts for all experiments.

### Weaknesses
- The system is relatively complex, raising concerns that it may require quite different configurations across tasks or domains, which could limit its generalizability.
- A potential concern with using embedding similarity for code novelty rejection-sampling is that it may favor semantic or structural similarity, overlooking small but critical code modifications. Also, the LLM-as-a-judge component seems redundant with only marginal improvements at a high cost.
- The paper claims "order of magnitudes" fewer iterations compared to baselines for the circle packing task. However, it seems like the observed efficiency is at least partially an artifact of optimizing a surrogate objective of a relaxed task (Section B.1, Figure 10).

### Questions
1. Could you provide an estimate of the system's sensitivity across domains? Does the performance observed in one domain generalize to a new one without requiring extensive retuning of the system’s parameters?
2. Could you please elaborate on the limitations of using an embedding model for code novelty rejection-sampling? Also, in Table 3 (ALE-Bench config) it seems like it’s not being used.
3. Could you further clarify the core technical novelty of ShinkaEvolve and how it distinguishes itself form existing state-of-the-art evolutionary-based methods for LLM scientific discovery (e.g. AlphaEvolve [1])?

[1] Novikov et al. "AlphaEvolve: A coding agent for scientific and algorithmic discovery." (2025).

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ShinkaEvolve, a framework that addresses sample inefficiency in LLM-driven program evolution through three key innovations: adaptive parent sampling balancing exploration and exploitation, code novelty rejection-sampling to avoid redundant mutations, and bandit-based LLM ensemble selection. The approach is validated across four domains including circle packing, mathematical reasoning, competitive programming, and neural architecture design, achieving state-of-the-art results with orders of magnitude fewer evaluations than existing methods.

Strengths:
1. The paper successfully demonstrates improvements across four diverse domains: geometric optimization, mathematical reasoning, competitive programming, and neural architecture design. It shows consistent performance gains with proper ablation studies validating each algorithmic component's contribution, and includes generalization experiments such as the AIME scaffold working across different LLMs that strengthen the practical applicability claims.
2. The work introduces a well-integrated three-component system combining adaptive parent sampling, embedding-based novelty rejection, and bandit-based LLM selection that addresses real limitations in existing approaches. The framework produces immediately applicable results like the discovered MoE load balancing loss for training large language models and competitive programming solutions that outperform expert baselines, effectively bridging evolutionary computation and modern AI research with both principled algorithmic design and strong empirical validation.

However, the bandit-based LLM ensemble approach uses a relatively simple UCB1 variant. More sophisticated multi-armed bandit algorithms or contextual bandits that consider the current state of the search could potentially yield better results. The paper doesn't justify why this approach was chosen.
In addition, this study is an applied engineering method of LLMs, the innovation of technical theory is relatively general.

### Strengths
1.	The study successfully demonstrates improvements across four diverse domains: geometric optimization, mathematical reasoning, competitive programming, and neural architecture design. It shows consistent performance gains with proper ablation studies validating each algorithmic component's contribution, and includes generalization experiments such as the AIME scaffold working across different LLMs that strengthen the practical applicability claims.
2.	The work introduces a well-integrated three-component system combining adaptive parent sampling, embedding-based novelty rejection, and bandit-based LLM selection that addresses real limitations in existing approaches. The framework produces immediately applicable results like the discovered MoE load balancing loss for training large language models and competitive programming solutions that outperform expert baselines, effectively bridging evolutionary computation and modern AI research with both principled algorithmic design and strong empirical validation.

### Weaknesses
the bandit-based LLM ensemble approach uses a relatively simple UCB1 variant. More sophisticated multi-armed bandit algorithms or contextual bandits that consider the current state of the search could potentially yield better results. The paper doesn't justify why this approach was chosen.
In addition, this study is an applied engineering method of LLMs, the innovation of technical theory is relatively general.

### Questions
the bandit-based LLM ensemble approach uses a relatively simple UCB1 variant. More sophisticated multi-armed bandit algorithms or contextual bandits that consider the current state of the search could potentially yield better results. The paper doesn't justify why this approach was chosen.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces ShinkaEvolve, an evolutionary framework to overcome the sample efficiency problems in prior works with three synergistic pieces: Parent sampling, code-novelty rejection sampling, and Bandit-based LLM ensemble selection. Experiments on several benchmarks and ablations prove its sample efficiency and broad utility.

### Strengths
1. **Clear Motivation**: The paper targets sample inefficiency in LLM-driven program evolution and proposes three synergistic components:  novelty-weighted parent selection, code-novelty rejection sampling, and bandit-based LLM ensemble selection—all described with algorithmic detail.

2. **Concrete Reproducibility Details**: The paper includes hyperparameters, task harness details, and states that anonymized code is provided with plans to open-source.

3. **Multi-domain Evidence**: Extends beyond a single toy task: evolves AIME reasoning scaffolds, improves ALE-Bench solutions, and proposes an MoE load-balancing regularizer for training—broadening the paper’s relevance.

### Weaknesses
**Heavy System Complexity and Tuning without Sufficient Sensitivity Analysis**: The framework has many moving parts and hyperparameters (islands, migration, thresholds, patch types, UCB1, etc.), increasing the risk that performance depends on careful engineering. Specifically, a fixed embedding-similarity cutoff (0.95) is used; aside from showing it helps, there’s little analysis of the sensitivity or cross-task robustness of that threshold.

**Modest Gains without Statistical Significance**: Average improvement for ALE-Bench is about 2.3% over a strong baseline across 10 tasks—useful but relatively small; variance or statistical significance isn’t discussed. The SOTA claim for Circle-packing comparisons is presented mainly relative to AlphaEvolve; the breadth of baselines in this domain isn’t deeply examined. 

**Effect-size Ambiguity for Ensembling**: Text/figure messaging is inconsistent, see question 1.

### Questions
1. Could you explain why it claims the bandit ensemble “significantly outperforms” both baselines (line 455), yet also notes it only “slightly improves” (Figure 8) over a fixed uniform ensemble? Quantitative deltas are not explicit.

2. What’s the exact parent-selection formula? The ablation plot labels “Novelty Weighted,” but the description elsewhere emphasizes fitness and offspring-count weighting—does novelty enter the score, and how?

3. Why choose a 0.95 embedding-similarity threshold—and how sensitive are results to it? The threshold is fixed for several tasks without a sweep or rationale.

4. Why is novelty filtering disabled on ALE-Bench but enabled on AIME/circle-packing? What trade-offs dictated this per-domain choice?

### Soundness
2

### Presentation
2

### Contribution
2
