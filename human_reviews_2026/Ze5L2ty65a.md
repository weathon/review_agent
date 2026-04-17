# A Hierarchical Probabilistic Framework for Incremental Knowledge Tracing in Classroom Settings

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6

## Abstract
Knowledge tracing (KT) aims to estimate a student's evolving knowledge state and predict their performance on new exercises based on performance history. Many realistic classroom settings for KT are typically low-resource in data and require online updates as students' exercise history grows, which creates significant challenges for existing KT approaches. To restore strong performance under low-resource conditions, we revisit the hierarchical knowledge concept (KC) information, which is typically available in many classroom settings and can provide strong prior when data are sparse. We therefore propose Knowledge-Tree-based Knowledge Tracing (KT$^2$), a probabilistic KT framework that models student understanding over a tree-structured hierarchy of knowledge concepts using a Hidden Markov Tree Model. KT$^2$ estimates student mastery via an EM algorithm and supports personalized prediction through an incremental update mechanism as new responses arrive. Our experiments show that KT$^2$ consistently outperforms strong baselines in realistic online, low-resource settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes $\mathrm{KT}^2$, a probabilistic KT method that models a student’s mastery over a hierarchical tree of knowledge concepts using a Hidden Markov Tree. 
It learns a global model from a small class-wide burn-in, then performs one-step EM updates per student as new answers arrive. 
Across two datasets, $\mathrm{KT}^2$ has consistent advantages over baselines and also provides interpretability.

### Strengths
Nice figures. The method writing is clear, though the formality of the notations could be improved. 
The motivation is valid, in proposing a personalized graph for each student and considering online learning in KT, which hasn’t been extensively explored yet.

### Weaknesses
- I like the presentation overall; it is clear. However, I strongly recommend that the authors improve the writing. For example, 1) avoid overusing LLMs for paraphrasing; 2) I assume lines 98–102 belong to the same paragraph; 3) The subheadings in the related works section should either be formatted as subsection titles or end with a full stop.

- What is gained beyond correlation graphs? Ths use of Hidden Markov Tree over KCs where each node’s mastery is a latent variable and has a hard entailment rule: if a parent is mastered, all children are mastered (Eq. 7). This is basically deterministic CPTs along the tree and a set of conditional independences that are observationally indistinguishable from certain non-tree graphical models when only passively observed correctness is available (no interventions). In other words, here adding latent variables is still giving you a correlational model. I am confused about where the extra modeling power comes from. Two main concerns: 
    - Missing comparisons to graph-based KT and non-hierarchical structure. To isolate what the tree (vs. a general graph or leaf-leaf links) buys you, please compare against graph-based KT baselines (e.g., GKT, SKT) fed the same induced structure. 
    - The hard constraint may affect the results and is risky especially when your trees are inferred by LLMs and may be imperfect. 

- This relates to the previous point. I think the work lacks analysis on why it works and when it fails. The qualitative Fig.3 is helpful, but there’s little error analysis (per-KC, per-item, or per-student). We don’t see calibration plots, learning curves beyond AUC, or examples where the hard hierarchy hurts.

- Emissions rely on three difficulty bins with shared parameters plus a single guessing rate $\varepsilon$(Eq. 9), and $\varepsilon$ clipped to 0.3 during training (Appendix H). This is quite a strong inductive bias. 

- Only small LLMs (3B/7B) with a simple 10-shot protocol are tried; no retrieval, cot, or structure-aware prompting. This likely underestimates LLM performance. 

- Table 2 lacks confidence intervals or significance tests as several margins are modest. This matters because differences of ~0.02-0.03 AUC may not be robust.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose a tree-based method for knowledge tracing that they update using EM updates.
They show using 2 lesser known datasets that they outperform existing approaches, notably deep learning approaches (some of them refitted in an online way) or LLMs.The proposed approach is elegant and does not require heavy computation nor a GPU.

### Strengths
The paper is well written. The proposed approach is elegant and does not require heavy computation nor a GPU. It seems to outperform existing approaches.

I enjoyed reading the paper, didn't believe the results at first read, then I understood that the proposed approach results in extra runtime as there is refitting at test time.

### Weaknesses
However, there is a large body of literature that seems missing from the paper. The authors mostly compare themselves to deep learning approaches and not simpler approaches. For example, this paper uses a hierarchical Bayesian network that is refitted on new observations, and matches the performance of DKT:

	Wilson, Kevin H., et al. "Back to the Basics: Bayesian Extensions of IRT Outperform Neural Networks for Proficiency Estimation." International Educational Data Mining Society (2016). https://www.educationaldatamining.org/EDM2016/proceedings/paper_145.pdf

More generally I am surprised that they didn't compare to cognitive diagnosis (DINA model, attribute hierarchy model) or recent approaches such as neural cognitive diagnosis:

	Wang, Fei, et al. "Neural cognitive diagnosis for intelligent education systems." Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 04. 2020.

The link between cognitive diagnosis and knowledge tracing has been established:

	Wang, Fei, et al. "Dynamic cognitive diagnosis: An educational priors-enhanced deep knowledge tracing perspective." IEEE Transactions on Learning Technologies 16.3 (2023): 306-323.

### Questions
The authors are aware of the pyKT benchmark, however it is surprising that the datasets used in the experiments are less encountered in the literature. Why didn't they compare themselves to more standard datasets such as ASSISTments?

Why didn't they compare to standard baselines such as NeuralCDM?

What is called transition probabilities is different from the transition probability in BKT "probability to acquire a skill". It is more related to the dependency structure on the tree. In the proposed approach, nothing models the latent evolution of knowledge. If some student fails some exercise at first attempt and tries again the same exercise, will the proposed model predict that the student will fail again? (The proposed model may overfit the past data and not be resilient to changes in distribution i.e. learning.)

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes KT2, a probabilistic framework designed for low-resource and online classroom scenarios. It models hierarchical knowledge concepts via a Hidden Markov Tree (HMT), treating student mastery as latent variables and exercise correctness as observed variables. The model uses an EM algorithm for initial parameter estimation and performs one-step EM updates for each new student interaction to enable incremental learning. Experiments on simulated subsets of XES3G5M and MOOCRADAR datasets show that KT² outperforms deep learning KT baselines  and LLM-based methods under low-resource conditions.

### Strengths
1. The paper addresses a practical problem: performing KT under low-resource and online settings.
2. The model formulation is mathematically consistent and clearly presented, with interpretable structure.
3. The use of a hierarchical KC tree adds intuitive interpretability compared to flat KT baselines.

### Weaknesses
1. While the model is conceptually clear, some assumptions—such as deterministic parent-to-child mastery—might be too strong; relaxing them could further improve realism. The model assumes full entailment between parent and child KCs (“if parent mastered → all children mastered”), which is unrealistic and oversimplifies real learning dynamics.
2. The current experiments rely on simulated subsets of existing datasets; validation on live classroom or streaming data would better demonstrate real-world applicability.
3. The paper could discuss more deeply how KT² might integrate with future LLM-based methdos.
4. The emission probability depends only on a fixed difficulty bin (easy/medium/hard), ignoring item discrimination and personalized learning rates.
5. The guessing and slipping parameters are fixed globally, which undermines personalization despite the “incremental” claim.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
4
