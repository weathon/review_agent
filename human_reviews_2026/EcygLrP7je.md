# MAVIS: Multi-Objective Alignment via Value-Guided Inference-Time Search

- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Large Language Models (LLMs) are increasingly deployed across diverse applications that demand balancing multiple, often conflicting, objectives--such as helpfulness, harmlessness, or humor. Aligning outputs to user-specific preferences in such multi-objective settings typically requires fine-tuning models for each objective or preference configuration, which is computationally expensive and inflexible. We introduce MAVIS - Multi-Objective Alignment via Value-Guided Inference-Time Search - a lightweight inference-time alignment framework that enables dynamic control over LLM behavior without modifying the base model's weights. MAVIS trains a set of small value models, each corresponding to a distinct objective. At inference time, these value models are combined using user-specified weights to produce a tilting function that adjusts the base model's output distribution toward desired trade-offs. The value models are trained using a simple iterative algorithm that ensures monotonic improvement of the KL-regularized policy. We show empirically that MAVIS outperforms baselines that fine-tune per-objective models and combine them post hoc, and even approaches the performance of the idealized setting where models are fine-tuned for a user's exact preferences.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes MAVIS, a value-guided decoding framework that enables a plug-and-play framework for multi-objective alignment. The main technical contribution involves resembling soft policy iteration to better estimate the Q function.

### Strengths
- LLM multi-objective alignment is important.

- The proposed method is overall reasonable.

### Weaknesses
- The presentation of this paper requires further refinement. Certain passages are overly informal; for instance, line 431 begins with “Of course…”. Additionally, some concepts are inadequately explained—for example, ζ is not properly defined in Section 3. Figure 2 in the methodology section seems unnecessary. Overall, the paper is a little hard to follow, especially for RL non-experts.

- The motivation of the work is unclear. Previous methods such as GenARM [1] and PARM [2] have already explored test-time multi-objective alignment in LLMs.

- The novelty of the paper is limited. The core contribution revolves around soft policy iteration, which has been explored in prior works. No fundamentally new algorithm is introduced.

- The current comparison set is narrow. The authors should include more comprehensive baselines: (1) For training-based methods: compare against MODPO, RiC, and other SOTA methods. (2) Compare against more methods of model merging. (3) Directly compare with other multi-objective decoding methods such as GenARM and PARM.

- The test set contains only 100 held-out prompts, which may limit robustness. Consider using benchmarks such as AlpacaEval 2 which leverage LLM-judge frameworks to assess helpfulness and alignment more reliably.

- The paper lacks quantitative analysis of the impact of guidance on generation quality, such as fluency and diversity.

- Lack of more analysis: (1) the role of Top-K sampling; (2) the performance change during each iteration when training value functions; (3) the cost analysis to estimate value functions.

- The authors claim that “MAVIS can be applied regardless of whether the weights of π_ref are available.” However, it is unclear how distribution adjustment can be performed without access to these weights.

- The process of training value functions may require more time compared to directly training the policy. Additionally, the training stability remains uncertain, and the performance under long output sequences has not been explored.

Reference

[1] GENARM: REWARD GUIDED GENERATION WITH AUTOREGRESSIVE REWARD MODEL FOR TEST-TIME
ALIGNMENT

[2] PARM: Multiobjective test-time alignment via preference-aware autoregressive reward model.

### Questions
See weaknessness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper *“MAVIS: Multi-Objective Alignment via Value-Guided Inference-Time Search”* tackles the problem of multi-objective alignment during decoding. Instead of the common *weighted-logit* formulation used in prior work such as MOD, the authors propose to perform **inference-time policy search** guided by a **weighted value function**. The key novelty is that the value function is *reward-based but includes a KL regularization term*, encouraging both objective satisfaction and adherence to the base model distribution. This leads to a more balanced trade-off between competing objectives without retraining or fine-tuning the LLM. Experimental results across diverse datasets, baselines, and language models show that MAVIS achieves superior Pareto fronts and improved controllability, with convincing ablation studies that isolate the contribution of each design choice.

### Strengths
- **Comprehensive empirical evaluation:** The experiments span multiple datasets, baseline methods, and backbone LLMs, clearly demonstrating that MAVIS consistently outperforms previous approaches and yields clean, interpretable Pareto fronts.  
- **Strong ablation studies:** The ablations effectively demonstrate how each design choice—value weighting, KL-regularized reward, and inference-time policy search—contributes to the observed improvements.  
- **Intuitive motivation:** The explanation of why MAVIS outperforms MOD is clear and intuitive, highlighting how weighting the *value function* instead of *logits* better aligns with downstream rewards.  
- **Principled extension:** The approach refines prior decoding-time alignment methods with subtle yet meaningful changes that improve both interpretability and empirical performance.

### Weaknesses
- **Limited theoretical novelty:** The theoretical modifications appear incremental. Earlier methods, such as Satisficing Alignment (Chehade et al., 2025), have also explored weighting value functions rather than logits.  
- **KL-based value functions are not new:** The idea of including KL regularization in the value formulation has been previously studied, for example in (Zhou et al., 2025).  
- **Positioning relative to MOD:** The introductory section does not sufficiently distinguish MAVIS from the closely related MOD framework (Shi et al., NeurIPS 2024). Clarifying the conceptual differences early would improve readability.  
- **Analytical justification:** The paper would benefit from stronger theoretical grounding or complexity analysis of the mentioned bandit-style derivation. Currently, the explanation of computational infeasibility is vague.  

### References

- Shi, Ruizhe, et al. *"Decoding-time language model alignment with multiple objectives."* NeurIPS 37 (2024): 48875–48920.  
- Chehade, Mohamad, et al. *"Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time."* arXiv preprint arXiv:2505.23729 (2025).  
- Zhou, Jin Peng, Kaiwen Wang, Jonathan Chang, Zhaolin Gao, Nathan Kallus, Kilian Q. Weinberger, Kianté Brantley, and Wen Sun. *"q♯: Provably Optimal Distributional RL for LLM Post-Training."* arXiv preprint arXiv:2502.20548 (2025).

### Questions
1. Why prefer this specific weighting of value functions over similar approaches such as Satisficing Alignment (Chehade et al., 2025) that also operate on value-based objectives?  
2. How computationally demanding is learning or estimating the value function in MAVIS?  
3. Equation (5) is described as “computationally infeasible” — could you provide an analytical interpretation or quantification of this infeasibility?  
4. Lines 84–86 mention that MOD requires fine-tuning large language models. Does MOD actually modify the model parameters, or does it remain a purely inference-time method?  
5. How does the performance of MAVIS change when the KL-based term is removed, i.e., when using a reward-only value function?  

### References

- Chehade, Mohamad, et al. *"Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time."* arXiv preprint arXiv:2505.23729 (2025).

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
5

### Summary
This paper introduces MAVIS, an inference-time alignment method that enables dynamic multi-objective control of large language models without fine-tuning. The approach trains small value models for each objective and combines them at inference time using user-specified weights to guide token selection. Experimental results on preference datasets show that MAVIS matches or exceeds the performance of baseline methods that require fine-tuning separate models for each objective, while offering greater flexibility and computational efficiency for multi-objective alignment tasks.

### Strengths
The paper addresses a meaningful problem by enabling a base model to adapt to different user preferences at inference time without extensive retraining, which provides significant practical value for deploying language models that can flexibly satisfy diverse user requirements.

### Weaknesses
1. A significant limitation is the insufficient motivation, as the core idea of using weighted multi-objective reward models or value functions to influence model logits during decoding for preference-aligned generation has been explored in prior work with similar approaches (e.g., PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model). The authors should better articulate what distinguishes their motivation and approach at a fundamental level from existing methods.

2. The experimental evaluation is insufficient as it lacks comparison with recent state-of-the-art baselines

   [1] Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment

   [2] Arithmetic Control of LLMs for Diverse User Preferences: Directional Preference Alignment with Multi-Objective Rewards

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces MAVIS — Multi-Objective Alignment via Value-Guided Inference-Time Search — a framework for aligning large language models to multiple, possibly conflicting objectives (e.g., helpfulness, harmlessness, humor) without fine-tuning. MAVIS trains small per-objective value (Q) models that estimate token-level returns under KL-regularization and combines them at inference using user-specified weights to steer decoding. The method provides a monotonic policy-improvement guarantee, integrates with beam or tree search, and empirically outperforms multi-objective fine-tuning baselines such as MORLHF, Rewarded Soups, and MOD on HH-RLHF and Summarize-from-Feedback benchmarks. It enables dynamic preference control at test-time while remaining computationally efficient compared to retraining-based alignment.

Conceptual overlap with Transfer Q★:
The core idea value-guided, KL-regularized inference-time decoding—is closely related to Transfer Q★ (Chakraborty et al., NeurIPS 2024). While MAVIS extends this to multi-objective settings with learned per-objective value models, the conceptual novelty beyond prior value-guided decoding frameworks could be clarified.

### Strengths
1. Introduces a principled method for multi-objective control of LLM behavior without modifying model weights — a clear step beyond single-objective inference-time methods.
2. Provides a formal derivation from KL-regularized policy optimization and proves monotonic improvement under iterative value updates, grounding the approach in RL theory.
3. Requires training only small per-objective value models instead of full fine-tuning, enabling dynamic preference mixing at inference and easy deployment on frozen LLMs.
4. Empirically Demonstrates superior Pareto fronts compared to MORLHF, Rewarded Soups, and MOD on HH-RLHF and Summarization Feedback datasets.

### Weaknesses
1. The evaluation focuses on numerical Pareto-front comparisons but lacks qualitative or human preference assessments. There is no demonstration of smooth or controllable trade-offs between objectives.
2. The conceptual overlap with Transfer Q★ (NeurIPS 2024).  Both rely on KL-regularized, value-guided inference-time alignment, yet the distinction in mechanism and novelty is not well-articulated.
3. The method is not generalizable, as it requires to train value model for each reward objective

### Questions
1. Could the authors clarify the conceptual distinction from Transfer Q★ and whether MAVIS can be interpreted as its multi-objective generalization?
2. Can you Compare Your method with --Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time

### Soundness
3

### Presentation
3

### Contribution
3
