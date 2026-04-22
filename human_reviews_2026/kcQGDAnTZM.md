# Surprise-Modulated Meta-Advantages in Reinforcement Learning: Towards Language-Neutral Post-Training for Code LLMs

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Large language models are more beneficial for code generation in mainstream languages such as Python and JavaScript, however, they are very ineffective for resource-constrained languages such as Fortran, OCaml, and R. We rephrase this discrepancy not as a consequence of inevitable data lack of information, but as a problem in learning efficiency. In this work, we present PolyCode, which is trained by a groupwise meta-normalised Proximal Policy Optimization (PPO) which we refer to as GMPO. GMPO is a standard PPO-clip objective that has two new additions: (i) Cross-Group Meta-Normalization (CGMN) that suppresses variance by collecting meta-statistics across prompt similarities, and (ii) Surprise-Based Advantage Modulation (SBAM) that gives preference to updates where the reward signal deviates from a relative confidence of the model. We consequently enforce language neutrality of evaluation by input and output only by binary reward r in either 0 or 1 for exact conformity, and thus avoid the need for unit test translation across languages. Empirically, PolyCode-4B always matches or significantly exceeds smaller baselines on our Ag-LiveCodeBench-X benchmark with considerable improvements over WPLL for Fortran and OCaml. For a standardised reporting, pass@1 is defined as a Monte Carlo estimate derived from multiple single-sample trials (single draw 20 times per prompt reactance at T=0.2), but the best of selection and voting were not used during implementation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents PolyCode, a multilingual Code LLM. It proposes GMPO, which is a modified PPO-like RL algorithm. Experiments with Qwen3-4B demonstrate improved performance compared to other models.

### Strengths
1. The approach is effective; experiments show improved performance.
2. The proposed GMPO may be an effective improvement for GRPO.

### Weaknesses
1. The presentation is disastrous. After reading, it isn't easy to discern the motivation and philosophy behind the proposed approach. 
2. Experiments are limited. It lacks comparison with other baselines or a necessary ablation.

### Questions
1. What is the main problem this paper tries to address? 
2. Why do the components of GMPO address the proposed problem? 
3. What is the relative performance compared to GRPO or DAPO approaches?

### Soundness
2

### Presentation
1

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
This paper addresses the performance disparity of large language models for code generation in low-resource programming languages (e.g., Fortran, OCaml, and R) by proposing a novel reinforcement learning framework, PolyCode, whose core component is an improved algorithm termed Groupwise Meta-Normalized Proximal Policy Optimization (GMPO).

### Strengths
1.	The paper explicitly targets the training imbalance problem in low-resource language scenarios, which represents a practical and significant challenge.
2.	This paper proposes a novel reinforcement learning framework, PolyCode, whose core component is an improved algorithm termed Groupwise Meta-Normalized Proximal Policy Optimization.
3.	 The experiments cover multiple programming languages (Lua, Julia, R, Fortran, and OCaml) and provide detailed reproducibility protocols, including seeds, container configurations, and compilation commands.

### Weaknesses
1. GMPO is described as a combination of “GRPO + PPO + meta-normalization,” but its distinction from GRPO (Shao et al., 2024) and the theoretical advantages are not rigorously analyzed.
2. The formulation of SBAM is largely heuristic, lacking evidence of convergence or formal theoretical guarantees.
3. Although the paper mentions that the binary reward may lead to sparse signals, it does not explore alternative reward shaping strategies or the potential complementarity between reward shaping and SBAM.
4. At times, the paper lacks clear intuitive explanations, particularly regarding the definition and effect of SBAM.

### Questions
1. When λ is large, SBAM may cause gradient explosion or unstable updates. Did the authors observe any training divergence or abnormal gradient norms? Is there an adaptive strategy for adjusting λ?
2. The similarity in CGMN is computed based on Encoder(π_old, x). Is this encoder fixed and pre-trained, or is it updated jointly during training? If it is fixed, does this limitation affect semantic generalization?
3. The paper claims that the evaluation is “language-neutral,” but differences in compilers, floating-point formats, and other language-specific factors may still introduce discrepancies. Has there been any quantitative verification of the framework’s fairness across different languages?
4. The paper demonstrates the ability to transfer from I/O-style tasks to function-style tasks. Do the authors attribute this transfer to the reward structure, the similarity of prompts, or the modulation mechanism of SBAM?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces PolyCode, a reinforcement learning framework for post-training large language models on code synthesis with a focus on language neutrality, particularly improving performance in low-resource programming languages. The method extends PPO with a new Groupwise Meta-Normalized PPO (GMPO), which integrates Cross-Group Meta-Normalization (CGMN) and Surprise-Based Advantage Modulation (SBAM). Evaluations on Ag-LiveCodeBench-X and MultiPL-E, using a language-neutral I/O setup, show that PolyCode consistently enhances performance under data-scarce conditions.

### Strengths
The paper tackles an underappreciated challenge: reducing the performance gap of LLMs in code generation for low-resource programming languages. This problem is of high practical significance.

### Weaknesses
1. The paper is dense and occasionally lapses into overly technical or informal phrasing (e.g., “greediness sophistication”, “interviewer-level interventions”, or “voltage on language-specific engineering”).
2. The experiments focus primarily on comparisons with Qwen3-4B and related LLMs, lacking evaluations against more recent RL-augmented code models or curriculum-based systems. In particular, there is no direct ablation against CodeRL-style or curriculum learning approaches for smaller models.

### Questions
1. How is the task embedding encoder ( $h_j$ ) implemented in practice? Is it frozen, shared with the backbone model, or dynamically learned? Is it language-agnostic or language-specific?
2. What is the empirical impact of the top-K neighborhood truncation in CGMN, especially under low-resource conditions?
3. Can the authors provide direct comparisons of GMPO or SBAM with CodeRL or curriculum learning methods?
4. Are there edge or failure cases where SBAM amplifies incorrect signals—for instance, overconfidence in noisy or rare reward spikes?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose GMPO, an optimization framework that combines cross-task meta-normalization with surprise-based modulation to enhance language model performance in multi-lingual code generation, particularly for low-resource programming languages such as Lua, Julia, R, OCaml, and Fortran. The core idea involves a normalization mechanism for computing baseline rewards, where normalization is performed over multiple responses sampled per prompt and further weighted across prompts from different tasks within the same batch. This design aims to reduce variance and improve learning stability for underrepresented programming languages.

### Strengths
- New optimisation technique to improve program synthesis on low-resource programming languages.
- Observable improvements in performance when models from different families, like Qwen and Phi4, are trained with GMPO.

### Weaknesses
### **1. Clarity and Writing Quality**

* Unclear notation:

  * What are R, L for which per-prompt sample statistics are being considered? Do they refer to Rewards and Log-Likelihood?
  * Encoder(π_old, x_j): Does this denote using π_old to compute the embedding of x_j or a separate encoder LLM?
  * What are μ_k and σ_k in Eq. 2 and 3? How are you defining and obtaining their values?
* Missing intuition:

  * What is the intuition behind introducing Cross-Group Meta-Normalization and Sequence Likelihood Normalization? It is unclear.
* Ambiguous training setup:

  * Are you training PolyCode separately on each language or pooling data for all languages?
* General writing issue:

  * Poor writing and less clarity.

---

### **2. Experimental Design and Missing Evidence**

* No supporting evidence such as error profiles of PolyCode and baselines when generating code in Sec. 7.4.
* Missing discussion for Figures 5, 6 and 7 — unclear what experiments they correspond to, observations, or conclusions.
* Missing Ablation on how useful Surprise-Based Advantage Modulation (SBAM) is — have you tried $A\_{j,i}^{meta}$ directly in the PPO objective?
* Per-batch statistics like batch-local softmax weights are critical, but no ablation on batch size is provided.
* No convergence plots for GMPO training; unclear sensitivity to batch size and G (responses per prompt).

---

### **3. Comparative Evaluation and Baselines**

* Missing comparison with foundation models trained with Supervised FineTuning, PPO, and GRPO. It is unclear whether the improvement is a result of more training on data for resource-constrained languages or design of GMPO itself.
---

### **4. Redundancy and Presentation**

* Redundant figures — Figures (1 & 2) and Algorithm 1 all communicate the same GMPO training.
---

### **5. Scope and Broader Evaluation**

* How good (in terms of pass@1) are closed-source models like GPT, Claude, Gemini on Lua, Julia, R, OCaml, and Fortran?
* Are there benefits from including data for popular languages (Python, Java)? What is the zero-shot performance of PolyCode on them before and after GMPO training?

### Questions
See Weaknesses Section

### Soundness
2

### Presentation
1

### Contribution
3
