# Memory Efficient Fine-Tuning of LLMs via Forward-Only Hessian-Free Coordinate Descent

- Avg Score: 5.20
- Decision: Reject
- Scores: 4, 6, 4, 8, 4

## Abstract
Fine-tuning large language models (LLMs) for specific downstream tasks has traditionally relied on memory-intensive optimizers using classical backpropagation, which demands substantial memory to store model states for gradient computation, motivating the development of memory-efficient zeroth-order optimizers that operate in a forward-only manner. However, the slower convergence of the zeroth-order optimizer remains a challenge, which recent research addresses by incorporating Hessian information to accelerate training, although storing even the diagonal Hessian requires memory equivalent to that of the model weights, leading to significant memory usage. To mitigate this problem, we propose a zeroth-order block coordinate descent (BCD)-Newton optimizer with coordinate updates adaptive to second-order information, allowing us to treat model layers as separate blocks and update only a greedily selected subset per training iteration, thereby reducing memory requirements while accelerating convergence. Specifically, at each iteration, an active set of layers is selected according to the block Gauss-Southwell-Diagonal rule, and their weights are updated while the other layers remain fixed, with compressed diagonal Hessian information stored and updated exclusively for the active layers. For fine-tuning foundation models across small to large sizes (OPT-1.3B and 30B, LLaMA-2-7B), our method achieves up to 40% memory reduction compared to existing Hessian-informed zeroth-order methods, while preserving baseline accuracy and memory usage to zeroth-order methods across various tasks, offering a memory-efficient alternative method for LLMs fine-tuning, especially on memory-constrained devices.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a forward‑only, Hessian‑aware BCD optimizer for memory‑efficient fine‑tuning of LLMs. Each transformer layer is treated as a block; at every step FOCUS selects a small subset of layers to update and stores/updates a diagonal curvature proxy only for the active blocks to avoid the O(d) memory overhead of keeping a Hessian proxy for all parameters as in HiZOO. The core estimator uses three forward passes to obtain a two‑point, preconditioned zeroth‑order update, while the embeddings and LM head are kept first‑order (MeZO‑style) for stability. A Gauss–Southwell‑Diagonal (GSD) saliency rule, approximated by a light‑weight bandit sampler, chooses which blocks to activate. Theoretical analysis gives an ergodic stationarity bound under block Lipschitz smoothness with sampling proportional to block gradient norms.

### Strengths
1. Addresses a real bottleneck: keeping second‑order benefits without paying the full O(d) memory for diagonal Hessian across all layers. 
2. The forward‑only, in‑place three‑probe loop (Algorithm 1) is consistent with MeZO practices; the block‑wise preconditioner is standard. 
3. Results on OPT‑1.3B LLaMA‑2‑7B, and OPT‑30B show FOCUS maintains MeZO‑level memory.

### Weaknesses
1. The two‑point estimator requires using the same perturbation direction $z$ for $+\mu$ and $-\mu$ evaluations; the pseudocode samples new $z$’s inside the loop and never explicitly restores parameters to $\theta$ before applying the update, which risks biasing the estimator and “drifting” the parameters (Alg. 1). 
2. Eq. (3) lacks a self‑contained derivation. The diagonal Hessian update looks adapted from NES/covariance‑score identities and uses an elementwise absolute value to ensure positivity. Without a derivation or explicit statement that $\Sigma$ is intended as an inverse‑Hessian proxy, it’s hard to assess bias/variance and stability trade‑offs. Provide a derivation or precise citation and show the scalar, per‑coordinate update actually used. 
3. For OPT‑1.3B, ZO methods use batch size 8 while FO baselines use batch size 2, which affects both throughput and memory. Some rows (e.g., BAdam bs=8 in Table 2) break that rule. 
4. LLaMA‑3‑8B table reports memory only w/ no accuracy. SQuAD’s metric is unspecified (“generation” column in Table 3). Some runtime cells are “–”. Add these to strengthen claims. 
5. The paper emphasizes GSD/bandit selection and re‑initialization of $\Sigma$ on first activation, but no ablation compares cyclic vs random vs GSD vs bandit, layers‑per‑update, or reuse vs reinit of $\Sigma$. These would isolate where the gains come from.

### Questions
1. In Algorithm 1, do you reuse the same $z$ for $+\mu$ and $-\mu$ evaluations, and do you explicitly restore $\theta$ before applying the update? Please provide the exact state‑reset logic (or code snippet) you use in practice. 
2. Is $\Sigma$ designed to approximate the inverse Hessian (so larger entries leads to lower curvature)?
3. Eq. (3) derivation: Could you supply a short derivation or precise citation connecting %\Delta L% to the diagonal update, and explain the effect of the absolute‑value operator on bias/stability?  
4. Which bandit algorithm do you use exactly? Please add the missing reference and provide the projection set $\mathcal P$ and step sizes. 
5. For OPT‑1.3B and OPT‑30B, can you report tokens/sec at matched peak memory and seed‑averaged accuracy with error bars? For LLaMA‑3‑8B, please add accuracy at the batch sizes shown in Table 5. 
6. What metric is used for SQuAD (EM/F1)? 
7. In Table 4, was LoRA BF16/FP16? Was activation checkpointing enabled? These affect the 41 GB figure. 
8. Ablations: (i) cyclic vs random vs GSD vs bandit; (ii) layers‑per‑update $={1,2,4}$; (iii) reuse vs reinit of $\Sigma$ upon re‑activation; (iv) $\mu$ and $\alpha_t$ schedules. These would localize the method’s gains.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose FOCUS, a method for fine-tuning large models with little memory requirements. The method uses only forward passes (as in MeZO), and leverages second order information (as in HiZOO) while adressing the memory issues of such methods by training only one layer at a time. FOCUS is evaluated on different fine-tuning settings with models ranging from 1.3 to 33 billion parameters, and is shown to outperform MeZO in terms of downstream accuracy with comparable memory and time requirements.

### Strengths
- The paper is well written and easy to follow.
- The method is well motivated, with GCD appearing as a natural idea for solving the memory overhead introduced by HiZOO.
- FOCUS achieves better convergence rates than MeZO, while requiring a similar amount of memory
- Experimental evaluation involves multiple models at different scales: OPT-1.3B, LLaMA-2-7B, LLaMA-3-8B, and OPT-30B.

### Weaknesses
- The BCD algorithm used to select the layer to update (a bandit method adapting the Gauss-Southwell-Diagonal rule) is not well motivated. How does it compare to other strategies, such as training layers in order, or with uniform distribution? Maybe an ablation study could help.
- Forward only methods still suffer from poor runtime compared to other methods: from Table 2, LoRA takes 55s to achieve better accuracy than FOCUS which took 51min (so about 55x slower).

Minor:
- The bold values in Table 3 are misleading: for CB HiZOO is better, for boolQ nothing is in bold, for WSC only FOCUS is in bold but all methods have the same accuracy, etc. Could the authors clarify what the bold values represent?

### Questions
- I struggle to understand why, in Tables 1 (with RTX 4090) and 2, FOCUS uses less memory than MeZO, and is even faster? Since FOCUS is based on MeZO I would expect at least the same cost. In particular, FOCUS uses 3 forward passes instead of 2 for MeZO, so I would have expected a higher memory and computation time.
- When selecting a new layer to be trained, you use the identity as an initial guess for the (diagonal) Hessian. However, when a layer has already been trained before, there already was a more refined estimation of the Hessian, but it was discarded to save memory (this is key to reducing HiZOO's memory). Could we imagine instead reusing this previous estimation, for instance by saving it on the disk until the layer is selected again? This way, no additional GPU would be required and the method would benefit from a better diagonal Hessian estimation.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FOCUS, a novel optimizer for memory-efficient fine-tuning of large language models (LLMs). FOCUS integrates a forward-only zeroth-order gradient estimation with block coordinate descent (BCD) and Hessian-informed (second-order) preconditioning. Instead of performing backpropagation, the method perturbs model parameters in a forward-only manner and updates only a subset of layers (blocks) per iteration. A diagonal approximation of the Hessian is maintained for active blocks, while a bandit-based Gauss–Southwell–Diagonal (GSD) rule probabilistically selects layers for update. Experiments on OPT-1.3B/30B, LLaMA-2-7B, and LLaMA-3-8B demonstrate up to 40% memory reduction compared to HiZOO, with comparable or faster convergence.

### Strengths
1.This paper proposes a memory efficient Zeroth-order optimizer for LLMs fine-tuning.By integrating Hessian-informed preconditioning with block coordinate descent,the optimizer can achieve similar accuracy to HiZOO but with lower memory usage and better convergence than MeZO.

2.By using block coordinate descent,the number of parameters updated per iteration has been much fewer than other methods like HiZOO, which also leads to reduced runtime.This makes FOCUS practical in LLMs fine-tuning.

### Weaknesses
1.The method shows limited novelty.This paper just simply combines block coordinate descent and Hessian-informed preconditioning, without new theory.

2.The experiments are not sufficient enough.The experiments are conducted on only three models , and there is only OPT-1.3B trained on multiple downstream tasks.

3.The reproducibility of the paper is debatable due to the lack of released code.

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
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes **FOCUS (Forward-Only Coordinate Updates with Second-Order Information)**, a *memory-efficient zeroth-order optimizer* for fine-tuning large language models (LLMs).  
It introduces a **Block Coordinate Descent (BCD)–Newton optimization scheme** that updates only a subset of layers per iteration, leveraging **compressed diagonal Hessian information** without storing full gradients or activations.

Traditional fine-tuning requires memory-heavy backpropagation, while recent forward-only (zeroth-order) optimizers like MeZO and HiZOO reduce memory cost but suffer from slow convergence or large Hessian storage overhead.  
FOCUS addresses these issues by:
1. Dividing model parameters into *blocks* (layers).
2. Selecting active blocks using a **Gauss–Southwell–Diagonal (GSD)** rule.
3. Updating only these layers with a Hessian-informed coordinate descent, while other layers remain frozen.

Key features include:
- **Forward-only optimization** (no backpropagation).
- **Hessian-free second-order adaptation** using compressed diagonal preconditioning.
- **Bandit-based probabilistic layer selection** for efficient block updates.

Empirical evaluation on **OPT-1.3B/30B**, **LLaMA-2-7B**, and **LLaMA-3-8B** shows:
- Up to **40% memory reduction** compared to HiZOO.
- Comparable or better accuracy than MeZO with faster convergence.
- Scalability to **OPT-30B** with 92.9–93.6% accuracy on SST-2 using 2–8 A100 GPUs.

Overall, the paper contributes a theoretically grounded and empirically validated framework for memory-efficient fine-tuning of large models on constrained devices.

### Strengths
1. **Methodological novelty:** Combines BCD, diagonal Hessian updates, and bandit-based layer selection.  
2. **Significant memory savings:** Achieves 30–40% reduction versus HiZOO, approaching MeZO-level efficiency.  
3. **Improved convergence:** Matches HiZOO’s speed while using less memory (see *Figure 2, page 8*).  
4. **Robust evaluation:** Benchmarks on GLUE/SuperGLUE tasks and across multiple model scales (OPT-1.3B → OPT-30B).  
5. **Scalability demonstrated:** Successful fine-tuning of 30B models on limited GPUs.  
6. **Theoretical soundness:** Formal convergence analysis and probabilistic block sampling guarantee stability.  
7. **Reproducibility:** Clear algorithms, hyperparameter tables (Appendix C), and implementation details (Appendix B).

### Weaknesses
1. **Limited exploration of stochasticity effects:**  
   The randomness in block sampling and Hessian approximation may lead to training variance, but no sensitivity study is provided.

2. **Overemphasis on memory metrics:**  
   While memory reduction is clear, the runtime and computational trade-offs (especially on multi-GPU setups) deserve deeper discussion.

3. **Restricted task diversity:**  
   Experiments are primarily classification tasks; more diverse NLP or multimodal settings (e.g., summarization, reasoning) would strengthen generality.

4. **No ablation on bandit vs. deterministic layer selection:**  
   It is unclear how much the bandit mechanism contributes versus static cyclic BCD.

5. **Minor reproducibility concerns:**  
   Some details (e.g., random seed management and number of forward passes per iteration) could be clarified for precision.

6. **Presentation density:**  
   Technical sections could better balance math with conceptual interpretation.

Overall, these are **non-critical limitations** that do not undermine the core contributions.

### Questions
1. How sensitive is the performance to the *block partitioning granularity* (e.g., per-layer vs. sub-layer)?  
2. Does the stochastic bandit-based selection introduce instability across different seeds or training runs?  
3. Can FOCUS be combined with parameter-efficient tuning (e.g., LoRA) for hybrid benefits?  
4. How does the method scale on extremely large models (≥70B) under mixed-precision training?  
5. Is there a theoretical link between your bandit selection and the importance-weighted Gauss–Southwell rule?  
6. Have you explored adaptive smoothing radii (µ) to improve curvature estimation stability?  
7. Could future work extend this to multimodal or reinforcement learning settings?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes FOCUS — a forward-only, zeroth-order (ZO) block coordinate descent (BCD) Newton optimizer for memory-efficient fine-tuning of LLMs. The key idea is to (i) partition parameters by layer, (ii) update only a subset of layers per step chosen via a Gauss–Southwell-Diagonal score with a bandit-style sampler, and (iii) keep a diagonal Hessian preconditioner only for the active blocks, thereby avoiding the large memory overhead of storing second-order statistics for the entire model. 
Embedding and LM-head layers are updated with MeZO-style ZO without second-order terms to avoid instability. Experiments on OPT and llama models across GLUE/SuperGLUE tasks report up to ~40% memory reduction vs. Hessian-informed ZO baselines while matching or surpassing their accuracy and improving wall-clock efficiency.

### Strengths
1. **Clear, practical objective**: Reduce second-order ZO memory overhead by storing diagonal Hessian only for currently active blocks, selected with a principled GSD rule; the scheme is well-motivated for low-memory settings. 
2. **Forward-only training pipeline**: Retains MeZO-level memory while gaining HiZOO-like curvature awareness, addressing ZO’s slow convergence without backprop activations. 
3. **Compelling empirical evidence**: Concrete GPU memory tables and wall-clock comparisons on OPT and llama models.
4. **Method details & theory**: Pseudocode, bandit-based layer selection, and a summarized convergence result for randomized GS-BCD.

### Weaknesses
1. **Batch-size and fairness controls**: Some reported setups use different batch sizes across FO/ZO families (e.g., ZO bs = 8 vs FO bs = 2 in OPT-1.3B SST-2), which can influence both memory and final accuracy; please normalize or include sensitivity analyses. 
2. **Ablation depth**: Multiple moving parts (block count **D**, **GSD** vs. random selection, **bandit** hyper-parameters, Hessian re-initialization policy, **#active layers per step**) are not fully disentangled with granular ablations/SEs. 
3. **Broader baselines**: Since the method is conceptually related to **blockwise full-gradient training**, comparisons against **BAdam** and **LiSA** would strengthen the case (even if they are FO), at least on equal memory budgets.
4. **Memory accounting granularity**: The paper makes a strong case for end-to-end memory savings; still, a breakdown (parameters, activations, Hessian buffers, fragments) across methods and models would clarify where the savings come from and when they vanish. 
5. **Task scope**: GLUE/SuperGLUE are convenient for controlled measurements, but additional *instruction-tuning* or *reasoning* tasks (MT-Bench, GSM8K) would increase external validity—especially because FO baselines like **LiSA** highlight strong downstream MT-Bench gains at low memory.

### Questions
1. **Ablations**: Could you report factorized ablations for *(a)* #active layers per step, *(b)* selection rule (GSD vs. random/cyclic), *(c)* bandit hyper-parameters (α, p_min), and *(d)* Hessian re-init frequency?
2. **Fairness & scaling**: Can you provide a *fixed-batch* comparison (same global batch, same precision) to isolate algorithmic gains from batch/precision differences?
3. **Cost profile**: What is the **overhead** of computing GSD scores and bandit updates per step, as a fraction of step time, for 7B/30B scales?
4. **Stability**: Any failure modes when repeatedly re-selecting the same blocks (e.g., catastrophic forgetting in inactive blocks)?
5. **Generalization**: Have you tried instruction-tuning/QA (e.g., MT-Bench, GSM8K) and multi-seed CIs to demonstrate robustness beyond GLUE/SuperGLUE?

### Soundness
2

### Presentation
3

### Contribution
2
