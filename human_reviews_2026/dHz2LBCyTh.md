# How to Cure Newton for Unlearning Neural Networks? An Empirical Study from the Hessian Perspective

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Machine unlearning enables AI practitioners to comply with data owners' ``Right to be Forgotten'' and post-hoc filter sensitive, noisy, or malicious data from trained models. As a theoretically justified algorithm, Newton unlearning is used in previous works to rigorously unlearn selected models, eliminating the need for expensive retraining. However, we found that Newton unlearning is highly sensitive to the Hessian degeneracy phenomenon in trained neural networks, including large language models (LLMs), leading to unlearning performance degradation. To address this challenge, we propose two new unlearning algorithms, CuReNU and CuReNUS, that tackle the Hessian degeneracy in principle based on cubic regularization and discuss their convergence guarantees. As a stochastic variant of CuReNU, CuReNUS offers an efficient second-order unlearning algorithm that is applicable even to the scale of LLMs. We demonstrated that CuReNUS can achieve comparable unlearning performance to state-of-the-art empirical algorithms across diverse settings, including batch and challenging sequential unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper identifies a key limitation in second-order unlearning methods like damped Newton, where Hessian degeneracy near local optima causes unstable and excessively large parameter updates that harm performance. To address this, the authors propose Cubic-Regularized Newton Unlearning (CuReNU) and its scalable variant StoCuReNU, which apply cubic regularization to automatically control the Hessian damping factor. This approach mitigates degeneracy, stabilizes update norms, and provides theoretical convergence guarantees to second-order stationary points.

### Strengths
Unlearning is an important topic and the authors clearly explain and evaluate their proposal.

### Weaknesses
The evaluation is limited to small models or LoRA ona small LLM. Thus it is unclear how this will scale to large models where retraining is not realistic, and approximate unlearning is needed. Some parts of the evaluation technique are unclear.

### Questions
I would have appreciated some analysis of the results in Figure 2. In particular, the experiment for unlearning on Llama-2 still has 40\% accuracy $D_e$ regardless of whether StoCuReNU or retraining is used, which suggests that the samples being targeted for unlearning are not unique within the training set. As a result, it is not clear if the experiments can be conclusive in this case. 

I was also confused on how the authors obtained a retraining time measurement for Llama2-7B. If I understand this should be the time to retrain Llama2-7B from scratch minus the data in the dataset they want forgotten. However, it's more likely this is the time to fine-tune Llama2-7B on the dataset. Thus, it seems misleading to call this the retraining time for Llama2-7B since that is not how a fully exact retraining method would work.

### Soundness
2

### Presentation
3

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
This paper investigates Newton-like unlearning algorithms to address the Hessian degeneracy challenge. It proposes two methods, CuReNU and StoCuReNU, based on cubic-regularized optimization and analyzes the convergence guarantee. StoCuReNU achieves performance comparable to state-of-the-art empirical unlearning methods across diverse settings. The authors demonstrate that StoCuReNU is scalable with comparable unlearning performance for various settings,including batchand sequential unlearning.

### Strengths
This paper investigates Newton-like unlearning algorithms to address the Hessian degeneracy challenge. It proposes two methods, CuReNU and StoCuReNU, based on cubic-regularized optimization. StoCuReNU achieves performance comparable to state-of-the-art empirical unlearning methods across diverse settings. Theoretic analysis of convergence guarantees seem solid.

### Weaknesses
1. This is not the first work to propose second-order unlearning for neural networks, so the contributions appear limited. For example, Qiao et al. (2025) and Zhang et al. (2024b) also introduce second-order unlearning algorithms for non-convex objectives. The paper therefore needs stronger motivation and a clearer articulation of its novel contributions.

2. The authors compare against only a subset of existing unlearning algorithms; for instance, Qiao et al. (2025) is not included. Moreover, in the experimental evaluations, the proposed method does not perform noticeably better than the baselines, which further limits the contributions of this work.

3. The experiments use only five unlearning rounds, averaged over three random runs, which seems quite limited.

4. Accuracy is not a very convincing metric to evaluate the unlearning performance, Other common evaluation methods, such as membership inference attacks (MIA), are not considered.

### Questions
1. Since this is not the first work to propose second-order unlearning for neural networks, the authors should restate the central question more clearly: How can we unlearn neural networks effectively using second-order methods?

2. StoCuReNU claims smaller space complexity than Qiao et al. (2025); how does its time complexity compare?

3. Can lower-complexity approaches be used to approximate or invert the Hessian in CuReNU, such as Hessian–vector products or related techniques?

4. Line 375: “Tug-of-War (ToW) score that aggregates these gaps (smaller is better)” — did you mean “larger is better”?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the machine unlearning problem and proposes cubic regularized method to handle the potential problem of Hessian degeneracy. The proposed method CuReNU introduces damping to stabilize the Newton's method and have a systematic way to avoid stability issues. The authors further propose a stochastic version StoCuReNU method to bypass the $O(d^3)$ involved in matrix inversion using efficient Hessian Vector Product computations. The numerical experiments also supports the method favorably.

### Strengths
Machine unlearning is a very timely topic and Hessian degeneracy can be serious issue for computation stability. The adaptation of tools from classical nonconvex optimization (cubic regularization) into ML context is very well motivated. The usage of second order information is something the community has often overlooked and it is great to see bring brought back to the stage. The paper is technically sound, clearly presented.

### Weaknesses
The CuReNU and StoCuReNU are both adapted from existing algorithm in different setting and hence the convergene and theoretical guarantees are inherited. The authors say that " this adaptation is both necessary and non-trivial to address failure modes". Please clarify further what exactly had to be modified.

While the theory shows favorable memory usage, the empirical results in Table 4 show STOCURENU's practical peak memory can be higher than its baselines. Is this due to a large constant factor (e.g., loading the base model, LoRA adapters, and HVP buffers) 1 and that the $\mathcal{O}(2d)$ benefit is about asymptotic scaling as the dataset size $n$ grows? Is the problem too small scale for the asymptotic to kick in?

In Appendix G, the authors test on overfitted models and note that while STOCURENU is effective, a performance gap remains in Membership Inference Attack (MIA) mitigation compared to the SOTA empirical method, SCRUB. Please discuss this further. Is this gap expected for other scenarios as well or in general there is a gap?

### Questions
In Appendix J.3.1 the authors  show that $T$ represents a direct trade-off between computational efficiency and unlearning effectiveness. Could you please provide a principled heuristic in selecting $T$? Or maybe an early stopping criterion?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the challenge of machine unlearning i.e. removing the influence of specific training data from a trained neural network without retraining from scratch. The authors focus on improving second-order unlearning methods such as Newton Unlearning, which rely on Hessian information to approximate retraining. They identify a critical limitation: Hessian degeneracy (presence of many small, zero or negative eigenvalues) in trained neural networks, which leads to unstable or divergent updates during unlearning. To overcome this, the paper proposes two novel algorithms: CuReNU (Cubic-Regularized Newton’s Unlearning) – uses cubic regularization to automatically determine an appropriate damping factor, thereby stabilizing updates and it's stochastic variant StoCuReNU (Stochastic CuReNU) – a scalable, Hessian-free variant using Hessian-vector products (HVPs). Empirically, on FashionMNIST, CIFAR-10, AG-News, and TOFU datasets, CuReNU and StoCuReNU achieve competitive unlearning performance compared to state-of-the-art empirical methods (e.g., SCRUB, DELETE) in both batch and sequential unlearning settings.

### Strengths
1. The paper is very well-written, including notational consistency, and is very easy to understand.
2. The problem formulation is very clean and I believe the authors point towards and study an important problem.
3. The proposed CuReNU and StoCuReNU are derived from established optimization theory. The adaptation for unlearning is technically sound and mathematically well-justified.
4. Experiments span both batch and sequential unlearning on diverse datasets (vision, text). Metrics (accuracy, JS divergence, ToW score, etc.) are carefully chosen and clearly reported.
5. The figures and tables are clear and informative.

### Weaknesses
1. The proposed methods are direct adaptations of known optimization techniques. The novelty lies primarily in contextual application rather than new algorithms.
2. The main results focus on small to medium models (CNN, ResNet-18, LoRA-tuned LLaMA-2). Full-scale LLM unlearning remains untested.
3. Ablation studies that isolate the impact of cubic regularization vs. stochasticity are missing.
4. Discussion around more efficient Hessian vector product is missing.
5. While there are a lot of results that are included, the proposed methods are not always a clear winner.

### Questions
1. How sensitive are CuReNU and StoCuReNU to the cubic regularization coefficient 𝐿?
2. Can the authors provide empirical Hessian spectra for larger models (e.g., LLaMA-2 layers) to substantiate degeneracy at scale?
3. How does the unlearning error accumulate across rounds? Is there a mechanism to “recalibrate” the model to prevent drift after many sequential unlearning steps?
4. For large foundation models with mixed data sources, how feasible is StoCuReNU in practice?
5. Can authors include runtime comparisons at least for a subset of experiments w.r.t. baselienes?

Any discussion around these will help improve the paper over it's current state.

### Soundness
3

### Presentation
3

### Contribution
2
