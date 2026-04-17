# FZOO: Fast Zeroth-Order Optimizer for Fine‑Tuning Large Language Models towards Adam‑Scale Speed

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 4

## Abstract
Fine-tuning large language models (LLMs) often faces GPU memory bottlenecks: the backward pass of first-order optimizers like Adam increases memory usage to more than 10 times the inference level (e.g., 633~GB for OPT-30B). Zeroth-order (ZO) optimizers avoid this cost by estimating gradients only from forward passes, yet existing methods like MeZO usually need tens of times more steps to converge. Can this trade-off between speed and memory in ZO be fundamentally improved? Normalized-SGD, for instance, demonstrates strong empirical performance with greater memory efficiency than Adam. In light of this, we introduce FZOO, a Fast Zeroth-Order Optimizer towards Adam-Scale Speed. On the one hand, FZOO reduces the total forward passes needed for convergence by employing batched one-sided estimates that adapt step-sizes based on the standard deviation of batch losses. On the other hand, it accelerates per-batch computation through the use of Rademacher random vector (±1) perturbations, which also enables further speedups through batched evaluation. Extensive experiments on diverse models (including RoBERTa-large, the OPT family (350M-66B), Phi-2, and Llama3) across 11 varied downstream tasks validate FZOO's effectiveness. On average, FZOO outperforms MeZO by +3% in accuracy while requiring 3$\times$fewer forward passes. Notably, for the RoBERTa-large model, FZOO achieves average improvements of +5.6% in accuracy and 18$\times$reduction in forward passes compared to MeZO, achieving convergence speeds comparable to Adam. We also provide theoretical analysis proving FZOO’s formal equivalence to a normalized-SGD update rule and establishing its convergence guarantees. Beyond full-parameter tuning, FZOO plugs smoothly into PEFT techniques, unlocking even larger memory savings. Taken together, our results make single-GPU, high-speed, full-parameter fine-tuning realistic today and point toward future work on memory-efficient pre-training. Code: https://github.com/DKmiyan/FZOO

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a new zeroth-order optimizer leveraging the idea from normalized-SGD. In comparison to the zeroth-order optimizer of MeZO,  the authors propose the following changes: (1) Generate the perturbation using Rademacher random vectors instead of a Gaussian distribution; (2) Average multiple one-sided difference estimates instead of using one two-sided difference estimate; (3) Normalize the gradient its estimated standard deviation; (4) Enable efficient implementation via batched forward pass.

The authors show that the proposed method can achieve better performance with a greatly improved the convergence rate.

### Strengths
1. The paper is generally well-written, clear, and, easy to follow. The proposed method is elegant, simple, but effective.

2. FZOO shows sizable improvements over MeZO in both convergence speed and model quality.

3. The authors conduct extensive experiments to support the effectiveness of the proposed method.

### Weaknesses
1. This paper doesn't have an extensive ablation study to test the effectiveness of each proposed change. (The ablation study only tests the effect of N, the number of perturbation directions).

### Questions
1. All the files under https://anonymous.4open.science/r/FZOO-5927 show the error message of "The requested file is not found.".

2. This paper doesn't include an extensive ablation study. In particular, the reviewer wonders which change contributes the most for improving the convergence speed. It seems the normalization step is the most important one. Does normalization + Gaussian perturbation work? Could the authors run further ablation studies to test the effectiveness of each proposed change?

3. It seems that averaging multiple estimates naturally enables using one-sided difference estimates. Have the authors tested the difference between averaging over N one-sided estimates v.s. averaging over N/2 two-sided estimates?

4. Could the authors elaborate a bit more on why batching recovers little speed-up for MeZO?

5. It would be useful to also compare against Addax (https://openreview.net/forum?id=QhxjQOMdDF).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper studies the memory and convergence tradeoff in zeroth-order optimization. The paper proposes FZOO (Fast Zeroth-Order Optimizer), a novel optimization method designed to fine-tune large language models with inference-level memory usage while achieving Adam-like convergence speed. The method combines batched Rademacher random perturbation with adaptive learning rate to mimic normalized-SGD (and Adam) behavior. The paper shows that the proposed algorithm converges in $O(\sqrt{d}/\sqrt{T})$ by properly choosing learning rate and perturbation stepsize.

Numerical results shows that the proposed method outperforms existing MeZO method in both final accuracy and convergence speed.

### Strengths
1. The paper proposed a novel approach to compute the activations for the perturbed model with Rademacher perturbation. The proposed method speeds up the loss computation.
2. The paper proposed using the batched loss variance to normalize the stepsize, which enables accelerated convergence speed.
3. The paper demonstrated promising numerical results showing FZOO outperforms MeZO and HiZOO in multiple settings.

### Weaknesses
1. Missing discussion on the effect of normalization with estimated variance. The author should further compare the convergence results with and without normalization. The current discussion in section 3.4 is insufficient to demonstrate the usefulness of the normalization. As we know, normalized SGD actually fails to converge to a stationary solution in specific problems.

2. Missing reference to existing ZO algorithms that use structured/directional perturbation. E.g., [R1-R4]. The proof technique is standard for ZOO with either Normal or Rademacher random perturbation.

3. The paper should compare the proposed FZoo method with other structural perturbation methods or variance-reduced methods.

[R1] Belouze, G. Optimization without backpropagation, 2022. URLhttps://arxiv.org/abs/2209.06302.

[R2] Rando M, Molinari C, Villa S, Rosasco L. Stochastic zeroth order descent with structured directions. Computational Optimization and Applications. 2024 Dec;89(3):691-727.

[R3] Ma S, Huang H. Revisiting zeroth-order optimization: Minimum-variance two-point estimators and directionally aligned perturbations. In The Thirteenth International Conference on Learning Representations 2025.

[R4] Shao, W., Albayrak, S. (2023). Adaptive Zeroth-Order Optimisation of Nonconvex Composite Objectives. In: Nicosia, G., et al. Machine Learning, Optimization, and Data Science. LOD 2022

### Questions
Please address the above weakness, especially the comparison with other structural perturbation methods.

Also, in the equations on page 20, I think some expectations are missing.

The final steps on page 21 (from line 1904 to lines 1104) do not look correct to me. I think a V term is missing in the first term, and thus, the main theorem 3.6 is incorrect. 

By having the term $\sigma_*/\sqrt{(\bar{\sigma}_t^{-2})^{1/2}} \geq 1$, it seems like the normalization is slowing down the convergence?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces FZOO, a zeroth‑order optimizer for LLM fine‑tuning that (i) uses batched one‑sided function evaluations and an adaptive step size normalized by the batch loss standard deviation, linking its update to normalized‑SGD, (ii) accelerates each step via Rademacher (±1) perturbations that enable efficient batched parallelism. Experiments show FZOO typically beats MeZO by ~+3% accuracy while using ~3× fewer forward passes across various datasets and tasks. On RoBERTa‑large reports Adam‑like convergence at inference‑level memory, with accompanying proofs of normalized‑SGD equivalence and convergence guarantees.

### Strengths
1. FZOO brings normalized‑SGD’s normalization into the ZO regime by scaling updates with the batch loss standard deviation and using one‑sided estimates, making ZO steps both more stable and step‑efficient.

2. The paper proves a formal link to normalized‑SGD and a convergence guarantee under standard smoothness/variance assumptions

3. Rademacher (±1) perturbations enable per‑layer sign‑flip/add operations and batched parallelism, yielding speed‑up.

4. Strong empirical gains across scales and tasks, and Inference‑level memory footprint

### Weaknesses
1. 3.3 argues Rademacher allows “bit‑level” sign flips so additions replace multiplies, there is no roofline or kernel profile to substantiate “addition beats multiply” benefits at scale.

2. The paper compares to prefix‑tuning, but omits head‑to‑head QLoRA/LoRA+Adam under matched memory/throughput budgets, which are widely used and could potentially alter the claimed end‑to‑end efficiency picture.

### Questions
1. Theorem 3.6 requires d framed as “parameter dimension”, what is the correct setting?

2. The advertised 1.92× speedup (OPT‑125M, N=8) compares against an 8‑perturbation sequential baseline, realistic strong baselines like MeZO with N=1 or two‑sided ZO with efficient batching. what causes this inflation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FZOO, a fast zeroth-order optimizer designed to reduce GPU memory in LLM fine-tuning. FZOO employs two primary strategies: it uses batched one-sided estimates to adapt its step-sizes based on loss variance and it utilizes Rademacher random-vector perturbations to accelerate per-batch computation. The authors demonstrate FZOO's effectiveness with experiments and provide theoretical convergence guarantees.

### Strengths
1. FZOO introduces a novel implementation that estimates more accurate zeroth-order gradients using batched one-sided estimates and Rademacher random-vector perturbations. This approach eliminates the two forward passes required by MeZO.
2. In experiments, FZOO demonstrates both improved speed-ups and higher model quality.
3. Similar to the original MeZO, FZOO is orthogonal to PEFT techniques.

### Weaknesses
1. I am unclear on the convergence results in Theorem 3.6 due to an apparent inconsistency in the definition of $d$ . The main text defines $d$ as the total number of model parameters; however, Appendix H. 1 clarifies that $d$ represents the per-layer input width, not the total parameter count. It is surprising that the zeroth-order method's convergence rate would be independent of the total number of model parameters.
2. The paper do not discuss and compare with an important, closely related line of work: hybrid first-order (FO) and zeroth-order (ZO) methods (e.g., [1,2]). A notable example, Addax [1], which adaptively combines FO and ZO gradient estimations based on input sequence length, is particularly relevant and warrants comparison.

[1] Li, Zeman, et al. "Addax: Utilizing zeroth-order gradients to improve memory efficiency and performance of sgd for fine-tuning language models." arXiv preprint arXiv:2410.06441 (2024).

[2] Chen, Jiahe, and Ziye Ma. "VAMO: Efficient Large-Scale Nonconvex Optimization via Adaptive Zeroth Order Variance Reduction." arXiv preprint arXiv:2505.13954 (2025).

### Questions
1. Could the authors provide results for Adam fine-tuning in their experiments to serve as a baseline comparison?
2. Would it be possible for the authors to include an experimental comparison with Addax [1]? Given that Addax is a prominent hybrid FO-ZO method, this comparison would provide valuable context for the proposed approach.
3. Could the authors provide a more detailed explanation of the term $d$ in their theoretical results? Specifically, clarification is needed on why  $d$ in Theorem 3.6 denotes the per-layer input width rather than the total number of model parameters, as the latter is a more common definition in this context.

I would be willing to reconsider my score if these concerns are satisfactorily addressed.

### Soundness
2

### Presentation
3

### Contribution
2
