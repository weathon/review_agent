# AlignFlow: Improving Flow-based Generative Models with Semi-Discrete Optimal Transport

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Flow-based Generative Models (FGMs) effectively transform noise into complex data distributions. Incorporating Optimal Transport (OT) to couple noise and data during FGM training has been shown to improve the straightness of flow trajectories, enabling more effective inference. However, existing OT-based methods estimate the OT plan using (mini-)batches of sampled noise and data points, which limits their scalability to large and high-dimensional datasets in FGMs. This paper introduces AlignFlow, a novel approach that leverages Semi-Discrete Optimal Transport (SDOT) to enhance the training of FGMs by establishing an explicit, optimal alignment between noise distribution and data points with guaranteed convergence. SDOT computes a transport map by partitioning the noise space into Laguerre cells, each mapped to a corresponding data point. During FGM training, i.i.d. noise samples are paired with data points via the SDOT map. AlignFlow scales well to large datasets and model architectures with negligible computational overhead. Experimental results show that AlignFlow improves the performance of a wide range of state-of-the-art FGM algorithms and can be integrated as a plug-and-play component. Code is available at: https://github.com/konglk1203/AlignFlow.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
AlignFlow introduces semi-discrete optimal transport (SDOT) as a deterministic way to align each noise-data pairs to ensure more straight flow paths in flow-based generative models (FGMs) training, reducing Number of Function Evaluations (NFE).

A dual-weight vector defines the transport plan between noise and data, corresponding to Laguerre cells that partition the noise space. Computes dual-weight vector by using Adam optimizer in maximizing an objective from the dual problem.

Training proceeds in two stages: (i) compute the SDOT map once (extra cost < 1 %), and (ii) train any FGM while re-using these fixed (noise, data) pairs .

Deterministic alignment yields straighter generative paths, fewer NFE, and bypasses the sample-complexity curse.

Plug-and-play integration with Flow-Matching, Shortcut Models, MeanFlow, etc. Improves FID across CIFAR-10 and ImageNet-256 at low NFE.

### Strengths
The paper is written clearly. It explains the problem settings, methodology, and contributions quite well.

Uses SDOT to build a deterministic coupling is a novel yet simple idea that avoids high-dimensional OT pitfalls. Similar ideas are rarely explored and has only one contemporary work published in 2025.

Demonstrated improvement across model sizes and datasets to imply the generality of the algorithm. Wall-clock times for experiment runs on CIFAR and ImageNet are reported to support claims for SDOT cost.

### Weaknesses
1. Although recommendations for setting SDOT hyper-parameters ($\epsilon$, lr, EMA β) is discussed qualitatively to help practitioners use the algorithm, sensitivity to hyperparameters is not discussed in the experiments. If the algorithm is to be further scaled, existing practices for choosing hyper-parameters could potentially fail.
2. Experiments are done on a rather limited data set size; scalability to larger datasets or infinitely sampled settings (e.g., synthetic data augmentation) is unclear. Paper mentioned that processing time is 8min 30s for CIFAR10, and that the algorithm scales quadratically w.r.t. number of targets. Extrapolating to industrial-sized datasets with typically millions of samples, the pre-processing could be astoundingly time-consuming.
3. The difference of FID-50k with and without AlignFlow for larger models and better FID is quite small according to Table 3. Even though AlignFlow improves FID across sizes, the improvement is not very significant. Perhaps the authors should consider using a new evaluation task or metric to highlight improvement.

### Questions
1. The paper states that "AlignFlow particularly suitable for large models that require small-batch training". This seems to weaken the contribution of this paper, as methods such as [1] works on small-batch training and [2] works on larger models already.

2. Could authors explain the reasons behind excluding discussions and experiments of AlignFlow on continuous normalizing flows (CNFs)? Demonstrating benefits there would strengthen the claim of generality.

3. Could the SDOT map be updated online to accommodate new data without recomputing from scratch?

##### Reference

[1] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid RectorBrooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport.

[2] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling.

### Soundness
3

### Presentation
3

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
The paper proposes AlignFlow, a two-stage recipe for FGM training: (1) precompute a semi-discrete OT (SDOT) map from continuous noise to the empirical data distribution by partitioning the noise space into Laguerre cells; (2) train any flow-matching-style model while pairing each sampled noise with its SDOT-assigned datum. The method is “plug-and-play,” deterministic (fixed coupling), and reported to (a) speed convergence and (b) improve FID across several FGM families (Flow Matching, Consistency/Shortcut, Live Reflow, MeanFlow) on CIFAR-10 and ImageNet256 (latent). The paper also argues that SDOT avoids the OT sample-complexity “curse of dimensionality” by optimizing against p_1 rather than the unknown population \tilde p_1.
	​

### Strengths
(1) Simple, general mechanism: drop-in coupling that works with multiple FGM targets (FM, Shortcut, MeanFlow, etc.). Algorithm 3 is clear.
(2) Practical engineering: seed-based noise storage; class-wise SDOT; lightweight Stage-1 overhead; handy MRE/L1 diagnostics and tuning guidance.
(3) Empirical gains: better FID at fixed NFE across multiple backbones; improvements persist from U-Net/CIFAR10 to DiT/SiT on ImageNet256. Tables 2–4 & Fig. 2.

### Weaknesses
(1) The paper acknowledges no assumption about SDOT ≈ population OT, but several claims and intuitions (e.g., straighter, “more optimal” paths) implicitly rely on population properties.
(2) Some parts of the presentation are unclear, e.g. the algorithm description.

### Questions
(1) Does the method resample noise every epoch?

### Soundness
3

### Presentation
3

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
The paper proposes AlignFlow, a method that calculates semi-discrete optimal transport (SDOT) between the source and the data distributions prior to the training of a flow-based generative model. It further uses the calculated SDOT plan as coupling in order to minimize the curvature of the resulting sampling trajectories.

### Strengths
The paper is well-written. The motivation is clear. The proposed approach is simple and can be used as a plug-and-play method to improve flow-based models. AlignFlow consistently improves sampling efficiency and convergence over the baselines.

### Weaknesses
1. The paper lacks discussion to some prior work that also attempted to improve minibatch-based OT coupling in flow-based generative models (see [1, 2, 4]).
2. The paper would benefit from additional analysis of how scalable the Algorithm 2 is with respect of the data size. Besides this, some additional details on how efficient the OT plan calculations are would be appreciated (Equation 5 and steps 6 and 8 in Algorithm 2).
3. The method assumes the dataset is available as whole prior to the training. However, modern generative models trained at scale sometimes deal with streaming data that is not available before the training. Could the authors provide some discussion on this?
4. While the paper suggests a way to deal with class-conditional training, it is not clear how it would extend to more complex conditioning signals, such as text. [3] propose a solution toward this. Could the authors provide some discussion whether this or similar techniques could be applied to AlignFlow?

[1] Davtyan, Aram, et al. "Faster inference of flow-based generative models via improved data-noise coupling." The Thirteenth International Conference on Learning Representations. 2025.

[2] Zhang, Stephen, et al. "On fitting flow models with large sinkhorn couplings." arXiv preprint arXiv:2506.05526 (2025).

[3] Cheng, Ho Kei, and Alexander Schwing. "The curse of conditions: Analyzing and improving optimal transport for conditional flow-based generation." arXiv preprint arXiv:2503.10636 (2025).

[4] Calvo-Ordonez, Sergio, et al. "Weighted conditional flow matching." arXiv preprint arXiv:2507.22270 (2025).

### Questions
1. At the first glance the optimization problem for rebalacing in Equation 12 has a large search space. Could the authors provide more details on how this is implemented in practice?

Some typos:
1. Line 131. The sentence seems to miss a word.
2. Line 208. smaple $\rightarrow$ sample.
3. Line 398. I believe Semi-Discrete Optimal Transport was meant instead of Stochastic Denoising Optimal Transport.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper AlignFlow: Improving Flow-Based Generative Models with Semi-Discrete Optimal Transport proposes a method to enhance flow-based generative models (FGMs) by explicitly aligning noise and data samples through Semi-Discrete Optimal Transport (SDOT). Traditional FGMs sample noise and data independently, leading to curved generative trajectories and high computational cost. AlignFlow introduces a two-stage training process: first computing a deterministic SDOT map that partitions the noise space into Laguerre cells matched to data points, and then training the FGM using these fixed correspondences. This approach bypasses the curse of dimensionality inherent to standard Optimal Transport methods, scales efficiently to large datasets, and adds negligible computational overhead. Experiments on CIFAR-10 and ImageNet show that AlignFlow consistently improves generation quality and convergence speed across multiple state-of-the-art models—including Flow Matching, Shortcut Models, and MeanFlow—while maintaining compatibility as a plug-and-play module for modern generative frameworks.

### Strengths
- The paper presents a new and relevant method to pair data and noise samples when training FGMs. The proposed method is similar in spirit to mini batch OT, but instead of computing the coupling between the empirical data distribution and an empirical version of a Gaussian distribution, it computes the coupling between the empirical data distribution and the Gaussian, using Semi-discrete OT.

- The experimental results show an improvement over the training performance of mini batch OT, and the proposed method is faster as well.

### Weaknesses
As in minibatch OT, it is unclear how the proposed approach can be adapted to settings like text-to-image, where we have one or few samples per “class”/prompt, as the plan must have Gaussian marginal for each class/prompt, but this is statistically hard to enforce when there are few samples. Yet, most large scale FGM are text-to-“modality” instead of class-based. To my understanding, this limits the applicability of the method. Could the authors comment on this? Have they found a way to apply their method in text-to-“modality” settings?

### Questions
The authors emphasize that unlike for minibatch OT, for which the OT plan is subject to the curse of dimensionality, semi-discrete OT bypasses the curse of dimensionality. However, that is with respect to the OT plan between $p_0$ and the empirical distribution $\tilde{p}_1$. However, arguably the OT plan that we want to be close to is the one between $p_0$ and the population distribution $p_1$, and I assume semi-discrete OT is still curse wrt this one. Is that so? If so, what are the advantages of not being cursed for the plan between $p_0$ and the empirical distribution $\tilde{p}_1$?

### Soundness
4

### Presentation
3

### Contribution
3
