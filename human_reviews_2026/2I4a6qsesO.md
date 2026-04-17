# Tight Bounds for Schrodinger Potential Estimation in Unpaired Data Translation

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Modern methods of generative modelling and unpaired data translation based on Schrodinger bridges and stochastic optimal control theory aim to transform an initial density to a target one in an optimal way. In the present paper, we assume that we only have access to i.i.d. samples from the initial and final distributions. This makes our setup suitable for both generative modelling and unpaired data translation. Relying on the stochastic optimal control approach, we choose an Ornstein-Uhlenbeck process as the reference one and estimate the corresponding Schrodinger potential. Introducing a risk function as the Kullback-Leibler divergence between couplings, we derive tight bounds on the generalization ability of an empirical risk minimizer over a class of Schrodinger potentials, including Gaussian mixtures. Thanks to the mixing properties of the Ornstein-Uhlenbeck process, we almost achieve fast rates of convergence, up to some logarithmic factors, in favourable scenarios. We also illustrate the performance of the suggested approach with numerical experiments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies tight generalization bounds for the empirical risk minimizer within a class of Schrödinger Bridge (SB) potentials. The authors focus on a fundamental yet widely studied setting where the risk functional is the KL divergence and the underlying dynamics follow an Ornstein–Uhlenbeck (OU) process. Under a set of explicit assumptions (Assumptions 1–5), they establish a fast convergence rate of order $O(\log^3 (n) / n$, which significantly improves upon prior results of order ($O(1\sqrt{n}$). As a practical instantiation, the paper adapts the LightSB algorithm to the OU reference process and presents experiments on Gaussian mixture models (GMMs) and single-cell datasets.

### Strengths
- The primary contribution lies in deriving a notably tighter generalization bound by adopting the OU reference process. The improved rate represents a substantial theoretical advance over previous works.

- The paper is well-structured and accessible, providing clear explanations of both the strengths and limitations of the proposed analysis. The authors carefully justify each assumption, discussing its practical implications and arguing convincingly that these assumptions are reasonable or attainable in practice.

### Weaknesses
While the theoretical contribution is valuable, my main concerns lie in the practical validation and experimental analysis.

- This is my main concern. The empirical section does not quantitatively assess the gap between the empirical and ground-truth SB potentials with respect to the number of samples $n$. Such analysis would strengthen the connection between theory and practice. It would be particularly insightful to evaluate this gap in the case of Gaussian marginals, where the exact SB potential is analytically tractable. Furthermore, constraining the neural network parameters to satisfy the boundedness and growth conditions (related to constants $L$ and $M$) should be discussed or enforced. A direct comparison with the Brownian-motion-based LightSB (with fixed number of samples $n$) should be discussed. Finally, the influence of the time horizon on the convergence rate should be investigated.

- In Table 2, the results focus solely on distributional alignment. It would be more comprehensive to also report transport cost metrics, as well as other measures such as the FID and transport cost in image-to-image translation tasks.

- The parametrization of the exponential SB potential through a GMM may not be practical for high-dimensional or complex data. In many applications, algorithms estimate the gradient of the log potential (i.e., the control function) rather than the potential itself.


Overall, I find the theoretical result strong and meaningful. With additional experiments addressing these points (especially the first bullet point), I would be inclined to raise my evaluation score.

### Questions
- Could the authors discuss the practical gap between the assumptions required by the theorem and the actual behavior of the LightSB implementation?

- Could the authors provide an intuitive explanation for why the OU process yields a tighter bound? Specifically, since the required time horizon  $T$ is roughly proportional to $1/b$, a larger $b$ would make the terminal reference distribution more Gaussian-like, and the joint distribution of the reference dynamics nearly independent (memoryless), as discussed in [1]. Is this correct? Could author provide the better intuition?

- Is it possible to extend the theoretical discussion to convergence guarantees for the gradient of the potential function? If so, are there existing works addressing this direction?

Reference

[1] Domingo-Enrich et al., Adjoint Matching.

### Soundness
2

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
3

### Summary
In this paper, the authors introduce novel bounds for the estimation of the Kullback-Leibler divergence between a candidate distribution defined by its potential and the Schrodinger Bridge target. More precisely, they show that under strong assumptions on both the potential and using a Ornstein-Uhlenbeck reference dynamics they can provide a non-asymptotic upper bound on the empirical risk minimizer and the target Schrodinger Bridge. The bound is of order $O(1/n)$ where $n$ is the number of data points used in the approximation and $O(d)$ (up to logarithmic terms in both cases). Some of the technical lemmas seem to be inspired by [2]. In addition, of this main theoretical contribution, the authors consider the procedure of [1] but instead replace the Brownian reference motion by a Ornstein-Uhlenbeck process. In that case, they present experimental results on several benchmarks. 

[1] Korotin et al. (2024) -- Light Schrodinger bridge

[2] Puchkin et al. (2025) -- Sample complexity of Schrodinger potential estimation

### Strengths
* The paper seems technical strong. In particular, there is an extensive discussion on the error bounds obtained in the literature and the ones obtained in this paper. In particular, the authors draw comparisons with [1]. 

* The theory on Schrodinger Bridge is still sparse and even though are extremely strong in the present work the authors derived a novel and interesting result. 

[1] Korotin et al. (2024) -- Light Schrodinger bridge

### Weaknesses
* The presentation of the paper is quite hard to follow. In particular, the introduction is extremely long and seems to merge both the contribution and the related works. In particular, while I understand that it is a choice of the authors, I found it quite hard to follow the related work section. In particular, I did not understand the discussion with [1,2]. Related to this issue, I found the related work to be quite poor. There is no mention on the impact of Schrodinger Bridge work and its application in Machine Learning. For example, Diffusion Schrodinger Bridge [3] but also Stochastic Interpolants [4] and Adversarial approaches [5] are competitive works which are not mentioned (some of them are compared with in the Numerical Experiments section but it is never mentioned why those works are relevant).

* The assumptions are extremely strong. While I understand that the results obtained by the authors are new, the class of target distributions and reference potentials that is under consideration is extremely limited. This strongly limits the theoretical impact of the paper. 

* The methodology contribution is incremental. In the applications, the authors consider Light SB which indeed fits their framework. However, the only modification to this work is in the reference process considered. The results obtained are quite underwhelming (see Table 2 for instance where the methods is similar to Light SB in performance and not as good as OT-CFM [7]). I do appreciate however that the authors highlight the limitations of their approach "We emphasize that these examples are not intended to suggest
universal superiority but rather to showcase specific strengths of our method in certain cases."

* In the theoretical results presented, even though this is discussed, I find the fact that the time must grow with the number of samples to be a bit concerning. Even though I agree that the doubly logarithmic dependence (which has a logarithmic effect on the regularisation due to the exponential convergence of the Ornstein-Uhlenbeck) can be mitigated, it is concerning that the time grows as the number of samples grows. I see this as one of  the main limitation of the paper. 

[1] Pooladian et al. (2024) -- Plug-in estimation of Schrodinger bridges

[2] Tang et al. (2024) -- Simplified diffusion Schrodinger bridge

[3] De Bortoli et al. (2021) -- Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling

[4] Albergo et al. (2023) -- Stochastic Interpolants: A Unifying Framework for Flows and Diffusions

[5] Gushchin et al. (2024) -- Adversarial Schrödinger Bridge Matching

[6] Korotin et al. (2024) -- Light Schrodinger bridge

[7] Tong et al. (2023) -- Improving and generalizing flow-based generative models with minibatch optimal transport

### Questions
* $\hat{\pi}$ in Page 3 hasn't been introduced yet. 

* See my question above on the time dependency. 

* Using a Ornstein-Uhlenbeck as a reference measure for Schrodinger Bridges is not new. For instance, it was used in [1]. In addition, it can be shown quite easily that using a Ornstein-Uhlenbeck as a reference path will lead to a quadratic cost function with a regularisation parameter that grows exponentially with the time of the process. It would be good if the authors could discuss this fact in the main paper. 

[1] Shi et al. -- Diffusion Schrödinger Bridge Matching

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
**Disclaimer**

*I do not have specific expertise in quantitative generalization bound estimation. Therefore, while I provide an assessment of the clarity, motivation, and overall contribution of the paper, I will defer to other reviewers with stronger expertise in the theoretical aspects when evaluating the technical soundness of the quantitative rate derivation.*

**Summary**

This paper studies the finite-sample generalization behavior of Schrödinger bridge (SB) estimation for unpaired data translation. While most prior works adopt Brownian motion as the reference process, this work introduces the Ornstein–Uhlenbeck (OU) process as the reference dynamics. The OU process exhibits exponential mixing, which enables a more stable and analytically tractable setting for sample-based SB estimation. The paper establishs the first non-asymptotic generalization bound for estimating Schrödinger potentials $\phi$ as the empirical risk minimizer and derives a convergence rate of $O(\log^{3} (n) / n)$ for the KL divergence between the optimal coupling $\pi^{*}$ and the empirical risk minimizer $\hat{\pi}$ (when the approximation error $\Delta$ is small). Empirically, the paper extends the Light Schrödinger Bridge framework to an OU-based version (LightSB-OU), demonstrating improved stability and translation quality in synthetic, biological (single-cell RNA), and unpaired image translation tasks.

### Strengths
- The paper provides a new tight finite-sample generalization bound for Schrödinger bridge (SB) estimation.
- Replacing Brownian motion with an OU process is both theoretically novel and practically beneficial.
- The proposed LightSB-OU algorithm effectively connects the theoretical framework to practical tasks. Notably, the OU process corresponds to the VP-SDE used in the diffusion model, while Brownian motion aligns with the VE-SDE. The experiments on both synthetic and real-world datasets demonstrate that the OU reference improves fidelity in unpaired data translation.
- The manuscript is well organized and clearly written. The motivation, the implications of assumptions, and the meaning of theoretical results are well described.

### Weaknesses
- Although the derived convergence rate of $O( \log^{3} (n) / n)$ is theoretically novel, there is no experimental study showing empirical convergence as a function of sample size. Including a convergence curve or ablation study on sample size would better demonstrate the practical significance of the bound.
- While the bound is tight in a theoretical sense, it remains unclear how much this improved rate translates into practical accuracy gains over the previous $O(n^{-1/2})$ regime established by LightSB.
- In the image-to-image translation experiments, only selected qualitative results are presented. Providing quantitative metrics, such as FID (for fidelity to target semantics) and LPIPS (for perceptual similarity between input and output), would significantly strengthen the empirical claims.

### Questions
- Beyond theoretical implications, are there practical improvements in unpaired translation performance (e.g., FID, MMD) that can be directly attributed to the tighter convergence rate?
- The title emphasizes "Unpaired Data Translation," but its primary contribution is theoretical. Perhaps a title that emphasizes the general results, such as "Tight Bounds for Schrödinger Potential Estimation between Empirical Distributions", would better reflect the scope of this paper? (Of course, this is left to the author's discretion.)

### Soundness
3

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
2

### Summary
This paper provides learning theory style high probability bounds for estimating Schrodinger bridge potentials when the reference dynamics is OU process. This is a theoretical paper where the main result is a nonasymptotic, high probability bound improving prior work.

### Strengths
- The paper is mathematically strong, well structured, well motivated on its own domain.
- There are mathematical novelties, for example, the high-probability bound is novel compared to prior work.
- The derived convergence rate is fast.

### Weaknesses
- Some assumptions of the paper are probably unrealistic for real-world scenarios
- The bound, while a nonasymptotic, high-probability bound, is not very clean.
- Practical implications of the result is not clear, there's a gap between theory and why how it explains the empirical improvements.

### Questions
- notation is dense. A short “notation table” early in the paper would help.
- Can Theorem 1 extend to sub-Gaussian cases? Please discuss Assumption 2 and its applicability in real-world settings.
- The authors claim the dependence of the bound to dimension is 'nearly linear'. The bound itself, however, is not clearly written. It would be great if the authors provide a corollary, perhaps cleaning up some unnecessary quantities and display the dimension dependence clearly.
- Please provide a discussion on when $\Delta$ can be made small (for the favourable convergence rate), i.e., for what classes of initial/target distributions, what families would make it small?
- The theoretical analysis is elegant, but it’s not clear what it tells practitioners: (i) Does the theorem suggest a way to choose the OU drift parameter b? (ii) Does it explain why LightSB-OU performs better numerically?

It is this last point I am more keen on seeing explained -- I appreciate the paper's main contribution is theoretical but it would be still good to understand how theoretical bound would give insights for practical improvements.

### Soundness
3

### Presentation
3

### Contribution
3
