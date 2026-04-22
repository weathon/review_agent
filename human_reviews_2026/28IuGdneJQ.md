# SFBD-OMNI: Bridge models for lossy measurement restoration with limited clean samples

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
In many real-world scenarios, obtaining fully observed samples is prohibitively expensive or even infeasible, while partial and noisy observations are comparatively easy to collect. In this work, we study distribution restoration with abundant noisy samples, assuming the corruption process is available as a black-box generator. We show that this task can be framed as a one-sided entropic optimal transport problem and solved via an EM-like algorithm. We further provide a test criterion to determine whether the true underlying distribution is recoverable under per-sample information loss, and show that in otherwise unrecoverable cases, a small number of clean samples can render the distribution largely recoverable. Building on these insights, we introduce SFBD-OMNI, a bridge model-based framework that maps corrupted sample distributions to the ground-truth distribution. Our method generalizes Stochastic Forward-Backward Deconvolution (SFBD; Lu et al., 2025) to handle arbitrary measurement models beyond Gaussian corruption. Experiments across benchmark datasets and diverse measurement settings demonstrate significant improvements in both qualitative and quantitative performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces SFBD-OMNI, a new framework for generating from a clean data distribution while training from corrupted samples and a limited number of clean samples. The proposed method generalizes SFBD to arbitrary corruption models, providing both theoretical guarantees and practical algorithms, including an online variant that supports end-to-end training. The authors handle non-identifiable corruptions by introducing a regularization that acts as a "projection" (in the KL sense), enforcing the solution to be close to a given distribution constructed from a small set of clean samples.

### Strengths
1. Overall, the proposed approach is clean and elegant, relying on well-established principles.
2. The paper is well-written and easy to follow. The paper is structured in a clear and logical manner, and the authors provide several intuitive examples throughout to help the reader understand the concepts presented.
3. The experimental results seem good -- the proposed approach improves upon previous methods, and it is able to handle non-identifiable degradations such as de-colorization.

### Weaknesses
1. Only two datasets (CIFAR-10 and CelebA) are evaluated. Testing on more complex real-world domains (e.g., MRI, satellite data) could have been more appealing from a practical standpoint. In my opinion, this is a major weakness.
2. While clean-sample weighting is explored (Figure 2), the effect of other hyperparameters such as $\gamma$ (update ratio) remains unclear.
3. The authors perturb the endpoints $y$ to "prevent degeneration", since flow matching is a deterministic map. But an explanation about the effects of such perturbations (both in terms of theory and practice) is missing.
4. The proposed approach still uses a small number of clean samples to train the generative model, which is a limitation. In real-world scenarios and scientific applications (e.g., astrophysics), we often only have corrupted samples.

### Questions
Why was flow matching selected as the primary instantiation? Would other stochastic bridges (e.g., Schrodinger) behave differently? Perhaps using another bridge could alleviate the need to perturb the endpoint samples? It would be interesting to assess several types of flows, especially since, from the beginning of the paper, the authors formulate the problem in the stochastic sense (using Brownian motion, eq. 8), but then implement it with flow matching (which does not involve Brownian motion).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper considers the problem of restoring clean signals from corrupted measurements in a setting with little to no access to clean samples and a black-box access to the corruption model. Taking inspiration from the Stochastic Forward-Backward Deconvolution (SFBD) algorithm for Gaussian noise corruption, the authors introduce SFBD-OMNI as its extension to arbitrary corruption processes. At the core of the method lies a reformulation of the problem using the Donsker-Varadhan variational principle, which leads to an explicit objective for the (augmented) Kullback-Leibler Ambient Projection (KLAP) problem. In addition to the resulting formulation of SFBD-OMNI, the authors also propose its online (flow) variant that performs only partial updates to the data buffer used during training. Both methods are empirically compared with a proper set of baselines on both identifiable and non-identifiable processes. In addition, an ablation regarding the influence of the clean sample weighting is presented.

### Strengths
S1. The paper is written extremely well considering that the topic is not the easiest to describe formally. The flow of the text is great, with additional intuitive explanations following the introduction of more difficult concepts (like Figure 1).

S2. The authors propose a novel reformulation of the considered objective using the Donsker-Varadhan variational principle, showing a connection to the entropic optimal transport objective. The theoretical results are interesting and relevant to the considered problem. Importantly, the convergence analysis applies to both the standard SFBD-OMNI and the online version.

S3. The proposed methods outperform the considered baselines and the empirical results align with the theoretical considerations.

### Weaknesses
W1. My main concern relates to the accuracy of one of the claims. The authors suggest that, under suitable identifiability conditions, the ground-truth clean data distribution can be recovered without access to clean samples (as lines 065-069 seem to suggest). Proposition 4 guarantees convergence under the considered update rules, even with an arbitrary initialization and full replacement in the training buffer $\Large \varepsilon$. Does this imply that SFBD works when no clean samples are used at all, i.e., even when the initial pretraining phase is skipped? Is this scenario empirically verified in the paper, or do the results only cover the case where clean samples are used for pretraining but omitted from the updates in line 4 of Algorithm 1/2? If this is not verified, I would ask the authors to provide results for this case.

W2. I understand that some methods in Table 1 are only applicable to specific corruption processes. Are there any instances where a baseline is theoretically suitable for one of the tested problems, but its results are not included in the table? If so, the paper would greatly benefit from evaluating these baselines in such cases, as this would provide a more complete comparison and support future work.

### Questions
I am generally very sympathetic to this work and consider the approach, together with its theoretical derivations, very elegant. I was leaning toward a score of 8. However, I am very interested in the authors' answer to W1 above and consider it an important clarification before raising my score from 6 to 8. Please also refer to W2. Below are some additional minor questions.

Q1. lines 046-048: Is this phrased correctly? Shouldn't it be *With only a limited number of corrupted samples but an abundance of clean ones (...)*?

Q2. line 160: The authors probably meant $g=0$ for Eq. 6.

Q3. I-projection (line 216) should be expanded to Information-projection for clarity.

Q4. line 253: What if $r(\mathbf{y}|\mathbf{x})=0$ under the $\log$?

Q5. lines 289-298: The subscripts and superscripts for $k$ are probably mixed up.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose SFBD-OMNI, a bridge model–based framework that maps corrupted sample distributions to the underlying clean data distribution for image restoration and generation recovery. The method extends the prior SFBD framework to handle arbitrary measurement models beyond the Gaussian corruption setting, providing a more general and theoretically grounded approach to learning from imperfect or partially observed data

### Strengths
- Theoretical generalization capability for any unbound noise. The paper extends prior diffusion-based deconvolution frameworks SFBD to non-identifiable measurement processes, providing a principled formulation via one-sided entropic optimal transport. This generalization makes the approach applicable to a broader range of real-world degradations 
- Data-efficient restoration without extensive ground-truth supervision. The proposed framework learns to recover clean image distributions using only a handful of clean samples, making image restoration more practical and label-efficient compared to traditional fully supervised diffusion approach

### Weaknesses
- Limited novelty via SFBD framework extension. While the paper provides a principled generalization of SFBD to non-identifiable measurement models via one-sided entropic OT, the overall formulation and optimization pipeline remain close to the original SFBD framework. The methodological increment feels more like an extension or refinement rather than a fundamentally new paradigm.

- Insufficient experimental evidence to substantiate the theoretical analysis. Despite a well-developed theoretical formulation, the experimental validation is narrow and mostly relies on simple synthetic corruption processes. The work lacks evaluation on real-world restoration datasets (e.g., **RainDrop (Qian et al., CVPR 2018)**, **AllWeather (Valanarasu et al., 2022)** ) that could better demonstrate generalization under uncontrolled degradations. Moreover, the notion of “limited clean data” (only 50 samples) is fixed and not systematically analyzed against traditional fully-supervised baselines, leaving unclear how much clean supervision is truly needed.

- Sensitivity to hyperparameter settings and practical complexity:
The algorithm involves several critical hyperparameters, such as the clean-sample weight λ and the online update ratio γ, which the authors fix empirically (λ / (1 + λ) = 0.2, γ = 0.002). The paper does not provide an ablation or sensitivity study, and training remains complex due to alternating optimization of all hyperparameters. This raises concerns about reproducibility and the need for manual tuning to achieve stable performance.

### Questions
1.	How does the method scale with different amounts of clean data?
The paper fixes the number of clean samples to 50 (~0.1%), but it would be informative to see how performance changes when this ratio varies, or how it compares to a small supervised diffusion baseline trained on the same subset?

2.	Do the authors plan to evaluate SFBD-OMNI on real-world restoration datasets (e.g., RainDrop, AllWeather) to demonstrate its applicability beyond synthetic corruptions, and to compare its performance against recent fully supervised diffusion-based restoration methods?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes SFBD-OMNI, a bridge-model framework to learn the clean data distribution when you have many corrupted samples and few clean ones, assuming black-box access to the corruption process. They cast “recover clean from corrupted” as a KL Ambient Projection (KLAP) problem and show two key variational views: (i) the classic AmbientGAN min–max formulation, and (ii) a new formulation via the Donsker–Varadhan principle that turns KLAP into a one-sided entropic optimal transport objective with a tractable alternating minimization (posterior update + prior update). 

This yields closed-form updates and naturally plugs into diffusion/bridge models to learn the posterior 
 $u_\theta(x∣y)$. To address non-identifiable corruption (e.g., grayscale or blur), they augment the objective with a KL regularizer toward a small set of clean samples $h$ and analyze the limit as the regularization weight vanishes, recovering the I-projection within the feasible set. They present SFBD-OMNI and an online variant (partial refresh of reconstructed samples) with convergence guarantees. Experiments on CIFAR-10 and CelebA show improved FID over baselines (including in non-identifiable cases) when a small number of clean samples is mixed in training.

### Strengths
- Unifying view + derivation: Recasting KLAP with Donsker–Varadhan to obtain a one-sided entropic OT objective is elegant and principled; it clarifies when/why alternating updates should work and connects to entropic OT literature (contrast with AmbientGAN’s min–max). 
- Identifiability analysis + “few clean” fix: Clear conditions for identifiability and a regularized KL that provably picks the best element in the feasible set, matching practice where a handful of clean samples are available.
- Convergence guarantees: Both batch and online SFBD-OMNI have convergence results (including a rate bound when $\lambda \to 0$). That’s stronger theory than many EM-like or Ambient-style methods.
- Empirical coverage: Results span identifiable (masking, Gaussian noise) and non-identifiable (grayscale, blur) regimes; OMNI competes with or beats specialized methods and EM-Diffusion. 
- Practical modeling choice: Using bridge/flow-matching to learn posteriors is a strong fit; the paper also addresses degeneracy of deterministic paths with noise injection.

### Weaknesses
- Assumption on black-box sampling from r(y|x): Many real sensors provide only forward passes on x, not easy sampling of y|x across conditions; measuring cost or calibration drift may complicate the assumed operator access. The paper could discuss robustness to operator mismatch (misspecified r). 
- Posterior learning scalability/details: Training $u_\theta(x|y)$ as a conditional bridge can be heavy; stability hinges on path choices and endpoint noise. More ablations on bridge choices, noise level, and compute vs. FID trade-offs would help (training time is non-trivial).
- Evaluation breadth: Only CIFAR-10 ($32^2$) and CelebA ($64^2$) with limited corruptions. Stronger evidence would include higher-res datasets, complex forward models (e.g., compressive sensing, Poisson noise), and domain tasks where distributional recovery helps downstream reconstructions.
- FIDs are fine but you really do need to show empirical samples for the learned model.
- Tuning $\lambda, \gamma$: While theory supports limits, practice shows a “sweet spot” for clean-weight; guidance for automatic selection (e.g., validation via corrupted likelihood) is missing.
- Identifiability test practicality: The paper discusses conditions and a criterion qualitatively; a procedural test practitioners can run on a new corruption (and with finite data) would increase usability.
- Minor nitpick, but Section 4.2 needs some cleaning up bc there's too much content in a small space and it's also a key contribution. Maybe reduce some content in Section 4.1 to give more space to 4.2? Also a small section with notation may be helpful, I had to search for the definition of $h$ which was last defined in Section 3.2 (maybe just say $p_data$?). Also, between Eqn (14) and (15), can you add a small phrase saying "taking \mathbb{E} over $y\sim q$ and subtracting E[ log q(y) ], we get..." It was a little confusing at first since there are many functions and it wasn't clear what's a function of y, x, etc.

### Questions
- Operator misspecification: How sensitive is SFBD-OMNI to errors in $r(y|x)$? Say you're doing pixel dropout and you have the wrong estimate of dropout probabilities?
- Automatic hyper-selection: How exactly do you select the $\lambda$ (clean-weight) and $\gamma$ (online update ratio) parameters?
- For my own understanding, what's the right way to think of $\Phi(p)$ in Proposition 3? Is it a penalty for how different the pushforwards in y are for $p^*$ and $p$?

### Soundness
3

### Presentation
1

### Contribution
3
