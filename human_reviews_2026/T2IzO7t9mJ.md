# IntervalGP-VAE: Learning Unobserved Confounders with Uncertainty for Personalized Causal Effect Estimation

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2

## Abstract
Estimating individual treatment effects (ITEs) in the presence of unobserved confounding remains a central challenge in causal inference. Existing proxy-based methods aim to recover latent confounders from observational proxies, but typically produce only point estimates without uncertainty quantification. This lack of uncertainty modeling leads to incomplete and potentially insufficient information for downstream decision-making, especially when uncertainty is inherent in the data. We propose IntervalGP-VAE, a novel framework that combines variational autoencoders with Gaussian Process (GP) to model both the latent confounders and their associated uncertainty. At the core of our method is an interval-valued GP prior, which enables the model to capture a distribution over plausible latent confounders and treatment responses, rather than relying on potentially unreliable point estimates. This approach accounts for uncertainty arising from noisy and imperfect proxy variables and yields calibrated ITE interval to support more robust causal decisions. We provide theoretical guarantees for identifiability of the latent confounder up to a smooth monotonic transformation under weak assumptions. Experiments on synthetic and semi-synthetic datasets demonstrate that IntervalGP-VAE achieves superior performance in ITE estimation and uncertainty calibration, outperforming existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses the challenge of estimating ITEs in the presence of unobserved confounders.

- **Methodology**: The authors introduce a novel framework named IntervalGP-VAE. This model integrates a VAE with GPs to infer latent confounders from proxy variables and, crucially, to quantify the uncertainty in this inference process.

- **Core Technical Contribution**: The central innovation is the use of an interval-valued GP prior over the latent space. This allows the model to represent the latent confounder as a distribution (an interval) rather than a deterministic point estimate. This uncertainty is then propagated through a second interval-valued GP that models the outcome, resulting in calibrated confidence intervals for the ITE.

- **Theoretical Contributions**: The paper provides a theoretical analysis of the conditions required for identifiability. It establishes that for a $d$-dimensional latent confounder, at least $k \ge 2d+1$ sufficiently non-redundant proxy variables are necessary to recover the confounder up to a smooth, invertible transformation. The authors also prove that the ITE estimation is invariant to such transformations of the latent space.

- **Experimental Validation**: The proposed method is evaluated on 24 synthetic datasets, which are specifically designed to meet the paper's theoretical identifiability conditions, and on the semi-synthetic IHDP. The performance is compared against several baselines, including TEDVAE, using standard metrics like PEHE and ATE error. A unique evaluation metric reported is the coverage rate of the estimated ITE intervals, which directly assesses the model's uncertainty quantification capabilities.

### Strengths
- The overall problem setup is clearly conveyed, and the proposed framework is well illustrated. The addressed problem is important and relevant to the causal inference community.
- It is interesting that the authors use features extracted through a VAE together with a Gaussian Process to provide uncertainty quantification for the estimated causal quantities. Although similar paradigms have been explored in previous studies, such as the Deep Kernel GP method in the CausalBALD paper and other GP-based approaches including CMGP and NSGP, this work appears to be the first to apply such a combination to the problem of causal inference with unmeasured confounders. This novelty represents a meaningful contribution.

### Weaknesses
**1. Theoretical analysis**

The theoretical analysis does not appear to provide any meaningful contribution to the CATE estimation task. The presented theorem seems disconnected from the main goal of estimating causal effects, as it only discusses the recovery of the latent variable \(U\) from \(Z\). Moreover, several derivations contain inconsistencies or potential errors. For instance, \(U\) is defined as a \(d\)-dimensional variable, but the invertible mapping \(h\) is described as a function from one dimension to one dimension, which is mathematically inconsistent.  

In addition, most results in the proof rely heavily on *Allman et al. (2009)*, whose theoretical development was made for **discrete** variables. This assumption is incompatible with the **continuous** latent setting in the current paper, making the imported results invalid in this context. Overall, the theoretical section does not convincingly support the claimed contributions and contains technical flaws that need substantial revision.

**2. Experimental design and evaluation**

The experimental results are unconvincing. The paper lists many baseline methods, including TEDVAE, but many of these baselines were originally designed for scenarios **without unmeasured confounders**. It is unclear why such baselines were selected if they are not intended for the same problem setting addressed by this work.  

Furthermore, TEDVAE is used as a baseline but itself targets the **unconfounded** case, which makes it an inappropriate comparator for the unmeasured-confounder setting emphasized by the paper. TEDVAE only outperforms these baselines under the authors’ specific experimental setup, and there is no evidence showing that the proposed method would consistently outperform appropriate baselines in general. As a result, the experimental evidence does not support the strong performance claims made in the paper.

### Questions
1. Some notations are introduced before their formal definitions. For example, PEHE and ATE are mentioned in the introduction without being defined. It would be clearer to define these terms when they first appear.

2. It would be helpful to illustrate the data-generating process using a causal graph. This would make the assumed causal structure and the role of latent variables more explicit.

3. It is unclear why the identifiability of the latent variable \(U\) should be related to the model outputs. In my understanding, identifiability should only depend on the relationship between the true latent variable \(U\) and its recovered version \(\hat{U}\), i.e., the learned representation. The current definition appears closer to the identifiability of causal quantities rather than that of the latent variable itself. It would be better to provide a clear mathematical formulation of the identifiability condition in the main text.

4. In the experimental results, the proposed method only outperforms TEDVAE in the fully synthetic case and only under the PEHE metric. For the ATE error, the proposed method does not surpass TEDVAE. On the IHDP dataset, TEDVAE performs better in both metrics. More importantly, both the baseline and the proposed method are unable to properly handle the unmeasured confounder case, indicating that the experimental design is fundamentally flawed.

### Soundness
2

### Presentation
2

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
This paper proposes IntervalGP-VAE, a variational framework that combines a Gaussian Process prior with interval-valued likelihoods to model unobserved confounders for personalized causal effect estimation. The method aims to quantify uncertainty in individual treatment effects (ITEs) by introducing an interval Gaussian Process in the latent space. Experiments on synthetic and semi-synthetic datasets (IHDP) show improved estimation accuracy.

### Strengths
- The experimental section is comprehensive, covering both synthetic and semi-synthetic datasets with multiple baselines and uncertainty-related metrics, which provides a convincing empirical evaluation.

- The related work section is thorough and well structured, covering prior studies on causal inference, VAE-based representation learning, and GP-based uncertainty modeling, which helps position the paper clearly within the existing literature.

### Weaknesses
- The paper describes a composite objective combining reconstruction and GP regularization, but it’s not clear how the model is actually optimized. Are all components trained jointly? Please clarify how the optimization is actually performed.

- The paper does not analyze the computational complexity of the proposed model. Given that Gaussian Process inference scales as 
$𝑂(𝑛^3)$ with the sample size, it remains unclear how the method performs or can be adapted for practical deployment

### Questions
See weakness

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
4

### Summary
This paper proposes a Bayesian method for identifying hidden confounders in causal models using noisy proxy variables. It treats proxy noise as uncertainty and employs Gaussian Processes (GP) to model the smoothness of the underlying functional mappings from proxies to the latent confounder $ U $. The approach addresses identifiability by discretizing continuous proxies and applying Kruskal rank conditions under assumptions of conditional independence and generic variability. Theoretical results claim identifiability up to injective transformations, enabling downstream estimation of individual treatment effects via GPs that preserve structural coherence in the latent space.

### Strengths
- Framing proxy noise as uncertainty within Bayesian methods is a sound conceptual approach, aligning well with probabilistic modeling principles.
- The general idea of employing Gaussian Processes (GPs) to capture smoothness in the underlying functions is sensible.

### Weaknesses
All theoretical statements and proofs contain significant issues, undermining the paper's claims.

- **Definition 1 (Identifiability):** The definition is incomplete. Identifiability requires not just the existence of a mapping from observed variables to the hidden confounder $ U $, but that this mapping (or its equivalence class) is uniquely recoverable from the distribution of the observed variables alone.
- **Theorem 1:** The assumptions of continuous differentiability and injectivity for the functions $ g_i $ are stated but never utilized in the proof. Instead, the proof relies on Kruskal's theorem, which applies exclusively to discrete variables and thus contradicts the smoothness assumption (as $ g $ would need to be discontinuous over a discrete domain).
    
    The proof is fundamentally flawed due to the mismatch between discrete and continuous variables in identification analyses. For instance, Kruskal ranks depend critically on the number of discrete values, rendering the authors' remark that "as bin widths shrink, the discrete model approaches the continuous distribution" meaningless in this context. Even granting the discretization approximation, the proof fails to demonstrate how conditional independence of proxies given $ U $, combined with sufficient variability (generic position) within each group, ensures the Kruskal rank conditions are met—this is precisely the core challenge the work claims to address (e.g., see [1] and its applications to causal inference [2]). Relatedly, the partition of $ Z $ into three sets appears arbitrary, as if any partition would suffice; it certainly does not. A rigorous argument would require detailed structural analysis to verify adequate Kruskal ranks for these sets. Finally, the concluding step (line 711) has a serious gap, as the proof nowhere addresses the identification of the prior $ p(U) $. Overall, the theorem and its "proof" resemble a superficial and erroneous adaptation of results from Allman et al. (2009) and related works (see LLM suspicion below).
    
- **Theorem 2:** This is a trivial consequence (identifying the hidden confounder up to an injective mapping naturally implies downstream identifiability) and irrelevant, as Theorem 1 does not establish the required identifiability (injective or otherwise).
- **Core Idea of Gaussian Processes (GPs):** The motivation for GPs is underdeveloped. The notions of (spatial or structural) coherence and similarity are not clearly articulated, nor is their real-world relevance. Mathematically, this intuition stems from the smoothness of $ g $, but—as detailed above—this smoothness assumption leads to internal contradictions within the paper.
- **Proposition 1:** This is an imprecise assertion relying on the aforementioned intuition that GPs model "smoothness" (or "inherit the same geometric structure via an induced kernel"). The proof offers only hand-wavy justifications, such as "similar inputs $ Z_i \approx Z_j $ induce similar latent values $ u_i \approx u_j $"—such approximations are invalid in formal identification proofs or any mathematical derivation. Even if smoothness is pursued, the key question is how to design a kernel suitable for $ g $, which demands additional assumptions on $ g $'s functional form (e.g., a Lipschitz constant).


### Other Issues

- **LLM Writing Suspicion (for Theorem 1 and Potentially More):** The terms "smooth" and "injective" are invoked at least five times before the proof of Theorem 1 (including in the statement itself). This is peculiar, as they bear no relation to the actual proof content. Assuming LLM involvement, this pattern aligns with common motifs in nonlinear ICA literature, a prominent area in machine learning (see below). The paper's overall "plausible but deeply flawed" quality mirrors experiences with LLM-generated text, warranting further scrutiny.

- "ITE" is used without formal definition; it appears to refer to the "treatment effect" in lines 102–103, but this is not the Individual Treatment Effect (ITE), which should be a deterministic function of all outcome-affecting variables (observed, hidden, and exogenous noise). This is a frequent misnomer in machine learning literature.
- Typo: In the boxed statement of Theorem 1, the domain of $ g_i $ should be $ \mathbb{R} $, not $ \mathbb{R}^d $.

---

### Related Work
All the identifiable or causal **VAE** papers below are fundamentally based on *smooth and injective* decoder maps. In particular, *the idea of "injective proxy" was considered in [3] (Definition 15)*.

[1] Khemakhem, Ilyes, et al. "Variational autoencoders and nonlinear ICA: A unifying framework." International Conference on Artificial Intelligence and Statistics. PMLR, 2020.

[2] Wu, Pengzhou Abel, and Kenji Fukumizu. "\beta-Intact-VAE: Identifying and Estimating Causal Effects under Limited Overlap." International Conference on Learning Representations (2022).

[3] Wu, Pengzhou, and Kenji Fukumizu. "Towards principled causal effect estimation by deep identifiable models." arXiv preprint arXiv:2109.15062 (2021).

### Questions
Please refer to the points in Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1
