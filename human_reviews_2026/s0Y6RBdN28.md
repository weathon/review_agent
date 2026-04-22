# Spatial Deconfounder: Interference-Aware Deconfounding for Spatial Causal Inference

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Causal inference in spatial domains faces two intertwined challenges: (1) unmeasured spatial factors, such as weather, air pollution, or mobility, that confound treatment and outcome, and (2) interference from nearby treatments that violate standard no-interference assumptions. While existing methods typically address one by assuming away the other, we show they are deeply connected: *interference reveals structure* in the latent confounder.
Leveraging this insight, we propose the **Spatial Deconfounder**, a two-stage method that reconstructs a substitute confounder from local treatment vectors using a conditional variational autoencoder (CVAE) with a spatial prior, then estimates causal effects via a flexible outcome model. We show that this approach enables nonparametric identification of both direct and spillover effects under weak assumptions—without requiring multiple treatment types or a known model of the latent field.
Empirically, we extend `SpaCE`, a benchmark suite for spatial confounding, to include treatment interference, and show that the Spatial Deconfounder consistently improves effect estimation across real-world datasets in environmental health and social science. By turning interference into a multi-cause signal, our framework bridges spatial and deconfounding literatures to advance robust causal inference in structured data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed a new method, CVAE-SPATIAL+ for spatial causal inference with two challenges in spatial settings: localized interference (spillovers from neighbors) and unobserved spatial confounding (latent fields like meteorology or socioeconomic context). This is a two-stage method. The key idea is to treat interference as a multi‑cause signal: by looking at a unit’s treatment/covariates together with its neighbors’ treatments/covariates, a CVAE is trained for recovering a smooth substitute confounder, then estimate direct and spillover effects with a flexible outcome regression model (e.g., U‑Net/GNN architecture). An important “latent‑field sufficiency” assumption is proposed, which shows that the Z representation of the observed assignment/covariates that is equivalent to the latent field for the purpose of adjustment. The paper provides results for both effects under localized confounding and spatial confounding and evaluates the approach on an extended SpaCE semi‑synthetic benchmark. Across air‑quality/health and PM₂.₅‑components tasks, the proposed method reduces standardized absolute bias relative to spatial econometric, spline/RSR, matching, GCNN, and U‑Net baselines.

### Strengths
1.	The methodology is pretty clear and straightforward. Easy to understand and follow.
2.	Using a Gaussian–Markov random‑field prior over the grid Laplacian to enforce smoothness is very reasonable
3.	Methods comparison is comprehensive, the proposed method shows clear advantage over existing methods versus S2SLS‑LAG1, SPATIAL/SPATIAL+, GCNN, DAPSM

### Weaknesses
1.	Especially what additional information has CVAE learned in Z. Since all inputs are also used in the second stage.
2.	Theoretical analysis is relatively weak, the intuition is not clear why adding a CVAE is better than just use SPATIAL+.
3.	The radius is set to only 1 or 2 in experimental settings, which may be quite impractical in real applications. The real data typically reply on irregular graphs.

### Questions
1.	Assumption 4 is very strong and even impractical given unmeasured confounder. Could the authors provide intuition and failure modes and show robustness when this is violated?
2.	In Theorem 1, How necessary is the additivity for Z? This also seems very strong assumption since Z is essentially a representation of A,As,X,Xs.
3.	Some deeper analysis is needed to show intuition of what type of additional information contained in the Z that is learned and contribute to the performance gain.
4.	A systematic sensitivity analysis over neighborhood radius r is very necessary.
5.	How data size and dimensionality of X affect the performance of the new method?
6.	How the dimension of Z affect the final results?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the problem of spatial confounding by proposing a method to recover an unobserved spatial confounder from observed neighbor treatments and covariates. The core methodology employs a Conditional Variational Autoencoder (CVAE), where the latent variable is intended to represent this hidden confounder. 

For identification, the paper relies on a key assumption (Assumption 4) that extends a single-site ignorability condition to hold uniformly across all sites if it holds for some. The proposed estimator's validity is assumed directly (Assumption 5), with a reference made to a multi-cause confounding model by Wang and Blei (2019) as a conceptual analogy. The paper's setting is ambitious, aiming to handle spatial confounding in a broad context.

### Strengths
- Relatively well written.
- Transparent about its assumptions.
- Ambitious general problem setting and goals.

### Weaknesses
### 1: Estimation
The paper assumes away the core difficulties of estimation. It directly posits that the CVAE estimator is valid (Assumption 5), implying the hidden confounder can be fully recovered. However, a valid estimator must be tied to a specific identification strategy. Even with identification in hand, designing a valid estimator is highly nontrivial—especially for VAEs, where identifiability remains an active research area [1, 2]. The current approach mirrors the CEVAE paper [3], which has faced similar critiques [4, 5]. 

The reference to Wang and Blei (2019) does not substantiate the estimator's validity. The proposed estimator diverges from the identification and modeling in Wang and Blei: the multi-cause structure it alludes to is merely a vague analogy, and the VAE does not leverage their factor model.

The estimator offers limited novelty and relies on arbitrary architecture choices. The broad idea of a VAE-based causal effect estimator appears in numerous works [2, 3] (see references in [2, 5] for more examples), and this paper contributes nothing new. Moreover, the loss in eq. (8) leaves unclear what observational distribution the variational inference targets and how it factorizes. It is also unclear why the encoder takes its current form rather than other viable options. These details matter, as they encode the independence relationships assumed by the inference procedure.

### 2: Identification
Assumption 4, crucial for identification, is unclear and appears unrealistic. It states that if eq. (14) (ignorability?) “holds for some sites, then it holds uniformly across sites.” This is unclear compared to standard ignorability of joint exposure given observed covariates. First, it should be explicitly stated as an additional assumption that single-site ignorability (eq. 14) must hold. The paper seems to imply that ignorability is typically assumed across all sites, but it fails to explain why a "some sites, then uniformly across sites" condition would hold in practice—an example (perhaps based on Figure 2) would help.

The confounding/spatial structure is underexplained. This seems to tie back to Assumption 4, but it reads more like "global sharing" than true structure. Readers would expect a graphical model, or some algebraic/analytical structure in the functional equations, to substantiate it.

### 3: Missing Related Work
Key omissions include [2, 5], which develop general ideas for identifiable deep models, particularly *conditional VAEs*, that could bolster the current work. Specifically, they include an application to network deconfounding that exploits spatial information from neighbor cities—relevant to spatial deconfounding here. They also analyze and demonstrate the role of the $\beta$ parameter, which appears in this paper. See further applications to real-world health data in [6].


### Minor Points
- "Exposure" and "treatment" appear to be used interchangeably, which is confusing. Clarify or standardize to one term.

### References
[1] Khemakhem, Ilyes, et al. "Variational autoencoders and nonlinear ICA: A unifying framework." *International Conference on Artificial Intelligence and Statistics*. PMLR, 2020.

[2] Wu, Pengzhou Abel, and Kenji Fukumizu. "\beta-Intact-VAE: Identifying and Estimating Causal Effects under Limited Overlap." *International Conference on Learning Representations* (2022).

[3] Louizos, Christos, et al. "Causal effect inference with deep latent-variable models." *Advances in Neural Information Processing Systems* 30 (2017). [Early work, *not* recommended]

[4] Rissanen, Severi, and Pekka Marttinen. "A critical look at the consistency of causal estimation with deep latent variable models." *Advances in Neural Information Processing Systems* 34 (2021): 4207-4217.

[5] Wu, Pengzhou, and Kenji Fukumizu. "Towards principled causal effect estimation by deep identifiable models." *arXiv preprint arXiv:2109.15062* (2021).

[6] Ma, Wenao, et al. "Treatment outcome prediction for intracerebral hemorrhage via generative prognostic model with imaging and tabular data." *International Conference on Medical Image Computing and Computer-Assisted Intervention*. 2023.

### Questions
Please refer to the points in Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the Spatial Deconfounder, a two-stage framework for spatial causal inference that handles both unmeasured spatial confounders and interference (spillovers) between neighboring units. It treats the vector of a site’s own and neighbors’ treatments as a multi-cause signal and uses a CVAE with a spatial prior to reconstruct a smooth substitute confounder; this proxy is then fed into a flexible outcome model (e.g., U-Net/GNN) to estimate direct and spillover effects via plug-in contrasts. Under localized interference, positivity, and a weak “latent-field sufficiency” assumption, the authors show nonparametric identification of both effects without specifying a parametric latent-field model. Empirically, they extend the SpaCE benchmark to include interference and demonstrate, on environmental health and social-science datasets, that Spatial Deconfounder variants consistently reduce bias versus spatial econometric, spline/RSR, matching, and GNN baselines—while uniquely recovering spillover effects. Conceptually, the work reframes interference from a nuisance into a source of information for uncovering hidden spatial structure.

### Strengths
- The paper tackles unmeasured spatial confounding and localized interference together and establishes identifiability of both direct and spillover effects under spatial consistency, spatial positivity, localized interference, and a latent-field sufficiency assumption.

- The paper relaxes full ignorability by allowing an unobserved latent spatial field while only requiring that purely local confounders are observed, formalizing this with a latent-field sufficiency condition that is plausible for lattice-structured data.

- The paper provides explicit, nonparametric identification expressions for the direct and spillover effects, which anchor the subsequent estimation strategy.

- The estimation procedure cleanly separates confounder reconstruction from outcome modeling by using an interference-aware CVAE with a Gaussian-Markov random-field prior to recover a smooth substitute confounder, and then plugging this proxy into flexible outcome models such as U-Nets or GNNs.

### Weaknesses
1/ The approach relies critically on accurate reconstruction of the latent confounder via a CVAE; if the substitute confounder is poorly recovered, both direct and spillover effect estimates can be biased. can the authors quantify how estimation error in the latent proxy translates into bias or variance of the effect estimators?

2/ As a VAE-based method, the assignment model is vulnerable to posterior collapse and latent non-identifiability, which could yield an uninformative latent and ineffective adjustment. What concrete safeguards (e.g., KL warm-up schedules, mutual-information terms, decoder constraints) are used, and how often do collapse diagnostics fail in practice?

3/ The identifiability story ultimately assumes a "sufficient" latent field and that the learned proxy recovers a valid transformation of it; this is a strong modeling assumption at the learning stage. Can the authors provide formal or empirical evidence that their training procedure consistently recovers an informative proxy under realistic misspecification?

4/ Performance may be sensitive to hyperparameters that directly affect identifiability and collapse (e.g., \beta in the KL term, temperature or strength of the spatial prior). Do the results remain stable under a systematic sensitivity analysis over these hyperparameters?

5/ The spatial smoothness prior (e.g., GMRF) imposes a particular structure on the latent field; if the true confounding varies non-smoothly or anisotropically, the proxy may be biased. Have the authors evaluated robustness to misspecified spatial priors (non-stationary/anisotropic fields, sharp boundaries)?

6/ The localized-interference assumption is central; if interference extends beyond the specified neighborhood, the multi-cause signal may be mis-modeled and the recovered latent may absorb spillovers rather than confounders. How sensitive are estimates to enlarging or shrinking the interference neighborhood, and do the authors provide misspecification experiments?

7/ The method’s benefits rely on interference providing a strong multi-cause signal; in sparse treatments or weak neighbor correlations, the signal may be too weak to recover the latent reliably. Please present stress tests varying treatment sparsity and network density to map the regime where the approach is reliable.

### Questions
Please refer to the questions in Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work focuses on learning a model for which to perform causal inference in settings where covariates, treatments and outcomes live withing a spatial grid (such that neighboring variables can cause each other), and where the goal is to do so under: 1) interference, where treatments can affect outcomes in neighboring cells; and 2) spatially structured hidden confounding, which affects both the treatment and the outcome in that grid cell. The key idea of this work is that interference can be interpreted as a multiple-cause setting, for which they can leverage the work on the deconfounder framework to enable causal inference. Then, the authors propose a model based on conditional VAEs to build up a substitute of the hidden confounder, which is then fed to a potential outcome model that uses that substitute to perform causal inference. Moreover, the authors show causal identifiability results and empirically test their proposed method.

### Strengths
- **S1.** The core idea behind this work is sound and interesting: Interference as observed in climate science applications can be interpreted as a multi-cause framework.
- **S2.** The paper is well-motivated and well-written, with clear text, explanations, and walking the reader through all the process.
- **S3.** The proposed model is sound, and the authors provide some causal identifiability results to back it up.
- **S4.** All the assumptions are clearly stated and they are reasonable and sound.
- **S5.** The experimental results show promise on the proposed model.

### Weaknesses
- **W1.** I have a few concerns regarding assumptions:
  - **W1.1.** Assumption 5 sounds a bit unrealistic, given that the model uses a Gaussian with positive variance to model the hidden confounder. I understand it can be interpreted in the "nearly-deterministic" setting if the variance is extremely small, but it needs to be ensured. ([This paper](http://arxiv.org/abs/2206.02416) could be of interest to the authors regarding this topic.)
  - **W1.2.** Causal identifiability in theorem 1 relies on the hidden confounding being separable from the rest of variables in the structural equation.
  - **W1.3.** Assumption 2 regard the learned hidden confounder as far as I understand, which depends on the model itself.
- **W2.** Related work does not include any causal generative models, which are relevant and related with the current work. This includes  [NCMs](https://arxiv.org/abs/2107.00793), [CNFs](https://arxiv.org/abs/2306.05415), or [Diff-SCM](http://arxiv.org/abs/2202.10166), among others (see references therein). [Follow-up work](https://proceedings.mlr.press/v139/wang21c.html) of the deconfounder framework by the same lab should also be relevant.
   - **W2.1.** Particularly related to this work is [DeCaFlow](https://arxiv.org/pdf/2503.15114), which combines the Deconfounder framework with Causal Normalizing Flows to perform causal inference under hidden confounding with a given yet general causal graph. DeCaFlow shares quite some similarities with the proposed model, where it also uses a encoder-decoder architecture to build a substitute of the hidden confounder, and trains with the ELBO. Indeed, DeCaFlow should be applicable in the experiments as a baseline. I'd suggest to relax the statements regarding being the "first framework" (line 70).
- **W3.** Following up on the experiments, I have three main concerns.
  - **W3.1.** I am not sure to understand what it means to "mask" some elements. Does it mean to zero-them out? Or to completely removed them from the model's input?
  - **W3.2.** Similarly, I am not sure why the exogenous noise was introduced in equations 19 and 20 as additive, rather than as another input to the function $f$.
  - **W3.3.** Finally, I am concerned about the selection of baselines, most of them looking rather weak as they are not causally-aware and, despite that, the differences in performance with the proposed model are not statistically significant, which concerns me the most. (Indeed, I find the bold numbering misleading.)

### Questions
- **Q1.** What do the authors mean by "Posterior draws of Zs yield uncertainty bands." in line 263?
- **Q2.** Are the samples of the treatment from the decoder in line 269 sampled using samples from the prior of z? or from the posterior?
- **Q3.** Where are the predictive checks used? 
- **Q4.** What is $L_Y$ in remark 1?

---
Other feedback:
- I'd be careful with using CVAE as name during the manuscript, as it closely resembles another causal inference work: [CEVAE](https://arxiv.org/abs/1705.08821).

I'll happily increase my score after my concerns are addressed.

### Soundness
3

### Presentation
3

### Contribution
3
