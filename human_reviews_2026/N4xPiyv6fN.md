# Generative Diffusion Models for High-Dimensional Time Series

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
We propose a two-stage pipeline for high dimensional time series generation: (i) nonparametric kernel estimation for the conditional first and second moments of the underlying data increments to recover residuals, and (ii) score-based diffusion model trained on these residuals. We derive finite-time convergence estimates for reverse-time sampling in both total variation (TV) and Wasserstein-2 ($W_2$), with explicit dependence on the variance preserving noise schedule. Experiments on synthetic multivariate processes validate: (a) empirical TV and $W_2$ track the theoretical upper bounds, and (b) Monte Carlo estimates of test functionals achieve the predicted standard errors.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper focuses on the problem of time series modeling. Specifically, the proposed approach can be viewed as a combination of non-parametric estimation (i.e., Nadaraya–Watson estimation) and diffusion models. As I understand it, the generation process proceeds sequentially over time steps, where at each step a $\Delta$ value is computed and added iteratively to the previous point. This delta consists of two parts: a fixed component (e.g., the mean) and a residual component. The authors delegate the generation of the residual part to the diffusion model, while the fixed component is pre-determined through non-parametric estimation. In the experimental section, the authors evaluate the effectiveness of the proposed method on synthetic datasets.

### Strengths
The proposed decomposition-based time series generation framework grounded in non-parametric estimation is distinct from existing approaches such as multi-scale or trend–cycle decompositions. From my perspective, this design represents a interesting direction.

### Weaknesses
Overall, my reading experience with this paper was not very positive. I summarize several major weaknesses as follows: 
1. **Unclear problem formulation and motivation.** 
The paper lacks a precise problem definition or clear motivation. The direct transition from the *Introduction* to the *Method* section leaves readers puzzled—what exactly is this paper trying to achieve? Moreover, it remains unclear why the authors chose to combine non-parametric estimation with diffusion models. Why not simply use a diffusion model to jointly fit all components? I strongly recommend that the authors clearly articulate the problem statement and the motivation for their design, either in the *Introduction* or in a dedicated section. 

2. **Poor writing and presentation quality.** 
To illustrate with examples: in the *Introduction*, the authors briefly introduce diffusion models, followed by a very limited review of related work that is neither comprehensive nor contextualized. The contribution part then appears abruptly, with no conceptual transition. 
In the *Method* and *Theoretical Analysis* sections, I strongly suggest including a table summarizing all mathematical symbols (e.g., (x), (X), (\hat{X}), etc.) and reorganizing the overall structure for better readability. 
In the *Experiments* section, the authors seem to confuse the distinction between experiment setup and experiment results. Excessive attention is given to describing model parameters, while the analysis of results is extremely brief. 

3. **The proposed method is rather trivial.** 
In essence, the approach appears to be a straightforward combination of existing methods, without any meaningful mechanism to make them complement each other (i.e., achieving synergy beyond simple stacking). Moreover, the method has inherent limitations: it cannot generate long sequences and requires a known starting point for data augmentation. From a higher-level perspective, this is not substantially different from a classical autoregressive model. 
Combined with the high inference cost of diffusion models, the proposed framework inherits the inefficiencies of both autoregressive and score-based models. 
As for the theoretical analysis, while I did not examine every detail, it seems that the main conclusions are largely derived from existing results. 
Additionally, *Algorithm 1* in the paper appears to be a pre-processing and joint training–generation framework rather than a “training algorithm” per se; I suggest renaming it accordingly. 

4. **Experiments are too limited.** 
The experimental evaluation is overly simplistic—there are no baselines, no real-world datasets, and no standard metrics such as KL divergence or FID. The authors themselves seem aware of these limitations, as the *Discussion* section lists many future directions, including comparisons with time-series diffusion models and sensitivity analyses. 
Overall, the current level of empirical evaluation falls far short of what is expected for a top-tier venue like ICLR.

5. **Concerns about reproducibility.** 
The paper does not release any source code or detailed implementation information, making it difficult for readers to reproduce or verify the results.

### Questions
See the Weaknesses part.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes generating time series by decomposing it into first and second moments and then fitting a score-based diffusion model. The authors provide some theoretical results. There is a simple synthetic experiment showing how it works.

### Strengths
The paper is decent to read. The method is somewhat sound and the theoretical results are good, if correct, I didn't check in detail. The experiment shows a good results.

### Weaknesses
There are no real experiments, only a single synthetic experiment. There are no baselines, only comparison to an oracle. This is not good enough for a machine learning conference, especially since there are so many different time series datasets that this can be applied to and so many models to compare to.

Formatting is bad. Algorithm 1 is too big, it should be 2 algorithms at least, one for sampling and one for training. Theoretical results don't seem that important to be in the main text. Figure 1 is low quality and takes too much space.

### Questions
No particular questions. Please see the weaknesses for my comments. If there were actual experiments I would ask about the runtime comparison, learning efficiency, choice of hyperparameters etc.

### Soundness
2

### Presentation
1

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
This paper introduces a two-stage pipeline for generating high-dimensional time series. The proposed method first employs a Nadaraya-Watson kernel estimator to decompose time series increments into conditional mean, covariance, and residuals. In the second stage, a score-based diffusion model is trained on these extracted residuals. The authors present a theoretical analysis providing finite-time convergence bounds in Total Variation (TV) and Wasserstein-2 (W2) metrics, and validate their approach on a synthetic 20-dimensional vector autoregressive process.

### Strengths
The primary strength of the paper lies in its elegant and intuitive approach. Decoupling the learning of deterministic dynamics from the modeling of the innovation distribution could potentially simplify the task for the score network, a promising direction for time-series generation. Furthermore, the effort to provide a rigorous theoretical analysis and to use quantitative diagnostics (such as empirical TV/W2 distances and test functionals) for evaluation, rather than relying solely on qualitative plots, is commendable in principle.

### Weaknesses
1. Incomplete convergence guarantees. The reverse-time bounds are derived under the true residual distribution, but training uses residuals estimated via kernel smoothing. The stage-one estimation errors in $\hat{\mu}$ and $\hat{a}$ never enter the analysis, so the theoretical guarantees do not reflect the actual pipeline.

2. Unsubstantiated regularity assumptions. Assumption 3 requires the true score $\nabla \log p_t(x)$ to be globally Lipschitz. The experiments use Gaussian-mixture residuals, yet the paper does not provide theoretical or empirical justification that these conditions are satisfied, leaving the applicability of Theorem 1 uncertain.

3. Implementation opacity of the kernel stage. Although a data-adaptive $k$-nearest-neighbour bandwidth is mentioned, the paper gives no guidance on how this scales in 20 dimensions nor on keeping the estimated covariance matrices positive definite. Reproducibility remains in doubt.

4. Limited and contradictory empirical evidence. The evaluation considers only one synthetic process without baselines or ablations, and Table 1 reports a ≈40 % bias on the basket functional despite tiny Monte Carlo errors, directly undermining the claim that the model preserves key statistics.

### Questions
1. How are the kernel bandwidths selected in 20 dimensions? What measures are taken to ensure the estimated covariance matrices remain positive definite and well-conditioned, and what is the computational complexity of this stage?

2. Can you provide a rigorous verification that your chosen OU process satisfies the one-sided Lipschitz condition in Assumption 3? More critically, can you provide theoretical justification or empirical evidence that the score function of a Gaussian Mixture Model, after undergoing the OU diffusion process, remains globally Lipschitz continuous across all time steps $t \in [0,T]$? Without such justification, how do you defend the validity of Theorem 1?

3. How would the theoretical analysis and the final convergence bounds change if the estimation error from the first stage (i.e., errors in $\hat{\mu}$ and $\hat{a}$) were to be properly propagated through to the reverse SDE?

4. How do you explain the large discrepancy between the oracle and model-generated functional estimates in Table 1? Does this significant bias fall within your theoretical error bounds, and what does it imply about the model's ability to capture the true data distribution?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a two-stage pipeline for multivariate time-series generation: first, estimate conditional mean and covariance via Nadaraya–Watson kernel regression to extract residuals; second, train a score-based diffusion (variance-preserving OU/DDPM) on those residuals.

### Strengths
- Clear, modular algorithm (nonparametric residualization + diffusion), with an explicit training-and-sampling procedure
- Theoretical bounds expose how noise schedules impact convergence; W2 analysis via coupled SDEs is carefully laid out
- Synthetic study includes interpretable probes (functionals) and visual checks of residual geometry, including multimodality

### Weaknesses
- **Metric-dependent behavior and ranking variance.** Figure 1 suggests markedly different trends: W2 steadily decreases while TV appears comparatively flat, implying that relative model quality could look very different depending on the metric used. This raises concerns about stability of conclusions under metric choice and whether improvements are universal or metric-specific. 
- **Limited datasets and scope.** Evaluation is confined to a single synthetic setting (vector AR(1) with mixture innovations) without real-world time-series or domain-standard baselines (e.g., time-series DDPMs, latent-SDEs) to contextualize gains. The Discussion itself calls for comparisons and scaling studies, underscoring the current evaluation gap.  
- **Lack of analysis on why it works.** While empirical distances and functionals are reported, there is little ablation or diagnostic analysis explaining *why* kernel residualization plus diffusion yields the observed behavior (e.g., how bandwidth choice, lag augmentation, or residual non-Gaussianity affect learning and bounds). The paper notes future stress tests but does not include them.  
- **Strong assumptions and practicality.** The convergence results rely on regularity conditions (e.g., Lipschitz scores, drift growth) that may be hard to verify for high-dimensional, non-Markov real data. It is unclear how violations (e.g., heavy tails, regime shifts) would affect guarantees or practice. 
- **Algorithmic sensitivity not quantified.** The first stage uses locally adaptive k-NN bandwidths; there is no sensitivity study for bandwidth, lag selection, or Cholesky factor ambiguity (rotation) that could materially change residual structure and thus the diffusion target.

### Questions
See the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2
