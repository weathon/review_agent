# Measurement-Consistent Langevin Corrector: A Remedy for Latent Diffusion Inverse Solvers

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
With recent advances in generative models, diffusion models have emerged as powerful priors for solving inverse problems in each domain. Since Latent Diffusion Models (LDMs) provide generic priors, several studies have explored their potential as domain-agnostic zero-shot inverse solvers. Despite these efforts, existing latent diffusion inverse solvers suffer from their instability, exhibiting undesirable artifacts and degraded quality. In this work, we first identify the instability as a discrepancy between the solver’s and true reverse diffusion dynamics, and show that reducing this gap stabilizes the solver. Building on this, we introduce *Measurement-Consistent Langevin Corrector (MCLC)*, a theoretically grounded plug-and-play correction module that remedies the LDM-based inverse solvers through measurement-consistent Langevin updates. Compared to prior approaches that rely on linear manifold assumptions, which often do not hold in latent space, MCLC operates without this assumption, leading to more stable and reliable behavior. We experimentally demonstrate the effectiveness of MCLC and its compatibility with existing solvers across diverse image restoration tasks. Additionally, we analyze blob artifacts and offer insights into their underlying causes. We highlight that MCLC is a key step toward more robust zero-shot inverse problem solvers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a Langevin corrector for inverse solvers using a latent diffusion model (LDM) as the prior. The correction step preserves measurement consistency by being orthogonal to the measurement gradient. The authors show results on many inverse problems with Stable Diffusion v1.5 as the LDM prior, adding the proposed corrector on top of many existing latent diffusion inverse solvers. They also compare to existing solvers that aren’t plug-and-play-compatible with their corrector. In many situations, they find improvement in PSNR, LPIPS, FID, and Patch FID. They also provide an experiment showing that “blob” artifacts in decoded images arise from scaled outliers in the latent space. They claim that their proposed corrector may help alleviate this type of artifact.

### Strengths
* The proposed corrector is a principled way to reduce artifacts that may arise with existing LDM solvers.
* The corrector is easy to plug in to existing solvers.
* The authors evaluate on a variety of inverse problems and plenty of baselines.

### Weaknesses
* The technical contribution is quite marginal; it is just a Langevin corrector step that follows directly from predictor-corrector sampling with Langevin dynamics, and the measurement consistency is a minor additional piece.
* The qualitative results in Figure 5 are not convincing. I have to really zoom in to see some artifacts. And I am not convinced that the baseline artifacts in Figure 1 are in fact artifacts. They look like they could be plausible artifacts (e.g., from JPEG compression) in images that the LDM was trained on.
* The proposed corrector doesn’t provide an obvious improvement to LatentDAPS.
* The blob analysis is informative, but it is unclear how and by how much the proposed corrector would alleviate such artifacts.
* Evaluating a Patch FID score with 3x3 patches does not seem meaningful to me.
* I’m confused about the fundamental assumption being demonstrated by Figure 3. I see that the KL divergence is quite bad at the beginning of reverse diffusion (without the corrector), but isn’t it fine as long as the KL divergence approaches 0, since what we care about is the distribution at time 0? The plot in Figure 3 shows that the KL divergences with and without the corrector approach the same value by time 0. I have a similar question based on Figure 2. I can see that the latents look messier with the baseline, but that doesn’t necessarily mean their decoded images are bad.

### Questions
* Please clarify the last weakness point. Why is it a problem for the KL divergence to be high at the beginning of reverse diffusion process, even if it approaches a small value by the time reverse diffusion is done?
* How can you be sure that what appear to be artifacts are not actually coming from the true prior?

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
This paper argues that existing LDM-based inverse solvers often suffer from artifacts and degraded quality. The authors attribute this to a divergence between the solver's dynamics and the true reverse diffusion process. They propose a plug-and-play module that applies a projected Langevin dynamics step to "correct" the latent state at each iteration, pulling it back toward the true data manifold. The evaluations are comprehensive.

### Strengths
1. The proposed MCLC is a clever solution. The use of Langevin dynamics to reduce KL divergence and the projection to maintain measurement consistency are both well-justified.

2. The method is designed as a modular add-on that can improve multiple existing solvers (LDPS, PSLD, ReSample), which is a plus.

3. The method achieves significant and consistent improvements in perceptual metrics (LPIPS, FID, and the proposed P-FID) across a wide range of tasks and datasets.

### Weaknesses
1. The appendix reveals that MCLC requires its own set of hyperparameters that must be individually tuned for every base solver and every restoration task. This is a massive tuning effort that directly contradicts the paper's claims of simplicity and generality.

2. The method fails to provide significant improvements for the more advanced LatentDAPS solver. If the method only works on older/simpler solvers and is incompatible with newer SOTA methods, its contribution is limited to being a "patch" for those specific methods rather than a general-purpose solution.

3. The authors concede that improvements in PSNR are "modest". This indicates the method is primarily an artifact-removal and perceptual enhancement technique, which may be less valuable for scientific or high-fidelity applications where accuracy (PSNR) is more important than visual pleasantness (LPIPS/FID).

4. The paper repeatedly critiques prior work (e.g., DiffStateGrad, MPGD) for relying on a strong linear manifold assumption that fails to hold in latent space. However, MCLC's own projection is based on the first-order gradient, which is itself a linear approximation of the measurement-consistent manifold. The paper even admits its consistency guarantee is only "up to the first-order Taylor expansion", effectively trading one linear approximation for another.

### Questions
1. Given the extensive task-specific tuning in Tables 4-7, how can the authors justify the zero-shot and plug-and-play claims? Is there a single, default set of MCLC hyperparameters that provides a reasonable improvement across all tasks?

2. Can the authors elaborate on why their first-order gradient projection is fundamentally superior to the linear manifold assumption they critique? Both appear to be linear approximations of a nonlinear reality.

3. The paper states MCLC is less compatible with LatentDAPS because LatentDAPS does not inherit reverse diffusion dynamics. Does this imply MCLC is fundamentally incompatible with any solver that breaks from the standard DDIM/DDPM reverse process, limiting its future applicability?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes the Measurement-Consistent Langevin Corrector (MCLC), a plug-and-play method that augments latent diffusion inverse solvers with an orthogonally projected Langevin corrector step to preserve measurement consistency. The authors claim this corrector mitigates artifacts, narrows the KL gap between solver dynamics and the true reverse diffusion, and improves perceptual metrics across various inverse problems.

### Strengths
1. The paper is clearly written and well-structured, with consistent notation and clear transitions between sections.
2. The authors provide both theoretical and empirical analyses, including derivations that connect the proposed correction step to reduced KL divergence, which helps ground the method conceptually.
3. The evaluation covers a diverse set of inverse problems (super-resolution, deblurring, inpainting, HDR, nonlinear deblurring) and includes comparisons with multiple baseline solvers, demonstrating good experimental thoroughness.

### Weaknesses
1. The core idea of MCLC, adding a Langevin correction step with an orthogonal projection, appears very similar to established approaches such as predictor-corrector sampling in Song et al., 2021. The paper acknowledges this but presents the orthogonal projection for “measurement consistency” as its main novelty. This modification feels incremental, raising the question of whether constraining the Langevin step geometrically constitutes a substantial conceptual advance.
2. While the paper attributes artifacts and “blob” patterns to inherent issues in latent diffusion inverse solvers, it does not rule out simpler causes such as suboptimal hyperparameter tuning. Competing methods may not exhibit these artifacts simply because their measurement-consistency step sizes or regularization terms are better tuned.
3. Using only 100 images from each dataset is a bit unconvincing. Standard validation splits are publicly available for both datasets (1k images) and are commonly used in related work. The paper should report results on these standard sets or, at minimum, provide a clear justification and description of how the 100-image subsets were selected.
4. The paper does not provide any analysis or discussion of the additional computational burden introduced by MCLC. Since the method applies an extra Langevin and projection step at each diffusion iteration, it likely increases runtime and memory usage, but no quantitative results or comparisons are given.

**Minor Comments:**
- The effectiveness of MCLC appears sensitive to the choice of the corrector step size $\eta_t$, which the authors note remains non-trivial to select. Since the method’s stability and measurement consistency likely depend on this tuning, it would be helpful to discuss how $\eta_t$  was chosen in practice and whether results are robust to this parameter.

### Questions
1. *Regarding weakness 2:* Could the observed artifacts instead stem from under-tuned baselines rather than a fundamental flaw in their dynamics? Have the authors verified that measurement-consistency step sizes (e.g., $\zeta$ and $\gamma$ in Alg. 4)  and solver parameters were optimized fairly for all methods?
2. *Regarding weakness 4:* Can the authors report the actual runtime and memory overhead of MCLC relative to the base solvers, and clarify whether the added cost is negligible or significant in practice? Did you select k=3 heuristically (in Appendix C2)?

### Soundness
3

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
This paper introduces the Measurement-Consistent Langevin Corrector (MCLC), a novel, plug-and-play module designed to remedy artifacts and degraded quality commonly seen in existing zero-shot inverse solvers based on Latent Diffusion Models (LDMs).

### Strengths
1. The plug-and-play nature is highly significant, showing that simple, post-update correction can elevate the performance of various base solvers (LDPS, PSLD, ReSample) to be comparable to, or even surpass, more complex, recent methods (LatentDAPS). The method is even shown to be applicable to Pixel Diffusion Models (PDMs)

### Weaknesses
1. Langevin correction is a well-known technique in diffusion models that can be used with any diffusion solver, which is not novel to me.

2. Measurement Consistency correction requires additional forward and backward propagation, lacking relevant analysis of computation time and consumption.


3. To my knowledge, we only need a simple Restart [1][2] to alleviate accumulated errors and adverse artifacts, which is simpler and more intuitive for me.

[1] RePaint: Inpainting using Denoising Diffusion Probabilistic Models

[2] Restart Sampling for Improving Generative Processes

### Questions
Why did the author only conduct experiments on latent  diffusion, while their baseline method DiffState also performed experiments in the pixel domain.

What are the advantages of the proposed method compared to Restart method.

### Soundness
2

### Presentation
2

### Contribution
2
