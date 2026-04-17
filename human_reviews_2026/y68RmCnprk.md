# Progressive Latent Calibration for Stable Score Distillation

- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
Recent advancements in Score Distillation Sampling (SDS) have significantly accelerated progress in text-to-3D generation by leveraging pre-trained 2D diffusion models to supervise 3D representations. However, SDS often suffers from high variance and produces over-smoothed outputs, limiting the quality of the synthesized 3D assets. While recent methods have introduced DDIM inversion to stabilize optimization, we identify that repeated DDIM inversion introduces discretization errors that accumulate over thousands of iterations, ultimately leading to severe artifacts such as structural distortions and color degradation. To address these limitations, we introduce a novel score distillation framework that eliminates the reliance on DDIM inversion by leveraging multi-step pseudo-ground-truth sampling with progressive latent calibration. Our approach explicitly estimates and reintegrates information loss about the original rendering from a 3D representation during multi-step sampling, thereby preserving semantic fidelity and reducing variance across training iterations. Extensive experiments show that our method consistently outperforms existing inversion-based and standard score distillation approaches in generating high-fidelity 3D assets from text prompts. The anonymous project page is available at https://anonymous-iclr-sd.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Progressive Latent Calibration (PLC), a novel score distillation framework for text-to-3D generation designed to overcome critical limitations in existing methods. The authors first identify that while standard Score Distillation Sampling (SDS) produces over-smoothed results due to high variance, recent inversion-based methods suffer from the accumulation of discretization errors from repeated DDIM inversion, leading to significant artifacts like color degradation and structural distortion. To solve this, PLC proposes an inversion-free strategy that utilizes multi-step pseudo-GT sampling. The core mechanism "progressively calibrates" this sampling process by explicitly estimating and reintegrating information loss about the original 3D rendering at each step; this is achieved by adding a corrective noise residual which is the difference between the actual noise and an unconditional prediction, back into the guided denoising step. This method successfully reduces variance and preserves high-frequency details without relying on DDIM inversion, demonstrably outperforming existing approaches in generating high-fidelity 3D assets.

### Strengths
1.	The paper clearly identifies and diagnoses a critical, yet previously overlooked, problem in score distillation: the accumulation of discretization errors from repeated DDIM inversion, which leads to artifacts like color degradation.
2.	The proposed Progressive Latent Calibration (PLC) framework is a highly novel inversion-free alternative.
3.	extensive and high-quality evaluations, including thorough quantitative comparisons against a strong set of baselines, compelling qualitative results, and a user preference study that validates the perceptual quality of the generated 3D assets .

### Weaknesses
1.	The proposed PLC method remains essentially a 2D operation focused on improving single-view fidelity, without introducing priors or constraints for multi-view consistency. As acknowledged, it still suffers from the Janus problem.
2.	The cost analysis is misleading—PLC scales linearly with sampling steps and requires multiple U-Net passes, making it substantially more expensive than single-step baselines.
3.	The introduced auxiliary loss $\mathcal{L}_{aux}$ adds complexity but offers marginal benefit. The paper’s own ablations show that most performance gains stem from the main PLC-guided loss $\mathcal{L}m$, while $\mathcal{L}{aux}$ yields negligible quantitative and qualitative improvements.

### Questions
The paper could be improved by comparing with flow matching based method like FLowDreamer[1] or RFDS[2], which addresses trajectory inconsistency via Rectified Flow rather than stochastic heuristics like PLC, offering a deterministic and potentially cleaner alternative.
[1]. Li H, Chu X, Shi D, et al. Flowdreamer: Exploring high fidelity text-to-3d generation via rectified flow[J]. arXiv preprint arXiv:2408.05008, 2024.
[2]. Yang X, Chen C, Yang X, et al. Text-to-image rectified flow as plug-and-play priors[J]. arXiv preprint arXiv:2406.03293, 2024.

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
This paper uses a score distillation approach to generate 3D models guided by text prompts.  In this framework, images of the 3D model are rendered and diffusion methods are used to guide the evolution of the 3D model by determining how to adjust the model to match denoised images.  The paper identifies a potential issue with current approaches and introduces a technique that adds a corrective noise during this iterative process to produce a more accurate trajectory.  Quantitative and qualitative experiments support the potential gains of this method.

### Strengths
The proposed method seems to be well motivated by solid intuitions.

Experiments indicate significant gains in performance.

### Weaknesses
The proposed method is somewhat incremental to existing approaches.  Results relative to ISM seem better, but not by a lot.  

The paper is not really self-contained, and assumes implicitly some knowledge of existing approaches.  This would be hard to avoid, as the method builds on fairly complex prior work.

### Questions
In the user preference test, why was the method not compared to VSD?

Is it possible to measure how much of the gain from the method is due to better color, and how much is due to other improvements?

### Soundness
4

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
Progressive Latent Calibration introduces an alternative to the Score Distillation Sampling (SDS) gradient in the Dreamfusion text-to-3D optimization framework. The method identifies problems with diffusion inversion (solving the PF-ODE forward in time) to obtain view-consistent latents and proposes to instead use the typical sampling-based diffusion forward process with a modified denoising process. This reduces the variance of the gradient, improving semantic and visual fidelity.

### Strengths
- Progressive latent calibration introduces a correction term in the denoising that pulls the denoising toward the current clean sample/render, an approach that has been effective in other settings
- Analysis on the deficits of diffusion inversion (particularly compounding errors) in this context is reasonable
- PLC is less computationally expensive than DDIM inversion because it doesn't require NFEs for the forward process
- User study is the most reliable way to evaluate such methods due to the lack of reliable metrics
- Results seem qualitatively good

### Weaknesses
- Latent calibration is mostly heuristically motivated, though it is reasonable
- Ablations do not probe sensitivity to many hyperparameters, for instance the residual scaling or the denoising schedule (step lengths)
- Methods in this problem setting are known to have inconsistent performance across text prompts or random seeds. It would be useful to see failure modes and how they compare to other methods.
- Scope of the method's contribution is relatively limited.

### Questions
- How well does it apply to other versions of Stable Diffusion, particularly SDXl, which is known to have problems with SDS

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
Although the current mainstream "Score Distillation Sampling" (SDS) method has made significant progress by utilizing 2D diffusion models, it generally suffers from problems such as over-smoothed output results, oversaturated colors, and excessively large update variance. To address these issues, some recent methods have introduced DDIM inversion technology to stabilize the optimization process. However, the authors found that DDIM inversion introduces and accumulates discretization errors, which, after thousands of iterations, lead to serious structural distortions and color distortion in the generated 3D models. The authors proposed a new score distillation framework called "Progressive Latent Calibration" (PLC). PLC abandons DDIM inversion and adopts a multi-step sampling strategy to generate high-quality "pseudo-ground truth" samples to replace the single-step denoising in SDS, thereby obtaining more stable and refined supervision signals.

### Strengths
The article is clear and fluent, making it easy to read. The "non-inversion" approach of PLC has opened up a new technical path for solving the stability problem of score distillation. The article clearly reveals its motivation through preliminary experiments in the appendix section. Whether from the perspective of quantitative data or visual effects, this method has achieved excellent results. The qualitative graphs (such as Figure 4) also clearly demonstrate its advantages in color and details.

### Weaknesses
- Does the number of sampling steps N have a linearly increasing impact on the memory consumption and computational efficiency of PLC? - - Does the model quality also increase with the increase of N? Why was N=4 chosen as the final number of sampling steps? Are there more examples of Ablation studies?
- There are some detailed questions. Please elaborate on why the auxiliary loss chooses the unconditional prediction after the first update? Why not use conditional prediction? Why wasn't supervised prediction continued afterward? Additionally, why is it matched with the original CFG prediction?

### Questions
Please see weakness

### Soundness
2

### Presentation
2

### Contribution
2
