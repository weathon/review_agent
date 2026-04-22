# Dual-Solver: A Generalized ODE Solver for Diffusion Models with Dual Prediction

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Diffusion models achieve state-of-the-art image quality.
However, sampling is costly at inference time because it requires a large number of function evaluations (NFEs).
To reduce NFEs, classical ODE numerical methods have been adopted.
Yet, the choice of prediction type and integration domain leads to different sampling behaviors.
To address these issues, we introduce Dual-Solver, which generalizes multistep samplers through learnable parameters that continuously (i) interpolate among prediction types, (ii) select the integration domain, and (iii) adjust the residual terms.
It retains the standard predictor-corrector structure while preserving second-order local accuracy.
These parameters are learned via a classification-based objective using a frozen pretrained classifier (e.g., MobileNet or CLIP).
For ImageNet class-conditional generation (DiT, GM-DiT) and text-to-image generation (SANA, PixArt-$\alpha$), Dual-Solver improves FID and CLIP scores in the low-NFE regime ($3 \le$ NFE $\le 9$) across backbones.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Diffusion models deliver state-of-the-art image quality. However, sampling is costly at inference time because it requires many model evaluations (number of function evaluations, NFEs). To reduce NFEs, classical ODE multistep methods have been adopted. Yet differences in the choice of prediction type (noise/data/velocity) and integration domain (half log-SNR/noise-to-signal ratio) lead to different outcomes. This paper introduces Dual-Solver, which generalizes multistep samplers by introducing learnable parameters that continuously (i) interpolate among prediction types, (ii) select the integration domain, and (iii) adjust the residual terms. It maintains the traditional predictor-corrector structure and guarantees second-order local accuracy. These parameters are learned with a classification-based objective using a frozen pretrained classifier (e.g., ViT or CLIP). On ImageNet class-conditional generation (DiT, GM-DiT) and text-to-image (SANA, PixArtα), Dual-Solver consistently improves FID and CLIP scores in the low-NFE regime (NFE ≤ 6) across backbones.

### Strengths
1. Generalizes multistep samplers by introducing learnable parameters that continuously interpolate among different prediction types (noise/data/velocity), select the integration domain, and adjust residual terms, addressing the issue of inconsistent outcomes caused by differences in prediction type and integration domain choices.
2. Structural compatibility and guaranteed accuracy: Maintains the traditional predictor-corrector structure (ensuring compatibility with existing multistep method frameworks) while guaranteeing second-order local accuracy, balancing structural familiarity and precision.
3. Learns the key parameters via a classification-based objective, leveraging frozen pretrained classifiers (e.g., ViT, CLIP) — this avoids the need for heavy independent training and capitalizes on existing well-performing models.

### Weaknesses
see Questions

### Questions
1. For a single prediction model, does Dual-Solver need to first transform the prediction into eps and $x_0$?

2. Is dual prediction necessary? Given that a predefined noise scheduler has deterministic prediction transforms( eps to v, v to eps, v to x0), I have doubts regarding the arguments for dual prediction. Although other works [1, 2] use a single prediction, no discriminator is employed for learning.

[1] Shuai Wang, Zexian Li, Tianhui Song, Xubin Li, Tiezheng Ge, Bo Zheng, Limin Wang, et al. Differentiable solver search for fast diffusion sampling. arXiv preprint arXiv:2505.21114, 2025.

[2] Neta Shaul, Uriel Singer, Ricky TQ Chen, Matthew Le, Ali Thabet, Albert Pumarola, and Yaron Lipman. Bespoke non-stationary solvers for fast sampling of diffusion and flow models. arXiv preprint arXiv:2403.01329, 2024

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Dual-Solver, a learnable, generalized ODE sampler that unifies noise/data/velocity predictions via dual prediction, and combines a log–linear domain transform with a second-order predictor–corrector and learnable residuals for stable few-step sampling.

It learns per-step parameters and the timestep schedule end-to-end using discriminative objectives (e.g., classifier/CLIP), yielding large quality gains at very low NFE across diverse backbones and tasks.

### Strengths
1. The proposed method is novel and solid theoretical. This work analyzes currently popular semi-linear samplers and different prediction targets and proposes a more general PC-based sampler. Contributions include:
    - unifies different prediction according to a dual-prediction parameterization. 
    - uses a learned change of variables in semi-linear family for integration
    - second order residual adjustment for p1c2

2.  Experimental sufficient. This work provides strong, multi-backbone results on FID. And there's enough ablations over predictor/corrector order and parameter freedoms

3. Limitations are acknowledged. 

Overall, this work proposed a well-motivated and clearly articulated solvers, especially for few-step sampling.

### Weaknesses
1. Personally, the article writing is somewhat difficult to follow up on. The overall structure is somewhat confusing to me.

2. The description of the parameter training part is insufficient. Especially the training process. 

3. The discussion of details for certain algorithms lacks intuitive understanding.

### Questions
1. About the training process, my current understanding is that after sampling M-1 steps, we directly optimize the corresponding 10*(M-1) groups of parameters with a set of labels or captions? Once these parameters are trained, the parameters can be reused for this backbone? Can the author provide a more specific training process?

2. Why classification objectives beat regression is not deeply unpacked. The paper shows a V-shaped accuracy–FID trend and argues “reaching the right decision region” is enough, but it stops short of a deeper mechanism: when/why classification consistently wins across backbones/NFE, and under what conditions it might overfit decision boundaries.

3. Consistency vs. transfer across backbones/settings.
It observes that learned parameter curves look similar across NFEs for a given backbone, yet recommends training per backbone/setting. Readers may wonder about the practical transfer value of those similarities (warm-starts, NFE interpolation, shared parameterizations), which isn’t explored methodologically.

4. Fig2 is not mentioned in the article. And I feel that the Sec 4 and 5 are somewhat abrupt logically, it seems subsection of Sec2? 


Overall, my main concern is 1. Please clarify the approach for obtaining parameters for all steps. If my concern addressed, I would be willing to raise my score.

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
This paper introduces Dual-Solver, which accelerates the sampling of diffusion models using learnable parameters $\gamma$, $\tau$, and $\kappa$. This framework incorporates and extends many existing multi-step and learning-based solvers, and abandons high-NFE (Number of Function Evaluations) teacher trajectories; instead, it updates the learnable parameters via classification loss or CLIP loss. The paper reports experimental results on ImageNet class-conditional and text-to-image diffusion models, effectively demonstrating the sampling performance of Dual-Solver under low NFE.

### Strengths
1. It constructs a generalized ODE solver based on a "predictor-corrector" structure. Through explicit interpolation of $\gamma$ and domain transformation via $\tau$, it unifies noise/data/velocity prediction methods within a single framework.
2. It proposes a solver parameter update method based on classification loss or CLIP loss, abandoning the parameter update learning strategy that relies on teacher trajectories. This reduces computational overhead and improves performance under low NFE.
3. The paper features rigorous mathematical derivations, covering the mathematical foundations of dual prediction and log-linear domain mapping. It also provides a complete proof of the second-order local truncation error.

### Weaknesses
1. The method is derived based on second-order accuracy, but it does not discuss potential issues when extending to higher-order schemes, which limits the further performance improvement of Dual-Solver.
2. It lacks comparative results with current advanced learning-based solvers, such as EPD-Solver[1] and AMED-Solver[2].
3. The parameter update of Dual-Solver relies on the cross-entropy loss of a classifier or CLIP loss, which means the parameter update of Dual-Solver will fail for unguided conditions.

[1] Zhu B, Wang R, Zhao T, et al. Distilling Parallel Gradients for Fast ODE Solvers of Diffusion Models[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025: 19557-19566.

[2] Zhou Z, Chen D, Wang C, et al. Fast ode-based sampling for diffusion models in around 5 steps[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 7777-7786.

### Questions
1. How does Dual-Solver perform compared to current advanced learning-based solvers (e.g., EPD-Solver[1], AMED-Solver[2])?
2. The paper proposes that using a "weak classifier" with lower accuracy can alleviate the problem of reduced sample diversity, but the results in Table 7 do not seem to clearly demonstrate the effectiveness of this strategy. Could you provide further explanation?
3. The paper learns parameters for each step but does not use ablation studies to quantify the effect of globally shared parameterization. Given that shared parameters can effectively reduce the number of learned parameters, could the authors conduct more detailed comparative experiments?

[1] Zhu B, Wang R, Zhao T, et al. Distilling Parallel Gradients for Fast ODE Solvers of Diffusion Models[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025: 19557-19566.

[2] Zhou Z, Chen D, Wang C, et al. Fast ode-based sampling for diffusion models in around 5 steps[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 7777-7786.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Dual-Solver, a learned predictor–corrector sampler for diffusion/flow-matching models that targets the low-NFE regime. It introduces three per-step learnable knobs: γ (interpolates prediction type across noise/data/velocity), τ (a log–linear domain transform that interpolates between λ- and ρ-domains), and κ (a residual term that preserves second-order local accuracy). The method keeps second-order local truncation error, learns the timestep schedule, and trains all solver parameters via a classification-based objective that uses a frozen classifier.

### Strengths
1. Unified, principled parameterization. γ cleanly bridges noise/data/velocity predictions; τ smoothly interpolates λ↔ρ domains; κ adjusts second-order residuals

2. Strong low-NFE results across backbones. Consistent FID/CLIP gains at ≤6 steps on DiT, SANA, PixArt-α, and GM-DiT.

3. Classification-based training removes the need for high-NFE teacher samples, reducing preparation cost and directly optimizing a perceptual proxy.

### Weaknesses
1. Training introduces ten learnable parameters per step (in addition to a learned schedule), which could increase brittleness or overfitting to specific backbones and guidance settings. Also, does the method require separate training runs for different NFE targets?

2. Because CE/CLIP is computed with a frozen classifier, the optimization might bias samples toward ‘classifier-friendly’ patterns. How do the authors mitigate proxy-objective leakage and ensure alignment with the intended generative quality?

### Questions
Does the method require separate training runs for different NFE targets?

### Soundness
4

### Presentation
3

### Contribution
3
