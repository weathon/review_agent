# Overshoot and Shrinkage in Classifier-Free Guidance: From Theory to Practice

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Classifier-Free Guidance (CFG) is widely used in diffusion and flow-based generative models for high-quality conditional generation, yet its theoretical properties remain incompletely understood. By connecting CFG to the high-dimensional framework of diffusion regimes, we show that in sufficiently high dimensions it reproduces the correct target distribution—a “blessing-of-dimensionality” result. Leveraging this theoretical framework, we analyze how the well-known artifacts of mean overshoot and variance shrinkage emerge in lower dimensions, characterizing how they become more pronounced as dimensionality decreases. Building on these insights, we propose a simple nonlinear extension of CFG, proving that it mitigates both effects while preserving CFG’s practical benefits. Finally, we validate our approach through numerical simulations on Gaussian mixtures and real-world experiments on diffusion and flow-matching state-of-the-art class-conditional and text-to-image models, demonstrating continuous improvements in sample quality, diversity, and consistency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper shows that the CFG artifacts of overshoot and variance shrinkage that have been observed in low dimensions vanish in high dimensions, via a theoretical analysis of dynamical regimes. They then propose a nonlinear extension of CFG that mitigates the low dimensional artifacts while preserving the high dimensional behavior.

### Strengths
The paper is clear and well-written. The problem is well-positioned within the literature (particularly wrt Chidambaram and Biroli). It is nice to see the question of why CFG 'works' despite the known issues in low dimensions being addressed. The dynamical regimes analysis is interesting. The improved diversity on ImageNet is promising.

### Weaknesses
Please see questions.

As the authors note in the limitations, although the theory can help improve diversity while maintaining fidelity/quality, it still cannot explain how CFG improves fidelity/quality.

### Questions
The Biroli dynamical regimes analysis applied to the CFG deals with a mixture of two Gaussians. How can we be sure the conclusions carry over to more complex multimodal distributions?

How does this analysis relate to the common empirical observation that conditional diffusion models often start to behave like unconditional denoisers at low noise levels? (Does the analysis actually rely on that behavior? Or can it help explain it?)

L328: Do you require $0 < \alpha \le 1$ or can $\alpha$ be greater than zero?

Clarification: (L332) When the difference between conditional and unconditional scores is zero, CFG automatically reproduces the conditional score. So the goal of the power-law is to further (nonlinearly) damp small (but nonzero) score differences?

Why is Power-Law CFG helpful in high-dimensional settings like ImageNet? I thought the analysis showed that CFG is already "correct" in high dimensions (i.e. recovers conditional distribution without distortion)?

Minor edits:
L 82-83 did you swap descriptions of mean overshoot and variance shrinkage?

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
4

### Summary
The paper analyzes Classifier-Free Guidance (CFG) for diffusion/flow generative models. Using a high-dimensional dynamical-regime framework, it proves that in sufficiently large dimensions CFG reproduces the correct conditional distribution and accelerates convergence before “speciation” (transition between Regime I and II). In finite dimensions, it formally characterizes mean overshoot and variance shrinkage and attributes them to a more confining effective potential. Motivated by this, it introduces a nonlinear “power-law” CFG that scales the score difference by $|\cdot|^\alpha$, which provably mitigates both artifacts while preserving the high-dimensional guarantee. Experiments span Gaussian-mixture simulations and ImageNet/T2IM models (DiT, EDM2, MMDiT), showing improved FID/recall and robustness to $\omega$.

### Strengths
1. The paper's main theoretical punch is the "blessing-of-dimensionality" result. The authors formally prove for the first time that CFG can generate the exact target distribution under appropriate conditions (infinite dimensions).This is a huge deal because it formalizes why this heuristic is so stable and effective in high-dimensional domains like images. (Eq. (4)–(6); Sec. 4.1; Appx. B–D).

2.  The authors' theory also provides a clear explanation for well-known flaws in CFGs. They show that they are mathematically necessary consequences of finite dimensionality and directly relate them to the curvature of the effective potential (Section 4.2; Equation (35)).

3.The authors propose a theoretically derived solution, the power-law CFG, which is very simple. It requires only the addition of a `$|\cdot|^\alpha$` term, and it also has theoretically driven reassurances (Equation (7)).

### Weaknesses
1. Theoretical assumptions regarding accurate scores; practical mismatch issues remain unresolved.
The paper obtains true scores under high-dimensional guarantees and finite-dimensional analysis assumptions. The paper acknowledges that it cannot explain why (linear) CFGs perform well with imperfect estimators (Section 3.2, last paragraph; Conclusions/Limitations). Direct robustness analysis of the scores to the error/noise of nonlinear rules (e.g., perturbation bounds) is also lacking.

2. The scope of the study focuses on Gaussian mixture structures.
The core derivation of this paper relies on mixture models with isotropic covariances and separable scaling (Section 4; Appendices B-D); extensions to non-GMM real data are controversial but unproven. While Appendix C discusses multi-component and non-centered mixture models, empirical validation of multi-class schemes focuses primarily on standard vision datasets rather than controlled non-isotropic cases.


3. While $\alpha$ is considered robust ($\alpha \approx 0.9$), the method introduces a second hyperparameter; more extensive ablations (dataset, sampling step size, $omega$ range) are shown in Appendix G, but some edge cases (e.g., extremely small $omega$ or very low step size) require minimal exploration. (Figure 4; Appendix G).

4.  I note that you cite an extension of flow matching with velocity prediction (Appendix G.2.1 notes), but the main theoretical section does not provide a complete derivation.

### Questions
refer to weakness

### Soundness
3

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
3

### Summary
This paper provides a theoretical analysis of how CFG affects the generation of mixtures of Gaussians via diffusion models. The authors arrive at the result that in the scenario of exact score estimation the effect of CFG on the generated distribution tends to zero as the number of data dimensions tends to infinity. Conversely, they find that in finite/small dimensional cases CFG causes the generated means to shift and the variances to shrink. The authors propose to adjust CFG according to the norm of the score difference in order to mitigate the above effects and evaluation on various image generation models/datasets.

### Strengths
- The theoretical contributions advance the understanding of CFG.
- The presentation of the paper is clean and professional. Theoretical results are presented in a structured and understandable manner.
- The scale and variety of the experiments are appropriate.

### Weaknesses
The paper load for ICLR this year has been large, and so I have not been able to spend as much time as I would like on reviewing. I encourage the authors to correct any errors/misunderstandings I may have with regards to the paper.  


1. **weak motivation for proposed approach**
    1. The authors motivate their Power-Law CFG as a method to reduce mean overshoot and variance shrinkage (implying generation is in the *finite dim* regime), but then justify their approach by saying it retains the *high dim* properties of CFG that they've derived. It seems like they are suggesting that their approach won't "break" the nice high-dimensional properties of CFG, but then are focused on addressing a problem that only exists in low/finite dim regime. I find this link between the theoretical and practical part of the paper quite weak.   
    1. It is unclear to me how the proposed approach reduces the mean overshoot and variance shrinkage, as the authors don't provide an explanation in the main body of the paper. 
    1. From what I can tell, any heuristic scaling of CFG [1] (e.g. according to timestep) will also retain the nice theoretical properties, which hardly makes power-law scaling unique. Besides, practically speaking it seems like the proposed approach behaves similarly to existing heuristics, gradually increasing CFG as generation progresses [1].
    1. The intuitive description of power-law scaling is misleading. First of all the assertion that score difference large -> reliable, small -> unreliable is not clearly explained or motivated. Secondly, $x_t=\alpha_t x_0 + \sigma_t \epsilon$, $S_\theta = -\epsilon_\theta/\sigma_t$, that is to say from a noise prediction perspective the magnitude of the score estimate depends on the noising schedule. As noise level $\sigma_t$ decreases over the course of generation this will have a direct impact on the power-law scaling. Importantly, the $\sigma$ scaling has nothing to do with the actual difference in spatial content predicted by the conditional and unconditional models and is instead dependent on the diffusion time. In fact, the authors acknowledge that applying their scaling directly to flow velocity results in different behaviour, but do not offer much additional insight. 


2. **Tepid experimental results**
    1. Although improvements over vanilla CFG seem notable, improvements over other improved CFG methods are much more marginal. Moreover, due to the aforementioned weak motivation there isn't really a clear explanation for why to prefer power-law scaling to competing approaches. 

3. **Lack of intuition to support theory (minor)**
   1. For a general reader interested in using/improving CFG, I feel like this paper would benefit from more intuitive explanations to help understanding. Personally, I still have not been able to build an intuition for why the effect of CFG disappears when dimensionality tends to infinity, making it harder to appreciate the contribution. (I have some vague ideas of distributions tending to hyperspherical shells, leading to different behaviour, but nothing concrete.)

[1] Wang et al, Analysis of Classifier-Free Guidance Weight Schedulers, TMLR 2024.

### Questions
See above

My main issues with the paper are (according to my current understanding)
1. The proposed approach is weakly linked with the theoretical contribution.
2. The concrete form of the power-law CFG is not well motivated.

I'm open to increasing my score if the authors can help me understand better the above issues. I have also not read the appendices in detail.

Additionally, I would like to clarify, is the score difference a vector norm or an elementwise difference?

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
3

### Summary
The paper observes the problem of mean overshooting and variance shrinkage resulted by classifier-free gudiance during Diffusion Sampling process. The main results are drawn from the connection of dynamical regimes and classifier-free guidance. Main observations are conducted on  the mixture of Gaussian (2 modes) with the same weight and variance. This gives us the division of CFG into three distinct steps (1) guidance before speciation (2) align before exiting regime I (3) follow unguided path. However, when the dimensionality is limited, the observation (2) is no longer correct and over push the conditional information. To solve this problem in lower-dimension setting, the authors propose power-law guidance where it will enhance the guidance when the guidance signal is strong, and purging the guidance when the guidance is weak. The results show the significant improvement

### Strengths
1. The paper is well-written
2. The theoretical part is well-written
3. The experiments are comprehensive and up-to-dated.
4. The improvement is quite significant
5. The proposed method is simple

### Weaknesses
The main drawback of the paper lies in its over-complication. In my opinion, simplicity should be viewed as a strength rather than a weakness. The proposed method is inherently simple, and the results are largely empirical. Therefore, I strongly suggest simplifying the presentation and avoiding unnecessary theoretical analysis that does not contribute to the core message.

1. **Overly Strong Assumptions:**  
   The assumptions made in the paper are excessively restrictive and unrealistic.  
   - The use of a mixture of two Gaussians is not representative of most real-world datasets.  
   - The assumption that both Gaussians share the same variance is also impractical and lacks justification.  
   - Furthermore, assuming the means to be +m and –m is arbitrary and overly specific.  
   In contrast, *Biroli (2024)* assumes only the existence of two classes without making any assumptions about their distributions, means, or variances. The authors may want to consider relaxing or revising these assumptions to enhance realism and generalizability.

2. **Unnecessary First Key Finding:**  
   The first key finding adds confusion rather than clarity. It does not appear essential to the overall narrative or conclusions. The paper would benefit from removing this section and instead starting directly from the second finding (e.g., Figures 8 and 12), focusing on the proposed method for addressing low-dimensional cases.

3. **Weak Connection to Previous Findings:**  
   The third key finding seems only loosely connected to the earlier results. It is unclear how the concepts of “Regime I” and “Regime II” relate to the presented equation. This section requires clearer logical linkage and explanation.

4. **Experimental Results and Resolution:**  
   The experiments report results for resolutions of 512 and 256. According to the paper’s own claims, one might expect the method to perform even better on lower resolutions such as 64×64 or 32×32. If this is indeed the case, the authors should provide empirical evidence to support this claim.

### Questions
Please see the weakness. I would encourage the authors to simplify the paper.

### Soundness
3

### Presentation
2

### Contribution
3
