# Mitigating Diffusion Model Hallucinations with Dynamic Guidance

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Diffusion models, despite their impressive demos, often produce hallucinatory samples with structural inconsistencies that lie outside of the support of the true data distribution. Such hallucinations can be attributed to excessive smoothing between modes of the data distribution. However, semantic interpolations are often desirable and can lead to generation diversity, thus we believe a more nuanced solution is required. In this work, we introduce $\textbf{Dynamic Guidance}$, which tackles this issue.  Dynamic Guidance $\textit{mitigates}$ hallucinations by selectively sharpening the score function only along the pre-determined directions known to cause artifacts, while preserving valid semantic variations.  To our knowledge, this is the first approach that addresses hallucinations at generation time rather than through post-hoc filtering. Dynamic Guidance substantially reduces hallucinations on both controlled and natural image datasets, significantly outperforming baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to mitigate hallucinations in diffusion-generated images. The core idea is to use a dynamic classifier guidance scale at each timestep to replace the previous fixed guidance. Such dynamic guidance is done by performing classification on the noisy sample per timestep. Huge and consistent gain is experimentally reported, and the authors also provide detailed empirical analyses on the effects of Dynamic Guidance.

### Strengths
1. This work is well-motivated, and the proposed method is closely related to the motivation.
2. The simplicity of the proposed method makes it very easy to integrate into current pipelines.
3. The experimental improvement is impressive, and there are also many empirical observation experiments.

### Weaknesses
1. (Major) This work is an extension of classifier guidance, but commercial-scale diffusion models mostly use classifier-free guidance nowadays, where there are uncountable classes, as semantics are continuous. Is there any possibility to extend the idea of Dynamic Guidance to classifier-free guidance?
2. (Minor) While the score sharpening effect is empirically observed with abundant experiments, could there be some more rigorous and theoretical discussion about why this can happen? In particular, why can Dynamic Guidance isolate dimensions related to hallucinations from unrelated ones? 
3. (Minor) Language-wise, many commas are missing where there should be a punctuation. A standard grammar check pass would easily fix this.
4. (Minor) Dynamic Guidance can fail if the classifier is not well trained. This is a natural result of being based on classifier guidance, so I won't blame the authors for this weakness.

### Questions
The authors mention that there are also studies trying to address hallucinations from the perspective of text-image misalignment. Why do the authors prefer to address this problem at the sampling stage rather than the training stage, which might be more fundamental in my opinion?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors analyze hallucination in diffusion models and argue that it arises from interpolation between data modes. They hypothesize that the learned score function becomes too flat in these interpolation regions, causing the sampler to fall into traps and fail to escape. To mitigate this, they propose a selective guidance method that pushes samples toward the nearest class direction at each step, thereby steering them away from regions where hallucination typically occurs.

### Strengths
- **Easy-to-follow and intuitive approach:**
    The proposed method dynamically applies classifier guidance during unconditional generation, sharpening the distribution and easing generation away from mode-interpolated regions, thus reducing hallucination.
    
- **Well-designed toy examples:** The toy examples provides clear observations and insightful analysis, helping establish the problem motivation.

### Weaknesses
- **Limited experimental setting and weak scalability:**
    The method is only validated for unconditional generation. In class-conditional generation, if the nearest class differs from the desired target class, conflicts may arise, limiting applicability. Moreover, because the approach relies on classifier guidance, it is unlikely to extend to text-conditioned generation, where classes are not explicitly defined. Hallucination becomes even more problematic in more complex conditional scenarios, yet such cases are not explored.
    
- **Lack of metrics such as FID:**
    In ImageNet experiments, reduced recall relative to classifier guidance (CG) suggests limited class coverage, which the authors acknowledge as a limitation. For unconditional generation, it is crucial to verify whether all classes are sufficiently covered (assuming a balanced dataset). If class coverage deteriorates, this could undermine one of diffusion models’ strengths: broad mode coverage. It is unclear why FID is not reported, as it reflects both coverage and fidelity. Furthermore, if the method’s main advantage is improved mode-seeking, comparisons to GAN-like models that excel at this should be included.
    
- **Insufficient motivation behind selective sharpening:**
    Since the dynamic guidance pushes toward a specific class present in the training data (even if the class changes at each step), it is somewhat expected that this would reduce hallucination. However, the paper does not clearly explain why this method avoids interfering with interpolation unrelated to hallucination. While toy experiments show that latent dimensions unrelated to class are unaffected, a stronger theoretical or intuitive justification would be helpful.

### Questions
- If hallucination is also reflected in the classifier, the proposed method may not be able to address it. How do the authors view this issue?
- In Algorithm 1, dynamic guidance is applied between T1T_1T1 and T2T_2T2. These appear to be hyperparameters, but there is no explanation of how they are selected or any ablation.
- In L.726, there seems to be a typo. According to the description, the latent dimension for the position of the shape in the right figure should be 5, not 0.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a new sampling approach for diffusion models that they call dynamic guidance. This is introduced to alleviate the mode interpolation based hallucination issues identified in Aithal et. al. 2024. The key idea is that dynamic guidance adjusts classifier guidance during sampling rather than using a fixed condition. This is done by re-aligning the guidance condition to the most likely mode at each denoising step. The authors find that the new method reduces hallucinations close to 70%, and improves precision on Imagnet.

### Strengths
1. The score function sharpening approach is well founded both mathematically and emprically (as see in Figure 3)
2. The dynamic scoring approach achieves better fidelity for both DDPM and DDIM when steps are small.
3. The evaluation is diverse across a set of toy + hands + imagenet tasks.

### Weaknesses
## Diversity Discussion
1. My biggest concern with this work is that there is no explicit discussion on diversity (or model) collapse. This is almost central to the work. Diversity and Interpolation are fundamentally at odds, and an investigation into the reduction of interpolation is incomplete without measuring its impact on diversity.
2. Of note, the paper briefly reports precision and recall metrics on ImageNet (Table 3) and shows some qualitative examples. It can be seen that the recall drops (e.g., 0.61 → 0.52), and Appendix Figures A.15–A.16 reveal that Dynamic Guidance leads to over-representation of certain ImageNet classes.
3. This indicates that while hallucinations are reduced, the model’s generative support becomes narrower, favoring a subset of modes where the classifier is more confident.
4. For hallucination-mitigation research, measuring this trade-off is essential, as overly “safe” sampling may degrade the creative or exploratory capacity of generative models.


## Need for a Classifier
I am concerned if we will always be able to know the classes of hallucination that the classifier could use for hallucination-prone regions. I am not sure how this approach will scale beyond toy tasks. Could you throw more light on its expansion to general purpose models?

### Questions
Can you include explicit diversity metrics such as LPIPS variance, embedding-space entropy, coverage/recall FID curves to quantify how Dynamic Guidance affects the distributional breadth of outputs.

### Soundness
2

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
4

### Summary
This paper aims to reduce hallucinations in diffusion generations. The authors identify the root cause as excessive smoothing of the learned score function between distinct modes of the data distribution, which leads to mode interpolation and unrealistic generations. To mitigate this, the authors propose Dynamic Guidance that continually re-estimates the most likely class based on the current noisy sample and incurs a classifier-based guidance for that class.
The paper evaluates the method on several settings: toy 2D Gaussian mixtures, synthetic shape datasets, human hand datasets, and large-scale ImageNet. Results show consistent hallucination reduction across different datasets.

### Strengths
1. The paper is very clearly-written and easy to follow. The dynamic guidance is an easy-to-implement and elegant technique in practice. The paper presented good visualizations (e.g. Figure 1 & 3) to exhibit the sharpening effect of dynamic guidance.

2. The paper studies an important topic for diffusion models and presents strong empirical evidence for the effectiveness of the guidance.

3. The paper presented certain limitations which is also the reviewer's concern.

### Weaknesses
1. The method of the paper is limited to solve hallucinating in certain realistic scenarios when there is too much classes (e.g. combinatorial classes https://openreview.net/forum?id=SKW10XJlAI) or continuous gestures of the hands that may not fall into discrete classes. The classifier-based methods are also training-expensive in the sense that we need a well-trained classifier at every noise level for good model performance. These limit the applicability of the technique in practice.

2. In the experiments, the paper exhibits that while classifier-based guidance is ineffective to reduce hallucination in some cases, dynamic guidance is effective. Due to the similarity between the two types of guidance, the paper should delve deeper into such a distinction, which is its key novelty and contribution. However not much observations are made in the paper like how "dynamic" is the classifying process and why dynamicity is crucial to reduce hallucination.

3. The paper does not contrast its technique with many existing baselines (e.g. https://arxiv.org/pdf/2402.08680) and sampling techniques. It also does not account for solving the limitations using methods like classifier calibrations.

### Questions
1. Why is the guidance only applied in an interval $[T_1,T_2]$ in the generation process in stead of the whole process? How are $T_1, T_2$ selected?

2. Why cannot classifier guidance eliminate mode interpolations?

### Soundness
4

### Presentation
4

### Contribution
3
