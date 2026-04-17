# How Diffusion Models Memorize

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Despite their success in image generation, diffusion models can memorize training data, raising serious privacy and copyright concerns. Although prior work has sought to characterize, detect, and mitigate memorization, the fundamental question of why and how it occurs remains unresolved. In this paper, we revisit the diffusion and denoising process and analyze latent space dynamics to address the question: "How do diffusion models memorize?" We show that memorization is driven by the overestimation of training samples during early denoising, which reduces diversity, collapses denoising trajectories, and accelerates convergence toward the memorized image. Specifically: (i) memorization cannot be explained by overfitting alone, as training loss is larger under memorization due to classifier-free guidance amplifying predictions and inducing overestimation; (ii) memorized prompts inject training images into noise predictions, forcing latent trajectories to converge and steering denoising toward their paired samples; and (iii) a decomposition of intermediate latents reveals how initial randomness is quickly suppressed and replaced by memorized content, with deviations from the theoretical denoising schedule correlating almost perfectly with memorization severity. Together, these results identify early overestimation as the central underlying mechanism of memorization in diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates why diffusion models memorize training images and identifies the early overestimation of the clean x when using classifier-free guidance. The authors show that classifier-free guidance amplifies the contribution of x, leading to memorization, and that this phenomenon arises from inference-time dynamics rather than training overfitting.

### Strengths
Provides a clear mechanistic understanding of memorization under classifier-free guidance, distinguishing it from traditional overfitting during training.

### Weaknesses
I think this paper mainly analyzes how classifier-free guidance amplifies memorization, rather than explaining how diffusion models memorize in the first place. Some experiments actually meet expectations. For instance, the diffusion model itself is not trained in the same way as the CFG inference — it learns to predict noise, either conditional or unconditional, not to generate the noise after the CFG amplification — so the fact that CFG increases loss is expected. The main issue, in my view, is that even after following the authors’ reasoning, I still don’t clearly understand how diffusion models memorize at all. The paper convincingly shows that conditional guidance can trigger memorization, but it stops short of explaining the fundamental memorization mechanism inherent to the diffusion process itself.

### Questions
See Weakness. Further:
   1. Have you examined whether unconditional diffusion models also memorize training samples？ In 3.3 you say B: Unconditional noise predictions contain no information about x, will it contain information about other training sample?
  2. What factors explain why some prompts trigger memorization while others do not ?

### Soundness
2

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
This paper studies the memorization of diffusion models. Specifically, the authors empirically show that classifier-free guidance (CFG) amplifies and strengthens memorization in diffusion models. Then, they attribute the effect of CFG to the overestimation of $x$ and show the correlation between the trajectory deviation and the memorization.

### Strengths
* The results and visualization are clear and convincing. They conduct a lot of experiments in Section 3.3 to validate their claims. They calculate the cosine similarity between noise predictions and the initial noise $x_T$ or the real image $x$, showing that the conditional noise predictor memorizes the training sample $x$, while the unconditional predictor does not. Then these findings are further used to formally explain the effect of CFG in Eqs. (11)-(13).

### Weaknesses
* My biggest concern is your claim that overfitting cannot explain memorization alone. For the CFG, if the conditional diffusion does not overfit, then CFG won’t amplify the effect. So, the memorization is still attributed to overfitting. In other words, the overfitting of conditional predictor is the necessary condition, and CFG just gives it a push.
* The preliminary is too long, introducing some basic concepts. Put some part in the appendix and only keep the most important part in your main paper, like the noise of the CFG process.
* In Section 3.2, you use the term "training loss". Are you using ||\hat_{x}_0, x_0|| as the “training loss”? To use a CFG method, we need to train conditional and unconditional diffusion models. Then, the training loss should just be the loss for training the conditional and unconditional models. But I don’t think it’s reasonable to plug in the CFG-inference x0 into the formula and define it as "training loss". CFG is applied during the inference time, and the training loss has no relation to the inference. If you want to claim that CFG will make the generation distribution even farther from the original distribution, this is definitely true and is common sense in the diffusion model field, since even the training process of CFG itself is not aimed to minimize ||\hat_{x}_0, x_0||.
* Some prior work has already shown similar insights. For example, [1] shows that CFG would cause memorization if we apply CFG in an early denoising stage. [2] also has similar conclusions. The insight behind this is somewhat intuitive: since the conditional diffusion memorizes the sample, stronger guidance would strengthen the memorization.
* The finding that memorization occurs at a very early stage of denoising is also discussed in some prior work [3]. 

[1]: Classifier-Free Guidance inside the Attraction Basin May Cause Memorization.
[2]: Feedback Guidance of Diffusion Models.
[3]: Exploring Local Memorization in Diffusion Models via Bright Ending Attention.

### Questions
Please refer to previous sections.

### Soundness
2

### Presentation
2

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
This paper aims to understand how diffusion models, especially for text-conditioned ones, memorise training data. Through the empirical evidence and theoretical derivation, the authors aim to attribute the memorisation to overestimation of memorised data during early denoising. And such an overestimation is driven by the text conditioned noise prediction. Afterwards, the authors demonstrate that such an overestimation leads to memorised images.

### Strengths
This paper attempts to deepen the understanding of memorisation of diffusion models, which is an important topic in the community. Through both empirical evidence and theoretical approximation, the authors showed that the text-conditioned noise prediction is the key to inject training image, and leads to memorisation.

### Weaknesses
I have the following concerns which require further clarifications from the authors' side. And please correct me if I am wrong. 

1. I am not very convinced by the relationship between memorisation and overfitting in this paper. As the training loss essentially should be $||\boldsymbol{\epsilon}-\epsilon\_{\theta}(\boldsymbol{x}_t, \boldsymbol{e}_C)||+\lambda ||\boldsymbol{\epsilon}-\epsilon\_{\theta}(\boldsymbol{x}_t)||$, with $\lambda$ could be a term representing the text prompts are randomly replaced with void during training. Although equation 3,5,6 could be used to transform it into the loss term 10, whether we can still have such a transformation under classifier-free guidance is not clear to me as now the noise prediction has two terms. Can we use classifier-free guidance to predict noise and have the denoising trajectory of $\boldsymbol{x}_t$ and then observe the loss in equation 6 but with text condition?

2. The authors' conclusions significantly rely on some assumptions of approximations, e.g., $||\hat{\boldsymbol{x}}_0^{(t)}-\boldsymbol{x}||\approx 0$, however, at least from Figure 1, this term is about $10^4$, why can we say it approximates 0. Additionally, $\hat{\boldsymbol{x}}_0^{(t)}\approx k\boldsymbol{x}$  is also a very strong assumption.

3. Normally $\boldsymbol{x}_T$ refers to the forward latents, and we assume it follows a gaussian distribution which is already has no information about the training data $\boldsymbol{x}$. During the generation, we will sample a new one based on gaussian distribution, it seems that the theoretical analysis still uses the same one $\boldsymbol{x}_T$ in the denoising process? How to justify this?

4. Given the intuitions in this paper, what are actionable suggestions to mitigate memorisation in diffusion models?

### Questions
See the weakness.

### Soundness
2

### Presentation
2

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
This paper tackles the memorization issue in diffusion models by providing a principled analysis of why and how diffusion models memorize their training data, which is a well-studied privacy concern in existing diffusion models.
Through theoretical derivations and empirical analyses, the authors show that memorization is driven by early overestimation of training samples during the denoising process, specifically when classifier-free guidance (CFG) amplifies the contribution of the memorized image. This overestimation reduces latent diversity, collapses denoising trajectories, and forces convergence to the memorized sample. 
The paper formalizes this process by decomposing intermediate latents and demonstrating near-perfect correlation between deviations from the theoretical denoising schedule and the severity of memorization. The results indicate that memorization is not merely due to overfitting, but to an intrinsic instability in guided denoising dynamics.

### Strengths
1. This paper contributes to the reasons for memorization. While prior works have primarily characterized memorization or proposed detection and mitigation strategies, this work focuses on the mechanistic cause underlying the phenomenon. It provides a causal explanation for memorization that unifies and explains previous empirical findings (e.g., from [1] and [2]). 
2. The decomposition analysis of intermediate latents, linking theoretical denoising trajectories to empirical memorization, is a novel methodological contribution.
3. The experiments are conducted systematically, that are across multiple diffusion models (Stable Diffusion v1.4, v2.1, and RealisticVision) and sampling schemes.
4. Overall, this work contributes to preserving the training data privacy of diffusion models.

[1] Detecting, explaining, and mitigating memorization in diffusion models. ICLR 2024.

[2] Classifier-free guidance inside the attraction basin may cause memorization. CVPR 2025.

### Weaknesses
1. Although the paper provides a precise mechanistic understanding, it stops short of proposing or testing new mitigation strategies derived from its findings (e.g., modified guidance schedules, adaptive early-denoising regularization). A concrete demonstration of how to exploit the discovered mechanism for reducing memorization would make the contribution even stronger, as most related work proposes the corresponding mitigation strategies based on their findings to further prove the insights.
2. Although the classifier-free guidance (CFG) being the key cause of memorization is convincing, the finding has already been well-studied in some early works in the field (e.g., [3] and [4]) that analyzed this to be one of the causes of memorization, and proposed corresponding mitigation strategies for diffusion memorization. It is suggested to elaborate on the difference in this finding to make the contribution clear and unique.
3. Some experiments involve large-scale sampling (e.g., 436 prompts × 50 images × multiple guidance scales). Runtime and resource requirements are not clearly reported, which are recommended to be included for better reproducibility.

[3] Understanding and Mitigating Copying in Diffusion Models. NeurIPS 2023.

[4] Towards Memorization-Free Diffusion Models. CVPR 2024.

### Questions
1. Do the authors suggest that similar overestimation phenomena appear in other modalities (audio, video, text-diffusion)?
2. How does the proposed “early overestimation” mechanism behave in unguided or LoRA-fine-tuned models, where classifier-free guidance is weak or absent?
3. Could the correlation between deviation and memorization severity be used as a real-time detection metric during generation?

### Soundness
3

### Presentation
2

### Contribution
3
