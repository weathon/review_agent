# Dormant Memories Undermine Safety: Initial Latent Variable Optimization for Attacking Unlearned Diffusion

- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Although diffusion models (DMs) have advanced image synthesis, they pose risks of generating Not-Safe-For-Work (NSFW) content. Recent unlearning-based defenses contend that they can eliminate NSFW concepts, and show promise in defending traditional attacks. However, we analyze unlearned models from a new perspective and reveal a key insight: unlearning does not really erase unsafe concepts, but only disrupts the mapping between linguistic symbol and corresponding knowledge. The knowledge itself remains intact, preserved as **dormant memories**. We further show that the distributional discrepancy in the denoising process serves as a measurable indicator of how much of the mapping is retained, reflecting the strength of unlearning. Inspired by this, we propose **IVO** (**I**nitial Latent **V**ariable **O**ptimization), a concise yet powerful attack framework that reactivates these dormant memories by reconstructing the broken mappings.  IVO uses optimized initial latent variables as triggers align the noise distribution of unlearned models with that of standard DMs while steering it toward NSFW content. It operates in three simple stages: *Image Inversion*, *Adversarial Optimization*, and *Reused Attack*. Extensive experiments across 6 widely used unlearning techniques demonstrate that IVO achieves the highest attack success rates while maintaining strong semantic consistency, indicating that dormant memories remain exploitable and exposing fundamental flaws in current defenses. The code is available at anonymous.4open.science/r/IVO/. **Warning**: This paper has unsafe images that may offend some readers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Initial Latent Variable Optimization (IVO), a novel method to assess the success of concept unlearning methods in text-to-image diffusion models. Instead of optimizing the text prompt or its latent representation (which related adversarial evaluations do), IVO optimizes the initial latent noise vector. The paper hypothesizes that existing unlearning methods only disrupt the mapping between natural language and unsafe content instead of actually removing the content from the model's generation capabilities. Based on this assumtpion, IVO first applies DDIM inversion to get a good starting point for the optimization, followed by an iterative, gradient-based refinement of the latent. The method is evaluated against multiple unlearning procedures and compared to other adversarial evaluation strategies on two prompt datasets. An ablation study concludes the experiments.

### Strengths
- Optimizing the initial noise vector instead of the text prompt or embedding to analyze the limitations of unlearning techniques is an interesting and promising research direction.
- The experimental results indicate a clear success of IVO, which outperforms related prompt-based optimization strategies. The number of investigated unlearning techniques and adversarial evaluation methods is sufficiently large.
- An ablation study analyzes the contributions of the individual components of IVO and demonstrates that each one contributes to the method's effectiveness.

### Weaknesses
The major weakness of the paper is that many parts are imprecise and lack sufficient detail.

- Below are specific examples from Section 4:
  - Is the loss function simply an MSE loss between the computed noise predictions?
  - Additionally, where does the additional surrogate model originate from, and what assumptions does the method make regarding it (e.g., same architecture, noise dimension, or capabilities)?
  - What is meant by "Simultaneously, P and z also input into another separate prediction process" (L300)? Which prediction process is being referred to here?
  - ... 

- The experimental protocol in Section 5 also lacks crucial details:
  - What is the underlying diffusion model used? 
  - Are all different unlearning methods applied to the same base model? 
  - How is the attack success rate defined (i.e., is an attack successful if only one nudity detector reports a match, or must all detectors classify the image as unsafe)? 
  - Over how many samples is the FID computed, and what is the reference dataset? What is the baseline FID score? 
  - How are image-to-image attacks exactly conducted?
  - ...

Minor weaknesses:
- Section 3 focuses on Stable Diffusion/LDMs. While these are the largest group of models, more recent approaches, such as flow models, should be mentioned. Furthermore, the paper does not discuss whether the method is, in principle, applicable to other Diffusion Model (DM) approaches like flow models.
- Some abbreviations, such as MMD, are not introduced in the main text (only in a table caption).

Suggestions:
- Write $\mathcal{L}_\mathit{DML}$ instead of $\mathcal{L}_{DML}$ for a nicer look of the symbol.
- Consider using KID (Kernel Inception Distance) instead of FID since it is more reliable on smaller datasets.

### Questions
- Since the method relies on two diffusion models (including the surrogate model), what is the memory consumption of the optimization?
- During which generation steps is the loss function computed, only during the first one?
- What are the numbers in Table 2 behind the method names?
- When using DDIM inversion to get the initial latents, how close are the final images after the optimization step compared to the images used during DDIM inversion? I could imagine that the final results are close to the initial images.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposed a new latent space attack framework for unlearned diffusion model, claiming the unlearning is because of dormant memory instead of really erase unsafe concepts. Based on this observation, the author proposed the IVO attack framework that can reactivate these dormant memories by reconstructing the broken mappings. The three step approach namely image inversion, adversarial optimization and reused attack, shows much improved attack successful rate (ASR) over other widely used unlearning techniques. In addition, the IVO framework is orthogonal so that it can be used standalone or integrate with other attack methods.

### Strengths
1. The concept of "dormant memories" is a good intuition. The Maximum Mean Discrepancy to quantify the distributional discrepancy between original and unlearned models is a clear and simple way to measure the strength of unlearning.
2. The method shows both text-to-image and image-to-image generation scenarios with good versatility. It also shows the modularity meaning it can be integrated with other attack method with better together.
3. expand the attack interface from prompt to latent space, and shows better results.

### Weaknesses
1. Even though optimization steps are greatly reduced by reduced attack method, the proposed framework still involves standard DM to generate DDIM inverted NSFW image embeddings, as well as surrogate model used in adversarial optimization, this indicates the further improvement on runtime performance optimization.
2. section 5.4 listed a few hand-pick examples on semantic consistency but without define a metrics.
3. section 5.5 ablation study in table 4, is safety an attack type or not? why sexy + violence ASR is higher than the matched violence + violence? Is there a ASR vs Opt step curve - how if you increase the Opt budget will it have higher ASR in the table?
4. table 4: assuming adversarial prompt in other literature is a further enhanced version of unsafe prompt, why the ASR and FID is worse than unsafe prompt case in your experiment?
5. equation (3) overall loss term is not clearly explained with details. It might be better to add additional hyper-parameter to balance the two loss terms.
6. latent space attack method has the disadvantage in real world application since the inversion from embedding back to prompt are not guaranteed to exist. It will be less practicable and realistic in most circumstances.

### Questions
1. The optimization steps reported in table 3 & 4, how much is contributed by reused attack?
2. FID score is good but have you thought about better metrics like CLIP score?

### Soundness
3

### Presentation
2

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
This paper proposed a novel attack framework which is motivated by incomplete concept erasure under existing unlearning methods for diffusion models and they denote the left knowledge intact as dormant memories. To be more specific, the proposed attack optimize the initial latent in the image latent space instead of discrete text space to reactivate the dormant memories.

### Strengths
1. The three-stage is simple but effective: Image inversion, adversarial optimization, and reused attack. 
2. Strong and consistent empirical results over different datasets and attack baselines.
3. Good ablation study on loss impact, prompt influence, and modularization in complex cases.

### Weaknesses
1. While the "dormant memory" perspective is new, however, the underlying technique (latent optimization and distribution alignment) resembles previous latent inversion and adversarial reparameterization methods.
2. Compared with the attack baselines, the proposed attack bring large improvements on ASR, however, the main improvements might come from the continuous (embedding) optimization space, while the chosen attack baselines optimize the adversarial prompts in the discrete text prompt space.
3. The proposed method is only evaluated on NSFW concepts. It is better to consider more different concepts such as style and objects.

### Questions
The proposed method is to optimize adversarial embedding in the continuous optimization space which is similar to the textual inversion but the goal of textual inversion is to obtain the optimized embedding which can enforce the model to generated some unseen customized concepts. In other words, there is an assumption that even if the model never see any NSFW images during training, continuous adversarial attacks are still able to enforce the model to utilize common knowledge to generate image containing NSFW concepts.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This study proposes a new jailbreak attack against concept erasure in diffusion models (DMs). It optimizes the initial latent variable, hence the name Initial Latent Variable Optimization (IVO), which differs from the optimization of the input prompt, prompt embedding, or textual inversion like other attacks do. This work is overall well-executed but lacks one or two important baselines on the erasure side (STEREO, RECE, Receler, RACE), and has one significant weak spot, which is left unaddressed. This weak spot is the assumption of having access to a harmful surrogate model, which is used as a teacher to guide the optimization of the initial latent to produce harmful content with the unlearned DM. This is a crucial assumption, which is neither elaborated further, nor is the exact choice of this harmful teacher model explained. If it turns out that this surrogate teacher is the base model (before unlearning), then there is no real practical applicability of the IVO attack. Its attack requires white-box access to the initial latent, but also access to a surrogate model, which then raises the question of why not use the surrogate model directly to generate harmful content. The good results of the IVO attack are thus also not a fair comparison to the related attacks, which are simply making fewer assumptions, such as only having access to a set of harmful images in the pixel-space, rather than a surrogate model to get score-based guidance in the latent space. Unfortunately, this study is limited to the NSFW scenario and appears to be outdated in light of existing works that demonstrate most erasure approaches fail to truly erase by optimizing inputs to the model.

### Strengths
- (S1) **Novel focus on the image latent for a restoration attack**, i.e. finding the most effective starting point in the latent space for the iterative denoising to rediscover previously erased concepts, instead of finding the most effective text prompt or embedding that independently of the starting point reliably leads to the erased content.
- (S2) **Image-to-image scenario** Figure 7 (b), together with the section is an interesting take on circumventing moderated image-to-image models

### Weaknesses
I find the following list of things to be major weaknesses:
- (W1) **The Surrogate Model Assumption**: There is no information on which model is used as the surrogate and no ablations on that. When the surrogate model is the original model (before unlearning), then the practicality of the whole attack methodology is fundamentally flawed.
- (W2) **Missing robust erasure baselines**: As this work proposes a new type of attack, it should be tested against more, reliable erasure methods besides just AdvUnlearn, which in itself is a bit special as a usually text-encoder-focused approach. The inclusion of STEREO, and RACE or Receler are required to further substantiate the claims of this work.
- (W3) **Only NSFW**: This study is exclusively limited to NSFW. Other typical scenarios in this area are style unlearning, object erasure, or celebrity erasure. With only a single application scenario, the generalizability of the presented results are questionable, especially since different erasure scenarios tend to often behave quite differently.

And the following is a minor weakness:
- (W4) **Only SD v1**: This study is also limited to SD v1 with no direct results or transferability experiments to more recent models. This is the case for many works in this area, but by now most of them include at least a preliminary experiment demonstrating transferability or applicability to newer architectures like SD v3.

### Questions
- (Q1) Why are the FIDs beyond 100? Knowing that FID gets biased for smaller sampling sets, it is likely that the numbers also have higher variance in general. I would appreciate more info on that.
- (Q2) How diverse are the images generated from a single derived IVO latent? Is there a structural bias?

### Soundness
2

### Presentation
1

### Contribution
1
