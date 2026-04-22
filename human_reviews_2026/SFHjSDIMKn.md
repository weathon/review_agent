# Seeing It Before It Happens: In-Generation NSFW Detection for Diffusion-Based Text-to-Image Models

- Avg Score: 3.60
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 2, 4

## Abstract
Diffusion-based text-to-image (T2I) models enable high-quality image generation but also pose significant risks of misuse, particularly in producing not-safe-for-work (NSFW) content. While prior detection methods have focused on filtering prompts before generation or moderating images afterward, the in-generation phase of diffusion models remains largely unexplored for NSFW detection. In this paper, we introduce In-Generation Detection (IGD), a simple yet effective approach that leverages the predicted noise during the diffusion process as an internal signal to identify NSFW content. This approach is motivated by preliminary findings suggesting that the predicted noise may capture semantic cues that differentiate NSFW from benign prompts, even when the prompts are adversarially crafted. Experiments conducted on seven NSFW categories show that IGD achieves an average detection accuracy of 91.32% over naive and adversarial NSFW prompts, outperforming seven baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes In-Generation Detection (IGD), a method that detects NSFW intent during diffusion-based image generation by analyzing predicted noise, enabling faster and more robust moderation than pre- or post-detection approaches.

### Strengths
1. The paper is well-organized with clear logic, 

2. Its method effectively detects NSFW content early during the generation process.

### Weaknesses
1. Can we understand it this way that they are indirectly classifying the prompts? Since the distribution of prompts themselves is different, classification on the noise are indirectly classifiying the prompts. Why not directly classify the text encoder output? How is this different from pre-detection? If we train a text classification model using i2p prompts with text encoder as input, I think it would still be useful. It means the ability and accuracy might come from the different of prompt instead of the genration. Please privde experiments to show your method is better than this.

2. How do they ensure that all i2p prompts can actually generate harmful content? Since i2p also sometimes generate benign images and sometimes it generate harmful images, wouldn’t that affect the training of the classification model?

3. The method is too simple and lacks novelty.

4. Beyond detection, concept removal can also generate other benign parts of the prompt after removing the harmful content. If we use concept removal as a baseline, it’s unfair, because it also has to handle this additional aspect.

### Questions
See weakness

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
In this work authors show that intermediate predictions from the diffusion denoiser can be used to detect the generation of NSFW content with text-to-image diffusion models. Based on that observation, a method is proposed to classify and block such generation.

### Strengths
- The topic evaluated in this work is extremely important given the plethora of recent publicly available text-to-image models

### Weaknesses
My main weakness concentrate on the novelty of the proposed approach which in my opinion is extremely limited.
“While existing methods primarily focus on pre-detection (prompt filtering) and post-detection (image moderation), the possibility of detecting NSFW content during the image generation process itself has, to our knowledge, been largely overlooked.” - This is simply not true. There is the whole branch of works exploring similar idea mostly with steering vectors. See for example:
- Gaintseva, Tatiana, et al. "Casteer: Steering diffusion models for controllable generation." arXiv preprint arXiv:2503.09630 (2025).
- Zhang, Hongxiang, Yifeng He, and Hao Chen. "Steerdiff: Steering towards safe text-to-image diffusion models." arXiv preprint arXiv:2410.02710 (2024).

Recent works also employ SAEs for the same task- see:
- Kim et al. Concept steerers: Leveraging k-sparse autoencoders for controllable generations
- Cywinski et al. SAeUron: Interpretable Concept Unlearning in Diffusion Models with
Sparse Autoencoders
- Cassano, Enrico, et al. "SAEmnesia: Erasing Concepts in Diffusion Models with Sparse Autoencoders."

The idea presented in this work (filtering NSFW content during generation) is also extremaly similar to the idea of unlearning. While this is acknowledged by the authors in section 5.3 it is unclear for me why the proposed approach is only compared against pioneering works in this field instead of the newest state-of-the-art solutions. (See for example  Zhang, Yihua, et al. "Unlearncanvas: Stylized image dataset for enhanced machine unlearning evaluation in diffusion models." arXiv preprint arXiv:2402.11846 (2024). - as a good benchmark evaluating more recent approaches

### Questions
Do I understand correctly that the input to the classifier is the “noise” predicted by the diffusion model? Why isn’t it just the recent state of the generation? Similarly to what was proposed in Universal guidance: Bansal, Arpit, et al. "Universal guidance for diffusion models." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.
“and apply five state-of-the-art attack methods to automatically generate adversarial prompts” - Can the authors discuss this in more details? What were those state-of-the-art adversarial methods?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents In-Generation Detection (IGD), an approach for identifying NSFW (not-safe-for-work) intent during the denoising process of diffusion-based text-to-image (T2I) models. Rather than filtering solely on the input prompt or the full synthesized image, IGD leverages the evolving predicted noise within the generation loop as a semantically rich internal signal and attaches a lightweight classifier to this signal. Through extensive experiments over seven NSFW categories and several adversarial prompting strategies, IGD achieves notably robust detection accuracy, outperforming existing pre- and post-generation baselines.

### Strengths
- The authors bring forward the notion of in-generation NSFW detection in diffusion models—monitoring predicted noise at intermediate denoising steps—whereas prior works focuses almost exclusively on pre-prompt and post-image detection.
- Experimental results (see Table 1, Table 2, Table 3) demonstrate strong robustness and high accuracy (92.45% mean on naive and adversarial prompts) across multiple challenging NSFW categories, substantially outperforming seven recent baseline systems.
- The classifier is light (5-layer MLP), incurs negligible runtime overhead, and is straightforward to implement—details well specified.

### Weaknesses
- The motivation for using predicted noise is supported mainly via qualitative t-SNE analyses (Figure 2 and related Appendix figures), but these visualizations are limited to a handful of classes. There is minimal theoretical discussion on why predicted noise at early timesteps reliably encodes semantic intent for all prompt regimes, especially as the denoising process is stochastic and intermediate signals could, at times, be altered by prompt perturbations.
- Table 11 explores layer count, but the impact of, for example, alternative architectures (CNNs on reshaped noise, regularization, larger-scale pretraining, or alternative loss functions) is not explored.
- While results with different Stable Diffusion versions are shown (Tables 5 and 6), the work stops short of evaluating on non-LDM architectures (e.g., DALL·E, Imagen) or with models with very different text embedding spaces, which calls into question universality.

### Questions
- Regarding miss-classifications observed in Figure 10 and in the confusion among certain NSFW sub-categories: What are the typical error cases (e.g., artistic nudes, ambiguous “shocking” prompts)? Could the authors qualitatively describe where IGD fails, and whether those failure cases are more severe than those of the best baselines?
- What impact would incorporating adversarial or paraphrased prompts into the training set (as opposed to only naive NSFW prompts) have on classifier robustness? Are there tradeoffs (e.g., overfitting, generality, or leakage)?
- Is the method’s internal signal available for non-open-source, closed API diffusion models, or is full access to the U-Net/denoiser module required? How practical is deployment in black-box commercial settings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces IGD, a method for detecting NSFW content in diffusion-based text-to-image models by analyzing predicted noise during the denoising process. Unlike existing pre-detection (prompt filtering) and post-detection (image moderation) approaches, IGD monitors intermediate representations and trains a lightweight MLP classifier to identify NSFW intent before image synthesis completes.

### Strengths
- NSFW content generation in T2I models is a legitimate safety concern that warrants research attention.
- The paper evaluates against multiple adversarial attack methods, which is important for assessing robustness.

### Weaknesses
- The section 4.2 is titled "In-Generation Detection Method" but provides almost no concrete methodological details, it mostly repeats motivation from section 3. The statement "we train a lightweight binary classifier" is insufficient. 
- Figure 2 shows several pairwise t-SNE comparisons but conspicuously omits the most important comparison: SFW vs. naive NSFW vs. adversarial NSFW all together. The paper claims adversarial prompts produce similar noise patterns to naive NSFW prompts, but Figure 3 only shows naive vs. adversarial NSFW without including SFW as a reference. This incomplete analysis raises concerns about whether the full picture would support the claims.
- Table 2 reports results across seven NSFW categories (sexual, violence, self-harm, harassment, hate, shocking, and illegal activity), yet the paper states that NudeNet, which only classifies nudity, was used for evaluation. This presents a fundamental inconsistency: how can one evaluate violence, harassment, hate, shocking content, and illegal activity using a nudity-only classifier? The paper provides no explanation. Similarly, for concept-erasing baselines, ESD only offers nudity checkpoints, and even the violence checkpoint is available only in third-party implementations. Yet the paper reports results for all seven categories. How were these results obtained?
- Critical details are missing or unclear:
    - Do you perform classification at every timestep (25 or 50 times per generation) or only once? This dramatically affects the interpretation of results and computational cost claims.
    - If classification is at a single timestep, is a separate classifier trained for each timestep? The paper is silent on this fundamental design choice.
    - How do you handle the I2P dataset preprocessing? Not all I2P prompts successfully jailbreak all target models. Did you verify attack success rates? If not, the evaluation may be comparing methods on different effective datasets.
    - For adversarial prompts generated by attack methods, did you verify they actually succeed in generating NSFW content? If unsuccessful attacks are included, the evaluation is meaningless.

### Questions
- The paper uses SD v1.5 for all experiments, but some concept-erasing methods and attack methods were originally designed for SD v1.4. Using mismatched versions when loading pre-trained safety components could invalidate the comparisons with concept-erasing baselines.
- Do the t-SNE separation patterns hold for concept-erasing models (ESD/SLD) as the backbone, or only for vanilla SD? Can your method complement with current defending method?
- For your evaluation, do you use the NudeNet classifier or the NudeNet detector? These are different tools with different characteristics and outputs. The NudeNet classifier is prone to misclassification. Have you experimented on the classifier's false positive rate?
- For most of the experiment results in the paper, you utilize SD v1.5 as the backbone model. However, to my knowledge, some concept-erasing methods and attacking methods initially use SD v1.4 in their official implementation. When you are comparing the results, especially for concept-erasing methods that need to load safety components, if you change the backbone from v1.4 to v1.5, will this affect the results? For example, if ESD was trained on SD v1.4, can you directly apply it to SD v1.5 and expect valid results?
- Some concepts are formed in earlier denoising steps and others form later. Based on the results you provided in Table 4, can you conclude that harmful concepts form in early timesteps? The accuracy is highest at t=5 (90.96%) and generally decreases for middle timesteps before recovering at later timesteps. Do different harmful categories (e.g., nudity and violence) show different temporal formation patterns? Maybe nudity concepts form early and violence concepts form later, or vice versa. Did you analyze Table 4 results separately by category? If the target model is a concept-erasing model (ESD or SLD) rather than vanilla SD, will the experimental results show different trends? Do erased concepts show altered temporal formation patterns compared to the base model?
- This paper proposes a lightweight NSFW classifier that could complement other safety measures like concept-erasing methods. The initial motivation studies the predicted noise of SFW prompts, naive NSFW prompts, and adversarial NSFW prompts in SD v1.5. Does the same observation hold in concept-erasing methods as well? Specifically, do you still see clear t-SNE separation between SFW, naive NSFW, and adversarial NSFW when using ESD or SLD as the backbone instead of vanilla SD v1.5?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces In-Generation Detection (IGD) for identifying NSFW content during the diffusion process itself.
To achieve this, the authors train a classifier on the noise predictions of a pre-trained diffusion model, distinguishing between NSFW and safe-for-training (SFT) samples.
This approach bridges the trade-off between speed (typical of pre-generation filtering) and accuracy (typical of post-generation detection).
Experimental results demonstrate that IGD significantly outperforms prior pre-generation methods, achieving stronger robustness against both naïve and adversarial NSFW content.

### Strengths
1. Clarity - The paper is well written and clearly structured.

2. Conceptual Simplicity - The method is conceptually simple yet effective.

3. Empirical Advantage - It substantially outperforms prior pre-generation NSFW detection methods.

4. Efficiency: Compared to post-generation detection approaches, IGD achieves conceptually faster detection though not shown.

### Weaknesses
1. Limited Novelty:
The research contribution is somewhat limited in scope. While the proposed approach offers potential speed advantages by detecting NSFW content. Classifiers on noisy intermediate representations are not new. 
a. Given that the approach involves training a classifier on predicted noise, it would be interesting to explore whether this classifier could also be used as a guidance signal to steer generation away from NSFW regions, potentially broadening the impact of the method.

2. Incomplete Comparison with Post-Generation Methods:
The paper does not include a direct comparison with post-generation NSFW detection techniques, leaving open questions about actual efficiency and performance trade-offs:
a. What is the measured gain in inference time relative to standard post-generation detection?
b. If one were to perform partial denoising (e.g., generating a pseudo-image with only 3–5 denoising steps), could post-generation detection achieve comparable accuracy and speed to the proposed in-generation method?

### Questions
See previous section 1a. 2a. 2b.

### Soundness
3

### Presentation
3

### Contribution
2
