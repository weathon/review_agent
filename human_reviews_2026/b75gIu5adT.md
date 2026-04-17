# Warfare: Breaking the Watermark Protection of AI-Generated Content

- Decision: Reject
- Scores: 8, 2, 2

## Abstract
AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services using advanced generative models to create realistic images and fluent text. Regulating such content is crucial to prevent policy violations, such as unauthorized commercialization or unsafe content distribution.
Watermarking is a promising solution for content attribution and verification, and numerous watermarking approaches have been proposed recently. However, we demonstrate its vulnerability to two key attacks: (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. 
We propose Warfare, a unified attack framework leveraging a pre-trained diffusion model for content processing and a generative adversarial network for watermark manipulation. Evaluations across datasets and embedding setups show that Warfare can achieve high success rates while maintaining the quality of the generated content. We further introduce Warfare-Plus, which enhances efficiency without compromising effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
A unified watermark attack method is proposed, which consists of a pre-trained diffusion model and a generative adversarial network. Watermark removal and forgery are achieved jointly via using both watermarked data and its denoised version, through which the GAN is implemented. Thorough evaluations have been provided with clearly articulated loss functions.

### Strengths
The proposed idea is clearly conveyed and it is quite enjoyable to read the paper.

The proposed method does not require the unwatermarked counterpart of the watermarked images or the watermarking schemes, due to the use of a pretrained diffusion model to create a denoised copy.

The proposed design of using GAN to discern watermarked data and denoised data is quite novel, which leads to the good advantage that both post-hoc and prior watermarking methods can be attacked.

### Weaknesses
It is unknown how the denoised images replicate their original counterparts. On the one hand, this depends on the capability of the pre-trained model. On the other hand, this also depends on the underlying watermarking model -- different watermarking models may render different denoised results, leading to certain xhat being not similar to the original x.

### Questions
Now that the proposed method does not require any information about the watermarking schemes, do the watermarked images have to be AIGC? Would the proposed method be applicable to other watermarked images? The answer to the latter question seems to be yes, due to the definition of x, x' and xhat. Please clarify.

How effective is the adopted pre-trained diffusion model H? It would be helpful if the authors could clarify what kinds of watermarks can/cannot be removed by H.

It seems to me that watermark removal and forgery have an intrinsic tradeoff, which is analogous to the tradeoff between type I and II errors (or that between false positive and negative). Can the authors provide some insights on this aspect, for example, whether and how the proposed method may also experience such a tradeoff?

I understand that visible watermarks are not considered because the denoised version would not be similar to the original version. However, I'm not sure why only the steganographic approach is considered within the invisible category. It might be helpful if the authors could explain what aspects of steganographic approaches are taken advantage of when designing the attack.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Warfare, an image watermark removal and forging attack. Warfare utilizes a public diffusion model (DM) to generate unwatermarked images, and subsequently trains GANs to learn mappings between watermarked and unwatermarked images in both directions.

### Strengths
1. The proposed method is clearly described, and the paper is well written and easy to follow.

2. The black-box attack setting is realistic and relevant to practical watermarking scenarios.

### Weaknesses
1. The goal of watermark removal appears largely accomplished in the first step, using the pre-trained diffusion model to generate unwatermarked images, a method already explored in prior work. The additional GAN mapping step seems redundant. The authors are encouraged to more clearly articulate the novelty and benefits of their approach relative to established methods such as DiffPure. In this light, statements like “the first work on…” feel somewhat overstated. 

2. Training a GAN typically requires thousands of diverse samples; otherwise, the model risks overfitting or mode collapse. It seems unrealistic that a GAN can successfully learn watermark patterns from as few as ten training examples. The authors should provide full details on the training of GANs (complete architecture, initialization, etc.) and convincingly explain the reason of success under such data scarcity. I also invite other reviewers and the AC to share their perspectives on this point. 

3. Following the above concerns, if the proposed method indeed requires a large number of training samples, its practicality and advantage would be diminished. In contrast, DiffPure can perform watermark removal on a single image without additional knowledge.

### Questions
Please see Weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Warfare and Warfare-Plus, as a unified framework for attacking watermarks in AI-Generated Content. The proposed method operates under a black-box threat model. The core idea involves three steps: (1) collecting watermarked images, (2) using a pre-trained diffusion model or unconditional sampling to generate non-watermarked mediator images, and (3) training a GAN to translate between the watermarked and mediator image distributions. This allows the framework to perform both watermark removal and watermark forgery attack.

### Strengths
1. The conceptualization of a single framework that can perform both watermark removal and forgery by reversing the mapping in a GAN is an elegant idea.
2. The paper proposes Warfare-Plus as a more time-efficient alternative to Warfare, correctly identifying data pre-processing as a potential bottleneck. This focus on practical efficiency is a positive aspect.

### Weaknesses
1. The paper's literature review is severely lacking. For a submission to ICLR 2026, it is surprising that there are no references to work published in 2025, especially in a fast-moving field like generative AI security. The authors appear to be unaware of the recent progress in both watermark removal and forgery, which leads them to make unsupported claims about their work's novelty.
2. The central claim that this is the first work to forge watermark is demonstrably false. Several prior works have explored black-box watermark forgery.
3. The experimental evaluation is not convincing because it omits comparisons with many recent and highly relevant black-box attack methods, such as CtrlRegen. The authors compare against simple image transformations and a few selected baselines but fail to benchmark against the true state-of-the-art in watermark removal and forgery. This omission makes it impossible to assess whether Warfare offers any meaningful improvement in performance, efficiency, or applicability over existing techniques. A thorough experimental comparison against recent literature is essential for a paper in this field.
4. The methodology rests on a critical but unsubstantiated claim. The authors state, "The mediator dataset $\hat{\mathcal{X}}$ can be seen as being drawn from the same 'non-watermarked' distribution as $\mathcal{X}$". However, the paper provides no theoretical analysis to support this.

### Questions
1. The reference list for this ICLR 2026 submission appears to stop in early 2024. Can you confirm if a thorough literature search was conducted for relevant work published in 2024 and 2025? If so, why were more recent state-of-the-art attack methods not included as baselines?
2. On what theoretical basis do you claim that the mediator dataset $\hat{\mathcal{X}}$ (generated by adding large noise and denoising) is drawn from the same distribution as the original clean dataset $\mathcal{X}$? Given that your own text states the resulting images are visually different, how does this assumption hold, and how does a potential distributional shift affect the validity of your framework?

### Soundness
1

### Presentation
2

### Contribution
1
