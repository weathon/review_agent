# Unlocking Noise-Resistant Vision: Key Architectural Secrets for Robust Models

- Decision: Reject
- Scores: 8, 2, 2, 4

## Abstract
While the robustness of vision models is often measured, their dependence on specific architectural design choices is rarely dissected. We investigate why certain vision architectures are inherently more robust to additive Gaussian noise and convert these empirical insights into simple, actionable design rules. Specifically, we performed extensive evaluations on 1,174 pretrained vision models, empirically identifying four consistent design patterns for improved robustness against Gaussian noise: larger stem kernels, smaller input resolutions, average pooling, and supervised vision transformers (ViTs) rather than CLIP ViTs, which yield up to 506 rank improvements and 21.6\%p accuracy gains. We then develop a theoretical analysis that explains these findings, converting observed correlations into causal mechanisms. First, we prove that low-pass stem kernels attenuate noise with a gain that decreases quadratically with kernel size and that anti-aliased downsampling reduces noise energy roughly in proportion to the square of the downsampling factor. Second, we demonstrate that average pooling is unbiased and suppresses noise in proportion to the pooling window area, whereas max pooling incurs a positive bias that grows slowly with window size and yields a relatively higher mean-squared error and greater worst-case sensitivity. Third, we reveal and explain the vulnerability of CLIP ViTs via a pixel-space Lipschitz bound: The smaller normalization standard deviations used in CLIP preprocessing amplify worst-case sensitivity by up to 1.91 times relative to the Inception-style preprocessing common in supervised ViTs. Our results collectively disentangle robustness into interpretable modules, provide a theory that explains the observed trends, and build practical, plug-and-play guidelines for designing vision models more robust against Gaussian noise.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper investigates why certain vision architectures are naturally more robust to additive Gaussian noise and translates these findings into simple, theoretically grounded design principles. The authors evaluate over 1,100 pretrained models and identify four recurring factors that consistently improve robustness: larger stem kernels, smaller input resolutions, average pooling instead of max pooling, and supervised ViTs rather than CLIP ViTs. They complement these large-scale empirical results with theoretical analyses showing that noise attenuation scales quadratically with kernel size and downsampling factor, that average pooling reduces variance while max pooling introduces bias, and that CLIP’s normalization amplifies sensitivity by up to 1.9× compared to standard preprocessing. Altogether, the paper connects architecture-level design choices to quantifiable noise robustness and provides actionable guidelines for building more stable vision models.

### Strengths
**Strengths**
- Evaluates over a thousand pretrained models, ensuring strong empirical backing.
- The paper, motivation and insights are clearly written.
- The practical guidelines are directly usable for robust model design and could influence model architecture choices in industry and academia.
- A nice overview that includes mathematical rigor and applied insight.

### Weaknesses
**Weaknesses:**
- Could you elaborate more in the differences between the different architecture of ResNet-{C,D,T,S}, maybe highlights the differences with a checkmark system in Table 2 to make the insight of Table 2 clearer because right now the takeaway by looking past the table is not clear. 
- The name in the Table 1 of the models is not very easy to parse, if you could make it clearer having column for each component, if would be much easier to compare. 
- The study focuses solely on Gaussian noise, which, while analytically convenient, may not capture real-world corruptions (e.g., blur, brightness, compression). Extending to more diverse perturbations would strengthen generality. Or at least add a limitation section mentioning that these findings might affect other perturbations in different ways. Building on that, I then find the title borderline misleading. The findings show some architectural changes to make the model robust to gaussian noise, not robust in general. Same for you conclusion:  "[...] consistently improved * robustness". l. 413 -> Replace: * = "**gaussian noise**" would be less misleading.
So, please tone down some claim.

### Questions
**Questions:**

- I am slightly bothered by the fact the the ICLR template format is not the same as the original template, any reason for that ? Line spacing seem different than the other papers. 
- Please add a limitation section mentioning the limitations. These architecture changes might work for this specific case, but they are not solution for all corruptions/OOD sources. 


Addressing these points (**Questions** and **Weaknesses**) would clarify the analysis, make the results more interpretable, and strengthen the positioning of the paper within the broader robustness literature.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors perform an empirical study on the effect of architectural design on the model robustness against Gaussian noise. The authors evaluate a large number of vision models from timm library and identify four design choices for improved robustness: larger stem kernels, smaller input resolutions, average pooling, and supervised ViTs. The authors also provide the theoretical analysis for each choice.

### Strengths
-	The paper is generally well-written and easy to follow.
-	The experiments are comprehensive, covering a large number of models.

### Weaknesses
-	The authors mainly focus on Gaussian noise, which makes the scope of the paper somewhat limited. There are many corruptions beyond Gaussian noise. For example, 15 corruptions in [1] are commonly used to evaluate the model robustness. Besides, there are many other robustness benchmarks such as ImageNet-Adversarial, ImageNet-Rendition, ImageNet-Sketch, etc. Broadening the current scope will add more insights to the paper.
-	Since the authors focus on architectures, the mode size is also one important dimension that should be considered, where the current analysis is missing.
-	Apart from ViT and ResNet, there are also many other architectures such as ConvNeXt, Swin, VMamba, Diffusion Classifier, etc. These more recent architectures should be considered and studied as well.

**Reference:**

[1] Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. In ICLR, 2019.

### Questions
I am concerned about the questions mentioned above. Given the current status of the paper, I am leaning towards rejection and hope the authors could address my concerns during the rebuttal.

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
3

### Summary
This paper tries to study the robustness of different image classification models towards Gaussian noise perturbation. By comparing models' performance rank changes in a big leaderboard before and after adding Gaussian noise to the input images, this paper identifies some architectural designs related to the model robustness. Through analyzing the in-domain performance on different dataset, authors claim that ViTs require larger patch size, smaller input resolution and supervised CE loss to be robust to gaussian noise. For ResNet models, average pooling contributes more to the robustness compared to the max pooling and nearest neighbor pooling. For each conclusion, the paper also presents theoretical analysis as explanations.

### Strengths
1. This paper presents a way to analyze the model robustness by looking at the rank difference in the leaderboard before and after applying noise.
2. This paper provides comprehensive theoretical analysis for the observed phenomena.

### Weaknesses
1. The organization and the writing for this paper require improvement. 
2. There are some significant factual mistakes in the paper, i.e. the authors claim CLIP is a self-supervised method.
3. The conclusion in this paper lacks reasonable support and looks unconvincing. For example, the authors claim larger patch size and lower input resolution image can benefit robustness of ViTs by looking at the performance of existing ViT checkpoints. However, the paper doesn't study the relationship between robustness with patch size and resolution on ViTs. Instead, authors try to verify this conclusion on ResNet backbone. I believe there exists some logic gap between the conclusions and the evidence presented in the paper.
4. This paper only discuss the in-domain setting. It is well known models quite differently for in-domain data and out-of-domain data. Only analyzing model behaviors on in-domain data cannot reflect model's robustness comprehensively.

### Questions
Please see the weakness part.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies various vision models to determine architectural choices that are related to higher robustness gaussian noise. The paper finds that larger kernels in the stem network, average pooling and larger normalization standard deviations are related to enhanced robustness.

### Strengths
- Extensive experiments
- Mostly clear presentation
- Good theoretical analysis

### Weaknesses
- It is not surprising, and known, that larger kernel sizes lead to increased robustness to zero-mean gaussian noise. As the kernel grows larger the (weighted) sum of the noise component in the pixels/features approaches 0.
- Likewise, it is expected that increasing the image resolution, while keeping the kernel size constant, leads to reduced robustness because the number of elements in the downstream feature maps increases, and thus the noise energy in downstream feature map is larger.
    - Perhaps it would be interesting to plot the kernel size as a %age of the image size.
- It is portrayed that unsupervised CLiP is more robust than supervised CLiP, but it is revealed that the only factor is the normalization constant, not the nature of the training objective.
- some clarity issue:
    1. Add gaussian parameters in 3.1
    1. Clarify what the patch size in Section 3 refers to. Is it the longest dimension? Or is it, the total number of pixels in the patch?
    1. In the tables, instead of writing the model name, it would be better to create separate columns for each parameter of the model.
    1. Add details of ResNet-{C,D,T,S} in 4.1.

### Questions
If we measure the size of the kernel size as a percentage of the image size, do the observations regarding robustness hold?

### Soundness
3

### Presentation
3

### Contribution
2
