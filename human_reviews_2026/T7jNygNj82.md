# Distribution-Aware Synergistic Evolution for Few-shot Discrimination and Generation

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Discrimination and generation are two distinct yet complementary paradigms in machine learning.
Generally, discriminative models are better at estimating the $\textit{class center}$, while generative models are better at modeling the $\textit{data variance}$.
To harness the strengths of both paradigms, we propose a synergistic evolution framework that allows discriminative and generative methodologies to cooperate in estimating feature distributions.
For one thing, the discriminative model incorporates synthetic samples from the generative model to improve the estimation of feature covariance, especially when the available data is limited.
For another, the generative model leverages calibrated class centers from the discriminative pathway as anchors to improve the semantic accuracy of the generated samples.
In summary, our framework enables the discriminative model and the generative model to jointly develop and collaborate within a few-shot learning scenario, thereby enhancing both of their individual capabilities.
Additionally, our design improves open-set learning by enhancing out-of-distribution detection through better covariance modeling in the discriminative space. 
Extensive experiments on the CUB-200 and miniImageNet datasets demonstrate performance gains in few-shot class-incremental learning (FSCIL), few-shot incremental generation (FSIG), and open-set recognition tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
1.  Uses generative samples to refine covariance estimation, improving few-shot class-incremental learning (FSCIL) and open-set recognition (OSR).

2. Leverages calibrated class prototypes from the discriminator to enhance semantic information in few-shot image generation

3. Enables mutual reinforcement between discrimination and generation, outperforming baselines on CUB200 and miniImageNet

### Strengths
1. Novel bidirectional interaction between discriminative and generative models

2. Introduces calibrated prototypes for generation and covariance refinement via synthetic samples

3. Rigorous experiments on FSCIL, OSR, and FSIG, with ablation studies validating each component

4. Unifies traditionally disjoint paradigms, enabling applications like continual learning with generative feedback

### Weaknesses
1. Experiments limited to Stable Diffusion 1.5 + LoRA. Larger diffusion models or non-CLIP backbones are untested.

2. Diagonal covariance ignores cross-feature correlations. A low-rank or sparse approximation could better capture structure.

3. No analysis of adversarial robustness such as distribution shifts.

### Questions
1. How does DASE perform with larger diffusion models or non-CLIP backbones?

2. Could structured (e.g., block-diagonal) covariance improve discrimination without overfitting?

3. How does OSR performance degrade under domain shifts (e.g., CUB200 → iNaturalist)?

4. Is there any theoretical support for the arguments presented in the abstract? "Generally, discriminative models are better at estimating the class center, while generative models are better at modeling the data variance."

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a unified framework called Distribution-Aware Synergistic Evolution (DASE) to integrate discriminative and generative learning for few-shot class-incremental learning (FSCIL), open-set recognition (OSR), and few-shot image generation (FSIG). It models each class as a Gaussian distribution in CLIP feature space and uses calibrated means and variances for both classification and generation. The approach consists of two phases: an initialization phase where visual prototypes are calibrated using base class statistics and text guidance, and a synergy phase where classifier and generator iteratively refine each other. Experiments on CUB-200 and miniImageNet demonstrate improvements across classification, out-of-distribution detection, and generative quality.

### Strengths
Presents a unified framework that leverages distribution-aware Gaussian modeling in CLIP feature space to support few-shot classification, open-set recognition, and image generation within a shared probabilistic structure.

### Weaknesses
1. The reliance on diagonal Gaussian assumptions, while practical for few-shot settings, may oversimplify class feature distributions.
2. The method depends heavily on heuristic filtering of synthetic data and text-guided prototype calibration, yet lacks a detailed ablation or sensitivity analysis to understand the robustness of these design choices.

### Questions
1. How sensitive is the overall performance to the quality of the synthetic samples selected during Phase II? Would the model degrade significantly if lower-quality generations were mistakenly included?
2. Have the authors considered or tested more expressive distribution models beyond diagonal Gaussians, such as full covariances, low-rank approximations, or mixture models? If so, what were the tradeoffs?
3. Could the text-guided calibration process introduce bias if semantic similarity does not align with visual similarity? How robust is the calibration step when text embeddings are noisy or ambiguous?
4. In the synergy phase, how often is feedback exchanged between the generator and classifier, and how does this frequency affect convergence and performance?
5. Does the method generalize well to non-fine-grained datasets or other domains (e.g., medical, synthetic imagery), or is it limited by assumptions baked into CLIP and Stable Diffusion’s pretraining?
6. Could the authors provide a more comprehensive ablation study that isolates the contributions of the synergy phase—specifically the generator-to-discriminator feedback via synthetic data and the discriminator-to-generator guidance via calibrated prototypes—across all three tasks (FSCIL, OSR, FSIG), as these components appear central to the claimed performance gains?

### Soundness
2

### Presentation
2

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
This paper proposes a unified framework called Distribution-Aware Synergistic Evolution (DASE) that integrates discriminative and generative paradigms for few-shot learning. The key idea is to enable mutual reinforcement between a CLIP-based discriminator and a  Diffusion-based generator. The discriminative pathway refines class mean and covariance estimation by incorporating generated samples, while the generative pathway uses calibrated class prototypes from the discriminator as semantic anchors to enhance the fidelity of synthesized images. The framework is applied to three tasks—few-shot class-incremental learning (FSCIL), open-set recognition (OSR), and few-shot image generation (FSIG)—demonstrating moderate improvements in accuracy, AUROC, and FID on CUB200 and miniImageNet. Overall, the paper aims to bridge the gap between discrimination and generation in few-shot scenarios through iterative distribution calibration.

### Strengths
1. The paper’s structure and writing are clear, making the method easy to follow. Figures and tables are properly formatted, and the overall presentation is neat.  
2. The research problem itself is meaningful and relevant. Exploring how discriminative and generative models can mutually benefit each other under few-shot conditions addresses a long-standing challenge in vision research, and the proposed bidirectional feedback idea has potential value for future work on hybrid learning frameworks.

### Weaknesses
1. The literature review is incomplete. Many recent works in few-shot image generation are not discussed, such as ADAM [1], RICK [2], and GenDA [3], as well as several key few-shot class-incremental learning methods from the past two years.
2. The experimental design is limited. Most comparisons are only comparing with the variants of the proposed method. For example, in Table 3, the method is evaluated on few-shot image generation but without comparing to any existing FSIG methods. Furthermore, well-known diffusion-based personalization methods like DreamBooth and Textual Inversion are mentioned in the related work but not included in experiments. This is important, since both CLIP and Stable Diffusion are pre-trained on massive paired datasets, and their inherent generalization ability could overshadow the claimed contribution of the proposed method.  
3. The analysis of the experimental results is not convincing. In Figure 2, the visual differences between the proposed method and the baseline are not significant; some images appear almost identical, and only a few qualitative examples are shown, making the analysis not statistically supported.  
4. There are minor typographical errors. For instance, the abstract ends with “few-shot incremental generation (FSIG),” but based on the paper’s context, it should be “few-shot image generation.”

[1] Few-shot Image Generation via Adaptation-Aware Kernel Modulation

[2] Exploring Incompatible Knowledge Transfer in Few-shot Image Generation

[3] FEW-SHOT CROSS-DOMAIN IMAGE GENERATION VIA INFERENCE-TIME LATENT-CODE LEARNING

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

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
This paper introduces the Distribution-Aware Synergistic Evolution (DASE), an approach designed to unify few-shot discrimination and generation tasks. The core idea is to establish a bidirectional, mutually beneficial relationship between a discriminative and a generative pathway. The framework operates in two phases: (I) The discriminative model calibrates the prototype for new few-shot classes, which then serves as a semantic anchor to guide the generative model in producing higher-fidelity images. (II) The discriminative model, in turn, leverages the synthetic samples from the generator to estimate a more robust feature covariance matrix, addressing the unreliability of covariance estimation from scarce data. Experiments demonstrate that this synergistic loop enhances performance across Few-Shot Class-Incremental Learning, Open-Set Recognition, and Few-Shot Image Generation on the CUB-200 and miniImageNet datasets.

### Strengths
The proposed framework is technically plausible and is supported by empirical validation.

### Weaknesses
1. The abstract makes a strong assertion that “discriminative models are better at estimating the class center, while generative models are better at modeling the data variance”. However, this claim lacks sufficient theoretical justification or citations from prior literature.
2. The method section does not present any explicit loss or objective function; the algorithm is only described procedurally, which weakens its mathematical rigor.
3. Experiments are conducted only on miniImageNet and CUB-200, lacking more challenging benchmarks, which limits the assessment of generalization.
4. The paper omits comparisons with recent state-of-the-art methods in open-set recognition and few-shot image generation, making it difficult to evaluate the competitiveness of the proposed approach.
5. The method introduces several critical hyperparameters, yet the paper provides no discussion or sensitivity analysis regarding their selection, which reduces reproducibility.

### Questions
1. The paper states that the calibrated visual prototype $\mu_c$ is injected into Stable Diffusion by replacing the $t_{EOS}$ and $t_{PAD}$ embeddings in the CLIP text encoder. This is a rather unusual and insufficiently justified design choice. The authors should provide ablation studies or theoretical reasoning to support this key component.
2. In Section 5.1, the authors mention using a “class-specific prior preservation loss” during LoRA training on the CUB-200 dataset, but not on miniImageNet. Why is this regularization applied to only one dataset? This inconsistency makes the comparison between datasets unfair and undermines the claimed generality of the method.
3. In Table 1, the $A_{last}$ score of DASE (Phase I) on CUB-200 is reported as 78.51%, whereas in Table 4, the corresponding method (“Calibrated Class Distribution”) achieves 78.15%. These two entries appear to represent the same experiment (the discriminative Phase I only), yet the results are inconsistent. Please clarify this discrepancy.

### Soundness
2

### Presentation
2

### Contribution
2
