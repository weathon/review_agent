# No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Latent diffusion models have achieved remarkable success in high-fidelity text-to-image generation, but their tendency to memorize training data raises critical privacy and intellectual property concerns. Membership inference attacks (MIAs) provide a principled way to audit such memorization by determining whether a given sample was included in training. However, existing approaches assume access to ground-truth captions. This assumption fails in realistic scenarios where only images are available and their textual annotations remain undisclosed, rendering prior methods ineffective when substituted with vision-language model (VLM) captions. In this work, we propose MoFit , a caption-free MIA framework that constructs synthetic conditioning inputs that are explicitly overfitted to the target model's generative manifold. Given a query image, MoFit proceeds in two stages: (i) model-fitted surrogate optimization, where a perturbation applied to the image is optimized to construct a surrogate in regions of the model’s unconditional prior learned from member samples, and (ii) surrogate-driven embedding extraction, where a model-fitted embedding is derived from the surrogate and then used as a mismatched condition for the query image. This embedding amplifies conditional loss responses for member samples while leaving hold-outs relatively less affected, thereby enhancing separability in the absence of ground-truth captions. Our comprehensive experiments across multiple datasets and diffusion models demonstrate that MoFit consistently outperforms prior VLM-conditioned baselines and achieves performance competitive with caption-dependent methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors are tackling the task of membership inference attack without access to ground truth captions. The idea is to find a way to distinguish between samples involved in the training of the generative diffusion model and those who don't. The problem become more feasible when not relying on ground truth captions since it is more applicable in real-life. The idea is to find cases of privacy violation of copyrights. The main conceptual idea of the method is that the difference between the conditional loss (conditioned on the caption, commonly referred as guidance-free generation), and the uncoditional one is at its highest on hold-out samples.

### Strengths
* The problem is stated explicitly and in a very understandable and accessible manner.
* The motivation of the method, and the intuition behind the structural choices is very clear and detailed, make it very accessible even for people which are less expert in this field.
* The experimental part is very impressive, the results of MoFit are better with a large margin then the competitors.

### Weaknesses
* The authors claim to be the first to tackle the caption-free MIA technique on generative diffusion models whereas [1], which not cited, explicitly claim to be the first (and already published). The authors should mention it and differentiate themselves from it, and to tone down their primacy in the field if it was already done previously.
* The evaluation time is super high - so the method is very slow and perhaps not feasible for set of images which is more than a few (7-9 minutes for a single image is reported in the Supp).
* I didnt see (correct me if Im wrong) any evaluation on a wide range of diffusion processes. Im not talking about the same architecture trained on several different datasets. I think that if you show that your approach is architecture-agnostic, namely when a novel diffusion mechanism will arrive your approach will highly likely work on it as well.
* I think it would be nice to analyze several fail cases, to better understand on which cases the method fails, even a short discussion on it (help follow-up works to understand the limitation of the method).


refs:

[1] Towards Black-Box Membership Inference Attack for Diffusion Models. Li et al. ICML 25.

### Questions
* for $\lambda$ optimization, in Eq. 5. The min is over all $\lambda$s, when there is no $\lambda$ in the formula itself. I assume that $z_t$ can be written as a function of $x_0$, it is somehow should be stated clearly, since minimizing on a variable which is not argument in the formula is confusing.
* it is not stated explicitly based on which LDM table 2 was produced?


I think that despite that caption-free methods were probably published earlier than this work, the paper is well written, and it seems novel to me. The paper flows nicely, there is a lot of elaboration on the intuition behind empirical decisions, which make it much more intuitive for the reader. The results of the paper are impressive, and I think that the authors cover relevant and timely topic with nice academic progress.

### Soundness
3

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
3

### Summary
This paper proposes a new framework, called MOFIT, for performing membership inference attacks on latent diffusion models without ground-truth captions. The paper first studies the differences in how member and non-member samples respond to mismatched conditioning during the denoising process. Based on this observation, the authors propose an efficient method to improve MIA in the caption-free setting.

### Strengths
- The problem being studied is practical. Privacy risks in generative diffusion models are an important topic. The work addresses a realistic limitation in existing MIAs by removing the dependency on ground-truth captions, making it applicable to real-world scenarios.

- Strong empirical performance: The results show consistent improvements over VLM-substituted baselines across multiple datasets and models.

- Comprehensive analysis: The paper provides detailed analysis and discussion of different components.

### Weaknesses
- Despite arguing that the proposed method has a more practical setting, it still relies on internal model signals (e.g., likelihoods or denoising trajectories required by CLiD) that may not translate to black-box APIs. While common in MIA research, this reduces practicality in real-world applications.


- The experiments focus on fine-tuned models on relatively small datasets (e.g., Pokémon with ~800 images), so generalization to large-scale, pre-trained models is unclear. It is important to conduct experiments on open-source pre-trained models with publicly available training sets. I am curious whether the proposed method still works on these models.


- It is also important to provide experiment on computational overhead. Surrogate optimization in high-dimensional conditioning spaces can be expensive and resource-intensive, especially for large models. Reporting runtime and scaling behavior would help assess practicality.


- Stability. The method relies on fixed noise bounds and timesteps for optimization, which might be sensitive to hyperparameters. The paper mentions using VLM initialization for stability but lacks ablation studies on failure cases or robustness to noise variations.

### Questions
Please see the weaknesses section

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
3

### Summary
Latent diffusion models (LDMs) have high text-to-image generation fidelity but risk training data memorization, audited by membership inference attacks (MIAs). Existing MIAs need ground-truth captions, ineffective with only images. This work proposes MOFIT, a caption-free MIA framework with two stages. Experiments show it outperforms VLM-conditioned baselines, even matching/surpassing caption-dependent methods.

### Strengths
1. Fills the "caption-free" gap, fits real scenarios with undisclosed training annotations, and has practical value.
2. Based on member samples’ higher sensitivity to mismatched conditions, its two-stage optimization is logically consistent.
3. Outperforms VLM-based baselines across datasets/models, and surpasses caption-dependent methods in some cases.

### Weaknesses
1. Its two-stage optimization takes 7-9 minutes per image on RTX 4090, far slower than VLM-based baselines, unable to handle large-scale data.
2. No tests on non-ideal images or small-sample LDMs; key parameters are only validated on Pokemon, lacking cross-scenario adaptability.
3. It excludes recent caption-free MIA studies, uses only 2 VLMs as baselines, and lacks theoretical explanation for member samples’ sensitivity.

### Questions
1. Optimize efficiency with adaptive stopping criteria and lightweight optimizers, and quantify efficiency-performance trade-offs for large-scale applicability.
2. Test on non-ideal images/small-sample LDMs, add more VLMs/recent studies as baselines to boost result robustness.
3. Derive member samples’ sensitivity from LDM mechanisms, analyze risks, and propose defense strategies and usage rules.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents new methods for membership inference attacks on diffusion models, focuses on generative/representation learning, and addresses privacy.

### Strengths
By tackling the MIA problem in the absence of ground-truth captions, the work fills a clear gap in current literature, reflecting realistic adversarial constraints faced in practice.

### Weaknesses
1. Limited diversity in experiments with large models: While SD v1.5 is included, the exploration of truly large-scale public models (e.g., with more challenging real-world splits or harder negative scenarios) is limited. The LAION-mi split required special curation and so does not test MoFit “in the wild” under extreme generalization.


2. The paper focuses heavily on the positive results of MoFit, but does not sufficiently examine its limitations, potential false-positive drivers (e.g., for out-of-distribution or near-duplicate samples), or how defender-side strategies (e.g., differential privacy or input randomization) could mitigate its efficacy.

3. No significant discussion of defense strategies: The paper states findings are for diagnostic/privacy audit purposes, but does not offer suggestions or preliminary results for possible countermeasures or design recommendations for more robust LDMs.

### Questions
1. How does MoFit’s performance transfer to other classes of generative models or different LDM architectures? Could the authors provide results (or analysis) for other types of datasets (e.g., medical or highly out-of-distribution images) not covered here?

2. What, in the authors' view, would be the most promising defense for LDMs against MoFit, aside from differential privacy? Are there results for even naïve defense schemes

3. Can the authors provide distributions or examples of hard false positives/negatives at very low FPRs? Are there query images or image properties where MoFit’s advantage is diminished, and if so, what characteristics drive this behavior?

### Soundness
3

### Presentation
3

### Contribution
3
