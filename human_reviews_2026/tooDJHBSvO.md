# DiffSDA: Unsupervised Diffusion Sequential Disentanglement Across Modalities

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Unsupervised representation learning, particularly sequential disentanglement, aims to separate static and dynamic factors of variation in data without relying on labels. This remains a challenging problem, as existing approaches based on variational autoencoders and generative adversarial networks often rely on multiple loss terms, complicating the optimization process. Furthermore, sequential disentanglement methods face challenges when applied to real-world data, and there is currently no established evaluation protocol for assessing their performance in such settings. Recently, diffusion models have emerged as state-of-the-art generative models, but no theoretical formalization exists for their application to sequential disentanglement. In this work, we introduce the Diffusion Sequential Disentanglement Autoencoder (DiffSDA), a novel, modal-agnostic framework effective across diverse real-world data modalities, including time series, video, and audio. DiffSDA leverages a new probabilistic modeling, latent diffusion, and efficient samplers, while incorporating a challenging evaluation protocol for rigorous testing. Our experiments on diverse real-world benchmarks demonstrate that DiffSDA outperforms recent state-of-the-art methods in sequential disentanglement.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents a new systematic approach to latent factor disentanglement of sequential data. It is based on a combination of several models 1) to extract static and dynamic components of latent factors, 2) to stochastically encode sequential data, 3) to conditionally decode sequential data based on latent factors. Additional components to deal with high-dimensional data (such as video) might be required as well, such as VQ-VAE.

The proposed method is described using the language of probabilities, akin to what we are used to when dealing with diffusion models.

The authors perform a thorough and extensive validation campaign, using an appropriate experimental protocol. They define both qualitative and quantitative metrics for the numerous tasks they address and show that the proposed method is superior to the state of the art.

### Strengths
* The experimental protocol used to assess DiffSDA and compare it to alternatives is extensive and globally well done. The authors have produced convincing qualitative and quantitative results indicating that the proposed method outperforms current SOTA approaches.

* Measuring quantitatively the effectiveness of disentanglement is a hard problem. The authors worked really hard on this issue and 1) tested an external judge (a classifier), and 2) proposed an unsupervised swapping metric to sidestep the limitations of 1).

* This work is self contained: as a reader, I greatly appreciate the quality and depth of the material in the appendix, especially the details on the metrics, as well as the additional results and ablations.

### Weaknesses
* It's minor, but I would argue that the narrative used in the introduction about the requirement for "labels" for generative models is a bit misleading. Text-to-image generation indeed requires pairs of image and prompts, but unconditional models have flourished before those work, and they do not require labels. In general, I find the problem of learning disentangled representation sufficiently recognized as an important issue to address in the community, that does not need additional motivations.

* I am a bit worried about the claim of "simplicity" of the proposed approach. I view the proposed method as a complex system, involving several models (some of which pretrained, others that require separate training) combined in a coherent design, which is far from being simple. Also, the lack of apparent hyper-parameters (a critic the authors advance for other methods on disentanglement from the state of the art), is in my humble opinion, another illustration of downplaying the complications of the proposed system. In practice, diffusion models are not so easy to train, as demonstrated by the choice of the authors to rely on an enormous amount of work that has been done to come up with "recipies" to elucidate the design space of diffusion models.

### Questions
* From the introduction, the requirement for "labeled" data stems from, e.g., text-to-image generation. Original work from Ho et al., and from Song et al., are unconditional models which do not require labeled data.

* Check your claims: when the authors say they propose a novel probabilistic approach, they are actually using standard practice in diffusion modeling. The main difference is that they combine two diffusion processes, one for learning (and generating) disentangled latents for static and dynamic components, and one for learning (and generating) the sequential data distribution, conditioned on the disentangled latents. Both models are learned concurrently using a single loss. While my remark does not want to diminish the technical contribution of the proposed method, I think that the claim for novelty is too strong.

* As stated in lines 212-213, eq. (3) does not require training for the latents $(s_0, d_0^{1:V})$ generative model, thus this optimization is separate. Then, the loss in eq. (3) is not the only loss you consider in your overall training of DiffSDA, right? Then, I would like to ask the authors to clarify why in lines 221-222 they claim that only a single loss term is required.

* In line 227, the last hidden representation $h^V$ is fed to a linear layer to produce $s_0$, but this is not reflected in Figure 1. This is just nitpicking, but for clarity, the figure could be easily amended to be coherent with the textual description of the architecture.

* In lines 265-269, the authors mention that for high-dimensional data such as video, they abuse the notation (for ease of reading), and use symbols that denote ambient space, to indicate vector-quantized latent variables. This is clear. The question is: when considering other modalities, such as audio or general timeseries, there is no need for the VQ-VAE embeddings right?

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
This paper proposes DiffSDA, a novel framework for unsupervised sequential disentanglement that leverages diffusion models to separate static and dynamic factors of variation across diverse modalities such as video, audio, and time-series data.
The key claimed contributions are:A novel, modal-agnostic probabilistic framework for sequential disentanglement based on diffusion processes, which models static and dynamic factors as interdependent.An efficient design that achieves disentanglement using a single, unified score estimation loss, simplifying optimization compared to methods requiring multiple loss terms.Demonstrated effectiveness on high-dimensional, real-world data, with capabilities for zero-shot disentanglement and further factorization into multiple interpretable factors via post-hoc PCA.The introduction of a new evaluation protocol for video-based disentanglement, including high-resolution datasets and unsupervised metrics (AED, AKD).

### Strengths
1. The work successfully bridges powerful diffusion models with the challenging problem of sequential disentanglement, a domain previously dominated by VAE/GAN-based approaches (e.g., SPYL, DBSE). The modal-agnostic claim is well-supported by experiments.
2. The use of a single diffusion loss term is a significant advantage, effectively circumventing the complex hyperparameter tuning and multi-term loss balancing required in prior works (e.g., C-DSVAE uses five loss weights). This makes the method more accessible and robust.
3. The paper provides extensive experiments. Qualitatively, the conditional/zero-shot swap and multifactor traversal results (e.g., Figs. 2, 3, 4, 23-47) are visually compelling and demonstrate clear disentanglement. 
4. Quantitatively, the model consistently outperforms strong baselines (SPYL, DBSE) across multiple metrics (AED, AKD, etc.) and datasets.

### Weaknesses
1. Why is the dependent modeling of static and dynamic factors theoretically justified or preferable for disentanglement? The empirical result (13% FVD improvement) is convincing, but an intuitive or formal explanation is lacking.
2. A more detailed analysis of why the single diffusion loss naturally leads to disentanglement, beyond the empirical observations that the static factor is shared and the dynamic factors are low-dimensional, would be highly valuable.
3. The paper highlights the efficiency of the EDM sampler (63 NFEs) but does not provide a comparative analysis of training or inference costs against the baselines. Given that diffusion models are generally more computationally intensive, a discussion of this trade-off would be helpful.

### Questions
1.What key property of the diffusion model or its learned representations enables such strong cross-dataset, zero-shot disentanglement?
2.Are the semantic factors discovered via PCA (e.g., gender, age) consistent and stable across different random seeds and model initializations?
3.From an optimization standpoint, please justify the efficiency of the single loss formulation over a multi-term objective, for instance, by analyzing its mitigation of gradient conflict.
4.Please provide a quantitative comparison of training/inference time and GPU memory footprint against key baselines like SPYL and DBSE.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
DiffSDA is a modal-agnostic diffusion sequential disentanglement autoencoder that factorizes static and dynamic factors using a single, unified score-estimation loss.

### Strengths
1. It allows static and dynamic factors to be interdependent, which makes the model more general.

2. It can be applied to video, audio and time-series data, and achieves sota performance.

3. It present a new evaluation protocol for high-quality visual sequential disentanglement.

### Weaknesses
1. I am curious to what extent this model can reduce hyperparameter tuning. The paper claims that other methods always rely on multiple loss terms, making the optimization process more complex, whereas this model only requires a single standard loss term. However, in Appendix B.2, it seems that the VQ-VAE is pre-trained with a perceptual loss and a patch-based adversarial objective, and in Appendix B.3, you train a latent DDIM prior. Could you elaborate on how this method is actually simpler than GAN- or VAE-based approaches that can also disentangle dynamic and static components?

2. In Figure 3, it seems that SPYL failed to reconstruct the image well. Since reconstruction is usually simpler than disentanglement, this might be due to bad hyperparameters. If such trivial failures like SPYL’s could be avoided, the comparison would be more meaningful.

### Questions
See Weaknesses.

### Soundness
3

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
4

### Summary
This paper propose a unsupervised diffusion model that can disentangle factors in a time series data. The model is insprired on a principled probabilistic model, with the ability of both dynamic-static and conponent level disentanglement. The extensive experiments results verify the conclusion.

### Strengths
1. A probabilitic model is provided, which provide a theoritical guarantee for disentanglement in diffusion models.
2. Extensivce experiments are provided, which makes the paper more convincing.
3. The paper is clear and easy to follow.

### Weaknesses
### Major Concerns
1. This paper assumes $d$ and $s$ are independent. However, in many real world senarios, they cannot be independent, as sometimes the content of a video can decide the type of actions to be taken in a video. This limits the generalization of thie model. 
2. The dataset is limited to simple scenarios like human face, where this assumption is reasonable. However, the performance in open-domain dataset is not verified.
3.  This paper claims the ability of disentanglement, but some traditional disentanglement metrics are not compared, i.e., SAP, modularity, e.t.c..
4. The baseline models in the paper are out-dated, especially for the video generation part. 
### Minor Concerns
1. Typo: line 203 pr-evious

### Questions
1. The video is generated frame by frame, which may reduce the spatio-temporal consistency in the video generation. Is it possible to adapt this model to LVDM or similar models to strength its video generation ability?
2. Can the unsupervisely learned concepts be combined with other supervised condition signals? For example, how this method can contribute to the current T2V generation framework?

### Soundness
2

### Presentation
3

### Contribution
2
