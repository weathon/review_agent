# CARL: Camera-Agnostic Representation Learning for Spectral Image Analysis

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 4

## Abstract
Spectral imaging offers promising applications across diverse domains, including medicine and urban scene understanding, and is already established as a critical modality in remote sensing.
However, variability in channel dimensionality and captured wavelengths among spectral cameras impede the development of AI-driven methodologies, leading to camera-specific models with limited generalizability and inadequate cross-camera applicability.
To address this bottleneck, we introduce CARL, a model for Camera-Agnostic Representation Learning across RGB, multispectral, and hyperspectral imaging modalities.
To enable the conversion of a spectral image with any channel dimensionality to a camera-agnostic representation, we introduce a novel spectral encoder, featuring a self-attention-cross-attention mechanism, to distill salient spectral information into learned spectral representations.
Spatio-spectral pre-training is achieved with a novel feature-based self-supervision strategy tailored to CARL. 
Large-scale experiments across the domains of medical imaging, autonomous driving, and satellite imaging demonstrate our model's unique robustness to spectral heterogeneity, outperforming on datasets with simulated and real-world cross-camera spectral variations. 
The scalability and versatility of the proposed approach position our model as a backbone for future spectral foundation models. Code and model weights are publicly available at https://github.com/IMSY-DKFZ/CARL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an architecture called CARL and a self-supervised learning (SSL) algorithm for it called CARL-SSL. CARL consists of a spectral encoder followed by a spatial encoder. The spectral encoder is essentially a transformer over the spectral dimension, i.e., a token is a spatial patch of a single spectral channel and attention occurs over tokens at the same spatial location (= inter-channel attention). The spectral encoder outputs K tokens per spatial location, which represent the spectral information (and some spatial information, since the patch size is not 1) at the location. The spatial encoder is a ViT. 

CARL-SSL consists of two learning objectives. The first adapts VICReg to pre-train the spectral encoder and the second adapts I-JEPA to pre-train the spatial encoder. 

The paper demonstrates CARL's superiority on many benchmarks and domains, such as remote sensing, medical imaging, and automotive data.

### Strengths
- The paper is dense with a lot of content, yet is well-written overall
- Both CARL and CARL-SSL are technically novel
- CARL is demonstrated across many domains
- The challenge that CARL aims to solve is important. Cameras with more colour channels contain more information that can be used to improve predictions, but the field has not settled on the optimal way to process many channels.

### Weaknesses
Major concern:
- Computational cost. It is not clear how much more or less expensive CARL is relative to baseline methods. I see a computational complexity section in the appendix, showing that for an input of 128x128x48, CARL is 15X the cost of DOFA. In my opinion, this makes the comparison unfair at this input size. The cost at other sizes is unclear. I believe that CARL is computationally expensive because it uses a separate spectral encoder and it may have more tokens, since it uses k=8 tokens per spatial location. 

Moderate concerns:
- Limited novelty. The spectral encoder is a transformer over the spectral dimension, in a sense it reminds me of axial transformers/attention (which attend to one axis at a time). It does include learned query tokens, differentiating it from a purely axial framework, which means it always outputs K tokens per spatial location. Have the authors thought about using a spectral encoder with only self-attention, then pooling all tokens along the spectral axis? This is a simpler means of achieving K=1 tokens per spatial location.
- Limited ablations. The paper includes some ablations, e.g., Table 3c (a single result that removes the spectral SSL task) and Table 6 (spectral position encoding, aggregation, K, and the feature dimension). These are informative but I'd like some more fundamental ablations, e.g., removing the spectral encoder and simply tokenizing all spatio-spectral patches with a vanilla transformer.
- Benchmarks and Baselines. Although CARL is evaluated on many domains, within a domain the experiments are not rigorous enough, in my view. For example, in remote sensing (one of the primary domains of the paper), EuroSat and BigEarthNet are not used despite being the most popular benchmarks. Regarding baselines, the paper cites Galileo (SOTA RS foundation model) but does not compare to it.

### Questions
Please see weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a method for camera-agnostic spatio-spectral
representation learning across RGB, multispectral, and hyperspectral images. It
introduces a spectral encoder with self- and cross-attention to learn
camera-agnostic spectral representations, combined with I-JEPA spatial
pretraining for self-supervised learning. This approach is evaluated across medical
imaging, autonomous driving, and satellite imaging, and claims to outperform
both camera-specific and channel-invariant baselines in terms of generalization
to unseen sensors.

### Strengths
- The paper tackles a relevant problem: variability across spectral cameras that limits generalization.
 - The combination of spectral tokenization, spatio-spectral aggregation, and self-supervised pretraining is reasonable.
 - Demonstrates cross-domain evaluation, including medical, automotive, and remote sensing datasets.

### Weaknesses
- The methodological contribution is limited: the components (spectral tokenization, 2D token aggregation, spatial feature extraction, I-JEPA pretraining) are standard components. Presenting it as a new ``framework'' is a bit of an overstatement.
 - The claims ``first spatio-spectral camera-agnostic representation learning'' is overstated; I am not sure why a model such as DOFA does not fall into this category. The related work Panopticon [1] seems to provide a similar channel encoding scheme (any comment on this will be appreciated). A clear positioning with respect to other models such as Copernicus-FM [2], SMARTIES [3], and Fleximo [4] would also be useful.
 - The remote sensing evaluation is a bit weak. A comparison with Panopticon would be helpful. Regarding the claim ``our approach outperformed both camera-specific and channel-invariant baselines'', for an equivalent compute budget, it is not clear that the proposed model performs better than one trained for a single sensor or a fixed set of sensors. The experimental section does not demonstrate this. Moreover, pretraining is conducted on SpectralEarth and a Sentinel-2 dataset (only two sensors), and three out of the four datasets used for evaluation contain only Sentinel-2 data, which was already seen during pretraining.
 - The evaluation on GeoBench does not use all the available datasets. Moreover, only SpectralGPT, DOFA, and SatMAE are compared to on one of the datasets. To support the claim to outperform sensor-specific models, comparisons with Terramind [5], CROMA [6], and others would be useful. (It may be possible to reuse results reported from previous papers for some of the datasets).
 - Additionally, in Table 4, linear probing is used across three datasets. This is unusual for remote sensing segmentation tasks, where models typically use an UpperNet decoder or at least a few convolutions and upscaling layers. That could explain the relatively weak results.
 - Code is not provided at submission time.


[1] Waldmann, Leonard, et al. "Panopticon: Advancing any-sensor foundation models for earth observation." CVPR 2025.

[2] Wang, Yi, et al. "Towards a unified copernicus foundation model for earth vision." arXiv preprint arXiv:2503.11849 (2025).

[3] Sumbul, Gencer, et al. "SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images." arXiv preprint arXiv:2506.19585 (2025).

[4] Li, Xuyang, et al. "Fleximo: A flexible remote sensing foundation model." arXiv preprint arXiv:2503.23844 (2025).

[5] Jakubik, Johannes, et al. "Terramind: Large-scale generative multimodality for earth observation." arXiv preprint arXiv:2504.11171 (2025).

[6] Fuller, Anthony, Koreen Millard, and James Green. "CROMA: Remote sensing representations with contrastive radar-optical masked autoencoders." NeurIPS 2023

### Questions
- How does CARL compare to Panopticon or Terramind on datasets not included in the pretraining corpus, specifically for true camera-agnostic generalization?
 - Why linear probing is used for segmentation tasks instead of standard decoders, and how this choice affects results?
 - How sensitive are results to pretraining dataset composition? For example, would adding more diverse sensors improve generalization?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces CARL, a novel framework for camera-agnostic representation learning across RGB, multispectral, and hyperspectral imaging modalities. The key innovation is a spectral encoder that uses wavelength positional encoding and learnable spectral representations with a self-attention-cross-attention mechanism to convert spectral images with arbitrary channel dimensionality into camera-agnostic representations. The authors also propose CARL-SSL, a self-supervised pre-training strategy combining spectral and spatial feature-based learning. Extensive experiments across medical imaging, autonomous driving, and satellite imaging demonstrate superior performance compared to camera-specific and channel-invariant baselines.

### Strengths
1. The paper addresses a critical bottleneck in spectral imaging—the inability of existing models to generalize across cameras with different spectral properties. This is particularly relevant for domains like medical imaging where sensor diversity is high.
2. CARL is the first approach to combine wavelength-awareness, channel-invariance, spatio-spectral encoding, and spatio-spectral SSL pre-training (Table 1). The wavelength positional encoding is an elegant solution for establishing cross-camera channel correspondences.
3. The evaluation spans three diverse application domains with both synthetic and real-world spectral variations. The progressive substitution experiments (Figure 5) provide compelling evidence of robustness to spectral heterogeneity.
4. CARL consistently outperforms strong baselines across all experiments. The cross-modality knowledge transfer demonstrated in Figure 4 (e.g., transferring "pole" labels from RGB Cityscapes to HSI) is particularly impressive.
5. The ablation studies (Table 6) validate key architectural decisions, and the variance decomposition analysis (Table 10) provides insight into what the model learns.

### Weaknesses
1.  While σ=3 and K=8 are validated through ablations, the paper doesn't discuss how sensitive these choices are across different domains or whether they need domain-specific tuning.
2. Scalability Limitations: The experiments use relatively modest model sizes (small and base versions). It's unclear whether the approach scales to larger foundation models or whether architectural modifications would be needed.
3.  While CARL-SSL shows improvements, the relative contribution of spectral vs. spatial self-supervision could be explored more thoroughly. The ablation in Table 3c is limited to a single dataset with reduced budget.

### Questions
1. The authors mention spatial resolution differences as a limitation. Could the wavelength positional encoding framework be extended to handle spatial resolution as metadata? Have you experimented with this?
2. How does CARL perform when test wavelengths fall outside the training distribution? For example, if trained on visible-NIR but tested on SWIR?
3. You mention sampling 32 channels during training when C>32. What is the impact of this sampling ratio? Could adaptive sampling based on wavelength diversity improve performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes CARL (Camera-Agnostic Representation Learning), a framework designed to achieve cross-camera generalization across RGB, multispectral, and hyperspectral imaging systems. CARL introduces a spectral encoder with self- and cross-attention modules to map arbitrary channel-dimensional spectral images into a camera-agnostic latent space using wavelength positional encoding. The model further incorporates a spatio-spectral self-supervised learning (CARL-SSL) scheme combining spectral masking and spatial predictive learning (based on VICReg and I-JEPA). Experiments are presented across three domains, such as medical imaging, automotive vision, and satellite imaging, demonstrating improved robustness under heterogeneous spectral conditions and outperforming several baselines such as DOFA, Hyve, and SpectralGPT.

### Strengths
Problem relevance: The challenge of camera-specific spectral variability is real and significant in multisensor spectral imaging and remote sensing.

Comprehensive evaluation: The authors conduct cross-domain tests (medical, automotive, satellite), showing a broad view of potential applications.

Framework completeness: CARL integrates both wavelength-aware encoding and SSL, aligning with ongoing developments in spectral foundation models.

Potential impact: If scalable and reproducible, the framework could contribute to harmonized spectral representation learning across heterogeneous sensors.

### Weaknesses
Limited novelty / Incremental contribution.
Despite a well-written motivation, the technical innovation is incremental rather than groundbreaking. The proposed spectral encoder (self-attention + cross-attention) and wavelength positional encoding are straightforward extensions of existing works such as DOFA (Xiong et al., 2024), Hyve (Varga et al., 2023), and SpectralGPT (Hong et al., 2024). The SSL design (masked prediction + VICReg) follows existing feature-level pretraining frameworks (I-JEPA, DINOv2) without novel loss functions or theoretical insight. The “camera-agnostic” formulation is largely a rebranding of channel-invariant learning.

Lack of clear incremental improvement.
Reported gains (typically 1–2 mIoU or OA points) over existing baselines are marginal and fall within confidence intervals. The work does not convincingly demonstrate why the proposed cross-attention or wavelength encoding leads to generalizable representations beyond what existing channel-adaptive or wavelength-aware layers already achieve.

Insufficient discussion of existing multimodal or foundation models.
The manuscript lacks substantial comparison or discussion of recent multimodal foundation models (e.g., OmniSat, CROMA, Galileo, SeaMo, etc.) and fails to position CARL relative to these stronger baselines. This omission weakens its claim of being a “backbone for future spectral foundation models.”

Poor figure readability.
Figures (e.g., Fig. 1–5) are small, densely packed, and use fonts unreadable at standard scale. Important architectural elements and result visualizations are nearly illegible, making it difficult to evaluate the technical contributions.

Experimental design limitations.

The model is only tested on moderate-scale datasets (≤800 k samples) and not on true foundation-model-scale data.

Cross-domain transfer is limited to a few camera pairs (e.g., Cityscapes–HSICity); results on other heterogeneous sensors (e.g., PRISMA, ZY1-HSI) are missing.

Computational efficiency and scalability analyses are superficial, despite the model’s significant complexity.

Overstated claims.
The claim of being the “first camera-agnostic framework” is inaccurate, as previous models have already introduced similar wavelength- and channel-adaptive mechanisms. The novelty is therefore overstated.

### Questions
How does CARL differ quantitatively and conceptually from prior channel-invariant or wavelength-aware transformers?

What is the true scale of CARL in parameter count and computational cost compared to foundation models such as SpectralGPT or SatMAE?

Can the authors show clearer visualizations (e.g., spectral attention maps) to support claims of camera-agnostic learning?

How robust is CARL to non-spectral variations (spatial resolution, atmospheric noise, illumination)?

### Soundness
2

### Presentation
2

### Contribution
2
