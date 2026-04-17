# Withanyone: Toward Controllable And

![0_Image_0.Png](0_Image_0.Png) Id Consistent Image Generation

Hengyuan Xu1,2 Wei Cheng2,† Peng Xing2 Yixiao Fang2 Shuhan Wu2 **Rui Wang**2 Xianfang Zeng2 Daxin Jiang2 Gang Yu2,‡ Xingjun Ma1,‡ **Yu-Gang Jiang**1 1 Fudan University 2 StepFun

![0_image_1.png](0_image_1.png)

Figure 1: **Showcases of WithAnyone.** WithAnyone is capable of generating high-quality, controllable, and ID-consistent images by leveraging ID-contrastive training on the proposed **MultiID-2M** dataset. IDs above are authors' and authors' friends'.

## Abstract

Identity-consistent (ID-consistent) generation has become an important focus in text-to-image research, with recent models achieving notable success in producing images aligned with a reference identity. Yet, the scarcity of large-scale paired datasets—containing multiple images of the same individual—forces most approaches to adopt reconstruction-based training. This reliance often leads to a failure mode we term *copy-paste*, where the model directly replicates the reference face rather than preserving identity across natural variations in pose, expression, or lighting. Such over-similarity undermines controllability and limits the expressive power of generation. To address these limitations, we (1) construct a large-scale paired dataset **MultiID-2M** tailored for multi-person scenarios, providing diverse references for each identity; (2) introduce a benchmark that quantifies both copypaste artifacts and the trade-off between identity fidelity and variation; and (3) propose a novel training paradigm with a contrastive identity loss that leverages paired data to balance fidelity with diversity. These contributions culminate in WithAnyone, a diffusion-based model that effectively mitigates copy-paste while preserving high identity similarity. Extensive experiments—both qualitative and quantitative—demonstrate that WithAnyone substantially reduces copy-paste artifacts, improves controllability over pose and expression, and maintains strong perceptual quality. User studies further validate that our method achieves high identity fidelity while enabling expressive, controllable generation. Our project is fully open-sourced at HTTPS://DOBY-XU.GITHUB.IO/WITHA**NYONE**/.

1

## 1 Introduction

With the rapid progress of generative AI, controllable image generation via reference images or image prompting (Ruiz et al., 2023; Hertz et al., 2022; Zhang et al., 2023a; Xiao & Fu, 2024; Hu et al., 2025b; Wu et al., 2024a) and identity-consistent (ID-consistent) generation (Ye et al., 2023; Guo et al., 2024; Wang et al., 2024c; Jiang et al., 2025a; Cheng et al., 2025; Zhang et al., 2025; He et al., 2024) have achieved remarkable advances: modern models can synthesize portraits that closely match the provided individual. Recent efforts (Cheng et al., 2025; Chen et al., 2025) push resemblance toward near-perfect reproduction. While pursuing higher similarity seems natural, beyond a certain point, excessive fidelity becomes counterproductive.

![1_image_0.png](1_image_0.png)

![1_image_1.png](1_image_1.png)

In real photographs of the same person, identity similarity varies substantially due to natural changes in pose, expression, makeup, and illumination (Fig. 2). By contrast, many generative models adhere to the reference image far more rigidly than this natural range of variation. Although such over-optimization may seem beneficial, it suppresses legitimate variation, reducing controllability and limiting practical usability. We term this failure mode the **copy-paste** artifact: rather than synthesizing an identity in a flexible, controllable manner, the model effectively copies the reference image into the output (see Fig. 2). In this work, we formalize this artifact, develop metrics to quantify it, and propose a novel training strategy to mitigate it. Mitigating copy-paste artifacts is fundamentally constrained by the lack of suitable training data. While numerous large-scale face datasets exist (Liu et al.,
2015; Stacchio et al., 2020; Chu et al., 2024; Zhang et al., 2015; Jiang et al., 2025b; Zhong et al., 2018; Wang et al., 2025), they remain ill-suited for controllable multi-identity generation. Critically, few datasets provide paired references for each identity—multiple images of the same person across diverse expressions, poses, hairstyles, and viewpoints. As a result, most prior work resorts to single-person, reconstruction-based training (Guo et al., 2024; Wang et al., 2024c), where the reference and target coincide. This setup inherently promotes copying and exacerbates copy-paste artifacts. Constructing datasets with multiple references per identity, particularly in group photos, and developing methods to effectively exploit such data remain open challenges. In this work, we introduce a large-scale open-source Multi-ID dataset, **MultiID-2M**, together with a comprehensive benchmark, **MultiID-Bench**, designed for intrinsic evaluation of multi-identity image generation. MultiID-2M contains 500k group photos featuring 1–5 recognizable celebrities. For each celebrity, hundreds of individual images are provided as paired references, covering diverse expressions, hairstyles, and viewing angles. In addition, 1.5M unpaired group photos without references are included. MultiID-Bench establishes a standardized evaluation protocol for multiidentity generation. Beyond widely adopted metrics such as ID similarity (Schroff et al., 2015; Deng et al., 2019), it quantifies copy-paste artifacts by measuring distances between generated images, references, and ground truth. Evaluation on 12 state-of-the-art customization models highlights a clear trade-off between ID similarity and copy-paste artifacts (see Fig. 5).

Furthermore, we present **WithAnyone**, a novel identity customization model built on the FLUX (Batifol et al., 2025) architecture, as a step toward mitigating copy-paste artifacts. WithAnyone maintains state-of-the-art identity similarity (with regard to target image) while substantially reducing copypaste, thereby breaking the long-observed trade-off between fidelity and artifacts. This advance is enabled by a paired-training strategy combined with an ID contrastive loss enhanced with a large negative pool, both made possible by our paired dataset. The labeled identities and their reference Prompt: A blonde lady, natural makeup Input InstantID PULID WithAnyone GT
Copy Copy **Customized**
Figure 2: **Our Observation**. Natural variations, such as head pose, expression, and makeup, may cause more face similarity decrease than expected. Copying reference image limits models' ability to respond to expression and makeup adjustment prompts.

images enable the construction of an extended negative pool (images of different identities), which provides stronger discrimination signals during optimization. In summary, our main contributions are:
- **MultiID-2M:** A large-scale dataset of 500k group photos containing multiple identifiable celebrities, each with hundreds of reference images capturing diverse variations, along with 1.5M additional unpaired group photos. This resource supports pre-training and evaluation of multi-identity generation models.

- **MultiID-Bench:** A comprehensive benchmark with standardized evaluation protocols for identity customization, enabling systematic and intrinsic assessment of multi-identity image generation methods.

- **WithAnyone:** A novel ID customization model built on FLUX that achieves state-of-the-art performance, generating high-fidelity multi-identity images while mitigating copy-paste artifacts and enhancing visual quality.

## 2 Related Work

Single-ID Preservation. Identity-preserving image generation is a core topic in customized synthesis (Wang et al., 2024a; Huang et al., 2024; Arar et al., 2024; Jones et al., 2024; Kumari et al., 2024; Zeng et al., 2023; Arar et al., 2023; Ma et al., 2024; Valevski et al., 2023; Wang et al., 2024b; Yan et al., 2023; Xiao et al., 2025; Wu et al., 2024b; Wang et al., 2024d; Chen et al., 2024; Hyung et al., 2024; Papantoniou et al., 2024). Many methods in the UNet/Stable Diffusion era inject learned embeddings (e.g., CLIP or ArcFace) via cross-attention or adapters (Ho et al., 2020; Ronneberger et al., 2015; Qian et al., 2024; Ye et al., 2023; Radford et al., 2021; Ren et al., 2023). With the rise of DiT-style backbones (Peebles & Xie, 2023; Esser et al., 2024; Labs, 2024) (e.g., SD3, FLUX), progress on ID preservation like PuLID (Guo et al., 2024), also attracts great attentions. Multi-ID Preservation. Multi-ID preservation remains relatively underexplored. Some works target spatial control of multiple identities (Kim et al., 2024; He et al., 2024; Zhang et al., 2025), while others focus on identity fidelity. Methods such as XVerse (Chen et al., 2025) and UMO (Cheng et al., 2025) use VAE-derived face embeddings concatenated with model inputs, which can produce pixel-level copy-paste artifacts and reduce controllability. DynamicID (Hu et al., 2025a)1achieves improved controllability but is constrained by limited task-specific data and evaluation standards. Other general-purpose customization and editing models (Parmar et al., 2025; Mou et al., 2025; Patashnik et al., 2025; Wu et al., 2025d; Xiao et al., 2024; Wu et al., 2025b;c; Batifol et al., 2025; Wu et al., 2025a) can also synthesize images containing multiple identities, but their ID similarity are often compromised for generality.

ID-Centric Datasets and Benchmarks. Although numerous single-ID datasets (Karras et al.,
2017; Wang et al., 2025) and multi-ID collections (Chu et al., 2024; Jiang et al., 2025b) exist, paired reference images are scarce, so reconstruction remains the dominant training objective for multi-ID
datasets. Representative datasets are listed in Table 4. Evaluation protocols are underdeveloped:
several works (e.g., PuLID (Guo et al., 2024), UniPortrait (He et al., 2024), and others (Xiao et al., 2025; Zhang et al., 2025)) construct test sets by sampling identities from CelebA (Liu et al., 2015), which undermines reproducibility. To address this, we release a curated multi-ID benchmark with standardized splits and comprehensive metrics to facilitate future research.

## 3 Multiid-2M: Paired Multi-Person Dataset Construction

MultiID-2M is a large-scale multi-person dataset constructed via a four-stage pipeline: (1) collect single-ID images from the web and construct a clean reference bank by clustering ArcFace (Deng et al., 2019) embeddings, yielding ∼1M reference images across ∼3k identities (averaging 400 per identity); (2) retrieve candidate group photos via multi-name and scene-aware queries and detect faces; (3) assign identities by matching ArcFace embeddings to single-ID cluster centers using cosine 1Excluded from our experiments due to unavailability of code and pretrained models.

![3_image_0.png](3_image_0.png)

## 4 Multiid-Bench: Comprehensive Id Customization Evaluation

MultiID-Bench is a unified benchmark for group-photo (multi-ID) generation. It samples rare, longtail identities with no overlap to training data, yielding 435 test cases. Each case consists of one ground-truth (GT) image containing 1–4 people, the corresponding 1–4 reference images as inputs, and a prompt describing the GT. Detailed statistics are provided in Appendix C. Evaluation considers both identity fidelity and generation quality. Let r, t, g denote the face embeddings of the reference identity, the target (ground-truth), and the generated image, respectively. We define similarity between two embeddings as Sim(a, b), specifically we term the generated image's face similarity with regard to GT as SimGT, and to reference as SimRef,

$$\mathrm{Sim}(\mathbf{a},\mathbf{b})={\frac{\mathbf{a}^{\top}\mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}},\quad\mathrm{Sim}_{\mathrm{GT}}={\frac{\mathbf{g}^{\top}\mathbf{t}}{\|\mathbf{g}\|\,\|\mathbf{t}\|}},\quad\mathrm{Sim}_{\mathrm{Ref}}={\frac{\mathbf{g}^{\top}\mathbf{r}}{\|\mathbf{g}\|\,\|\mathbf{r}\|}}.$$
(1)  $\frac{1}{2}$ ................................. (1)  ... 
Prior works (Zhang et al., 2025; He et al., 2024; Guo et al., 2024; Cheng et al., 2025) has largely reported only SimRef, which inadvertently favors trivial copy-paste: directly replicating the reference appearance maximizes the score, even when the prompt specifies changes in pose, expression, or viewpoint. In contrast, MultiID-Bench uses SimGT—the similarity to the ground-truth identity described by the prompt—as the primary metric. This design penalizes excessive copying when natural variations (e.g., pose, expression, occlusion) are expected, while rewarding faithful realization of the prompted scene.

![4_image_0.png](4_image_0.png)

$$\left(2\right)$$

We define the angular distance as θab = arccos(Sim(a, b)) (geodesic distance on the unit sphere).

The Copy-Paste metric is given by

$$\mathbf{M}_{\mathrm{CP}}(\mathbf{g}\mid\mathbf{t},\mathbf{r})={\frac{\theta_{g t}-\theta_{g r}}{\operatorname*{max}(\theta_{t r},\,\varepsilon)}}\in[-1,1],$$
∈ [−1, 1], (2)
where ε is a small constant for numerical stability. The metric thus captures the relative bias of g toward the reference r versus the ground truth t, normalized by angular distance of r and t. A score of 1 means g fully coincides with the reference (perfect copy-paste), while −1 means full agreement with the ground truth. We additionally report identity blending, prompt fidelity (CLIP I/T), and aesthetics; formal definitions and further details are provided in Appendix D.

## 5 Withanyone: Controllable And Id-Consistent Generation

Building on the scale and paired-reference supervision of the MultiID-2M, we devise training strategies and tailored objectives that transcend reconstruction to enable robust, identity-conditioned synthesis. This rich, identity-labeled supervision not only substantially improves identity fidelity but also suppresses trivial copy–paste artifacts and affords finer control over multi-identity composition. Motivated by these advantages, we introduce WithAnyone - a unified architecture and training recipe designed for controllable, high-fidelity multi-ID generation. Architectural schematics and implementation details are provided in Fig. 4 and Appendix E.

## 5.1 Training Objectives

Diffusion Loss. We adopt the mini-batch empirical flow-matching loss. For each batch, we sample a data latent x1 ∼ pdata, Gaussian noise x0 ∼ N (0, I), and a timestep t ∼ U(0, 1). We then form the interpolated latent xt = (1 − t)x0 + tx1 and regress the target velocity (x1 − x0):

$${\mathcal{L}}_{\mathrm{diff}}=\left\|v_{\theta}(x_{t}^{(i)},t^{(i)},c^{(i)})-(x_{1}^{(i)}-x_{0}^{(i)})\right\|_{2}^{2},$$
, (3)
where c
(i) denotes the conditioning signal.

Ground-truth-Aligned ID Loss. Since ArcFace embedding requires landmark detection and alignment, directly extracting landmarks from Igen is unreliable because generated images are obtained through noisy diffusion or one-step denoising. Prior methods compromise: PortraitBooth (Peng et al., 2024) applies the loss only at low noise levels (t < 0.25), discarding supervision at higher noise, while PuLID (Guo et al., 2024) fully denoises generated results at significant computational cost. In contrast, we align the generated image using GT landmarks, thereby avoiding noisy landmark extraction. We minimize the cosine distance between GT-aligned ArcFace embeddings of the generated and ground-truth (GT) faces:
LID = 1 − cos(g, t) (4)
where g and t are ArcFace embeddings of the generated and GT images. This design (1) enables applying the ID loss across all noise levels, (2) incurs negligible overhead throughout training, and (3) implicitly supervises generated landmarks. Ablation studies (Sec. 6.3) demonstrate more accurate

$$({\mathfrak{I}})$$

identity measurement and substantially improved identity preservation. Further explaination and notations are provided in Appendix E.1. ID Contrastive Loss With Extended Negatives. To further strengthen identity preservation, we introduce an ID contrastive loss that explicitly pulls the generated image closer to its reference images in the face embedding space while pushing it away from other identities. The loss follows the InfoNCE (Oord et al., 2018) formulation:

$${\cal L}_{\rm CL}=-\log\frac{\exp(\cos({\bf g},{\bf t})/\tau)}{\sum_{j=1}^{M}\exp(\cos({\bf g},{\bf n}_{j}))/\tau)},\tag{5}$$
(6) $\frac{}{}$
where t is the embedding of the target, nj are embeddings of M negatives from different identities, and τ is a temperature hyperparameter. This formulation relies on ID-labeled datasets, which make it possible to draw thousands of negatives per sample from the reference bank, thereby greatly enriching the diversity of negative examples. The overall training objective is a weighted sum of the above losses:
L = Ldiff + λIDLID + λCLLCL, (6)
where λID and λCL are hyper-parameters controlling the contributions of the ID loss and contrastive loss, respectively. Both are set to 0.1 across all training phases described below.

## 5.2 Training Pipeline

Copy–paste artifacts largely arise from reconstruction-only training, which encourages models to replicate the reference image rather than learn robust identity-conditioned generation. Leveraging our paired dataset, we employ a four-phase training pipeline that gradually transitions the objective from reconstruction toward controllable, identity-preserving synthesis. Phase 1: Reconstruction pre-training with fixed prompt. We begin with reconstruction pretraining to initialize the backbone, as this task is simpler than full identity-conditioned generation and can exploit large-scale unlabeled data. For the first few thousand steps, the caption is fixed to a constant dummy prompt (e.g., "two people"), ensuring the model prioritizes learning the identityconditioning pathway rather than drifting toward text-conditioned styling. The full MultiID-2M is used in this phase, which typically lasts for 20k steps, at which point the model achieves satisfactory identity similarity. To further enhance data diversity, CelebA-HQ (Karras et al., 2017), FFHQ (Karras et al., 2019), and a subset of FaceID-6M (Wang et al., 2025) are also incorporated. Phase 2: Reconstruction pre-training with full captions. This phase aligns identity learning with text-conditioned generation and lasts for an additional 40k steps, during which the model reaches peak identity similarity. Table 1: **Quantitative comparison on the single-person subset of MultiID-Bench and OmniContext**. , ,
and indicate the first-, second-, and third-best performance, respectively. For Copy-Paste ranking, only cases with Sim(GT) > 0.40 are considered.

a **MultiID-Bench**

MethodIdentity Metrics **Generation Quality**

Sim(GT) ↑ Sim(Ref) ↑ CP ↓ CLIP-I ↑ CLIP-T ↑ Aes ↑

DreamO 0.454 0.694 0.303 0.793 0.322 4.877 OmniGen 0.398 0.602 0.248 0.780 0.317 5.069 OmniGen2 0.365 0.475 0.142 0.787 0.331 4.991 FLUX.1 Kontext 0.324 0.408 0.099 0.755 0.327 5.319 Qwen-Image-Edit 0.324 0.409 0.093 0.776 0.316 5.056 GPT-4o Native 0.425 0.579 0.178 0.794 0.311 5.344 UNO 0.304 0.428 0.141 0.765 0.314 4.923 USO 0.401 0.635 0.286 0.790 0.329 5.077 UMO 0.458 0.732 0.359 0.783 0.305 4.850 UniPortrait 0.447 0.677 0.265 0.793 0.319 5.018 ID-Patch 0.426 0.633 0.231 0.792 0.312 4.900 InfU 0.439 0.630 0.233 0.772 0.328 5.359 PuLID 0.452 0.705 0.315 0.779 0.305 4.839 InstantID 0.464 0.734 0.337 0.764 0.295 5.255 Ours 0.460 0.578 0.144 0.798 0.313 4.783 GT 1.000 0.521 -0.999 N/A N/A N/A Ref 0.521 1.000 0.999 N/A N/A N/A

MethodQuality Metrics **Overall**

PF ↑ SC ↑ Overall ↑

DreamO 8.13 7.09 7.02 OmniGen 7.50 5.52 5.47 OmniGen2 8.64 8.50 8.34 FLUX.1 Kontext 7.72 8.60 7.94 Qwen-Image-Edit 7.66 8.16 7.51 GPT-4o Native 7.98 9.06 8.12 UNO 7.22 7.72 7.04 USO 6.96 7.88 6.70 UMO 6.56 7.92 6.79

UniPortrait 6.62 6.00 5.55

ID-Patch N/A N/A N/A InfU 7.69 4.62 4.70

PuLID 6.62 6.83 5.78

InstantID 4.89 5.49 4.35

Ours 7.43 7.04 6.52

Phase 3: Paired tuning. To suppress trivial copy–paste behavior, we replace 50% of the training samples with paired instances drawn from the 500k labeled images in MultiID-2M. For each paired

![6_image_0.png](6_image_0.png) 
sample, instead of using the same image as both input and target, we randomly select one reference image from the identity's reference set and another distinct image of the same identity as the target. This perturbation breaks the shortcut of direct duplication and compels the model to rely on high-level identity embeddings rather than low-level copying. Phase 4: Quality tuning. Finally, we fine-tune on a curated high-quality subset augmented with generated stylized variants to (i) enhance perceptual fidelity and (ii) improve style robustness and transferability. This phase refines texture, lighting, and stylistic adaptability while preserving the strong identity consistency established in earlier phases.

## 6 Experiments

In this section, we present a comprehensive evaluation of baselines and our WithAnyone model on the proposed MultiID-Bench. Baselines. We evaluate two categories of baseline methods: general customization models and face customization methods. The general customization models include OmniGen (Xiao et al., 2024), OmniGen2 (Wu et al., 2025b), Qwen-Image-Edit (Wu et al., 2025a), FLUX.1 Kontext (Batifol et al., 2025), UNO (Wu et al., 2025d), USO (Wu et al., 2025c), UMO (Cheng et al., 2025), and native GPT-4o-Image (OpenAI, 2025). The face customization methods include UniPortrait (He et al., 2024), ID-Patch (Zhang et al., 2025), PuLID (Guo et al., 2024) (referring to its FLUX (Labs, 2024) implementation throughout this paper), and InstantID (Wang et al., 2024c). All models were evaluated on the single-person subset of the benchmark, while only those supporting multi-ID generation were additionally tested on the multi-person subset. Further implementation details are provided in Appendix F.1. Table 2: **Quantitative comparison on the multi-person subset of MultiID-Bench**. , , and indicate the first-, second-, and third-best performance, respectively. For Copy-Paste ranking, only cases with Sim(GT) > 0.35 are considered. GPT exhibits prior knowledge of identities from TV series in subsets with more than two IDs, leading to abnormally high similarity scores.

a **2-people Subset**

## 6.1 Quantitative Evaluation

The quantitative results are reported in Tables 1 and 2. We observe a clear trade-off between face similarity and copy-paste artifacts. As shown in Fig. 5, most methods align closely with a regression curve, where higher face similarity generally coincides with stronger copy-paste. This indicates that many existing models boost measured similarity by directly replicating reference facial features rather than synthesizing the identity. In contrast, WithAnyone deviates substantially from this curve,

![7_image_0.png](7_image_0.png)

![7_image_1.png](7_image_1.png)

ground-truth image shown on the leftmost side.
achieving the highest face similarity with regard to GT while maintaining a markedly lower copy-paste score. WithAnyone also achieves the highest score among ID-specific reference models on the OmniContext (Wu et al., 2025b) benchmark. However, VLMs (Bai et al., 2025; OpenAI, 2025) exhibit limited ability to distinguish individual identities and instead emphasize non-identity attributes such as pose, expression, or background. Despite that general customization and editing models often outperform face customization models on OmniContext, WithAnyone still has best performance among face customization models.

## 6.2 Qualitative Comparison

To complement the quantitative results, Fig. 6 presents qualitative comparisons between our method, state-of-the-art general customization/editing models, and face customization generation models.

| Ablation          | Identity Metrics   | Generation Quality   |       |       |       |       |       |
|-------------------|--------------------|----------------------|-------|-------|-------|-------|-------|
| Sim(G) ↑ Sim(R) ↑ | CP ↓               | CLIP-I ↑ CLIP-T ↑    | Aes ↑ |       |       |       |       |
| Phases            | w/o Phase 3        | 0.406                | 0.625 | 0.239 | 0.755 | 0.307 | 4.955 |
| Loss              | w/o GT-Align       | 0.385                | 0.549 | 0.175 | 0.763 | 0.317 | 4.754 |
| w/o Ext. Neg.     | 0.368              | 0.455                | 0.074 | 0.740 | 0.304 | 4.984 |       |
| Data              | FFHQ only          | 0.224                | 0.246 | 0.027 | 0.658 | 0.330 | 5.039 |
| Ours              | Full Setting       | 0.405                | 0.551 | 0.161 | 0.770 | 0.321 | 4.883 |

![8_image_0.png](8_image_0.png)

![8_image_1.png](8_image_1.png)

It shows that identity consistency remains a significant weakness of general customization or editing models, consistent with our quantitative findings. Many VAE-based approaches—where references are encoded through a VAE, such as FLUX.1 Kontext and DreamO—tend to produce faces that either exhibit copy-paste artifacts or deviate markedly from the target identity. A likely reason is that VAE embeddings emphasize low-level features, leaving high-level semantic understanding to the diffusion backbone, which may not have been pre-trained for this task. ID-specific reference models also struggle with copy-paste artifacts. For example, they fail to make the subject smile when the reference image is neutral and often cannot adjust head pose or even eye gaze. In contrast, WithAnyone generates flexible, controllable faces while faithfully preserving identity.

## 6.3 Ablation And User Studies

![8_image_2.png](8_image_2.png)

To better understand the contribution of each component in WithAnyone, we conduct ablation studies on the training strategy, the GT-aligned ID loss, the InfoNCE-based ID loss, and our dataset. Due to space constraints, we report the key results here, with additional analyses provided in Appendix G. As shown in Table 3, the paired-data fine-tuning phase reduces copy-paste artifacts without diminishing similarity to the ground truth, while training on FFHQ performs significantly worse than on our curated dataset. Fig. 7 further demonstrates that the GT-aligned ID loss lowers denoising error at low noise levels and yields higher-variance, more informative gradients at high noise, thereby strengthening identity learning. By ablating extended negatives, leaving only 63 negative samples from the batch (originally extended to 4096), the effectiveness of ID contrastive loss is greatly reduced. More ablation results can be found in Appendix G. We conduct a user study to evaluate perceptual quality and identity preservation. Ten participants were recruited and asked to rank 230 groups of generated images according to four criteria: identity similarity, presence of copy-paste artifacts, prompt adherence, and aesthetics. The results, shown in Fig. 8, indicate that our method consistently achieves the highest average ranking across all dimensions, demonstrating both stronger identity preservation and superior visual quality. Moreover, the copy-paste metric exhibits a moderate positive correlation with human judgments, suggesting that it captures perceptually meaningful artifacts. Further details of the study design, ranking protocol, and statistical analysis are provided in Appendix H.

Figure 8: **User study.** Bigger bubbles indicate higher ranking and better performance.

## 7 Conclusion

Copy-paste artifacts are a common limitation of identity customization methods, and face-similarity metrics often exacerbate the issue by implicitly rewarding direct copying. In this work, we identify and formally quantify this failure mode through MultiID-Bench, and propose targeted solutions. We curate MultiID-2M and develop training strategies and loss functions that explicitly discourage trivial replication. Empirical evaluations demonstrate that WithAnyone significantly reduces copy-paste artifacts while maintaining—and in many cases improving—identity similarity, thereby breaking the long-standing trade-off between fidelity and copying. These results highlight a practical path toward more faithful, controllable, and robust identity customization.

## Ethics Statement And Disclaimer

Data Source. The images used in this work were collected exclusively from publicly accessible sources through search engines that provide explicit filtering based on Creative Commons (CC) licensing. Our dataset focuses on publicly known figures, and we restricted data collection to images released under CC licenses that explicitly permit reuse and derivative works (e.g., CC-BY, CC- BY-SA, or CC0 where applicable). We excluded content with restrictive terms such as "NoAI", non-derivative, or unclear licensing conditions. No private datasets, login-restricted sources, or personally sensitive images were used. These license-based permissions provide authorization consistent with creator-defined reuse terms, and we follow applicable copyright and data-usage regulations to ensure responsible research practice. Anonymization. To further mitigate privacy and misuse risks, we applied an anonymization procedure to the dataset processing and training pipeline. No personal names, textual identifiers, or explicit identity labels were included during training. All individuals are represented solely through internal numeric identifiers and corresponding ID embeddings, without any direct linkage to real-world names or metadata. The model therefore operates on abstract identity representations rather than explicit personal information, reducing the risk of unintended identity disclosure. Potential Ethical Risks and Mitigation. Identity-consistent image generation is inherently dualuse. While enabling legitimate applications such as creative media and virtual avatars under proper authorization, it may also facilitate identity cloning, impersonation, misattribution, or deceptive synthetic media—particularly if applied without consent. To mitigate these risks, the models are released strictly for non-commercial academic research; training data is limited to publicly known figures under reuse-permitted licenses; and no personal names or explicit identity labels are used in training. We further recommend that downstream deployments implement consent verification, authorization controls, disclosure or watermarking mechanisms, and abuse monitoring. Responsible use must comply with applicable legal, institutional, and ethical standards.

Disclaimer. The *WithAnyone* models and associated datasets (the "Project") are provided solely for research and non-commercial use under the FLUX.1 [dev] Non-Commercial License v1.1.1. All base models and third-party components remain subject to their original licenses. Any underlying content derived from publicly available sources remains the property of its respective rights holders, and no ownership, endorsement, or additional rights are claimed or granted. The Project is provided
"as is" without warranties of any kind, express or implied. Users are solely responsible for ensuring compliance with all applicable laws, regulations, and third-party rights. The providers of this Project shall not be liable for any claims, damages, or losses arising from its use. Under no circumstances shall the authors or the affliated organization be liable for any claims, damages, losses, or other liabilities arising from or related to the use of the Dataset.

## References

Moab Arar, Rinon Gal, Yuval Atzmon, Gal Chechik, Daniel Cohen-Or, Ariel Shamir, and Amit H. Bermano. Domain-agnostic tuning-encoder for fast personalization of text-to-image models. In SIGGRAPH Asia 2023 Conference Papers, pp. 1–10, 2023.

Moab Arar, Andrey Voynov, Amir Hertz, Omri Avrahami, Shlomi Fruchter, Yael Pritch, Daniel Cohen-Or, and Ariel Shamir. Palp: prompt aligned personalization of text-to-image models. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–11, 2024.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pp. arXiv–2506, 2025.

Bowen Chen, Mengyi Zhao, Haomiao Sun, Li Chen, Xu Wang, Kang Du, and Xinglong Wu. Xverse:
Consistent multi-subject control of identity and semantic attributes via dit modulation. arXiv preprint arXiv:2506.21416, 2025.

Weifeng Chen, Jiacheng Zhang, Jie Wu, Hefeng Wu, Xuefeng Xiao, and Liang Lin. Id-aligner:
Enhancing identity-preserving text-to-image generation with reward feedback learning. arXiv preprint arXiv:2404.15449, 2024.

Yufeng Cheng, Wenxu Wu, Shaojin Wu, Mengqi Huang, Fei Ding, and Qian He. Umo: Scaling multi-identity consistency for image customization via matching reward. arXiv preprint arXiv:2509.06818, 2025.

Jiaming Chu, Lei Jin, Yinglei Teng, Jianshu Li, Yunchao Wei, Zheng Wang, Junliang Xing, Shuicheng Yan, and Jian Zhao. Uniparser: Multi-human parsing with unified correlation representation learning. T-IP, 2024.

Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4690–4699, 2019.

Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. Retinaface:
Single-shot multi-level face localisation in the wild. In CVPR, 2020.

discus0434. aesthetic-predictor-v2-5. https://github.com/discus0434/
aesthetic-predictor-v2-5, 2023. Accessed: 2025-05-12.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, Peng Zhang, and Qian He. Pulid: Pure and lightning id customization via contrastive alignment. In Advances in Neural Information Processing Systems, 2024.

Junjie He, Yifeng Geng, and Liefeng Bo. Uniportrait: A unified framework for identity-preserving single-and multi-human image personalization. arXiv preprint arXiv:2408.05939, 2024.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Promptto-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Xirui Hu, Jiahao Wang, Hao Chen, Weizhan Zhang, Benqi Wang, Yikun Li, and Haishun Nan.

Dynamicid: Zero-shot multi-id image personalization with flexible facial editability. arXiv preprint arXiv:2503.06505, 2025a.

Yuqi Hu, Longguang Wang, Xian Liu, Ling-Hao Chen, Yuwei Guo, Yukai Shi, Ce Liu, Anyi Rao, Zeyu Wang, and Hui Xiong. Simulating the real world: A unified survey of multimodal generative models. arXiv preprint arXiv:2503.04641, 2025b.

Ziqi Huang, Tianxing Wu, Yuming Jiang, Kelvin CK Chan, and Ziwei Liu. Reversion: Diffusionbased relation inversion from images. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–11, 2024.

Junha Hyung, Jaeyo Shin, and Jaegul Choo. Magicapture: High-resolution multi-concept portrait customization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 2445–2453, 2024.

Liming Jiang, Qing Yan, Yumin Jia, Zichuan Liu, Hao Kang, and Xin Lu. Infiniteyou: Flexible photo recrafting while preserving your identity. arXiv preprint arXiv:2503.16418, 2025a.

Qing Jiang, Lin Wu, Zhaoyang Zeng, Tianhe Ren, Yuda Xiong, Yihao Chen, Qin Liu, and Lei Zhang.

Referring to any person, 2025b. URL https://arxiv.org/abs/2503.08507.

Maxwell Jones, Sheng-Yu Wang, Nupur Kumari, David Bau, and Jun-Yan Zhu. Customizing text-toimage models with a single image pair. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–13, 2024.

Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196, 2017.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4401–4410, 2019.

Chanran Kim, Jeongin Lee, Shichang Joung, Bongmo Kim, and Yeul-Min Baek. Instantfamily:
Masked attention for zero-shot multi-id image generation. arXiv preprint arXiv:2404.19427, 2024.

Minchul Kim, Anil K Jain, and Xiaoming Liu. Adaface: Quality adaptive margin for face recognition.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp.

18750–18759, 2022.

Nupur Kumari, Grace Su, Richard Zhang, Taesung Park, Eli Shechtman, and Jun-Yan Zhu. Customizing text-to-image diffusion with object viewpoint control. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–13, 2024.

Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

Black Forest Labs. Flux.1 krea. https://huggingface.co/black-forest-labs/FLUX.

1-Krea-dev, 2025.

Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. Photomaker:
Customizing realistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8640–8650, 2024.

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.

Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, et al. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–12, 2024.

Chong Mou, Yanze Wu, Wenxu Wu, Zinan Guo, Pengze Zhang, Yufeng Cheng, Yiming Luo, Fei Ding, Shiwen Zhang, Xinghui Li, et al. Dreamo: A unified framework for image customization. arXiv preprint arXiv:2504.16915, 2025.