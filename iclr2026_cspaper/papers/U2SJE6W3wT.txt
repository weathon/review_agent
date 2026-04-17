# Improved Adversarial Diffusion Compression For Real-World Video Super-Resolution

Bin Chen12∗, Weiqi Li12∗, Shijie Zhao2⋄, Xuanyu Zhang12, Junlin Li2, Li Zhang2**, Jian Zhang**13†
1School of Electronic and Computer Engineering, Peking University 2ByteDance Inc.

3Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology, Shenzhen Graduate School, Peking University

## Abstract

While many diffusion models have achieved impressive results in real-world video super-resolution (Real-VSR) by generating rich and realistic details, their reliance on multi-step sampling leads to slow inference. One-step networks like SeedVR2, DOVE, and DLoRAL alleviate this through condensing generation into one single step, yet they remain heavy, with billions of parameters and multi-second latency. Recent adversarial diffusion compression (ADC) offers a promising path via pruning and distilling these models into a compact AdcSR network, but directly applying it to Real-VSR fails to balance spatial details and temporal consistency due to its lack of temporal awareness and the limitations of standard adversarial learning. To address these challenges, we propose an improved ADC method for Real-VSR. Our approach distills a large diffusion Transformer (DiT) teacher DOVE equipped with 3D spatio-temporal attentions, into a pruned 2D Stable Diffusion (SD)-based AdcSR backbone, augmented with lightweight 1D temporal convolutions, achieving significantly higher efficiency. In addition, we introduce a dual-head adversarial distillation scheme, in which discriminators in both pixel and feature domains explicitly disentangle the discrimination of details and consistency into two heads, enabling both objectives to be effectively optimized without sacrificing one for the other. Experiments demonstrate that the resulting compressed **AdcVSR** model reduces complexity by 95% in parameters and achieves an 8× acceleration over its DiT teacher DOVE, while maintaining competitive video quality and efficiency.

## 1 Introduction

Real-world video super-resolution (Real-VSR) (Tao et al., 2017; Nah et al., 2019) is a fundamental and long-standing problem in computer vision. It targets at recovering high-resolution (HR) videos xHR from their low-resolution (LR) counterparts xLR degraded by unknown factors in real-world cases. Traditional non-generative (Yi et al., 2019; Chan et al., 2021; Yang et al., 2021; Chan et al., 2022b;a; Wang et al., 2019; Liang et al., 2024) and generative adversarial network (GAN)-based (Chu et al., 2020; Lucas et al., 2019; Xu et al., 2025) approaches have achieved notable progress, yet most of them struggle under complex, mixed degradations, producing over-smoothed results or artifacts. To enhance the detail richness of super-resolution outputs xSR, many Real-VSR studies (Zhou et al., 2024; Yang et al., 2024a; He et al., 2024; Wang et al., 2025b; Xie et al., 2025b;a; Kong et al., 2025; Wang et al., 2025c;f; Zhao et al., 2025; Wang et al., 2025e; Bai et al., 2025) have already developed diffusion-based approaches which can generate video frames with richer and more realistic details. However, these methods are hindered by long inference time, as they need multiple sampling steps.

∗Equal contribution. ⋄Project lead. †Corresponding author: zhangjian.sz@pku.edu.cn. This work was financially supported in part by National Natural Science Foundation of China (62372016), Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology (2024B1212010006), Shenzhen Science and Technology Program (SYSPG20241211173440004), and Outstanding Talents Training Fund in Shenzhen.

![0_image_0.png](0_image_0.png)

Figure 1: Comparison of methods in compressing diffusion networks for Real-VSR. (a) Traditional ADC (Chen et al., 2025a) distills an SD network with 2D spatial attentions into a pruned student using a single adversarial signal without temporal modeling, suffering from frame flickering. (b) Our improved ADC distills a larger DiT-based teacher with heavier 3D spatio-temporal attention into the same 2D student, augmented by 1D temporal convolutions, using dual-head discriminators D in pixel and feature domains. Through disentangling the discriminations of detail richness and temporal consistency into different heads, it balances the optimization of both.

1 Recently, researchers have shifted focus to one-step diffusion (Wang et al., 2024a; Wu et al., 2024; Xie et al., 2024; Lin et al., 2025c) for achieving efficient and high-quality Real-VSR. Building on the pretrained Stable Diffusion (SD) models (Rombach et al., 2022), originally developed for image generation, UltraVSR (Liu et al., 2025) enhances the temporal consistency in xSR by propagating and fusing features along the temporal dimension, while DLoRAL (Sun et al., 2025) aligns the structure of the previous frame with the current one using the estimated optical flow between them. Building on video diffusion models, SeedVR2 (Wang et al., 2025a) progressively distills a pretrained 64-step SeedVR (Wang et al., 2025b) into one sampling step and further enhances it through adversarial posttraining. DOVE (Chen et al., 2025b) adapts pretrained CogVideoX networks (Yang et al., 2024b) to Real-VSR by fine-tuning them on a curated high-quality video dataset. Despite these advances, such approaches still suffer from high complexity due to large-scale parameters and heavy computation. Meanwhile, two recent methods, AdcSR (Chen et al., 2025a) and TinySR (Dong et al., 2025b), have explored compressing the diffusion networks OSEDiff (Wu et al., 2024) and TSD-SR (Dong et al., 2025a) via structural pruning and adversarial distillation to reduce complexity for real-world image super-resolution (Real-ISR), as shown in Fig. 1 (a). However, it is non-trivial to extend them to Real- VSR. One may use them to compress Real-VSR networks such as SeedVR2, DOVE, and DLoRAL, but two challenges arise: the compressed models are still too large, or video quality is compromised due to the conflicts between optimizing detail richness and temporal consistency (Sun et al., 2025),
which are difficult to balance with existing distillation methods. Therefore, it remains underexplored and of great value to design more effective compression approaches for diffusion-based Real-VSR. To improve effectiveness, in this paper, we propose **AdcVSR**, a novel Real-VSR network that compresses the one-step model DOVE using an improved method of Adversarial Diffusion Compression (ADC) (Chen et al., 2025a). Unlike previous networks that rely on computationally costly 3D spatiotemporal attentions or additional frame alignment modules, we hypothesize that a 2D diffusion backbone (*e.g.*, SD2.1) is sufficient to generate rich details, while temporal consistency can be maintained with a few lightweight 1D temporal convolutional layers, and that their combination is also effective in removing degradations. Guided by this insight, as Fig. 1 (b) illustrates, AdcVSR adopts the same pruned SD2.1 backbone as AdcSR, augmented with 1D temporal convolution layers, achieving significantly lower complexity than its heavy 3D teacher DOVE. To improve video quality and address the conflict between optimizing details and consistency (Sun et al., 2025), we introduce a new adversarial distillation scheme that leverages the strong teacher DOVE together with numerous temporally consistent real videos and detail-rich real images. Specifically, two discriminators are employed for adversarial learning: one discriminating in a feature space of variational autoencoder (VAE) decoder and the other in pixel space, each with a "detail" head and a "consistency" head sharing a common backbone, to separately assess the realism of spatial details and temporal consistency. This enables the student to generate super-resolution results which are both detail-rich and temporally consistent. By integrating our "2D + 1D" architectural design with the dual-head, dual-discriminator adversarial distillation scheme, AdcVSR effectively compresses DOVE, yielding substantial efficiency gains while maintaining competitive video quality. Our contributions are summarized as follows: ❑
 (1) We propose a novel improved ADC approach that combines an effective network design with adversarial distillation to compress a heavy Real-VSR model into an efficient diffusion-GAN hybrid. ❑
 (2) We demonstrate that a 2D image diffusion backbone augmented with lightweight 1D temporal convolutions can effectively learn Real-VSR mapping from 3D diffusion Transformer (DiT) teacher. ❑
 (3) We introduce a new adversarial distillation scheme that decouples the discriminations of details and consistency into two heads sharing a common network backbone, applied in both VAE decoder's feature space and the pixel space. This design enables balanced optimization, avoiding collapse into either over-smoothed outputs (loss of spatial details) or flickering (loss of temporal consistency). ❑
 (4) Extensive experiments show that our AdcVSR model reduces parameters by 95% and achieves an 8× acceleration over its teacher DOVE, while maintaining competitive performance on Real-VSR and striking a balance among fidelity, detail richness, temporal consistency, and model efficiency.

## 2 Related Work

Real-VSR. A main challenge in this field lies in modeling the diverse, complex degradations of LR inputs, which can usually not be well represented by the bicubic downsampling (Mou et al., 2024; Hu et al., 2023; Wang et al., 2022; Jiang et al., 2025; Li et al., 2025b). To address this, two strategies for collecting LR-HR video pairs to train deep Real-VSR networks have been developed: one captures pairs using different camera settings (Yang et al., 2021; Wang et al., 2023b), while the other synthesizes LR inputs from HR videos by composing degradations including noise, blur, resizing, and image/video coding compression (*e.g.*, JPEG, H.264, and MPEG-4) in random, high-order processes (Wang et al., 2021; Zhang et al., 2021; Chan et al., 2022b). These approaches enrich the degradation space and can synthesize large amounts of training data. Building upon these, a number of expressive Real-VSR networks have been developed (Shi et al., 2022). Non-generative methods like BasicVSR (Chan et al., 2021; 2022a), EDVR (Wang et al., 2019), and RVRT (Liang et al., 2024; 2022) excel at distortion removal, but often yield over-smoothed results under severe degradations. GAN-based methods including TecoGAN (Chu et al., 2020) and VideoGigaGAN (Xu et al., 2025) enhance visual quality with sharper details, but could introduce visible artifacts. Recently, diffusionbased methods have shown stronger performance by generating more realistic videos. For instance, Upscale-A-Video (Zhou et al., 2024), VEnhancer (He et al., 2024), STAR (Xie et al., 2025b), RealisVSR (Zhao et al., 2025), SeTe-VSR (Wang et al., 2025e), and Vivid-VR (Bai et al., 2025) integrate transform/control modules into diffusion backbones to exploit the LR inputs as a condition. MGLD- VSR (Yang et al., 2024a) enforces temporal consistency via motion guidance, SeedVR (Wang et al., 2025b) introduces shifted window-based DiTs to handle varying spatial sizes, LiftVSR (Wang et al.,
2025c) adopts attentions and memory for modeling short- and long-term dependencies, SimpleGVR
(Xie et al., 2025a) performs cascaded latent-domain upscaling, and DiffVSR (Li et al., 2025c) decomposes Real-VSR learning in three progressive stages. Although these approaches improve detail richness, their reliance on multi-step sampling makes inference slow and computationally expensive. One-Step Diffusion. Reducing step number while keeping output quality is a widely adopted strategy to accelerate inference (Wang et al., 2025f). In the context of Real-ISR and Real-VSR, several works have pushed this idea to the extreme by compressing multi-step generation into a single step (Yue et al., 2025; Gong et al., 2025; Wang et al., 2025d). For example, SinSR (Wang et al., 2024a) and SeedVR2 (Wang et al., 2025a) distill a 15-step ResShift (Yue et al., 2024) and 64-step SeedVR (Wang et al., 2025b) into one step through bidirectional or progressive distillation (Salimans & Ho, 2022). OSEDiff (Wu et al., 2024), S3Diff (Zhang et al., 2024a), D3SR (Li et al., 2024), and HYPIR (Lin et al., 2025c) adopt variational score distillation (Wang et al., 2024b), degradation-guided lowrank adaptations (LoRAs) (Hu et al., 2022), and adversarial post-training (Lin et al., 2025a;b), enabling one-step Real-ISR with high image quality. UltraVSR (Liu et al., 2025) aggregates temporal features via recurrent shifts, while PiSA-SR (Sun et al., 2024) and DLoRAL (Sun et al., 2025) introduce residual one-step diffusion and dual-LoRA learning, to alternately optimize spatial details and temporal consistency. Leveraging large-scale pretrained DiT-based text-to-image/-video (T2I/T2V) generation models, FluxSR (Li et al., 2025a), DiT4SR (Duan et al., 2025), and DOVE (Chen et al., 2025b) fine-tune FLUX (Labs, 2024), SD3 (Esser et al., 2024), and CogVideoX (Yang et al., 2024b) under the flow matching framework (Lipman et al., 2022) to enhance fine structures and text regions. To further reduce complexity, AdcSR (Chen et al., 2025a), PassionSR (Zhu et al., 2025), and TinySR (Dong et al., 2025b) compress one-step Real-ISR networks using pruning, weight quantization, and distillation. However, these techniques are tailored for Real-ISR and struggle in Real-VSR, as they do not account for consistency. As a result, applying them directly would compromise video quality. To mitigate this problem, we propose to learn a diffusion network **AdcVSR** for Real-VSR, based on our improved ADC method that compresses the large DOVE teacher by distilling it into a pruned 2D SD2.1 backbone, augmented with lightweight 1D temporal convolutions. Additionally, we develop a new dual-head, dual-discriminator adversarial distillation scheme with decoupled detail-consistency discrimination, enabling the student network to achieve competitive video quality and efficiency.

## 3 Method 3.1 Preliminary

Conflict in Optimizing Details and Consistency. Detail enrichment in video outputs requires synthesizing fine-grained structures like textures and edges with significant pixel-level variations, while temporal consistency demands constraining these variations across frames to ensure visually pleasant transitions and suppress flickering. These objectives are empirically found to be in conflict (Chu et al., 2020; Li et al., 2025c; Xu et al., 2025): many generative models emphasizing perceptual qual-

![3_image_0.png](3_image_0.png)

ity tend to prioritize details, leading to visible flickers, while propagation or alignment mechanisms for consistency may over-smooth or attenuate details. Recent work (Sun et al., 2025) also highlights that details and consistency are competing objectives, where optimizing one could degrade the other. AdcSR, and Current Methods' Limitations. AdcSR (Chen et al., 2025a) demonstrates that onestep Real-ISR diffusion network (Wu et al., 2024) can be compressed by an ADC method: removing VAE encoder and text components, pruning channels of denoising UNet and VAE decoder, and then applying adversarial distillation to restore outputs' quality, as shown in Fig. 1 (a). However, AdcSR is designed for Real-ISR and does not account for temporal modeling. When applied frame by frame to videos, it inevitably introduces flickers (Zhou et al., 2024; Rota et al., 2024). One-step Real-VSR networks (Liu et al., 2025; Wang et al., 2025a; Sun et al., 2025; Chen et al., 2025b) are still heavy, with ≥1.3B parameters and ≥4s latency even for a 25-frame 512 × 512 video (Fig. 4). An intuitive idea is to compress them by combining ADC, but existing learning approaches like dual-LoRA (Sun et al., 2024; 2025), adversarial/score-based distillation (Xu et al., 2025; Liu et al., 2025; Chen et al., 2025a) are ineffective under aggressive pruning, failing to resolve the detail-consistency conflict.

## 3.2 Network Architecture Design

To compress large Real-VSR diffusion networks while balancing quality and efficiency, we propose an improved ADC approach which combines an effective architecture with an adversarial distillation scheme. Our key intuition is that although 3D spatio-temporal DiTs (Wang et al., 2025a; Chen et al., 2025b) achieve impressive results, their attention mechanisms aim to capture long-range space-time dependencies, which are important for T2V generation (Yang et al., 2024b; Wan et al., 2025), where such global information must be inferred from scratch. In Real-VSR, however, the LR video already provides much of this information (Sun et al., 2025), e.g., structural layout and temporal continuity. Its main objectives are: (1) synthesizing details and (2) ensuring that they are temporally consistent to prevent flickering. In this setting, heavy 3D attentions might introduce redundancy, as much of its capacity is devoted to generating global spatio-temporal structures that are already present in xLR.

Building on this insight, we hypothesize that (1) a 2D diffusion backbone is capable of synthesizing details, and (2) consistency can be maintained with several 1D temporal convolutions. The rationale behind the second point is that maintaining consistency is inherently less challenging than synthesizing details: the objective is to constrain pixel-level variations across consecutive frames, rather than generate new fine structures. By enabling adjacent frames to be processed with temporal awareness, these convolutions are hypothesized to be sufficient to suppress flickering and yield temporally consistent recoveries. This motivates a principled "2D + 1D" network design that is expressive enough to learn the powerful Real-VSR mappings of large 3D DiTs while reducing redundant overhead. Specifically, as Fig. 2 (a, bottom) exhibits, we adopt AdcSR (Chen et al., 2025a) as the 2D backbone, composed of channel-pruned SD2.1 UNet and VAE decoder. To add temporal awareness, we insert residual blocks after each UNet block, each consisting of a 1D temporal convolution, a ReLU activation, and a second convolution with a skip connection. Unlike flow- and motion-guided approaches (Zhou et al., 2024; Yang et al., 2024a) or alignment/propagation strategies (Liu et al., 2025; Xu et al., 2025; Sun et al., 2025), this simple yet effective design equips the resulting network, **AdcVSR**, with temporal modeling capacity, while keeping network architecture compact and inference efficient.

## 3.3 Adversarial Distillation Scheme

To achieve competitive video quality, we distill the large pretrained 3D DiT model DOVE (teacher) into our "2D + 1D" network (student). Specifically, as illustrated in Fig. 2 (a, top), we adopt DOVE's outputs as the learning target and conduct distillation in two domains: the pixel domain and the feature domain of the AdcSR VAE decoder's middle block. In the latter, DOVE's output pixels xteacher are re-encoded by SD2.1 VAE encoder and fed into the middle block to obtain aligned features fteacher for supervision, using the regression losses ∥xstudent−xteacher∥1 and ∥fstudent−fteacher∥1, where xstudent and fstudent denote the student's corresponding outputs. Compared with the original ADC framework (Chen et al., 2025a), which distills only in a single feature domain corresponding to fstudent and fteacher
with the remaining decoder blocks frozen, our method uses richer supervisory signals and fine-tunes the entire network end-to-end, thereby activating its full capacity to learn Real-VSR mappings better. However, although minimizing pixel and feature errors provides strong supervision, it is insufficient because our "2D + 1D" AdcVSR is significantly smaller and architecturally different from 3D DiT teacher, making exact fitting impractical while causing optimization difficulties and degraded reconstructions. Unlike the original setup of ADC, where the student was a streamlined variant of teacher with closer capacity and simple error minimization could still be effective, our case is far more challenging due to much larger gaps in both architecture design and parameter scale. We therefore retain
error-minimizing distillations as a foundation, but augment them using adversarial learning to relax
the requirement of exact replication, enabling the student to benefit from guidance of teacher DOVE while enjoying the flexibility to generate outputs that are feasible for its capacity and of high quality. To achieve this, a straightforward approach would introduce a standard discriminator that adversarially aligns output distributions with real data (*e.g.*, xHR) (Sauer et al., 2023; 2024; Lin et al., 2024).
However, this couples the objectives of details and consistency into one single adversarial signal. In practice, the discriminator D often tends to prioritize one aspect (typically details) at the expense of
the other (typically consistency), leading to detail-rich but flickering result xSR. This reveals a fundamental issue: a traditional single-head discriminator entangles these conflicting objectives, yielding gradient that can not balance both. To overcome this, we propose a dual-head and dual-discriminator scheme that disentangles the assessments of details and consistency. Concretely, as Fig. 2 (b) shows,
one discriminator operates in the pixel domain on xstudent, while the other in decoder feature domain on fstudent, forming a more comprehensive dual-domain supervision than single-domain approaches.
Each discriminator is built upon a separate frozen pretrained backbone (ConvNeXt (Liu et al., 2022;
Lin et al., 2025c) for xstudent and the same augmented SD UNet as our designed AdcVSR for fstudent)
to provide strong representations and stabilize training, followed by three additional alternating 2D and 1D convolutional layers to jointly capture spatial and temporal features. Finally, each discriminator branches into two linear heads ("detail" and "consistency") that project the last-layer features into two adversarial signals for detail realism and consistency, respectively. Formally, the adversarial distillation loss for the student generator is defined as follows, where we also include a perceptual DISTS (Ding et al., 2020) loss term as in DOVE to further strengthen the pixel-domain supervision:
$$\mathcal{L}=\lambda_{\text{pixel}}\mathcal{L}_{\text{pixel}}+\lambda_{\text{feature}}\mathcal{L}_{\text{feature}},$$ $$\mathcal{L}_{\text{pixel}}=\|\mathbf{x}_{\text{student}}-\mathbf{x}_{\text{cascher}}\|_{1}+\text{DISTS}(\mathbf{x}_{\text{student}},\mathbf{x}_{\text{scatter}})+\lambda_{\text{abs}}\text{Softplus}(-\mathcal{D}_{\text{pixel}}(\mathbf{x}_{\text{student}}))\,,$$ $$\mathcal{L}_{\text{feature}}=\|\mathbf{f}_{\text{student}}-\mathbf{f}_{\text{Face}}\|_{1}+\lambda_{\text{abs}}\text{Softplus}(-\mathcal{D}_{\text{feature}}(\mathbf{f}_{\text{student}}))\,,$$
where Dpixel and Dfeature denote pixel- and feature-domain discriminators, Softplus(−Dpixel(xstudent)) and Softplus(−Dfeature(fstudent)) implement non-saturating adversarial losses (Yin et al., 2024), while λpixel, λfeature, and λadv are weights controlling the relative contributions of corresponding loss terms.
To train the dual-head discriminators, we curate five carefully designed data types with head-specific
labels that vary detail and consistency independently. First, the student's outputs (xstudent and fstudent)
are always labeled as "fake" for both heads, ensuring that adversarial feedback persistently pushes
$\left(1\right)$. 
$$(2)^{\frac{1}{2}}$$
$\left(3\right)$. 
the generator toward improvement. Second, real videos (xvideo and fvideo) are adopted and labeled as
"real" for consistency, as they preserve coherent temporal dynamics of the same underlying scene. Third, temporally shuffled versions of these videos (x
∗
video and f
∗
video) obtained via randomly permuting frame order along the temporal dimension, destroy frame-to-frame continuity, and are therefore labeled as "fake" for consistency. Fourth, we exploit detail-rich real images by repeating each one to construct static pseudo-videos (ximage and fimage), which enjoy both high-quality details and perfect temporal stability, and are thereby labeled as "real" for both heads. Finally, we randomly sample and crop real images without temporal correspondences (x
∗
image and f
∗
image). These sequences are detailrich but inherently inconsistent across frames, so they are labeled as "real" for details but "fake" for

consistency. To be formal, the losses for dual-head discriminators Dpixel and Dfeature are defined as: Ldisc =X (s,yd,yc)∈S hSoftplus(−yd[D(s)]d) + Softplus(−yc[D(s)]c) i, (4) S = n(xstudent, −1, −1),(fstudent, −1, −1),(xvideo, 0, 1),(fvideo, 0, 1),(x ∗ video, 0, −1), (f ∗ video, 0, −1),(ximage, 1, 1),(fimage, 1, 1),(x ∗ image, 1, −1),(f ∗ image, 1, −1)o. (5)
Here, [D(s)]d and [D(s)]c denote the outputs of the "detail" and "consistency" heads of discriminator D for input s, while yd, yc *∈ {−*1, 0, 1} encode "fake", "unlabeled", and "real" labels, respectively.

The set S enumerates the five curated data types and corresponding labels in both pixel and feature domains. It is worth noting that we leave real video details unlabeled, and rely on real images as the positive supervision for "detail" head, encouraging the generator to produce more detail-rich frames. In contrast to standard GAN discriminators (Goodfellow et al., 2014) which provide a single binary signal for real versus generated samples, our design restructures adversarial supervision into a multiattribute form, producing fine-grained and disentangled signals for both details and consistency, with outputs that preserve the same spatial resolution as the inputs. This approach moves beyond existing adversarial distillation methods (Sauer et al., 2023; 2024; Chen et al., 2025a) by explicitly requiring the dual-head discriminators to evaluate two aspects of video realism. As a result, neither aspect can be disregarded or down-weighted, as the two dedicated heads consistently provide supervisions, ensuring that the model receives separate weight gradients for both. This prevents AdcVSR generator from collapsing toward one objective at the expense of the other, instead guiding it to optimize both jointly, and produce reconstructions that are simultaneously detail-rich and temporally consistent.

## 4 Experiment 4.1 Experimental Setting

Implementation Details. We employ AdcSR (Chen et al., 2025a) pretrained by compressing PiSA- SR (Sun et al., 2024) as 2D backbone, in which the SD2.1 denoising UNet and VAE decoder (Luo et al., 2023) are channel-pruned by 25% and 50%, respectively, and augmented with zero-initialized (Zhang et al., 2023) 1D temporal convolutions to form our AdcVSR network. Each 1D convolution has a kernel size of 3 and the same channel number as its preceding UNet block. For discriminators, the channel numbers of first convolutions are adjusted to match the dimensions of input images and features, while both channel numbers of last-layer features are set to 256. The "detail" and "consistency" heads are implemented by 1 × 1 convolutions with 192 and 64 output channels, respectively.

Similar to Wang et al. (2021), AdcVSR model is trained in two consecutive stages. In the first stage, we perform only error-minimizing distillation from the pretrained DOVE teacher without adversarial learning for 200K iterations. In the second stage, AdcVSR (generator) is initialized with the weights from the first stage and fine-tuned for another 200K iterations. During this stage, the pixel-domain discriminator uses the pretrained ConvNeXt backbone from the OpenCLIP library (kept frozen) (Liu et al., 2022; Lin et al., 2025c), while the feature-domain discriminator exploits the same pretrained augmented SD UNet from the first stage (also frozen). The initial learning rates for AdcVSR are set to 1×10−4in the first stage and 1×10−5in the second stage, each halved after 100K iterations, while the trainable parts of discriminators (first and tail convolutions, as well as heads) adopt learning rate 1 × 10−7. Loss weights are set to λpixel = 0.1, λfeature = 1.0, and λadv = 1.0, respectively.

In both stages, we fully fine-tune the entire AdcVSR network end-to-end, following Liu et al. (2025); Sun et al. (2025); Chen et al. (2025b), using randomly sampled and cropped temporally consistent

Table 1: **Quantitative comparison of Real-VSR performance.** Inference time is measured on an NVIDIA H20 GPU for generating a 25-frame video at spatial resolution 512×512. The best, secondbest, and third-best results are labeled in **bold red**, underlined blue, and *italic green*, respectively.

Method RealBasicVSR Upscale-A-Video MGLD-VSR STAR SeedVR2 DOVE DLoRAL PiSA-SR AdcSR HYPIR **AdcVSR (Ours)**

Test Dataset: UDM10 (Synthetic)

PSNR↑ 24.39 23.03 24.28 22.61 25.92 **26.00** 22.49 23.21 23.39 22.55 *25.36* SSIM↑ 0.7376 0.6189 0.7491 0.6534 *0.7674* **0.7805** 0.7130 0.6799 0.6772 0.6995 0.7697 LPIPS↓ 0.3283 0.4218 0.3103 0.5055 0.2653 **0.2645** 0.3201 0.3658 0.3781 0.3736 *0.3065* DISTS↓ 0.2078 0.2360 *0.1909* 0.2665 **0.1532** 0.1732 0.2066 0.2213 0.2287 0.2125 0.2112 MANIQA↑ 0.5725 0.5331 0.5558 0.3468 0.5232 0.5133 0.5679 **0.6257** 0.5696 0.5856 *0.5793* CLIPIQA↑ 0.4422 0.4661 0.4640 0.2346 0.3471 0.5420 0.4667 **0.7055** *0.6693* 0.6006 0.6818 MUSIQ↑ 57.10 52.06 58.12 33.93 50.09 60.68 55.54 **66.42** *61.30* 59.85 63.88

E∗

warp↓ 3.36 3.68 3.31 *2.51* 2.56 2.22 3.51 6.96 6.19 10.68 **1.67**

DOVER↑ 0.2610 0.4002 0.3311 0.2717 0.3296 0.4731 0.3637 **0.5010** 0.4364 *0.4851* 0.4878

Test Dataset: VideoLQ (Real-World)

MANIQA↑ 0.5609 0.5366 0.5530 0.4356 0.4389 0.4336 0.5976 0.6319 0.6017 **0.6424** *0.6121* CLIPIQA↑ 0.3444 0.3594 0.3446 0.2497 0.2318 0.3258 0.4211 **0.6199** 0.6098 0.5937 *0.6024* MUSIQ↑ 56.47 55.20 55.90 41.01 40.56 50.03 58.50 **67.31** 66.14 63.69 *64.55*

E∗

warp↓ 9.27 14.54 8.99 10.65 11.32 8.41 *8.94* 12.65 12.47 23.45 **6.74**

DOVER↑ 0.2239 0.3278 0.2830 0.3013 0.2027 0.3790 0.3192 *0.4131* 0.4100 **0.4711** 0.4319

Efficiency

#Steps↓ - 30 50 15 1 1 1 1 1 1 1

#Param. (B)↓ **0.04** 14.44 1.43 2.49 8.24 10.55 1.30 1.30 0.46 1.55 *0.57* Time (s)↓ **0.35** 66.39 32.34 96.38 60.61 4.42 6.36 2.94 0.52 2.81 *0.55*

real video and detail-rich real image data from OpenVid-1M (Nan et al., 2024) and LSDIR (Li et al.,
2023). We use a batch size of 8, with 25 frames per video clip and a spatial resolution of 512 × 512.

The RealBasicVSR degradation pipeline (Chan et al., 2022b) is applied to synthesize LR-HR video pairs. All experiments are implemented in PyTorch, and trained with the Adam optimizer (Kingma, 2014) on 8 NVIDIA H20 GPUs (96GB each), with the full training process taking about one day. Test Datasets. Following Liu et al. (2025); Wang et al. (2025a); Sun et al. (2025), we test AdcVSR and compare it with other methods using the same synthetic and real-world datasets as DOVE (Chen et al., 2025b). The three synthetic test datasets include UDM10 (Yi et al., 2019) (10 videos), SPMCS (Tao et al., 2017) (30 videos), and YouHQ40 (Zhou et al., 2024) (40 videos), which are synthesized with the same degradation pipeline as during training, using a scaling factor of 4 for Real-VSR task. The three real-world datasets are RealVSR (Yang et al., 2021) (50 videos), MVSR4x (Wang et al., 2023b) (15 videos), and VideoLQ (Chan et al., 2022b) (50 videos). All videos are pre-processed via clipping the first 25 frames and applying center-cropping, fixing the output spatial size to 512×512. Evaluation Metrics. We employ both full-reference and no-reference metrics for performance evaluations. Full-reference metrics include PSNR and SSIM (Wang et al., 2004) for fidelity, as well as LPIPS (Zhang et al., 2018) and DISTS (Ding et al., 2020) for perceptual quality. No-reference metrics include MANIQA (Yang et al., 2022), CLIPIQA (Wang et al., 2023a), and MUSIQ (Ke et al., 2021). In addition, following Yang et al. (2024a); Zhang et al. (2024b); Sun et al. (2025), we report the flow warping error E∗warp (Lai et al., 2018), scaled by 10−3, as in DOVE (Chen et al., 2025b), and employ DOVER (Wu et al., 2023), to evaluate temporal consistency and video quality, respectively.

## 4.2 Comparison With State-Of-The-Arts

Compared Methods. Following Liu et al. (2025); Wang et al. (2025a); Sun et al. (2025); Chen et al. (2025b), we compare the proposed AdcVSR model with seven representative Real-VSR approaches: the non-generative RealBasicVSR (Chan et al., 2022b); multi-step diffusion-based Upscale-A-Video (Zhou et al., 2024), MGLD-VSR (Yang et al., 2024a), and STAR (Xie et al., 2025b); as well as onestep diffusion networks SeedVR2 (Wang et al., 2025a), DOVE (Chen et al., 2025b), and DLoRAL (Sun et al., 2025). Additionally, we include three state-of-the-art one-step diffusion-based Real-ISR approaches: PiSA-SR (Sun et al., 2024), AdcSR (Chen et al., 2025a), and HYPIR (Lin et al., 2025c), which super-resolve video frames independently, for comprehensive evaluation and comparison. Video Quality Comparison. Tab. 1 quantitatively demonstrates that our AdcVSR achieves competitive performance across a broad range of metrics. First, it ranks within the top three in most cases, surpassing the majority of competing approaches and confirming its effectiveness in restoring highquality video frames. Second, AdcVSR achieves strong temporal consistency with smallest warping errors, as indicated by its superior E∗warp results. In contrast, Real-ISR models perform the worst on this metric because they lack temporal modeling, which leads to inconsistent content across frames. Third, when compared with the previous best approaches DOVE (teacher) and PiSA-SR in fidelity, perceptual quality, and warping error, AdcVSR remains highly competitive across all these aspects.

![7_image_0.png](7_image_0.png)

The advantages of AdcVSR are also verified by the qualitative comparison in Fig. 3. Our network reconstructs sharp and realistic details, while RealBasicVSR, Upscale-A-Video, MGLD-VSR, DOVE, and DLoRAL yield over-smoothed outputs. STAR and AdcSR bring artifacts to boat's windows and facial regions, whereas PiSA-SR and HYPIR produce fewer details on building and water surface, or generate visually unrealistic boat textures. Moreover, they suffer from significant temporal instability, as evidenced by the erratic fluctuations in their temporal profiles, resulting in unpleasant flickers that degrade overall video quality. In comparison, our AdcVSR not only reconstructs natural details on buildings, boat structures, water textures, facial components, clothing, hat, and signage with less distortion, but also maintains smooth transitions across consecutive frames with reduced flickering. It is worth highlighting from Tab. 1 and Fig. 3 that, despite their poor temporal consistency, Real-ISR
diffusion networks PiSA-SR, AdcSR, and HYPIR, with only 2D spatial attentions and convolutions, are highly effective at removing degradations in individual video frames and generating rich details. This results in high-quality outputs with strong scores on no-reference perceptual metrics, including MANIQA, CLIPIQA, MUSIQ, and DOVER, which often align better with human perception than traditional metrics like PSNR. This observation is consistent with hypothesis (1) in Sec. 3.2. Building on this insight, our method employs lightweight temporal convolutions together with dual-head adversarial distillation, allowing the preservation of strong per-frame detail quality while improving fidelity and temporal consistency across all frames. Efficiency Comparison. The final 4 rows of Tab. 1 and the bubble plot in Fig. 4 compare Real-VSR performance in temporal consistency (E∗warp), step numbers, parameter numbers, and inference times across different approaches. Built upon an effective "2D + 1D" architecture, and learned from the large 3D DiT teacher DOVE using our dual-head adversarial distillation scheme, AdcVSR delivers both strong temporal consistency with the best E∗warp results and efficiency gains. Specifically, it reduces parameters by 96%, 60%, and 77%, and accelerates inference by 121×, 59×, and 175× over the multi-step diffusionbased methods Upscale-A-Video, MGLD-VSR, and STAR. Against one-step diffusion-based Real-VSR models SeedVR2 and DLoRAL, it achieves parameter reductions of 93% and 56%, and accelerations of 110× and 308×, respectively. Compared with its teacher DOVE, AdcVSR achieves **a 95% reduction** in parameters and an 8× **speedup** while maintaining very competitive video quality. Overall, AdcVSR is substantially faster and more lightweight than most existing approaches, verifying the effectiveness of our improved ADC method and the high efficiency of the resulting compressed diffusion network.

![7_image_1.png](7_image_1.png)

Flow W
arpi ng Error Figure 4: **Performance comparison among** diffusion-based Real-VSR methods in temporal consistency and complexity (parameter number and inference time) (see Tab. 1). AdcVSR attains the lowest warping error E∗warp, the second-lowest parameter number, and the second-highest inference speed. Bubble colors represent method types: green for multistep, blue for one-step, and red for AdcVSR.

4.3 ABLATION STUDY Table 2: **Comparison of network designs** on UDM10.

Method DISTS↓ E∗warp↓ \#Param. (B)↓
3D (A Pruned DOVE) **0.2098** 2.53 8.36 2D (AdcSR) 0.2418 4.43 **0.52** 2D + 1D (Ours) 0.2112 **1.67** 0.55 Input 3D 2D **2D + 1D (Ours)**
Effect of "2D + 1D" Network Design. Tab. 2 and Fig. 5 compare three student architectures: a pruned 3D DiT (based on DOVE) obtained by the original ADC approach, a 2D SD backbone (AdcSR), and our AdcVSR. The 3D DiT model delivers the best DISTS and strong E∗warp but remains heavy. Replacing it with a 2D backbone severely degrades performance, showing that a model without temporal modeling cannot effectively learn from a temporally aware teacher, or maintain both frame quality and inter-frame coherence, leading to visible flickering. By introducing 1D convolutions, our design restores temporal modeling capacity while preserving efficiency: it achieves the lowest E∗warp, narrows DISTS gap to 3D model to 0.0014 with 7% parameters, and produces sharp textures and smooth temporal profiles.

Table 3: **Comparison of discriminators** on YouHQ40.

Method CLIPIQA↑ E∗
warp↓
Single-Head, Dual-Domain 0.6745 6.32 Dual-Head, Single-Domain 0.6421 3.59 Dual-Head, Dual-Domain (Ours) 0.6861 **2.22**
Figure 5: **Comparison of different network designs**.

Effect of Dual-Head Discriminators. Tab. 3 compares three AdcVSR variants with different settings for the discriminators: (1) singlehead (only one output without distinguishing

details and consistencies), (2) single-domain (only one feature-domain discriminator as in the original ADC), and (3) our proposed scheme. The

single-head variant preserves frame quality but shows much worse E∗warp, indicating that consistency

is less optimized during its adversarial training. The single-domain variant improves consistency but reduces perceptual quality due to the absence of pixel-domain supervision. In contrast, our method achieves best performance on both metrics, effectively balancing details and temporal consistency.

Table 4: **Comparison of distillation setups** on MVSR4x.

Method PSNR↑ LPIPS↓ MUSIQ↑

No Adversarial Loss 23.97 0.3596 54.33 No Teacher (HR GT Only) **24.85** 0.3641 50.32 SeedVR2 as Teacher 23.24 0.3489 60.74 DLoRAL as Teacher 23.08 0.3554 54.61 DOVE as Teacher (Ours) 23.81 0.3337 **61.48**

Effect of Adversarial Distillation. In Tab. 4, we compare AdcVSR variants under different setups of adversarial learning and distillation. Removing adversarial losses or relying solely on ground truth (GT) supervisions noticeably degrades LPIPS and MUSIQ, indicating that both adversarial training and a teacher's guidance are essential for perceptual quality. Using SeedVR2 or DLoRAL as teachers yields promising but weaker results. By contrast, our adversarial distillation with DOVE as teacher strikes a favorable balance across three metrics, demonstrating that adversarial learning combined with an appropriate teacher is important for improving Real-VSR performance in both fidelity and perceptual realism.

Due to page limitations, more experimental results, analyses, and discussions are presented in the **Appendix**.

## 5 Conclusion

In this work, we proposed an improved Adversarial Diffusion Compression (ADC) method for realworld Video Super-Resolution (Real-VSR). Instead of relying on computationally heavy 3D spatiotemporal attentions as in existing diffusion Transformer (DiT)-based approaches, our model adopted a compact "2D + 1D" design: a pruned 2D Stable Diffusion (SD) backbone for synthesizing details, augmented with lightweight 1D temporal convolutions to enforce inter-frame coherence, while their combination also proved effective in removing degradations. To address the conflicts between optimizing detail richness and temporal consistency in Real-VSR while leveraging the knowledge of a large 3D DiT teacher DOVE as well as diverse real video and image data, we introduced a dual-head, dual-discriminator adversarial distillation scheme that disentangles and jointly optimizes details and consistency via pixel- and feature-domain supervision. Across synthetic and real-world benchmarks, the resulting **AdcVSR** model achieved competitive video quality while being substantially more efficient than its 3D DiT teacher, offering a 95% parameter reduction and an 8× inference acceleration, striking a strong balance among fidelity, detail richness, temporal consistency, and model efficiency. Beyond Real-VSR, our work provides a systematic recipe for building efficient video reconstruction systems, delivering practical guidelines for diffusion model compression and real-world application.

## References

Haoran Bai, Xiaoxu Chen, Canqian Yang, Zongyao He, Sibin Deng, and Ying Chen. Vivid-vr:
Distilling concepts from text-to-video diffusion transformer for photorealistic video restoration. arXiv preprint arXiv:2508.14483, 2025.

Kelvin CK Chan, Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. Basicvsr: The search for essential components in video super-resolution and beyond. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4947–4956, 2021.

Kelvin CK Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Basicvsr++: Improving video super-resolution with enhanced propagation and alignment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5972–5981, 2022a.

Kelvin CK Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Investigating tradeoffs in real-world video super-resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5962–5971, 2022b.

Bin Chen, Gehui Li, Rongyuan Wu, Xindong Zhang, Jie Chen, Jian Zhang, and Lei Zhang. Adversarial diffusion compression for real-world image super-resolution. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 28208–28220, 2025a.

Zheng Chen, Zichen Zou, Kewei Zhang, Xiongfei Su, Xin Yuan, Yong Guo, and Yulun Zhang.

Dove: Efficient one-step diffusion model for real-world video super-resolution. arXiv preprint arXiv:2505.16239, 2025b.

Mengyu Chu, You Xie, Jonas Mayer, Laura Leal-Taixe, and Nils Thuerey. Learning temporal coher- ´
ence via self-supervision for gan-based video generation. *ACM Transactions on Graphics (TOG)*, 39(4):75–1, 2020.

Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure and texture similarity. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(5):2567–2581, 2020.

Linwei Dong, Qingnan Fan, Yihong Guo, Zhonghao Wang, Qi Zhang, Jinwei Chen, Yawei Luo, and Changqing Zou. Tsd-sr: One-step diffusion with target score distillation for real-world image super-resolution. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 23174–23184, 2025a.

Linwei Dong, Qingnan Fan, Yuhang Yu, Qi Zhang, Jinwei Chen, Yawei Luo, and Changqing Zou. Tinysr: Pruning diffusion for real-world image super-resolution. arXiv preprint arXiv:2508.17434, 2025b.

Zheng-Peng Duan, Jiawei Zhang, Xin Jin, Ziheng Zhang, Zheng Xiong, Dongqing Zou, Jimmy S
Ren, Chun-Le Guo, and Chongyi Li. Dit4sr: Taming diffusion transformer for real-world image super-resolution. *arXiv preprint arXiv:2503.23580*, 2025.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam ¨
Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In *Forty-first international conference on machine learning*, 2024.

Jue Gong, Tingyu Yang, Jingkai Wang, Zheng Chen, Xing Liu, Hong Gu, Yulun Zhang, and Xiaokang Yang. Haodiff: Human-aware one-step diffusion via dual-prompt guidance. arXiv preprint arXiv:2505.19742, 2025.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in Neural Information Processing Systems, 27, 2014.

Jingwen He, Tianfan Xue, Dongyang Liu, Xinqi Lin, Peng Gao, Dahua Lin, Yu Qiao, Wanli Ouyang, and Ziwei Liu. Venhancer: Generative space-time enhancement for video generation. arXiv preprint arXiv:2407.07667, 2024.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. *ICLR*, 1(2):3, 2022.

Yujie Hu, Yinhuai Wang, and Jian Zhang. Dear-gan: Degradation-aware face restoration with gan prior. *IEEE Transactions on Circuits and Systems for Video Technology*, 33(9):4603–4615, 2023.

Xu Jiang, Gehui Li, Bin Chen, and Jian Zhang. Multi-agent image restoration. *arXiv preprint* arXiv:2503.09403, 2025.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5148–5157, 2021.

Diederik P Kingma. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*,
2014.

Zhe Kong, Le Li, Yong Zhang, Feng Gao, Shaoshu Yang, Tao Wang, Kaihao Zhang, Zhuoliang Kang, Xiaoming Wei, Guanying Chen, et al. Dam-vsr: Disentanglement of appearance and motion for video super-resolution. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pp. 1–11, 2025.

Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

Wei-Sheng Lai, Jia-Bin Huang, Oliver Wang, Eli Shechtman, Ersin Yumer, and Ming-Hsuan Yang.

Learning blind video temporal consistency. In Proceedings of the European conference on computer vision (ECCV), pp. 170–185, 2018.

Jianze Li, Jiezhang Cao, Zichen Zou, Xiongfei Su, Xin Yuan, Yulun Zhang, Yong Guo, and Xiaokang Yang. Unleashing the power of one-step diffusion based image super-resolution via a large-scale diffusion discriminator. *arXiv preprint arXiv:2410.04224*, 2024.

Jianze Li, Jiezhang Cao, Yong Guo, Wenbo Li, and Yulun Zhang. One diffusion step to real-world super-resolution via flow trajectory distillation. *arXiv preprint arXiv:2502.01993*, 2025a.

Runyi Li, Bin Chen, Jian Zhang, and Radu Timofte. Ctsr: Controllable fidelity-realness trade-off distillation for real-world image super resolution. *arXiv preprint arXiv:2503.14272*, 2025b.

Xiaohui Li, Yihao Liu, Shuo Cao, Ziyan Chen, Shaobin Zhuang, Xiangyu Chen, Yinan He, Yi Wang, and Yu Qiao. Diffvsr: Revealing an effective recipe for taming robust video super-resolution against complex degradations. *arXiv preprint arXiv:2501.10110*, 2025c.

Yawei Li, Kai Zhang, Jingyun Liang, Jiezhang Cao, Ce Liu, Rui Gong, Yulun Zhang, Hao Tang, Yun Liu, Denis Demandolx, et al. Lsdir: A large scale dataset for image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1775–1787, 2023.

Jingyun Liang, Yuchen Fan, Xiaoyu Xiang, Rakesh Ranjan, Eddy Ilg, Simon Green, Jiezhang Cao, Kai Zhang, Radu Timofte, and Luc V Gool. Recurrent video restoration transformer with guided deformable attention. *Advances in Neural Information Processing Systems*, 35:378–393, 2022.

Jingyun Liang, Jiezhang Cao, Yuchen Fan, Kai Zhang, Rakesh Ranjan, Yawei Li, Radu Timofte, and Luc Van Gool. Vrt: A video restoration transformer. *IEEE Transactions on Image Processing*, 33:2171–2182, 2024.

Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. *arXiv preprint arXiv:2402.13929*, 2024.

Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation. *arXiv preprint arXiv:2501.08316*, 2025a.

Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. *arXiv preprint arXiv:2506.09350*, 2025b.

Xinqi Lin, Fanghua Yu, Jinfan Hu, Zhiyuan You, Wu Shi, Jimmy S Ren, Jinjin Gu, and Chao Dong.

Harnessing diffusion-yielded score priors for image restoration. *arXiv preprint arXiv:2507.20590*, 2025c.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. *arXiv preprint arXiv:2210.02747*, 2022.

Yong Liu, Jinshan Pan, Yinchuan Li, Qingji Dong, Chao Zhu, Yu Guo, and Fei Wang. Ultravsr:
Achieving ultra-realistic video super-resolution with efficient one-step diffusion space. *arXiv* preprint arXiv:2505.19958, 2025.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie.

A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11976–11986, 2022.

Alice Lucas, Santiago Lopez-Tapia, Rafael Molina, and Aggelos K Katsaggelos. Generative adversarial networks and perceptual losses for video super-resolution. IEEE Transactions on Image Processing, 28(7):3312–3327, 2019.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. *arXiv preprint arXiv:2310.04378*, 2023.

Chong Mou, Xintao Wang, Yanze Wu, Ying Shan, and Jian Zhang. Empowering real-world image super-resolution with flexible interactive modulation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(11):7317–7330, 2024.

Seungjun Nah, Sungyong Baik, Seokil Hong, Gyeongsik Moon, Sanghyun Son, Radu Timofte, and Kyoung Mu Lee. Ntire 2019 challenge on video deblurring and super-resolution: Dataset and study. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pp. 0–0, 2019.

Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. *arXiv* preprint arXiv:2407.02371, 2024.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨
resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Claudio Rota, Marco Buzzelli, and Joost van de Weijer. Enhancing perceptual quality in video superresolution through temporally-consistent detail synthesis using diffusion models. In European Conference on Computer Vision, pp. 36–53. Springer, 2024.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. *arXiv* preprint arXiv:2202.00512, 2022.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. *arXiv preprint arXiv:2311.17042*, 2023.

Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. In SIG- GRAPH Asia 2024 Conference Papers, pp. 1–11, 2024.

Shuwei Shi, Jinjin Gu, Liangbin Xie, Xintao Wang, Yujiu Yang, and Chao Dong. Rethinking alignment in video super-resolution transformers. *Advances in Neural Information Processing Systems*, 35:36081–36093, 2022.

Lingchen Sun, Rongyuan Wu, Zhiyuan Ma, Shuaizheng Liu, Qiaosi Yi, and Lei Zhang. Pixellevel and semantic-level adjustable super-resolution: A dual-lora approach. *arXiv preprint* arXiv:2412.03017, 2024.

Yujing Sun, Lingchen Sun, Shuaizheng Liu, Rongyuan Wu, Zhengqiang Zhang, and Lei Zhang.

One-step diffusion for detail-rich and temporally consistent video super-resolution. *arXiv preprint* arXiv:2506.15591, 2025.