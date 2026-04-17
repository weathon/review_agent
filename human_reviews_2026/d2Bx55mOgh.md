# TerraCodec: Compressing Earth Observations

- Decision: Reject
- Scores: 2, 4, 4, 6, 6

## Abstract
Earth observation (EO) satellites produce massive streams of multispectral image time series, posing pressing challenges for storage and transmission. Yet, learned EO compression remains fragmented and lacks publicly available, large-scale pretrained codecs. Moreover, prior work has largely focused on image compression, leaving temporal redundancy and EO video codecs underexplored. To address these gaps, we introduce TerraCodec (TEC), a family of learned codecs pretrained on Sentinel-2 EO data. TEC includes efficient multispectral image variants and a Temporal Transformer model (TEC-TT) that leverages dependencies across time. To overcome the fixed-rate setting of today's neural codecs, we present Latent Repacking, a novel method for training flexible-rate transformer models that operate on varying rate-distortion settings. TerraCodec outperforms classical codecs, achieving 3-10x stronger compression at equivalent image quality. Beyond compression, TEC-TT enables zero-shot cloud inpainting, surpassing state-of-the-art methods on the AllClear benchmark. Our results establish EO-trained neural codecs and temporal compression as a promising direction for Earth observation. Code and model weights will be released under a permissive license.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper propose TerraCodec, a series of neural codecs for EO data based on well-designed architectures. The author use Factorized Prior variant, ELIC, and VCT architecture and adapt them for multispectral and temporal EO images. What’s more, the author propose flexible-rate codec by selecting different latent channels for transmission. The proposed codecs outperform classical codes and can be utilized for downstream tasks and zero-shot cloud inpainting.

### Strengths
- The author consider the multispectral and temporal dependency of EO images.
- The writing and presentation of the paper are excellent, with a clear and logical flow.

### Weaknesses
- The proposed codecs are **lack of novelty**, with no specific design for EO images.
- More performance comparison tests of neural compression need to be included, such as [1,2,3]
- **The setting for downstream tasks evaluation is impractical.** To demonstrate that the compressed images have minimal impact on downstream tasks, previous works, including task-oriented compression, use well-trained AI models and tested with compressed images without finetuning. First, downstream tasks may involve images from various satellites with different codecs, making it impossible to fine-tune for a single codec's reconstructed images. Second, users may not have sufficient data for fine-tuning. Additionally, the authors don’t demonstrate that finetuning with TerraCodec-FP’s reconstructed images would not negatively affect the task accuracy for images from other sources.

[1]  Remote sensing image compression based on high-frequency and low-frequency components

[2] COSMIC: Compress Satellite Images Efficiently via Diffusion Compensation

[3] Map-Assisted Remote-Sensing Image Compression at Extremely Low Bitrates

### Questions
- Is there any specific design for EO images, considering multispectral, 16-bit inputs and temporal dependency, distinct from video? Or the proposed TerraCodec only change the input channels and pretrained on EO dataset?
- In line161, TEC-TT tries to model temporal dependencies of seasonal EO data, however, for LEO satellite, a single orbit revolution is only 90 minutes. Therefore, the dependency between EO data is not seasonal. The angle, jitter, and cloud cover factors of satellite shooting between time-series images should be considered more.
- In Figure6, why the results of TerraCodec-TT (image only) is not equal with TerraCodec-ELIC? When TEC-TT reduces to an image codec, the architecture is the same as ELIC.
- FlexTEC exhibits poor RD performance, even underperforming TerraCodec-FP at low bitrates. Previous work [4] has shown that better RD performance can be achieved when selecting different channels to achieve variable-rate.

[4] Slimmable Compressive Autoencoders for Practical Neural Image Compression

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
TerraCodec targets EO compression with three models (FP/ELIC/TT) and a single-checkpoint variable-rate variant (FlexTEC) via Latent Repacking. Experiments on Sentinel-2 show RD gains vs. JPEG/JPEG2000/WebP/HEVC, plus zero-shot cloud removal and downstream robustness.

### Strengths
The paper aligns well with practical EO data characteristics, focusing on multispectral and temporal redundancy and addressing real-world demands such as variable-rate compression and downstream validation. The overall engineering implementation is solid, with clear structure and detailed configurations. The topic is timely and relevant, and the work provides potential value for large-scale EO data storage and transmission scenarios.

### Weaknesses
The experimental design does not fully support the claimed scope. Validation is limited to Sentinel-2 and lacks evaluation on datasets with different spatial, spectral, and temporal resolutions, which raises questions about generalizability. Some experiments are not sufficiently convincing, particularly the absence of quantitative metrics such as end-to-end encoding and decoding latency. The choice of baselines is somewhat outdated, lacking comparisons with the latest codec standards and learned compression frameworks. In addition, the novelty appears relatively weak—TEC-FP and TEC-ELIC seem to be adaptations of existing architectures rather than newly proposed methods.

### Questions
1. Dataset and Scope Expansion: Please include additional EO datasets (e.g., fMoW, USMapping, Landsat, MODIS) to verify the method’s performance under different spatial, spectral, and temporal resolutions, demonstrating its generalization ability.

2. Baseline Completeness: Extend the comparisons to include VTM, JPEG XL, and other recent standards, as well as diffusion-based and INR-based learned compression methods. In addition, comparisons with state-of-the-art EO-specific compression models (e.g., HL-RSCompNet) are needed for fairness and completeness.

3. Algorithm Efficiency: Provide quantitative evaluations of end-to-end encoding and decoding efficiency, including runtime and resource usage. A speed–quality curve comparing TEC variants would help illustrate their practical advantages.

4. Downstream Task Validation: Beyond classification and segmentation, please include regression-based downstream tasks (e.g., NDVI or vegetation index estimation) to show the general usability of compressed data across different task types.

### Soundness
2

### Presentation
4

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
The paper presents TerraCodec, a family of learned codecs for multispectral and temporal Earth Observation (EO) imagery, including TEC-FP, TEC-ELIC, and the temporal Transformer-based TEC-TT. It further proposes FlexTEC, a single-checkpoint variable-rate scheme leveraging Latent Repacking. Experiments on Sentinel-2 demonstrate improvements over classical codecs (JPEG, JPEG2000, HEVC) and maintain downstream performance for tasks such as land-cover classification and flood mapping.

### Strengths
1. The work targets multispectral and temporal EO imagery, incorporating per-band normalization and temporal modeling, demonstrating good engineering practice for EO data.
2. The paper provides detailed model and training setups, improving reproducibility and practical applicability.
3. The proposed FlexTEC offers a practical mechanism to achieve variable bitrates within a single model, which is relevant for real EO applications.
4. The authors verify that compressed data remain useful for EO downstream tasks, demonstrating functional robustness.

### Weaknesses
1. The title and abstract suggest “Compressing Earth Observation” in general, but experiments are limited to Sentinel-2. The work would be more accurately described as “Sentinel-2 multispectral image compression.” Broader validation (e.g., Landsat, MODIS) would strengthen the generalization claim.
2. The related work section omits several influential recent studies, such as C3, PnVC (INR-based), and diffusion-based compression models. In addition, recent SOTA compression methods for remote sensing imagery are not discussed. Including both general and EO-specific works would make the review more complete.
3. Unclear contribution boundary:  TEC-FP and TEC-ELIC are adaptations of existing frameworks (Factorized Prior and ELIC) to EO imagery and should not be presented as novel contributions. The methodological innovation primarily lies in Latent Repacking/FlexTEC and in systematizing EO compression practice, not in the architectural variants themselves.
4. Incomplete experimental comparisons:  The study lacks modern baselines—neither the latest codec standard VVC (H.266) nor contemporary learned methods (e.g., diffusion-based or INR-based) are included. It also omits compression approaches in the remote sensing domain. Extending the comparisons to these would significantly improve credibility.

### Questions
1. The title and abstract suggest general EO compression, yet all experiments use Sentinel-2. Can the authors include results from at least one additional EO source (e.g., Landsat, MODIS) to support claims of generalization?
2. Are TEC-FP and TEC-ELIC introducing new algorithmic components or primarily applying existing compression backbones to EO data? Please delineate what is new beyond domain-specific adaptations.
3. The Related Work section omits influential recent studies such as C3, PNVC, and diffusion-based compression frameworks. Please update this section and re-situate TerraCodec in the current research landscape.
4.  The comparison set lacks modern codecs and neural approaches. Please include evaluations against VVC, C3, or diffusion-based compression methods, as well as recent SOTA compression methods for remote sensing imagery,  to ensure fair and up-to-date benchmarking.
5. Analysis No runtime or complexity metrics are provided. Please add encoding/decoding latency, throughput, and resource usage comparisons among TEC-EP/ELIC/TT/FLEX variants to assess deployment feasibility.
6. The “latent repacking” mechanism resembles existing flexible bitrate methods. Please specify the conceptual or technical differences and provide evidence of improved generality or efficiency.
7. Figure 1 lacks TEC-ELIC visual results. Please include side-by-side reconstructions.
8. It would be valuable to evaluate how different temporal reference settings affect reconstruction quality — including the number of reference frames, temporal intervals, degree of land-cover change, and cloud coverage ratio within the temporal context.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents TerraCodec (TEC), a family of neural compression models made specifically for Earth Observation (Sentinel-2 L2A) imagery. TEC is designed for multispectral time-series Earth observation data, where existing image codecs (like JPEG) fail to capture temporal redundancy, and video codecs rely on motion priors that are unsuitable for the radiometric evolution of largely static EO scenes.

TerraCodec includes:
- TEC-FP: a simple, efficient CNN-based codec using a factorized prior.
- TEC-ELIC: a stronger version using spatial–channel attention for better rate–distortion trade-offs.
- TEC-TT: a temporal transformer that learns relationships across time, capturing how EO scenes change seasonally rather than through motion.
- FlexTEC: a flexible-rate model based on TEC-TT, trained with Latent Repacking and masking, allowing users to control bitrate from a single checkpoint.

### Strengths
The paper offers a well-motivated and effective solution in Earth Observation data storage and transmission, offering a real-world impact accompanied by technical innovation. The work introduces temporal transformers and a flexible-rate compression mechanism through latent repacking. The experimental setup is comprehensive and carefully executed, covering rate–distortion performance, zero-shot cloud inpainting, and clear qualitative analyses. Moreover, the methodology is transparent and well-documented, with extensive implementation details and a commitment to open-source release, which enhances the work’s reproducibility and value to the community.

### Weaknesses
**Generality and scope**: It would be useful to evaluate the transferability of TerraCodec’s pretrained models to other sensors (e.g., Sentinel-1 SAR or Landsat-8) without retraining. If such a transfer is not feasible, clarifying the underlying technical limitations would strengthen the discussion. Moreover, assessing how TerraCodec performs on Sentinel-2 L1C imagery would be valuable, as it would enable direct application to datasets like SEN12MS-CR-TS [R1] for cloud inpainting.

**Computational cost**: Training TEC-TT and FlexTEC appears computationally demanding and may be impractical for non-research users. Including runtime or inference-time benchmarks would help assess the models’ operational viability.

**Limited temporal depth in cloud inpainting experiments**: Evaluation uses 4-frame sequences, shorter than real-world EO time series, so long-term compression benefits may be underrepresented.

**Downstream extensions**: The paper could further exploit the temporal modeling capacity of TEC-TT by exploring zero-shot change detection as an additional downstream task. Additionally, since many downstream EO tasks utilize only the RGB bands of the L2A product, exploring this setting could allow the model to interface with RGB-only datasets such as CloudTran++ [R2] for cloud-removal tasks.

### Overall assessment
This paper presents a well-motivated and technically solid contribution to learned compression for Earth observation data. The work is clearly presented and experimentally convincing. I recommend acceptance, although some aspects could be further investigated.

### Minor clarity issues:
- Fig. 3 could benefit from a clearer caption with a step-by-step flow explanations.
- A minor labeling inconsistency in Figure 9. “Context for TerraCodec-TT”: the second image is marked a x_(t-2) again, but it should likely be labeled as x_(t-1) to correctly represent the temporal sequence. 

### References
[R1] P. Ebel et al., "SEN12MS-CR-TS: A Remote-Sensing Data Set for Multimodal Multitemporal Cloud Removal," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-14, 2022
[R2] Christopoulos et al., "CloudTran++: Improved Cloud Removal from Multi-Temporal Satellite Images Using Axial Transformer Networks". Remote Sensing, 17, 86, 2025

### Questions
- Can the method be applied to other sensors (e.g., Sentinel-1 SAR or Landsat-8) without retraining?
- Have the authors tried the method on Sentinel-2 L1C data? What is the expected performance?
- How efficient are the introduced models during inference?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces TerraCodec, a family of learned compression models specifically designed for Earth observation (EO) data, with a focus on multispectral and multi-temporal satellite imagery. The core idea is to learn a compact latent representation of EO data using a transformer-based autoencoder and exploit temporal dynamics to improve compression efficiency and reconstruction fidelity. A key technical contribution is Latent Repacking, which enables flexible bitrate control without retraining. The method is evaluated on large-scale EO datasets and shows strong performance in both compression metrics (e.g., bits per pixel, BPP) and downstream tasks like cloud inpainting—specifically on the AllClear benchmark—without task-specific fine-tuning.

### Strengths
- The proposed Latent Repacking mechanism is elegant and practical. By dynamically truncating or quantizing latent codes based on importance (e.g., entropy or variance), the model supports continuous rate adaptation—a significant advantage over fixed-rate neural codecs.
- TerraCodec achieves state-of-the-art rate-distortion performance on EO data. More impressively, it demonstrates zero-shot generalization to downstream tasks

### Weaknesses
- While comparisons to image codecs and EO-specific baselines are provided, the paper does not benchmark against modern neural video codecs
- Transformer-based models can be computationally expensive. The paper lacks details on encoding/decoding latency, memory footprint, or inference speed—critical factors for deployment on ground stations or edge devices. A complexity vs. performance trade-off analysis would be valuable.

### Questions
- Can Latent Repacking be extended to hierarchical or spatially adaptive compression?

### Soundness
3

### Presentation
3

### Contribution
3
