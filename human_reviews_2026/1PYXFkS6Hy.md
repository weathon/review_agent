# TaCo:  A Benchmark for Lossless and Lossy Codecs of Heterogeneous Tactile Data

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 4

## Abstract
Tactile sensing is crucial for embodied intelligence, providing fine-grained perception and control in complex environments. However, efficient tactile data compression, which is essential for real-time robotic applications under strict bandwidth constraints, remains underexplored. The inherent heterogeneity and spatiotemporal complexity of tactile data further complicate this challenge. To bridge this gap, we introduce TaCo, the first comprehensive benchmark for Tactile data Codecs. TaCo evaluates 30 compression methods, including off-the-shelf compression algorithms and neural codecs, across five diverse datasets from various sensor types. We systematically assess both lossless and lossy compression schemes on four key tasks: lossless storage, human visualization, material and object classification, and dexterous robotic grasping. Notably, we pioneer the development of data-driven codecs explicitly trained on tactile data, TaCo-LL (lossless) and TaCo-L (lossy). Results have validated the superior performance of our TaCo-LL and TaCo-L. This benchmark provides a foundational framework for understanding the critical trade-offs between compression efficiency and task performance, paving the way for future advances in tactile perception.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces TaCo, the first large-scale benchmark for tactile data compression, evaluating 30 codecs (off-the-shelf and neural) on five heterogeneous tactile datasets. It also proposes two data-driven codecs, TaCo-LL (lossless) and TaCo-L (lossy), trained end-to-end on tactile data. The benchmark spans four task types: lossless storage, human visualization, classification, and dexterous grasping. Extensive experiments show that the proposed codecs outperform existing methods across all tasks.

### Strengths
Timeliness and Relevance 
Tactile sensing is rapidly maturing, yet compression remains fragmented and under-studied. A unified benchmark is sorely needed; 
TaCo fills this gap convincingly.
Comprehensive Evaluation
The paper tests 30 codecs on >250 k frames from five datasets covering vision-based (GelSight, DIGIT) and force-based sensors, providing the broadest coverage to date.
Novel Data-Driven Codecs
TaCo-LL and TaCo-L are, to my knowledge, the first codecs trained exclusively on tactile data. They achieve new state-of-the-art bit-rates (e.g., 0.447 bits/Byte on Touch-and-Go, 18× compression) while preserving task accuracy.

### Weaknesses
Limited Physical Interpretation of Distortion
PSNR and MS-SSIM are used for “human vision” evaluation, but tactile images are seldom viewed by humans; their perceptual space is task-dependent (e.g., shear vs. normal force). The paper does not validate that a 3 dB PSNR drop translates to an imperceptible change for either human fingers or downstream controllers. A small discrimination experiment (e.g., JND for contact edge localization) would strengthen the lossy compression claims.

Dataset Bias and Generalisation
All five datasets are collected on rigid, Lambertian objects. Soft or textured materials (foam, fabric, skin) exhibit elasto-plastic deformation and specularity that violate the stationary-signal assumption of both CNN and transformer codecs. The claim “superior on heterogeneous tactile data” is therefore premature. At minimum, the authors should report cross-dataset transfer (e.g., TaCo-L trained on Touch-and-Go, tested on ActiveCloth).
Baseline Fairness for Neural Codecs
Pre-trained image codecs (ELIC, LALIC, TCM) are evaluated zero-shot, whereas TaCo-L is fine-tuned on the target domain. This gives an unfair advantage. A fair comparison would fine-tune all neural baselines for the same number of epochs or report the “pre-trained → tactile” transfer gap.

Absence of Latency/Complexity Metrics
Real-time tactile feedback for tele-operation requires <1 ms end-to-end latency. The paper omits encode/decode runtime, GPU/CPU memory, and whether the arithmetic coder is single-pass. Without these numbers, a 190× compression ratio is meaningless for closed-loop control. Please add throughput (fps) vs. bitrate curves on an embedded GPU (Jetson Orin) or DSP.
Statistical Significance in Grasping Experiments
Table 6 reports success rates on 100 simulated objects. With only 8 deformable objects, the 95 % CI on the 62.2 % success rate is ≈ ±9 %, overlapping the baseline 63.8 %. The authors should use paired t-tests across seeds or bootstrap CIs to confirm that TaCo-L is not worse than uncompressed data.

### Questions
Figure 6 shows “visual” quality at 0.06–0.09 bpp, but tactile images are not human-interpretable. Replace with a tactile-specific error map (e.g., normal-force RMSE) or skip visual inspection altogether.
Table 3 uses BD-rate with HM-Intra as anchor. Please also quote absolute BPP at equivalent PSNR (e.g., 40 dB) so readers can judge real-world bandwidth (e.g., 0.04 bpp ≈ 1.2 Mbps at 30 fps).
Section 4.5 employs Isaac Sim tactile signals that are artificially sparse (hence 1000× ratio). Clarify that physical GelSight data achieve at most 22× (ObjTac) to avoid misleading roboticists.
Typos: “intra-frame compressors” header in Table 4 is misspelled “Jlrra-gt”; reference “Deletang et al. 2024” appears three times with inconsistent years.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduce TaCo, the first comprehensive benchmark for Tactile data Codecs.  It comprises five publicly tactile datasets, 30 codecs, and four tactile-related tasks. It  introduce two data-driven tactile codecs TaCo-LL and TaCo-L, and conducts experiment to validate their performance.

### Strengths
1. The proposed Toco benchmark is the first systematic benchmark specifically designed for tactile data compression.
2. The paper addresses the problem of tactile data compression, which is indeed an important yet overlooked issue.
3. The dataset encompasses a variety of tactile sensors.
4. The work evaluates the impact of tactile data compression on several downstream tasks.

### Weaknesses
1. The practical contribution of this work is rather limited. The proposed TaCo benchmark is essentially a simple aggregation of several existing datasets, without introducing any new data or labels. The proposed TaCo-L and TaCo-LL are in fact direct applications of two existing methods, DualComp-I and LALIC, trained on tactile data. Essentially, this work only integrates existing datasets to form a new dataset specifically for the tactile data compression task.
2. In both Figure 1 and the introduction, the paper highlights the significant intra-frame and inter-frame correlations in tactile videos, which should be a key distinction between tactile videos and natural RGB videos. However, disappointingly, the paper barely discusses this crucial aspect in its methodology, experiments, or metric design. Instead, it simply adopts existing approaches from natural image and video compression and applies them directly to the tactile domain.
3. An important distinction between tactile data and natural image or video modalities is that tactile sensing mainly serves object interaction. Therefore, validating performance only on material classification tasks or simple grasping tasks in simulation environments, which are not closely related to real tactile interaction, is insufficient to demonstrate its effectiveness. More real-world object manipulation experiments should be included.
4. A prominent issue in the tactile domain is the low diversity of available data and the significant heterogeneity among sensors. Does the proposed tactile data compression algorithm possess cross-sensor and unseen object generalization capabilities (the latter being particularly important)? Because in real-world applications, the objects that tactile sensors come into contact with can be highly diverse.
5. Another key point is that tactile data cannot exist independently of objects. Therefore, it is important to compare and discuss methods such as ObjectFolder, which use implicit neural representations to model objects. Compared with these methods, do tactile compression algorithms also offer advantages in terms of quality and efficiency?
6. Would tactile data compression negatively affect the spatially dependent information in tactile data? This aspect is crucial for tactile sensing, which is a highly fine-grained, sensitive, and localized modality during object manipulation.
7. When applying tactile compression algorithms to compress tactile data for downstream tasks, how does this differ from and what advantages does it have over using pretrained tactile encoders (such as UniTouch [1], T3 [2], or AnyTouch [3]) for feature preprocessing (which also can be seen as compressing)? What kinds of problems specifically require tactile compression algorithms rather than these pretrained encoder-based methods? Possible experimental results should be provided to support the claims, along with thorough and detailed discussions.
8. There are some confusing aspects in the experimental setup. For the Touch and Go dataset, since each recording segment contains many frames of the same object, the official split is based on collection trajectories. Why does this work not follow the official split and instead choose to perform a random split? In addition, the ObjectFolder dataset is a multimodal object dataset based on implicit neural representations, so the procedure for the material recognition task in this part needs to be explained in detail.

Overall, I believe the contribution of this work is quite limited. It simply combined several existing tactile datasets and revalidated existing techniques and experiences from image and video compression in the tactile domain, while overlooking many unique characteristics of tactile sensing. Therefore, considering the numerous issues present in the paper and the high quality standards of ICLR, I am inclined to reject this paper unless the authors can fully address these problems within a short period of time, including adding additional experiments and extensive discussions.



[1] Yang, Fengyu, et al. "Binding touch to everything: Learning unified multimodal tactile representations." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2024.

[2] Zhao, Jialiang, et al. "Transferable tactile transformers for representation learning across diverse sensors and tasks." *arXiv preprint arXiv:2406.13640* (2024).

[3] Feng, Ruoxuan, et al. "Anytouch: Learning unified static-dynamic representation across multiple visuo-tactile sensors." *arXiv preprint arXiv:2502.12191* (2025).

### Questions
1. In the appendix's motivation section, it is mentioned that one key challenge is that tactile data from high-resolution sensor arrays on robotic hands can consume a large portion of the available bandwidth. I would like to know how this problem can be addressed using tactile compression algorithms, especially for vision-based tactile sensors that are essentially cameras.
2. There are already some LLMs capable of handling tactile inputs [1,2]. Why not consider using these methods instead of using LLaMA, which has never been exposed to tactile data?
3. The experimental results on YCB-Slide are extremely high, with the uncompressed baseline reaching 99%. Is there an issue with the data split in the experiment?



[1] Yang, Fengyu, et al. "Binding touch to everything: Learning unified multimodal tactile representations." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2024.

[2] Yu, Samson, et al. "Octopi: Object property reasoning with large tactile-language models." *arXiv preprint arXiv:2405.02794* (2024).

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces TaCo, a comprehensive benchmark for tactile data compression. TaCo evaluates 30 codecs (17 off-the-shelf, 13 neural) across five heterogeneous tactile datasets comprising over 250K samples.The benchmark measures both lossless and lossy compression, connecting performance to four downstream tasks: data storage, human visualization, material/object classification, and robotic grasping. Furthermore, the authors train two tactile-specific codecs, TaCo-LL (lossless) and TaCo-L (lossy), adapted from DualComp and LALIC, showing that modality-specific training can substantially improve compression efficiency and downstream task fidelity.

### Strengths
1. Addresses a clear and underexplored bottleneck.
TaCo fills a critical gap in the tactile perception community: the absence of a standardized, quantitative framework for evaluating tactile data compression.The scope—five tactile datasets, multiple codecs, and four downstream tasks—demonstrates strong engineering execution and community value.
2. Task-grounded evaluation beyond compression metrics.
The inclusion of real downstream metrics (classification, grasping) moves the discussion beyond raw rate–distortion numbers and demonstrates practical significance for embodied and robotic systems.
3. Evidence that tactile-specific retraining is beneficial.
The adapted models (TaCo-LL, TaCo-L) consistently outperform generic codecs across modalities, indicating that learning tactile-specific priors is nontrivial and valuable. This establishes a solid empirical foundation for future work on modality-aware compression.

### Weaknesses
1. Limited methodological originality.
 While the benchmark is extensive, the codec architectures themselves largely reuse existing designs (DualComp and LALIC) with dataset-specific retraining. The paper’s main novelty lies in benchmarking and empirical synthesis rather than proposing new modeling principles or theoretical insights into tactile compression.
2. Insufficient analysis of tactile data characteristics.
 The paper shows that tactile-specific codecs outperform generic ones but does not investigate the underlying statistical or structural differences that drive this improvement. Analyses such as entropy distribution, temporal redundancy, or spatial frequency spectra could provide insight into why certain codecs perform better on specific tactile modalities.
3. Lack of robustness and generalization analysis.
 The models are trained on a subset of datasets and evaluated on others, but there is no quantitative assessment of domain shift or variability (e.g., performance variance across seeds, datasets, or other sensor types such as vibrotactile, event-based, or thermal sensing are not included). Without such analysis, the extent to which the proposed codecs generalize beyond the training distribution remains unclear.
4. Missing runtime and deployment evaluation.
 Although model sizes are reported, the inference time such as KB/s or other metrics related to time efficiency is not analyzed. These aspects are important for understanding whether the proposed codecs are suitable for real-time robotic applications.

### Questions
1. What specific reasons motivated the choice of ObjectFolder (v1) instead of the newer ObjectFolder 2.0, which features higher scene diversity and richer material representations?
 Were there technical constraints (e.g., data volume, format consistency), or does ObjectFolder 2.0 differ in ways that make it unsuitable for this benchmark?
2. What specific modifications were made to DualComp and LALIC architectures when adapting them to tactile data?
 Please clarify in detail whether the entropy modeling, tokenization, or context mechanisms differ from the originals.
3. How were the grasping experiments conducted with respect to compression?
Section 4.5 describes a grasping simulation built in NVIDIA Isaac Sim using DexHand-13 with 11 tactile sensors, where pre-compressed tactile signals are fed into a tactile-aware RL controller. Could you provide additional details such as the tactile sampling rate, control frequency, and how compression latency was incorporated into the control loop?
 Moreover, robotic grasping tasks are particularly sensitive to real-world tactile performance—success rate, contact stability, and control frequency often differ substantially between simulation and physical setups. It would therefore be valuable to discuss or empirically examine how the proposed codecs might behave on real robotic hardware, even at small scale, to assess practical applicability beyond simulation.
4. Could you report the inference-time throughput (e.g., KB/s) and latency of TaCo-LL and TaCo-L compared with existing codecs?
 Since DualComp—one of the models adapted in this work—emphasizes runtime efficiency and reports detailed KB/s metrics, including similar results here would clarify whether the proposed tactile codecs preserve comparable efficiency. This metric is also particularly important for real-time robotic systems.
5. Have you examined how sensitive TaCo-LL and TaCo-L are to the training data distribution (e.g., varying tactile sensor types or object categories)?
 This could clarify whether performance gains arise from data diversity or from architectural suitability.

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
This paper proposed TaCo as the first comprehensive benchmark for evaluating tactile data codecs. 
TaCo evaluates existing classical and neural codecs on lossless storage, human visualization, material and object classification, and dexterous robotic grasping. 
The authors also proposed TaCo-LL (lossless) and TaCo-L (lossy) models and achieved superior performance on the benchmark.

### Strengths
The motivation is important for the community. Bandwidth requirement has been one of the main bottlenecks for vision-based tactile sensors nowadays, whose temporal resolution is usually limited despite the high spatial resolution. A benchmark for tactile codecs is necessary to push forward the research in this direction. The benchmark design is comprehensive and covers a wide range of tasks that are important for tactile data usage, making it a valuable resource for the community.

### Weaknesses
1. Still not enough discussion/experiment to support the motivation. To decrease the bandwidth requirement for tactile hands in reality, a discussion/metric on whether/how these methods are computationally feasible on the sensor side hardware is necessary. Without that, the motivation of this direction is vague.
2. Lack details about the method itself. "Symbol" predicted by $f_a$ is not defined anywhere nor is its dimension or quantization level for AE clarified. Neural nets used in either setting not explained. Lack high level explanation in Sec 3.2.2.
3. Writing issues, including but not limited to: Typos; Citation formats messed up; Missing full stops; Lack of explanation of abbreviations; Eq. 1 is incomplete; Table 2 lacks explanation on what does your -12M/48M/96M mean, and why some of the results has a ratio after it but some doesn't. How is Table 4 related to the topic of the paper is unclear.
4. Lack qualitative results. As the only figure in the paper that shows qualitative results, Fig 6 is way too subtle that I cannot see the difference. Also I highly doubt the quality of tactile signals you generated within Isaac Sim. On the codec aspect, the Sim2Real gap for tactile simulation is not ignorable in this case. The lack of artifacts and much simpler optical setup could make signal behavior far from real world scenarios. On simulation itself, can you show qualitative examples to prove the results are reasonable, especially for those deformable-deformable contacts you claimed?
5. For the quantitative results, especially the dexterous hand grasping experiment, what's the purpose of introducing $s_{disturb}$ experiment? Can you explicitly explain that in your experiment design section? I also don't think the result is "comparable" to VTM-Intra.

### Questions
Covered in "Weaknesses".

### Soundness
2

### Presentation
2

### Contribution
3
