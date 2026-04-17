# GSCV: Compressing Gaussian Splatting Sequence with Video Codec

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
This paper presents a novel effective Gaussian Splatting (GS) sequence Compression method that utilizes the Video codec (GSCV). Existing video-based GS sequence compression relies on the Parallel Linear Assignment Sorting (PLAS) to convert GS into smooth 2D maps. Using the vanilla PLAS, however, can generate images exhibiting weak inter-frame correlation, due to its stochastic nature. GSCV incorporates a simple yet efficient Inter-PLAS method to produce close images between the I- and P-frames of GS, enhancing the inter-frame performance of video codec greatly. GSCV also realizes a new pipeline based on the state-of-the-art video codecs with high bit-depth GS images, achieving higher compressibility while simultaneously raising the upper limit of the quality. Experiment results show that the proposed GSCV exhibits evidently improved performance over MPEG video and point cloud-based anchors in GS sequence compression.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
## Summary
* This paper proposes a more effective approach towards GS compression using video codec. 
* The proposed method consists of an improved inter-frame consistency module and a support for higher bit depth, which enhance the efficiency and upperbound of video codec.
* The authors provide convincing empirical evidence to support their claim.

### Strengths
## Strength
* The authors have shown that their initialization and refinement successfully stablize the GS representation between I and P frames, through clear demonstration and examples.
* The empirical result is convincing as the advantage of the proposed method over baseline is very obvious as shown in Fig 7. Besides, the authors have show that their approach outperforms other PC codec such as GPCC v1 in high bitrate regime.

### Weaknesses
## Weakness
* The adoptation of heavy video codec such as HM and VTM might increase the encoding time in a non trivial way. The adoptation of YUV444 format and 10 bits depth might hinders the adoptation of faster variant such as x265. Then, it is better to discuss the temporal complexity of different methods in Fig 7 for fair comparsion.
* It is a little bit surprising to see that VVC is outperformed by HEVC, considering the performance gap on RGB videos. Additional explaination might strengthen the result.
* As the input images are from different domains, it is probabily benefical to consider pre/post processing those images before/after the video codec, such as [One-click upgrade from 2D to 3D: Sandwiched RGB-D video compression for stereoscopic teleconferencing]. Afterall, HM and VTM are tuned for RGB videos.

### Questions
## Questions
* The weird relative performance between VTM and HM might comes from the fact that the images domain deviants too much from RGB. Is that possible that simpler codec, such as H264 works as well?

### Soundness
3

### Presentation
3

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
The paper introduces GSCV, a novel method for compressing 3D Gaussian Splatting sequences using standard video codecs. Unlike previous approaches that rely on optimization-based compression (A-3DGS), GSCV focuses on compressing the trained GS data (I-3DGS).The core innovation is Inter-PLAS, an improved version of the Parallel Linear Assignment Sorting (PLAS) method and anchor-based refinement, which ensures that consecutive GS frames (I- and P-frames) produce similar images, enhancing inter-frame prediction in video codecs like HEVC and VVC. Experiments on MPEG datasets show that GSCV outperforms existing video- and point-cloud-based methods in both tracked and semi-tracked GS sequences at some bitrate region.

### Strengths
1. Effective Inter-Frame Compression.  Spatial-Stable Initialization (SSI) and anchor-based refinement reduce randomness in PLAS. These modules  improve frame similarity, leading to better inter-frame-prediction 

2. Compatibility with Standard Codecs. Compression techniques for pre-trained Gaussian Splatting models are practical in certain scenarios, while leveraging existing video encoders also offers better adaptability, making them applicable in environments lacking specialized acceleration hardware.

### Weaknesses
1. Limited Compression Efficiency on PLAS Images. PLAS-generated images are structurally different from natural images, limiting the full potential of standard video codecs.

2. The experimental comparisons are insufficient. Although the technical paradigms are not entirely identical, there are already several studies on Gaussian Splatting sequences or 4D Gaussians. Conducting more comparisons on more dataset would more comprehensively reflect the performance of the proposed method in this paper. 

some previous work:

[1] 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

[2] 4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable Free-Viewpoint Video

### Questions
1. Generally, methods leveraging existing codecs, even if not achieving optimal overall performance, still significantly outperform training-based approaches in certain aspects, such as encoding time. This allows for some compromise in rate-distortion (RD) performance. However, PLAS-based methods still require thousands of iteration steps—does this undermine the practical value of such approaches? Furthermore, could some encoding time results be provided in the experiments?

2. In the low-bitrate region of the rate-distortion curve (bartender, cinema), the performance of the GSCV method falls below that of the GPCC all-intra mode. Does this indicate that the proposed inter-frame model still has certain limitations?

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
The paper targets I-3DGS (optimization-free) compression of Gaussian Splatting (GS) sequences using standard video codecs (HEVC/VVC). The core idea is to convert each GS frame to a set of smooth 2D “PLAS images” and then exploit inter-frame prediction in video codecs. The authors identify a weakness in the vanilla PLAS pipeline for sequences, i.e., independent stochastic initializations make I/P frames misaligned, hurting inter prediction. They propose Inter-PLAS, comprising (i) Spatial-Stable Initialization (SSI) via 3D Morton-order sorting plus reuse of the I-frame’s PLAS index as context for the P-frame, and (ii) an anchor-based PLAS refinement that aligns the P-frame’s PLAS image directly toward the I-frame to raise inter-frame similarity. They also introduce a practical bit-depth design: coordinates are quantized to 20-bit and split into two 10-bit images, other attributes are 10-bit, forming 22 attribute videos per GS sequence and enabling compatibility with HEVC/VVC. Experiments on MPEG tracked & semi-tracked GS sequences (bartender, cinema, breakfast) report gains over a video-based anchor (GSCodec Studio) and a point-cloud anchor (GPCC v1).

### Strengths
1. The paper squarely addresses the gap in inter-frame GS compression with canonical codecs, aligning with I-3DGS industrial constraints (CPU-centric decoding, codec re-use). It discusses A-3DGS vs I-3DGS and focuses on I-3DGS compression with optimization-free pipelines, together mature video codecs such as H.265/266.

2. Engineering that makes video codecs work. The 10-bit per-attribute plus 20-bit (split) for coordinates is a pragmatic choice that avoids quality ceilings while keeping compatibility (YUV444; careful avoidance of RGB↔YUV rounding). The paper also explains why VVC’s usual advantage over HEVC diminishes on PLAS images, indicating distribution shift vs natural video.

### Weaknesses
1. Limited methodological novelty.
The paper presents a straightforward engineering implementation that leverages standard video codecs (HEVC/VVC) for Gaussian Splatting sequence compression. While practical and well-structured, the core contribution—mapping GS attributes into 2D PLAS images and feeding them to an existing codec—is conceptually simple and incremental rather than a fundamentally new compression principle.

2. Incomplete comparison with learning-based and adaptive GS compression.
The work lacks comparisons with A-3DGS methods that optimize Gaussian kernels directly for compression efficiency, as well as with learning-based I-3DGS compressors such as FCGS in the inter-frame setting. Although the appendix briefly touches FCGS in an all-intra setting, obviously sequence-level modification of FCGS produces much better performance.

3. Shallow codec-level modeling.
Although the framework maintains compatibility with HEVC/VVC, it does not perform any explicit rate–distortion modeling or entropy analysis. The system relies entirely on codec defaults without quantifying the optimality of its design choices. The 10-bit attribute and 20-bit coordinate configuration appear heuristic; a principled justification via per-component rate–distortion curves, entropy measurements, or bit-allocation optimization would substantially strengthen the technical rigor.

### Questions
Please see in Weaknesses Part.

In addition, Why does VVC ≈ HEVC here (Figure 7)? Have you profiled residual statistics (e.g., heavier tails, non-natural textures) to quantify how PLAS images deviate from natural video?

The layout of statistics in some figures could be improved a lot. It is hard to distinguish some data points there.

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
2

### Summary
The paper proposes a new 3D Gaussian splat compression scheme, where a new variant of the Parallel Linear Assignment Sorting (PLAS) procedure is used to increase inter-frame similarities between mapped I- and P-frames, thus improving compression efficiency when a video-based codec is used.

### Strengths
Compression of 3D Gaussian splats is a relevant topic. The technical description of the proposal in section 4 is fairly easy to follow. Reuse of existing video / point cloud-based codec means standard-compliant and easy adaptation.

### Weaknesses
1. In the big picture of 3D Gaussian splats compression, given an existing video or point cloud-based codec is used downstream, the proposal to tweak PLAS (which the authors themselves admit is itself "heuristic" (pg.3)) in order to increase inter-frame similarities as a pre-processing step is a pretty minor hack that is not theoretically grounded. Specifically, in Section 4.1, the proposed spatial-stable initialization (SSI) is basically sorting via Morton code, which is long used in voxels when 3D point clouds voxelized into successively embedded cubes. This seems quite minor. Second, in Section 4.2, the anchor-based PLAS refinement amounts to "disabling the generation of the blurred target $T_i$ in PLAS and use $I_{PLAS}$ instead as the target to progressively smooth $P_{ini}$", which is also a minor tweak.

2. The description is not always clear, even for compression non-experts outside the Gaussian splats topic. What's meant by "end-to-end entropy supervision" What does "stable" mean in "stable I-frame PLAS image generation"? What does "anchor-based PLAS refinement" mean? There are also awkward / unnatural English expressions throughout the manuscript, such as "raising the upper limit of the quality" (abstract), "exhibits evidently improved performance" (abstract), "experiment results" rather than "experimental results" (abstract), "motivates commensurate attention" (intro).

### Questions
1. How would the I- and P-frame (and possibly B-frames) sequencing affect the proposal? For example, if P-frames $t$ can reference earlier frame $t-k$ for $k \geq 2$ for motion prediction in an IPPP frame sequence?

2. How should the blur parameters like $\sigma$, which control the speed and extent of the blurring processing, be optimized given the length of the generated IPPP sequence?

### Soundness
2

### Presentation
2

### Contribution
1
