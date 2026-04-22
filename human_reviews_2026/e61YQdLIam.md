# RelayFormer: A Unified Local-Global Attention Framework for Scalable Image and Video Manipulation Localization

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 8

## Abstract
Visual manipulation localization (VML) aims to identify tampered regions in images and videos, a task that has become increasingly challenging with the rise of advanced editing tools. Existing methods face two central issues. The first is resolution diversity. Resizing or padding can distort subtle forensic cues and introduce unnecessary computational cost. The second is the difficulty of extending spatial models for images to spatio-temporal inputs in videos, which often results in maintaining separate architectures for the two data types.
To address these challenges, we propose RelayFormer, a unified framework that adapts to varying resolutions and naturally handles both static and temporal visual data. RelayFormer partitions inputs into fixed-size sub-images and introduces Global Local Relay (GLR) tokens that propagate structured context through a relay-based attention mechanism. This design enables efficient exchange of global cues, such as semantic or temporal consistency, while preserving fine-grained manipulation artifacts.
Unlike prior approaches that depend on uniform resizing or sparse attention, RelayFormer scales to variable resolutions and video sequences with minimal overhead. Experiments across diverse benchmarks demonstrate superior performance and strong efficiency, combining resolution adaptivity without interpolation or excessive padding, unified processing for images and videos, and a favorable balance between accuracy and computational cost. Code is available at~\href{https://github.com/WenOOI/RelayFormer}{https://github.com/WenOOI/RelayFormer}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces RelayFormer, a unified attention framework designed to address two core challenges in Visual Manipulation Localization (VML): resolution diversity and the modality gap across images and videos. The core of RelayFormer is a novel Global-Local Relay Attention (GLRA) mechanism. This mechanism partitions inputs of arbitrary resolution into fixed-size sub-images and introduces a small set of learnable "relay" tokens to absorb fine-grained manipulation artifacts locally before engaging in a sparse information exchange at the global level to propagate context. Furthermore, the framework incorporates a parameter-efficient design, leveraging a shared Transformer backbone with task-specific LoRA adapters to optimize efficiency with minimal impact on performance. The authors validate the framework's efficiency and performance through extensive experiments on multiple image and video benchmarks.

### Strengths
1. The core GLRA mechanism introduces a small set of learnable [GLR] (Global-Local Relay) tokens to act as information proxies. This design reduces the computational complexity of global attention from quadratic in the total number of patches to quadratic in the number of sub-images, achieving an exponential reduction in computational cost. The framework employs an elegant information fusion strategy, iteratively executing a two-step process of "local perception" and "global relay" within each Transformer layer. 

2. The paper ingeniously employs a strategy that combines a shared Transformer backbone with two independent, lightweight LoRA adapters. By dynamically switching between these adapters, the design achieves functional specialization for the distinct local and global attention tasks at a minimal additional parameter cost.

### Weaknesses
1. Information Bottleneck of Relay Tokens:
Since global information exchange is mediated through a small set of relay tokens, this mechanism may introduce mild information loss when manipulations span multiple sub-images with complex cross-region dependencies (e.g., fine-grained copy-move forgeries).
Although this trade-off is computationally justified, it would be useful to quantify how much contextual precision is sacrificed compared to a full-attention baseline.


2. Missing CAT Protocol Comparison:
The image manipulation localization experiments primarily follow the MVSS protocol (training on CASIA-v2 and testing on standard benchmarks).
However, stronger and more comprehensive evaluation settings, such as the CAT protocol,should also be considered.

### Questions
1. Quantifying Relay Compression Loss:
How much performance degradation, if any, occurs when comparing GLRA against a full global-attention model (even at a reduced resolution)?
Such an experiment could empirically validate the trade-off between efficiency and precision.

2. Evaluation under CAT Protocol:
Given that the paper currently reports results under the MVSS protocol, could the authors also train and evaluate a CAT-protocol version of RelayFormer?

3. Temporal Sparsity in Manipulation Frames
In real-world video forgeries, only a few frames may be manipulated while the majority remain authentic.
Given that RelayFormer aggregates temporal information through relay tokens across frames, how effectively can the model detect and localize sparse manipulations that occur in only a small subset of frames?

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
This paper proposes the RelayFormer framework for unified visual manipulation localization in both images and videos. The authors claim their method addresses two primary challenges: resolution diversity and the image-video "modality gap." The core innovation is the Global-Local Relay Attention (GLRA) mechanism, which propagates global context across fixed-size sub-images through learnable Global-Local Relay (GLR) tokens. The paper divides inputs into overlapping sub-images to avoid interpolation, introduces GLR tokens as an information bottleneck for efficient global-local interaction, and employs a LoRA-based parameter-efficient strategy. Experimental validation is conducted on multiple image and video manipulation datasets.

### Strengths
1.The research problem is well-motivated. The paper correctly identifies that manipulation detection requires simultaneous consideration of both fine-grained local artifacts and coarse-grained global consistency cues (such as illumination mismatches and structural redundancies). This provides a sound foundation for proposing an efficient global information propagation mechanism and offers valuable insights for the forensics community.

2.The input unification strategy (Section 3.1) is practical. By partitioning variable-resolution images and videos into fixed-size overlapping sub-images, the method avoids interpolation operations that could destroy forensic traces. This strategy is both simple and effective, supports parallel processing, and demonstrates practical engineering value.

3.The experimental validation is comprehensive. The paper evaluates the approach on 8 datasets covering both image and video scenarios. Ablation studies systematically analyze GLR token quantity (Table 4), temporal dimension (Table 5), and interpolation effects (Table 6). Robustness evaluation (Figure 5) tests against common perturbations, with generally sound experimental design.

### Weaknesses
1. The authors emphasize that images and videos belong to different modalities, which is imprecise. Multimodality typically refers to different data types (e.g., vision and text, vision and audio). Both images and videos belong to the visual modality, differing only in the presence or absence of temporal dimension. This conceptual confusion weakens the theoretical positioning of the paper. It would be more appropriate to frame this as a "temporal dimension extension problem" rather than a "modality gap."

2. The technical novelty is limited. The technical contributions of this work are relatively incremental. The use of fixed-size sub-images and Global-Local Relay (GLR) tokens represents fairly standard techniques. Fixed-size sub-images are a common operation in image and video processing (standard practice in ViT and similar architectures). The proposed GLR tokens show no significant distinction from existing prompt tuning methods [1][2][3][4], as they essentially all perform information aggregation through learnable tokens. The paper fails to adequately differentiate GLR tokens from prior work such as Dgl [3] and Visual Prompt Tuning [4]. The innovative contributions are primarily at the application level rather than the methodological level.
[1] Liu, Xiao, et al. "P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks." arXiv preprint arXiv:2110.07602 (2021).
[2]Zhang, Ji, et al. "Dept: Decoupled prompt tuning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
[3] Yang, Xiangpeng, et al. "Dgl: Dynamic global-local prompt tuning for text-video retrieval." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 7. 2024.
[4] Jia, Menglin, et al. "Visual prompt tuning." European conference on computer vision. Cham: Springer Nature Switzerland, 2022.

3. The readability needs improvement and the related work section is weak. The writing quality requires enhancement. For example, lines 103-104 and 152-153 employ excessive use of dashes. The Related Work section (Section 2) is overly brief, appearing more as a literature listing than a critical analysis. It lacks in-depth discussion of why existing methods fail. The writing should be more direct and professional to enhance clarity.

4. The 4D RoPE design lacks sufficient justification. The 4D RoPE positional encoding design (Section 3.2) lacks adequate theoretical foundation. How are the dimension allocations (d_T, d_id, d_H, d_W, d_rem) determined (lines 261)? The paper only provides formulas without sensitivity analysis. More importantly, the paper claims "strong extrapolation capability" (line 270) but provides no experimental verification. For instance, what is the performance on 2K or 4K images after training at 512×512 resolution? This claim lacks empirical support.

5. Hyperparameter selection lacks rigor. The choice of n=2 for GLR token quantity (line 311) is determined only through ablation studies, but why does n=3 lead to performance degradation? The paper merely speculates "due to redundancy" without deeper analysis. Similarly, the choices of sub-image sizes (512×512 for images, 224×224 for videos) and overlap size (16 pixels) lack theoretical or experimental justification. Additional experimental analysis of hyperparameter selection should be provided.

### Questions
Refer to the weaknessses.

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
4

### Summary
The paper tackles visual manipulation localization for both images and videos, addressing resolution diversity and modality gaps. It proposes RelayFormer, which partitions inputs into fixed-size sub-images and introduces Global-Local Relay (GLR) tokens to propagate global context via a Global-Local Relay Attention (GLRA) mechanism. The method includes a parameter-efficient strategy using shared backbone layers with task-specific adapters, a 4D RoPE positional encoding, and a lightweight query-based mask decoder. Experiments on multiple image and video benchmarks show improved F1/IoU performance against prior work, with analyses of FLOPs/parameters and robustness.

### Strengths
A new framework for unified image and video forensics.

Achieves top average F1 on image benchmarks, and consistently high IoU/F1 across video inpainting methods on MOSE.

Robustness curves for Gaussian blur, noise, and JPEG compression.

### Weaknesses
The proposed approach relies on fusing local and global information, a well-established technique in computer vision whose effectiveness is often assumed.

While results from training on both are presented, there is no clear explanation or investigation into how these modalities mutually influence each other. An intuitive hypothesis is that images can be considered single frames of videos, and training on image forgery could enhance video forgery detection (and vice-versa). To truly demonstrate the benefit of a unified approach, it is suggested to include experiments evaluating model performance when trained exclusively on one modality (image or video) and then tested on both. This would shed light on the synergy between the two data types.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes RelayFormer, a unified framework for visual manipulation localization across both images and videos. It addresses two major challenges: resolution diversity and the image–video modality gap. The key innovation is the Global-Local Relay Attention (GLRA), where Global-Local Relay (GLR) tokens propagate global context efficiently between local sub-images, enabling scalability without interpolation or dense attention. A lightweight query-based decoder produces precise masks. Extensive experiments show state-of-the-art accuracy, strong robustness, and excellent efficiency on multiple benchmarks.

### Strengths
- The insight of bridging image and video manipulation localization holds novelty. It will offer a benchmark case for future multi-modal VML tasks.
- The proposed method’s resolution strategy offers a significant computational efficiency advantage within the current field of manipulation detection.
- The experiments in this paper are comprehensive and well-organized, effectively demonstrating the proposed claims.

### Weaknesses
- The proposed method shares some similarities with Visual Prompt Tuning[A], as it introduces additional trainable tokens to transmit information and assist decision-making. It is recommended that the authors discuss this connection in an appropriate section of the paper.
- To the best of my knowledge, RoPE is introduced for the first time in a manipulation localization model. The authors may consider analyzing the advantages of this positional embedding for a pure computer vision task like manipulation detection, ideally supported by additional experiments.
- We are currently in an era that values scaling up as an important perspective. Although the authors claim that the proposed structure bridges image and video manipulation localization, the experiments are conducted separately for the two modalities. It would be interesting to explore whether joint training on both images and videos could lead to further improvements or provide additional insights into scaling behavior.

## Reference
- [A] Jia, Menglin, et al. "Visual prompt tuning." European conference on computer vision. Cham: Springer Nature Switzerland, 2022.

### Questions
Will the code be publicly available?

### Soundness
3

### Presentation
4

### Contribution
4
