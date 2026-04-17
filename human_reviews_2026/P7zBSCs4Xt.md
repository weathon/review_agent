# FastAvatar: Towards Unified and Fast 3D Avatar Reconstruction with Large Gaussian Reconstruction Transformers

- Decision: Accept (Poster)
- Scores: 4, 4, 2, 8

## Abstract
Despite significant progress in 3D avatar reconstruction, it still faces challenges such as high time complexity, sensitivity to data quality, and low data utilization. We propose~\textbf{FastAvatar}, a feedforward 3D avatar framework capable of flexibly leveraging diverse daily recordings (e.g., a single image, multi-view observations, or monocular video) to reconstruct a high-quality 3D Gaussian Splatting (3DGS) model within seconds, using only a single unified model. The core of FastAvatar is a Large Gaussian Reconstruction Transformer (LGRT) featuring three key designs: First, a 3DGS transformer aggregating multi-frame cues while injecting initial 3D prompt to predict the corresponding registered canonical 3DGS representations; Second, multi-granular guidance encoding (camera pose, expression coefficient, head pose) mitigating animation-induced misalignment for variable-length inputs;
Third, incremental Gaussian aggregation via landmark tracking and sliced fusion losses. Integrating these features, FastAvatar enables incremental reconstruction, i.e., improving quality with more observations without wasting input data as in previous works.  This yields a quality-speed-tunable paradigm for highly usable 3D avatar modeling. Extensive experiments show that FastAvatar has a higher quality and highly competitive speed compared to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a feedforward framework for rapid 3D avatar reconstruction based on 3D Gaussian Splatting (3DGS). It can generate high-quality avatars from diverse inputs (single image, multi-view, or video) within seconds using a single unified model. The proposed Large Gaussian Reconstruction Transformer (LGRT) introduces (1) a 3DGS transformer for canonical reconstruction, (2) multi-granular guidance to handle pose and expression variations, and (3) incremental Gaussian aggregation for quality refinement. Experiments demonstrate that FastAvatar achieves good quality and speed compared to existing approaches.

### Strengths
1. Supports the reconstruction of high-quality 3D heads from various input types, including a single image, multiple frames, or multi-view observations.
2. Designs a full-attention mechanism and a differentiable sampling strategy to effectively fuse image information from multiple inputs.
3. Proposes a sliced fusion loss to further optimize reconstruction quality.

### Weaknesses
1. Ambiguous description: In Lines 200–209, the authors state that h_i represents the concatenated feature of a single image I_i with its expression z_i^{exp}. However, in Figure 2, h_i appears to denote the feature corresponding to different image/expression pairs under the same camera pose. The authors should clarify this inconsistency.
2.  Missing ablation studies: In Lines 234–236, the authors claim that the naïve fusion strategy underperforms compared to their proposed Gumbel-Softmax sampling. However, no ablation study is provided to substantiate this claim. The authors are encouraged to include experiments quantifying the contribution of the Gumbel-Softmax sampling to the final performance.
3. Lack of visual comparisons: The supplementary video only presents results from the proposed method. Including visual comparisons with other state-of-the-art approaches would more convincingly demonstrate the superiority of the proposed technique.
4. Unclear contribution of key modules: It remains unclear how much the full-attention mechanism and the final Canonical 3DGS Model Fusion contribute to multi-frame aggregation. The authors are encouraged to perform an ablation study or provide quantitative evidence demonstrating their effectiveness. Since multi-view and multi-frame fusion form a core contribution of this paper, a more detailed analysis and discussion of these components would significantly strengthen the work.
5. Lack of ethics statements.

### Questions
1. Why does your method achieve faster inference speed than LAM? Since multi-view fusion involves full attention and sampling operations, wouldn’t this introduce significant additional computational overhead?

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
4

### Summary
The paper presents a feed-forward 3D avatar head reconstruction framework that performs high-quality 3DGS-based modeling within seconds from arbitrary-length inputs (single/multi-view or video). They propose Large Gaussian Reconstruction Transformer (LGRT), which integrates multi-granular positional encoding (camera, expression, pose), and Landmark Tracking Loss and Sliced Fusion Loss, enabling incremental reconstruction. Extensive experiments on multiple datasets demonstrate that FastAvatar achieves superior reconstruction quality and rendering fidelity compared to state-of-the-art works.

### Strengths
- The proposed LGRT architecture is effective in aggregating multi-view / multiple image cues and aligning variable-length inputs, achieving consistent geometric and appearance coherence across frames.
- The method shows substantial quantitative improvements in PSNR/SSIM/LPIPS, outperforming existing baselines across all view settings (1, 4, 8, and 16 frames).

### Weaknesses
- The tracking loss (Eq. 9) includes the term $y$, but its definition is missing. It should explicitly state how ground-truth and predicted landmarks $y_{j,i}$, $\hat{y}_{j,i}$ are obtained.
- Similarly, Eq. 10 introduces $L_{\text{mask}}$, but no definition or explanation of this loss term is provided. A clarification of its role and formulation is necessary.
- The framework claims to support arbitrary input lengths, but experiments are limited to at most 16 views. It is unclear whether the model generalizes beyond 16-frame inputs or whether GPU memory becomes a constraint.
- The paper highlights incrementality as a core advantage; however, quantitative improvements saturate after 4 views (Table 1). The authors should discuss why the performance gain diminishes beyond 4~8 views.
- In Table 2, the ablation only considers fixed-view settings. Since Sliced Fusion Loss is designed for multi-input benefit, it would be informative to show how its impact scales with increasing frame counts (e.g., 1, 4, 8, 16 frames w/ or w/o the Sliced Fusion Loss).

I will reconsider the score when all those concerns are handled well.

### Questions
What are typical failure cases observed in reconstruction or animation (e.g., inconsistent geometry, occluded regions, or identity drift) in this work?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes FastAvatar, a feedforward framework for fast and unified 3D avatar reconstruction from variable-length inputs (single images, monocular sequences, or sparse multi-view frames). The core model, termed LGRT, uses alternating frame/global attention blocks and injects camera pose, expression, and head-pose encodings into image tokens. The model directly predicts 3DGS attributes for avatar reconstruction, while two new losses, including sliced fusion loss and landmark tracking loss, encourage consistent multi-frame fusion and geometric alignment. The paper claims incremental reconstruction capability, where quality improves as more observations are provided.

### Strengths
- Practical motivation: Addresses fast, unified avatar reconstruction from variable-length inputs without per-identity optimization.

- Architecture: Extends VGGT with multi-granular encodings (pose, expression, camera) suitable for dynamic faces.

- Loss design: The sliced fusion and landmark tracking losses are reasonable to promote frame consistency and alignment.

- Feedforward inference potentially enables better avatar generation compared to optimization-based methods.

### Weaknesses
- Dependence on external camera pose tracking:
Unlike VGGT, which infers relative geometry and camera pose implicitly through attention, FastAvatar requires explicit camera parameters and FLAME-derived head/expression tracking as inputs. This reliance on external preprocessing weakens the method’s claim of being a fully feed-forward, generalizable system. In practice, the need for accurate tracking limits applicability in real-world scenarios (e.g., in-the-wild videos) and reduces the robustness advantage that a unified transformer architecture should ideally provide.
- Methodological inconsistency: Despite criticizing parametric proxies, the method depends heavily on FLAME tracking for pose/expression priors, limiting robustness and novelty.
- Visual evidence lacking:
The qualitative effects aren’t striking. Some reconstructions lack detail and natural dynamics (see video).
All qualitative examples show mainly frontal or near-frontal views, without large-angle or side-view results. This omission makes it difficult to assess whether the method truly captures consistent and complete 3D geometry rather than merely fitting frontal appearances. Given that the paper claims to produce fully animatable 3D avatars, the absence of wide-angle and rotational visualizations significantly weakens the empirical validation of this claim.
- Incremental claim overstated: The model recomputes from all available inputs rather than performing genuine streaming updates.
- Evaluation issues:
   - Modified baselines (e.g., LAM) may yield unfair comparisons.
   - Missing key baselines such as Arc2Avatar, HeadGap, and SynShot.
   - no identity evaluations.
   - The video results are too short.

### Questions
- During fusion, how exactly does the Gumbel-Softmax operate? Does it directly set the Gaussian attribute masks for some frames to zero? Please provide more details about its mechanism.
- As the number of input images increases, the performance improvements are limited; interestingly, in Figure 4, the visual results become sharper with more input views, which contradicts the common expectation that multiple views usually cause smoothing. Could this be related to some Gaussian attribute fusion strategy? Where do the authors think this gain originates from?
- Training time? The paper does not report key training details such as the hardware configuration, batch size, training time, or number of GPU hours required for training.
- How many Gaussians are predicted per frame and per fused model? Are they anchored to FLAME vertices or generated freely in canonical space?
- How are LBS weights for Gaussians derived—learned, interpolated from FLAME, or transferred directly?
- What does “incremental” precisely mean? Does the system reuse previous results or recompute from scratch with new inputs?
- Why was LAM modified for multi-frame input, and how was fairness ensured in this comparison?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
FastAvatar is introduced as a novel feedforward framework for 3D avatar reconstruction, designed to overcome existing challenges in high time complexity, sensitivity to data quality, and inefficient data utilization in contemporary methods.

This single unified model flexibly leverages diverse daily recordings, such as a single image, multi-view observations, or monocular video, to reconstruct a high-quality animatable 3D Gaussian Splatting (3DGS) model typically within seconds. The central component of the framework is the Large Gaussian Reconstruction Transformer (LGRT), which is specifically adapted from the VGGT structure to meet the demanding registration and aggregation needs of 3D avatar tasks. The LGRT incorporates designs including a 3DGS transformer that aggregates multi-frame cues by injecting initial 3D positional prompts, and multi-granular guidance encoding utilizing camera pose, expression coefficients, and head pose to mitigate animation-induced misalignment for variable-length inputs. Crucially, the model utilizes a Landmark Tracking Loss and Sliced Fusion Loss during training, which supervise the combination of frame-wise Gaussian representations to ensure consistency and enhanced aggregation accuracy.

Integrating these contributions, FastAvatar uniquely pioneers incremental 3D avatar reconstruction, allowing the model to continuously ingest new observational data to progressively refine modeling quality while achieving highly competitive quality and speed compared to existing state-of-the-art approaches.

### Strengths
- FastAvatar uses a single unified model capable of processing diverse daily recordings, including a single image, multi-view observations, or monocular video. FastAvatar demonstrates greater model flexibility and higher data utilization efficiency compared to other feedforward methods like LAM or Avat3r.

- Sliced Fusion Loss is a key component of the FastAvatar framework. This loss enables the model $G$ to leverage richer information from multiple inputs and to handle an arbitrary number of frames. It supervises the fusion of per-frame Gaussian representations, ensuring cross-frame consistency.

- FastAvatar introduces the capability for incremental 3D avatar reconstruction, a feature currently unattainable by existing approaches. This means the model can continuously ingest new observational data to progressively and reliably refine modeling quality.

### Weaknesses
- Although the method handles variable lengths, the practical input size N is explicitly limited. This constraint (max 16 frames) suggests that processing longer videos (which previously required 30 seconds at 25fps, for optimization-based methods) still requires sampling or chunking, potentially limiting true incremental modeling for extended footage.

- The precision of these proxy models (FLAME/3DMM) is known to be sensitive to limitations like representational capacity and data quality, often failing to produce highly accurate proxy 3D models. Although FastAvatar uses this information for alignment, the quality of the final reconstruction is fundamentally tied to the accuracy of these initial estimates. The current ablation study only focuses on the two proposed losses ($L_{track}$  and $L_{sliced}$). It is recommended to include an ablation on the estimated FLAME parameters. Measuring performance degradation would quantitatively assess robustness to real-world FLAME tracking inaccuracies. For example, the authors could test different tracking methods. When tracking is erroneous, does the method propagate these errors, or can the multi-frame image evidence mitigate them?

### Questions
The specific contribution of the initial 3D prompt, which distinguishes this approach from other Transformer designs, is not quantitatively demonstrated. The description is unclear: is the LGRT initialization based on standard FLAME vertices, subject-specific identity-shaped FLAME, or another form? Clarification and discussion would be helpful.

### Soundness
4

### Presentation
3

### Contribution
3
