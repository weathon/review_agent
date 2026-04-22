# Faster Vision Transformers with Adaptive Patches

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 8

## Abstract
Vision Transformers (ViTs) partition input images into uniformly sized patches regardless of their content, resulting in long input sequence lengths for high-resolution images. We present Adaptive Patch Transformers (APT), which addresses this by using multiple different patch sizes within the same image. APT reduces the total number of input tokens by allocating larger patch sizes in more homogeneous areas and smaller patches in more complex ones.
APT achieves a drastic speedup in ViT inference and training, increasing throughput by 40\% on ViT-L and 50\% on ViT-H while maintaining downstream performance.
It can be applied to a previously fine-tuned ViT and converges in as little as 1 epoch.
It also significantly reduces training and inference time without loss of performance in high-resolution dense visual tasks, achieving up to 30\% faster training and inference in visual QA, object detection, and semantic segmentation.
We will release all code and trained models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes the Adaptive Patch Transformer (APT): it adaptively employs multiple patch sizes within the same image based on content—using larger patches for flat/redundant regions and smaller patches for detail-rich areas, thereby reducing input tokens and boosting throughput. Core Implementation: Multi-scale histogram entropy (Eq.(1)) serves as compressibility metric, while hierarchical thresholds (ω) determine whether to “stop” at a layer or continue subdivision. For large patches, dual-path information is simultaneously utilized: “Embedding resized to base dimensions + sub-patch embeddings aggregated via Conv,” fused through a zero-initialized MLP. Inference/training side: “Sequence packing + block-diagonal mask” accommodates variable sequence lengths, accelerated by FlashAttention. Experiments show: Full ImageNet and 1-epoch fine-tuning achieve 20%–90% throughput gains while maintaining accuracy, with accelerated performance on LLaVA's VQA, COCO detection, and ADE20K segmentation. Authors also report: Without token reduction, APT incurs non-zero overhead; zero-initialized connections deliver the most stable “plug-and-play” fine-tuning convergence.

### Strengths
1. Comparison tables are provided across four task categories—classification, VQA, detection, and segmentation—clearly demonstrating how variable tokens are transformed into regular feature maps for dense prediction tasks.

2. Key methodological formulas and structural diagrams are presented, along with component ablation studies (zero initialization vs. non-zero/residual; system overhead without compression) to facilitate reproducibility and pinpoint sources of improvement.

### Weaknesses
1. The adaptive patch size relies on a hierarchical entropy threshold (Eq. 1, §3.1), which is fixed and manually tuned for each scale. The paper gives no data-driven method to set these thresholds, and poor choices can cause information loss or accuracy drops.

2. To use FlashAttention, some baselines were re-implemented with modifications like disabling weighted attention (§4.1). These changes may alter results, so runtime and throughput comparisons might not be fair without full implementation details or code release.

3. The reconstruction method (§3.3) repeats large-patch features to form dense grids, which may create block artifacts and hurt small-object accuracy. The paper reports only overall mAP/mIoU and shows no failure examples, leaving fine-detail loss unverified.

4. Table 6 shows that APT is slower than the baseline when no compression is applied, suggesting preprocessing overhead from entropy computation and token packing. The paper omits CPU/GPU timing breakdowns, and speed gains are smaller than FLOPs reductions, implying memory or pipeline bottlenecks.

5. The method mainly benefits high-resolution, large-model settings. Key parameters (thresholds ω, binning strategy, search range) are missing, and code is unreleased, making reproduction and fair comparison difficult.

### Questions
1. How are the hierarchical thresholds (ωᵢ) determined for different datasets or tasks? Are they manually tuned or selected automatically?

2. How is the histogram entropy computed — what bin settings and value ranges are used? Have other texture or frequency-based criteria been compared?

3. What is the preprocessing overhead of multi-scale entropy computation and token packing on CPU and GPU? Are these times included in the reported runtime measurements?

4. Since token counts vary across samples, what is the distribution of sequence lengths, and how does this variation affect throughput and latency?

5. Can the object detection results be further analyzed by object size (small, medium, large) to better understand performance on fine-grained details?

6. Does the “repeat 2^{2i}” reconstruction step cause aliasing or checkerboard artifacts? Has any boundary or contour accuracy been evaluated to confirm visual quality?

7. How are positional embeddings handled for variable patch sizes and packed sequences? Is there any scale-aware interpolation or adjustment applied?

8. What is the parameter and memory overhead introduced by the zero-initialized MLP fusion layer, and how stable is it during longer fine-tuning?

9. For baselines adapted to support FlashAttention, can the authors provide a detailed list of implementation changes and verify that all models were tested under identical conditions?

10. Does the aggressive merging of smooth regions lead to errors when fine-grained or background details are required for prediction? Has any failure case analysis been performed?

**If you address my concerns, I will consider raising my score.**

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces an adaptive patching (tokenization) method for ViTs. The main idea is to measure the entropy (how variable the pixel values are) of local regions of images and use finer (smaller) patches to tokenize the regions with higher variation. The patches of different sizes are resized (and split) to the same size for embedding. It reduces the number of tokens compared to the ViT using the same-sized patches for the whole image. This method is adapted to different ViTs and used in different vision tasks such as ImageNet classification, VQA, and object detection. The method generally shows better accuracy-effeciency trade-off than the original ViTs and some efficient ViTs.

### Strengths
- This paper is well written overall; the adaptive patching idea (larger patches for low-entropy regions, smaller for high-entropy regions) is intuitive and easy to follow. The entropy formulation and hierarchical quadtree patchification are clearly described, with alternatives noted for the appendix.
- The zero-initialized MLP lets the model incorporate high-res details without hurting initialization, enabling quick convergence from existing ViTs.
- The proposed method APT plugs into several ViT backbones and tasks, including classification, VQA, detection, and segmentation. It also works with window attention (e.g., EVA/ViTDet).
- APT reported 40–50% throughput gains on large models/resolutions while matching accuracy, and also some speedups on dense tasks, achieving better accuracy-effeciency trade-off than several well-known efficient ViTs such as EViT and ToMe.
- The paper re-implements layer-level merging baselines with FlashAttention for a fairer comparison (and shows APT outperforms across compute budgets).

### Weaknesses
- Table 3 shows +22–26% throughput on LLaVA-1.5 (7B/13B), with some metrics slightly down (e.g., VQAv2 −0.6 for 13B) and others on par or up; overall the Pareto looks close but not strictly better across all benchmarks. Table 4 shows +14–30% throughput on detection with essentially unchanged mAP/AP50. This is positive, but the improvements are less decisive than in classification and would benefit from a Pareto plot analogous to Fig. 4 for these tasks.
- Currently APT uses entropy to measure the variation of pixels in image regions. It would be beneficial to add ablations for different measures, such as standard deviation of the pixels and local frequency (e.g., DCT-band energy).
- The writing of the experimental setup can be clearer, specifically on the difference between Full Fine-Tuning and Short Fine-Tuning (Section 4.2). Sometime it could be a bit confusing as to what is the pre-trained MAE; is it one only trained with masked autoencoding or one with both masked autoencoding and classification training?

### Questions
- For dynamic input size (seciton 3.3), you concatenate the tokens of a batch of images into a single sequence and use block attention. Why not padding the sequence of each image into the same length?
- In the Input-level Merging Baselines, what do you mean by saying Resizing represents a stronger version of Quadformer? It seems that Resizing is a variant of APT by removing the zero initialized layer, and is not really related to Quadformer. Similar question for "Random". Clearer explantion is needed.
- For the dense prediction tasks (section 4.3), you mentioned training only the newly added component. Does it mean only training the conv layers and ZeroMLP as in Fig 3? For example, for LLaVa, the language model and the projection layer are frozen. Is that correct?

### Soundness
3

### Presentation
3

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
This paper proposes Adaptive Patch Transformers (APT) , which accelerates Vision Transformers (ViTs) by replacing uniform patch splitting with content-aware, multi-granularity patching   based on entropy calculation . The method utilizes a Resize + ZeroMLP mechanism to fuse features from different scales into a unified embedding space, significantly reducing the input token count . The key contributions include achieving drastic throughput speedup (up to 50% on ViT-H) while maintaining accuracy across classification and dense prediction tasks, and ensuring   fast, stable adaptation to fine-tuned models via its zero-initialized fusion layer.

### Strengths
1. Strong Experimental Validation: The paper features a comprehensive set of ablation studies   across various tasks, effectively demonstrating the efficacy of the proposed mechanisms.
2. Significant Efficiency and Generalization: APT delivers substantial throughput improvements (up to 50% on ViT-H) on large models, exhibits fast convergence (1 epoch fine-tuning), and shows   robust generalization   across classification and dense prediction benchmarks.
3. Clarity and Presentation: The paper is well-structured, and the figures are high-quality.

### Weaknesses
1. Missing Similar Methods Comparison: The evaluation is incomplete because it fails to include a direct comparison against methods addressing the same task, such as MG-ViT[1] and PPT[2]. A detailed analysis of the methodological and empirical differences among APT, MG-ViT and PPT would substantially strengthen the paper.

2. Hyperparameter Dependency: The performance is sensitive to the entropy threshold , which appears to be a manually tuned hyperparameter . This dependency might complicate achieving optimal efficiency across different downstream tasks, as the definition of "salient information" can vary significantly between tasks.

3. In object detection tasks, does the use of entropy to determine patch size risk ignoring subtle object boundaries? Entropy measures pixel intensity distribution diversity, which may not perfectly align with semantically critical edges, especially when compared to gradient-based measures.

4. For higher resolution images, which naturally result in a larger number of base patches, could the authors explore further patch fusion/aggregation operations after the initial adaptive patching step, particularly when several adjacent low-entropy patches exhibit similar entropy scores

5. When patch size change, should entropy threshold change? That means one size patch correspond to one entropy threshold, and different size patch correspond to different entropy threshold.

[1]Zhang Y, Liu Y, Miao D, et al. MG-ViT: a multi-granularity method for compact and efficient vision transformers[J]. Advances in Neural Information Processing Systems, 2023, 36: 69328-69347.

[2]Wu, Xinjian, et al. "Ppt: Token pruning and pooling for efficient vision transformers." arXiv preprint arXiv:2310.01812 (2023).

### Questions
Refer to weakness. If these concerns are well addressed, I will raise the rating to a positive one.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The manuscript proposes Adaptive Patch Transformers (APT) considering using multiple different patch sizes within the same image processed by a Vision Transformer (ViT). Larger patch sizes are allocated in more homogeneous areas while smaller patches are allocated to more complex ones. The proposed approach accelerates the ViT with 40%.

Entropy is used as a measure of a patch’s compressibility with lower entropy indicates higher redundancy.

Patch aggregation is also employed aggregating embeddings from sub-patches back to size 

Token merging approaches is done at input-level. Input-level merging reduces tokens directly from image patches before entering the model. The method is also compared to Layer-level token merging.

### Strengths
* A more efficient and adaptive transformer model is proposed, where the adaptation refers to the fact that complex information is processed in more detail with smaller patches. Meanwhile less complex regions are processed with larger patches.
* Extensive experimental results are provided.

### Weaknesses
* Lack of theoretical analysis
* Lack of computational analysis

### Questions
How is the adaptive patch sizes work with other transformer models, such as for example the Swin transformer.
Z. Liu et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, ICCV 2021

Can other methods be used of detecting complex regions for using smaller patches?

### Soundness
3

### Presentation
3

### Contribution
3
