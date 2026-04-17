# Generalizable and Consistent Granular Edge Prediction

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
We introduce a new task in edge detection: Granular Edge Prediction. Unlike traditional binary edge maps, this task aims to predict a categorical edge map, where each edge pixel is assigned a granularity level reflecting the likelihood of being recognized as an edge by a human annotator. Our contributions are threefold: 1) we construct a large-scale synthetic dataset for granular edge prediction, where each edge is labeled with a quantized granularity level, and introduce a graph-based edge representation to enforce consistency in edge granularity across the dataset, 2) we develop a novel edge consensus loss to enforce granularity consistency within individual edges, and 3) we propose a comprehensive evaluation framework, including granularity-aware edge evaluation and two quantitative metrics to assess the consistency of granular edge prediction. Extensive experiments demonstrate that our method generalizes well in zero-shot evaluation across four standard edge detection datasets, closely aligns with human perception of edge granularity, and ensures high consistency in edge-wise granularity estimation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper constructs a synthetic dataset for granular  edge detection and proposes an efficient model that outputs edge maps of multiple granularity levels in a single forward pass, treating the task as a multi-class prediction problem. In addition, the paper introduces an edge consensus loss to enforce granularity coherence within edges, as well as  granularity-aware edge evaluation to demonstrate the effectiveness of the proposed approach.

### Strengths
1. The dataset is built by first extracting object boundaries using SAM and then performing refinement via graph-based representation, resulting in a large-scale granularity-aware edge detection dataset.
2. The proposed formulation effectively avoids the need for multiple forward passes required by previous methods to infer different granularity levels. By converting the problem into a multi-level classification task, the method can produce all granularity results with a single inference.
3. A new loss function and evaluation metric are introduced, which can potentially promote further research in granularity-aware edge detection.

### Weaknesses
1. Choice of granularity levels. The dataset is initially annotated with 36 granularity levels, but experiments later quantize these into 4 or 6 levels due to distribution imbalance (line 325–327). (1) How were the *original 36 levels* chosen? What advantage does the approach of annotating 36 levels *first and then merging* have over *directly annotating fewer granularity levels*? (2) When the granularity level is extremely small (level = 1), how does this differ from conventional deterministic edge detection? (3) In Fig. 12, one would expect that the smallest granularity level preserves only instance-related boundaries without background clutter. however, the results still appear influenced by background textures. Could the authors comment on this?
2. Model architecture clarification. The main paper does not clearly describe the model architecture. It only becomes clear in Appendix Table 3 that it follows a diffusion-model U-Net backbone. Were pretrained diffusion model weights used?
3. Inconsistent cross-dataset generalization. In Table 1, MuGE trained on SGED performs worse on NYUD compared to zero-shot inference from models trained on BSDS. However, performance improves on BIPED and Multicue. (1) Does this suggest that SGED yields weaker cross-domain generalization than BSDS? (2) Similarly, in Table 7, the performance gap between naive SAM and GEPM is small on BSDS and NYUD but large on BIPED and Multicue. Why does SGED training benefit some datasets but not others? (3) How does the current SOTA DiffusionEdge model perform when trained on SGED and tested on BSDS?
4. Model size comparison. In Table 7, what architecture does *naive SAM* refer to? Is it SAM-ViT-B? The base model used in the paper contains substantially more parameters than both SAM-ViT-B and the EfficientNet models used in MuGE. A direct parameter comparison table would clarify fairness in evaluation.
5. Class imbalance handling (Eq. 2). Since larger granularity levels have fewer annotated samples, does Eq. 2 account for this imbalance? Would class-balanced weighting or focal-type reweighting help further alleviate the imbalance noted in line 325–327?

### Questions
The strength of this paper lies in proposing a dataset, introducing a granularity consistency loss, and transforming prediction tasks of different granularities into multi-class classification tasks.

The main issue, however, lies in the experiments. The results on BSDS are nearly comparable to those of SAM's zero-shot performance. Additionally, the constructed dataset was expected to be entirely at the instance level when the granularity is set to 0, but this requirement has not been fully met.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new edge detection paradigm, termed Granular Edge Prediction (GEP), which redefines edge detection from binary classification into a granularity-aware prediction problem. Instead of simply detecting whether a pixel belongs to an edge, the proposed framework predicts an edge granularity level that reflects edge consistency and perceptual thickness.

To support this task, the authors construct a large-scale synthetic dataset called SGED, generated via SAM 2 segment masks and multi-level granularity transformations. They also design a novel Generalizable Edge Prediction Model (GEPM) equipped with a graph-based edge representation ensuring per-edge consistency, and an Edge Consensus Loss that enforces distributional agreement along the same edge.

### Strengths
1. Results show that GEPM achieves near or better than supervised methods in zero-shot settings on several benchmarks.

2. The Edge Consensus Loss based on Jensen–Shannon divergence effectively enforces intra-edge consistency in predictions.

### Weaknesses
1.  SGED is entirely synthetic; the paper lacks rigorous validation of how well its granularity annotations align with human perceptual judgments.

2. The distinction between “granular consistency” and previously studied “multi-level edge fusion” (e.g., in UAED or MuGE) could be more clearly articulated.

3. The quality of the SGED dataset depends heavily on SAM’s segmentation quality, which may introduce structured bias.

### Questions
1. How closely do the generated granularity annotations approximate human perceptual edge thickness? Have user or psychophysical studies been considered?

2. Could SGED overfit to SAM’s segmentation priors, limiting real-world generalization?

3. What is the training cost (GPU hours, parameter count), and how scalable is the proposed GEPM framework?

4. Would combining GEP with downstream tasks (e.g., boundary-aware segmentation) improve overall performance?

5. How sensitive is the model to the granularity level discretization (e.g., 36 vs. 10 levels)?

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
4

### Summary
This work systematically defines and solves the "granular edge prediction" task for the first time, constructs the first large-scale, structured granular edge dataset (SGED) and a novel edge consensus loss and a comprehensive evaluation framework, providing a new paradigm for subsequent edge perception, controllable generation, and other tasks.

### Strengths
This work construct a large-scale synthetic dataset for granular edge prediction, where each edge is labeled with a quantized granularity level, and introduce a graphbased edge representation to enforce consistency in edge granularity across the dataset. The approach develop a novel edge consensus loss to enforce granularity consistency within individual edges, and propose a comprehensive evaluation framework, including granularity-aware edge evaluation and two quantitative metrics to assess the consistency of granular edge prediction.

### Weaknesses
1. The paper mentions "making it particularly valuable and has potential for applications where edge prominence varies," which shows that the authors recognize the importance of downstream applications. However, if the full text does not include specific experiments.
2. The SGED dataset relies on the Segment Anything Model (SAM) to generate synthetic edges. However, the core capability of SAM is to detect object boundaries, which leads to the dataset's insufficient capture of "non-object edges" (such as textures, contour boundaries, and edges with severe material differences).

### Questions
The core value of granular prediction is to support downstream tasks. Have the authors tried applying the granular output of GEPM to depth estimation or artistic rendering? Are there any empirical results demonstrating "performance improvement in downstream tasks "?

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
3

### Summary
This paper introduces Granular Edge Prediction, extending binary edge detection to predict granularity levels reflecting perceptual saliency. Main contributions: (1) SGED dataset with 376K SAM-generated images and graph-based refinement for consistency; (2) GEPM model with Edge Consensus Loss; (3) new consistency metrics. The method achieves competitive zero-shot performance across four benchmarks. However, the work lacks human validation, inherits SAM's biases toward object boundaries, and shows limited advantages over supervised methods.

### Strengths
1. The granular edge prediction task is well-motivated, addressing the inherent subjectivity in edge annotation with clear practical applications. 
2. Creating 376K images addresses severe data scarcity (only 600 in existing datasets), enabling robust training and generalization. 
3. Table 1 shows competitive cross-dataset results, with Multicue ODS of 0.843 approaching supervised methods (0.904).

### Weaknesses
1. Zero human studies validating predicted granularities align with perception. 
2. Critical omission for a paper claiming to predict "human-recognized" edge granularity. 
3. Some figures hard to see, granularity reduction (36→6) under-motivated in main text.

### Questions
1. Can you provide human studies validating predicted granularities?
2. What is supervised fine-tuning performance?
3. When is zero-shot granular prediction preferable to supervised binary detection?

### Soundness
3

### Presentation
3

### Contribution
3
