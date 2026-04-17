# VISTA: Visual-Semantic Disentanglement and Dynamic Spatial-Temporal Asynchrony for Brain Decoding

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Electroencephalogram (EEG) offers a portable, low-cost, and millisecond-scale window into neural dynamics, making it an attractive alternative to functional magnetic resonance imaging (fMRI) for real-world brain visual decoding. Yet fine-grained visual representations and high-level semantic concepts emerge in distinct temporal intervals and spatial topology connections, creating asynchronous patterns that hinder their joint visual-semantic disentanglement. We present VISTA, an EEG-centric neural decoding framework that disentangles visual-semantic modalities along asynchronous spatial-temporal dimensions. Temporally, VISTA divides EEG into non-overlapping time patches and employs an attention mechanism to assign soft weights to each slice, enhancing EEG to capture the heterogeneous temporal distributions of visual and semantic activations. Spatially, it learns modality-specific brain topology connections and derives spatial representation via low-rank decomposition and normalized Laplacian spectral decomposition. The resulting visual and semantic embeddings are each aligned with CLIP’s image and text spaces to leverage rich pretrained knowledge. On the large-scale and widely used EEG-visual dataset THINGS-EEG, VISTA outperforms prior EEG methods in zero-shot object recognition. Moreover, on the magnetoencephalogram (MEG) dataset THINGS-MEG, it demonstrates cross-modal generality beyond EEG, achieving comparable gains. Our results underscore the value of asynchronous, disentangled feature extraction and cross-modal alignment for robust neural decoding. Code and pretrained models will be available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
VISTA presents a novel EEG-based neural decoding framework that disentangles visual and semantic representations of brain activity while modeling their inherent spatial-temporal asynchrony. The model divides EEG signals into time patches to capture asynchronous temporal activations and learns separate brain connectivity graphs for visual and semantic modalities via low-rank and Laplacian spectral decomposition. These representations are then aligned with CLIP’s image and text embeddings through contrastive learning to leverage large-scale visual-semantic priors. Experiments on THINGS-EEG and THINGS-MEG datasets show that VISTA achieves state-of-the-art zero-shot object recognition accuracy.

### Strengths
1. VISTA explicitly models asynchronous neural dynamics by separating EEG into time-specific and spatially distinct pathways for visual and semantic features, providing biologically plausible and interpretable representations.
2. By aligning EEG-derived embeddings with CLIP’s image and text spaces, the model leverages pre-trained multimodal representations for zero-shot decoding, enhancing generalization to unseen categories.
3. The framework achieves consistent state-of-the-art performance across EEG and MEG datasets.

### Weaknesses
1. The multi-branch architecture involving spatial-temporal encoders, graph learning, and CLIP alignment increases training overhead and may limit practical scalability for real-time brain–computer interfaces. Also, such an over-complex architecture seems lack principal novelty and application potential
2. The reliance on CLIP embeddings ties performance to the representational quality and domain coverage of CLIP, potentially restricting generalization to novel or non-visual cognitive tasks.

### Questions
1. How well do the learned visual and semantic brain networks correspond to actual cortical pathways known from neuroscience, and can this be quantitatively validated?
2. Could VISTA’s disentanglement and alignment approach extend to other cognitive decoding tasks (e.g., language comprehension or emotion recognition)?

### Soundness
3

### Presentation
3

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
The paper proposes VISTA, an EEG visual decoding framework that explicitly disentangles visual and semantic components and models their spatio-temporal asynchrony. The disentangled EEG embeddings are then aligned with CLIP’s image and text spaces through contrastive learning for both retrieval and reconstruction tasks.

### Strengths
1. Clear motivation from neurocognitive mechanism.
2. The overall framework design is well visualized. The experiments are extensive, showing solid performance in single-subject settings and reasonable generalization across subjects.

### Weaknesses
1. While the paper effectively introduces separate visual and semantic branches that correspond to human visual processing pathways, the core of visual decoding task is to improve retrieval accuracy or reconstruction fidelity. The two branches are evaluated independently, and the paper does not explore how their fusion could further enhance decoding performance.
2. The experimental comparison lacks several SOTA baselines, such as UBP[1].
3. A few minor issues:
  - L291: should be 1654 categories.
  - Section 4.4.1 and 4.4.3: discuss results not reflected in Table 4.
  - FLOPs inconsistency: VISTA shows 89.9M in Table 6 but 79.9M in L1192.
  - Appendix A.3.3: contains duplicated content.
[1] Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior

### Questions
1. The Introduction mentions that the temporal attention is supervised by a Gaussian-kernel distance loss, but the paper does not provide a formal definition or implementation details.
2. Although not explicitly stated in the main text, Figure 3 suggests that the two convolutional layers share params. What is the motivation for this design?
3. In Section 4.4.2, what exactly does “both off” mean? How are the two modules disabled during ablation?
4. Could the authors elaborate on how Figure 5 was visualized? Specifically, what data and methods were used?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduce VISTA, a framework for neural decoding. This approach disentangles the visual-semantic modalities and incorporate temporal-spatial modeling to capture brain representations. The framework is evaluated on THINGS-EEG and THINGS-MEG datasets, reporting improvements over prior work such as CognitionCapturer.

### Strengths
1. The proposed framework employs an attention mechanism to capture temporal representations and models spatial asynchrony through brain network learning and  Laplacian decomposition.
2. This paper provide comprehensive experimental validations, on EEG and MEG datasets. Moreover, the authors analyzes the temproal asynchrony modeling on visual and semantic information.
3. The step-by-step breakdown of the model's components (EEG encoder, attention mechanism, spatial-temporal modeling) makes the methodology clear and reproducible.

### Weaknesses
1. The novelty of this work is fair, since extracting the low-level and high-level representations, spatial-temporal modeling, contrastive learning are commonly used in viusal decoding frameworks. 
2. In Section 4.4.1, the authors describe “visual-only” and “semantic-only” variants of their model. However, these ablation settings are not reported in Table 4. Moreover, the paper lacks a clear comparison between these simplified configurations and the full VISTA model.
3. The paper does not clearly specify the inference protocol for different tasks: in particular, it remains ambiguous whether the visual and semantic embeddings are fused into a single representation or used separately depending on the target task.

### Questions
1. Does the design of shared parameters affect the decoupling effect? The paper mentions that shared parameters were used in the EEG encoder, which may reduce the independence between visual and semantic features to some extent. 
2. In the provided Figure 4, the visual and semantic information appears to overlap at certain time intervals. Given this overlap, how does the model ensure effective disentanglement during these periods? If visual and semantic representations are mixed at certain time points, does this impact the disentangling process, and if so, how does the model address this issue?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an EEG visual decoding framework named VISTA, which aims to address the problem of spatio-temporal asynchrony between visual and semantic information in EEG signals. By partitioning the EEG signal into non-overlapping temporal segments and applying a weighted attention mechanism, VISTA achieves asynchronous modeling of visual and semantic components in the temporal dimension. Concurrently, it constructs visual- and semantic-specific brain network graphs and performs graph Laplacian spectral decomposition to achieve asynchronous modeling in the spatial dimension. Finally, VISTA aligns the visual and semantic representations from EEG with the image and text spaces of CLIP, respectively, enabling cross-modal contrastive learning. Experiments are conducted on the THINGS-EEG and THINGS-MEG datasets, validating VISTA's superior performance on zero-shot object recognition tasks.

### Strengths
1.Originality and Innovation: introduces the concept of visual-semantic decoupling, overcoming the limitations of traditional EEG decoding methods that confound these two streams of information.

The introduction of "spatio-temporal asynchronous modeling" aligns with neuroscientific findings regarding the different processing timelines for visual and semantic information.

2.High-Quality Methodology: the temporal modeling employs a soft-gating mechanism, preventing information loss. The spatial modeling uses graph Laplacian spectra to extract structural information, enhancing spatial representation capabilities. The alignment mechanism with CLIP effectively leverages large-scale pre-trained knowledge.

3.Sufficient Experimentation: effectiveness is validated on both EEG and MEG modalities. Detailed ablation studies confirm the contribution of each module. Auxiliary experiments, such as parameter analysis and visualizations, enhance interpretability.

4.Practical Significance of Results: The zero-shot recognition capability indicates strong generalization potential. Image reconstruction experiments demonstrate potential application value in BCI and neural interfaces.

### Weaknesses
1.The motivation for the temporal segmentation is not sufficiently strong. Given that visual and semantic features appear sequentially in the EEG signal, wouldn't applying contrastive learning to the entire signal segment be more effective for adaptively learning the correspondence between semantic/visual EEG features for different object stimuli?

2.The experimental results are not significantly better than those of methods from the last two years. This (relative lack of significant improvement) somewhat diminishes the paper's technical contribution.

3.Lack of cross-dataset generalization validation: Although the authors validated their method on THINGS-EEG and THINGS-MEG, it was not tested on other EEG datasets (e.g., ImageNet-EEG or DEAP). This limits the verification of its generalizability.

### Questions
1.Cross-Dataset Generalization: Have the authors considered validating VISTA's generalizability on other EEG datasets (e.g., DEAP, ImageNet-EEG)? Is there a risk of overfitting to the THINGS datasets?

2.Semantic Embedding Improvement: The semantic representations perform relatively weakly on fine-grained tasks. Have the authors considered incorporating more granular text descriptions (e.g., attributes, parts) to enhance semantic modeling?

3.Handling Individual Differences: While the model performs well in cross-subject tasks, have the authors considered methods to further model inter-subject variability (e.g., adaptation layers, meta-learning)?

4.Control Mechanisms for Image Reconstruction: In the image reconstruction task, how can the semantic consistency of the generated images be better controlled? Have the authors considered introducing stronger conditional controls (e.g., text guidance or attention-based control)?

### Soundness
3

### Presentation
2

### Contribution
2
