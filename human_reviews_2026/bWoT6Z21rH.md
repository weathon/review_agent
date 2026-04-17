# From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Unsupervised object-centric learning models, particularly slot-based architectures, have shown great promise in decomposing complex scenes. However, their reliance on reconstruction-based training creates a fundamental conflict between the sharp, high-frequency attention maps of the encoder and the spatially consistent but blurry reconstruction maps of the decoder. We identify that this discrepancy gives rise to a vicious cycle: the noisy feature map from the encoder forces the decoder to average over possibilities and produce even blurrier outputs, while the gradient computed from blurry reconstruction maps lacks high-frequency details necessary to supervise encoder features. To break this cycle, we introduce Synergistic Representation Learning (SRL) that establishes a virtuous cycle where the encoder and decoder mutually refine one another. SRL leverages the encoder's sharpness to deblur the semantic boundary within the decoder output, while exploiting the decoder's spatial consistency to denoise the encoder's features. This mutual refinement process is stabilized by a warm-up phase with a slot regularization objective that initially allocates distinct entities per slot. By bridging the representational gap between the encoder and decoder, SRL achieves state-of-the-art results on video object-centric learning benchmarks. Codes are available at github.com/hynnsk/SRL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces three novel approaches to improving the quality of segmentation maps in Slot-Attention based approaches. The authors note that the attention maps from the Slot Attention mechanism are sharp, but noisy, while the decoder provides maps that are not noisy but overly smooth. To alleviate these issues the authors propose adding two hierarchical contrastive losses over constructed sets of positive, semi-positive and negative feature map patches constructed from the segmentation maps from the Slot Attention and Decoder modules. The encoder’s contrastive loss tries to align the feature map prior to slot attention with the feature map from the decoder. The decoder similarly tries to align with patches from the encoder. Additionally, the authors propose an approach to alleviate over-clustering by regularizing slots. Overall, these approaches lead to gains on the reported segmentation metrics.

### Strengths
The paper describes the proposed approaches in a clear manner and performs sufficient ablations to show that each component contributes to improving segmentation performance. The presentation is good and the figures help to understand the method.

### Weaknesses
The paper should be more specific in their experimental setup.

It is unclear if the poor decoder segmentation maps are a result of the choice of MLP, or if other types of decoders suffer the same issue. 

Furthermore, it is unclear if this is a result of MSE, which the paper implies, and if it can be resolved through the use of a different reconstruction loss.

It should be stated which version of DINOv2 was used.

The authors make claims of the gradients flowing from the decoder being “corrupted” and
“low-frequency”. It would be good if the authors could give some evidence to back up this statement.

The paper does not provide examples of cases where the proposed approach fails and leaves room for improvement.

Some of the writing is superfluous. For example: “This is because under MSE, the loss is calculated from the square of the error, meaning that the penalty grows quadratically with the error’s magnitude.”

### Questions
- Do you use an MLP decoder as is alluded to in the introduction?
- Do other types of decoders suffer from the same overly smooth segmentation maps that you try to alleviate?
- Does using a different loss, e.g. L1, for the reconstruction objective, rather than MSE, lead to less blurry decoder segmentations?

### Soundness
2

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
5

### Summary
The authors identified an objectively existing phenomenon, which I believe widely observed but never addressed by most OCL researchers (including me): 
- The slot attention maps tend to segment the visual scene with noises;
- while the decoder attention maps tend to segment it with blurred boundaries.

Then they formulate the issues behind it:
- Noisy Encoder Reinforces the Decoder’s Blurriness; 
- Blurry Decoder Corrupts the Encoder’s Learning Signal.

Accordingly, the authors propose two ternary contrastive losses that make the most of intrinsic supervision signals between the slot attention and decoding. Specifically,
- The positive pseudo labels, which are the overlap between the slot attention and decoding;
- The semi-positive pseudo labels, which are the inconsistent segmentations between the slot attention and decoding;
  - Differently chosen for their two losses
- The negative pseudo labels, which are the different segmentations between the slot attention and decoding.

Respectively, they designed two losses to achieve:
- DENOISING CONTRASTIVE LEARNING: REFINING THE ENCODER VIA DECODER COHERENCE
- DEBLURRING CONTRASTIVE LEARNING: REFINING THE DECODER VIA ENCODER SHARPNESS

Combined with their novel slot regularization, under staged training, their method achieved sota on video datasets in object discovery tasks.

### Strengths
Intuitve formulation on a widely observed phenomenon and feasible solutions.

### Weaknesses
(Writing issues first but not most important)

W1
---
Line 017 "We identify that this discrepancy gives rise to a vicious cycle; the noisy ...": ";" should be ":".

W2
---
Line 248 Equation (4) vs (6): Compared with (6), Equation (4) seems missing the $\Sigma$ and $\frac{1}{|P|}$ on the first term.

W3
---
Line 276, 287 Equation (6) and (7): Should be written in one Equation, not two.

W4
---
Line 144-145 "representational conflict between the slot attention maps and **reconstruction maps**": "reconstrcution maps" is a misleading term (First sight, I was wondering if you are mentioning the reconstructed feature maps?), better to use ones like "reconstraction/decoding attention maps".

W5
---
Sect 3.1, "Noisy Encoder Reinforces the Decoder’s Blurriness" and "Blurry Decoder Corrupts the Encoder’s Learning Signal": These two parts lack mathematical or statistical analysis/probing. Providing such analyses will surely make your problem definition more convincing.

W6
---
Sect Experiment. No OCL results on image datasets. Why not testing on image datasets like COCO? Theoretically your method supports OCL on both images and videos.

W7
---
More hyperparameters, Line 341, i.e., $\lambda reg$ and $\lambda CL$, and top-$K$, as well as $M$ at some where. Anyway, not a big deal. Most top conference fancy works would introduce one or two hyperparameters.

W8
---
More computation cost. According the two contrastive losses, pairwise cosine similarities need to be calculated and the three subsets need to be determined. So there must be most latency and VRAM consumption. It is necessary to provide quantitative results on these.

W9
---
Slot regularization contributes a large part to the total performance gain as shown in Tab 2. But this is not included in the main storyline. 

W10
---
Slot regularization is similar to slot pruning-related techniques, which can boost the performance almost in any cases. Thus related works should be compared or at least discussed:
- SOLV: Self-supervised Object-Centric Learning for Videos. NeurIPS 2023
- MetaSlot: Break Through the Fixed Number of Slots in Object-Centric Learning. NeurIPS 2025.

W11
---
The proposed contrastive losses are actually a kind of **fine-grained** ***self-distillation***, compared with the following existing works. Thus related works should be compared or at least discussed:
- SPOT: Self-Training with Patch-Order Permutation for Object-Centric Learning with Autoregressive Transformers. CVPR 2024.
  - offline "self"-distillation on slot attention masks
- DIAS: Slot Attention with Re-Initialization and Self-Distillation. ACM MM 2025.
  - online self-distillation on slot attention masks
- SlotMatch: SlotMatch: Distilling Temporally Consistent Object-Centric Representations for Unsupervised Video Segmentation.
  - offline "self"-distillation on slots
- SmoothSA: Smoothing Slot Attention Iterations and Recurrences.
  - online self-distillation on slots

Interesting works are worth encouragement. So I will change my ratings if the authors can address my concerns.

### Questions
Please refer to the former part.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper modifies SlotContrast, a framework for extracting object-centric representations from pretrained vision foundation models, to produce sharper and less noisy outputs for each slot. The proposed approach consists of three stages: in the first stage (the initial 10% of training), slot–patch correspondence maps are extracted, based on which sets of positive and negative pairs are formed. The decoder loss is then regularized with a contrastive loss derived from these sets. In the second stage (10–20% of training), no regularization is applied. Finally, after 20% of training, the encoder features are regularized, but the sets are now extracted from the decoder’s object maps. Empirical results demonstrate improvements over the baseline, yielding sharper and cleaner maps.

### Strengths
1 - The problem discussed in the paper is an interesting observation, which has not been explored in the previous works. 

2 - The paper shows better empirical results than state-of-the-art.

3 - The paper is clean, and the method has been elaborated in details for different stages.

### Weaknesses
1 - Although I agree with the identified problem, the proposed solution appears overly complex. It consists of three stages defined by the proportion of training iterations completed. Since different backbones or even datasets may require varying numbers of iterations, the approach seems unlikely to generalize well. Could you please report results with other foundation models such as Dino3 [1] or Franca [2] to verify generalizability?

2 - Recently, several post-training methods have been introduced to address the issue of noisy dense features in foundation models [3, 4]. A simpler alternative might be to replace the backbone with such improved checkpoints and apply SlotContrast directly. This approach could provide a more natural and straightforward solution than introducing complex regularization. Could you also compare performance using these backbones?

3 - An ablation study on the parameter η, which determines the stage at which each loss function is applied, would be valuable. This would further demonstrate the robustness of the method under different training schemes.

1 - Dinov3." arXiv preprint arXiv (2025)

2 - Franca: Nested Matryoshka Clustering for Scalable Visual Representation Learning, arXiv preprint arXiv (2025)

3 - MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning, ICCV25

4 - Near, far: Patch-ordering enhances vision foundation models' scene understanding, ICLR25

### Questions
I explained my question in the weaknesses, please refer to them.

### Soundness
2

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
3

### Summary
This paper introduces Synergistic Representation Learning (SRL), a method designed to resolve the conflict between the sharp encoder attention maps and the blurry decoder reconstructions in unsupervised object-centric learning. SRL enables mutual refinement between the encoder and decoder, leveraging the encoder's sharpness to enhance the decoder's output and the decoder's spatial consistency to improve the encoder's features. The method is validated on three video object-centric learning benchmarks.

### Strengths
1. The paper presents a novel opinion: a vicious cycle between the encoder and decoder in video object-centric learning.
2. The paper is well-organized, and the experimental design is clear.

### Weaknesses
1. The authors propose a vicious cycle in unsupervised video object-centric learning, where noisy encoder inputs lead to blurry, low-frequency decoder outputs, which in turn fail to refine the encoder's features. However, it remains unclear whether this phenomenon truly exists during training, and whether it worsens as training progresses. Qualitative or quantitative experiments are necessary to justify this claim.
2. In the comparison experiments presented in Table.1, SRL does not show a clear advantage across multiple metrics (with three out of six metrics showing actually lower performance). It would be beneficial to evaluate the method's effectiveness on additional datasets.
3. The implementation details and ablation studies suggest that the model's performance on a benchmark is somewhat sensitive to hyperparameters, such as the number of positive patches (K) and the number of slots. This raises a question: whether the proposed method is task-specific.
4. How does the proposed method's efficiency (in terms of both time and space cost) compare to previous methods?
5. From the visualization results, both the proposed method and the comparison methods seem to produce relatively coarse object discovery and segmentation results, with limited delineation of object boundaries. Thus, what advantages do these methods offer over models specifically designed for object or instance segmentation in real-world scenarios?

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
