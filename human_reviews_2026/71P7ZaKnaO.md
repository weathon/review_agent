# BoundaryDPT: Pushing the Boundaries of Depth Pruning for Vision Transformers

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
While prior studies have successfully compressed vision Transformers (ViTs) through various pruning techniques, most have concentrated on width pruning to achieve significant reductions in model size. Depth pruning, which involves the removal of entire layers from a ViT, is notoriously difficult for accuracy recovery, although depth pruning usually leads to higher speedups of compressed ViTs. Consequently, existing joint approaches that incorporate both width and depth pruning have exhibited limited acceleration ratios due to the inefficiencies of previous depth pruning methods.

To tackle the challenges in depth pruning, this work introduces BoundaryDPT, a novel depth pruning method by targeting redundancy of both attention layers and non-linearity within ViTs. To the 
best of our knowledge, we are the first to propose the pruning of activation function layers in ViTs. By reducing the redundancy of nonlinearity, instead of directly targeting linear layers in ViTs, the depths of ViTs are naturally reduced without incurring dimension mismatch. Moreover, we present a two-stage joint pruning method designed to address the heterogeneity of attention layers and activation function layers.

Comprehensive experiments on ImageNet1k, CIFAR-100, and ADE20K have validated our methods. Firstly, BoundaryDPT achieves a 1.58$\times$ speedup for DeiT-B while maintaining accuracy, and a 1.39$\times$ speedup for DeiT-S with nearly lossless accuracy degradation. Furthermore, when combined with width pruning (referred to as BoundaryDPT+), our method sets a new state-of-the-art record in ViT pruning. For instance, BoundaryDPT+ enhances the acceleration ratio from 4.24$\times$ to 5.19$\times$ for the Isomorphic-Pruning-2.6G configuration while maintaining near-lossless accuracy, establishing new benchmarks in extreme ViT compression.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a solution to the deep pruning challenge in Visual Transformers (ViTs), achieving optimization by removing entire layers rather than reducing width. The authors propose BoundaryDPT, a structured pruning framework that decouples attention layer pruning from activation layer (GELU) pruning. A lightweight Model Accuracy Predictor (MAP) guides pruning ratios for both layers, while learnable importance scores filter specific layers. Post-pruning depth reduction is further achieved through fine-tuning and linear layer fusion. On the ImageNet-1K dataset, BoundaryDPT achieves up to 1.6x acceleration with no accuracy loss. Its extended variant (BoundaryDPT+) delivers over 5x acceleration while maintaining orthogonality with width pruning and label pruning methods.

### Strengths
1. This paper proposes a decoupled pruning strategy that simultaneously considers attention layers and activation layers, addressing their distinct pruning behaviors.
2. The maximum a posteriori probability (MAP)-based ratio estimation method provides efficient guidance for pruning decisions without requiring exhaustive search.
3. Experiments across multiple ViT variants demonstrate that this approach achieves significant computational reduction while incurring minimal accuracy degradation.

### Weaknesses
1.   The paper relies on a quadratic polynomial predictor (MAP) fitted to a small sample size to determine pruning ratios, but its prediction error has not been systematically evaluated. This approach may fail across datasets or different architectures, fundamentally compromising the accuracy of pruning strategies.

2.   The paper overlooks residual and LayerNorm structures in ViT. The assumption of directly merging linear layers may not hold in Pre-Norm or gated variants, introducing structural risks.

3.   The core finding claiming faster recovery for activation layer pruning versus slower recovery for attention layer pruning is based solely on short-term (10 epoch) fine-tuning. It remains unverified whether this holds under longer training durations or different hyperparameters.

4.   Experiments primarily focus on ImageNet classification tasks, with insufficient evaluation of performance in detection, segmentation, robustness, or out-of-distribution scenarios. The conclusion that “it is orthogonal to token pruning” lacks adequate validation.

5. Training and inference utilize different hardware platforms (Ascend NPU vs NVIDIA GPU), leading to inconsistent sources for some throughput results. Additionally, the MAP and TE sampling processes are complex and not fully open-sourced, making result reproducibility challenging.

### Questions
**Q1:** The paper uses a quadratic polynomial as the Model Accuracy Predictor (MAP). Does this MAP require refitting across different architectures (e.g., Swin, ViT-Large) or datasets? If directly transferred, how significant would the prediction error be?

**Q2:** If MAP predictions exhibit bias (e.g., error ±0.5% Top-1), would the final pruning ratio decision be completely altered? Did the paper evaluate MAP's sensitivity to pruning strategies?

**Q3:** When performing “prune activations + merge linear layers,” how is it ensured that residual connections and LayerNorm do not compromise model stability? Are there strict mathematical conditions guaranteeing this merging is harmless or nearly harmless?

**Q4:** If applied to ViT variants with gating mechanisms (e.g., SwiGLU) or Post-Norm, can layer fusion still be performed directly? Did the paper validate effectiveness in such cases?

**Q5:** The paper notes that clipping activations results in larger initial losses but faster recovery, while clipping attention layers shows the opposite pattern. Does this phenomenon persist across multiple random seeds or longer training cycles? If fine-tuning is extended to 50 epochs, does this “asymmetry” remain pronounced?

**Q6:** What is the fundamental cause of this recovery disparity? Is it solely due to gradient magnitude differences, or is it related to information flow paths or residual distribution? The paper does not provide a theoretical explanation.

**Q7:** BoundaryDPT is evaluated only on ImageNet classification. If applied to downstream tasks (e.g., COCO detection, ADE20K segmentation, or ImageNet-A robustness testing), can it maintain the same accuracy-acceleration trade-off?

**Q8:** The claim that “BoundaryDPT is strictly orthogonal to token pruning” is validated only under a single token method (GTP-ViT) and a single pruning rate. Does this orthogonality hold if the token strategy is changed or the pruning rate is increased?

**Q9:** The paper trained using Ascend 910B NPUs, yet inference throughput was measured on NVIDIA H800 GPUs. Were all baselines remeasured? If not, is the throughput comparison fair?

**Q10:** Constructing MAP requires multiple rounds of “pruning–fine-tuning–evaluation” sampling. The paper does not specify the exact number of samples or computational cost. How many GPU days are required to train MAP in a practical environment? Can it be reproduced under limited computational resources?


**If you address my concerns, I will consider raising my score.**

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduce BoundaryDPT, which considers pruning not only linear layers but activation function layers for model compression. The authors present several evaluations across multiple datasets, and the results look good.

### Strengths
This paper proposed pruning non-linear activation function layers, and found finetuning can  mitigate degration caused by pruning activation function layers.
The proposed method achieves some improvement in experimental results among different models on various datasets.

### Weaknesses
1.Limited contribution
The overall contribution of this paper is quite incremental. The authors make an effort to prune activation function layers, the motivation behind this idea is not clear. It remains unclear why pruning nonlinear layers is worth considering, and what advantages it offers compared to pruning linear layers. In addition, the proposed method lacks innovation or novel insights that would make it stand out.

2. Unclear structure and layout
The former of the main text is acceptable, the latter is disordered and confusing. For section 4.2 and the context behind it, the authors fail to depict their methods clearly about the details. From the fact that Section 4.3 is titled “Stage 2 AND 3”, it is evident that the authors did not pay sufficient attention to the overall structure and presentation of the paper. This gives the impression that the manuscript is prepared hastily and lacks the level of polish and rigor expected for acceptance.

3.Poor presentation and informal writing 
The manuscript suffers from several formatting and clarity issues — for example, many formulas and algorithms do not clearly define the meaning of their variables, references are not cited correctly or consistently. And the logical flow between sections is weak, making the paper hard to follow. These issues significantly reduce readability and make it difficult to understand.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes BoundaryDPT, a new depth pruning framework for ViTs that jointly prunes attention layers and activation function layers. The key idea is to exploit the redundancy in nonlinearity to naturally reduce depth with minimal accuracy loss. The authors identify two critical challenges in joint pruning, gradient disparity and recovery asymmetry. BoundaryDPT addresses these via a three-stage method: (1) redundancy identification with a Model Accuracy Predictor, (2) pruning and fine-tuning with optional self-distillation, and (3) MLP layer merging for inference speedup. An extended version, BoundaryDPT+, integrates width pruning for extreme compression, achieving up to 5.19× speedup on DeiT-B with near-lossless accuracy.

### Strengths
1.  Experimental results shows negligible performance drop with better compression/speed up ratio compared to the previous depth pruning method
2. Experimental results are comprehensive, different architectures are tested (DeiT and Swin-Transformer ), different datasets are tested (ImageNet, Cifar, ADE)
3. Targeting activation function redundancy, which is a rarely explored dimension in ViT pruning.

### Weaknesses
1. Although I guess K in eq3 means the sparsity constraint of the layer/model, but the author didn't define K in the text
2. I'm not sure about the meaning of section3, which is the observation section. How the observation leads into the intuition or motivation of the methodology?
3. The writeup of part 1 is a little messy and hard to understand
4. Although pruning activation functions seem to be novel, but there is no motivation supporting why we are considering pruning activations. 
5. The MAP polynomial approximation, though effective, feels heuristic without theoretical grounding

Overall, the contribution is well-motivated, but the technical novelty is moderate. The pruning framework mainly integrates known ideas  into a structured pipeline.

### Questions
1. The performance and more details about MAP
2. Although activation layer pruning is new in ViT, the effect may be task-dependent and should be validated beyond classification, like detection etc.
3. The runtime and memory costs for the MAP fitting

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a ViT pruning method by depth pruning on activation layers of ViT, on top of the usual attention layer in depth pruning. This is performed by firstly calibrating layer's impact on model accuracy (MAP predictor) to decide the pruning ratio of each layer, then learning a layerwise importance factor during finetuning to progressively remove the current most redudant layers torwads the pruning ratio target. This is done iteratively interleved with finetuning. Experiment results shows the method can achieve more than 1.5x speedup for ViT.

### Strengths
Paper is well written, with clear organization and presentations.
The results seems comprehensive on the conducted models and data.
The motivation of this paper also seems well supported.

### Weaknesses
The overall idea of leveraging learned importance for incorporating activation pruning seems preexisted. Although the paper did provide some insights regarding the attention v.s. activation gradient magnitude, but i'm not sure how universal these discoveries are beyond standard ViT arch.

Although the method seems effective from the results, whether the improvement still worth it when considering the training cost to obtain those models, especially that the method requires iterative pruning with finetuning over 400 epochs.

The paper also didn't include a dedicated limitation discussion.

### Questions
1. Are the compared baselines all requiring the similar finetuning costs? It would be more meaningful to include the comprasion on the training costs to obtain the pruned structure. 
2. I wonder why the gradient disparity is observed on the additional learnable layer importance added by author, rather than on the native ViT behavior, not sure how other ways of observation would affect the conclusion of this observation.

### Soundness
3

### Presentation
3

### Contribution
3
