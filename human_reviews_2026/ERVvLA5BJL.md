# HexFormer: Hyperbolic Vision Transformer with Exponential Map Aggregation

- Decision: Reject
- Scores: 6, 4, 4, 8

## Abstract
Data across modalities such as images, text, and graphs often contains hierarchical and relational structures, which are challenging to model within Euclidean geometry. Hyperbolic geometry provides a natural framework for representing such structures. Building on this property, this work introduces HexFormer, a hyperbolic vision transformer for image classification that incorporates exponential map aggregation within its attention mechanism. Two designs are explored: a hyperbolic ViT (HexFormer) and a hybrid variant (HexFormer-Hybrid) that combines a hyperbolic encoder with an Euclidean linear classification head.
HexFormer incorporates a novel attention mechanism based on exponential map aggregation, which yields more accurate and stable aggregated representations than standard centroid based averaging, showing that simpler approaches retain competitive merit. Experiments across multiple datasets demonstrate consistent performance improvements over Euclidean baselines and prior hyperbolic ViTs, with the hybrid variant achieving the strongest overall results.
Additionally, this study provides an analysis of gradient stability in hyperbolic transformers. The results reveal that hyperbolic models exhibit more stable gradients and reduced sensitivity to warmup strategies compared to Euclidean architectures, highlighting their robustness and efficiency in training.
Overall, these findings indicate that hyperbolic geometry can enhance vision transformer architectures by improving gradient stability and accuracy. In addition, relatively simple mechanisms such as exponential map aggregation can provide strong practical benefits.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes **HexFormer**, a Vision Transformer formulated entirely in **Lorentzian hyperbolic space**. The key novelty lies in the **exponential map aggregation (ExpAgg)** mechanism within the attention module, replacing centroid-based averaging to improve geometric consistency and training stability. Two model variants are explored: a fully hyperbolic **HexFormer** and a **HexFormer-Hybrid** that combines a hyperbolic encoder with a Euclidean linear classification head.  

Empirical evaluations on **CIFAR-10, CIFAR-100**, and **Tiny-ImageNet** demonstrate that HexFormer and its hybrid variant achieve slightly higher accuracy than Euclidean ViTs and prior hyperbolic ViTs (HVT, LViT). The paper also presents a detailed **gradient stability analysis**, showing smoother gradients and reduced dependence on warmup schedules in hyperbolic models.

### Strengths
1. **Stable and thorough implementation.** Training hyperbolic transformers is notoriously difficult, and the authors’ success in stabilizing training is valuable. The detailed explanation of why exponential aggregation avoids NaN/inf issues is practically helpful.

2. **Well-structured and clearly written.** The paper is organized effectively, with comprehensive details in both the main text and appendix, aiding reproducibility.

3. **Gradient stability analysis.** The study of gradient histograms and warmup sensitivity is one of the strongest aspects, giving useful empirical insight into hyperbolic optimization dynamics.

4. **Insightful use of exponential aggregation.** The proposed ExpAgg offers modest but consistent improvements and helps mitigate numerical instability, which future hyperbolic models can benefit from.

5. **Reproducibility and transparency.** Detailed hyperparameters, architecture specifications, and clear training setups are provided, suggesting reproducibility will be straightforward once code is released.

### Weaknesses
1. **Limited novelty.** The proposed idea — using exponential and logarithmic maps for aggregation in hyperbolic attention — is a **natural and incremental extension** of prior hyperbolic ViTs such as HVT, LViT, and HypFormer. The conceptual contribution feels modest.

2. **Marginal performance gains.** The improvements over Euclidean and prior hyperbolic baselines are small (often within 0.2–1%), which does not convincingly demonstrate a strong benefit.

3. **No runtime or efficiency analysis.** Hyperbolic models are known to train significantly slower. The paper lacks a comparison of **training time, FLOPs, or parameter efficiency**, which is important for practical evaluation.

4. **Limited theoretical explanation.** While the empirical stability is shown, the underlying reasons for why ExpAgg helps are described mainly via numerical examples rather than rigorous geometric or analytical reasoning.

5. **Hierarchical structure. ** An understanding of how or even if the model is indeed capturing hierarchical structure has not been made.

### Questions
1. **Runtime comparison:** Please include wall-clock training time, parameter counts, and FLOPs for HexFormer vs. Euclidean ViT. What is the computational overhead of the Lorentz operations?

2. **Geometric explanation:** Can you provide more intuition on why ExpAgg yields slightly higher accuracy compared to centroid aggregation beyond numerical precision effects?

3. **Hybrid performance:** Why does the hybrid model outperform the fully hyperbolic variant? Is it due to optimization simplicity in the Euclidean classifier? Because, in my mind it doesnt make sense how a hierarchical model automatically learnt 

4. **Curvature parameter:** How sensitive are the results to the curvature parameter \( K \)?

5. **Generalization:** Have you tested HexFormer on larger-scale datasets such as ImageNet, or in transfer-learning settings? I am assuming this is where the issue of time taken per epoch would come up. 

6. **Hierarchical structure**: Claim was made about the model being able to learn Hierarchical structure, how would one be able to see it and how does attention specifically help or aid in achieving it?

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
3

### Summary
This paper introduces HexFormer, a novel vision transformer that embeds visual tokens into a hyperbolic feature space to better capture hierarchical and compositional relationships in visual data. The authors propose a Hyperbolic Attention mechanism and a Geometric Projection Layer to align Euclidean features with the hyperbolic manifold, enabling efficient curvature-aware representation learning. Theoretical analysis shows that hyperbolic embeddings can preserve structural similarity and relational distance, while experiments across multiple vision benchmarks demonstrate consistent improvements over Euclidean counterparts.

### Strengths
**Novel geometric perspective:** This paper introduce hyperbolic geometry into vision transformers, which is original and conceptually meaningful, extending non-Euclidean representation learning to high-dimensional visual domains.

**Theoretical Support:** The mathematical formulation of the hyperbolic projection and attention mechanism is clearly presented and theoretically sound.

**Well-organized:** The paper is well organized and easy to follow, with intuitive figures and insightful visualizations of curvature and embedding behaviors.

### Weaknesses
**Limited experimental scope and generality:** The experiments are limited to image classification tasks on a small number of datasets, which makes the empirical validation narrow and the method’s generality questionable. Since the proposed hyperbolic representation aims to capture hierarchical relationships, demonstrating its benefits on diverse tasks such as detection, segmentation, or retrieval would be crucial for broader applicability.

**Ablation:** The empirical study lacks a systematic analysis of individual module contributions. For instance, it is unclear how much improvement comes from the hyperbolic projection layer versus the hyperbolic attention mechanism, or whether curvature learning plays a significant role. A detailed ablation or sensitivity study (e.g., varying curvature c) would help isolate the key driving factors of performance gains.

**Efficiency and numerical stability concerns:** It seems that the introduction of hyperbolic operations, exponential and logarithmic mappings, may introduce additional computational overhead and numerical instability. The paper does not include runtime, memory, or FLOPs analysis, leaving uncertainty about the practicality of the method for large-scale vision applications.

### Questions
**Q1:** How sensitive is the performance to the curvature parameter c? Is it fixed or learned, and how does it influence optimization stability?

**Q2:** Could the authors provide training time or FLOPs comparison with Euclidean ViT/DeiT models? It is unclear of the method's efficiency.

**Q3:** How does HexFormer perform on datasets with weak or flat hierarchies?

**Q4:** How does the HexFormer applied with linear attention vision transformers, such as PolaFormer[1] and FLatten  Transformer[2]?

This is an interesting work, and I would consider raising my score if the authors adequately address the concerns above.

[1] W. Meng, Y. Luo, X. Li, D. Jiang, and Z. Zhang, “Polaformer: Polarity-aware linear attention for vision transformers,” in Proc. International Conference on Learning Representations (ICLR), 2025.

[2] Han D, Pan X, Han Y, Song S, Huang G. Flatten transformer: Vision transformer using focused linear attention. In Proceedings of the IEEE/CVF international conference on computer vision 2023 (pp. 5961-5971).

### Soundness
3

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
The authors introduce HexFormer and conduct comprehensive comparisons with standard Euclidean Vision Transformers (ViTs) and previous Hyperbolic ViTs (such as HVT and LViT) across multiple datasets, including CIFAR-100, ImageNet, and Tiny-ImageNet. The authors particularly emphasize the stability and advantages provided by exponential mapping aggregation, demonstrating that it effectively avoids NaN/Inf issues across multiple random seeds and achieves significantly superior training convergence compared to traditional aggregation schemes.

### Strengths
The main advantages of this paper are: (1) exploring more forms of hyperbolic-Euclidean hybrid structures; (2) emphasizing the stability and advantages brought by exponential map aggregation

### Weaknesses
The paper's experimental validation focuses primarily on small image classification datasets (CIFAR-10, CIFAR-100, Tiny-ImageNet), lacking verification on more challenging downstream tasks such as object detection, segmentation, or real-world scenario data. This limitation raises questions about the model's generalization capabilities and practical deployment value.

The paper explicitly acknowledges that comprehensive hyperparameter optimization was conducted only for the ViT-Tiny scale, while configurations for ViT-Small, ViT-Base, and different datasets were largely adopted without modification. This approach constrains both the improvement potential and fairness of subsequent model evaluations, with results potentially lacking representativeness due to limited tuning strategies.

Although the paper emphasizes that ExpAgg demonstrates greater stability than centroid aggregation in floating-point operations (reduced susceptibility to NaN/Inf errors), this analysis remains largely confined to theoretical derivations and limited seed or small-scale experiments. The stability performance under more extreme conditions—such as larger models, extensive datasets, or prolonged training scenarios—has not been thoroughly investigated.

The methodological innovation centers primarily on replacing centroid aggregation with "exponential mapping aggregation," while other model components (including topological encoding, residual design, and normalization) largely follow existing approaches. The work does not explore integration with or comparison to more diverse Transformer improvements such as pooling layers, encoder-decoder architectures, or cross-modal extensions.

While the paper indicates plans to open-source the code (noting "Paper under double-blind review"), implementation details, source code, and pre-trained weights remain unavailable at present.

### Questions
How does the author understand hyperbolic geometry's ability to address hierarchical relationships? What type of hierarchical relationships are being examined, and how do CIFAR-10, CIFAR-100, and Tiny-ImageNet relate to this analysis?

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
This paper introduces HexFormer, a novel Vision Transformer (ViT) architecture implemented entirely in the Lorentz model of hyperbolic space. The paper uses the Logarithmic and Exponential maps to safely perform weighted summation in the stable tangent space, solving the critical numerical instability issues associated with centroid aggregation in the Lorentz manifold. The framework also introduces a high-performing hybrid variant (HexFormer-Hybrid). Experiments show HexFormer consistently surpasses Euclidean ViTs and prior hyperbolic ViT SOTA, demonstrating significant parameter efficiency and providing compelling evidence of improved training stability (reduced warmup dependency and smoother gradients) across multiple datasets.

### Strengths
The Exponential Map Aggregation (ExpAgg) scheme is a robust, mathematically sound and original.

The paper provides strong quantitative and qualitative evidence that hyperbolic representations enhance optimization dynamics. HexFormer exhibits minimal reliance on warmup schedules compared to Euclidean baselines (Table 2) and demonstrates smoother, more consistent gradient distributions (Figure 3). This is a major finding for the practical adoption of hyperbolic models.

HexFormer-Tiny (∼3M params) achieves higher accuracy than the cited ViT-Base results (∼85M params) of prior hyperbolic models (HVT and LViT).

### Weaknesses
The HexFormer-Hybrid variant performs best, raising the question of why a strictly Euclidean classification head is superior after a hyperbolic encoder.

Providing a training time comparison would also be beneficial.

While Table 2 cifar 10 supports the claim, Tiny-Imagenet does not fully show the difference between Euclidean and hyperbolic 


I would recommend citing hyperbolic learning surveys such as “Hyperbolic Deep Learning in Computer Vision: A Survey”


Multiple citations in lines 105-106 can be aggregated.
Sentence in line 197 is not complete.

### Questions
Please check the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
