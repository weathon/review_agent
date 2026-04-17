# Octic Vision Transformers: Quicker ViTs Through Equivariance

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Why are state-of-the-art Vision Transformers (ViTs) not designed to exploit natural geometric symmetries such as 90-degree rotations and reflections? In this paper, we argue that there is no fundamental reason, and what has been missing is an efficient implementation. To this end, we introduce Octic Vision Transformers (octic ViTs) which rely on octic group equivariance to capture these symmetries. In contrast to prior equivariant models that increase computational cost, our octic linear layers achieve 5.33x reductions in FLOPs and up to 8x reductions in memory compared to ordinary linear layers. In full octic ViT blocks the computational reductions approach the reductions in the linear layers with increased embedding dimension. We study two new families of ViTs, built from octic blocks, that are either fully octic equivariant or break equivariance in the last part of the network. Training octic ViTs supervised (DeiT-III) and unsupervised (DINOv2) on ImageNet-1K, we find that they match baseline accuracy while at the same time providing substantial efficiency gains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Octic Vision Transformers (octic ViTs) that leverage D8 group equivariance (90-degree rotations and reflections) to achieve computational efficiency gains in Vision Transformers. The key innovation is implementing equivariant linear layers in the Fourier domain of D8, achieving 5.33× FLOPs reduction and 8× memory reduction per feature dimension compared to standard linear layers. The authors propose two ViT families: I8 (fully equivariant) and H8 (breaking equivariance in later layers), demonstrating ~40% FLOP savings on ViT-H while maintaining accuracy on ImageNet-1K using both supervised (DeiT-III) and self-supervised (DINOv2) training recipes.

### Strengths
1. **Solid theoretical foundation**: The application of group representation theory to ViTs is mathematically rigorous, with clear exposition of the D8 group structure and isotypical decomposition.

2. **Comprehensive experimental validation**: 
   - Evaluation on both supervised (DeiT-III) and self-supervised (DINOv2) training
   - Extensive ablations: invariantization methods (6 variants), number of octic blocks, effect of breaking equivariance
   - Additional evaluations on white blood cell classification (DinoBloom)

3. **Practical implementation details**: Custom Triton kernels for fused operations, efficient FFT implementation, and clear computational complexity analysis.

4. **Reproducibility**: Detailed hyperparameters, compute resources documented (Table 10), commitment to release code and weights.

5. **Honest reporting**: Authors acknowledge limitations regarding throughput not matching FLOPs savings.

### Weaknesses
### Major Issues:

1. **Theory-practice gap is severe**: 
   - The 5.33× FLOPs reduction only translates to 1.47-2.37× throughput gains
   - Appendix B reveals octic layers only have better arithmetic intensity at C > 3200, well beyond experimental scales
   - This fundamentally undermines the efficiency claims and suggests limited practical applicability

2. **No accuracy improvements**: 
   - All octic models only *match* baselines, never exceed them
   - For a top-tier conference with <10% acceptance, matching performance with some efficiency gains is insufficient
   - The accuracy-efficiency trade-off is not favorable enough to warrant publication at this venue

3. **Limited experimental scope**:
   - Only ImageNet-1K pretraining (1.2M images)
   - No evaluation on modern large-scale datasets (ImageNet-21K, JFT, etc.)
   - No real-world application demonstrations beyond standard benchmarks
   - Missing comparisons with other efficiency methods (pruning, quantization, knowledge distillation)

4. **Incremental novelty**: 
   - Direct extension of Bökman et al. (2025) from D2 to D8
   - The key technical contribution (Fourier domain implementation) is relatively straightforward application of known theory
   - No fundamental architectural innovations

5. **Scalability concerns**:
   - Authors acknowledge they "do not scale the size beyond ViT-H" (Limitations, Section 5)
   - No experiments on ViT-G, ViT-e, or ViT-22B despite including them in Table 1
   - Unclear if benefits hold at truly large scales where efficiency matters most

### Minor Issues:

6. **Missing baselines**: No comparison with other structured/efficient transformers (e.g., local attention, sparse attention, linear attention variants)

7. **Evaluation limitations**: 
   - OOD rotation evaluation is limited—only random 90° rotations, not continuous angles
   - Segmentation results show I8 models sometimes underperform (ADE20K: 33.9 vs 34.7 mIoU)

8. **Implementation dependency**: Requires custom Triton kernels, limiting accessibility and reproducibility for the broader community

### Questions
1. **Critical**: Can you explain why the arithmetic intensity analysis (Appendix B) shows octic layers only become advantageous at C ≈ 3200, yet you claim efficiency gains at C=1280? This appears contradictory.

2. **Scalability**: Table 1 projects improvements for ViT-G through ViT-22B. Why weren't these models actually trained? What evidence suggests the benefits will hold at larger scales?

3. **Comparison with other efficiency methods**: How do your models compare with:
   - Standard pruning/quantization techniques?
   - Knowledge distillation from larger models?
   - Other efficient ViT architectures (e.g., DeiT-distilled, EfficientViT)?

4. **Throughput analysis**: Can you provide a detailed breakdown of where the gap between FLOPs savings (5.33×) and throughput gains (1.47×) comes from? Is this fundamental or implementation-limited?

5. **Accuracy ceiling**: Is there a theoretical reason octic models cannot exceed baseline accuracy? Have you tried longer training or different optimization strategies?

6. **Real-world applications**: Can you demonstrate benefits on practical downstream tasks (object detection, instance segmentation, video understanding)?

7. **Continuous rotations**: How do I8 models perform under continuous rotation angles, not just 90° increments?

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
4

### Summary
The paper proposes a new, efficient D8 equivariant vision transformer. The authors report extensive experiments which supports their method.

### Strengths
The experiments show that the method has good efficiency, though not so comprehensive, which will be discussed below.

The proposed method is interesting. Applying octic group to ViT is reasonable and looks have wide application domain.

### Weaknesses
My first concern is that the mechanism is not well motivated. I would suggest the authors adding in more explaination why the proposed method has better efficiency in the introduction and method sections. Currently, the authors justify the method that leveraging a larger group can yield "faster, stronger, and more compact models" without further establishment of this claim. I agree this could be right, but this leads to two questions: (1) Why not use a group that is even larger? (2) Would it be possible to quantitatively link the size of group and efficiency? 

Ideally, theoretical analysis would be beneficial. For example, it would be interesting to see convergence or generalisation analysis linking the method with octic group.

Second, the experiments can be improved. The authors only use ImageNet-1k for classification, which is not sufficient. The compared methods are only Touvron et al. (2022) and Bökman et al. (2025). I would suggest the authors giving more results.

### Questions
Please see above.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Octic ViT, a Vision Transformer leveraging octic group equivariance. The core innovation is an efficient octic-equivariant linear layer that operates in the Fourier domain, reducing the linear layer's FLOPs by 5.33x and parameters by 8x. Experiments (DeiT-III, DINOv2) show that Octic ViTs match or exceed the accuracy of ViT-H/L baselines while saving ~40% of total FLOPs.

### Strengths
- Significant Compute Efficiency: Drastically reduces FLOPs for SOTA ViTs without sacrificing accuracy.

- Efficient Equivariance: Unlike prior work where equivariance adds overhead, this paper uses Fourier domain sparsity to accelerate ViTs.

- SOTA Validation: Validated at scale on SOTA training recipes, proving practical utility.

- Architectural Insights: The ablation between hybrid and fully invariant models provides valuable design insight.

### Weaknesses
- FLOPs vs. Throughput Mismatch: The large FLOPs reduction does not fully translate to throughput/speedup (only 1.47x).

- Implementation Complexity: The method is complex, requiring knowledge of group representation theory, Fourier transforms, and custom Triton kernels.

- Non-linearity Overhead: GELU activations must be applied in the spatial domain ($\rho_{reg}$), requiring costly round-trips via Fourier transforms.

### Questions
- Throughput Bottleneck: What is the primary bottleneck causing the gap between FLOPs savings (4.6x) and throughput gain (1.5x) for ViT-H? Is it the custom Triton kernel or the Fourier transforms?

- Hybrid vs. Invariant: Why do the hybrid $\mathcal{H}_{8}$ models almost always achieve slightly better accuracy than the fully invariant $\mathcal{I}_{8}$ models? Does this imply ImageNet requires non-symmetric features?

- Hyperparameter Tuning: You did not retune hyperparameters despite an 8x parameter reduction in the linear layers. Does this suggest the model is robust, or could accuracy be further improved by tuning regularization (e.g., weight decay)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes using octic group equivariant ViTs for better performance in downstream tasks. The authors use custom Triton kernels to speed up training equivariant ViTs and introduce different variants where $I_8$ is exactly invariant in the later layers and $H_8$ break equivariance in the later layers. Experiments on ImageNet-1k show that $H_8$ can outperform standard ViTs.

### Strengths
- Good comparison of end-to-end equivariant ($D_8$), late invariant ($I_8$), late non-equivariant ($H_8$) and their performance.
- Extensive experiments on ImageNet classification and an SSL task show that equivariance can help.

### Weaknesses
A critical weakness of this paper in my opinion is the limited novelty. Equivariant ViTs have been proposed previously [1, 2, 3] and octic ViTs are a special case. It is also well known that weight sharing/weight tying reduces the number of FLOPs as a direct consequence of the reduction of the number of parameters [4, 5]. Furthermore, several papers have shown that some form of symmetry breaking, especially in the later layers, can be beneficial for CNNs [4, 5, 6], albeit not for ViTs. Thus the novelty is minimal and thus I feel that the main contribution of this paper lies on the fact that the authors provide a custom Triton kernel to speed up wall clock time and do extensive FLOPs analysis on the variants.

Another critical weakness is that the argument that equivariance is beneficial in ViTs goes against the results. The variant $H_8$, which uses non-equivariant layers in the later blocks performs the best, beating regular ViTs, but the exactly equivariant ViT $D_8$ or even the invariant ViT $I_8$ perform worse than a standard ViT. There should be a nuanced discussion about why equivariance, specifically in the earlier layers, helps over the blanket statement that equivariance is beneficial and should be used more if it could be sped up.

Lastly, the FLOPs seems to decrease but the wall clock time is higher for all variants. Table 10 in the Appendix shows that the ViT-L/16 baseline takes 24 hours, compared to 39 for $H_8$ or 37 for $I_8$. This contradicts the statement that the proposed methods are indeed faster. The reason why equivariant ViTs require lower number of FLOPs is due to the weight tying/sharing reducing the number of parameters, but the wall clock time is often a lot slower than regular ViTs because the representation must be transformed to the spatial domain ($\rho_{iso}$ to $\rho_{reg}$) in order to apply the nonlinearities pointwise, which are then mapped back to $\rho_{iso}$ after the nonlinearity. This is true for equivariant steerable convolutions as well.

One way I can see to improve the paper in the future would be to focus on the symmetry breaking aspect, and deeply investigate why breaking equivariance in later layers, for ViTs, helps (perhaps by looking at the intermediate representations).

References
1. Romero, D. W., & Cordonnier, J. B. (2020). Group equivariant stand-alone self-attention for vision. arXiv preprint arXiv:2010.00977.
2. Xu, R., Yang, K., Liu, K., & He, F. (2023, July). $ E (2) $-Equivariant Vision Transformer. In Uncertainty in Artificial Intelligence (pp. 2356-2366). PMLR.
3. Klee, D., Park, J. Y., Platt, R., & Walters, R. A Comparison of Equivariant Vision Models with ImageNet Pre-training. In NeurIPS 2023 Workshop on Symmetry and Geometry in Neural Representations.
4. Weiler, M., & Cesa, G. (2019). General e (2)-equivariant steerable cnns. Advances in neural information processing systems, 32.
5. Kondor, R., & Trivedi, S. (2018, July). On the generalization of equivariance and convolution in neural networks to the action of compact groups. In International conference on machine learning (pp. 2747-2755). PMLR.
6. Vadgama, S., Islam, M. M., Buracas, D., Shewmake, C., Moskalev, A., & Bekkers, E. (2025). Probing Equivariance and Symmetry Breaking in Convolutional Networks. arXiv preprint arXiv:2501.01999.

### Questions
I'll write some minor comments here.
- The octic group is more commonly denoted as $D_4$ rather than $D_8$. I would recommend changing to the more common notation.
- The preliminaries seem a touch too long, given that these are known results. Some of the preliminaries also don't seem to be necessary to understand the method.

### Soundness
3

### Presentation
2

### Contribution
2
