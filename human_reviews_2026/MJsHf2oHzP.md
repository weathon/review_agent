# UniSVD: Unilateral Weight Decomposition for Attention-based Vision Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 4, 2, 4

## Abstract
Transformers have achieved remarkable success across diverse domains, but their ever-growing scale results in prohibitive computational and memory costs. Low-rank matrix decomposition with Singular Value Decomposition (SVD) has emerged as an effective compression technique. Recent studies, such as ASVD, SVD-LLM, and FLAR-SVD, have improved decomposition quality by incorporating activation-aware method. However, these methods do not consider the unique mechanism of MHA, where query-key ($Q$-$K$) and value–output ($V$-$O$) computations are linear and allow pre-computation. To effectively leverage this mechanism, we propose Unilateral Singular Value Decomposition (UniSVD), a novel framework that applies decomposition to only one side of the $Q$-$K$ or $V$-$O$ weight pairs in a head-wise manner.  Since $Q$-$K$-$V$-$O$ weights exhibit varying sensitivities to low-rank approximation across heads and layers, UniSVD adaptively selects which side to be decomposed according to the rank sensitivity, thereby preserving the important information of weights. Extensive experiments demonstrate that UniSVD seamlessly integrates with existing decomposition methods and consistently achieves superior trade-offs between parameter reduction, FLOPs, and model performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The key idea is that to decompose only one side of (Q,K) and (V,O) per head, chosen by rank-sensitivity, to improve the accuracy–efficiency trade-off vs per-weight and combined decompositions and positions against per-weight SVD inefficiency and COMCAT’s combined scheme. 

The paper claims broad plug-in gains across FWSVD, ASVD, SVD-LLM, FLAR-SVD without fine-tuning. This is backed up by some impressive results - notably in table 2 - beats per-weight and combined methods across 20/40/60% MHA reduction (Table 2); large gains at 60%. 

Error analyses across layers support the mechanism

### Strengths
Simple idea which results in some good results. 

The idea is simple, training-free, and plugs into multiple SVD families, improving both Top-1 and latency without fine-tuning. Under heavy MHA reduction, unilateral consistently outperforms combined by large margins. I think this is a nice result which is presented in a very clear way.

### Weaknesses
End-to-end efficiency reporting -  provide VRAM and throughput across batch/sequence lengths (besides latency) for each baseline + UniSVD; include wall-clock for the selection pass. 
The following would improve the paper I feel 

Compare Frobenius-error selection with alternatives (spectral norm, activation-aware criteria like ASVD/SVD-LLM whitening) to confirm robustness.

Add ViT-L/16 or Swin-T, and a second dataset to show generality beyond DeiT/ImageNet-1K. 

Granularity of ranks. Release per-layer/head ranks and selection maps; add sensitivity curves (Top-1 vs rank) to guide practitioners. 

Theory–performance link. The appendix notes combined-space error may not track accuracy; expand analysis on why individual-space optimisation aligns better with performance. 

Ablations on latency sources. Separate gains from fewer sublayers vs cache/matmul effects to clarify where the speedup comes from.

### Questions
Can you provide full VRAM/throughput and batch/sequence sweeps, and quantify the one-off cost of computing head-wise SVD errors for selection? 

Do you have results on non-DeiT architectures or another dataset to confirm portability? 

Could unilateral decomposition extend to MLP blocks (e.g., only decompose one of in/out projections) or to LLM attention with RoPE? Any problems you foresee? 

The appendix shows combined-space error isn’t predictive of accuracy. Can you formalise why individual-space optimisation is better aligned, or give counter-examples?

### Soundness
3

### Presentation
4

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
This paper proposes UniSVD, a novel low-rank decomposition method for efficient compression of Transformer-based models. UniSVD exploits the linear nature of Multi-Head Attention by selectively decomposing only one side of each Q–K or V–O weight pair, while dynamically determining which side to factorize based on Frobenius-norm-based rank sensitivity analysis at the head and layer levels. This approach preserves information in rank-sensitive weights while significantly reducing parameter count and computational cost. Experiments on DeiT-Small and DeiT-Base models with the ImageNet-1K dataset demonstrate that UniSVD consistently improves accuracy and reduces latency compared to prior SVD-based methods such as FWSVD, ASVD, SVD-LLM, and FLAR-SVD, achieving over 40% parameter reduction with only a 2–3% accuracy drop. These results validate UniSVD as a simple yet generalizable low-rank decomposition framework that effectively balances efficiency and performance for Transformer architectures.

### Strengths
1. The proposed UniSVD maintains a level of parameter and computational efficiency comparable to existing combined decomposition methods, while introducing a unilateral decomposition strategy and a dynamically sensitivity-aware weight selection mechanism. Through this design, the method effectively preserves model accuracy without requiring any fine-tuning process.

2. The proposed UniSVD demonstrates strong practicality and applicability in real deployment and model compression scenarios by maintaining stable performance without any fine-tuning process. This highlights the proposed method’s contribution not only as a theoretically efficient approach but also as a training-free, low-cost compression framework that can be readily applied in real-world applications.


3. The theoretical explanation of the proposed method is presented in a logical and convincing manner. The mathematical formulation and analysis are well aligned with the underlying intuition, effectively supporting the validity and reliability of the proposed approach.

### Weaknesses
1. While Table 1 reports reductions in latency and GFLOPs, the paper does not provide sufficient analysis of the underlying causes. A more detailed explanation is needed to clarify which computational characteristics of the proposed method lead to reduced operations and latency. In addition, the paper lacks discussion on whether such acceleration effects are specific to certain hardware environments or can be consistently observed across different device types.  
  
  
2. The proposed method leverages the characteristics of the MHA structure. Accordingly, it would be important to discuss whether this method can be extended to large-scale generative models such as Vision-Language Models (VLMs) or Large Language Models (LLMs) that also rely on MHA. Furthermore, the experiments are confined to the relatively simple DeiT–ImageNet setting, which limits the evaluation of the method’s generality and effectiveness for large-scale models. In particular, restricting the experiments to DeiT models does not sufficiently demonstrate the proposed method’s generalization capability, representing a critical limitation that diminishes the overall contribution of the work. As recent research trends emphasize validating applicability and generalization across diverse model architectures and datasets, broader experiments and validation are necessary to reinforce the paper’s claims.


Overall, the proposed method presents a novel and technically interesting idea that deserves recognition. However, empirical validation regarding the range of applicable models and the generalization performance remains limited. As mentioned in the future work, extending the approach to large-scale models such as LLMs or VLMs and demonstrating consistent generalization would significantly strengthen the contribution of this study. Incorporating additional experiments or analyses addressing these limitations in a future revision would greatly enhance the completeness and overall impact of the paper.

### Questions
1) The paper reports in Figure 4 that $W^{K}$and $W^{O}$are frequently selected due to their low rank sensitivity, and Table 3 shows that a fixed-selection strategy using these weights achieves nearly identical accuracy to the dynamic UniSVD variant. Given this, could the authors clarify whether the additional overhead of performing per-head SVD and loss computation twice in the dynamic selection process is practically justified, considering the marginal accuracy improvement observed?

### Soundness
3

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
4

### Summary
This paper proposes UniSVD, a simple and training-free low-rank compression method for attention-based vision transformers. Instead of decomposing all attention weights, UniSVD decomposes only one side of each Q/K or V/O pair, based on which side is less sensitive to rank reduction. This “unilateral” strategy keeps more information from the important weights and reduces FLOPs and parameters. Experiments on DeiT-Small and DeiT-Base show consistent improvements over several existing SVD-based decompositions, without any fine-tuning.

### Strengths
UniSVD offers a novel insight into attention decomposition: discovering and utilizing the asymmetric sensitivity of Q/K and V/O matrices to improve compression without retraining.
The method is conceptually simple and easy to follow, requiring only SVD and a dynamic selection rule based on Frobenius loss.

### Weaknesses
The experimental results mainly highlight two aspects:
(1) UniSVD is orthogonal to existing SVD-based decompositions and consistently improves upon them;
(2) it outperforms simpler variants such as Per-Weight Decomposition and Combined Weight Decomposition.
However, the evaluation lacks comparison against stronger and more recent low-rank decomposition baselines from the literature, making it unclear whether UniSVD achieves SOTA performance beyond the specific SVD variants tested here.

From a methodological perspective, while the core intuition is reasonable, the technical novelty is modest. The method essentially adds a dynamic selection between Q/K and V/O before applying SVD. The results suggest that decomposing K over Q and O over V tends to be optimal, and the additional head or layerwise dynamic selection provides only marginal benefit, since Table 3 shows negligible differences. 

|        | 20%  | 40%  | 60%  | 20%  | 40%  | 60%  | 20%  | 40%  | 60%  |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| K-O  | 78.3  | 74.2  | 65.3  | 79.6  | 76.1  | 69.3  | 81.7  | 80.3  | 71.7  |
| DynamicSelection | 78.4  | 74.4  | 65.4  | 79.9  | 76.5  | 69.3  | 81.7  | 80.4  | 72.1  |

Thus, the main contribution appears to lie in identifying the relative sensitivity of Q/K/V/O weights to decomposition rather than introducing a fundamentally new decomposition framework.

Finally, the method is evaluated only on DeiT models and ImageNet classification. It remains uncertain whether UniSVD generalizes to other ViT architectures e.g. Swin Transformer, downstream tasks such as detection or segmentation, or also transformer-based LLMs. If demonstrated, such generalization could substantially strengthen the paper’s contribution and broader impact.

### Questions
q1: In your experiment, why do you use the distilled version of DeiT-S only but not the other two model sizes?
q2: In figure 4, it seems that in most cases, one component will dominate the other choice. so I'm curious how the following settings works: just calculate per-layer error and always choose the dominant choice regardless of the head. 
q3: I'm curious about what is the ratio of choosing Q/K and choosing V/O, it seems that K is dominating Q and O is dominating V.
q4: What is the computational cost of the head-wise dynamic selection during inference

### Soundness
3

### Presentation
4

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
This paper introduces UniSVD, a novel weight decomposition method for compressing attention modules in vision transformers. Unlike conventional approaches that decompose all weight matrices or combined weight products, UniSVD selectively decomposes only one side of the Q-K and V-O weight pairs based on their sensitivity to low-rank approximation. The selection is performed dynamically at the head and layer level using Frobenius norm-based approximation error. The authors demonstrate consistent improvements when UniSVD is integrated with existing SVD-based compression methods on DeiT models.

### Strengths
1 - The unilateral decomposition idea cleverly exploits the linear nature of Q-K and V-O operations in attention mechanisms.

2 - Table 1 shows substantial gains across multiple baseline methods (e.g., 8.5% improvement for ASVD on DeiT-Small).

3 -  The paper includes thorough ablation studies examining dynamic vs. fixed selection strategies and head-wise selection patterns.

### Weaknesses
1. **Limited scope and evaluation**: 
   - Only evaluated on DeiT models with ImageNet-1K
   - No experiments on larger/recent models (ViT-Large, DINOv2, etc.)
   - Cannot be applied to MLP layers, limiting overall compression potential

2. **Lack of theoretical analysis**: No formal analysis of why unilateral decomposition preserves more information than alternatives. The singular value analysis in Figure 2 is superficial and doesn't rigorously support the claims.

3. **Computational overhead concerns**: The dynamic selection requires computing SVD for both weights to make decisions, potentially negating efficiency gains. This overhead is not analyzed.

4. **Missing critical comparisons**: 
   - No wall-clock time measurements for the selection process
   - Limited baseline comparisons (only SVD-based methods)

5. **Inconsistent experimental details**:
   - Table 1 shows different parameter counts for FLAR-SVD (49.2M) vs others (44.1M) without explanation
   - The "hierarchical strategy" for rank assignment is mentioned but not detailed

6. **Weak empirical analysis**: The claim that combined weights have "noticeably larger singular values" (Figure 2) lacks quantitative support and statistical testing.

### Questions
1. What is the actual computational overhead of the dynamic selection process? Computing SVD for both weights seems expensive. Can you provide wall-clock time comparisons?

2. Why limit evaluation to DeiT models? Have authors tested on ViT-Large, Swin Transformers, or recent models like DINOv2? How does the method scale?

3. Can authors please provide a theoretical analysis?  Why should unilateral decomposition preserve more information than combined decomposition from an information-theoretic perspective?

4. How does UniSVD compare with structured pruning or quantization methods in terms of accuracy-efficiency trade-offs?

5. What is the sensitivity to rank selection? How robust is the dynamic selection to different target ranks?

6. Why do parameter counts differ in Table 1 between FLAR-SVD and other methods for the same model?

7. Can you provide quantitative analysis for Figure 2? Statistical tests comparing singular value distributions would strengthen your claims.

### Soundness
2

### Presentation
3

### Contribution
2
