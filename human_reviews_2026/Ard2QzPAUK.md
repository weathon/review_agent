# BeliefFormer: Belief Attention in Transformer

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
In this paper, we consider modifying the attention layer in Transformer to improve its generalization performance. Conceptually speaking, the standard attention layer takes the softmax-based weighted summation of V vectors as the residual signal (with a linear mapping for post-processing) when performing the skip-connection operation. Inspired by distributed optimization, we propose to first perform an orthogonal projection of the softmax-based weighted summation of V vectors with respect to the original V vectors and then take the perpendicular component instead as the residual signal (with a linear mapping for post-processing) when performing the skip-connection operation.  By doing so, the token vectors are modified relatively more along their tangent directions compared to their magnitudes. Intuitively speaking, the perpendicular component reflects a belief about the discrepancy between the weighted summation of V vectors and the V vectors themselves. We refer to the newly modified layer and the overall architecture as the belief-attention and the BeliefFormer, respectively. To further improve performance, we also design a variant of belief-attention by incorporating both the per attention-head based and global orthogonal projections, referred to as belief-attention$^{\ast}$.  Extensive experiments show that the two new variants of attention layer in Transformers lead to better performance than the standard attention for image classification over ImageNet and natural language processing when training nano-GPT2 and Llama.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper replaces the usual attention residual MH(X)Wo with an orthogonalized residual that projects the softmax-weighted value summation onto the subspace orthogonal to the original V’s direction, argued to update tokens tangentially and shows modest gains on ImageNet, OpenWebText, and CIFAR-10. A second variant (Belief-Attention*) further orthogonalizes each head’s output w.r.t. its own 
Vm, concatenates these per-head tensors and applies an extra projection Ws, yielding slightly larger gains at a small cost in parameters and compute.

### Strengths
1) Both the introduced Variants are easy to implement in PyTorch/Jax.
2) The new variants don't introduce heavy compute/wall-clock overhead.

### Weaknesses
The paper lacks empirical validation; values are reported at early to mid-training checkpoints, and the models do not appear to be well-tuned. Typically, well-tuned 20M ViT scores at least 70% accuracy on ImageNet classification, but the paper reports numbers close to 60%. It is the same with CIFAR-10, we expect the model to score above 90-95% but the paper's numbers are below 90. In addition, the paper didn't evaluate Language models' performance on downstream tasks. The paper also produced no empirical evidence that the proposed variants scale.

### Questions
How can we conclude that the performance improvements of Belief-attention* come from orthogonalization rather than simply increasing the number of parameters? For example, in some of my experiments, concatenating (MH, V) along dim = −1 and feeding it to Wo improves performance. In this setup, the input dimension of  Wo doubles, matching the parameter count of Belief-attention*. Providing ablations of parameters-matched variants like the above would increase confidence that orthogonalization is the source of the improvement.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes BeliefFormer, a drop-in modification to the Transformer attention layer that replaces the usual residual signal (the softmax-weighted sum over V) with an orthogonal-projection-based discrepancy between the aggregated value MH(X) and the original value vectors V. Concretely, it computes, per token, the component of MH(X) that is orthogonal to V and uses that as the residual before the output linear map. A variant, BeliefFormer*, adds per-head orthogonal projections and an extra linear map to capture both global and per-head discrepancies. The method is motivated by an analogy to distributed optimization (PDMM), where updates explicitly incorporate constraint residuals; here, the “belief” is the discrepancy between aggregated and original values. The authors argue this leads to updates that change token directions more than magnitudes, potentially improving generalization.

### Strengths
1. The PDMM analogy is thoughtfully developed, and the geometric analysis of the orthogonal projection (directional vs magnitude changes) is sound and intuitive, with supportive empirical diagnostics.
2. The change is minimal (a per-token orthogonal projection), adds no parameters for BeliefFormer and only one extra linear map for BeliefFormer*, and can be implemented with a few lines of code as a drop-in replacement.
3. The paper is clearly written and well structured.

### Weaknesses
1. The evaluation is limited to small-to-mid-scale settings (ViT-small/DeiT-small, nano-GPT2). There are no results on large-scale LLMs or larger ViT backbones/long-context settings. As a result, the main claim of broadly improved generalization and scalability across Transformers is not convincingly supported.
2. The PDMM analogy remains heuristic, there is no formal mapping of attention+FFN updates to a constrained optimization scheme, no convergence or generalization guarantees, and no theory showing that orthogonal projection necessarily improves optimization stability or generalization. Moreover, the desired “orthogonality” is not preserved after the output linear map W_o, undercutting the claimed mechanism without a formal treatment of how W_o and W_V interact with the geometry.

### Questions
See Weakness

### Soundness
3

### Presentation
3

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
This paper proposes the BeliefFormer architecture (with the base version belief-attention and variant belief-attention∗ ) for the improvement of the attention layer of Transformer. The core idea is to borrow the PDMM algorithm in distributed optimization, and use the orthogonal projection of the weighted sum of the V-vectors and the original V-vectors as the residual signal (instead of the direct weighted sum of the standard attention), so as to make the token vectors update more along the tangent direction and reduce the amplitude change, thus improving the generalization performance. Experiments are verified in ImageNet/CIFAR10 image classification, and nano-GPT2 for NLP tasks, and both variants outperform the standard Transformer, and the base version of BeliefFormer does not add extra parameters.

### Strengths
- In 3 types of tasks (image classification, NLP), the verification accuracy/loss of BeliefFormer and variants are better than that of the standard Transformer, with no task adaptation failure problem.
- The scheme is easy to implement and improves performance without introducing additional significant computational overhead increases

### Weaknesses
- Slight increase in computational complexity: training/reasoning time for the variants is  higher than the standard Transformer and overhead may accumulate in long sequence scenarios.
- Tested only in ViT (small models), nano-GPT2 and 3 types of basic tasks, generalization to complex tasks such as large LLMs, very long sequences or speech/image generation was not verified. Insufficient validation of applicability.
- The core of the paper I think lies in treating attn as a distribution optimization problem on a connected graph, and therefore (following the logic of the paper) how the difference metric is constructed is key. Then why orthogonal projection is used, the paper does not give a detailed argument for this, which is a problem.
- In addition, the paper tries to explain the attn process by adopting the idea of PDMM for distribution optimization, but this explanation part lacks detailed arguments, and the authors directly regard the residual part of the attn computation as the residual part of the Lagrange multiplier, and again regard the MHA part as the information cohabitation part. So, the question here is, for the input X in attn is it regarded as the optimization objective X. Then is the V-vector regarded as the multiplier? The paper does not describe these details clearly, and the proposed role of the PDMM is confusing, as I could not make a proper connection between the PDMM and the computation of attn in the way it is described in the paper.

### Questions
see Weakness

### Soundness
2

### Presentation
2

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
This paper proposes Belief-Attention, a variant of Transformer attention inspired by ideas from the Primal-Dual Method of Multipliers (PDMM). Instead of directly using the standard residual connection from the attention output, Belief-Attention orthogonalizes this residual with respect to the value vectors $V$, encouraging updates that change direction rather than magnitude. A variant, Belief-Attention*, performs projection per head and introduces an additional learnable matrix $W_s$. Experiments on ViT (ImageNet), nano-GPT2 (OpenWebText), and CIFAR-10 show consistent but modest accuracy and loss improvements with limited computational overhead. The method is easy to implement and generalizes across both vision and language tasks.

### Strengths
The paper’s main strengths lie in its simplicity, generality, and clarity. The proposed modification is theoretically motivated and readily applicable across various architectures without requiring retraining or a structural overhaul. Empirical results show consistent gains in both vision and language tasks, and the discussion of limitations is transparent. The approach offers an intuitive geometric interpretation, emphasizing angular changes over magnitude, which may provide insights for understanding and stabilizing deep residual learning.

### Weaknesses
The main limitation is the lack of rigorous theoretical grounding. The PDMM analogy is suggestive but not formalized, and it remains unclear when or why orthogonal projections should improve optimization or generalization. The experiments, while broad, are shallow in statistical rigor—missing repetitions, variance reporting, and ablation details (e.g., projection strength, per-layer effects, normalization variants). Numerical stability and computational cost analyses are also underexplored. Finally, the paper does not discuss prior related works on orthogonality or residual reparameterization (e.g., ReZero, cosine similarity regularization), which would help situate the contribution.

### Questions
Could the authors explicitly map PDMM variables (residuals, multipliers, discrepancy) to quantities in the attention module (MH, V, X), clarifying what is retained and what is replaced in the Belief-Attention model?

Can the authors formalize or empirically justify conditions under which orthogonalizing residuals improves generalization or convergence, perhaps by connecting to Lipschitz continuity or residual scaling analyses?

How are LayerNorms positioned relative to the projection step, and are results sensitive to pre-LN versus post-LN configurations?

Would introducing a learnable projection strength (e.g., scaling factor $\gamma$) improve stability or adaptability?

How does the model handle near-zero norms in $V$ during the orthogonalization process? Are numerical safeguards (e.g., $\epsilon$-stabilization) applied, and what impact do they have?

Do the reported gains hold for larger-scale language models or long-context causal decoders?

Are the proposed modifications compatible with efficient attention variants such as FlashAttention or linear attention kernels?

### Soundness
2

### Presentation
3

### Contribution
2
