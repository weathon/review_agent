# SubTrack your Grad: Gradient Subspace Tracking for Memory-Efficient LLM Training and Fine-Tuning

- Avg Score: 4.75
- Decision: Reject
- Scores: 5, 5, 6, 3

## Abstract
Training and fine-tuning Large Language Models (LLMs) demand significant computational resources and time due to their large model sizes and optimizer states. To mitigate these challenges and improve accessibility, several memory-efficient methods have been developed. Methods such as Low-Rank Adaptation (LoRA) optimize model weights within a low-rank subspace, while Gradient Low-Rank Projection (GaLore) projects gradients into a lower-dimensional space to decrease memory footprint. In this paper, we propose Gradient Subspace Tracking (SubTrack-Grad), a method that confines optimization to a compact core subspace of the gradient matrices and dynamically tracks its changes using the geometry of Grassmannian manifolds. SubTrack-Grad efficiently updates its subspace estimation by leveraging estimation errors and previously identified subspaces. Our results demonstrate that even with rank-1 updates to the underlying subspace, SubTrack-Grad achieves comparable or superior performance to GaLore, while reducing runtime by approx. 15% on an average and up to 20.57% on some datasets. Furthermore, SubTrack-Grad exhibits only a minimal runtime increase compared to GaLore when the update frequency is increased, while controlling the extent of changes via rank-1 updates, allows more frequent updates without negatively impacting convergence.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a method, SubTrack-Grad, to update gradient subspaces using rank-1 estimation of tangent vector on the Grassmannian manifold. SubTrack-Grad reduces disruptive changes in the gradient update process, leading to more stable learning dynamics. Experimental results show that SubTrack-Grad accelerates convergence and achieves superior performance compared to GaLore.

### Strengths
1. SubTrack-Grad improves GaLore by introducing a method for updating gradients in a low-dimensional space. It addresses the issue that not all layer gradients evolve within stable low-rank subspaces, thus accelerating convergence through rank-1 estimation of tangent vectors on the Grassmannian manifold.
2. Experimental results show that SubTrack-Grad outperforms GaLore in both wall-time efficiency and accuracy, demonstrating faster convergence and improved performance.

### Weaknesses
1. The method rank-1 gradient update needs further explanations about why choosing rank-1 rather than rank-r, where r can be any integer, and more supporting related works. Providing empirical comparisons with higher rank updates, or discussing potential tradeoffs between update rank and performance/efficiency can make this point valid. Additionally, examples of related work that use rank-1 updates in similar contexts would strengthen this update.
2. SubTrack claims to reduce the memory usage compared to GaLore, but still periodically (at each update interval $k$) utilizes SVD to obtain the largest singular value and the corresponding singular vectors, which is also used in GaLore. Rather than inferring from the algorithmic description, specific memory usage measurements or GPU footprint comparisons between SubTrack-Grad and GaLore would provide more concrete evidence for or against the memory efficiency claims.
3. Experimental results are insufficient to fully evaluate SubTrack-Grad’s performance. Additional experiments are needed to provide a more comprehensive assessment (please see Questions for further details).
4. Some figures require clearer presentation and explanation, for example, Figure 1 lacks legends and uses an unclear line to describe the tangent vector, and Figure 2 needs more explanation for the right part.

### Questions
1. In the introduction section, the authors proposes that SubTrack-Grad is more memory-efficient than GaLore, and in the experiments, the authors evaluates the wall-time of both methods, but can you please provide the memory usage or GPU footprint to compare these two methods? Since in SubTrack-Grad, the authors utilizes SVD when computing the rank-1 estimation of tangent vector, while GaLore also utilizes SVD in their gradient projection. Can authors please explain more about the memory saving in the SubTrack-Grad, compared to GaLore?
2. SubTrack-Grad utilizes rank-1 update to reduce the abrupt changes in the optimization process and the performance and wall-time is better than GaLore, but can authors explain and give more explanation about why choosing rank-1 rather than rank-2 or rank-3? Or can authors provide more related works which utilize rank-1 update to support the method of SubTrack-Grad?
3. For the subspace update interval $k$ in the SubTrack-Grad, the authors compare the wall-time with different intervals, but can authors provide the comparison of the performance (accuracy) affected by the different update intervals? I think this can be better to help understand the influence of this factor.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Galore's original paper employs SVD to compute the projection matrix. To enhance the efficiency of this calculation, the paper introduces Gradient Subspace Tracking (SubTrack-Grad), which optimizes within a compact core subspace of the gradient matrices and dynamically tracks changes using Grassmannian manifold geometry. Experiments test the effectiveness of SubTrack-Grad.

### Strengths
- The paper clearly presents its ideas, and the proposed SubTrack-Grad is straightforward to understand.

- The paper analyzes the convergence of the proposed SubTrack-Grad.

-  Experiments demonstrate that the proposed SubTrack-Grad's performance is comparable to Galore's, with improvements in computational efficiency.

### Weaknesses
- When compared to Galore using SVD, the actual reduction in running time is very small, and there appears to be no advantage in running time when compared to Galore using QR decomposition. The Galore compared in the paper only uses a projection matrix, and it is not necessary to use SVD, but only QR decomposition, which is more efficient.

- The paper lacks experimental comparisons with Galore on memory usage, i.e., the peak memory consumption. The reduction of running time only makes sense if the memory increase is not large, otherwise why not use a larger batch size. It is better to test the memory usages when different model sizes are used.

- The proposed SubTrack-Grad does not show a significant advantage over Galore. The paper needs to compare memory usage, running time, and performance at the same time. The reduction in running time is meaningful only when performance and memory usage are both comparable to Galore.

### Questions
- Galore can compute the projection matrix P using QR decomposition, which is significantly more efficient. The paper needs a thorough comparison with Galore using QR decomposition for projection matrix calculation.

- Memory usage should be compared with Galore. Galore requires only a low-rank projection matrix, while the proposed SubTrack-Grad necessitates storing two matrices, Gacc and the projection matrix S.  

- When reporting running time, it is better to present the total fine-tuning or training time for the same number of iterations, allowing readers to accurately assess the speedup ratio over Galore.

- Since Galore is the main compared method, it is a reasonable choice to strictly follow Galore's experimental setups.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SubTrack-Grad, a novel method for memory-efficient training and fine-tuning of large language models (LLMs). This technique focuses on confining the optimization process to a core subspace of the gradient matrices and tracking its evolution dynamically using Grassmannian manifold geometry. The key improvements claimed over existing methods, like GaLore, include computational efficiency and better runtime with minimal accuracy trade-offs. The authors demonstrate that SubTrack-Grad achieves comparable or superior performance while reducing computational costs by up to 20% in some cases.

### Strengths
The introduction of the SubTrack-Grad method, which leverages Grassmannian manifold geometry and rank-1 updates, presents an innovative approach to tracking gradient subspaces efficiently. This approach reduces computational overhead compared to periodic SVD in similar methods.
 The experimental results achieves comparable or superior performance to existing methods like GaLore.
The paper includes a theoretical analysis with convergence guarantees.

### Weaknesses
The derivation of Equation 7 appears disconnected from the preceding content. Include a reference to the original source or derivation details. Additionally, clarify the rationale for applying cosine and sine functions to the singular values in the projection update.

I can't confirm whether \( S_{t+1} \) maintains orthonormality post-update, as it is essential for the method's validity.

The justification for reduced computational load is unclear. Specifically, computing \( G_t \) (the full gradient) and updating \( S_{t-1} \) to project to \( A \) may introduce substantial overhead. This undermines claims of computational benefits. Strengthen arguments or provide concrete runtime comparisons.

The element-wise regularizer mentioned in Section 4 is undefined in the pseudocode and algorithmic description. 

The notations \( N \), \( L_A \), \( L_B \), and \( L_C \) in Theorem 4.1 require explicit definitions for clarity.

The Appendix proof claiming convergence to the global minimum during training is questionable. Provide stronger theoretical backing or clarify the assumptions that make this feasible.

Theorem 4.1 suggests projecting the gradient twice, differing from the algorithm's single-projection description. Ensure alignment between theoretical claims and practical implementation.

The proof assumes fixed projection matrices throughout training, contradicting the stated goal of dynamically tracking subspace changes. Address this inconsistency.

The paper refers to \( w \) as components from the gradient decomposition rather than model parameters. Clarify why \( w_{t+1} = w_t - \alpha \nabla \) is used in Equation 16 to avoid confusion.

### Questions
The paper has some technical issues which sounds not tangible and correct. Please see weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces SubTrack-Grad, a method to make LLM training more memory and time-efficient. Methods like GaLore reduce memory by projecting gradients into low-rank spaces using SVD. But SVD is computationally expensive. So, instead SubTrack-Grad updates a compact gradient subspace with lightweight, rank-1 adjustments on the Grassmannian manifold.

### Strengths
- SubTrack-Grad  shows reduction in memory footprint and computational cost of training LLMs . This method also reports 15-20% reduction in runtime is substantial and valuable for scaling model training.  
- The use of Grassmannian manifold geometry to track gradient subspaces is innovative. This allows the method to dynamically adapt to shifts in gradient directions avoiding frequent SVD operations.
- By controlling subspace adjustments through rank-1 updates, this method reduces abrupt changes and noise in the optimization process.

### Weaknesses
The paper in its current form has several gaps in fully addressing and validating the main claims. For instance, the main benefit of the method, which is to improve memory efficiency isn't validated in the experiments.  Could you report the peak memory usage for  SubTrack-Grad and other baselines along with perplexity and time in the experiments? 

 The benefits behind the gradient accumulation that SubTrack-Grad proposes isn't intuitively clear from the paper. The experiments are a bit vague and limited on this. In fact, the experiments in general are quite limited. The paper could benefit from experiments on "larger" models >3b parameters.

Minor: Some other related works that can be cited:

1. Adam-mini: Use Fewer Learning Rates To Gain More, Zhang et al
2. BlockLLM: Memory-Efficient Adaptation of LLMs by Selecting and Optimizing the Right Coordinate Blocks: Ramesh et al

### Questions
Refer to weaknesses section

### Soundness
2

### Presentation
2

### Contribution
2
