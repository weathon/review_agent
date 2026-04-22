# PiCa: Parameter-Efficient Fine-Tuning with Column Space Projection

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Fine-tuning large foundation models is essential for building expert models tailored to specialized tasks and domains, but fully updating billions of parameters is computationally prohibitive. Reducing the number of trainable parameters using Parameter-Efficient Fine-Tuning (PEFT), such as Low-Rank Adaptation (LoRA), is therefore crucial not only to reduce training costs but also to mitigate storage, caching, and serving overheads during deployment. Prior works, such as Singular Vectors-guided Fine-Tuning (SVFT), have shown that exploiting the geometry of pre-trained weights based on Singular Value Decomposition (SVD) can significantly improve parameter-efficiency, but they lack a solid theoretical foundation. In this paper, we introduce Parameter-Efficient Fine-Tuning with Column Space Projection (PiCa), a novel theoretically grounded PEFT method. We prove that projecting gradients onto the principal column space of pre-trained weights provides an effective inductive bias for adaptation and further enhance parameter efficiency through a novel weight-sharing strategy. Across diverse NLP and vision tasks, PiCa consistently outperforms state-of-the-art baselines under comparable or smaller parameter budgets, demonstrating both theoretical rigor and practical effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PiCa, a novel and theoretically grounded method for parameter-efficient fine-tuning (PEFT) of large foundation models. The core idea is to project gradients onto the principal column space of pre-trained weights, a space empirically and theoretically shown to preserve fine-tuning directions with minimal update error. To further reduce trainable parameters, PiCa integrates a weight-sharing mechanism, where shared compact parameters are used across layers within the same functional group. The proposed method is evaluated across a wide range of NLP (Mathematical Reasoning, Commonsense Reasoning, GLUE) and vision tasks (VTAB-1K, DreamBooth). PiCa outperforms state-of-the-art PEFT baselines, including LoRA, DoRA, SVFT, and VeRA, under equal or lower parameter budgets.

### Strengths
- **Innovative method with practical benefits:** The combination of gradient projection and parameter sharing is novel and effective. The layer-specific projection matrices allow for structural adaptivity, while shared compact parameters reduce redundancy, enabling PiCa to use up to 13× fewer parameters than LoRA without sacrificing performance.

- **Comprehensive experiments:** PiCa is extensively evaluated across three NLP subfields and two vision domains, including strong baselines. The method shows superior average performance across tasks such as GLUE, GSM-8K, and VTAB-1K, highlighting its general applicability beyond a single modality.

### Weaknesses
- **Inference-time overhead not deeply addressed:** PiCa requires computing (or storing) the top singular vectors of each pre-trained layer for projection. Although the authors mention this as a minor load-time overhead, more discussion or benchmarking of real-world inference-time latency/storage trade-offs would be helpful.

- **Lack of comparison on long-context or sequence-heavy tasks:** While the NLP evaluations are diverse, no experiments are shown on long-context reasoning, e.g., document-level QA or summarization. Since PiCa modifies fine-tuning dynamics, it remains unclear how it scales in such scenarios.

### Questions
- **How sensitive is PiCa to the choice of projection rank $r$?** Although some rank settings are explored, it would be helpful to know if PiCa is **robust across a range of $r$**, especially in lower-resource settings. Could adaptive rank selection (e.g., via energy thresholding) be incorporated?

- **Can PiCa be combined with LoRA or orthogonal fine-tuning?** Given its theoretical grounding, could PiCa act as a drop-in enhancement to existing PEFT layers like LoRA by refining the directionality of updates?

- Can the authors comment on scalability in terms of actual wall-clock latency or GPU memory?

- The shared representation $\theta_f$ and structured projection matrix offer intriguing possibilities for compositional fine-tuning. Could this architecture be extended to **task-conditioned adaptation**?

### Soundness
3

### Presentation
2

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
This paper introduces PiCa, a theoretically grounded PEFT method that exploits of the spectral geometry of the pre-trained weights. PiCa projects gradient updates onto the principal column space (top-rank singular vectors) of each pretrained weight matrix, supported by theoretical analysis showing this projection yields a near-optimal low-rank update. To further reduce trainable parameters, PiCa also shares the projected update matrix across layers of the same type. Experiments across multiple NLP and some vision tasks demonstrate consistent state-of-the-art performance compared to previous SoTA PEFT baselines with fewer trainable parameters.

### Strengths
- By projecting gradients onto the top singular vectors’ subspace, PiCa leverages a mathematically principled inductive bias. The authors prove that this yields a near-optimal low-rank update (minimizing Frobenius error) up to small residual terms.

- Extensive experiments validate that PiCa achieves competitive or superior performance while using fewer parameters than baselines.

### Weaknesses
- PiCa’s theoretical guarantee (Theorem 1) assumes that the transformation from pre-trained to fine-tuned singular vectors is almost identity, i.e. fine-tuning doesn’t drastically alter the singular directions. While empirically supported for conducted experiments, this might not universally hold. If a downstream task requires introducing entirely new features or directions that were not significant in the pre-trained model, limiting updates to the original top-r singular space could be restrictive. In cases where the task’s optimal solution exists outside the column space of the pretrained weights (e.g., OOD tasks), PiCa would then incur an approximation error equal to the omitted singular components.

- Weight-sharing theoretically reduces flexibility; broader ablations on when sharing hurts would strengthen claims. Running PiCa on some OOD downstream tasks, or on less capable base LLM models would offer more insights.

### Questions
1. Could you apply the same weight-sharing scheme to SVFT so both methods use a comparable trainable-parameter budget? This would isolate the effect of PiCa’s rank-r projection versus SVFT’s full-rank update.

2.  Have you assessed robustness to decoding hyperparameters (e.g., temperature, top-p) and reported mean ± std over multiple seeds? Re-running with varied sampling settings would strengthen claims of statistical reliability.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a PEFT method which involves, for any given weight matrix, using the singular vector spaces of that matrix to determine which parameters to update. In particular, if a matrix W has SVD given by USV', and the top-r components are U_r and V_r, this paper proposes a low-rank update of the form U_r P V_r' - where P is a small square matrix of their trainable parameters. 

Further, to decrease parameter count, they do weight-tying of these P across groups of layers.

They show improvements over a few other methods that do similar things for PEFT.

### Strengths
The method is clearly written. While not exactly theory, there is mathematical grounding of their method presented. They have compared against the correct set of competing methods.

### Weaknesses
The main issue with this paper is that it is a special case of the SVFT method, with weight tying. In particular (see the notation above in the "Summary" of this review), in SVFT the update is of the form U Q V' where Q is a matrix with <some> pre-determined sparsity pattern. The only difference in this paper is that this sparsity pattern is fixed to be the r x r "top left" square corresponding to the main singular values.

The other innovation is weight tying across layers, a standard parameter-saving technique. Indeed it is likely the gains over SVFT arise mainly from weight-tying, and not some fundamental algebraic structural difference of the update. A fairer comparison would be with weight-tying applied to other methods as well.

### Questions
What does specifically does Theorem 2 add to the method from an algorithmic perspective? It is a pretty standard derivation from Lipschitz analysis (which, also does not hold for these models which have ReLUs in them).

How is it that PEFT methods beat fullFT even in accuracy for the 7B and 8B models (Table 2)? Was the FullFT done properly?

### Soundness
3

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
4

### Summary
This paper introduces PiCa, a new Parameter-Efficient Fine-Tuning (PEFT) method. The core idea is to improve parameter efficiency by providing a theoretically-grounded inductive bias. The method operates by projecting gradients onto the principal column space of the pre-trained weights, which the authors argue is the most effective subspace for adaptation. This gradient projection is then combined with a novel weight-sharing strategy, where layers with the same function (e.g., all "query" matrices) share a single set of trainable parameters, further reducing the parameter budget.


The authors provide a theoretical analysis to support this approach. Theorem 1 aims to justify that the principal column space captures the ideal update, based on the observation that fine-tuned weights are a small perturbation of the pre-trained weights . Theorem 2 argues that PiCa's practical "sequential projection" algorithm is a high-fidelity approximation of an "ideal" but computationally infeasible "accumulated projection" . Empirically, the paper demonstrates that PiCa achieves strong results on a wide array of NLP and vision tasks, often outperforming baselines like LoRA and SVFT with a significantly smaller parameter count

### Strengths
1. The paper tackles the problem of parameter-efficient fine-tuning, which is of critical importance for the practical adaptation and deployment of large-scale foundation models.

2. The paper is well-written and clearly structured. The authors effectively articulate the components of their method (gradient projection and weight-sharing) and the high-level intuition behind their theoretical claims.

3. The experimental evaluation is commendable for its broad coverage of foundation models. The method is tested across a diverse set of backbones, including both vision and language foundation models.

### Weaknesses
My main concerns lie with the theoretical justification, which feels less convincing than presented, and the comprehensiveness of the experimental baselines, which are insufficient to fully support the claims of state-of-the-art parameter efficiency.

1. The paper claims to provide a strong theoretical foundation, contrasting with prior "heuristic" SVD-based works. However, I am not fully convinced by Theorem 1's argument for projecting onto the column space. The justification relies on the finding that Full Fine-Tuning (FFT) produces a small deviation (gap) from the pre-trained weights . This "small deviation" is often an artifact of the FFT setup itself (e.g., low learning rates, few epochs) which is intentionally designed to prevent catastrophic forgetting. Thus, the argument feels somewhat circular: it observes that a "forgetting-constrained" method produces a small change, and then builds a theory based on this small-change assumption, rather than proving that the optimal unconstrained solution must lie in this space.

2. Both theorems provided upper bounds for the errors. These bounds look incomplete to me, in the sense that they are not proven to be optimal or tighter than bounds that could be derived for other projection (or even pruning) strategies. They do not, by themselves, demonstrate that this specific projection policy is inherently superior to any other. 

3. The entire theoretical analysis is based on matrix norms (e.g., Frobenius norm). A small approximation error in the matrix norm does not necessarily or directly translate to higher parameter efficiency or better final task performance. This represents a significant gap between the theoretical claims and the practical outcomes.

4. The paper's core hypothesis—that the principal column space (the "head" of the weights) is the correct subspace for adaptation—is debatable. A growing body of work (e.g., [1], [2]) has suggested the opposite: that the tail singular vectors or the null space of the pre-trained weights may be more important for learning new task-specific knowledge while mitigating forgetting. The paper fails to acknowledge or address this contradictory line of research.


In addition, I feel that the experiments can be enhanced: While the backbone model diversity is a strength, the diversity of the PEFT baselines is a significant weakness. The paper claims state-of-the-art parameter efficiency but omits comparisons to many crucial, high-performance PEFT methods. To truly situate PiCa's efficiency, comparisons against methods like MiLoRA ([1]which focuses on the tail-space), PiSSA ([3] which also uses principal singular vectors), AdaLoRA ([4] which dynamically allocates rank), , and RoseLoRA ([5] which prunes LoRA entries in an unstructured way) are necessary. Without these baselines, the claim to superior efficiency is not fully substantiated.



[1] MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning.

[2] AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.

[3] PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models.

[4] AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.

[5] RoseLoRA: Row and Column-wise Sparse Low-rank Adaptation of Pre-trained Language Model for Knowledge Editing and Fine-tuning

### Questions
Please see my comments above.

### Soundness
2

### Presentation
3

### Contribution
2
