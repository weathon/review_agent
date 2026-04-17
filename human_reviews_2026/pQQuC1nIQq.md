# Understanding and improving Shampoo and SOAP via Kullback-Leibler Minimization

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Shampoo and its efficient variant, SOAP, employ structured second-moment estimations and have shown strong performance for training neural networks (NNs). In practice, however, Shampoo typically requires step-size grafting with Adam to be competitive, and SOAP mitigates this by applying Adam in Shampoo’s eigenbasis---at the cost of additional memory overhead from Adam in both methods. Prior analyses have largely relied on the Frobenius norm to motivate these estimation schemes. We instead recast their estimation procedures as covariance estimation under Kullback-Leibler (KL) divergence minimization, revealing a previously overlooked theoretical limitation and motivating principled redesigns. Building on this perspective, we develop \textbf{KL-Shampoo} and \textbf{KL-SOAP}, practical schemes that match or exceed the performance of Shampoo and SOAP in NN pre-training while achieving SOAP-level per-iteration runtime. Notably, KL-Shampoo does not rely on Adam to attain competitive performance, eliminating the memory overhead introduced by Adam. Across our experiments, KL-Shampoo consistently outperforms SOAP, Shampoo, and even KL-SOAP, establishing the KL-based approach as a promising foundation for designing structured methods in NN optimization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes KL-Shampoo and KL-SOAP, which modify the Shampoo and Soap algorithms to try to find the best approximation to the gradient second moment in the KL norm rather than the Frobenius norm. In practice this yields an update rule where current estimates of the second moment of the RHS covariance are used to update the LHS covariance and visa-versa. The paper also proposes to use QR updates, but combined with frequent estimates of eigenvalues via EMA. Empirical results show gains over the baselines.

### Strengths
1. The paper provides a new perspective on pre-conditioning for LLM optimization that yields principled algorithms.
2. The presentation of related work and how the KL methods fit in is very nice.
3. The method seems to work better than baselines in the experiments. And hyper parameter tuning seems relatively fair across baselines and runtime/memory is considered.

### Weaknesses
1. It’s a lot to cover, but the presentation of the algorithm itself could be improved. It is too nonlinear with options here and there. Some clear, linear algorithm boxes for each separate algo in the appendix could provide clarity. 
2. I understand there are computational constraints, but the experiments are small-scale, so it is not clear whether the gains with scale. Perhaps even more than scaling up model size (which can get expensive), I would be interested to see substantially longer runs at the same size. The current experiments are effectively *very* early in training (less than chinchilla and far below the >>100 token-to-param rations that are more standard now), so it is not clear whether the gains matter later on as well.

### Questions
1. In figure 3, why do the delta definitions have two lines each?
2. In figure 3 bottom right, where is the modification that defines “augmented”?
3. Are the output layers and/or layernorms treated differently in the empirical implementation (as is often the case with similar methods)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors recast diagonal estimation as covariance estimation via KL minimization: they treat the true second moment $E[gg^\top]$ as a Gaussian covariance and seek the vector $d$ that minimizes the KL divergence between $\mathcal{N}(0, E[gg^\top])$ and $\mathcal{N}(0, Q,\mathrm{Diag}(d),Q^\top)$. They prove that the KL-optimal $d$ equals the expected squared gradient in the eigenbasis, which provides a clean justification for SOAP’s “RMSProp in an eigenbasis” update as the solution to a KL problem. From this view, the classic Kronecker estimators in Shampoo and SOAP are suboptimal for the KL objective, which motivates revised estimators. They propose KL-SOAP, which keeps SOAP’s basic pattern but replaces Shampoo’s factor estimation with KL-Shampoo’s estimation to obtain a better eigenbasis $Q$.

### Strengths
The paper tackles a well motivated problem and offers a clear explanation. The literature typically justifies SOAP with Frobenius norm approximations [1]. The authors replace that with a covariance estimation view that minimizes KL divergence between zero mean Gaussians. Under this view, the usual Kronecker estimators in Shampoo and SOAP do not solve the right objective when both factors are learned jointly, which motivates a two sided estimator. This reveals a limitation in the standard Kronecker update and leads to principled fixes, presented as KL Shampoo and KL SOAP. The result is better performance at roughly SOAP level iteration cost.


[1] Morwani, Depen, et al. "A New Perspective on Shampoo's Preconditioner." arXiv preprint arXiv:2406.17748 (2024).

### Weaknesses
The paper argues that the Frobenius-norm view misses the right geometry and proposes a KL objective instead. Since methods like SOAP ultimately aim to approximate a preconditioner, it is natural to assess estimation quality by a distance such as the Frobenius norm, and it should not be surprising that different distance functions produce different optimal approximations. I appreciate the point that reconstruction error is not the final goal, yet the justification still feels incomplete. The paper does not fully persuade me that the KL objective, rather than Frobenius or another principled choice, is the uniquely appropriate metric for evaluating structured preconditioners in this setting.



The paper presents KL-SOAP as a principled improvement to SOAP, but the mechanical change is small. In practice the algorithmic delta is limited to swapping SOAP’s Shampoo-based eigenbasis for a KL-Shampoo eigenbasis and adding an EMA rule for factor eigenvalues under basis staleness. The step form, the augmented diagonal, and the memory profile remain essentially the same. The narrative should reflect that KL-SOAP is best understood as SOAP with a stronger basis construction and more careful eigenvalue tracking, not as a distinct optimizer.

The experiments never report approximation error under the proposed KL objective.

### Questions
Please address my concerns above

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a KL-divergence perspective of looking at the updates provided by Shampoo and SOAP. It provides a new update rule for the kronecker factored preconditioner based on the KL divergence perspective and provides empirical results for the same. Empirically, KL-Shampoo and KL-SOAP outperform Shampoo and SOAP on various language modeling benchmarks.

### Strengths
1. The paper provides a novel perspective of viewing the Kronecker factorization updates based on KL-divergence.

2. It provides a new update rule for the kronecker factor updates based on this perspective, and also provides a computationally efficient way for implementing it.

3. The empirical results of the method are pretty strong, outperforming recent second order methods such as Shampoo and SOAP.

### Weaknesses
1. The paper is missing comparisons to Muon, another popular second order optimizer proposed recently.

### Questions
1. Can the authors add comparison to Muon as well - for both runtime and number of iterations?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper reinterprets the preconditioner estimation in Shampoo and SOAP as a covariance estimation problem solved via Kullback-Leibler (KL) divergence minimization.  This perspective may reveal a previously overlooked limitation in the one-sided estimation approach of Shampoo. Building on this insight, the authors propose new estimation rules, leading to KL-Shampoo and KL-SOAP. Through  experiments on multiple language models, the authors demonstrate that KL-Shampoo and KL-SOAP outperform Shampoo and SOAP.

### Strengths
- The KL divergence view is a well-motivated approach. The connection to covariance estimation and the justification via proximal-gradient steps are interesting. 

- The proposed methods are not just theoretical constructs; they are practical algorithms that deliver comparable performance over Shampoo and SOAP.

- The paper is generally well-written.

### Weaknesses
- The authors only demonstrate that KL-Shampoo and KL-SOAP outperform Shampoo and SOAP. They do not compare with other state-of-the-art large scale algorithms.

- The authors do not provide any theoretical convergence guarantees.

### Questions
Page 3, Figure 1 Caption: "pre-iteration runtime",  "per-iteration runtime"?


Appendix G: It might be helpful to briefly summarize the conclusions of Fig. 7 and Fig. 8 in the main text, as they provide strong support for the choice of KL divergence over VN divergence.

Could the authors provide theoretical guarantees, showing the advantage of the proposed algorithms?

Could the authors provide more numerical results to compare with other emerging large scale algorithms?

### Soundness
3

### Presentation
3

### Contribution
3
