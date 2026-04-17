# OD-LoRA: Overcoming the Dilemma between Weight Representation and Gradient Approximation in Low-Rank Adaptation

- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
Low-Rank Adaptation (LoRA) enables efficient adaptation of large pre-trained models to downstream tasks by representing weight updates with trainable low-rank (LR) matrices.
Recent studies have shown a different perspective that learning with LoRA is equivalent to using a low-rank approximation of the full fine-tuning gradient, obtained by mapping it onto low-rank subspaces through the LR matrices.
In this paper, we theoretically show that LoRA faces a dilemma between these two perspectives: weight update representation and gradient approximation.
We first demonstrate that the quality of gradient approximation is improved if the LR matrices have uniform singular values, since non-uniform singular values anisotropically distort the projection of the full gradient onto the subspaces.
However, this condition entails a strict constraint on the weight updates, significantly compromising their representational capacity.
To Overcome this Dilemma, we introduce a new method, named OD-LoRA, which decouples the approximated gradient from the singular values of the LR matrices.
Specifically, OD-LoRA ensures that the full gradient is mapped through the orthonormal bases of the low-rank subspaces defined by LR matrices, achieving perfect projection onto the subspaces, while still allowing the singular values to represent the weight updates.
Consequently, OD-LoRA achieves both the optimal condition for accurate gradient approximation and unconstrained representation of weight updates simultaneously.
The experimental results on natural language and vision benchmarks demonstrate that OD-LoRA improves loss convergence and gradient approximation quality, significantly enhancing the adaptation performance of LoRA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper identifies a theoretical dilemma in LoRA between weight update representation and gradient approximation. To address this, the authors propose OD-LoRA, which decouples the approximated gradient from the singular values of low-rank matrices, enabling accurate gradient approximation while preserving full representational capacity. The paper includes theoretical analysis and comprehensive experiments.

### Strengths
- The paper provides a novel and insightful perspective on LoRA by analyzing the alignment between the full gradient and low-rank subspaces.

 - The presentation is clear and logically structured, making the theoretical development easy to follow.

 - The experimental evaluation is comprehensive, covering multiple model sizes and domains, and demonstrates consistent improvement over baselines.

### Weaknesses
- W1. The paper argues that uniform singular values benefit gradient approximation, but this condition is not strictly necessary for minimizing angular distance. Prior works ([1] and [2]) on gradient projection in LoRA suggest the scaled gradient descent can project the full gradient onto low-rank spaces perfectly without enforcing uniform singular values. The paper also fails to compare with these key references, making its claimed novelty somewhat suspicious. The authors should clearly justify the necessity of this condition and clarify how OD-LoRA fundamentally differs from existing gradient-projection formulations.

- W2.  The statement (Line 266) that low-rank adapters must be initialized to zero is not valid. Several methods (e.g., LoRA-GA) adopt non-zero initialization by modifying pretrained parameters. Moreover, OD-LoRA comes to be the standard LoRA for the first few steps, which does not address gradient misalignment during early training. It would be valuable to discuss integrating the uniform singular value condition with alternative optimizers (e.g., Muon) and contrast against AdamW.


- W3. The algorithm is theory-inspired, but the paper lacks formal guarantees (e.g., convergence). Even a modest convergence statement or stability analysis would strengthen the theory.

- W4. Gradient-based baselines (LoRA-GA, LoRA-Pro) are compared on LLaMA2-7B, while the main experiments use LLaMA3-8B. Including these baselines on LLaMA3-8B (Instruct) would make the empirical case more convincing.


[1] Zhang F, Pilanci M. Riemannian preconditioned lora for fine-tuning foundation models[J]. arXiv preprint arXiv:2402.02347, 2024.

[2] Yu X, Wang Y, Chen J, et al. AltLoRA: Towards Better Gradient Approximation in Low-Rank Adaptation with Alternating Projections[J]. arXiv preprint arXiv:2505.12455, 2025.

### Questions
- Q1. The analysis focuses on the gradient alignment but ignores the effect of optimizer momentum (e.g., AdamW). Since momentum changes the effective update direction, how would this affect the theoretical claims?


- Q2. Typo error: Line 314: "trauncated" $\rightarrow$ "truncated".

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper provides a theoretical analysis of LoRA, demonstrating that LoRA’s use of low-rank matrices A and B attempts to simultaneously achieve two conflicting objectives: (1) accurately approximating the full gradient, which requires uniform singular values, and (2) flexibly representing weight updates, which requires the singular values to be freely learned. To resolve this conflict, the paper proposes the OD-LoRA method, which decouples low-rank gradient approximation from the learning of singular values by utilizing an orthogonal basis defined by the LoRA subspace. Experimental results across multiple tasks show that OD-LoRA achieves superior performance compared to existing approaches.

### Strengths
1.The paper clearly and theoretically reveals the inherent trade-off in LoRA between accurate gradient approximation and weight representation capability.

2.It proposes the OD-LoRA method, which leverages an orthogonal basis defined by the LoRA subspace to decouple low-rank gradient approximation from the learning of singular values, effectively resolving the aforementioned dilemma.

3.Extensive experiments on multiple benchmark datasets demonstrate the superiority of the proposed method compared to LoRA and its variants.

### Weaknesses
1.In OD-LoRA, the initialization stage involves two hyperparameters, N and T, which are fixed in the experiments. It would be helpful to include ablation studies or sensitivity analyses to verify how different choices of these values affect performance.

2.Since LoRA-based fine-tuning often suffers from catastrophic forgetting, it would be valuable to investigate whether OD-LoRA also exhibits similar issues after fine-tuning.

### Questions
Please refer to the Weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new LoRA-based PEFT method called OD-LoRA. OD-LoRA addresses a fundamental dilemma between the gradient approximation and maintaining full representational capacity in low-rank weight updates. The authors first prove that LoRA achieves minimal gradient approximation error only when its low-rank matrices have uniform singular values, but such uniformity severely limits their representational capacity. To overcome this, OD-LoRA decouples the gradient approximation process from singular values by leveraging orthonormal bases of the LoRA subspaces to preserve the gradient direction.

### Strengths
* The idea of decoupling through orthonormal bases is conceptually elegant and technically novel.
* The paper is built on a strong theoretical foundation, which provides valuable insights into the geometric characteristics of LoRA.

### Weaknesses
* The frequency of computing the SVD of $\Delta W$ is not specified. Moreover, could you provide the time cost associated with the SVD steps in the ablation study? Intuitively, performing SVD frequently may yield more precise directions for $A$ and $B$, but it may also frequently reset the optimizer states, which may not help to improve the overall performance.

* In Algorithm 1, the initialization $A = 0, B = 0$ is confusing. Similarly, in Line 308, the statement “since $A = B = 0$” is unclear. If $A$ and $B$ are both set to zero in the algorithm, how are the backward and update steps for $A$ and $B$ performed? In the original LoRA design, the initializations for $A$ and $B$ are not both zero.

* Gradient-based methods (LoRA-GA, LoRA-Pro) are compared only on LLaMA2-7B with two tasks, which is not consistent with the main experimental results presented in Tables 1 and 2. I think comparing with these methods is important to show the effectiveness of the proposed method. 

* Moreover, what are the learning rate and experimental settings for Table 5? If the settings used in Table 5 correspond to those shown in Table 9, then the comparison in Table 5 may be unfair. As shown in Table 9, OD-LoRA adopts a learning rate of 2e-4, 1 training epoch, and a batch size of 32. However, this configuration is not consistent with the original LoRA-GA paper, while the LoRA-GA row in Table 5 is exactly the same as that in Table 2 of the LoRA-GA paper.

### Questions
See questions in the weaknesses part.

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
The paper argues that LoRA faces a dilemma between two roles: weight representation, as it must flexibly represent diverse low-rank updates, and gradient approximation, as its gradient corresponds to a low-rank projection of the full fine-tuning gradient. The authors claim this dilemma arises because the gradient is best approximated when the singular values of the LoRA factors are uniform, but this condition simultaneously limits representational capacity. To address this, they propose OD-LoRA, which decouples gradient computation from the singular values by using orthonormal bases obtained via QR decomposition. This reparameterization aims to achieve accurate gradient approximation and unrestricted weight representation, leading to slightly faster convergence and modest but consistent improvements over standard LoRA and recent variants across NLP and vision benchmarks.

### Strengths
1. Clear theoretical framing: The paper formalizes an internal tension in LoRA between representing flexible low-rank updates and approximating the full fine-tuning gradient, presenting the argument in a clean, mathematically consistent way.
2. Simple and efficient modification: OD-LoRA is easy to implement and incurs negligible computational overhead.
3. Comprehensive experiments: Evaluation spans language, reasoning, code, and vision tasks with several model sizes, showing the method's generality.
4. Consistent empirical gains: OD-LoRA typically converges faster and outperforms LoRA and recent variants (rsLoRA, DoRA, PiSSA) by small but steady margins.

### Weaknesses
1. Conceptual overreach of the “dilemma”: The paper’s central claim that LoRA’s non-uniform singular values cause a fundamental trade-off between gradient approximation and representational capacity is formally true but empirically unsubstantiated. There is no evidence that this dilemma actually affects LoRA’s training dynamics or performance in practice. The analysis basically assumes that better approximation of the full fine-tuning gradient improves performance, but this is not necessarily true. But why is full fine-tuning the gold standard when it often overfits or destroys pretrained representations? LoRA’s success arguably stems from restricting learning to low-dimensional subspaces rather than reproducing full fine-tuning behavior. Can you provide some empirical evidence that connects gradient misalignment to LoRA performance?

2. Questionable metric for gradient approximation: The cosine similarity between the full gradient and its low-rank projection is a weak and potentially misleading metric. It operates in euclidean space and measures direction only, not loss changes. This might be ignoring the fact that some directions are more important than others. I would suggest exploring alternative notions of "approximation" and justifying why the one chosen by the authors is more adequate here. 

3. Limitations of the experimental evidence: while the experiments are extensive and carefully executed, they do not convincingly validate the paper’s central claims. The reported accuracy gains are modest and consistent but never traced to the proposed mechanism: the experiments measure only downstream performance, not whether LoRA actually suffers from the claimed trade-off or whether OD-LoRA resolves it. The ablation studies confirm internal consistency but remain correlational, as multiple confounding factors such as orthogonalization, initialization resets, and altered optimization dynamics could explain the improvements. Appendix D’s singular-value analysis merely shows that full fine-tuning produces anisotropic spectra, without comparing LoRA or OD-LoRA or linking spectral shape to performance. As a result, the empirical section supports that OD-LoRA works slightly better in practice but fails to establish why it does or whether the theoretical “dilemma” is a real limitation in existing LoRA methods.

### Questions
1. Can the authors provide empirical evidence that the proposed “dilemma” between representational capacity and gradient approximation actually arises during standard LoRA training?

2. Why should approximating the full fine-tuning gradient be considered desirable, given that LoRA’s efficiency often comes from deviating from full fine-tuning?

3. Are there theoretical conditions under which better gradient alignment provably improves convergence or generalization?

4. Could the authors show that OD-LoRA’s improved performance correlates with reduced gradient misalignment, rather than to unrelated optimization effects?

### Soundness
2

### Presentation
3

### Contribution
2
