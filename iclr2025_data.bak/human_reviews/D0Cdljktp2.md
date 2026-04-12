## Human Reviewer 1

### Summary
This paper examines the use of Memformers for optimization in the setting of linear regression. The authors focus on two different type of Memformers with updates that similar to Momentum Methods and Conjugate gradient descent (in terms of operations required).  They test these models experimentally and compare them with Linear Transformers and the equivalent optimization methods.

### Strengths
1. The idea of exploring how different architectures perform in various optimization problems could lead to the discovery of new algorithms and can enhance our understanding on the limitations and capabilities of those architectures. 

2. A wide range of optimization methods are considered as a baseline.

### Weaknesses
1. There are no formal proofs of the two propositions, but only proof sketches which are not detailed enough. 
2. The experiments consider only up to 4 layers and compare only with linear attention transformers and not softmax based. 
3. The paper doesn't have a clear contribution. Even though the authors show that memformers can perform better than optimization methods, their results only hold for 4 layers and it's unclear whether using more steps/layers this would still be the case. It is also unclear whether Memformers perform some type of optimization algorithms of find a shortcut solution.

### Questions
1. I would suggest to the authors to add the full proofs of the propositions. In the current version it is unclear to me which is the exact theoretical statement. Is it that Memformers with the specific updates are able to perform the corresponding optimization methods ? For the result of [1]  the authors proved that the global minima for one layer of transformer is indeed one step of preconditioned gradient. For the case of multiple layers (Lemma 1), [1] assumes a specific parameterization of the weight matrices. Do the authors get an equivalent result for Memformers and assume that the weight matrices have the specific parameterization? 
2. In proposition 1 how the quantities $a_l$ and $\gamma_l$ are calculated with the Memformer? How many layers and width is needed for the simulation of the algorithm ? 
3. In the proof sketch of proposition 2 the authors state that "The full proof follows from the cumulative memory structure and the connection between attention and preconditioned gradients, as discussed in the proof steps of Lemma 1." Could the authors explain how exactly the proof follows? 
4. Did the authors tried to train more than 4 layers? If so is it observed that there is an error floor for Memformers? This has been observed in the prior work that this paper builds upon. How does Memformers perform compared to softmax based attention Transformers? 
5. Did the authors test how these models perform in out-of-distribution data? For example input values that belong in the tails of the gaussian distribution. 
6. I think suggestions 4,5 would improve the claims of the paper and would clarify whether these models learn some type of optimization algorithm or not. 
7. I understand that the main motivation of the work is to explore "what augmented Transformers can learn, as opposed to looking for “the best” algorithm.", but I think that the current experimental and theoretical results do not clarify what these models can actually learn. They seem to perform better than the considered optimization algorithms for a few steps, but this does not provide a concrete result on what they actually learn.
Could the authors clarify a bit which is the main contribution of their work?

[1]: Ahn, Kwangjun, et al. "Transformers learn to implement preconditioned gradient descent for in-context learning." Advances in Neural Information Processing Systems 36 (2023): 45614-45650.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper studies the representation power of memory-augmented Transformers (Memformers) in terms of implement linear first-order optimization methods for in-context learning of linear regression.
The authors provide theoretical constructions showing that Memformers can simulate methods like conjugate gradient descent and momentum methods that linearly combine past gradients.
Numerical experiments are conducted to show that Memformers can achieve better performance than conjugate gradient descent on random linear regression tasks.

### Strengths
1. Overall the paper is well written and easy to follow.
2. The paper studies an interesting topic on the representation power of Transformers for simulating algorithms solving in-context learning problems. The current results provide a theoretical understanding of memory-augmented Transformers.
3. The contributions of the paper are clearly summarized, and the limitations of the current study are appropriately discussed.

### Weaknesses
1. The architecture of Memformer is not well explained. The role of the memory $\{\mathbf{R}_l\}$ should be clarified.
2. Related to the above point, it would be helpful to clarify which parts of the architectures in Equation (19) and (21) are trainable (though they are mentioned in Section 3.3).
3. The results are restricted to in-context learning of linear regression.
4. The discussion about the benefit of using multi-head attention from line 456 to 460 seems interesting, but there is no formal analysis or heuristic explanation to support the claim. It would be helpful to provide more details. For example, why there is implicit regularization effect?

### Questions
1. It seems plausible to replace the memory register by using a larger hidden size in the Transformer. Can the authors compare these two approaches?
2. From the experiment results in Section 4, it seems that the trained Memformer outperforms CGD. What are the implications of this given the optimality results for CGD?
3. From Figure 4(a), it seems that the Memformer basically solves the linear regression task (log(loss)=-30) with two layers. Based on this, it seems hard to justify that Memformer is simulating certain optimization algorithms, and it is unclear how this is achieved.
4. Comparing Figure 4(a) and 4(b), it seems that batch size has a significant impact on the performance of Memformer. Can the authors provide some insights on this?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper explores the algorithmic capabilities of Transformers and investigates the potential of memory-augmented Transformers (Memformers) to learn linear first-order optimization methods. It provides theoretical justification and empirical evidence that Memformers can learn more advanced optimization algorithms based on prior work that demonstrates how Transformers can implement preconditioned gradient descent. Experimental results of training on random linear regression tasks show that Memformers are able to learn a class of optimization methods.

### Strengths
1. The paper is easy to follow and explores the algorithmic capabilities of memory-augmented Transformers.

2. Experiments show that linear first-order methods (LFOMs) learned by Memformers outperform conjugate gradient descent on training data while maintaining generalization performance. Additionally, multi-headed attention enhances Memformers’ test performance.

### Weaknesses
1. Lemma 1 demonstrates that multi-layer Transformers learn to implement preconditioned gradient descent under suitable parameterization, but the result and the full proof is directly from [Ahn et al. (2024)](https://arxiv.org/pdf/2306.00297). 

2. Proposition 1 and Proposition 2 in Section 3 should be the main theoretical results of this paper. However, the authors provide only proof sketches for these propositions rather than presenting detailed and rigorous proofs.

3. Figure 1(a) shows that LFOMs perform worse than preconditioned gradient descent on general quadratic problems. Additionally, Figure 2 and Figure 3 indicate that LFOMs’ performance on isotropic test data falls short of conjugate gradient descent, contradicting the claimed good generalization performance in the main contributions.

### Questions
1. Could you provide full, detailed proofs for Proposition 1 and Proposition 2? Without these, the theoretical results lack sufficient rigor and are less convincing.

2. It is mentioned in Section 6.1 that Transformers can implement second-order methods like Newton’s method, which typically outperform LFOMs in convergence speed and accuracy. However, first-order methods are more popular than second-order methods in practice, especially in deep learning. Could you provide an explanation for that?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
4