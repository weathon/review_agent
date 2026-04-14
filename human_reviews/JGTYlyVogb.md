# Self-supervised Transfer Learning via Adversarial Contrastive Training

- Decision: Reject
- Scores: 3, 3, 8

## Abstract
Learning a data representation with strong transferability from an unlabeled scenario is both crucial and challenging. In this paper, we propose a novel unbiased self-supervised transfer learning approach via Adversarial Contrastive Training (ACT). Additionally, we establish an end-to-end theoretical understanding for self-supervised contrastive pretraining and its implications for downstream classification tasks in a misspecified, over-parameterized setting. Our theoretical findings highlight the provable advantages of adversarial contrastive training in the source domain towards improving the accuracy of downstream tasks in the target domain. Furthermore, we illustrate that downstream tasks necessitate only a minimal sample size when working with a well-trained representation, offering valuable insights on few-shot learning. Moreover, extensive experiments across various datasets demonstrate a significant enhancement in classification accuracy when compared to existing state-of-the-art self-supervised learning methods.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This paper considers the problem of contrastive learning that requires negative samples. The authors propose a variation of contrastive loss that removes the terms contrasting negative pairs from the loss and adds a new regularization term that is unbiased. Experimental results on small image datasets support the claim. The proposed loss comes with theoretical analysis, where the error for downstream tasks is proven to be bounded.

### Strengths
The proposed method is paired with a theoretical upper bound of the error for downstream tasks.

### Weaknesses
- Overall writing/organization of the paper could be improved, especially about the math part. The paper is generally not easily understandable.

- For example, <A,B>_F should be defined when first introduced.

- While G appears to be crucial throughout the paper, G is never explicitly defined, such that the overall flow of the paper is not easily understandable.

- While authors discuss biased/unbiased populations, there is no clear explanation on how exactly they are biased/unbiased.

- Experimental results are limited to small image datasets and the performance gain is overall marginal.

- L400: isn't one of the asterisk a typo?

### Questions
Please address concerns above.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes a contrastive learning loss that does not require negative sampling, with the objective framed as a min-max problem. The authors state that the empirical loss of the proposed method is an unbiased estimator of the population loss and provide a theoretical analysis guaranteeing its performance. Experiments are conducted, comparing the proposed method with some existing techniques and demonstrating improvements.

### Strengths
- The authors state that the proposed loss function does not suffer from bias between the empirical loss and the population loss.  
- Theoretical analysis is conducted on the proposed loss.

### Weaknesses
- The authors seem to claim that one advantage of the proposed method is that it does not require negative sampling. However, there are already many existing techniques that do not require negative sampling, such as Barlow Twins [3], VICREG [4], and others referenced in [5]. The paper lacks a detailed discussion and comparison with these methods, and there is no clear advantage demonstrated over them, nor any comparison in terms of performance in the experiments section.

- Another point of emphasis in this paper appears to be that the proposed loss function is an unbiased estimator of the population loss. However, this is not explicitly and clearly presented. For example, (1) is there any theoretical analysis directly showing that this proposed loss is unbiased? (2) Is there a theoretical analysis showing that all existing losses, including those mentioned in the previous points, are biased, and why?

- There needs to be a more intuitive explanation of how and why the regularization term is designed in this specific way.

- More specifically, looking at the regularization term in equations (1) or (2), to solve the inner max problem, it seems that we could set $ G = \mathbb{E}[f(x_1)f(x_2)^T] - I_d $. Then the regularization reduces simply to the norm of $ \mathbb{E}[f(x_1)f(x_2)^T] - I_d $, so I don’t see why it is necessary to frame it as a min-max problem or as some form of adversarial training. Moreover, if the above is true, after converting the regularization term, it becomes the same as the one analyzed in [2] (in fact, the entire loss is also the same; and also the same as the one proposed by [1] as mentioned by [2]), which is also very similar to the well-known Barlow Twins [3]. In this sense, I don’t see any novelty in the proposed method.

- To motivate the need for proposing an unbiased loss, the authors mention that [2] requires too strong an assumption, although it has already performed sample-level analysis. Could this point be elaborated further? Also, given the similarity between the proposed loss and the loss analyzed in [2], how does this paper address the issues present in [2]?

- According to [5], there is some unification between contrastive and non-contrastive losses. It would be beneficial to discuss the connection and differences compared to prior work in the context of [5], so that the uniqueness of the proposed loss can be more clearly understood.

- Regarding the experiments, beyond the lack of many baseline comparisons mentioned earlier, the reported improvements also seem marginal.



[1] HaoChen, Jeff Z., et al. "Provable guarantees for self-supervised deep learning with spectral contrastive loss." Advances in Neural Information Processing Systems 34 (2021): 5000-5011.

[2] HaoChen, Jeff Z., and Tengyu Ma. "A theoretical study of inductive biases in contrastive learning." arXiv preprint arXiv:2211.14699 (2022).

[3] Zbontar, Jure, et al. "Barlow twins: Self-supervised learning via redundancy reduction." International conference on machine learning. PMLR, 2021.

[4] Bardes, Adrien, Jean Ponce, and Yann LeCun. "Vicreg: Variance-invariance-covariance regularization for self-supervised learning." arXiv preprint arXiv:2105.04906 (2021).

[5] Garrido, Quentin, et al. "On the duality between contrastive and non-contrastive self-supervised learning." arXiv preprint arXiv:2206.02574 (2022).

### Questions
See the questions in the Weaknesses section.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a new contrastive self-supervised technique via ACT for transfer learning with both practical results and theoretical guarantees. The model seems to be motivated by the theoretical challenge posed by the non-commutativity between expectation and the Frobenius norm, inspiring the authors to vary the mathematical formulation into an adversarial form and adopt an alternative optimization algorithm to solve it. Moreover, they give an rigorous  analysis at  the sample level. The results reveal that the misclassification rate solely relies on the efficacy of data augmentation with high probability when the pretraining data is sufficiently large, despite the amount of data from the downstream task is quite limited as the order with respect to downstream sample size is $\frac{1}{\sqrt{\min_k n_t(k)}}$, which coincides with the definition of $K$-way, $\min_k n_t(k)$-shot learning.

### Strengths
- The way of rewriting the Frobenius norm in an adversarial form is very ingenious, which allows them to conduct error decomposition for excess risk without the assumption that $f^* \in \mathcal{F}$, referred to as the misspecified setting, which is fundamentally different from previous theoretical works. Meanwhile, in practice, this method introduces a novel discretization approach for problems where the expectation is taken within the Frobenius norm.
- This paper construct a representation $\tilde{f}$ making $\mathcal{L}(\tilde{f})$ vanished under a mild assumption 4.5, giving a great understanding to the population optimizer. The misspecified setting allows them to handle the task without having to consider the expressiveness of the network, thus eliminating the need for strong assumptions.  
- The theoretical result of this paper looks solid and impressive. Under the case of domain adaptation, the final theoretical result clarified the role of sample size from both the source domain and the target domain. It reveals that, under a strong data augmentation, the downstream classification error will be small enough even when the sample size from the target domain is pretty limited, which can be regarded as a strong explanation for few-shot learning.
- The indication for the width $W$ and depth $L$ can explain the reason why over-parameterized setting is effective in practice.

### Weaknesses
- The title of Table 2 should be upon table.
- It is  better to consider dedicating some space to provide a brief explanation of misspecified and over-parameterized setting. Additionally, it would be beneficial to clarify how the final theoretical results achieve over-parameterization.
- The definition of latent classes lacks clarity. The semantics of latent classes $C_s(k)$ should be determined by the semantic meaning of $C_t(k)$ in downstream tasks when considering the domain adaptation scenario.

### Questions
- I suggest that the numerical comparison with BarlowTwins would be  added  since the proposed  model is a little bit similar to theirs. By the way, it would be  better to  add  additional experiments with  domain adaptation although  the existing experiments are acceptable for this paper.
- Why do you adopt the extra requirement $P_t(\cup_{k=1}^KC_t^*(k)) \geq \sigma_t$ for augmentation instead of **Correctly augmented parts** (See Definition 2 in Appendix F of [1]) to tackle the case where the $\mathcal{A}(C_k)$ intersects?
- Is it possible to obtain a better order with respect to $n_s$ for downstream classification error is still unclear. I mean, whether a tighter error bound exists remains an unresolved issue.

If the authors can appropriately address the weaknesses and questions I raised, I will consider updating my score.

[1] Towards the generalization of contrastive self-supervised learning. ICLR 2023.

### Soundness
4

### Presentation
3

### Contribution
3
