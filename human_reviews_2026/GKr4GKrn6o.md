# Training Transformers with Enforced Lipschitz Bounds

- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Neural networks are often highly sensitive to input and weight perturbations. This sensitivity has been linked to pathologies such as vulnerability to adversarial examples, divergent training, and overfitting. To combat these problems, past research has looked at building neural networks entirely from Lipschitz components. However, these techniques have not matured to the point where researchers have trained a modern architecture such as a transformer with a Lipschitz certificate enforced beyond initialization. To explore this gap, we begin by developing and benchmarking novel, computationally-efficient tools for maintaining norm-constrained weight matrices. Applying these tools, we are able to train transformer models with Lipschitz bounds enforced throughout training. We find that optimizer dynamics matter: switching from AdamW to Muon improves standard methods---weight decay and spectral normalization---allowing models to reach equal performance with a lower Lipschitz bound. Inspired by Muon's update having a fixed spectral norm, we co-design a weight constraint method that improves the Lipschitz vs. performance tradeoff on MLPs and 2M parameter transformers. Our <=2-Lipschitz transformer on Shakespeare text reaches validation accuracy 60%. Scaling to 140M parameters, our <=10-Lipschitz transformer reaches 21% accuracy on internet text. When matching the NanoGPT baseline accuracy of 37.4%, our Lipschitz-bounded network achieves a maximum activation norm of 112, compared to about 1,872 for the unconstrained network. Our Lipschitz transformers train without stability measures, such as layer norm, QK norm, and logit tanh softcapping.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents methods for training transformers with Lipschitz bounds enforced throughout training. The authors develop spectral soft cap and spectral hammer techniques, demonstrate that Muon optimizer works better than AdamW for Lipschitz-constrained training, and scale to 140M parameter transformers. However, competitive performance requires astronomically loose Lipschitz bounds (10^122), significantly undermining the practical utility.

### Strengths
The motivation of the paper is well-rooted and is the subject of reliability for AI systems.

The paper presents an interesting avenue of research regarding Lipschitz transformers.

The authors do explore several empirical avenues for their proposed methodology and build on recent work.

### Weaknesses
My first and foremost issue is that the paper does not establish the baselines and the necessary frameworks appropriately. Especially, as this work ventures into semi-rigorous mathematics, it is critical to establish the frameworks that this work builds on properly. 
Critically, the optimizers that this paper talks about should be properly established, and their steps (or at least the relevant steps) must be clearly stated so that one can easily verify the claims of the authors and follow through the logic of the work. Despite my relatively deep knowledge of mathematical optimization, I do not have the steps of AdamW, or the very novel work Muon, in memory, and this is not acceptable for a paper that seeks to build upon these works and expand them, and study a phenomenon under these two different optimizers.
This is a concern that covers sections 3 and 4.

In my view, unless the paper fixes this issue so that it is easier to read and follow, it would not rise to the level of ICLR.

It is indeed new and somewhat odd that I see the use of the RMS norm in the literature of Lipschitz continuity. Nonetheless, this is not a big issue. However, it does cast some uncertainty on certain experimental results.
Firstly, it makes Figure 3 slightly harder to compare with existing baselines.
Secondly, it casts doubt on the Lipschitz bounds calculated for the LipsFormer in Tab1. Have the authors taken caution to make sure that the Lipschitz calculations are correct with respect to this new norm that the authors use?

I do not understand the focus of the authors on the activations. What is the significance of having small activations??
This question is regarding both the experiments and the main body.

Another point regarding activations is regarding eq (1). A residual connection as in eq 1 is always 1-Lipschitz, given block(x) is 1-lipschitz. Why do you need a bound on the activation norms??? 
I would also like to point the authors to works such as [5] that establish conditions on simple feed-forward residual connections that can be 1-Lipschitz with a non-modified residual connection. This can help the authors to make modifications to their text so that it is technically correct.


Finally, the experimental results of the paper seem subpar. I am aware that Lipschitz regularization reduces the natural accuracy of a classifier. However, the accuracy reported in this paper for CIFAR-10 is not acceptable. For example, adversarially trained CNNS [1] or resnets [2] present much higher accuracies on both $ l_2 $ and $ l_\infty $ perturbations, and certified trained methods such as those of [3] and [4] present better results. Thus, unless there is a significant difference between the setups, this is not acceptable.

The same argument follows for the results of Table 1, though I am less familiar with the baselines in this space. Unfortunately, a Lipschitz constant that surpasses the 1000s is potentially useless, and thus, the only relevant entry, I think, is the 10-Lipschitz model, which seems to have a significantly lower accuracy. Care to further explain why such a model could be useful?








[1] Zhang H, Yu Y, Jiao J, Xing E, El Ghaoui L, Jordan M. Theoretically principled trade-off between robustness and accuracy. InInternational conference on machine learning 2019 May 24 (pp. 7472-7482). PMLR.

[2] Xu Y, Sun Y, Goldblum M, Goldstein T, Huang F. Exploring and exploiting decision boundary dynamics for adversarial robustness. arXiv preprint arXiv:2302.03015. 2023 Feb 6.

[3] Fazlyab M, Entesari T, Roy A, Chellappa R. Certified robustness via dynamic margin maximization and improved lipschitz regularization. Advances in Neural Information Processing Systems. 2023 Dec 15;36:34451-64.

[4] Wang R, Manchester I. Direct parameterization of lipschitz-bounded deep networks. InInternational Conference on Machine Learning 2023 Jul 3 (pp. 36093-36110). PMLR.

[5] Araujo A, Havens A, Delattre B, Allauzen A, Hu B. A unified algebraic perspective on lipschitz neural networks. arXiv preprint arXiv:2303.03169. 2023 Mar 6.

### Questions
See weaknesses.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies training transformers with small Lipschitz constants with two optimizers: AdamW and Muon. To enforce a strong Lipschitz constraint without losing expressivity too much, the authors propose two new techniques: spectral soft capping (for Muon) and spectral hammer (for AdamW). Spectral capping works by reducing all singular values of the weight matrices with stronger effects for the larger singular values. Spectral hammer works by targeting the highest singular value in each of the weight matrices (estimated through power iteration) and setting it below a certain threshold. Since spectral hammer only targets one singular value at a time, it is more suitable for weight updates that exhibit low rank (and hence most singular values are close to zero anyways) which is the regime that Adam mostly operate in. This is not the case for Muon which encourages the update to stay high-rank, where soft spectral capping (which brings down all singular values at the same time) becomes more suitable. Empirically, both versions of  spectral capping (soft and hard) work effectively at maintaining a low Lipschitz bound while achieving high validation accuracy/low validation loss on FineWeb10B and Tiny Shakespear.

### Strengths
- The paper contains thorough empirical evaluations across over a wide range of Lipschitz bounds, allowing a fair assessment of the robustness/expressiveness trade-off of the proposed methods and other Lipschitz-regularization methods like spectral normalization

- The proposed methods are well-motivated and allow for low Lipschitz bounds while maintaining reasonable validation accuracy/loss.

### Weaknesses
- The paper does not contain enough information (especially in the main text) for me to fully assess the technical validity of the proposed methods. To name a few issues,

  - The details on how spectral hard capping is implemented are missing in the main body. The paper only says ``spectral hard cap uses the matrix sign function to approximate $\sigma \rightarrow \min(\sigma_{\mathrm{max}}, \sigma)$ on the singular values''. Matrix sign function is not defined anywhere/missing a reference. I am guessing it means $\mathrm{msign}(M) = U\mathrm{sign}(\Sigma) V^\top $ where $M=U\Sigma V^\top$, but it is never explicitly stated or referenced anywhere. It would be good if the authors could clarify this in more details. 

  - To figure out what spectral hard capping is, I read through most of Appendix C, which was referenced from the main body. It was still unclear after this point what spectral hard capping does exactly. It seems that at L938 (in the appendix), the authors referenced Cesista (2025) for the implementation details of spectral clipping (a generalization of spectral hard capping), which leads to a blogpost. After examining this reference material, the blogpost seems to propose the same method as done in the paper (\emph{e.g.}, spectral hard capping). This raises not only clarity concern but also novelty concern. 

- The proposed methods do not seem to push the Lipschitz vs. loss frontier. From both Figure 2 (transformer), 4, and Table 1, the standard spectral normalization method and the prior Stiefel manifold projection method seem to cover the same frontier achieved by the proposed methods. While it is nice to see a comprehensive study of different Lipschitz-constrained methods, it remains unclear to me what the significance of the proposed methods is.

### Questions
(1) In Appendix C.1, Line 989 to Line 992, why is $\mathrm{msign}(\beta\mathrm{msign}(W) - W) = U\mathrm{sign}(\beta I - \Sigma)V^\top $? When I tried to verify it, I could only get to the following step:

$$
\begin{align}
    &\mathrm{msign}(\beta\mathrm{msign}(W) - W)\\\\
    &= \mathrm{msign}(\beta U\mathrm{sign}(\Sigma)V^\top - U\Sigma V^\top)\\\\
    &= U\mathrm{sign}(\beta \mathrm{sign}(\Sigma) - \Sigma)V^\top 
\end{align}
$$

Perhaps I am missing something here, but this seems to be only true when $\Sigma$ has strictly positive singular values, which is not in the assumption as far as I understood. Same thing applies for the second term (i.e., $\beta\mathrm{msign}(W) - W = U\mathrm{sign}(\beta I - \Sigma) V^\top$

(2) In Table 1, spectral normalize is studied extensively. How well does your proposed methods (e.g., spectral capping) do for $\sigma_{\mathrm{max}}=1$ compared to spectral normalize?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper develops new, computationally efficient tools—spectral capping and spectral hammer—to maintain norm-constrained weight matrices during training. These techniques enable the successful training of Lipschitz-bounded transformer models with up to 140M parameters. The authors also identify that the Muon optimizer plays a crucial role in achieving stability and improving the Lipschitz–performance tradeoff.

### Strengths
1.	This paper is well-organized and clearly written.
2.	The paper addresses a practical problem, enabling the training of Lipschitz-bounded transformers.

### Weaknesses
1.	There is a growing line of work on Lipschitz neural networks, which tightly bound the Lipschitz constant through architectural design or explicit Lipschitz regularization. These represent the mainstream direction in Lipschitz-bounded models, but the submission appears to omit key references, such as:

    [1] Hu, Kai, et al. "A Recipe for Improved Certifiable Robustness." ICLR, 2024.

    [2] Xu, Xiaojun, Linyi Li, and Bo Li. "Lot: Layer-wise orthogonal training on improving l2 certified robustness." NeurIPS, 2022.

2.	The contribution is somewhat limited. The proposed approach primarily leverages norm-normalization-based regularization of weight matrices, which may not meet the ICLR standard for conceptual novelty.
3.	The experimental section is relatively weak. The paper would benefit from more extensive experiments and in-depth discussion.
4.	The study focuses solely on language models. Including results on vision transformers would make the work more comprehensive and competitive.
5.	In Table 1, the results mainly show improved training stability without clear gains in performance. If the proposed methods merely stabilize training while sacrificing accuracy, the contribution appears insufficient. Since it is commonly understood that constraining the Lipschitz constant can enhance model robustness, the paper should discuss or empirically demonstrate this benefit (or others such as low-precision stability); otherwise, the practical significance of the proposed method remains unclear.

### Questions
1.	In Table 1, the condition number of the weight matrix using spectral capping or spectral hammer is not close to 1 but rather high. Does this lead to a looser Lipschitz bound?
2.	Standard methods often use layer normalization or QK normalization, whereas this paper adopts spectral-norm-based regularization. Is this understanding correct? If so, does the proposed norm-based method serve a similar role to layer norm or related normalization techniques?
3.	Has the paper considered incorporating the concept of orthogonality, where all singular values are equal to 1? Otherwise, the Lipschitz constant may grow with network depth. How does the proposed approach address this issue?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the effects of enforcing Lipschitz bounds beyond initialization during the training of Transformer networks. Specifically, the paper removes activation normalization and constrains the weights throughout training, and finds that weight decay and spectral normalization are more effective with Muon than AdamW. Based on the conclusion, this paper further proposes spectral capping and spectral hammer to ensure the Lipschitz bounds during training. Comprehensive experiments are conducted to illustrate the impact of the technique on practical performance.

### Strengths
1. The impact of Lipschitz constraints during training of transformer-based networks is highly relevant for practical applications.
2. This paper is well-organized and easy to read.

### Weaknesses
1. More comprehensive experments on ImageNet-1K are needed to further demonstrate the generalization of the conclusion on Muon and AdamW.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
