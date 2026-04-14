# Differentially private optimization for non-decomposable objective functions

- Decision: Accept (Poster)
- Scores: 5, 3, 5, 1, 6

## Abstract
Unsupervised pre-training is a common step in developing computer vision models and large language models. In this setting, the absence of labels requires the use of similarity-based loss functions, such as the contrastive loss, that favor minimizing the distance between similar inputs and maximizing the distance between distinct inputs. As privacy concerns mount, training these models using differential privacy has become more important. However, due to how inputs are generated for these losses, one of their undesirable properties is that their $L_2$ sensitivity grows with the batch size. This property is particularly disadvantageous for differentially private training methods, such as DP-SGD. To overcome this issue, we develop a new DP-SGD variant for similarity based loss functions --- in particular, the commonly-used contrastive loss --- that manipulates gradients of the objective function in a novel way to obtain a sensitivity of the summed gradient that is $O(1)$ for batch size $n$.  We test our DP-SGD variant on some CIFAR-10 pre-training and CIFAR-100 finetuning tasks and show that, in both tasks, our method's performance comes close to that of a non-private model and generally outperforms DP-SGD applied directly to the contrastive loss.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a variant of DPSGD tailored for DP-training with non-decomposable loss functions, such as the CLIP loss. The authors theoretically bound the l2 sensitivity of the gradient, demonstrate the privacy guarantee, and introduce their Logit-DP algorithm. They also discuss prior methods like per-batch and per-sample clipping and evaluate their approach on CIFAR-10 pretraining and CIFAR-100 finetuning tasks.

### Strengths
The problem addressed is both important and practically valuable, and the method provides an interesting solution for DP-training models with non-decomposable losses, such as CLIP-based VLMs.

### Weaknesses
See below.

### Questions
1. The main result, Theorem 4.2, seems to be a simple expansion with some triangle inequlities. Much of the theoretical complexity is deferred to parameters L, G1, and G2, which seem to be technical artifacts rather than providing meaningful generalization of the theory.

2. The algorithm involves per-sample clipping, computing the batch gradient using Theorem 4.2, and adding noise. This approach somehow feels somewhat redundant to me, as it seems to enforce clipping of the per-sample loss explicitly. Since per-batch clipping alone should suffice to ensure privacy (see [1]), it seems that per-sample clipping is not an essential requirement from a privacy perspective, but rather a technical necessity due to the proof requiring bounded per-sample gradients. While the authors discuss the empirical limitations of per-sample clipping (see 5 below), it would be helpful if the authors could further clarify the reasoning behind this choice, particularly any theoretical intuitions beyond this empirical justification.

3. The DP guarantee in Corollary 4.4 also seems loose. The noise magnitude of order $n \sqrt{\log(1/\delta)} / \epsilon$ could be reduced to $\sqrt{\log(1/\delta)} / \epsilon$, similar to [1], given that the loss is not divided by $1/n$, while the loss in [1] is divided by $1/n$.

4. The paper lacks discussion on the privacy-utility tradeoff. While noise can always be added to ensure privacy, it’s crucial to evaluate its impact on model performance.

5. In Sections 5.1 and 5.2, the values of $\epsilon$ is missing (maybe I missed it...?) Additionally, the authors suggest that “clipping the batch gradient (which they term Naive-DP) does not significantly reduce $L_X(w)$, regardless of batch size or clip norm.” They imply that Naive-DP leads to poor convergence: excessive noise for small batches and stability issues for large batches. However, is there a rationale beyond Figure 1? Also, was the learning rate kept constant across different batch sizes? Typically, larger batch sizes requires larger learning rates (see [2]).

6. Lastly, in Figure 2, Logit-DP performs better than the non-private model, which seems counterintuitive.


[1] Huang et al. Safeguarding Data in Multimodal AI: A Differentially Private Approach to CLIP Training, https://arxiv.org/pdf/2306.08173
[2] Ponomareva et al. How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy, https://arxiv.org/pdf/2303.00654

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This papers discusses the challenge of training computer vision and language models with differential privacy, especially when using unsupervised pre-training and similarity-based loss functions like contrastive loss. The primary issue with contrastive loss in a privacy-preserving setting is that its sensitivity increases with the batch size, which negatively impacts differentially private methods like DP-SGD (Differentially Private Stochastic Gradient Descent). To address this, the authors propose a modified DP-SGD method specifically designed for similarity-based losses, achieving a constant sensitivity level regardless of batch size. Their experiments on CIFAR-10 and CIFAR-100 datasets demonstrate that this approach nearly matches the performance of non-private models and surpasses the results of directly applying DP-SGD to contrastive loss.

### Strengths
The paper considers developing DP algorithms for problems where the objective function is coupled, which often happens in problems with contrastive loss. The setting appears to be new, and the algorithms are specifically design and analyzed for this class.

### Weaknesses
However, the derivation of the \ell_2 sensitivity is rather trivial, and the resulting algorithm is still just a simple variation of the traditional DP-SGD algorithm. Most of the derivations are mechanical. I do not see too much insights or contribution in this work. Further, the paper has not been well-written, as there are places that have obvious typo and inconsistencies. For example in line 156 Eq. (2.4) has been mentioned, but no (2.4) has been defined, nor the gradient of K_z(w) has been defined either.

### Questions
n/a

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work studies DP optimization of non-decomposable functions in neural networks, trying to add less noise with respect to the batch size. A new algorithm with proofs are proposed with CIFAR experiments.

### Strengths
The paper is reasonably easy to follow and rigorous as far as I am concerned. A significant portion of the paper is devoted to background introduction. The empirical results are clear. The algorithmic limitation (the computation bottleneck) is carefully discussed.

### Weaknesses
The algorithm is not straightforward: $\bar g$ comes from (5) which does not relate to $\bar g$ explicitly nor to $\bar g_{ij}$ in the line above. There should be a hidden for loop between line 4 and 5 in Algorithm 1 (for i,j in 1...n) to highlight the computational bottleneck.

The experiments are not very convincing, especially given the first sentence in abstract "Unsupervised pre-training is a common step in developing computer vision mod- els and large language models." Some experiments are fine-tuning not pre-training, and no language modeling is shown. I would encourage the authors to enhance the experiments with ImageNet (at least a subset) or Transformers.

Minors: 
1. Most new contents are only presented after page 4, which may come too late and is slightly distracting.

2. Title should not use abbreviations like DP and, since the method applies to other optimizers besides SGD, maybe the title can highlight this.

3. Line 3 of Algorithm 1, the for loop counting is incorrect.

### Questions
See "Weaknesses".

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
This paper proposes a variant of DP-SGD for loss functions that cannot be described as an average (or sum) of individual losses, such as contrastive loss and spread-out regularization. The authors apply a Gaussian Mechanism scheme, theoretically derive an upper bound for $L_2$-sensitivity and experimentally claim their findings on computer vision tasks.

### Strengths
**S1:** The method is consistent with prior studies on DP-SGD.

**S2:** The implementation of the proposed Logit-DP algorithm is the same as for DP-SGD.

**S3:** The authors also prove several lemmas regarding the conditions on $L_2$-sensitivity.

### Weaknesses
**W1:** The main theoretical findings of the paper focus only on "vanilla" contrastive loss and spread-out loss. However, there is an increasing demand for more sophisticated types of loss functions, such as SimCLR [1] and InfoNCE [2]. These functions align well with **Definition 2.2** and benefit from a large batch size. The paper lacks additional commentary on these and other contrastive objectives, for which analogous lemmas can be easily proved.

**W2:** The experimental validation is unclear:

- In Line 326 the authors note that they train model without BatchNorm layers because of they are not privacy-preserving, at the same time, no comparison is provided with a baseline model that includes BatchNorm.
- Modern NLP and CV models often use transformer architectures that incorporate LayerNorm. Could the authors clarify whether they consider LayerNorm more privacy-preserving or not?
-  It is hard to understand figures: in Figure 1 the authors report averaged relative loss along with its bounds/percentiles for Logit-DP, but no bounds/percentiles provided for Non-Private and Naive-DP. Also there are no mentioned standard deviation in all Tables, while it is obvious that the authors report an averaged results.
- There are several drawbacks regarding the hyperparameters sweep: the authors train a ResNet18 model on CIFAR10 for only 2 epochs and a generic convolutional model for 20 epochs, whereas training from scratch on CIFAR10 typically requires hundreds of epochs. At the same time, they fine-tuned both the generic convolutional model and ResNet18 for 10 epochs, which is much longer than regular fine-tuning.
- In Table 4, the authors report an accuracy score below 20% for the CIFAR100 fine-tuning task. Needs further clarifications.

**W3:** In Lines 380-383, the authors do not address the issue of private fine-tuning of the publicly available (i.e., non-privately trained) model,  which is almost more popular in the community than private training from scratch. Additionally, the authors do not clarify what is meant by a ''privately pre-trained'' embedding model: how is it pre-trained?

**W4:** The code is not publicly available.


**Minor Comments:**

**C1:** There is no need to write ''non-private'' twice in Line 405,  as the implication is clear that the Non-Private optimizer is SGD (Line 324).

**C2:** The section title in Line 824 is incorrect; it should be  ''Fine-tuning on CIFAR100''.

**C3:** In Line 834 you denote $L_2$-sensitivity as $\ell_2$-sensitivity, which is not self-consistent.





[1] Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton. ''A Simple Framework for Contrastive Learning of Visual Representations''. ICML 2020

[2] Aaron van den Oord, Yazhe Li, Oriol Vinyals. ''Representation Learning with Contrastive Predictive Coding''.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a new variant of Differentially Private Stochastic Gradient Descent (DP-SGD) designed for similarity-based loss functions, such as contrastive loss, which are common in unsupervised pre-training. The core claimed contribution is a modified DP-SGD method that achieves sensitivity of $O(1)$ for the summed gradient with respect to batch size. The paper also provides experimental validation of this new method on CIFAR-10 and CIFAR-100, showing performance close to non-private models and generally better than the naive DP-SGD approach.

### Strengths
- The problem is very well-motivated. Differentially private pre-training is an important and interesting direction for learning foundation models privately. Most existing work [PBV2022, YSM+2024] focus on training decomposable loss with DP-SGD. Investigating new efficient and effective techniques for training DP models with non-decomposable loss is very important.


*References*

[PBV2022] Training text-to-text transformers with privacy guarantees. Ponomareva, N., Bastings, J., and Vassilvitskii, S.  In Findings of the Association for Computational Linguistics: ACL 2022, pp. 2182–2193, 2022.

[YSM+2024] ViP: A Differentially Private Foundation Model for Computer Vision. Yaodong Yu, Maziar Sanjabi, Yi Ma, Kamalika Chaudhuri, Chuan Guo. Proceedings of the 41st International Conference on Machine Learning, PMLR 235:57639-57658, 2024.

### Weaknesses
The main question (and potential weakness) I have is: 
>Is the proposed method indeed independent of the batch size $n$? As proved in Theorem 4.2, the $L_2$ sensitivity is upper bounded by $(G_1 + G_2 + (n-1)L) B$, where $n$ is the batch size. Then as described in Lemma 4.5, for contrastive loss, $(G_1 + G_2 + (n-1)L) B$ is upper bounded by $B_{\text{contrastive}}=2(1+\frac{(n-2)e^2}{e^2+(n-1)})$. Therefore, when $n$ is large, isn't $B_{\text{contrastive}} = O(n)$? And this is not independent of the batch size $n$. 

I may have some misunderstandings here. I would like have the authors to clarify this during the discussion. If the independent argument is correct, I would raise my score.

(minor weakness) The proposed method is computationally more expensive than standard DP-SGD, I suggest the authors to provide some results on [training loss vs training time].

### Questions
A few missing references on unsupervised DP-SGD pre-training:

[PBV2022] Training text-to-text transformers with privacy guarantees. Ponomareva, N., Bastings, J., and Vassilvitskii, S.  In Findings of the Association for Computational Linguistics: ACL 2022, pp. 2182–2193, 2022.

- [YSM+2024] ViP: A Differentially Private Foundation Model for Computer Vision. Yaodong Yu, Maziar Sanjabi, Yi Ma, Kamalika Chaudhuri, Chuan Guo Proceedings of the 41st International Conference on Machine Learning, PMLR 235:57639-57658, 2024.

### Soundness
2

### Presentation
3

### Contribution
2
