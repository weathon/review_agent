# Deep Weight Factorization: Sparse Learning Through the Lens of Artificial Symmetries

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 8, 5

## Abstract
Sparse regularization techniques are well-established in machine learning, yet their application in neural networks remains challenging due to the non-differentiability of penalties like the $L_1$ norm, which is incompatible with stochastic gradient descent. A promising alternative is shallow weight factorization, where weights are decomposed into two factors, allowing for smooth optimization of $L_1$-penalized neural networks by adding differentiable $L_2$ regularization to the factors. 
In this work, we introduce deep weight factorization, extending previous shallow approaches to more than two factors. We theoretically establish equivalence of our deep factorization with non-convex sparse regularization and analyze its impact on training dynamics and optimization. Due to the limitations posed by standard training practices, we propose a tailored initialization scheme and identify important learning rate requirements necessary for training factorized networks.
We demonstrate the effectiveness of our deep weight factorization through experiments on various architectures and datasets, consistently outperforming its shallow counterpart and widely used pruning methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Building upon the fact that regularizing neural networks with an L1 penalty is not adequate due to its non-smoothness and its limited compatibility with SGD, the authors investigate the application of the weight factorization framework to induce sparse weights upon training of deep learning models.

They extend the ideas of [Hoffn, 2017] and [Ziyin & Wang, 2023] to arbitrary factorization depths of neural network weights, and provide theoretically grounded equivalence of training deep factorized neural nets and sparse optimization problems (Theorem 1).

The authors then address the difficulties in optimizing their model with standard optimization practices and suggest more suited strategies. Furthermore, they analyze the training dynamics of these networks (namely the accuracy-sparsity interactions during training) and provide an empirical evaluation of the performances of their approach against other sparsification frameworks.

### Strengths
The presentation is very clear, with a nice balance of theoretical investigations coupled with empirical evaluations.
I enjoyed reading this paper.

I find Figure 2 to be especially clear and informative, offering a nice intuitive overview of the approach.

I would also like to highlight the thoroughness of the appendix which provides appreciated details and references to related literature.

Overall, factorizing the weights of a network to arbitrary depths to ensure the sparsity of the trained model upon collapsing the weights is an interesting idea that I find worth exploring.

### Weaknesses
I will provide here some suggestions on the presentation as well as technical questions.

* Figure 1 displays accuracy as function of compression ratio (CR), but we don't know what that refers to at that point (it is introduced later in section 2.1). I would suggest defining CR in the caption of the figure or at least point to its definition.

* I am not sure to understand what Figure 3 aims to show. Are the curves' colors referring to the value of $c$ in Definition 1 ?

* I do understand the inevitability of a sparsity-accuracy tradeoff, and acknowledge the link made by the authors regarding the use of large LRs and generalization performance. However, I still fail to wrap my head around why the LR has such a significant impact on the sparsity of the collapsed weights, given that the authors previously show the equivalence between DWF and sparse regularization. At convergence, shouldn't sparsity be present no matter what ? I would greatly appreciate if you could perhaps provide me some more intuition on this.

* I acknowledge the memory and computational advantages in using sparser neural networks, but I was wondering how relevant these advantages would be if one could instead just directly train a dense but smaller (with less parameters) network to begin with. If a large but sparse network can achieve competitive performance with its dense counterpart, wouldn't it be fair to assume that a small but dense network could also be competitive ? If so, what would be the point in sparsifying the large network ?

* This is more of a research question, but there seems to be intricate interactions between accuracy and sparsity in DWF models. Do you plan on investigating these from a theoretical standpoint ? Perhaps it could be interesting to study the drops in accuracy observed in Figure 1 after a certain compression ratio is reached ?

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper extends earlier results on weight factorization and proposes a differentiable (though non-convex) regularized training objective for neural networks whose global and local optima coincide with those of a problem with the same training loss and with $\frac{2}{D}$-regularization on the network weights. For $D \geq 2$, the naive regularizer is sparsity-inducing but non-differentiable, and solutions to the latter problem fail to achieve competitive results at high compression ratios. The proposed solution is to factorize the network weights into a product of $D$ matrices and apply $L_2$ regularization to each factor independently; the authors call their factorization “deep weight factorization” (DWF). The paper’s key result, Theorem 1, conclusively supports the correctness of their method in theory – though there is no guarantee that one reaches equally desirable local minima in practice while solving each problem.

The authors then note that standard neural network weight initializations yield poor results for DWF and propose an alternative initialization that fixes the variance of the weights while avoiding weights that are too close or far from 0. They also study the impact of learning rate schedules on the trained networks’ sparsity-accuracy tradeoffs. They finally present empirical results to show that DWF outperforms shallow weight factorizations (albeit marginally for datasets such as CIFAR10/100 and Tiny Imagenet) and post-training pruning methods.

### Strengths
- The paper develops a natural extension of existing shallow weight factorization methods to construct differentiable regularizers whose corresponding optima are equivalent to those obtained by non-convex and non-differentiable $L^{(2/D)}$ regularization. This is a valuable – and to my knowledge original – theoretical contribution to the literature.
- The authors present extensive empirical results to study the impact of choices such as initializations and learning rates on their method’s performance. Doing this legwork to resolve critical pitfalls in the implementation of their method enables other researchers to use their work in practice, and substantially increases their method’s potential for impact.
- The paper is generally well-written, and I was able to follow its exposition without too much trouble.

### Weaknesses
My primary critique of this paper is that DWF’s performance improvements over shallow weight factorization and Synflow (in terms of compression ratio achieved at acceptable cost to accuracy) seem marginal for CIFAR10/100 and Tiny Imagenet. DWF does improve over competing methods by large margins on toy problems such as MNIST and its variants – but the gap between its relative performance on MNIST and on somewhat more complex datasets such as CIFAR and Tiny Imagenet makes me wonder whether the claimed improvements would become even narrower once one leaves the realm of standard ML benchmarks and tackles real-world problems. In light of this and my first question below, the authors may consider tempering their claims regarding the empirical benefits of their method over shallow weight factorization and pruning.

### Questions
- On lines 475-6, the authors claim that “For VGG-19 on CIFAR100 at 10% tolerance, DWF (D = 4) achieves a compression ratio of 1014, surpassing the runner-up (SynFlow at 218) by a factor of almost 5.” Perhaps I’m misreading the bottom-left plot in Figure 9, but don’t $D=2$ and D=3 both outperform $D=4$ at the 10% tolerance threshold (i.e. ~60% accuracy)? Furthermore, $D=2$ may be the best-performing method in this regime (though it is very close to $D=3$) – and this corresponds to shallow weight factorization, which is not an original contribution of this paper.
- Why use LeNet architectures for the MNIST experiments? To my knowledge, these are essentially never used in practice. Is there a particular scientific reason for using a very simple architecture in these experiments?
- The plots in Figures 6 and 7 are quite cluttered. Would the authors consider depicting curves for fewer values of $\lambda$ in Figure 7? They could also be more selective with the information they present in Figure 6.

### Soundness
3

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
This paper is concerned with sparsity-inducing training of neural networks. The fundamental (pre-existing) technical tool is the transformation of a (non-differentiable) L1-penalized problem, into an “equivalent” (differentiable) L2-penalized problem via a Hadamard-product reparametrization of the weights. This paper introduces Deep Weight Factorization (DFW), an extension of the previous idea to reparametrizations including D \ge 3 Hadamard factors, and shows an equivalence with non-convex sparse regularization. The authors present an analysis of the interplay between DFW and common optimization aspects such as the weight initialization and the choice of step size. Experimentally, the paper includes several comparisons against existing sparsity methods on vision benchmarks, favorably to DWF.

### Strengths
S1. The paper is well structured, and the presentation of the ideas is easy to follow.

S2. While the idea of generalizing “weight factorization” to multiple factors is quite natural, the authors do a good job at motivating it, and providing theoretical and empirical insights on their proposed DWF technique.

S3. The authors transparently explore the empirical uncertainties stemming from the change in parametrization. In particular, I appreciated the section on weight initialization: it identifies challenges with traditional initializations, discusses a potential explanation (based on the kurtosis of the distributions) and designs/prescribes a new initialization scheme that is more suitable for use in conjunction with DWF.

S4. The problem of neural network sparsity is very relevant in the current large model context. The simple core principles of the DWF approach, along with the theoretical insights presented by the authors, make it an appealing technique for the deep learning community.

### Weaknesses
W1. The paper exclusively focuses on vision-related tasks, of relatively small scale. 
  * The submission does not provide results on ImageNet-scale vision tasks.
  * The submission does not include any experiments on language or image generation.

W2. Theorem 1 presents an equivalence between the DWF and “original” non-convex penalized problem, claiming that they “have the same global and local minima”. However, the current discussion does not explore whether the relative quality of a local DWF minimum is or not comparable to the quality of its related minimum for the original problem. In other words, let $\hat{\omega}_1 \odot \ldots \odot \hat{\omega}_D = \hat{w}$ be a local minimum of the DWF problem. How good of a local minimum is $\hat{w}$ for the original problem?

W3. I found the visual presentation of experimental results –in particular, Fig 6, and Fig 7 bottom– too crowded, making it difficult for the reader to extract the relevant information.

### Questions
The questions marked with * are most important.

*Q1. I think the contributions of the work are sufficient for publication, but I would like to know the authors’ position on W1.

*Q2. In Fig 7 (and related plots in App E), what is the step-size selection protocol? Is it tuned individually for each depth?

Q3. For Sec 5.1, why is the magnitude pruning without fine-tuning protocol justified? Especially at high compression levels.

*Q4. The paper mentions explicitly (L151) “in principle [...] the factorization can also be selectively applied to arbitrary subsets of the parameters w” in the context of unstructured sparsity (in implicit contrast with structured sparsity). Orthogonal to the structured vs unstructured discussion, what is the relationship between DWF and the type of layers involved? Do certain layers benefit or suffer disparately with different factorization depths?

Q5A. I was pleasantly surprised to read the conclusions of the runtime analysis in Sec 5.2 and App I.2. Given (1) the claimed low computational overhead induced by DWF and (2) the change in optimization dynamics caused by DWF, what do the authors think could be the effects of this factorization idea on the training of non-regularized objectives?

*Q5B. As I stated in W1, the paper focuses on relatively small-scale experiments. How do the authors think the experimental conclusions of their runtime analysis would transform at extremely high parameter counts (>1B)?

Q5C. Along the same line of high parameter counts as in Q2B, do we need to “persistently hold” a (multi-factor) reparametrization during training? During the forward phase, it makes no difference to use the collapsed or not-collapsed parametrization, the only difference is for the gradient application. Would it be possible to hold an “ephemeral reparametrization”, sampled on demand during the parameter update? The idea would be to locally simulate the reparameterized GD dynamics, without requiring the persistent footprint of multiple model copies.

Q6. I noticed the absence of optimization “momentum” (as in Nesterov/Polyak) in the experimental details (as presented in App G.2). I understand that this could be a consequence of the specific architectures and tasks considered in the submission. However, I would like the authors to expand on any potential consequences/challenges of using DWF with momentum-based optimizers.

*Q7. The paper begins almost exclusively concerned with the case of the Lasso regularizer (q=1). Eventually, the theoretical connection presented in Thm 1 is established with a fractional regularized (q=2/D). Is this departure from the Lasso case “necessary”, or merely an artifact of the analysis/proof? In other words, if the problem I truly care about is the Lasso, can I still take advantage of the DWF-induced dynamics?

Minor comments/suggestions (No response expected)
* In Fig 7 (and more generally App E), the case “Depth=1” would be a useful baseline to understand how much of the improvement is afforded by DWF.
* There are several instances in the paper where the authors mention that “Appendix X contains information on the relationship between Y and Z”, but do not highlight any core insight on the main paper. I believe the current appendices are quite valuable, and the submission could benefit from (very!) briefly summarizing said key discoveries in the main body.
* Given the many appendices present, the authors may consider including a table of contents for the appendix to aid navigation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper extends existing (shallow) weight factorization techniques to deep weight factorization(DWF), and demonstrates the equivalence of DWF  with the $L_\frac{2}{D}$ (where $D$ denotes the depth of the factorization) regularization, thus potentially achieving higher sparsity compared to its shallow counterpart. The authors also provided enhanced strategies for  initialization and  learning rate to ensure the performance of DWF. Additionally, they analyzed the impact of DWF on training dynamics and optimization by empirical experiments, which show that DWF outperforms the shallow weight factorization and several existing pruning approaches.

### Strengths
1. The author presented a generalization of the shallow weight factorization  to overcome the non-differentiability of $L_1$ regularization and proved its equivalence to the $L_\frac{2}{D}$ regularization.
2. The strategies like initialization and large learning rate ensure the application of DWF.
3. The experiments show that DWF outperforms the original weight factorization and several pruning methods.1. The motivation of the paper is kind of vague. Why extending the shallow weight factorization in [1] to the deep one could potentially result in performance gain?  One  suggestion is: building on the equivalence of shallow weight factorization with the $L_1$ regularization, and taking into account that $L_p (p<1)$ regularization would potentially yield higher sparsity thanks to it is closer to the $L_0$ regularization compared to $L_1$ regularization, it is kind of natural to conjecture that deep weight factorization is equivalent to the $L_\frac{2}{D}$ (where $D$ denotes the depth of the factorization) regularization.

### Weaknesses
1. The motivation of the paper is kind of vague. Why extending the shallow weight factorization in [1] to the deep one could potentially result in performance gain?  One  suggestion is: building on the equivalence of shallow weight factorization with the $L_1$ regularization, and taking into account that $L_p (p<1)$ regularization would potentially yield higher sparsity thanks to it is closer to the $L_0$ regularization compared to $L_1$ regularization, it is kind of natural to conjecture that deep weight factorization is equivalent to the $L_\frac{2}{D}$ (where $D$ denotes the depth of the factorization) regularization.

2. In Figure 1, does the different compression ratios of vanilla $L_1$ regularization relevant to different values of $\lambda$? To my knowledge, achieving the highest test accuracy at various compression ratios often requires selecting different $\lambda$ values. If the value of $\lambda$  is fixed across different compression ratios, it would be beneficial to search $\lambda$  at different compression ratios.

3. Considering DWF is a pruning-before-training method, it is not much fair to only conduct comparison with those pruning-before-training methods, such as SNIP and SynFlow. It is suggested to add comparison with some high-performance pruning-after-training methods, such as [2] and [3] (pruning with $L_1$ regularization).

[1] Liu Ziyin and Zihao Wang. spred: Solving l1 penalty with sgd. In International Conference on Machine Learning, pp. 43407–43422. PMLR, 2023.

[2]Renda, A., Frankle, J., and Carbin, M. Comparing rewinding and fine-tuning in neural network pruning. arXiv preprint arXiv:2003.02389, 2020. 3, 4, 6, 9. 

[3]Zhang, Q., Zhang, R., Sun, J., & Liu, Y. (2023). How Sparse Can We Prune A Deep Network: A Fundamental Limit Viewpoint. *arXiv preprint arXiv:2306.05857*.

### Questions
1. The equivalence of DWF to the $L_p$ regularization largely relies on the fact that balanced factorization (or zero factor misalignment) is optimal. Would you provide some intuitive explanation that regularization by the balanced DWF would imply (or encourage) the sparsity of weights?

2. Whether the training for DWF could achieves a global minima, given that the weight factorization approach is  non-convex?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces deep weight factorization, an extension of shallow weight factorization for neural network sparsification. The key idea is to decompose network weights into D factors (D≥2), allowing the use of standard $L_2$ regularization to achieve a similar optimization problem to that of the regularized $L_p$ loss, while avoiding smoothness problems. The authors study the performance of this algorithm as well as analyze the dynamics and representations learned during sparsification.

### Strengths
The paper has some strengths:

1) The topic of sparsification during training is of great interest with growing model sizes.
2) The paper is well-written and presented.
3) The equivalence between the different regularization schemes is clearly explained and shown.
4) Section 4 is particularly interesting to me, as I am not aware of other studies which attempt to analyze the sparsification dynamics, or attempt to interpret unstructured sparsification algorithms.
5) The ideas presented in the paper are very good, but unfortunately suffer from a lack of novelty as I discuss in the weaknesses section.

### Weaknesses
The main weakness of this paper is that, unfortunately, the method proposed has already been implemented before with minimal difference, both in its current form based on Hadamard products [1] for linear networks and in a simpler completely equivalent form [2] incorporated directly to weight decay for any $p$-norm, including $2/d$. 

Therefore the paper offers no real novelty besides the analysis of the onset of sparsity, and since that is not the main focus of the paper it does not justify acceptance. I would suggest the authors reframe the work as an exploratory study of the dynamics of sparsification in deep networks, and submit to another venue.

Minor weaknesses:

1) The datasets and tasks are not studied at large scale, which is the most interesting setting in which sparsification should be performed. In particular, focusing either on small models or too simple data results in untrustworthy results when scaling up. Since this is not the first time this method is proposed, a potential avenue would be studying these types of $L_p$ methods in very large models, or for fine tuning.
2) The baseline of random pruning is not a useful baseline, basic magnitude pruning during training should at least be explored.

**References:**

[1] *Representation Costs of Linear Neural Networks: Analysis and Design*, Zhen Dai, Mina Karzand, Nathan Srebro, NeurIPS 2021 (https://openreview.net/forum?id=3oQyjABdbC8)

[2] *Decoupled Weight Decay for Any p Norm*, N. Outmezguine, N. Levi (https://doi.org/10.48550/arXiv.2404.10824)

### Questions
As stated in the weaknesses section,  I would suggest the authors reframe the work as an exploratory study of the dynamics of sparsification in deep networks and learned representations, and submit to another venue, after conducting a more thorough literature survey.

### Soundness
3

### Presentation
2

### Contribution
1
