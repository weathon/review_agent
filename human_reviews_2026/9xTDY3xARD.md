# Interpretable Kernel Representation Learning at Scale: A Unified Framework Utilizing Nyström Approximation

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Kernel methods provide a theoretically grounded framework for non-linear and non-parametric learning, with strong analytic foundations and statistical guarantees. Yet, their scalability has long been limited by prohibitive time and memory costs. While progress has been made in scaling kernel regression, no framework exists for scalable kernel-based representation learning, restricting their use in the era of foundation models where representations are learned from massive unlabeled data. We introduce KREPES---a unified, scalable framework for kernel-based representation learning via Nyström approximation. KREPES accommodates a wide range of unsupervised and self-supervised losses, and experiments on large image and tabular datasets demonstrate its efficiency. Crucially, KREPES enables principled interpretability of the learned representations, an immediate benefit over deep models, which we substantiate through dedicated analysis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a scalable framework, called KREPES, for kernel-based representation learning.  This framework relies on the Nystrom approximation, such that a representation f(x) lies in the span of some anchor points functions, say phi(tilde{x}_i), where the points \tilde{x}_i are called landmarks. Unlike the classical Nyström method, KREPES does not minimizes the residual in the RKHS norm. Instead, it allows for alternative self-supervised loss functions, which are optimized using gradient-based methods. The paper further explores preconditioning strategies and scalable techniques for efficient landmark selection.

### Strengths
- The overall pipeline makes sense and the combination of kernel representation with self-supervised learning is novel and interesting. That being said, it may be useful to position the paper with respect to other work relying on kernel and self-supervised learning such as  Zheng et al. Self-supervised learning with rotation-invariant kernels. ICLR 2023.
 - The code is provided and seems efficient.

### Weaknesses
- The empirical validation is not convincing. Using self-supervised loss functions within the Nystrom approximation is new, but we are missing an experimental case where the formulation/method is actually useful. Scalability is demonstrated in Section 4, to some extent, but the reported numbers are actually very far from the state of the art. I am also confused by the baseline NN, which performs particularly badly (60% accuracy on CIFAR-10 is **very** low). Why is that? How is NN trained? on which data? with which loss function? It is of course not directly comparable, but in the context of self-supervised learning, we are used to see extremely high accuracies on CIFAR-10 (e.g., 99% accuracy for DINO when pretrained on ImageNet1k).

  - It may be useful to provide computational time that include kernel precomputation (and report breakdown: kernel computation vs optimization). I was a bit lost by what average time means in Table 1.

  - The claims about interpretability are to be taken with caution. The decomposition is indeed explicit, but it does not necessarily translate
    into human-understandable features. It would strengthen the paper to show case studies where one inspects landmark contributions and relates them to meaningful patterns beyond the experiment on CIFAR-10 which is rather toyish.

### Questions
- is there any hope to extend existing theoretical results about Nystrom approximation to the loss functions used in this paper?

  - by changing the loss in the Nystrom method, the resulting optimization problems become nonconvex. Other difficulties may arise such as loss function with trivial solutions (called collapsed solutions in the SSL literature). Is it a problem within the KREPES framework?

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
3

### Summary
This paper presents KREPES, a scalable framework for kernel-based self-supervised representation learning. The key idea is to apply Nyström approximation to reduce the computational cost of kernel methods so they can operate on large datasets. The framework supports multiple SSL objectives (e.g., SimCLR, BYOL, VICReg, Barlow Twins) and is optimized via gradient-based training on a reduced set of landmark points. The authors propose a PCA-based initialization and second-order preconditioning strategies to improve optimization efficiency. Experiments on some tabular and vision datasets are presented. Additionally, the model naturally supports interpretability via representer landmarks, enabling analysis of influential samples and concept-level explanations.

### Strengths
-The paper tackles an important and timely challenge: enabling scalable kernel-based representation learning in the self-supervised setting. This represents a meaningful contribution.

-The proposed framework is flexible and unifies a wide range of SSL objectives.

-The interpretability component is interesting. Leveraging representer points to analyze landmark influence and concept alignment provides clear explanatory benefits compared to standard deep SSL models.

-The empirical results suggest that the approach can be competitive across different data modalities, particularly in settings where interpretability is a priority.

### Weaknesses
-The literature review on Nyström methods is quite outdated, and many recent works are not mentioned.

-Lines 190–193 discuss Rudi et al. (2017), which itself builds on Less Is More (Rudi et al., 2015). However, this earlier work shows that 
m=O(\sqrt{n}) landmarks are required under standard assumptions (not O(logn) as the paper implies).

-In lines 239–240, the authors correctly point out that their setting differs from Rudi et al. (2017) because no closed-form solution exists for these SSL losses. However, the literature review remains incomplete, as other works addressing Nyström approximation for general losses—such as for example, "The Nyström method for convex loss functions" by Della Vecchia et al. (2024) or “Fast and Accurate Refined Nyström Based Kernel SVM” by Li et al. (2015) and others—are not cited.

-Again, in lines 318–320, while discussing landmark selection, only Rudi et al. (2017) is referenced. Recent work aiming to mitigate this bottleneck, such as "On fast leverage score sampling and optimal learning" by Rudi et al. (2018) with the BLESS algorithm, is missing.

-Some choices are quite obscure and not well-motivated, for example at lines 276-279 where the preconditioner used in FALKON is mentioned but a simpler one is indeed used because "empirically it improves optimization ... ". Does this hold in general? Is it limited to the analyzed datasets and losses?

-The experiments in Section 4 are somewhat limited, and the results are not fully convincing. For instance, it is stated that the neural network baselines are “generally faster”, but no discussion is given on the magnitude of this gap — e.g., on CoverType the NN appears to be up to 1000× faster, which is substantial. Similarly, on CIFAR-10 the proposed method outperforms the NN baselines by up to 40 accuracy points, which seems extreme and suggests a potential issue with the fairness or quality of the NN baselines.

### Questions
Some perplexities are expressed above. Additionally:

Table 1 shows differences of up to 40 accuracy points between the proposed method and the neural network baselines. How is such a large performance gap possible? Were the neural networks trained correctly with standard and competitive SSL protocols? Is the comparison fair?

-Prior work (e.g., Rudi et al., 2015; Della Vecchia et al., 2024, among others) provides both theoretical and empirical guidance on properly selecting the number of Nyström landmarks m, typically 
m=O(\sqrt{n}), to ensure near-optimal accuracy while reducing computation. In the present paper, however, no theoretical guarantees are provided for the chosen m; the experimental results suggest the opposite of the expected trade-off: the Nyström-based model seems to achieve higher accuracy (sometimes significantly so), while being 100× to 1000× slower than the suggested baseline; the procedure for selecting m is not discussed. A deeper clarification and justification of these aspects is needed to support the claims regarding scalability and fairness of the evaluation.

I will reconsider my rating once the authors have addressed the questions above.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces KREPES, a scalable kernel-based self-supervised learning algorithm that optimizes modern SSL losses (SimCLR, BYOL, VICReg, Barlow Twins) directly in kernel space using stochastic gradient descent. It achieves scalability via a Nyström approximation and stabilizes training with a preconditioner inspired by Rudi et al. (2017), extended to non-quadratic losses through Gauss–Newton updates. Using kernels like the empirical NTK of pretrained networks, KREPES matches neural network performance while remaining interpretable and computationally efficient.

### Strengths
1. Investigating self-supervised losses within kernel methods is a relatively underexplored area with strong potential.

2. The paper explores a practical approach to scaling kernel models to larger datasets rather than a theoretical advance in scalability.

3. Evaluating multiple modern SSL losses (SimCLR, BYOL, VICReg, Barlow Twins) within one unified kernel framework is important for robustness and broad applicability.

### Weaknesses
1. The authors emphasize scalability, but I remain unconvinced. The method builds on a preconditioner from Rudi et al. (2017), which still requires an $O(m^2)$ memory footprint. This scaling becomes impractical when the number of landmarks m grows large. For true scalability, I would expect to see experiments demonstrating how the method performs when m reaches millions of landmarks not just $n >> 10^6$.


2.  If the paper’s goal is to show that allowing diverse self-supervised losses (beyond MSE) leads to better performance, the reported results do not convincingly support that claim. The method uses an **empirical NTK (eNTK)** computed from a pretrained network, but it is unclear which model produced this eNTK. If it was derived from a strong architecture like **ResNet-50**, then even a simple MSE-based kernel regression on those features should yield strong results or even simpler get the last layer embeding of the trained model and use a laplace kernel. 


3. Table 1 raises questions — the CIFAR-10 neural network baseline reportedly achieves only about 60–70% accuracy, which seems unusually low for a ResNet-50-sized model (~25 M parameters), as such a network typically exceeds 90% accuracy. Clarifying which neural network was used for both the baseline and the eNTK would greatly help interpret these results.


4.  The main contribution of the paper is not fully clear. It would help if the authors included an explicit algorithm box summarizing what **KREPES** actually does and what is novel about it. As I understand it, KREPES applies stochastic optimization (e.g., Adam) to kernel models using the **preconditioner from Rudi et al.**, while extending it to non-quadratic self-supervised losses. However, that preconditioner was originally derived for the MSE loss in kernel ridge regression; applying it empirically to other, non-convex SSL losses seems somewhat ad-hoc. It would strengthen the paper if the authors could justify why this preconditioner is expected to help under those more general losses, or provide theoretical or empirical support for this choice.

### Questions
See weakness.

### Soundness
2

### Presentation
1

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
This paper introduces a scalable framework for kernel-based self-supervised representation learning through Nystrom approximation. The proposed framework achieves comparable learning performance as DNNs with much less training parameters, therefore should be more computationally efficient. The proposed framework also provides some interpretation through representer landmarks. The overall writing is fluent the generalization of kernel methods to a group of loss functions in SSL is appreciated.

### Strengths
1. The proposed scalable kernel-based method for representation learning is novel.
2. The improvement of the algorithm through second-order method with pre-conditioner and Hessian approximation is effective. 
3. The writing is relatively good.

### Weaknesses
1. The paper claims to find a scalable method in the data-hungry demands era, which however, is not reflected through their simulation. 
2. The theoretical advantage of kernel methods in representation learning is also not reflected. Without that, based on the current presentation, what we can see is kernel-based SSL achieves comparable performance as, not better than, neural networks. Combined with the first weakness, the work could not persuade us to choose kernel based method over DNNs.
3. The proposed method is claimed to scale efficiently to large dataset, which could be reflected through training time, is missing. Moreover, the test time of the proposed framework in most of the cases (Table 1) is much longer than NN. What would be the benefits choosing KREPES over NN, without explicit theoretical benefits and computation efficiency?

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2
