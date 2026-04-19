# A New Perspective on Shampoo's Preconditioner

- Decision: Accept (Poster)
- Scores: 3, 8, 8, 6

## Abstract
Shampoo, a second-order optimization algorithm that uses a Kronecker product preconditioner, has recently received increasing attention from the machine learning community. Despite the increasing popularity of Shampoo, the theoretical foundations of its effectiveness are not well understood. The preconditioner used by Shampoo can be viewed as either an approximation of the Gauss--Newton component of the Hessian or the covariance matrix of the gradients maintained by Adagrad. Our key contribution is providing an explicit and novel connection between the optimal Kronecker product approximation of these matrices and the approximation
made by Shampoo. Our connection highlights a subtle but common misconception about Shampoo’s approximation. In particular, the square of the approximation used by the Shampoo optimizer is equivalent to a single step of the power
iteration algorithm for computing the aforementioned optimal Kronecker product approximation. Across a variety of datasets and architectures we empirically
demonstrate that this is close to the optimal Kronecker product approximation. We also study the impact of batch gradients and empirical Fisher on the quality of Hessian approximation. Our findings not only advance the theoretical understanding of Shampoo but also illuminate potential pathways for enhancing its practical performance.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
1

### Summary
This work provides a theoretical foundation for understanding Shampoo from both Adagrad and Hessian perspective. It also establishes a novel connection between Shampoo’s approximation and the optimal Kronecker product approximation, clarifying misconceptions and highlighting potential improvements.

### Strengths
This work analyzes and understands Shampoo from the perspective of Adagrad and Hessian.\
A well-structured work.

### Weaknesses
-

### Questions
-

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Training modern deep neural networks is challenging due to their size and the inherent non-convexity of the training objective. 
The most popular optimizers for training these models are first-order methods such as SGD, Adam, and AdamW. 
In classical convex optimization, it is well-established empirically and theoretically that methods that incorporate second-order curvature information outperform first-order methods. 
However, directly extending these methods to the Deep Learning setting leads to prohibitive computational and memory costs. 
Fortunately, more computationally friendly approximate second-order optimizers have emerged in recent years, and they have shown promising performance relative to methods like Adam and AdamW. 
Foremost among these methods is Shampoo, which recently won the Algoperf benchmark for training neural networks commonly deployed in practice.

Despite its excellent empirical performance, the reasons underlying Shampoo's success have remained a mystery. 
This paper helps explain the method's success by providing a better theoretical understanding of what Shampoo's preconditioner approximates.
The paper provides experiments that corroborate their theory and show it can explain the observed practical performance of variants of Shampoo.
Finally, the paper provides empirical studies investigating how common tricks employed in practice for computational efficiency impact the quality of the Hessian approximation.

### Strengths
**Clarity:**

Overall, I found the paper easy and enjoyable to read. 
The paper is very well-organized, and the authors make all their points clearly.
For instance, usually, each theoretical result comes with a plot demonstrating the result empirically. This was very nice, made the presentation more concrete, and helped to immediately validate the theory. 

**Contribution:**

The theoretical insights into the Shampoo preconditioner are valuable and timely, as Shampoo is arguably the most practically effective approximate second-order method currently available.
This paper's results can possibly lead to improved convergence analyses of Shampoo-like optimizers and inspire the design of new optimizers in this area.
This paper will be of interest to anyone in the ICLR community who is interested in more efficient model training and deep learning optimization.

The experiments in Section 4 give useful practical insights into how computational tricks commonly employed in practice impact the approximation quality of $H_{GN}$. 
I think these results can help practitioners to develop better heuristics that balance the tradeoff in the preconditioner delivering better approximation quality vs. computation time.

**Correctness:**

I read through all the proofs of the various results in the paper. They are clear and correct.

**Overall:**

I enjoyed the paper and think it makes a nice contribution to the deep learning optimization literature, so I recommend acceptance.

### Weaknesses
**Introduction**

The first paragraph's discussion of the cost of 2nd-order methods should be conditioned a bit; otherwise, it is a bit misleading. 
A quadratic space requirement and a cubic computational complexity only arise if you naively try to apply classical techniques like Newton's method to Deep Learning.
By leveraging automatic differentiation, we can compute hvps (which is usually all we need) without forming the Hessian at a cost of $\mathcal O(np)$, where $n$ is the number of samples and $p$ is the number of parameters.
The same holds for the Fisher and Gauss-Newton matrices.
Of course, if no minibatching is done, this too is prohibitively expensive as n and p are both very large. 
However, in practice, good performance is often obtained using subsampling. 
Many methods in the literature exploit this trick and are called Hessian-free. 
The authors are aware of such methods, as they discuss them in the appendix.
Given this, in the final version, I would like the statements of the first paragraph to be qualified.

**Experiments**

One area where the paper could improve is if, in addition to plotting the approximation quality against the iteration count, it also gave plots showing how well the optimizer with that approximation performed on the actual objective loss. 
Preferably, these would appear side-by-side. 
This would give better insight into the impact of the preconditioner's approximation quality on the optimizer's convergence speed. 
For instance, in the setting of Fig 3. it would be interesting to see if $Shampoo^2$-Real Labels exhibited much worse performance than $Shampoo^2$-Sampled Labels, as $Shampoo^2$-Real Labels provides a much worse approximation to $H_{GN}$ in the minibatch setting.   

**Related Work**

The authors have been thorough in their list of related work in the appendix, but I feel they should add the citation:
    
Yao, Z., Gholami, A., Shen, S., Mustafa, M., Keutzer, K. and Mahoney, M., 2021, May. AdaHessian: An adaptive second order optimizer for machine learning. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 12, pp. 10665-10673).

AdaHessian is another popular stochastic second-order optimizer based on a stochastic approximation to the Hessian diagonal combined with Adam-style momentum and weighted averaging.

### Questions
1. In the revision, could you also include plots showing training loss, as per my comment in Weaknesses?
2. Do the authors have any thoughts on why $Shampoo^2$-Real Labels provides a much better approximation quality in the minibatch setting on CIFAR-10 than on ImageNet and Binary MNIST? This is not an essential question; I'm merely curious.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors analyze the similarity between the optimal Kronecker product approximation and the approximation made by Shampoo and show that the square of the Shampoo preconditioner is equivalent to a single step of the power iteration algorithm for computing the optimal Kronecker product approximation. The authors show that the cosine similarity with H_{Ada} (and Gauss-Newton) of preconditioners over training iterations is very similar for Shampoo^2 and the optimal Kronecker product approximation on MNIST-2, CIFAR-5M and ImageNet. However, this similarity can decay with real labels using a large batch size.

### Strengths
Novel insight interpreting a popular optimization algorithm, with extensions to batches and a real data regime. Insights have already been used to create new optimization algorithms which perform well. A good number of numerical experiments clearly verify the claims made about the similarity with the optimal Kronecker product approximation and the reason for choosing L and R as multiples of the identity. The work is succinct with a clear plotting style.

### Weaknesses
Experimental details are contained in the Appendix, and we see that cosine similarity changes over the training steps, an investigation into whether this is optimization trajectory independent (or at least using the Shampoo or Shampoo^2 algorithm) could be performed.

### Questions
Why was the decision made to compare matrices using cosine similarity? For example, what do the plots look like when the Frobenius norm \| A-B\|_F is used instead? In Figure 3, do you know how Shampoo would perform in the same setting?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a theoretical examination of the Shampoo optimizer, a second-order optimization method widely used in machine learning for its efficiency in handling large-scale problems. Shampoo’s preconditioner, based on a Kronecker product, is analyzed in this work to address a gap in understanding its effectiveness. The authors establish that Shampoo's preconditioner approximates the Gauss-Newton matrix or gradient covariance matrix via a novel connection to the power iteration method. Specifically, they show that Shampoo’s approach can be viewed as a single iteration of power iteration aimed at producing an optimal Kronecker product approximation, which they demonstrate empirically to be close to the ideal. Their study further explores the role of batch gradients and the empirical Fisher matrix in improving Hessian approximation, suggesting potential enhancements to Shampoo’s performance in various applications.

### Strengths
The authors theoretically and empirically demonstrate that the square of Shampoo’s approximation of $H$ is exactly equivalent to one round of the power iteration method, aimed at obtaining the optimal Kronecker-factored approximation of the matrix $H$. This finding offers a clear theoretical basis for Shampoo’s preconditioner, revealing that its approximation is more rigorous and methodical than previously understood, with direct ties to the power iteration process. This insight deepens our understanding of Shampoo's structure and effectiveness, and it sets the groundwork for further enhancements based on this relationship.

### Weaknesses
Despite the fact that I find the results really interesting, I still have my concerns about the relevance of this article to the conference. Below I will try to explain what I mean (if I am wrong, I am open to discussion):

- This paper can hardly be called theoretical, since the authors cite previous work for all lemmas in the paper;

- The whole paper is built on one result (see Proposition 1), which seems to be a mynor result.  

I also noticed some typos:

- Line 104: Kronecker product ($\otimes$) is used before definition;
- Line 176 and following: $I_n$ is nowhere defined;
- Line 235: Frobenius norm ($|| \cdot ||_F$) undefined;
- Line 374: PSD is not clear what it means.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
2
