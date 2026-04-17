# Training-Free Determination of Network Width via Neural Tangent Kernel

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Determining an appropriate size for an artificial neural network under computational constraints is a fundamental challenge. This paper introduces a practical metric, derived from Neural Tangent Kernel (NTK), for estimating the minimum necessary network width with respect to test loss *prior to training*. We provide both theoretical and empirical evidence that the smallest eigenvalue of the NTK strongly influences test loss in wide but finite-width neural networks. Based on this observation, we define an NTK-based metric computed at initialization to identify what we call *cardinal width*, i.e., the width of a network at which generalization performance saturates. Our experiments across multiple datasets and architectures demonstrate the effectiveness of this metric in estimating the *cardinal width*.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper shows that the test error of infinite and finite-width networks is upper bound by a value related to the smallest eigenvalue of the (empirical) NTK. Based on the theory, the authors provide a training-free method to determine the cardinal width, which is the critical width where the test loss saturates, through examining the saturation of the empirical NTK smallest eigenvalue at initialization. Experiments show that the proposed method can predict cardinal width pretty well.

### Strengths
1. There are many empirical works trying to utilize the empirical NTK spectrum to study the neural network properties at initialization. The theory derived in the paper offers a principled way of choosing the proper width without model training, which could be useful for practitioners in the field.
2. The paper is well-organized. The theory is accompanied with proper experiments.

### Weaknesses
1. The main concern of the paper is that whether the cardinal width is useful in practice. In several plots of figure 2, in CNN/MNIST from width 50 to 125 I do not observe obvious slope change. Similar observations can be seen in CNN/CIFAR10 width 50-100 and  ResNet/MNIST width 25-75. Hence, it makes the “plateau point” ambiguous. In such cases, a coarse manual random guess could possibly achieve similar conclusions. In conclusion, I am unsure about the superiority of the proposed method compared to heuristic rule.
2. Based on the figure 1, the observed plot is closer to $1/\sqrt{\mu_{min}}$, raising the concern of whether the bound is loose in real settings. Could the authors provide a detailed comparison/discussion with existing theoretical results regarding the NTK-based test error bound?

### Questions
In scaling law studies (Kaplan et al. 2020), the authors show that width and depth have minimal effects within a wide range compared to overall parameter count. Can the authors provide more insights on the superiority of the proposed methods in practical uses in more details, for example, how can the proposed method be used in real world experiment settings, like scaling law experiments.

### Soundness
2

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
The paper studies generalization loss of neural nets in the kernel regime and derives a generalization bound that depends on the minimum eigenvalue of the kernel at initialization ($\mu_{min}$). In particular their results indicate an upperbound proportional to $1/\mu_{min}^2$ for the test loss. Moreover, the result holds for both infinite and finite width neural networks. This leads to an algorithm for a training-free selection for the network width where the point where increasing the width does not significantly help with increasing $\mu_{min}$ is selected as the best choice for width.

### Strengths
Rigorous theory and several experiments to verify the effectiveness of the proposed algorithm and theory. Bounding the test loss based on a single computable parameter can be insightful even in the restricted setup of NTK regime and gradient flow considered in this paper. The authors also present a simple algorithm for width selection which is computationally efficient and does not require training.

### Weaknesses
-The paper mentions the importance of kernel condition number for both optimization and generalization as specified in literature. Can the authors please explain more the relation to prior works in line 90 of the paper? Moreover, the authors claim that they are the first to show the relation between $\mu_{min}$ and test error in the kernel regime. In my opinion, it needs more clarification by authors as there is already a link between these two in the literature. In particular, it is known that the neural network objectives of sufficiently large width with square-loss satisfy a PL condition in the kernel regime (e.g, Thm 4. in [1] where the authors show the relation between PL parameter and the minimum eigenvalue of kernel) and therefore one can obtain test loss bounds based on $\mu_{min}$ (e.g., see [2]). The minimum eigenvalue also seems to be proportional to class margin for classification tasks as discussed by [3] (section 5). 

-How does $\mu_{min}$ scale with number of samples ($n$)? Also, It's better to clarify whether $C_1,C_2$ in theorem 3.2. scale with $n$. can the authors please clarify how the bound scales with $n$, or specify relevant work in literature?

-What is the benefit of bounding test loss only based on $\mu_{min}$ beyond computational efficiency reasons? can including more information beyond the minimum eigenvalue (while not using the whole spectrum) help with generalization bounds? I seems the bounds based only on $\mu_{min}$ can be loose (or at least the authors do not characterize tightness) and moreover the authors observe a tighter $1/\sqrt{\mu_{min}}$ relation in their experiments. 

-Do the authors suspect that the theory can be extended to the feature-learning regime using DMFT framework of [Bordelon and Pehlevan, 2023] and the resulting equations to specify whether there is relation between generalization gap and $\mu_{min}$ of dynamical (time-varying) kernel? Has such a relation been observed in experiments? 

-Can the authors verify their theory on stylized data (e.g. boolean XOR data or multi-index models (see references[3-4])) to see how their theory predicts compared to current theoretical lower bounds for the network width in the kernel regime? 

-Does Thm 3.8 hold for any $m$? or Should the network be sufficiently wide for this results to hold? It seems to me that this detail is not emphasized in the statement of the theorem. 

1- Loss landscapes and optimization in over-parameterized non-linear systems and neural networks, Liu et al, 2021

2- Sharper Generalization Bounds for Learning with Gradient-dominated Objective Functions, Lei and Ying, ICLR 2021. 

3- Polylogarithmic width suffices for gradient descent to achieve arbitrarily small test error with shallow relu networks. Ji and Telgarsky, ICLR 2020.

4- Feature selection and low test error in shallow low-rotation relu networks, Telgarsky, ICLR 2023.

### Questions
Please see above section for questions.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a training-free criterion for selecting an appropriate network width. The key observation is that, when the width is sufficiently large, test error is governed by the smallest eigenvalue of the NTK and authors justify this both theoretically and empirically. Based on this, authors introduce a method that estimates the cardinal width, the point at which generalization performance saturates, by tracking the smallest NTK eigenvalue at initialization through Locally Optimal Block Preconditioned Conjugate Gradient . Experiments across multiple architectures and datasets validate effectiveness of this method.

### Strengths
The paper first makes an non trivial claim, test error is controlled by the smallest NTK eigenvalue, and supports it both theoretically and empirically. Leveraging this insight and pointing out the highlighting of using the smallest NTK eigenvalue rather than the full spectrum, the authors devise a practical procedure to estimate the cardinal width and experimentally validate the approach on two-layer networks. This line of research is compelling and provides well-balanced results.

### Weaknesses
Assumption 3.6, invariance of NTK, is central to the NTK framework, yet the authors assume it without proof. Although the empirical evidence is supportive, a full theoretical justification would substantially strengthen the work.

Although the effectiveness of the cardinal width is demonstrated experimentally, I’m concerned about how closely the experimental setup reflects real-world training conditions.

### Questions
Considering the eigenvalues of the NTK is a nice approach, but eigenvalues are sensitive quantities in numerical analysis. Did the authors report any uncertainty in calculating this eigenvalue? If not, do you have insights on how to compute it robustly in this setup?

In the EXPERIMENTAL VERIFICATION subsection, the weights seem to be initialized as ( \mathcal{N}(0,1) ), which appears different from NTK initialization. Could the authors explain this point?

### Soundness
3

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
4

### Summary
This paper introduces a principled, training-free method to determine an appropriate network width, termed "cardinal width," where generalization performance is expected to saturate. The core contribution is a novel metric based on the smallest eigenvalue ($\mu_{min}$) of the Neural Tangent Kernel (NTK) computed at initialization. The authors provide a theoretical justification, linking an upper bound on the test error to $\mu_{min}^{-2}$ in both infinite-width and, under certain assumptions, finite-width regimes. The proposed algorithm identifies the cardinal width by finding the point at which the growth of $\mu_{min}$ saturates as width increases, offering a practical tool to avoid unnecessary computational costs from over-provisioning network width without requiring any training.

### Strengths
- **Theoretical Rigor:** The paper's primary strength lies in its strong theoretical foundation in NTK theory. Unlike many zero-cost (ZC) proxies that are based on heuristics derived from pruning or empirical observations, this method provides a clear, analytical connection between a spectral property of the network at initialization ($\mu_{min}$) and its generalization capability. This adds a significant degree of interpretability and trustworthiness.

- **Actionable and Concrete Goal:** Most ZC proxies provide a relative ranking of architectures, which is useful for search but does not answer the fundamental question: "How wide is wide enough?". This paper's proposal of a "cardinal width" offers a specific, actionable target for practitioners, which is a distinct and highly valuable contribution.

- **Novel Theoretical Connection:** To our knowledge, this is the first work to formally connect the smallest eigenvalue of the empirical NTK in finite-width networks directly to an upper bound on generalization error. This provides a new and potentially powerful analytical tool for the community.

### Weaknesses
- **Fragile Theoretical Assumption for Practical Regimes:** The key theoretical result for finite-width networks (Theorem 3.8) explicitly depends on the "lazy training" assumption (Assumption 3.6), a regime where the network behaves like a linear model and does not perform significant feature learning. The authors commendably acknowledge their experiments operate outside this regime. However, this creates a critical disconnect between the theory that provides the method's justification and its demonstrated practical application. The method's success in the feature learning regime is thus an empirical observation, not a theoretically guaranteed outcome.

- **Significant Scalability Concerns:** The methodology requires constructing an $N \times N$ NTK matrix for a dataset of size $N$, a process with at least $O(N^2)$ complexity. This makes the approach computationally infeasible for large-scale datasets (e.g., ImageNet), where many modern ZC proxies that operate on a single minibatch (e.g., Synflow, GraSP) are orders of magnitude faster and more scalable. While subsampling is suggested as a remedy, its impact on the accuracy of the estimated cardinal width is not fully explored.

- **Narrow Scope of Application:** The paper focuses exclusively on selecting a single hyperparameter (width) for regression tasks. The broader and more common application of ZC proxies is in Neural Architecture Search (NAS) across complex search spaces, often for classification tasks. The paper does not demonstrate the method's utility as a general-purpose NAS proxy, and its theoretical underpinnings for classification losses are not provided.

- **Static Nature:** The proposed method is static, determining the width before training commences. This overlooks recent dynamic approaches that learn or adapt network width during training (e.g., Adaptive Width Networks) or allow for efficient width reduction after training (e.g., Triangular Dropout), which may find more data-efficient architectures.

### Questions
- Could you please elaborate on the observed alignment between the saturation of $\mu_{min}$ and test loss in the feature learning regime, despite Theorem 3.8 relying on the lazy training assumption? Is there an intuition or a potential theoretical argument that could bridge this gap?

- Regarding scalability: Could you provide a more quantitative analysis of the trade-off between the data subsampling rate, the computational cost, and the stability/accuracy of the predicted cardinal width? How sensitive is the method to the random choice of the subset?

- How do you envision this method being extended to classification tasks? What are the primary theoretical hurdles (e.g., defining the NTK for cross-entropy, the validity of the error bounds) and have you conducted any preliminary experiments in this direction?

- While the method provides an absolute value ("cardinal width"), how does it perform as a relative ranking metric on a standard NAS benchmark? For instance, does ranking architectures by their proximity to the predicted cardinal width yield a strong correlation with final accuracy compared to established proxies like Synflow or GraSP?

### Soundness
4

### Presentation
3

### Contribution
3
