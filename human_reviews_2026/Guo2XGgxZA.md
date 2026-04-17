# LoRA-S: An Efficient Low Rank Adaptation scheme via Sylvester equation

- Decision: Accept (Poster)
- Scores: 2, 4, 4, 8

## Abstract
Numerous studies on low-rank adaptation (LoRA) emerged in recent years, with the aim of accelerating the convergence of the LoRA framework. In this paper, we leverage the horizontal lift theory from differential geometry to establish the general iteration scheme on the quotient manifold  \mathbb{R}\_\*^{m \times r} \times \mathbb{R}\_\*^{n \times r}/\sim. 
By endowing the LoRA framework with Riemannian quotient geometries, our theory not only guarantees efficient feature learning but also bridges the LoRA algorithms and the pre-training algorithms for large models. 
Furthermore, we theoretically analyze the role of the weight decay matrix $\epsilon_{decay}I$ in efficient feature learning and then replace it with the Sylvester matrix $K$, indicating that the theory helps remove an important hyperparameter while generating accurate and computationally efficient optimizers. 
Based on the general scheme, we propose two efficient LoRA optimizers with runtime analysis, Adam-Sylvester (AdamS) and LRACS, then conduct experiments on the transformer-based networks. The results demonstrate evident improvements over existing optimizers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this work, the authors propose and analyze a Riemannian-based optimizer to fine-tune pretrained models using Low-rank adapters.

In particular, the framework is the one of [Absil et al., 2014], in which the manifold 
$$
\mathcal M_r =\{ W \in \mathbb R^{n \times m} \,| \, \mathrm{rank}(W) = r \}
$$
is seen as a quotient of $\mathbb R_*^{n \times r} \times \mathbb R_*^{n \times r}$ through the projection $\pi(A,B) = AB^\top$. The framework proposed is essentially a general Riemannian optimization algorithm over the fibers of the projection, with the metric on the product defined through the metric of a chosen metric on $\mathcal M_r$. In practice, then optimization is performed on liftings in the horizontal space. After this, the authors discuss the effect of $L^2$ regularization in their framework.

Finally, the framework proposed allows us to easily lift preconditioners from $\mathcal M_r$ to the quotient space, allowing us to easily precondition the dynamics and therefore to reproduce classical optimizers such as Adam.

### Strengths
The approach the authors propose is fairly general, and the theory is well-founded. The notation is pretty precise, and the authors often give good intuitions behind the formalism.
The proposed framework allows for easy extension for quasi-Newton methods, as the preconditioning in $\mathcal M_r$ can be lifted to horizontal spaces.

### Weaknesses
1. A lot of the theory described in the majority of the work (including the horizontal lifting) was already presented in [1]. It is not clear what the actual contributions of the work are, as even the "Efficient feature learning" proposed in [Hayou et al., 2024] seems in this case a simple consequence of performing optimization in the quotient space, and this is well known in the Riemannian optimization literature (see [1]). 
2. Section 4.1 is not clear at all, as it seems the discussion is all to point out again that preconditioning is "well-defined" again because of the quotient, therefore one can use Gauss-Newton-like methods similar to Adam, and the preconditioner to use won't depend on the fiber of $\pi$ we are working on. I guess also the title "ANY PRECONDITIONING WORKS WELL ON THE MATRIX TANGENT SPACE" is not clear, as it does not seem to reflect the content of the section.
3. The claims of the theory are correct, but they don't seem to be reflected in the experimental section. In particular, numerical improvement with respect to much simpler methods is minor; therefore, one can ask if all this machinery has a use case in practice. I believe the authors should focus the experimental section more on showing the advantages of the proposed framework with respect to vanilla SGD or Adam on LoRA instead of just benchmark performances.

[1] B. Mishra et al., "Fixed-rank matrix factorizations and Riemannian low-rank optimization", Computational Statistics, Volume 29, pages 591–621, (2014).

### Questions
I would appreciate it if the authors could discuss the points I raised in the "weaknesses" section, as I believe the line of research per-se is interesting and valuable, but in my opinion, the work needs to be better presented.

Minor comments and typos:
1. As a general rule, I suggest that the authors not number equations if they are not referenced in the main text, as this creates unnecessary confusion.
2. I would suggest including an initial paragraph summarizing the contributions of the work
3. All initial quotes are inverted, use `` ... " instead of ""... "

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The reviewed work is concerned with low rank adaptation. An important feature of low-rank based parametrizations is that many pairs of left and right factors can give rise to the same weight matrix. Likewise, many pairs of infinitesimal changes of these factors give rise to the same tangent vector in weight space.

The "horizontal lift theory" (Avsil et al., 2014) proposes to select the representation that is orthogonal to the kernel of the map from pairs of factors to weight matrices. The reviewed work suggests using this mechanism to select updates in low rank training algorithms. Computing these updates requires solving a Sylvester equation. The authors provide a high-level argument for why the Sylvester-based update already accounts for the need of weight-decay, thus allowing to remove the need for explicit regularization and in particular hyperparameter optimization. 

The numerical examples show small improvements over baselines.

### Strengths
The use of Riemannian geometric approaches for improving low rank learning is a natural approach.

### Weaknesses
I found this paper difficult to read. This may be partly due to technical nature of its contribution, but unnecessarily aggravated by the way in which the heavy notation is introduced (or not). For instance in line 122 the paper talks about "the metric $\bar{g}$", but this metric is only introduced in the next section.

The paper also claims benefits "Our metric induces update rules with richer structure and interpretability than the metric (13)" without (in the main text) substnatiating these claims. In contrast, the numerical improvements are very minor. I will confess that I was unable to follow the argument why the weight decay parameter can be removed.

### Questions
1. How is it possible for $\bar{g}$ to be a metric (as opposed to a pseudometric). After all, the map $(\dot{M}, \dot{N}) \mapsto \dot{X}$ has a kernel by dimensional considerations.

2. I have a hard time understanding why the proposed scheme is transformation invariant. After all, the definition of the horizontal complement relies on the "orthogonal complement" of the kernel of $D\pi(M, N)$.
What this orthogonal complement is depends on the choice of inner product on $T_(M, N)$. If the Euclidean inner product is used, then doesn't this again result in transformation dependence? If an inner product is prescribed, then doesn't this play the same role as describing a coordinate transformation? 

3. Could you help me appreciate the empirical contribution? As it stands, it seems that there is very little difference to the reference.

### Soundness
2

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
This paper presents LORA-S, a novel LoRA optimization framework that leverages horizontal lift theory and Sylvester equations, achieving efficient feature learning (EFL) and transformation invariance. The authors propose two optimizers, AdamS and LRACS, both free of weight decay hyperparameters and shown to outperform existing methods in transformer-based models.

### Strengths
**Theoretical Innovation and Novelty:**

This paper correctly establishes a general iterative scheme that generalizes conventional optimizers. The use of horizontal lift theory and quotient manifolds is both mathematically sound and novel in the context of LoRA. Besides, the removal of weight decay and the introduction of Sylvester-based updates reduce the need for hyperparameter tuning, offering a deeper understanding of how regularization interacts with low-rank parameterization.

**Clarity and Presentation:**

The paper demonstrates a clear and logical structure. The authors properly define equivalence relations, tangent spaces, and induced metrics, and the derivations from the quotient manifold formulation to the horizontal lift operator are clearly and coherently presented.

### Weaknesses
- Although replacing the weight decay parameter with a sylvester matrix is mathematically sound, can you provide a practical intuition behind why it promotes efficient feature learning?
- How do other hyperparameters affect the proposed framework? For instance, can a large learning rate lead to instability in the results, or can gradient noise from a small batch size undermine the underlying assumptions? 
- Does the computational overhead become a bottleneck for larger models, and does the scheme remain efficient in large-scale distributed training environments, where synchronization is a critical factor?
- Sections 3 and 4 can be simplified, and unnecessary parts moved to the appendix, leaving more analysis of results in the main text.

### Questions
Please see weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a general iteration scheme for LoRA that works on a quotient manifold of pairs $(M, N)$ modulo the equivalence which preserves products $MN^\top$ and updates them via "horizontal lifts" of a descent direction in the product space. The horizontal lift gives the unique update with no component along the equivalence directions. Computing this update requires a small Sylvester equation solve which can be done in $O(r^3)$ time where $r$ is the LoRA rank. This scheme wraps conventional optimizers to produce updates that by construction enforce desirable properties such as transformation invariance and efficient feature learning (EFL). The authors also observe that the Sylvester solution introduces a decay term and remark that this decay may obviate the need for explicit weight decay. The performance of this scheme applied to Adam and RACS are demonstrated across several experiments.

### Strengths
The framework introduced by this paper is very interesting and is a novel application of quotient spaces used to "convert" optimizers to a principled LoRA update. The generality of the approach and the connection to prior ideas such as transformation invariance and EFL is valuable. Discussions of practical concerns are thorough and informative.

### Weaknesses
The diversity and scale of empirical validations is lacking. 

When arguing about removing the weight decay parameter, it doesn't appear that the baseline weight-decay in Adam is tuned.

Some of the technical exposition is hard to parse, especially to readers not familiar with the mathematical notions. More high-level exposition and background would be helpful.

### Questions
What sufficient conditions ensure the Sylvester equation solution exists and is unique?

Are there any numerical stability issues concerning computation of the inverses or the Sylvester solution?

Are there metrics other than the Grassmannian quotient metric considered which can also be used?

Are there any ablations on using the Riemannian moment accumulation (Eq. 36) vs Euclidean moment accumulation (Eq. 35)?

### Soundness
4

### Presentation
3

### Contribution
3
