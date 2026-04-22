# The Sparse Matrix-Based Random Projection: Exploring Optimal Sparsity for Classification

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 4

## Abstract
In the paper, we study the sparse $\\{0,\pm1\\}$-matrix-based random projection, a technique extensively applied in diverse classification tasks for dimensionality reduction and as a foundational model for each layer in the popular deep ternary networks. For these  sparse matrices, determining the optimal sparsity level, namely the minimum number of nonzero entries $\pm1$ needed to achieve the optimal or near-optimal classification performance, remains an unresolved challenge.  To investigate the impact of matrix sparsity on classification, we  propose to analyze the mean absolute deviation (MAD) of projected data points, which quantifies their dispersion. Statistically, a higher degree of dispersion is expected to improve classification performance by capturing more intrinsic variations in the original data.  Given that the MAD value depends   not only on the sparsity level of random  matrices but also on the distribution of the original data,  we  evaluate two  representative  data distributions for generality: the Gaussian mixture distribution, widely used to model complex real-world data; and the two-point distribution, available for modeling discretized data. Our analysis reveals that sparse matrices with only \textit{one} or \textit{a few} nonzero entries per row can achieve MAD values comparable to, or even exceed, those of denser matrices,   provided  the  matrix size satisfies $m\geq\mathcal{O}(\sqrt{n})$, where $m$ and $n$ denote the projected and original dimensions, respectively. These extremely sparse matrix structures imply significant computational savings.  This finding  is further validated through classification experiments on diverse real-world datasets, including images, text, gene data, and binary-quantized data, demonstrating its broad applicability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies sparse $\{0,\pm1\}$ random projection matrices and investigates the optimal sparsity level for classification tasks. By analyzing the mean absolute deviation (MAD) of projected data under two representative distributions (two-point and Gaussian mixtures), the authors show that extremely sparse matrices---with only one or a few nonzero entries per row---can achieve comparable or even superior classification performance to denser matrices when $m \ge O(\sqrt{n})$. The results imply significant computational savings without loss of accuracy.

### Strengths
S1 (Quality): Theoretical derivations are detailed, with convergence bounds and conditions for optimal sparsity explicitly characterized. Empirical results cover diverse data modalities and scales.

S2 (Clarity): The motivation, assumptions, and implications are clearly stated, and the paper is generally well written.

S3 (Efficiency): The results imply potentially large computational savings for random-projection-based pipelines, and offer insights relevant to large models and quantized deep networks.

### Weaknesses
W1 (Limited Novelty): The conceptual contribution appears incremental. Extremely sparse random projections and ternary/binary structures have been extensively explored in the context of model compression and quantized neural networks, raising concerns about whether the paper provides sufficiently new theoretical insight beyond existing literature.

W2 (Strong Distributional Assumptions): The analysis relies on two idealized data models—Gaussian mixture and two-point distributions. It remains unclear how well the theoretical conclusions hold under more realistic, heavy-tailed, multimodal, or correlated distributions commonly observed in large-scale datasets.

W3 (Weak Link Between MAD and Accuracy): MAD is used as a surrogate for classification performance, but the paper does not rigorously justify why maximizing MAD directly correlates with improved accuracy. The connection is empirical and intuitive, lacking a formal link to decision boundaries, margin analysis, or downstream classifier behavior.

W4 (Limited Baselines): Experiments only compare against Gaussian random projection. Without comparisons to other state-of-the-art dimensionality reduction or projection techniques, it is difficult to evaluate the practical advantage and competitiveness of the proposed approach.

### Questions
Q1 (Novelty Clarification):
Prior work has extensively investigated ternary/binary projections and extreme sparsity in the context of compression and quantized neural networks. Could the authors more clearly distinguish what new theoretical insight this paper provides beyond existing analyses? 

Q2 (Distributional Robustness):
The theoretical analysis assumes Gaussian mixture and two-point distributions. How robust are the results if data deviates from these assumptions (e.g., heavy tails, multimodal density, correlated features)? Can the authors provide empirical evidence, theoretical arguments, or discussion on how far the conclusions generalize under real-world distributional shifts?

Q3 (Justification of MAD as a Surrogate):
MAD is used as a proxy for classification accuracy, but the paper currently offers only intuitive justification. Can the authors provide a more rigorous argument or reference connecting MAD to downstream accuracy—such as its relationship to class separability, classifier margins, or error bounds? Under which conditions might maximizing MAD not translate into improved accuracy?

Q4 (Baseline Coverage):
Experiments only compare against Gaussian random projection. To better evaluate practical competitiveness, could the authors include additional baselines?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new random projection method, where the projection matrix is constructed by $\{0, \pm 1\}$. The authors establishes a connection between matrix sparsity and classification performance via MAD analysis, providing a new perspective beyond traditional distance preservation. It seems that the analysis in this work relies heavily on specific data distributions. Therefore, I recommend weak accept.

### Strengths
The authors conduct numerical experiments using six datasets (images, text, genes, binary) to support the theoretical claim.

### Weaknesses
The limitation of the assumption for the original data: The analysis relies heavily on specific data distributions (e.g. Gaussian mixture in the manuscript)

### Questions
How to generalize your framework to more general original data distributions?

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
The manuscript suggests that sketching with sparse matrices tends to preserve classification accuracy in practice. Motivated by this observation, the authors study the maximum absolute deviation (MAD) of the projected data and argue that this framework explains the effectiveness of sparse sketching compared to dense matrices, such as Gaussian ones.

### Strengths
The paper reads well.

### Weaknesses
I think certain parts of the paper should be explained better, and some statements appear to be incorrect:

- In lines 58–61, the authors suggest that sketching with sparse matrices preserves accuracy well and often performs comparably to or better than dense sketching. However, no citation is provided. A citation should be provided, as this is the motivation of the paper.  
- I could not follow why MAD is a good metric for understanding classification performance. The authors suggest that MAD is a more robust metric for quantifying dispersion; however, neither the text nor the examples involve heavy-tailed models or outliers. As written, the claim that MAD is a better alternative is not convincing.  
- There are some incorrect statements in the paper that should be addressed:  
  - In line 175, it is argued that when the original data $h$ follow the Gaussian mixture distribution described above, the projected data $z$ remain Gaussian. This appears to be incorrect.  
  - In line 179, it is argued that this relationship also holds approximately for original data $h$ drawn from other distributions, since by the Central Limit Theorem, the projected data $z \in \mathbb{R}^m$ can be approximated by a Gaussian distribution. This claim is too broad, since it depends heavily on the dimension and on the tail properties of the data in high-dimensional settings.  
  - In line 364, the authors suggest that MNIST follows a two-point distribution. On what basis? Please clarify.

### Questions
See above

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the optimal sparsity level for $\{0, \pm 1}$-matrix random projections in classification tasks by analyzing the dispersion of projected data points under two representative data distributions. The main finding is that extremely sparse matrices with only one or a few nonzero entries per row can match or exceed the classification performance of much denser alternatives, offering significant computational savings. This result is validated experimentally across diverse real-world datasets including images, text, gene expression data, and binary data.

### Strengths
- The paper tackles why extremely sparse random projections can still perform well in classification despite violating traditional distance-preservation guarantees, which is an interesting problem to tackle.
- The proposal to use Mean Absolute Deviation (MAD) as a proxy for classification performance seems to be a novel departure from standard $l_2$ or variance-based analyses. It potentially provides a new lens through which to analyze the problem.
- The theoretical findings are tested against a wide and diverse range of datasets, including images (YaleB, CIFAR100, ImageNet1000), text , and gene data.

### Weaknesses
1. **Connection between Disperson and Classification:** A core concern I have about the proposed view is the connection between dispersion and classification. The authors mention in the abstract that, *"Statistically, a higher degree of dispersion is expected to improve classification performance by capturing more intrinsic variations in the original data."*  However, I could not find anywhere in the paper where the author theoretically examines this. Since, this is a theory paper proposing this central hypothesis of MAD being a proxy for classification, I think this should be theoretically explored. 
    - I am also not convinced that PCA is a good example when supporting this view, as the goal of PCA is reconstruction and not class separability. 
   - If dispersion is the metric, it means everything is dispersed by the projection. I think inter/between-class variance relative to intra/within-class variance would be the proper metric for classification? How does MAD capture these nuances and how do the authors study this point?

2. **Claim about $z$ being Gaussian:**  The paper invokes the closure of Gaussian distributions under linear transformations to claim that the projected variable $z = Rh$ “remains Gaussian.” However, in section 2.2.2 $h$ is modeled as a Gaussian mixture (not Gaussian). Hence, $z$ is, in general, a *mixture* of Gaussians rather than a single Gaussian. This misapplication of the closure property undermines the validity of later derivations that rely on i.i.d. Gaussian assumptions (e.g., the MAD identity). Can the authors clarify?

3. In terms of proof presentation, the central point of the proof could be summarized much more neatly in the main paper and more of the technical details could be stowed away in the appendix.

### Questions
Please, see the weakness section.

### Soundness
2

### Presentation
2

### Contribution
3
