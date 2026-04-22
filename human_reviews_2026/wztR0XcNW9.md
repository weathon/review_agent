# TopoCore: Unifying Topology Manifolds and Persistent Homology for Data Pruning

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Geometric coreset selection methods, while practical for leveraging pretrained models, are fundamentally unstable. Their reliance on extrinsic geometric metrics makes them highly sensitive to variations in feature embeddings, leading to poor performance when transferring across different network architectures or when dealing with noisy features. We introduce TopoCore, a novel framework that resolves this challenge by leveraging the principles of topology to capture the intrinsic, stable structure of data. TopoCore operates in two stages, (1) utilizing a _topology-aware manifold approximation_ to establish a global low-dimensional embedding of the dataset. Subsequently, (2) it employs _differentiable persistent homology_ to perform a local topological optimization on the manifold embeddings, scoring samples based on their structural complexity. We show that at high pruning rates (e.g., 90\%), our _dual-scale topological approach_ yields a coreset selection method that boosts accuracy with up to 4$\times$ better precision than existing methods. Furthermore, through the inherent stability properties of topology, TopoCore is (a) exceptionally robust to noise perturbations of the feature embeddings and (b) demonstrates superior architecture transferability, improving both accuracy and stability across diverse network architectures. This study demonstrates a promising avenue towards stable and principled topology-based frameworks for robust data-efficient learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the coreset selection problem: choosing a small subset of training data that maintains nearly the same model performance as the full dataset. It introduces a training-free approach that operates on frozen embeddings, viewing the dataset as a point cloud and using both global manifold density and local topological persistence to identify samples essential to the intrinsic structure. On benchmark image datasets, the method consistently achieves higher retention and lower variance than geometric baselines, especially under high pruning, showing stability and robustness across architectures, though its experiments are limited to vision tasks and lack compute analysis.

### Strengths
1. **Originality and clarity.** A training free coreset that combines manifold density with local topological persistence is a fresh, well motivated idea and the method is clearly described for replication.

2. **Strong empirical results.** Consistently high retention and lower variance than geometric baselines, especially at high pruning, plus sensible ablations on mixing weights and optimization depth.

3. **Practical robustness.** Works across multiple backbones with good transfer and noise robustness, indicating the selection signal is less model dependent than distance based methods.

### Weaknesses
1. **Limited scope.** Evaluation is restricted to vision benchmarks; no NLP or other modalities are tested, which weakens generality claims.

2. **Dependence on embeddings and projection**. Results hinge on the quality of frozen features and the chosen manifold projector, with limited guidance on hyperparameters or stability across settings.

3.  **Unclear computational costs.** No clear wall clock, memory, or scaling analysis for kNN construction and persistence steps, so the cost–accuracy tradeoff is unclear.

4. **Quantitative evidence is incomplete.** The paper’s claims, like *“up to 4× better precision” and improved proxy-to-target transfer*, are not consistently backed by tables, and the sensitivity of results to k-NN and manifold-projection settings remains largely unexplored.

### Questions
1. How sensitive are results to the manifold projection choice and its hyperparameters?

2. Please provide runtime and memory comparisons vs baselines on CIFAR and ImageNet to clarify scalability.

3. Do you have any non-vision results to support generality, for example ANLI or IMDB with a frozen RoBERTa encoder (D2 paper).

I like this paper and find the direction promising. I am at marginal accept and am willing to increase my score if the authors address my concerns in the rebuttal with concrete evidence.

### Soundness
3

### Presentation
4

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
The paper introduces TopoCore, a method for coreset selection. It is a combination of dimensionality reduction, non-parametric density estimation and persistent homology. Then, coresets are used for further training of ResNet-18. 
Experimental results show that the proposed method slightly outperforms baseline.

### Strengths
The paper proposes a new approach to coreset selection. 
This is one of the few applications of multipersistence to deep learning.

### Weaknesses
1) The paper is hard to understand. Some notions like "Hilbert decomposition signed measure" are not defined.
2) Some details of the method are missing (see Questions)
3) The difference is no statistically significant w.r.t. baselines in many cases (Table 5 in Appendix).
Please include statistical tests to validate significance.
4) Improvements over Random selection is quite small. I doubt that the method is of practical importance.
5)  A relevant publications is missing:

Trofimov, I., Cherniavskii, D., Tulchinskii, E., Balabin, N., Burnaev, E., & Barannikov, S. (2023). Learning topology-preserving data representations. arXiv preprint arXiv:2302.00136.

### Questions
1) As far as I understood, persistence scores are calculated for every class separately.  Are they summed next? 
2) The optimization of L_{pers} can naturally lead to a degenerate solution, like points very far from each other, which maximizes persistence. How do you handle it?
3) Is TopologyScore maximized or minimized or minimized?
4) In Table 1, why TopoCore exhibits different metrics in "no training dynamics" and "with training dynamics" blocks?
I assume that the difference must be only in baselines.
5) Some important details are hard to understand from the paper. How L_{proj} is optimized? Together with TopologyScore or not?
How the coreset is selected? Is should be a subset of a dataset, but I can't find details.
Where are similarities p_{ij} are taken from? etc.

### Soundness
2

### Presentation
2

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
The paper addresses the problem of coreset selection, i.e. a small representative subset of a large dataset that minimizes the degradation in model performance and allows for faster training and reduced storage. Although existing geometry-based methods do not require an expensive training, they rely on extrinsic metrics that make them sensitive to variations in feature embeddings. The authors propose TopoCore, a two-stage method for coreset selection that utilizes topology to accurately approximate the underlying manifold of the data. To preserve the global structure, during the first stage, feature embeddings of deep neural network are projected onto a low-dimensional manifold with UMAP. To preserve the local structure, during the second stage topological persistence of points is maximized independently for each class. The coreset selection is based on the TopologyScore that combines Density Score, reflecting global representativeness, and Persistence Score, reflecting local topological complexity. The empirical evaluation includes comparison with several baseline methods in both training-based and training-free scenarios, analysis of method’s performance when feature embedding model is varied. The authors also analyze the TopoCore robustness to the noise injected into feature embeddings.

### Strengths
- The paper proposes a novel topology-based view on the problem of coreset selection that leverages the geometric methods.
- Experimental results demonstrate that TopoCore outperforms benchmark methods, especially at high pruning rate and on more complex datasets.
- TopoCore is more robust to noise in the feature space, especially at the higher pruning rates (70-90%). 
- TopoCore provides better results across a wide range of embedding model choice.

### Weaknesses
- Although the paper provides some evidence for the choice of UMAP, a more recent works [1][2][3], which were shown to outperform UMAP with better preservation of data topology, are not considered for comparison and/or improvement of TopoCore.
- Experimental part is limited. As far as I understand, the experiments focus on the test accuracy of the ResNet-family models (ResNet-18, ResNet-50) for different pruning rates and embedding models. The evaluation lacks results for more recent architectures, for example, transformers, and estimation of other properties such as quality of transfer learning / domain adaptation.
- The paper does not provide any estimate on the computational cost of the proposed procedure. Is TopoCore more computationally intensive than the benchmark methods?

Minor: The notion of prototype is often used in the main text but formal definition is given only in the appendix.

[1] M. Moor et al. Topological autoencoders. ICLR, 2020.
[2] I. Trofimov et al. Learning topology-preserving data representations. ICLR, 2023.
[3] E. Tulchinskii et al. RTD-Lite: scalable topological analysis for comparing weighted graphs in learning tasks. AISTATS, 2025.

### Questions
Please, see weaknesses.

### Soundness
2

### Presentation
3

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
The paper proposes TopoCore, a novel training-free coreset selection framework that leverages topological representations of data rather than purely geometric ones. The method addresses the geometric brittleness of prior approaches that rely on extrinsic distances in feature space.
TopoCore operates in two stages:
1. Global topology-aware manifold embedding using UMAP-like projection to capture intrinsic structure and compute a density-based representativeness score.
2. Local topological optimization via differentiable multi-parameter persistent homology to compute a per-sample persistence score.
The two are combined into a unified TopologyScore for coreset selection. Experiments show that TopoCore outperforms geometric, gradient-based, and score-based baselines, especially under high pruning rates and noisy embeddings. The method also demonstrates strong cross-architecture stability.

### Strengths
The paper shows strong originality by unifying manifold learning with differentiable topology for coreset selection, introducing a novel use of persistent homology as an optimization objective. Its technical soundness is reinforced by comprehensive experiments across multiple datasets and architectures, demonstrating consistent gains—especially at high pruning rates. The method is also practically relevant, as it works directly with pretrained embeddings and scales to large real-world settings. Finally, the authors support reproducibility through thorough documentation and a forthcoming anonymous code release.

### Weaknesses
1. The paper does not provide a clear runtime or computational complexity analysis of the differentiable multi-parameter persistent homology component, which is generally more expensive than geometric or graph-based baselines. Without wall-clock comparisons or complexity estimates, it is difficult to evaluate the practical feasibility and scalability of TopoCore.
2. The paper reports mean performance but omits confidence intervals or statistical significance tests, making it difficult to assess the claimed stability improvements over baselines.
3. The paper does not clearly state which homology dimensions (e.g., $H_{0}$, $H_{1}$, or higher) are used to compute the persistence-based importance scores. Because different homology groups capture different types of topological structure, this omission makes it harder to interpret what the method is actually measuring.
4. The term “Hilbert decomposition signed measure” is used but is not clearly defined, and it does not appear in the cited reference (Botnan & Lesnick, 2022), making its mathematical meaning and implementation ambiguous. This lack of alignment with standard terminology in multiparameter persistence may confuse readers and hinders reproducibility.
5. Grammar errors:
- Line 171-172: "signifying it's structural importance" -> "signifying its structural importance"
- Line 316-317: "this approach is diverges" -> "this approach diverges"
- Line 995: "Coresest" -> "Coreset"

### Questions
1. Could the authors provide runtime measurements or a complexity analysis comparing the computational cost of the differentiable persistent homology component to that of existing coreset baselines, so that the practical scalability of TopoCore can be more clearly understood?
2. Could the authors clarify whether the concept “Hilbert decomposition signed measure” is directly borrowed from Stable Vectorization of Multiparameter Persistent Homology using Signed Barcodes as Measures (Loiseaux et al., 2023), which defines this object formally? In any case, could you include a clear definition for “Hilbert decomposition signed measure”?
3. How sensitive is TopoCore to the choice of UMAP parameters (e.g., Number Neighbors, Minimum Distance)? If you think the choice Number Neighbors=15, Minimum Distance=0.1 is optimal, could you explain?

### Soundness
3

### Presentation
3

### Contribution
3
