# Autoencoder with Distribution Preservation

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
This paper proposes an improved autoencoder method. On the basis of maintaining the reconstruction accuracy, we introduce a data distribution preservation mechanism to improve the performance of the dimensionality reduction of the model. Traditional autoencoders only focus on the point-to-point distance between the input sample and its reconstruction result, ignoring the preservation of the overall distribution structure of the data. To solve this problem, we introduced the Kernel Mean Embedding (KME) term based on a kernel function with good topological properties into the loss function to measure the difference between the original data distribution and the reconstructed data distribution. This method effectively maintains the topological features and distribution characteristics of the global data. Experimental results show that compared with traditional autoencoders and existing topological autoencoders, our method performs better on multiple datasets, especially in terms of dimensionality reduction quality and structural preservation of latent representations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a novel approach to improving autoencoder-based dimensionality reduction by introducing a distribution preservation mechanism. The work addresses a significant limitation of traditional autoencoders and shows promising results.

### Strengths
The introduction of Kernel Mean Embedding (KME) with a dimensionality-unbiased Λ-kernel represents a meaningful advancement in distribution-aware dimensionality reduction.
The paper clearly identifies that traditional autoencoders focus only on point-to-point reconstruction error, neglecting the preservation of global data distribution structure - a crucial insight for improving representation quality.
The proposed Distribution Autoencoder (DAE) framework effectively integrates distribution loss with reconstruction loss, demonstrating improved performance over existing methods.

### Weaknesses
Some experimental data presentation is unclear. For example, Table 1 fails to explicitly map each metric to corresponding datasets and methods, hindering result interpretation.
There is no comparison of computational efficiency (e.g., training time, memory usage) between DAE and TAE. Although the paper mentions that DAE avoids complex topological computations, it lacks quantitative data to support its advantages on large-scale data.
While the paper provides mathematical proofs for the core logic of the Λ-kernel achieving dimensional unbiasedness through Voronoi diagram mapping, it lacks intuitive geometric or illustrative explanations (e.g., using simple low-dimensional data examples to show how Voronoi partitioning eliminates the impact of dimensional differences). This makes it difficult for readers to understand the working principle of the Λ-kernel. Additionally, the inherent connection between distribution loss and topological feature preservation (e.g., why distribution alignment can implicitly maintain topological structures) could be further strengthened through theoretical analysis or visual comparisons.

### Questions
It is unclear why the CH index shows such a large performance gap across different methods. The authors should provide an explanation or analysis of the factors contributing to this discrepancy.
There is no comparison of computational efficiency (e.g., training time, memory usage) between DAE and TAE. 
 The inherent connection between distribution loss and topological feature preservation (e.g., why distribution alignment can implicitly maintain topological structures) could be further strengthened through theoretical analysis or visual comparisons.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces the Distribution Autoencoder (DAE), a model designed to preserve the global structure of data during dimensionality reduction. It adds a novel "distribution loss" to a standard autoencoder, which compares the distribution of the original input data to that of the low-dimensional latent data. To enable this comparison between different dimensions, the authors propose using a "dimensionality-unbiased" kernel.

### Strengths
- The paper addresses a clear limitation of standard AEs (ignoring global structure). The proposed distribution loss, which explicitly compares the input and latent distributions, is a sound approach to regularizing the latent space in a dimensionality-reduction setting.

- The formalization of a "dimensionality-unbiased" kernel is a valuable contribution. The proofs demonstrating that the Gaussian kernel is biased and the proposed $\Lambda$-kernel is unbiased provide a sound theoretical foundation for the method's design.

### Weaknesses
- A primary claim of the paper is that DAE avoids the high computational complexity of TAE. However, the paper provides no empirical comparison of training time, inference time, or computational complexity between DAE and TAE. This omission is a major weakness, as it leaves the central efficiency claim unverified.

- The ablation study shows that performance almost consistently improves as $\lambda$ increases, with optimal results on MNIST, SPHERES, and RINGS achieved at $\lambda=1.0$. This means completely ignoring the reconstruction loss $\mathcal{L}_r$. It suggests the decoder and reconstruction loss are unnecessary and that the model is effectively just an encoder trained via KME. This undermines the "autoencoder" framing and warrants a much more critical discussion.

 - The method is presented solely as a technique for autoencoder-based dimensionality reduction. Its core objective is to force the latent distribution $Z$ to faithfully mimic the (often complex and sparse) manifold of the input data $X$. While this seems effective for preserving cluster separation and topological features, it also means the latent space becomes just as sparse and "gappy" as the original data. This design choice limits the method's applicability, as the resulting latent space is not well-suited for tasks that benefit from a continuous, dense, or "filled" representation, such as latent space interpolation or exploring variations.

- The overall contribution could be viewed as somewhat incremental. The idea of using KME or MMD to regularize latent spaces is not new (https://arxiv.org/abs/1711.01558), and the novelty here lies in the specific application (input vs. latent) and the "dimensionality-unbiased" kernel. The framework is largely an adaptation of existing components.

- There are several minor issues and typos throughout the paper that need to be addressed for better readability. In particular, the abstract states that the auxiliary loss has the purpose to "measure the difference between the original data distribution and the reconstructed data distribution", but it's evident from the paper that the comparison happens between the input and the latent.

### Questions
- Could the authors provide a direct performance benchmark to substantiate the claim that DAE is a more efficient alternative to TAE?
- Could the authors elaborate on Weakness 2? Does this imply the model is effectively just an encoder trained via KME, and if so, what is the justification for framing the method as an "autoencoder"?
- The paper argues that preserving the distribution implicitly preserves topology. Could you provide more theoretical justification for this link? Why should minimizing the KME discrepancy with the $\Lambda$-kernel be expected to faithfully preserve topological features?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces a new autoencoder-based dimensionality reduction method called Distribution Autoencoders (DAE). The method aims to preserve the global structure and topology of the data in the latent space. For that, the authors diverge from common topology-based losses and develop an RKHS-based loss that embeds the data points into a Hilbert space. The kernel mean embedding (KME), which describes the whole distribution, is obtained by averaging the kernel embedding across the dataset. KME are computed for both latent space and the original space, and the loss term is obtained as the product of these vectors. The theoretical contributions provide the proof of ``dimensionality-unbiasness'' of the $\Lambda$-kernel and bias present in the Gaussian kernel. To support the claims, authors present several experiments on synthetic (Rings & Spheres) and real-world (e.g. MNIST) datasets.

### Strengths
- A new method for dimensionality reduction that is positioned to preserve global structure without computing persistent homology.
- Results show improvements in clustering-based metrics compared to the AE and TopoAE, while visually, the global structure is respected much better than UMAP or t-SNE.

### Weaknesses
- **Topology preservation**: While the results seem to show some global structure preservation, it is not described which exactly topological features are preserved (e.g., 0-dim homology, or any dimension?). There is no theoretical grounding for why the method preserves topology, unlike TopoAE or RTD-AE (arxiv:2302.00136). It is also not clear why the property of the kernel dimensionality unbiasedness matters for global structure preservation. Additionally, when describing results on the Spheres dataset, the authors state that TopoAE is wrong in making the outer ring a ring and not a disk, while their method presents the correct picture. However, this is incorrect -- the topological type, more precisely, homology groups, for the high-dimensional sphere are more "similar" to the ring (non-trivial) than to the ball (which are trivial). So there should be 11 spheres (rings in 2D), with one. While TopoAE and AE are not perfect, DAE is worse since the outer ring/disk does not encircle the inner ones, which should be the case.
- **Computational complexity**: The introduction positions the method to preserve global structure while avoiding expensive PH computation. However, it is not clear how fast the method actually is, since the $\Lambda$-kernel seems to require computation of Voronoi cells for every point AND the distances from every point in the dataset to that cell. While autoencoder methods that compute PH can use batching to compute the PH partially, the kernel embedding dimension depends on the number of points, and doesn't scale well. This crucial limitation is not addressed.
- **Text & formatting**: the text has a substantial number of grammatical and formatting errors (e.g., lines 073, 088-089, \citep vs. \citet, etc.). There is a paragraph dedicated to describing AE, which is well-known and studied; however, related works lack detailed descriptions of TopoAE (as the main baseline), as well as the less commonly used RKHS, Voronoi cells, and Maximum Mean Discrepancy.
- **Missing related works**: since TopoAE (btw, wrong naming), there have been several works dedicated to preserving topological structure (for example, arxiv:2302.00136, arxiv:2306.17638), even with a focus on faster PH computation (arxiv:2503.11910). These are not mentioned or described in the paper.
- **Results**: Some ablation studies are missing, for example, proving that the Gaussian kernel indeed performs worse than the $\Lambda$-kernel. All the metrics used in the paper concern clustering, not the global structure (e.g., relations between clusters). It would be nicer to use some metrics from the previous papers (e.g. TopoAE) that tackle the global structure for adequate comparison.

### Questions
- Why is it important for the kernel to be dimensionality unbiased? Intuitively, how does it affect the latent representation and preservation of global structure?
- In case of the $\Lambda$-kernel, how is $\phi(x_i)$ computed? Do you calculate the distances for each data point to the Voronoi cell of any other data point? How are these distances calculated exactly, and how much time and memory does it require?
- Please, provide some results with the Gaussian kernel and explain why the property of dimensionality unbiasness matters for the global structure preservation and for constructing the loss function.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the problem of dimensionality reduction in terms of Autoencoder that promotes the preservation of the distribution of data. It was designed to replace the complicated measurement of Vietoris–Rips Complex with the Kernel Mean Embedding (KME) term based on a kernel function to maintain the topological features of the data.

### Strengths
The paper proves that the Gaussian kernel is dimensionality-biased for the task of dimensionality reduction, and that the proposed Λ-kernel is dimensionality-unbiased. Necessary experiments illustrate the performance of the so-called distribution autoencoder compared to either classic dimensionality reduction methods like PCA, t-SNE, or TAE.

### Weaknesses
The authors of the paper seem to be completely unaware of the works about Variational autoencoder (VAE), which is one of the most well known autoencoder model aiming to match the distribution of data and its low dimensional representations. Particularly, fair comparison between the proposed Kernel-based distribution autoencoder and VAEs is completely missing.

### Questions
What are the reasons for the complete missing of VAE in the discussion of the paper? Can the authors justify their missing discussion or comparison to the VAEs? 

Without clear and convincing reasons for leaving the VAEs out of discussion, the reviewer is not convinced that the present paper is completely fair to be accepted, despite the fact that that its development of Gaussian kernel and Λ-kernel is still valuable.

### Soundness
2

### Presentation
3

### Contribution
2
