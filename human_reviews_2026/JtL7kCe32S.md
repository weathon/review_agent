# Efficient Synthetic Network Generation via Latent Embedding Reconstruction

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 2, 8

## Abstract
Network data are ubiquitous across the social sciences, biology, and information systems. Generating realistic synthetic network data has broad applications, from network simulation to scientific  discovery. However, many existing black-box approaches for network generation tend to overfit observed data while overlooking characteristic network structure, and incur substantial computational overhead at scale. These practical challenges call for synthetic network generation methods that are both efficient and capable of capturing structural properties of networks. In this paper, we introduce Synthetic Network Generation via Latent Embedding Reconstruction (SyNGLER), a general and efficient framework for synthetic network generation that builds on latent space network models. Given an observed network, SyNGLER first learns low-dimensional latent node embeddings via a latent space network model and then reconstructs the latent space by building a distribution-free generator over these embeddings. For generation, SyNGLER first samples (or resamples) node embeddings from the generator in the latent space and then produces synthetic networks using the latent space network model. Through the latent space framework, SyNGLER preserves unique characteristics in networks such as sparsity and node degree heterogeneity, while allowing for efficient training with lower computational cost than many existing deep architectures. We provide theoretical guarantees by developing consistency results regarding the distance between the true and synthetic edge distributions. Empirical studies further demonstrate the effectiveness of SyNGLER, where SyNGLER efficiently produces networks that better preserve key network characteristics such as network moments and degree distributions compared with existing approaches.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a new generative network model, SyNGLER, that can replicate the network's structure. SyNGLER embeds the nodes into a low-dimensional space of dimensions r. Each edge A_{ij} is sampled according to the parameter \pi_{ij}, which can have a different distribution depending on the edge type (binary, continuous, etc.). \pi_{ij} is defined by the sum of \z_i^Tz_j+\alpha_i+\alpha_j+\rho_n, where \z_i is the embedding of the node, \alpha_i is the degree parameter, and \rho_n is a global sparsity parameter.

### Strengths
The introduction and related work are directly focused on the main contribution. 

The main contribution is interesting. It presents a general framework based on low-dimensional spatial embeddings that possesses several important properties. 

Algorithm 1 appears to be general enough to work with any embedding used in the process. It is also a very simple and understandable pseudocode.

The synthetic experiment shows that the model is able to replicate networks using the same method.

### Weaknesses
The assumptions must be improved. I greatly appreciated the simple explanation for Assumption 2.1. However, given that all these assumptions appear to come from the paper by Li et al. (2025), which is on ArXiv and has not been peer-reviewed, this paper should provide a demonstration. 

The replicability of the main paper is very low. While the algorithm is simple to follow, the embedding learning process is not described in the main paper. For example, subsection 2.3 starts with a learned embedding that is not described in the main paper. Similarly, the training process for \alpha and \rho and the SAMPLER process are not described. I understand that some cases are considered, such as the replication of the embedding, and others, but more details are needed. I also understand that the main idea is the proposition of a general framework, but some ideas must be given to allow the reproducibility of the paper. 

Theorem 3.1 is not demonstrated. Eq. (3) states a separation of the d_{KL} into three terms, which are later explained, but I could not see the proof of the theorem. If the demonstration is in another paper (Chen et al.), it must explicitly state that this was demonstrated in that paper to avoid confusion about a potential contribution. 

Most of the theoretical contributions are based on a reference to other papers that have not been peer-reviewed (Chen et al., Wu et al., Li et al.), and similar to Theorem 3.1, I could not figure out their demonstration. 

One of the main claims of the paper is its low training time complexity. Unfortunately, there is no theoretical analysis of the process, only an empirical analysis of some embedding process. 

The results seem to be misleading. The real networks have millions of nodes, but these are reduced to thousands by selecting the largest connected component. So, the final application is applied only to a small network (thousands of nodes), rather than millions of nodes, which is inconsistent with the main claim of the paper. 

Several baselines use a similar approach (assuming a probability for each edge) and should be considered baselines (e.g., Chung-Lu, mKPGM, BTER, etc.). This will improve the paper's results.

Figure 2 is misleading; a network can be shown in different forms using different visualization algorithms. This should not be used for comparisons. 

As shown in Figure 3, the proposed model is applied to very small networks. Another issue is the comparison against deep network models. Considering the similar process to the classical method (sampling an edge with a specific probability), it should be compared against these classical algorithms. Also, the training process seems to explode for a large number of nodes; therefore, the paper must increase the size of the network to consider millions of nodes. 

Table 2 is not discussed in the main paper. The metrics are not discussed in the main paper either.

In general, the main contribution has potential, but an important rewriting of the paper must be done. The main paper is not self-contained, lacking important details, which reduces the replicability and analysis of the results. 

Minor comments
The details of the Real-World datasets are in Appendix B.2

### Questions
Please the weaknesses.

### Soundness
2

### Presentation
1

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
This paper presents SyNGLER, a new framework for efficient and realistic synthetic network generation. The key idea is to separate representation learning and generative modeling: SyNGLER first fits a latent space network model to obtain low-dimensional node embeddings and degree parameters, and then trains a distribution-free generator (either via resampling or a score-based diffusion model) in the latent space. Synthetic networks are generated by sampling new embeddings from the generator and reconstructing edges through the latent model likelihood.

The authors provide theoretical guarantees on the consistency between the true and synthetic edge distributions, decomposing the KL divergence into interpretable terms related to sparsity, latent estimation, and generative modeling errors. Experiments on both simulated and real-world datasets (YouTube, DBLP, Yelp, PolBlogs) demonstrate that SyNGLER achieves superior fidelity to key network statistics (e.g., triangle density, clustering coefficient, eigenvalue and degree distributions) while significantly reducing computational cost (measured in e-FLOPs) compared to deep graph generative models such as VGAE, GRAN, and EDGE.

Overall, the paper proposes a conceptually elegant and computationally efficient method grounded in a statistical foundation.

### Strengths
* Good motivation and positioning: The paper clearly identifies computational and structural limitations in existing deep graph generation methods and motivates the latent-space approach convincingly.
* Clear exposition and well-organized presentation: The paper is clearly written, with the model pipeline, algorithm, and theoretical results systematically presented. 
* Theoretical rigor: The decomposition of KL divergence into interpretable components and the corresponding asymptotic analysis (Theorems 3.1–3.4) provide a solid theoretical underpinning.

### Weaknesses
* Limited empirical comparison with latest graph generative models. The baselines are somewhat dated (VGAE, GRAN, EDGE). Including the latest graph generators (e.g., those mentioned in the Introduction) would strengthen the empirical evaluation.
* Lack of comparison with network reconstruction methods. Since SyNGLER relies on fitting a latent space model, it would be informative to compare it with classical network reconstruction techniques.
* Unclear sensitivity to latent dimension and model choice. The performance of SyNGLER may depend on the latent dimension (r) and the choice of likelihood (Bernoulli vs Gaussian). A sensitivity analysis or ablation would help clarify robustness.

### Questions
1. Assumption 2.1(iv) requires the network not to be too sparse. How restrictive is this in practice for real-world sparse graphs?
2. Why was XGBoost chosen for the score network instead of neural parameterization? Does it affect sample quality or training stability?
3. Could the authors include metrics such as graphlet frequency distance or spectral entropy to capture more structural nuances?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper focuses on the topic of synthetic network generation, aiming to address a key challenge: how to generate graphs that preserve the structural properties of a real-world network while remaining computationally efficient. To tackle this, the authors propose SyNGLER, a two-stage framework. First, a latent space network model is used to learn low-dimensional node embeddings, then, a distribution-free generator is trained in this latent space to produce new node embeddings, from which synthetic graphs are generated. The paper provides theoretical guarantees via KL divergence decomposition and consistency analysis, and empirically shows on synthetic and real datasets that SyNGLER better preserves structural statistics while being more computationally efficient than existing methods.

### Strengths
1. The paper addresses an important problem — generating synthetic graphs that preserve structural properties while being computationally efficient.
2. Provides theoretical analysis.
3. The experiments cover both synthetic and multiple real-world networks (YouTube, DBLP, Yelp, PolBlogs), and evaluate diverse structural metrics (degree, clustering, eigenvalues, triangle density)

### Weaknesses
1. While synthetic graph generation is indeed a meaningful and important research direction, this paper focuses only on generating non-attributed graphs that replicate structural statistics (e.g., degree, clustering, spectrum) of a real network. First of all, in modern applications, most graphs are attributed, and the interaction between attributes and structure is often essential to the task, how could the proposed SyNGLER benefit such graphs remain undiscussed. Moreover, the paper does not demonstrate or validate any downstream usage, so it remains unclear what practical purpose these synthetic structure-only graphs serve beyond matching statistical properties.

2. Would be better if code could be provided.

### Questions
I was curious to see what would be the practical impact of the SyNGLER generated graphs, and how to validate these practical impacts. 

1. For the practical impact, the method only focuses on generating network structure (adjacency), without modeling node or edge attributes. In most of the real-world applications, graphs are attributed, and structure alone is insufficient. Ignoring attributes may lead to significant information loss, and it is unclear how the generated synthetic graphs could be used for tasks that rely on feature–structure interactions (e.g., GNN training, attributed diffusion). I wonder if the authors has any empirical or insights on how SyNGLER could be extended to attributed graphs?

2. For validating the practical impact, the paper demonstrates that the generated networks match several structural statistics of the input graph. However, structural similarity alone does not guarantee that synthetic networks can serve as valid surrogates for scientific analysis or downstream decision-making. How can we validate functional equivalence of the input graph and the synthetic graph?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
SyNGLER is an efficient framework that generates synthetic networks via latent embedding reconstruction. It learns low-dimensional node embeddings and trains a lightweight generator in the latent space. This approach reduces computational cost and preserves key network structural properties, such as sparsity and degree heterogeneity.

### Strengths
The SyNGLER framework efficiently generates synthetic networks using latent embedding reconstruction. It significantly reduces computational cost and preserves key structural properties like sparsity and degree heterogeneity. It also provides theoretical consistency guarantees.

### Weaknesses
The SyNGLER framework needs extensions for conditional generation and support for complex network types like directed, dynamic, or multilayer networks. The resampling variant (SyNG-R) risks exposing sensitive information and creating unrealistic duplicate node embeddings. Future development must include rigorous privacy-preserving mechanisms.

### Questions
Why was the XGBoost-based score approximation chosen for SyNG-D's generator?

### Soundness
3

### Presentation
3

### Contribution
3
