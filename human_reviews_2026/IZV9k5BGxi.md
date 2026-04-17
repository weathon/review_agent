# Global and Local Topology-Aware Graph Generation via Dual Conditioning Diffusion

- Decision: Accept (Poster)
- Scores: 2, 6, 6

## Abstract
Graph generation plays an important role in various domains such as molecular design, protein prediction, and drug discovery. However, generating graph-structured data poses challenges due to the complex dependencies inherent in graphs, spanning from intricate local substructures to broad global topologies.  Although recent advances in graph-generative models have made notable progress, traditional node-level generative paradigms may have difficulty simultaneously capturing the multiscale dependencies in graphs. To address these challenges, we propose a unified latent diffusion model that jointly learns local and global topological information, enabling effective and efficient graph generation. Besides, our approach introduces a dual conditioning mechanism designed to promote dynamic interaction between local and global information, equipping the generative model with global and local awareness to better capture the coupled dependencies within graphs. Our method can largely promote the joint modeling of global and local information and substantially improve the quality of the generated graphs. Extensive experiments consistently demonstrate the effectiveness of our method.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes DualDiff, a latent diffusion model designed for graph generation that jointly captures both local and global topological information through a two-branch diffusion process and a dual conditioning mechanism.

The model encodes input graphs via a pretrained graph autoencoder, extracts global representations via graph clustering (spectral or K-means), and performs parallel diffusion on node-level and cluster-level latent variables, alternately conditioning one on the other. Experiments are conducted on generic graph datasets (Ego-small, SBM, etc.) and molecular datasets (ZINC250k, QM9, MOSES).

### Strengths
- The motivation—capturing multi-scale (global and local) structural information in graph generation—is both timely and important.

- The paper is very well-written and full of details.

- The authors conduct comprehensive experiments, comparing with many recent diffusion-based and autoregressive graph generators.

### Weaknesses
1. The paper defines “global information” simply as the cluster centroids obtained from node embeddings via K-means or spectral clustering. This is a very coarse, heuristic, and outdated approximation of global structure. It is not truly “topology-aware” as claimed. The authors do not justify why a fixed, non-learned clustering is preferable.

2. The statement that “existing methods still leverage node-level generative paradigms” is too absolute. Many recent approaches already incorporate hierarchical or subgraph-level generation.

3. Although the authors emphasize that the encoder takes 3D coordinate information as input for molecular graphs, the model ultimately generates only 2D molecular structures (bond graphs) rather than full 3D geometries.

4. The authors claim Eq. (7) “implies that modeling the joint distribution of global and local information can be decomposed as two complementary processes” . In fact, Eq. (7) is merely a standard conditional-probability identity—any joint distribution can be written that way. Therefore, it provides no theoretical evidence that the proposed alternation scheme meaningfully models $p(Z_l​,Z_g​)$.

5. On ZINC250k (Table 2), the reported Validity = 92%, which is much lower than baselines such as GruM (99%), GraphArm (100%), and GDSS (97%).  This indicates a serious failure in generating chemically valid molecules and undermines the claim that DualDiff captures local chemical constraints (e.g., valence, functional groups).

6. Protein datasets, which contain larger graphs (100 < |V| < 500) are also widely used in generic graph generation. Moreover, related baselines such as GruM have reported results on these datasets. The authors are encouraged to include experiments on protein graphs to more comprehensively demonstrate the effectiveness and scalability of their proposed approach.

7. The paper does not provide a code release or supplementary implementation details.  
Given the model complexity and multi-stage training (autoencoder + dual diffusion), reproducibility is questionable. Code should be provided for verification.

### Questions
1. Why do the authors believe that a fixed, non-learnable clustering method (K-means or spectral) can define “global topology” more effectively than end-to-end hierarchical pooling approaches in modern GNNs?

2. How sensitive are the results to the choice of clustering algorithm and the number of clusters $K$?

3. Why not train the clustering jointly with the autoencoder or in the diffusion process to obtain adaptive global representations?

4. The model’s validity on ZINC250k is only 92%. Does this reflect an inherent inability to capture local chemical rules? Have the authors examined failure cases?

5. Can the framework be extended to explicitly generate 3D molecular structures rather than only 2D topologies?

6. Is there any theoretical basis to justify Eq. (8) as a valid probabilistic factorization of $p(Z_l, Z_g)$?

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
3

### Summary
This paper proposes a latent graph diffusion model that concurrently denoises node-level and cluster-level representations, coupled with a bidirectional conditioning mechanism to exchange global and local topological cues during generation. Extensive experiments on eight generic and molecular graph benchmarks demonstrate competitive or state-of-the-art performance in both unconditional and property-conditional generation while requiring fewer diffusion steps than prior diffusion counterparts.

### Strengths
1. The paper is well-written and easy to follow.
2. Using global & local latent features in graph diffusion models with learnable cross-conditioning is a reasonable design for the task.
3. The empirical results across diverse datasets show the effectiveness of the proposed method.

### Weaknesses
1. The paper uses different global feature extraction methods in different tasks, but how to choose the method is unclear, which may lead to difficulty in generalization to new tasks.
2. In ablation study, what if there is only local-to-global condition?
3. At the beginning of sampling process, the number of nodes and clusters should be pre-determined, how to decide these numbers? What about the generalization ability of these numbers?

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes DualDiff, a dual-branch latent diffusion model that learns both global and local graph structures. The key idea is to run two diffusion processes, one at the node level and one at the cluster level, and let them interact through a dual conditioning mechanism. This design helps the model capture both fine-grained details and overall topology. Experiments on multiple benchmarks show clear gains over prior graph diffusion models, proving that the method effectively models multi-scale dependencies in graphs.

### Strengths
1. Good motivation and clear design idea
The paper starts from a real problem that most graph diffusion models only handle node-level stuff. Splitting the process into local and global branches makes sense and is well explained.
2. Thoughtful architectural design
Alternating between the two diffusion branches is clever. It’s like doing local updates and then global aggregation, which keeps the training stable and avoids conflicts between the two processes.
3. Strong and broad experiments
They test on both synthetic graphs and real molecular datasets, and the gains are consistent. It shows the method isn’t overfitted to one type of data.

### Weaknesses
1. too dependent on clustering
The global branch comes from clustering nodes, so if the clustering is poor, the “global info” might be misleading. They don’t analyze this sensitivity much.
2 . Efficiency claim not fully convincing
They say the model is efficient in the latent space, but with two diffusion networks and alternating steps, it’s unclear how much heavier it actually is.
3. No solid theoretical grounding for stability
The alternating process is explained intuitively (like server-client updates), but there’s no formal guarantee or analysis of convergence.

### Questions
1. The paper mentions that the alternating scheme improves stability, but is there any quantitative or theoretical evidence to support this? What happens if the two processes are trained simultaneously, rather than alternately?
2. How sensitive is the model to the choice of clustering algorithm or the number of clusters K? Did you try other global extraction methods besides K-means or spectral clustering?
3. Computation and scalability: Since you have two denoising networks and alternating updates, how much additional compute or memory does this introduce compared to a standard latent diffusion model, such as Latent Graph Diffusion?

### Soundness
3

### Presentation
3

### Contribution
2
