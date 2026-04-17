# Latent Geometry-Driven Network Automata for Complex Network Dismantling

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
Complex networks model the structure and function of critical technological, biological, and communication systems. Network dismantling, the targeted removal of nodes to fragment a network, is essential for analyzing and improving system robustness. Existing dismantling methods suffer from key limitations: they depend on global structural knowledge, exhibit slow running times on large networks, and overlook the network’s latent geometry, a key feature known to govern the dynamics of complex systems. Motivated by these findings, we introduce Latent Geometry-Driven Network Automata (LGD-NA), a novel framework that leverages local network automata rules to approximate effective link distances between interacting nodes. LGD-NA is able to identify critical nodes and capture latent manifold information of a network for effective and efficient dismantling. We show that this latent geometry-driven approach outperforms all existing dismantling algorithms, including spectral Laplacian-based methods and machine learning ones such as graph neural networks and . We also find that a simple common-neighbor-based network automata rule achieves near state-of-the-art performance, highlighting the effectiveness of minimal local information for dismantling. LGD-NA is extensively validated on the largest and most diverse collection of real-world networks to date (1,475 real-world networks across 32 complex systems domains) and scales efficiently to large networks via GPU acceleration. Finally, we leverage the explainability of our common-neighbor approach to engineer network robustness, substantially increasing the resilience of real-world networks. We validate LGD-NA's practical utility on domain-specific functional metrics, spanning neuronal firing rates in the Drosophila Connectome, transport efficiency in flight maps, outbreak sizes in contact networks, and communication pathways in terrorist cells. Our results confirm latent geometry as a fundamental principle for understanding the robustness of real-world systems, adding dismantling to the growing set of processes that network geometry can explain.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Latent Geometry-Driven Network Automata (LGD-NA) for network dismantling—sequentially removing nodes to fragment a graph. The key idea is to estimate latent geometric distances on the graph using local, training-free automata rules, convert these to edge dissimilarities, sum per-node to obtain a geometric strength score, and perform dynamic dismantling (recompute scores after each removal). The study evaluates on an ATLAS of 1,475 real-world networks across 32 domains, by AUC of the LCC curve until 10% LCC, and reports top mean-field ranks for the latent-geometry family vs. centrality, message-passing and ML baselines. It also provides a GPU implementation and a robustness-engineering experiment.

### Strengths
1. Latent-geometry-driven dismantling, realized via local automata rules (RA2; CND ablation) that use only first-hop structure, is a clean and useful angle.
2. The paper is clearly written. Pipeline, metrics (AUC to 10% LCC), and reinsertion constraints are explicit.
3. The proposed work is evaluated on large-scale set, including 1,475 networks across 32 domains.

### Weaknesses
1. Paper alternates between “LGD-NA outperforms all” and “NBC achieves better dismantling but is slower.”
2. Results are for undirected, unweighted graphs; many domains are directed or weighted (transport flows, trade). Please consider adding experiments (or a clearly stated limitation) for those cases. 
3. Missing tuning/compute budgets and accelerated-NBC baselines.
4. Reinsertion and robustness-engineering analyses need stronger controls.

### Questions
1. NBC vs LGD-NA performance kind of unclear. Please provide a table with absolute AUC and mean-field rank side-by-side for NBC, RA2, CND under identical dynamic/reinsertion settings, and list fields where one strictly dominates.
2. Please show results for at least one directed (e.g., trade) and one weighted (e.g., power grid loads) domain, in order to make the paper more convincing.
3. I am kind of interested in the sensitivity analysis of the proposed method. For 3–4 strong methods, please plot AUC with no reinsertion vs 3 reinsertion policies and report ΔAUC. So that we can confirm rankings are stable or not.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Latent Geometry-Driven Network Automata Framework
The LGD-NA framework introduces a novel approach to network dismantling by leveraging local network automata rules to estimate effective link distances. ​

LGD-NA utilizes local automata rules to approximate node distances, enhancing dismantling efficiency. ​
It identifies critical nodes and captures latent manifold information for effective network dismantling. ​
The framework outperforms existing algorithms, including machine learning methods, across 1,475 real-world networks. ​
A common-neighbor-based rule achieves near state-of-the-art performance, demonstrating the effectiveness of minimal local information. ​
GPU acceleration enables efficient scaling to large networks, significantly reducing running times. 

The paper empirically validates the approach.

### Strengths
Contributions:
 1.  Latent Geometry-Driven (LGD) dismantling, where methods estimate effective node distances on a network’s latent manifold to expose critical structural information. 

2.  LGD-NA framework uses local network automata rules to approximate these geometric distances; a node’s summed distance to its neighbors estimates how critical it is for dismantling. 

3.  simple common neighbors-based rule, which we term Common Neighbor Dissimilarity (CND), is highly effective, achieving performance close to the state-of-the-art method, NBC. 

4.  comprehensive experimental validation on an ATLAS of 1,475 real-world networks across 32 complex systems domains, the largest and most diverse collection to date, showing that LGD-NA consistently outperforms all other existing dismantling algorithms, including machine learning methods.

### Weaknesses
I have issues with the objectives of a framework for LGD dismantling: what is the purpose of knowing about dismantling?

--fault tolerance: will a system fail?

--communications: will communications be disrupted?

--security: can a system's security be compromised?

The metrics used are totally abstract, and I would like to see a real-world application that shows the significance of your results. At present, it is an extension of small-world theory with no clear application.




The paper makes overly general claims about manifolds and the applications of this approach that must be scaled back, e.g, "network geometry captures essential structural and dynamical properties of complex systems". There is also the strong claim: "a novel strategy to engineer network robustness". The latter claim is unsubstantiated. Robustness is undefined; you are engineering graph-theoretical properties, not a precise notion of robustness.

Originally, small-world graph research showed that systems possess shared graph metrics. Now, you are using dismantling methods, but for what purpose? You need to show how dismantling impacts system performance. All you do is apply small-world theory with an application driven framework, and you MUST look at the application.

You need to be more precise about your use of manifolds and manifold theory. In complex systems, a latent manifold is a hidden, often lower-dimensional structure that captures the essential dynamics or configurations of the system. These manifolds are not directly observable but can be inferred from data using techniques like:

--Nonlinear dimensionality reduction (e.g., t-SNE, UMAP, Isomap)

--Autoencoders and variational autoencoders (VAEs)

--Diffusion maps or spectral embeddings

--Manifold learning in dynamical systems (e.g., Koopman operator theory)

Graph metrics like small-worldness, degree heterogeneity, clustering coefficient, and community structure are NOT manifolds themselves, but they characterize the topology of networks that may be embedded in or arise from latent manifolds.

--Community structure can hint at stratification or clustering on a manifold.
--Small-worldness suggests short geodesic distances on a latent space.
--Degree heterogeneity may reflect curvature or singularities in the manifold.

 Manifolds in complex systems can be:

--Non-Euclidean: Curved, with nontrivial Riemannian metrics.

--Stratified: Composed of multiple manifolds of different dimensions glued together (e.g., hybrid systems with discrete modes).
--Singular: Containing points where the manifold structure breaks down (e.g., bifurcations, phase transitions).

Multi-scale: Different dynamics dominate at different scales, requiring nested or hierarchical manifolds.

So while these metrics are topological descriptors, they can be indirect indicators of underlying manifold geometry.

For example, a system that can operate in multiple modes may have different structural properties important per mode. This corresponds to a system (in your sense) having different small-world metrics representing different modes. Typically, people use persistent homology to show structural consistency across dynamics---what you compute is different from this.

### Questions
Your notion of NETWORK ROBUSTNESS is theoretical only. Please define it precisely. How does NETWORK ROBUSTNESS apply to the 3 applications pointed out above, in a precise sense: fault tolerance, communications,  security? I am looking for validation of the strong claim: "a novel strategy to engineer network robustness".

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
The paper proposes Latent Geometry-Driven Network Automata (LGD-NA) for network dismantling. The general idea behind is to estimate effective link distances using local “network automata” rules and rank nodes by the sum of their incident dissimilarities, where highest-scoring nodes are removed with dynamic recomputation. This paper evaluates the proposed methods across a very large “ATLAS” of 1,475 real-world networks covering 32 domains, using AUC of the LCC curve (10% threshold) as the metric. Additionally, it also studies optional reinsertion strategies, where a GPU implementation is reported to yield substantial speedups and latent-geometry estimators (including betweenness as a global estimator) can explain why these strategies work. Finally, this paper leverages CND’s explainability to “engineer robustness” by adding edges among neighbors of critical nodes.

### Strengths
1. The method offers an intuitive latent geometry framing with dynamic recomputation, which remains simple, general, and effective in practice.

2. The evaluation spans 1,475 networks across 32 domains and follows a clear LCC AUC metric and protocol.

3. Matrix formulations together with a GPU implementation yield speedups that make the approach scalable.

4. The paper translates insights into an easy robustness intervention by closing triangles among neighbors of critical nodes.

### Weaknesses
1. There are no theoretical guarantees that connect the proposed geometry estimators to near optimal dismantling orders, leaving the case largely empirical.

2. Runtime comparisons underrepresent strong GPU or approximate betweenness baselines, weakening claims about practical efficiency.

3. Mean field ranking and a single ten percent threshold may hide domain specific behavior, and guidance on when CND versus RA2 is preferable is limited.

### Questions
1. Could you clarify whether CND or RA2 actually outperform NBC in accuracy across the dataset or whether their advantage is primarily speed, and include a consolidated table with AUC gaps with and without reinsertion.

2. What explains the lack of GPU gains for NBC and would approximate betweenness or hybrid CPU GPU pipelines change the outcome.

3. How sensitive are the conclusions to thresholds other than ten percent and to alternative fragmentation metrics, especially when stratified by domain or graph statistics.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The submission makes a contribution in the area of network dismantling, which largely has to do with understanding vulnerabilities of large and complex networks.  The paper proposes a new framework that provides insight into the geometry of the latent manifold and applies it to a wide range of networks with excellent performance.

### Strengths
The paper is quite well written.  The experimental study also appears to be extensive with accompanying code that will be freely distributed.

The experimental study is extensive and compares to a lot of other methods.

### Weaknesses
Limitations were not discussed in the main body and relegated to the appendix.  In the revision, it is important to include at least a brief discussion of the challenges and limitations of the contribution in the main paper.

### Questions
Although I do not work in the area of network dismantling, a classical approach that is inherently geometric and is also known to be powerful for robust network design is spectral graph theory, leveraging the spectra of graph Laplacians.  Is there any existing work on the use of Laplacians for this task?

### Soundness
3

### Presentation
3

### Contribution
3
