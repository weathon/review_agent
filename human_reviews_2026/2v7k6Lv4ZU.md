# Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation

- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
Graph Domain Adaptation (GDA) aims to bridge distribution shifts between domains by transferring knowledge from well-labeled source graphs to given unlabeled target graphs.
	One promising recent approach addresses graph transfer by discretizing the adaptation process, typically through the construction of intermediate graphs or stepwise alignment procedures.
	However, such discrete strategies often fail in real-world scenarios, where graph structures evolve continuously and nonlinearly, making it difficult for fixed-step alignment to approximate the actual transformation process.
	To address these limitations, we propose \textbf{DiffGDA}, a \textbf{Diff}usion-based \textbf{GDA} method that models the domain adaptation process as a continuous-time generative process. We formulate the evolution from source to target graphs using stochastic differential equations (SDEs), enabling the joint modeling of structural and semantic transitions. 
	To guide this evolution, a domain-aware network is introduced to steer the generative process toward the target domain, encouraging the diffusion trajectory to follow an optimal adaptation path.
	We theoretically show that the diffusion process converges to the optimal solution bridging the source and target domains in the latent space. 
Extensive experiments on 14 graph transfer tasks across 8 real-world datasets demonstrate DiffGDA consistently outperforms state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces DiffGDA, a novel approach for Graph Domain Adaptation (GDA) that frames the problem as a continuous-time generative process. Instead of relying on discrete alignment steps or the construction of intermediate graphs, DiffGDA models the evolution from the source to the target domain using stochastic differential equations (SDEs), i.e., a diffusion model. The core contributions include: (1) A formulation of GDA as a continuous-time process that jointly models structural and semantic (feature) transitions. (2) A "domain-aware guidance network" designed to steer the reverse diffusion process, effectively learning an adaptation path from the source to the target distribution. (3) A theoretical justification (Theorem 1) showing that this guidance mechanism, by learning the density ratio between domains, converges to an optimal adaptation trajectory. (4) Extensive experiments on 14 transfer tasks, demonstrating that DiffGDA consistently outperforms state-of-the-art GDA baselines.

### Strengths
1. The core idea of modeling GDA as a continuous, time-driven generative process is novel and compelling. It directly addresses a clear limitation of existing data-oriented methods, which often assume a discrete or linear transformation between domains. The argument that real-world graph evolution is continuous and nonlinear provides strong motivation for this diffusion-based approach.

2. The method is thoroughly evaluated against a wide array of recent GDA baselines (both model-oriented and data-oriented) across 8 datasets and 14 transfer tasks. The results in Tables 1 and 4 show that DiffGDA achieves state-of-the-art performance, often by a significant margin.

3. The paper provides a solid theoretical foundation for its proposed guidance mechanism. Theorem 1 connects the optimal reverse SDE for the target domain to the source domain's score function plus a guidance term based on the density ratio $q(G_0^{\mathcal{T}})/p(G_0^{\mathcal{S}})$. This provides a principled basis for the guidance network's objective.

### Weaknesses
1. The primary concern is the practical efficiency and scalability of the proposed method. The framework involves training multiple components: a score network $\mathbb{P}(l)$, a guidance network $\mathbb{Q}(\delta)$ (which itself relies on a pre-trained domain classifier $\mathcal{C}_{gnn}$ for density ratio estimation), and a final GNN classifier. This represents a significant increase in complexity over simpler GDA methods. The "Discussion on Computational Cost" remark and the complexity analysis $\mathcal{O}(T\cdot n^{2} + L\cdot(|\mathcal{V}^{\mathcal{S}}|+|\mathcal{E}^{S}|))$ reveal a quadratic dependency on the number of sampled nodes $n$. This scalability issue is confirmed in the hyperparameter analysis for $\alpha$ (Figure 3), where the authors report running "out-of-memory" on tasks $D\rightarrow A$ and $D\rightarrow C$ for sampling ratios above 50%. This practical limitation seems to contradict the claim that the method is efficient and suitable for "modest computational resources."

2. The implementation of the guidance network depends on estimating the density ratio $q/p$ by training a GNN classifier $\mathcal{C}_{gnn}$ and using the approximation $(1-y)/y$. The stability and accuracy of this estimation, especially in the high-dimensional, sparse space of graphs, is a potential weak point that is not fully explored. The paper could be more explicit about the generation process. It appears the model generates a "labeled graph" $G'=(X', A', Y')$ by diffusing and reversing the concatenated input $\tilde{X}^{\mathcal{S}}=[X^{\mathcal{S}}||Y^{\mathcal{S}}]$. This is a key detail and implies that the model is learning to generate node labels as part of the diffusion process, which is an interesting but non-trivial aspect of the design.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the problem of graph domain adaptation, where a model trained on a labeled source graph is expected to perform well on an unlabeled target graph with different structures and semantics. The authors propose a new method called DiffGDA, which views the adaptation process as a continuous-time generative evolution rather than a step-by-step alignment.
The method uses stochastic differential equations (SDEs) to model how graphs evolve from the source domain to the target domain, capturing both structural and semantic changes. A domain-aware guidance network is introduced to guide the reverse diffusion process toward the target distribution. DiffGDA is trained jointly with a graph neural network (GNN) classifier, and additional MMD alignment and adjacency constraints are used to maintain consistency across domains.
Experiments on 14 cross-domain tasks from citation, airport, and social network datasets show that DiffGDA achieves higher accuracy and better efficiency compared with existing state-of-the-art methods.

### Strengths
1. It formulates GDA as a continuous-time generative process via SDEs, unifying structural and semantic evolution.
2. It provides a theoretical proof that the process works, which gives the method a solid foundation.
3. It demonstrates consistent superiority over state-of-the-art baselines on 14 graph transfer tasks across 8 real-world datasets.

### Weaknesses
1. It models domain transfer as a continuous graph evolution process but lacks explicit interpretability or concrete tracking of graph structural changes.

2. Diffusion-based methods are generally computationally expensive, and the scalability of the proposed approach to large-scale attributed graphs (e.g., ogbn-Products) remains questionable.

3. Experiments are conducted only on homogeneous graphs, lacking evaluations on more realistic heterogeneous graphs (e.g., IMDB) to demonstrate broader applicability.

### Questions
1. How does modeling the continuous generative process via SDE differ from standard DDPM-based diffusion approaches in terms of performance, efficiency, and stability?
2. Could the proposed method be extended to handle heterogeneous or large-scale graphs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes DiffGDA, a diffusion-based Graph Domain Adaptation (GDA) method that models domain adaptation as a continuous-time generative process via Stochastic Differential Equations (SDEs). It integrates a domain-aware guidance network to steer the diffusion trajectory toward the target domain, jointly capturing structural and semantic transitions. Theoretical analysis proves the diffusion process converges to the optimal adaptation solution, and experiments on 14 transfer tasks across 8 datasets demonstrate consistent superiority over SOTA baselines. The paper’s formulation is rigorous, with clear definitions and solid mathematical proofs, and the implementation details are sufficiently detailed.

### Strengths
Graph Domain Adaptation (GDA) is a very interesting and meaningful research direction. The writing of this article is clear. The author has given proof of relevant theories and conducted a relatively comprehensive experimental design. DiffGDA’s advanced effects are worth checking out.

### Weaknesses
1. Applying the diffusion model (Diff) to the Graph scene is interesting,the author did not discuss in depth the advancement and challenges of symmetric diffusion processes, as well as the key differences with the advanced methods of Diff applied in Graph scenarios; at the same time, the difference between the diffusion model in the image/video generation field and GDA has not been deeply discussed, which makes me doubt the innovation and contribution of this paper.
2. Lack of cross-domain (inter-domain) experiments: Current experiments focus on intra-domain transfers (e.g., citation → citation, airport → airport), where the data distribution has a similar pattern. Cross-domain transfer (e.g., citation → airport, social → citation) is more challenging and better reflects generalization, but it was not included, limiting the validation of the method's robustness to large distribution changes.
3. Incomplete hyperparameter analysis of diffusion steps: The number of diffusion steps is a very important parameter. In image generation, the number of diffusion steps often exceeds 3,000, but the paper only tested up to 100 steps (is the node representation in the graph less difficult to learn than the pixel representation in the image?).
4. The results do not clearly indicate whether the model has converged, and the impact of larger diffusion step sizes (e.g., 500, 1000) on performance and efficiency has not been explored. (If the author's GPU supports it)
5. The necessity of using MLP is not explored: in image diffusion, UNet has been shown to effectively capture spatial dependence. The paper uses MLP for scoring and guiding networks, but does not discuss why MLP is superior or necessary for graph data, nor does it compare it to graph-specific architectures such as GNN-based scoring networks.

### Questions
I hope the author can provide in-depth responses to the weaknesses and questions raised.

### Soundness
3

### Presentation
3

### Contribution
2
