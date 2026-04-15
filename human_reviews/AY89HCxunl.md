# Cometh: A continuous-time discrete-state graph diffusion model

- Decision: Reject
- Scores: 5, 3, 3, 5

## Abstract
Discrete-state denoising diffusion models led to state-of-the-art performance in graph generation, especially in the molecular domain. Recently, they have been transposed to continuous time, allowing more flexibility in the reverse process and a better trade-off between sampling efficiency and quality. Here, to leverage the benefits of both approaches, we propose Cometh, a continuous-time discrete-state graph diffusion model, tailored to the specificities of graph data. In addition, we also successfully replaced the set of structural encodings previously used in the discrete graph diffusion model with a single random-walk-based encoding, providing a simple and principled way to boost the model's expressive power. Empirically, we show that integrating continuous time leads to significant improvements across various metrics over state-of-the-art discrete-state diffusion models on a large set of molecular and non-molecular benchmark datasets. In terms of VUN samples, Cometh obtains a near-perfect performance of $99.5$% on the planar graph dataset and outperforms DiGress by $12.6$% on the large GuacaMol dataset.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces a graph generation model named COMETH, which combines the advantages of discrete state denoising diffusion models with continuous-time dynamics. COMETH integrates graph data into a continuous-time diffusion framework, achieving a better balance between sampling efficiency and quality. Experimental evidence shows that this continuous-time integration significantly improves performance on various molecular and non-molecular benchmark datasets compared to related graph generation models.

### Strengths
1. This paper proposes a new continuous-time discrete state diffusion model framework, integrating discrete state diffusion models into a continuous-time framework to leverage the efficiency enhancement techniques of continuous diffusion models during training and sampling.
2. This model demonstrates strong performance across multiple benchmark datasets, surpassing state-of-the-art models in both molecular and non-molecular graph synthesis.

### Weaknesses
1. The proposed framework mainly combines existing techniques, such as continuous-time Markov chains (CTMC) with marginal transitions and random-walk encoding.
2. There is a lack of comparison with continuous diffusion models on large-scale datasets like MOSES and GuacaMol.
3. Although Appendix D demonstrates the performance of COMETH with fewer number of steps, there is a lack of comparison with the training and sampling times of related works.

### Questions
Mentioned in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors propose a continuous-time discrete-state diffusion model for graphs. As an additional contribution, the authors replace the graph-specific features often used to augment GNN-based denoisers (e.g. in Vignac et al. (2022) [1]) by a more-principled random-walk encoding.

### Strengths
- **Clarity of presentation**: the paper has a clear goal (introducing continuous time discrete-state graph diffusion to enable additional sampling methods) and suggests a suitable approach to achieve it (adapting the continuous-time framework proposed by Campbell et al. (2022) [2])
- **Empirical improvement**: COMETH seems to be competitive with existing baselines, achieving better results on at least a couple of metrics in all experiments.

### Weaknesses
The main weakness of this work is its incremental nature. Below I analyze the contributions of the work and explain my assessment.

**TL;DR** Since the continuous time framework for discrete data has already been proposed, a good direction for the present work is to provide a thorough evaluation of the design choices relevant to adapting this framework to graph diffusion. This can include for example both theoretical and empirical evaluations of the choice of loss function, noise schedule, and sampling procedures, to only name a few. A theoretical analysis of why continuous time can benefit graph diffusion in particular could also be beneficial here.

- **Limited technical novelty in adapting the continuous-time framework**: The paper applies Campbell et al. (2022)'s [2] continuous-time discrete-state framework to graph diffusion by treating nodes and edges as independent components. This adaptation follows naturally from Campbell's CTMC for handling independent dimensions and was already established for graphs in discrete time by Vignac et al. (2022) [1]. The paper does not introduce fundamental changes to either framework, making this contribution largely incremental. Perhaps a thorough empirical study demonstrating how the continuous-time process is a critical choice for graph-diffusion could make this contribution more relevant.
- **Insufficient justification for loss function choice**: The authors replace Campbell's ELBO loss with cross-entropy, citing only that "preliminary experiments using this loss led to poor empirical results" (page 5). No theoretical analysis is provided for why ELBO performs poorly or what guarantees might be lost by this substitution. While following Vignac et al. (2022)'s [1] choice of cross-entropy, this seems more like an engineering decision than a principled modification of the continuous-time framework. A theoretical (and empirical) analysis of the effects of each loss could be beneficial here.
- **Discussion of the noise schedule is misleading**: The authors claim that Campbell et al. (2022) [2] "used a constant noise schedule for categorical data" which they replace with a cosine schedule to ensure more gradual decay of data (Figure 2, Appendix B.1.). However, this misrepresents Campbell's work - their main text does not prescribe constant schedules, and they successfully use an exponential schedule β(t) = ab^t log(b) in most experiments. The only mention of a constant schedule appears in Appendix H.3 for one music generation experiment, noted as "sufficient for this dataset." By overstating Campbell's commitment to constant schedules, the authors inflate their switch to a cosine schedule as addressing a fundamental limitation, when it is simply a design choice made without theoretical or empirical justification over the exponential alternative. I recommend rephrasing the discussion of the noise schedule in the main text and the appendix to rectify this situation.
- **Proposition 1 is an adaptation of DiGress[1]'s forward process formulation** Proposition 1 adapts DiGress[1]'s forward process to continuous time through a direct substitution of rate matrices, without leveraging any continuous-time properties. The proof relies solely on matrix algebra, making this change incremental rather than a fundamental theoretical contribution.

- **The properties of the random walk encoding follow directly from Ma et al. (2023) [3]** the properties in question are the size of the largest component, whether two nodes lie in the same component, and counting the number of p-cycles in which a node is contained. 
    - While the first property extends Ma et al. [3]'s results from 'RWPP encodes distances' to 'RWPP captures structural properties', the result itself still follows from an argument regarding distances. A similar remark was made by Ma et al. [3] briefly in page 4: "[...] for K = n, we recover all shortest path distances (with disconnected nodes getting a distance of n, which is higher than the maximum distance n − 1 between connected nodes)." 
    - The second property is a direct consequence of the first: we can count the size of the largest component if we know the connectivity of all the nodes. 
    - The estimation of p-cycles uses DiGress [1]'s inference of p-cycles from powers of the adjacency matrix of the graph, which can be directly deduced from the random walk encoding. The result regarding the inability to identify p-cycles larger than 8 comes from an interesting combination of previous results, but constitutes a minor contribution in itself.
    - Even if these new properties are worth stating explicitly beyond the results of Ma et al. (2023) [3], I do not think this constitutes an important enough contribution for the present work, not to mention that it is not in fact its main focus. If the authors would like to emphasize this contribution, they can change the text of the paper to bring more focus to it, perhaps also comparing to other encodings in the process.
- **Absence of code/implementation**: it would be nice if the authors could share an implementation of their method + checkpoints for the model to replicate their results.
- **Improving presentation**: I believe including additional figures can make the main text richer and improve its flow. For instance, the authors could add figures showing the effect of denoising steps on the sample quality of COMETH, the denoising progression on example graphs for COMETH vs DiGress (to highlight the benefits of continuous time processes), etc.

### Questions
**clarifications**
- Can you explain why cross-entropy is a better loss when diffusing graphs compared to other discrete data? Is anything lost by foregoing ELBO in favor of CE? Sharing results comparing both approaches would be helpful as well.

- Can you expand on your noise schedule choice as it relates to graph diffusion in continuous time in particular? Why do you think the cosine schedule fits better here compared to other schedules? Empirical evidence could be useful.

- Could you elaborate on why the random walk encoding is a better choice than other encodings (e.g. Laplacian)? 

- Can you discuss why you think the random walk encoding is a sufficient replacement for the additional features used by DiGress? Namely, DiGress uses features beyond the graph structure (e.g. chemical properties of molecules), which might still be beneficial to the denoiser even with a continuous time process.

**suggestions**
- The authors should include campbell 2022 [2] in the related work section since their framework is heavily adapted in COMETH. 
- I found figure 1 a bit confusing. Perhaps the 'sample G^(t - tau)' can be directly between 'compute R' and 'predict G(0)'? 

**nitpicks**
- I recommend adding numbers to all equations.
- The first sentence in 'Present work' (first page) => graph is duplicated
- The first line in section **2. Background** has 'discrete-state' repeated twice.
- The numbers in table 3 and 4 are missing bolding
- The second part in the equation given in A.1 after "The rate matrix must satisfy the following conditions:" the second z(t) should be \tilde{z}
- The last sentence in 4.3 is ambiguous: 'This may be due to the fact that we train on a subset of the original dataset, whereas the LSTM is trained directly on SMILES.': is  the difference with LSTM the data modality or the training subsets?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a generative model for graph data using a discrete state space within a continuous time framework. It extends the previously proposed marginal transition noisy scheme, used for generating graphs, to a continuous-time setting. Additionally, the paper adopts a random-walk-based feature, which appears to enhance performance to some extent. Empirically, this approach achieves good results on synthetic and one molecular benchmark.

### Strengths
1. This paper have achieved state-of-the-art results on synthetic and one of the molecular benchmarks.
2. It makes notable machine learning contributions by adapting continuous-time discrete diffusion from [1] to develop a rate matrix, resulting in a noise model of marginal transition. The paper proposes using RRWP features to effectively model the graph, and appears to enhance performance to some extent.

### Weaknesses
1. From a machine learning perspective, the contribution of the proposed extension for marginal transition seems minor, and its impact on improving graph generation remains unclear.
2. While the RRWP feature seems to enhance generation performance, the lack of an ablation study makes it difficult to determine the extent to which each component contributes and which factors are responsible for the improvements.
3. There are inconsistencies in the abbreviations "RWPP" and "RRWP" on lines #314 and #847. Additionally, the last minus sign in the proof of Proposition 1 on line #82 appears to be incorrect.
4. The structure of the paper could be improved by making "Related Work" an independent section.

### Questions
1. On line #84, why is the discrete-time sampling scheme referred to as ancestral sampling, while continuous-time sampling is not?
2. How does the continuous-time Markov chain (CTMC) version of discrete-state marginal transition diffusion aid in graph generation?
3. On line #178, why is the difference between nodes and edges considered a challenge when the treatment appears to be the same throughout the paper?
4. The model performs well on QM9 but not as well on the other two molecular datasets. What might be the reason for this discrepancy?

[1] Andrew Campbell, Joe Benton, Valentin De Bortoli, Tom Rainforth, George Deligiannidis, Arnaud Doucet. "A Continuous Time Framework for Discrete Denoising
Models"

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents Cometh, a continuous-time discrete-state diffusion model for graph generation that combines the flexibility of continuous time with the structure-aware benefits of discrete-state modeling. The authors replace prior structural encodings with a simple random-walk-based encoding to enhance model expressiveness. Experiments show that Cometh achieves state-of-the-art performance across multiple benchmarks, notably outperforming existing models like DiGress on the GuacaMol dataset.

### Strengths
1. The discrete-state continuous-time framework for graph generation offers flexibility and is well-suited to the structural properties of graphs.
2. The authors conducted extensive experiments across datasets of varying scales, performed ablation studies, and presented results for conditional generation.

### Weaknesses
1. In lines 70-71, the authors claim that this is the first work using a continuous-time discrete-state diffusion model for graph generation. However, this is inaccurate as there is a concurrent work, [R1] (Xu et al., 2024). While a direct comparison is not required for Arxiv paper, it would enhance the paper's completeness to mention this work in the submission.

[R1] Xu, Zhe, et al. "Discrete-state Continuous-time Diffusion for Graph Generation." arXiv preprint arXiv:2405.11416 (2024).

2. The writing and organization of this paper need improvement. There is no whole picture of the problem and the proposed model. In the method section, there are many statements like “we followed Vignac et al. (2022)” and “Following the design choice of Campbell et al. (2022).” Although the authors improve upon previous methods, it would be clearer to distinguish the proposed method from prior work using a summary table or a separate preliminary section.

3. Some experimental results lack sufficient interpretation. For instance, in Table 1, DiGress appears to outperform the proposed COMETH in several metrics (Degree and Orbit), yet no explanation is provided. Additionally, the performance variation between COMETH and COMETH-PC for the Orbit metric is unexplained. Furthermore, the COMETH-PC results are omitted from Table 1 (SBM) and Table 2. Although the authors state that COMETH-PC does not improve SBM performance, including these results would contribute to completeness.

### Questions
1. In lines 130-131, the forward process starts with the marginal distribution, whereas in most diffusion models, it starts from a (discrete) uniform distribution. What is the reason for choosing the marginal distribution?
2. The computational resources and time requirements are not reported in the submission.

### Soundness
3

### Presentation
2

### Contribution
2
