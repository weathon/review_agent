# EvA: Evolutionary Attacks on Graphs

- Decision: Accept (Poster)
- Scores: 8, 2, 4, 4

## Abstract
Even a slight perturbation in the graph structure can cause a significant drop in the accuracy of graph neural networks (GNNs). Most existing attacks leverage gradient information to perturb edges. This relaxes the attack's optimization problem from a discrete to a continuous space, resulting in solutions far from optimal. It also prevents the adaptability of the attack to non-differentiable objectives. Instead, we introduce a few simple, yet effective, enhancements of an evolutionary-based algorithm to solve the discrete optimization problem directly. Our Evolutionary Attack EvA works with any black-box model and objective, eliminating the need for a differentiable proxy loss. This allows us to design two novel attacks that reduce the effectiveness of robustness certificates and break conformal sets.
EvA uses sparse representations to significantly reduce memory requirements and scale to larger graphs.
We also introduce a divide and conquer strategy that improves both EvA and existing gradient-based attacks.
Among our experiments, EvA shows $\sim$11\% additional drop in accuracy on average compared to the best previous attack, 
revealing significant untapped potential in designing attacks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes EvA (Evolutionary Attack), a black-box, gradient-free attack on graph neural networks (GNNs) that directly solves the discrete edge-perturbation problem using a tailored genetic algorithm. Unlike dominant gradient-based attacks (PRBCD, LRBCD), EvA does not relax the adjacency matrix to a continuous space and can therefore optimize non-differentiable objectives such as accuracy, certified ratio, and conformal coverage.

### Strengths
The paper convincingly argues that gradient-based structure attacks are fundamentally misaligned with the discrete problem, e.g. gradients are local, ignore edge interactions, require relaxations, and can be obfuscated. The paper's empirical evaluation is comprehensive and the results are significant, not just marginal. EvA drastically outperforms all baselines, including the SOTA PRBCD, across multiple datasets.

### Weaknesses
The most significant weakness, which the authors acknowledge in the limitations, is the high query complexity. Genetic algorithms are inherently query-intensive.

### Questions
Could the authors provide a direct comparison of the total number of forward passes (queries) used by EvA versus the number of forward/backward passes used by PRBCD to achieve the results in Table 1?

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
This paper proposed a genetic adversarial attack. Experiments compared with a few gradient based attack are conducted with many ablation study.

### Strengths
**1.** Experiments in various aspects are done to validate the method performance with abundant figures for illustrations.

### Weaknesses
**1.** The presentation of the paper is bad. There's neither formulation nor algorithm written, or any figure to completely show the pipeline of the proposed attack. The description of method is just split around all section 3 and 4 without a clear introducing logic, instead just describing sentences and paragraphs concatenated. The methodology description highly relies on comparison with a previous baseline "PRBCD" which was not formulated introduced as well, making the part harder to follow. There are also too many verbal definitions which are lack of clear expression and only used once. In all, the bad writing makes me really hard to have a full view of the proposed method in details, and I recommend the author to rewrite and reformulate the paper entirely. 
 
**2.** The experiments lack fully comparison with other works. The paper only compares with a few gradient based attack baselines, excluding experimental comparison with all other kind of attack by just stating "gradients method are SOTA" "out perform others". Considering the gradient baselines raised contain only one in 2023 while all others are before 2020, and the proposed method itself is indeed one of "beaten" genetic method, this lack of new baselines and ones from other attack types are unacceptable.

### Questions
Please see weakness.

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
4

### Summary
This paper introduces EvA (Evolutionary Attack), a new framework for edge-based adversarial attacks on GNNs through a discrete evolutionary search rather than gradient-based optimization. EvA formulates the attack as a genetic algorithm that evolves a population of perturbation candidates, where each candidate encodes a small set of edge flips. The algorithm iteratively applies Selection, Crossover, and Mutation. They also design a divide-and-conquer strategy to handle large graphs. Because it only requires model evaluations, not gradients, EvA is model-agnostic and applicable to black-box settings. EvA consistently achieves larger drops in classification accuracy than gradient-based attacks. The evolutionary framework can also attack non-differentiable objectives, where gradient-based methods cannot operate.

### Strengths
This paper uses a discrete evolutionary search method for graph edge-based adversarial attacks on GNNs, without the need of gradients, making it model-agnostic and applicable to black-box settings. The evolutionary framework can also attack non-differentiable objectives. They show strong empirical performance comparing with gradient-based methods.

### Weaknesses
1. The proposed evolutionary search is highly heuristic and not guaranteed to find globally optimal perturbations. Many of its design--mutation rate, crossover scheme, and selection strategy--lack principled justification or ablation analysis. It remains unclear which components are critical for performance and how sensitive the attack is to hyperparameter choices.
2.  The algorithm is difficult to follow from the current text presentation. Including clear pseudo-code or an algorithm box would greatly improve readability and reproducibility.
3. The scalability and efficiency analysis are underdeveloped (e.g., runtime, total queries, memory usage). Currently there's not statistics showing the time and memory consumptions of PRBCD and EvA on various datasets.
4. Using evolutionary search is not entirely new. The main novelty here lies in engineering and scaling rather than in conceptual advances. The paper would benefit from a clearer discussion of how its evolutionary design specifically differs from prior heuristic or search-based attacks.

### Questions
I wonder how EvA performs on larger graphs like arXiv, Products, and Papers 100M [1].

[1] W. Hu, M. Fey, M. Zitnik, Y. Dong, H. Ren, B. Liu, M. Catasta, and J. Leskovec. Open Graph Benchmark: Datasets for Machine Learning on Graphs. 2020.

### Soundness
2

### Presentation
2

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
This paper introduces EvA, a black-box adversarial attack method for graph neural networks that uses a genetic algorithm to perturb the graph structure. Unlike gradient-based attacks, EvA directly optimizes discrete, non-differentiable objectives(such as classification accuracy) without relying on gradient approximations or domain relaxation. EvA highlights the limitations of gradient-based methods and establishes evolutionary search as a powerful, underexplored paradigm for adversarial attacks on graphs.

### Strengths
> 1. The paper's primary strength is its successful revival of evolutionary search, a paradigm previously dismissed as inferior. It demonstrates that with careful design, this approach can decisively outperform state-of-the-art gradient-based methods, challenging a core assumption in the field and opening a new direction for research.
> 2. The work is supported by comprehensive experiments and thorough ablation studies that validate every design choice. 
> 3. The significance of the work is greatly amplified by applying EvA to novel objectives beyond accuracy, such as breaking robustness certificates and conformal predictions.

### Weaknesses
> 1. The paper rightly notes the high query complexity as a limitation, but does not conduct a rigorous quantitative trade-off analysis between computational cost and performance improvement. In my opinion, this is essential for the practical evaluation of the method.
> 2. The paper presents compelling empirical evidence for EvA's superiority over gradient-based methods. However, the explanatory depth for this success seems limited, primarily resting on the well-established notion of gradient unreliability. A more profound analysis examining whether EvA's advantage stems from superior navigation of non-convex loss landscapes or effective exploitation of higher-order edge interaction effects would significantly strengthen the work and provide foundational insights for future research. For more details, please refer to the question section.

### Questions
- While EVA demonstrates that Genetic Algorithms can achieve considerable effectiveness in conducting adversarial attacks, I still feel I don't fully grasp its fundamental mechanisms.

- In section 3, the sentence "We hypothesise that EvA, leveraging the exploratory capabilities of GA, can explore the search space more effectively and avoid local optima, while PRBCD gets stuck." "Exploratory capabilities" and "avoid local optima" are essentially standard claims for all GAs, bordering on being tautological. The paper seems to fail to specify how exactly the exploration capability of the EvA manifests in the specific context of discrete graph perturbation spaces. However, the analysis of perturbation patterns might provide the most relevant clues. As shown in Appendix D.1 "Label diversity", this section compares the statistical characteristics of perturbation solutions found by EvA and PRBCD. The analysis reveals that EvA's perturbation connections are more uniformly distributed across nodes with different labels, and demonstrate a greater tendency to connect to high-degree nodes and high-margin nodes. While this analysis is highly valuable, it primarily describes "what the solution looks like" rather than "how this solution was progressively discovered." It would be enlightening if the authors could demonstrate how the genetic algorithm guides the search process, which could potentially enhance the paper's readability and conceptual clarity.

### Soundness
3

### Presentation
3

### Contribution
3
