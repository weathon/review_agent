# MCbiF: Measuring Topological Autocorrelation in Multiscale Clusterings via 2-Parameter Persistent Homology

- Decision: Accept (Poster)
- Scores: 6, 8, 2

## Abstract
Datasets often possess an intrinsic multiscale structure with meaningful descriptions at different levels of coarseness.  Such datasets are naturally described as multi-resolution clusterings, i.e., not necessarily hierarchical sequences of partitions across scales. To analyse and compare such sequences, we use tools from topological data analysis and define the Multiscale Clustering Bifiltration (MCbiF), a 2-parameter filtration of abstract simplicial complexes that encodes cluster intersection patterns across scales. The MCbiF is a complete invariant of (non-hierarchical) sequences of partitions and can be interpreted as a higher-order extension of Sankey diagrams, which reduce to dendrograms for hierarchical sequences. We show that the multiparameter persistent homology (MPH) of the MCbiF yields a finitely presented and block decomposable module, and its stable Hilbert functions characterise the topological autocorrelation of the sequence of partitions. In particular, at dimension zero, the MPH captures violations of the refinement order of partitions, whereas at dimension one, the MPH captures higher-order inconsistencies between clusters across scales. We then demonstrate through experiments the use of MCbiF Hilbert functions as interpretable topological feature maps for downstream machine learning tasks, and show that MCbiF feature maps outperform both baseline features and representation learning methods on regression and classification tasks for non-hierarchical sequences of partitions. We also showcase an application of MCbiF to real-world data of non-hierarchical wild mice social grouping patterns across time.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a method to analyze the defect if hierarchically in a
clustering using novel filtrations for multiparameter persistent homology.

The multipersistence module induced by this filtration is shown to satisfy
several desirable properties, such as being decomposable in
rectangles, or characterize defects in a nonhierarchical clustering.

### Strengths
- Detailed mathematical analysis. The proof seem sound.
  - The resulting filtration is mathematically satisfying. 
  Rectangle decomposability a good to have.
  - The analysis with the 0,1-conficts is also interesting

### Weaknesses
- Code not available.
 - Maybe a bit math-heavy for this conference. I also think a little drawing
 could help for some definition/result.
 - Several definitions / proofs are hard to follow.
 - Experiments.
   - I'm not familiar with the state of the art, but from what I see, the only
   competitor is the conditional entropy introduced in 2003. It is thus hard
   to judge the practical impact of this construction.
   - I guess that this filtration can become easily very large but $N,M$ feel
   a bit small. How does this filtration scale w.r.t. these parameters? and
   what about its nerve equivalent? 



minor comments.
- l157. I think there is an issue with the indices. shouldn't it be
$m,m,m+1,m+1$ ?
- l173. Shouldn't the $P_i$s be totally ordered? "operation of building
subsets" this is a bit vague.
- l216. Every multifiltration is multi-critical?
- l248. Unless I missed something, Rivet can handle arbitrary module
presentation, so MPH can be computed for arbitrary homological degree. Rivet is
also slightly outdated to compute the hilbert function. For instance, `mpfree`
can compute the betti numbers significantly faster (and for arbitrary degree as
well), for which there is a python interface in `multipers`. Furthermore, as
the induced module rectangle decomposable (Prop 21), the rectangle
decomposition (computable in `multipers`) should recover the module.
- Def 7. Not very intuitive.
- l269. What is a strictly hierarchical clustering? Is the relation strict on
the restriction on which it is non-constant?
- l303. $C$
- l333. $y$
- Table 1. Who is raw $\theta$, and what about corollary 15 instead of feeding
it to a regresser?
- prop 21. finitely presentable? and in the proof, I think it requires prop 23.
- l701. since $\theta$ is piecewise constant?
- prop 23. I think this should be detailed a bit more.

### Questions
see weaknesses.
 - Are there some stability result w.r.t. $\theta$?
 - Is there a link with the interlevelset filtration (which has similar
 guarantees)? I'm not sure this is possible since IIC, the Mayer-Vietoris
 sequence isn't valid for the non-nerve version, but this might be possible
 with the nerve version?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper considers the following setting. Given is finite set $X$. For each real number $t \geq 0$, there is a partitioning of $X$, in other words, we have a family of different clustering ways. Importantly, it is not assumed to be hierarchical. The paper proposes a way to use 2-parameter persistent homology to describe such a family. The construction is elegant: given values $s \leq t$, one looks at all the clusters occurring between $s$ and $t$ and inserts a corresponding simplex into the complex $K_{s,t}$.
The paper proves algebraic properties of the resulting 2-persistence modules (pointwise finite-dimensional, finitely presentable and block-decomposable). Then the paper concentrates on the Hilbert functions (just pointwise dimensions) of the resulting bifiltrations and derives new measures of conflict and interpretations of the information captured by the TDA techniques. There are also experiments, offering new insights for, e.g., a real-world dataset showing how the grouping of mice changes over time.

### Strengths
- The construction of bifiltration is really nice and elegant. While the formal definition might be hard to digest for non-experts, the illustration in Fig. 1 does a great job explaining what is really going on.
- The paper provides new quantitative measures to clustering.
- The results generalize the previously known Sankey diagrams.

### Weaknesses
- As the paper admits, Hilbert functions are a rather crude invariant that loses some information. The paper poses exploration of other invariants as future work, which is fair enough.
- The first experiment (predicting crossing number) seems to be a bit weak, considering that even for MCBif the Pearson is still 0.544. I mean, the whole set up looks a bit artificial, but I am not an expert in clustering.

### Questions
I wonder if every block-decomposable module can be obtained from some clustering by the proposed construction? Of course, the module must also be assumed to be zero outside of the $s \leq t$ triangle.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces the Multiscale Clustering Bifiltration (MCbiF), a topological framework for analyzing multi-resolution, non-hierarchical clusterings that describe data at different levels of coarseness. MCbiF models how clusters intersect and evolve across scales using a two-parameter filtration of simplicial complexes, whose structure is studied through multiparameter persistent homology. The resulting stable Hilbert functions capture key topological properties: dimension-0 features measure the nestedness of clusters, while dimension-1 features quantify higher-order inconsistencies between them. The method generalizes dendrograms (for hierarchical cases) and extends Sankey diagrams to represent higher-order relationships.

### Strengths
The paper is well-grounded in theory and method, with clear mathematical development and interpretable ideas. Its main strength lies in the clarity and rigor of the framework, which extends topological concepts in a coherent way to analyze multiscale clustering.

### Weaknesses
However, the empirical evaluation is limited and lacks modern benchmarking. Experiments are mostly conducted on synthetic datasets and a single small-scale real-world example (wild mice social groups), which limits generalizability. The comparative analysis is restricted to information-theoretic baselines (conditional entropy), with no inclusion of state-of-the-art representation learning methods such as graph neural networks or topological embeddings. Performance variability and robustness analyses are not reported, and code availability is not mentioned, hindering reproducibility. While the theoretical development is strong, the translation to practical machine learning applications is limited, reducing the paper's accessibility and impact for a broader representation learning audience.

### Questions
- Could the authors elaborate on how MCbiF could be applied to modern large-scale or high-dimensional datasets (e.g., image, text, or graph data)? This would help clarify the method's practical relevance beyond theoretical or synthetic settings.

- Why were only information-theoretic measures (e.g., conditional entropy) used as baselines? Including comparisons with recent representation learning or topological deep learning methods (e.g., graph neural networks, persistence-based embeddings) could strengthen the empirical validation.

- While the paper provides theoretical insight into Hilbert functions, could the authors give more intuitive examples or visualizations showing how specific topological patterns correspond to interpretable data behaviors?

### Soundness
2

### Presentation
2

### Contribution
2
