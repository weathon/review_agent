# EUGENE: Explainable Structure-aware Graph Edit Distance Estimation with Generalized Edit Costs

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
The need to identify graphs with small structural distances from a query arises in domains such as biology, chemistry, recommender systems, and social network analysis. Among several methods for measuring inter-graph distance, Graph Edit Distance (GED) is preferred for its comprehensibility, though its computation is hindered by NP-hardness. Optimization based heuristic methods often face challenges in providing accurate approximations. State-of-the-art GED approximations predominantly utilize neural methods, which, however: (i) lack an explanatory edit path corresponding to the approximated GED; (ii) require the NP-hard generation of ground-truth GEDs for training; and (iii) necessitate separate training on each dataset. In this paper, we propose EUGENE, an efficient, algebraic, and structure-aware optimization based method that estimates GED and also provides edit paths corresponding to the estimated cost. Extensive experimental evaluation demonstrates that EUGENE achieves state-of-the-art GED estimation with superior scalability across diverse datasets and generalized cost settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Eugene, a new method for graph edit distance (GED) approximation. While the approach is clearly presented and compared with several existing methods, its claimed contribution of explainability is not supported by experiments, and its generalizability to other distance metrics is unclear. The method also requires manual parameter tuning and lacks efficiency results, with some recent relevant methods omitted from comparison. Overall, Eugene introduces a new GED approximation technique, but its practical benefits and novelty need further validation.

### Strengths
1.	The problem addressed is significant and has been extensively studied in the literature related to GED computation and estimation.
2.	The paper presents experimental comparisons with several existing methods.
3.	The paper is well-written and easy to follow.

### Weaknesses
1.	Why is explainability important? GED is just a score.
2.	Can your method generalize to other distance metrics in addition to GED? For example, maximum common subgraph (MCS)? If not, the generalizability of the method is limited, rather than the statement on line 062 that existing methods lack generalizability.
3.	Is Theorem 1 derived by you, or is it mainly from existing theoretical results in (Skitsas et al. 2023)? If Theorem 1 is from an existing method and you just use it for approximation, this makes the technical contribution limited.
4.	There are many parameters to be decided manually at line 1 in Algorithm 1. This may make it hard to use. What about changing the parameter values? Will this have a significant impact on the performance?
5.	There are no efficiency results in Section 4. This also suggests that the paper is not well written.
6.	The following recent methods are not compared:
o	Exploring Attention Mechanism for Graph Similarity Learning (2023)
o	Multilevel Graph Matching Networks for Deep Graph Similarity Learning (TNNLS)
o	GRASP: Simple yet Effective Graph Similarity Predictions (AAAI)
7.	The paper emphasizes explainability in the introduction as a major contribution, but there is no experiment on explainability at all. This may also suggest that, for GED estimation that is just a score, explainability is not important.
8.	The running time of the proposed method is on the longer side.

### Questions
Please see the eight issues above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduced a novel method to estimate the graph edit distance between two graphs, as well as the edit path.

### Strengths
The major strength of this method is that it does not require training, compared to other neural network approximation methods from the literature.

The proposed methodology follows the same approach introduced by Bougleux et al. (2017), by formulating GED as a quadratic assignment problem. However, there are some contributions, such as providing a structural-aware formulation and providing a deterministic resolution algorithm with the so-called Modified Adam algorithm.

### Weaknesses
The introduced novelties could be viewed as incremental, compared to the literature following the IPFP of Bougleux et al. (2017). It is also not a major progress that the proposed method requires a specific resolution algorithm, whereas the IPFP can benefit from off-the-shelf optimizers.

Some affirmations are straightforward, such as “the returned GED upper-bounds the true GED”, which is obvious.
Moreover, one could integrate some perturbation in the algorithm in order to enhance the estimation, thus getting closer to the true GED. Thus, the affirmation that the proposed Modified Adam algorithm is deterministic cannot be always viewed as a strength of the algorithm.

The paper is missing an in-depth study of the step values for the lambda, beyond the experiments given in Table T with several values, the best one being 0.5 for all datasets.

One of the 3 highlights of the proposed method, as stated by the authors in the paper, is the use of the CPU as opposed to GPU. This motivation of resource-efficient GPU-free execution pipeline is demonstrated in the appendix C.7 through Carbon emissions estimation. However, this analysis can be misleading. The used modifier Adam solver can efficiently explore GPU resources, in the same spirit as the Adam solver implemented in PyTorch, for instance.

The experimental results demonstrate that the proposed method outperforms the other compared methods. However, it is a bit weird how some old methods are often second-best, mainly H2MN of Zhang (2021) and ERIC by Zhuo & Tan (2022).
It is also not clear and unfair that the proposed method considers non-uniform edit costs, while the other compared methods are restricted to uniform edit costs (except GraphEdx).

The authors have chosen not to evaluate the 3 cost settings in all the tables, but only 2 out of 3. For instance, Cost Setting Case 3 is only given in Table J, while most tables explore Cases 1 and 2.

The captions of most tables are misleading because they do not present “Accuracy”, but errors when considering the MAE.

### Questions
No further questions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents EUGENE, a graph edit distance algorithm (without ML modules) based on ADAM and the so-called unrestricted graph alignment formulation. The authors perform rigorous experimental evaluations with various learning-based baselines and show that EUGENE performs better than several deep neural networks.

### Strengths
* This paper presents Eugene, an optimization-based heuristic algorithm that seems to outperform existing methodologies. Given the importance of graph edit distance computation in science and engineering, this is an important and significant effort.
* The experimental evaluation is comprehensive, covering learning-based neural network approximators and existing heuristic algorithms, and the evaluation is performed on nine diverse datasets.

### Weaknesses
* My main concern with this manuscript is with the writing and presentation: the authors seem to make several bold claims without rigorous evidence and justifications. Specifically,
    * In this paper, the authors linked efficiency to carbon emissions. While it is an interesting perspective, the estimation of carbon emissions is too rough: it just computes a fixed power with run time. Also, the comparison is unfair: the authors include both training and inference for GPU-based methods. But in practice, one can load a pre-trained model and only run inference, therefore, it is arguably unfair to simply add the one-time training cost.  Reporting inference time (and training time, if applicable) of all peer methods is a more reasonable and standard metric if you want to claim the efficiency of Eugene. 
    * The authors cited the survey paper (Skitsas et al, 2023) for the formulation of _Unrestricted Graph Alignment_, but I do not find the same formulation of Eq (3) and Definition 7 in that survey. Please provide the proper reference where this formulation is proposed.
    * The authors claimed that Eq (4) can handle arbitrary edit costs. This is not true given that $\kappa$ is a fixed value—if edge 1 in graph 1 has cost 1, 2, 3 to edges a, b, c in graph 2; but edge 2 in graph 1 all have the same cost to edges a, b, c in graph 2, I don't think it can be handled by Eq (4). Instead, the quadratic assignment problem formulation in IPFP can handle situations like this.
     * On L170, the authors claimed that their formulation has a time complexity of $O(n^3)$. I cannot see any justification for this claim. Also, the Frobenius norm term seems to be equivalent to $$tr(AA^\top) + tr(BB^\top) - 2tr(PBP^\top A^\top),$$ where $tr(AA^\top)$ and $tr(BB^\top)$ are constants, therefore, equivalent to maximizing $tr(PBP^\top A^\top)$, the same as the Koopmans-Beckmann QAP formulation. As a proven NP-hard problem, I don't think there is an $O(n^3)$ algorithm for that. Also, there should not be an $O(n^4)$ algorithm for the problem IPFP is solving, either.
    * On L171, the authors claim that C (used in IPFP's QAP formulation) is a dense matrix of size $n^2×n^2$, which is wrong. If the graphs are sparse, in the same way as the authors describe their own formulation, C is also a sparse matrix. The paper "Factorized Graph Matching" (TPAMI 2015) utilized this important property. 
* In evaluations, though Engene is claimed to be an efficient algorithm, an inference runtime comparison is missing. Also, the authors claimed that methods like Genn-A* cannot scale, but datasets such as AIDS have been reported in the original paper of Genn-A*. Even if Eugene does not show better accuracy, demonstrating a better accuracy-time trade-off is also important, as it provides readers with a complete picture of which algorithm to select for their specific applications. 
* It's a minor formatting issue, but you should not use parentheses for textual citations. For example, the parentheses should be removed in L296: 
  > As in **(Jain et al., 2024)**, we remove isomorphic graphs from the datasets ......

  There are more examples like this in this paper; please fix all of them in future revisions.

### Questions
In the comparison with baselines, did you reimplement and retrain the baselines or use previously reported results? If retrained, what efforts on hyperparameter tuning did you make to make sure it's a fair comparison?

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes EUGENE, an and non-neural optimization-based  method for approximating the GED between two graphs.  GED estimation is tackled  as a relaxed graph alignment problem over the space of doubly stochastic matrices, with a trace-based regularization that gradually encourages solutions toward quasi-permutation matrices. The method operates via a modified gradient-based optimization scheme. The authors present extensive experiments across multiple datasets and edit-cost regimes, showing that EUGENE matches or surpasses existing supervised neural approaches while requiring no training, no GPUs, and comparatively low computational resources. The paper positions EUGENE as an efficient, interpretable alternative to learning-based GED approximation methods.

### Strengths
The relaxation to doubly stochastic matrices and gradual regularization toward quasi-permutations is well-motivated.

EUGENE aims to explicit explicit node alignments to demonstrate explicit node edit paths. 

Benchmarks across many datasets, cost regimes, and baselines,  show EUGENE to be a better performer.

### Weaknesses
EUGENE is shown to outperform deep neural approaches by large margins, despite not using any data-driven learning. Neural methods typically require expensive supervision signals and are expected to excel in settings tied to specific datasets, so such a result is both striking and counterintuitive. What then accounts for EUGENE’s success? The paper currently does not provide a clear theoretical explanation for this performance gap. This needs a deeper analysis examining the geometry of the relaxed solution space, or the behavior of the optimization dynamics, or the implicit biases introduced by the regularization schedule,  to understand why this approach works so well. As it stands, the empirical effectiveness is quite surprising but not yet well-explained.

The optimization is also carried out directly on adjacency matrices, which suggests potential sensitivity to initial node ordering. It is unclear whether performance was evaluated across systematic graph permutations. If the method is indeed robust to such reorderings, this would be an important and nontrivial finding that deserves explicit discussion.

Finally, the overall relax–optimize–round strategy follows a well-established pattern in the graph alignment and quadratic assignment literature. The work currently reads as a strong optimization result rather than a machine learning contribution. This type of work requires thorough analysis regarding contributions relative  to prior convex/LP-based GED relaxations.

### Questions
Please refer to the weaknesses.
Overall, I remain genuinely puzzled by how such strong performance is achieved with virtually no supervision signal and minimal computational overhead.

### Soundness
2

### Presentation
2

### Contribution
2
