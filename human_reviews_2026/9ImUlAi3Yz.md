# Fair Bayesian Model-Based Clustering

- Decision: Reject
- Scores: 8, 4, 4, 6, 4

## Abstract
Fair clustering has become a socially significant task with the advancement of machine learning and the growing demand for trustworthy AI.
Group fairness ensures that the proportions of each sensitive group are similar in all clusters.
Most existing fair clustering methods are based on the $K$-means clustering and thus require the distance between instances and the number of clusters to be given in advance.
To resolve this limitation, we propose a fair Bayesian model-based clustering called Fair Bayesian Clustering (FBC).
We develop a specially designed prior which puts its mass only on fair clusters, and implement an efficient MCMC algorithm.
The main advantage of FBC is its flexibility in the sense that it can infer the number of clusters, can process
data where the choice of a reasonable distance is difficult (e.g., categorical data), 
and can reflect a constraint on the sizes of each cluster. We illustrate these advantages by analyzing real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a fair clustering algorithm based on a mixture model, namely FBC, which can automatically infer the number of clusters. In addition, an MCMC inference algorithm is proposed to obtain fair clustering results. The convergence of the algorithm is evaluated experimentally. Experimental results on three real-world datasets and three baseline methods demonstrate that the proposed method outperforms existing approaches.

### Strengths
1. The proposed fair clustering method based on a mixture model is novel and clearly presented.
2. The experimental results demonstrate the superior of the proposed fair clustering method in comparison with 3 well-known fair clustering models.

### Weaknesses
1. The paper does not explain the method to find the fixed number of cluster k* (in section 5.1).
2. The paper does not analyze the complexity (in big O notation) of the proposed MCMC algorithm (section 4).
3. The experiments are conducted on only several selected attributes of the datasets which might significantly affect to the results. Furthermore, the paper does not explain how those attributes are selected for experiments.

### Questions
See the weaknesses.
A minor comment: The paper states that “SFC is the first fair clustering method based on fairlets” (SFC (Backurs et al., 2019)). But this is not correct. The first fair clustering method based on fairlets is Chierichetti et al., (2017). Backurs et al., 2019 is an extension with “Scalable fair clustering”.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Fair Bayesian Clustering (FBC), a model-based approach for fair clustering that builds fairness directly into a Bayesian mixture model via a fairness prior whose mass concentrates on clusterings that respect proportional representation of sensitive groups in each cluster. Unlike most fair K-means variants, FBC can infer the number of clusters, works with data where a metric is hard to specify, and supports cluster-size constraints. The authors develop an MCMC inference scheme for the fair posterior and show on real datasets that FBC is competitive when K is known and can recover a “fair” K when K is unknown.

### Strengths
This paper introduces a Bayesian approach to fair clustering with a fair prior that concentrates posterior mass on proportionally fair clusterings, distinct from the K-means algorithms that dominate prior fair clustering work. This framing lets fairness be enforced probabilistically rather than via hard constraints.

The paper provides a concrete model specification and an MCMC inference scheme to approximate the posterior over the number of clusters, centers, and assignments under the fairness prior.

### Weaknesses
The paper does not clearly differentiate its Bayesian formulation from recent strong non–K-means fair clustering methods such as Fair Clustering via Alignment (FCA) [1] and FairDen [2], which already achieve strong fairness–utility trade-offs through alignment-based and density-based principles.

The proposed method, FBC. enforces only a proportional representation constraint across sensitive groups, which limits its flexibility in modeling fairness definitions that reflect real-world diversity, such as bounded or multi-attribute fairness. However, existing methods like FairKM [3] explicitly address multi-attribute fairness and could serve as benchmarks for evaluating FBC’s generality.

The MCMC-based inference proposed for FBC may not scale effectively to large or high-dimensional datasets, limiting its practicality relative to lighter fair clustering methods.

[1] Kim, Kunwoong, et al. "Fair Clustering via Alignment." arXiv preprint arXiv:2505.09131 (2025).

[2] Krieger, Lena, et al. "FairDen: Fair Density-Based Clustering." The Thirteenth International Conference on Learning Representations. 2025.

[1] Abraham, Savitha Sam, and Sowmya S. Sundaram. "Fairness in clustering with multiple sensitive attributes." arXiv preprint arXiv:1910.05113 (2019).

### Questions
Please refer to the weaknesses.

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
The work presents fair bayesian clustering, a novel method to address the limitation in existing fair clustering literature where users must specify the number of clusters, $k$, ahead of time. The proposed finite mixture model with unknown number of concepts is used in tandem with a "fair prior" to enact group fairness in the traditional sense.
The core contribution is a reduction of the group fairness constrain on balanced datasets to the existence of a matching map between instances of sensitive groups.
To avoid computational complexity in model parameter enumeration, the method places on a prior on this matching map instead with fairness enforced by construction.

### Strengths
### Compelling Theoretical Model + Results
- The connection between perfect group fairness and a matching map is the paper's strongest point. This is a very interesting reformulation of the fairness constraint, though I have some questions / concerns on this connection (see the following sections).
- To my knowledge this if the first work to take the Bayesian non-parametric approach to group fairness, which is. a clear and important contribution.

### Experimental Results
- The model is demonstrated to be competitive with non-bayesian methods when $K$ is fixed or reasonably inferred. These simulations seem consistent with the above theoretical insights.

### Weaknesses
### Model Issues
- The authors note early in the paper that "most existing fair clustering methods are based on the K-means clustering and thus
require the distance between instances and the number of clusters to be given in advance," but the matching map and "energy" is directly a function of distances between matched points. It seems contradictory to re-introduce such a major dependency when the paper claims to be avoiding the limitations of $k$-means. If a reasonable distance is difficult to define, then this prior becomes just as ill-defined as a $k$-means objective. This seemingly undermines one of the paper's central motivations--please correct me if I am wrong on this.
- The space of matching maps is massive and while verification checking may be fast, it seems the methods noted in the paper (like swapping in MH) would likely get stuck in local optima. The authors note that for faster convergence, they need to partially initalize $\mathcal{T}$ using optimal transport, which suggests that they are aware of this serious limitation in the methodology.
- The theoretical results seemingly breakdown unless very strong assumptions are imposed when $r > 0$. To circumvent the issues here, the solution is seemingly a "heuristic modification" or a very strong cardinality assumption (Proposition A.2). Thus, seemingly the main breakthroughs of this formulation fail to hold in the more common/general settings encountered within fair clustering.
- The lack of extension to new data points (not in training data) is seemingly a major practical limitation. A new test point has not corresponding clustering assignment under the matching map, so the model-based clustering algorithm seems severely weakened for practical utility beyond the fixed dataset.

### Missing Literature
The authors cite very few of the fair clustering papers, making it difficult to situate this result in the current landscape of known results. Specifically, it's odd that the authors note the main problem of fair clustering as pre-specification of the number of clusters, but many existing lines of work circumvent this exact issue. For example, fair hierarchical clustering is a major object of study in fair clustering that does not require such pre-specification but is not mentioned in this work. I encourage the authors to incorporate more review of the literature (see for example https://www.fairclustering.com)

### Questions
- What, if any, are the convergence guarantees of his MCM step procedure on a high-dimensional combinatorial space?
- Why is the distance computation in energy not the same issue as the noted central issue in $k$-means based clusterings?
- Can you discuss the hurdles in working on imbalanced data and why this framing is still useful in spite of them?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses significant limitations in the field of fair clustering, where most existing methods are derivatives of K-means. These methods typically require (1) the number of clusters ($k$) to be specified in advance and (2) a pre-defined distance metric, rendering them inflexible for datasets with an unknown cluster structure or non-trivial data types (e.g., categorical data). To solve this, the authors propose a "Fair Bayesian Clustering" (FBC) model.  The core technical innovation is a "fair prior", constructed to put its mass only on fair clusters, ensuring that the model's posterior distribution naturally favors solutions that satisfy group fairness with moderate violations. They develop an efficient MCMC algorithm for posterior approximation. Experimental evidence on an extend number of datasets show the method to beat most benchlines, achieving high fairness with minimal loss in performance. Especially, the algorithm's performance matches that of the fixed-number flat clustering algorithms in the best case and infers the number of clusters correctly, showing it to be effective.

### Strengths
The paper tackles a well-known and practical shortcoming of the dominant K-means paradigm in fair clustering. The inflexibility of fixed $k$ is a major hurdle in real-world applications. The move to a Bayesian mixture model is also a natural and powerful way to solve these problems. The ability to infer $k$ is a significant advantage for exploratory analysis, and the model-based framework is inherently more adaptable to diverse data types than distance-based algorithms.

The idea of a "fair prior" is very appealing. Instead of enforcing fairness as a hard constraint during optimization (which can be computationally complex) or as a post-processing step (which can be suboptimal), this approach bakes fairness directly into the model's assumptions. If the prior is well-designed, the MCMC sampler will naturally explore a "fair" solution space.

The authors explicitly acknowledge the difficulty of MCMC on constrained spaces and present their "matching" prior as a solution. This focus on developing a novel prior that also leads to an efficient MCMC algorithm is a key strength.

Finally, the empirical evidence is strong in the sense that the inferred number of clusters is mostly correct, and it achieves comparable performance with the previous methods in the best scenario, with even less running time.

### Weaknesses
Despite the earlier claim that the paper focuses on the "perfect fairness" scenario, in section 2.1.2., when $n_1 = \beta n_0 + r$, the fairness constraint can actually be moderately violated. I think the constraint violation should be further quantified in the main body.

Although the Bayesian fair prior and posterior update framework is new, the idea of "coupling" points from different sensitivity groups follows the tradition in the fair clustering community, but is more difficult to follow in the presentation of this paper. This paper introduces the mapping of points in the larger sensitivity group to the smaller, but then they evolved to messy notations like $T, T_0, E$ etc. Also in general the notation system can benefit from some clean-ups.

### Questions
Can you elaborate more on the case where $n_1 = \beta n_0 + r$ and why fairness is not guaranteed when you pick functions from the set of mappings $\mathcal{T}$? I think this is an important point to clarify and it'll be better if you can give concrete examples. Is the problem that the cluster assignment might not put the mapped sets of different sizes proportionally into different clusters? If so, can't you incorporate that constraint into the model?

I'm wondering if you can phrase the problem and solution differently using the language of fairlets as introduced in previous papers. The mapping set $\mathcal{T}$ that you are constructing seems to be equivalent to the definition of all possible decomposition of fairlets. It would be easier to understand in that sense.

I didn't find discussion on how well the new method scales compared to other methods as dataset size increases. Any ideas on that?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Fair Bayesian Clustering (FBC), a Bayesian model-based clustering framework that incorporates group fairness into mixture models. The key idea is to design a fair prior over cluster assignments based on a matching map between samples from different sensitive groups, ensuring that matched samples are assigned to the same cluster. An efficient MCMC inference algorithm is developed to approximate the posterior distribution over the number of clusters, cluster parameters, and the fairness-related latent variables. The method allows automatic inference of the number of clusters, supports both continuous and categorical data, and can incorporate size constraints. Experiments on benchmark datasets show that FBC achieves comparable fairness–utility trade-offs compared with existing fair clustering algorithms, while being able to infer the number of clusters automatically.

### Strengths
1. The paper is among the first to integrate group fairness constraints into a Bayesian mixture modeling framework (an existing model for clustering without fairness constraints), providing new insights into fairness-aware clustering and Bayesian non-parameter.

2. The proposed  fair prior through the matching map is sound, which allows inference via standard Bayesian sampling tools without strict constraints on parameters.

3. Unlike prior fair clustering methods (which are typically combined with $k$-means objectives), the proposed FBC model can infer the number of clusters $k$ through a hierarchical prior, which is an interesting extension.

### Weaknesses
1. The matching based ideas have also been used in fairlet and fair $k$-means methods, and here it is mainly reformulated as a Bayesian prior combined with MCMC. Although the paper claims to be the first to apply this idea in a Bayesian clustering setting, the core mechanism remains very similar to previous approaches, making it difficult to identify clear Bayesian advantages beyond inferring the number of clusters.

2. This paper lacks a comprehensive comparison with previous fair clustering models. Integrating clustering objectives with fairness is natural as it can guarantee the fraction of data points from each protected group while minimizing the clustering costs to make the grouping (clustering) results convincable. However, the proposed model in this paper does not consider optimizing the clustering costs, where a detailed discussion should be included.

3. The proposed method does not include a principled way to assign new data points at test time, which the authors acknowledge as future work. For a clustering method intended for practical use, this omission is significant.

4. Without explicit pseudo-code or a flow diagram, it is difficult for readers to fully understand the operational steps and practical implementation of the algorithm. 

5. A stated advantage of FBC is its capability to balance clustering utility and group fairness within a Bayesian framework. Nevertheless, the experiments only report individual metrics but do not present evidences that can highlight the trade-off between utility and fairness. Many tables do not bold or highlight the best results among the compared algorithms. This makes it difficult for readers to quickly identify performance differences and assess the relative strength of FBC.

### Questions
1. Can the authors clarify why inferring k is particularly important in fair clustering, and demonstrate cases where this leads to qualitatively different or better fairness outcomes compared to fixing k?

2. Can the authors provide a detailed discussion on the advantages of not simultaneously considering optimizing the clustering objectives in the proposed model?

3. Can the proposed FBC algorithm efficiently handle large-scale datasets such as KDDCUP99? Moreover, can FBC still infer a reasonable number of clusters K when applied to such large datasets? 

4. How sensitive are the experimental results to the choice of hyperparameters?  Could different parameter settings significantly change the fairness–utility balance or the inferred number of clusters?

### Soundness
2

### Presentation
2

### Contribution
2
