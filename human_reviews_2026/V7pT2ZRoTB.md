# Theoretical Guarantees for Causal Discovery on Large Random Graphs

- Decision: Accept (Poster)
- Scores: 4, 2, 8, 4

## Abstract
We investigate theoretical guarantees for the \emph{false-negative rate} (FNR)—the fraction of true causal edges whose orientation is not recovered, under single-variable random interventions and an $\epsilon$-interventional faithfulness assumption that accommodates latent confounding. For sparse Erdős--Rényi directed acyclic graphs, where the edge probability scales as $p_e = \Theta(1/d)$, we show that the FNR concentrates around its mean at rate $O\bigl(\tfrac{\log d}{\sqrt d}\bigr)$, implying that large deviations above the expected error become exponentially unlikely as dimensionality increases. This concentration ensures that derived upper bounds hold with high probability in large-scale settings. Extending the analysis to generalized Barabási--Albert graphs reveals an even stronger phenomenon: when the degree exponent satisfies $\gamma > 3$, the deviation width scales as $O\bigl(d^{\beta - \frac{1}{2}}\bigr)$ with $\beta = 1/(\gamma - 1) < \frac{1}{2}$, and hence vanishes in the limit. This demonstrates that heterogeneous, heavy-tailed degree structures commonly observed in empirical networks can intrinsically regularize causal discovery by reducing variability in orientation error. These finite-dimension results provide the first dimension-adaptive, faithfulness-robust guarantees for causal structure recovery, and challenge the intuition that high dimensionality and network heterogeneity necessarily hinder accurate discovery. Our simulation results corroborate these theoretical predictions, showing that the FNR indeed concentrates and often vanishes in practice as dimensionality grows.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors derive high-probability deviation bounds for the (normalized) number of true edges missed due to incorrect causal ordering — that is, for the false negative rate — in the context of causal discovery with interventional data on random graphs. Their theory applies to the score-optimal causal order (as defined by the score function introduced in Chevalley et al., 2025c) and covers both Erdős–Rényi and Barabási–Albert graph models.

### Strengths
The presentation of the technical contributions of the paper is clear. The technical contribution themselves seem solid (although I haven’t checked the proofs myself). The empirical evidence of figure 1 convingly support the theory.

### Weaknesses
- **Presentation.**
    - In the abstract and introduction, it is not clear that the FNR concentration refers to the causal order that optimizes the specific score in Eq. (2) (this only becomes precise later via the notation and Assumption 2). This oversells the scope: the bounds apply to the score-optimal order for Eq. (2).
    - The goal is scattered: L48–49 suggests a broad “analysis of interventional CD,” whereas L52–53 focuses on finite-sample deviation bounds for FNR. Please state *up front* (abstract + intro) that the paper derives deviation bounds for $D_{\text{top}}$ and its normalization (FNR) **for the Eq. (2) optimizer**, on ER and BA random graphs. The problem definition currently appears only in Related Work (L135–143); it belongs in the introduction.
- **Motivation**. I’d appreciate a clearer case for the value and motivation of this work.
    - Why does the authors believe that it is valuable to prove this results for optimizers of the score of eq. (2)? Should we expect similar behaviour for the causal order found by other interventional causal discovery methods? Can we show that empirically?
    - Why do the authors believe that it is interesting to study concentration bounds for synthetically generated graphs? Would the theoretical results on ER/BA be relevant for real world causal discovery? Can we have an empirical analysis on real world / semi synthetic data (e.g. syntren) supporting the claim that these results would be useful for real world problems?
    
    The technical contributions are sound. I don’t see why they are valuable for the causal discovery community, due to their narrow scope (they refer specifically to optimizers o the score of eq.(2)) and empirical evidence limited to artificial settings.

### Questions
- **Experiments.** I’d like to see the plots of figure 1  also for $D_{top}$, to visualize whether there is any additional information we can gain from knowing concentration results on FNR on top of what we already know for $D_{top}$
- **Assumption 2.** Is there any dependency of the results on the tie breaking rule chosen?
- **L427-428**: authors write that “prior work shows” that DiffiIntersort “achieves close approximations [..] and is scalable”. What prior work?

### Soundness
3

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
4

### Summary
This paper proposes a new framework for causal structure discovery based on a finite-dimensional permutation representation of DAGs. The authors develop a differentiable objective function in which acyclicity is enforced implicitly through the permutation parameterization rather than via explicit matrix constraints as in NOTEARS. They derive theoretical results showing that this formulation admits dimension-adaptive and faithfulness-robust guarantees for causal structure recovery under ideal conditions. The work thus contributes a mathematically elegant alternative to existing continuous-optimization approaches, linking causal identifiability to permutation-based relaxations. Although the focus is primarily theoretical, the paper positions the framework as a potential foundation for scalable causal discovery in high-dimensional or heterogeneous systems.

### Strengths
1.	Ambitious theoretical framing.
The paper tackles a central challenge in causal structure discovery—learning DAGs in high-dimensional or heterogeneous settings—by proposing a finite-dimensional permutation formulation that encodes acyclicity by construction. The theoretical framing is ambitious and conceptually interesting, even if the practical payoff is not yet demonstrated.
2.	Sound mathematical development.
The formal derivations appear internally consistent, and the proofs provide a logically coherent set of identifiability and faithfulness-robustness results for the proposed objective. The mathematics is solidly presented, though its empirical implications remain uncertain.
3.	Conceptual link between identifiability and optimization.
By connecting permutation-based relaxations with identifiable scoring properties, the paper makes a reasonable attempt to bridge causal identifiability theory and differentiable optimization, offering a potentially useful theoretical synthesis.
4.	Alternative view of acyclicity enforcement.
The permutation-based representation provides an alternative to matrix-based continuous optimization methods like NOTEARS, handling acyclicity implicitly rather than through explicit constraints. This framing offers a different theoretical perspective, though the comparative advantages are not empirically explored.
5.	Foundation for future empirical work.
The finite-dimensional formulation could, in principle, enable more scalable algorithms once effective optimization strategies are developed. As it stands, the framework provides a theoretical starting point for such future work.
6.	Clear high-level motivation.
The goal of achieving dimension-adaptive, faithfulness-robust causal discovery in complex systems is well articulated, and the paper situates itself within a coherent line of theoretical progress, even if the empirical validation is lacking.

### Weaknesses
1. Connection between theory and search.
The paper presents strong theoretical results on identifiability and robustness for a permutation-based DAG formulation, but it remains unclear how these guarantees translate to the actual optimization procedure. The analysis establishes properties of the objective function and representation rather than of the search itself. Clarifying whether the theoretical results apply to the implemented algorithm—or only to its ideal global optimum—would strengthen the paper.
2. Relation to existing permutation-based approaches.
The method builds on a permutation representation of DAGs, similar in spirit to prior work such as Lam, Andrews & Ramsey (2022). That literature already discusses both the strengths and limitations of permutation-based formulations, including the sensitivity of search to scoring design and high-dimensional heterogeneity. The authors should clarify what aspects of their framework overcome these earlier difficulties and in what precise sense their theoretical guarantees are novel.
3.	Comparison with continuous acyclicity formulations.
The paper positions its method as distinct from continuous-optimization approaches such as NOTEARS, DAG-GNN, and GOLEM, but the practical differences are not clearly explained. Since those methods also encode acyclicity differentiably, a clearer contrast—both conceptually and empirically—would help establish the contribution.
4.	Claims of scalability.
The paper repeatedly describes its framework as “large-scale” and “dimension-adaptive,” yet these appear to refer to theoretical rather than empirical scalability. The experiments do not extend beyond a few hundred variables, whereas recent permutation-based methods such as BOSS (Andrews et al., 2023) have been validated at scales of up to 1000 nodes. It would strengthen the work if the authors clarified whether “large-scale” refers only to asymptotic complexity or whether they expect genuine computational advantages in practice. If the latter, additional empirical evidence or runtime analysis would be needed to substantiate the claim.
5.	Scope of empirical evaluation.
The empirical results are limited to moderate-size graphs and do not provide comparisons to recent permutation-based or continuous-relaxation algorithms. Demonstrating competitive recovery or runtime behavior would help connect the theoretical framework to its practical utility.
6. Presentation and framing.
The exposition is difficult to follow, and the central algorithmic contribution is not clearly stated in the abstract or introduction. Key distinctions—particularly between the theoretical guarantees and the practical optimization behavior—are left implicit. Several claims, especially those concerning robustness, scalability, and novelty, would benefit from clearer qualification and direct comparison to prior work.

### Questions
1.	Clarifying the algorithmic contribution.
It is not clear from the abstract or introduction that the paper proposes a new algorithm rather than purely a theoretical framework. Could the authors explicitly describe the practical procedure—what is optimized, how the permutation variable is represented or relaxed, and how the optimization proceeds? A concise summary of the algorithmic steps would greatly aid comprehension.
2.	Connection between theory and practical search.
The theoretical guarantees are stated in terms of identifiability and dimension adaptivity, but it remains unclear how these relate to the performance of the implemented optimization. Are the guarantees properties of the objective itself (assuming global optimization), or do they also hold for the proposed search procedure in finite-sample, non-ideal conditions?
3.	Relation to prior permutation-based work.
Prior studies such as Lam, Andrews & Ramsey (2022) identified both strengths and limitations of permutation-based DAG formulations. How does the present approach overcome those issues—particularly the tendency toward local minima and sensitivity to score design? Clarifying this connection would help establish novelty.
4.	Comparison to continuous acyclicity formulations.
The paper briefly mentions NOTEARS and related work, but the empirical and theoretical distinctions between matrix-based and permutation-based relaxations are not fully developed. Could the authors elaborate on what advantages their finite-dimensional permutation formulation provides relative to these existing continuous-optimization methods?
5.	On claims of scalability.
The paper describes the method as “large-scale” and “dimension-adaptive,” but the experiments are limited to moderate-size graphs. Could the authors clarify whether “large-scale” refers only to asymptotic complexity, or whether they expect tangible computational benefits in practice? If the latter, are there empirical results or runtime analyses supporting this claim?
6.	Faithfulness-robust guarantees.
The abstract mentions “faithfulness-robust guarantees for causal structure recovery.” Could the authors specify what kind of robustness is meant—e.g., tolerance to near-unfaithful parameterizations, violations of Markov equivalence, or resilience of recovery under approximate conditional independencies?
7.	Terminology and framing.
The terms “dimension-adaptive,” “finite-dimensional,” and “summary formulation” are used prominently but somewhat interchangeably. Could the authors provide a more precise definition of each and clarify how they differ conceptually?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies identifiability properties of causal graphs under single-node interventions from a probabilistic perspective. In particular, the authors study the number of misoriented edges (or, normalizing, the false negative rate), when the topological ordering of the graph is estimated via the InterSort algorithm. 

The authors study three random graph models: dense Erdos-Renyi (ER) models, sparse ER models, and a generalized Barabasi-Albert (BA) model, and a single intervention model where each single-node intervention is present with probability $p_{\text{int}}$. For graphs on $d$ nodes, they show that the expected false negative rate (FNR) under the dense ER is $O(d^{-1})$, and for the latter two models, they show that the expected FNR is $O(1)$. Further, they provide deviation bounds on the FNR: in the dense ER model, they show that the FNR has deviations of order $O(d^{-1/2})$, in the sparse ER model, deviations are of order $O(\log(d) \cdot d^{-1/2})$, and under the BA model, the deviations are of the order $O(d^{\beta - 1/2})$, where $\beta = (1 + \kappa/m)^{-1} \in (0, 1)$ for an attractiveness parameter $\kappa > 0$ and a link-count parameter $m > 0$. These theoretical findings are corroborated by experiments on synthetic data using the DiffInterSort extension of the InterSort algorithm.

### Strengths
**Originality and significance:** To the best of my knowledge, this paper is the first to study deviation bounds for identifiability metrics in random causal graph models. Compared to just expectation results, these results are more informative and provide stronger guidance for downstream applications (e.g., the development of causal discovery algorithms which are targeted towards more easily identifiable cases).

**Quality and clarity:** The theoretical results are strong under the relatively weak assumptions (random intervention targets and single-node interventions only identify neighbors). The presentation is well-structured, with the results presented in a very logical order, the writing is clear, with related work being particularly well-described.

### Weaknesses
In my opinion, the main weakness of the paper (shared by similar papers in the area, and even related areas like average-case complexity theory) is a somewhat shaky relation between theory and practice. In the "Limitations" section, the authors acknowledge that the random graph models are somewhat realistic, and I would also prefer the paper to emphasize that the "random intervention" model may be overly pessimistic. 

However, for the sake of argument, assume that both the random graph and intervention models were realistic. In practice, causal discovery is not often applied to a large number of different high-dimensional datasets with interventions, both due to a lack of interventional data and a lack of enough samples per intervention for reliability in high-dimensional settings. Hence, it is somewhat difficult to translate these forms of theoretical results into practical relevance. While the authors make a good effort to connect their theory with potential practical implications in their "Conclusion" section, I still feel that this flavor of work verges towards "theory for its own sake".

### Questions
**Intervention probability:** It was somewhat hard to track how the probability of having data from any particular single-node intervention plays a role in the bounds, e.g., this probability does not appear in Table 1. I see this dependence in other places (e.g. Theorem 8), but it would be nice to make this dependence more explicit and transparent throughout, as I would consider it one of the key quantities.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper considers the problem of  causal discovery under random interventions on large random graphs. The authors study how accurately causal directions can be identified when single-variable interventions are applied to networks drawn from Erd\H{o}s--R\'{e}nyi (ER) and Barab\'{a}si--Albert (BA) models. A key contribution is the  derivation of the deviation bounds for the false-negative rate (FNR)---the fraction of causal edges whose orientation is not correctly recovered. The paper shows that, under an $\varepsilon$-interventional faithfulness assumption (milder than $d$-separation based faithfulness), the FNR not only remains small in expectation but also concentrates sharply around its mean as the number of nodes grows. For dense ER graphs, the expected FNR decays as $\Theta(1/d)$ with concentration rate $O(d^{-1/2})$; for sparse ER graphs, it stabilizes at $O(1)$ with $O((\log d)/\sqrt{d})$ deviations. The paper also give bound for and BA (scale-free) graphs, though they are bit more technical to write. The paper also empirical illustrates the predicted concentration trends.

### Strengths
Causal structure discovery is a fundamental and challenging problem in machine learning and scientific inference. Any progress in understanding its limits and guarantees is valuable. Thus the broad topic of the  paper is well motivated. The paper is primarily theoretical, providing detailed proofs in the appendix and clearly stated assumptions. They also give a very good description of related work. While the main results are theoretical, the authors also include empirical experiments illustrating the concentration of the FNR on both ER and BA graphs. These simulations, though modest, help validate the theoretical findings.

### Weaknesses
The main weakness I find is that it is not clear whether causal discovery on random graphs corresponds to realistic application domains. Most real-world causal systems are I presume are more structured rather than randomly generated. Without concrete scenarios where random-graph analysis informs practice, the practical utility of these results remains uncertain. Thus the paper would be strengthened by examples of settings where such asymptotic guarantees could guide real causal-discovery in practice. While the paper appears technically sound, I am not able to enthusiastically recommend acceptance without a clearer bridge to practical utility of the modeling. That said, I am open to other expert reviewers’ perspectives, especially those with stronger background in causal-discovery theory that the paper is addressing.

### Questions
My main question is related to the weakness I mentioned. Can the authors provide motivating examples where causal-discovery performance on random-graph models offers insights applicable to real-world causal networks? Are there particular domains where random interventions approximate realistic experimental designs?

### Soundness
3

### Presentation
3

### Contribution
2
