# E(n) Equivariant Topological Neural Networks

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Graph neural networks excel at modeling pairwise interactions, but they cannot flexibly accommodate higher-order interactions and features. Topological deep learning (TDL) has emerged recently as a promising tool for addressing this issue. TDL enables the principled modeling of arbitrary multi-way, hierarchical higher-order interactions by operating on combinatorial topological spaces, such as simplicial or cell complexes, instead of graphs. However, little is known about how to leverage geometric features such as positions and velocities for TDL. This paper introduces E(n)-Equivariant Topological Neural Networks (ETNNs), which are E(n)-equivariant message-passing networks operating on combinatorial complexes, formal objects unifying graphs, hypergraphs, simplicial, path, and cell complexes. ETNNs incorporate geometric node features while respecting rotation, reflection, and translation equivariance. Moreover, being TDL models, ETNNs are natively ready for settings with heterogeneous interactions.  We provide a theoretical analysis to show the improved expressiveness of ETNNs over architectures for geometric graphs. We also show how E(n)-equivariant variants of TDL models can be directly derived from our framework. The broad applicability of ETNNs is demonstrated through two tasks of vastly different scales: i) molecular property prediction on the QM9 benchmark and ii) land-use regression for hyper-local estimation of air pollution with multi-resolution irregular geospatial data.  The results indicate that ETNNs are an effective tool for learning from diverse types of richly structured data, as they match or surpass SotA equivariant TDL models with a significantly smaller computational burden, thus highlighting the benefits of a principled geometric inductive bias. Our implementation of ETNNs can be found at https://github.com/NSAPH-Projects/topological-equivariant-networks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
**Method:** The paper introduces a framework, termed ETNN, for processing combinatorial complexes with geometric cell features in an $\mathrm{E}(n)$-equivariant way (explicitly introduces a method for $0$-cell geometric features and explains how to generalize to geometric features of cells of arbitrary rank). ETNNs are equivariant to the action of both $\mathrm{Sym}(\mathcal{X})$ (cell renaming) and $\mathrm{E}(n)$ isometries acting on the geometric features. The architecture builds on higher-order message-passing over CCs and has both invariant and equivariant versions. ETNNs generalizes previous work on $\mathrm{E}(n)$-equivariant graph neural networks, and $\mathrm{E}(n)$-equivariant cellular complex networks.

**Theory:** The expressivity of the proposed methods is evaluated based on its ability to distinguish $k$-hop distinct geometric graphs. The paper proves that in *most cases* ETNN is strictly more expressive than baseline geometric graph methods (i.e. message-passing architectures on geometric graphs that do not operate on higher-order cells).

**Experiments:** 
- **QM9:** The first application considered in the paper is QM9 molecular property prediction, where molecules are represented as combinatorial complexes with geometric features. ETNN variants demonstrate clear performance increase over standard geometric graph methods.
- **Air Pollution Benchmark:** The second application introduces a new benchmark for air pollution prediction. CCs are constructed from point measurements (0-cells), road measurements (1-cells), and census data (2-cells). 

**Notes.**
- Typo in equation (13); should be "$x \subset z$, and $y \subset z$".

### Strengths
- **Framework and Theoretical Contributions:**
  - The ETNN architecture introduced in the paper is a novel framework for $\mathrm{E}(n)$-equivariant processing of a wide class of topological data objects.
  - The framework elegantly unifies and generalizes existing architectures -- e.g. EGNN [1] and EMPSN [2].
  - The paper theoretically proves expressivity improvements over $\mathrm{E}(n)$-equivariant graph methods.

- **Empirical Results:**
  - ETNN variants achieve clear performance gains over standard $\mathrm{E}(n)$-equivariant graph methods for QM9 molecular property prediction tasks; ETNNs also improves upon EMPSNs while using less memory and having a faster runtime.

- **Applications and Dataset Contributions:**
  - The paper introduces a principled approach to modeling irregular multi-resolution geospatial data using combinatorial complexes.
  - The *air pollution downscaling* benchmark introduced in the paper is a new dataset for benchmarking TDL architectures. Additionally, the construction and analysis of geospatial combinatorial complexes is in itself an interesting contribution of the paper.

[1] Satorras et al. "E(n) equivariant graph neural networks", ICML 2021.

[2] Eijkelboom et al. "E(n) equivariant message passing simplicial networks", ICML 2023.

### Weaknesses
- **Novelty:** While the proposed method generalizes previous work to general combinatorial complexes, the architectural changes are incremental.
  
- **Theoretical analysis:**
  - The expressivity analysis focuses solely on distinguishing geometric *graphs*, despite defining expressive power in terms of separating more general non-isomorphic *CTSs*.
  - Proposition 2's statement is imprecise: the claim of improved expressivity "in most cases" lacks formal qualification, and the relationship between the expressivity gap and choice of lifting method needs clearer formulation and concrete characterization. A clearer restatement would be helpful.
  - No formal comparison to expressivity of other TDL methods (e.g. equivariant simplicial networks).

- **Empirical evaluation:**
   - Results lack statistical significance analysis (no standard deviations reported).
  - On the air pollution benchmark, improvements over the MLP baseline are modest (~1.5% RMSE reduction) and their statistical significance cannot be assessed due to unreported standard deviations.

### Questions
- Can the authors clarify the motivation for using "k-hop distinct" graphs as the primary measure of expressivity? Is it possible extend the expressivity results to CCs? E.g. is it possible to define a notion of "k-hop distinctness" for geometric combinatorial complexes and analyze ETNN's ability to distinguish between such complexes?
- How does ETNN's expressivity compare to that of simplicial complex networks in distinguishing non-isomorphic geometric graphs/complexes?
- In proposition 2, can you specify the conditions under which ETNNs are provably more expressive than EGNNs? 
- Could you include standard deviations for the air pollution benchmark and QM9 results to assess statistical significance?

### Soundness
3

### Presentation
2

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
This paper extends message passing supported on geometric graphs [3] and geometric simplicial complexes [2] to the more general setting of geometric combinatorial complexes. Theoretically, similar to how [1] establishes that higher-order message passing is more expressive than standard graph message passing, this paper demonstrates that the same holds true in the geometric setting. The paper also highlights the effectiveness of the proposed architecture through two novel real-world applications: (1) property prediction over geometric graphs representing molecules, where higher-order cells consist of rings and active groups, and (2)  a regression task over geospatial combinatorial complexes. The proposed architecture outperforms previous geometric graph architectures on these tasks.


[1] Cristian Bodnar, Fabrizio Frasca, Nina Otter, Yuguang Wang, Pietro Lio, Guido F Montufar, and Michael Bronstein. Weisfeiler and lehman go cellular: Cw networks. Advances in neural information processing systems, 34:2625–2640, 2021.

[2]  Floor Eijkelboom, Rob Hesselink, and Erik J Bekkers. E (n) equivariant message passing simplicial networks. In International Conference on Machine Learning, pages 9071–9081. PMLR, 2023.

[3] Vıctor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E (n) equivariant graph neural networks. In International conference on machine learning, pages 9323–9332. PMLR, 2021.

### Strengths
1. The paper offers a simple and straightforward way to adapt higher order message passing to respect $O(d)$ symmetries.

2. The experimental section effectively demonstrates the architecture's performance and introduces two new, interesting real-world TDL benchmarks, addressing a need highlighted in a recent position paper [1]. 



[1] Theodore Papamarkou, Tolga Birdal, Michael M Bronstein, Gunnar E Carlsson, Justin Curry, Yue Gao, Mustafa Hajij, Roland Kwitt, Pietro Lio, Paolo Di Lorenzo, et al. Position: Topological deep learning is the new frontier for relational learning. In Forty-first International Conference on Machine Learning, 2024.

### Weaknesses
1. The novelty of the proposed architecture and the theoretical section is somewhat limited. The architecture closely resembles those presented in [4] and [2]. Additionally, the theoretical contributions feel somewhat straightforward, offering limited new insights.

2. Building on the previous comment about the theoretical section, the paper lacks an analysis of how the choice of invariant function (see Equation (6)) affects the architecture's expressivity. It would be valuable to examine whether simpler, computationally efficient invariant functions could result in architectures that are  as expressive as architectures which use more complex alternatives, thereby guiding the choice of which invariant function to use in practice. Additionally, it would have been insightful to analyze whether there exist any natural geometric invariant functions that  geometric graph models are unable to compute while the proposed model succeeds in doing so.

3. The end of Proposition 2 states "In most of the cases, an ETNN is strictly more powerful than an EGNN". I tried finding the proof to this in appendix F and had a hard time. I think the authors refer to proposition 2 in appendix F but I'm not sure. A clearer framing of this result in the appendix, and perhaps an illustrative example or plot in the main body, would improve readability and support this claim.

4. The paper refers to the architecture proposed in [2] for geometric simplicial complexes. A comparison of the expressive power of ETNN using different lifts compared to this architecture would be interesting.

5. The paper [3] benchmarks higher-order message passing on several geometric benchmarks, using data augmentation to address $O(d)$ symmetries. An empirical comparison to this approach would provide valuable insight.

[1] Cristian Bodnar, Fabrizio Frasca, Nina Otter, Yuguang Wang, Pietro Lio, Guido F Montufar, and Michael Bronstein. Weisfeiler and lehman go cellular: Cw networks. Advances in neural information processing systems, 34:2625–2640, 2021.

[2]  Floor Eijkelboom, Rob Hesselink, and Erik J Bekkers. E (n) equivariant message passing simplicial networks. In International Conference on Machine Learning, pages 9071–9081. PMLR, 2023.

[3] Mustafa Hajij, Ghada Zamzmi, Theodore Papamarkou, Nina Miolane, Aldo Guzm´an-S´aenz, Karthikeyan Natesan Ramamurthy, Tolga Birdal, Tamal K Dey, Soham Mukherjee, Shreyas N Samaga, et al. Topological deep learning: Going beyond graph data. arXiv preprint arXiv:2206.00606, 2022.

[4] Vıctor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E (n) equivariant graph neural networks. In International conference on machine learning, pages 9323–9332. PMLR, 2021.

### Questions
see weaknesses.

### Soundness
3

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
3

### Summary
The authors propose an equivariant Topological Deep Learning framework that deals with geometric node features. The frameworks can be generalized to many topological domains including simplicial, cell, combinatorial, and path complexes. The authors provide theoretical analysis regarding his design choice as well as expressiveness of the proposed method. Lastly, the authors support his arguments via real-world datasets QM9 and his proposed benchmark - air pollution downscaling benchmark.

### Strengths
The authors add an important piece of work for the Topological Deep Learning (TDL) community as there is not much literature on Equivariant TDL. The work is well-formulated with clear motivations. The theoretical contributions are well-written. The paper is also self-contained and easy to follow, given the substantial explanations from related prior literature. The ablation studies are well-conducted via many synthetic graphs and additional information (hyperparameters, data statistics, etc.) are provided. The novel benchmark based on geospatial information is novel.

### Weaknesses
Novelty is the key disadvantage of the paper. It seems that the work just extends prior works on graphs to TDL. Even though the theoretical insights are important, yet they are mostly an extension from graphs. Another important weakness is scalability and practicability  of the problem. There are only two real-world datasets evaluated, and in both cases, graph sizes are small. Furthermore, the performance isn’t convincing given there are only minor improvements over the graph counterparts. The performance could be due to extra parameters for higher-order filters. Perhaps an experiment on model comparison with constraints on parameter budgets and an ablation study on higher-order features masked out are needed to prove your arguments. Lastly, even when the framework makes sense, it is unclear how we can obtain geometric features for higher-order cells. Please refer to question 6 for my concern.

### Questions
1. If I understand correctly, your argument on “heterogeneous interactions” focuses on different relationships between cells. Meanwhile, this property was actually mentioned in prior literature ([3] to name a few). I don’t think it is fair to claim that ETNNs are set up for this characteristic, but more like TDL in general already possesses this property.
2. The paper mentioned that Combinatorial Complex subsumes Path Complex. I believe there are two distinct lines of work regarding complexes arising from paths. One is path-based Combinatorial Complex [3], and another work is a simplified path complex [2] based on path complex [1]. [2] can’t be derived directly from [3] because [2] reserves the sequential information of paths. From my understanding, your work focuses more on path-based combinatorial complexes.
3. A relevant work [4] is not discussed in the paper.
4. It seems to me that $\mathcal{N}_{A, \text{max}}$ isn’t supported by any experiments. It would be better if the authors provide a set of experiments to support this design choice.
5. A minor subjective feedback on writing. I think it is better to have different notations for “containment” and “is a subset of” operation for clarity (Equation 6 and 7).
6. Regarding section I.1.2, it seems that 2-cell features do not encode any geometric features (velocity for example) but only invariant features. I think it is a missing piece to convince the audience that your framework can work with geometric features. Also, suppose that even when we have geometric features at node levels, it is unclear how to lift these features into higher-order spaces.

[1] Grigor’yan, A.A., Lin, Y., Muranov, Y.V. et al. Path Complexes and their Homologies. J Math Sci 248, 564–599 (2020).

[2] Truong, Q., & Chin, P. (2024). Weisfeiler and Lehman Go Paths: Learning Topological Features via Path Complexes. Proceedings of the AAAI Conference on Artificial Intelligence, 38(14).

[3] Hajij, M. et al. Topological Deep Learning: Going Beyond Graph Data (2023).

[4] Li, L. et al. Path Complex Neural Network for Molecular Property Prediction. ICML 2024 Workshop GRaM (2024).

### Soundness
2

### Presentation
4

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
This paper introduces an equivariant model within the framework of topological deep learning. The architecture generalizaes the equivariant graph neural network architecture from Santorras at al. from the setting of graphs to message passing over combinatorial complexes. Notably, this architecture allows for message passing with cells that have heterogeneous node feature over differening ranks. 

The authors first introduce the relevant theory of combinatorial complexes, topological deep learning and important noitions of equivariance.  In section 3, they introduce their architecture using the formalism from the previous section. The authors then discuss important theoretical aspects of their model; proving equivariance, comparing expressiveness with traditional equivariant, scalarization-based techniques and discussing computational complexity of their design. 

The authors discuss in section 4 two complementary examples of how to attain combinatorial complexes: firstly in molecular data and secondly in geospatial data. In section 5, the authors use said combinatorial complexes to benchmark their method against common architectures on the QM9 dataset. Secondly, the authors introduce a novel benchmark for predicting airpollution from annoted geospatial data.

### Strengths
Presentation, clarity and experimental transparency

I found the paper to be written clearly, and the mathematical statements and proofs were precise, well formulated and easy to follow. Further, the authors included a lot of helpful background about topological deep learning, and explained concepts clearly. 

One of the biggest strengths of this paper is the much appreciated transparency around testing in the appendix. This allowed a confident and clear understanding of what the authors actually did to obtain their results. I really appreciated the detailled description of cell features in section I.1.2 and I.2.2. The detailled ablation studies on the were also super helpful to understand and evaluate overall performance.

Novel geospatial task

I really like that the authors introduced a novel geometric prediction task into the literature. In reading the literature for this review, it struck me that many of the benchmarks for TDL were somewhat old and outdated. It seemed appropriate that the task featured integration of data over different dimensional regions (points, lines, cells) in a way that showcased the central feature of the paper — reconciling data with features on subspaces of differing dimension i.e. the ‘heterogeneous interactions’ promised in the abstract. 

Computational

I think one of the main strengths of the paper is the computational benefits, and I would personally focus on this more in the introduction. The tailored lifting of molecular graphs to the higher order CCs dramatically decreases the number of higher order cell, which is a problem that plagues many architectures in learning on simplicial/cellular complexes. The section on computational complexity also demonstrates this in a robust mathematical setting.

### Weaknesses
TDL vs GDL: novelty as a conceptual framework

For me personally I find it hard to understand the framing of this as a part of an entirely new conceptual field of topological deep learning beyond GNNs, and question the genuine novelty of papers like this. This is a concern I have of the field of TDL more generally, but I hope that the authors may be able to help clarify given their excellent communication skills demonstrated in the paper. 

Unless I’m mistaken, the basic content of proposition (1) is that an ETNN can be reformulated as an EGNN. This means that the main novelty is the clever choice of ‘lifting' the data into a certain graph and some delination of the learning based on ‘rank’. Indeed, even the proof of theorem (1) is basically a straightforward adaption of the corresponding result for EGNNs. I think the need to reformulate everything as an ETNN then show it’s equivalent to some specific EGNN needs more justification.

I still think that the experiments, results and set-up of the paper are interesting, but I personally get the sense that the ’topology’ part — and hence the novelty of these kinds of architectures — is overplayed a little.  One gets the impression from reading the introduction that there is some topological thing deep down somewhere that is making the difference in performance, whereas my feeling is that the inclusion of domain-specific data — functional groups, rings, etc. — along with the design of the graph is doing most of the work. 

On a similar theme, I don’t personally see a strong connection with the ‘lifted’ combinatorial complexes used in this paper and topology in the classical sense. It’s true that cell complexes and simplicial complexes also have the structure of a combinatorial complex (as in Appendix C), but these classical objects have additional connections to topology — i.e. they are stratifications of a genuine topological space, posses a homology theory, etc. 

QM7 Benchmarking comments

I find the approach to testing for this benchmark slightly unfair on other methods. Searching through so many iterations of hyperparameters, it seems inevitable that eventually some parameter set will outperform current methods at least once just on the balance of probabilities rather than model strength. ﻿Surely a fairer test would be run the competing methods along a similarly vast set of hyperparameters? At minimum, this experiment should be run multiple times, with the variance on results included, to test the robustness of these results.

Could the authors elaborate on why they chose to focus only on comparing the ETNN only with EGNN in the bottom line of Table 1? Upon first look, I assumed that ETNN outperformed SotA, but then on closer look saw that Equiformer outperformed ETNN in (almost)( every category. I would recommend highlighting the best performing method in bold in each column so that it’s more immediately clear that the while ETNN is improving on EGNN, it is still behind the SotA methods like DimeNet++ and Equiformer. My personal opinion is that the bottom line should be removed — the fact that it outperforms EGNNs is not surprising considering the fact that EGNNs are somehow a sub-architecture given the results of the expressiveness appendix.

### Questions
Experiments

Could you provide more details on how functional groups were defined or identified in the QM7 dataset? Were these pre-annotated in the data or determined through some other process?

It also seems like many of the best results are given in hyper parameter configurations without the use of higher order cells. Can the authors comment on this? Does this detract from the main thrust of the paper, which is the inclusion of higher order cells?

Novelty: TDL vs GDL

Turning my comments into the weaknesses section in a specific question: what do we gain from introducing the notion of a combinatorial complex? As above, (1) the procedures for turning graphs into combinatorial complexes in this paper are not objects commonly studied in topology — i.e. do not correspond to a simplicial or cellular complex — so do not have access to additional mathematical theory and (2) we could seemingly equally well-formulate this paper as an EGNN over a specific procedure for turning the data into a graph﻿ as per Prop. 1. Why not just say this is a specific example of a EGNN using the construction in Prop 1?

I am very open to hear the author’s thoughts/be corrected on this!

Virtual cells

The virtual cell seems like to dramatically increase performance. However, my understanding is that this totally change the connectivity of the underlying data, meaning that
everything is now connected (as per the comment on 389). ﻿What does this say about the actual importance of the graph structure? 

It seems to suggest the ’topology’ of the underlying molecular graph is basically ignored once the virtual cell is included. Why not just define the architecture as a family of fully connected EGNN over the nodes, edges and potentially the faces, given that the hyperparameters including the virtual cell almost always are superior?

On a related note: "it is clear how ETNN naturally handles heterogeneity, e.g., the same pair of bonds could be connected because part of the same functional group (up adjacency) and the virtual cell (max adjacency), but the two messages will be different across the neighborhood”. Unless I’ve missed something, how are we certain that the two types of propagation add to performance, rather than just overcomplicating the situtation?

Miscellaneous

Line 354: In the defintiion of geospatial CC, the mapping function s : X \to T takes cells to points in T, but the geographic space s(X) is now a subset of T rather than an element. Should s be mapping to the powerset of T, rather than T, if indeed it takes each cell to a subspace of T? 

In general, it would better to have a more concrete definition of what is meant by a ‘geospatial combinatorial complex’. I’m also unfamilar with the term polyline. 

Line 375: "This work is the first to explore combinatorial topological modeling of multi-resolution irregular geospatial data..” The claim that this is the first work to explore combinatorial topological modeling of multi-resolution irregular geospatial data seems overstated given existing literature in Topological Data Analysis applied to geospatial data. Could you clarify how your approach differs from or advances beyond these existing works?" https://arxiv.org/pdf/2104.00720, https://www.researchgate.net/publication/366891451_Topological_data_analysis_for_geographical_information_science_using_persistent_homology, https://pubmed.ncbi.nlm.nih.gov/26353267/

Table 2: It would be helpful to have the variance included over the multiple runs

Line 242: "geometric invariants should make use of the underlying topological structure.” The pairwise distance is more of a geometric structure than a topological one.

### Soundness
3

### Presentation
4

### Contribution
2
