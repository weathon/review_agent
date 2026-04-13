## Human Reviewer 1

### Summary
The authors tackle the challenge of systematically defining new Topological Deep Learning (TDL) architectures and to enlarge the accessibility of the latter to the broader community. The way they approach this endeavour is by (i) proposing a new class of TDL architectures that generalises previously proposed ones, and by (ii) implementing a software module that encapsulates architectural search over this class.

As for (i), the authors build upon the concepts of “strictly augmented Hasse Graphs” and “Per-rank neighborhoods”. The former ones are employed to model the structure of a combinatorial complex via an ensemble of augmented Hasse graphs, one for each neighbourhood. The latter ones prescribe defining a specific set of neighbourhoods for each rank. The authors propose GCCN as architectures which process ensembles of strictly augmented Hasse graphs with per-rank neighbourhoods with specific neural models and “synchronisation” components.

As for (ii), the module is called TopoTune and is a configuration-oriented component integrated with other TDL frameworks.

Experiments are conducted on graph datasets, lifted to either simplicial or cellular complexes. Results show that GCCNs can outperform standard architectures with a smaller number of parameters or lower computational cost.

### Strengths
- The submission tackles an interesting research topic in a timely manner.
- The implemented TopoTune module can be helpful to practitioners and researchers outside of the specific field of TDL.

### Weaknesses
- From the perspective of the framework generality, it is not clear how GCCNs would unlock new interesting operations or computational patterns.
    - Eq. 3 and 8 look particularly alike, and it is not evident what kind of advantage the latter brings. In particular, in Eq. 3, the message function $\psi$ can be specific to a particular neighbourhood (and rank), similarly to the neighbourhood message function $\omega$ in Eq. 8 — which, incidentally, is not rank specific.
    - Specific information about ranks and neighbourhoods could be specified by features akin to ”marks” over nodes and edges of an augmented Hasse graph, and a general enough neural architecture could then make use of these for neighbourhood and rank specific updates.
- Proposition 3 appears to be quite trivial given Proposition 1. What is it telling us in addition to that?
- It is not clear how the proposed contributions would help “democratising” TDL, as the authors claim. The proposed approach appears to significantly enlarge the hyper-parameter space by considering a plethora of possible architectural designs arising from the combination of neighbourhood and rank specific neural modules. Although TopoTune lowers the practical effort of searching over these spaces, these large parameter searches may still require large computational capabilities to be satisfactorily performed in a reasonable time frame.
- The value and/or interest of some experimental questions and emerging results is not clear.
    - “GCCNs outperform CCNNs”: It is not clear what the outperformance is due to when comparing to “standard” CCNNs, which could have, potentially, neighbourhood and rank-specific message functions. What is the take-home message for readers?
    - “GCCNs are smaller than CCNNs”: the authors do not explain why this is the case, and it is seemingly the first time this concept emerges in the manuscript
    - “GCCNs improve over existing CCNNs”: the results seem to be merely a matter of additional hyper-parameter search?
    - “Performance-cost tradeoff”: The authors highlight the reduced number of parameters of GCCN models, but they do not expand into how this actually translates into lower computational cost (e.g. because run-time experiments are not discussed in this section).
- Generally speaking, the manuscript would benefit from a clearer and more punctual presentation in regards to the motivations behind the proposed contribution and how these precisely address the research questions put forwards by the authors.

### Questions
- Can the authors expand on whether CCNNs can capture GCCNs? Are there functions expressed by GCCNs that cannot be expressed by CCNNs? If the two classes are equivalent, can the authors discuss more in detail what is the effective advantage of considering their proposed GCCN class?
- Can the authors better explain what were the research questions addressed in their experimental section and how their results contribute to answer them?
- Can the authors better discuss how TopoTune goes beyond merely a hyper-parameter search tool?

Please also see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 2

### Summary
In this work, the authors propose a generalization of Combinatorial Complex Neural Networks (CCNNs) called GCCNs and an accompanying software library called TopoTune, to generalize works on CCNNs into one computational framework and streamline the training and tuning of TDL architectures. Both theoretical and empirical results indicate that the proposed framework is indeed a useful generalization of previous efforts in TDL.

### Strengths
- The main strength of this work is that the authors are able to subsume TDL architectures under a single framework.
- The empirical results indicate to me that the framework matches existing works, thus validating the claim that the framework is indeed general.
- The framework allows the use of GNNs, which should bring the two fields closer together and have TDL research benefit from progress in GNNs.

### Weaknesses
- L458: The authors state that “GCCNs outperform CCNNs”. Out of the 8 presented datasets, I can only find two instances (NCI1, ZINC) where GCCNs actually perform better than the best CCNN baseline (accounting for one standard deviation). I could be convinced that the benefit of TopoTune is that one must only sweep over the GNN sub-modules to obtain an (at least) on-par model. However, this would still require some effort to find the best sub-module; see question 3 for more on this.
- In L468 and Figure 5, the authors discuss performance vs. number of parameters. However, I don not find this comparison convincing as a smaller number of parameters may not necessarily be more cost-efficient. Instead, I would like to see a comparison in terms of runtime and memory usage of the different models.
- Since the authors argue their approach to be superior to works on higher-order GNNs, a comparison of GCCNs and higher-order GNNs would be very useful. For example, PPGN++ (https://arxiv.org/abs/1905.11136), a higher-order GNN, performs much more on par with the best GCCN on ZINC than most CCNN baselines presented in the paper.

### Questions
- In the introduction you say “However, constrained by the pairwise nature of graphs, GNNs are limited in their ability to capture and model higher-order interactions […]”. I would expect that higher-order GNNs (https://arxiv.org/abs/1905.11136, https://arxiv.org/abs/1810.02244, https://arxiv.org/abs/1905.11136) are able to capture higher-order interactions. Could you elaborate on how TDL differs from higher-order GNNs?
- Related to the first question, in L88-L93 you mention the work of Jogl et al. (https://openreview.net/forum?id=HKUxAE-J6lq) on Cell Encodings, which is equivalent to using the standard Weisfeiler-Leman test on a transformed graph, but your argument for the shortcomings of this approach is not clear to me. In particular, you state that “However, although these architectures over the resulting graph-expanded representations are as expressive as their TDL counterparts […] the former are neither formally equivalent to nor a generalization of the latter”. What is “the former”? What is “the latter”? Assuming the former are Cell Encodings and the latter topological GNNs, why is it important that they are formally equivalent or one being a generalization of the other? Are they different in their runtime or memory requirements? Do we expect better learning behavior from TDL methods?
- As outlined in the weaknesses, in Table 1, only on two datasets GCCNs outperform the best CCNN from TopoBenchmarkX. Can you further elaborate on the benefits of TopoTune in this context?
- Related to the third question, can the authors provide an overview over the runtime and memory complexity of the compared CCNNs, as well as GCCNs, possibly in relation to the complexity the underlying GNN submodules?
- Am I correctly assuming that the ZINC dataset used in this work is the full ZINC dataset with 250K graphs, rather than the ZINC (12K) version frequently benchmarked in graph learning?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
The authors propose a general topological deep learning (TDL) architecture called Generalized Combinatorial Complex Network (GCCN). It aims to unify prior work on TDL under a common mathematical framework.
Additionally, the authors provide the TopoTune library, a reusable software implementation of the proposed GCCN method.
The experiments show that the flexibility of the GCCN framework allows it to match or outperform previously proposed TDL methods while, oftentimes, requiring fewer model parameters to do so.

### Strengths
First, the proposed GCCN architecture (while fairly straight-forward) provides a useful framework for describing a large variety of TDL methods and it enlarges the design space for such methods.
The experiments illustrate how this simplifies the optimization of TDL models and improving upon the state-of-the-art.
Additionally, the authors show that GCCN can match or even outperform previously proposed approaches while requiring fewer parameters to do so.

Second, the provided TopoTune implementation of GCCN integrates with existing GNN and TDL libraries.
This simplifies the exploration of novel TDL architectures and, as stated by the authors, could help accelerate research on TDL.
However, since I am not deeply familiar with the current literature on TDL and open problems, I can not confidently assess the relevance of this contribution.

Last, I want to highlight the presentation. The paper is well structured and written. The figures are of high quality and helpful.

### Weaknesses
In Section 4 the authors show a number of theoretical properties of their proposed GCCN framework.
While certainly desirable, the value of those properties is limited. 
As stated by the authors themselves in the proofs in the supplement, those properties are, for the most part, fairly straight-forward.
As far as I can tell, the GCCN framework is an intuitive generalization of prior work which only provides relatively small theoretical insights.
The overall value of the contribution therefore seems to depend on the relevance of the previously described strengths of the paper, in particular, on the relevance of the provided TopoTune implementation.
However, as mentioned, I cannot fully assess this aspect.
Thus, one potential general concern might be the overall relevance of the paper.

Apart from this point I have only minor suggestions for improvement:
1. I would have found a (brief) explanation of the evaluated types of combinatorial complexes (cellular vs simplicial) to be helpful.
2. There seem to be two small errors in the formal definitions in Section 2:
	- p. 3 (127): At $\mathcal{P}(S) \setminus \{\emptyset\}$ it should probably read $\mathcal{V}$ instead of $S$.
	- p. 3, eq. 2 (146): $\mathrm{rk}(\tau)$ after $\exists\ \delta$ should be probably $\mathrm{rk}(\delta)$.

### Questions
1. In Figure 5, it did not become entirely clear to me why the parameter size is reduced by changing the neighborhoods. I would expect that the total number of parameters of the GNN modules are independent of the specific types of neighborhood used. However, as shown in Figure 5 this does not appear to be the case. Can you elaborate on what exactly you mean by parameter size and how it relates the the choice of neighborhoods?
2. It is not clear to me how exactly the GCCN models are parameterized in the different experiments. In particular, which intra- and inter-neighborhood aggregators were used for the different experiments?
3. In the conclusion, you state that you hope that TopoTune might help "bridge the gap with other machine learning fields". Apart from the connection GNNs (and possibly Transformer models), are there any specific fields you envision that might profit from such a connection?

### Soundness
4

### Presentation
4

### Contribution
2

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper focuses on the topological deep learning (TDL) models in particular CCNNs and proposes a new powerful graph-based methodology for new TDL architectures, named GCCNs. The paper proves that GCCNs generalize and subsume CCNNs. The paper conducts extensive experiments and shows that the GCCN architectures achieve comparable performance with CCNNs. An efficient toolkit, TopoTune, is also introduced to accelerate the development of TDL models.

### Strengths
1. The paper proposes a new method to generalize any neural network to TDL architectures. 
2. The proposed GCCNs formally generalize CCNNs and have the same expressiveness as CCNNs. 
3. A new toolkit, TopoTune, has been developed to make it easy to design and implement GCCNs.

### Weaknesses
1. For node-level tasks, the paper only considers three very small datasets, which might limit the application of the method. 
2. The complexity analysis of the method is missing and the paper does not report any training time in the experiment. 
3. The experiment of "performance versus size" is not well analyzed especially for the graph-level datasets (i.e., PROTEINS, ZINC).

### Questions
1. Could the authors use larger node-level datasets for experiments?
2. What is the time complexity of the proposed GCCNs compared with CCNNs?
3. The GNN models perform very different results in Figure 5. More analysis is needed.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3