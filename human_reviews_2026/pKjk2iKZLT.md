# Scaling Higher-Order Graph Learning with Maximal Clique Complexes

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Graph neural networks (GNNs) are widely used for learning on graphs but are fundamentally limited to modeling pairwise relationships.
Topological models based on simplicial or cell complexes can capture higher-order structure and match or surpass the expressive power of the Weisfeiler–Leman (WL) test, but they are difficult to scale because they require constructing higher-order complexes.
In this paper, we ask how to retain the expressivity of cellular Weisfeiler networks (CWNs) while improving their scalability, and how to exploit cliques efficiently on large graphs. First, we introduce simplified and factored cellular Weisfeiler–Leman (sCWL and fCWL) tests, and show that they are as expressive as the original CWL test, while achieving better scalability properties. We then define the maximal clique complex, a cell complex whose higher-order cells are the maximal cliques of the graph, and apply the corresponding simplified and factored CWNs (sCWN and fCWN) on this structure, achieving improved time and memory complexity. To avoid explicit enumeration of all maximal cliques, we propose CliqueWalk, a biased random walk that samples (maximal) cliques and scales quasi-linearly with the number of nodes.
Combining maximal clique complexes with CliqueWalk yields scalable clique-based architectures that preserve CWL-level expressivity.
Experiments on node and graph classification benchmarks, including large-scale datasets, show that our models are competitive with or better than GNN and higher-order baselines, while substantially reducing computational and memory costs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces simplified Cellular Weisfeiler Netowrks (sCWN), a novel architecture designed to overcome the limitation of Graph Nueral Networks (GNNs) in capturing relationships that go beyond pairwise interaction. In terms of message passing mechanism, sCWN follow the framework of CW Networks (from [Bodnar et al., NeurIPS 2021]) but operate on a "maximal clique complex" which the authors define as a simplified complex containing only the nodes (as the 0-cells) and the maximal cliques of the graph.
As exhaustively identifying the maximal cliques in a graph is too expensive computationally, the authors introduce a sampling algorithm based on random walks.
The experimental section tests sCWN on both node and graph classification tasks and compares against popular GNNs and higher-order architectures. Results show that most benefits appear in datasets with specific structures, while on many standard benchmarks results are on par with simple GNNs. An ablation study is provided to show the scalability of the proposed model.

### Strengths
- The authors introduce a well motivated theoretical background for the method that highlight its theoretical expressivity
- The method puts focus on practical scalability of the method, which is good for real world applications

### Weaknesses
- The chosen benchmarks are a bit outdated. State-of-the-art GNN papers have typically moved to Open Graph Benchmark (OGB).
- Experimental results show that for many graphs the proposed model is actually less performant than simple baselines. It seems that the proposed method performs well only on specific datasets (seemingly the ones with the most cliques). A better analysis of the datasets (and their properties) in which the model really outperforms baselines would be useful.
- Unclear novelty of the algorithm proposed for maximal clique enumeration.

### Questions
- From the text it is not clear whether the authors used the standard splits for the benchmark datasets, could you please clarify if the splits were randomly sampled or if they followed the standard ones?
- Oversmoothing is a well known problem in GNNs (e.g., see "A Survey on Oversmoothing in Graph Neural Networks", Rusch et al.). If a graph has large maximal cliques, would the proposed method encourage oversmoothing as it adds more message passing among already fully connected nodes? Could this be the reason why it underperforms on some of the benchmarks?
- If a graph has very few maximal cliques, and maybe they are also far from each other, would the proposed method provide any benefit over a simple GNN? I think it could be useful to show some study of how maximal cliques are distributed in real world graphs
- The authors report that they initialize clique features using clique length; it would be useful to provide some ablations to justify this choice
- There are many algorithms for approximately enumerating cliques, like the ones cited at the beginning of page 6. Is there a specific reason why there is no comparison against them for the proposed method?
- Related to the question above, could the authors expand on the novelty of CliqueWalk? In particular what are the difference to existing algorithms, like the ones cited at the beginning of page 6?

### Soundness
3

### Presentation
3

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
The paper tackles the problem of scaling topological neural networks to larger complexes. The main idea is that of only considering maximal cliques as higher-order cells. The authors propose a specific, simplified architecture to work on the so-defined "maximal clique complexes", and advance some results in terms of expressive power and computational complexity. In addition to this, they propose a method to only sample a small number of maximal cliques and improve efficiency of the overall approach. Then, the authors turn to experimental analyses, run on node and graph-wise property prediction tasks. They additionally run some sensitivity analyses and ablations. The results are relatively mixed, especially in graph classification.

### Strengths
[S1] The main motivation behind the approach is clear enough and relevant.

### Weaknesses
[W1] There is quite a fundamental confusion between general cell complexes and complexes that can be obtained by considering maximal cliques only. Maximal cliques are constituted by *complete subgraphs*, thus do not include, e.g., induced cycles (rings). It does not seem to be the case from Figures 1 and 2, where the authors shade cells which clearly do not correspond to complete subgraphs. See, e.g., $\sigma_1$ in Fig. 2.

[W2] The relation between the original CWN network (Bodnar et al., 2021) and the CWN network discussed in the manuscript is not clear. Eq. 1 and 2 seem not to correspond to the original formulation, because, there, higher-order cells could also aggregate information from boundary, non-node cells. This seems not to be allowed in Eq. 1. Additionally, in the original formulation, higher-order cells can also aggregate from upper-adjacent ones, here in Eq. 2, it seems they can only aggregate from boundary cells. This is very confusing and puzzling. So they are supposed to be different models?

[W3] If the model only uses maximal cliques, it seems very unlikely it can generally match the expressive power of CWL from the original paper (Bodnar et al., 2021), as it could use, for example, ring-based lifting and distinguish the cliqueless graphs in the right most pair of figure 8 here (https://arxiv.org/pdf/2106.12575). 

[W4] The original justification behind the factored CWN models is lacking. What is its advantage w.r.t., say simplified CWNs?

[W5] Line 270 – the authors say their Clique Walk is inspired by existing clique sampling techniques, but what is exactly the relation? And what is its advantage over those? finding maximal cliques is a common well studied problem in computer science.

[W6] Some experimental details are rather uncommon. What does it mean to carve out an additional 20% of data as internal test set on top of 20% of validation (line 323)? And why not treating Batch Normalisation as just another hyperparameter (line 347)?

[W7] Results are not very conclusive.
- In node classification, it appears as standard graph models are often better or on par with the proposed approaches. fCWN also seems significantly stronger than standard CWN in some datasets, how is this expected? In the -School datasets some methods obtain suspiciously low results (see e.g. SAGEConv).
- In graph classification, results are generally low compared to baselines graph models. CWN with ring-based lifting is expected to work relatively better or on par there, especially in chemical tasks. What is the lifting procedure utilised in the CWN baseline?
- Figure 4a seems to suggest there is no real use-case where it is not convenient to use CWN in terms of acc-complexity tradeoff, but also, what is SCCN exactly?

### Questions
Please see weaknesses and, in particular, W2, W4, W5, W7.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Topological graph neural networks increase the expressivity of standard graph neural networks by considering cell complexes beyond edges and nodes. This paper suggests two improvements to existing topological networks: (a) It shows that certain aggregation operations can be removed without sacrificing expressive power, and as a result achieve a new mechanism with  reduced complexity (b) as finding maximal cliques in a graph is required for this method, and finding all such cliques is time consuming, they suggest a sampling method to replace the full enumeration. Empirical results seem mostly comparable to more expensive topological methods, which is encouraging.

### Strengths
* Overall, the paper is fun to read
* The theoretical results seem correct (I didn't check them very carefully) and can be helpful in reducing the complexity of using topological gnns.
*  Empirical results mostly support the conclusion that the reduction in complexity, obtained by both aggregation simplification and clique sampling, leads to comparable results

### Weaknesses
* I have some technical issues with the writing described in the questions section, but I believe they can be easily revised for the camera ready version

### Questions
* Practically, it seems that for standard node level tasks standard MPNN are comparable topological GNN, and the latter are more useful for datasets where the task clearly requires multi-agent interaction (the XXXschool tasks). Do you agree with this statement?

Some comments and questions on writing:
* I realize this may go back to the Bodnar paper, but in Section 3 you say a graph is a cell complex. This is strange to me, in your definitions, because a graph is a purely combinatorial object. What is the topology on the graph? When you say an edge {i,j} is a 1-cell, then by definition 1 this should mean it is isomorphic to [0,1]. How so? I would imagine describing a graph as an "abstract (combinatorial) simplicial complex"

*In definition 3: you talk about cell complexes. How is this test used for checking isomorphisms of graphs?

*In definition 7, in the words before the equatoin you say "and a cell c" but in the equations to my understanding you use sigma instead of c?

* equations 2,4,6 confused me, because you used AGG for an operation which people would usually call COMBINE or UPDATE. AGG, for AGGREGATION, usually means some aggregation over a multiset (which you use a $\oplus$ sign for) did I unserstand correctly? 

* In the first expression in equation 3, there is a some over $\sigma \ni i$ but inside the summation there is a $j$. What did you mean there?

* I didn't understand the subsection on CliqueWalk very well. In particular, in Proposition 13 I don't recall you defined what $\omega_{max}$ and $\omega(G)$ are.

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
2

### Summary
This paper addresses scalability challenges in higher-order graph learning by introducing maximal clique complexes and a sampling algorithm called CliqueWalk. The authors propose simplified cellular Weisfeiler networks (sCWN) that maintain CWL-level expressivity while reducing computational costs. The key innovation is using only maximal cliques rather than all cliques up to a fixed size, combined with efficient random walk sampling.

### Strengths
1. Strong theoretical contributions. 
2. The overall presentation is good and audiences can capture the key ideas of the work.
3. The experiments are comprehensive and the results are convincing.

### Weaknesses
1. Some format inconsistencies: 
    - Definitions does not share a same format, e.g., Definition 1 and Definition 2. 
    
2. What would happened if the $\omega_{max} < \omega(G)$? The walks won't be truly maximal in this case.

3. Can you provide examples where maximal clique CWL distinguishes graphs that WL cannot?

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
