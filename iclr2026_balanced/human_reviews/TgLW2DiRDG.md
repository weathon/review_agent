## Human Reviewer 1

### Summary
ReLU networks partition a $d$-dimensional input space into linear regions and therefore have an associated canonical *polyhedral complex* in which the $d$-cells correspond to individual linear regions and lower-dimensional cells arise from their iterative intersections.
By considering only top level polyhedra ($d$-cells) as nodes and their faces ($(d-1)$-cells) as edges, the polyhedral complex can be simplified into a *connectivity graph*, retaining information of neighoring linear regions.
The core contributions of the paper are to prove theoretical results: (1) an upper bound on the average number of neighbors of $k-cells$ in the complex, (2) a convergence result towards this upper bound for $d-cells$ with the increase of network size (number of neurons) and (3) lower and upper bounds on the diameter of the connectivity graph.
Empirical experiments confirm these results and, in addition, reveal an intriguing observation: linear regions that contain data points tend to be more connected i.e. have more neighboring regions, than those that do not.

### Strengths
The paper appears technically sound and is written in a clear, engaging style, with informative and visually appealing figures and illustrations (except Figure 6; see my specific comment below).
The overall structure and section breakdown are logical, and the paper’s main message is conveyed smoothly. Experimental details are provided in the appendix, and code availability supports reproducibility.

The work contributes new theoretical results in an active research area. It complements existing studies on counting linear regions by considering instead their adjacency relations.
This perspective has ties with other work branches such as tropical geometry of ReLU networks and error bounds.
While the immediate actionable implications of the theoretical results are not obvious, they could inform studies on expressivity or serve as guideline to use quantities like degree of the connectivity graph as features. 
Each claim is well supported both by proofs and empirical validation and the paper never overreaches.

While not new the polyhedra search method is interesting on its own.

### Weaknesses
The paper feels somewhat incomplete on one point.
The observation that polyhedra containing data tend to have, on average, more neighbors than those that do not is intriguing and calls for further exploration.
It would be helpful to confirm this finding with additional experiments to confirm it is not an artifact.
Intuitively, one might expect that regions containing data are more constrained and thus require higher expressivity and more complex boundaries to better fit the data.
While a theoretical explanation may be out of reach, additional empirical checks would, in my opinion, strengthen the paper, for instance:
- Check that this distribution difference emerge gradually during training by measuring the same statistics as in Fig. 6 at multiple training steps (including initialization).
- If region sizes can be estimated, are data-containing regions systematically smaller (indicating higher flexibility) ?
- Equivalently, is the spatial density of regions correlated with data density ? For example by taking a representative point by linear regions and measuring overlap with train/test data.
- Could the observed difference partly reflect the presence of peripheral, unbounded regions that would tend to be empty?

## Minor and additional feedback
- the links don't work
- line 76: unclear definition of the diameter
- typo missing "to" line 216
- line 825 lemma 3.4 does not exist in the paper
- strange syntax line 316
- vizualizing the lower bound or the number of regions $N_d(\mathcal C)$ would be nice on fig 5
- Fig 6 is a bit confusing: if the gray bars represent polyhedra traversed by BFS and colored bars are the subset of these polyhedra containing at least 1 data point, why is are gray bars below the colored ones on the right tail of plots (b) and (c) ?

### Questions
1. **Neighbor Count Distribution and Training Data** One of the most intriguing findings is the shift in the distribution of neighbor counts between regions that contain training data and those that do not. Do the authors have an intuition (or theoretical explanation in simple cases) for why regions containing data tend to have higher connectivity? (see also my comment above)
1. **Expressivity and Local Degree** To what extent could the number of neighbors in the connectivity graph be interpreted as a local measure of expressivity? On one hand more neighboring regions suggest finer partition of the input space, but on the other hand a simplicial partition of the input space would have maximal connectivity graph degree $d+1$ while still being able to be a good approximation (e.g. 3D meshes).
1. **Bounded vs unbounded regions** There is no discussion of bounded vs unbounded (peripheral) linear regions and while it does not change theoretical results they coexist implicitly throughout the paper. Moreover the unbounded regions can have less neighoring regions as you note in the proof of Theorem 3.4 in appendix for the base case, so I was curious to know if you had some general comment on this distinction and whether it is a useful framing.
1. **Diameter Bound and Implicit Dependence on Input Dimension** One of the hypothesis of theorem 3.6 is "as many first-layer neurons as input dimensions" and the derived upper bound is in $O(m^l)$. In the limit $d\mapsto \infty$ this implies $m=m_1\geq d$, which seems to be inconsistent with the claim of line 309 (also in the abstract) that the upper bound does not depend at all on the input dimension d. Could you clarify this point ?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper provides new results on the geometric characterisation of the polyhedral complex generated by the linear regions of ReLU neural networks.  The theoretical results provide, in some sense, a universality result in that the bound on the degree of the connectivity graph does not depend on depth and width of the network.  Experimental results are provided alongside.

### Strengths
The paper is very well written.  The presentation is very nice and the ideas and concepts are presented very well.

### Weaknesses
- Limitations of the work were not discussed.
- Although there is quite a number of references cited and a further detailed section on related work in the appendix, I still felt that there wasn't a strong connection between the contributions of the submission and previous work.  For example, the cited work by Zhang et al. on the tropical geometry of neural networks provides a theoretical upper bound on the number of linear regions.  I think that the submission could have done some computational comparisons on the proposed approach versus this existing well-known tropical geometry paper.

### Questions
What can be said about the computational complexity of the approach versus some recent work, e.g., by Stargalla et al. "The Computational Complexity of Counting Linear Regions in ReLU Neural Networks"?  Also to other existing work, e.g., by Lezeau et al. "Tropical Expressivity of Neural Networks" that gives the exact linear region count as well as discusses geometry of the linear regions?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper studies the geometry of piecewise-linear regions induced by ReLU networks, focusing on how these regions connect to each other rather than merely counting them. The authors define and analyze the connectivity graph, where nodes correspond to linear regions and edges represent shared facets between adjacent regions. They prove two main structural results. First, the average degree of this graph is at most $2d$, where $d$ is the input dimension, independent of network depth and width. Second, its diameter is upper-bounded by a quantity that depends only on the network’s width and depth, but not on input dimension. These bounds reveal a type of geometric regularity that holds across architectures.

On the algorithmic side, the paper proposes a method to enumerate regions and reconstruct the adjacency graph using linear programming feasibility checks. Experiments on small networks empirically validate the theoretical results and suggest that regions containing data points tend to have higher connectivity. Overall, the paper contributes new geometric understanding of ReLU networks, extending previous work.

### Strengths
The paper provides clear and rigorous geometric insights into the structure of ReLU networks by analyzing the connectivity of linear regions. The results are conceptually interesting, mathematically clean, and supported by both proofs and experiments. They are well presented and also lay a solid foundation for future work.

### Weaknesses
- The paper should explicitly state the basic assumption adapted from Masden (2022). Although it holds for a full-measure set, combinatorially it is arguably a strong restriction.  
- While the paper gives helpful intuition and pictorial explanations, including a rigorous definition of the canonical polyhedral complex (at least in the appendix) would improve clarity.  
- The results are not particularly deep; nonetheless, they are conceptually interesting, clean, and provide meaningful insights into neural network structure.  

Minor: 
- In the proof of Theorem 3.1, Lemma 3.4 is referenced but does not appear in the paper.

### Questions
- Is the connectivity graph the 1-skeleton of the sign sequence complex, which under the weight assumptions forms a cubical complex? Would your results hold more generally for graphs that are the 1-skeleton of a cubical complex? Have you checked the literature for related results on 1-skeleta of cubical complexes?  
- How exactly do you define a ``face''? It seems that you mean a codimension-1 face, which is usually called a facet.  
- Can you say anything about the non-generic case? Understanding the class of possible connectivity graphs for networks with 2 hidden layers would be very interesting.  
- The observation that data points lie in cells with more neighbors is interesting. Do you have any idea why this might be the case? It has also been observed that data points tend not to lie close to the boundaries of cells after successful training (e.g., https://openreview.net/forum?id=NpufNsg1FP). Do you see any connection between these observations?

### Soundness
4

### Presentation
4

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper establishes fundamental results about the geometry of ReLU networks. It shows that the number of neighbors a linear region has is always upper bounded by 2 times the input dimension of the network. This is somewhat surprising because using a deep architecture can sometimes produce an exponential number of these regions, yet the configuration still obeys this graph property. The other main result is that the diameter of the connectivity graph has upper bounds that are independent of the input dimension, which is again surprising given the role the input dimension played in the first result.

### Strengths
1. Excellent presentation. This paper has great figures and clearly and concisely explains a lot of mathematical concepts.

2. Theoretical results were thought-provoking and intriguing, providing the reader with new ways to think about ReLU configurations.
 
3. Empirical results are well connected to the paper and make sense in light of the theory.

4. Theorem 3.6’s reasoning about connecting regions with lines does a good job explaining why the graph diameter really isn’t about the input dimension.

5. I was wondering about the connection with dividing space into cubes when I read the abstract and saw connectivity = 2d, it was cool to see that connection in Theorem 3.5

### Weaknesses
1. Line 80, missing the word ‘and’ 
2. 147, 169 ‘exactly’ should be ‘at least’ since a line can lie in the intersection of as many bent hyperplanes as it wants. It seems that most of the time we’re supposed to assume the BHs are in generic configuration though, mentioning that sooner could also fix it
3. 216 corresponding ‘to’ a network. ‘To’ is missing.
4. 269 ‘two d-ells’ typo
5. 316 ‘On a polyhedral region where the network’s behavior is affine.’ 
6. 408 complexes complexes
7. 825 appendix, is Lemma 3.4 supposed to be Theorem 3.4?
8. 839 ‘at once of the d-cells’ 
9. 863 did you mean to say each neuron adds d times more d-1 cells than d cells?
10. 936 contain should be contains

For Theorem 3.5, if the subgraph from the vertex cut has minimum degree d-1, doesn’t that mean that the number of new edges is only at least (d-1)/2 times the number of new vertices rather than (d-1)? I think this might affect the correctness of this proof. It would seem that if average degree (d-1) rather than 2(d-1) were all it took for a new (d-1) complex to contribute d times as many (d-1)-cells as d-cells to a d complex, that that would break the induction in theorem 3.4.

### Questions
One thing that confuses me about the proof of Theorem 3.4 (appendix A) is that it says it inducts over the ‘input dimension’ of the network, rather than the dimension of the ReLU complex. It establishes the base case as d=1, where all the 1-cells lie along a single line with two unbounded end segments, which is true of input dimension 1, but not generally true of an arbitrary 1-d ReLU complex, which could have multiple loops (such as the blue line in Figure 3b). It seems like it ought to be inducting over the ReLU complex dimension, since inductively adding neurons would need to check that the intersections of the existing BHs onto the new BH obey (d-2 cells) < (d-1)*(d-1 cells), which is different from having one less input dimension. Are you maybe using ‘input dimension’ to describe restriction onto a d-1 dimensional set of input space where the new neuron’s pre-activation output is 0? I guess I’m confused because the body of the paper makes the point that the dimension of the ReLU complex is more about the highest dimensional cells it has rather than the space it’s embedded in, but the induction seems to be over the embedding space. Some clarification here would be helpful.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4