# When GNNs meet symmetry in ILPs: an orbit-based feature augmentation approach

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 5, 6, 5

## Abstract
A common characteristic in integer linear programs (ILPs) is symmetry, allowing variables to be permuted without altering the underlying problem structure. Recently, GNNs have emerged as a promising approach for solving ILPs. 
However, a significant challenge arises when applying GNNs to ILPs with symmetry: classic GNN architectures struggle to differentiate between symmetric variables, which limits their predictive accuracy. In this work, we investigate the properties of permutation equivalence and invariance in GNNs, particularly in relation to the inherent symmetry of ILP formulations. We reveal that the interaction between these two factors contributes to the difficulty of distinguishing between symmetric variables.
To address this challenge, we explore the potential of feature augmentation and propose several guiding principles for constructing augmented features. Building on these principles, we develop an orbit-based augmentation scheme that first groups symmetric variables and then samples augmented features for each group from a discrete uniform distribution. Empirical results demonstrate that our proposed approach significantly enhances both training efficiency and predictive performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors propose a novel feature augmentation method for ILP's with symmetries that are described by bipartite graphs for solving with GNN's. The augmentations obey some important symmetry properties but are also more parsimonious than existing methods. Empirical results suggest that these augmentations help the GNN's break the symmetry better than competing methods.

### Strengths
The main idea in Section 4.2 is explained well. The numerical results are decent and convey the practical benefits of the method. The introduction is also nicely written.

### Weaknesses
While the numerical results do suggest that the method helps break symmetries better than others, it is hard to tell if this makes a difference in the end result (objective of rounded solution). I think such a comparison should be added (either way). A more complete description or explanation of the augmentation procedure in general could be useful for those not in the field. Some minor improvement for minor writing weaknesses are suggested below.

### Questions
Figure 1 appears too soon, before the map between ILP and graphs...
I don't think you need to define a permutation.
I don't understand the h_i^v definitions in 2.3, and how they're related to c, b, and A in the original problem? Isn't c_i a scalar? How is this a feature vector? Can you please clarify the entries of \mathcal{A} and how they relate to the bipartite graph? (Are you using the columns and rows of \mathcal{A} as the feature vectors for the nodes V and W? If this is true, I think it should be explicitly stated somewhere.) 
I think you mean {V,W,E} as the bipartite graph? (You have C in one place and W in another)
In figure 1, I don't yet understand why the GNN must make x1 and x2 equal? How was the integer constraint enforced?
The notation in proposition 1 seems a bit overkill - this seems like a simple & intuitive result, so perhaps this proof can be made simpler. 
Am I missing an additional assumption of Proposition 1 that the initial feature embedding of points in the same orbit are equal? (Otherwise this proposition extends to the proposal in section 4?)
Corollary 1 seems overly general (and not quite correct). I think you mean that it cannot ALWAYS predict the optimal solution. Of course there are instances that it can solve correctly (like min 0 s.t. [no constraints])
The number of instances in the problems 5.1 is quite low, isn't it?
How is the GNN enforcing integer constraints?
Do you have any other metrics of comparison? Don't you also care if the rounded solution is any good? (It's OK if it's not -- I'm just curious.)
Minor: "By some mathematical derivations" is an awkward phrase; typo "cloest" on page 9.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes to augment the vertice features in ILP graph representation based on orbit of the symmetry group. Some theory is provided and numerical results are conducted with the comparison with the random feature technique (Chen et al. 2022) and the positional ID technique (Han et al. 2023).

### Strengths
1. The paper is in general well-written and is easy to follow.

2. The idea of augmenting features based on orbits is reasonable. I agree with the authors that only vertices in the same orbit need to be separated with augmented features, and the cardinality of the augmented feature space is much smaller than $n!$ if the number of orbits is much larger than one.

3. The reported numerical results look better than baseline methods.

### Weaknesses
1. There is no comparison with conventional methods based on symmetry group and orbits, such as Ostrowski et al. (2011) cited by the authors. In addition to Ostrowski et al. (2011), there should actually be a much richer literature in this direction.

2. There is no report on the cost of computing the symmetry group. I expect to see a trade-off between the size of the symmetry group and the improvement from "no augmentation".

### Questions
None

### Soundness
2

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
5

### Summary
This paper improves a weakness of graph network approaches for predicting the solutions of integer linear programs (ILPs) with symmetric variables. Because graph networks are permutation equivariant, they cannot distinguish between exchangeable variables in the ILPs (or more concretely, variables such that, when permuted, the cost constraint is still satisfied and the objective is unchanged). The authors use feature augmentation to break the symmetries between these equivalent variables, specifically emphasizing distinguishability, “isomorphic consistency”, and “augmentation parsimony”. Past works have used feature augmentation to break these symmetries, but without adhering to these principles. They demonstrate the effectiveness of their techniques relative to alternatives on solving synthetic ILPs, with training data generated by a classical solver.

### Strengths
This paper addresses a meaningful issue (symmetry-breaking in integer linear programs) in a new way (using inputs to GNNs that break only the required orbit symmetries, in accordance with the three defined desiderata). Although I believe these two pieces are individually not new (see “weaknesses”), their combination is. The “Orbit+” approach is also a novel way of enhancing “augmentation parsimony”. The paper is generally clear, and the writing style and notations are both enjoyable to read. Their experimental results on the chosen tasks beat the chosen ML baselines. Although I am not convinced by the necessity of isomorphic consistency for most applications, in which a logical loss function can be chosen (which is unaffected by swapping equivalent nodes), the use of SymILO to enforce it by adjusting training labels is also new.

### Weaknesses
I think the biggest weakness of this paper is that, in short, many of its central ideas have already been introduced and (in some cases) thoroughly explored in papers of which the authors seem unaware. (This is understandable, given that they do not appear in the ILP literature and use different terminology to describe the problem, but nonetheless they exist — I hope the authors may be inspired by the perspectives of these papers, and can also articulate the novelty of their work relative to them.) This line of work (see the references below; although not the earliest, [2] or [4] may be the most accessible starting points) goes by the name “symmetry-breaking”, and articulates the precise issue that the authors encounter for ILPs, but in a much more general way, for all group equivariant networks. The principles of distinguishability and augmentation parsimony are explored under different names, e.g. in [3]. There is a related line of work on breaking symmetries of sets, termed “multiset equivariance” [5]. Works such as these and [4] make clear subtleties of the problem that aren’t discussed in this paper, such as the difference between the graph automorphism group and the node orbits, and articulate methods for addressing the equivalent nodes of ILPs in ways that subsume the method presented here.

I believe that orbit-equivariant graph neural networks [6] are also a slightly more fleshed out version of the “Orbit” approach. 

As noted under questions, I also find the discussion of isomorphic consistency confusing, as it is (assuming I understand correctly) not well-motivated under orbit-invariant loss functions, and importantly, not necessarily even possible to achieve. Is “relaxed equivariance” [2] more suitable? 

Finally, as also noted under questions, there seem to be weaknesses with the experiments — namely, the choice of loss function, and the premise/lack of comparison to non-ML baselines. 

References:
1. Smidt, T. E., Geiger, M., and Miller, B. K. Finding symmetry breaking order parameters with euclidean neural networks. Phys. Rev. Research, 3: L012002, Jan 2021. doi: 10.1103/PhysRevResearch
2. Kaba, S.-O. and Ravanbakhsh, S. Symmetry breaking and equivariant neural networks. In Symmetry and Geometry in Neural Representations Workshop, NeurIPS, 2023.
3. Xie, Y. and Smidt, T. Equivariant symmetry breaking sets. TMLR 2024.
4. Hannah Lawrence, Vasco Portilheiro, Yan Zhang, and Sekou-Oumar Kaba. Improving equivariant networks with probabilistic symmetry breaking. In ICML 2024 Workshop on Geometry-grounded Representation Learning and Generative Modeling, 2024.
5. Zhang, Y., Zhang, D. W., Lacoste-Julien, S., Burghouts, G. J., and Snoek, C. G. M. Multiset-equivariant set prediction with approximate implicit differentiation. In International Conference on Learning Representations, 2022.
6. Morris, M., Grau, B. C., & Horrocks, I. (2024). Orbit-equivariant graph neural networks. ICLR 2024.

### Questions
Questions:
1. *Value of ML versus classical ILP method*: The authors motivate the use of GNNs for ILPs by referencing recent works which use GNNs as part of the solution process, e.g. as an oracle in branch-and-bound algorithms or predicting initial solutions. However, the experiments they run directly output solutions of the input ILPs, and there is no comparison to classical ILP solvers (eg int terms of accuracy), because the training data itself is generated by the ILP solver SCIP. Thus, my question is: what value does machine learning add to these problems? For example, are the trained models’ forward passes on test data supposed to be faster than the solver? (If so, does this take into account the time to detect the symmetry of the input ILP?) Does the noted symmetry-breaking problem arise when ML is used to predict branching decisions or node selections?
2. *Choice of loss function in experiments*: why use a loss function that tries to learn the exact solution instance, when there are several degenerate, “equally good” solutions? It seems far more natural to use the objective function $c^Tx$ of the ILP itself, or the fraction of instances that satisfy the constraint $Ax\leq b$, over $x$ from the test set.
3. *Motivation for “isomorphic consistency” principle*: The isomorphic consistency principle (which is dataset-dependent) strikes me as odd — the whole problem explored in this paper is that it is not even **possible** to satisfy while outputting optimal solutions, except for on certain training sets. Can the authors elaborate on this? IN particular, if one uses a loss function like the one suggested above, I can’t tell what the value of isomorphic consistency is.
4. *Practicalities of the experiments*: Line 468 states that multiple augmentations need to be drawn per sample. Is this done for baselines too? Also, is there an ablation result for the importance of using SymILO to enforce isomorphic consistency? Does this remain under the choice of loss function suggested above?
5. *Minor clarification about formulation symmetry definition*: As defined around line 118, a formulation symmetry “retains the description $Ax \leq b$”. One way of achieving this is if $Ag(x)=Ax$, indeed this is what I would have expected as the definition. Is this more accurate?
In the methodology paragraph, around lines 228-230, the paper says “…the approach did not exploit the underlying symmetry properties, leading to suboptimal performance on ILPs with strong symmetries”. What does this mean? The paper would be improved by making this claim precise, and including references to evidence (eg a specific result in the cited paper).
6. Why not use MIPLIB, the dataset cited in the first paragraph of the paper, in the experiments? For the datasets used in experiments, what fraction of inputs exhibit no symmetric variables at all?

Typos:
As I read through the paper, I noticed some minor typos. These did not affect my evaluation of the paper, but I’m noting them here in case it’s useful to the authors.
* line 74: “fully use of” —> “full use of”
* Subtitle on line 191: “issues occur”, not “occurred”
* Line 207: “correspond”, not “corresponds”
* Line 218: “oribit”
* Line 219: “have distinct values”, not “has distinct values”
* Line 445: “cloest”

Minor notation/writing notes:
* Definition 3 should define that $I \in I^n$ and should also define $\mathcal{O}_i$. 
* The standard term for Assumption 1 is permutation equivariance, not equivalence
* It would be good to explain lines 322-323 (“Accordingly…other orbits”) in mathematically precise language

On the writing side, I would also recommend making the “Motivations” paragraph more concrete.

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
2

### Summary
The authors study the problem of solving Integer Linear Programs (ILPs) with symmetry among variables. They first show that if a permutation from the set of all variable permutations is a formulation symmetry of the ILP, then under an assumption of permutation equivalence and invariance for the GNN, the network cannot predict the optimal solution of the ILP.

To address this, they propose a feature augmentation algorithm that assigns unique augmented features to each orbit, sampling a distinct feature value within an orbit without replacement. They compare their methods against previously proposed augmentation schemes empirically and based on some principles for three ILP benchmark problems.

### Strengths
Their method appears to perform better in terms of their proposed metric than existing methods on certain tasks. They also introduce the problem clearly, making it easily understandable for someone outside the field, while establishing a good motivation through negative results about GNNs and formulation symmetries.

### Weaknesses
They don't discuss any limitation of their orbit-based augmentation, making their method's application scope appear narrow restricting to ILPs with formulation symmetry. 

Additionally, the approach relies on detecting symmetry groups and orbits, which as they note may be computationally expensive.  It would also be interesting to see how their method performs for different evaluation metrics (maybe beyond $\ell_1$ distances). 

As this area is new to me, their contributions do not seem sufficiently novel in terms of the algorithm, and the experiments provided also seem limited and hence I recommend a reject but with a confidence score of 2.

### Questions
Could the authors comment on the computational complexity of detecting symmetries and how the established methods help handle it?

### Soundness
2

### Presentation
2

### Contribution
2
