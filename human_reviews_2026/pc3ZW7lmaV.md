# How hard is learning to cut? Trade-offs and sample complexity

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
In the recent years, branch-and-cut algorithms  have been the target of data-driven approaches designed  to enhance the decision making in different phases of the algorithm such as branching, or the choice of cutting planes (cuts). In particular, for cutting plane selection two score functions have been proposed in the literature to evaluate the quality of a cut: branch-and-cut tree size and gap closed.  In this paper, we present new sample complexity lower bounds, valid for both scores. We show that for a wide family of classes $\mathcal{F}$ that maps an instance to a cut, learning over an unknown distribution of the instances to minimize those scores requires at least (up to multiplicative constants) as many samples as learning from the same class function $\mathcal{F}$ any generic target function (using square loss). Our results also extend to the case of learning from a restricted set of cuts, namely those from the Simplex tableau. To the best of our knowledge, these  constitute the first lower bounds for the learning-to-cut framework. We compare our bounds to known upper bounds in the case of neural networks and show they are nearly tight, suggesting that both scores (gap closed and tree size) are of comparable difficulty from a learning standpoint. Guided by this insight, we provide empirical evidence -- by using a  graph neural network cut selection evaluated on various integer programming problems -- that gap closed is a practical and effective proxy for minimizing the tree size. Although the gap closed score has been extensively used in the integer programming literature, this is the first principled analysis discussing both scores simultaneously both theoretically and computationally.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper provides the first sample-complexity lower bounds for the “learning-to-cut” framework in branch-and-cut MILP solvers. It proves that, for neural-network policies mapping an ILP instance to a Chvátal–Gomory cut, minimising either tree-size or gap-closed requires  
Ω(VCdim(F)/ε) samples, i.e. as many samples as learning any generic real-valued target with squared loss. The bounds are shown tight up to log factors for ReLU nets. Complementary experiments on four NP-hard problems indicate that a GNN trained on the cheap “gap-closed” proxy achieves tree-size reductions close to the expensive oracle, corroborating the theoretical insight that the two scores are equally learnable.

### Strengths
1. Theoretically pioneering: First distribution-free lower bounds for learning-to-cut; closes the loop with recent upper bounds [Balcan et al. 2021, Cheng et al. 2024].  
2. Generality: Results hold for any encoder–network family that satisfies minimal closure assumptions; not restricted to CG or tableau cuts.  
3. Tightness: Lower bound Ω(WL log(W/L)/ε) matches the best-known upper bound Õ(WL/ε²) for ReLU nets, showing no exponential gap.  
4. Empirical validation: Carefully designed GNN experiments demonstrate that gap-closed is a practical surrogate for tree-size, bridging theory–practice.  
5. Clarity: Proof roadmap and supplementary document are rigorous and well written.

### Weaknesses
1. Restricted concept class: The lower bounds are proved only for single-cut selection, whereas modern solvers add batches (rounds) of cuts. It remains unclear whether the lower bound still holds when the learner outputs a *set* of cuts.
2. CG-only cutting regime: The lower bounds rely exclusively on Chvátal–Gomory (CG) cuts. Extending the results to more general cut families such as split cuts, GMI cuts, or knapsack cuts would significantly strengthen the theoretical claims.
3. Instance distribution: overly general setting: The lower bounds are worst-case over all possible distributions, without considering structured or parametric families (e.g., IPs generated from stochastic block models or Gaussian-distributed `A, b`). This may lead to overly pessimistic bounds.
4. No information-theoretic upper bound for gap-closed: While the authors conjecture that only cuts separating the fractional solution matter, they do not provide a distribution-dependent upper bound based on this insight. Thus, the sample complexity of gap-closed remains only partially characterized.
5. Limited empirical scope: Experiments are conducted only on small synthetic instances (≤ 50 variables). There is no validation on medium-scale realistic benchmarks such as the “easy” subset of MIPLIB 2017, leaving open whether the proxy quality degrades on real-world IPs.
6. Missing baselines: The empirical evaluation lacks comparison with recent learning-based cut selectors, such as LLM-guided methods (LIFT-FE, TransGPT-FE) or imitation learning with tree-size experts (Paulus et al., 2022), making it hard to assess the relative strength of the proposed approach.

### Questions
1. Do the lower bounds hold for batch cut selection?  
   If the learner outputs a set of cuts instead of a single one, does the sample complexity lower bound still apply? Is a new theoretical framework needed?
2. Can the bounds be extended to general cut families?  
   Is it possible to generalize the lower bounds beyond CG cuts to split cuts, GMI cuts, or other stronger cut classes?
3. Can tighter bounds be obtained under structured distributions?  
   If instances are drawn from structured generative models (e.g., stochastic block-model IPs or Gaussian `A, b`), can we prove tighter sample complexity bounds?
4. Can we derive a distribution-dependent upper bound for gap-closed?  
   Can the conjecture — that only cuts separating the fractional solution matter — be leveraged to derive a tighter, distribution-dependent upper bound on the sample complexity of learning with gap-closed?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies lower bounds on the sample complexity of learning cut policies for Branch and Cut solvers for integer linear programs.
A key step in the Branch and Cut framework is adding a cutting plane, which is an additional constraint that tighten the constraint set without eliminating any integral solutions.
There are many valid cutting planes to choose from, so traditional B&C implementations have a collection of hand-crafted heuristics designed to choose cuts that lead to fast solution times.
Recently a line of work has explored the idea of learning cutting plane selection policies from data: given a training collection of ILP instances sampled from some distribution, each instance is solved and the resulting B&C trees are used to train cut selection policies that optimize a utility metric attempting to capture the running time required to solve the problem after the cut is applied.
The two most commonly considered metrics are tree-size (the actual size of the resulting B&C tree, which is strongly predictive of the runtime), and the gap-closed metric, which measures the reduction in the optimality gap after introducing the cut.
Some prior work has provided upper bounds on the sample complexity of learning such cutting policies when the utility metric is the size of the resulting tree (but the authors claim that it can be generalized to the gap-reduction metric straight forwardly).

This paper has two main high-level contributions:
1. The authors derive sample complexity lower bounds for learning cut selection policies from data under fairly realistic assumptions. In particular, together with the upper bounds from prior work, they establish that the sample complexity is comparable for both the the gap-closed and tree-size metrics.
2. The authors empirically find that the gap-closed metric is a suitable proxy for the tree-size metric. This is useful because the gap-closed metric is much more computationally efficient to evaluate when generating training data.

Taken together, these results support the use of gap-closed for learning cut selection policies. It is not harder to learn than the tree-size metric from a sample complexity point of view, still provides a reasonable proxy for solution time, and is easier to calculate.

### Strengths
The problem of learning cut selection policies for the branch and cut framework is interesting and very practical, and this paper provides the first lower bounds on sample complexity for this problem.
The findings that the gap-reduction metric is a suitable approximation to tree-size also useful.
Overall I found the paper well written and feel like the main contributions were clearly communicated.

### Weaknesses
I would have liked for more of the key sample complexity lower bound argument to be sketched in the main body of the paper.
For example, in the discussion of Proposition 3.4, it is stated that the approach ignores $n \times m + m$ of the inputs, and I was curious to understand why.
There are also a number of minor typos throughout the paper, and a few significant ones in the experimental results (unless I have misunderstood something).

I think at the beginning of Section 2.2, the authors could include a little bit more detail about the notation. At first I had some trouble identifying the role played by the functions $f$ and $h$ in equation (2).

A few of the minor typos I noticed:
- Line 120: "Ou results" -> "Our results"
- Line 169: "that can derived at any iteration" -> "that can be..."?
- Line 181: Maybe "so this motivates the idea of learning such a score"?

### Questions
- On line 308, should it say "returns a vector in $[0,1]^m$" instead of $[0-1]^m$?
- On line 309, should it say that the function is "continuous at 1/2"?
- Table 2 seems to be missing some results for Facility Location compared to Table 3 in the appendix (i.e., the first column is all dashes instead of having an entry for GNN).
- In both Table 2 and Table 3, should the second column be "gap closed" and the third column be "B&C tree". The titles aren't consistent, but if I understand what is being reported, it seems like the dashes should always appear in the "gap closed" column.
- In Tables 2 and 3, since only GNN has a gap-closed variant, I wonder if it would make sense to have a single column and just include two versions of GNN: "GNN-Tree-Size" and "GNN-Gap-Closed".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides the first sample complexity lower bounds for learning to select cutting planes in integer programming. The bounds apply to two performance scores: branch-and-cut tree size and gap closed. The authors show these lower bounds are nearly tight with known upper bounds for neural networks, indicating both scores have comparable learning difficulty. Empirical results demonstrate that gap closed serves as a practical proxy for tree size reduction when training graph neural networks on tableau Chvatal-Gomory cuts.

### Strengths
1. First sample complexity lower bounds for learning-to-cut, establishing theoretical foundations
2. Lower bounds hold for wide function classes and are nearly tight with upper bounds up to logarithmic factors
3. Theoretical equivalence between scores provides formal justification for using gap closed as proxy
4. Proof construction for shattering ILP instances is non-trivial and well-executed
5. Experiments directly test the core thesis about proxy effectiveness

### Weaknesses
1. Gap between $\Omega(1/\epsilon)$ lower bound and $\Omega(1/\epsilon^2)$ upper bound for tableau case
2. Assumptions 1 and 2 restrict generality of theoretical claims
3. Empirical validation uses small-scale problems in controlled environment and restricted solver configuration
4. Performance varies: on Facility Location, Efficacy heuristic (123.63) outperforms GNN (134.61)
5. Experimental setup disables key solver components (presolve, heuristics, default cuts)
6. Theoretical analysis confined to Chvatal-Gomory cuts from simplex tableau
7. Learned policy selects single cut at root node, not dynamic cut selection
8. Worst-case bounds do not leverage structure in real-world MILP distributions
9. Computational cost of obtaining training samples not addressed theoretically

### Questions
1. Can the $\epsilon$-dependence gap be closed? Is the true sample complexity $\Theta(1/\epsilon)$ or $\Theta(1/\epsilon^2)$?
2. How would sample complexity scale for learning policies that select multiple rounds of cuts?
3. Would results transfer to production solver configurations with presolve and heuristics enabled?
4. Do similar lower bounds hold for other cut families beyond Chvatal-Gomory cuts?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proves sample complexity bounds in the setting of learning to cut for solving integer programs. The bounds are for two different scores, branch and cut tree size and gap closed. The theoretical results show that both scores are similarly difficult to learn. Additionally, the paper provides

### Strengths
- The paper is well written: it illustrates its core ideas through intuitive examples and explanations without sacrificing technical details.
- The bounds derived don't seem to be vacuous/uninformative. The proofs are also relying on (as far as I can tell) novel constructions that are specific to the problem.
- To the best of my knowledge, these are the best known nontrivial bounds for this learning problem.
- The paper also provides an experimental comparison of the learned cuts of a GNN compared to well known heuristics and shows that the GNN can learn to generate cuts that are on par or better compared to those heuristics.

### Weaknesses
- In the proof of 3.2 (Around line 600), you set $a = \epsilon$ and use the Transfer Lemma. 
Doesn't that mean you take $$\text{fat} _{\mathcal{F} _{s,\sigma'}} (\frac{\epsilon}{\epsilon}) \geq \text{VCdim}(\mathcal{F}[n]) ?$$
However the transfer lemma holds for $\gamma \in (0,1/2)$ so I'm not sure you want to do it this way. Maybe set $a=c\epsilon$ for c at least 2?
- How necessary is assumption 2 for your result? Are there any architectures that don't qualify? That's perhaps one part I'm a little unclear about and I woud like to see some more comments on.
- There are a bunch of typos in the document so more careful proofreading is required. For example see lines: 120, 224, 459-460, 
- I am not sure how the experimental section ties into the overall contribution here. I guess it's good to see that GNNs can learn a practically viable heuristic by training on gap closed but it is a bit unclear how that connects to the theoretical findings of the paper.

Overall this is a good paper that I lean towards accepting. There are some (I think minor) issues with the proof and I'm a bit unclear about assumption 2 so I start with a tentative score that I will reconsider after the rebuttal.

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3
