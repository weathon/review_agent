# Poisson-Algebraic Parallel Scan: A fast symplectic framework for neural Hamiltonians

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 4, 2

## Abstract
Learning Hamiltonian neural networks (HNNs) that respect the intrinsic symplectic structure of physical systems has emerged as a foundational framework for robust long-term predictions in scientific machine learning. Nevertheless, existing HNN methods face critical limitations: (i) inherent sequential integration prevents parallel computation, causing significant computational bottlenecks, and (ii) unconstrained neural architectures lead to instability when extrapolating dynamics beyond training regimes. To address these fundamental challenges, we introduce $\textit{Poisson-Algebraic Parallel Scan (PAPS)}$, a novel framework that leverages a carefully constructed Poisson algebraic decomposition of the learned Hamiltonian. By embedding polynomial generators explicitly closed under Poisson brackets, PAPS induces an associative Lie-group structure that naturally facilitates parallel-scan (prefix-sum) computation. Our method achieves exact symplectic integration with up to three orders-of-magnitude ($1000\times$) speedup at $10^3$ integration steps, significantly outperforming existing HNN approaches. Moreover, the structured algebraic representation inherent in PAPS ensures intrinsic physical consistency, delivering stable and reliable extrapolation far beyond the training distribution. Extensive theoretical analyses and rigorous numerical experiments validate the superior computational scalability of our approach, highlighting PAPS as a powerful new direction for scalable and physically consistent neural Hamiltonian modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Poisson-Algebraic Parallel Scan (PAPS) that can be used for efficiently and effectively training parametrized Hamiltonian dynamics. The authors embeds the Hamiltonian in a polynomial function space, forming a Neural Poisson algebra whose flows compose as a Lie group, enabling an associative efficient prefix-scan. This construction guarantees structure preservation—symplectic form, Liouville volume, and energy—at every step of the composed flow. Across quantum spin dynamics and molecular dynamics, PAPS offers higher trajectory accuracy while robustly preserving invariants, and dramatically reduces inference time compared with baselines.

### Strengths
- In data-driven simulations, approaches that incorporate physical constraints such as symplectic structures are extremely important. Also, learning Hamiltonians with numerical integration schemes often incurs high computational costs, making this research address a significant problem.
- The method of limiting the Hamiltonian to polynomials and using the parallel-scan technique while preserving Poisson brackets is novel.
- Theoretical considerations regarding approximation error and generalization performance have been made.
- Experiments were conducted on relatively large-scale systems, demonstrating the effectiveness of the proposed method.
- The manuscript is very well written.

### Weaknesses
- There is no mention of differences from previous studies with similar motivations (e.g., SympNets).
- There is no discussion of limitations (see Questions).

### Questions
- There is no mention of methods that directly model symplectic flows (e.g., SympNets [R1]). This approach enables simulation without numerical integration by directly modeling the symplectic map, making it fast. Furthermore, the symplectic structure is preserved. While the problem is similar to this research, could you please add a discussion?
- A comparison with SympNets' universal approximation theorem might be interesting. It could clarify the impact of assuming polynomials, as done in this study.
- Looking at Table 1, the HNN-based approach shows the best performance for the violation of $\Delta \omega$. Could you explain why the proposed method cannot accurately preserve the symplectic structure? Is it because the proposed method preserves the Poisson brackets, meaning it targets the Poisson system (i.e., generalization of Hamiltonian systems), and thus does not necessarily satisfy the symplectic structure?
- As the system becomes more complex, what is the expected order of the appropriate polynomial? Also, is there a way to determine it automatically?
- In the experiment, what time interval constitutes one step? $\epsilon=?$
- This paper assumes canonical coordinates, but is future extension to general coordinate systems possible, as in [R2]?

[R1] P. Jin et al., SympNets: Intrinsic Structure-Preserving Symplectic Networks for Identifying Hamiltonian Systems, Neural Networks, 132:166–179, 2020.

[R2] Yuhan Chen et al., Neural Symplectic Form: Learning Hamiltonian Equations on General Coordinate Systems, NeurIPS, 2021.

### Soundness
3

### Presentation
4

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
This work extends HNN-like modeling approches by making the generators to a polynomial be closed under the Poisson bracket, and uses the fact that the flow map derived in this way forms a Lie group under composition (i.e., it satisfies associativity) to enable parallelization via Parallel Scan. Whereas standard HNNs require $O(M)$ computation for $M$ steps, the proposed parallelization compresses the time to roughly $O(log M)$. The experiments show that, while the method is not always the strongest, it is consistently very fast, so it clearly offers a Pareto-optimal solution.

### Strengths
The proposed method is carefully designed and supported by solid theory, offering clear reasons for both its speed and its accuracy. The contribution is highly insightful.

It is compared against a wide range of approaches, including SE(3)-equivariant models, HNNs, and machine-learning potentials, which verifies confidence in the reported results.

### Weaknesses
(1) The statement in Theorem 4.1 reads as if the proposed method strictly preserves symplectic structure and energy, but that only holds when the neural network has learned so that $\\{H, g_m\\} = 0$. Of course, training on trajectories might lead to such a result, and such parameters exist within the hypothesis space, but it is not guaranteed. The condition is truly satisfied only when the true solution is learned perfectly. In the extreme, any neural network-based predictor would satisfy all desired properties if it learned with zero error, which makes the statement meaningless. In this sense, the explanation feels overclaiming or misleading. By contrast, the Hamiltonian nature of HNNs and the symplecticity of SympNets hold by definition. The paper should make clear what is guaranteed by definition and what is obtained only through learning.

(2) The function is restricted to polynomials, so this method lacks generality for learning unknown potentials. Moreover, because it is polynomial, the combinations can explode, raising concerns when the number of molecules grows. One also wonders whether an HNN constrained to polynomial potentials would achieve even better accuracy; a more balanced comparison seems possible.

(3) Thies paper compares against SE(3)-equivariant models, HNNs, and machine-learning potentials, but not against combinations with them. For example, building an SE(3)-equivariant HNN and comparing should be straightforward.

(4) There is prior work on parallelization for Hamiltonian systems: Dai et al., Symmetric parareal algorithms for Hamiltonian systems, ESAIM: Mathematical Modelling and Numerical Analysis, 2013.
For parallelizing dynamics with Prefix Sum, see: Yang et al., Parallel Dynamics Computation Using Prefix Sum Operations, IEEE RAL, 2017.
These are not for HNNs, but they could be combined. Thies paper lacks discussion and comparison (theoretical or intuitive would suffice, not necessarily experimental) with such existing parallel methods (parareal, MGRIT, etc.) applied to HNNs.

(5) The original HNNs are proposed in Greydanus et al., NeurIPS 2019. SympNets (Jin et al., Neural Networks, 2020) are neural networks with symplectic structure. These key references are missing. In terms of preserving the symplectic form, SympNets may be closer in spirit to the proposed method than HNNs.

(6) The paper uses the citet command where the citep command is appropriate, which causes the citations to be mixed with the text and reduces readability.

The efficiency of the proposed method is beyond doubt and should be highly valued. However, in its current form, some descriptions and comparisons are misleading or limited, and I do not think it should be accepted as is. If I have not misunderstood, I would be happy to raise my rating to 8 once appropriate explanations and clarifications are added.

### Questions
See Weaknesses.

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
This paper proposes Poisson-Algebraic Parallel Scan (PAPS), a new framework to overcome two key limitations of Hamiltonian Neural Networks (HNNs): (i) the sequential nature of standard symplectic integrators (which hinders parallel computation), and (ii) the instability of unconstrained neural models when predicting far beyond the training data. The authors address these issues by introducing a structured polynomial decomposition of the learned Hamiltonian that is closed under Poisson brackets. The paper provides extensive empirical validation on physics tasks, demonstrating dramatically improved scalability and long-term accuracy compared to existing approaches.

### Strengths
The core strength is the novel parallel prefix-sum integration scheme for Hamiltonian systems. By exploiting an associative Lie-group structure in the learned dynamics, the method reduces trajectory computation from linear to logarithmic depth. This yields orders-of-magnitude speedups in practice.

The paper is rigorously grounded in theory. Beyond preserving invariants, the authors derive error bounds and universal approximation guarantees for their polynomial integrator.

The experimental evaluation is thorough and demonstrates clear advantages of PAPS. The authors test the framework on multiple domains, including quantum spin dynamics and molecular dynamics (MD),

### Weaknesses
1. A potential concern is the restrictive function class imposed on the learned Hamiltonian. PAPS requires the Hamiltonian $H_\theta$ to lie in a specific polynomial algebra ($\mathcal{H} = \text{Poly}_r$ of degree $\le r$)

2. The paper’s notation and theoretical formalism, while rigorous, may be overwhelming for many readers. It presumes familiarity with advanced concepts from symplectic geometry and Lie algebras, which are not commonly part of the background of the machine learning community. Moreover, after carefully reading the manuscript, I did not find the necessity of introducing some definitions to be clearly justified, leaving their role in the overall argument somewhat unclear. Please Question part below.

3. The method introduces new hyperparameters, chiefly the polynomial degree $r$ and the DAG decomposition depth $K$ (number of sub-Hamiltonian partitions). The authors conduct an ablation showing that a moderate $r=5$ and $K=2$ worked well on their tasks. However, the paper could better articulate how these hyperparameters affect performance and how one might choose them for a new problem. For instance, increasing $r$ increases the model’s capacity (and the number of polynomial basis terms), does this always improve accuracy, or are there diminishing returns / overfitting observed beyond a certain point? The ablation suggests there is an optimal middle ground, but more explanation would help: e.g., higher $r$ might fit training trajectories better but could generalize worse if it starts fitting noise or spurious dynamics.

### Questions
1. The most significant concern lies with the discussion of Lie Group Associativity in Eq. 6 and the subsequent remark. It is not clear to me why this property is claimed to hold only for functions within the polynomial hypothesis space proposed in the article. The manuscript does not provide sufficient justification for this restriction. Moreover, in my view, the associativity of composition is a property that holds for arbitrary mappings, which makes it difficult to see why the introduction of Lie group machinery in this section is necessary. Is there a counterexample?

2. The introduction of the Neural Poisson algebra in Definition 2.2 and the subsequent verification of the three structural properties in Proposition 2.3—namely, closure under product, Poisson bracket, and Jacobi identity—are not sufficiently clear in terms of their motivation and necessity. As far as I understand, these properties are already guaranteed for the classical Poisson bracket under a fixed skew-symmetric matrix J, and the paper does not seem to propose learning or parameterizing J (e.g., as in some prior works on generalized Poisson structures). If so, then verifying these properties again seems more like algebraic bookkeeping rather than offering new theoretical insight. It is unclear what additional meaning or benefit this verification provides for the method, e.g., generalization, learning, or numerical stability.

Therefore, I suggest that the authors clarify the role and purpose of Definition 2.2 and Proposition 2.3.  

3. The parameterization of the Hamiltonian as a polynomial in state variables with coefficients produced by a neural network that takes the initial condition and time as input (Definition 2.1 and Algorithm 1) is quite puzzling and lacks sufficient justification or clarity. While choosing a polynomial basis for the Hamiltonian is reasonable, the decision to make the polynomial coefficients depend on both the initial condition and the time raises several questions:

3.1 Is the Hamiltonian explicitly time-dependent? If so, this should be clearly stated early in the paper, and the time variable should be included within the domain of the Hamiltonian itself (i.e., H(q,p,t)). However, the main text seems to treat the Hamiltonian as time-invariant in the classical sense.

3.2 If the Hamiltonian is not time-dependent, then making the coefficients depend on time and the initial state seems contradictory. In particular, the Hamiltonian function should be a property of the physical system, not of a specific trajectory. A Hamiltonian that changes across different initial conditions or depends on absolute time contradicts the foundational assumption of autonomous Hamiltonian dynamics.

3.3 How are the “initial condition” and “time” defined in the context of training data? Are they referring to the starting point of the particular trajectory being modeled and the time elapsed since then? If so, this implies that the learned Hamiltonian is trajectory-dependent, rather than a global model of the system. This seems to undermine the generalization claim and could break important symmetries such as time-translation invariance.

3.4 Does this formulation prevent standard data augmentation along trajectories? For example, in classical mechanics, a segment 
z1 z2 z3 of a trajectory can be split into overlapping sub-trajectories like z1 z2 and z2 z3 with consistent dynamics. If the Hamiltonian now depends on the initial point, then such overlapping reuse becomes invalid, limiting data efficiency and possibly introducing bias.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
1

### Summary
The general idea consists of working with a special class of polynomials truncated to degree r to approximate Hamiltonian dynamics. The polynomial class is shown to be closed under (truncated) products and Poisson brackets, allowing for the construction of a hierarchy of models of increasing complexity.  

Alas this is about all that I understand. The authors chose to write the paper in a jargon-heavy style that assumes a strong knowledge of symplectic theory.  Maybe they actually think in these terms, but your typical ICLR reader will not.  Reading the algorithms provided in the appendix gives more information about what they are actually doing, but I would like to have seen this explained in the main text.

In conclusion, my ability to understand this paper as written is extremely limited. Yet I understand enough to be quite certain that a more accessible explanation could have be given. When I am rating the contribution as low, I do not mean that the subject matter is problematic, but that very few ICLR readers will get anything out of this paper as written.

### Strengths
- Experiments cover a number of potentially interesting systems. What is being measured is unclear though.

### Weaknesses
- I wish I knew understood definition 2.1  I would have to know what all the symbols mean in there. Come on, there must be a simpler way to write this, something closer to equation (3) and the following unnumbered equation, for instance.

- Regardless of its intrinsic merits, very few ICLR readers will get anything out of this paper as written.

### Questions
- Can you rewrite the paper in a way that will both educate and enlighten the ICLR readers?

### Soundness
3

### Presentation
1

### Contribution
1
