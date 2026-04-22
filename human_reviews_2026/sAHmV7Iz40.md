# Topology Explains Loss Barriers: Quotient Homology of Neural Loss Landscapes

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 0, 4, 6

## Abstract
Although neural network parameter spaces are contractible, training exhibits global phenomena that elude purely Euclidean accounts. We show that these effects arise up to symmetry after factoring out ubiquitous reparameterizations, the low-loss regions of the quotient landscape acquire nontrivial homology. For semi-algebraic losses and common symmetry groups, we prove that quotient sublevel sets $S_c/G$ have finite Betti numbers and that $\beta_k(S_c/G)>0$ yields a topological certificate of barriers. Also, we operationalize these insights with symmetry-aware trajectories (permutation alignment, scale normalization, Stiefel-consistent updates) that remove spurious obstacles and expose genuine connectivity in the quotient space. Experiments on Stiefel-constrained autoencoders and residual networks support the theory: homology summaries of sublevel sets predict the presence or absence of interpolation barriers, and quotient-aware paths recover robust mode connectivity. Taken together, our results provide a principled and testable account of why weight matching works and when loss barriers are intrinsic rather than artifacts of parameter redundancy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a rigorous algebraic-topological framework showing that after quotienting neural network parameter spaces by natural reparameterization symmetries (permutations, scalings, orthogonality, etc.), low-loss sublevel sets can have nontrivial homology (nonzero Betti numbers), which certifies intrinsic loss barriers, and demonstrates via small-scale experiments and symmetry-aware trajectory heuristics that many apparent barriers are removable by symmetry alignment.

### Strengths
Well written and clearly presented: The paper is impressively well organized for such a mathematically demanding topic. The authors maintain clarity throughout by introducing formal definitions only when needed, summarizing key intuitions before each theorem, and connecting abstract topology to intuitive geometric analogies.

Theoretical rigor: The paper grounds intuitive empirical observations about mode connectivity and weight matching in solid algebraic topology; proofs (equivariant triangulation, transfer, Alexander duality) are provided and carefully connected to the neural setting.

Conceptual clarity and novelty: Whereas persistent-homology or Hessian-based analyses merely observe the local “shape of the mountain” (and may see spurious peaks created by parameter symmetries), the quotient-homology framework explains which mountains are real. By factoring out reparameterization symmetries, it distinguishes fake loss barriers from intrinsic topological obstructions that truly separate functionally distinct minima. This theoretical distinction provides deep explanatory power for why weight matching and permutation alignment succeed where naive interpolation fails.

Bridges theory and practice: Though chiefly theoretical, the paper proposes concrete symmetry-aware operations and supports claims with toy experiments that show the practical relevance of quotient-aware analysis.

### Weaknesses
Computational tractability: The paper does not provide practical, scalable algorithms to compute Betti numbers or other homological summaries for high-dimensional modern networks; proofs rely on triangulation and semialgebraic theory that are infeasible to implement at scale.

Limited empirical validation: Experiments are small-scale (Stiefel-constrained AE, mini-ResNet) and do not show performance or behavior on larger architectures (e.g., full ResNets, transformers) or real-world training pipelines. Quantitative TDA results (explicit persistence diagrams / Betti estimates) are not reported.

From certificate to algorithmic guidance: While nonzero Betti numbers certify barriers, the paper leaves open how to use homological information to design concrete, scalable optimization algorithms (beyond heuristic alignment and retraction).

### Questions
Computation & approximation: Do the authors have a concrete plan or preliminary experiments for estimating Betti numbers or persistence diagrams of S_c/G in practice? For example, can they (i) provide a toy pipeline that computes persistent homology on symmetry-aligned parameter samples, or (ii) report persistence diagrams/Betti estimates for the presented experiments?

Robustness to BN and optimizer state: BatchNorm running statistics and optimizer states (batch-norm, Adam moments) introduce stateful, dataset-dependent structure. How would these affect the semialgebraicity assumptions and the validity of the homology-based certificates?
Algorithmic use of homology: Beyond alignment heuristics, do the authors envision concrete ways to integrate homological information into optimization (e.g., topology-aware regularizers, homology-guided reparameterizations)? Any preliminary ideas or toy experiments would strengthen practical impact?

Scope of experiments: Can the authors expand empirical validation to larger models or provide a quantitative TDA-based analysis (persistence summaries) on the current experiments to support claimed Betti-related diagnoses?

Local (Hessian) vs global (quotient) approaches: Could the authors comment directly on when Hessian-based low-dimensional PH approximations (e.g., PH in a top-k Hessian subspace) are expected to agree or disagree with quotient-homology conclusions? Any guidance for practitioners on how to combine these diagnostics?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper gives a topological approach to the study of symmetries of the loss landscape. The main claim is that quotienting out the symmetries of a neural network creates an effective weight space whose topology is non-trivial, contrary to the standard Euclidean structure. This non-trivial topology is claimed to be responsible for well known and commonly observed features of the loss landscape, like loss barriers between minima. The paper proves properties about the topology: the finiteness of the Betti numbers (well known topological invariants) for sublevel sets of the loss and for the quotient and the exact structure of the quotient when we normalize the neurons' weights.

### Strengths
- The introduction does a good job in setting up the problem.
- Although I don't agree with it (see below), the paper provides an interesting approach to give a meaning to the topological structure resulting from symmetry, in particular the presence of nonzero Betti numbers beyond the order 0.

### Weaknesses
I believe that, in its current form, there are severe issues that make the paper not suitable for publication.

### Writing issues
I think that the most critical issue is the overall lack of clarity in the writing.

The technical results are presented in a way that is not appropriate for this venue. Notation is often undefined and technical notions from topology are taken for granted and not explained. The preliminary section is a good example of this fact that, however, holds throughout the paper.
Even in the first definition of geometric support, objects such as "topological space", "singular chain", and even the simplex symbol $\Delta^k$ are all undefined. In Definition 2.2. the homology group is presented with its symbol $H_{k}(X;G)$ without definition and the notions of *carrier* and *closed geometric support* are defined without any intuition or explanation. I don't think it is fair to assume that ICLR readers, even the ones that are mathematically more well-versed, are familiar with these notions and their meaning.

The appendix, especially section A which should cover algebraic topology bases, does define the objects used in the main text but still reads as extremely and unmotivatedly technical in its language and in the notation. I understand that algebraic topology is not a simple topic and, to a certain extent, requires some technicality, but I believe that a better job could have been made in simplifying the necessary material to the strictly necessary for the paper's claims. The feeling I get is that too much material is presented too quickly, with no apparent effort in easing the reader into understanding the concepts and how they relate to the main text material and the problem at hand.

### Interpretation issues
The significance of the main results of the paper leverages on the fact that having non-trivial homotopy groups in the loss landscape is relevant to understanding training dynamics and loss barriers.
This is claimed multiple times in the paper in different ways, e.g.: 
- L60: "Nonvanishing Betti numbers signal essential cycles and homological obstructions that any continuous path between certain solutions must intersect [...]"
- L118: "[...] there exists a noncontractible k-cycle whose neighborhood any path between certain solutions must intersect, certifying unavoidable loss barriers"
- L125: "Lower bounds on $\beta_{k−1}(S_c/G)$ give lower bounds on the number of attraction basins separated by representatives of (k − 1)-cycles"
- L211: "[...] It implies trivial homotopy groups meaning that there are no noncontractible loops or higher-dimensional cycles in the raw parameter space, so any potential topological barriers must arise from level sets of the loss function rather than from the ambient space."
- L233: "[...] Since the ambient parameter space is contractible, any local minima or saddle points encountered during training must arise from the geometry of the loss function rather than from topological obstructions in the domain"
- L388: "[...] Any continuous path joining minima or saddles that lie on different sides of such a cycle must exit the sublevel set and incur higher loss."

My feeling reading the paper is that these claims are rather vague, and the exact reason why studying these topological features should be important is not clear at all. 
From the different interpretations of I see in these sentences, the one that makes more sense to me is that the presence of a $k$-dimensional cycle of equal loss can provide a barrier that stops the training trajectory. However, given that the training trajectory is a curve, this seems to me to be true only in the case where $k$ is equal to the dimension of the ambient space. In that case, the cycle can "encapsulate" the trajectory, preventing it from escaping, but in all other cases it seems to me that no obstruction would result from it. To make this more intuitive, a sphere (2-dim hole) can stop a curve in 3d, but a circle (1-dim hole) cannot.


Moreover, even taking the interpretation to be (in some sense) true, I feel that the main theorem (3.4) is not particularly strong as it proves that the sublevel sets of the loss and the quotient space both have finite Betti numbers but doesn't prove that they are actually nonzero in general.
Prop. 3.2 and Theorem 3.3 give examples of situations where they are, in fact, nonzero, but they both rely on the assumption that we are constraining the weight matrices' rows to have unit norm. The motivation for assuming this is not clearly explained in the paper.


### Experiment issues
The paper motivation for the numerical analysis is not clear. In Example 4.1 why should we study a network with weights constrained on Stiefel manifolds? 
The setup and results themselves are, at least to me, not understandable. "random Stiefel retraction perturbation of $W_{1},W_{2}$" (L332) are undefined and the summarization as "In short, the local landscape is broadly well–conditioned modulo the group symmetry." (L335) doesn't clarify the experiment result.


Overall, the text is not clear in explaining how the experiments relate to the theory above.

### Questions
Following my comments in the weaknesses section, can the authors clarify exactly what is their interpretation of how higher-dimensional topological features impact the learning process?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper provides a topological explanation for why symmetry-aware methods are necessary in analyzing neural network loss landscapes and in particular their sublevel sets.
While the ambient space is contractible i.e it has no interesting structure, the paper shows that symmetries or other constraints can induce topological features.
An analysis of the homology or homotopy groups then characterizes loss barriers that are present in the quotiented parameter space.
To have meaningful summaries, Betti numbers of both the original and quotiented sublevels sets are proven to be finite under some assumptions.
This finding could explain why high loss is observed on interpolating paths between minima in (linear) mode connectivity and why these disappear after alignment.
The authors also provide a cautionary example showing that naive aligmnent can increase loss barrier if other factors such as batch norm statistics are no properly handled.

### Strengths
The point made that meaningful analysis of neural loss landscapes requires working in the quotient space, where symmetries are factored out, is convincing and the use of topology as a tool of choice to abstract from coordinate representations is relevant in this context.

The work's originality lies in its application of advanced algebraic topology to establish negative (impossibility) results about interpolation in parameter space, for instance that some barriers are unavoidable.
The focus on mode connectivity, which has been studied a lot in the literature is still relevant.

### Weaknesses
The theoretical setups feel a bit disconnected from practical deep learning scenarios.
Although neural networks appear throughout, the deep learning motivation is not clear.
For instance, in proposition 3.2 and theorem 3.3 a constraint $\|W_{j,*}\|=1$ is studied, but the rationale from a deep learning perspective is unclear.
Since this constraint typically reduces expressivity it is not a symmetry and therefore seems arbitrary.
Similar concerns apply to the Stiefel autoencoder setup.

Example 4.2 (the mini-ResNet experiment) seems somewhat disconnected from the main theoretical narrative, and its role in the overall argument could be clarified.
Additionally, while the paper claims that quotient-aware interpolation with corrected batch normalization stats yields smooth, low-loss paths, the corresponding plot is not provided.

However, the main drawback in my opinion is on accessibility and presentation.
While the paper includes a primer on algebraic topology, the exposition remains quite involved.
Many advanced concepts are introduced rapidly without a lot of context including references without warm up to: cycle representative, singular k-simplex on which depends the singular k-chain, cw complex and homotopy type, fundamental group, Morse index, transfer map, Stiefel retraction perturbation, etc. This breaks the reading flow for readers without substantial prior background in homology and algebraic topology, potentially limiting the paper's impact in the deep learning community.
Less crucially, omitting details or relying on notational conventions makes some parts ambiguous e.g. (1) The norm line 244 is unspecified though probably Euclidean, see minors below as well (2) The experimental complexity is reasonable, but some implementation details needed for reproducibility are missing, and no code is provided. For instance, in Example 4.2, the permutation computation method (weight alignment, activation alignment, or another approach?) is not specified.

## Minor and additional feedback
- First two paragraphs of the introduction, consider distinguishing explicitly mode connectivity from linear mode connectivity
- Remark 2.3: the first line introduces $z$ but it is not used afterwards
- Section 3, the equivalence relation on ReLU network is not the same as the pointer to appendix B.3
- Possible typo on example 4.1 in the definition of Stiefel matrix there is both (2,3) and (3,2) shapes, which in my understanding is inconsistent
- Assumption A2 of theorem 3.4: $G$ is defined line 280 and redefined line 283
- A citation to Git Re-Basin: Merging Models modulo Permutation Symmetries by Ainsworth et al could make sense in the discussion on linear mode connectivity.
- To enhance the reach of the paper, a possibility could be to try to explain results at different mathematical levels or convey a more intuitive picture.
- Overall the flow could be improved e.g.:
    - line 220: "in a one-hidden-layer network with ReLU one can simulatneously scale..." -> "in a one-hidden-layer network with ReLU activation, one can simulatneously scale..."
    - line 226: it is not clear in the sentence whether the homeomorphic statement is a results or a hypothesis without looking at appendix B.3
    - line 230->240: going back and forth makes the delivery confusing
    - line 334: missing "of"
    - line 393: strange use of "even"

### Questions
I have a few questions:

1.**Intersection of paths with cycles** (lines 61, 119): "Nonvanishing Betti numbers signal essential cycles and homological obstructions that any continuous path between certain solutions must intersect" Could you clarify what "intersect" means in this context?


2.**Nature of loss barriers**: Throughout the paper should the term "loss barrier" always be interpreted as referring to barriers along linear interpolation paths specifically (as opposed to curved trajectories)?


3. **Non-linear mode connectivity without symmetry alignment**: in mode connectivity, low-loss paths can often be found between independently trained networks without explicit symmetry matching if the path is non-linear (computed with Nudged Elastic Band for instance). How does this contrast with your paper ?


4. **Example 4.1 and learning dynamics** the decoder's inability to reconstruct the full sphere appears to be a consequence of the rank-2 bottleneck architecture. Is this limitation stable throughout training ?


5. **"Sides of a cycle" terminology** Could you provide a more precise explanation of what you mean by "sides of a cycle" line 388-389 ? 


6. **Example 4.2 interpretation**: In Figure 4, at $\alpha=1$ the orange curve represents $P\theta^B$ and the blue and orange curves don't coincide at this endpoint. Can this discrepancy be entirely attributed to mismatched batch normalization statistics? Put differently: if BN statistics were also linearly interpolated would the orange curve decrease toward $\alpha=1$ instead of increasing?ease ?

Being unfamiliar with multiple concepts in the paper and its appendices I choose a low confidence score and look forward exchanges, other reviews and answer to my questions to eventually reassess both my rating and confidence scores.

### Soundness
3

### Presentation
1

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
This paper develops a rigorous topological framework to explain loss barriers and mode connectivity in neural network training. Although neural parameter spaces are topologically simple, the authors show that after factoring out symmetries, the resulting low-loss regions acquire nontrivial topology. Using algebraic topology, they prove that nonzero Betti numbers in these quotient sublevel sets provide certificates of unavoidable barriers between minima, while symmetry alignment can remove spurious ones. Experiments on Stiefel-constrained autoencoders and small ResNets confirm these predictions: once symmetries are accounted for, apparent barriers vanish and smooth low-loss paths emerge. The work thus reframes mode connectivity as a consequence of the topology of quotient loss landscapes, offering a principled link between mathematical invariants and practical training behavior.

### Strengths
- The paper offers a clear and rigorous quotient‑space viewpoint that makes global phenomena like mode connectivity and apparent barriers mathematically precise rather than anecdotal. 
- The case studies isolate symmetry effects cleanly: the Stiefel autoencoder exhibits bowl‑like local curvature with scatter explained by an O(2) gauge, and the ResNet example shows how naive interpolation produces artificial bumps that diminish once quotienting and BN handling are applied. 
- The presentation is careful and self‑contained, with preliminaries and full proofs that make the homological arguments traceable, and with illustrative figures that tie the intuition to the results.

### Weaknesses
- The strongest theorems assume finite symmetry groups and semi‑algebraic losses, which may not fully cover the experiments or common training pipelines. 
- The empirical section is small‑scale and partly confounded: the ResNet interpolation curves degrade due to stale BN statistics, and while Algorithm 1 proposes recalibration, the paper does not convincingly show smooth low‑loss interpolation after proper BN correction. 
- The connection from topological signatures to generalization or training speed remains qualitative. No experiments demonstrate predictive power of Betti‑number summaries beyond detecting barriers.

### Questions
- Can you extend Theorem 3.4 beyond finite groups to compact Lie group actions (e.g., layer‑wise scalings or orthogonal gauges), or clarify what parts extend via orbifold/stratified arguments?

### Soundness
3

### Presentation
3

### Contribution
3
