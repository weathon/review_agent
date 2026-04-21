

{0}------------------------------------------------

# BEYOND TURING: TOPOLOGICAL CLOSURE AS A FOUNDATION FOR COGNITIVE COMPUTATION

Anonymous authors

Paper under double-blind review

## ABSTRACT

Classical models of computation, epitomized by the Turing machine, are grounded in *enumeration*: syntactic manipulation of discrete symbols according to formal rules. While powerful, such systems are intrinsically vulnerable to Gödelian incompleteness and Turing undecidability, since truth and meaning are sought through potentially endless symbolic rewriting. We propose an alternative foundation for non-enumerative computation based on *topological closure* of semantic structures. In this view, cognition operates by promoting transient fragments into closed cycles, where  $\partial^2 = 0$  ensures that only invariants persist. This shift reframes computation from *syntax* to *structure*: memory and reasoning arise not by enumerating all possibilities, but by stabilizing relational invariants that survive perturbations and generalize across contexts. We formalize this principle through the dot-cycle dichotomy: dots or trivial cycles ( $H_0$ ) serve as high-entropy scaffolds for exploration, while nontrivial cycles ( $H_1$  and higher) encode low-entropy invariants that persist as memory. Extending this perspective, we show how Memory-Amortized Inference (MAI) implements an anti-enumerative principle by storing homological equivalence classes rather than symbolic traces, yielding robust generalization, energy efficiency, and structural completeness beyond Turing-style models. We conclude that *topological closure* provides a unifying framework for perception, memory, and action, and a candidate foundation for cognitive computation that transcends the limits of enumeration.

## 1 INTRODUCTION

Since the early 20th century, formal models of computation have been grounded in *enumeration*. The Turing machine, along with its close relatives in the Church–Turing framework, exemplifies this paradigm: computation is conceived as the syntactic manipulation of discrete symbols on an infinite tape, with new results obtained only through the stepwise application of formal rules Turing (1936). This model has proven enormously successful, forming the foundation of digital computing, automata theory, and modern complexity classes. Deep learning architectures, despite their connectionist implementation, inherit this enumerative character: generalization emerges by statistical interpolation over enumerated training examples, and inference requires repeated evaluation across contexts Goodfellow et al. (2016). Yet the power of enumerative systems is matched by their intrinsic limitations. Gödel’s incompleteness theorem demonstrates that no sufficiently expressive formal system can be both complete and consistent: there will always exist true statements that cannot be proven within the system. Turing’s halting problem further establishes that no algorithm can decide, in finite time, whether arbitrary programs will terminate Sipser (1996). Both results reveal a deeper structural fragility: *enumeration can never guarantee closure*. Each attempt to list or decide the totality of possible outcomes leaves residual boundaries, open fragments that escape formal capture. This fragility manifests as brittleness in symbolic AI, combinatorial explosion in search Minsky (1961), and distributional failures in data-driven models. Enumeration fails because it cannot stabilize residual structures, open chains proliferate without ever closing. Topological closure reframes this failure: what enumeration leaves dangling, closure promotes into invariants.

In this paper, we propose an information topological framework for intelligence in which *cycle closure* is the fundamental mechanism of memory. Building on the first principle, we argue that memory is best understood not as a static store of representations, but as the ability to *re-enter and traverse latent cycles* in neural state space. We identify these invariant cycles as the natural carriers

{1}------------------------------------------------

of meaning across scales: they act as *alignment checkpoints* between context ( $\Psi$ ) and content ( $\Phi$ ), filtering out order-specific noise, enforcing closure, and preserving only what remains consistent across variations. A key principle underlying this framework is the *dot-cycle dichotomy*: trivial cycles collapse to dots ( $H_0$ ), serving as transient contextual scaffolds ( $\Psi$ ), while nontrivial cycles ( $H_1$  and higher) encode low-entropy content invariants ( $\Phi$ ) that persist as memory. This dichotomy clarifies how cognition achieves both adaptability and stability: dots support exploration, while cycles carry persistent knowledge across contexts. From this perspective, cognition is not tape-based symbol manipulation but the promotion of transient fragments into closed cycles that survive perturbation and generalize across contexts. This shift from *syntax* to *structure* reframes memory, learning, and reasoning as processes of stabilizing invariants, not enumerating sequences. Under this new conceptual framework, we develop the following arguments:

- We explore the physical origin of intelligence inspired by the first clue in Wheeler (2018):  $\partial^2 = 0 \Rightarrow$  Cycles (invariants)  $\Rightarrow$  Memory  $\Rightarrow$  Prediction (intelligence).
- We introduce the *dot-cycle dichotomy*: dots ( $H_0$ ) encode disconnected fragments, while cycles ( $H_1$  and higher) represent nontrivial order invariants that persist as memory.
- We introduce *Structure-before-Specificity* principle as the guidance of memory organization. Structural content is represented by low-entropy homology and specific context serves as high-entropy scaffolding.
- We show how *Memory-Amortized Inference (MAI)* implements a context-content uncertainty principle (CCUP) by bootstrapping and retrieval operators, yielding energy efficiency and robust generalization.

## 2 MOTIVATION: INTELLIGENCE AS TOPOLOGICAL CLOSURE

A unifying way to interpret both Gödel’s incompleteness theorem and Turing’s halting problem is to see them as demonstrations of the failure of *countable closure*. Any attempt to exhaustively enumerate truths or procedures inevitably leaves a residue, a diagonal element, an undecidable program, that lies outside the reach of the list. From a topological perspective, this means that enumerations generate fragments that remain open boundaries, unable to close into global invariants. What escapes enumeration is not accidental but principled: closure requires invariants beyond counting. This reinterpretation shifts the focus from the fragility of syntactic lists to the robustness of semantic cycles. The implication is profound: if Wheeler’s dictum It-from-Bit Wheeler (2018) highlights the informational substrate of reality, then for intelligence the relevant unit is not the fleeting bit but the persistent *cycle* that survives across variations Davatolhagh et al. (2024). Formally, we have

**Principle 1** (First Principle of Intelligence). *Intelligence is the capacity to stabilize invariants by cycle closure. At its core, cognition operates by minimizing joint context-content uncertainty  $H(\Psi, \Phi)$ , eliminating dangling boundaries and promoting them into closed cycles. These cycles constitute the fundamental units of meaning, memory, and prediction.*

Our guiding claim is that *cycle is all you need*: the organization of cognition, memory, and abstract thoughts in neural systems follows from the universal role of cycles as the algebraic residue of broken symmetry and the topological skeleton of information flow. This claim is supported by the hierarchical organization of cycles in mammalian brains, such as *Theta-gamma nesting* (e.g., hippocampus-entorhinal Buzsáki (1996)) and perception-action cycles Fuster (2004). In the spirit of Wheeler Wheeler (2018), we propose the following four No’s for cognition.

1. **No isolated information.** Bits are never standalone: they acquire meaning only through relations that close into cycles. Information without recurrence dissipates as noise.
2. **No privileged order.** The cognitive system must be robust to permutations of local steps. What matters is closure into a cycle, not the linear order of micro-events.
3. **No specificity before structure.** Persistent structures must stabilize first as the backbone of memory and prediction, while contextual specificities become scaffolding later to provide adaptive flexibility.
4. **No prediction without invariance.** Forecasting future states requires reducing entropy by filtering order-dependent variations; only invariant cycles can stabilize the predictive substrate.

{2}------------------------------------------------

![Figure 1: Three diagrams illustrating the dot-cycle dichotomy. Left: 'Open chain ⇒ dot in H₀' shows a path σ with endpoints that collapse to a single dot in H₀(Z). Middle: '∂² = 0 (boundary of boundary vanishes)' shows a chain σ whose boundary ∂σ (a pair of endpoints) collapses to zero. Right: 'Closed cycle ⇒ class in H₁' shows a closed loop γ in C₁(Z) with ∂γ = 0, representing a nontrivial cycle in H₁(Z).](1b7d539e02a202c2cf2d97698b911447_img.jpg)

Figure 1: Three diagrams illustrating the dot-cycle dichotomy. Left: 'Open chain ⇒ dot in H₀' shows a path σ with endpoints that collapse to a single dot in H₀(Z). Middle: '∂² = 0 (boundary of boundary vanishes)' shows a chain σ whose boundary ∂σ (a pair of endpoints) collapses to zero. Right: 'Closed cycle ⇒ class in H₁' shows a closed loop γ in C₁(Z) with ∂γ = 0, representing a nontrivial cycle in H₁(Z).

Figure 1:  $\partial^2 = 0$  enforces the dot-cycle dichotomy. *Left:* An open chain  $\sigma$  has a nonzero boundary  $\partial\sigma$  and collapses to a dot (class in  $H_0$ ), carrying no relational content. *Middle:* The boundary operator squares to zero:  $\partial(\partial\sigma) = 0$ . *Right:* A closed chain  $\gamma$  with  $\partial\gamma = 0$  persists as a homology class  $[\gamma] \in H_1$ , i.e., a cycle that encodes order-invariant structure.

**From constraints to clues.** These four principles define cognition as a non-ergodic information process Walters (2000): rather than averaging over all possible trajectories, the mind concentrates its dynamics onto recurrent, invariant cycles that persist across perturbations. Taken together, the Four No’s funnel cognition toward recurrent organization: items must close into cycles (no isolated information), be insensitive to micro-order (no privileged order), support re-entry (no static storage), and stabilize invariants for prediction (no prediction without invariance). The lightest formalism that enforces all four at once is the chain complex with boundary operator  $\partial$  Hatcher (2002): its nilpotency,  $\partial^2 = 0$ , cancels stray endpoints so that only closed traversals remain. This is the key new insight underlying the dot-cycle dichotomy, as shown in Fig. 1, and it sets up our first clue.

**Theorem 1** (The Boundary of a Boundary Vanishes). *Under the First Principle, intelligence is realized through cycle closure. This closure is only possible because the boundary operator  $\partial$  satisfies the fundamental identity  $\partial^2 = 0$ . That is, the boundary of a boundary vanishes. Cognitively, this law ensures that when cognition promotes boundaries into cycles, no further inconsistencies remain at the next level: every open edge is paired, every fragment canceled. This guarantees the existence of stable invariants (cycles), which are the carriers of meaning, memory, and communication. Therefore,  $\partial^2 = 0$  constitutes the First Clue of intelligence: coherence arises because boundaries consistently vanish when lifted, enabling cycles to persist.*

The vanishing of boundaries guarantees that what remains in memory is not arbitrary fragments but coherent cycles: the minimal invariants that bind context and content into an intelligible whole. From a computational standpoint, this marks a profound departure from the Turing paradigm. Traditional machines rely on symbolic tokens and sequential operations, where meaning is assigned externally to states of a register or tape. In contrast, a cycle-based architecture derives meaning intrinsically from topological closure: invariants are not “written” into memory but emerge from the very dynamics of neural interaction Gerstner et al. (2014). This dot-cycle dichotomy, where trivial cycles collapse and only nontrivial cycles persist, provides a natural mechanism for error correction, generalization, and energy efficiency without requiring exhaustive symbolic manipulation or gradient descent over high-dimensional parameter spaces. Rather than preserving measure by averaging over all paths, intelligent systems learn to concentrate probability mass onto order-invariant cycles (i.e., cycle-preserving structure replaces measure-preserving flow).

**Example 1** (Toy Navigation Loop). *In a  $5 \times 5$  grid with a square obstacle (a “hole”), trajectories that poke the obstacle and backtrack are open 1-chains ( $\partial\sigma \neq 0$ ) and collapse to trivial  $H_0$  “dots.” By contrast, any homing route that circles the hole and returns to start yields a closed 1-chain  $\gamma$  with  $\partial\gamma = 0$  and  $[\gamma] \neq 0$  in  $H_1$ . Crucially, reordering the same edges (e.g., north-first vs. east-first) produces the same class  $[\gamma]$ : the loop is order-invariant and reusable as a navigation template.*

## 3 MEMORY AS STRUCTURED TRAJECTORIES IN THE LATENT SPACE

Classical ergodic theory is built on the notion of a measure-preserving transformation Walters (2000). A dynamical system  $(X, \mathcal{B}, \mu, T)$  consists of a probability space  $(X, \mathcal{B}, \mu)$  and a measurable transformation  $T : X \rightarrow X$  satisfying  $\mu(T^{-1}A) = \mu(A), \quad \forall A \in \mathcal{B}$ . This measure invariance guarantees that long-term time averages along almost every trajectory coincide with ensemble averages

{3}------------------------------------------------

with respect to  $\mu$ . In this setting, entropy (e.g., Kolmogorov-Sinai entropy Cornfeld et al. (2012)) quantifies the unpredictability of the evolution under the assumption of ergodicity. Intelligent systems, however, are fundamentally non-ergodic: they retain memory, exhibit path dependence, and actively reduce uncertainty. In such systems, the measure  $\mu$  is not preserved, but typically concentrated onto lower-dimensional recurrent structures through learning and adaptation Spisak & Friston (2025). This concentration corresponds to entropy minimization rather than entropy conservation.

We propose that the appropriate generalization of “measure-preservation” in the non-ergodic setting is *cycle-preservation*. That is, while probability measures are not conserved globally, the system preserves *topological invariants* encoded in cycles that represent memory traces and recurrent behavioral motifs Gromov (1999). Formally, let  $(X, T)$  be a discrete-time dynamical system on a topological state space  $X$ . A  $k$ -cycle is a chain  $\gamma \in Z_k(X)$  satisfying  $\partial\gamma = 0$ . Under the induced map  $T_*$  on chains, invariance of  $\gamma$  requires that  $T_*\gamma - \gamma = \partial\beta$  for some  $(k+1)$ -chain  $\beta$ . Equivalently,  $[T_*\gamma] = [\gamma]$  in  $H_k(X)$ , where  $H_k(X)$  denotes the  $k$ -th homology group of the topological space  $X$ , so that  $\gamma$  is invariant up to homology class. In this way, although trajectories deform under dynamics (e.g., refer to the example of Wilson-Cowan model below), the *memory* encoded by the homology class persists.

**Example 2 (Wilson-Cowan Model).** The Wilson-Cowan system Wilson & Cowan (1972)  $\dot{E} = -E + S(w_{ee}E - w_{ei}I + P)$ ,  $\dot{I} = -I + S(w_{ie}E - w_{ii}I + Q)$  (with sigmoidal  $S$ ) undergoes a supercritical Hopf bifurcation for an open set of parameters, yielding a hyperbolic limit cycle  $\Gamma$ . Under small bounded input/parameter perturbations, trajectories deform (phase/amplitude modulation) but structural stability preserves a nearby periodic orbit  $\Gamma_\varepsilon$ ; thus the cycle, and its homology class  $[\Gamma_\varepsilon] \in H_1$ , persists even as paths vary.

This shift in perspective reframes the role of entropy reduction. In ergodic systems, entropy is managed by distributing trajectories uniformly across the entire state space  $X$ , ensuring statistical equivalence of time and ensemble averages. By contrast, in non-ergodic, adaptive systems, entropy reduction is achieved through *measure concentration* Gorbán & Tyukin (2018): rather than exploring all of  $X$ , trajectories are funneled toward lower-dimensional recurrent sets. These recurrent sets correspond to *persistent cycles* that remain stable under perturbations and across variations in initial conditions. In this sense, cycles act as the carriers of invariant information, preserving structural regularities across history-dependent dynamics and filtering out order-specific noise. The outcome is that intelligence emerges not from uniform exploration, but from the ability to stabilize information flow through the persistence of these invariant structures Ayzenberg et al. (2025). Formally, we have

**Principle 2 (Non-Ergodic Invariance Principle).** Let  $(X, T)$  be a dynamical system on a topological state space  $X$ . Then the natural counterpart of measure-preservation in ergodic theory is cycle-preservation:  $T_* : H_k(X) \rightarrow H_k(X)$ ,  $[\gamma] \mapsto [\gamma]$ . That is, an intelligent system preserves homology classes of cycles even while its measure evolves non-uniformly. These invariant cycles formalize memory persistence as the structural backbone of cognition.

When a non-ergodic system with many symmetric possibilities is forced to choose one outcome, symmetry is broken Anderson (1972). In neural and cognitive dynamics, this choice does not erase the unselected alternatives; instead, it organizes them into a closed cycle of relations: the chosen state, its competitors, and the transitions among them. In other words, the brain does not simply “pick a winner” among symmetric options. It establishes a cycle that records the selection, keeps the alternatives accessible for recall or switching, and stabilizes the outcome through recurrent interaction Hochreiter & Schmidhuber (1997). Broken symmetry, therefore, inevitably produces cycle formation, since the invariant residue of selection is a cycle connecting choice, memory, and potential revision.

This perspective reframes the role of entropy in prediction. Principle 2 establishes that non-ergodic systems preserve homology classes of cycles as their structural invariants. From an information-theoretic viewpoint, symmetry corresponds to maximal uncertainty: if all outcomes are equivalent under a symmetry group  $G$ , the induced distribution is uniform (entropy is maximized). Symmetry breaking reduces this uncertainty by eliminating redundant possibilities, thereby lowering entropy and concentrating probability mass around residual invariant cycles. In high dimensions, this process can be understood through the *theory of measure concentration* Ledoux (2001): instead of spreading trajectories uniformly, the dynamics of learning and memory focus trajectories around persistent cycles. To make this precise, we introduce the notion of *residual invariants* Beekman et al. (2019):

{4}------------------------------------------------

216  
217  
218  
219  
220  
221  
222  
223  
224  
225  
226  
227  
228  
229  
230  
231  
232  
233  
234  
235  
236  
237  
238  
239  
240  
241  
242  
243  
244  
245  
246  
247  
248  
249  
250  
251  
252  
253  
254  
255  
256  
257  
258  
259  
260  
261  
262  
263  
264  
265  
266  
267  
268  
269

![Figure 2: Three diagrams illustrating cycles. Left: 'Trivial 1-cycle' shows a blue filled circle labeled 's' with the caption '[\gamma] = 0 in H_1'. Middle: 'Nontrivial 1-cycle' shows a red circle around a white hole labeled 'hole' with the caption '[\gamma] \neq 0 in H_1'. Right: 'Order-invariant cycle' shows a green square with arrows on its edges and the caption '[\gamma] independent of order'.](edd10d3006553f0a7a5a7f844ed8cd01_img.jpg)

Figure 2: Three diagrams illustrating cycles. Left: 'Trivial 1-cycle' shows a blue filled circle labeled 's' with the caption '[\gamma] = 0 in H\_1'. Middle: 'Nontrivial 1-cycle' shows a red circle around a white hole labeled 'hole' with the caption '[\gamma] \neq 0 in H\_1'. Right: 'Order-invariant cycle' shows a green square with arrows on its edges and the caption '[\gamma] independent of order'.

Figure 2: **Trivial, nontrivial, and order-invariant cycles.** *Left:* A boundary of a filled region is trivial in  $H_1$ . *Middle:* A loop around a hole cannot bound any 2-chain, so it represents a nontrivial homology class. *Right:* Once a trajectory closes into a cycle, its homology class depends only on the multiset of moves, not their order: order permutations yield the same  $H_1$  class.

the structural survivors of symmetry breaking concentrate probability mass onto persistent cycles and formalize what remains stable under the reduced symmetry subgroup.

**Definition 1** (Residual Invariants under Symmetry Breaking). *Let a system evolve on a state space  $\mathcal{Z}$  with symmetry group  $G$ . Suppose a perturbation  $\varepsilon$  breaks  $G$ -equivariance, reducing the symmetry to a subgroup  $H \subset G$  and forcing selection of a representative state  $\Phi_\varepsilon \in \mathcal{Z}$ . The residual invariants are those structures that remain preserved under  $H$  despite the breaking of  $G$ . Formally, they are equivalence classes of cycles  $[\gamma] \in H_k(\mathcal{Z})$  that are stable under  $H$ -action and persist under perturbations of  $\varepsilon$ .*

Intuitively, residual invariants encode what remains stable after a decision or perturbation: in physics, they correspond to conserved quantities or Goldstone modes Beekman et al. (2019); in topology, to persistent homology classes Edelsbrunner et al. (2008); and in cognition, to cycles that bind chosen outcomes with unchosen alternatives, enabling recall, revision, and reuse Chen & Wilson (2023). This intuition can be formalized by showing that residual invariants emerging from symmetry breaking necessarily take the form of closed cycles, which persist as homology classes and provide the structural foundation of memory.

**Lemma 1** (Symmetry Breaking Generates Invariant Cycles). *Let a system evolve on a state space  $\mathcal{Z}$  with symmetry group  $G$ . Suppose a perturbation  $\varepsilon$  breaks  $G$ -equivariance by forcing the selection of a representative state  $\Phi_\varepsilon$ . Then: 1) The broken symmetry induces residual structures (orbits) invariant under residual transformations  $H \subset G$ . 2) These residual invariants manifest as closed cycles  $\gamma \subset \mathcal{Z}$  stabilized by feedback (i.e.  $\partial\gamma = 0$ ). 3)  $\gamma$  defines a homology class  $[\gamma] \in H_k(\mathcal{Z})$  that is stable under perturbations of  $\varepsilon$ , formalizing memory persistence.*

The proof for the above lemma can be found in Appendix A. This lemma establishes that symmetry breaking inevitably leaves behind residual invariants in the form of cycles, which act as stable memory traces of past selections. To fully understand their cognitive function, one must ask: What advantage does the system gain from organizing dynamics into such closed cycles? The key lies in the fact that cycles identify equivalence classes of trajectories, collapsing many superficially different paths into the same topological invariant Hatcher (2002). In other words, once dynamics are organized into homology classes, prediction and memory no longer depend on the precise order of steps, but only on the closure of the cycle due to the Abelian property of addition operators. This observation leads directly to the following theorem: cycles serve as the structural basis of *order invariance*, ensuring robustness in navigation, perception, action, and more abstract cognitive computations Hawkins (2021).

**Theorem 2** (Cycles Encode Order Invariance). *Let  $(\mathcal{Z}, x_0)$  be a pointed state space (latent manifold or graph) with base state  $x_0$  (“home”). Let  $\mathcal{A} = \{a_1, \dots, a_m\}$  denote a finite set of local moves inducing paths  $\{\alpha_i\}$  starting and ending in a neighborhood of their endpoints. For any finite sequence of moves  $w = a_{i_1} \cdots a_{i_k}$  that yields a cycle  $\gamma_w$  at  $x_0$  (i.e., a homing trajectory), the first homology class  $[\gamma_w] \in H_1(\mathcal{Z}; \mathbb{Z})$  depends only on the multiset of moves used (and their net orientations), not on their order. Equivalently, all order permutations of  $w$  that remain cycles at  $x_0$  determine the same element in  $H_1$ .*

Theorem 2 establishes that once trajectories are organized into cycles, their predictive value no longer depends on the precise ordering of steps but only on the closure of the cycle. This reduction

{5}------------------------------------------------

reflects a deeper topological dichotomy in memory formation. Algebraically, the identity  $\partial^2 = 0$  ensures that boundaries of boundaries vanish Edelsbrunner & Harer (2010): incomplete chains cannot accumulate meaning unless they close, and only closed cycles can survive as invariants. Cognitively, this corresponds to the fact that exploratory fragments either collapse into trivial points (dots) with no relational content, or are stabilized into nontrivial cycles that encode order-invariant memory Babichev et al. (2025). In this sense,  $\partial^2 = 0$  acts as the algebraic filter that separates forgotten scaffolds from consolidated invariants. To make this distinction explicit, we now formalize the roles of  $H_0$  and  $H_1$  in the following lemma (refer to Fig. 2).

**Dot-Cycle Dichotomy.** At the chain level, a “dot” (0-simplex) records isolated content, whereas a “cycle” (1-cycle) captures a closed relation in which endpoints cancel. The rule  $\partial^2 = 0$  formalizes this passage: boundaries of fragments do not compose, but pairwise cancellation at endpoints yields a cycle that survives in homology. Cognitively, this is the move from token to trace Spens & Burgess (2024): contents  $\Phi$  are registered as dots, yet only when linked by contextual relations  $\Psi$  into a closed cycle do they consolidate as durable memory. Details regarding biological implementations can be found in Appendix B.

**Lemma 2** ( $\partial^2 = 0$  Enforces the Dot-cycle Dichotomy). *Let  $C_*(\mathcal{Z})$  denote the chain complex of a neural state space  $\mathcal{Z}$ . The homological identity  $\partial^2 = 0$  implies that: 1) Any open chain  $\sigma \in C_1(\mathcal{Z})$  with  $\partial\sigma \neq 0$  must collapse to a trivial 0-cycle in  $H_0(\mathcal{Z})$ , encoding mere connectivity without relational content. 2) Any closed chain  $\gamma \in C_1(\mathcal{Z})$  with  $\partial\gamma = 0$  defines a homology class  $[\gamma] \in H_1(\mathcal{Z})$ . If  $\gamma$  is not the boundary of a higher-dimensional chain, it represents a nontrivial cycle that persists as a stable memory trace. Thus,  $\partial^2 = 0$  acts as a topological filter: boundaries of boundaries vanish, ensuring that only two outcomes are possible, collapse into trivial dots ( $H_0$ ) or persistence as nontrivial cycles ( $H_1$ ).*

Lemma 2 provides the algebraic gate for memory:  $\partial^2 = 0$  prunes open, order-sensitive fragments and admits only closed loops as meaningful carriers. To connect this structural pruning with predictive power, we now view closure through an information-theoretic lens Cover (1999). When many orderings of the same events are possible, their variability behaves as symmetry-induced noise. Closure collapses these degrees of freedom onto a residual loop, thereby concentrating probability mass on what is repeatable and compressing description length. In effect, cycles are the *sufficient statistics* of paths: once a trajectory closes, order fluctuations become irrelevant for forecasting Friston (2018). The algebraic identity  $\partial^2 = 0$  has an information-theoretic counterpart: broken symmetry reduces entropy by collapsing many equivalent paths into one invariant cycle. The next proposition formalizes this entropy-prediction link via symmetry breaking that leaves an invariant cycle.

**Proposition 1** (Entropy Minimization Improves Prediction by Cycles). *Let a system generate trajectories in a state space  $\mathcal{Z}$ . Suppose initially, the system has a symmetry  $G$  (e.g. different orders of moves or observations are treated as equivalent). A perturbation breaks this full symmetry, but leaves behind an invariant cycle  $\gamma \subset \mathcal{Z}$  with  $\partial\gamma = 0$ . Then we have: 1) The cycle  $\gamma$  encodes what is stable across different orders or paths; 2) Predictions about future outcomes need only depend on  $\gamma$  (and context), not on the detailed order of past steps; 3) Thus, broken symmetry reduces noise from order-specific variations and improves prediction by preserving only what remains invariant.*

Proposition 1 identifies *what* survives order variability: the residual invariant cycle  $\gamma$ . To pass from structure to statistics, note that discarding order-specific fluctuations is equivalent to an entropy drop: probability mass that was spread over many orderings is reassigned to the closed loop that summarizes them. In a non-ergodic system, this manifests as *measure concentration* on the surviving cycles Ledoux (2001). Therefore, predictive sufficiency (dependence only on  $[\gamma]$ ) coincides with entropy reduction (symmetry breaking) and with the asymptotic concentration of  $\mu_t$  on invariant classes. The following corollary makes this equivalence explicit.

**Corollary 1** (Prediction as Concentration on Cycles). *For a non-ergodic system  $(X, T)$ , prediction is possible iff the probability measure  $\mu_t$  concentrates on invariant cycles  $[\gamma] \in H_k(X)$  as  $t \rightarrow \infty$ . Equivalently, Prediction  $\iff$  Entropy Reduction via Symmetry Breaking  $\iff$  Measure Concentration on Cycles. Therefore, the structural invariants revealed by broken symmetry are precisely the carriers of predictive information, ensuring reliable memory and generalization across time.*

Corollary 1 identifies *what* supports prediction: global dynamics must collapse onto persistent cycles. How such cycles arise is local: symmetry breaking forces a choice among equivalent alterna-

{6}------------------------------------------------

tives, and the discarded possibilities are reorganized into recurrent loops. These loops stabilize the selected outcome while retaining counterfactual access, thereby creating the invariant structures that concentrate probability mass and convert uncertainty into predictive stability.

## 4 MEMORY-AMORTIZED INFERENCE FOR TOPOLOGICAL CLOSURE

To operationalize this picture in cognition, we adopt the *Context-Content Uncertainty Principle (CCUP)* Li (2025a): stable memory traces correspond to low-entropy *content variables*  $\Phi$  (persistent homological cycles), while transient variability is captured by high-entropy *context variables*  $\Psi$ . In what follows, we show how *Memory-Amortized Inference (MAI)* implements cycle formation by holding  $\Phi$  fixed as reusable structure and adapting  $\Psi$  until residual boundaries cancel ( $\partial^2 = 0$ ), thereby achieving topological closure.

**Content variable  $\Phi$  as low-entropy homology.** Within CCUP, the content variable  $\Phi$  corresponds to information that is both specific and stable. Mathematically,  $\Phi$  is identified with nontrivial homology classes: cycles  $[\gamma] \in H_k(\mathcal{Z})$  that cannot be reduced to boundaries. Such cycles encode persistent, low-entropy structures because many possible trajectories or micro-states collapse into the same equivalence class. In neural terms,  $\Phi$  reflects patterns of activity that recur reliably across different contexts, such as a learned motor primitive, a familiar spatial route, or a well-established object representation. By filtering away order-dependent variability,  $\Phi$  preserves only the invariant relational structure that remains after symmetry breaking. This makes  $\Phi$  the stable substrate of memory and the carrier of predictive power: once identified, it can be recalled, reused, and composed into higher-order cognitive structures.

**Context variable  $\Psi$  as high-entropy scaffolding.** In contrast, the context variable  $\Psi$  captures the transient, exploratory, and often noisy aspects of cognition. Topologically,  $\Psi$  is associated with trivial cycles or short-lived features in the persistence barcode: loops that quickly vanish under perturbation or deformation. These cycles act as *scaffolding*, supporting the discovery and stabilization of  $\Phi$  but not themselves persisting as memory. In information-theoretic terms,  $\Psi$  is high-entropy: it reflects a large space of possibilities, many of which will be pruned away as the system concentrates its measure on low-entropy  $\Phi$  structures. Biologically,  $\Psi$  is implemented by slow, contextual rhythms (e.g. theta oscillations) or exploratory neural activity that supplies diverse scaffolds for binding. Through dynamic alignment and phase-resetting, these high-entropy contextual structures are folded into persistent content loops, allowing cognition to maintain flexibility while ensuring stability in memory formation.

Taken together,  $\Phi$  and  $\Psi$  form a complementary pair:  $\Phi$  supplies the order-invariant backbone that can be reused across contexts, while  $\Psi$  provides the exploratory variability from which such backbones are discovered. CCUP therefore prescribes an operational loop: hold candidate content steady, let context range, and accept only those pairings that close into cycles (i.e., cancel boundaries). This suggests a general law of cognitive economy in which *structure leads* and *specificity follows*: stable invariants guide, while transient scaffolds adapt until closure is achieved. We now make this heuristic precise as a principled statement.

**Principle 3** (Structure-Before-Specificity Principle). *Let  $\Phi$  denote low-entropy content variables corresponding to nontrivial homology classes  $[\gamma] \in H_k(\mathcal{Z})$ , and let  $\Psi$  denote high-entropy contextual scaffolds corresponding to transient or trivial cycles. Then cognition obeys the following principle: 1) (Structure before specificity) Stable content  $\Phi$  arises from nontrivial cycles that persist across perturbations. These cycles define the backbone of memory and predictive power. 2) (Specificity from scaffolding) Context  $\Psi$  supplies a high-entropy exploratory substrate: transient cycles that may collapse but provide the variability needed to refine, adapt, or recombine  $\Phi$ . 3) (Dynamic alignment) The interaction of  $\Psi$  and  $\Phi$  via cycle closure ( $\partial^2 = 0$ ) ensures that contextual exploration is funneled into persistent content loops, transforming noisy scaffolds into stable memory traces.*

The above principle prescribes an operational recipe: stabilize  $\Phi$  as reusable structure and let  $\Psi$  explore until closure cancels residual boundaries. *Memory-amortized inference (MAI)* is the algorithmic embodiment of this recipe. Instead of re-solving each inference problem from scratch, MAI retrieves a candidate invariant (a cycle-level template for  $\Phi$ ), then adapts  $\Psi$  until the pair  $(\Psi, \Phi)$

{7}------------------------------------------------

378 closes (i.e.,  $\partial^2 = 0$ ), pruning order-specific noise. In effect,  $\Phi$  functions as a low-entropy prior over  
379 solutions, while  $\Psi$  supplies the high-entropy search that is guided and terminated by topological  
380 closure. We formalize MAI as a general strategy for reducing the computational cost of inference by  
381 storing and reusing structured latent representations. The key idea is to construct a memory of prior  
382 inference results such that new inference problems can be approximated by querying and adapting  
383 from this memory, rather than solving the full problem from scratch. Let  $\Psi \in \mathcal{X}$  denote the ob-  
384 servable context and  $\Phi \in \mathcal{S}$  the latent content to be inferred. Let  $\mathcal{L}(\Psi, \Phi)$  denote a loss or cost  
385 function encoding the fidelity or predictive value of  $\Phi$  under context  $\Psi$ . We assume that inference  
386 corresponds to solving the following optimization:  $\Phi^* = \arg \min_{\Phi \in \mathcal{S}} [\mathcal{L}(\Psi, \Phi)]$ . Formally, we start  
387 with the following definition (refer to Fig. 3).

388 **Definition 2** (Memory-Amortized Inference). *Let  $\mathcal{M} = \{(\Psi^{(i)}, \Phi^{(i)})\}_{i=1}^N$  be a memory of prior  
389 context-content pairs, and let  $\mathcal{R} : \mathcal{X} \times \mathcal{M} \rightarrow \mathcal{S}$  be a retrieval-and-adaptation operator and  $\mathcal{F} : \mathcal{S} \times \mathcal{X} \rightarrow \mathcal{S}$  be the bootstrapping update operator implemented via generative simulation. Inference  
390 is said to be memory-amortized if it is formulated as a structural cycle between content  $\Phi$  and  
391 context  $\Psi$ , where memory acts as a reusable substrate for inference:  $\Phi_{t+1} = \mathcal{F}(\Phi_t, \Psi_t)$ ,  $\Phi_t \approx$   
392  $\mathcal{R}(\Phi_{t+1}, \Psi_t)$  in lieu of directly optimizing  $\Phi^*$ , such that the expected cost satisfies  $\mathbb{E}_\Psi [\mathcal{L}(\Psi, \hat{\Phi})] \leq$   
393  $\mathbb{E}_\Psi [\mathcal{L}(\Psi, \Phi^*)] + \varepsilon$ , for some amortization gap  $\varepsilon \ll \mathcal{L}(\Psi, \cdot)$ , and where the runtime cost of  $\mathcal{R}$  is  
394 substantially lower than full inference.*

### Memory-Amortized Inference Cycle

![Diagram of the Memory-Amortized Inference Cycle. It shows a flow starting from Context Psi_t to Bootstrapping Phi_t = F(Phi_t, Psi_t) to Predictive Update Phi_{t+1}. A dashed line labeled 'reuse' goes from Predictive Update to Retrieval Phi_t = R(Phi_{t+1}, Psi_t). A dashed line labeled 'M = {(Psi^{(i)}, Phi^{(i)})}' goes from Retrieval back to Context Psi_t.](35a7554182eb055209552843f341a1ae_img.jpg)

```

graph TD
    Context["Context  
Ψ_t"] --> Bootstrapping["Bootstrapping  
Φ_t = F(Φ_t, Ψ_t)"]
    Bootstrapping --> PredictiveUpdate["Predictive Update  
Φ_{t+1}"]
    PredictiveUpdate -.-> Retrieval["Retrieval  
Φ_t = R(Φ_{t+1}, Ψ_t)"]
    Retrieval -.-> Context
    Retrieval -.-> Memory["M = {(Ψ^{(i)}, Φ^{(i)})}"]
    style Context fill:#d9d9ff,stroke:#333,stroke-width:1px
    style Bootstrapping fill:#ffcc99,stroke:#333,stroke-width:1px
    style PredictiveUpdate fill:#ffffff,stroke:#333,stroke-width:1px
    style Retrieval fill:#ccffcc,stroke:#333,stroke-width:1px
  
```

Diagram of the Memory-Amortized Inference Cycle. It shows a flow starting from Context Psi\_t to Bootstrapping Phi\_t = F(Phi\_t, Psi\_t) to Predictive Update Phi\_{t+1}. A dashed line labeled 'reuse' goes from Predictive Update to Retrieval Phi\_t = R(Phi\_{t+1}, Psi\_t). A dashed line labeled 'M = {(Psi^{(i)}, Phi^{(i)})}' goes from Retrieval back to Context Psi\_t.

Figure 3: Cycle of MAI. Instead of recomputing  $\Phi^* = \arg \min \mathcal{L}(\Psi, \Phi)$ , the system reuses prior  
trajectories:  $\Phi_{t+1}$  and  $\Psi_t$  guide memory-based retrieval via  $\mathcal{R}$ , and bootstrapping  $\mathcal{F}$  updates the  
latent state  $\Phi_t$ . The process forms a self-consistent loop grounded in structured memory.

**The Retrieval-and-Adaptation Operator  $\mathcal{R}$ .** The retrieval-and-adaptation operator  $\mathcal{R} : \mathcal{X} \times \mathcal{M} \rightarrow \mathcal{S}$   
serves as the core mechanism by which inference avoids re-computation. Given an input  
query (typically latent or perceptual),  $\mathcal{R}$  retrieves relevant elements from the memory  $\mathcal{M} =$   
 $\{(\Psi^{(i)}, \Phi^{(i)})\}_{i=1}^N$  and performs a lightweight adaptation to generate a candidate solution  $\Phi$ . Op-  
erationally,  $\mathcal{R}$  consists of two stages: 1) **Retrieval:** Identify a relevant subset of memory entries  
 $\{(\Psi^{(j)}, \Phi^{(j)})\} \subset \mathcal{M}$  based on similarity to the current context  $\Psi_t$ . This can be performed via  
kernel-based attention, similarity search in latent space, or topological proximity under homological  
constraints. 2) **Adaptation:** Modulate or interpolate the retrieved  $\Phi^{(j)}$  values conditioned on  $\Psi_t$ ,  
resulting in a candidate  $\hat{\Phi}_t = \mathcal{R}(\Phi_{t+1}, \Psi_t)$ . This step often involves gradient-free adjustments (e.g.,  
feature warping, parameter blending) and is significantly cheaper than full inference.

The retrieval-and-adaptation operator  $\mathcal{R}$  in MAI generalizes the classical notion of key-value memory  
used in neural attention and memory-augmented models. In conventional key-value memory  
systems Weston et al. (2014); Sukhbaatar et al. (2015), memory is structured as a set of key-value  
pairs:  $\mathcal{M} = \{(\Psi^{(i)}, \Phi^{(i)})\}_{i=1}^N$ , where a context vector  $\Psi$  acts as a key to retrieve values  $\Phi$  via  
similarity-based soft addressing:  $\hat{\Phi} = \sum_i w_i \Phi^{(i)}$ ,  $w_i = \frac{\exp(-d(\Psi, \Phi^{(i)}))}{\sum_j \exp(-d(\Psi, \Phi^{(j)}))}$ . This model sup-  
ports one-shot retrieval but lacks structural consistency or bidirectional inference. By contrast,  
the operator  $\mathcal{R}(\Phi_{t+1}, \Psi_t; \mathcal{M})$  in MAI performs a more general operation: it *retrieves* a candi-  
date latent representation from memory based on both the current context  $\Psi_t$  and a target la-  
tent code  $\Phi_{t+1}$ , and then *adapts* it to produce a consistent approximation of the preceding latent

{8}------------------------------------------------

state  $\Phi_t$ . This supports inference in reverse time and satisfies the memory-amortized constraint:  $\Phi_t \approx \mathcal{R}(\Phi_{t+1}, \Psi_t)$ ,  $\Phi_{t+1} = \mathcal{F}(\Phi_t, \Psi_t)$ . The operator  $\mathcal{R}$  thereby enables cycle-consistent inference, crucial for temporal coherence and structural reuse. Unlike key-value memory, which operates over flat vector spaces,  $\mathcal{R}$  may act over structured memory (e.g., graphs, latent manifolds, or topological complexes) and is inherently adaptive. A summary of the distinction is provided below:

**The Bootstrapping Update Operator  $\mathcal{F}$ .** The bootstrapping operator  $\mathcal{F} : \mathcal{S} \times \mathcal{C} \rightarrow \mathcal{S}$  governs the internal dynamics of inference by iteratively updating the latent content representation  $\Phi_t$  given the context  $\Psi_t$ . It defines a recurrence:  $\Phi_{t+1} = \mathcal{F}(\Phi_t, \Psi_t)$ , where  $\mathcal{F}$  encodes the system’s structural prior, capturing the directionality, topology, and dynamic consistency of inference over time. Unlike standard update rules that minimize a loss from scratch,  $\mathcal{F}$  performs bootstrapping: each update is initialized from a prior memory-induced state, often already close to the optimal solution due to cycle recurrence. Here are several key properties of  $\mathcal{F}$ : 1) **Cycle-Consistency:** If  $(\Phi_t, \Psi_t) \in \gamma$  for some memory cycle  $\gamma \subset \mathcal{Z}$ , then  $\Phi_{t+T} \approx \Phi_t$ , enabling amortization via structural recurrence. 2) **Structural Biasing:** Updates follow latent paths constrained by prior topology (e.g., flow fields over homology classes or attention-modulated latent graphs), enforcing low-entropy generalization. 3) **Minimal Cost Gradient:** Because the initialization  $\Phi_t$  already lies near an attractor, the subsequent update  $\Phi_{t+1}$  requires only a small corrective shift, further amortizing the inference process.

The bootstrapping update operator  $\mathcal{F}$  in MAI is structurally analogous to the *half-step down* trick used in Q-learning Watkins & Dayan (1992) and temporal difference (TD) methods Sutton & Barto (1998). In Q-learning, the value function is updated by approximating the current value via a one-step lookahead:  $Q(s_t, a_t) \leftarrow r_t + \gamma \max_{a'} Q(s_{t+1}, a')$ , which yields the approximation  $Q(s_t) \approx Q(s_{t+1})$ . This forward-directed value propagation allows reinforcement learning agents to estimate long-term outcomes without simulating entire trajectories. By contrast, MAI reverses the time direction: the update operator  $\mathcal{F}$  bootstraps latent inference forward using structured memory and contextual cues:  $\Phi_{t+1} = \mathcal{F}(\Phi_t, \Psi_t)$ , and this is inverted by retrieval:  $\Phi_t \approx \mathcal{R}(\Phi_{t+1}, \Psi_t)$ . This dual relationship forms the backbone of the MAI half-step trick: the current latent content  $\Phi_t$  generates the next-step prediction  $\Phi_{t+1}$ , which in turn can be used to reconstruct  $\Phi_t$ . While Q-learning bootstraps value via reward-driven transitions, MAI bootstraps inference through latent memory and context, yielding a cycle-consistent structure that reduces entropy. Both approaches use bootstrapping to manage uncertainty and amortize computational cost, but in opposite directions, highlighting a deeper time-reversed duality between learning and inference (refer to Appendix C). This recursive formulation enables stable inference trajectories that converge toward contextually relevant attractors, effectively amortizing the cost of learning across time. The underlying dynamics of this process can be formalized as a contractive map over a structured retrieval cycle, leading to provable convergence under mild assumptions. We now state the following result, which captures the fixed-point stability of the MAI loop:

**Proposition 2** (Topological Closure via Structural Recursion). *Let  $\mathcal{T}(\Phi, \Psi) := \mathcal{F}(\mathcal{R}(\Phi, \Psi), \Psi)$  be the composite update in MAI. Suppose  $\mathcal{T}$  is contractive in its first argument for fixed context  $\Psi$ . Then there exists a unique fixed point  $\Phi^*$  such that:  $\Phi^* = \mathcal{T}(\Phi^*, \Psi)$ . Moreover, the inference trajectory  $\{\Phi_t\}_{t=0}^{\infty}$  forms a closed loop in latent space as:  $\lim_{t \rightarrow \infty} \|\Phi_t - \Phi^*\| = 0$ . This latent recurrence corresponds to a nontrivial 1-cycle, representing topological closure in the MAI manifold.*

Proposition 2 establishes closure at the level of latent dynamics: a contractive structural recursion yields a fixed point and a recurrent trajectory that “homes” to it, i.e., a geometric 1-cycle in the MAI manifold. We now lift this geometric closure to the algebraic level of chains. Specifically, the same retrieve–update loop can be read as a chain-homotopy correction that cancels residual boundaries in the context–content complex. In this view, latent recurrence (fixed-point closure) and homological recurrence (boundary cancellation) are two faces of the same mechanism. The next theorem formalizes this equivalence by showing that MAI implements topological closure via  $\partial^2 = 0$  (its proof can be found in Appendix A).

**Theorem 3** (MAI as Computational Realization of Topological Closure). *Let  $(\mathcal{C}_*, \partial)$  be a chain complex encoding context–content relations, with  $\Psi$  as high-entropy scaffolds and  $\Phi$  as candidate content variables. In Memory-Amortized Inference (Definition 1), the iterative cycle  $\Phi_{t+1} = \mathcal{F}(\Phi_t, \Psi_t)$ ,  $\Phi_t \approx \mathcal{R}(\Phi_{t+1}, \Psi_t)$  implements a homotopy update that cancels residual boundaries:  $\partial(\Psi_t, \Phi_t) \mapsto \partial(\Psi_{t+1}, \Phi_{t+1}) \approx 0$ . Thus, amortization prunes misaligned, order-dependent fragments (open boundaries) and preserves only reproducible cycles  $[\gamma] \in H_k(\mathcal{C}_*)$ . Equivalently, MAI realizes topological closure by enforcing  $\partial^2 = 0$  in computation: context–content updates that fail to close are discarded, while those that re-enter memory persist as invariants.*

{9}------------------------------------------------

## REFERENCES

- Philip W Anderson. More is different: Broken symmetry and the nature of the hierarchical structure of science. *Science*, 177(4047):393–396, 1972.
- Anton Ayzenberg, Thomas Gebhart, German Magai, and Grigory Solomadin. Sheaf theory: from deep geometry to deep learning. *arXiv preprint arXiv:2502.15476*, 2025.
- Andrey Babichev, Vladimir Vashin, and Yuri Dabaghian. Spaces and sequences in the hippocampus: a homological perspective. *bioRxiv*, 2025.
- Aron Beekman, Louk Rademaker, and Jasper Van Wezel. An introduction to spontaneous symmetry breaking. *SciPost Physics Lecture Notes*, pp. 011, 2019.
- M.Á. Belluscio, K. Mizuseki, R. Schmidt, R. Kempter, and G. Buzsáki. Cross-frequency phase–phase coupling between theta and gamma oscillations in the hippocampus. *Journal of Neuroscience*, 32(2):423–435, 2012.
- Guo-qiang Bi and Mu-ming Poo. Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience*, 18 (24):10464–10472, 1998. doi: 10.1523/JNEUROSCI.18-24-10464.1998.
- György Buzsáki. The hippocampo-neocortical dialogue. *Cerebral cortex*, 6(2):81–92, 1996.
- György Buzsáki and Xiao-Jing Wang. Mechanisms of gamma oscillations. *Annual Review of Neuroscience*, 35:203–225, 2012. doi: 10.1146/annurev-neuro-062111-150444.
- Ryan T. Canolty and Robert T. Knight. The functional role of cross-frequency coupling. *Trends in Cognitive Sciences*, 14(11):506–515, 2010. doi: 10.1016/j.tics.2010.09.001.
- Ryan T. Canolty, Edward Edwards, Sarang S. Dalal, Alireza Soltani, Srikantan S. Nagarajan, Heidi E. Kirsch, Michel S. Berger, Nicholas M. Barbaro, and Robert T. Knight. High gamma power is phase-locked to theta oscillations in human neocortex. *Proceedings of the National Academy of Sciences*, 103(19):9674–9679, 2006. doi: 10.1073/pnas.0600418103.
- Natalia Caporale and Yang Dan. Spike timing–dependent plasticity: a hebbian learning rule. *Annual Review of Neuroscience*, 31:25–46, 2008a. doi: 10.1146/annurev.neuro.31.060407.125639.
- Natalia Caporale and Yang Dan. Spike timing–dependent plasticity: a hebbian learning rule. *Annu. Rev. Neurosci.*, 31(1):25–46, 2008b.
- Zhe Sage Chen and Matthew A Wilson. How our understanding of memory replay evolves. *Journal of Neurophysiology*, 129(3):552–580, 2023.
- Isaac P Cornfeld, Sergei Vasilevich Fomin, and Yakov Grigor’evic Sinai. *Ergodic theory*, volume 245. Springer Science & Business Media, 2012.
- Thomas M Cover. *Elements of information theory*. John Wiley & Sons, 1999.
- S Davatollahgh, A Sheykhi, and MH Zarei. ‘it from bit’: How does information shape the structures in the universe? In *Proceedings A*, volume 480, pp. 20240024. The Royal Society, 2024.
- Kamran Diba and György Buzsáki. Forward and reverse hippocampal place-cell sequences during ripples. *Nature Neuroscience*, 10(10):1241–1242, 2007. doi: 10.1038/nn1961.
- Herbert Edelsbrunner and John Harer. *Computational topology: an introduction*. American Mathematical Soc., 2010.
- Herbert Edelsbrunner, John Harer, et al. Persistent homology-a survey. *Contemporary mathematics*, 453(26):257–282, 2008.
- R. Douglas Fields. A new mechanism of nervous system plasticity: activity-dependent myelination. *Nature Reviews Neuroscience*, 16(12):756–767, 2015. doi: 10.1038/nrn4023.
- David J. Foster and Matthew A. Wilson. Reverse replay of behavioural sequences in hippocampal place cells during the awake state. *Nature*, 440:680–683, 2006. doi: 10.1038/nature04587.

 Rest of paper (reference and Appendix) is removed.