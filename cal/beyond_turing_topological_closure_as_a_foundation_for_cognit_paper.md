# BEYOND TURING: TOPOLOGICAL CLOSURE AS A FOUNDATION FOR COGNITIVE COMPUTATION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Classical models of computation, epitomized by the Turing machine, are grounded
in _enumeration_ : syntactic manipulation of discrete symbols according to formal
rules. While powerful, such systems are intrinsically vulnerable to G¨odelian
incompleteness and Turing undecidability, since truth and meaning are sought
through potentially endless symbolic rewriting. We propose an alternative foundation for non-enumerative computation based on _topological closure_ of semantic
structures. In this view, cognition operates by promoting transient fragments into
closed cycles, where _∂_ [2] = 0 ensures that only invariants persist. This shift reframes computation from _syntax_ to _structure_ : memory and reasoning arise not
by enumerating all possibilities, but by stabilizing relational invariants that survive perturbations and generalize across contexts. We formalize this principle
through the dot–cycle dichotomy: dots or trivial cycles ( _H_ 0) serve as high-entropy
scaffolds for exploration, while nontrivial cycles ( _H_ 1 and higher) encode lowentropy invariants that persist as memory. Extending this perspective, we show
how Memory-Amortized Inference (MAI) implements an anti-enumerative principle by storing homological equivalence classes rather than symbolic traces, yielding robust generalization, energy efficiency, and structural completeness beyond
Turing-style models. We conclude that _topological_ _closure_ provides a unifying
framework for perception, memory, and action, and a candidate foundation for
cognitive computation that transcends the limits of enumeration.


1 INTRODUCTION


Since the early 20th century, formal models of computation have been grounded in _enumeration_ .
The Turing machine, along with its close relatives in the Church–Turing framework, exemplifies
this paradigm: computation is conceived as the syntactic manipulation of discrete symbols on an
infinite tape, with new results obtained only through the stepwise application of formal rules Turing
(1936). This model has proven enormously successful, forming the foundation of digital computing, automata theory, and modern complexity classes. Deep learning architectures, despite their
connectionist implementation, inherit this enumerative character: generalization emerges by statistical interpolation over enumerated training examples, and inference requires repeated evaluation
across contexts Goodfellow et al. (2016). Yet the power of enumerative systems is matched by their
intrinsic limitations. G¨odel’s incompleteness theorem demonstrates that no sufficiently expressive
formal system can be both complete and consistent: there will always exist true statements that cannot be proven within the system. Turing’s halting problem further establishes that no algorithm can
decide, in finite time, whether arbitrary programs will terminate Sipser (1996). Both results reveal
a deeper structural fragility: _enumeration can never guarantee closure_ . Each attempt to list or decide the totality of possible outcomes leaves residual boundaries, open fragments that escape formal
capture. This fragility manifests as brittleness in symbolic AI, combinatorial explosion in search
Minsky (1961), and distributional failures in data-driven models. Enumeration fails because it cannot stabilize residual structures, open chains proliferate without ever closing. Topological closure
reframes this failure: what enumeration leaves dangling, closure promotes into invariants.


In this paper, we propose an information topological framework for intelligence in which _cycle_
_closure_ is the fundamental mechanism of memory. Building on the first principle, we argue that
memory is best understood not as a static store of representations, but as the ability to _re-enter and_
_traverse latent cycles_ in neural state space. We identify these invariant cycles as the natural carriers


1


of meaning across scales: they act as _alignment checkpoints_ between context (Ψ) and content (Φ),
filtering out order-specific noise, enforcing closure, and preserving only what remains consistent
across variations. A key principle underlying this framework is the _dot-cycle_ _dichotomy_ : trivial
cycles collapse to dots ( _H_ 0), serving as transient contextual scaffolds (Ψ), while nontrivial cycles
( _H_ 1 and higher) encode low-entropy content invariants (Φ) that persist as memory. This dichotomy
clarifies how cognition achieves both adaptability and stability: dots support exploration, while
cycles carry persistent knowledge across contexts. From this perspective, cognition is not tapebased symbol manipulation but the promotion of transient fragments into closed cycles that survive
perturbation and generalize across contexts. This shift from _syntax_ to _structure_ reframes memory,
learning, and reasoning as processes of stabilizing invariants, not enumerating sequences. Under
this new conceptual framework, we develop the following arguments:


    - We explore the physical origin of intelligence inspired by the first clue in Wheeler (2018):
_∂_ [2] = 0 _⇒_ Cycles (invariants) _⇒_ Memory _⇒_ Prediction (intelligence).

    - We introduce the _dot-cycle_ _dichotomy_ : dots ( _H_ 0) encode disconnected fragments, while
cycles ( _H_ 1 and higher) represent nontrivial order invariants that persist as memory.

    - We introduce _Structure-before-Specificiy_ principle as the guidance of memory organization.
Structural content is represented by low-entropy homology and specific context serves as
high-entropy scaffolding.

    - We show how _Memory-Amortized_ _Inference_ _(MAI)_ implements a context-content uncertainty principle (CCUP) by bootstrapping and retrieval operators, yielding energy efficiency
and robust generalization.


2 MOTIVATION: INTELLIGENCE AS TOPOLOGICAL CLOSURE


A unifying way to interpret both G¨odel’s incompleteness theorem and Turing’s halting problem is
to see them as demonstrations of the failure of _countable closure_ . Any attempt to exhaustively enumerate truths or procedures inevitably leaves a residue, a diagonal element, an undecidable program,
that lies outside the reach of the list. From a topological perspective, this means that enumerations
generate fragments that remain open boundaries, unable to close into global invariants. What escapes enumeration is not accidental but principled: closure requires invariants beyond counting.
This reinterpretation shifts the focus from the fragility of syntactic lists to the robustness of semantic cycles. The implication is profound: if Wheeler’s dictum It-from-Bit Wheeler (2018) highlights
the informational substrate of reality, then for intelligence the relevant unit is not the fleeting bit but
the persistent _cycle_ that survives across variations Davatolhagh et al. (2024). Formally, we have

**Principle 1** (First Principle of Intelligence) **.** _Intelligence is the capacity to stabilize invariants by cy-_
_cle closure. At its core, cognition operates by minimizing joint context-content uncertainty H_ (Ψ _,_ Φ) _,_
_eliminating dangling boundaries and promoting them into closed cycles._ _These cycles constitute the_
_fundamental units of meaning, memory, and prediction._


Our guiding claim is that _cycle is all you need_ : the organization of cognition, memory, and abstract
thoughts in neural systems follows from the universal role of cycles as the algebraic residue of
broken symmetry and the topological skeleton of information flow. This claim is supported by
the hierarchical organization of cycles in mammalian brains, such as _Theta–gamma_ _nesting_ (e.g.,
hippocampus–entorhinal Buzs´aki (1996)) and perception-action cycles Fuster (2004). In the spirit
of Wheeler Wheeler (2018), we propose the following four No’s for cognition.


1. **No** **isolated** **information.** Bits are never standalone: they acquire meaning only through
relations that close into cycles. Information without recurrence dissipates as noise.

2. **No privileged order.** The cognitive system must be robust to permutations of local steps.
What matters is closure into a cycle, not the linear order of micro-events.

3. **No specificity before structure.** Persistent structures must stabilize first as the backbone of
memory and prediction, while contextual specificities become scaffolding later to provide
adaptive flexibility.

4. **No** **prediction** **without** **invariance.** Forecasting future states requires reducing entropy
by filtering order-dependent variations; only invariant cycles can stabilize the predictive
substrate.


2


**Open chain** _⇒_ **dot in** _H_ 0 _∂_ [2] = 0 **(boundary of boundary vanishes)** **Closed cycle** _⇒_ **class in** _H_ 1


Figure 1: _∂_ [2] = 0 **enforces the dot-cycle dichotomy.** _Left:_ An open chain _σ_ has a nonzero boundary
_∂σ_ and collapses to a dot (class in _H_ 0), carrying no relational content. _Middle:_ The boundary
operator squares to zero: _∂_ ( _∂σ_ ) = 0. _Right:_ A closed chain _γ_ with _∂γ_ = 0 persists as a homology
class [ _γ_ ] _∈_ _H_ 1, i.e., a cycle that encodes order-invariant structure.


**From** **constraints** **to** **clues.** These four principles define cognition as a non-ergodic information
process Walters (2000): rather than averaging over all possible trajectories, the mind concentrates
its dynamics onto recurrent, invariant cycles that persist across perturbations. Taken together, the
Four No’s funnel cognition toward recurrent organization: items must close into cycles (no isolated
information), be insensitive to micro-order (no privileged order), support re-entry (no static storage),
and stabilize invariants for prediction (no prediction without invariance). The lightest formalism
that enforces all four at once is the chain complex with boundary operator _∂_ Hatcher (2002): its
nilpotency, _∂_ [2] = 0, cancels stray endpoints so that only closed traversals remain. This is the key
new insight underlying the dot-cycle dichotomy, as shown in Fig. 1, and it sets up our first clue.

**Theorem 1** (The Boundary of a Boundary Vanishes) **.** _Under the First Principle, intelligence is re-_
_alized through cycle closure._ _This closure is only possible because the boundary operator ∂_ _satisfies_
_the_ _fundamental_ _identity_ _∂_ [2] = 0 _._ _That_ _is,_ _the_ _boundary_ _of_ _a_ _boundary_ _vanishes._ _Cognitively,_ _this_
_law ensures that when cognition promotes boundaries into cycles, no further inconsistencies remain_
_at_ _the_ _next_ _level:_ _every_ _open_ _edge_ _is_ _paired,_ _every_ _fragment_ _canceled._ _This_ _guarantees_ _the_ _exis-_
_tence of stable invariants (cycles), which are the carriers of meaning, memory, and communication._
_Therefore,_ _∂_ [2] = 0 _constitutes the_ First Clue _of intelligence:_ _coherence arises because boundaries_
_consistently vanish when lifted, enabling cycles to persist._


The vanishing of boundaries guarantees that what remains in memory is not arbitrary fragments
but coherent cycles: the minimal invariants that bind context and content into an intelligible whole.
From a computational standpoint, this marks a profound departure from the Turing paradigm. Traditional machines rely on symbolic tokens and sequential operations, where meaning is assigned
externally to states of a register or tape. In contrast, a cycle-based architecture derives meaning
intrinsically from topological closure: invariants are not “written” into memory but emerge from the
very dynamics of neural interaction Gerstner et al. (2014). This dot–cycle dichotomy, _where trivial_
_cycles collapse and only nontrivial cycles persist_, provides a natural mechanism for error correction,
generalization, and energy efficiency without requiring exhaustive symbolic manipulation or gradient descent over high-dimensional parameter spaces. Rather than preserving measure by averaging
over all paths, intelligent systems learn to _concentrate_ probability mass onto order-invariant cycles
(i.e., cycle-preserving structure replaces measure-preserving flow).

**Example 1** (Toy Navigation Loop) **.** _In a_ 5 _×_ 5 _grid with a square obstacle (a “hole”), trajectories_
_that poke the obstacle and backtrack are open_ 1 _-chains (∂σ_ = 0 _) and collapse to trivial H_ 0 _“dots.”_
_By_ _contrast,_ _any_ _homing_ _route_ _that_ _circles_ _the_ _hole_ _and_ _returns_ _to_ _start_ _yields_ _a_ _closed_ 1 _-chain_ _γ_
_with ∂γ_ = 0 _and_ [ _γ_ ] _̸_ = 0 _in H_ 1 _._ _Crucially, reordering the same edges (e.g., north-first vs. east-first)_
_produces the_ same _class_ [ _γ_ ] _:_ _the loop is order-invariant and reusable as a navigation template._


3 MEMORY AS STRUCTURED TRAJECTORIES IN THE LATENT SPACE


Classical ergodic theory is built on the notion of a measure-preserving transformation Walters
(2000). A dynamical system ( _X, B, µ, T_ ) consists of a probability space ( _X, B, µ_ ) and a measurable
transformation _T_ : _X_ _→_ _X_ satisfying _µ_ ( _T_ _[−]_ [1] _A_ ) = _µ_ ( _A_ ) _,_ _∀A ∈B_ . This measure invariance guarantees that long-term time averages along almost every trajectory coincide with ensemble averages


3


with respect to _µ_ . In this setting, entropy (e.g., Kolmogorov-Sinai entropy Cornfeld et al. (2012))
quantifies the unpredictability of the evolution under the assumption of ergodicity. Intelligent systems, however, are fundamentally non-ergodic: they retain memory, exhibit path dependence, and
actively reduce uncertainty. In such systems, the measure _µ_ is not preserved, but typically _concen-_
_trated_ onto lower-dimensional recurrent structures through learning and adaptation Spisak & Friston
(2025). This concentration corresponds to entropy minimization rather than entropy conservation.


We propose that the appropriate generalization of “measure-preservation” in the non-ergodic setting
is _cycle-preservation_ . That is, while probability measures are not conserved globally, the system
preserves _topological_ _invariants_ encoded in cycles that represent memory traces and recurrent behavioral motifs Gromov (1999). Formally, let ( _X, T_ ) be a discrete-time dynamical system on a
topological state space _X_ . A _k_ -cycle is a chain _γ_ _∈_ _Zk_ ( _X_ ) satisfying _∂γ_ = 0. Under the induced
map _T∗_ on chains, invariance of _γ_ requires that _T∗γ_ _−_ _γ_ = _∂β_ for some ( _k_ + 1)-chain _β_ . Equivalently, [ _T∗γ_ ] = [ _γ_ ] in _Hk_ ( _X_ ), where _Hk_ ( _X_ ) denotes the k-th homology group of the topological
space _X_, so that _γ_ is invariant up to homology class. In this way, although trajectories deform under
dynamics (e.g., refer to the example of Wilson-Cowan model below), the _memory_ encoded by the
homology class persists.

**Example** **2** (Wilson-Cowan Model) **.** _The_ _Wilson-Cowan_ _system_ _Wilson_ _&_ _Cowan_ _(1972)_ _E_ [˙] =
_−E_ + _S_ ( _weeE_ _−_ _weiI_ + _P_ ) _,_ _I_ [˙] = _−I_ + _S_ ( _wieE_ _−_ _wiiI_ + _Q_ ) _(with_ _sigmoidal_ _S)_ _undergoes_
_a supercritical Hopf bifurcation for an open set of parameters, yielding a hyperbolic limit cycle_ Γ _._
_Under_ _small_ _bounded_ _input/parameter_ _perturbations,_ _trajectories_ deform _(phase/amplitude_ _modu-_
_lation) but structural stability preserves a nearby periodic orbit_ Γ _ε; thus the cycle, and its homology_
_class_ [Γ _ε_ ] _∈_ _H_ 1 _,_ persists _even as paths vary._


This shift in perspective reframes the role of entropy reduction. In ergodic systems, entropy is
managed by distributing trajectories uniformly across the entire state space _X_, ensuring statistical
equivalence of time and ensemble averages. By contrast, in non-ergodic, adaptive systems, entropy
reduction is achieved through _measure concentration_ Gorban & Tyukin (2018): rather than exploring
all of _X_, trajectories are funneled toward lower-dimensional recurrent sets. These recurrent sets
correspond to _persistent cycles_ that remain stable under perturbations and across variations in initial
conditions. In this sense, cycles act as the carriers of invariant information, preserving structural
regularities across history-dependent dynamics and filtering out order-specific noise. The outcome is
that intelligence emerges not from uniform exploration, but from the ability to stabilize information
flow through the persistence of these invariant structures Ayzenberg et al. (2025). Formally, we have


**Principle 2** (Non-Ergodic Invariance Principle) **.** _Let_ ( _X, T_ ) _be a dynamical system on a topological_
_state_ _space_ _X._ _Then_ _the_ _natural_ _counterpart_ _of_ _measure-preservation_ _in_ _ergodic_ _theory_ _is_ cyclepreservation _:_ _T∗_ : _Hk_ ( _X_ ) _→_ _Hk_ ( _X_ ) _,_ [ _γ_ ] _�→_ [ _γ_ ] _._ _That is, an intelligent system preserves homology_
_classes_ _of_ _cycles_ _even_ _while_ _its_ _measure_ _evolves_ _non-uniformly._ _These_ _invariant_ _cycles_ _formalize_
_memory persistence as the structural backbone of cognition._


When a non-ergodic system with many symmetric possibilities is forced to choose one outcome,
symmetry is broken Anderson (1972). In neural and cognitive dynamics, this choice does not erase
the unselected alternatives; instead, it organizes them into a closed cycle of relations: the chosen
state, its competitors, and the transitions among them. In other words, the brain does not simply
“pick a winner” among symmetric options. It establishes a cycle that records the selection, keeps
the alternatives accessible for recall or switching, and stabilizes the outcome through recurrent interaction Hochreiter & Schmidhuber (1997). Broken symmetry, therefore, inevitably produces cycle
formation, since the invariant residue of selection is a cycle connecting choice, memory, and potential revision.


This perspective reframes the role of entropy in prediction. Principle 2 establishes that non-ergodic
systems preserve homology classes of cycles as their structural invariants. From an informationtheoretic viewpoint, symmetry corresponds to maximal uncertainty: if all outcomes are equivalent
under a symmetry group _G_, the induced distribution is uniform (entropy is maximized). Symmetry
breaking reduces this uncertainty by eliminating redundant possibilities, thereby lowering entropy
and concentrating probability mass around residual invariant cycles. In high dimensions, this process
can be understood through the _theory of measure concentration_ Ledoux (2001): instead of spreading
trajectories uniformly, the dynamics of learning and memory focus trajectories around persistent
cycles. To make this precise, we introduce the notion of _residual invariants_ Beekman et al. (2019):


4


Figure 2: **Trivial,** **nontrivial,** **and** **order-invariant** **cycles.** _Left:_ A boundary of a filled region is
trivial in _H_ 1. _Middle:_ A loop around a hole cannot bound any 2-chain, so it represents a nontrivial
homology class. _Right:_ Once a trajectory closes into a cycle, its homology class depends only on
the multiset of moves, not their order: order permutations yield the same _H_ 1 class.


the structural survivors of symmetry breaking concentrate probability mass onto persistent cycles
and formalize what remains stable under the reduced symmetry subgroup.
**Definition 1** (Residual Invariants under Symmetry Breaking) **.** _Let a system evolve on a state space_
_Z with symmetry group G. Suppose a perturbation ε breaks G-equivariance, reducing the symmetry_
_to a subgroup H_ _⊂_ _G and forcing selection of a representative state_ Φ _ε_ _∈Z._ _The_ residual invariants _are those structures that remain preserved under H_ _despite the breaking of G._ _Formally, they_
_are_ _equivalence_ _classes_ _of_ _cycles_ [ _γ_ ] _∈_ _Hk_ ( _Z_ ) _that_ _are_ _stable_ _under_ _H-action_ _and_ _persist_ _under_
_perturbations of ε._


Intuitively, residual invariants encode what remains stable after a decision or perturbation: in
physics, they correspond to conserved quantities or Goldstone modes Beekman et al. (2019); in
topology, to persistent homology classes Edelsbrunner et al. (2008); and in cognition, to cycles that
bind chosen outcomes with unchosen alternatives, enabling recall, revision, and reuse Chen & Wilson (2023). This intuition can be formalized by showing that residual invariants emerging from
symmetry breaking necessarily take the form of closed cycles, which persist as homology classes
and provide the structural foundation of memory.
**Lemma 1** (Symmetry Breaking Generates Invariant Cycles) **.** _Let a system evolve on a state space Z_
_with symmetry group G._ _Suppose a perturbation ε breaks G-equivariance by forcing the selection_
_of_ _a_ _representative_ _state_ Φ _ε._ _Then:_ _1)_ _The_ _broken_ _symmetry_ _induces_ _residual_ _structures_ _(orbits)_
_invariant under_ _residual transformations H_ _⊂_ _G._ _2) These residual invariants manifest as closed_
_cycles γ_ _⊂Z_ _stabilized by feedback (i.e. ∂γ_ = 0 _)._ _3) γ_ _defines a homology class_ [ _γ_ ] _∈_ _Hk_ ( _Z_ ) _that_
_is stable under perturbations of ε, formalizing memory persistence._


The proof for the above lemma can be found in Appendix A. This lemma establishes that symmetry breaking inevitably leaves behind residual invariants in the form of cycles, which act as stable
memory traces of past selections. To fully understand their cognitive function, one must ask: What
advantage does the system gain from organizing dynamics into such closed cycles? The key lies
in the fact that cycles identify equivalence classes of trajectories, collapsing many superficially different paths into the same topological invariant Hatcher (2002). In other words, once dynamics are
organized into homology classes, prediction and memory no longer depend on the precise _order_
of steps, but only on the closure of the cycle due to the Abelian property of addition operators.
This observation leads directly to the following theorem: cycles serve as the structural basis of _or-_
_der_ _invariance_, ensuring robustness in navigation, perception, action, and more abstract cognitive
computations Hawkins (2021).
**Theorem 2** (Cycles Encode Order Invariance) **.** _Let_ ( _Z, x_ 0) _be a pointed state space (latent manifold_
_or graph) with base state x_ 0 _(“home”)._ _Let A_ = _{a_ 1 _, . . ., am} denote a finite set of local moves in-_
_ducing paths {αi} starting and ending in a neighborhood of their endpoints. For any finite sequence_
_of moves w_ = _ai_ 1 _· · · aik_ _that yields a cycle γw_ _at x_ 0 _(i.e., a homing trajectory), the first homology_
_class_ [ _γw_ ] _∈_ _H_ 1( _Z_ ; Z) _depends only on the_ multiset _of moves used (and their net orientations), not_
_on_ _their_ _order._ _Equivalently,_ _all_ _order_ _permutations_ _of_ _w_ _that_ _remain_ _cycles_ _at_ _x_ 0 _determine_ _the_
_same element in H_ 1 _._


Theorem 2 establishes that once trajectories are organized into cycles, their predictive value no
longer depends on the precise ordering of steps but only on the closure of the cycle. This reduction


5


**Trivial 1-cycle**


**Nontrivial 1-cycle**


**Order-invariant cycle**


reflects a deeper topological dichotomy in memory formation. Algebraically, the identity _∂_ [2] =
0 ensures that boundaries of boundaries vanish Edelsbrunner & Harer (2010): incomplete chains
cannot accumulate meaning unless they close, and only closed cycles can survive as invariants.
Cognitively, this corresponds to the fact that exploratory fragments either collapse into trivial points
(dots) with no relational content, or are stabilized into nontrivial cycles that encode order-invariant
memory Babichev et al. (2025). In this sense, _∂_ [2] = 0 acts as the algebraic filter that separates
forgotten scaffolds from consolidated invariants. To make this distinction explicit, we now formalize
the roles of _H_ 0 and _H_ 1 in the following lemma (refer to Fig. 2).


**Dot–Cycle Dichotomy.** At the chain level, a “dot” (0–simplex) records isolated content, whereas a
“cycle” (1–cycle) captures a closed relation in which endpoints cancel. The rule _∂_ [2] = 0 formalizes
this passage: boundaries of fragments do not compose, but pairwise cancellation at endpoints yields
a cycle that survives in homology. Cognitively, this is the move from token to trace Spens & Burgess
(2024): contents Φ are registered as dots, yet only when linked by contextual relations Ψ into a
closed cycle do they consolidate as durable memory. Details regarding biological implementations
can be found in Appendix B.
**Lemma** **2** ( _∂_ [2] = 0 Enforces the Dot-cycle Dichotomy) **.** _Let_ _C∗_ ( _Z_ ) _denote_ _the_ _chain_ _complex_
_of_ _a_ _neural_ _state_ _space_ _Z._ _The_ _homological_ _identity_ _∂_ [2] = 0 _implies_ _that:_ _1)_ _Any_ _open_ _chain_
_σ_ _∈_ _C_ 1( _Z_ ) _with ∂σ_ = 0 _must collapse to a trivial 0-cycle in H_ 0( _Z_ ) _,_ _encoding mere connectivity_
_without relational content._ _2) Any closed chain γ_ _∈_ _C_ 1( _Z_ ) _with ∂γ_ = 0 _defines a homology class_

[ _γ_ ] _∈_ _H_ 1( _Z_ ) _._ _If γ is not the boundary of a higher-dimensional chain, it represents a nontrivial cycle_
_that_ _persists_ _as_ _a_ _stable_ _memory_ _trace._ _Thus,_ _∂_ [2] = 0 _acts_ _as_ _a_ _topological_ _filter:_ _boundaries_ _of_
_boundaries vanish, ensuring that only two outcomes are possible, collapse into trivial dots (H_ 0 _) or_
_persistence as nontrivial cycles (H_ 1 _)._


Lemma 2 provides the _algebraic_ _gate_ for memory: _∂_ [2] = 0 prunes open, order-sensitive fragments
and admits only closed loops as meaningful carriers. To connect this structural pruning with predictive power, we now view closure through an information-theoretic lens Cover (1999). When many
orderings of the same events are possible, their variability behaves as symmetry-induced noise. Closure collapses these degrees of freedom onto a residual loop, thereby concentrating probability mass
on what is repeatable and compressing description length. In effect, cycles are the _sufficient statis-_
_tics_ of paths: once a trajectory closes, order fluctuations become irrelevant for forecasting Friston
(2018). The algebraic identity _∂_ [2] = 0 has an information-theoretic counterpart: broken symmetry
reduces entropy by collapsing many equivalent paths into one invariant cycle. The next proposition
formalizes this entropy–prediction link via symmetry breaking that leaves an invariant cycle.

**Proposition 1** (Entropy Minimization Improves Prediction by Cycles) **.** _Let a system generate tra-_
_jectories_ _in_ _a_ _state_ _space_ _Z._ _Suppose_ _initially,_ _the_ _system_ _has_ _a_ _symmetry_ _G_ _(e.g._ _different_ _orders_
_of_ _moves_ _or_ _observations_ _are_ _treated_ _as_ _equivalent)._ _A_ _perturbation_ _breaks_ _this_ _full_ _symmetry,_ _but_
_leaves behind an invariant cycle γ_ _⊂Z_ _with ∂γ_ = 0 _._ _Then we have: 1) The cycle γ encodes what is_
_stable across different orders or paths; 2) Predictions about future outcomes need only depend on γ_
_(and context), not on the detailed order of past steps; 3) Thus, broken symmetry reduces noise from_
_order-specific variations and improves prediction by preserving only what remains invariant._


Proposition 1 identifies _what_ survives order variability: the residual invariant cycle _γ_ . To pass from
structure to statistics, note that discarding order-specific fluctuations is equivalent to an entropy
drop: probability mass that was spread over many orderings is reassigned to the closed loop that
summarizes them. In a non-ergodic system, this manifests as _measure concentration_ on the surviving
cycles Ledoux (2001). Therefore, predictive sufficiency (dependence only on [ _γ_ ]) coincides with
entropy reduction (symmetry breaking) and with the asymptotic concentration of _µt_ on invariant
classes. The following corollary makes this equivalence explicit.

**Corollary** **1** (Prediction as Concentration on Cycles) **.** _For_ _a_ _non-ergodic_ _system_ ( _X, T_ ) _,_ _predic-_
_tion_ _is_ _possible_ _iff_ _the_ _probability_ _measure_ _µt_ _concentrates_ _on_ _invariant_ _cycles_ [ _γ_ ] _∈_ _Hk_ ( _X_ ) _as_
_t_ _→∞._ _Equivalently,_ _**Prediction**_ _⇐⇒_ _**Entropy Reduction via Symmetry Breaking**_ _⇐⇒_
_**Measure Concentration on Cycles**_ _._ _Therefore, the structural invariants revealed by broken symme-_
_try are precisely the carriers of predictive information, ensuring reliable memory and generalization_
_across time._


Corollary 1 identifies _what_ supports prediction: global dynamics must collapse onto persistent cycles. _How_ such cycles arise is local: symmetry breaking forces a choice among equivalent alterna

6


tives, and the discarded possibilities are reorganized into recurrent loops. These loops stabilize the
selected outcome while retaining counterfactual access, thereby creating the invariant structures that
concentrate probability mass and convert uncertainty into predictive stability.


4 MEMORY-AMORTIZED INFERENCE FOR TOPOLOGICAL CLOSURE


To operationalize this picture in cognition, we adopt the _Context–Content_ _Uncertainty_ _Principle_
_(CCUP)_ Li (2025a): stable memory traces correspond to low-entropy _content variables_ Φ (persistent
homological cycles), while transient variability is captured by high-entropy _context variables_ Ψ. In
what follows, we show how _Memory–Amortized_ _Inference_ _(MAI)_ implements cycle formation by
holding Φ fixed as reusable structure and adapting Ψ until residual boundaries cancel ( _∂_ [2] = 0),
thereby achieving topological closure.


**Content variable** Φ **as low-entropy homology.** Within CCUP, the content variable Φ corresponds
to information that is both specific and stable. Mathematically, Φ is identified with nontrivial homology classes: cycles [ _γ_ ] _∈_ _Hk_ ( _Z_ ) that cannot be reduced to boundaries. Such cycles encode
persistent, low-entropy structures because many possible trajectories or micro-states collapse into
the same equivalence class. In neural terms, Φ reflects patterns of activity that recur reliably across
different contexts, such as a learned motor primitive, a familiar spatial route, or a well-established
object representation. By filtering away order-dependent variability, Φ preserves only the invariant
relational structure that remains after symmetry breaking. This makes Φ the stable substrate of memory and the carrier of predictive power: once identified, it can be recalled, reused, and composed
into higher-order cognitive structures.


**Context** **variable** Ψ **as** **high-entropy** **scaffolding.** In contrast, the context variable Ψ captures
the transient, exploratory, and often noisy aspects of cognition. Topologically, Ψ is associated with
trivial cycles or short-lived features in the persistence barcode: loops that quickly vanish under perturbation or deformation. These cycles act as _scaffolding_, supporting the discovery and stabilization
of Φ but not themselves persisting as memory. In information-theoretic terms, Ψ is high-entropy:
it reflects a large space of possibilities, many of which will be pruned away as the system concentrates its measure on low-entropy Φ structures. Biologically, Ψ is implemented by slow, contextual
rhythms (e.g. theta oscillations) or exploratory neural activity that supplies diverse scaffolds for
binding. Through dynamic alignment and phase-resetting, these high-entropy contextual structures
are folded into persistent content loops, allowing cognition to maintain flexibility while ensuring
stability in memory formation.


Taken together, Φ and Ψ form a complementary pair: Φ supplies the order-invariant backbone that
can be reused across contexts, while Ψ provides the exploratory variability from which such backbones are discovered. CCUP therefore prescribes an operational loop: hold candidate content steady,
let context range, and accept only those pairings that close into cycles (i.e., cancel boundaries). This
suggests a general law of cognitive economy in which _structure_ _leads_ and _specificity_ _follows_ : stable invariants guide, while transient scaffolds adapt until closure is achieved. We now make this
heuristic precise as a principled statement.


**Principle** **3** (Structure-Before-Specificity Principle) **.** _Let_ Φ _denote_ _low-entropy_ _content_ _variables_
_corresponding_ _to_ _nontrivial_ _homology_ _classes_ [ _γ_ ] _∈_ _Hk_ ( _Z_ ) _,_ _and_ _let_ Ψ _denote_ _high-entropy_ _con-_
_textual_ _scaffolds_ _corresponding_ _to_ _transient_ _or_ _trivial_ _cycles._ _Then_ _cognition_ _obeys_ _the_ _following_
_principle:_ _1) (_ _**Structure before specificity**_ _) Stable content_ Φ _arises from nontrivial cycles that per-_
_sist_ _across_ _perturbations._ _These_ _cycles_ _define_ _the_ _backbone_ _of_ _memory_ _and_ _predictive_ _power._ _2)_
_(_ _**Specificity**_ _**from**_ _**scaffolding**_ _)_ _Context_ Ψ _supplies_ _a_ _high-entropy_ _exploratory_ _substrate:_ _transient_
_cycles_ _that_ _may_ _collapse_ _but_ _provide_ _the_ _variability_ _needed_ _to_ _refine,_ _adapt,_ _or_ _recombine_ Φ _._ _3)_
_(_ _**Dynamic alignment**_ _) The interaction of_ Ψ _and_ Φ _via cycle closure (∂_ [2] = 0 _) ensures that contex-_
_tual_ _exploration_ _is_ _funneled_ _into_ _persistent_ _content_ _loops,_ _transforming_ _noisy_ _scaffolds_ _into_ _stable_
_memory traces._


The above principle prescribes an operational recipe: stabilize Φ as reusable structure and let Ψ
explore until closure cancels residual boundaries. _Memory–amortized inference (MAI)_ is the algorithmic embodiment of this recipe. Instead of re-solving each inference problem from scratch, MAI
retrieves a candidate invariant (a cycle-level template for Φ), then adapts Ψ until the pair (Ψ _,_ Φ)


7


closes (i.e., _∂_ [2] = 0), pruning order-specific noise. In effect, Φ functions as a low-entropy prior over
solutions, while Ψ supplies the high-entropy search that is guided and terminated by topological
closure. We formalize MAI as a general strategy for reducing the computational cost of inference by
storing and reusing structured latent representations. The key idea is to construct a memory of prior
inference results such that new inference problems can be approximated by querying and adapting
from this memory, rather than solving the full problem from scratch. Let Ψ _∈X_ denote the observable context and Φ _∈S_ the latent content to be inferred. Let _L_ (Ψ _,_ Φ) denote a loss or cost
function encoding the fidelity or predictive value of Φ under context Ψ. We assume that inference
corresponds to solving the following optimization: Φ _[∗]_ = arg minΦ _∈S_ [ _L_ (Ψ _,_ Φ)]. Formally, we start
with the following definition (refer to Fig. 3).

**Definition** **2** (Memory-Amortized Inference) **.** _Let_ _M_ = _{_ (Ψ [(] _[i]_ [)] _,_ Φ [(] _[i]_ [)] ) _}_ _[N]_ _i_ =1 _[be]_ _[a]_ _[memory]_ _[of]_ _[prior]_
_context–content_ _pairs,_ _and_ _let_ _R_ : _X_ _× M_ _→S_ _be_ _a_ _retrieval-and-adaptation_ _operator_ _and_ _F_ :
_S ×X_ _→S be the bootstrapping update operator implemented via generative simulation. Inference_
_is_ _said_ _to_ _be_ memory-amortized _if_ _it_ _is_ _formulated_ _as_ _a_ _structural_ _cycle_ _between_ content Φ _and_
context Ψ _,_ _where memory acts as a reusable substrate for inference:_ Φ _t_ +1 = _F_ (Φ _t,_ Ψ _t_ ) _,_ Φ _t_ _≈_

                                     -                                     _R_ (Φ _t_ +1 _,_ Ψ _t_ ) _in lieu of directly optimizing_ Φ _[∗]_ _, such that the expected cost satisfies_ EΨ _L_ (Ψ _,_ Φ) [ˆ] _≤_

EΨ [ _L_ (Ψ _,_ Φ _[∗]_ )] + _ε,_ _for_ _some_ _amortization_ _gap_ _ε_ _≪L_ (Ψ _, ·_ ) _,_ _and_ _where_ _the_ _runtime_ _cost_ _of_ _R_ _is_
_substantially lower than full inference._


**Memory-Amortized Inference Cycle**


similarity-based soft addressing: Φ = [�] _i_ _[w][i]_ [Φ] _[,]_ _wi_ = - _j_ [exp(] _[−][d]_ [(Ψ] _[,]_ [Ψ][(] _[j]_ [)][))] [.] [This] [model] [sup-]

ports one-shot retrieval but lacks structural consistency or bidirectional inference. By contrast,
the operator _R_ (Φ _t_ +1 _,_ Ψ _t_ ; _M_ ) in MAI performs a more general operation: it _retrieves_ a candidate latent representation from memory based on both the current context Ψ _t_ and a target latent code Φ _t_ +1, and then _adapts_ it to produce a consistent approximation of the preceding latent


Figure 3: Cycle of MAI. Instead of recomputing Φ _[∗]_ = arg min _L_ (Ψ _,_ Φ), the system reuses prior
trajectories: Φ _t_ +1 and Ψ _t_ guide memory-based retrieval via _R_, and bootstrapping _F_ updates the
latent state Φ _t_ . The process forms a self-consistent loop grounded in structured memory.


**The Retrieval-and-Adaptation Operator** _R_ **.** The retrieval-and-adaptation operator _R_ : _X ×M →_
_S_ serves as the core mechanism by which inference avoids re-computation. Given an input
query (typically latent or perceptual), _R_ retrieves relevant elements from the memory _M_ =
_{_ (Ψ [(] _[i]_ [)] _,_ Φ [(] _[i]_ [)] ) _}_ _[N]_ _i_ =1 [and] [performs] [a] [lightweight] [adaptation] [to] [generate] [a] [candidate] [solution] [ˆΦ][.] [Op-]
erationally, _R_ consists of two stages: 1) **Retrieval:** Identify a relevant subset of memory entries
_{_ (Ψ [(] _[j]_ [)] _,_ Φ [(] _[j]_ [)] ) _}_ _⊂M_ based on similarity to the current context Ψ _t_ . This can be performed via
kernel-based attention, similarity search in latent space, or topological proximity under homological
constraints. 2) **Adaptation:** Modulate or interpolate the retrieved Φ [(] _[j]_ [)] values conditioned on Ψ _t_,
resulting in a candidate Φ [ˆ] _t_ = _R_ (Φ _t_ +1 _,_ Ψ _t_ ). This step often involves gradient-free adjustments (e.g.,
feature warping, parameter blending) and is significantly cheaper than full inference.


The _retrieval-and-adaptation operator R_ in MAI generalizes the classical notion of key-value memory used in neural attention and memory-augmented models. In conventional key-value memory
systems Weston et al. (2014); Sukhbaatar et al. (2015), memory is structured as a set of key-value
pairs: _M_ = _{_ (Ψ [(] _[i]_ [)] _,_ Φ [(] _[i]_ [)] ) _}_ _[N]_ _i_ =1 [,] [where] [a] [context] [vector] [Ψ] [acts] [as] [a] _[key]_ [to] [retrieve] [values] [Φ] [via]


_i_ _[w][i]_ [Φ][(] _[i]_ [)] _[,]_ _wi_ = �exp( [exp(] _−d_ _[−]_ (Ψ _[d]_ [(Ψ] _,_ Ψ _[,]_ [(][Ψ] _[i]_ [)][(] )) _[j]_ [)]


similarity-based soft addressing: Φ [ˆ] = [�]


8


state Φ _t_ . This supports inference in reverse time and satisfies the memory-amortized constraint:
Φ _t_ _≈R_ (Φ _t_ +1 _,_ Ψ _t_ ) _,_ Φ _t_ +1 = _F_ (Φ _t,_ Ψ _t_ ). The operator _R_ thereby enables cycle-consistent inference, crucial for temporal coherence and structural reuse. Unlike key-value memory, which operates
over flat vector spaces, _R_ may act over structured memory (e.g., graphs, latent manifolds, or topological complexes) and is inherently adaptive. A summary of the distinction is provided below:


**The Bootstrapping Update Operator** _F_ **.** The bootstrapping operator _F_ : _S_ _× C_ _→S_ governs the
internal dynamics of inference by iteratively updating the latent content representation Φ _t_ given the
context Ψ _t_ . It defines a recurrence: Φ _t_ +1 = _F_ (Φ _t,_ Ψ _t_ ), where _F_ encodes the system’s structural
prior, capturing the directionality, topology, and dynamic consistency of inference over time. Unlike
standard update rules that minimize a loss from scratch, _F_ performs bootstrapping: each update
is initialized from a prior memory-induced state, often already close to the optimal solution due to
cycle recurrence. Here are several key properties of _F_ : 1) **Cycle-Consistency:** If (Φ _t,_ Ψ _t_ ) _∈_ _γ_ for
some memory cycle _γ_ _⊂Z_, then Φ _t_ + _T_ _≈_ Φ _t_, enabling amortization via structural recurrence. 2)
**Structural Biasing:** Updates follow latent paths constrained by prior topology (e.g., flow fields over
homology classes or attention-modulated latent graphs), enforcing low-entropy generalization. 3)
**Minimal Cost Gradient** : Because the initialization Φ _t_ already lies near an attractor, the subsequent
update Φ _t_ +1 requires only a small corrective shift, further amortizing the inference process.


The bootstrapping update operator _F_ in MAI is structurally analogous to the _half-step_ _down_ trick
used in Q-learning Watkins & Dayan (1992) and temporal difference (TD) methods Sutton &
Barto (1998). In Q-learning, the value function is updated by approximating the current value
via a one-step lookahead: _Q_ ( _st, at_ ) _←_ _rt_ + _γ_ max _a_ _[′]_ _Q_ ( _st_ +1 _, a_ _[′]_ ), which yields the approximation
_Q_ ( _st_ ) _≈_ _Q_ ( _st_ +1). This forward-directed value propagation allows reinforcement learning agents to
estimate long-term outcomes without simulating entire trajectories. By contrast, MAI reverses the
time direction: the update operator _F_ bootstraps latent inference forward using structured memory
and contextual cues: Φ _t_ +1 = _F_ (Φ _t,_ Ψ _t_ ), and this is inverted by retrieval: Φ _t_ _≈R_ (Φ _t_ +1 _,_ Ψ _t_ ). This
dual relationship forms the backbone of the MAI half-step trick: the current latent content Φ _t_ generates the next-step prediction Φ _t_ +1, which in turn can be used to reconstruct Φ _t_ . While Q-learning
bootstraps value via reward-driven transitions, MAI bootstraps inference through latent memory and
context, yielding a cycle-consistent structure that reduces entropy. Both approaches use bootstrapping to manage uncertainty and amortize computational cost, but in opposite directions, highlighting
a deeper time-reversed duality between learning and inference (refer to Appendix C). This recursive
formulation enables stable inference trajectories that converge toward contextually relevant attractors, effectively amortizing the cost of learning across time. The underlying dynamics of this process
can be formalized as a contractive map over a structured retrieval cycle, leading to provable convergence under mild assumptions. We now state the following result, which captures the fixed-point
stability of the MAI loop:
**Proposition 2** (Topological Closure via Structural Recursion) **.** _Let T_ (Φ _,_ Ψ) := _F_ ( _R_ (Φ _,_ Ψ) _,_ Ψ) _be_
_the composite update in MAI. Suppose T_ _is contractive in its first argument for fixed context_ Ψ _. Then_
_there exists a unique fixed point_ Φ _[∗]_ _such that:_ Φ _[∗]_ = _T_ (Φ _[∗]_ _,_ Ψ) _Moreover,_ _the inference trajectory_
_{_ Φ _t}_ _[∞]_ _t_ =0 _[forms]_ _[a]_ _[closed]_ _[loop]_ _[in]_ _[latent]_ _[space]_ _[as:]_ [lim] _[t][→∞]_ _[∥]_ [Φ] _[t]_ _[−]_ [Φ] _[∗][∥]_ [=] [0] _[This]_ _[latent]_ _[recurrence]_
_corresponds to a nontrivial 1-cycle, representing topological closure in the MAI manifold._


Proposition 2 establishes closure at the level of latent dynamics: a contractive structural recursion
yields a fixed point and a recurrent trajectory that “homes” to it, i.e., a geometric 1-cycle in the
MAI manifold. We now lift this geometric closure to the algebraic level of chains. Specifically,
the same retrieve–update loop can be read as a chain-homotopy correction that cancels residual
boundaries in the context–content complex. In this view, latent recurrence (fixed-point closure)
and homological recurrence (boundary cancellation) are two faces of the same mechanism. The
next theorem formalizes this equivalence by showing that MAI implements topological closure via
_∂_ [2] = 0 (its proof can be found in Appendix A).
**Theorem** **3** (MAI as Computational Realization of Topological Closure) **.** _Let_ ( _C•, ∂_ ) _be_ _a_ _chain_
_complex_ _encoding_ _context–content_ _relations,_ _with_ Ψ _as_ _high-entropy_ _scaffolds_ _and_ Φ _as_ _candi-_
_date_ _content_ _variables._ _In_ _Memory-Amortized_ _Inference_ _(Definition_ _1),_ _the_ _iterative_ _cycle_ Φ _t_ +1 =
_F_ (Φ _t,_ Ψ _t_ ) _,_ Φ _t_ _≈R_ (Φ _t_ +1 _,_ Ψ _t_ ) _implements a homotopy update that cancels residual boundaries:_
_∂_ (Ψ _t,_ Φ _t_ ) _�→_ _∂_ (Ψ _t_ +1 _,_ Φ _t_ +1) _≈_ 0 _._ _Thus,_ _amortization_ _prunes_ _misaligned,_ _order-dependent_
_fragments (open boundaries) and preserves only reproducible cycles_ [ _γ_ ] _∈_ _Hk_ ( _C•_ ) _._ _Equivalently,_
_MAI realizes_ topological closure _by enforcing ∂_ [2] = 0 _in computation:_ _context–content updates that_
_fail to close are discarded, while those that re-enter memory persist as invariants._


9


REFERENCES


Philip W Anderson. More is different: Broken symmetry and the nature of the hierarchical structure
of science. _Science_, 177(4047):393–396, 1972.


Anton Ayzenberg, Thomas Gebhart, German Magai, and Grigory Solomadin. Sheaf theory: from
deep geometry to deep learning. _arXiv preprint arXiv:2502.15476_, 2025.


Andrey Babichev, Vladimir Vashin, and Yuri Dabaghian. Spaces and sequences in the hippocampus:
a homological perspective. _bioRxiv_, 2025.


Aron Beekman, Louk Rademaker, and Jasper Van Wezel. An introduction to spontaneous symmetry
breaking. _SciPost Physics Lecture Notes_, pp. 011, 2019.


M. A. [˜] Belluscio, K. Mizuseki, R. Schmidt, R. Kempter, and G. Buzs´aki. Cross-frequency
phase–phase coupling between theta and gamma oscillations in the hippocampus. _Journal_ _of_
_Neuroscience_, 32(2):423–435, 2012.


Guo-qiang Bi and Mu-ming Poo. Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. _Journal of Neuroscience_, 18
(24):10464–10472, 1998. doi: 10.1523/JNEUROSCI.18-24-10464.1998.


Gy¨orgy Buzs´aki. The hippocampo-neocortical dialogue. _Cerebral cortex_, 6(2):81–92, 1996.


Gy¨orgy Buzs´aki and Xiao-Jing Wang. Mechanisms of gamma oscillations. _Annual Review of Neu-_
_roscience_, 35:203–225, 2012. doi: 10.1146/annurev-neuro-062111-150444.


Ryan T. Canolty and Robert T. Knight. The functional role of cross-frequency coupling. _Trends in_
_Cognitive Sciences_, 14(11):506–515, 2010. doi: 10.1016/j.tics.2010.09.001.


Ryan T. Canolty, Edward Edwards, Sarang S. Dalal, Alireza Soltani, Srikantan S. Nagarajan,
Heidi E. Kirsch, Mitchel S. Berger, Nicholas M. Barbaro, and Robert T. Knight. High gamma
power is phase-locked to theta oscillations in human neocortex. _Proceedings_ _of_ _the_ _National_
_Academy of Sciences_, 103(19):9674–9679, 2006. doi: 10.1073/pnas.0600418103.


Natalia Caporale and Yang Dan. Spike timing–dependent plasticity: a hebbian learning rule. _Annual_
_Review of Neuroscience_, 31:25–46, 2008a. doi: 10.1146/annurev.neuro.31.060407.125639.


Natalia Caporale and Yang Dan. Spike timing–dependent plasticity: a hebbian learning rule. _Annu._
_Rev. Neurosci._, 31(1):25–46, 2008b.


Zhe Sage Chen and Matthew A Wilson. How our understanding of memory replay evolves. _Journal_
_of Neurophysiology_, 129(3):552–580, 2023.


Isaac P Cornfeld, Sergei Vasilevich Fomin, and Yakov Grigor’evˇıc Sinai. _Ergodic_ _theory_, volume
245. Springer Science & Business Media, 2012.


Thomas M Cover. _Elements of information theory_ . John Wiley & Sons, 1999.


S Davatolhagh, A Sheykhi, and MH Zarei. ‘it from bit’: How does information shape the structures
in the universe? In _Proceedings A_, volume 480, pp. 20240024. The Royal Society, 2024.


Kamran Diba and Gy¨orgy Buzs´aki. Forward and reverse hippocampal place-cell sequences during
ripples. _Nature Neuroscience_, 10(10):1241–1242, 2007. doi: 10.1038/nn1961.


Herbert Edelsbrunner and John Harer. _Computational topology:_ _an introduction_ . American Mathematical Soc., 2010.


Herbert Edelsbrunner, John Harer, et al. Persistent homology-a survey. _Contemporary mathematics_,
453(26):257–282, 2008.


R. Douglas Fields. A new mechanism of nervous system plasticity: activity-dependent myelination.
_Nature Reviews Neuroscience_, 16(12):756–767, 2015. doi: 10.1038/nrn4023.


David J. Foster and Matthew A. Wilson. Reverse replay of behavioural sequences in hippocampal
place cells during the awake state. _Nature_, 440:680–683, 2006. doi: 10.1038/nature04587.


10


Karl Friston. Does predictive coding have a future? _Nature Neuroscience_, 21(8):1019–1021, 2018.


Joaquin M. Fuster. Upper processing stages of the perception–action cycle. _Trends_ _in_ _Cognitive_
_Sciences_, 8(4):143–145, 2004.


Wulfram Gerstner, Werner M Kistler, Richard Naud, and Liam Paninski. _Neuronal dynamics:_ _From_
_single neurons to networks and models of cognition_ . Cambridge University Press, 2014.


Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. _Deep_ _learning_, volume 1.
MIT press Cambridge, 2016.


Alexander N Gorban and Ivan Yu Tyukin. Blessing of dimensionality: mathematical foundations of
the statistical physics of data. _Philosophical Transactions of the Royal Society A: Mathematical,_
_Physical and Engineering Sciences_, 376(2118):20170237, 2018.


Misha Gromov. Topological invariants of dynamical systems and spaces of holomorphic maps: I.
_Mathematical Physics, Analysis and Geometry_, 2(4):323–415, 1999.


Allen Hatcher. _Algebraic topology_ . Cambridge University Press, 2002.


Jeff Hawkins. _A thousand brains:_ _A new theory of intelligence_ . Hachette UK, 2021.


Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. _Neural_ _computation_, 9(8):
1735–1780, 1997.


Eugene M. Izhikevich. Polychronization: computation with spikes. _Neural_ _Computation_, 18(2):
245–282, 2006. doi: 10.1162/089976606775093882.


M.L. Kolomiets and A. [˜] L. Shilnikov. [˜] Poincar´e return maps in neural dynamics: Three examples. In
_Computational Methods for Understanding Spike Patterns_, pp. 45–57. Springer, 2020.


Peter K¨onig, Andreas K. Engel, and Wolf Singer. Integrator or coincidence detector? the role of
the cortical neuron revisited. _Trends_ _in_ _Neurosciences_, 19(4):130–137, 1996a. doi: 10.1016/
S0166-2236(96)80019-1.


Peter K¨onig, Andreas K Engel, and Wolf Singer. Integrator or coincidence detector? the role of the
cortical neuron revisited. _Trends in neurosciences_, 19(4):130–137, 1996b.


Michel Ledoux. _The concentration of measure phenomenon_ . Number 89. American Mathematical
Soc., 2001.


Xin Li. On content-context uncertainty principle. _Neural_ _Information_ _Processing_ _Symposium_,
2025a. under review.


Xin Li. Cycle-consistent helmholtz machine: Goal-seeded simulation via inverted inference. _arXiv_
_preprint arXiv:2507.03065_, 2025b.


Xin Li. Memory as structured trajectories: Persistent homology and contextual sheaves. _arXiv_
_preprint arXiv:2508.11646_, 2025c.


John Lisman and Ole Jensen. The theta–gamma neural code. _Neuron_, 77(6):1002–1016, 2013. doi:
10.1016/j.neuron.2013.03.007.


Henry Markram, Joachim L¨ubke, Michael Frotscher, and Bert Sakmann. Regulation of synaptic
efficacy by coincidence of postsynaptic aps and epsps. _Science_, 275(5297):213–215, 1997. doi:
10.1126/science.275.5297.213.


Marvin Minsky. Steps toward artificial intelligence. _Proceedings of the IRE_, 49(1):8–30, 1961.


Marcelo A. Montemurro, Malte J. Rasch, Yusuke Murayama, Nikos K. Logothetis, and Stefano
Panzeri. Phase-of-firing coding of natural visual stimuli in primary visual cortex. _Current Biology_,
18(5):375–380, 2008. doi: 10.1016/j.cub.2008.02.023.


John O’Keefe and Michael L. Recce. Phase relationship between hippocampal place units and the
EEG theta rhythm. _Hippocampus_, 3(3):317–330, 1993. doi: 10.1002/hipo.450030307.


11


Sinisa Pajevic, Peter J. Basser, and R. Douglas Fields. Role of myelin plasticity in oscillations and
synchrony of neuronal activity. _Neuroscience_, 276:135–147, 2014. doi: 10.1016/j.neuroscience.
2013.11.007.


Wolfram Schultz, Peter Dayan, and P Read Montague. A neural substrate of prediction and reward.
_Science_, 275(5306):1593–1599, 1997.


Michael Sipser. Introduction to the theory of computation. _ACM Sigact News_, 27(1):27–29, 1996.


Eleanor Spens and Neil Burgess. A generative model of memory construction and consolidation.
_Nature human behaviour_, 8(3):526–543, 2024.


Tamas Spisak and Karl Friston. Self-orthogonalizing attractor neural networks emerging from the
free energy principle. _arXiv preprint arXiv:2505.22749_, 2025.


Greg J. Stuart and Bert Sakmann. Active propagation of somatic action potentials into neocortical
pyramidal cell dendrites. _Nature_, 367:69–72, 1994. doi: 10.1038/367069a0.


Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. End-to-end memory networks. _Advances_
_in neural information processing systems_, 28, 2015.


Richard S Sutton and Andrew G Barto. _Reinforcement Learning: An Introduction_ . MIT Press, 1998.


Alan Turing. On computable numbers, with an application to the entscheidungsproblem. _J. of Math_,
58(345-363):5, 1936.


Peter Walters. _An introduction to ergodic theory_, volume 79. Springer Science & Business Media,
2000.


Christopher JCH Watkins and Peter Dayan. Q-learning. _Machine learning_, 8:279–292, 1992.


Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. _arXiv_ _preprint_
_arXiv:1410.3916_, 2014.


John Archibald Wheeler. Information, physics, quantum: The search for links. _Feynman and com-_
_putation_, pp. 309–336, 2018.


Hugh R Wilson and Jack D Cowan. Excitatory and inhibitory interactions in localized populations
of model neurons. _Biophysical journal_, 12(1):1–24, 1972.


12


A PROOFS OF LEMMAS, THEOREMS AND PROPOSITIONS


**Proof of Lemma 1**


_Proof (sketch)._ Let _Z_ be a smooth manifold with a smooth (left) action of a Lie group _G_, and let
_fε_ : _Z_ _→_ _T_ _Z_ be a _C_ [1] family of vector fields such that _f_ 0 is _G_ –equivariant ( _f_ 0( _g · z_ ) = _Dg · f_ 0( _z_ ))
and _ε_ _�→_ _fε_ breaks equivariance to a proper subgroup _H_ _⊂_ _G_ by selecting a representative Φ _ε_ in
each _G_ –orbit near Φ0.


_(1) Residual invariant structure._ At _ε_ = 0, the _G_ –equivariance implies that the _G_ –orbit _O_ 0 := _G·_ Φ0
is invariant for _f_ 0. For _|ε|_ small, equivariance with respect to _H_ persists, and the _residual_ _H_ orbit _Oε_ := _H_ _·_ Φ _ε_ _⊂Z_ is _fε_ –invariant. By the slice theorem, a neighborhood of _O_ 0 is _G_ equivariantly diffeomorphic to _G ×H S_ for some slice _S_, hence the residual structure is modeled on
the homogeneous space _G/H_ near Φ _ε_ .


_(2) Emergence of closed cycles under feedback._ Assume a stabilizing feedback (or dissipation) renders _Oε_ _normally hyperbolic_ . Then the invariant manifold _Oε_ persists for small _ε_ and the restricted
flow _fε|Oε_ is _H_ –invariant. If _π_ 1( _Oε_ ) = 0 (e.g. when _Oε_ contains an _S_ [1] factor, as is typical for
residual phase symmetries), there exist periodic orbits _γ_ _⊂Oε_ _⊂Z_ representing nontrivial classes
in _π_ 1. As 1–chains, periodic orbits are cycles, hence _∂γ_ = 0 in the singular chain complex. More
generally, if the residual invariant manifold contains an embedded _k_ –dimensional compact submanifold _N_ _[k]_ _⊂Oε_ invariant under the restricted flow, its fundamental class yields a closed _k_ –cycle in
_Ck_ ( _Z_ ).


_(3) Homological persistence under perturbation._ Normal hyperbolicity plus smooth dependence on
parameters implies that _Oε_ and its periodic orbits (or invariant submanifolds) vary continuously for
small _ε_ ; hyperbolic periodic orbits persist (Structural Stability). Consequently, any closed chain _cε_
carried by _Oε_ depends continuously on _ε_ and its homology class [ _cε_ ] _∈_ _Hk_ ( _Z_ ) is invariant under
the induced homotopy. Hence [ _γ_ ] _∈_ _Hk_ ( _Z_ ) is stable for small perturbations, formalizing memory
persistence.


Combining (1)–(3) proves the three claims: symmetry breaking selects a residual invariant structure,
feedback stabilizes closed cycles on it ( _∂γ_ = 0), and the resulting homology classes persist under
small perturbations of _ε_ .


**Proof of Lemma 2**


_Proof._ Write _Zk_ := ker( _∂k_ ) and _Bk_ := im( _∂k_ +1) so that _Hk_ ( _Z_ ) = _Zk/Bk_ and _∂k−_ 1 _◦_ _∂k_ = 0 for
all _k_ .


(1) Let _σ_ _∈_ _C_ 1( _Z_ ) with _∂σ_ = 0. Then _σ_ _∈/_ _Z_ 1, so it cannot define a class in _H_ 1. The only
homological information it induces is via its boundary _∂σ_ _∈_ _C_ 0( _Z_ ). But by definition _∂σ_ _∈_ _B_ 0 =
im( _∂_ 1), and since _H_ 0 = _Z_ 0 _/B_ 0 with _B_ 0 _⊆_ _Z_ 0 (because _∂_ [2] = 0), we have [ _∂σ_ ] = 0 in _H_ 0. Thus
the open chain contributes no nontrivial _H_ 1 content and collapses, at best, to the trivial _H_ 0 class that
encodes mere connectivity (membership in a component), not a relational invariant.


(2) Let _γ_ _∈_ _C_ 1( _Z_ ) with _∂γ_ = 0. Then _γ_ _∈_ _Z_ 1 and its homology class [ _γ_ ] _∈_ _H_ 1 = _Z_ 1 _/B_ 1 is
well-defined. If moreover _γ_ _∈/_ _B_ 1 = im( _∂_ 2), then [ _γ_ ] = 0 in _H_ 1, i.e., _γ_ represents a nontrivial
1-cycle. Such a class is invariant under addition of boundaries ( _γ_ _∼_ _γ_ + _∂c_ 2), hence persists under
deformations that do not cross a filling 2-chain, formalizing stability of the memory trace.


Finally, _∂_ [2] = 0 implies _Bk_ _⊆_ _Zk_ for all _k_, so every boundary is a cycle but not conversely.
Consequently any 1-chain is either (i) non-closed, in which case it reduces to a trivial element in
_H_ 0, or (ii) closed, in which case it defines a class in _H_ 1 that is nontrivial precisely when it is not a
boundary. This is the dot–cycle dichotomy.


**Proof of Theorem 1**


_Proof._ We give a standard proof in simplicial (or singular) homology, then note two equivalent
formulations (cubical/differential forms) for completeness.


13


**Simplicial** **chains.** Let _Ck_ be the free abelian group generated by oriented _k_ -simplices _σ_ =

[ _v_ 0 _, . . ., vk_ ] of an oriented simplicial complex. Define the boundary operator _∂k_ : _Ck_ _→_ _Ck−_ 1
by


**Differential** **forms** **(Stokes** _⇒_ _d_ [2] = 0 **).** On smooth manifolds, Stokes’ theorem implies

- _∂_ ( _∂_ Ω) _[ω]_ [=] �Ω _[d]_ [(] _[dω]_ [)][.] [Since] _[∂]_ _[◦]_ _[∂]_ [=] [0] [as] [a] [current,] [it] [follows] [that] _[d]_ [2] [=] [0][.] [By] [de] [Rham’s]

theorem this is dual to the chain-level statement _∂_ [2] = 0.


**Cognitive interpretation (corollary).** Because _∂_ [2] = 0, any attempt to “promote” boundary fragments (order- and context-dependent specifics) to stable carriers necessarily eliminates dangling
inconsistencies: open edges are paired and cancel, and only _closed_ cycles persist. These persistent
cycles are precisely the invariants that can be stored as memory and reused for prediction. Hence
the closure identity guarantees the existence of stable semantic carriers and underwrites the claim
that intelligence (as memory-based prediction) rests on cycle closure.


This completes the proof that the boundary of a boundary vanishes: _∂_ [2] = 0.


**Proof of Theorem 2**


_Proof:_ The key insight is the Abelian property of the addition operator. Concatenate local moves
to form cycles based at _x_ 0, producing elements of the fundamental group _π_ 1( _Z, x_ 0). The Hurewicz
map _h_ : _π_ 1( _Z, x_ 0) _→_ _H_ 1( _Z_ ; Z) abelianizes path composition: commutators vanish in _H_ 1. Hence
for cycles _γ, η_, [ _γ·η_ ] = [ _η·γ_ ] and, more generally, any permutation of cycle segments yields the same
homology class, provided the path remains closed. Thus [ _γw_ ] is invariant to the _order_ of constituent


14


_∂k_ [ _v_ 0 _, . . ., vk_ ] =


_k_
�( _−_ 1) _[i]_ [ _v_ 0 _, . . .,_ - _vi, . . ., vk_ ] _,_

_i_ =0


where the hat indicates omission and each face inherits the induced orientation. Apply _∂k−_ 1 once
more:


( _−_ 1) _[j]_ [ _v_ 0 _, . . .,_   - _vi, . . .,_   - _vi_ + _j′, . . ., vk_ ] _,_
_j_ =0


_∂k−_ 1 _∂k_ [ _v_ 0 _, . . ., vk_ ] =


_k_ _k−_ 1
�( _−_ 1) _[i]_ 

_i_ =0 _j_ =0


_k_


where _j_ _[′]_ denotes the corresponding original index in _{_ 0 _, . . ., k} \ {i}_ .


Every ( _k −_ 2)-face of [ _v_ 0 _, . . ., vk_ ] arises _twice_ in this double sum: once by deleting _vi_ then _vj_
with _i_ _<_ _j_, and once by deleting _vj_ then _vi_ . These two occurrences have opposite signs and thus
cancel. Formally, fix 0 _≤_ _i_ _<_ _j_ _≤_ _k_ . The face obtained by deleting _vi_ then _vj_ appears with sign
( _−_ 1) _[i]_ ( _−_ 1) _[j][−]_ [1] = ( _−_ 1) _[i]_ [+] _[j][−]_ [1], while deleting _vj_ then _vi_ yields sign ( _−_ 1) _[j]_ ( _−_ 1) _[i]_ = ( _−_ 1) _[i]_ [+] _[j]_ . Hence
the two contributions sum to zero:


( _−_ 1) _[i]_ [+] _[j][−]_ [1] + ( _−_ 1) _[i]_ [+] _[j]_ = 0 _._


Since every ( _k−_ 2)-face of _σ_ appears exactly in such canceling pairs, all terms vanish and therefore
_∂k−_ 1 _∂k_ = 0, i.e. _∂_ [2] = 0.


**Singular chains (same combinatorics).** For singular homology, _Ck_ is generated by singular simplices _σ_ : ∆ _[k]_ _→_ _X_, and the boundary uses face inclusions _di_ : ∆ _[k][−]_ [1] _�→_ ∆ _[k]_ :


_∂kσ_ =


_k_
�( _−_ 1) _[i]_ _σ ◦_ _di._


_i_ =0


Then
_∂k−_ 1 _∂kσ_ =      

_i<j_


�( _−_ 1) _[i]_ [+] _[j]_ _σ ◦_ _di ◦_ _dj−_ 1 + ( _−_ 1) _[i]_ [+] _[j]_ _σ ◦_ _dj_ _◦_ _di_ - = 0 _,_


because _di ◦_ _dj_ = _dj_ +1 _◦_ _di_ and the two terms cancel in pairs.


**Cubical** **chains** **(face** **maps).** In cubical homology, the boundary is an alternating sum of
front/back faces along each coordinate. The same “each ( _k_ _−_ 2)-face appears twice with opposite
sign” cancellation proves _∂_ [2] = 0.


moves and depends only on their cumulative 1-chain (the signed sum of traversed edges/segments).
Intuitively, homology collapses all order-specific reparameterizations and commutator structure, retaining only the closed-cycle content.


**Proof of Theorem 3**


_Proof._ Let ( _C•, ∂_ ) encode context–content relations and write the _residual_ _boundary_ at step _t_ as
_rt_ := _∂_ (Ψ _t,_ Φ _t_ ) _∈_ _Ck−_ 1. Define the MAI update _U_ := _R ◦F_ so that (Ψ _t_ +1 _,_ Φ _t_ +1) = _U_ (Ψ _t,_ Φ _t_ )
with Φ _t_ +1 = _F_ (Φ _t,_ Ψ _t_ ) and Φ _t_ _≈R_ (Φ _t_ +1 _,_ Ψ _t_ ). Assume (i) _boundary-aware_ updates: there exists
a linear operator _H_ : _Ck_ _→_ _Ck_ +1 (a homotopy) and _η_ _∈_ (0 _,_ 1] such that, up to the amortization
error _ϵt_,
(Ψ _t_ +1 _,_ Φ _t_ +1) = (Ψ _t,_ Φ _t_ ) _−_ _η H rt_ + _ϵt,_ _∥ϵt∥≤_ _ε,_

and (ii) _∂_ [2] = 0 on _C•_ . Then


_rt_ +1 = _∂_ (Ψ _t_ +1 _,_ Φ _t_ +1) = _∂_ (Ψ _t,_ Φ _t_ ) _−_ _η ∂H rt_ + _∂ϵt_ =      - _I_ _−_ _η ∂H_      - _rt_ + _∂ϵt._


Choose _H_ so that _P_ := _I_ _−_ _∂H_ _−_ _H∂_ is the standard chain-homotopy projector onto _Zk_ := ker _∂_
(e.g. a Moore–Penrose choice on a chosen splitting). Using _∂_ [2] = 0,


_rt_ +1 =      - _I_ _−_ _η ∂H_      - _rt_ + _∂ϵt_ =      - _P_ + _H∂_ _−_ _η ∂H_      - _rt_ + _∂ϵt_ = ( _I −_ _η_ ) _rt_ + _∂ϵt,_


since _Prt_ = 0 and _∂rt_ = _∂_ [2] ( _·_ ) = 0. Hence _∥rt_ +1 _∥≤_ (1 _−_ _η_ ) _∥rt∥_ + _∥∂∥_ _ε_ . If _η_ _∈_ (0 _,_ 1] and
_ε_ is the small amortization gap from Definition 1, the residuals converge: _∥rt∥→_ 0 as _t_ _→∞_
(exactly if _ε_ = 0, or to an _O_ ( _ε_ ) neighborhood otherwise). Thus, any limit point (Ψ _∞,_ Φ _∞_ ) satisfies
_∂_ (Ψ _∞,_ Φ _∞_ ) = 0, i.e. it lies in _Zk_ and represents a closed cycle [ _γ_ ] _∈_ _Hk_ . Moreover, because _R_
retrieves from memory and _F_ bootstraps by simulation while satisfying the amortization inequality
EΨ[ _L_ (Ψ _,_ Φ)] [ˆ] _≤_ EΨ[ _L_ (Ψ _,_ Φ _[∗]_ )] + _ε_, open, order-dependent trajectories (with large _∥rt∥_ ) are not
retained, while reproducible closures are. Therefore MAI acts as a homotopy-based projection onto
ker _∂_, canceling boundaries and preserving precisely the invariant cycles, i.e. it realizes topological
closure computationally.


**Proof of Proposition 1**


_Proof (sketch)._ Let _P_ be the set of finite trajectories (paths) in _Z_ and let _G_ act on _P_ by the symmetry
that permutes orderings of local moves/observations. Define an equivalence relation _p_ _∼G_ _p_ _[′]_ iff
_p_ _[′]_ = _g · p_ for some _g_ _∈_ _G_ . Suppose a perturbation breaks _G_ to a residual subgroup _H_ and induces
a continuous, _H_ –invariant map

_q_ : _P_ _−→_ _Zk_ ( _Z_ )� _Bk_ ( _Z_ ) _[∼]_ = _Hk_ ( _Z_ ) _,_ _p �→_ [ _γ_ ( _p_ )] _,_


that sends each path _p_ to the homology class of its closing cycle _γ_ ( _p_ ) (if _p_ does not close, _q_ maps
it to the trivial class). By assumption there exists a nontrivial invariant cycle _γ_ with _∂γ_ = 0 that
survives the perturbation (i.e. [ _γ_ ] _̸_ = 0 and _H_ –invariant).


(1) _Cycle_ _encodes_ _order-invariant_ _stability._ If _p_ _[′]_ _∼G_ _p_, then _q_ ( _p_ _[′]_ ) = _q_ ( _p_ ) because permutations
of the same local moves that remain closable yield homologous loops. Hence [ _γ_ ] is constant on
_G_ –orbits and captures precisely what is invariant under order rearrangements. This proves (1).


(2) _Predictive sufficiency of_ [ _γ_ ] _._ Let _Y_ denote a future outcome (or next observation) to be predicted
from the past path _P_ _∈P_ and any ambient context variable _C_ (slow parameters). Assume the
perturbation enforces the residual symmetry so that _conditional on the homology class_ we have


P( _Y_ _| P, C_ ) = P� _Y_ �� _q_ ( _P_ ) _, C_              - _._


That is, order-specific information in _P_ beyond its cycle class does not affect the conditional law of
_Y_ . Then _q_ ( _P_ ) is a (Blackwell) sufficient statistic for predicting _Y_ given _C_ . By the data-processing
inequality,
_I_ ( _Y_ ; _P_ _| C_ ) _≥_ _I_                - _Y_ ; _q_ ( _P_ ) _| C_                - _,_


with equality under the displayed conditional independence, which shows that prediction needs only
depend on the invariant [ _γ_ ] = _q_ ( _P_ ) (and _C_ ), not on the detailed order in _P_ . This proves (2).


15


(3) _Entropy reduction improves prediction._ Let _G_ be the _σ_ -algebra generated by _P_ and let _G_ inv be
that generated by _q_ ( _P_ ). Since _q_ maps many orderings to the same class, _H_ ( _q_ ( _P_ )) _≤_ _H_ ( _P_ ) and
E[ _ℓ_ ( _Y,_ _Y_ [ˆ] ) ] for any Bayes-optimal predictor _Y_ [ˆ] is the same whether conditioning on _P_ or _q_ ( _P_ ) (by
sufficiency). Thus collapsing order-specific variability to [ _γ_ ] strictly reduces the description length
of the predictor while preserving optimal predictive risk. Interpreting entropy as uncertainty, the
symmetry breaking acts to concentrate probability mass onto [ _γ_ ]–classes (measure concentration on
invariant cycles), thereby removing order noise and improving generalization: predictions depend
only on what remains invariant. This proves (3).


Altogether, the perturbation-induced residual symmetry yields an invariant cycle _γ_ ( _closure ∂γ_ = 0)
whose class [ _γ_ ] summarizes all order permutations of closable paths; [ _γ_ ] is a sufficient statistic for
forecasting _Y_ (given context), and the associated entropy drop reflects the elimination of orderspecific noise. Hence entropy minimization via symmetry breaking improves prediction by preserving only invariant cycles.


**Proof of Proposition 2**


_Proof (sketch)._ Fix a context Ψ and define the self-map _T_ Ψ(Φ) := _F_ ( _R_ (Φ _,_ Ψ) _,_ Ψ). By assumption,
_T_ Ψ is a contraction in its first argument on a complete metric space ( _S, ∥·∥_ ): there exists 0 _< κ <_ 1
such that _∥T_ Ψ(Φ) _−T_ Ψ(Φ _[′]_ ) _∥≤_ _κ ∥_ Φ _−_ Φ _[′]_ _∥_ for all Φ _,_ Φ _[′]_ _∈S_ .


_(Existence, uniqueness, and convergence)._ By the Banach fixed-point theorem, there exists a unique
fixed point Φ _[∗]_ _∈S_ with Φ _[∗]_ = _T_ Ψ(Φ _[∗]_ ), and for any initialization Φ0 the iterates Φ _t_ +1 = _T_ Ψ(Φ _t_ )
satisfy _∥_ Φ _t −_ Φ _[∗]_ _∥≤_ _κ_ _[t]_ _∥_ Φ0 _−_ Φ _[∗]_ _∥→_ 0 as _t →∞_ . This proves the first two claims.


_(Latent recurrence as a closed 1-cycle)._ Form the polygonal 1-chain


where [Φ _t,_ Φ _t_ +1] denotes the oriented edge in latent space joining successive iterates. Its boundary
is _∂cn_ = Φ _n −_ Φ0. Close the polygon by adding the short edges _en_ = [Φ _n,_ Φ _[∗]_ ] and _e−_ 1 = [Φ _[∗]_ _,_ Φ0]
to obtain

_c_ ˜ _n_ := _cn_ + _en_ + _e−_ 1 _,_ _∂c_ ˜ _n_ = (Φ _n −_ Φ0) + (Φ _[∗]_ _−_ Φ _n_ ) + (Φ0 _−_ Φ _[∗]_ ) = 0 _._

Thus each ˜ _cn_ is a 1-cycle. Since _∥_ Φ _n −_ Φ _[∗]_ _∥→_ 0, the closing edges _en, e−_ 1 have lengths _→_ 0, and
the sequence _{c_ ˜ _n}_ converges (in the 1-chain norm induced by edge length) to a limit 1-chain _γ_ with
_∂γ_ = 0. Hence the MAI trajectory defines a closed loop (a 1-cycle) in latent space.


_(Nontriviality and topological closure)._ If the image of the trajectory lies in a region whose chosen
2-chain complex (e.g. a Vietoris–Rips or Cech complex at some scale [ˇ] _ε_ ) contains no filling 2-chain
for _γ_, then _γ_ _∈/_ im _∂_ 2 and [ _γ_ ] = 0 in _H_ 1, yielding a nontrivial cycle. This expresses _topological_
_closure_ : the structural recursion contracts to a fixed point while the induced 1-chain closes with
vanishing boundary; nontriviality holds precisely when the loop does not bound any 2-chain in the
MAI manifold at the working scale.


In summary, contractivity yields a unique fixed point and convergence; the polygonal chain of iterates closes in the limit to a 1-cycle _γ_ with _∂γ_ = 0, which is nontrivial whenever no 2-chain fills it.
Hence structural recursion realizes topological closure as a latent recurrence.


B BIOLOGICAL IMPLEMENTATION OF TOPOLOGICAL CLOSURE


Oscillations discretize time on a circle ( _S_ [1] ), providing phase bins within which coincidence detection collapses fragments into recurrent traversals. Mathematically, the boundary calculus enforces
this filtration: _∂_ [2] = 0 cancels unmatched endpoints so that only closed chains survive as persistent
cycles. Cognitively, isolated tokens (dots) do not stabilize memory; only when linked by contextual
relations into cycles do they consolidate as durable traces. In this section, we show how oscillatory
phase coding and coincidence detection implement temporal scaffolding and boundary cancellation
in spiking networks, turning temporal fragments into cycles.


16


_cn_ :=


_n−_ 1

- _et_ with _et_ := [Φ _t,_ Φ _t_ +1] _,_


_t_ =0


B.1 OSCILLATION PHASE CODING AS TEMPORAL SCAFFOLDING


Neural oscillations instantiate the closure principle by quotienting linear time to a circle: an oscillator implements _t_ _�→_ _e_ _[iωt]_ _∈_ _S_ [1], so events are registered by _phase_ rather than absolute time.
Biologically, this scaffold is realized at multiple, coupled timescales. (i) _Theta–gamma_ _nesting_
(e.g., hippocampus–entorhinal) provides a macrocycle ( _θ_, 4–12 Hz) that segments experience and
a microcycle ( _γ_, 30–100 Hz) that tiles each _θ_ bin with ordered subevents; phase–amplitude coupling thus lays out a toroidal code _Sθ_ [1] _[×][ S]_ _γ_ [1] [in which winds index recurrent cycles Lisman & Jensen]
(2013); Canolty & Knight (2010); Canolty et al. (2006). (ii) _Coincidence detection_ sharpens edges of
these cycles: NMDA nonlinearity, backpropagating spikes, and fast interneuron circuitry (PV/ING,
PING) create narrow _O_ (1 _−_ 10 ms) windows so that only spikes aligned within a phase bin form effective synaptic links; misaligned fragments fail to bind and are pruned K¨onig et al. (1996a); Stuart
& Sakmann (1994); Buzs´aki & Wang (2012). (iii) _Spike-timing dependent plasticity (STDP)_ orients
these links by phase lead/lag, turning phase offsets into directed edges in a chain; repeated traversal within a cycle consolidates these edges, canceling stray endpoints and favoring closed walks
Markram et al. (1997); Bi & Poo (1998); Caporale & Dan (2008a). (iv) _Conduction_ _delays_ _and_
_myelin plasticity_ tune effective phase lags, enabling polychronous assemblies: axonal/dendritic delays align distributed spikes into reproducible phase patterns that complete cycles despite spatial
dispersion Izhikevich (2006); Pajevic et al. (2014); Fields (2015). (v) _Phase-of-firing_ _coding_ _and_
_precession_ (e.g., hippocampal place cells) map position or task progress to phase on _Sθ_ [1][,] [so] [that] [a]
behavioral episode corresponds to a return map on the Poincar´e section; complete laps close in phase
space, incomplete traversals do not O’Keefe & Recce (1993); Montemurro et al. (2008). (vi) _State-_
_dependent reentry_ (sharp-wave ripples during NREM/quiet wake) replays phase-ordered sequences
on a faster carrier, tightening weights along already-closed paths and suppressing nonclosing detours
Foster & Wilson (2006); Diba & Buzs´aki (2007).


_Interpretation._ Oscillations supply the contextual scaffold Ψ that folds timelines into cyclic coordinates; coincidence and plasticity then implement boundary cancellation in synaptic space. What
persists are cycles, phase-locked traversals whose endpoints identify on _S_ [1], while unmatched fragments dissipate. This sets up the formal lemma below, which recasts phase-binned spiking as a chain
whose boundary vanishes after a full cycle.


**Lemma** **3** (Oscillatory Phase Coding as Temporal Scaffolding) **.** _Let_ _θ_ ( _t_ ) = _ωt_ (mod 2 _π_ ) _denote_
_the phase of a neural oscillator, with events encoded relative to θ_ ( _t_ ) _on the circle S_ [1] _._ _Then oscilla-_
_tory phase coding induces the following invariants:_ _1)_ _**Binding:**_ _Events occurring within the same_
_phase window θ_ ( _t_ ) _∈_ [ _ϕ, ϕ_ +∆] _are grouped together, forming a coherent representation; 2)_ _**Order-**_
_**ing:**_ _Sequences of events are represented by their relative phase offsets_ (∆ _θ_ 1 _,_ ∆ _θ_ 2 _, . . ._ ) _, embedding_
_linear order into a cyclic scaffold; 3)_ _**Closure:**_ _After a full cycle θ_ ( _t_ + _T_ ) = _θ_ ( _t_ ) _with T_ = [2] _ω_ _[π]_ _[, the]_

_system_ _resets,_ _ensuring_ _that_ _trajectories_ _are_ _organized_ _into_ _cycles_ _rather_ _than_ _unbounded_ _chains._
_Together, these properties enforce the topological identity ∂_ [2] = 0 _at the temporal level:_ _the bound-_
_ary of one temporal segment becomes the beginning of the next,_ _so that each cycle closes before a_
_new one begins._ _Consequently,_ _oscillatory phase coding guarantees consistency of memory traces_
_by embedding them in recurrent temporal cycles._


The formal statement of Lemma 3 captures how oscillatory phase coding transforms linear time
into a cyclic scaffold, guaranteeing binding, ordering, and closure. To visualize this principle, Fig. 4
illustrates how linear time _t_ is wrapped onto the circle _S_ [1] (theta phase), with discrete gamma packets
embedded at distinct phases. Events that fall into the same phase window (green arc) are bound
together, while relative phase offsets encode ordering. The reset at the end of each _θ_ cycle ensures
closure, embodying the algebraic identity _∂_ [2] = 0 in biological timekeeping. Let _θ_ : R _→_ _S_ [1] be
the phase map _θ_ ( _t_ ) = _ωt_ mod 2 _π_, so _T_ = [2] _ω_ _[π]_ [identifies] _[t]_ _[∼]_ _[t]_ [ +] _[ T]_ [and] [quotients] [linear] [time] [to]

a circle. Partition _S_ [1] into _L_ phase bins _{φℓ}_ _[L]_ _ℓ_ =1 [and] [let] _[v][ℓ]_ [denote] [the] [(phase-binned)] [latent] [state]
aggregated within bin _φℓ_ . Define oriented edges _eℓ_ = [ _vℓ, vℓ_ +1] with _vL_ +1 _≡_ _v_ 1. The phaseordered chain _c_ = [�] _ℓ_ _[L]_ =1 _[e][ℓ]_ [has] _[∂c]_ [=] [�] _ℓ_ _[L]_ =1 [(] _[v][ℓ]_ [+1] _[−]_ _[v][ℓ]_ [)] [=] _[v][L]_ [+1] _[−]_ _[v]_ [1] [=] [0][,] [so] [a] [full] [2] _[π]_ [sweep]
closes into a 1–cycle. Coincidence detection enforces this construction: only events aligned within
a phase window of width _ε_ create edges, pruning stray fragments whose endpoints would otherwise
fail to cancel. Conduction delays implement modular jumps _eℓ_ = [ _vℓ, vℓ_ + _k_ ], yielding a winding
number _k_ on _S_ [1] ; after _L_ such steps the path returns to _v_ 1, again giving _∂c_ = 0 and a homology class

[ _c_ ] _∈_ _H_ 1( _S_ [1] ) _[∼]_ = Z.


17


3 _π_

2


Figure 4: **Oscillatory phase coding as topological closure.** Linear time is wrapped onto the circle
_S_ [1] (theta phase), placing events by _phase_ rather than absolute time. Gamma packets ( _g_ 1 _, . . ., g_ 5) at
distinct phases encode _order_ via phase offsets, while events within the same phase window (green
arc) are _bound_ . Cycle reset at the end of each theta period enforces topological closure ( _∂_ [2] = 0),
supporting consistent memory cycles.


Two useful views follow. (i) _Poincar´e/return_ _map:_ sampling at phase _ϕ_ 0 defines _Fϕ_ 0 : _x_ ( _t_ ) _�→_
_x_ ( _t_ + _T_ ); fixed points and periodic points of _Fϕ_ 0 are closed orbits on _S_ [1], i.e., cycles Kolomiets &
Shilnikov (2020). (ii) _Cross-frequency nesting:_ with _θ_ and _γ_ phases, time quotients to a torus _Sθ_ [1] _[×][S]_ _γ_ [1]
and _H_ 1 = _[∼]_ Z [2] ; winds ( _kγ, kθ_ ) encode hierarchical cycles Belluscio et al. (2012). _Summary._ Phase
coding turns linear sequences into cyclic invariants: the absolute start/end times are identified on
_S_ [1], so boundaries telescope away and only closed traversals persist. Coincidence gates which edges
exist; _∂_ [2] = 0 guarantees unmatched endpoints cannot accumulate into memory, while completed
cycles survive as stable traces.


B.2 COINCIDENCE DETECTION AS TOPOLOGICAL CLOSURE


Lemma 3 establishes that oscillatory phase coding furnishes a natural scaffold for aligning events
on the circle _S_ [1], ensuring that candidate trajectories can be organized into cyclic frames. However,
phase alignment alone does not guarantee stability: without a mechanism to prune misaligned or
inconsistent events, spurious boundaries would accumulate and prevent reliable cycle formation.
Lemma 4 addresses this gap by showing how coincidence detection enforces closure at the level of
spike trains, cancelling mismatches as boundary terms and preserving only those cycles that survive
across trials. Together, these results formalize the complementary roles of phase scaffolding and
coincidence pruning in transforming transient alignments into reproducible cognitive invariants.


**Lemma** **4** (Coincidence-Induced Closure and Survival of Reproducible Cycles) **.** _Let_ _N_ =
_{_ 1 _, . . ., n} be a set of units (neurons) producing a spike train S_ = _{_ ( _i, tk_ ) _} with phases ϕ_ ( _tk_ ) _∈_ _S_ [1] _._
∆
_Fix_ _a_ _coincidence_ _window_ ∆ _∈_ (0 _, π_ ) _and_ _define_ _the_ coincidence relation _i_ _↔_ _j_ _iff_ _there_ _ex-_
_ist_ _spikes_ ( _i, t_ ) _,_ ( _j, t_ _[′]_ ) _∈_ _S_ _with_ _|ϕ_ ( _t_ ) _−_ _ϕ_ ( _t_ _[′]_ ) _|S_ 1 _≤_ ∆ _and_ _t_ _<_ _t_ _[′]_ _(to_ _orient_ _time)._ _Construct_
_the_ _directed_ _1–skeleton_ _G_ ∆( _S_ ) _whose_ _vertex_ _set_ _is_ _N_ _and_ _whose_ _(possibly_ _multiple)_ _oriented_
∆
_edges_ _are_ _e_ = ( _i_ _→_ _j_ ) _for_ _every_ _coincident_ _pair_ _i_ _↔_ _j._ _Let_ _C_ 1( _G_ ∆) _be_ _the_ _free_ _abelian_
_group_ _on_ _edges_ _and_ _C_ 0( _G_ ∆) _the_ _free_ _abelian_ _group_ _on_ _vertices,_ _with_ _boundary_ _∂_ : _C_ 1 _→_ _C_ 0
_given_ _by_ _∂_ ( _i_ _→_ _j_ ) = _j_ _−_ _i._ _Define_ _the_ coincidence aggregation _c_ ∆( _S_ ) _∈_ _C_ 1( _G_ ∆) _by_ _sum-_
_ming_ _all_ _oriented_ _edges_ _(with_ _multiplicities)_ _generated_ _by_ _coincident_ _pairs,_ _and_ _the_ coincidence
projection Π∆ : _C_ 1( _G_ ∆) _−→_ _Z_ 1( _G_ ∆) := ker _∂_ _as_ _the_ _(linear)_ _projection_ _onto_ _the_ _cycle_ _space_
_(e.g.,_ _orthogonal_ _projection_ _with_ _respect_ _to_ _any_ _inner_ _product_ _on_ _C•_ _or_ _the_ _canonical_ _decompo-_
_sition_ _C_ 1 = _Z_ 1 _⊕_ _B_ 1 _[⊥][).]_ _[Then]_ _[we]_ _[have:]_ _[1)]_ _**[Closure]**_ _**[by]**_ _**[coincidence.]**_ _[The]_ [coincidence] [detector]
_K_ ∆ := Π∆ _◦_ ( _·_ ) _enforces_ _closure:_ _z_ ∆( _S_ ) := _K_ ∆� _c_ ∆( _S_ )� _∈_ _Z_ 1( _G_ ∆) _and_ _∂z_ ∆( _S_ ) = 0 _._ _More-_
_over, the edges removed by K_ ∆ _are precisely those whose net contribution appears in ∂c_ ∆( _S_ ) _; i.e._


18


_π_
2


**1000**


**1001**

**1002**

**1003**

**1004**

**1005**

**1006**


**1007**

**1008**

**1009**

**1010**

**1011**

**1012**


**1013**

**1014**

**1015**

**1016**

**1017**

**1018**


**1019**

**1020**

**1021**

**1022**

**1023**


**1024**

**1025**


_misaligned_ _spikes_ _are_ _canceled_ _as_ _boundary_ _terms_ _and_ _do_ _not_ _survive_ _in_ _z_ ∆( _S_ ) _._ _2)_ _**Survival**_ _**of**_
_**reproducible**_ _**cycles**_ _**(stability).**_ _Suppose_ _S_ [(1)] _, . . ., S_ [(] _[T]_ [ )] _are_ _trials_ _with_ _phase_ _jitter_ _at_ _most_ _ε_ _<_ ∆
_(i.e._ _every_ _coincidence_ _in_ _one_ _trial_ _has_ _a_ _matched_ _coincidence_ _within_ _phase_ _distance_ _ε_ _in_ _all_ _oth-_
_ers,_ _with_ _the_ _same_ _orientation)._ _Then_ _for_ _all_ _t,_ [ _z_ ∆( _S_ [(] _[t]_ [)] )] = [ _z_ ∆( _S_ [(1)] )] _∈_ _H_ 1( _G_ ∆; Z) _,_ _so_ _the_
_homology class is trial-invariant._ _In particular,_ _in the persistence module obtained by varying the_
_window_ _δ_ _∈_ ( _ε,_ ∆] _,_ _this_ _class_ _has_ _positive_ _lifetime_ _and_ _therefore_ survives _while_ _nonreproducible_
_coincidences die as boundaries._


_Proof sketch._ (1) By construction, _∂c_ ∆( _S_ ) counts net imbalance of incident coincidences at each
vertex (incoming minus outgoing). Projecting onto ker _∂_ removes exactly those components whose
boundary is nonzero; hence _z_ ∆( _S_ ) _∈_ _Z_ 1 and _∂z_ ∆( _S_ ) = 0. Informally, coincidences that do not
close are eliminated as boundary terms; only closed flow persists. (2) Phase jitter _ε_ _<_ ∆ induces
edge correspondences between the _G_ ∆( _S_ [(] _[t]_ [)] ) that preserve orientation and incidence, yielding chain
homotopic _c_ ∆( _S_ [(] _[t]_ [)] ). Projection to _Z_ 1 commutes with these homotopies, so the resulting _z_ ∆( _S_ [(] _[t]_ [)] )
are homologous. Viewing _δ_ as a filtration parameter, unmatched (nonreproducible) edges vanish at
_δ_ _↘_ _ε_, whereas reproducible cycles define a bar of positive length in _H_ 1, hence survive.


Lemma 4 established that coincidence detection enforces closure by cancelling misaligned spikes,
ensuring that only reproducible cycles survive. Fig. 5 illustrates this principle: when presynaptic
spikes align within a coincidence window ∆ (top), their inputs sum coherently and trigger a postsynaptic spike, corresponding to _∂γ_ = 0 (closure). When spikes fall outside the window (bottom),
they remain as unmatched boundaries that cancel one another, yielding no output. The inset shows
the topological analogy: different paths that bound the same face _σ_ cancel in homology, just as
misaligned temporal fragments fail to stabilize into persistent cycles. Formally, we have


**Definition 3** (Topological Closure) **.** _Let_ ( _X, τ_ ) _be a topological space and A_ _⊆_ _X._ _The_ closure _of_
_A, denoted A, is defined as A_ = [�] _{C_ _⊆_ _X_ _| C_ _is closed and A ⊆_ _C}._ _Equivalently, A consists of_
_all points x ∈_ _X_ _such that every open neighborhood U_ _∈_ _τ_ _with x ∈_ _U_ _satisfies U_ _∩_ _A ̸_ = ∅ _._


With the formal notion of closure in hand, we now _operationalize_ it in neural dynamics: replace open
neighborhoods by temporal coincidence windows and subsets _A_ by sets of candidate spikes. Under
this identification, the “points in the closure” are precisely spikes that recurrently co-occur within
a window, and the homological reading of closure ( _∂_ [2] = 0) corresponds to cancelling unmatched,
out-of-window events as boundary terms. This yields a direct bridge from topological closure to
coincidence-driven cycle formation in neural circuits. For a PNG to persist, spikes from multiple
presynaptic neurons must converge within a narrow temporal window at their postsynaptic targets.
Coincidence detection acts as a filter: inputs that arrive in synchrony are integrated, while those that
fall outside the coincidence window are effectively cancelled. This selective integration implements
the algebraic identity _∂_ [2] = 0: misaligned spikes behave like open boundaries that fail to connect,
whereas synchronous arrivals cancel boundary terms and enforce cycle closure. In this way, only
temporally coherent activity contributes to a closed 1-cycle in the neural state space. Once closure is
achieved, spike-timing dependent plasticity (STDP) reinforces the recurrent pathways that produced
coincident input Caporale & Dan (2008b). Potentiation strengthens the synapses along routes that
consistently deliver spikes within the window ∆, while depression weakens those that fail to align.
Over repeated activations, this differential plasticity stabilizes the trajectory as a reentrant cycle: the
cycle not only replays reliably, but also becomes resistant to perturbations of individual spike times.
In summary, coincidence detection, together with STDP, extracts the low-entropy content variable
Φ: a reproducible invariant that persists as a memory trace and can later be recalled or recombined
into higher-order structures Li (2025c).


The principle “coincidence detection = boundary cancellation” can now be made explicit. When
presynaptic spikes converge within the coincidence window, their temporal boundaries align and
cancel, producing a closed cycle that can drive a stable postsynaptic response K¨onig et al. (1996b).
In contrast, when spikes arrive outside the window, they leave residual unmatched boundaries that
fail to close, and no postsynaptic output is generated. Figure 5 illustrates this correspondence: in
the neural case, misaligned spikes cancel each other’s contributions and disappear from the effective
cycle; in the topological case, paths that differ by the boundary of a 2-simplex _σ_ cancel in homology.
In both settings, coincidence detection enforces the identity _∂_ [2] = 0, ensuring that only closed cycles
survive as memory-bearing invariants.


19


**1026**

**1027**


**1028**

**1029**

**1030**

**1031**

**1032**

**1033**


**1034**

**1035**

**1036**

**1037**

**1038**

**1039**


**1040**

**1041**

**1042**

**1043**

**1044**


**1045**

**1046**

**1047**

**1048**

**1049**

**1050**


**1051**

**1052**

**1053**

**1054**

**1055**

**1056**


**1057**

**1058**

**1059**

**1060**

**1061**

**1062**


**1063**

**1064**

**1065**

**1066**

**1067**


**1068**

**1069**

**1070**

**1071**

**1072**

**1073**


**1074**

**1075**

**1076**

**1077**

**1078**

**1079**


**coincidence** _⇒_ **closure:**
_∂γ_ = 0 preserved


_A_

_C_


_A_

_C_


|Col1|Col2|Col3|
|---|---|---|
||||


Figure 5: **Coincidence** **detection** **=** **boundary** **cancellation.** _Left:_ Three presynaptic spike trains
( _A, B, C_ ). **Top:** Spikes align within a coincidence window ∆ (green band), summate, and produce
a postsynaptic spike (closure). **Bottom:** Spikes are misaligned; inputs fail to coincide in ∆, so no
output occurs (boundaries do not cancel). _Right inset:_ Topological analogy - when two paths differ
by the boundary of a face ( _∂σ_ ), their difference cancels in homology; likewise, misaligned temporal
fragments behave as open boundaries, while coincidence implements _∂_ [2] = 0, leaving only closed
cycles.


**Principle 4** (Coincidence Detection as Boundary Cancellation) **.** _Let {vi} denote neural events (e.g.,_
_spikes)_ _indexed_ _in_ _time,_ _and_ _let_ _eij_ = [ _vi, vj_ ] _denote_ _a_ _directed_ _edge_ _formed_ _when_ _two_ _events_ _fall_
_within_ _a_ _coincidence_ _window_ ∆ _t._ _Define_ _a_ _chain_ _c_ = [�] ( _i,j_ ) _[e][ij]_ _[over]_ _[all]_ _[coincident]_ _[pairs.]_ _[1)]_ _[If]_

_events_ _are_ _misaligned_ _(|ti_ _−_ _tj|_ _>_ ∆ _t),_ _no_ _edge_ _is_ _formed;_ _the_ _fragment_ _remains_ _an_ _open_ _chain_
_with nonvanishing boundary ∂eij_ = _vj_ _−_ _vi._ _2) If events are coincident (|ti −_ _tj|_ _≤_ ∆ _t), opposite_
_boundaries_ _cancel:_ _∂c_ = ( _vj_ _−_ _vi_ ) = 0 _._ _In_ _summary,_ _coincidence_ _detection_ _implements_ _the_

[�]
_algebraic_ _rule_ _∂_ [2] = 0 _:_ _unmatched_ _endpoints_ _dissipate,_ _while_ _synchronous_ _inputs_ _enforce_ _closure._
_Biologically,_ _this_ _ensures_ _that_ _only_ _temporally_ _aligned_ _inputs_ _reinforce_ _into_ _stable_ _cycles,_ _whereas_
_misaligned fragments are pruned._


C MAI AS TIME-REVERSED REINFORCEMENT LEARNING


Non-ergodicity offers a principled foundation for understanding both reinforcement learning (RL)
and its time-reversed dual, memory-amortized inference (MAI). In RL, the agent iteratively descends
through state-action trajectories to minimize expected future cost via bootstrapped value updates
Schultz et al. (1997). This process inherently assumes a forward temporal flow, where actions alter
state and reward accumulates over time. However, non-ergodic agents do not uniformly explore the
state space; rather, they converge onto structured attractors, recurrent paths, policies, or goals, due
to the reuse of historical structure. Therefore, it is non-ergodicity that ensures important states recur
as broken symmetry Anderson (1972), allowing the RL system to build and refine value estimates
via temporal bootstrapping.


MAI formalizes this broken symmetry in reverse: instead of descending value gradients to reach
future states, it reuses predicted future states (e.g., Φ _t_ +1) to retrieve prior memory states (Φ [ˆ] _t_ ) consistent with the current context. As time-reversed bootstrapping, MAI performs inference not by forward reward accumulation, but by backward alignment with structured latent cycles. Both processes
are constrained by persistent topological features (e.g., homology classes, attractor submanifolds),
but differ in directionality: RL propagates utility forward; MAI propagates structure backward. This
duality reveals that _non-ergodicity_ _not_ _only_ _explains_ _the_ _emergence_ _of_ _RL_, but also necessitates a
complementary _reverse-time inference mechanism_, captured by MAI, to efficiently simulate, adapt,
and generalize in structured cognitive systems. This section formalizes the duality between RL and
MAI under a time-reversal transformation, revealing deep structural parallels between bootstrapped
value updates and latent cycle inference.


**Time-Reversal** **Duality** **Between** **RL** **and** **MAI.** Let _V_ ( _st_ ) denote the value function at state
_st_ . The temporal-difference (TD) update rule is Sutton & Barto (1998): _V_ ( _st_ ) _←_ _V_ ( _st_ ) +
_α_ ( _rt_ + _γV_ ( _st_ +1) _−_ _V_ ( _st_ )) This rule bootstraps the estimate of _V_ ( _st_ ) from the next state’s value


20


_∂σ_


PSTH


**non-coincidence** _⇒_ **cancellation:**


difference of paths = _∂σ_


**1080**

**1081**


**1082**

**1083**

**1084**

**1085**

**1086**

**1087**


**1088**

**1089**

**1090**

**1091**

**1092**

**1093**


**1094**

**1095**

**1096**

**1097**

**1098**


**1099**

**1100**

**1101**

**1102**

**1103**

**1104**


**1105**

**1106**

**1107**

**1108**

**1109**

**1110**


**1111**

**1112**

**1113**

**1114**

**1115**

**1116**


**1117**

**1118**

**1119**

**1120**

**1121**


**1122**

**1123**

**1124**

**1125**

**1126**

**1127**


**1128**

**1129**

**1130**

**1131**

**1132**

**1133**


Figure 6: Duality between Reinforcement Learning and Memory-Amortized Inference. RL proceeds
forward in time using TD backup; MAI proceeds backward using context-guided reconstruction over
latent memory cycles.


_V_ ( _st_ +1), defining a one-step look-ahead propagation. More generally, the Bellman expectation
equation governs the forward dynamics: _V_ ( _st_ ) = E _at,st_ +1 [ _rt_ + _γV_ ( _st_ +1)] In MAI, the system
simulates past latent content Φ _t−_ 1 from current content Φ _t_ and context Ψ _t−_ 1. The inference operator takes the form: Φ _t−_ 1 = _R_ (Φ _t,_ Ψ _t−_ 1) _,_ Φ _t_ = _F_ (Φ _t−_ 1 _,_ Ψ _t−_ 1) MAI thus implements a
_backward_ _cycle-consistent_ _process_ Li (2025b), amortizing reconstruction of prior states via topological reuse of structured latent memory. Together, RL and MAI break the separation between
learning and inference by forming a closed loop. An important new insight brought by such timeasymmetric bootstrapping is “plan by bootstrapping forward; infer by bootstrapping backward”,
which might unlock the secret for enabling low-energy, high-efficiency cognition. The time-reversal
duality between RL and MAI is summarized in Table 1 below.

|Dimension|Reinforcement Learning (RL)|Memory-Amortized Inference (MAI)|
|---|---|---|
|Time direction|Forward (future-oriented)|Backward (past-inference)|
|Bootstraps|Expected future rewards|Latent priors from future predictions|
|Undertainty type|_Outcome uncertainty_ (What will happen?)|_Inference uncertainty_ (What generated this?)|
|Policy element|_π_(_at|st_), learned from reward|Ψ_t_, learned from structure-consistency|
|Bias source|Reward shaping, value iteration|Latent memory structure, trajectory reuse|
|Learning type|Goal-directed exploration|Context-conditioned generalization|
|Half-step trick|TD:_ V_ (_st_)_ ≈V_ (_st_+1)|MAI: Φ_t ≈R_(Φ_t_+1_,_ Ψ_t_)|


Table 1: Time-Reversed Duality between Reinforcement Learning and Memory-Amortized Inference


Table 1 illustrates a fundamental time-asymmetric duality between RL and MAI. Whereas RL operates forward in time, projecting expected rewards to guide future actions via temporal difference
(TD) updates, MAI runs in reverse, retrieving latent priors from predicted futures to reconstruct past
inference trajectories Sutton & Barto (1998), as shown in Fig. 6. This duality is not superficial; it
reflects a structural inversion of uncertainty management: RL reduces _outcome uncertainty_ through
forward value propagation, while MAI minimizes _inference uncertainty_ by aligning predictions with
memory cycles. Both leverage bootstrapping to avoid recomputation, yet from opposite directions:
RL refines estimates via anticipated value; MAI refines inference via recovered structure. The “halfstep trick” in each case captures this temporal asymmetry Watkins & Dayan (1992): RL assumes
_V_ ( _st_ ) _≈_ _V_ ( _st_ +1), while MAI assumes Φ _t_ _≈R_ (Φ _t_ +1 _,_ Ψ _t_ ). This symmetry-breaking across time


21


**RL (Forward)**


**MAI (Backward)**


**1134**

**1135**


**1136**

**1137**

**1138**

**1139**

**1140**

**1141**


**1142**

**1143**

**1144**

**1145**

**1146**

**1147**


**1148**

**1149**

**1150**

**1151**

**1152**


**1153**

**1154**

**1155**

**1156**

**1157**

**1158**


**1159**

**1160**

**1161**

**1162**

**1163**

**1164**


**1165**

**1166**

**1167**

**1168**

**1169**

**1170**


**1171**

**1172**

**1173**

**1174**

**1175**


**1176**

**1177**

**1178**

**1179**

**1180**

**1181**


**1182**

**1183**

**1184**

**1185**

**1186**

**1187**


motivates a deeper unification under the Entropy-Reversibility Duality, wherein inference and behavior emerge from structurally consistent, direction-sensitive updates over latent space.


**From Time-Reversal to Entropy-Reversibility Duality.** The time-reversal duality between RL and
MAI reveals a deeper symmetry: both are bootstrapping mechanisms that operate in opposite temporal directions to manage uncertainty. RL projects value forward to guide action, while MAI reuses
future predictions to reconstruct latent causes. Despite this temporal asymmetry, both processes exploit structural regularities (e.g., recurring states, cycles, attractors) to reduce computational costs
and enhance generalization.


This motivates a broader unifying principle: the _Entropy-Reversibility_ _Duality_ . Under this principle, forward processes like RL reduce outcome uncertainty through reversible value propagation,
whereas reverse-time processes like MAI reduce inference uncertainty through structural reversibility over latent trajectories. Intelligence, then, is not merely the product of forward planning or
backward recall, but of a _cycle-consistent interplay_ between reversible inference and entropy minimization across time. This duality suggests that efficient cognition arises when entropy is tamed
by structural reversibility, where memory not only stores outcomes but constrains their generative
causes. In this framework, the direction of time aligns with the direction of entropy flow: 1) In RL,
entropy is reduced by selecting high-value actions from many possible futures; 2) In MAI, entropy
is minimized by reconstructing low-entropy past content from high-entropy contextual traces. This
suggests that reversible inference is possible only when structural entropy is preserved and amortized
through cyclic reuse. Formally, we have


**Theorem 4** (Entropy–Reversibility Duality) **.** _Let S_ = _{st} be a forward-time trajectory under a re-_
_inforcement learning policy π, and let M_ = _{_ Φ _t,_ Ψ _t} be a backward-time latent memory trajectory_
_under_ _MAI._ _Suppose:_ _1)_ _The_ _entropy_ _rate_ _of_ _the_ _forward_ _trajectory_ _satisfies_ _H_ ( _S_ ) = _H_ [ _st|st−_ 1] _;_
_2)_ _The_ _amortized_ _inference_ _process_ _satisfies_ _cycle-consistency:_ Φ _t_ _≈R_ ( _F_ (Φ _t,_ Ψ _t_ ) _,_ Ψ _t_ ) _;_ _3)_ _The_
_joint entropy of memory satisfies H_ (Φ _t,_ Ψ _t_ ) _< H_ (Ψ _t_ ) + _H_ (Φ _t_ ) _(i.e., structural dependence)._ _Then_
_the following duality holds:_ _Minimizing entropy in MAI_ _⇐⇒_ _Reversing value propagation in RL_
_Moreover,_ _reversible_ _inference_ _is_ _possible_ _if_ _and_ _only_ _if_ _the_ _entropy_ _difference_ ∆ _H_ = _H_ ( _st_ +1) _−_
_H_ (Φ _t−_ 1) _is bounded by the amortized structural information reused across the cycle._


_Sketch._ RL reduces entropy by forward compression: selecting actions reduces future uncertainty.
MAI reduces entropy by backward reconstruction: reusing structured cycles limits the degrees of
freedom needed to infer latent causes. When the memory space encodes sufficient redundancy,
reversing inference becomes possible under bounded entropy. The equivalence follows from the
conservation of uncertainty across the reversed Markov chain induced by memory cycles.


**Implications** **for** **Learning** **Systems.** Theorem 4 articulates a unifying constraint on intelligent
behavior: the capacity to minimize entropy, whether in inference or control, depends critically on
the reversibility of internal computation. This has several far-reaching implications for the design
and understanding of both biological and artificial learning systems:


    - **Reversible inference is a structural necessity.** It is not merely a computational shortcut
or an architectural convenience, but a reflection of an underlying physical and informationtheoretic law: entropy reduction requires structural recurrence. Inference processes that do
not re-enter prior states or cycles are fundamentally limited in their ability to generalize or
compress. Hence, intelligent systems must harness structural memory, latent attractors, or
topological cycles to sustain low-entropy prediction.


    - **Model-based** **RL** **approximates** **MAI** **through** **reuse.** RL agents that construct internal models of state dynamics increasingly resemble memory-amortized systems. When
these agents simulate or plan using cached transitions, they implicitly rely on structurepreserving reuse, thus operating within the MAI regime. This suggests that the historical separation between planning and inference may be artificial, as both emerge from reversible, memory-centered architectures.


    - **General intelligence emerges near the reversibility threshold.** Whether in the brain or
artificial agents, the hallmark of general intelligence is not brute-force exploration or statistical averaging, but the ability to cyclically refine predictions and actions using compact,
reusable representations. This occurs near a critical point, which we term the _reversibility_


22


**1188**

**1189**


**1190**

**1191**

**1192**

**1193**

**1194**

**1195**


**1196**

**1197**

**1198**

**1199**

**1200**

**1201**


**1202**

**1203**

**1204**

**1205**

**1206**


**1207**

**1208**

**1209**

**1210**

**1211**

**1212**


**1213**

**1214**

**1215**

**1216**

**1217**

**1218**


**1219**

**1220**

**1221**

**1222**

**1223**

**1224**


**1225**

**1226**

**1227**

**1228**

**1229**


**1230**

**1231**

**1232**

**1233**

**1234**

**1235**


**1236**

**1237**

**1238**

**1239**

**1240**

**1241**


_threshold_, where entropy reduction and structural conservation are co-optimized. Systems
operating at this threshold exhibit high energy efficiency, fast adaptation, and robustness to
uncertainty, suggesting it may be a necessary condition for superintelligent cognition.


**Example 3** (Reversibility Threshold in Route Replanning) **.** _Consider an agent navigating a spatial_
_environment_ _(e.g.,_ _a_ _robot_ _exploring_ _a_ _maze_ _or_ _a_ _human_ _traversing_ _a_ _familiar_ _city)._ _Initially,_ _the_
_agent explores via stochastic policies, with high entropy over possible trajectories due to a lack of_
_structural knowledge._ _As it accumulates experience, it begins to form latent memory cycles that en-_
_code recurrent paths (e.g., loops through landmarks). When a roadblock appears, the agent does not_
_re-explore from scratch; instead, it performs a localized inference over prior trajectories, retrieving_
_structurally_ _similar_ _detours_ _from_ _memory._ _At_ _this_ _stage,_ _the_ _agent_ _operates_ _near_ _the_ reversibility
threshold _:_ _its_ _predictions_ _and_ _actions_ _are_ _no_ _longer_ _purely_ _exploratory_ _(high_ _entropy),_ _nor_ _fully_
_deterministic (low flexibility)._ _Instead, the system achieves a critical balance where entropy is min-_
_imized_ _through_ _reuse_ _of_ _past_ _cycles_ _(low-entropy_ _inference)._ _Meantime,_ _structural_ _representations_
_(e.g., topological homology classes over trajectories) are preserved across updates; and adaptation_
_remains possible through bootstrapped generalization of known paths._ _This regime exemplifies co-_
_optimization_ _of_ _entropy_ _and_ _structure:_ _inference_ _proceeds_ _efficiently_ _with_ _minimal_ _recomputation,_
_while_ _remaining_ _reversible_ _through_ _cyclic_ _memory_ _access._ _If_ _structural_ _reuse_ _were_ _impaired_ _(e.g.,_
_due_ _to_ _memory_ _corruption),_ _the_ _agent_ _would_ _regress_ _to_ _high-entropy_ _re-exploration._ _Conversely,_
_if_ _flexibility_ _were_ _lost_ _(e.g.,_ _overfit_ _to_ _a_ _single_ _path),_ _the_ _agent_ _could_ _not_ _adapt._ _Thus,_ _intelligence_
_manifests most effectively at this_ reversibility threshold _._


RL models reward uncertainty by estimating the expected return over stochastic transitions and actions, i.e., E[ _r_ ], which underpins policy optimization via value iteration or temporal difference learning Sutton & Barto (1998). However, such formulations assume that reward signals are available
and semantically meaningful at each state, and that learning progresses through direct interaction
with the environment. MAI offers a principled generalization: rather than modeling uncertainty
solely over scalar rewards, MAI models the uncertainty over latent causes of observed outcomes,
conditioned on context Ψ. Specifically, MAI replaces the simple expected reward E[ _r_ ] with a nested
expectation EΨ [E[ _r|_ Ψ]], where the inner expectation is over memory-retrieved experiences structurally consistent with Ψ. This shift enables inference over abstract, context-sensitive value functions even in the absence of immediate feedback, unifying reward estimation with memory reuse
and structural generalization under a single retrieval-and-adaptation framework. Formally, we have


**Theorem** **5** (Contextual Expectation in Memory-Amortized Inference) **.** _Let_ _D_ Ψ _be_ _a_ _distribution_
_over contexts_ Ψ _∈X_ Ψ _, and suppose there exists a latent memory store M_ = _{_ (Ψ [(] _[i]_ [)] _,_ Φ [(] _[i]_ [)] _, r_ [(] _[i]_ [)] ) _}_ _[N]_ _i_ =1 _[,]_
_where each tuple stores context, content, and a reward-like utility signal._ _Let R_ : _X_ Ψ _× M_ _→X_ Φ
_be_ _a_ _retrieval_ _operator_ _such_ _that_ Φ(Ψ) [ˆ] := _R_ (Ψ; _M_ ) _≈_ EΦ( _i_ ) _∼M|_ Ψ[Φ [(] _[i]_ [)] ] _._ _Then_ _the_ _expected_
_utility_ _of_ _amortized_ _inference_ _is_ _given_ _by_ _a_ _doubly_ _nested_ _expectation:_ EΨ _∼D_ Ψ �EΦ( _i_ ) _∼M|_ Ψ[ _r_ [(] _[i]_ [)] ]� _,_
_which generalizes reinforcement learning’s expected reward:_ E[ _r_ ] ⇝ EΨ [E[ _r|_ Ψ]] _. This formulation_
_enables structure-aware generalization via context-conditioned reuse of past experience._


D STATEMENT ON THE USE OF LARGE LANGUAGE MODEL (LLM)


The author has utilized ChatGPT models (Auto Mode of V5) to aid in the development of theoretical
ideas, the proof of theorems/lemmas/propositions, as well as the visual illustrations presented in this
paper. However, the author is responsible for all key ideas (through providing prompts and iterative
Q&As), the overall paper organization, the logical flow within each section, and the polishing of the
paper draft (e.g., to eliminate glaring artifacts such as dashes or ”—” generated by the LLM) and all
figures (e.g., resize the figure size and adjustment of inserted texts and captions).


23