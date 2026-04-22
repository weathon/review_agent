# Just-in-Time Piecewise-Linear Semantics for ReLU-type Networks

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
We present a JIT PL semantics for ReLU‑type networks that compiles models into a guarded CPWL transducer with shared guards. The system adds hyperplanes only when operands are affine on the current cell, maintains global lower/upper envelopes, and uses a budgeted branch‑and‑bound. We obtain anytime soundness, exactness on fully refined cells, monotone progress, guard‑linear complexity (avoiding global $\binom{k}{2}$), dominance pruning, and decidability under finite refinement. The shared carrier supports region extraction, decision complexes, Jacobians, exact/certified Lipschitz, LP/SOCP robustness, and maximal causal influence. A minimal prototype returns certificates or counterexamples with cost proportional to visited subdomains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces a new neural network reasoning method based on a compilation of the network into an automaton with polyhedral edge guards and node and edge weight matrices. The authors show that any neural network with piecewise linear activation functions can be represented exactly in this automata language. A dynamic ("just in time") expansion of the automaton is developed to analyze different properties (e.g., robustness, equivalence, etc) over the network's input-output behavior. To this end, the method features mechanisms to propagate input constraints and to introduce dynamically additional hyperplanes refining the guards if necessary. A thorough theoretical analysis studies a wide range of properties of this method, such as correctness and completeness for different analysis tasks, and the connection to concepts from geometry. A small empirical study show cases the flexibility of the developed method.

### Strengths
Owed to requirements in many applications, the formal analysis of learned functions, particularly neural networks, is receiving and has received significant attention in ML research in the recent years. This paper introduces a (to my knowledge) completely novel idea in this context. The algorithm is highly flexible in tackling a wide range of different analysis tasks. The paper evaluates this new method thoroughly from a theoretical perspective. Detailed proofs are provided in an appendix.

### Weaknesses
The write-up is very dense. Essentially, the main text is just a lose connection of different theoretical results, without providing a clear coherent story, or even explaining how the proposed method works exactly. Some of the introduced processing steps seem to be needed for some specific analysis task only, yet the paper lacks any sort of explanation when some steps are executed, in which order, and what context they are relevant. An overview of the overall method such as in the form of pseudocode would greatly improve comprehensibility. As it stands now, the paper is impossible to digest for non-expert readers. The general telegraph style of the write-up is not exactly beneficial for the clarity either.

The purpose of the experimental evaluation is also unclear. Given that the authors propose a new method for existing and well-researched problems, I would expect to see a comparison to state-of-the-art methods. Nevertheless, the authors just ran their methods on three different analysis tasks and different network architectures. This might show case the versatility of the proposed method, but apart from this it is not clear what to conclude from the results. Questions like the following remain open, but are really what should be shown in the empirical study: How does the method scale? Are the shown results in any sense "impressive", e.g., in terms of dealing with network sizes so far beyond reach or handling properties that were so far not considered? How is the method placed with respect to other approaches? How effective is the algorithm, in terms of bound propagation and in terms of the number on-demand splits generated vs. unfolding the complete computation tree.

The paper certainly makes a major contribution, but given its current presentation and the unclear experiment design, it needs another thorough revision.

### Questions
1. Have you compared your algorithm to other state-of-the-art methods, and if so, how does it perform?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Just-in-Time Symbolic Weighted Transducers (JIT-SWT) as an execution semantics for CPWL neural networks.

Rather than completely expanding a ReLU network's exponentially large region decomposition, JIT-SWIT builds a guarded CPWL transducer and refines it on demand during the analysis.

The framework support sound any time envelopes, exactness on fully refined cells, formal reasoning tasks, and some experiments on FFNs, CNNs, GNNs.

### Strengths
Originality: The formalization seem novel and the idea of bridging algebraic formalisms and neural verification seems new.

Quality: Comprehensive theoretical framework with proofs of soundness, decidability, and budgeted complexity bounds.

### Weaknesses
Accessibility: The main text is quite dense and does not provide enough background, motivation, or illustrations for several of the definitions and proofs in the paper. A brief paragraph before each definition or at the beginning of each subsection providing an intuition on why the concepts are needed, or how they work at an intuitive level would be appreciated.

Scalability: Experiments are on small networks and restricted settings. A clearer discussion on when this can be applied and when it cannot would be appreciated.

Related work: It is not clear if this is the only solution out there for neural verification, geometry, and causal analysis. Are there other solutions that could be used for this? If yes, how does this compare to them? In which regards is this solution better?

### Questions
Is this approach the first one of its kind? Have other solutions been proposed in the past? How does JIT-SWT compare to them? If it is the first of its kind, what were the challenges in the past that prevented JIT-SWT from happening? What were the insights that allowed to overcome these challenges?

What are the scalability limits of this approach?

Can more concepts be moved into the appendix so there is room for more intuitions and examples that can help readers who are not very familiar with this topic understand the quality of this work?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a "Just-in-Time" (JIT) Symbolic Weighted Transducer (SWT) framework to model ReLU-based neural networks. Unlike traditional static analysis that suffers from exponential explosion by eagerly materializing all regions, this approach performs on-demand refinement. It maintains sound anytime lower/upper envelopes and guarantees exactness when cells are locally fully refined. The authors demonstrate that this unified "carrier" can support diverse tasks—including region extraction, exact Jacobian computation, Lipschitz estimation, robustness verification, and causal influence analysis—by reducing them to sequence of local Linear Programming (LP) or Second-Order Cone Programming (SOCP) problems.

### Strengths
1. Unified Theoretical Framework: The primary strength is the unification of disparate geometric and formal tasks into a single symbolic carrier. Instead of specialized algorithms for Lipschitz estimation vs. robustness verification, the JIT-SWT provides a common substrate where these become differing objectives on the same refinement engine.
2. Anytime Guarantees & Monotonicity: The proofs for sound anytime envelopes (DYN-1) and monotonic progress without regression (DYN-3) are theoretically satisfying. This allows the method to act as both an exact verifier (given enough time) and an approximate certifier under budget constraints.
3. Budgeted Complexity: Explicitly bounding the complexity (number of active GuardSets, library size) linearly by the number of splits and inserted guards (DYN-4) is a strong theoretical contribution, distinguishing it from purely exponential worst-case static analysis.
4. Decidability: The proof that properties expressible with finite affine/SOC constraints are decidable under fair JIT refinement strategies (DYN-6) places the method on solid formal ground.

### Weaknesses
1. Scalability Concerns: While "JIT" mitigates immediate exponential explosion, the reliance on solving LPs/SOCPs in the inner loop of the Branch-and-Bound driver 8 represents a significant computational bottleneck. The experimental validation is limited to very small-scale problems (MNIST FFNs, subsets of CIFAR-10 with tiny CNNs, Karate Club GNNs). There is little evidence this approach can scale to standard Deep RL policy networks (e.g., ResNet-50 size or Transformer-based policies), where the number of active paths can be overwhelmingly large even for local properties.
2. Restriction to CPWL: The framework is rigorously defined only for CPWL networks (affine + Max/ReLU type gates). Modern state-of-the-art architectures, particularly in RL (Transformers, world models), rely heavily on non-CPWL operations like Softmax, Attention, and LayerNorm. The paper notes this limitation and suggests "sound PL relaxations" as future work11, but without this, its applicability to current high-impact models is severely limited.
3. Practicality of "Exact" Metrics: While exact Lipschitz or $I_{max}$ computation is theoretically appealing, in many DL/RL applications, tight upper bounds (obtainable faster via specialized abstract interpretation methods like $\alpha,\beta$-CROWN) are often sufficient. The paper does not extensively compare the cost-benefit ratio of JIT-SWT against state-of-the-art incomplete verifiers on larger benchmarks.

### Questions
1. You mention "certified PL lifts for multiplicative modules" (like Attention/Softmax) as future work. Do you envision these relaxations fitting neatly into the current SWT guard structure, or will they require a fundamental redesign of the semantic carrier to handle fundamentally non-linear/non-convex regions efficiently?
2. Comparison with Incomplete Verifiers: How does JIT-SWT compare in wall-clock time against state-of-the-art incomplete verifiers (such as $\alpha,\beta$-CROWN) when the goal is only to find a sound bound rather than an exact one? Does the overhead of the unified SWT structure make it competitive for purely bounded verification tasks on larger networks?
3. Sensitivity to Scheduler & Budget: The decidability guarantees rely on a "fair scheduler" and sufficient budgets. In practice, how sensitive is the quality of the returned bounds to the choice of micro-policy (e.g., max-gap vs. round-robin) when operating under tight computational budgets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes JIT-SWT, a just-in-time semantics for analyzing ReLU-type networks as continuous piecewise-linear (CPWL) functions. The main idea is to compile networks into guarded transducers with shared guards, then refine on-demand rather than enumerating all fragments upfront. This method could avoid the computational explosion and will benefit the scalability aspect of tasks, including computing Lipschitz constant, robustness checking, etc. The main contributions are summarized as follows:
1. The framework for calculus and static compilation rules for CPWL NNs. 
2. JIT refinement with soundness guarantee and complexity bounds. 
3. Geometry and verification procedures leveraging the shared carrier
4. Experiments on MNIST/CIFAR-10/Karate Club demonstrating compiler correctness and correlations with network properties

### Strengths
- The problem formulation is clear. Unrolling ReLU layer-by-layer results in exponentially growing computational complexity. This paper proposed a principled approach with formal guarantees. 
- Guard-sharing and unified representation are neat contributions. Inserting comparator faces only when needed yields better computational complexity. Having a single object support multiple verification tasks (Lipschitz, robustness) through a common LP/SOCP interface could meaningfully simplify verification pipelines.

### Weaknesses
- The paper is hard to read. The main text still feels like navigating a theorem catalog rather than a coherent story. The paper is dense with theorem families and heavy notations. It lacks descriptions of theoretical results and their role in the proposed method. Table 1 helps a bit, but a major streamlining pass is needed. 
- Results AF-1 through AF-5 appear to be standard CPWL preliminaries (closure, continuity, differentiability). The paper should explicitly delineate which theoretical results are novel versus which are foundational background, making it easier to assess the core technical contributions.
- The experiments lack comparisons with existing works. We have no information about runtime and memory consumption relative to state-of-the-art tools, like benchmarks in VNN-COMP. Without baseline comparisons, the practical value remains unclear.

### Questions
1. Can the authors provide a clearer roadmap of how the theorem families (AF, SWT, DYN, GEO, VER, CAU) relate to each other and to the overall method? Specifically, which results are novel contributions versus foundational CPWL preliminaries?
2. The scope of the paper in the literature is lacking; there are literally hundreds of papers on neural network verification now, and the paper cites <10, none more recent than 2020. Either can the authors place the paper better in the existing vast literature and compare it qualitatively to what exists, or add comparisons against VNN-COMP benchmarks (and participating tools, eg, Marabou, alpha-beta-Crown, nnenum, NNV, NeuralSAT, Pyrat, etc.) and report wall-clock time, memory usage, and certified accuracy compared?
3. The decidability result (DYN-6) assumes finite refinement and fairness. In practice, how often does the system return UNKNOWN under realistic budgets?

### Soundness
2

### Presentation
1

### Contribution
3
