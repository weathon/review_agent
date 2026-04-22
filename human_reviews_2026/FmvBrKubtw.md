# Negotiated Reasoning: On Provably Addressing Relative Over-Generalization

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
We focus on the relative over-generalization (RO) issue in fully cooperative multi-agent reinforcement learning (MARL). Existing methods show that endowing agents with reasoning can help mitigate RO empirically, but there is little theoretical insight. We first prove that RO is avoided when agents satisfy a consistent reasoning requirement. We then propose a new negotiated reasoning framework connecting reasoning and RO with theoretical guarantees. Based on it, we develop an algorithm called Stein variational negotiated reasoning (SVNR), which uses Stein variational gradient descent to form a negotiation policy that provably bypasses RO under maximum-entropy policy iteration. SVNR is further parameterized with neural networks for computational efficiency. Experiments demonstrate that SVNR significantly outperforms baselines on RO-challenged tasks, including Multi-Agent Particle World and MaMuJoCo, confirming its advantage in achieving better cooperation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes a “negotiated reasoning” with an SVGD-based algorithm that proves RO-free convergence under strict conditions. The authors validate the approach on Differential Games, Particle Gather, and MaMuJoCo, improving on returns versus baseline approaches.

### Strengths
* Clean conceptual split (PRO vs. ERO) and a “consistent reasoning” criterion that ties the pathology to modeling assumptions.
* A (mostly) coherent theory: SVNR via nested negotiation + MaxEnt iteration.
* Comprehensive experiments show that ERO consistently improves over reasoning baselines and mainstream MARL on MaMuJoCo.

### Weaknesses
* Guarantees rely on strong assumptions, namely annealing $\alpha$ to 0, and at times finite action spaces, which do not match continuous-control practice.
* Early theory assumes access to the optimal joint policy, then uses an estimator. The gap between proof conditions and the amortized neural implementation is not fully bridged.
* Decentralized execution relies on amortization faithfully reproducing multi-round negotiation, a strong limiting assumption in practice.
* Sample efficiency and robustness are underexplored. At the minimum I would expect training curves for the method versus the baseline approaches.

### Questions
* Clarify which theorems still hold in continuous action spaces without discretization.
* Quantify sensitivity to $\alpha$-annealing with an ablation.
* Tighten the link between Theorem 3.2’s $\alpha \rightarrow 0$ requirement and continuous-control results.
* Will the authors release their code?

### Soundness
3

### Presentation
2

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
The paper tackles the problem of relative over-generalization (RO) in cooperative multi-agent reinforcement learning. The authors point out that under CTDE settings, RO arises in two stages: during training (when agents form a perceived joint policy) and during execution (when they act without seeing others’ final actions). They formalize these as Perceived RO (PRO) and Executed RO (ERO), and show that if all agents achieve consistent reasoning—reasoning about others in a way consistent with their optimal or executed policies—then RO can be avoided. To realize this condition, they propose a Negotiated Reasoning (NR) framework where agents iteratively update joint action beliefs through structured “negotiation.” They instantiate this idea as Stein Variational Negotiated Reasoning (SVNR), which leverages SVGD updates and a nested negotiation structure. A neural amortized version enables efficient decentralized execution. Experiments on several MARL benchmarks demonstrate that SVNR avoids sub-optimal equilibria and performs better than prior reasoning-based methods.

### Strengths
1. Well-motivated and conceptually clear. The paper tackles a long-standing issue in cooperative MARL—relative over-generalization (RO)—from a fresh theoretical angle. The decomposition into Perceived RO (PRO) and Executed RO (ERO) is a intuitive way to separate the effects of exploration during training and coordination failures during execution. This framing alone makes the paper stand out conceptually.

2. Theoretical depth. The authors do not stop at defining RO but go on to establish a sufficient condition—consistent reasoning—under which RO provably disappears. The reasoning flow (formalization → condition → constructive algorithm) is solid and self-contained, which is rare in the MARL literature where many “theoretical” claims are hand-wavy.

3. Negotiated Reasoning framework is novel. The idea of embedding a negotiation mechanism among agents—modeled via particle-based updates and linked to SVGD—is original and technically well-grounded. It provides a new way to understand coordination as an iterative reasoning process rather than just communication or credit assignment.

4. Empirical validation matches the theory. Experiments across both simple and complex cooperative environments (Differential Games, Particle Gather, MaMuJoCo) demonstrate consistent gains.

### Weaknesses
1. Assumptions may be restrictive. The theoretical results rely on strictly nested negotiation sets and the maximum-entropy policy iteration framework. These are strong assumptions, and it’s unclear how sensitive the algorithm is to relaxing them. More empirical discussion of non-strict or sparse negotiation topologies would strengthen the practical claim.

2. Scalability and computational overhead are under-discussed. While the amortized neural version helps, the original SVNR involves particle-based updates and iterative negotiation steps. The paper lacks a quantitative analysis of training cost, memory footprint, or runtime.

3. Limited connection to broader MARL literature.
The paper positions itself mainly against “reasoning-based” methods, but doesn’t fully clarify how its negotiation differs in spirit from other opponent-modeling or communication-based approaches. A clearer conceptual contrast would make the contribution easier to situate.

4. Experimental evaluation is strong in breadth but shallow in analysis.
While performance improvements are clear, the paper could provide more interpretability: e.g., what negotiation dynamics emerge, how many rounds are effectively used, or how PRO/ERO evolve during training. Without this, the reader must take the “negotiation” story largely on faith.

5. Writing density. Some proofs and definitions could use more intuitive explanation or visual support (especially PRO/ERO and the consistent reasoning condition). The theoretical sections are mathematically heavy, which may alienate non-specialist readers.

### Questions
1. Generalization to partial observability: The consistent reasoning definition assumes full observability of state. Do you foresee PRO/ERO or the theoretical guarantees extending to POMDP settings?

2. Practical guidance: For someone implementing SVNR, how should they select the number of particles 𝑀 and negotiation rounds K? Is there a heuristic or convergence indicator?

3. Interpretability: Can you visualize or quantify the evolution of “agreement” or “negotiation” during training to make the proposed reasoning process more transparent?

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
3

### Summary
The paper reframes relative over-generalisation (RO) in cooperative MARL into two operational notions: perceived RO (PRO) during policy updates and executed RO (ERO) during decentralised execution. It argues that if agents reason consistently about teammates (during training and at test), RO can be avoided. To operationalise this, it proposes Negotiated Reasoning and an instantiation, SVNR, which uses (message-passing) SVGD to negotiate joint actions, embedded in a maximum-entropy policy-iteration loop, with a practical amortised neural version for speed. Theory shows PRO-free negotiation under a strictly nested conditional factorisation and (stated) finite action spaces; experiments on RO-heavy games and multi-agent MuJoCo show strong empirical gains.

### Strengths
- Clear, useful split of RO into PRO vs ERO, making the pathology diagnosable during training and execution.
- Principled negotiation mechanism tied to MaxEnt policy iteration (not just a heuristic add-on)
- Stated convergence/guarantee story under a strictly nested factorisation 
- Amortised implementation to distil many negotiation steps into one forward pass; practical ablations on particles, team size, topology.
- Strong empirical wins on PRO/ERO-challenged settings and competitive continuous-control benchmarks.

### Weaknesses
- Core theorems are written for finite action spaces, while key experiments use continuous actions; the theory–practice gap should be tightened or clearly scoped.
- The paper states communication-free execution, yet the pseudocode shares noise variables between neighbours at test time- I think this needs unambiguous clarification.
- Guidance on schedules/sensitivity is thin and could affect stability/robustness.
- The idealised policy-iteration description initially assumes a known model, then switches to critic learning; the implications of this shift aren’t fully analysed.
- Compute/tuning parity vs baselines isn’t fully tabulated 
- Scope limits: assumes CTDE, full observability; robustness under partial observability is left open.

### Questions
- Continuous actions: which parts extend beyond finite $|U|$?
- Do agents share any variables at test time? If not, please fix the pseudocode; if yes, how is this still communication-free?
- How sensitive are results to final $\alpha$ and the annealing schedule across tasks?
- Please provide wall-clock/GPU hours and tuning budgets/ranges for all baselines, and confirm identical exploration/α schedules where applicable.
- Beyond strictly nested, what guarantees (or empirical scaling laws) hold for partial DAGs/peer sampling?
- Can you characterise the error introduced when replacing the model-based policy-iteration view with the critic-based practical algorithm?

### Soundness
3

### Presentation
3

### Contribution
3
