# Is my action policy safe? PolIC3 to the rescue

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
The use of machine learning in sequential decision-making tasks has grown substantially, intensifying concerns regarding the safety of learned policies and motivating research on policy verification. We present a new policy verification method based on the well-known IC3 algorithm. Unlike existing approaches, ours decouples reasoning about policy decisions from reasoning about the effects of these decisions on the environment in which the policy is executed. This separation allows us to leverage the latest advances in machine learning certification tools to handle the former subproblem, whilst relying on specialized solvers for the latter. Experiments confirm that our approach scales better and supports a wider variety of policy architectures than current state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work deals with policy verification.
A new method for policy verification is presented, based on the established IC3 algorithm.

### Strengths
* The topic is relevant.
* The idea of using and adapting a technology established in another domain is interesting.
* The abstract is written very well.

### Weaknesses
* The presentation seems somewhat tedious to me. A didactic revision would be desirable here. To achieve this, some details would need to be moved to the appendix.

The description is very technical. For my taste, it reads too much like a software description. I think the motivation and the individual design decisions should be explained in more detail.

Furthermore, what specific steps make the novelty should be better explained.

I am having difficulty with the presentation. I would appreciate a more detailed introduction to the problems encountered to date and the innovations presented. I would also like to see more text comparing this work with existing studies. This would be possible in a journal. In my opinion, this work, which is both highly specialized and theory-heavy, would require a complete overhaul in order to be presented as a conference paper.

Further comments:

Sometimes it says “action policy,” sometimes “policy,” sometimes “decision policies.” I think using “policy” consistently would be clearer.

“argmax” should not be italicized.

Since “a priori” is italicized, “i.e.,” “e.g.,” and “et al.” should also be italicized for consistency.

„markov“ -> „Markov“

### Questions
What still needs to be done so that the method can be used in a practical application?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies policy safety verification for discrete environments. It adapts IC3 and proposes POLIC3, which decouples reasoning about the policy and the environment. The environment is a guarded command model; the policy is a neural net or a tree ensemble. The key step replaces an exact “policy frame transition” test with a necessary-condition surrogate that splits action feasibility and policy selection so that standard SMT handles environment reachability, and standard certification tools handle policy decisions. Experiments on several integer-variable planning benchmarks show higher coverage and faster runtimes than prior policy verification baselines, and the method can handle more complex policies such as ASNets through interface nets.

### Strengths
The paper writes a generic IC3 with four abstract subroutines and gives the assertions needed so that termination and correctness follow. It then maps these to environment models and to policy-constrained transition systems.

The paper explains why a direct SMT encoding of \pi is brittle, then proposes a split test with theorems that show rejection is sound: if the relaxed test fails, the exact test is false. It also gives practical constructions that reduce the policy side to LiRPA for nets and VERITAS for trees.

On known FFNN and DTE policy benchmarks, POLIC3 matches or improves solved instances and often reduces runtime, and it scales to ASNet policies through interface functions. A coverage table and per-instance runtime plots support this.

### Weaknesses
The core split is only a necessary condition, not equivalent to the exact policy frame test.
The method replaces the exact test with separate checks over transitions and over policy choice. Theorem 2 and Corollary 1 show only that failures of the relaxed test imply failures of the exact test, not the converse. This can produce many false positives in the relaxed test and weak pruning. The paper claims the IC3 assertions still hold, but the split gives no completeness guarantee about when refinement will cut enough states to ensure practical convergence speed. A short discussion that quantifies this gap would help.

The policy side check replaces max over applicable actions with max over an under approximation  $A\inA(s)$ that must hold for all states in r. How A is computed with SMT, how conservative it is under large guards, and what happens when guards use disjunctions or derived predicates are not discussed. Errors here directly affect the soundness of the rejection step and the strength of pruning.

For successor selection the paper enumerates commands rather than solving the approximate policy frame test, to avoid SMT complexity. This is simple but it can be expensive when branching is high. A cost bound or empirical trace would be useful to show that this choice does not dominate runtime on hard instances.

### Questions
For which classes of r and guards is the relaxed policy frame test equivalent to the exact test? Can you give a counterexample where it is loose and how often that arises in practice?

How do you compute A in SMT when guards include disjunctions or action precondition structure is complex? How sensitive is pruning to the size of A?

Could you share data on clause sizes and variable orders for the hard FFNN cases and show a heuristic that consistently shrinks reasons?

What guarantees do you have that F and G preserve the decisions of the original ASNet over the JANI model semantics? If none, can you bound the deviation or add a check that flags states where the mapping is ambiguous?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a policy verification method based on the IC3 algorithm. The algorithm answers the reachability question - that is, whether a particular state is reachable in at least one of the executions of the policy.
Essentially, the paper suggests to treat policy verification as a model checking problem for a safety (reachability) property.

### Strengths
The approach appears novel - at least I didn't see any previous work with an overlapping contribution. The paper is well-written, and the claims about efficiency are believable. The approach is implemented and demonstrates good experimental results.

### Weaknesses
The theoretical contribution is weak: it consists of applying an existing algorithm to a known problem.

Moreover, the claim of solving policy verification with IC3 overstates the contribution somewhat, as the paper presents verification of reachability properties of a policy only.

### Questions
1. Can you use a SAT solver for this problem as well (performance problems notwithstanding)?
2. Can you verify liveness properties?
3. Traditionally, verification of policies is probabilistic. It seems that your approach cannot provide probabilistic guarantees, is that correct?

### Soundness
3

### Presentation
3

### Contribution
2
