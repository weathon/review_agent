# Neural Interactive Proofs

- Decision: Accept (Poster)
- Scores: 5, 10, 5

## Abstract
We consider the problem of how a trusted, but computationally bounded agent (a 'verifier') can learn to interact with one or more powerful but untrusted agents ('provers') in order to solve a given task. More specifically, we study the case in which agents are represented using neural networks and refer to solutions of this problem as neural interactive proofs. First we introduce a unifying framework based on prover-verifier games (Anil et al., 2021), which generalises previously proposed interaction protocols. We then describe several new protocols for generating neural interactive proofs, and provide a theoretical comparison of both new and existing approaches. Finally, we support this theory with experiments in two domains: a toy graph isomorphism problem that illustrates the key ideas, and a code validation task using large language models. In so doing, we aim to create a foundation for future work on neural interactive proofs and their application in building safer AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper considers a prover-verifier game to improve the correctness of the output of ML models. The verifier, which cannot direct prove the correctness, engages with a set of provers whose outputs are not trustable. The interaction between verifier and provers can make the final answer more trustable. It also formulates their interactions as a zero-knowledge proof. 

Simple experiments are conducted on two use cases: testing graph isomorphism and code validation.

### Strengths
The formulation of the problem into a zero-knowledge based verifier-prover interaction.

### Weaknesses
While I like the formulation, I have some concerns about this paper: 

1. It is unclear on the zero-knowledge aspect. Why do we need to have the zero-knowledge in their interactions? The argument on the potential model stealing needs to be more carefully consolidated. 

2. The actual technical contribution is limited. Neural interactive proof is technically explained with worst-case loss and Stackelberg game. First, I wasn't able to understand the paragraph under Proposition 2, due to the confusing and undefined notations. This seriously affects my understanding about the worst case loss. Second, the bi-level optimisation used to solve the Stackelberg game is obvious, and for a verifier-prover game, I don't think we need to go through a definition of Stackelberg game to reach the bi-level optimisation. 

3. Many technical developments in the paper are not actually implemented and experimented, including the zero-knowledge interactions. This, combined with many typos in the paper (e.g., "a node a similar node the second graph"), shows that the paper requires more developments before it becomes publishable. 

4. The experiments were conducted on two simple experiments. It is unclear if the method is actually better and can generalise to more complex settings. 

In summary, I like the formulation of the problem as a zero-knowledge based verifier-prover game, but such reduction is not well motivated and the actual technical contribution (e.g., zero-knowledge) is not consolidated, neither theoretically nor practically.

### Questions
Please see the weaknesses, and I would like to see more discussions, and technical developments, related to the zero-knowledge proofs.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
2

### Summary
This paper introduces "Neural Interactive Proofs" (NIPs), a framework for enabling a computationally bounded but trusted agent (verifier) to learn to interact with more powerful but untrusted agents (provers) to solve decision problems. The work bridges concepts from interactive proofs in complexity theory with modern machine learning approaches. The paper makes four main contributions:

1. A unifying game-theoretic framework based on prover-verifier games that generalizes existing neural interactive proof protocols
2. Several new protocols, including single-prover (NIP), multi-prover (MNIP), and zero-knowledge variants, each with theoretical guarantees about their properties and computational power
3. A theoretical and empirical comparison of both existing and new protocols
4. Empirical validation on two domains: a graph isomorphism task, and a code validation task using large language models

The work aims to create a foundation for future research on neural interactive proofs and their application in building safer AI systems by providing ways for weaker systems to verify the outputs of more powerful ones. The authors provide theoretical results establishing correspondences between their protocols and valid proof systems, as well as empirical evidence.

### Strengths
This paper stands out for bridging theoretical and practical contributions while maintaining high standards of rigor throughout, as well as for the generality of the neural interactive proofs framework.

1. Originality:
    - Novel unification of interactive proofs with neural networks, bridging complexity theory and machine learning
    - Creative adaptation of game-theoretic concepts to create learnable verification protocols, particularly use of Nash and Stackelberg equilibria and their relationship to proof validity
    - The careful treatment of approximate equilibria to handle practical ML settings
2. Quality:
    - Strong theoretical foundations with rigorous proofs of protocol properties
    - Documentation of hyperparameters, architecture choices, and training procedures
    - Release of reproducible codebase for future research
3. Clarity:
    - Excellent high-level exposition of the key ideas and their implications throughout the paper.
        - I want to call out in particular the description of practical applications of zero-knowledge proofs (L368-369: preventing model stealing via black-box access): this was enlightening.
4. Significance:
    - Addresses a crucial challenge in AI safety: verifying outputs of powerful AI systems when the verifier is computationally bounded but trusted, and the prover is powerful but potentially untrustworthy
    - Provides a theoretical framework that can be extended to many verification scenarios
    - Demonstrates the approach through experiments on both toy problems (graph isomorphism) and large language models (coding challenge)
    - Opens new research directions in combining interactive proofs with machine learning
    - Potential impact on development of safer AI systems

### Weaknesses
I believe there are two central weaknesses of this paper, beyond what is listed in section 7 (lack of use of advanced methods, evaluation only on two domains, lack of evaluation of all protocols):

The biggest weakness is that the scope, motivation, and impact of NIP are not made clear.  Perhaps this is just a lack of context on interactive proofs and game theory on my part, but I cannot tell whether the combination of game-theoretic equilibria with generalization-from-examples for specifying the verifier will be widely applicable and revolutionary, or turn out to be relatively niche in practice.

Some more specific points in an attempt to drive at my confusion:

- Line 107 says “Importantly, we assume that it is impractical or impossible to convert the task description into a specification amenable to standard formal verification tools.” This assumption seems wrong.  Insofar as we are capable of training a non-interactive verifier, and willing to trust that verifier, we can just use the verifier itself (however large it may be) as the formal specification.  Applying more-or-less standard formal verification techniques to theorem statements that contain a full neural net is considered, for example, in [1].  Instead, it seems the assumption to make here is that the task is complicated enough that any system which is powerful enough to (non-interactively) verify the solution is larger or more complicated than what we’d be willing to trust as part of our specification.
- Proposition 1 (L242—245, L770) is used to motivate the insufficiency of `adp`, but the example constructed in the appendix is contrived as far as I can tell.  Sure, it is possible to gerrymander the set of possible prover protocols and verifier protocols so that Stackelberg equilibria are insufficient, but this seems to me to be severely unrealistic.  In realistic settings, we expect that computationally unbounded verifiers always have a winning strategy of “ignore the provers and solve the problem”, and so sufficiency of `adp` or `nip` must be evaluated with regard to some reasonable model of computational limitations.  I did not find any justification in the paper for believing that the game used to prove Proposition 1 is a reasonable toy model of computationally limited provers.
- I would very much like to see an example of a problem where the interactivity permits significant compression of the specification / theorem statement (thereby making the specification easier to trust, as per the motivation of Example 1).  In the absence of this, the case for `nip` over `adp` is much weaker.  The graph isomorphism example is too weak: a verifier for graph isomorphism can be hand-coded to run in polynomial time.  In problems like graph non-isomorphism, where interactivity allows bypassing an exhaustive loop with random checking, we don’t expect significant compression in an interactivity-based theorem statement over a non-interactive theorem statement.  However, transformers cannot use loops, so this theorem statement compression is not relevant to NIP over IP.  Can NIP be fully factored into the classical compression that interactivity gives in IP (that of loops, as far as I’m aware) and the generalization-from-examples discussed on lines 266—269, or is there some interaction between the interactivity component and the neural / game-theory equilibrium component that would suggest that Example 1 is realistic rather than hyperbolic?

Additionally, perhaps relatedly, there is no exploration of how performance degrades as the complexity of verification tasks increases.

A secondary weakness is that, while a valiant attempt is made to present the theory rigorously, the notation is not fully explained and the constraints in one of the central definitions (5) are contradictory.  I call this weakness secondary because it is obvious that the ideas are sound and the definitions are repairable.

Definition 3 in general is under-defined:

- Line 173: ∆ is used without being defined.  What is it?
- Line 174—175: The notation $C(i) := \{c \in C : i \in N \}$ is being used non-standardly.  Strictly interpreting set-builder notation would give $C(i) := C$ or $C(i) := \prod_{i\in N} C$ (depending on whether we are taking a union or a disjoint union), but this is clearly not what is mean.  Perhaps you meant to write $C(i) := \{c \in C : c(i) = 2 \}$?  (Note that defining $N = \{1, \ldots, n\}$ is inconsistent with taking $2 = \{0, 1\}$, so for consistency you have to use 1 for false and 2 for true unless you want to change the definition of $N$ or call out $2$ as a different set.)
- Line 174: “When $\mu(c, t)$ is deterministic” how is non-determinism defined here?  What indication was there that $\mu$ could be non-deterministic?
- Line 176: What does $\sim$ in $i \in N’ \sim \mu(t, c)$ mean?
- Line 178: What is $\rho$?
- Line 178: Is the message chosen randomly per-channel, per-timestep, per channel per-timestep, or per-game?

Definition 5 is also confusing:

- Line 223—225: What is $m^T$?  Is it the same as $m_T$ in L140?  And I presume $1_S$ is the indicator function on membership in $S$?
- Line 223—225: The direction of the inequality seems wrong.  If we are trying to minimize loss, then we want $\sigma$ to have lower verifier loss (be better) when the expected match with the indicator function is higher.
- Line 228—230: Criterion 3 is inconsistent.  L219 says $\mu(c, 0) = \emptyset$ for all $c\in C$, but here we say that $\mu(0, c^\dagger)$ (which I assume is a typo for $\mu(c^\dagger, 0)$, as otherwise it does not type-check) is non-empty.  L220 is also contradicted: $c^\dagger = \{ p \}$ for $p \in N^p$ but $c^\dagger$ is assumed on L220 to equal $\{ v \}$ for some $v \in N^v$, which is only possible if $N^p \cap N^v \neq \emptyset$.  But even if this is the case, it would still contradict criterion 2, which asserts that no verifier can solve the problem, precluding the overlap of provers and verifiers.

[1] Jason Gross, Rajashree Agrawal, Thomas Kwa, Euan Ong, Chun Hei Yip, Alex Gibson, Soufiane Noubir, and Lawrence Chan. “Compact Proofs of Model Performance via Mechanistic Interpretability.” 2024. [arXiv:2406.11779](https://arxiv.org/abs/2406.11779).

### Questions
The main confusion I want addressed is that described above in the weaknesses section as the primary weakness.

Beyond that, some minor questions and comments:

Questions:

- On lines 141—142 the message sequence in the prover-verifier game is stochastic, but on line 165 the loss function on the product strategy space is deterministic without any access to a random seed.  How is this tension resolved?  Does the loss function fix a random seed ahead of time?
- L218: Should $\cup$ be $\sqcup$?  That is, are provers and verifiers required to be distinct?  Otherwise “as $p$ or $v$ respectively” on L219 seems a bit misleading without a caveat that we may sometimes have $p = v$.

Minor comments:

- On line 140, $m_T \in {\top, \bot}$, but on 143—146, 0 and 1 are used instead
- On lines 143—146, $\langle p, v\rangle(x) \in \{0, 1\}$, but on lines 153—155, $\langle p, v\rangle(x)$ is an entire sequence of messages ($\mathbf{m}$).
- Line 179: “assume” should be “say” or “declare” (or just remove “we assume”); it seems that this is not an assumption so much as a definition of what it means for play to terminate.
- Line 228—229: $\mu(0, c^\dagger)$ should be $\mu(c^\dagger, 0)$.
- L218—219 uses $N_p$ and $N_v$ while most other lines use $N^p$ and $N^v$.  I assume this is a typo.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper discusses various games which could be used as multi-agent training setups. The objective of these games is to come up with ways of training models by getting models to interact with each other; this is justified by reference to the problem of training models that are untrusted, and the problem of training models that are smarter than human labelers ("scalable oversight").

The paper provides some formalism for a variety of games that could be used for this purpose, proving various results about them (e.g. about cases where you can bound the worst-case risk).

It introduces some new protocols, such as the neural interactive proof protocol, and some zero-knowledge protocols.

It then evaluates some of these protocols in empirical settings; an algorithmic graph isomorphism setting and an APPS code verification setting.

### Strengths
- I think the problem discussed by this work is important
- I'm not aware of the NIP protocol being investigated in the past (though it is an obvious generalization of other things that have been studied). I think that the NIP protocol is plausibly better than other protocols that have been investigated, so it's a contribution to suggest it and empirically investigate it.

### Weaknesses
This paper has two main contributions: theory and empirical results. I'm concerned that the theoretical results aren't very important, they aren't crucially related to the empirical results, and the empirical results (while promising) aren't very detailed/strong.

## The theoretical results aren't very important

Firstly, I was not persuaded that a lot of the theory is very useful to understand. I assume that the results are correct, but I don't think they're very informative about how to construct better protocols or how to analyze the safety of protocols in this setting.

- I'm not sure that the proofs about worst-case performance can be used in practice, because the conditions are very strong.
- As a less important example, the paper introduces the possibility of zero-knowledge proofs, but only very briefly gestures at why it would be helpful to have a protocol be zero-knowledge in this context.

## The theoretical results aren't crucially related to the empirical results

I don't see a strong connection between the theoretical contributions and the empirical experiments. E.g. the zero-knowledge proof idea is proposed, very briefly justified, and then doesn't appear in the empirical results.

## The empirical results aren't very detailed/strong

The results on graph isomorphism are somewhat interesting but it's hard to take much away from them except as a demonstration that the nip protocol isn't buggy somehow.

The results on APPS have the potential to be really interesting--I can totally imagine a version of this paper where, along the lines of various scalable oversight papers in the past, you implement nip in the APPS setting and demonstrate that it works better than debate etc. But I don't feel like this paper establishes this conclusively or provides substantial insight into why the technique works (e.g. section 6.2 here is much less detailed than section 3 in Khan et al, "Debating with More Persuasive LLMs Leads to More Truthful Answers", https://arxiv.org/pdf/2402.06782).

----

I'm also not persuaded that prover verifier games are a good formalism for developing oversight techniques, compared to e.g. the type of formalism in "AI Control: Improving Safety Despite Intentional Subversion" https://arxiv.org/abs/2312.06942.

### Questions
- How is Figure 3, which shows (nip > adp > solo verifier > debate), consistent with Figure 8, which appears to show that at the end of training that (solo verifier = nip > adp > debate)?
- Can we get error bars in Figure 3, please?
- What future research directions are enabled by the theoretical contributions of this paper, and why are they important?

### Soundness
3

### Presentation
1

### Contribution
2
