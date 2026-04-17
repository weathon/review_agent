# Global Resolution: Optimal Multi-Draft Speculative Sampling via Convex Optimization

- Decision: Accept (Oral)
- Scores: 6, 8, 6, 6

## Abstract
Speculative sampling reduces the latency of autoregressive decoding for target model LLMs without sacrificing inference quality, by using a cheap draft model to suggest a candidate token and a verification criterion to accept or resample this token. To improve acceptance and decoding efficiency, recent work has explored the multi-draft extension, where at each step $n$ draft tokens are generated, and the verification criterion is a distribution conditioned on these. When this criterion maximizes the probability of accepting some draft token, it is called the optimal transport (OT). However, finding the OT is difficult, as it is the solution of a linear program (OTLP) in over $V^n$ variables, with $V$ being the vocabulary size. Two recent theoretical works have reframed the OTLP in terms of importance sampling or subset selection. In this work, we prove that these formulations are equivalent to an exponentially large relaxed OTLP, so it remains infeasible to solve. Then, we reverse engineer subset selection to formulate the OTLP as a max-flow problem. With a novel application of polymatroid theory, we reduce the exponentially large OTLP to a convex optimization problem in at most $V$ variables. This allows us to devise an algorithm for optimal $n$-draft speculative sampling when the $n$ tokens are chosen i.i.d. from a single draft model, which can be tuned to arbitrary accuracy. Finally, we measure acceptance rates and algorithm runtimes for various $n$ and top-$k$ draft sampling settings. Our findings give the first multi-draft algorithm with 90\% acceptance and under 100 ms of overhead per generated token with negligible deviation from the target model distribution.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies multi-draft speculative sampling for autoregressive decoding. It (i) proves that recent importance-sampling and subset-selection formulations are equivalent to an exponentially large relaxed OTLP, which is still intractable, (ii) reverse-engineer subset selection to a max-flow formulation, and (iii) apply polymatroid theory to reduce the problem to a convex program for i.i.d. drafts from a single draft model. Theoretical guarantees are given and empirical results demonstrate the effectiveness of the proposed method.

### Strengths
- The paper turns an exponential OTLP into a convex program in tolerable number of variables (under i.i.d. drafts) is a substantial conceptual and practical advance.
- It also provides approximation error guarantees, which helps to determine the time-precision trade-off.
- The experiments demonstrated the strong performance of the proposed method.

### Weaknesses
- The main convex reduction requires i.i.d. drafts from a single $q$. There are some works adopt mixture drafts across experts. The iid configuration may be one concern of the proposed method.
- The analysis focuses on solver runtime rather than full end-to-end throughput.

### Questions
- When extending to the multi-step case, can the authors comment on how the compounding error is expected to scale?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the computational intractability of optimal multi-draft speculative sampling. Previous work either computed optimal acceptance analytically (for restricted cases) or estimated it without recovering the optimal transport (OT) itself. The authors prove the equivalence of prior formulations (subset selection and canonical decomposition) and introduce Global Resolution, a convex-optimization-based solver that achieves near-optimal OT in the i.i.d. draft setting. The method reduces an exponentially large LP to a convex problem, with guaranteed deviation and practical runtimes (< 100 ms/token).

### Strengths
Important problem. Tackles a key bottleneck in speculative decoding efficiency for LLM inference.

Theoretical soundness. Clear equivalence proofs (Sec. 4), a max‑flow reduction with complementary slackness (Sec. 5), and convex programs for inner/outer systems with explicit error bounds (Theorems 6.4–6.5, Lemma 6.6). 

Practical algorithm. Global Resolution achieves < 100 ms/token OT‑solve time while maintaining near‑optimal acceptance. This is a substantial empirical improvement over generic LP/max‑flow baselines in the tested regime.

### Weaknesses
Not a major weakness, but the current method is limited to i.i.d. drafts. In practice, sampling without replacement (i.e., enforcing distinct drafts) typically yields better performance. Extending the approach to that regime, as well as to multi‑step setups, is left for future work. However, overall, the paper already makes great progress.

Typos:
Theorem 4.6, Equation (17): missing a summation on the LHS.
Equations (22), (23): the conditional distributions are omitted in the notation.

### Questions
Do Global Resolution extends (with guarantees) to independent but non‑identical drafts or sampling without replacement drafts?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the inefficiency of *multi-draft speculative decoding* for large language models (LLMs), where computing the optimal transport (OT) verification criterion requires solving an exponentially large linear program (OTLP) over \(V^n\) variables. The authors first unify two prior theoretical formulations—importance sampling and subset selection—showing that both are equivalent to a relaxed exponential OTLP. They then reverse-engineer the subset-selection view and reformulate the problem as a max-flow optimization, which, via a novel application of polymatroid theory, is further reduced to a convex minimization problem with at most \(V\) variables.  This reduction yields a new algorithm called Global Resolution, which achieves provably optimal acceptance rates in the i.i.d. single-distribution setting where \(n\) draft tokens are sampled from the same draft model. Empirically, the paper measures acceptance rates and solver runtimes across different numbers of drafts \(n\) and top-\(k\) sampling configurations. Results show that the proposed solver can achieve over 90% acceptance with less than 100 ms overhead per token, and negligible deviation from the true target distribution.

### Strengths
The theoretical contribution of this paper is substantial. It successfully unifies two previously disjoint theoretical perspectives on speculative decoding—importance sampling and subset selection—into a single coherent framework. Building upon this, the authors further derive a novel convex minimization formulation via polymatroid theory, which drastically reduces the exponential complexity of the original OT linear program to a problem in at most \(V\) variables.

### Weaknesses
This paper does not discuss the robustness of the proposed algorithm with respect to temperature. Intuitively, different temperature values shape different draft and target distributions, which could significantly affect the optimal acceptance rate. It remains unclear how the proposed approach performs under varying temperature settings. Moreovere, this paper does not intergrate their algorithm in real-world speculative decoding systems to show how the algorithm/theory can improve the latency of LLM decoding.

### Questions
1. Can you give more details about how you construct the distribution in your experiments, e.g., tempearture. 
2. Can you add some temperature experiments?

### Soundness
3

### Presentation
3

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
First, I apologize for the late review; I was invited to join late.

The paper studies the problem of speculative sampling with a draft sequence length of one (single-step), which can be formulated mathematically as an optimal transport (OT) problem. The paper studies when the optimal transport problem can be efficiently solved (which lets one hence compute the optimal acceptance ratio). In particular cases, the authors derive an efficient algorithm to solve the OT problem. Several experiments demonstrate the empirical efficacy of the proposed method in solving the OT problem vs other approaches, and also demonstrate the effect of changing various parameters on the resulting acceptance rates.

### Strengths
The paper establishes an algorithm for solving the optimal transport (OT) problem arising from speculative sampling. The algorithm has some compelling theoretical properties and has interesting connections with seemingly unrelated topics like max flow and linear programming. Experimental results comparing the resulting proposed algorithm vs other methods of solving the OT problem are convincing of the algorithm's efficacy in solving the OT problem compared to off-the-shelf solving methods. While I had some issues with the theory (see below), the experimental improvements combined with the proposed theory for solving the OT are convincing.

### Weaknesses
The main weakness of the work is as follows:
1. It would be interesting to see that the proposed algorithm improves on one way to demonstrate this improvement would be to demonstrate that the proposed algorithm is better empirically. However there is no empirical comparison to Hu et al (2025) or other papers that have studied this problem. 

2. The proposed algorithm seems to require enumerating over $(H^{\star} \cup T)^n - (H^{\star})^n$ (see Theorem 6.4) (e.g. to compute gradients of $\Phi_T$). Yes $T$ can be chosen, but as $|T| \ge 1$ this still takes time at least $|H^{\star}|^{n-1} \cdot |T| \cdot n \ge |H^{\star}|^{n-1}$. As such, how is the proposed algorithms computationally efficient (which is the point of the paper), am I missing something? It would be great if the authors could clarify this point.

3. The writing could be significantly improved. It seems more conventional to present Theorems 6.4 and 6.5 to begin with, describe (at least at a high level) the algorithm, and then argue for its correctness. The current presentation does so backwards. It also took several reads for me to understand some of the key points of the paper, like how $H^{\star}$ can be obtained efficiently in $O(V \log V)$ time from Hu et al 2025 so therefore it is reasonable to assume knowledge of $H^{\star}$, and that the crucial point is to solve the OT rather than compute $\alpha^{\star}$. Overall the paper has interesting ideas but it was hard to read.

### Questions
Could the authors please clarify weakness 2 above, on the computational efficiency of the proposed algorithm?

It would be also nice if the authors could add some more references and discussion on why solving the single-step speculative sampling problem is important in practice.

### Soundness
3

### Presentation
2

### Contribution
3
