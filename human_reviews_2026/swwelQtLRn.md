# Diversified Multinomial Logit Contextual Bandits

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Existing contextual multinomial logit (MNL) bandits model relevance-driven choice but ignore the potential benefits of within-assortment diversity, while submodular/combinatorial bandits encode diversity in rewards but lack structured choice probabilities. We bridge this gap with the *diversified multinomial logit* (DMNL) contextual bandit, which augments MNL choice probabilities with a generally submodular diversity function, thereby formalizing the relevance—diversity trade-off within a single model.
Incorporating diversity renders exact MNL assortment optimization intractable. We propose a *white-box* UCB-based algorithm, `OFU-DMNL`, that constructs assortments item-wise by maximizing optimistic marginal gains, avoids black-box optimization oracles, and provides end-to-end guarantees.
We show that `OFU-DMNL` achieves at least a $(1-\tfrac{1}{e+1})$-*approximate* regret bound $\tilde{O}\big(d \sqrt{T/K}\big)$, where $d$ is the context dimension, $K$ the maximum assortment size, and $T$ the horizon, and attains an improved approximation factor over standard submodular baselines. 
Experiments demonstrate consistent gains and, relative to exhaustive enumeration, comparable regret with substantially lower runtime. Overall, DMNL bandits provide a principled and practical foundation for diversity-aware assortment optimization under uncertainty, and `OFU-DMNL` offers a statistically and computationally efficient solution.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Diversified Multinomial Logit (DMNL) Contextual Bandits, extending the Multinomial Logit (MNL) choice model to explicitly promote diversity in the chosen assortments. The authors incorporate a submodular diversity function directly into the choice probability formulation. Finding optimal assortment in DMNL is intractable and requires exhaustive search. To tackle with this difficulty, the authors developed a UCB-based method that achieves a (1-1/(e+1)) approximate regret guarantee. The claimed theoretical result parallels known regret bounds for standard MNL bandits under uniform revenue assumptions.

### Strengths
The DMNL is a novel model that directly incorporates diversity as a submodular function added to the choice probability. This integration is conceptually appealing, as it allows diversity to influence user choice in a principled way rather than as a post-hoc balancing approach, seen in existing work.

To approximately solve the assortment selection problem, the authors constructed a UCB-based item-wise greedy algorithm that estimates the unknown relevance utility and diversity parameters jointly, avoiding exhaustive enumeration. The joint estimation of both parameter sets is a notable contribution. The authors further prove a regret bound of the algorithm that achieves comparable performance to existing MNL bandits, despite the added complexity of estimating diversity parameters.

### Weaknesses
The main weakness lies in the technical depth of the paper. While the formulation of the DMNL model is conceptually novel, the theoretical contributions are somewhat incremental. The improved approximation rate is a relatively straightforward derivation from the structure of the submodular function, which should be stated more of a proposition rather than theorem.
The item-size optimistic construction largely follows from UCB based algorithms that are commonly used in semi-bandit combinatorial multi-armed bandit literature. Although the adaptation to the DMNL context is sound, it seems not introducing new insights beyond existing frameworks. The joint estimation of $\theta$ and $\lambda$ comes from a relatively simple reparameterization of the choice probability that effectively treating diversity as an additional feature dimension. In addition, the presentation could be improved. The novelty is somewhat obscured by lengthy discussions that follow definitions and proofs, which destroys the flow of the paper.

### Questions
1. For the regret upper bound, is it matching in all parameters? It would be helpful to discuss how the additional diversity parameter introduces looser constraints.
2. A discussion on lower bound is also desirable. 
3. What is the exact setup of the numerical experiments shown in Figure 2? A brief introduction would be desirable. Providing this context would make the figure much easier to understand without referring back to the appendix.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes DMNL, an extension of the MNL choice model with a diversity parameter $g_t(S)$, where the diversity of the assortment in some sense modifies the probability of the outside option. A major challenge of this new model is that it is not possible to "greedily" solve the offline best assortment problem. As a result, the authors present $\gamma-$regret (approximate regret bound)

### Strengths
The paper is overall well-written and easy to understand. The model of DMNL is novel and bridges the gap between MNL choice model and diversity-focused assortment

### Weaknesses
1. I hope the authors can motivate the model better. Modeling the diversity via the submodular function $g_t$ seems a bit contrived. Is there a good understanding why should be the diversity of assortment be modelled in this way? e.g. Appendix C.2 gives some mathematical examples, can authors suggest if there are any examples where those functions may be appropriate?

2. The strict submodularity assumption is quite strong

3. Why is uniform exploration of lines 4-5 (Algorithm 1) required? What step of the proof fails without it?

4. Line 240 (Claim of "composition of a submodular function with a non-decreasing .....preserve submodularity" requires a proof or reference . it is not obvious.

5. What is utility of Theorem 1? In what scenarios/instances is it helpful to give a stronger constant in regret?

### Questions
1)$g_t$ needs to be sub-modular? It is not specified explicitly

also see the above weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Proposes a diversified MNL (DMNL) contextual bandit: standard MNL utilities plus a submodular diversity term; gives a greedy OFU algorithm with an improved approximation constant over 1−1/e and √T-type regret; synthetic results show better relevance–diversity trade-offs than MNL baselines.

### Strengths
- Bridges choice modeling and diversity via DMNL; captures the relevance–diversity tension within the click probabilities rather than via ad-hoc reward shaping.
- Item-wise optimistic greedy avoids black-box combinatorial oracles yet has provable guarantees

### Weaknesses
- The algorithm takes \(g_t(S)\) as given. Many applications may only have noisy diversity signals; robustness to misspecification or learned $g_t$ is not explored.
- Experiments appear synthetic; adding a real recommendation/assortment dataset would strengthen the contribution

### Questions
- Is \(g_t(S)\) assumed exactly known at decision time? How would OFU-DMNL adapt if \(g_t\) is observed with noise or must be learned from implicit feedback?
- Can guarantees be given under submodularity (ω=0) ?

### Soundness
3

### Presentation
3

### Contribution
3
