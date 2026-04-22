# Nonparametric Contextual Online Bilateral Trade

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
We study the problem of contextual online bilateral trade. At each round, the learner faces a seller-buyer pair and must propose a trade price without observing their private valuations for the item being sold. The goal of the learner is to post prices to facilitate trades between the two parties. Before posting a price, the learner observes a $d$-dimensional context vector that influences the agent's valuations. Prior work in the contextual setting has focused on linear valuation models, where valuations are linear functions of the context. We provide the first characterisation of a general nonparametric setting in which the buyer’s and seller’s valuations behave according to arbitrary Lipschitz functions of the context. We design an algorithm that leverages contextual information through a hierarchical tree construction and guarantees regret $\widetilde{O}(T^{(d-1)/d})$. Remarkably, our algorithm operates under two stringent features of the setting: (1) one-bit feedback, where the learner only observes whether a trade occurred or not, and (2) strong budget balance, where the learner cannot subsidize or profit from the market participants. We further provide a matching lower bound in the full-feedback setting, demonstrating the tightness of our regret bound.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work studies contextual online bilateral trade under strong budget balance and one-bit feedback. At each round t, a d-dimensional context $x_t ∈ [0,1]^d$ arrives, buyer and seller valuations are generated deterministically by unknown L-Lipschitz functions $f_b$, $f_s: [0,1]^d → [0,1]$, and the learner posts a single price p_t; a trade occurs if and only if $f_s(x_t) ≤ p_t ≤ f_b(x_t)$. The objective is to maximize cumulative gain-from-trade, $GFT(p_t|s_t,b_t) = I(s_t ≤ p_t ≤ b_t)(b_t − s_t)$, equivalently minimize regret against the in-hindsight optimal policy that knows $f_b,\ f_s$. The main contribution is a nonparametric, context-sensitive algorithm with a hierarchical dyadic partition (an adaptive tree) of the context space. For each node (level $l$, reference point $z$), the algorithm first runs a REDUCE routine (probing two prices separated by $L·2^{−(l−1)}$ to force rejections), which geometrically constrains future GFT to $O(L·2^{−l})$ in that region by Lipschitz arguments; then a GUESS routine samples uniformly from an ε-grid around the parent’s marking price to discover an acceptable price and mark the node. With a careful per-node regret decomposition and tuning $ε = L·T^{−1/d}$, the algorithm achieves regret $\tilde{O}(L·T^{(d−1)/d})$ under one-bit feedback and strong budget balance. A multi-scale variant removes the need to know L a priori with only a $polylog(L,T)$ loss, giving $\tilde{O}(L^2·T^{(d−1)/d})$. Finally, the paper proves a matching (up to logs) minimax lower bound $\Omega(L·T^{(d−1)/d})$ even under full feedback, establishing the optimal rate in this noise-free Lipschitz setting.

### Strengths
1, Novel problem formulation: The REDUCE+GUESS design neatly exploits the problem’s structure—deterministic Lipschitz valuations, discontinuous GFT, and one-bit feedback—using geometric constraints to tame local regret and randomized price probing to quickly mark nodes.

2, Tight regret bounds: The upper bound $\tilde{O}(L·T^{(d−1)/d})$ with one-bit feedback and strong budget balance, plus the matching lower bound under full feedback, gives a compelling and crisp rate-optimal result in a challenging, nonparametric setting.

### Weaknesses
The most crucial issue that occurs to me is the dependence on exact Lipschitz structure (deterministic, noise-free L-Lipschitz f_s, f_b), which is central to both the algorithm and the regret analysis. However, it is not entirely clear how robust the guarantees are to even mild model misspecification (e.g., small additive noise, piecewise-Lipschitz with a few discontinuities, or approximate Lipschitzness with slack). The core REDUCE lemma (bounding future GFT by O(L·2^{−ℓ}) after two targeted rejections) drives the entire regret accounting across the tree, yet it relies critically on tight Lipschitz geometry and the absence of stochasticity in (s_t, b_t). In many practical bilateral trade scenarios, context-to-valuation mappings can exhibit small noise, context-specific irregularities, or Lipschitz violations near local kinks. Under such perturbations, the argument used to confine valuations “near the diagonal” after REDUCE may no longer hold with the same strength, potentially causing (i) GUESS to spend substantially longer at certain nodes, (ii) per-node regret to accumulate beyond the current bounds, and (iii) the overall O~(T^{(d−1)/d}) rate to degrade. While the paper’s setting is clearly stated and the results are strong within that model, this fragility leaves uncertain how well the approach carries over to slightly more realistic contexts where, for example, valuations incorporate measurement error, institutional frictions, or adversarial but small perturbations.

Besides, some typos to notice:
1, Line 079: “breaching” → “branching”?
2, Line 461: “Ramark”
3, Overall: “nonparametric” versus “non-parametric”

### Questions
The GUESS routine uses a uniform ε-grid in an interval of length O(L·2^{−(ℓ−1)}), and the analysis leverages a success probability of order 1/|grid| to bound expected time-to-mark. In high levels (large ℓ), the interval shrinks, but the number of active nodes grows; conversely, at shallow levels, the interval is wider but fewer nodes contribute. Could the authors elaborate on how sensitive the overall bound is to the specific grid spacing ε = L·T^{−1/d}? In particular, if contexts are adversarially clustered so that the algorithm revisits certain “hard” nodes disproportionately, does the node-wise decomposition still preclude superlinear accumulation at those nodes? A short intuition (beyond the formal summations) on how the per-node “mark fast or suffer only small GFT” dichotomy prevents pathological concentration would help readers build geometric intuition for the optimal T^{(d−1)/d} rate.

(Will increase the score if the authors properly address my concerns.)

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
The problem of bilateral trade is a new context for online pricing problems. In essence, compared to online pricing, both a buyer and a seller present themselves with private valuations, and the agent must propose a clearing price, which, if accepted by both parties, will generate some social welfare that the agent pockets. One can also view this as a market maker pocketing the bid-ask spread, but in a setting where there is no market and thus bid/ask prices are hidden. 

The objective in this paper is to study the problem with features and, in particular, the case of Lipschitz valuation models when the agent doesn’t observe his own reward, only whether or not a sale happened. The specific online learning set-up considered is adversarial contexts, from which fixed (i.e. stationary in time) but unknown Lipschitz functions generate the hidden bid and ask prices. The observability set-up is what is known as _one-bit_ feedback in online pricing, wherein only a 0/1 signal is observed corresponding to whether or not the transaction cleared. 

Using sequential discretisation of the context space, the authors propose an algorithm whose regret (with respect to an omniscient planner) is of order $\tilde{\mathcal{O}}(T^{\frac{d-1}{d}})$, while using Yao’s principle to provide a matching lower bound. This generalises a prior work on linear valuation models.

### Strengths
The results claimed are clearly presented and their proofs are succinct yet rigourous. Overall the paper's soundness is commendable from a mathematical standpoint. I would also like to highlight that the authors openly point out a limitation of their main algorithm (dependence on knowledge of the Lipschitz constant of the valuation functions) and provide an adapted algorithm.

The presentation is clear and the text is easy to follow, mostly due to the relatively new nature of the online bilateral trade problem: the problem setting is easy to explain and related works are few and well-differentiated. Nevertheless, the quality of the writing is plain to see: it is fluid, and I spotted only very few typos.

### Weaknesses
The main weakness for me is in the contribution, or perhaps its contextualisation. While technically sound, the results of the paper didn’t bring me any exciting new insight into online learning/estimation. While bilateral trade is an interesting setting, I didn’t find a clear characterisation of what key properties it has, to what extent they are unique or generalise, say to/from online auction problems. I think a broader characterisation of the problem and its properties (or a clear application if the authors would rather go that route) would add to the conversation, helping to differentiate this setting from the myriad other bandit-like settings.

The following are minor feedback:

- I wouldn’t use ${\bf Area}(N_{\ell,z})$ to denote a set, as area refers to a scalar quantity in geometry. I would pick another synonym or word which doesn’t have a common use in mathematics, such as region, footprint, shadow…

- Technically, regret was only defined for a sequence of times from $1$ to $T$, so expressions such as $\ell$657 are under-defined, though they are comprehensible. At a minimum $(x)\_{t=1}^T$ and $(p)\_{t=1}^T$ should be passed as arguments along with $(f_b,f_s,T)$.

### Questions
While I remain sceptical of accepting this paper, the following questions might affect my view on the paper:

i) Technical challenges discuss the difficulties inherent in the setting, but only compared to classical online learning settings ($\ell 70$). What complexities are unique and insightful to bilateral trade / Gain From Trade when compared with online one-bit learning in a posted-price auction, for example? 

ii) The algorithm obtained is not anytime (it depends on prior knowledge of $T$. While it is always possible to add a doubling trick, what is fundamentally in the way of making the algorithm properly anytime?

### Soundness
4

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
This paper studies contextual online bilateral trade in a nonparametric setting in which the buyer’s and seller’s valuations behave according to arbitrary Lipschitz functions of the context. The authors develop an algorithm through a hierarchical tree construction, achieving a low regret of $O(T^{\frac{(d-1)}{d}})$, assuming the valuation functions are Lipschitz.

The regret is better than the lower bound of $O(T^{\frac{(d+2)}{d+3}})$ for the general cases by exploiting the specific shape of the gains-from-trade function. The authors also develop an algorithm agnostic to the Lipschitz constant, and provide a matching lower bound.

### Strengths
The paper is generally well-written and clear. The collection of theoretical results is non-trivial and technically strong. The authors provide a complete picture for regret in contextual online bilateral trade in a nonparametric setting.

### Weaknesses
1. It seems that the overhead in the regret bound is pretty large, which might make the proposed algorithm not applicable in practice.

2. The paper would be stronger if the authors can conduct empirical studies to validate the effectiveness of the proposed algorithm for practical instances.

3. This paper could be better suited for other theory and/or econ conferences, such as EC/SODA/STOC/FOCS.

### Questions
1. Could the authors comment on the overhead in the regret bound and applicability of the proposed algorithm in practice?

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
3

### Summary
This paper studies the problem of contextual online bilateral trade in a general nonparametric setting. At each round $t$, a learner observes a context vector $x_t \in [0, 1]^d$ and must post a single price $p_t$ to a seller-buyer pair with private valuations $s_t$ and $b_t$. The valuations are assumed to be arbitrary L-Lipschitz functions of the context, i.e., $s_t = f_s(x_t)$ and $b_t = f_b(x_t)$. The learner's goal is to minimize regret, which is the difference between the total possible gain from trade and the gain achieved by the learner's prices.

### Strengths
- This paper extends the contextual online bilateral trade framework from linear valuations to a general nonparametric setting where valuations are arbitrary Lipschitz functions of the context. The main results are achieved under the restrictive setting of one-bit feedback and strong budget balance, a setup where prior work often required relaxing one of these constraints.
- A major strength is the tightness of the results. The paper establishes an upper bound on regret of $\tilde{O}(T^{(d-1)/d})$ and provides a matching lower bound of $\Omega(T^{(d-1)/d})$ up to logarithmic terms.
-  The related work section is comprehensive and clearly situates the paper's contributions within the existing literature.
- The section “1.1 TECHNICAL CHALLENGES AND TECHNIQUES” very clearly describes the details about the problem, algorithm, results, and the proof techniques at a high level.
- The algorithmic components are presented in a clear, modular fashion (Algorithm 1 and its subroutines), and the accompanying descriptions in Sections 3.3 and 3.4 effectively explain the purpose of each routine.

### Weaknesses
- The assumption that valuations are deterministic functions of the context ($s_t = f_s(x_t)$, $b_t = f_b(x_t)$) is a significant limitation. Real-world valuations are typically subject to noise. The authors correctly note in the conclusion that adding noise is an open question, but the implications of this assumption on the main results could be discussed more prominently. The deterministic nature is precisely what allows certain proof techniques to work and is a key reason why the tight regret rate, which is better than in some noisy nonparametric bandit models, is achievable. A clearer discussion of this tradeoff in the main body would better contextualize the results.
- While the contributions are substantial, their novelty could be framed more sharply against existing techniques in nonparametric bandits. The paper mentions using a hierarchical tree structure similar to adaptive zooming (Slivkins, 2014) or chaining (Cesa-Bianchi et al., 2017). However, the paper could be improved by more explicitly articulating why those methods are insufficient on their own and how the proposed "tricks" (especially the REDUCE routine) are essential modifications to handle the specific structure of the bilateral trade problem (e.g., the reward discontinuity and one-bit feedback).
- The manuscript would benefit from a figure illustrating the hierarchical tree partitioning of the context space. This would provide a helpful visual aid for the core data structure of the algorithm.
- The terminology for interacting with the market is inconsistent. The text and algorithms variously use "post price", "play", and "pick". Standardizing to "post" or "propose" would improve consistency.

### Questions
- In Section 1.1, you mention that a first "trick" leads to a suboptimal regret of $\tilde{O}(T^{(d-2)/d})$, and a second trick improves this to the optimal rate of $\tilde{O}(T^{(d-1)/d})$. However, for any dimension $d \ge 2$, the exponent $(d-2)/d$ is smaller than $(d-1)/d$, which would imply that the "suboptimal" rate is asymptotically better. Could you clarify this point? Is there a typo, or is there a nuance in the definition of "suboptimal" in this context?
- What are the time and space complexities of Algorithm 1 per round? While this is a theory paper, the contributions would be strengthened by empirical validation, even on synthetic data, to demonstrate how the regret scales in practice and to verify the theoretical guarantees. Have any experiments been conducted?
- Your work focuses on maximizing the gain from trade (a social welfare objective) under strong budget balance. Could your techniques be extended to a revenue-maximization setting, where the learner posts different prices to the buyer ($p_b$) and seller ($p_s$) and aims to maximize the cumulative revenue $\sum_t (p_b-p_s)$?

### Soundness
3

### Presentation
3

### Contribution
3
