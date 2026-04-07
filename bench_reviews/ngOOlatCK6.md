## Summary
This paper introduces the conditional causal bandit problem, where arms are single-node conditional interventions in a known causal graph aimed at maximizing a target variable Y. Its core contribution is a graphical characterization of the minimal set of nodes (mGISS) guaranteed to contain the optimal intervention node, and a linear-time algorithm (C4) to compute this set. Empirical results demonstrate significant search space reduction and accelerated convergence when integrated into a bandit algorithm.

## Strengths
- **Novel and well-motivated problem formulation.** The paper clearly argues for the importance of conditional over hard interventions in real-world decision-making (e.g., medical treatment) and rigorously defines the novel setting of single-node conditional causal bandits.
- **Strong theoretical foundation.** The graphical characterization via Λ-structures and the equivalence between conditional and deterministic atomic intervention superiority (Proposition 4) are elegant and provide the basis for the minimal search space result (Theorem 13). Proofs are provided and appear rigorous.
- **Efficient and correct algorithm.** The proposed C4 algorithm runs in O(|V|+|E|) time, is simple to implement using the connector concept, and is proven correct (Theorem 16), making the theoretical contribution practically usable.

## Weaknesses
- **Strong assumption of no latent confounders.** The entire analysis assumes causal sufficiency. This is a significant limitation for real-world applicability, as latent confounding is common. While acknowledged as future work, its absence curtails the current contribution's direct utility.
- **Lack of theoretical bounds on pruning effectiveness.** The paper provides no theoretical guarantees on the size of mGISS relative to the set of all ancestors of Y. Without such bounds, it is difficult to predict the utility of the method for an arbitrary graph class.
- **Empirical evaluation could be broader and more comparative.** The synthetic graph analysis uses only the Erdős-Rényi model; evaluation on other generative models (e.g., scale-free) would strengthen generalizability claims. The bandit experiments show improvement over a brute-force search but do not compare against natural baselines (e.g., intervening only on the parents of Y), making the added value of the full characterization less clear.
- **Experimental details in the main text are sparse.** While code is provided, key details for reproducibility—such as the exact specification of the synthetic structural equations and reward functions for the bnlearn datasets, and the hyperparameters for the CondIntUCB algorithm—are insufficiently detailed in the paper itself.

## Nice-to-Haves
- Integration of the mGISS pruning step with more advanced causal or contextual bandit algorithms to demonstrate its utility beyond a simple UCB adaptation.
- Runtime measurements on very large graphs to empirically confirm the linear-time scalability in practice.
- A sensitivity analysis investigating how the choice of the conditioning sets **Z_X** (within the assumed constraints) impacts the bandit performance.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness:** The abstract does not explicitly mention the assumption of no latent confounders. *This limitation is clearly stated in the introduction.*
- **Weakness:** The introduction provides insufficient intuition for why single-node interventions are "more challenging." *The paper explains that with multi-node interventions one can intervene on all parents of Y, which is not possible in the single-node case, justifying the complexity.*
- **Weakness:** The paper does not discuss the complexity of the policy space after node pruning. *The paper's contribution is precisely to prune the node space; the subsequent policy selection for a given node is a separate problem explicitly left to standard bandit algorithms.*
- **Nitpick:** Concerns about notation clarity (e.g., E_n f̄_Y^{do(X=g(Z_X))}(n)). *The notation is standard in the causality literature and is further clarified in the appendix.*

## Novel Insights
The equivalence established between conditional intervention superiority and deterministic atomic intervention superiority (Proposition 4) is a key insight that simplifies the problem and enables the subsequent graphical analysis. Furthermore, the characterization of the minimal search space via Λ-structures (Theorem 12) provides an intuitive and computationally tractable graphical criterion that directly leads to the efficient C4 algorithm.

## Suggestions
- Provide a theoretical bound or a worst-case family of graphs illustrating the potential size of mGISS relative to An(Y)\{Y\}.
- Enhance the bandit experiments by including a comparison to a baseline that only considers intervening on the parents of Y, to better isolate the value of the full mGISS characterization.
- Broaden the synthetic graph analysis to include other common graph models (e.g., preferential attachment networks) to strengthen claims about performance on realistic structures.