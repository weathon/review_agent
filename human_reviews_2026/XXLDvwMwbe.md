# On the trade-off between expressivity and privacy in graph representation learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
We investigate the trade-off between expressive power and privacy guarantees in graph representation learning. Privacy-preserving machine learning faces growing regulatory demands that pose a fundamental challenge: safeguarding sensitive data while maintaining expressive power. To address this challenge, we leverage homomorphism density vectors to obtain graph embeddings that are private and expressive.
Homomorphism densities are provably highly discriminative and offer a powerful tool for distinguishing non-isomorphic graphs. By adding noise calibrated to each density’s sensitivity, we ensure that the resulting embeddings satisfy formal differential privacy guarantees. Our theoretical construction preserves expressivity in expectation, as each private embedding remains unbiased with respect to the true homomorphism densities. We demonstrate the usefulness of our embeddings through experiments on molecular and social network datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the trade-off between privacy and expressivity in graph representation learning. It proposes using homomorphism density vectors as graph embeddings to achieve both expressive power and formal privacy guarantees. The paper derives the trade-off in two parts: firstly, if homomorphism density vectors are used as graph embeddings, then with any noise added to them, in expectation, they will achieve expressiveness. Secondly, it proposed to use a $\beta$-smoothing sensitivity bound and leverage Gaussian noise to achieve tCDP. Extensive theoretical and experimental results are conducted to highlight and illustrate the advantage of using homomorphism density vectors as embeddings.

### Strengths
- Provide a strong theoretical guarantee for the expressivity and privacy of graph representation learning.
- The experimental results highlight the validity and soundness of the theoretical claims.
- The proposed method has been proven to be useful through experimental results and is robust against attacks.

### Weaknesses
- The privacy guarantee is provided for local sensitivity. Although it is demonstrated that very similar graphs can still be distinguished by the noisy homomorphism density vectors, the privacy guarantee does not meet the DP standard. 
- The expressivity guarantee is analyzed in expectation, which is orthogonal to the privacy analysis. Since the DP noise is only added once, the expressivity will be biased to that sampled noise, leading to the degradation of the guarantee.
- Some of the critical theoretical annotations are provided in the Appendix, making it hard to follow the flow of the paper.

### Questions
- In line 206, why is the sensitivity of Theorem 3.10 a local sensitivity if the neighboring database is considered across the input domain?
- By the definition of F-expectation-expressivity and Expectation-completeness, what are the $X$ for your analysis and proposed method?
- By expectation w.r.t $X$, the F-expectation-expressivity and Expectation-completeness have to consider all of the possible inputs of the input space? Then, can you elaborate on how to use this expectation in your expressivity guarantee?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the theoretical trade-off between expressivity and privacy in graph representation learning. The authors propose constructing graph embeddings using noisy homomorphism densities, where homomorphism counts are perturbed with Gaussian noise calibrated via smooth sensitivity to ensure tCDP. Experiments are conducted on several OGB molecular datasets and synthetic graphs, showing that the proposed method maintains competitive predictive performance and effectively resists privacy attacks under reasonable privacy budgets.

### Strengths
1. Graph privacy is an important and practically relevant research direction.
2. The paper is relatively well-organized and logically coherent.

### Weaknesses
1. The paper emphasizes the use of homomorphism density vectors as a key tool. However, while it claims to fill the gap in understanding the interplay between expressivity and privacy in graph representation learning, it does not clearly define what this gap uniquely refers to. Prior studies (Sajadmanesh et al. 2021, 2023, 2024, as well as others) have already explored expressivity under differential privacy and aimed to design utility-preserving private graph learning frameworks.
2. The paper focuses on edge-level privacy and graph-level tasks without explaining why this specific setting was chosen. In practice, graph privacy leakage often involves node attributes, and for edge-level privacy, link prediction tasks are crucial for comprehensive evaluation. The current setup limits the practical relevance of the work.
3. The paper cites Lovász’s theorem (1967), which states that homomorphism counts can distinguish all non-isomorphic graphs. However, it later uses homomorphism densities, and in Remark B.1 admits that these cannot distinguish a graph from its blow-up. Although the authors propose adding “node counts” as a correction, they provide no theoretical proof that this modification preserves differential privacy guarantees or avoids leaking information about graph size. Moreover, the proposed fix only applies when graphs have different node counts—what about when they are the same?
4. Experimental details are incomplete. For instance, the paper sets $δ=10^{-6}$, $β=ρ'/5$, and $d=50$, but does not justify these choices or explain their sensitivity.
5. Experiments are limited to OGB molecular graphs and synthetic graphs, lacking evaluation on other types of graph data such as social networks (large, sparse) or medical graphs (sensitive node attributes). Since molecular graphs are relatively regular, the effectiveness of the proposed method on structurally complex graphs remains unverified, reducing generalizability.

### Questions
see weaknesses.

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
This paper investigates the trade-off between expressivity and privacy in graph representation learning. The authors propose using noisy homomorphism density vectors to achieve both distinguishability and differential privacy guarantees. The theoretical part derives upper bounds on sensitivity and smooth sensitivity, while the experimental section validates the privacy–utility trade-off on the OGBG and SBM datasets.

### Strengths
1. The paper demonstrates rigorous logical consistency from definitions and lemmas to theorem derivations, with solid mathematical details.
2. The proposed framework can be mapped to different GNN architectures (e.g., message-passing and subgraph-based GNNs) and allows for quantitative analysis of the trade-off.
3. The experiments show clear performance differences across pattern classes, validating the theoretical trends.

### Weaknesses
1. It only considers edge-level DP.
2. Experiments are conducted only on small-scale, toy-level datasets and do not demonstrate the feasibility of the approach on real-world large graphs or complex tasks.
3. The paper lacks comparison with established DP-GNN methods such as edge-level GAP, using only a randomly perturbed GNN as the baseline.
4. Computing homomorphism densities is computationally expensive. Please provide a detailed analysis of computational overhead.
5. The paper is clearly written, but it appears that the authors used last year’s ICLR template.

### Questions
See weakness above.


Since only edge-level privacy is considered, can this analysis be extended to node-level or subgraph-level differential privacy in the future?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates the trade-off between expressivity (ability to distinguish non-isomorphic graphs) and privacy (edge-level differential privacy) in graph representation learning. It proposes using homomorphism density vectors as embeddings, adding Gaussian noise calibrated via smooth sensitivity to achieve truncated concentrated DP (tCDP) guarantees. The authors argue that expectation-expressivity is preserved because the noise is mean-zero. Experiments on OGB molecular datasets and a synthetic SBM illustrate privacy–utility trade-offs and compare against a randomized-response baseline.

### Strengths
- The paper addresses an underexplored intersection of privacy and expressivity in graph learning.
- Formalizes expectation-expressivity and privacy guarantees using established DP frameworks.
- Demonstrates basic privacy–utility trade-offs and sanity-checks the insufficiency of global sensitivity.

### Weaknesses
- Experiments mostly use tree patterns; higher treewidth and complex pattern classes are barely explored. The empirical section lacks depth: pattern classes beyond treewidth 1 are barely tested, and runtime implications of homomorphism counting (#P-hard) are not addressed.
- No comparison to state-of-the-art DP-GNNs (e.g., ProGAP) or strong inference attacks like LinkTeller. 
- Non-private use of degree bounds ($\Delta max$) for sensitivity calibration undermines DP guarantees. Furthermore, Δmax is estimated from data without privatization, which violates DP assumptions

### Questions
Given the strengths, I have the following concerns:

1. The authors claims that "Our embeddings match, in expectation, the expressive power of a broad range of graph neural networks (GNNs), such as
message-passing and subgraph GNNs, while providing formal privacy guarantees." but there is no comparison with GNN-based privacy methods like ProGAP or [1]. These are widely cited and provide edge-level DP guarantees. Can the authors include these baselines?

2. The paper evaluates privacy via nearest‑neighbor graph re‑identification on embeddings, ignoring stronger edge inference attacks. Why restrict to nearest-neighbor re-identification? Stronger attacks (e.g., LinkTeller) has shown non‑trivial leakage even against DP defenses for modest $\epsilon$ or modern membership/link inference analyses for GNNs; the empirical privacy validation thus seems under‑powered. 

3. The sensitivity bound with degree dependence (Theorem 4.5) assumes a known max degree Δmax (e.g., fixed per dataset) or estimated (Chernoff) on synthetic data. Using estimated Δmax can itself leak information unless privatized or externally bounded, and the paper sets Δmax from data or heuristics in experiments, which is not privacy‑preserving if used to calibrate noise. This critical point is not addressed. Can the authors clarify this?

4. This is subtle but very important, does the method allow private release of embedding?

5. Most OGB experiments use tree patterns (tw=1), which undermines the empirical claim that “more expressive pattern classes” help under fixed privacy budgets. The brief MOLHIV experiment with small treewidth still tops out at tw=3; no cycles/graphlets on large graphs, no runtime/memory profiling of homomorphism computations, despite #P‑hardness beyond bounded treewidth. 

6. Blow‑up invariance of densities is acknowledged in the appendix, but because densities cannot distinguish graph blow‑ups, the completeness claims require appending |V(G)| or switching to counts; this subtle but important caveat should be explicit in the main text where expressivity claims are made.



**Missing references:**

*On DP graph learning:*

[1]Olatunji et al. Releasing graph neural networks with differential privacy guarantees

[2]Xiang et al. Preserving Node-level Privacy in Graph Neural Networks

*On graph reconstruction attack:*

[3]Olatunji et al. Private graph extraction via feature explanations

[4]Zhou et al. On Strengthening and Defending Graph Reconstruction Attack with Markov Chain Approximation

### Soundness
3

### Presentation
3

### Contribution
3
