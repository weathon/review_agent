# Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Decentralized Learning (DL) enables users to collaboratively train models without sharing raw data by iteratively averaging local updates with neighbors in a network graph. This setting is increasingly popular for its scalability and its ability to keep data local under user control. Strong privacy guarantees in DL are typically achieved through Differential Privacy (DP), with results showing that DL can even amplify privacy by disseminating noise across peer-to-peer communications.
Yet in practice, the observed privacy-utility trade-off often appears worse than in centralized training, which may be due to limitations in current DP accounting methods for DL. In this paper, we show that recent advances in centralized DP accounting based on Matrix Factorization (MF) for analyzing temporal noise correlations can also be leveraged in DL. By generalizing existing MF results, we show how to cast both standard DL algorithms and common trust models into a unified formulation. This yields tighter privacy accounting for existing DP-DL algorithms and provides a principled way to develop new ones. To demonstrate the approach, we introduce MAFALDA-SGD, a gossip-based DL algorithm with user-level correlated noise that outperforms existing methods on synthetic and real-world graphs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies Differentially Private Decentralized Learning and proposes a matrix-factorization view that unifies attacker observations and common trust models (LDP, PNDP, SecLDP). It generalizes prior MF privacy results to a broader class of workloads while preserving adaptivity, enabling tighter privacy accounting for existing DP-DL methods. The framework also disentangles the matrix governing privacy from the one driving optimization, clarifying how to design useful noise correlations in decentralized settings. Building on this, the authors introduce MAFALDA-SGD, which optimizes node-local noise correlations under LDP, and they report tighter PNDP accounting and improved utility on synthetic and real graphs and a regression task.

### Strengths
- The work offers a unifying template to express the attacker across multiple decentralized trust models, which generalizes beyond prior algorithm-specific analyses.

- The theoretical contribution extends matrix-factorization privacy guarantees to column-echelon workloads, preserves adaptivity, and formalizes sensitivity, providing principled guarantees that support the framework.

- The empirical results indicate tighter PNDP accounting and improved utility for MAFALDA-SGD under LDP on the reported setups, suggesting practical benefits.

### Weaknesses
- The paper’s accessibility is limited for readers without prior exposure to matrix-factorization in DP; the notation is dense, the definition and role of “column-echelon” are introduced with minimal intuition in the main text, and the transition from the general theorem to MAFALDA is abrupt. A short primer with a toy $T=3$ derivation (showing how $A = B C$ and $C^\dagger$ shape correlations), a step-by-step “apply-the-framework” recipe (choose trust model and participation, form the attacker view and objective, impose constraints, solve for correlations), a brief glossary of symbols, and one worked example mapping a familiar DP-D-SGD variant into the proposed matrices would substantially improve readability.

- The empirical scope is narrow, focusing on one regression task and a few graphs; adding a classification task and ablations over participation schemes and correlation patterns would better demonstrate robustness.

- The presentation would benefit from precise definitions and cleanup, including defining the weighted/semi-norm $|\cdot|_{B^\dagger B}$ when first introduced and making the Cholesky orientation consistent throughout (for example, state clearly whether $H = L L^\top$ or $H = L^\top L$ and keep that convention)

- The set of LDP baselines could be expanded or justified; in particular, including DECOR as a comparison (or explaining why it is out of scope) would help situate MAFALDA-SGD among methods that also exploit correlations under LDP.

- Typo: “Independent” in Figure 1

### Questions
- Could the authors clarify whether attackers who can read some local models $\theta$ (for example, compromised nodes) can be incorporated by augmenting $O_A = A G + B Z$, and whether the column-echelon and adaptivity arguments still apply in that setting?

- Can you bound the limits of generality by stating which settings fall outside the framework, for example, non-linear message transforms, asynchronous or stale models, quantization or compression, or additional side information that might violate linearity or the column-echelon assumption?

- Do you plan to add further experiments, such as a classification task, ablations over participation schemes $(k,b)$ and training epochs, and sensitivity to graph topology, to substantiate the generality and robustness of the reported improvements?

- Do the authors plan to include an LDP baseline, such as DECOR (or provide a rationale if not), to clarify the relative benefits of optimized local correlations under LDP?

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
The paper proposes a unified privacy framework for decentralized learning based on the Matrix Factorization (MF) mechanism, bridging recent advances in centralized DP accounting to decentralized, peer-to-peer training. This paper presents a new algorithm, MAFALDA-SGD, which achieves tighter privacy guarantees.

### Strengths
1. The background information and technical introduction are very detailed and the writing is well-organized.
2. The issue of distributed privacy protection is of great practical significance, and the utility-privacy trade-off faced by existing methods is more severe compared to that in the centralized approach.
3. Sufficient theoretical evidence shows that each step has detailed proofs.

### Weaknesses
1. The gap between theory and method implementation: The theoretical framework in Sections 4–5 establishes a very general condition for the MF mechanism. This suggests wide applicability to diverse decentralized algorithms and trust models. However, the proposed algorithm MAFALDA-SGD in Section 6 only instantiates this framework under a restricted setting: Local Differential Privacy (LDP) combined with node-local temporal correlation.
2. Additional computational overhead: MAFALDA-SGD algorithm requires an offline optimization of the noise correlation matrix. This process involves computing the Gram matrix and its Cholesky factorization, which may introduce non-negligible computational overhead.
3.	Limited empirical diversity: Experiments focus on small to mid-scale graphs (≤ 300 nodes). Realistic, large-scale federated networks (e.g., > 10⁴ users) would better demonstrate practicality.
4.	Minor clarity issues: Heavy notation could be simplified, and intuitive explanations (especially for Eq. 9 and Lemma 10) would help non-specialists.
5.	Evaluation scope: Only MLP on Housing dataset — a deeper model (e.g., CNN or Transformer under DL setting) would show robustness of MAFALDA-SGD.

### Questions
1. Why is a direct multiplicative objective preferable in Eq. 9? Could the authors provide empirical evidence or theoretical justification that this formulation does not suffer from scale imbalance in larger graphs or longer training horizons?
2. Why does the method not explore cross-node correlation or richer participation structures, which the theory explicitly allows?
3. Does the method remain computationally practical when gradients are extremely high-dimensional, as in Transformers?
4. Can your generalized privacy theorem (Theorem 7) apply to asynchronous or time-varying gossip matrices W_t?
5. Did you explore non-Gaussian noise distributions (e.g., Laplace or sub-Gaussian) within the MF framework?
6. Could MAFALDA-SGD be extended to heterogeneous privacy budgets across nodes?
7. Is the matrix-factorization optimization offline-only, or could it adapt online to changing participation patterns?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper builds a unified privacy-accounting framework for decentralized learning (DL) by recasting DL updates as a Matrix Factorization (MF) mechanism. The paper extends centralized MF-style DP accounting to decentralized settings and multiple trust models. This achives tighter guarantees (especially under PNDP) and motivates a new algorithm, Mafalda-SGD, with optimized, node-wise correlated noise that outperforms prior DP-DL baselines on synthetic and real graphs.

### Strengths
1. The paper derives neat theory with clean abstraction and provides a unified lens and extends to adaptive and rectangular cases.
2. The paper presents substantially tighter bounds on PNDP.
3. The paper presents a concrete algorithm Mafalda-SGD that is adoptable and easy to run after an offline correlation evaluation step.
4. Empirically results show strong performance across a lot of graphs.

### Weaknesses
1. Seemingly strong assumptions (e.g., linear DL (definition 3), column-echelon, and the knowledge of gossip matrix for MAFALDA-SGD)
2. Limited experimental scale
3. Computational overhead is not discussed

Please see the questions section below for more details.

### Questions
1. The theory assumes linear DL where "all observable quantities can be expressed as a linear combination of the concatenated gradients G and the concatenated noise Z." This is often not true in practice. For example, in practice, quantization or compression is often used when exchanging information, which would render them non-linear. Also, if cryptographic techniques are further used to encrypt the weights or to perform secure aggregation, the weights/gradients will usually be converted into a discrete space.
2. In practice, how do we verify/enforce the column-echelon condition on real DL workloads?
3. A client needs to know the gossip matrix and the participation pattern to optimize for correlation. Is this a standard assumption? I guess a client may just know its local patterns. Also the topology may be time-varying.
3. What’s the computational/communication overhead of computing and refreshing the optimal correlation on large graphs and long horizons? How sensitive is the performance to approximate solutions if optimization is hard.
4. While experiments are conducted on a wide range of graphs (both synthetic and real-world), they are limited to small tabular datasets with small models. How would the proposed method for vision or NLP tasks?
5. The paper seems to use GDP and Renyi terms mostly, I recommend the authors to also include (epsilon,delta)-DP guarantees that will be of the interest to many practitioners.

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
This is an interesting paper that addresses differentially private (DP) decentralized learning (DL) by generalizing recent advances in centralized DP. In particular, it achieves privacy be means of correlated noise through the matrix factorization mechanism which is shown to apply also in decentralized settings. The proposed framework is applicable for different trust models, e.g., Local DP, Pairwise Network DP, and Secret-based LDP, providing tighter privacy guarantees and covering all known DP-DL algorithms under a single formulation.

### Strengths
- Extends the centralized matrix factorization mechanism to decentralized learning, unifying several existing trust models (LDP, PNDP, SecLDP) and encompassing all known DP-DL algorithms within one framework.

- Generalizes DP analysis to adaptive and non-square workload matrices through Theorem 7, enabling rigorous privacy guarantees for realistic decentralized and potentially other distributed systems.

- Introduces a novel algorithm (MAFALDA-SGD) that optimizes local noise correlation, achieving tighter privacy accounting and better privacy–utility trade-offs than previous methods.

- Empirically validated on multiple graph topologies and datasets, showing consistent performance gains and more accurate privacy bounds compared with prior accounting techniques.

### Weaknesses
- The experimental section only include small graphs (the largest including only 271 nodes). Moreover, the models considered in section 7.2 is very small, an MLP with a single hidden layer. 

- There are no ablations on the effect of graph topology, participation patterns, or colluding attackers. 

- There is no complexity analysis of the new algorithm. 

- It is not clear how restrictive definition 3 is.

### Questions
1) How restrictive is Definition 3? For example, it seems to roll out adaptive optimization like ADAM, quantization techniques, personalization (often achieved via regularization) etc. 

2) In section 6, when defining MAFALDA-SGD, the authors assume that all nodes follow the same local participation pattern. This assumption seem to rule out dynamic topologies that are, in fact, prevalent in DL settings. For example within potential space applications or where nodes may exhibit dropout.
a) Is this assumption necessary for MAFALDA?
b) How does this assumption limit the applicability of MAFALDA?
c) Is the framework applicable to dynamic graphs?

3) Can the authors provide an analysis of the complexity of the approach? Is the framework applicable in realistic scenarios?

4) Algorithm 1 and 2 requires that each node has access to $W$, $\Delta_g$, and $\sigma$.
a) Why is it reasonable for each node to have access to the entire topology?
b) How would $\Delta_g$ be decided in practice? If all nodes must use the same value, it requires coordination that will add to the complexity. How does data heterogeneity among nodes play a role?

5) It would be interesting to see more holistic experiments including i) more realistic settings with larger models, ii) the effect of graph topology, and larger graphs.

### Soundness
3

### Presentation
2

### Contribution
2
