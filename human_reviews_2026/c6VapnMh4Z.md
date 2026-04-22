# Causal-GNN SupplyNets Enabling Resilient Semiconductor Supply Chains with Causal World Models and Lyapunov-Safe Control

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
The inherent cyclicality of semiconductor supply chains and the associated severe volatility pose a significant challenge to the global electronics ecosystem. During periods of tight capacity, micro-level disruptions (e.g., tool failures, yield fluctuations) are rapidly amplified through the complex network structure, leading to protracted order delivery delays and system-wide disruptions. The core problem for achieving resilience lies in making decisions based on partial, incomplete information while providing high-probability guarantees that critical operational constraints (e.g., capacity, work-in-process inventory) are satisfied. Existing approaches often decouple forecasting and decision-making, lacking either a causal understanding of intervention effects or the ability to provide provable safety guarantees, resulting in suboptimal performance in turbulent environments. To overcome these challenges, we present **Causal-GNN SupplyNets**, a framework that unifies causal reasoning with *safe constrained optimization*. Our approach introduces three key innovations: (1) We learn a graph neural network-based "world model" that incorporates macro-level causal structural priors, enabling accurate prediction of the causal effects of sudden shocks and local interventions (e.g., adjusting dispatch policies) throughout the supply network; (2) We design a Lyapunov-based safe reinforcement learning controller that *provably* optimizes material dispatch and replenishment policies while satisfying safety constraints with high probability; (3) We introduce a privacy-preserving federated distillation mechanism, allowing different organizations to collaboratively improve their interventional knowledge without sharing raw sensitive data. Extensive experiments in simulated environments and on anonymized real-world manufacturing data demonstrate that our method significantly outperforms baseline models across various load and shock scenarios. It consistently improves on-time delivery rate (**up to 17 percentage points at peak load**), shortens cycle times, and accelerates post-shock recovery. Ablation studies further confirm that the causal constraints are crucial for accurate counterfactual prediction, and the Lyapunov safety guard is necessary for ensuring *near-zero* constraint violations. Our work provides a new pathway for achieving provable resilient control in highly uncertain and dynamic complex networks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an integrated framework that couples a causal world model with Lyapunov-based safe control. A structural causal model (SCM) constrains GNN time-series forecasting, and a Lyapunov-guided safe RL controller performs dispatching and replenishment decisions in re-entrant semiconductor supply chains. The authors claim the method improves OTIF, reduces CT tail risk, and shortens recovery time while satisfying hard constraints.

### Strengths
1. The work combines multiple perspectives in a unified framework—causal priors, heteroscedastic forecasting, safe control, and federated distillation—with clear engineering relevance.

2. It uses an SCM mask to constrain message passing. This design is novel, intuitive, and highly interpretable; coupled with IRM, it should improve robustness across environments.

3. The experimental metrics closely match engineering needs—for example, OTIF/CTP95/recovery time/ACE/violation rate—and the paper includes extensive ablations, which strengthens the empirical evidence.

### Weaknesses
1. The abstract and introduction repeatedly state that constraints are strictly guaranteed to be non-violated, whereas the main text and appendix provide only finite-horizon high-probability guarantees, further relying on a contractive Lyapunov function and smoothness assumptions. These claims are not of the same strength.

2. The safety condition depends on a learned V that must be contractive, but the paper does not explain how V is trained to ensure contractiveness. Under model bias or OOD shocks, when $\mathbb{E}[V(s_{t+1})]$ under the true dynamics deviates from the world-model estimate, how is non-increase of V still guaranteed?

3. The main text explicitly claims bounded regret relative to the optimal safe policy, but Appendix A.2 contains no regret theorem (only two informal statements on IRM/ safety). This is a core theoretical claim without a proof and should not be presented as a theorem.

### Questions
1. See Weaknesses above.

2. How is the alignment between the mask MMM and Attn implemented in practice?

3. I am uncertain about the necessity of RDAG. If M is fully specified by an external SCM-DAG and applied as a layer-wise mask, can the learned graph still contain cycles? Why is an additional acyclicity penalty (RDAG) imposed on the learned structure?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Causal-GNN SupplyNets, a framework for improving resilience in semiconductor supply chains by combining causal world modeling, Lyapunov-safe RL, and federated causal distillation. It models supply chain dependencies as a DAG learned from domain knowledge, using it to regularize a GNN that predicts and controls cascading disruptions. A Lyapunov-guided RL agent is introduced to enforce stability and constraint satisfaction in dynamic control tasks, while federated distillation enables collaborative learning across sites using interventional queries.

### Strengths
1. The paper addresses resilience in supply chain networks, a critical yet underrepresented problem in the machine learning community. Most ML research has focused on well-studied domains like vision, language, or simple graph tasks, while the complex, interdependent dynamics of supply chains, especially their cascading failure effects remain relatively unexplored. Tackling this class of problems has strong real-world relevance and societal impact.
2. Learning a Directed Acyclic Graph (DAG) informed by domain knowledge from supply-chain processes and using it to train a Graph Neural Network (GNN) is an insightful approach. The authors clearly separate the physical supply chain from its acyclic causal abstraction (via the SCM/DAG formulation), which is conceptually elegant and aligns with causal reasoning principles in dynamic networked systems.
3. Although the methodology is insufficiently detailed, the idea of using interventional (counterfactual) queries to train clients in a federated distillation setup is unique and promising. If the mathematical formulation and algorithmic steps are fully developed in future versions, this approach could meaningfully advance how distributed agents learn causal models without sharing data.

### Weaknesses
This is a very superficial paper that plugs in multiple domains of ML/Engineering, without a deep dive into innovation in any. The contributions are not significant at all, except for the problem setting (which I admire a lot). The presentation quality is poor. Much of the paper lacks formal mathematical expression or clear derivations. Many core ideas are described only narratively, assuming a high level of background knowledge and leaving essential preliminaries undefined. I have divided the main critique of the paper into the three key contribution areas:  
## I. Causal Model
1. The definition of the loss function is unclear. It is not explicitly stated what $L_{forecast}$ corresponds to. 
2. Regularizers $R_{DAG},$ and $R_{IRM}$ have no mathematical expression. Their formulation and role in training are missing, leaving the section incomplete. 
3. The “causal mask” $M$ is vaguely described. It is unclear if this is simply the GNN adjacency matrix or an additional learned mask.
4. The GNN architecture appears to use attention layers (thus resembling a GAT), but this is never clarified. The paper should explicitly specify the architecture (e.g., GCN vs. GAT).
5. Equation (2) is claimed to enable counterfactual inference, but there is no supporting mathematics connecting it to counterfactual reasoning.
6. The statement that the “predictive head is heteroscedastic” is ungrounded. The predictive head is not defined in Eq. (1), leaving the claim ambiguous.

## II. Safe RL Agent 
1. The use of the term “safe RL” is misleading. In this context, “safety” refers to satisfying production or service-level constraints, not to the formal notion of safety in reinforcement learning (i.e., avoiding unsafe exploration or catastrophic outcomes). The terminology risks confusing the ML audience and should be revised throughout.
2. The Lyapunov-guided optimization is presented with only a single inequality and no derivation or numbered equation. The paper lacks a clear buildup or preliminaries explaining how Lyapunov stability theory applies to the CMDP formulation.
3. The theoretical and algorithmic connection between the Lyapunov condition and the claimed safety guarantee is not substantiated.

## III. Federated Causal Distillation
1. The explanation is extremely superficial. It is unclear how counterfactual queries are generated and used for training. 
2. No mention of what information is exchanged between clients and server. This is really strange for a decentralized/federated learning manuscript.  
3. The paper doesn't provide any insights as to why the KL-divergence is the appropriate distillation objective. Are the authors using distributions of student (server) and teacher (clients) as opposed to model updates/gradients. If yes, they must be clearly mentioned with the associated maths. 

**A one-paragraph treatment of such a complex topic is inadequate and makes the section appear conceptually weak.**

## Other Comments 
1. Several metrics are poorly defined. For example, OTIF (On-Time-In-Full) is introduced as a “control performance” metric without an explanation of what control variables affect it.
2. “Constraint violation” is equated with “safety,” which is conceptually incorrect in reinforcement-learning terms. Meeting production targets does not equate to operating a “safe” policy.

### Questions
I will use this section to paraphrase the main weakness of the paper I identified and pose the related questions for the authors: 
1. Provide theoretical preliminaries and derivations for the Lyapunov-guided policy update.
2. Describe the federated learning process in detail—what is exchanged, how privacy is enforced, and how counterfactual queries are used.
3. Define all evaluation metrics (e.g., OTIF, constraint violation) and explain why they reflect control or safety performance.
4. Provide explicit definitions and mathematical forms for $L_{forecast}, $ $R_{DAG},$ and $R_{IRM}$. 
5. Clarify what the causal mask $M$ represents and how it differs from the standard GNN adjacency matrix.
6. Specify the exact GNN architecture used (GCN, GAT, or hybrid) and justify this choice.
7. Explain how Equation (2) supports counterfactual inference.
8. Define the predictive head and show how heteroscedasticity is modeled.
9. Clarify the notion of “safety” used in this context and justify the terminology.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses resilience in semiconductor supply chains by proposing Causal-GNN SupplyNets, which integrates three components: (1) a GNN-based causal world model constrained by a Structural Causal Model (SCM), (2) a Lyapunov-guided safe reinforcement learning controller, and (3) a federated causal distillation mechanism. The key innovation is using an SCM-derived mask to constrain GNN message-passing, forcing the model to respect causal relationships. Experiments demonstrate improvements in on-time delivery (up to 17pp), cycle time reduction, and faster shock recovery.

### Strengths
The paper tackles an important real-world problem with significant practical implications. The semiconductor supply chain domain is timely and challenging, with its re-entrant queueing dynamics and cascading failure modes providing excellent motivation for both causal modeling and safe control. The heavy traffic stress testing framework (d80-d95 scenarios) is particularly well-designed, systematically evaluating performance as systems approach capacity limits.

The technical approach shows genuine novelty in its integration. While individual components (causal GNNs, safe RL, federated learning) exist separately, their synthesis here is meaningful. The use of an SCM to generate a causal mask M that constrains GNN message-passing is creative, and the integration with Lyapunov-based safety certificates addresses real operational constraints. The federated causal distillation mechanism is innovative, sharing interventional knowledge rather than just predictive models.

The experimental evaluation is comprehensive and methodologically sound. The ablation studies (Table 2) effectively demonstrate that each component contributes meaningfully removing the SCM mask degrades ACE by 23%, removing the Lyapunov guard increases violations by 189%. The multi-scenario testing across different demand levels shows robustness, and statistical rigor with 7 seeds and significance testing strengthens the claims.

### Weaknesses
The most critical limitation is the assumption of a known causal structure. The paper states that "This SCM is represented by a Directed Acyclic Graph (DAG) D" but never adequately addresses how this DAG is obtained in practice. While RDAG penalties are mentioned for structure learning, the interplay between learning and exploiting causal structure remains unclear. For real deployment, obtaining ground-truth causal graphs is extremely difficult, and the method's robustness to SCM misspecification is inadequately characterized. The brief mention in Section 7 of "developing methods for online causal discovery" acknowledges but doesn't resolve this fundamental challenge.

The theoretical foundations lack rigor. Appendix A.2 provides only "informal statements" of theorems. Assumption 1 (bounded shocks, known intervention subsets) is quite strong but not validated empirically. Theorem 2's safety guarantee requires a "contractive" Lyapunov function and "sufficiently smooth" dynamics conditions that may not hold in practice but are not verified. The gap between theoretical claims and empirical results needs better reconciliation.

Methodological details are insufficient for reproducibility despite the reproducibility statement. The construction of the mask M from DAG D is described only conceptually in Equation 1, without algorithmic details. How exactly does one go from macro-level causal relationships to micro-level message-passing constraints on the physical supply network graph G? The relationship between the physical graph G and causal graph D needs clarification. Hyperparameter choices (λdag, λinv) appear to require extensive tuning but the sensitivity analysis is minimal.

The federated learning component feels underdeveloped. While conceptually introduced in Section 5, the federated causal distillation is barely evaluated empirically. The experimental results focus on single-site performance. How much does federated learning actually improve over local models? What is the communication overhead? The privacy claims rely on differential privacy but (ε, δ) values are not reported.

The evaluation has limitations in scope and depth. Most experiments use synthetic data where ground-truth SCMs are available by construction. The "anonymized operational logs from real-world semiconductor fabs" receive minimal treatment. Baseline comparisons are incomplete Table 2 shows ablations of the proposed method but doesn't compare against other safe RL algorithms (CPO, Lagrangian-SAC are mentioned in Section 6.1 but not evaluated). The computational cost analysis is entirely absent, yet scalability is crucial for industrial deployment.

The presentation suffers from trying to accomplish too much. Combining causal discovery, causal forecasting, safe RL, and federated learning in one paper makes it difficult to assess the contribution of each piece. The writing is generally clear but some sections are dense (Section 4 could be more pedagogical). Figure 2's innovation diagram could better illustrate the mask construction process.

### Questions
This paper addresses an important problem with a novel integrated approach and demonstrates promising empirical results. However, it suffers from critical weaknesses: the assumption of known causal structure is not adequately addressed, theoretical guarantees lack rigor, and the federated learning component is under-evaluated. The work makes contributions to both causal modeling in spatiotemporal GNNs and safe control of complex networks, but the breadth of scope compromises depth of treatment. With substantial revisions addressing the causal structure specification, stronger theoretical grounding, and more complete empirical evaluation, including federated learning and computational costs, this could become a strong contribution.

### Soundness
2

### Presentation
3

### Contribution
2
