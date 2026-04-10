## Summary
This paper presents an empirical investigation and theoretical analysis of temporal communication scheduling in decentralized learning. It shows that concentrating limited communication budgets in later training stages, particularly through a single global model averaging step at the end, can dramatically improve global test performance under high data heterogeneity. Theoretically, it offers a novel convergence analysis suggesting that the discrepancy among local models can be partially reinterpreted as constructive rather than purely detrimental noise.

## Strengths
- **Compelling and well-validated empirical finding:** The core observation—that decentralized training with extremely sparse peer-to-peer gossip, followed by one final global averaging step, recovers performance close to more communication-intensive federated learning—is demonstrated robustly across multiple models (ResNet-18, CLIP), datasets (CIFAR-100, TinyImageNet), and heterogeneity levels (Figures 1, 2). This is a simple, surprising, and practically significant insight.
- **Novel perspective on model discrepancy:** The theoretical analysis (Section 5) introduces a fresh conceptual framework. It attempts to move beyond treating consensus error purely as harmful noise by exploring conditions under which it might contribute beneficially to the convergence of the averaged model, linking it to concepts like "progressive sharpening."
- **Clear experimental design for temporal allocation:** The methodology of dividing training into windows and activating AllReduce only in specific periods (Section 4.1) provides a direct and effective way to study the impact of communication timing. The contrast between "low communication" and "no communication" in Figure 2c cleanly isolates the enabling role of minimal synchronization.

## Weaknesses
### Major:
- **Theoretical explanation is misaligned with observed geometry and relies on strong, unverified assumptions.** The analysis hinges on **Assumption 4 (Progressive Sharpening)**, which posits a negative alignment between the loss gradient and the gradient of sharpness for the averaged model. However, the paper's own visualization (Figure 1c) depicts local models in a "ring-like high-loss region surrounding a central low-loss basin." The benefit of averaging appears to come from *interpolating into a flatter, lower-loss region*, not from a sharpening dynamic driven by the averaged model's trajectory. This core disconnect means the proposed theoretical mechanism does not convincingly explain the primary empirical phenomenon. Furthermore, other assumptions (fourth-order smoothness, gradient norm lower bounds) are stringent and not validated within the experiments.
- **Convergence claims are not empirically substantiated for the experimental setting.** **Theorem 1** and **Proposition 2** claim the merged model from decentralized SGD can match the rate of parallel SGD. This is a strong theoretical statement derived under the aforementioned assumptions. There is no empirical validation within the paper that the intricate sufficient conditions (e.g., Inequality (8), (11)) hold during the ResNet/CLIP training runs. The theory remains a potential explanation under specific conditions, not a demonstrated mechanism for the observed results.
- **Practical deployment challenges of the final merge are under-discussed.** The strategy requires a single, synchronous all-to-all averaging step at the end. While cost is analyzed as *O(2mP)*, the practical orchestration of this global operation in a truly decentralized, asynchronous, and potentially unreliable network (with churn, latency variation, firewall restrictions) is a significant systems challenge. The brief mention of approximating it via gossip (Appendix C.3.4) lacks a thorough evaluation of the performance-cost trade-off in such scenarios.

### Minor:
- **Empirical scope could be broader.** Experiments are confined to vision tasks with Dirichlet-based label distribution skew. Testing on other data modalities (e.g., language), different heterogeneity patterns (e.g., feature skew), or with larger models would help assess the generality of the finding.
- **Clarity of communication protocol description.** Section 4.1 could be more explicit in stating that the low-communication gossip mode (probabilistic communication with a random peer) runs continuously, and the high-communication AllReduce windows are *superimposed* on this baseline.

### Trivial:
- The paper is generally well-written and structured.

## Nice-to-Haves
- Extending the experimental suite to include more diverse tasks and heterogeneity patterns to better establish the finding's generalizability.
- A deeper analysis quantifying "mergeability" in weight space (e.g., measuring linear mode connectivity or loss barriers between local models) to move beyond performance-based definitions.
- Prototyping and evaluating the adaptive communication algorithm suggested by the theoretical intuition in Proposition 3, which would strengthen the link between theory and practice.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

**Strength (from Neutral Reviewer):** "Novel Theoretical Contribution... provides a mechanistic explanation for the empirical mergeability." *Removed because the review judges the theoretical explanation to be flawed and not substantiated, contradicting this strength.*

**Weakness (from Harsh Critic):** "The description of the communication schedule in Section 4.1 is slightly confusing... should be explicitly stated..." *Moved to Minor Weaknesses above as a clarity issue, not removed.*

**Weakness (from Spark/Human Finder):** "Demonstrate the claimed mergeability for modern, highly heterogeneous LLMs." *Removed as scope creep. The paper's empirical evidence on standard vision benchmarks is sufficient to support its claims; demanding LLF fine-tuning experiments is an expansion beyond its current scope.*

**Weakness (from Spark):** "Systematically compare against strong federated learning baselines with equal total communication cost." *Weakened to a Nice-to-Have. While a useful comparison, the paper already shows its method approaches FL performance (Figure 1), and a full cost-equated comparison, while ideal, is not a fundamental flaw in the presented evidence.*

**Weakness (from Human Finder):** "Convergence analysis provides implicit rather than explicit bounds... limits its interpretability." *Removed. Theorem 1's implicit bound is explicitly noted by the authors (Remark 3) as a deliberate step to connect to dynamics; it is a stylistic choice of the analysis, not a weakness.*

**Weakness (from Human Finder):** "Insufficient exploration of correlated computation-communication dynamics..." *Removed as scope creep. The paper studies temporal *scheduling* of communication, not joint stochastic modeling of computation and communication delays. This is a different research problem.*

## Suggestions
- **Reframe the theoretical contribution.** The current theory, while novel, does not successfully explain the main empirical result. The paper would be significantly strengthened by either: (1) providing direct empirical validation that the assumptions (especially progressive sharpening) hold in the studied setting and that the theoretical conditions are met, or (2) pivoting the theoretical discussion to a more plausible explanation aligned with the observed interpolation/flattening geometry (Figure 1c), perhaps connecting more directly to the literature on loss landscape geometry and model merging.
- **Expand the discussion of practical implementation.** Dedicate a subsection to honestly address the systems challenges of executing a final global merge in a decentralized environment. Discuss and evaluate the gossip-based approximation suggested in Appendix C.3.4 as a more feasible alternative, analyzing its convergence properties and overhead.

**Novelty:** High. The core empirical finding about late-stage merging is surprising and counter-intuitive, and the attempt to theoretically reframe model discrepancy is conceptually novel.
**Technical Soundness:** Moderate. The empirical methodology is sound and well-executed. The theoretical analysis is technically sophisticated but built on assumptions that appear misaligned with the empirical evidence, undermining its explanatory power.
**Empirical Support:** Strong. The central empirical claim is supported by comprehensive experiments across multiple variables (models, datasets, heterogeneity).
**Significance:** High. The finding has clear practical implications for designing communication-efficient decentralized learning algorithms and offers a new perspective for model merging research.
**Clarity:** Good. The paper is clearly written, with minor room for improvement in describing the experimental protocol.