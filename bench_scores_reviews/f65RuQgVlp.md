---

## Summary

This paper formalizes **Online Federated Continual Learning (Online-FCL)**, a setting where clients learn from streaming mini-batches that can each be processed only once, in contrast to the offline multi-pass assumption underlying existing generative FCL methods. The authors propose an uncertainty-aware memory management scheme that uses **Bregman Information (BI)**—derived from a bias-variance decomposition of the cross-entropy loss (Gruber & Buettner, 2023)—to select representative (low-epistemic-uncertainty) samples for local replay buffers. The method is evaluated on CIFAR-10/100, TinyImageNet, two biomedical image datasets with class imbalance, and three text classification benchmarks, consistently outperforming both FL baselines and generative FCL methods that are shown to fail in the online regime.

---

## Strengths

- **Genuinely novel problem formulation:** Online-FCL is a meaningful and underexplored gap between existing offline FCL and the established online-CL literature. The formal specification of streaming mini-batch processing within a federated continual learning setting, with class-incremental evaluation, is a concrete and useful contribution that the community was missing.
- **Principled use of BI for memory selection:** The distinction between aleatoric uncertainty (captured by entropy / confidence scores) and epistemic uncertainty (captured by BI) is clearly motivated. Figure 2 effectively illustrates why a data point near the decision boundary may have low BI (high data density, low DGP uncertainty) while having low confidence — a non-trivial and practically important insight that justifies choosing BI over standard uncertainty metrics for identifying *representative* replay samples.
- **Rigorous demonstration that generative FCL fails online:** The paper not only argues theoretically that generative methods (FedCIL, MFCL) need multi-pass offline access, but shows empirically that they collapse to near-FedAvg performance in the online setting (Table 5, Figure 3). This is an informative negative result that concretely motivates the memory-based approach.
- **Multi-domain evaluation including realistic biomedical data with imbalance:** Moving beyond CIFAR and EMNIST to include CRC-Tissue and KC-Cell with realistic class imbalance and different image statistics is a specific design choice that most FCL papers omit and that strengthens the real-world relevance of the claims.

---

## Weaknesses

### Fatal
None.

### Major

- **Marginal and statistically fragile advantage of BI over competing uncertainty metrics.** The core claimed advantage — that BI is *superior* to entropy, least confidence, margin sampling, and ratio of confidence for memory selection — is inconsistently supported. On CIFAR-100 (Table 2), BI-bottom (14.04 ± 0.62) is barely better than MS-bottom (13.77 ± 1.31) and LC-bottom (13.43 ± 1.00), with overlapping standard deviations. On KC-Cell (Table 3), BI-bottom at M=120 achieves 20.91 ± 1.32 accuracy, while LC-bottom achieves 19.66 ± 0.83 and MS-bottom achieves 20.05 ± 1.79 — differences well within noise. Only 3 random seeds are used throughout, and the high variance (e.g., F values in Table 3 with std > 9) renders many pairwise differences statistically inconclusive. The paper does not perform any significance testing. This is important because the headline claim is specifically that BI outperforms other uncertainty metrics; if this cannot be robustly demonstrated, the core methodological novelty is substantially weakened.

- **Text experiments use frozen sentence embeddings, undermining the multi-modality claim.** Section 4.1 uses a frozen pre-trained sentence encoder (e5-small-v2) with only an MLP linear head trained on top. Because the backbone representations do not shift during training, the continual learning problem is fundamentally different (no representational catastrophic forgetting, only linear decision boundary drift). The BI scores computed via Gaussian noise on frozen embeddings also have a very different interpretation than BI computed on dynamic visual representations. The claim that the method "works with different data modalities" is partially undermined by this shortcut; the results are informative about replay in a frozen-embedding setting, but not about multi-modal continual learning in general.

### Minor

- **Gaussian noise perturbation for text TTA is ad hoc and unvalidated.** Adding i.i.d. Gaussian noise $\mathcal{N}(0, 0.1)$ to fixed sentence embedding dimensions is used as TTA to estimate BI for text. There is no analysis of whether this perturbation correlates with actual model epistemic uncertainty, nor is sensitivity to the noise standard deviation ablated. Different noise magnitudes could systematically change which samples are selected, and the paper provides no empirical or theoretical justification for the specific value of 0.1.

- **No joint-training upper bound.** The paper provides FedAvg as a lower bound but does not include an oracle that trains on pooled data from all tasks simultaneously, which is standard practice in continual learning to contextualize how much performance is sacrificed by all sequential methods. Without this, the 35.83% accuracy on CIFAR-10 and persistent high forgetting on KC-Cell are hard to interpret — it is unclear whether the gap is due to the online constraint, the federated setting, or the class-incremental evaluation itself.

- **Early-training instability of BI estimates during memory population.** The model is poorly calibrated in the early batches of each task, which is precisely when memory is first populated. The burn-in period delays communication but does not prevent early (potentially unreliable) memory entries from persisting, since memory is updated throughout training. There is no analysis of how much the initial memory population quality degrades with a smaller burn-in, nor whether a delayed memory-write strategy would help.

- **Task boundary assumption not discussed.** The formulation assumes known task boundaries, which is a standard simplification but is at odds with the "realistic" framing of the paper. Online streams in healthcare or edge settings typically do not come with clean task boundary signals. This limitation is not acknowledged in the paper.

- **Scalability to non-synchronized task boundaries and heterogeneous client settings not addressed.** All experiments assume all 5 clients follow synchronized task schedules. FCL deployments (e.g., multiple hospitals) typically involve asynchronous task arrivals and heterogeneous class distributions across clients (non-IID beyond the current setup). Given the paper's positioning as a realistic FCL framework, this assumption deserves at least a discussion.

### Tiny

- The burn-in period (30) and jump parameter (q=5) are set uniformly across all datasets. While Appendix A.7 contains ablations, the main text provides no guidance for practitioners on how to set these for new datasets or data stream velocities.
- The claim of "competitive communication efficiency" in the abstract is not substantiated in the main results section; the comparison (rounds and wall-clock time) is relegated to the discussion and appendices. Either the claim should be moved to a limitation or the evidence promoted to the main experimental tables.

---

## Nice-to-Haves

- **DER++ or SCR adapted to the federated online setting** as a stronger memory-based baseline would help isolate whether improvements stem from the BI selection mechanism itself versus using uncertainty-guided replay in general, as opposed to comparing primarily against standard ER.
- **Non-IID client heterogeneity experiments** using Dirichlet label splits across clients (varying $\alpha$) would stress-test whether the class-conditioned averaging and BI selection remain effective under more realistic data heterogeneity.
- **Task-wise forgetting curves** (accuracy on previous tasks as each new task is introduced) would reveal whether forgetting is gradual or occurs catastrophically at specific task transitions — information that is currently masked by the aggregate forgetting metric.
- **Visualization of BI-selected vs. entropy-selected samples** side-by-side would make the core claim about representative vs. boundary-hugging samples visually compelling and directly validate the hypothesis behind the approach.
- **Quantitative communication cost table** (total bytes transferred, total communication rounds) in the main paper to back the abstract's claim of competitive communication efficiency.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"Comparison with MFCL/FedCIL is misleading" (Harsh Critic):** The paper explicitly and clearly labels MFCL and FedCIL as offline methods and uses the comparison to demonstrate they fail in the online setting. The asymmetry intentionally favors the baselines (they are operating outside their designed regime), which makes the comparison a stronger argument for the paper's thesis, not a deceptive one. This is not a legitimate criticism per evaluation norms for empirical systems papers.
- **"Privacy of class-conditioned averaging" (Harsh Critic, Positive Reviewer):** The paper directly addresses this: "if sharing the class information is not possible, we can rely on standard averaging strategies (e.g., FedAvg or FedProx) without hampering the performance (see Table 9)." The concern is adequately handled.
- **"Non-overlapping classes assumption is too restrictive" (Harsh Critic):** The paper explicitly follows the convention of Qi et al. (2023) which is the standard formulation for federated class-IL. Criticizing an assumption that matches the established literature convention is scope creep.
- **"Contributions list conflation" (Harsh Critic):** Pure stylistic complaint with no technical content.
- **Missing specific centralized uncertainty-based replay baselines (Harsh Critic):** Per review policy, we do not flag missing related works without external sources to confirm their relevance.
- **"Absolute accuracy numbers are too low" (Harsh Critic):** These numbers reflect the genuine difficulty of online class-incremental federated learning (5 clients, streaming mini-batches of 10, one-pass only). The low absolute values characterize the setting, not a failure of the method. All methods, including generative FCL, achieve similarly low numbers, confirming this is a hard problem.

---

## Novel Insights

The most genuinely interesting insight in this work — partly surfaced by the spark finder and partly implicit in the paper — is the **distinction between aleatoric and epistemic uncertainty for memory selection in online class-incremental learning**. The argument that samples near the decision boundary have *low* epistemic uncertainty (BI) but *high* aleatoric uncertainty (entropy), and that storing low-BI (representative, well-identified) samples is more useful for replay than storing high-entropy (ambiguous, boundary) samples, runs counter to the intuition from active learning and some CL literature (which favors uncertain / boundary samples as informative). The paper motivates this as being about *representativeness for class identity* rather than *informativeness for boundary sharpening*, and the empirical results directionally support this across multiple datasets. Fully resolving whether this advantage is specifically tied to the federated aggregation context (where inter-client representation mismatch makes boundary samples less transferable) or is a more general phenomenon would be a valuable follow-up.

---

## Suggestions

1. **Increase seeds to at least 5 and add statistical testing** (e.g., Wilcoxon signed-rank across datasets) for the BI vs. other uncertainty metrics comparison. Given the variance levels observed (std sometimes exceeding the point estimate differences), this is the single most important fix to make the core methodological claim credible.
2. **Add a joint-training oracle** and ideally a single-client (non-federated) online-CL upper bound to give readers reference points for interpreting the absolute performance levels.
3. **Either justify the Gaussian noise magnitude for text TTA empirically** (e.g., ablate noise std ∈ {0.01, 0.1, 0.5}) **or use a more principled perturbation** (e.g., random dropout on embedding dimensions, or sampling from the local empirical distribution of embeddings).
4. **Add an experiment with a non-frozen backbone for text** (e.g., fine-tuning the sentence encoder or using a trainable embedding layer) to make the multi-modality claim more robust.
5. **Report memory buffer class composition over time** as an ablation to separate the contribution of BI uncertainty estimation from the class-balanced update strategy inherited from Chrysakis & Moens (2020), which is currently conflated.
6. **Discuss task boundary detection** in the limitations section and acknowledge that the known-boundary assumption is a key gap between the current setting and fully unsupervised deployment.

---

**Overall evaluation:** The paper makes a concrete and useful contribution by formalizing Online-FCL and demonstrating the failure of generative baselines in this regime. The BI-based memory management is a reasonable and motivated methodological choice with good multi-dataset empirical coverage. However, the marginal statistical advantage of BI over simpler uncertainty metrics — the core technical claim — is not convincingly demonstrated with only 3 seeds and overlapping standard deviations on CIFAR-100 and KC-Cell. The text modality experiments are weakened by the frozen-embedding setup. The paper sits at the boundary of acceptable quality for the venue: the problem formalization and empirical breadth are genuine contributions, but the technical soundness of the central claim needs stronger statistical support to be fully convincing.