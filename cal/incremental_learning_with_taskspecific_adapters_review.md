=== CALIBRATION EXAMPLE 24 ===

# Final Consolidated Review
---

## Summary
This paper proposes integrating task-specific bottleneck adapters with standard regularization-based incremental learning (EWC, LwF, MAS, etc.). Unlike prior adapter-based approaches that freeze the backbone, the method co-trains the backbone alongside per-task adapters, supplemented by an additional backbone regularization term that encourages the backbone to learn task-invariant features. Experiments on CIFAR-100 (primary) and ImageNet (secondary) in the Task-IL setting show consistent improvements over non-adapter counterparts across multiple regularization families, task scales, and task orderings.

---

## Strengths

- **Consistent, cross-family empirical improvement on CIFAR-100.** Figure 3 shows adapter-enhanced versions outperform their baselines for every regularization family tested (EWC, MAS, PathInt, LwF, LwM), with ~3% gains for weight-regularized and up to ~5% for prediction-regularized methods. This breadth and consistency across 5 methods is more convincing than a single ablation.

- **Empirically validated co-training hypothesis.** Table 2 directly compares co-trained vs. frozen-backbone variants (LwF-A: 74.0% vs. LwF-A-FrB: 72.9%), providing clear evidence that co-training the backbone is the right design choice and distinguishing the method from frozen-backbone adapter work (Liang & Li, 2024).

- **Robustness to task ordering, including adversarial ordering.** Figure 5 evaluates three distinct orderings, including the higher-diversity coarse-grained ordering that most methods struggle with. Adapter-enhanced methods maintain advantages across all orderings, suggesting the benefit is not an artifact of a favorable data sequence.

- **Plug-and-play compatibility.** The method requires no architectural overhaul to existing regularization methods: for weight-regularized methods, adapter parameters are simply excluded from the regularization term; for prediction-regularized methods, an additional backbone distillation term is added. This simplicity and generality are genuine engineering contributions.

- **Motivated empirical premise.** Figure 1 provides clear and concrete motivation: the coarse-grained task ordering (higher inter-task diversity) causes substantially more forgetting under LwF than other orderings, grounding the argument that inter-task differences drive catastrophic forgetting.

---

## Weaknesses

### Fatal
None.

### Major

- **Class-IL results relegated to appendix.** The paper evaluates almost exclusively in the Task-IL (task-ID oracle) setting, where a separate head is invoked per task at inference time. The adapters themselves require knowing which adapter to invoke at test time, making the method inherently oracle-dependent. Class-IL is the harder, more realistic, and increasingly standard ICLR benchmark for incremental learning. Relegating these results to Appendix B without summarizing them in the main paper significantly undermines the breadth-of-contribution claim. Given the paper's framing that it "eliminates the stability-plasticity dilemma," the omission of the more practical and challenging evaluation setting is a major gap.

- **ImageNet results partially contradict the headline claim.** Table 1 shows that at Task 10: LwF-A (67.2%) < LwF (68.2%), and LwM-A (56.9%) < LwM (58.0%) — i.e., the adapter versions are *worse* than the baselines on these methods by the end of the sequence. EWC-A also starts 4.3 points below EWC (76.0 vs. 80.3 at Task 2), recovering only partially by Task 10. The paper acknowledges reusing CIFAR-100 hyperparameters for ImageNet, but then asserts "methods with adapters yield the best performance across all incremental tasks," which is directly contradicted by these numbers. This selective framing undermines confidence in the ImageNet conclusions.

- **The core disentanglement claim is entirely unverified.** The central narrative — that the backbone learns task-invariant features while adapters capture task-specific information — is the paper's conceptual foundation, but no evidence is provided that this separation actually occurs. No feature similarity analysis (e.g., CKA, centered kernel alignment), representation probing, mutual information measurement, or visualization is offered. Without this, the proposed mechanism remains an untested hypothesis rather than a demonstrated phenomenon.

- **Overclaiming the resolution of the stability-plasticity dilemma.** The abstract says "effectively addressing the stability-plasticity dilemma," but both the introduction and conclusion escalate this to "eliminating the stability-plasticity dilemma." All figures (3, 4, 5) show monotonically declining accuracy as more tasks are added — including with adapters. The dilemma is softened, not eliminated. This is a significant overclaim that should be corrected throughout.

### Minor

- **Figure 2 has a mislabeling inconsistency.** The figure's alt-text/diagram describes "Our method with adapters" as having a **frozen** adapter β, but the paper's entire contribution is that adapters are *co-trained* (not frozen). The text caption below Figure 2 correctly states "we allow adapters to be co-trained," but the figure diagram contradicts this. Since co-training vs. freezing is the central distinguishing claim from prior work, this labeling error in the figure needs to be corrected urgently.

- **Backbone regularization term (R_φ) is underspecified.** Equation (113) involves φ^{t'}(x), which refers to the backbone at previous task t'. This apparently requires storing a snapshot of the backbone after each task, which is a non-trivial memory overhead that is never discussed or quantified. Additionally, the Linear_{d×c} projection used in R_φ is not described: are its weights shared, task-specific, or per-pair? Their management during training is never specified. The assumption that c equals the number of classes per task (which must be equal across all tasks) is also not explicitly stated as a requirement of the method.

- **Bottleneck width inconsistency between ablation and ImageNet experiment.** Figure 6 shows that width 256 performs best on CIFAR-100, but Table 1 (ImageNet) uses width 128, reportedly from CIFAR-100 settings. The paper does not explain why 128 was chosen for ImageNet rather than the empirically best width of 256.

- **No forgetting metrics reported.** The paper claims to address the stability-plasticity dilemma but only reports average accuracy A_t. Explicit backward transfer (forgetting) metrics are needed to substantiate the claim that stability is genuinely improved, not just that the average accuracy happens to be higher.

- **No confidence intervals or standard deviations despite 10-seed averaging.** Results are averaged over 10 random seeds, but no standard deviations are reported in any table or figure. Given that some claimed improvements are 1–3%, and that Figure 5 (coarse ordering) shows gaps as small as ~1%, statistical uncertainty bands are needed to confirm significance.

- **No comparison with strong rehearsal-based or modern methods in main text.** The paper compares adapters vs. no-adapters within regularization-based families. While DualPrompt and iTAML appear in Table 2, no strong replay-based methods (iCaRL, DER++, etc.) are included in the main text, making it hard to judge where the adapter-enhanced regularization methods stand relative to the broader SOTA.

### Tiny

- **Abstract typo:** "compromising two distinct components" should be "comprising."
- **Limitations section is absent.** The paper does not explicitly discuss known limitations (oracle dependence, linear adapter count growth, mixed ImageNet results), which are important for future researchers.

---

## Nice-to-Haves

- **Hyperparameter tuning for ImageNet.** Even a lightweight search over bottleneck widths (e.g., 64, 128, 256) on ImageNet would significantly strengthen the ImageNet conclusions. The current transfer of CIFAR-100 hyperparameters is an acknowledged shortcut that limits the ImageNet findings.

- **Ablation on adapter placement depth.** The adapters are placed only between the backbone and the task head (single bottleneck at the top). Testing placement at intermediate backbone layers would clarify whether this specific placement is optimal or just convenient, and would better support the architectural narrative.

- **Longer task sequences (e.g., 20+ tasks).** All experiments use 10 tasks. Evaluating on longer sequences would demonstrate scalability and strengthen claims about lifelong learning applicability.

- **Computational overhead analysis.** A brief analysis of training time and total memory cost (backbone snapshots + multiple adapters) relative to baselines would help practitioners assess the practical trade-offs, given that memory efficiency is part of the motivation.

- **Feature similarity analysis.** Even a simple CKA heatmap or t-SNE visualization comparing backbone representations across tasks before and after adapter training would provide the first empirical support for the claimed invariant/task-specific separation.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

- **[Removed] Adapter placement differs from Houlsby et al. (2019).** The critic complains that the paper's single post-backbone adapter is "architecturally quite different" from Houlsby-style per-layer adapters. This is factually true, but the paper clearly states adapters are used as "feature modifiers" between the feature extractor and the classifier head — this is the paper's stated design choice, not an oversight. Evaluating the paper for not replicating a different paper's architecture is scope creep.

- **[Removed] EWC Fisher information may be invalid post-adapter.** The critic argues that Fisher information computed at task t' may be meaningless for post-adapter representations at task t. While theoretically interesting, this is a known and broadly accepted limitation of EWC itself (not specific to this paper's contribution), and the method demonstrably improves over EWC-without-adapters empirically. This is a theoretical concern that applies equally to standard EWC and does not specifically invalidate the paper's contribution.

- **[Removed] Request for confidence intervals as essential standard.** For CIFAR-100 task-IL with 10 seeds, standard deviations should be reported (kept as minor weakness above). However, the harsh critic frames this as making all results invalid — that framing is too strong. The 10-seed average is a reasonable practice; the weakness is the missing reporting of variance, not the absence of formal significance tests.

- **[Removed] "Inter-task differences are the *primary* driver of forgetting" lacks controlled evidence.** While technically correct that the paper provides no controlled ablation separating inter-task diversity from other forgetting mechanisms (gradient interference, layer geometry, etc.), this is a motivational framing claim, not a core methodological claim the evaluation is built on. The method does not *require* this claim to be true; it simply motivates the approach.

- **[Removed] Nonlinearity after up-projection is non-standard.** The up-projection uses a nonlinear activation g, unlike standard Houlsby adapters. While the harsh critic notes this is non-standard, the paper never claims to replicate Houlsby adapters exactly — it explicitly re-purposes adapters. This is a design choice, and the empirical results support its effectiveness.

- **[Removed] Comparison with iTAML is unfair (uses exemplars).** The harsh critic notes that iTAML uses exemplars while the paper is positioned in the no-exemplar regime. However, the paper tests whether adapters *improve* iTAML (which they do: 79.0 → 80.1), not whether their method beats iTAML as a baseline. This is not an unfair comparison — the asymmetry favors the baseline (iTAML with exemplars) and the paper still shows improvement.

---

## Novel Insights

The most actionable insight beyond the paper's own stated contributions is the interaction between inter-task diversity and adapter benefit: Figure 5 shows that while adapter-enhanced methods remain superior on the coarse-grained (high-diversity) ordering, the margin shrinks to ~1% in some cases. This suggests the benefit of task-specific adapters is most pronounced when tasks are highly dissimilar and least pronounced when they share substantial structure — a nuance the paper acknowledges in passing but does not explore deeply. A future investigation into *when* adapters are most beneficial (as a function of inter-task distance, dataset scale, or task duration) could be a valuable follow-up. The Figure 2 labeling error also hints at a possible conceptual ambiguity in the paper regarding whether the adapter parameters should be viewed as frozen after task training for use at inference (i.e., task-specific fixed modules) versus continually trained — clarifying this distinction may reveal further design choices.

---

## Suggestions

1. **Move Class-IL results to the main body.** Promote the Appendix B results to Section 4.2, even as a compact table. Discuss frankly whether adapter-enhanced methods improve or match non-adapter methods in Class-IL, and if there are gaps, explain why.

2. **Correct Figure 2.** The "Our method with adapters" panel of Figure 2 should clearly label the adapter β as *fine-tuned* (blue), not frozen (white). This is the single most confusing element in the paper.

3. **Fix the overclaim.** Replace "eliminating the stability-plasticity dilemma" throughout with language consistent with the evidence: "substantially mitigating," "reducing," or "alleviating."

4. **Add standard deviations to all tables and figures.** With 10 seeds already run, reporting σ is trivial and necessary to validate small but claimed improvements.

5. **Clarify the backbone snapshot requirement in R_φ.** Explicitly state whether previous backbone states φ^{t'} are stored, and report the memory overhead. If they are not stored, explain what φ^{t'} refers to in practice.

6. **Reconcile the bottleneck width choice for ImageNet.** Either use width 256 (empirically best per Figure 6) for ImageNet experiments, or justify why 128 was retained. Report results with width 256 on ImageNet if feasible.

7. **Add at least one forgetting/backward-transfer metric.** Even a single table reporting per-task forgetting rates alongside average accuracy would substantially strengthen the stability-plasticity claim.

---

**Axes evaluation:**
- **Novelty:** Moderate. Co-training adapters with the backbone in an IL setting is a distinct and useful variation over frozen-backbone adapter methods, but the adapter concept itself is borrowed and the application is straightforward.
- **Technical soundness:** Adequate but with gaps — the R_φ regularization is underspecified and the disentanglement mechanism is asserted but unverified.
- **Empirical support:** Moderate. CIFAR-100 results are convincing and well-replicated; ImageNet results are mixed and under-tuned; Class-IL is hidden in an appendix.
- **Significance:** Moderate. The plug-and-play compatibility with regularization methods is practically useful, but the method's oracle dependence and untested Class-IL performance limit its broader significance.
- **Clarity:** Adequate overall, but significantly hurt by the Figure 2 labeling error and the overclaiming language around "eliminating" the dilemma.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 5.0, 3.0, 1.0]
Average score: 3.0
Binary outcome: Reject
