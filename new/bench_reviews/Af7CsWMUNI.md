Now let me look at the calibration papers to determine appropriate scoring.Now I have enough information to synthesize the final review. Let me check the suspicious identical results across models more carefully before writing it up.Now I have a complete picture of the paper and calibration anchors. Let me write the final review.

---

## Summary

This paper proposes a new inference paradigm called "Representation-Level ICL via Unlabeled Texts." The core idea is to replace traditional in-context learning (concatenating labeled demonstrations at the text level) with a representation-level approach: extract hidden states from other unlabeled test inputs, and fuse them with the test sample's hidden states via an attention-like mapping, before predicting using the frozen language model head. The authors motivate this by identifying two gaps between ICL and pretraining—label appearance and weak semantic relevance—and demonstrate that their method outperforms zero-shot baselines across 8 datasets and 4 model families, and on average outperforms traditional ICL with labeled demonstrations on general-domain datasets.

---

## Strengths

- **Novel framing of the pretraining-inference gap.** The paper cleanly identifies and formalizes two gaps (label appearance, weak semantic relevance) and the controlled analysis in Section 2.2 (Table 2) providing empirical evidence that removing labels helps general-domain but hurts specific-domain tasks is a genuinely interesting and actionable observation.

- **Empirical breadth.** Results span 8 datasets (general and specific domain), 4 open-source model families of varying size (2.7B–13B), and multiple ablation dimensions (pooling strategy, retrieval count, mapping function). The coverage is broad for this type of work.

- **Simple, training-free design.** The method uses frozen LLM representations and a non-parametric attention-like mapping, requiring no labeled data, no training, and no access to the training set—attributes highly practical for low-resource scenarios.

- **Consistent gains on most datasets.** On 6 of the 8 datasets, the method shows real numerical variation across models and meaningful improvements over zero-shot, supporting a genuine signal in the representation-level fusion idea.

---

## Weaknesses

### Fatal
*None that single-handedly invalidate all results.*

### Major

- **Apparent majority-class collapse on MRPC and COLA (2 of 8 datasets).** Every model (GPT-Neo-2.7B, Mistral-7B, Llama2-7B, Llama2-13B) and every ablation variant (all 3 pooling strategies, Tables 4, 6, 7) produces *exactly* 66.49% on MRPC and *exactly* 69.13% on COLA—identical to two decimal places, regardless of the underlying model architecture or configuration. These are the known majority-class baselines for these datasets (MRPC is ~66.5% positive-paraphrase majority, COLA is ~69.1% acceptable-sentence majority). For Mistral-7B, whose zero-shot MRPC is only 35.19%, the method "improves" it by 31.3 points to 66.49—but if this is simply defaulting to the majority class, that is not a genuine improvement. This systematic collapse, which persists across every model and every variant without a single exception, is a major empirical concern. It means that the claims of "broad and consistent improvement" and "small models beating larger models" may partly rest on majority-class trivial predictions for 2 of 8 datasets. The authors provide no analysis of per-class accuracy, per-prediction behavior, or why all model families converge to this exact number. This must be investigated and resolved.

- **Informational asymmetry between proposed method and baselines.** The method explicitly retrieves from the *full test set* (Eq. 7, Step 2, Figure 2: "unlabeled texts from test set"), performing transductive inference where every test example sees the entire remaining test corpus at prediction time. Zero-shot baselines predict each sample in isolation; standard ICL baselines draw from a training pool. This is acknowledged in Table 6's header ("test set" vs. "training set"), and the paper does not present it as hidden—but the resulting comparisons are nonetheless not apples-to-apples. A zero-shot or training-set ICL method does not have access to *any* unlabeled target-domain examples, while the proposed method has access to all of them. Gains of 30–42 points on some datasets could plausibly arise purely from domain-adaptive smoothing via test-corpus retrieval, not from the representation-level mechanism per se. Critically, there is no baseline that uses the same test-set retrieval corpus but applies it at the text level (standard unlabeled ICL with test-set context)—which would isolate the representation-level contribution.

- **Unjustified 0.4/0.6 mixing coefficient (Eq. 11).** The formula `0.4 × mean(retrieved) + 0.6 × original` is a central step in the pipeline and is presented with no derivation, theoretical motivation, or sensitivity analysis. Given the number of design choices already tuned (pooling strategy, k∈{16,32,64}, τ∈{1,1.5}, 5 mapping functions), this fixed ratio adds to concerns that the method's performance was shaped by implicit optimization on the reported datasets. An ablation over this ratio is necessary.

### Minor

- **Theoretical analysis limited to a single attention layer.** Section 2.3 analyzes "Method 1 vs. Method 2" using a single self-attention layer for a single demonstration (m=1). Modern LLMs have dozens of layers with nonlinear activations and multi-head attention. The paper does not discuss how this simplification relates to the full multi-layer setting, and the presented analysis cannot fully justify the design choices in Step 2 (which operates on the final layer's hidden states, not an intermediate cross-attention between layers). This limits the theoretical grounding to an illustrative analogy rather than a rigorous justification.

- **No baseline from the same retrieval regime.** The fairest comparison to isolate the representation-level contribution would be "unlabeled text-level ICL where the retrieval corpus is also the test set." Table 5 shows "Text" vs. mapping methods, but the "Text" baseline there is concatenation of unlabeled test inputs at the text level—which is the right ablation. Curiously, this is included only in Table 5 (two models) rather than Table 6, where the comparison against traditional ICL (training set) is the headline claim. Reporting this baseline consistently would help.

- **Inconsistent gains on specific-domain datasets undermine headline claims.** The abstract states the method "outperforms traditional ICL with extra information of gold labels." Table 6 shows −48.68 on PHRASE for GPT-Neo vs. Topk-ICL, and −23.06 for Llama2-13B. These are severe failures on a specific-domain dataset, which are correctly noted in Section 4.3 but do not adequately qualify the headline abstract claim.

### Trivial

- **Garbled Figure 1 text.** The figure caption text in the extracted version references "MNPC" instead of MRPC and "COQA" instead of COLA—this appears to be a parser artifact per the instructions, but it indicates the figure description may need to be verified in the submission.

---

## Nice-to-Haves

- **t-SNE / UMAP of original vs. reconstructed representations** colored by class label to verify whether the representation-level fusion genuinely reshapes class geometry or averages indiscriminately.
- **Online/streaming evaluation**: evaluate whether the method works when test instances arrive one-at-a-time rather than in a full batch, as the current batch-retrieval setup is unrealistic for many deployment scenarios.
- **Majority-class baseline** explicitly reported alongside results, so readers can verify the scale of genuine improvement vs. majority-class convergence.
- **Evaluation on at least one larger-scale or harder benchmark** (e.g., MMLU, HellaSwag) to test generalizability beyond small GLUE-style tasks.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic – "Data Leakage" framing.** The critic frames the use of test-set data as hidden data leakage and calls it a "structural flaw." This is too strong: the paper *explicitly* discloses in Eq. 7, Step 2, Figure 2, and Table 6's header that it uses "unlabeled texts from the test set." This is a transparent design choice, not a deceptive practice. The informational asymmetry is real (and kept as a Major concern above), but framing it as undisclosed "data leakage" mischaracterizes the paper. Removed the "deceptive" framing while keeping the substance.

- **Harsh Critic – "ICL paradigm" label is fundamentally misleading.** The critic argues the method is "not ICL at all" because the LLM forward passes are independent per example. However, the paper explicitly frames the method as a *new ICL paradigm* that operates on representations rather than text sequences—a definitional extension, not a terminological fraud. Whether one agrees with the framing is a matter of scope, but it is not a factual error or mischaracterization that warrants removal of the paper's central claim. This is an overclaim in degree but not kind; it fits better as a nuance to the theoretical motivation discussion (the single-layer analysis) rather than a fatal structural flaw.

- **Harsh Critic – Missing related works.** Per the meta-review rules, missing related work criticisms are removed as we cannot verify existence of external references.

- **Harsh Critic – Statistical p-value under-specification.** The critic argues that p-values are spurious (paired over only 8 datasets). While the statistical analysis is weak, the paper's empirical improvements are large and visible enough to stand without lean-on-p-values rhetoric. This is a trivial presentation issue, not a substantive weakness.

- **Neutral Reviewer / Spark – Variance reporting / confidence intervals.** Requesting standard deviations or confidence intervals for large-scale single-run LLM evaluations is not standard practice in this sub-community. Moved to nice-to-have.

- **Spark – Hyperparameter selection protocol and potential data leakage.** The Spark reviewer asks whether hyperparameters were selected using test-set performance. The paper says "We initially set k=64, τ=1 to identify the optimal pooling strategy. Following that, we adjust the value of k, and finally, we tweak the value." This describes a sequential tuning protocol but does not clarify whether the tuning dataset is the reported test set. This is a legitimate concern but is a reproducibility/disclosure issue rather than a confirmed flaw; it is captured under the "unjustified 0.4/0.6 weighting" point which is the most salient symptom.

- **Human Finder – Missing related works such as G7u4ue6ncT (I2CL).** Per the rules, missing related work is not flagged.

---

## Novel Insights

The observation that removing labels from ICL demonstrations benefits general-domain tasks (where the label-appearance gap dominates) but hurts specific-domain tasks (where genuine input-label mapping signals are necessary) is a useful empirical regularity. The paper's identification of this domain-type moderation effect is more actionable than typical "ablate labels" studies in the ICL literature. Additionally, the finding that representation-level fusion consistently outperforms text-level concatenation within the same retrieval corpus (Table 5) is a concrete, reproducible empirical result that supports the "weak semantic relevance" hypothesis, even if the theoretical justification is underspecified.

---

## Suggestions

1. **Investigate and explain the identical MRPC=66.49 and COLA=69.13 results across all models and all variants.** Report per-class accuracy (not just overall accuracy) on these datasets. If the method is predicting the majority class for all inputs, state this explicitly. If not, explain the mechanism that causes four architecturally distinct models to produce the identical number to two decimal places.

2. **Add a text-level unlabeled test-set retrieval baseline to Table 6** (alongside Table 5), so the representation-level vs. text-level comparison is directly visible against the same retrieval regime as the "no labels" claim.

3. **Provide a sensitivity analysis or principled derivation for the 0.4/0.6 coefficient** in Eq. 11. At minimum, run a grid search with α ∈ {0.1, 0.2, …, 0.9} and show the landscape.

4. **Clearly separate the two claims**: (a) representation-level fusion is better than text-level concatenation given the same unlabeled test-set corpus (Table 5 supports this), and (b) unlabeled test-set retrieval beats labeled training-set ICL (Table 6—but this comparison is partly confounded by corpus source and domain match). Currently, both claims are merged, which weakens the paper's precision.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Decision | Scores | Similarity |
|---|---|---|---|
| "Calibrate to Discriminate" (RUn41kd6i0) | Reject | 3, 6, 3 (avg ~4) | Label-free ICL, unfair zero-shot comparison |
| "Disentangling Latent Shifts of ICL" (pdf6MbXnAS) | Reject | 8, 3, 6, 6 (avg ~5.75) | Representation manipulation for ICL, mixed reception |
| "Exploring ICL and IT" (SSItGuNLD2) | Reject | 5, 3, 8, 5, 3 (avg ~4.8) | ICL analysis limited scope, limited evaluation |
| "Revisiting ICL Inference Circuit" (xizpnYNvQq) | Accept Poster | 8, 6, 6, 6 (avg 6.5) | Mechanistic ICL analysis—stronger grounding |
| "Unsupervised Meta-Learning via ICL" (Jprs1v2wPA) | Accept Poster | 6, 6, 6, 6 | Representation-level ICL—cleaner methodology |

The paper under review has a novel, interesting idea and breadth of experiments that place it above the pure-reject "Calibrate to Discriminate" territory. However, the major concerns—especially the systematic majority-class collapse on 2 of 8 datasets (which is an unacknowledged empirical anomaly, not merely a framing issue), the absence of a fair comparison baseline within the same retrieval regime, and the unjustified design choice in Eq. 11—place it solidly below the accepted poster papers in similar territory. Papers like "Revisiting ICL Inference Circuit" earned acceptance through rigorous ablations and thorough circuit characterization; this paper's ablations do not resolve its core ambiguities.

**Axes:**
- **Originality:** Moderate — the idea of representation-level ICL is new; the specific method is straightforward.
- **Importance of research question:** Good — label-free, training-free ICL is an important practical direction.
- **Claims supported by experiments:** Weak-to-moderate — 6 of 8 datasets are plausible; 2 datasets appear to collapse to majority class; comparisons are not fully fair.
- **Soundness of experiments:** Weak — no majority-class baseline, no same-regime text-level comparison, unjustified mixing coefficient, no per-class breakdown.
- **Clarity:** Good — well-written and transparent about the transductive setting.
- **Value to research community:** Moderate — the observations about domain-type moderation of label effects are useful; the method itself needs cleaner validation.

**Final score: 4.5 (Reject)**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>