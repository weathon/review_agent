Now let me look at the human reviews for calibration purposes.## Summary
MM-R³ is a benchmark for evaluating the consistency of Multimodal Large Language Models (MLLMs) under semantically equivalent but superficially different inputs. The paper introduces three tasks—Question Rephrasing, Image Restyling, and Context Reasoning—covering both linguistic and visual perturbation axes, evaluates nine open- and closed-source MLLMs, and proposes an adapter-based mitigation strategy. The central finding is that accuracy and consistency are not tightly aligned across models, and that even state-of-the-art models show large consistency drops compared to their sampling-regime performance.

---

## Strengths

- **Timely and important problem framing.** Consistency is an understudied axis for MLLM evaluation. The paper's framing—that consistency is a *necessary but not sufficient* condition for trustworthy deployment—is conceptually sound and well-motivated throughout.

- **Multi-task benchmark covering distinct perturbation axes.** Three tasks probe different dimensions: linguistic surface form (Question Rephrasing), visual domain transfer (Image Restyling), and visual occlusion reasoning (Context Reasoning). This multi-faceted design provides a richer picture than single-perturbation evaluations.

- **Meaningful "Sampling vs. All" decomposition.** Showing that BLIP-2 and LLaVA 1.5M drop from 100% sampling consistency to ~48–50% consistency under rephrasing (Table 2) is an illuminating diagnostic—stochasticity is not the bottleneck, prompt sensitivity is.

- **Comprehensive empirical coverage.** Nine models including GPT-4V, GPT-4o, and Gemini are evaluated. Supplementary analyses of model size (Table 5), temperature (Figure 3), and resolution (Figure 2) add meaningful depth.

- **Human validation of data quality.** A forced-choice experiment on 100 samples each (92% rephrasing equivalence, 86% restyling equivalence) provides at least initial credibility for the benchmark inputs.

---

## Weaknesses

### Fatal
None.

### Major

- **Adapter improvement is standard task fine-tuning, not a consistency-specific mechanism.** The adapter is trained with `CrossEntropyLoss`—standard conditional language modeling against ground-truth tokens—rather than any objective that couples outputs across semantically equivalent variants. There is no loss term enforcing output invariance across rephrasings/styles/masks. Crucially, the paper itself admits the source of the largest gains: *"This is largely because original MLLMs are not trained on data of this form."* The Image Restyling and Context Reasoning accuracy improvements (BLIP-2: 13.0→36.7% and 27.9→54.6%) are far too large for a lightweight adapter on frozen weights, and are consistent with domain adaptation to a new data distribution—not learning general consistency invariance. The "consistency improvement" framing thus overstates what the experiments demonstrate. To support the claimed contribution, the authors would need either a contrastive/consistency-specific loss, or evaluation on held-out perturbation families unseen during training.

- **No baselines for the adapter.** The adapter is compared only against the bare base model. There is no comparison to: (a) standard LoRA/adapter fine-tuning on the same data with a conventional architecture placement, (b) fine-tuning the base model directly on the same number of examples, or (c) prompt ensembling or self-consistency decoding. Without such comparisons, it is impossible to determine whether the adapter's architectural design (Bi-LSTM + MLP + learnable prefix) contributes anything beyond simply seeing more task-similar data.

- **Adapter tested on only two model architectures.** Despite the benchmark evaluating nine models and the paper claiming the adapter "can be added to any MLLM," experiments are limited to BLIP-2 and LLaVA 1.5M—both in the low-consistency regime that makes absolute gains easier to show. Models with divergent consistency profiles (e.g., Qwen-VL-Chat, MoE-LLaVA) are untested, severely limiting the generalizability claim.

### Minor

- **Question rephrasing prompt is answer-leading.** The GPT-3.5 rephrasing prompt explicitly includes the ground-truth answer: *"Please give me three different types of rephrased questions to which the answer would be (Answer)."* This introduces potential lexical bias—generated rephrasings may orbit near the answer string, making the task partially a reading-the-prompt test rather than a pure VQA test. This is not discussed as a limitation and may inflate apparent model differences on this task.

- **The 0.7 similarity threshold for Con is not validated for this setting.** The threshold is borrowed from the STS benchmark (Cer et al., 2017) without any pilot study confirming it captures human-perceived "same answer" judgments in the multimodal open-ended output regime. No sensitivity analysis across alternative thresholds (e.g., 0.5–0.9) is provided.

- **Consistency metrics reward "consistently wrong" outputs without acknowledgment in quantitative analysis.** The paper explicitly notes in qualitative analysis (Figure 5) that some models score Con=100 while Acc=0 (Qwen-VL-Chat always predicting "columbia"; BLIP-2 always predicting "bat"). While the paper correctly frames consistency as necessary-but-not-sufficient, there is no systematic quantitative decomposition of how much reported Con improvement comes from "correctly consistent" vs. "incorrectly consistent" behavior—leaving the aggregate numbers ambiguous.

- **Speculative cross-model interpretations in §4.3.2.** Statements like "Qwen-VL-Chat's better language representations explain its rephrasing consistency" and "BLIP-2's pre-training aids context reasoning" are purely narrative—no ablations or controlled comparisons support them.

### Trivial

- **Human semantic equivalence evaluation covers only 100 samples per task**, which leaves 8–14% noisy pairs that could affect aggregate consistency metrics. The paper acknowledges this but does not report sensitivity of conclusions to filtering these pairs.

---

## Nice-to-Haves

- **Out-of-distribution generalization test for the adapter**: Train on one task's perturbation data and evaluate on the other two tasks to see whether any general consistency invariance is learned.
- **Per-example accuracy–consistency scatter plots**: Aggregated model-level comparisons are not sufficient to substantiate the claim that accuracy and consistency are decoupled. Per-example scatter plots or correlation coefficients would be far more convincing.
- **Embedding-space visualization**: A t-SNE or PCA plot showing whether adapter embeddings of semantically equivalent inputs cluster closer together would provide mechanistic evidence for the stated goal of "modifying embeddings to be invariant to surface form variations."
- **Computational cost of the adapter**: Parameter count, training time, and inference overhead would help practitioners assess feasibility.
- **Evaluation on standard downstream VQA benchmarks**: Running the adapter on VQAv2, GQA, or TextVQA would help rule out capability regression from fine-tuning.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**Harsh Critic §1 — "Consistency metrics are self-referential and do not match semantic consistency"** (partially removed/demoted): The paper explicitly frames consistency as a *necessary but not sufficient* condition, and Figure 5 directly showcases "consistently wrong" models as a deliberate qualitative finding. The harsh critic overstates this as a fatal flaw; the paper's conceptual framing is sound and honestly acknowledges the limitation. Retained only the valid concern about the lack of quantitative decomposition between "correctly consistent" vs. "incorrectly consistent" as a minor weakness.

**Harsh Critic §3 — "Sampling vs All confounded by decoding parameters"**: The paper directly states in §4.3.1 that BLIP-2 and LLaVA 1.5M are set with temperature=0, which explains their Sampling Con=100. This is explicitly discussed, not hidden. The criticism misreads this as an unreported confound.

**Harsh Critic §4 — "Claims about accuracy–consistency trade-off unsupported without statistics"**: The paper makes a qualitative empirical observation, not a causal statistical claim. Requesting confidence intervals and correlation analyses is a nice-to-have for a benchmark paper of this scale, not a fatal flaw.

**Harsh Critic — Context reasoning masking changes inferential difficulty**: While partially valid in principle, the paper's design (controlling mask object size to 0.1–0.25 of image) mitigates the most extreme cases, and the premise that identical context should yield identical inference regardless of mask type is reasonable. This is a soft limitation, moved to Nice-to-Haves.

**Human Finder — Overlap between benchmark tasks**: The claim that Question Rephrasing and Context Reasoning overlap is not well supported—one tests linguistic variation on the same image, the other tests visual mask variation with the same question. These are clearly distinct axes.

---

## Novel Insights

The paper's most genuinely novel observation is the dissociation between *sampling consistency* and *prompt-driven consistency*: deterministic models (BLIP-2, LLaVA 1.5M at temperature=0) that achieve Con=100 under repeated identical prompts collapse to Con≈48% when the question is paraphrased. This demonstrates that model stochasticity and prompt sensitivity are orthogonal failure modes, and that prompt-sensitivity is the dominant source of inconsistency in these models—a distinction that has implications for how future benchmarks and fine-tuning objectives should be designed. The observation that closed-source models (GPT-4o, Gemini) show substantially smaller Sampling→All drops than open-source models is also empirically useful for the community.

---

## Suggestions

1. **Reframe the adapter contribution honestly**: Acknowledge in the abstract and conclusion that the adapter achieves improvements partly through task-specific fine-tuning (especially for Image Restyling and Context Reasoning), not solely through consistency invariance. Run at least one ablation that tests whether vanilla fine-tuning on the same data achieves similar gains.
2. **Add a contrastive loss term**: A simple loss that minimizes the divergence between adapter outputs for different rephrasings/styles/masks of the same item would genuinely operationalize the stated consistency objective.
3. **Conduct threshold sensitivity analysis** for S_C across 0.5–0.9 and report whether model rankings change.
4. **Evaluate the adapter on at least 2 additional architectures** (e.g., MoE-LLaVA, Qwen-VL-Chat) before claiming generality.
5. **Revise the Question Rephrasing prompt** to avoid including the ground-truth answer, or at minimum analyze whether this biases model outputs.

---

## Score and Decision

**Calibration against comparable papers:**

- *Measuring Free-Form Decision-Making Inconsistency* (0pbxX2jatP): Rejected, scores 5/3/5 (avg ~4.3). Similar problem framing (consistency measurement via semantic similarity), no mitigation strategy, single domain. The present paper is broader (multimodal, 3 tasks, 9 models) and adds a mitigation attempt—slightly stronger.
- *TP-Eval* (QnjUf0VytI): Rejected, scores 5/3/6 (avg ~4.7). Prompt sensitivity in MLLMs, evaluation-focused. The present paper's benchmark is more comprehensive.
- *Permutation Sensitivity in LLMs* (H8Qg1IIMaR): Rejected, scores 6/5/6/5 (avg ~5.5). More focused scope (MCQA permutations), similar issues with limited mitigation. That paper is tighter and more rigorous.
- *Prompt Formatting Sensitivity* (RIu5lyNXjT): Accepted poster, scores 6/8/6 (avg ~6.7). More rigorous statistical treatment, FormatSpread algorithm, formal grammar of prompt formats—substantially more methodologically sound.
- *MuirBench* (TrVYEZtSQH): Accepted poster, scores 3/5/6/6/6 (avg ~5.2). More comprehensive (20 models, 2600 QA pairs, pairwise design), no mitigation strategy; still borderline accepted due to benchmark value alone.

The present paper is below MuirBench in benchmark rigor (fewer models, smaller test sets, weaker pairwise validation) and the adapter contribution—claimed as a second pillar—is not credibly validated as a consistency-specific mechanism. It sits comfortably below the rejected permutation paper in methodological rigor and close to TP-Eval quality. Score: **4.5**.

**Originality:** Moderate — consistency evaluation for MLLMs is novel but closely related to LLM consistency work; the three-task multimodal framing is a genuine extension.  
**Importance of research question:** High — consistency is a real and understudied concern for MLLM deployment.  
**Claims vs. support:** Below standard — the adapter claims are overclaimed relative to what the experiments demonstrate.  
**Soundness of experiments:** Adequate for the benchmark; weak for the adapter.  
**Clarity of writing:** Good overall, readable and well-structured.  
**Value to community:** Moderate — the benchmark fills a real gap; the adapter contribution is unconvincing in its current form.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>