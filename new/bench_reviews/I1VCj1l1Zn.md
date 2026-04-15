Now let me search for calibration papers.Now I have enough information. Let me write the comprehensive final review.

---

## Summary

DLP-LoRA proposes a sentence-level dynamic LoRA fusion method using a lightweight external mini-MLP classifier (5M parameters) to route among a bank of task-specific LoRA adapters via top-p sampling. The method avoids per-token gating overhead by making routing decisions once per sentence and exploits parallel CUDA GEMM operations for multi-LoRA computation. Evaluations span 26 tasks (17 MCQ + 9 QA) across four LLM backbones, showing performance close to single-task LoRA at roughly 1.24× its inference overhead, and large gains over a basic backbone and a single jointly-trained LoRA in composite task settings.

---

## Claims and Support

**Claim 1: Sentence-level routing via mini-MLP is sufficient because within-sentence token routing is unnecessary.**
- *Status: Assumed, not demonstrated on own tasks.* The paper cites prior work (Xu et al., 2024; Lin et al., 2024b; Muqeeth et al., 2024) for this observation but provides no direct evidence on its own 26-task suite. No token-level assignment analysis, no comparison to a token-level router on the same tasks. This is the methodological foundation of the paper and it is taken on faith.

**Claim 2: DLP-LoRA achieves performance comparable to single-task LoRA across 26 tasks and 4 backbones.**
- *Status: Supported (weakly positive framing overstated).* Table 1 shows DLP-LoRA consistently 0–0.94% below single LoRA on MCQ averages. Table 2 shows mixed QA results, occasionally above, occasionally below. The paper's own numbers support "close to but slightly below on average," which is a reasonable result; the abstract framing of "matches or exceeds" is somewhat overstated.

**Claim 3: DLP-LoRA significantly improves multi-task composite performance.**
- *Status: Partially supported, but against a weak baseline.* Table 3 shows large gains over basic backbone and a single r=64 LoRA trained on all 26 tasks. The single combined LoRA is a known-weak baseline. No comparison against contemporaneous dynamic routing methods in the composite setting. The gains therefore show "specialist LoRAs + classifier > one shared LoRA," which is unsurprising, not that DLP-LoRA's fusion mechanism is an advance.

**Claim 4: Dynamic fusion (not just hard task routing) contributes to performance.**
- *Status: Undemonstrated.* There is no ablation comparing: (a) hard top-1 routing, (b) oracle task-ID routing, (c) random fusion, (d) varying top-p. The paper never isolates the contribution of multi-LoRA fusion from simple task classification. The case study is anecdotal.

**Claim 5: DLP-LoRA is more efficient than token-level MoE/gating alternatives.**
- *Status: Partially supported for latency; unfair quality-matched comparison.* Table 7 shows clear latency and memory advantages over MOLA, PESC, MoRAL, LoRA-Switch. However, Table 7 uses a different setup (7 LoRAs, ShareGPT) from the main evaluation, and no task-quality scores are reported for any baseline in that comparison. A method cannot claim efficiency superiority without demonstrating quality parity. The paper does disclose in footnote 1 that the LoRA-Switch assumption is a lower bound, which shows transparency.

**Claim 6: Qwen-2 1.5B + DLP-LoRA outperforms LLaMA-2 13B (unadapted).**
- *Status: Empirically supported but scientifically uninformative.* An adapted small model with 26 task-specific LoRAs vs. a much larger unadapted base model is not a meaningful scientific comparison. It demonstrates that task adaptation matters.

---

## Strengths

- **Genuinely lightweight and fast plugin.** Table 4 shows the mini-MLP variant adds only ~18.19% overhead over single LoRA on average, and Table 6 shows the ratio stays below 2× even with 100 LoRAs. This is a concrete and reproducible engineering result that distinguishes the approach from prior methods with much heavier routing overhead (Table 7: MOLA at 10.54×, PESC at 3.54×).

- **Multi-backbone breadth.** Evaluating across Qwen-2 1.5B, Qwen-2 7B, LLaMA-2 7B, and LLaMA-3 8B — with results averaged over 10 runs — is a systematic effort that many multi-LoRA papers skip. Performance within ~0.35% of individually fine-tuned single LoRAs across this range confirms robustness of the approach.

- **Scalability data at 50 and 100 LoRAs.** Table 6 provides a concrete scaling curve for inference time, which is practically relevant and rare in papers of this type.

---

## Weaknesses

### Fatal
*(None that fully invalidate the basic methodology; the method is functional and the efficiency results are real.)*

### Major

- **No task-performance comparison against the very methods it claims to outperform.** Table 7 compares only latency and memory against MOLA, PESC, MoRAL, and LoRA-Switch — never accuracy, BLEU, or ROUGE. This is the central evaluative gap. An efficiency-quality trade-off cannot be assessed without measuring both legs on the same tasks. A method can be fast because it does less or because it is smarter; the paper never establishes which is true relative to its competitors. This undermines the core positioning of the paper.

- **No ablation isolating dynamic fusion from hard task routing.** The paper's headline claim is "dynamic fusion," yet there is no comparison to (a) always selecting the top-1 LoRA (hard routing), (b) oracle routing to the correct task LoRA, or (c) uniform weighting of top-p LoRAs. Without this, the paper cannot support the claim that fusion, rather than classification, drives performance. If top-p almost always selects a single LoRA, then DLP-LoRA is effectively a lightweight task classifier + LoRA loader — a much more modest contribution.

- **Composite-task evaluation uses insufficiently competitive baselines.** Table 3 compares DLP-LoRA against the basic backbone and a single LoRA (r=64) jointly trained on all 26 tasks — a known weak multi-task baseline. No dynamic multi-LoRA baseline is evaluated in the composite setting. The natural comparisons would be: (1) oracle task-ID + load the single correct LoRA; (2) LoRA-Switch or another dynamic method on the same composite benchmark. Without these, Table 3 cannot distinguish DLP-LoRA's value from the trivially true claim that "many specialist LoRAs beat one generic LoRA."

- **No top-p sensitivity analysis.** The top-p threshold controls how many LoRAs are fused per sentence — the paper's central hyperparameter — yet no experiment varies it. The exact value of p used in all experiments is not stated. Readers cannot determine whether the reported results are robust or tuned to a specific p.

### Minor

- **92.34% MCQ headline figure is misleading.** The abstract reports "achieves an average accuracy of 92.34% on multiple-choice datasets" without specifying this is for LLaMA-3 8B only (Table 1). The cross-backbone average is lower. Similarly, the "92.95% relative improvement" is over the *unadapted baseline*, not over dynamic alternatives, which should be made explicit upfront.

- **Sentence-level routing premise is inherited, not validated.** The method's key justification — that token-level routing is unnecessary within a sentence — is cited from prior work but never verified on the paper's own task suite. This weakens the mechanistic rationale.

- **Classification accuracy of 98.45% reported without per-task breakdown or held-out analysis.** The routing accuracy is central to the method's integrity, but is reported as a single aggregate figure. There is no analysis of which tasks are confused or how misclassifications affect generation quality.

- **Efficiency comparison setup differs from the main evaluation.** Table 7 uses 7 LoRAs on ShareGPT (following LoRA-Switch's setup), while the main evaluation uses 26 tasks on domain-specific benchmarks. This is disclosed but limits the interpretability of the efficiency claims in the context of the main results.

### Trivial

- Eq. 4–5 notation is ambiguous: I_p appears to be a set of probability values, but Softmax is then applied to re-normalize them. The mechanism is likely correct (select above-threshold, renormalize) but could be stated more precisely.

---

## Nice-to-Haves

- Analyze how often top-p selects exactly one LoRA vs. multiple LoRAs per task, to quantify when genuine fusion occurs and characterize which task types benefit from it.
- Provide a concrete experiment on adding a new (27th) task: retraining time for the mini-MLP, and whether prior task routing is disrupted.
- Report standard deviations across the 10 runs in Tables 1–3, given that observed differences (e.g., DLP-LoRA vs. single LoRA at sub-1%) are small enough that noise matters for interpretation.
- Empirically evaluate behavior on genuinely mixed-task inputs (e.g., a math word problem in a non-English language) that require cross-domain reasoning within one sentence.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Smaller backbone vs. larger backbone" as a scientific contribution (Harsh Critic Claim 7):** The claim that comparing Qwen-2 1.5B + DLP-LoRA against unadapted LLaMA-2 13B is a meaningful scientific result was correctly identified. However, this is moved to Minor rather than removed entirely; it's kept as a framing issue rather than a standalone weakness.

- **Mathematical inconsistency in Softmax(I_p) (Harsh Critic):** After reading Eq. 4–5, the design is plausible: I_p contains the raw probability values that exceed the threshold p, and Softmax re-normalizes them as fusion weights. This is a legitimate (if under-described) design, not a methodological error. This is moved to Trivial notation clarity.

- **Efficiency comparison Table 7 as "completely unfair" (Harsh Critic):** The paper actually discloses in footnote 1 that the LoRA-Switch comparison is a conservative lower bound. The concern is real but the severity is overstated; moved to Minor.

- **"Reproducibility of exact top-p value" as a major weakness:** Per hard rules, this is a hyperparameter/implementation detail that is moved to Nice-to-Haves rather than kept as a substantive weakness.

---

## Novel Insights

The most genuinely interesting structural observation — raised across all reviewers but not fully explored — is whether DLP-LoRA is actually doing fusion or classification. The paper shows a 98.45% task-classification accuracy while achieving performance within 0.35% of single-task LoRAs; this strongly suggests the classifier is doing near-perfect task identification and the "fusion" component may be near-vacuous in practice. If true, the paper's real contribution is: *a standalone, ultralight classifier (5M parameters, <10 min to train) is sufficient to automate LoRA selection for a large task bank at negligible overhead* — an arguably cleaner and more honest framing than "dynamic multi-LoRA fusion." The authors should run the hard-routing ablation; if the gap is zero, the paper should be reframed, which would actually make it a stronger and more honest contribution.

---

## Suggestions

1. **Add task-accuracy comparisons against at least LoRA-Switch and one MoE LoRA baseline (e.g., PESC or MoRAL) on the same 26-task composite setup.** This is the single most important experiment missing.
2. **Run the top-1 hard routing ablation.** Compare DLP-LoRA (top-p) vs. DLP-LoRA (top-1, always load only the highest-probability LoRA) on all tables. This validates or refutes the fusion claim.
3. **State and sweep the top-p threshold.** Report p and show a table/figure of performance and average LoRAs selected across p ∈ {0.5, 0.7, 0.8, 0.9, 0.95}.
4. **Reframe Table 5** as an application result ("LoRA adapters enable much smaller models to punch above their weight") rather than a headline result about DLP-LoRA's superiority over larger models.
5. **Clarify the 92.34% and 92.95% figures in the abstract** with proper conditioning on backbone and comparison target.

---

## Score and Decision

**Calibration:**

- **LoRAHub** (Reject, avg. 5.3): A LoRA composition paper with missing task-performance baselines and unclear scope. DLP-LoRA has similar or larger gaps (no performance comparison with any dynamic alternative), but a broader empirical evaluation.
- **ELREA** (Accept Poster, avg. 5.8): An accepted multi-expert LoRA paper with missing baseline comparisons noted but acceptable experimental scope. ELREA has a more novel routing mechanism (gradient clustering). DLP-LoRA is methodologically simpler and has the larger missing-baseline gap.
- **MeteoRA** (Accept Poster, avg. 6.2 per Human Finder): The primary competitor cited in the paper. MeteoRA provides multi-baseline comparison including performance numbers. DLP-LoRA does not meet this bar.
- **MORE** (Reject, avg. 4.0): A weak LoRA method with marginal performance and thin experiments. DLP-LoRA is better executed but shares the missing-comparison problem in a more fundamental way.

**Assessment:** DLP-LoRA is a real, functional engineering contribution with genuine efficiency results and a broad empirical evaluation. However, the central scientific claim — that dynamic multi-LoRA *fusion* is beneficial — is unvalidated due to the missing hard-routing ablation. The composite-task evaluation relies on weak baselines, and the paper never compares task quality against any of the dynamic baselines it claims to outperform. This places it below LoRAHub (rejected) in terms of comparative rigor, even though the overall experimental scope is broader. The paper sits below the threshold for ICLR acceptance in its current form.

- **Novelty:** Incremental. Sentence-level routing has prior art (Polytropon, Flix); the main contribution is an efficient external classifier.
- **Technical soundness:** Partially sound. The efficiency mechanism is well-described; the fusion mechanism is underspecified and unablated.
- **Empirical support:** Adequate for the efficiency claim; insufficient for the fusion and multi-task adaptation claims.
- **Significance:** Moderate practical value if the engineering contribution is correctly scoped.
- **Clarity:** Reasonable overall; abstract overstates some results.

**Final Score: 4.0**

Compared to LoRAHub (rejected, ~5.3 avg.) which at least evaluates on the tasks it claims to address, DLP-LoRA's failure to compare task quality against any dynamic competitor is a more fundamental gap — pushing it below that threshold. Compared to MORE (rejected, ~4.0 avg.) which has thin experiments and marginal performance, DLP-LoRA is more carefully executed but has equivalent evaluative gaps in the critical multi-LoRA comparison dimension.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>