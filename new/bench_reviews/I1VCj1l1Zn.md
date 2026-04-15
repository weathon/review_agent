## Summary
This paper proposes DLP-LoRA, a lightweight external router for a bank of task-specific LoRAs: a 5M-parameter sentence classifier predicts task probabilities from the input sentence, then uses top-\(p\) selection and weighted fusion of selected LoRAs at sentence boundaries. Empirically, across 26 tasks and four backbones, the method achieves near-parity with manually selected single-task LoRAs while preserving relatively low latency, and it substantially outperforms unfine-tuned base models or a single LoRA trained jointly on all tasks in the composite setting.

## Strengths
- **The paper identifies and operationalizes a concrete systems insight:** instead of routing per token inside every transformer layer, it performs routing once per sentence via an external plugin. This is a specific design choice, not a generic efficiency claim, and it is well aligned with the paper’s reported runtime results in Table 4 and the implementation discussion in Sec. 3.3.
- **Efficiency is the strongest part of the submission.** The paper supports the claim that DLP-LoRA keeps inference overhead modest relative to single-LoRA inference: Table 4 reports mini-MLP-based DLP-LoRA at 1.11–1.60× the base model, versus 1.00–1.15× for single LoRA, and Sec. 3.3/4.2 consistently emphasizes this tradeoff.
- **The method preserves most of the accuracy of task-specific LoRAs despite automatic routing.** Tables 1 and 2 show that DLP-LoRA is generally very close to single-LoRA performance across 17 MCQ and 9 QA tasks over four backbones, with average deltas typically under 1% relative on MCQ and mixed but small changes on QA.
- **The composite-task result demonstrates a practically meaningful point:** a bank of specialist LoRAs plus lightweight routing is much stronger than either an unfine-tuned base model or a single LoRA trained jointly on all 26 tasks (Table 3). This does not fully validate the fusion mechanism by itself, but it does show the practical value of external routing to specialized adapters.
- **The evaluation breadth is a real asset here.** The paper does not only test one backbone or one task family; it reports results on Qwen-2 1.5B/7B, LLaMA-2 7B, and LLaMA-3 8B, spanning MCQ-style classification and generation-style QA tasks.

## Weaknesses

###: Fatal
- **The paper does not isolate whether “fusion” is actually responsible for the gains.** The title and method center on dynamic LoRA fusion, but the experiments never compare against the most important ablations: classifier-driven top-1 routing without fusion, hard selection of a single LoRA, or oracle task-ID selection of the correct single LoRA. As written, the evidence is equally consistent with a simpler story: the mini-MLP is just a strong closed-set task classifier that picks the right specialist adapter. This is a central evidential gap because the core claimed contribution is fusion, not merely routing.

### Major:
- **The broader framing overstates what is actually demonstrated.** The method in Sec. 3.1 is explicitly a classifier over a fixed set of known tasks, and Sec. 4.1 evaluates on exactly those known tasks. That supports “closed-set routing among known task-specific LoRAs,” but not a broad claim of general “dynamic fusion based on contextual inputs” in an open-ended sense. The paper should present this more narrowly and accurately.
- **The composite-task evidence in Table 3 does not disentangle where the gains come from.** DLP-LoRA is compared mainly against (i) an unfine-tuned base model and (ii) a single LoRA trained jointly on all 26 tasks. Those are informative baselines, but they do not test the key question: among a bank of specialist LoRAs, is top-\(p\) fusion better than simply selecting one? Without top-1 and oracle-routing baselines, the composite-task improvement mostly establishes the value of specialization, not the value of fusion.
- **Several claims of superiority over single LoRA are overstated relative to the tables.** The actual pattern in Tables 1–2 is near-parity, not consistent improvement. For MCQ in Table 1, DLP-LoRA is slightly below single LoRA on average for three of four backbones and equal on one; on QA metrics in Table 2, averages are mixed with small gains/losses. The paper should claim “comparable to single LoRA” rather than implying robust outperformance.
- **The key methodological choice, top-\(p\) fusion, is insufficiently analyzed.** Sec. 3.2 presents top-\(p\) selection as central, but the paper provides no sensitivity study over \(p\), no reporting of how many LoRAs are typically activated, and no evidence about when multi-LoRA fusion helps versus hurts. This omission is especially problematic because the claimed contribution hinges on dynamic fusion rather than single-adapter selection.
- **The evidence for sentence-level sufficiency versus token-level routing is indirect.** The paper cites prior observations that tokens in a sentence often share task identity, but it does not directly validate this on its own benchmark or compare against a matched token-level router on the same setup. Since this assumption motivates the whole architecture, stronger direct evidence would materially strengthen the paper.

### Minor
- **Some headline numbers and wording are confusing or inconsistent.** For example, the abstract reports “92.34%” average MCQ accuracy, while Table 1 shows backbone-specific DLP-LoRA averages of 89.89, 92.89, 90.65, and 95.93. It is unclear what aggregation yields 92.34, and the paper should define it explicitly if retained.
- **Tiny average deltas are reported without uncertainty estimates.** The paper says results are averaged over 10 runs, but Tables 1–3 omit standard deviations/confidence intervals. Given that many differences from single-LoRA baselines are very small, uncertainty reporting matters for interpreting whether those differences are meaningful.
- **The use of an ALBERT tokenizer for the router, while the backbones use their own tokenizers, is under-motivated.** Sec. 3.1 states this choice, but the paper provides no ablation or discussion of whether tokenizer mismatch affects routing accuracy or robustness.
- **The practical story around adding new tasks is somewhat oversold.** The paper does support that the router is lightweight and trains quickly, but “easily adaptable to new domains” is only partially demonstrated. In practice, adding a new task appears to require both a new LoRA and retraining or expanding the classifier; the paper does not evaluate that workflow.
- **Some task-level failures deserve discussion.** There are visible per-task drops versus single LoRA in Tables 1–2 (e.g., TopChat and several MCQ datasets for some backbones), but the paper largely highlights only wins and averages.

### Trivial
- **The top-\(p\) notation/description could be clearer.** Eq. (4) reads more like thresholding class probabilities than standard nucleus sampling; clarifying this would help reproducibility.
- **Sentence-boundary handling is not fully specified.** Since the router activates “once the first token of every new sentence is generated,” more detail on how sentence boundaries are detected during autoregressive generation would improve methodological clarity.

## Nice-to-Haves
- Add direct ablations for **top-1 routing**, **oracle task-ID routing**, and **top-\(p\) fusion** to isolate the benefit of fusion.
- Provide a **sensitivity analysis over \(p\)**, including average number of active LoRAs, latency, and accuracy/quality.
- Report a **confusion matrix or per-task routing accuracy** for the mini-MLP and analyze how misrouting propagates to downstream failures.
- Include a more direct matched comparison to the specific token-level dynamic routing baseline used to motivate the work, under the same backbones / LoRA pool / hardware.
- Clarify the aggregate-number computations in the abstract and provide **variance over the 10 runs**.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The comparison to single composite-26 LoRA is unfair because DLP-LoRA has more parameters.”** This is only partially useful as stated. The paper is explicitly comparing a bank of per-task LoRAs plus a router against a single multitask LoRA to make a specialization point; that comparison is not invalid. The real issue is not “unfairness” per se, but that this does not isolate the value of fusion over simpler routing.
- **“Missing comparison to Meteora means the paper’s efficiency claim is unsubstantiated.”** Overstated. The paper does provide some dynamic-LoRA efficiency comparison in Table 7, so the efficiency claim is not unsubstantiated. The valid criticism is narrower: the most directly motivated token-level baseline is not evaluated under fully matched conditions on the main benchmark.
- **“The method must handle unseen tasks / zero-shot new tasks.”** This is mostly scope creep. The paper studies routing among a fixed bank of known task-specific LoRAs, and should be judged primarily on that setting. It is fair to say the framing should be narrower and that extensibility is not demonstrated, but not fair to fault it for not solving open-world unseen-task adaptation.
- **“Doubts about cited baselines or assumptions about release/existence.”** Per instruction, such points are removed.

## Novel Insights
The paper is stronger as a systems paper than as a method paper. Its real demonstrated contribution is not that top-\(p\) multi-LoRA fusion is uniquely effective, but that **external sentence-level routing to specialist adapters can recover most single-task-LoRA quality at much lower overhead than heavy in-model routing schemes**. Put differently: the submission’s evidence currently supports a valuable deployment lesson—*move routing outside the transformer and do it infrequently*—more strongly than it supports the headline claim that dynamic fusion itself is the source of the gains. Reframing around that would make the paper more honest and, paradoxically, more convincing.

## Suggestions
- Add the crucial ablations: **classifier → top-1 single LoRA**, **oracle task-ID → correct single LoRA**, and **classifier → top-\(p\) fusion**.
- Report how often more than one LoRA is selected, and on which tasks multi-LoRA fusion actually helps.
- Tone down the framing from broad “context-aware dynamic fusion” to **closed-set sentence-level routing among known tasks**, unless broader evidence is added.
- Add uncertainty measures for Tables 1–3, since many differences are small.
- Clarify the \(92.34\%\) abstract number and ensure all headline statistics directly map to visible tables.
- Discuss failure cases where DLP-LoRA underperforms the corresponding single-task LoRA, rather than only emphasizing wins.
- If space permits, include a matched comparison to a token-level routing method on at least a representative subset of tasks/backbones.

## Score and Decision
**Novelty:** Moderate. Sentence-level external routing for a LoRA bank is a sensible and practically useful twist, but the paper does not convincingly establish that the novel *fusion* component is essential.  
**Technical soundness:** Mixed. The core implementation idea is plausible and the runtime evidence is credible, but the main scientific claim is under-supported because the key ablations are missing.  
**Empirical support:** Good breadth, but incomplete causally. The paper evaluates many tasks/backbones, yet the most important comparisons for validating fusion are absent.  
**Significance:** Moderate to high if framed as an efficiency-oriented routing design; lower if framed as a strong new fusion mechanism.  
**Clarity:** Generally understandable, though some claims are overstated and a few key details/statistics are ambiguous.

**Calibration against similar papers:**
- **Mixture of LoRA Experts (uWvKBCYh4S.md)**: marginal accept/reject band (3/5/6/6). Like this paper, it had a plausible LoRA-composition idea with some empirical support but concerns about limited or only marginal improvements. DLP-LoRA is somewhat stronger on efficiency and task breadth, but weaker on isolating its claimed fusion mechanism.
- **MORE (LWvgajBmNH.md)**: reject-leaning (3/3/5/5). Similar pattern: mixture-of-LoRA idea, some gains, but concerns about novelty/justification and incomplete evaluation. DLP-LoRA is stronger empirically than MORE, especially on runtime and breadth.
- **LoraHub (w8eCnnq57m.md)**: borderline reject/accept (5/6/5). LoraHub’s claims were more tightly matched to its evidence, even if limited in other ways. DLP-LoRA has broader experiments, but its central fusion claim is less cleanly validated.
- **UnoLoRA (49ti6LOUw5.md)**: strong reject (3/3/3/3). DLP-LoRA is clearly better than this calibration point: more coherent motivation, broader experiments, and a practically meaningful efficiency result.
- **CITER (J2FyEVg8HR.md)**: mixed borderline (5/6/5/3). Like CITER, this paper has a real efficiency idea but leaves important overhead/mechanism questions unresolved. DLP-LoRA sits in a similar borderline region, though for different reasons.

Overall, I view this as a **borderline paper with a real practical idea and good runtime evidence, but with a central evidential hole that matters too much for ICLR acceptance**. If the main claim were reframed around efficient closed-set routing to specialist LoRAs, I would be more positive. As submitted, the title and strongest claims overreach what the experiments prove.

**Score: 4.8 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>4.8</pineapple>
MY FINAL DECISION: <orange>Reject</orange>