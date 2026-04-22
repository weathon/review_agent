Now let me do calibration searches concurrently.Now I have enough to write the final review.

---

## Summary
MM-R³ introduces a three-task benchmark (Question Rephrasing, Image Restyling, Context Reasoning) for evaluating the consistency of nine MLLMs (6 open-source + 3 proprietary) under semantically equivalent but surface-form-varied inputs. The paper demonstrates empirically that accuracy and consistency are largely decoupled and that visual perturbations degrade consistency more than linguistic ones. A lightweight Bi-LSTM adapter is proposed to improve consistency of BLIP-2 and LLaVA 1.5M.

## Strengths
- **Accuracy–consistency decoupling is concrete and non-obvious** (Tables 2–4): Gemini achieves 55.31% accuracy on Context Reasoning but only 45.22% Con; BLIP-2 achieves 27.91% accuracy but 82.44% Con. Qwen-VL-Chat outperforms GPT-4V on consistency in Question Rephrasing despite lower accuracy. These specific model-level divergences are informative and reproducible.
- **Visual > linguistic perturbation finding** (Section 4.3.2): The benchmark reveals that image restyling and masking cause substantially larger consistency drops than question rephrasing, a new and actionable insight about MLLM brittleness to visual domain shifts.
- **Broad model coverage**: Nine models (BLIP-2, mPLUG-Owl2, LLaVA 1.5M, MoE-LLaVA, Qwen-VL-Chat, BLIP-3, GPT-4V, GPT-4o, Gemini) under a unified evaluation is a genuine strength.
- **Model-size / consistency finding** (Section 4.3.4): Consistency does not scale with model size the way accuracy does—for some models the larger variant is actually *less* consistent on Context Reasoning. This is a novel, practically important observation.
- **Human validation of semantic equivalence** (Section 3.2): 92% for rephrasing, 86% for restyled images confirms perturbations preserve semantic content, so observed consistency failures are genuine model failures, not annotation noise.

## Weaknesses

### Fatal
None.

### Major

- **Adapter lacks a fine-tuning ablation baseline (Section 5).** The paper's own Section 5.2 acknowledges the improvement for Image Restyling and Context Reasoning is partly because "original MLLMs are not trained on data of this form. Hence, the introduced adapter can both fine-tune performance on the new data *and* improve consistency." Training with standard CrossEntropyLoss on same-distribution data will produce gains via domain adaptation alone; the Bi-LSTM+prefix architecture's specific contribution to *consistency* (vs. simply fitting the new domain) is never isolated. There is no comparison against a vanilla LoRA or MLP adapter trained on identical data with the same loss. The conclusion that the adapter "help[s] MLLMs overcome variability… by making them invariant to surface form variability" is a mechanistic claim unsupported by the experimental design. The adapter's loss function also provides no formal invariance guarantee—CrossEntropyLoss maximizes accuracy on training examples, not input-invariance.

- **Image Restyling accuracy metric is fundamentally mismatched.** Section 3.2 instructs MLLMs to "describe the depicted places in two sentences," but accuracy is computed via substring match against the original dataset's single-word category labels (e.g., "kindergarten"). A model producing "a colorful children's playroom filled with toys and small furniture" correctly describes a kindergarten but scores 0. This explains uniformly poor accuracy figures (8–25% in Table 3) across all nine models and makes the accuracy column for this task uninterpretable as a measure of visual understanding. Note that this does *not* invalidate the consistency metrics (Con and S_C), which are computed pairwise between model responses without reference to GT labels; but it renders cross-model accuracy comparisons on this task meaningless.

- **Temperature confound undermines Sampling vs. All cross-model comparisons.** Section 4.3.1 acknowledges that BLIP-2 and LLaVA 1.5M run at temperature=0, producing 100% sampling consistency trivially (identical input → identical deterministic output). The paper then interprets the large gap between 100% sampling Con and ~48% All Con as evidence these models are "sensitive to input prompts," but the gap is mechanically inflated by the deterministic setting: any prompt perturbation forces a completely new deterministic trajectory. Other models run at higher temperatures, producing noisier sampling scores but smoother All scores. Tables 2–4 mix temperature=0 and temperature>0 models without controlling for this, making the Sampling/All comparison framework misleading for the two highest-profile models in the study. Temperatures used during All evaluation are not disclosed in the main tables.

### Minor

- **Rephrasing prompt includes the ground-truth answer** (Section 3.2): The GPT-3.5 prompt is *"…to which the answer would be (Answer)."* Providing the target answer to the rephrasing model may produce answer-leading rephrasings rather than naturalistic paraphrases. This is not acknowledged as a limitation and could affect the difficulty (and validity) of the rephrasing test.

- **Uniform 0.7 consistency threshold applied across tasks with very different output lengths.** Context Reasoning produces short noun-phrase answers while Image Restyling produces two-sentence descriptions. Pairwise sentence similarity between short phrases and between full descriptive paragraphs behaves quite differently, yet a single 0.7 threshold governs both. No sensitivity analysis on this threshold is provided.

- **Table 6 vs. Table 2 numerical discrepancy.** The "Ori." LLaVA 1.5M baseline in Table 6 reports substantially different numbers from Table 2 for Question Rephrasing (e.g., Acc=26.9, Con=32.5 vs. Acc=31.01, Con=48.47). The most likely explanation is that Table 6 uses the 7B variant while Table 2 uses the 13B—but this is never stated explicitly, which creates confusion about what is being compared.

- **"Abductive reasoning" labeling.** The paper describes Context Reasoning as "abductive reasoning" and "perceptual inference," but the task is asking "What kind of object is in the masked region?" given full surrounding context—this is occluded object identification, not formal abductive inference. Mislabeling the task inflates conceptual novelty claims.

### Trivial
- The abstract reports "absolute improvements of 5.7% and 12.5%, on average on BLIP-2 and LLaVA 1.5M in terms of consistency" without identifying which metric (S_C or Con) is cited, and without noting that LLaVA S_C on Image Restyling actually *decreases* (56.9→52.6) — this decline should be disclosed.

## Nice-to-Haves
- A consistency-specific training objective (e.g., contrastive loss requiring similar prefix representations across semantically equivalent inputs) would make Section 5 a genuine methodological contribution rather than an engineering demonstration.
- Out-of-distribution evaluation of the adapter (different style-transfer methods, adversarial paraphrases not seen during training) to test whether the adapter generalizes or simply memorizes the training distribution.
- A human-performance baseline for the actual tasks (Con/S_C numbers for human respondents), not just the semantic-equivalence validation, to make the model–human gap concrete.
- Analysis separating "consistent and correct" from "consistent and incorrect" responses after adapter training, to check whether adapter gains come from genuine invariance or mode collapse.
- Controlled temperature comparison: re-running all models at matched temperature to disentangle temperature effects from intrinsic consistency properties.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic: "The adapter cannot be attributed to its design because CrossEntropyLoss provides no formal consistency guarantee."** — PARTIALLY RETAINED. The core validity of this concern (missing ablation baseline) is kept as a Major weakness. The sub-claim about formal guarantees is weakened to a Minor point since most applied adapter papers do not provide theoretical guarantees.
- **Harsh Critic: "Section 3.2 Context Reasoning is mislabeled as abductive reasoning and is just occluded recognition."** — RETAINED but downgraded to Trivial/Minor. The task is indeed not formal abductive reasoning, but this is a presentation issue, not a validity-undermining error.
- **Strength Finder: "Clear task design with complementary perturbation strategies."** — DROPPED as generic (no specific table/equation citation, one-size-fits-all praise).
- **Strength Finder: "LLaVA S_C Context Reasoning improves from 25.9→62.0."** — KEPT in context but the +36.1 point gain is partly confounded by domain adaptation (admitted by authors), so it is presented with that caveat.

## Novel Insights
The most genuinely novel observation is the asymmetry between visual and linguistic perturbation sensitivity: visual domain changes (style transfer, masking) degrade MLLM consistency substantially more than linguistic rephrasing, suggesting that the vision encoder—not the language decoder—is the primary consistency bottleneck in current MLLMs. The finding that model size does not improve consistency (and may worsen it for certain tasks) is a second non-obvious result with practical implications for deployment. These two findings together constitute a meaningful empirical contribution even if the adapter methodology has gaps.

## Suggestions
1. Add a simple fine-tuning baseline (LoRA or continued pre-training) trained on identical data with the same CrossEntropyLoss, and compare it to the Bi-LSTM+prefix adapter in Table 6 to isolate the architectural contribution.
2. Fix or replace the Image Restyling accuracy metric to align with the free-form two-sentence task (e.g., use S_GT with GT descriptions, or a VQA-style metric).
3. Disclose and standardize temperature settings across all models in Tables 2–4; run a controlled matched-temperature comparison.
4. Add a brief ablation on the Con threshold (e.g., 0.5, 0.7, 0.9) to show robustness of findings to threshold choice.
5. Report model variant (7B vs 13B) explicitly in Table 6 to resolve the discrepancy with Table 2.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Human Score | Comparison to MM-R³ |
|------|----------------|---------------------|
| `/human_reviews/RTHbao4Mib.md` (WDCT LLM consistency benchmark) | 6.25, Accepted Poster | Most similar in spirit — consistency benchmark with empirical findings and an improvement strategy. WDCT has smaller dataset and no ablation studies but fewer methodological flaws (no metric mismatch, no temperature confound). MM-R³ is broader in scope but more flawed in execution. |
| `/human_reviews/mMaQvkMzDi.md` (Beyond task performance, LMM evaluation) | 5.50, Accepted Poster | Empirical LMM analysis with ICL improvement; similarly lacks strong ablations, outdated model set. Comparable in rigor to MM-R³. |
| `/human_reviews/ZuYvrjh2od.md` (ReForm-Eval) | 5.00, Rejected | Larger benchmark but unfair comparisons and limited novelty. MM-R³ has more novel findings but comparably serious methodological issues. |
| `/human_reviews/X1OfiRYCLn.md` (VL Bootstrapping dynamic eval) | 7.50, Accepted Oral | Much stronger: more rigorous protocol, properly calibrated metrics. MM-R³ falls clearly below this. |
| `/human_reviews/g9diuvxN6D.md` (LLM paraphrasing robustness) | 7.50, Accepted Spotlight | Clean methodology with no metric mismatch or temperature confound. MM-R³ is less rigorous. |
| `/human_reviews/gNoqEdT2wO.md` (Multimodal CIL benchmark) | 2.33, Withdrawn | Weak benchmark with no novel findings. MM-R³ has clearly more substantive empirical contributions. |
| `/human_reviews/MGceYYNvXp.md` (MPG LLM benchmark aggregation) | 1.50, Rejected | Trivial contribution; no comparison to MM-R³ which has genuine novel findings. |

**Reasoning:** The benchmark contribution (accuracy–consistency decoupling, visual > linguistic perturbation effect, model-size finding) is genuine and specific. However, three substantive methodological issues—the metric mismatch for Image Restyling accuracy, the missing adapter ablation baseline, and the temperature confound in cross-model comparisons—collectively undermine roughly half the paper's claimed contributions. The core consistency analysis (Con and S_C) remains valid. MM-R³ sits below RTHbao4Mib (6.25) due to more severe methodological flaws, closer to ZuYvrjh2od (5.0), but above the withdrawn papers (2–2.5). Placing at **4.5**: the paper has real ideas but the methodological gaps are too significant to recommend acceptance without revisions.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>