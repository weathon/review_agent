Now let me search for calibration papers to properly score this submission.Now I have enough information to write the final consolidated review. Let me synthesize everything.

---

## Summary

SimE proposes a simple class-incremental learning framework that combines a frozen CLIP image encoder with lightweight adapters (AdaptFormer and a proposed Multi-Adapter extension). The key design is: (1) fine-tune adapters only on Task 1's base classes, then freeze the entire encoder for all subsequent tasks, appending class prototypes into a growing cosine-similarity classifier. The paper also reports an empirical observation that within-block adapter connections do not always positively correlate with IL performance, and presents a systematic study of CLIP backbone/pre-training dataset choices.

---

## Strengths

- **Efficient and practically useful design**: SimE achieves strong performance on CIFAR-100 and TinyImageNet without a memory bank and with far fewer trainable parameters than methods like ZSCL (which fine-tunes 140M parameters). This simplicity is appealing for practitioners with strong frozen pre-trained models.
- **Systematic CLIP configuration study**: Tables 3 and 4 consolidate useful empirical results on how pre-training dataset scale (WIT-400M vs. LAION-2B) and ViT backbone size (ViT-B/16 vs. ViT-L/14) affect prototype-based CIL performance — a practically useful reference, even if unsurprising in direction.
- **Interesting adapter placement ablation (Sec. 4.3, Fig. 5)**: The finding that earlier-block adapters contribute more to downstream performance, and that more within-block adapters can degrade performance at small step settings (Table 2), is a non-trivial empirical observation even if modestly evidenced.
- **Strong empirical performance vs. comparable CLIP-based methods**: SimE (with ViT-B/16, same backbone as CLIP-based baselines) outperforms ZSCL by ~5.7% avg accuracy on CIFAR-100/10-step, which is a meaningful margin.

---

## Weaknesses

### Fatal
*None.* The paper has real issues, but they do not rise to the level of invalidating the core contribution entirely — prototype-based classification on an adapter-tuned frozen CLIP encoder is a valid and useful paradigm. However, several major issues substantially weaken the claims.

### Major

- **Low novelty: the method is essentially "SimpleCIL + adapter fine-tuning on Task 1."** The core pipeline — freeze a pre-trained model, compute per-class prototypes, use cosine-similarity for classification — is identical to the SimpleCIL paradigm (i.e., the 0-parameter baseline row in the authors' own Table 2: frozen CLIP alone achieves 79.69 avg / 70.08 last; adding adapters brings this to 85.94 / 77.10). The paper does not engage with this well-known approach or compare against it as a named baseline, leaving readers unable to assess how much of the improvement is due to the adapter fine-tuning vs. simply using a better CLIP variant. The Multi-Adapter extension provides only marginal gains (at best ~0.34% over AdaptFormer at 10 steps in Table 2, and often hurts).

- **Missing critical and obvious baselines.** Methods like SimpleCIL, EASE, DualPrompt, CODA-Prompt, and SLCA — which are standard in the PTM-based CIL space — are entirely absent. Without these comparisons, the claim of "surpassing state-of-the-art" is not credible. In particular, SimpleCIL (prototype classification on a frozen PTM) is the closest conceptual baseline and would help isolate how much Task-1 adapter fine-tuning actually contributes.

- **Parameter count claim is internally inconsistent.** The abstract and Sec. 4.2 state SimE uses "only thousands of trainable parameters." However, Table 2 explicitly lists the default SimE configuration (AdaptMLP ✓, others ✗) as **1.19 million** parameters, and the fuller Multi-Adapter variants reach 3.57M. Figure 4(a) further shows "Ours ~10 Milio" training parameters. No configuration in the paper clearly uses "thousands" of parameters; even the smallest bottleneck dimensions would yield tens of thousands, not thousands. This discrepancy — between the headline efficiency claim and the actual numbers — is a factual inconsistency that undermines a key selling point.

- **Evaluation limited to two small, low-resolution datasets.** Experiments are confined entirely to CIFAR-100 and TinyImageNet (32×64px). No ImageNet-scale evaluation, domain-shift benchmarks (ImageNet-R, ImageNet-A, VTAB, OmniBenchmark), or challenging multi-domain settings are included. This calls into question whether the benefits — particularly of the freeze-after-Task-1 strategy — generalize beyond distributions close to CLIP's own pre-training data.

- **The "freeze after Task 1" design is presented ambiguously and its limitations are not acknowledged.** The paper clearly states (Fig. 1) that all parameters are frozen after Task 1 and only prototypes are computed for subsequent tasks. This is a legitimate and simple design, but the paper does not: (a) discuss what happens under domain shift between Task 1 and later tasks, (b) compare to the obvious alternative of adapting incrementally for each task, or (c) explicitly frame this as a trade-off. The presented method is, in practice, nearest-centroid classification on a one-time-adapted CLIP encoder — this should be acknowledged as a design constraint, not glossed over.

### Minor

- **No statistical significance testing.** Table 2's "remarkable phenomenon" about adapter connections shows differences of 0.01–0.66%, yet no standard deviations or multi-seed results are reported. At these effect sizes, results may be within random noise from initialization or class ordering. This weakens the empirical claim.

- **Confounded ablations on CLIP components (Sec. 4.4).** The comparisons across CLIP pre-training datasets and ViT backbone sizes (Tables 3–4) confirm the well-known result that larger models and larger pre-training data give better features. Because the encoder is entirely frozen after Task 1, these ablations measure static representation quality — not anything CIL-specific. The claim that this is a "systematic study under IL" is overstated.

- **Writing and notation are sometimes unclear.** Eq. (3) presents the encoder output as a sum over blocks rather than the standard sequential ViT composition, which is confusing. Eq. (6) for Multi-Adapter is hard to parse. The paper would benefit from cleaner formalization that maps to standard ViT notation.

### Trivial

- The distinction between "Avg" and "Last" accuracy could be better explained upfront for readers unfamiliar with CIL evaluation conventions.

---

## Nice-to-Haves

- Compare against CLIP zero-shot directly (i.e., no fine-tuning, just text-name class prototypes) to establish the true benefit of adapter fine-tuning on Task 1.
- Analyze robustness to Task 1 choice: what happens if fewer or more base classes are used, or if the task order is shuffled?
- Provide a representational analysis (CKA, t-SNE) of what the Task-1 adapter actually learns and why freezing it suffices for later classes.
- Multi-seed results and confidence intervals, especially for Table 2 where differences are <0.5%.
- Evaluate on at least one large-scale benchmark (e.g., ImageNet-100 or ImageNet-R) to establish broader applicability.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**Harsh critic Issue 1 (cross-paradigm comparison as entirely invalid)**: Partially removed. The harsh critic argues the entire comparison to CLIP-based baselines is unfair. However, the SimE (Ours) row in Table 1 uses the same ViT-B/16 CLIP checkpoint as CoOP, Continual-CLIP, LwF-VKD, and ZSCL — so the comparison is not fundamentally unfair at the backbone level. The comparison to from-scratch methods (UCIR, PASS, DyTox, DER) is standard practice to demonstrate the gap attributable to pre-training. The real issue is missing baselines (SimpleCIL), not an invalid experimental design. Downgraded to the missing-baselines weakness above.

**Harsh critic Issue 2 (freeze-after-Task-1 as "not CIL")**: Partially removed as framed. The design is a legitimate engineering choice for memory-free prototype-based CIL. It is not conceptually invalid to train adapters on Task 1 and use a growing prototype bank thereafter — this is simply a different (simpler) instantiation of CIL. However, the critique that the paper does not acknowledge this limitation is valid and retained above.

**Harsh critic Issue 6 (backbone/dataset ablations are not CIL-specific)**: Retained as a minor weakness but not fatal — the paper explicitly frames these as "systematic study to enhance utilization of zero-shot capabilities of CLIP," and practitioners value this even if it does not test continual-learning mechanisms per se.

**Harsh critic notation complaints (Eq. 3, Eq. 6)**: Downgraded to trivial. The equations are unusual but interpretable, and the parser artifacts in the PDF do not reflect actual paper quality.

**Human Finder's data contamination concern (CIFAR-100 in CLIP training set)**: This is a generic concern affecting all CLIP-based CIL papers equally; it is not specific to SimE's design choice and is not actionable for this paper alone. Removed.

---

## Novel Insights

The most genuinely interesting empirical finding is the asymmetric effect of within-block adapter placement depending on task granularity: at 10-step (coarse) task splits, adding within-block adapters (AdaptAttn, AdaptAll) marginally degrades performance, while at 50-step (fine-grained, many small increments) they help. This aligns intuitively with the idea that coarse task boundaries give larger distributional shifts per step, where a simpler (less over-fit) adapter generalizes better. This observation is modest in effect size but could motivate adapter design guidelines in the broader PEFT-for-continual-learning literature. It deserves more careful analysis (multiple seeds, mechanistic probing) before it can be elevated to a principle.

---

## Suggestions

1. **Add SimpleCIL as a named baseline** (or acknowledge the related design explicitly) and isolate the contribution of Task-1 adapter fine-tuning vs. pure frozen CLIP prototypes. The 0-param row in Table 2 is functionally SimpleCIL but is not labeled as such.
2. **Fix the parameter count discrepancy** — report a single consistent number per configuration across all figures and tables, and remove the "thousands" claim unless a specific configuration truly achieves that.
3. **Expand to at least one large-scale benchmark** (ImageNet-R, ImageNet-A, or VTAB) to establish generalizability.
4. **Add multi-seed results** for Table 2 adapter ablations to support or temper the "remarkable phenomenon" claim.
5. **Reframe the paper's positioning** more honestly: SimE is a "one-time adapter fine-tuning + prototype expansion" CIL method, not a general incremental representation learner. Framing it this way is honest and still compelling from an efficiency standpoint.

---

## Score and Decision

**Calibration anchors:**

- **mrRbIcyouU** (SimpleCIL/APER — same core idea, richer analysis, new benchmarks): Scores 6, 3, 5, 5 → average ~5; this is a more systematic and complete paper than SimE yet still rejected by most reviewers.
- **sb7qHFYwBc** (C-CLIP — multimodal CIL with new benchmark, comprehensive framework): Accepted (Poster), scores 6, 6, 8, 6 → average 6.5. Much higher novelty than SimE.
- **k9NYnsC4Mq** (PROOF — VLM-based CIL with cross-modal fusion): Rejected, scores 5, 6, 6. Genuinely more novel than SimE but still rejected partly due to narrow evaluation.
- **04TRw4pYSV** (ModalPrompt — rejected, 3, 3, 3, 5): Weaker presentation, similar issues with missing baselines.

SimE sits **below APER** (which already had a mixed reception despite much richer experiments) and **below PROOF** (which had a more technically novel contribution). SimE shares PROOF's weaknesses (limited benchmark scope, missing key baselines, no standard deviations) without PROOF's novelty of cross-modal fusion. The parameter count inconsistency is an additional specific problem. The paper is closer to the 4–4.5 range.

**Axes summary:**
- *Originality*: Low — combination of known techniques (CLIP + AdaptFormer + prototype classification), no clearly novel algorithmic contribution.
- *Importance of research question*: Moderate — efficient memory-free CIL is relevant, but the specific design is narrow.
- *Claims well-supported*: Partially — good on CIFAR-100/TinyImageNet, unsupported at scale; parameter efficiency claim is inconsistent.
- *Soundness of experiments*: Weak — missing key baselines, no error bars, limited scope.
- *Clarity of writing*: Fair — main ideas clear but notation inconsistencies and the "thousands" claim are problems.
- *Value to community*: Moderate — the CLIP backbone study and adapter placement analysis are useful; the SimE framework itself is straightforward but practical.

**Final score: 4.0 (Reject)**

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>