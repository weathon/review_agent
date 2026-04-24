Now let me do calibration searches to score this paper appropriately.Now I have sufficient context to write the final review.

---

## Summary

SimE is a class-incremental learning (CIL) framework that combines CLIP's image encoder with lightweight adapter modules. The key design is: fine-tune adapters on Task 1 only, then freeze all weights and compute class prototypes using the frozen encoder for all subsequent tasks. The paper also explores "Multi-Adapter" configurations (placing adapters at different positions within transformer blocks) and conducts a systematic study of how CLIP backbone size and pre-training dataset affect CIL performance.

---

## Strengths

- **Systematic CLIP pre-training study (Tables 3–4):** The paper provides a clean, principled comparison across five pre-training datasets (WIT-400M, LAION-400M, LAION-2B, DataComp-1B, CommonPool-1B) and four backbone sizes (ViT-B/16, ViT-B/32, ViT-L/14, ViT-L/14-336px) in a CIL setting. This is underexplored in the literature and yields actionable recommendations (LAION-2B + ViT-L/14 is best). This is the paper's most credible empirical contribution.

- **Genuine efficiency advantage over Continual-CLIP and ZSCL at matched backbone:** Table 2 (ViT-B/16 with WIT-400M, matching baselines) shows SimE achieving 85.94% Avg on CIFAR-100 10 steps with only 1.19M trainable parameters, matching ZSCL (85.94%) while ZSCL requires ~140M parameters. The improvement over Continual-CLIP (78.81%) is substantial (+7.13%). This efficiency claim is real and meaningful when backbone-matched.

- **Transparency of "train once, freeze forever" design:** Figure 1 and its caption are explicit that adapter fine-tuning only occurs on Task 1, with all weights frozen for Tasks 2 through T. This design choice is documented, not hidden.

---

## Weaknesses

### Fatal
None that completely invalidate the core efficiency contribution at matched backbone.

### Major

- **Backbone mismatch in the primary results table (Table 1):** The headline SimE result of 91.66% Avg on CIFAR-100 10 steps does not correspond to the ViT-B/16 setting claimed for all non-† methods. Table 4 shows that ViT-B/16 gives 85.94% and ViT-L/14 (WIT-400M alone) gives 88.79%; reaching 91.66% requires the ViT-L/14 + LAION-2B combination. Figure 3 explicitly confirms: "The result of 'Ours' in left [CIFAR-100 10 steps, showing the 91+% performance] is based on ViT-L/14." Meanwhile, the SimE(Ours)† row, which by the table note should hold the ViT-L/14 LAION-2B numbers, is completely empty. This creates an actively misleading table: the headline number uses a ~4× larger backbone than all CLIP baselines (ZSCL, LwF-VKD, Continual-CLIP, CoOP), but is presented in the same row format as if it is a ViT-B/16 result. The abstract claim that SimE "outperforms CLIP-based methods by 5.3% on CIFAR-100" is only true at this mismatched backbone. At matched ViT-B/16, SimE and ZSCL tie at 85.94%.

- **The "train only on Task 1" design sidesteps catastrophic forgetting entirely:** The paper's framework (Panel B/C of Figure 1) fine-tunes adapters only on Task 1 and never updates any weights again. For Tasks 2 through T, the encoder is frozen and only prototype vectors are appended. This means SimE has zero plasticity after Task 1 and cannot suffer catastrophic forgetting. Compared baselines (ZSCL, LwF-VKD, Continual-CLIP) actually update across the task sequence and must balance plasticity/stability. The paper frames this design as "preserving pre-trained prior knowledge," which is a re-framing of zero plasticity. The comparison is legitimate if the paper is explicit that it represents a different operating point (efficiency via reduced plasticity), but the paper presents it as if SimE has "solved" catastrophic forgetting while doing less of what the baselines do. A clear discussion of this trade-off is absent.

- **"Fren-time" baseline is uncited and unexplained:** Table 1 and Figure 3 include "Fren-time" as a competitive baseline (79.35% Avg on CIFAR-100 10 steps, higher than Continual-CLIP), but the method is never cited, named in full, or described anywhere in the paper. Readers cannot evaluate what this comparison represents.

### Minor

- **The "remarkable phenomenon" of multi-adapter connections is weakly supported:** The observed effect in Table 2—that adding more intra-block adapters hurts at 10 steps (85.94% → 85.54%, a 0.4% difference) but slightly helps at 50 steps (84.09% → 85.00%, 0.91%)—is presented as a "significant phenomenon" and a primary contribution bullet. No statistical testing or multi-run variance is reported. For CIFAR-100 CIL, differences of 0.4–0.9% are within expected run-to-run variance. The paper also offers no mechanistic explanation for why this pattern would exist. Claiming this as a key insight without statistical support is an overstatement.

- **Equation (3) uses a sum over blocks that contradicts standard sequential ViT composition:** The equation $E(\mathbf{x}) = \sum_i^B (g_i(\phi_i, f_i(\theta_i, \mathbf{x}_i)) + d_i(\tilde{\eta}_i, \mathbf{x}_i))$ implies the outputs of all blocks are summed, whereas standard ViT composes blocks sequentially (the output of block $i$ is the input to block $i+1$). The notation $\mathbf{x}_i$ for $i > 0$ is not defined—it should be the output of the previous block. The underlying implementation is likely correct, but the formalization is incorrect and may confuse readers.

- **Abstract claims no memory bank, but Figure 4(c) shows SimE with memory bank size ~100:** The abstract states SimE uses "no memory bank." Figure 4(c) plots SimE at a memory bank size of approximately 100. The likely explanation is that SimE stores class prototypes (one per class, 100 classes for CIFAR-100), which are not an episodic replay buffer. This distinction should be stated clearly—the current presentation creates an apparent self-contradiction.

### Trivial

- The training objective for the adapter on Task 1 (cross-entropy? contrastive?) is never stated in the main text (referenced to Appendix A in Section 3.2).

---

## Nice-to-Haves

- A version of Table 1 where all rows use the same backbone (one ViT-B/16 table, one ViT-L/14 table) would make comparisons interpretable and show the effect of scale fairly.
- A baseline of "ViT-L/14 + frozen CLIP + prototypes (no adapter, no fine-tuning)" would isolate what the adapter contributes beyond simply having a larger backbone.
- Statistical variance across multiple runs for Table 2, given that the "remarkable phenomenon" hinges on sub-1% differences.
- A mechanistic analysis of why more intra-block adapters might not help at few incremental steps (e.g., feature similarity analysis, gradient norm analysis during Task 1 fine-tuning).
- A per-task accuracy curve (not just cumulative avg/last) to show whether the method degrades specifically on earlier tasks or maintains them trivially due to freezing.

---

## Removed Points

These points are flagged to be removed, treat them with caution.

1. **Harsh Critic: "Abstract directly contradicts itself on memory bank."** Removed as a standalone "contradiction" criticism. The paper's "no memory bank" language clearly refers to an episodic replay buffer; storing prototypes (class means) is a widely understood alternative. However, the lack of any clarifying language is moved to Minor.

2. **Strength Finder: "Strong empirical gains over SOTA with orders-of-magnitude fewer parameters."** Removed as stated because it relies on the backbone-mismatched Table 1 comparison (91.66% vs 85.94%). The more defensible version—efficiency gains at matched ViT-B/16—is preserved in Strengths.

3. **Strength Finder: "Novel and counterintuitive finding about adapter placement."** Downgraded—the effect size is <1% without statistical testing, and is preserved as a Minor weakness rather than a strength.

4. **Strength Finder: "Consistent improvements across varying granularity."** Removed—this repeats the backbone-mismatched comparison issue across settings.

5. **Harsh Critic: Criticism of SimE not being "real" incremental learning.** Kept as a Major weakness but contextualized: the design is disclosed, and the efficiency contribution (matched backbone) is real. The framing criticism stands as a major concern about the claims in the abstract/intro, not an invalidation of the technical work.

---

## Novel Insights

The most genuinely insightful empirical observation—cleanly documented in Tables 3 and 4—is that CIL performance with a frozen (or nearly frozen) encoder is primarily determined by the encoder's feature quality, and that pre-training data volume matters as much as backbone size. The distinction between DataComp-1B and LAION-2B despite both exceeding 1B samples (DataComp slightly underperforms LAION-2B despite larger curation) hints that data quality interacts non-trivially with dataset scale for downstream CIL generalization. This is a useful finding for the community independently of the SimE framework.

---

## Suggestions

1. **Fix Table 1 immediately.** Fill in the SimE(Ours)† row with the ViT-L/14 + LAION-2B numbers, and replace the SimE(Ours) row with ViT-B/16 + WIT-400M numbers (matching all baselines). All abstract/introduction claims about outperforming ZSCL by 5.3% should be re-scoped to the correctly matched comparison.

2. **Add a direct "Continual-CLIP with ViT-L/14 + LAION-2B" baseline.** This is the minimum needed to show that the adapter contributes something beyond what a larger frozen backbone provides alone.

3. **Cite and describe Fren-time** or remove it from Tables and Figures.

4. **Clarify the Task 1 training objective** in the main text rather than deferring entirely to the appendix.

5. **Reframe the "remarkable phenomenon" claim** based on its actual effect size, or add multi-run variance; avoid calling a 0.4% difference "significant."

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Human Score | Comparison to SimE |
|---|---|---|---|
| C-CLIP: Multimodal CL for VLM | `sb7qHFYwBc.md` | **6.50** (Accept) | More rigorous comparisons, proper methodology; SimE has backbone mismatch SimE doesn't meet this bar |
| Prediction Error-based CIL (PEC) | `DJZDgMOLXQ.md` | **6.50** (Accept) | Novel classification paradigm, fair comparisons; SimE lacks comparable rigor |
| Dual-Modality Prompt for CL | `04TRw4pYSV.md` | **3.50** (Reject) | Simple method, weak baselines, under-evaluated; similar profile to SimE |
| ProCEED: Prototype CIL | `gDZd8UGaxS.md` | **3.00** (Reject) | Novelty concerns, methodology issues; SimE has real CLIP study that ProCEED lacks |
| YoooP (prototype-based IL) | `H6pf70GZVU.md` | **5.00** (Reject) | Prototype-based IL with solid baselines; better experimental rigor than SimE |

**Positioning:** SimE's core efficiency contribution (adapter once + prototypes ≈ ZSCL accuracy at 1/100 the parameters, matched backbone) is real and non-trivial. However, the primary results table contains a backbone mismatch that inflates the headline claim by ~5–6%, the main comparison structure is misleading, the "remarkable phenomenon" claim is unsupported at the reported effect size, and "Fren-time" is uncited. The CLIP study (Tables 3-4) is genuinely useful but secondary. This profile aligns more closely with `04TRw4pYSV` (avg 3.5) than with `H6pf70GZVU` (avg 5.0), because the backbone mismatch undermines the primary empirical claim—a more serious problem than the issues in YoooP. The real efficiency contribution at ViT-B/16 prevents a lower score. I set the final score at **3.5**.

**Originality:** Low-to-moderate. The approach is a combination of well-known components (CLIP + AdaptFormer + prototype classifier) with a specific design choice (train only on task 1). The CLIP study is original.

**Importance of research question:** Moderate. Efficient CIL is an important problem.

**Claims vs. support:** Weak. The primary quantitative claim in the abstract is misleading at the headline backbone.

**Soundness of experiments:** Weak for the main comparison; reasonable for the CLIP ablation study.

**Clarity:** Moderate. The Table 1 backbone inconsistency and the Equation 3 formulation reduce clarity.

**Value to community:** Limited in current form; the CLIP backbone study could be valuable if extracted and presented honestly.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>