Good. I have sufficient anchors now. Let me finalize my review.

## Summary of calibration anchors:
- **High (>7)**: SD-LoRA (7.5, Oral) - CIL with foundation models, actual incremental learning with LoRA; SEED (7.0, Poster) - CIL with expert selection; Budgeted CL (7.5, Spotlight)
- **Medium (4-6)**: SimpleCIL/APER (4.75, Withdrawn) - extremely similar paper; frozen PTM + prototypes for CIL, only finetunes on first task; LVLM-CL (2.5) was low; Js5PJPHDyY (6.0) - training-free CLIP; OVOR (6.0) - prompt-based CIL; YoooP (5.0) - prototype CIL; CREATE (4.80) - CIL with questionable baselines
- **Low (<3)**: Projected Subnetworks (2.0, Reject) - ambiguous CL framing; DIRAD/PREVAL (2.33) - CL with structural growth; LVLM-CL (2.5) - missing evaluation

The paper under review is very similar to SimpleCIL (4.75), but adds the Multi-Adapter ablation and backbone study. However, it has the Table 1 vs Table 2 inconsistency (potential unfair comparison using ViT-L/14+LAION-2B against ViT-B/16 baselines) and the "thousands of parameters" factual error. These are significant issues.

Let me now write the final review.</think>Now I have a thorough understanding of the paper and calibration anchors. Let me compose the final review.

## Summary

SimE proposes a simple class-incremental learning (CIL) framework that fine-tunes CLIP's image encoder with adapters on only the first task, then freezes all parameters for subsequent tasks and merely computes class prototypes (mean features) as classifier weights. The paper also introduces Multi-Adapter, which adds multiple adapter sub-modules within and between transformer blocks, and reports a "remarkable phenomenon" that more within-block adapter connections do not improve and may degrade performance at smaller incremental steps. A systematic study of CLIP backbones and pre-training datasets is included.

## Strengths

- **Strong empirical gains over prior CLIP-based CIL methods on the ablation configuration**: Table 2 shows that even with the standard ViT-B/16+WIT-400M backbone, SimE with AdaptMLP achieves 85.60% avg (vs. 79.69% without any adapter), demonstrating that Task-1-only adapter fine-tuning meaningfully closes the domain gap compared to frozen CLIP. Table 4 and Table 3 further show systematic gains from larger backbones and pre-training sets, providing useful practical guidance.

- **Zero catastrophic forgetting by construction**: Since the encoder is frozen after Task 1, there is no forgetting for tasks 2–T, and no memory bank is needed. This is a legitimate and practical design choice, especially given the strength of CLIP's zero-shot features (Section 3.1, Eq. 4; Figure 1 panels B and C).

- **Comprehensive backbone and pre-training dataset study**: Tables 3–4 evaluate 5 pre-training datasets and 4 backbone architectures under a unified CIL protocol, offering concrete and actionable guidance (e.g., LAION-2B + ViT-L/14 is the strongest configuration). This is a useful empirical contribution for the community.

## Weaknesses

### Fatal

None.

### Major

- **Unfair comparison in Table 1 — headline results likely use ViT-L/14+LAION-2B against ViT-B/16 baselines**: Table 1 reports SimE achieving 91.66% avg on CIFAR-100/10 steps, but Table 2 (which the paper states uses the default ViT-B/16 — Section 4.4: "Our default model uses ViT-B/16") shows at most 85.94% under the same 10-step setting. Table 4 shows ViT-L/14 on CIFAR-100/10 steps yields 88.79%, and Table 3 shows LAION-2B+ViT-B/16 yields 88.34%. The 91.66% result is only achievable with the ViT-L/14+LAION-2B configuration (indicated by † in the table caption). However, Table 1's caption states "the remaining methods use CLIP ViT-B/16 as the backbone," meaning all baselines use a significantly smaller model and pre-training corpus. Comparing SimE with a ~3× larger backbone and ~5× larger pre-training dataset against ViT-B/16 baselines is fundamentally unfair and inflates the claimed 5.3% improvement over ZSCL. This discrepancy undermines the paper's central accuracy claims and must be clarified or corrected.

- **Factually incorrect "thousands of parameters" claim**: The abstract and Section 4.2 both state SimE operates with "only thousands of parameters." However, Table 2 clearly reports trainable parameter counts of 1.19M–3.57M, and Figure 4(a)'s data shows "Ours" at ~10M training parameters. This is a three-orders-of-magnitude discrepancy between the claimed and actual parameter counts. Such a misrepresentation in the abstract is a significant factual error that directly supports the paper's efficiency narrative.

### Minor

- **"Remarkable phenomenon" is overclaimed for sub-1% effect sizes without variance**: The core finding about within-block adapter connections (Table 2) shows effect sizes below 1% (e.g., 85.94% vs. 85.54% at 10 steps), and no statistical significance or variance across runs is reported. While the within-block vs. between-block distinction (Figure 5 vs. Table 2) is a valid empirical observation, calling this "remarkable" is overclaiming for differences well within typical run-to-run variance. The finding is more accurately described as: modest effect sizes where more within-block capacity provides diminishing or slightly negative returns at finer task granularity — consistent with standard overfitting on limited Task 1 data.

- **Missing prompt-based CIL baselines**: Among CLIP-based methods, key prompt-incremental approaches (L2P, DualPrompt, CODA-Prompt) that use parameter-efficient adaptation of ViT/CLIP for CIL are absent from Table 1. These are the most direct comparators to an adapter-based approach, and their absence leaves a gap in showing SimE's positioning relative to the full landscape of CLIP-based CIL methods.

- **Table 2 parameter counts for different adapter combinations appear inconsistent**: Rows showing AdaptMLP+AdaptAtten and AdaptMLP+AdaptAll both report 1.19M parameters despite adding different modules, which suggests possible errors or unexplained parameter sharing. This minor inconsistency should be clarified.

### Trivial

- Eq. 3 notation uses $\mathbf{x}_i$ in a way that could be read as each block receiving the same raw input, though context and the note "when $i=0$, $\mathbf{x}_i$ is the reprocessed image $\mathbf{x}$" slightly mitigate this — it would be clearer if $\mathbf{x}_i$ explicitly denoted the output of block $i-1$.

## Nice-to-Haves

- Report SimE results on standard ViT-B/16 alongside the † configuration in Table 1, or provide a separate table comparing all methods under identical backbones, to enable fair like-for-like comparison.
- Add per-class or per-task accuracy breakdowns to reveal whether the frozen encoder discriminates poorly for specific later-task classes with no path to improvement.
- Include a direct ablation comparing "Continual-CLIP + Task-1 prototype-only update" vs. SimE to isolate the adapter's contribution (the jump from 79.69% to 85.60% in Table 2).
- Report variance across multiple runs for Table 2's sub-1% differences.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The method does not genuinely perform incremental learning"** (Harsh Critic Issue 1) — The paper is transparent about training only on Task 1 (Figure 1, Section 3.1). This is a design choice, not an oversight. The SimpleCIL paper (human-reviewed at 4.75) employs the same approach and reviewers considered it legitimate though limited. However, the limitation that the representation cannot adapt for later tasks is real and is captured in the minor weakness about per-class analysis. The framing concern is partially valid but not "fatal" — prototype-based CIL with frozen encoders is an established paradigm.

- **"The 'remarkable phenomenon' is simply overfitting"** (Harsh Critic Issue 2) — While the effect sizes are small, the critic's dismissal as "simply overfitting" is itself an unproven assertion. The within-block vs. between-block distinction IS a genuine empirical finding with practical implications for adapter design, even if the effect is modest. Overclaiming the "remarkable" nature is addressed in the minor weakness above.

- **"Fren-time and iCaRL listed alongside CLIP methods"** (implied from section notes) — Mixing methods trained from scratch with CLIP-based ones is standard in CIL tables for comprehensive coverage. Not a real weakness.

- **"Notation in Eq. 6 is hard to parse"** — This is a presentation nitpick. The case notation, while not elegant, is functional.

- **Requests for t-SNE visualizations, EWC/SI regularization extensions** — These are nice-to-haves, not weaknesses. The paper's scope is the SimE framework; requesting extensions beyond that scope is scope creep.

- **Requests for missing related works** — Per rules, I cannot verify what works might be missing.

- **"Table 1 comparison mixes scratch-trained and CLIP-based methods"** — Standard practice in the field, not a valid weakness.

## Novel Insights

The paper's most interesting empirical finding is the asymmetry between within-block and between-block adapter scaling in the CIL context — adding more adapters *between* transformer blocks consistently helps (Figure 5), while adding more *within* blocks shows diminishing or slightly negative returns at finer task granularity (Table 2). This asymmetry, while modest in effect size, has practical implications for how to allocate adapter capacity when fine-tuning large models for CIL. However, this insight is partially undercut by the small effect sizes and lack of significance testing.

## Suggestions

- **Most critical**: Add a fair-comparison table where all methods (SimE and baselines) use the same backbone. If the 91.66% result uses ViT-L/14+LAION-2B, either re-run baselines with the same backbone or clearly present SimE under ViT-B/16 in the main comparison table.
- **Correct the "thousands of parameters" claim** in the abstract and Section 4.2 to match the actual 1.19M–3.57M figures from Table 2, or explain precisely what subset of parameters the claim refers to.
- Report standard deviations across 3+ runs for Table 2 to support the "phenomenon" claim.
- Add prompt-based CIL baselines (L2P, DualPrompt, CODA-Prompt) to the comparison table.

## Evaluation

**Originality**: Low-to-moderate. The core idea (freeze CLIP after Task 1, use prototypes) closely resembles SimpleCIL/Continual-CLIP-type approaches. The Multi-Adapter contribution and the within-vs-between-block finding add modest novelty, but the effect sizes are small.

**Importance**: The question of how to efficiently leverage CLIP for CIL is important, but the paper's contribution is more benchmarking/study than algorithmic innovation.

**Claims support**: The headline accuracy claims (91.66%) are undermined by the apparent use of a larger backbone against smaller baselines, and the efficiency claims are factually misstated ("thousands" vs. millions of parameters). These significantly weaken the evidentiary basis.

**Experimental soundness**: The systematic backbone study (Tables 3–4) is sound and useful. The main comparison (Table 1) has the fairness issue noted above. The ablation (Table 2) needs variance.

**Clarity**: The paper is generally clear with good figures, though some notation (Eq. 3, 6) could be improved.

**Community value**: The backbone/dataset study provides practical guidance. The prototype+frozen-CLIP baseline is useful but overlaps significantly with prior work (SimpleCIL).

## Calibration

Papers compared against:
1. **SimpleCIL/APER** (mrRbIcyouU, avg 4.75, Withdrawn): Very similar approach — frozen PTM + prototypes for CIL, fine-tune only on first task. Reviewers criticized unfair comparison and limited novelty. SimE is comparable or slightly worse due to the factual errors and Table 1 unfairness.
2. **SD-LoRA** (5U1rlpX68A, avg 7.5, Oral): CIL with foundation models but genuine incremental learning (LoRA updated every task) with theoretical grounding. SimE is clearly below this.
3. **Training-free CLIP GDA** (Js5PJPHDyY, avg 6.0, Poster): Simple training-free CLIP baseline. SimE has the advantage of the systematic backbone study but the factual errors and unfair comparison are worse issues.
4. **Projected Subnetworks** (WM5G2NWSYC, avg 2.0, Reject): Ambiguous CL framing and fundamental methodology issues. SimE is clearly above this — it does provide useful empirical insights despite its problems.
5. **SEED** (sSyytcewxe, avg 7.0, Poster): CIL with expert selection, simple method but genuine algorithmic contribution. SimE is below this due to limited novelty and the factual/comparison issues.

SimE sits in a similar space to SimpleCIL (4.75) but with additional backbone studies. However, the "thousands of parameters" error and the Table 1 fairness issue are serious problems that simple revisions cannot fully fix without re-running experiments. This places it somewhat below SimpleCIL's withdrawn score.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>