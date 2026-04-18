Now let me read some calibration papers to properly score this submission.Now I have enough information to write the final review. Let me synthesize everything.

---

## Summary

SimE is a class-incremental learning (CIL) framework that fine-tunes lightweight adapters (AdaptFormer / a proposed Multi-Adapter) on the CLIP image encoder during the **first task only**, then freezes all weights and uses class-mean (prototype) classifiers for all subsequent tasks. The paper additionally presents a Multi-Adapter design that extends AdaptFormer with additional attention-side and full-block adapter paths, an ablation on adapter placement within vs. between transformer blocks, and a systematic study of CLIP pre-training datasets and ViT backbone sizes for CIL scenarios.

---

## Strengths

- **Solid empirical results on the tested benchmarks:** SimE achieves clearly better numbers than all CLIP-based baselines in Table 1 (e.g., 91.66 avg on CIFAR-100 10-step vs. ZSCL's 85.94), and the per-task accuracy curves in Fig. 3 show stable performance for SimE while other methods degrade. The improvement is genuine and sizeable.
- **Genuine parameter efficiency:** Even though "thousands of parameters" overstates things (see below), Table 2 shows ~1.19M trainable parameters for the base configuration, significantly smaller than the ~100M+ for full-model fine-tuning methods. Fig. 4(a)(b) clearly illustrates this efficiency advantage.
- **Practical and interpretable design:** The freeze-after-task-1 paradigm is simple, has low training overhead, and is easy to reproduce. This is a meaningful practical property.
- **Useful systematic study:** Tables 3 and 4 provide an organized sweep over CLIP pre-training data sources (WIT-400M through LAION-2B, DataComp-1B) and ViT sizes (B/32, B/16, L/14), producing consolidated practical guidance for practitioners.
- **Informative adapter placement ablation:** Fig. 5 and Table 2 provide concrete empirical data on how placing adapters in earlier vs. later transformer blocks and between vs. within blocks affects CIL performance—useful for adapter practitioners even if the "phenomenon" framing is overstated.

---

## Weaknesses

### Fatal

**None.** The method works and the empirical numbers are not fabricated. However, several major issues collectively undermine the strength of the core claims.

---

### Major

**1. Core efficiency claim is factually inconsistent (thousands vs. millions of parameters).**
The abstract and Section 4.2 repeatedly state: *"SimE, with only thousands of parameters…"* and *"our method requires only thousands of trainable parameters."* However, Table 2 plainly reports `Para(M)` = **1.19M** for the simplest (AdaptMLP-only) configuration, rising to 3.57M. Even the smallest bottleneck configuration shown in Fig. 4(d) is ~0.5M. These are *hundreds of thousands to millions* of parameters, not "thousands." This discrepancy is not a rounding issue—it is three orders of magnitude off. The efficiency advantage over ZSCL (~140M) is real but the paper inflates it through incorrect labeling, undermining trust in the quantitative claims. The paper must correct this across the abstract, Section 4.2, and all figures.

**2. Structural asymmetry: SimE is not a genuine continual learner after Task 1.**
As Fig. 1 and Section 3.1 make explicit, SimE only trains the adapter during Task 1 and **freezes all parameters for every subsequent task**, merely appending prototype vectors. Methods like ZSCL, LwF-VKD, and LwF-VR update the backbone at every task and must manage catastrophic forgetting continuously—a strictly harder regime. SimE's near-flat per-task accuracy curves (Fig. 3) are almost trivially a consequence of freezing the encoder, not of superior forgetting mitigation. The paper frames its advantage as *incremental learning superiority* but is comparing an "adapt-once + frozen prototype retrieval" system against fully dynamic CIL methods. This asymmetry is the dominant factor behind SimE's performance gap and must be discussed honestly as a fundamental scope difference, not treated as a triumph of the IL algorithm. A fair comparison requires baselines operating under the same "no parameter update after task 1" constraint.

**3. Table 1 contains an empty row for the headline result (SimE†).**
The abstract's headline claim—*"SimE surpasses traditional methods by 9.6% on TinyImageNet"*—refers to the ViT-L/14 + LAION-2B configuration (SimE†), yet the SimE† row in Table 1 is completely blank. The most strongly advertised result has no tabulated evidence. This is either an omission that must be rectified or suggests the result is derived from a separate setting not clearly connected to the table.

**4. Missing recent PTM-based CIL baselines.**
The CLIP-based baselines in Table 1 are drawn from 2022–2023. The PTM-based CIL literature has since produced multiple parameter-efficient methods (prompt-based, LoRA-based, adapter-based) that are similarly memory-free and lightweight. Without comparison to these methods—well-established in contemporary human reviews of closely related papers at the same venue—the state-of-the-art claims cannot be confidently assessed. The paper's positioning as outperforming "state-of-the-art" is premature given the incomplete baseline set.

---

### Minor

**5. The "remarkable phenomenon" is overstated given the evidence.**
The claim that more adapter connections within transformer blocks *degrade* CIL performance at small step counts is framed as a "remarkable phenomenon" and a key contribution. Looking at Table 2, the relevant differences are within ~0.1–0.5% accuracy (e.g., AdaptMLP-only 85.60 avg vs. AdaptMLP+AdaptAtten 85.94 at 10 steps; AdaptMLP+AdaptAtten+AdaptAll 85.54). No standard deviations or multiple random seeds are reported, and the parameter count also increases across configurations (1.19M → 2.38M → 3.57M), so it is impossible to attribute the observed non-monotonicity to "placement" rather than capacity/optimization effects. This should be presented as a preliminary empirical observation, not a "remarkable phenomenon" constituting a scientific contribution.

**6. No sensitivity analysis on Task 1 composition.**
The entire method hinges on Task 1 adapter training generalizing to all future tasks. There is no experiment varying the size of Task 1 (e.g., 10 vs. 50 vs. 100 base classes), the domain overlap between Task 1 and later tasks, or the distribution within Task 1. Since "task ordering is often somewhat arbitrary" in CIL, the absence of this analysis is a notable gap in the empirical coverage.

**7. Evaluation limited to two small, low-resolution benchmarks.**
All main results are on CIFAR-100 and TinyImageNet. These are relatively small-scale and low-resolution. The CIL community expects evaluation on at least one larger-scale benchmark to validate generalizability.

**8. Notational inaccuracy in encoder formulation.**
Equation (3) writes the encoder output as $E(\mathbf{x}) = \sum_i^B (\ldots)$, i.e., the *sum* of outputs across all transformer blocks. A ViT's representation is the output of the *last* block fed sequentially through all prior blocks, not a sum. Equation (7) inherits this notation. This is either a misleading abstraction that obscures the actual implementation, or a genuine design deviation that is never explained.

---

### Trivial

- The GPU usage metric "GFP" is used in Fig. 4 without definition; it should be clarified (e.g., GFLOP, GPU memory in GB, etc.).
- Notation in Eq. (6): $\mathbf{c}_{ij}$ and $\mathbf{s}_{ij}$ carry double duty as both inputs and outputs, making the multi-adapter block hard to parse; a cleaner notation with separate input/output variables would aid readability.

---

## Nice-to-Haves

- Feature-space visualization (t-SNE/UMAP) across tasks showing prototype separability after task-1 adaptation would provide intuitive support for the "frozen encoder generalizes" claim.
- Analysis of per-task backward transfer metrics would help readers assess whether SimE's stable curve reflects genuine anti-forgetting or simply no learning after task 1.
- A deeper investigation of *why* early-block adapters benefit performance more than late-block adapters (Fig. 5) would convert an empirical curiosity into an actionable insight.
- Experiments with different Task 1 configurations (varying number of base classes, domain diversity) would bound the method's applicability.
- Report mean ± standard deviation over at least 3 seeds, particularly for Table 2's small-margin comparisons.

---

## Removed Points

*These points are flagged to be removed; treat them with caution as they either misread the paper, are unfairly applied, or violate the rules.*

- **Harsh Critic Issue 1 (mixing from-scratch vs. CLIP methods):** The paper explicitly acknowledges the distinction in Table 1's footnote ("UCIR, PASS, DyTox, and DER train from scratch, while the remaining methods use CLIP ViT-B/16"). The comparison to traditional from-scratch methods is presented as illustrating the overall benefit of CLIP-based approaches, and the main competitive claim is against CLIP-based baselines. This is not a hidden or deceptive comparison.

- **Harsh Critic Issue 4 (memory bank comparisons in bytes):** Framing prototype storage as comparable to replay buffers in bytes is a legitimate point, but the paper's primary claim is "no exemplar replay," which is structurally true. The exact byte-for-byte accounting is excessive scope for this critique.

- **Harsh Critic's claim that "GPU usage is anecdotal":** The ~1/3 GPU claim (Fig. 4(b)) is approximate visualization data rather than a core empirical finding; calling it "anecdotal" is fair but this is a trivial presentation issue, not a structural flaw.

- **Harsh Critic's claim about LAION-2B recommendation being trivial:** Calling the recommendation for larger CLIP models "unsurprising" is a scope-creep criticism—the paper's systematic study may confirm known trends, but providing that consolidated evidence on CIL benchmarks has standalone practical value.

- **Human Finder's novelty critique (citing verbatim quotes from other papers about prototypes being common):** While the novelty concern is valid, the argument that "prototype-based classification is well-known" does not erase the contribution of showing that freeze-after-task-1 with prototypes outperforms CLIP-finetuning methods. The concern is better framed as "limited novelty" (kept in Weaknesses) rather than citing other papers' reviewer quotes verbatim.

- **Harsh Critic's equation 3/7 "sum over blocks" claim as a major issue:** While it is a real notational inaccuracy (kept as Minor), the harsh critic characterizes this as reducing reproducibility severely. The actual implementation is standard ViT + adapters, and the notation is an imprecise shorthand rather than an unreproducible design.

---

## Novel Insights

The review process surfaces one genuine insight worth highlighting for the authors: **SimE's core advantage may be entirely attributable to the "adapt-once, freeze-forever" paradigm rather than to the Multi-Adapter design.** The ablation in Table 2 shows that even the baseline configuration with zero adapters (frozen CLIP + prototypes) achieves 79.69 avg, already outperforming several baselines in Table 1. The gap from 79.69 to 91.66 when adding adapters trained on Task 1 indicates that the dominant factor is base-task adaptation, not the adapter architecture itself. If the paper re-framed its contribution as "a simple, principled baseline for CLIP-based CIL via single-task adaptation + frozen prototype retrieval," it would be a more honest and still valuable contribution—especially with appropriate comparisons against methods sharing the same operational constraint.

---

## Suggestions

1. **Fix the parameter count claim immediately.** Replace "thousands of parameters" with the correct millions figure throughout, and recalibrate all relative efficiency claims accordingly.
2. **Fill in the SimE† row in Table 1** with actual numbers or remove the reference to this configuration from the abstract headline.
3. **Add a "fair constraint" ablation:** Re-run the strongest CLIP-based baselines (at minimum Continual-CLIP and ZSCL) under the same "no parameter update after task 1" constraint to isolate SimE's genuine advantage from the "never update = never forget" free lunch.
4. **Reframe the IL setting explicitly** in the paper as "single-task adaptation + frozen prototype retrieval," distinguishing it from fully dynamic CIL and clearly noting the advantages and limitations of this formulation.
5. **Add variance estimates** (at minimum over 3 seeds) to Table 2 before claiming a "phenomenon" from sub-1% differences.
6. **Broaden evaluation** to at least one larger-scale benchmark to allow the community to assess generalizability.
7. Fix the encoder notation (Eqs. 3 and 7) to reflect the sequential block structure of ViT rather than the sum-of-all-blocks form.

---

## Score and Decision

**Calibration:**
- *mrRbIcyouU* (Revisiting CIL with PTMs, Withdrawn/Reject, scores 6,3,5,5): Very similar "train on first task + prototype" design. That paper was stronger—it proposed 4 new benchmarks and was more systematic—yet was rejected. The paper under review is **weaker** than mrRbIcyouU on all dimensions: no new benchmarks, limited to 2 small datasets, parameter inconsistency, and an empty result row.
- *HyperAdapter (29sul3tAEa, Withdrawn/Reject, scores 5,3,5,5,6)*: Also rejected for limited evaluation and missing recent baselines, despite a more novel core mechanism (hypernetwork). The paper under review has comparably limited evaluation.
- *n8UYBYE9SQ (Selective Parameter Update, Withdrawn/Reject, scores 5,3,3,5)*: Rejected for missing comparisons and insufficient theoretical support. Shares the limited-baselines weakness.
- *C-CLIP (sb7qHFYwBc, Accept, scores 6,6,8,6)*: Accepted due to a new benchmark, comprehensive multimodal evaluation, and a proper CL design. The paper under review is clearly below this level.

The paper under review is closest to mrRbIcyouU (which the community rejected at avg ~4.75) and is weaker than that paper. The major parameter inconsistency, empty headline result row, and methodological asymmetry (freeze-after-task-1 compared against dynamic CIL methods) push the score below the borderline rejection zone for similar papers. The empirical results are real and the work has practical value, but the framing is sufficiently misleading and the evaluation sufficiently incomplete that it does not meet acceptance standards in its current form.

**Final Score: 4.0**

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>