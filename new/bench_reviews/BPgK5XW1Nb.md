Now let me search for calibration anchors.Now I have enough information to write the final consolidated review.

---

## Summary
SPA (Spread Preference Annotation) is a framework for LLM alignment that iteratively expands a small seed preference dataset (~3.3% of UltraFeedback) by judging preferences directly via the DPO implicit reward using a fixed SFT reference model (Eq. 7), then mitigates label noise through a confidence-based self-refinement with de-coupled noise detection via logit extrapolation (Eqs. 10–12). Experiments on Mistral-7B show an AlpacaEval 2.0 length-controlled (LC) win rate improvement from 7.58% to 15.39%, and a raw win rate improvement from 4.72% to 21.13%, with generalization demonstrated across Phi-2-2.7B, LLaMA-3-8B, and Phi-3-14B.

---

## Strengths

- **Direct preference judgment via DPO implicit reward with fixed SFT reference is clean and well-motivated.** Table 7 provides a controlled single-iteration comparison showing this design (LC: 15.08%) clearly outperforms using the previous iteration model as reference (13.73%), no reference model (12.83%), and PairRM (13.57%), validating the design choice on its own terms.

- **Cross-model generalization is broad and convincing.** Table 5 shows consistent LC and WR gains across architecturally diverse models (Phi-2-2.7B, LLaMA-3-8B-Instruct, Phi-3-14B-Instruct) with the same training pipeline, supporting that SPA is not tailored to Mistral-7B.

- **Robustness across seed sizes is well-demonstrated.** Table 3 shows SPA consistently outperforms DPO at every tested seed fraction (0.8%, 1.7%, 3.3%, 10%), with the LC win rate doubling relative to DPO at every setting.

- **Low variance across random seeds (Table 4).** Three random seeds show WR variance of only 0.03 and LC variance of 2.10, and the LC lower bound (13.36%) exceeds the strongest external-judge baseline (11.87%), providing some statistical confidence in the gains.

- **Computational efficiency of de-coupled noise detection is genuine.** The logit extrapolation in Eq. 12 reuses h_θ and h_ref already computed for the DPO objective, so DND incurs no meaningful additional compute.

---

## Weaknesses

### Fatal
None.

### Major

- **The primary headline metric (21.13% raw win rate) is inflated by response length, and this inflation is not adequately surfaced.** Verified from Table 1: for every Mistral-7B baseline (SFT, DPO, PairRM, Zephyr), the LC win rate ≥ raw win rate (e.g., SFT: 7.58% LC vs. 4.72% WR; DPO: 9.03% vs. 7.68%; Zephyr: 11.75% vs. 10.03%). SPA is the sole exception, where WR (21.13%) dramatically exceeds LC (15.39%) by 5.74 pp — the hallmark of response-length gaming in LLM judge evaluations, which AlpacaEval 2.0's LC metric was specifically designed to correct. The same inversion appears for LLaMA-3-8B SPA in Table 5 (WR: 34.84% vs. LC: 25.03%). The abstract, Figure 2, and Section 5.2 all lead with and foreground the 21.13% figure; the LC metric — which is the more reliable quality indicator — is treated as secondary throughout. This overstates the headline contribution. The LC gains (7.58% → 15.39%) remain substantial and should be framed as the primary result.

- **Table 2's main comparative claim is confounded: it cannot isolate the contribution of the judgment method.** As explicitly stated in Section 5.1, the Iterative DPO baselines in Table 2 have self-refinement "removed." This means the comparison between SPA (full: DE + SR + DND) and Iterative DPO (PairRM) (DE only) conflates three factors simultaneously: judgment method, reference model choice, and presence of SR+DND. Table 6 shows SR+DND add LC +0.98% / WR +1.22% on top of DE alone. Table 7 provides a more controlled one-iteration comparison (judgment method only), which does support SPA's design, but that evidence covers only a single transition. The paper's stated claim that "direct preference judgment outperforms external reward models" cannot be cleanly established from Table 2 as currently structured.

- **The dominant driver of all gains is iterative on-policy data expansion (DE), not the novel SPA components.** Table 6 explicitly shows: DE alone achieves LC 14.41% / WR 19.91%, while full SPA achieves LC 15.39% / WR 21.13%. Self-refinement without DND adds only LC +0.29% / WR +0.03%; DND adds an additional LC +0.69% / WR +1.19%. Thus DE alone accounts for ~87% of the total LC improvement. The paper does not compare against contemporaneous self-play / online DPO methods (e.g., SPIN, SPPO) that also use the model's own generations iteratively, which would determine whether the DE contribution is itself novel relative to the prior art.

### Minor

- **The `λ` hyperparameter schedule for de-coupled noise detection (1/2, 1/4, 1/8 across iterations) is presented without ablation.** Given that DND is a named contribution, the sensitivity to λ across iterations is not examined in the main paper, making it unclear whether the specific schedule matters or whether a fixed value would perform similarly.

- **The no-seed comparison (Figure 4) uses a different base model.** The "0% seed data" experiment starts from Mistral-7b-instruct-v0.1 (already RLHF-tuned), while the main experiments start from the SFT model. This makes the "no seed" result illustrative but not a clean ablation of seed data quantity.

- **The LLM-as-judge baseline specification is deferred to the appendix** (Section 5.1: "Details are presented in Appendix B"), leaving it unclear in the main paper which model is used as the judge, at what scale, and with what prompt. This matters because the comparison partly depends on whether the LLM-as-judge is a small 7B model or a larger capable judge.

### Trivial
None beyond items already noted.

---

## Nice-to-Haves

- **Response length analysis across iterations.** Given the WR vs. LC inversion, reporting average response lengths per iteration and per method would directly diagnose whether SPA's WR gains reflect quality or length drift, and would clarify whether DE or SR/DND is responsible for length increase.

- **Comparison to self-play baselines (SPIN, SPPO).** The DE component—iterative on-policy DPO with model-self-labeling—is closely related to prior self-play methods. A direct comparison would sharpen the claim about what is novel about SPA's judgment mechanism.

- **Ablation of the `λ` schedule and the K% noise threshold.** Even a 2×2 grid over a few values would substantially strengthen confidence in the DND contribution.

- **Analysis of why iteration 3 marginally decreases LC (from ~16% at Iter. 2 to ~15% at Iter. 3, per Figure 3).** Understanding the iteration-level degradation would be practically useful for deploying SPA.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "Per-token logit extrapolation doesn't approximate sequence-level policy."** The critique that Eq. 12 is theoretically unjustified because per-token extrapolation doesn't equal a sequence-level policy shift is pedantic and standard in the logit-editing literature. The paper's footnote 4 acknowledges the approximation and empirical results validate its effectiveness. Given the modest but consistent gains from DND (Table 6), this theoretical precision concern doesn't undermine the contribution. **Removed as scope-creep into theoretical territory for an empirical systems paper.**

- **Harsh Critic: "The 3.3% label efficiency comparison conflates label efficiency with training efficiency."** The paper is explicit in Section 5.1 and Section 5.2 about using 3.3% gold labels vs. 100% for Zephyr, and is clear that total prompts are the same (60K). The framing in Section 5.2 ("using 3.3% vs. 100%") refers specifically to labeled preference pairs; this is not misleading in context. **Removed as a strawman.**

- **Harsh Critic / Strength Finder: Several strengths about "noise-aware preference learning being simple and backward-compatible" and "practical virtue of few lines of code."** These are too generic to stand as meaningful strengths without evidence tying them to measurable gains. Given that SR alone contributes negligible LC improvement, these are weakened and not listed as primary strengths above. **Moved to Removed.**

- **Strength Finder: "SPA works even without seed preference data."** The "0% seed" experiment (Figure 4) uses Mistral-7b-instruct (already instruction-tuned) as π_0, not the SFT model. The setup is structurally different from the main experiments, making this claim partially misleading. The modest gains (LC: 10.14% → 11.59%, WR: 6.31% → 9.79%) are real but modest. Retained as minor supporting evidence rather than a primary strength.

---

## Novel Insights

The most insight-generating aspect of the paper is the empirical demonstration that a fixed SFT reference model in Eq. 7 substantially outperforms using the previous iteration model as the reference (Table 7: 15.08% vs. 13.73% LC). This is non-obvious: one would expect the more recently trained model to better reflect accumulated preferences, but the fixed reference appears to provide a more stable anchor that prevents reward miscalibration under distribution shift. This finding has practical implications for other iterative DPO pipelines and deserves more theoretical attention than it receives.

---

## Suggestions

1. **Restructure main results to lead with LC win rate.** Table 1's abstract statement and Figure 2's framing should be reordered so that LC (15.39%) is the headline metric, with raw WR as a secondary indicator. The abnormal WR > LC pattern should be explicitly acknowledged, with a brief response-length analysis added.

2. **Add a fair comparison variant in Table 2.** Apply SR+DND to Iterative DPO (PairRM) and Iterative DPO (LLM-as-judge) to isolate whether the judgment mechanism, or the denoising components, account for the gap. If SR+DND applied to PairRM still underperforms SPA, the judgment-method claim becomes much stronger.

3. **Add a self-play/SPIN baseline.** Even a single comparison against SPIN (which also uses only self-generated data with no external labels) would contextualize whether DE's gains require SPA's specific design.

---

## Score and Decision

**Calibration anchors:**
- `/home/wg25r/review_agent/human_reviews/NtAXAvIYuN.md` (iREPO, avg 3.40): Rejected — uses same implicit-reward iterative DPO framework as SPA but with weaker experiments and more fundamental theoretical issues. SPA is clearly better.
- `/home/wg25r/review_agent/human_reviews/uIGnuyDSB9.md` (SeRA, avg 6.00): Accepted poster — highly analogous (implicit reward, iterative on-policy preference bootstrapping on UltraFeedback). SeRA and SPA have comparable breadth of experiments; SeRA had cleaner framing; SPA has broader cross-model validation and the DND component.
- `/home/wg25r/review_agent/human_reviews/Pujt3ADZgI.md` (INPO, avg 6.00): Accepted oral — iterative alignment via Nash policy game, more principled theoretical grounding than SPA but comparable empirical scope.
- `/home/wg25r/review_agent/human_reviews/1oijHJBRsT.md` (Instruction Backtranslation, avg 8.00): Accepted oral — self-alignment with seed data, iterative, but substantially stronger methodology and cleaner claims. SPA is well below this anchor.
- `/home/wg25r/review_agent/human_reviews/aYYZBPoSHb.md` (avg 3.40, LLM alignment with self-judgement): Low anchor — rejected for incremental contribution and conflated experiments. SPA has more substance.

**Positioning:** SPA sits between iREPO (3.40) and SeRA/INPO (6.00). The major weaknesses—length-inflated headline metric, Table 2 confound, absent self-play baselines—pull it below SeRA. However, unlike iREPO, SPA has genuine and replicable LC improvements, solid cross-model validation, and a controlled Table 7 experiment that partially supports its claims. The paper is borderline: a strong LC improvement (~2× over SFT) with genuinely clean ablations, but major comparative claims are overclaimed or not cleanly supported. This lands at **5.5**, below the typical accept threshold for an ICLR poster but not clearly in reject territory.

**Originality:** Moderate. The core idea (DPO implicit reward as labeler with fixed SFT reference) is clean and well-executed but not conceptually far from concurrent work. The fixed-reference insight is the most original finding.

**Importance:** The label-efficiency question is practically important. The LC gains are real and the cross-model evidence is solid.

**Claims vs. support:** Partially supported. The LC gains are well-supported; the judgment-method superiority claim is confounded.

**Soundness:** Good experiments, but the main comparative table (Table 2) has a methodological flaw.

**Clarity:** Generally clear, but the paper's narrative is distorted by privileging the inflated WR metric.

**Value to community:** Useful as a practical recipe for iterative DPO with self-labeling, even if not all claims hold up rigorously.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>