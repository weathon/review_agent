=== CALIBRATION EXAMPLE 7 ===

# Final Consolidated Review
## Summary

HUBBLE is a fully open-source suite of LLMs (1B and 8B parameters, trained on 100B or 500B tokens) designed for controlled scientific study of LLM memorization. Standard models are pretrained on a DCLM-filtered English corpus, while perturbed models additionally receive controlled insertions of sensitive text (book passages, synthetic biographies, test sets) at randomized duplication levels to enable causal measurement of memorization. The paper establishes two main empirical findings—dilution (larger corpora reduce memorization risk) and ordering (sensitive data inserted earlier is memorized less)—and demonstrates HUBBLE's utility as a benchmark for membership inference attacks and machine unlearning.

## Strengths

- **Controlled perturbation design enabling causal claims.** The randomized assignment of duplication levels (0×, 1×, 4×, 16×, 64×, 256×) to perturbation examples, combined with decontamination of the base corpus, allows measurement of causal quantities impossible in observational studies. For instance, the MIA benchmark avoids the temporal confounds that undermined WikiMIA (§6.1; Duan et al., 2024). This is a genuine methodological advance over prior model suites like Pythia.

- **Comprehensive open-source release.** All models (including checkpoints), training code, configuration, datasets, and evaluation code are publicly released. This is a significant community resource: the 200k A100 GPU-hours of compute required makes independent reproduction impractical for most academics, so the open release fills a real access gap.

- **Dilution and ordering findings with practical import.** The dilution finding (Figure 2) generalizes Bordt et al. (2025) beyond test set contamination to copyright and privacy domains. The ordering finding (Figure 14), showing that data inserted only in the first quarter of training is largely forgotten, is practically actionable and complements the deduplication best practice.

- **MIA and unlearning benchmarks address known limitations.** HUBBLEMIA provides clean member/non-member splits without spurious features (§6.1). HUBBLEUNLEARNING tests unlearning on pretraining data with known duplication rates, extending beyond the fine-tuning-only settings of TOFU and MUSE (§6.2). The finding that current unlearning methods degrade near-neighbor knowledge rather than targeting specific data (Figure 3) is a valuable negative result.

- **Interference validation strengthens causal claims.** The three single-domain perturbation models (copyright-only, privacy-only, test-set-only) confirm that the combined perturbation model matches per-domain behavior (§4, Figure 20), lending credibility to the domain-level findings.

## Weaknesses

### Major:

- **Dilution claim rests on only two corpus sizes.** The core dilution finding compares 100B vs. 500B tokens—a single binary contrast. This cannot establish whether the relationship between corpus size and memorization is monotonic, linear, or follows a different scaling law. Without intermediate points (e.g., 200B, 300B), the generalizability of "train on more tokens to reduce memorization risk" as a prescriptive best practice is uncertain, especially at the trillion-token scales used in practice.

- **Ordering claim demonstrated only on 1B models.** The timing runs (§4, Figure 14) are exclusively 1B-parameter models. Given that the paper shows larger models memorize at lower duplication levels (§4), the ordering effect may differ substantially at the 8B scale. Presenting ordering as a general best practice without verifying it at the larger scale weakens the claim's credibility.

- **Claim calibration is inconsistent with evidence strength.** The abstract states memorization risks are "determined by" frequency relative to corpus size—causal language stronger than a two-point comparison warrants. Section 4 states that early-inserted data "does not memorize," but Figure 14 shows non-zero (though reduced) memorization. These overstatements matter because the paper explicitly aims to inform policy and practice; imprecise claims could mislead practitioners.

- **Practical limitations of best practices underdiscussed.** Dilution requires 5× more tokens (100B→500B), incurring enormous additional compute cost. Ordering requires knowing which data is sensitive *before* training begins, which is precisely the information practitioners often lack. The paper frames these as "best practices" (§4) without adequate discussion of when they are feasible. A brief honest assessment of these constraints would strengthen rather than weaken the contribution.

### Minor:

- **Popular vs. unpopular book memorization contradicts data density hypothesis without investigation.** The paper finds "no noticeable difference" at 1B and "only a slight increase" at 8B (§5, Appendix D.1), contradicting Kirchenbauer et al. (2024). This is noted but not investigated—does the base DCLM corpus already contain enough discussion of popular books that the inserted passages are redundant? This finding deserves more than a passing mention.

- **PII type variation is observed but unexplained.** Figure 8 shows occupation, email, and UUID have distinct memorization patterns (e.g., occupation is harder to extract; email is easier). The paper attributes some effects to position in the biography template but does not fully analyze whether semantic predictability or token-level rarity drives the differences. Understanding *why* certain PII types are more vulnerable would substantially increase the privacy implications.

- **MinK%++ underperformance on highly duplicated members is flagged but uninvestigated.** Table 1 shows MinK%++ achieves lower AUC than simple Loss on 256× duplicated examples (0.949 vs. 1.0 for Gutenberg Unpopular). This is noted as "surprising" but not analyzed. Since MinK%++ is the most effective attack at lower duplication levels, understanding its failure mode at high duplication could reveal important properties of the benchmark or the attack.

- **ELLie dataset has a confound that invalidates its use for dilution analysis.** Appendix D.3 acknowledges that ELLie examples are minimal pairs sharing first sentences, causing 0-duplicate examples to show high accuracy because sibling examples were inserted at higher duplication levels. This design flaw was identified but ELLie was not excluded from the core perturbation set, which could mislead users who do not read the appendix carefully.

### Trivial:

- The 8B model uses 36 layers instead of Llama 3's standard 32 (justified by GPU utilization), and the OLMo tokenizer replaces Llama's. These minor deviations from the base architecture are documented but slightly reduce direct comparability to Llama-based results elsewhere.

## Nice-to-Haves

- **Intermediate corpus sizes** (e.g., 200B, 300B tokens) to establish a scaling curve for dilution, which would transform the finding from a binary observation into a quantitative relationship.

- **8B timing runs** to validate that the ordering effect persists at larger model scales, or documentation of why they were infeasible.

- **Mechanistic analysis of the ordering/forgetting effect**—even speculative discussion of whether this resembles catastrophic forgetting, optimizer dynamics, or a primacy effect would make the recommendation more actionable and less purely empirical.

- **Comparison of artificial vs. natural duplicate memorization** to validate that the splicing insertion method (Figure 1) produces memorization dynamics similar to naturally occurring duplicates in web data.

- **Error analysis of unlearning failures** identifying which examples or PII types resist unlearning, to help the community diagnose whether the limitation is methodological or fundamental.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Temporal distribution confounds in MIA evaluation (from transferred weaknesses):** This criticism applies to WikiMIA (which uses temporal splits), not HUBBLE. HUBBLE explicitly designs around this by using randomized duplication for member/non-member splits (§6.1). The criticism is factually wrong for this paper's design.

- **Benchmark susceptibility to future data contamination:** While a valid concern for any released benchmark, HUBBLE's primary value is as a controlled research environment with known training data, not as a forever-evaluation benchmark. The randomized insertions are the key feature, and future models trained on HUBBLE's data would simply be detectable as contaminated by design. This is inherent to any fully open benchmark and not a unique flaw.

- **Key results in appendix / fragmented narrative:** This is a presentation/formatting concern. The rules state to remove pure formatting nitpicks.

- **Inconsistent terminology ("perturbed models" vs. "Hubble models"):** Formatting/style nitpick.

- **Citation formatting inconsistencies:** Formatting nitpick.

- **Decontamination manual inspection protocol too vague:** The paper provides decontamination details in Appendix A.3 with a two-phase procedure. Demanding more granular protocol details is a nitpick about reproducibility of a preprocessing step.

- **Reproducibility concerns about compute costs:** The hard rules state to remove nitpicks about reproducibility such as large artifacts impractical to include. The open release of all artifacts addresses this.

- **Unlearning benchmark negative results as a weakness:** That no current method succeeds is a feature of the benchmark—it reveals a genuine limitation in the field. This is not a weakness of the paper.

- **Test set contamination design is "too strong" (answers appended):** This is by design to establish upper bounds on memorization, consistent with the paper's stated goal of measuring worst-case risk.

- **Missing related works:** Hard rule—do not mention missing related works.

## Novel Insights

The ELLie contamination issue reveals a subtle but important design pitfall for memorization benchmarks: when perturbation examples share structure (e.g., same first sentence in minimal pairs), random assignment to different duplication bins creates cross-contamination that invalidates the 0-duplicate control condition. This is a generalizable lesson for any benchmark that uses randomization to create member/non-member splits—deduplication must account for shared substructure, not just exact matches. Separately, the finding that popular and unpopular books are memorized similarly challenges the intuitive "data density" hypothesis and suggests that in controlled settings with decontaminated base corpora, the base corpus may not provide the indirect exposure that makes popular texts more memorizable in observational studies. This hints that the data density effect observed in prior work may operate through base-corpus confounds rather than intrinsic properties of popular text.

## Suggestions

- Recalibrate language: replace "determined by" with "strongly influenced by," and "does not memorize" with "shows substantially reduced memorization." These changes preserve the findings' impact while matching evidence strength.

- Add a brief paragraph (even 3–4 sentences) discussing practical constraints of the best practices: when dilution is computationally infeasible, and what practitioners can do when sensitive data cannot be identified in advance for ordering.

- Investigate and explain the MinK%++ underperformance at high duplication—even a hypothesis (e.g., the method's calibration assumptions break down for near-perfectly memorized examples) would strengthen the MIA benchmark discussion.

- Add a warning label or exclusion of ELLie from dilution analyses in any documentation or code, since the confound invalidates its use for that purpose.

## Axis Evaluations

- **Novelty:** Moderate. The controlled perturbation insertion framework is the key innovation over prior suites (Pythia, OLMo). The empirical findings largely confirm prior observations (dilution generalizes Bordt et al. 2025; ordering is consistent with More et al. 2025; larger models memorize more is consistent with Tirumala et al. 2022) but in a more controlled and multi-domain setting. The MIA and unlearning benchmark constructions are useful extensions.

- **Technical soundness:** Good. The methodology is careful—decontamination, randomized duplication, interference checks, and standard vs. perturbed model comparisons provide solid experimental controls. The main issues are claim calibration (overstating findings) and limited experimental conditions (two corpus sizes, 1B-only timing runs).

- **Empirical support:** Adequate for dilution (clear effect across domains and model sizes) and ordering (clear effect at 1B), but insufficient for establishing general scaling relationships. The domain-specific analyses are thorough but largely relegated to the appendix.

- **Significance:** High as a community resource. HUBBLE fills a genuine gap between synthetic small-model studies and observational large-model studies. The benchmarks address known methodological issues in MIA and unlearning evaluation. The practical impact of the best practices is tempered by their feasibility constraints.

- **Clarity:** Good. The paper is well-organized with clear section structure. The main weakness is that several critical figures (timing runs, 1B comparisons, interference checks) appear only in the appendix, which fragments the empirical narrative for readers evaluating core claims.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept
