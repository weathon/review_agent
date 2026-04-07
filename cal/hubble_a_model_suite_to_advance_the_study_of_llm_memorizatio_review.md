=== CALIBRATION EXAMPLE 87 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title is accurate and the abstract is generally well-written. The core claims are supported by the experiments. One concern: the abstract promises "two best practices for addressing memorization risks" (dilute and order), but these are essentially confirmations of already-suspected phenomena rather than genuinely new prescriptions. The abstract would benefit from being more honest that the findings *formalize* prior intuitions rather than discover them. The statement that "a password appearing once in a smaller corpus is memorized better than the same password in a larger corpus" is technically the definition of relative frequency and risks sounding tautological to sophisticated readers.

---

### Introduction & Motivation (§1)

The positioning of HUBBLE on a spectrum between "controlled studies of smaller models" and "observational studies of large pretrained models" is cogent and well-motivated. The gap is real. The claim that "most causal quantities on memorization are impossible to estimate" in observational studies is accurate and persuasive.

**Concern:** The introduction emphasizes the policy relevance of HUBBLE (copyright law, GDPR, safe harbors), but the models trained on ≤500B tokens are roughly 30× smaller than production models like Llama-3 (15T tokens). This gap is acknowledged in §3 but is not adequately addressed in the introduction—the policy-relevance framing risks over-selling findings that may not transfer to the scales where the policy stakes actually lie. The paper never explicitly argues (with evidence or theory) why memorization findings from 500B-token models should inform copyright or GDPR policy as applied to models trained at 15T+ tokens.

---

### Perturbation Design (§2)

This section is the strongest in the paper. The literature survey is comprehensive, and the choice of perturbations is well-motivated across the three risk domains. Using both popular and unpopular Gutenberg books to test the data-density hypothesis is clever. ECtHR biographies complementing synthetic YAGO biographies is a thoughtful design choice.

**Concern 1 (Privacy domain – YAGO UUIDs):** The YAGO biographies include randomly generated UUIDs as one of the 9 PII attributes (Table 8, referenced but not in the read portion). Memorizing a UUID is closer to memorizing a random string than memorizing real PII with social significance. The authors should clarify whether UUID memorization behaves differently from other PII types (e.g., nationality, occupation) and whether its inclusion inflates or deflates the practical significance of the privacy results. This distinction matters for policy claims about PII leakage.

**Concern 2 (Paraphrases):** MRPC and PAWS paraphrases are used to test whether models prefer the inserted version over the held-out version. The evaluation (§3.3) measures loss-based preference. However, both paraphrases in a pair were presumably present in the pretraining corpus at different frequencies before decontamination. The decontamination removes matches, but it removes matching *documents* rather than individual sentences—it is not clear the held-out paraphrase was truly absent from training. The paper should discuss this more carefully.

**Concern 3 (Test sets – ELLie/MUNCH):** The paper describes these as "new test sets created after the DCLM dataset cutoff," which reduces unintended contamination. However, ELLie (Testa et al., 2023) was released at ACL 2023, which may fall close to or within the DCLM crawl window. The paper should be more explicit about the DCLM cutoff date and verify both datasets postdate it.

---

### The HUBBLE Suite (§3)

**Pretraining data (§3.1):** The decontamination procedure is sound and well-described. Removing 7,540 documents (<0.002% of total) is reasonable. The choice to use a *random* subset of the DCLM pool (not the highest-scoring documents) rather than the official DCLM-BASELINE subset is appropriate for generalizability, and the paper is transparent about this trade-off.

**Insertion procedure (§3.1):** Figure 1 explains that perturbations are inserted as whole units, never broken across sequences, with at most one per sequence. This is a methodological choice that departs from how sensitive data naturally appears in the wild—real copyrighted content often spans multiple sequences, may appear partially within a document, and appears in many documents at once. The implication is that the experimental setting may underestimate memorization of naturally occurring sensitive data (since real data could appear across more contexts). This limitation is not discussed.

**Models (§3.2):** The 2×2×2 factorial design for the core models is clean and appropriate. 

**Concern:** The 8B model uses 36 layers instead of the standard 32 in Llama-3.1-8B, to "maximize GPU utilization." This is a hardware-motivated deviation that makes the architecture non-standard and slightly complicates comparisons with Llama-3 results cited elsewhere. The paper does not discuss whether this depth modification could systematically affect memorization behavior, which is the object of study.

**Duplication levels {0, 1, 4, 16, 64, 256}:** This grid is justified by initial 1B experiments showing "a range of memorization," but there is no theoretical or empirical argument for why this particular grid (roughly powers-of-4 with some asymmetry) is the right choice. For example, the gap between 64 and 256 is large—finer resolution in this range might reveal non-linearities that are currently invisible.

**Evaluations (§3.3):** The three evaluation modes (loss, loss-based choice, generative) are appropriate and well-described. The acknowledgment that evaluations "establish lower bounds" on memorization is epistemically honest. However, the paper does not evaluate stronger, model-access-informed attacks (e.g., beam-search extraction, beam search with boosted temperature), which would reveal whether the lower bounds are tight.

---

### Domain-Agnostic Results (§4)

**Dilution finding:** The main finding—that training on a larger corpus at the same absolute number of perturbation insertions reduces memorization—is intuitive and largely follows from the definition of relative frequency. The paper correctly situates this as formalizing and extending Bordt et al. (2025) and Kandpal et al. (2022). The result is convincing but incremental as a scientific finding.

**Concern:** The practical "best practice" of "dilute sensitive data by increasing the size of the training corpus" is not operational advice. In practice, a company cannot choose to train on an arbitrary amount of extra data just to dilute a few sensitive examples. A more useful operationalization might be: given a fixed corpus with sensitive data, what is the maximum tolerable absolute frequency of the sensitive data under different corpus sizes? The paper provides the data to answer this but does not synthesize it into actionable guidance.

**Ordering finding:** The timing results show that data seen only in the first quarter of training is largely forgotten by the end, while data seen only in the last quarter is more memorized. This is directionally consistent with prior work (Jagielski et al., 2023; More et al., 2025 as cited). The practical recommendation to "order sensitive data early" is interesting but problematic: in real training pipelines, ordering is typically constrained by data availability and shuffle protocols, not by deliberate choice of when to expose sensitive data.

**Interference check:** Three 1B models trained on single-domain perturbations match the all-domain model on the corresponding domain. This is a necessary sanity check, but it is acknowledged as a domain-level check only. The paper correctly notes that exhaustive interference characterization would be impractical, but the current check does not rule out subtler cross-domain interference effects (e.g., whether high-duplication test-set contamination affects the forgetting dynamics of biographical PII).

**Larger models memorize at lower duplications (Figure 19):** This is consistent with Tirumala et al. (2022) and is an important confirmation at production-relevant scales. However, the 8B models here are still far from production scale (Llama-3 at 8B is trained on 15T tokens, vs. HUBBLE's 500B). The paper does not attempt to extrapolate or bound how this relationship continues.

---

### Domain-Specific Results (§5 and Appendix D)

**Copyright:** The finding that the metric choice (loss vs. k-eidetic) affects whether memorization is detected is genuinely valuable for ongoing copyright debates. The divergence between loss-based and k-eidetic results at 4 duplicates for Wikipedia 8B (100B) is a concrete example that researchers and policymakers should take seriously.

**Concern:** The finding that "popular and unpopular books are memorized similarly" is surprising given the data-density hypothesis (Kirchenbauer et al., 2024). The paper attributes the small difference to the fact that DCLM is already somewhat deduplicated. But popular Gutenberg books are well-represented on the web and in DCLM even after deduplication—the explanation that DCLM's deduplication erases the density difference is plausible but not verified. This finding contradicts some assumptions in the copyright literature and deserves more investigation.

**Privacy (YAGO biographies):** The result that "attack accuracy on the Hubble 8B (100B) perturbed model is close to 100% with just 16 duplications" is alarming and practically significant. However, the YAGO biographies use synthetic, fictional persons—the attributes are conditionally sampled to appear plausible but do not correspond to real individuals. Whether these results transfer to memorization of real PII (which may be embedded in much more complex web contexts) is an open question the paper does not address.

**Test set contamination:** The finding that "memorizing test set examples does not translate into generalization on that task" (and that contamination can even hurt performance on minimal pairs) is the most scientifically novel result in the paper. This is surprising and counter-intuitive, and has direct implications for the contamination-detection literature. It deserves more prominence in the main text rather than being buried in §5 and Appendix D.

---

### Use Cases (§6)

**HUBBLEMIA:** The benchmark addresses a real weakness in WIKIMIA (temporal confounders). The controlled experimental design with known member/non-member status is a genuine methodological advance. The result that MinK%++ paradoxically underperforms simpler methods at high duplication (AUC < 1.0 when Loss and MinK% achieve 1.0) is interesting and the paper does not fully explain it—this would benefit from more analysis.

**Concern (non-member selection):** Members are defined as perturbations duplicated >0 times, and non-members are perturbations duplicated 0 times. This means members and non-members are drawn from the same distribution of texts—which is a strength for avoiding spurious features. However, perturbations duplicated 0 times were still candidates for insertion (chosen not to be inserted). If the selection of what to insert vs. hold out is correlated with any text properties (e.g., length, perplexity), this could introduce subtle confounders. The paper should clarify whether the 0× duplication assignment was random with no selection bias.

**HUBBLEUNLEARNING:** The setup is clean, with the standard model as an oracle for "desired" post-unlearning performance. The finding that all three methods fail to achieve precision unlearning (degrading the Keep set as well as the Unlearn set) is practically significant. The use of WikiText as the retain set (following prior work) is reasonable but introduces a distribution shift—the authors test with the in-distribution Keep set in Appendix G and note patterns are "consistent," which is reassuring.

**Concern:** Only three unlearning methods (RMU, RR, SatImp) are benchmarked. More recent gradient-based methods or methods specifically designed for pretraining unlearning are not included. The paper frames this as a "case study" rather than an exhaustive benchmark, which is acceptable, but the selection justification is thin.

---

### Discussion & Conclusion (§7)

The paper appropriately identifies three research questions HUBBLE can help address. The aspiration to become "an anchor point" for the memorization community is reasonable given the scale of the release.

**Missing discussion:** The paper does not discuss the ethical implications of releasing detailed memorization benchmarks that include personal biographical data (ECtHR court cases). While the data is public, using it as a training target for evaluating PII extraction attacks could facilitate the development of more powerful privacy attacks against real production models. This is worth at least a brief acknowledgment.

**Missing limitation:** The models are pretrained from scratch, not instruction-tuned. Modern production models undergo RLHF and instruction tuning, which may affect memorization dynamics. The degree to which HUBBLE's findings apply to RLHF-tuned models is not discussed.

---

### Writing & Clarity

Section 5 is overly brief—the domain-specific results are described in 6 short paragraphs for three domains, each summarizing findings that are detailed in appendices. For an ICLR submission, the key novel findings (particularly the test-set contamination ≠ generalization result and the metric-dependence finding) deserve more direct exposition in the main body. Readers should not have to navigate to Appendix D to understand the paper's substantive scientific contributions.

---

### Overall Assessment

HUBBLE is a well-executed infrastructure paper that makes a genuine and valuable contribution: it provides the community with the first suite of academically-trainable LLMs with *controlled* perturbations for studying memorization causally. The release is comprehensive (models, datasets, code, checkpoints), and the perturbation design is thoughtful across three policy-relevant risk domains. The domain-agnostic findings (dilution, ordering, scale effects) are largely confirmations of prior intuitions at new scales rather than surprises, but the formal causal identification they enable is a real methodological advance. The most novel scientific finding—that test set contamination does not consistently improve generalization and can hurt it on minimal pairs—is somewhat buried in the paper and deserves more prominence. Key weaknesses include: (1) the 30× scale gap between HUBBLE and commercial models limits the policy-relevance of the stated best practices, which the paper somewhat overstates; (2) the practical operationalization of "dilute" and "order" as best practices is underdeveloped; (3) some design choices (YAGO UUIDs, insertion as atomic units, the duplication grid) receive insufficient justification. Nonetheless, the artifact contribution is substantial and should be valuable to the research community regardless of the novelty of the individual findings. The paper is appropriate for ICLR's datasets and benchmarks track, and the contribution clears the bar for acceptance, though the paper would be strengthened by foregrounding its most novel empirical results and being more careful about the scope of its policy claims.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents HUBBLE, a fully open-source suite of LLMs designed to enable controlled scientific study of memorization risks across copyright, privacy, and benchmark contamination. By systematically varying corpus size and the timing/frequency of inserted sensitive data, the authors establish empirical best practices for mitigating memorization, specifically demonstrating that diluting data and ordering it early in training reduces risk. The release provides a comprehensive, reproducible benchmark for membership inference and unlearning research, bridging the gap between controlled toy studies and observational analyses of frontier models.

### Strengths
1.  **High Reproducibility and Openness:** Unlike many safety studies that rely on black-box commercial APIs, HUBBLE releases all models, training code, checkpoints, and perturbations publicly (Section 3, link in Abstract). This aligns perfectly with ICLR values on open science and allows others to audit the memorization claims directly.
2.  **Systematic Causal Design:** The paper moves beyond observational studies (which struggle to disentangle complexity from frequency) by using controlled insertions. The factorial design in the "Core" experiments (1B/8B × Standard/Perturbed × 100B/500B tokens) and the "Timing" runs allow for precise causal inference on memorization dynamics (Section 3.2, Section 4).
3.  **Actionable Policy-Relevant Findings:** The empirical results offer specific, interpretable mitigation strategies. The paper clearly demonstrates that memorization scales predictably with duplication and corpus size (Figure 2), suggesting "dilution" and "ordering" as valid risk reduction practices that are practical for developers to implement.

### Weaknesses
1.  **Scale Gap relative to Frontier Models:** While 500B tokens is robust for academic study, it is significantly smaller than frontier models (e.g., Llama-3 at 15T+ tokens per Section 3). The authors acknowledge this, but the memorization mechanisms and scaling laws for unlearning may differ substantially at the 1T+ token scale, limiting the direct applicability of findings to production systems.
2.  **Reliance on Synthetic Privacy Data:** The privacy perturbations heavily rely on templated YAGO biographies (Section 2.2). While useful for control, synthetic data may not exhibit the same distributional complexity or correlation structures as real-world web-scraped PII, potentially underestimating leakage risks in realistic scenarios.
3.  **Benchmark Validity Issues:** The analysis of the ELLie test set reveals a structural confounder where examples share minimal pairs with different duplication rates (Appendix D.3). This makes the "dilution" results for ELLie invalid without careful adjustment, highlighting a difficulty in designing contamination studies for certain dataset types.

### Novelty & Significance
*   **Novelty:** The primary novelty lies in the *resource* and *benchmarking framework*. While the concept of data dilution is not new (cited as Bordt et al., 2025), the execution of a full Llama-based suite with systematic domain-specific perturbations comparable to the scale of Pythia and Olmo is a significant contribution. It transforms memorization study from a theoretical or small-scale endeavor into a reproducible, industrial-grade experiment.
*   **Significance:** The work addresses a critical safety bottleneck: understanding when and why models memorize private or copyrighted data. By providing a "testbed" (Section 6), it facilitates further research in unlearning and membership inference, potentially lowering the barrier for community-wide audits of LLM safety.
*   **Clarity:** The paper is well-structured, with clear separation between the design of perturbations, the training setup, and the evaluation metrics.
*   **Reproducibility:** Excellent. Full transparency regarding decontamination (Appendix A.3) and random seeds ensures others can replicate the core experiments.

### Suggestions for Improvement
1.  **Extrapolation Analysis:** To increase impact on frontier research, include a discussion or limited experiments on how memorization metrics might scale relative to frontier models (e.g., using scaling law extrapolations from 1B/8B to larger parameter counts).
2.  **Real-World PII Validation:** Complement the YAGO synthetic data with a small subset of real, anonymized PII (e.g., from common crawl dumps) to validate that the leakage patterns hold for non-synthetic distributions.
3.  **Benchmark Curation:** For the "New Test Sets" section, specifically address the minimal pair confounders in ELLie more robustly in the main text to ensure future users do not inadvertently draw similar invalid conclusions about contamination metrics.
4.  **Mechanism Interpretation:** While the paper establishes *that* ordering sensitive data early reduces memorization (Section 4), adding a brief interpretability analysis (e.g., logit lens or attention analysis) on *why* this happens (Section 7 invites this, but a preview would strengthen the paper) would add depth.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **8B timing experiments** — The ordering recommendation (a core contribution) is only validated on 1B models, while dilution uses 8B. Without 8B timing runs, the claim that ordering is a general best practice is unsupported at the primary model scale.

2. **Statistical significance testing on core claims** — The dilution and ordering effects lack p-values, confidence intervals, or bootstrap estimates. ICLR requires statistical rigor; without it, observed trends could be noise from limited model runs.

3. **Validation that perturbations behave like natural memorization** — No experiment shows synthetic biographies/book passages memorize similarly to real PII/copyrighted text in the wild. This undermines whether HUBBLE findings generalize to actual deployment risks.

4. **Quantitative unlearning benchmark comparison** — Unlearning results are shown only against an internal "desired target." Must compare to TOFU/MUSE benchmarks with standard metrics (ROUGE, utility retention) to establish whether HUBBLE reveals anything new about unlearning.

5. **Cross-architecture validation** — All models use Llama architecture. One additional architecture (e.g., GPT-2 style or OLMo) at 1B scale would test whether findings are architecture-specific or general LLM phenomena.

### Deeper Analysis Needed (top 3-5 only)
1. **Mechanism of dilution** — The paper claims larger corpora reduce memorization but provides no analysis of why (relative frequency? gradient noise? convergence dynamics?). Without mechanistic insight, the finding is observational, not explanatory.

2. **Quantitative interference bounds** — The "minimal interference" claim relies on qualitative plot matching across 3 single-domain models. Need quantitative metrics (e.g., correlation coefficients, KL divergence) to substantiate that multi-domain training doesn't confound results.

3. **PII attribute variation explanation** — Emails and UUIDs memorize differently than occupations, but no analysis explains why. This limits practical privacy guidance—if some PII types resist mitigation, practitioners need to know which.

4. **Forgetting dynamics characterization** — The timing experiments show early data is forgotten, but don't quantify the forgetting rate or identify what determines retention vs. forgetting. This limits the ordering recommendation's actionability.

5. **Synthetic vs. real PII gap analysis** — YAGO biographies are templated with conditional sampling. Need analysis of how these differ from real PII distributions and whether memorization patterns transfer.

### Visualizations & Case Studies
1. **Concrete memorization examples with confidence intervals** — Show actual text passages that are/isn't memorized at different duplication levels with error bars. Current plots show aggregate metrics without revealing what memorization looks like in practice.

2. **Unlearning failure case visualization** — Figure 3 shows methods don't reach the target, but no examples show what the model actually outputs after unlearning. Are failures due to over-generalization or incomplete removal?

3. **Layer-wise memorization localization** — No analysis of which layers store memorized information. This is critical for interpretability claims and would enable targeted mitigation strategies.

4. **Forgetting curves with training loss overlay** — Figure 13 shows memorization decay but doesn't correlate with general training loss. Need to show whether forgetting is specific to perturbations or part of broader optimization dynamics.

### Obvious Next Steps
1. **8B timing runs should have been included** — Given the 200k GPU hours already spent, adding six 8B timing models (~45k GPU hours) was feasible and would have made the ordering claim credible at scale.

2. **Real PII validation on a subset** — Even one experiment with real leaked PII (e.g., from known data breaches) would ground the synthetic biography findings in actual privacy risk.

3. **Standardized unlearning metrics** — Report forget quality, utility retention, and neighborhood preservation using established unlearning evaluation frameworks rather than custom plots.

4. **Decontamination audit** — The paper claims <0.002% document removal but doesn't show what was removed or verify no accidental contamination remains in the "standard" models.

# Final Consolidated Review
## Summary

HUBBLE presents a suite of fully open-source LLMs (1B and 8B parameters, trained on 100B or 500B tokens) designed for controlled scientific study of memorization risks. The key innovation is systematic insertion of perturbation data (copyrighted passages, synthetic biographies with PII, test sets) at controlled duplication levels and training phases, enabling causal identification of memorization dynamics. The paper establishes two main findings: (1) dilution—training on larger corpora reduces memorization of sensitive data at fixed absolute frequency; (2) ordering—sensitive data inserted early in training is more likely to be forgotten. The release includes benchmarks for membership inference attacks and machine unlearning.

## Strengths

- **First comprehensive infrastructure for causal memorization study.** HUBBLE fills a genuine gap between small-scale controlled studies and observational analyses of frontier models. The controlled perturbation design with known duplication counts enables causal inference that is impossible with naturally-occurring data. The factorial design (model size × standard/perturbed × corpus size) allows systematic investigation of memorization determinants.

- **Thoughtful perturbation design across three risk domains.** The literature survey in §2 is comprehensive, and the design choices are well-motivated: popular/unpopular Gutenberg books to test data-density effects, synthetic YAGO biographies plus real ECtHR court cases for privacy, multiple contamination formats (infill vs MCQ) for test sets. The inclusion of paraphrase variants and timing experiments demonstrates attention to relevant research questions.

- **Novel empirical finding on test set contamination.** The result that "memorizing test set examples does not translate into generalization on that task" (§5, Appendix D.3) and can even hurt performance on minimal pairs is counter-intuitive and has direct implications for contamination detection research. This finding deserves more prominence than it receives.

- **Metric dependence for copyright interpretation.** The observation that loss-based measures detect memorization at lower duplication levels than k-eidetic measures (Appendix D.1) has practical significance for copyright debates—numerical measures of memorization are metric-dependent and cannot be interpreted in isolation.

- **Validated unlearning benchmark with known ground truth.** HUBBLEUNLEARNING provides a clean setup for evaluating unlearning on pretraining data (distinct from fine-tuning unlearning in TOFU/MUSE), with the standard model serving as an oracle for desired post-unlearning behavior.

## Weaknesses

- **Timing experiments limited to 1B models.** The ordering finding—"sensitive data to appear early in training reduces memorization risks"—is validated only on 1B models (§3.2, Figure 14). The paper does not provide evidence that this result extends to the 8B scale, where memorization dynamics may differ. The dilution finding is validated at both scales, but ordering is not.

- **Scale gap from production models limits practical applicability.** The models are trained on at most 500B tokens, approximately 30× smaller than Llama-3 (15T tokens). While the paper acknowledges this, the stated "best practices" for copyright and privacy risk mitigation are not validated at scales where these stakes actually arise. Extrapolation to production scales is assumed rather than demonstrated.

- **Synthetic PII may not capture real-world privacy risks.** YAGO biographies use templated text with conditionally sampled but fictional attributes. While this enables controlled study, the paper provides no validation that memorization patterns transfer to real PII (which may be embedded in complex web contexts, have different attribute correlations, or exhibit different memorization dynamics). The finding that emails and UUIDs behave differently from other PII types (Figure 8) underscores that PII type matters for memorization, but this variation is not analyzed.

- **Insertion procedure differs from natural data distribution.** Perturbations are inserted as atomic units (Figure 1)—never broken across sequences, at most one per sequence—with controlled duplication. Real sensitive data appears in more varied contexts (partial documents, multiple co-occurring instances, spanning sequence boundaries). This may underestimate memorization of naturally-occurring sensitive data. The paper does not discuss this limitation.

- **Lack of statistical significance testing.** The main figures (Figure 2, Figure 14, etc.) show trends without confidence intervals, error bars, or significance tests. While Appendix A.2 mentions aiming for "small error bars" by inserting >1000 examples at low duplication levels, the plots do not visualize uncertainty. Observed differences could reflect noise from limited model runs (single training run per configuration).

- **ELLie benchmark has structural confounder.** Appendix D.3 acknowledges that "examples in ELLie are minimal pairs" sharing the same first sentence but assigned to different duplication bins, causing high accuracy on "unseen" examples. This invalidates ELLie for studying dilution, though the paper correctly identifies this limitation.

- **Key findings buried in appendices.** The test set contamination results (contamination doesn't improve generalization, can hurt minimal pairs) and metric-dependence findings for copyright are scientifically novel but relegated to §5 and Appendix D. Readers should not need to navigate appendices to understand the paper's substantive contributions.

## Nice-to-Haves

- 8B timing experiments to validate ordering at the primary model scale
- Validation that synthetic PII memorization patterns transfer to real PII distributions
- Standardized unlearning metrics (beyond custom visualizations) to facilitate comparison with prior work
- Mechanistic analysis of why dilution and ordering work (e.g., gradient noise, convergence dynamics)

## Removed Points

- *Demand for TOFU/MUSE benchmark comparison for unlearning.* HUBBLE's unlearning setting is distinct (pretraining vs. fine-tuning), and the paper correctly situates it. This is not a fair comparison.

- *Demand for cross-architecture validation.* Testing whether findings generalize beyond Llama architecture would be valuable but is scope creep for a benchmark paper.

- *Criticism that dilution finding is "tautological" or "definition of relative frequency."* The paper goes beyond relative frequency to show empirical memorization curves at specific duplication levels, enabling practitioners to estimate risk quantitatively.

- *Criticism that policy relevance is overstated.* The paper frames findings appropriately as scientific contributions with policy implications, not as regulatory guidance.

- *Demand for mechanistic interpretation of dilution/ordering.* Interesting suggestion, but not a core flaw of an empirical benchmark paper.

- *Demand for stronger extraction attacks.* The paper appropriately frames evaluations as "lower bounds" and acknowledges this limitation. Demanding more sophisticated attacks is a research direction, not a weakness.

## Novel Insights

The finding that test set contamination does not improve—and can even harm—generalization to minimal pairs (Appendix D.3) challenges the assumption that contaminated benchmark performance inflates perceived capabilities in straightforward ways. This suggests that contamination may introduce spurious correlations that interfere with genuine learning, a hypothesis that deserves systematic investigation. Additionally, the metric dependence for copyright interpretation (loss vs. k-eidetic memorization) highlights that legal or policy decisions based on single memorization metrics may be arbitrary—numerical thresholds cannot substitute for qualitative analysis of what counts as problematic reproduction.

## Suggestions

- Move the test set contamination findings and metric-dependence analysis from appendices to the main text. These are among the most novel scientific contributions and should be prominently featured.

- Add confidence intervals or error bars to main figures, or explicitly state in figure captions that results are from single training runs. Statistical uncertainty affects how readers should interpret claimed differences.

- Include a brief discussion of limitations in generalizing to production scales (15T+ tokens), particularly for the ordering finding which is only validated at 1B scale. Readers should understand where evidence ends and extrapolation begins.

- For the timing experiments, report results on intermediate checkpoints (not just final models) to characterize the forgetting curve more precisely. The current analysis in Appendix E.1 is helpful but could be expanded.

- Consider releasing a technical report specifically on the unlearning benchmark with standardized metrics, to facilitate comparison with future work in pretraining-data unlearning.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept
