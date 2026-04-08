=== CALIBRATION EXAMPLE 44 ===

# Final Consolidated Review
## Summary

HUBBLE is a fully open-source suite of LLMs (1B and 8B parameters, trained on 100B or 500B tokens) designed to enable controlled scientific study of LLM memorization. Standard models are pretrained on a DCLM English corpus, while perturbed models add controlled insertions of sensitive text (book passages, synthetic biographies, test sets) at randomized duplication levels (0×–256×). The core empirical findings are two best practices: diluting sensitive data by training on larger corpora reduces memorization, and ordering sensitive data early in training leads to forgetting. The release also establishes benchmarks for membership inference attacks (HUBBLEMIA) and machine unlearning (HUBBLEUNLEARNING).

## Strengths

- **Controlled experimental design enabling causal inference about memorization.** By randomizing which perturbations are inserted and at what duplication rates, HUBBLE allows measurement of causal quantities (e.g., duplicates required to memorize a specific passage) that are impossible to estimate from observational studies of production models. This is a genuine methodological advance over prior work that either studied small synthetic models or made uncontrolled observations of large models.

- **Comprehensive, policy-relevant perturbation design spanning three risk domains.** The perturbations cover copyright (passages, paraphrases), privacy (biographies, chats), and test set contamination (standard and new test sets), grounded in a survey of the memorization literature. This breadth makes HUBBLE a holistic resource rather than a narrow benchmark, and the policy framing (e.g., safe harbors, GDPR rights) connects empirical findings to practical regulatory questions.

- **Fully open-source release with all components.** All models, training configurations, checkpoints, perturbation datasets, and evaluation code are publicly released. The explicit accounting of GPU hours (Appendix B.3) and detailed training setup (Table 4) support community reuse. This follows and extends the model suite tradition of Pythia and OLMo with a memorization-specific focus.

- **Practical best practices with empirical support.** The dilution finding (memorization increases slower with duplication when trained on larger corpora) and ordering finding (early-inserted data is forgotten without continued exposure) offer actionable guidance for practitioners. The interference check (Figure 20) showing minimal cross-domain interference increases confidence in the multi-domain setup.

- **Soundly designed MIA benchmark addressing confounders in prior work.** HUBBLEMIA avoids the spurious temporal features that undermined WIKIMIA (Duan et al., 2024) by using randomized duplication assignments. The benchmark reveals that MIAs achieve near-random performance on singly-duplicated members (AUC ~0.54), confirming and extending existing understanding of MIA limitations with controlled data.

## Weaknesses

- **The ordering best practice is validated only on 1B models, not 8B.** The paper explicitly shows that 8B models memorize more readily at lower duplication levels (Figure 19), yet the six timing runs that establish the "ordering" recommendation are all 1B models trained on 100B tokens. Since model scale affects memorization strength, the generalizability of the ordering finding to the 8B setting—and by extension to larger models—is unsupported by the current experiments. This is a significant gap for what is presented as a general best practice.

- **Scale gap limits confidence in generalizability to production models.** The largest HUBBLE models are trained on 500B tokens with 8B parameters, while the paper notes Llama 3 is trained on 15T+ tokens—a ~30× gap. The dilution finding is established by comparing 100B vs. 500B tokens, a 5× difference, whereas production scenarios involve corpora orders of magnitude larger. Whether the quantitative memorization thresholds and the effectiveness of dilution transfer to 15T-token training regimes remains unknown. The paper acknowledges this gap but does not discuss how findings might extrapolate.

- **Ecological validity of synthetic and controlled perturbations is limited.** The YAGO biographies are templated text populated by sampling from a knowledge base (Appendix A.1), and PersonaChat dialogues use randomly assigned usernames. These are structurally very different from how PII naturally appears in web-scraped corpora (embedded in varied contexts with heterogeneous formats). The near-100% PII reconstruction at 16 duplications on YAGO biographies may overestimate real-world privacy risks for organically distributed PII. Similarly, perturbation insertions are always surrounded by EOS tokens and never broken across sequences (Figure 1), which differs from how regular documents are processed and could create structural artifacts.

- **ELLie test set contains minimal pairs that invalidate dilution analysis for that dataset.** Section D.3 acknowledges that "examples in ELLie are minimal pairs" with the same first sentence placed in different duplication bins, meaning models achieve high accuracy on examples duplicated 0×. The paper honestly notes "This invalidates the use of ELLie for studying dilution," but this finding is buried in the appendix rather than discussed in the main text. This is a data quality issue that affects one of the "new test sets" prominently featured in Section 2.3, and raises the question of whether similar structural issues affect other perturbation datasets.

- **Dilution claim is supported by only two corpus sizes.** The core dilution finding compares models trained on 100B vs. 500B tokens. With only two data points, the relationship between corpus size and memorization rate could be non-monotonic or exhibit different scaling behavior at intermediate or larger sizes. The paper generalizes this as a broad principle, but the empirical support is limited to a single 5× comparison.

- **Interference check is at the domain level only, not the example level.** Figure 20 shows that the full perturbed model matches single-domain models at the domain level, but this does not rule out example-level interference—for instance, a highly duplicated test set example affecting memorization of a nearby biography within the same training sequence. The paper acknowledges that "exhaustively characterizing such interference...would be impractical" (Section 4), but the claim of "minimal interference" is stronger than the domain-level evidence supports.

- **Key figures lack confidence intervals or statistical tests.** The memorization curves in Figures 2, 13, and 14 do not include error bars or confidence intervals, making it difficult to assess whether observed differences (e.g., between 100B and 500B models at low duplication levels, or between timing conditions) are statistically significant. Given the small perturbation fraction (0.08% of the corpus), statistical uncertainty could be non-trivial.

## Nice-to-Haves

- Combined dilution + ordering experiments to verify the two best practices are complementary rather than interacting unexpectedly.
- At least one intermediate corpus size (e.g., 250B tokens) to establish monotonicity of the dilution effect.
- Comparison of dilution/ordering against established mitigations like deduplication or differential privacy training.
- 8B-scale timing runs to validate the ordering finding at a scale where memorization is stronger.
- Example-level interference analysis (e.g., examining whether highly duplicated examples in one domain affect low-duplication examples in the same training batch).
- Intermediate checkpoints for all core models (currently only timing runs have them) to enable broader memorization dynamics research.
- Concrete extracted text examples at each duplication level to demonstrate practical memorization risk beyond aggregate metrics.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Fully open-source" claim questioned due to OLMo tokenizer/untied embeddings.** The harsh critic argued the models are not "faithful Llama reproductions" due to architecture modifications. This misreads the claim: "fully open-source" means all components are publicly released, not that the architecture is identical to Llama 3. The modifications are clearly documented in Section 3.2 and Table 4.

- **Copyright perturbations use public domain Gutenberg texts, not copyrighted books.** The critic argued this is a "significant mismatch." However, the paper never claims the inserted texts are copyrighted; Section 2.1 explicitly calls them "open-domain texts" inserted "to study the measurement and mitigation of LLM memorization on books and articles." The copyright section studies memorization *relevant to* copyright risk, using book-like passages as proxies.

- **Chinchilla comparison uses nominal "1B/8B" rather than actual 1.2B/8.3B parameter counts.** This is a minor naming convention issue. The paper uses "1B" and "8B" as model designations throughout, and the actual parameter counts are listed in Table 4 and evaluation tables.

- **Compute barriers (200k GPU hours) undermine "open-source" practical meaning.** The checkpoints and all training data are released, so the resource is usable without retraining. The compute cost of training is a reality of large model research, not a deficiency of this paper's openness.

- **Multilingual limitation.** The paper scopes its study to English pretraining on the DCLM corpus. Criticizing the absence of multilingual analysis is scope creep.

- **Negative societal impact / misuse concerns.** Generic one-size-fits-all concern. The models contain only synthetic PII and public-domain texts, and the paper's explicit purpose is to help *mitigate* memorization risks.

- **Standard model PII inference "undermines perturbation-specific memorization claims."** The fact that standard models achieve non-trivial PII inference accuracy (Figure 8) is actually an important *finding*—it shows models learn associations from the base corpus. It does not undermine the perturbation analyses, which compare standard vs. perturbed models and measure the *additional* memorization from inserted data.

- **Unlearning "all methods fail" as a weakness.** The paper honestly reports this negative result, which is itself a contribution—demonstrating that current unlearning methods cannot precisely remove pretraining data without degrading related knowledge. The benchmark's value lies in enabling future method development.

- **MinK%++ anomaly not explained.** This is a missed analysis opportunity but not a methodological flaw. Moved to nice-to-have.

## Novel Insights

The ELLie minimal-pair issue reveals a subtle but important design challenge for memorization benchmarks: standard deduplication procedures based on exact n-gram matching can miss structural overlaps (shared first sentences with different queries) that create cross-duplication-bin information leakage. This suggests that future memorization benchmark designs need structural deduplication checks that go beyond surface-level string matching—particularly for tasks that use minimal-pair or contrastive evaluation formats. Additionally, the finding that standard (non-perturbed) models already achieve non-trivial PII inference accuracy from corpus-level associations alone raises a deeper question: in real-world settings, the *incremental* privacy risk from inserting specific PII-containing documents may be smaller than the *baseline* risk from associations already learned from the general corpus, suggesting that memorization mitigation efforts should consider not just whether data was seen, but whether the model could have inferred the same information from other sources.

## Suggestions

- Run at least one 8B timing experiment (e.g., inserting perturbations only in the first quarter vs. last quarter of 500B-token training) to validate the ordering best practice at the larger scale where memorization is demonstrably stronger.
- Add error bars or confidence intervals to the core memorization curves (Figures 2, 14, 19), at minimum for the key dilution and timing comparisons where the paper draws its main conclusions.
- Move the ELLie invalidation discussion from Appendix D.3 to the main text (Section 5), since it directly affects the validity of one of the prominently featured "new test sets" and serves as an important cautionary note for future benchmark design.
- When presenting the ordering best practice, explicitly qualify that it has been validated only at 1B scale and its generalizability to larger models remains an open question.

---

**Axis Evaluations:**

- **Novelty:** Moderate-to-high. The controlled perturbation framework at 1B–8B scale is novel and fills a genuine gap between toy-model studies and observational analyses. However, the two core empirical findings (dilution, ordering) explicitly extend recent concurrent work (Bordt et al., 2025; More et al., 2025), and the primary contribution is the extension across domains rather than discovery of new phenomena.

- **Technical soundness:** Good with notable gaps. The experimental design is rigorous in its randomization and control structure, and the interference check provides some validation. However, the ordering finding lacks validation at 8B scale, key figures lack statistical uncertainty quantification, and the ELLie data quality issue was not caught before release.

- **Empirical support:** Adequate for the main claims but thin in places. The dilution effect is consistent across multiple tasks but supported by only two corpus sizes. The ordering effect is shown clearly at 1B but untested at 8B. The MIA and unlearning benchmarks provide useful baselines though the unlearning results are primarily negative.

- **Significance:** High as a community resource; moderate for empirical findings. HUBBLE fills an important infrastructure gap for memorization research, and the open-source release will likely become a standard testbed. The best practices (dilution, ordering) are practically useful but represent confirmatory extensions of existing evidence rather than paradigm-shifting discoveries.

- **Clarity:** Good. The paper is well-organized with clear separation of domain-agnostic and domain-specific results. The perturbation design and evaluation methodology are thoroughly documented. The main weakness is that key supporting figures (timing runs, interference check) are relegated to the appendix.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept
