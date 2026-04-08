=== CALIBRATION EXAMPLE 45 ===

# Harsh Critic Review
Now I have enough material to write a thorough review. Let me compose it.

---

## Section-by-Section Critical Review

### Title & Abstract

The title accurately describes the contribution. The abstract is well-structured and the claims are well-supported by the paper's content. One minor concern: the abstract presents "dilution" and "ordering" as novel best practices, but both are largely confirmatory of prior work (Bordt et al., 2025 for dilution; Jagielski et al., 2023 for forgetting; More et al., 2025 for late-training memorability). The abstract would benefit from more clearly scoping these as *confirmations at scale and across domains* rather than novel discoveries.

---

### Introduction & Motivation (§1)

The framing of a spectrum between controlled small-model studies and observational large-model studies is sharp and accurate. The motivation for HUBBLE as filling a gap is compelling. The three memorization risk domains (copyright, privacy, test set contamination) are well-chosen and tied to real-world legal and regulatory concerns.

**Concern:** The intro claims HUBBLE enables measurement of "causal quantities (e.g. the number of duplicates required to memorize a test set example)." This is somewhat overstated. Because duplication levels are randomized at the example level but not at the model level (there is one perturbed model per scale/size), the design identifies *correlational* trends between duplication level and memorization, not strict causal effects in the Pearl sense. There is no way to know whether a specific example would have been memorized at a different duplication level with the same training trajectory. The authors should hedge this language.

---

### Perturbation Design (§2)

The survey of memorization risks across copyright, privacy, and test set contamination is thorough and well-cited. The choice of data sources is generally principled.

**Concern 1 – Gutenberg as a copyright proxy.** Gutenberg texts are public domain, not copyrighted. The paper uses them to study copyright memorization. While this is a pragmatic choice (using copyrighted text would itself raise issues), the implicit assumption that memorization of public-domain text generalizes to copyrighted material is not tested. Popular copyrighted books often have richer cultural context in the pretraining corpus (reviews, citations, summaries) compared to public domain texts; this could produce systematically different memorization dynamics.

**Concern 2 – YAGO biographies as privacy proxies.** The YAGO biographies are synthetic (fictional) and templated. The biographies are self-described as having correlated attributes (nationality → name → birthplace). This correlation structure means that even a model performing poorly on direct attribute recall might leak privacy through attribute inference via correlations already present in the base corpus. This is acknowledged implicitly in §D.2 but not treated as a fundamental limitation of the privacy perturbation design.

**Concern 3 – Paraphrase task scope.** The paraphrase datasets (MRPC, PAWS) are used to study "non-literal memorization" and to test whether models prefer the version they saw during training. However, MRPC/PAWS are short paired sentences, typically a single sentence. The relevance to copyright's "expressive elements" doctrine (which applies to longer creative works) is limited, and the paper would benefit from acknowledging this scope restriction.

---

### The HUBBLE Suite (§3)

**Section 3.1 – Pretraining Data**

The use of DCLM as the base corpus is well-motivated. The decontamination procedure (§A.3) is clearly described and the two-phase approach for long vs. short perturbations is sensible. Removing only 7,540 documents (< 0.002%) is reassuring about the scale.

**Concern 1 – Duplication schedule gaps.** Perturbations are assigned duplication levels of {0, 1, 4, 16, 64, 256}. There is a 4× jump at each level. The gap between 64 and 256 is substantial; the jump from 16 to 256 skips intermediate regimes that may be policy-relevant (e.g., 30–100 duplicates, which corresponds to data appearing once per ~1–5B tokens at the 100B scale). The choice of this geometric schedule is not justified, and it creates low resolution precisely in the range where memorization transitions from weak to strong.

**Concern 2 – Perturbation fraction and ecological validity.** The inserted perturbations total to 0.016–0.08% of training tokens. This is intentionally small to avoid degrading model performance (referencing Hernandez et al., 2022, who found degradation >3%). However, real-world problematic datasets may constitute substantially larger fractions of training data. The authors do not discuss whether their findings generalize to higher contamination regimes, and the claim that dilution is a "best practice" implicitly assumes the sensitive data is already a small fraction.

**Concern 3 – Insertion procedure and attention artifacts.** Figure 1 shows that at most one perturbation is inserted per sequence, and perturbations are never truncated. This means highly-duplicated examples appear with the exact same surrounding EOS context and sometimes similar positional patterns within sequences. It is not clear whether the insertion procedure introduces systematic positional biases that could artificially inflate (or deflate) memorization detection via loss-based evaluations.

**Section 3.2 – Models**

The factorial design (2 sizes × 2 conditions × 2 corpus sizes = 8 core models) is clean. The architecture deviation from Llama 3.1 (36 layers for 8B instead of 32, untied embeddings, reduced vocabulary) is well-justified and the rationale explained.

**Concern:** The "gradient accum dtype" column in Table 4 lists both FP32 and BF16 for the 8B model but FP32 only for the 1B. It is unclear whether this difference in gradient accumulation precision affects the comparability of memorization results across scales.

**Section 3.3 – Evaluations**

The three evaluation modes (loss, loss-based choice, generative) are appropriate and each has a clear use case. Applying lower-bound framing to all memorization evaluations is honest.

**Concern 1 – Evaluation consistency across domains.** The domain-agnostic results in §4 use different primary metrics for different data types: loss for passages, loss-based choice for paraphrases, accuracy for test sets, and generative evaluation for biographies. Comparing "memorization strength" across these metrics is not straightforward, which makes the headline claim that "memorization risks are determined by the frequency of sensitive data relative to corpus size" a somewhat over-simplified characterization of what are qualitatively different phenomena.

**Concern 2 – Generative evaluation prompting.** The generative evaluation gives the model a prefix and measures exact-match or word recall on the continuation. For biographies, the prefix is a "partial biography." The choice of prefix length and content is consequential—more context means easier reconstruction—and the paper does not report sensitivity to this design choice in the main text (though §D.2 examines different amounts of auxiliary information).

---

### Domain-Agnostic Results (§4)

**Finding 1: Dilution.** The dilution finding is well-supported by Figure 2: the 500B model consistently shows weaker memorization than the 100B model at the same duplication level. However:

**Concern:** The comparison between 100B and 500B token models conflates two things: (a) increased corpus size (more dilution of the sensitive data's relative frequency), and (b) increased training compute (5× more gradient steps, which affects weight distribution and convergence). The 500B model is not just a "more diluted" version of the 100B model—it is also a more trained model that sees each non-perturbation token approximately 5× more. The paper does not disentangle these effects. A model trained on 500B tokens but with the same number of gradient steps (i.e., with a larger batch size) would be a more controlled comparison.

**Finding 2: Ordering.** The timing runs show that data inserted only in the first quarter of training is subsequently forgotten (Figure 13/14). This is interesting, but:

**Concern 1:** The mechanism of forgetting is not investigated. Is this standard catastrophic forgetting? Is it driven by the cosine LR schedule (which spends most of its budget at lower LRs after warmup)? Weight decay regularizing away low-frequency features? The paper should at minimum discuss plausible mechanisms and cite relevant mechanistic work.

**Concern 2:** The recommendation to "order sensitive data to appear early" is practically ambiguous. In real training pipelines, sensitive data (e.g., scraped personal data) is discovered after training begins, not before. The actionable framing would be: "avoid late-stage exposure"—which is more relevant for second-pass training, fine-tuning, or continued pretraining. The paper does not frame the finding this way.

**Finding 3: Larger models memorize at lower duplications.** This is consistent with Tirumala et al. (2022) and is correctly presented as a replication. No new methodological contribution here.

**Finding 4: Interference check.** Training three single-domain models and showing they match the corresponding domain in the joint perturbed model is a reasonable sanity check. However, the check only validates domain-level aggregate metrics—it does not rule out example-level interference (e.g., whether training on 256× duplicated biographies affects how specific Gutenberg passages are memorized). The paper appropriately acknowledges this limitation.

---

### Domain-Specific Results (§5 and Appendix D)

**Copyright (§D.1).** The finding that loss-based and k-eidetic metrics disagree (loss sensitive at 4× but k-eidetic only at 16×) is an important methodological contribution with direct relevance to legal contexts. This is one of the more novel findings in the paper.

**Concern:** The authors find that popular and unpopular Gutenberg books are memorized similarly at the 1B scale, with only minor differences at 8B. The expectation from the data density hypothesis (Kirchenbauer et al., 2024) was that popular books would be memorized better due to in-corpus discussion. This null/weak result is important and hints that the effect of data density on memorization may be smaller than previously thought. However, the paper somewhat downplays this finding: "more sensitive methods may reveal subtler forms of memorization." The paper should be more direct about what this means for the data density hypothesis.

**Privacy (§D.2).** The PII reconstruction results are well-documented. The finding that certain PII types (occupation, email, UUID) are memorized differently is interesting and ecologically valid.

**Concern:** The indirect leakage via PersonaChat usernames is evaluated using 10-way multiple-choice (the model selects the correct persona from 10 candidates). With random chance at 10% and the paper saying "inference is difficult but possible," the absolute accuracy numbers matter. The paper reports qualitative results without giving the actual accuracy numbers in the main text for this task, making it difficult to evaluate the severity of the indirect leakage.

**Test Set Contamination (§D.3).**

**Concern 1 – Format sensitivity.** The finding that perturbed models perform worse when the test-time format differs from the inserted format is important for practitioners (using different evaluation harnesses can mislead about contamination). This is well-documented. However, the paper notes models can achieve worse accuracy than the standard model when formats mismatch—this "anti-memorization" effect could reflect distribution shift from the duplicated incorrect-format examples. More analysis here would strengthen the contribution.

**Concern 2 – Test set contamination and generalization.** The paper states that "memorizing test set examples does not translate into generalization on that task, and for WinoGrande, perturbed models achieve worse accuracy on minimal pairs." This finding is somewhat inconsistent with the zero-shot evaluation in Table 5/6, where perturbed and standard models perform similarly. The paper should clarify whether the within-distribution degradation on minimal pairs is a general phenomenon or specific to WinoGrande's minimal pair structure.

---

### Use Cases: HUBBLEMIA and HUBBLEUNLEARNING (§6)

**HUBBLEMIA.** The benchmark design is sound: randomized insertion eliminates temporal confounders that plagued WIKIMIA, and the controlled duplication levels enable systematic evaluation at different memorization strengths. The choice of 4 off-the-shelf MIA methods is appropriate for a benchmark paper (not intended to be a new MIA method).

**Concern:** The paper tests MIAs on only the 8B 500B-token model in Table 1, reporting results for the other three model variants in Appendix F. The full table (Table 11 in the appendix) reveals an important fact: on the standard model (which never saw the perturbations), all MIA methods also show near-random AUC for Dup=64 and Dup=256 (0.54 and 0.55 for loss on Gutenberg Unpopular). But wait—the standard model was *not trained on these perturbations*, so why is AUC not 0.50? The paper should explain this residual deviation.

**HUBBLEUNLEARNING.** The finding that no method achieves the "desired target" (erase Unlearn set while preserving Keep and Test sets) is expected given the state of the art, but the visualization in Figure 3 is informative. The use of the same-distribution Keep set as a secondary retain set (instead of WikiText) is a valuable addition.

**Concern:** The unlearning experiments are run only on the 8B 500B-token perturbed model with data duplicated 256 times. This is the regime where memorization is strongest (AUC approaching 1.0 for MIAs). Testing at lower duplication levels (e.g., 16×) would be valuable to understand whether unlearning is easier when memorization is less entrenched. Also, the forget set consists of 256× duplicates—these examples have very high memorization. The unlearning benchmark as defined may be measuring an extreme case. Reporting results across duplication levels would make the benchmark more useful.

---

### Writing & Clarity

The paper is well-written and the structure is logical. The policy framing in §2 is engaging. The appendices are detailed and well-organized.

**Concern:** The distinction between "standard" and "perturbed" models is used to simultaneously mean (a) models with/without perturbation data and (b) the oracle reference point for unlearning. This dual use of "standard model" as both an experimental control and an "ideal" unlearning target could confuse readers. In §6.2, the "desired model" symbol differs from "standard model" but the text conflates them.

**Minor concern:** The paper mentions the architecture experiments (8-layer vs. 32-layer models) in §3.2 and Appendix E.3 but provides no main-text discussion of the results. If architecture runs are a released model collection, at least a one-sentence summary of the key finding belongs in the main text.

---

### Limitations & Broader Impact

The paper is commendably honest about limitations: the gap to commercial model scale, the use of public-domain proxies for copyright, synthetic data for privacy, etc. The discussion in Appendix H raises good future research questions.

**Concern 1 – Ethical implications of inserting PII-like data into models.** The authors insert YAGO synthetic biographies and ECtHR case summaries (involving real defendants). The ECtHR data is already public, but inserting it into a training corpus and training models to memorize it (up to 256×) raises ethical questions about re-exposing personal data. The paper does not include an ethics statement discussing whether ECtHR participants consented to this use.

**Concern 2 – Dual-use of the HUBBLE benchmark.** By providing a controlled testbed for testing how to extract memorized data, HUBBLE also makes it easier for adversaries to benchmark and improve their extraction attacks. The paper briefly mentions "lower bounds" on memorization but does not discuss whether releasing the benchmark itself increases attack capabilities against existing deployed models.

**Concern 3 – Generalizability of "ordering" advice.** The timing experiments show that early-inserted data is forgotten. But "forgotten" here means not extractable at the end of training. Whether this data is truly unlearned (e.g., cannot be elicited via fine-tuning, in-context learning, or model merging) is not tested. The claim that ordering provides "a form of privacy" (citing Jagielski et al., 2023) overstates the strength of the evidence.

---

### Overall Assessment

HUBBLE is a substantial, well-executed infrastructure contribution to the LLM memorization literature. The fully open-source release of 14+ models, training data, insertion code, and evaluation harnesses represents genuine scientific value, particularly for researchers studying copyright, privacy, and benchmark contamination without access to commercial training infrastructure. The controlled perturbation design—with randomized insertion rates, decontamination, and interference checks—is methodologically rigorous by the standards of this type of resource paper.

That said, the empirical findings themselves (dilution reduces memorization risk, early data is forgotten, larger models memorize more) are largely confirmatory of prior work at a scale still far below commercial LLMs. The core §4 findings do not individually represent novel scientific discoveries; their value lies in simultaneous confirmation across three domains in a single controlled setup. The dilution finding conflates corpus size with training compute, the ordering finding lacks mechanistic analysis, and the MIA/unlearning benchmarks test only extreme memorization conditions (256× duplication). For ICLR, which expects contributions beyond resource creation, the paper would be strengthened by either (a) deeper mechanistic analysis of *why* forgetting occurs during training, or (b) a clear scaling law relating duplication frequency, corpus size, and model size to memorization probability that goes beyond descriptive trend plots. As is, the paper is a valuable and well-presented resource contribution that sits at the threshold of ICLR's acceptance bar—likely acceptable given the breadth of open-source release and multi-domain empirical coverage, but reviewers who prioritize novel technical insight may find the empirical takeaways insufficiently surprising.

# Neutral Reviewer
## Balanced Review

### Summary
The authors introduce HUBBLE, a fully open-source suite of LLMs (1B and 8B parameters, up to 500B training tokens) designed to enable controlled, causal study of memorization across copyright, privacy, and test-set contamination domains. Through systematic pretraining experiments with randomized, decontaminated insertions, the work establishes that memorization risks can be mitigated by diluting sensitive data in larger corpora and ordering it to appear early in training. The release further provides standardized benchmarks for membership inference attacks and machine unlearning, offering a foundational resource for the research community.

### Strengths
1. **High-Quality, Fully Open Scientific Release:** The suite matches the rigor of prior landmark open releases (e.g., Pythia, OLMo) while specifically targeting memorization. The release includes all model weights, training code, perturbation datasets, evaluation harnesses, and detailed compute accounting (~200k GPU hours), ensuring exceptional reproducibility (§3, App B).
2. **Rigorous Experimental Design & Causal Grounding:** The paper employs careful decontamination, randomizes duplication levels (0× to 256×), and explicitly checks for cross-domain interference (Fig. 20, §4). This controlled design allows for reliable estimation of memorization scaling and spacing effects that are otherwise impossible in observational studies.
3. **Actionable, Generalizable Empirical Findings:** The work clearly demonstrates two scalable pretraining best practices: dilution (memorization decreases with larger corpus size for fixed duplication rates, Fig. 2) and ordering (early-exposed data is more likely to be forgotten without continued exposure, Fig. 13/14). These findings are directly relevant to dataset curation and training pipelines.
4. **Well-Constructed Downstream Benchmarks:** HUBBLEMIA and HUBBLEUNLEARNING address critical gaps in the literature. By providing clean member/non-member splits without spurious temporal cues and known duplication levels for unlearning targets, the benchmarks eliminate common confounders that plague existing MIA and unlearning evaluations (§6.1, §6.2).
5. **Strong Baseline General Performance:** Evaluation on standard benchmarks (Tables 5–7, App C) confirms that injecting controlled perturbations does not severely degrade general capabilities, validating the suite's utility for both memorization research and as a competitive open-weight model family.

### Weaknesses
1. **Scale Gap to Frontier Models:** While the 8B/500B-token runs are substantial for academic research and exceed Chinchilla-optimal compute for those sizes, they remain 1–2 orders of magnitude smaller than modern foundation models (e.g., Llama 3 at ~15T tokens). Memorization dynamics, especially regarding dilution thresholds and interference, may exhibit non-linear scaling behaviors at larger capacities or with different architectural innovations (§3 intro).
2. **Reliance on Synthetic/Structured Data for Privacy:** The privacy domain heavily relies on templated YAGO biographies and Personachat dialogues. Real-world PII leakage occurs in unstructured, noisy web text with varying contextual plausibility. Synthetic biographies may overestimate attack success due to consistent formatting and may not fully capture the statistical entanglement of real personal data (§2.2, App A.1).
3. **Limited Scope of Unlearning Evaluation:** The unlearning benchmark tests only three methods (RMU, RR, SatImp) with a basic grid search over hyperparameters. The analysis focuses primarily on likelihood/accuracy drop on forget/retain sets but lacks broader utility metrics (e.g., downstream task degradation, representation similarity, or more recent unlearning techniques), limiting immediate conclusions about method efficacy (§6.2, Fig 3, App G).
4. **Lack of Actionable Thresholds for Policy/Practice:** The paper notes that metric choice (e.g., loss vs. k-eidetic) changes the interpretation of memorization (§5, App D.1) but does not propose standardized thresholds or a decision framework for when dilution/ordering reduces risk to "acceptable" levels, leaving practitioners to extrapolate from relative trends.

### Novelty & Significance
**Novelty:** High. While open model suites are increasingly common, HUBBLE is the first specifically architected for *causal, randomized memorization experimentation* at a non-trivial scale. Integrating perturbation design directly into the pretraining corpus, rather than relying on fine-tuning or synthetic post-hoc injections, is a distinct and valuable methodological contribution.
**Clarity:** Excellent. The paper is well-structured, with clear motivation, thorough perturbation design documentation, and logical progression from domain-agnostic findings to specialized use cases. Figures and tables are effectively used to support claims.
**Reproducibility:** Outstanding. Full transparency in training configurations, data filtering/decontamination steps, evaluation scripts, and compute resources ensures that any research group can replicate or extend the suite. The auxiliary release of TokenSmith further lowers the barrier for dataset manipulation research.
**Significance:** High. Memorization sits at the intersection of model capability, copyright law, data privacy, and benchmark validity. By providing a standardized, open testbed and empirically validating two practical mitigation strategies, the work directly advances foundational ML safety and dataset curation practices. The benchmarks for MIAs and unlearning in pretraining settings fill a major gap in the literature, making this highly relevant to ICLR's focus on rigorous, open, and safety-conscious foundational research.

### Suggestions for Improvement
1. **Expand Unlearning Evaluation & Utility Tracking:** Incorporate additional unlearning baselines (e.g., gradient ascent, contrastive, or circuit-breaking methods) and report standardized utility metrics beyond the immediate forget/retain sets, such as downstream general task performance, representation drift (e.g., CKA distances), and computational overhead.
2. **Provide Scaling Extrapolations or Theoretical Grounding:** Include a discussion or preliminary analysis on how dilution and ordering might scale with model size and corpus magnitude. Even simple empirical fits or references to influence function theory/data dynamics would help practitioners estimate required corpus expansion to hit specific memorization thresholds at larger scales.
3. **Release Intermediate Checkpoints or Dynamic Memorization Scripts:** While intermediate checkpoint analysis is discussed in §E.1, releasing a curated subset of checkpoints or a lightweight script to compute memorization curves dynamically would enable broader study of memorization evolution without requiring full pretraining runs.
4. **Propose Concrete Risk Thresholds or Metric Guidelines:** Given the paper's observation that metric choice dictates policy conclusions (loss vs. k-eidetic), consider adding a standardized evaluation guideline or threshold framework. For example, define what duplication levels and corpus sizes correspond to specific risk tiers (e.g., "low extraction probability under black-box attacks") to bridge the gap between empirical trends and practical deployment guidelines.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Test ordering/timing findings at 8B scale** — The timing runs only use 1B models, but the core dilution experiments use both 1B and 8B. Without 8B timing results, the claim that "ordering sensitive data early reduces memorization" lacks the same evidentiary support as the dilution claim.

2. **Compare against existing memorization mitigation baselines** — The paper proposes dilution and ordering as "best practices" but doesn't compare these against alternatives (e.g., differential privacy training, gradient clipping, or dedicated unlearning during pretraining). Without this, readers cannot assess whether these practices are actually superior or just convenient.

3. **Add a solvable unlearning setting** — All three unlearning methods fail similarly in Figure 3. Include at least one method or setting where unlearning partially succeeds, otherwise the benchmark cannot distinguish between methods and the "failure" result may reflect benchmark design rather than method limitations.

4. **Direct comparison to prior controlled memorization studies** — The paper positions HUBBLE as advancing beyond prior controlled studies (Zhang et al., 2023; Allen-Zhu & Li, 2024), but doesn't replicate their key experiments on HUBBLE to show what new insights the suite enables.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantify interference between perturbation domains more rigorously** — The interference check (3 single-domain models) is minimal. Analyze whether high-duplication test set examples affect copyright passage memorization, or whether biographyinsertions affect test set contamination measurements. Without this, the multi-domain design's validity is uncertain.

2. **Analyze why unlearning methods fail in pretraining vs. fine-tuning** — The paper notes existing unlearning benchmarks target fine-tuning (TOFU, MUSE), but doesn't analyze whether the pretraining setting fundamentally changes the unlearning problem or whether methods just need adaptation.

3. **Explain the mechanism behind dilution** — The paper shows dilution works but doesn't analyze why (e.g., reduced gradient signal per example, attention distribution changes, or representation dilution). Without mechanistic insight, the finding remains purely empirical and less actionable.

4. **Validate that memorization metrics correlate with actual extraction risk** — Show that examples flagged as "memorized" by loss-based metrics can actually be extracted via prompting. Otherwise, the memorization measurements may not reflect real privacy/copyright risks.

### Visualizations & Case Studies
1. **Show actual extracted text at different duplication levels** — Include concrete examples of what can be generated from models trained with 1×, 16×, and 256× duplication. This makes the privacy/copyright risk tangible rather than abstract metric differences.

2. **Visualize memorization evolution during training** — Figure 13 shows forgetting curves but doesn't show when memorization first appears or how it stabilizes. A training trajectory visualization would clarify the timing findings.

3. **Embedding space or attention visualization of memorized vs. non-memorized examples** — Show whether memorized examples occupy distinct regions in representation space or receive distinct attention patterns. This would validate that the method captures genuine memorization phenomena.

### Obvious Next Steps
1. **Demonstrate a complete research workflow using HUBBLE** — Show one end-to-end example of how an external researcher would use HUBBLE to answer a new memorization question (e.g., "Does deduplication before or after tokenization matter?"). This validates the suite's usability claim.

2. **Release intermediate checkpoints for all runs, not just timing** — The paper mentions intermediate checkpoints for timing runs enable memorization evolution studies, but doesn't release these for core runs. This limits reproducibility of training dynamics analysis.

3. **Test whether dilution ordering findings generalize beyond English** — All experiments use English corpora. Given the policy-relevant framing (GDPR, copyright law), at least discuss or test whether findings hold for multilingual settings where memorization risks differ.

# Final Consolidated Review
## Summary

HUBBLE is a fully open-source suite of language models (1B and 8B parameters, trained on 100B and 500B tokens) designed to enable controlled, causal study of LLM memorization across copyright, privacy, and test set contamination domains. The paper releases models with systematically inserted perturbation data at controlled duplication rates, demonstrating that memorization risk can be reduced by diluting sensitive data in larger training corpora and by ordering such data to appear early in training. The release includes benchmarks for membership inference attacks and machine unlearning in pretraining contexts.

## Strengths

- **Comprehensive open-source scientific release.** The paper provides 14 model variants, all training code, perturbation datasets, and evaluation harnesses with detailed compute accounting (~200k GPU hours). This matches the rigor of prior landmark releases like Pythia and OLMo while specifically targeting memorization research (§3, Appendix B).

- **Rigorous experimental design with controlled perturbations.** The paper employs careful decontamination of the base corpus, randomizes duplication levels per example, and explicitly validates that perturbations from different domains minimally interfere with each other through dedicated interference experiments (Figure 20, §4). This design enables more reliable estimation of memorization effects than observational studies.

- **Actionable empirical findings with practical relevance.** The paper demonstrates two pretraining practices that reduce memorization risk: dilution (memorization decreases with larger corpus size for fixed duplication rates, Figure 2) and ordering (data appearing early in training is more likely to be forgotten without continued exposure, Figures 13–14). These findings have direct implications for dataset curation pipelines.

- **Novel benchmarks filling critical gaps.** HUBBLEMIA provides clean member/non-member splits without temporal confounders that plagued prior benchmarks, and HUBBLEUNLEARNING enables unlearning evaluation in pretraining contexts where current benchmarks (TOFU, MUSE) focus on fine-tuning (§6).

- **Methodological contribution on metric choice.** The copyright domain analysis finds that loss-based metrics detect memorization at lower duplication levels (4×) than k-eidetic metrics (16×+), demonstrating that metric choice materially affects policy conclusions about whether a model "memorizes" (Appendix D.1). This has direct relevance to legal debates.

## Weaknesses

- **The dilution experiment conflates corpus size with training compute.** The comparison between 100B and 500B token models shows weaker memorization at 500B, but this could result from either (a) relative frequency reduction of sensitive data, or (b) the model being more thoroughly trained (5× more gradient steps). The paper does not disentangle these effects, though the finding that additional training on non-sensitive data reduces memorization remains valid (§4).

- **Timing experiments are conducted only at the 1B scale.** The ordering/forgetting findings rely exclusively on 1B parameter models (§3.2, Appendix E.1). Without 8B timing results, it is unclear whether the "early data is forgotten" finding scales to the larger models where memorization is more pronounced (§4 notes larger models memorize at lower duplications).

- **The duplication schedule has large gaps at policy-relevant ranges.** Duplication levels jump from 64× to 256×, a 4× gap in the regime where memorization transitions from moderate to strong. The paper does not justify this geometric schedule, creating lower resolution precisely where practitioners might want to estimate risk thresholds (Appendix A.2).

- **The unlearning benchmark tests only extreme memorization conditions.** HUBBLEUNLEARNING evaluates data duplicated 256 times, where memorization is strongest. Results at lower duplication levels (e.g., 16× or 64×) would be valuable for understanding whether unlearning is more tractable when memorization is less entrenched. Additionally, all three tested methods fail similarly, so the benchmark cannot currently distinguish between method capabilities (§6.2, Figure 3).

- **No comparison to alternative memorization mitigation techniques.** The paper recommends dilution and ordering as best practices but does not compare these against existing mitigation methods such as differential privacy training, gradient clipping, or targeted data removal. Without such baselines, readers cannot assess relative efficacy (§4).

## Nice-to-Haves

- Intermediate checkpoints for core runs would enable training dynamics analysis beyond the timing experiments, which currently only release checkpoints for the 1B timing models.

- Analysis of whether the 100B vs. 500B dilution effect persists when controlling for total compute (e.g., by comparing 100B with 5× batch size vs. 500B with standard batch size).

- Extraction examples showing actual model outputs at different duplication levels, making the privacy/copyright risks concrete rather than abstract metric differences.

## Removed Points

- *Critic claimed AUC values on standard model should be exactly 0.5.* Table 11/12 shows AUC values around 0.50±0.05 for the standard model, which is near-random as expected. Small deviations are inherent to sampling noise; this is not a substantive issue.

- *Critic claimed "standard model" terminology is confusing.* The paper clearly distinguishes "standard" (no perturbations) from "perturbed" models and uses "desired model" to denote the ideal unlearning target. The terminology is standard and well-defined (§6.2).

- *Critic questioned ethics of ECtHR court data insertion.* The ECtHR dataset is already publicly available and specifically curated for PII research (Pilán et al., 2022). The paper uses it in accordance with its intended research purpose.

- *Critic demanded mechanistic explanation of why forgetting occurs.* While mechanistic analysis would be valuable, the paper's contribution is establishing the empirical phenomenon and providing the resource; mechanistic interpretation is a natural future direction, not a requirement.

## Novel Insights

The paper reveals an important asymmetry between memorization and generalization in test set contamination: models can achieve worse accuracy on contaminated examples than on unseen examples when the evaluation format differs from the inserted format (Appendix D.3). This "negative transfer" from contamination suggests that practitioners using different evaluation harnesses may be misled about contamination levels. Additionally, the finding that popular and unpopular books are memorized similarly at the 1B scale contradicts the data density hypothesis (Kirchenbauer et al., 2024), suggesting that pretraining corpus context may matter less than previously thought for verbatim memorization of isolated passages (Appendix D.1).

## Suggestions

- Add 8B timing experiments to strengthen the ordering finding, or explicitly acknowledge this as a limitation and scope the current finding to 1B models.

- Include unlearning results at multiple duplication levels (e.g., 16×, 64×, 256×) to create a more informative benchmark that can distinguish between methods of varying capability.

- Provide preliminary scaling extrapolations—simple empirical fits relating duplication rate, corpus size, and model size to memorization probability—to help practitioners estimate thresholds for their own settings without running full experiments.

- Compare dilution and ordering against at least one alternative mitigation method (e.g., deduplication alone, or differential privacy training) to contextualize the effectiveness of the recommended practices.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept
