=== CALIBRATION EXAMPLE 33 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately signals that this is a methodological critique / blueprint paper, but it is somewhat broader than the actual content. In practice, the paper is primarily a forensic re-analysis of one case study (min-p) rather than a general blueprint for “more rigorous science” in empirical ML.
- The abstract clearly states the problem, the method of inquiry, and the authors’ main findings. It is transparent that the paper is a re-examination of Nguyen et al. (2024) and that the authors identify issues in four evidence streams.
- Some abstract claims are strong and would need especially careful support for ICLR’s bar, e.g. “its conclusions are invalidated by its own data” and “the paper’s own evidence invalidates its central claim.” These are plausible if the analyses are correct, but they are high-stakes claims and should be framed with more explicit nuance about scope and dependence on reanalysis choices. The abstract also bundles community-adoption claims into the main scientific conclusion, which is somewhat distinct from the core evaluation of min-p itself.

### Introduction & Motivation
- The paper is well-motivated in the sense that it addresses a real and important problem for ICLR: weak statistical practice, selective reporting, and irreproducibility in empirical ML.
- The gap in prior work is clear: the paper positions itself as a detailed case study showing how a high-profile method paper can present apparently strong evidence that does not survive scrutiny.
- The introduction does overreach slightly in claiming the case study “provides a blueprint” and “general lessons” for the field before establishing how broadly the specific failure modes transfer beyond this example. ICLR will likely ask whether the lessons are truly general or mainly domain-specific to LLM sampling and evaluation.
- The framing is intentionally adversarial toward the target paper. That is acceptable for a critique paper, but the introduction should more clearly distinguish between factual reanalysis, interpretive judgment, and broader claims about the state of ML science.

### Method / Approach
- The paper is much stronger as a critique than as a new method paper, but its methodological rigor still matters greatly for ICLR. The reanalysis is generally well-motivated and specific, especially in Sections 2–5.
- In Section 2.1, the identification of omitted human-evaluation data is serious and concrete. The paper states that one-third of collected scores were excluded without justification, which is a major transparency issue if accurate.
- In Section 2.2, the authors use paired one-sided t-tests across 12 comparisons and also discuss Bonferroni correction and an Intersection-Union Test. This is a sensible direction, but the paper needs to be very precise about whether the original experimental unit is independent, how paired observations were constructed, and whether the tests match the design. The paper states the tests are paired t-tests with df = 52, but the rationale for pairing and the consequences of the missing baseline data deserve fuller explanation.
- The argument in Section 2.2 about pooling across conditions is important, but it is somewhat underdeveloped statistically. If the authors claim the original paper’s pooled test was invalid because it ignored heterogeneity across temperature/diversity settings, they should ideally give a formal model or at least a clearer justification for why the pooled analysis is the wrong estimand.
- In Section 2.4, the “new human evaluation” is complicated by multiple changes at once: sampler implementation, participant pool, hyperparameters, reading time, sampled text, and rubric. This makes the interpretation of the second study difficult. The paper correctly notes this, but then still draws fairly strong conclusions from the new study. Because many variables changed simultaneously, the paper should be more cautious about treating it as a clean corroboration of the critique.
- Section 3’s “fair comparison by controlling for hyperparameter volume” is the most methodologically novel part of the paper. However, the approach raises important questions:
  - Is “number of hyperparameters swept” a fair proxy for tuning budget, especially when the parameters may have different effective search granularity or relevance?
  - Best-of-N over randomly subsampled hyperparameter sets can be informative, but it depends on how the search space is sampled. The paper should justify why uniform subsampling of hyperparameter configurations is the right normalization.
  - It is not fully clear whether the analysis is comparing methods under equal human/computational tuning effort, equal grid size, or equal opportunity to discover strong settings. Those are related but not identical.
- In Section 4, the critique of the LLM-as-a-judge evaluation points to genuine reproducibility problems, especially under-specification of the sampling model, judge model, and hyperparameter selection. That is a strong concern.
- The Section 5 critique of community-adoption claims is more epistemic than methodological. It is relevant to the paper’s narrative, but it is not a direct scientific evaluation of min-p’s performance and should not be weighted as heavily as the empirical analyses.
- Overall, the approach is serious and detailed, but the paper would benefit from a more explicit accounting of assumptions, analysis choices, and the limits of what can be inferred from each reanalysis.

### Experiments & Results
- The experiments do test the paper’s core claims, which is exactly what ICLR would expect from a critical reanalysis paper. This is a major strength.
- The human-evaluation reanalysis in Section 2 is compelling, especially because it revisits the published data rather than relying only on new experiments. The missing baseline data and the multiple-comparisons issue are substantive.
- However, the paper’s strongest conclusion—“min-p does not outperform baseline samplers across all settings”—depends heavily on how the statistical tests are set up. The paper should be careful not to overstate what the tests prove. Failure to reject superiority across all tests is not identical to proving equality or lack of advantage.
- The hyperparameter-volume analysis in Section 3 is interesting, but I am not fully convinced the conclusion is airtight. The results suggest sensitivity to tuning budget, but the paper needs to show that its tuning-volume normalization is not itself favoring or disadvantaging particular samplers.
- I would also want a clearer ablation on the exact effect of the benchmark prompt-format correction in Appendix C. The main text says the results were nearly identical, which is reassuring, but this deserves concise quantitative summary in the main paper if it matters to the interpretation.
- The LLM-as-a-judge results in Section 4 are useful mainly as a reproducibility critique. The reported methodological ambiguity is itself a finding. Still, the paper’s interpretation of win rates would be stronger if it included a more direct reproduction of the original benchmark protocol or a clearly specified alternative protocol.
- The community-adoption section does not really test a scientific performance claim, so it should be treated as ancillary. It does, however, support the broader narrative that some rhetorical evidence in the original paper was weak or overstated.
- Error bars and uncertainty are partially reported, but not consistently across all analyses in a way that fully satisfies the best ICLR standards. In particular, the paper would benefit from clearer statement of sampling variability, multiple-testing corrections, and robustness across alternative reasonable analytical choices.
- Overall, the empirical sections are substantial and likely to be taken seriously, but some conclusions would be stronger if the paper more explicitly separated “the original paper’s evidence is insufficient” from “the method is not better.”

### Writing & Clarity
- The core arguments are understandable, but some sections are dense and occasionally hard to follow because many claims are packed together quickly.
- The structure is coherent: human evaluations, benchmark evaluations, LLM-judge evaluations, and adoption claims are separated cleanly.
- Figures and tables generally communicate the intended message, but several are difficult to interpret without careful reading. For example:
  - Figure 1 is meant to show overlap in confidence intervals, but the visual summary is not self-explanatory enough to support the strong interpretive claim on its own.
  - Table 1 is important and informative, but the logic of the 12 comparisons and the relation to the original paper’s single pooled test should be stated even more explicitly.
  - Figures 4–6 communicate the hyperparameter-volume point, but the normalization procedure is not visually obvious from the plots alone.
- The main clarity issue is not prose quality; it is that some of the paper’s strongest claims depend on nuanced statistical or methodological judgments that are not fully unpacked in the main text.
- A reader unfamiliar with the original min-p paper may have difficulty distinguishing what is being reanalyzed, what was added by these authors, and what exactly changed after the authors interacted with Nguyen et al.

### Limitations & Broader Impact
- The paper does acknowledge one key limitation: it reanalyzes only the evidence in Nguyen et al. (2024) and additional evidence generated using the original code, so its conclusions are conditional on that evidence base.
- That said, the limitations section is still too limited for ICLR standards. The paper would benefit from explicitly discussing:
  - dependence on the correctness and completeness of the publicly available data/code;
  - sensitivity of conclusions to the choice of statistical tests and tuning-budget normalization;
  - the fact that some arguments concern the original paper’s reporting practices rather than the underlying method itself;
  - the possibility that min-p could still be useful in regimes not covered by the reanalysis.
- Broader impact is not discussed in a conventional way, but the paper is essentially a meta-scientific critique. The main societal impact is positive if it improves rigor, but the authors should still acknowledge the risks of reputational harm, discouraging exploration of new methods, or incentivizing overly adversarial reading of papers without proportional evidence.
- Since this is an ICLR submission, I would expect a more explicit discussion of how such critique papers should be used responsibly and how to distinguish genuine methodological corrections from stylistic disagreement.

### Overall Assessment
This is a serious and potentially important critique paper that addresses a highly relevant ICLR topic: rigor in empirical ML. The strongest parts are the reanalysis of the human-evaluation data and the identification of methodological ambiguities and reporting issues. The paper plausibly shows that several pieces of evidence in the original min-p paper were weaker than claimed. However, the central scientific conclusion is still somewhat stronger than the evidence shown in the manuscript justifies, mainly because several analyses hinge on nontrivial methodological choices—especially the hyperparameter-volume normalization, the handling of missing/omitted human-evaluation data, and the interpretation of pooled versus condition-specific tests. I think the contribution is meaningful and likely publishable in spirit, but to meet ICLR’s bar it needs a more careful separation of “insufficient evidence for the original claim” from “the claim is definitively false,” plus a clearer justification of the reanalysis design choices.

# Neutral Reviewer
## Balanced Review

### Summary
This paper is a critique/re-analysis of Nguyen et al. (2024)’s min-p sampling paper, arguing that the original claims of min-p’s superiority are not supported by the available evidence. It examines four evidence streams—human evaluation, benchmark sweeps, LLM-as-a-judge results, and adoption claims—and concludes that min-p does not consistently outperform baselines once methodological issues, omitted data, and hyperparameter-tuning volume are accounted for.

### Strengths
1. **Timely and relevant topic for ICLR standards on rigor and reproducibility.**  
   ICLR typically values careful empirical scrutiny, and this paper directly targets common failure modes in ML evaluation: incomplete reporting, multiple comparisons, unfair tuning budgets, and unclear experimental design. The paper’s focus is aligned with the conference’s interest in reproducible and trustworthy ML research.

2. **Broad and systematic re-analysis of multiple evidence streams.**  
   The paper does not rely on a single criticism; it re-examines human studies, benchmark sweeps, LLM-judge evaluation, and adoption claims. This breadth strengthens the case that the critique is not an isolated complaint but a systematic challenge to the original paper’s conclusions.

3. **Concrete methodological concerns are backed by specific evidence.**  
   The paper identifies omitted human-evaluation data, incorrect/insufficiently corrected statistical testing, under-specified judge-evaluation methodology, and selective reporting. These are not vague objections; the manuscript points to specific tables, figures, data files, and public interactions with the original authors.

4. **The hyperparameter-volume argument is a useful and generalizable methodological point.**  
   The proposed “Best-of-N” style analysis to equalize hyperparameter-search volume addresses a real confound in comparing methods that have different tuning capacities. Even if one disagrees with some details, this is a meaningful contribution to evaluation methodology and may be broadly useful beyond min-p.

5. **The paper offers a clear normative message about scientific practice.**  
   Its final “blueprint” section distills lessons on transparency, fair comparison, statistical testing, and consistent reporting. For an ICLR audience, this kind of prescriptive guidance can be valuable if it is supported by careful evidence.

### Weaknesses
1. **The manuscript is closer to a strong critique than to a standalone methodological contribution.**  
   ICLR generally expects either a novel technical method, a significant empirical result, or a broadly impactful methodological insight. Much of the paper’s content is a re-analysis of a prior work rather than an independently developed framework; the novelty is therefore limited unless the “blueprint” is shown to generalize in a more formal way.

2. **Several central claims depend on adversarial framing and interpretation choices.**  
   For example, the benchmark analysis argues that min-p’s advantage disappears after controlling for hyperparameter volume, but the paper’s own sampling and selection protocol for the “Best-of-N” comparison introduces a chosen fairness criterion that may not be universally accepted as the right one. The conclusion may be reasonable, but the methodological choice itself needs stronger justification and sensitivity analysis.

3. **The paper sometimes overreaches from critique to broad dismissal.**  
   Some statements suggest that the original paper’s evidence “invalidates” the method’s value overall, even though many of the issues shown are about the strength of evidence for superiority rather than proof that min-p is ineffective in all settings. For ICLR, reviewers often prefer measured claims that precisely match the evidence.

4. **Reproducibility is mixed because the paper relies heavily on outside artifacts and public interactions.**  
   The critique references public GitHub repositories, author communications, and later edits/retractions. This is helpful for transparency, but it also means some conclusions depend on materials not fully packaged in the paper itself. A reviewer would likely want a more self-contained artifact and clearer provenance for each derived result.

5. **The paper’s empirical analyses could be more statistically disciplined.**  
   The human-evaluation section correctly highlights multiple comparisons, but the paper itself uses several exploratory analyses, manual annotations, and post hoc reruns. Those analyses may be reasonable, yet the manuscript would benefit from a stronger separation between confirmatory and exploratory findings, with uncertainty quantified consistently.

6. **Potential significance is somewhat narrow if viewed purely as a paper-specific rebuttal.**  
   The work is impactful as a correction to one influential paper, but ICLR generally favors contributions that also advance the field more broadly. The “blueprint” is promising, but its generality beyond this case study is not fully demonstrated.

### Novelty & Significance
**Novelty:** Moderate. The specific findings about omitted data, inconsistent reporting, and unfair tuning comparisons are new and important as a case study, but the core methodological ideas—replication, multiple-comparison correction, and controlling search budget—are well established. The most novel element is the combination of these critiques into a single comprehensive re-analysis and the attempt to formalize a fair comparison by hyperparameter-volume matching.

**Clarity:** Moderate to good. The paper’s thesis is easy to understand, and the section structure is logical. However, the argumentative style is very assertive, and some key methodological decisions would benefit from more careful framing to distinguish fact, inference, and interpretation.

**Reproducibility:** Mixed. The paper provides many concrete references to code/data and publicly available artifacts, which is a plus. But the heavy dependence on external links, manual annotation, and evolving public discussions makes full reproduction of the critique somewhat harder than it should be.

**Significance:** High as a corrective/reproducibility contribution, but more limited as a general machine learning advance. For ICLR, it is potentially acceptable if the conference values rigorous rebuttals and methodological lessons; however, its acceptance would hinge on how strongly the reviewers value field-corrective work versus new ML methods.

### Suggestions for Improvement
1. **Separate confirmatory claims from exploratory or interpretive claims.**  
   Clearly label which results are direct re-analyses of released data, which are post hoc, and which are interpretive judgments. This would strengthen credibility and make the paper easier to evaluate.

2. **Provide a more explicit justification for the hyperparameter-volume fairness criterion.**  
   Explain why “equalized hyperparameter space” is the right benchmark, discuss alternatives, and report sensitivity analyses. ICLR reviewers will likely want to know whether the conclusion holds under several reasonable fairness definitions.

3. **Package a fully reproducible artifact.**  
   Include a self-contained repository with scripts, fixed data snapshots, and exact instructions to regenerate every figure and table in the paper. This would greatly improve the paper’s reproducibility score.

4. **Tone down categorical language where the evidence supports a narrower conclusion.**  
   Replace broad claims like “invalidates” or “offers no advantage” with more precise statements tied to the specific experiments. This will make the paper more rigorous and less vulnerable to criticism over overstatement.

5. **Quantify uncertainty more consistently across all analyses.**  
   The manuscript would benefit from uniform uncertainty estimates for the benchmark sweeps, manual annotations, and judge-evaluation comparisons. This would help distinguish robust patterns from potentially noisy ones.

6. **Strengthen the general methodological contribution.**  
   Turn the “blueprint” into a more formal checklist, taxonomy, or workflow for evaluating empirical ML claims. If the authors can abstract the lessons beyond this single case, the paper’s ICLR-level contribution would be stronger.

7. **Clarify provenance for claims that depend on external communications or public edits.**  
   Whenever a conclusion rests on author correspondence, retracted camera-ready changes, or public issue threads, summarize the evidence in the paper itself so readers can assess it without following multiple external links.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Re-run the core human-evaluation claims with a preregistered, full-data analysis that includes all collected responses and a clear correction for multiple testing. Without an analysis that uses the complete dataset and appropriate inferential control, the claim that min-p fails to outperform baselines is not fully credible under ICLR’s reproducibility bar.

2. Add direct, matched comparisons in the LLM-as-a-Judge setting: min-p vs top-p and min-p vs basic at the same temperatures, same judge, and same prompt set. The current indirect comparison through a fixed reference sampler is too confounded to support the paper’s claim that min-p is not better.

3. Reproduce the benchmark study with a fair tuning budget matched across samplers, and report the best-performing settings under equal compute/hyperparameter search. The paper’s “min-p loses once tuning volume is controlled” claim depends on the sweep design; without an explicit budget-matched baseline, the conclusion is not secure.

4. Evaluate the method on at least one independent, non-GSM8K benchmark and one non-creative-generation task where sampling quality matters. Right now the empirical critique is narrow enough that it is hard to tell whether the concern is about min-p specifically or about the authors’ chosen task/setup.

5. Add a replication of the community-adoption claim using a transparent query protocol and an external source of evidence beyond GitHub keyword search. The paper argues the original adoption claim is invalid, but without a defensible alternative measurement, that part of the critique remains incomplete.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify sensitivity of the conclusions to analysis choices: one-sided vs two-sided tests, Bonferroni vs FDR, pooled vs stratified tests, and inclusion/exclusion of the omitted baseline data. ICLR reviewers will want to know whether the paper’s main refutation survives reasonable statistical conventions or hinges on a particular choice.

2. Separate “method failure” from “evaluation failure” by analyzing whether min-p’s apparent parity comes from weak implementation details, poor hyperparameter grids, or fundamentally no advantage. As written, the paper sometimes shows that the original experiments were flawed, but does not always distinguish that from a substantive lack of method value.

3. Provide uncertainty and effect-size analyses for all headline comparisons, not just p-values and point estimates. For a refutation paper at ICLR, reviewers will expect evidence that the observed differences are practically negligible, not merely statistically non-significant.

4. Analyze whether the authors’ own sweeps are biased toward the conclusion they want, especially in the choice and density of hyperparameter grids. The paper’s “control hyperparameter volume” idea is central, but without a principled justification for the sampled grids, the analysis risks becoming another form of selective tuning.

5. Audit the reported claims for consistency across manuscript versions and data releases, with a clear provenance table for every result. The paper alleges selective reporting in the original work; it needs a similarly rigorous traceability standard for its own claims to be maximally convincing.

### Visualizations & Case Studies
1. Add side-by-side plots showing every sampler’s performance distribution across temperatures and hyperparameters, with raw scores, confidence intervals, and the chosen best point highlighted. This would expose whether min-p truly dominates, merely ties, or wins only in isolated settings.

2. Show per-item human-evaluation agreement and disagreement examples, including cases where basic sampling was preferred over min-p. Concrete cases would make it obvious whether the original qualitative summary was misleading or whether the preference signal is genuinely mixed.

3. Plot win-rate curves with uncertainty bands for direct min-p-versus-baseline comparisons rather than indirect comparisons through a reference sampler. This would reveal whether the alleged superiority is robust or disappears once the comparison is made directly.

4. Provide a “hyperparameter-volume vs best-score” visualization with matched budgets across samplers and models. That would make it clear whether min-p’s advantage is real or simply a byproduct of a larger search space.

5. Include a provenance/forensics table for every disputed claim: original statement, source data, reanalysis method, corrected value, and whether the claim survives. For a critique paper, this would make it much easier to assess which parts are strong refutations and which are interpretation.

### Obvious Next Steps
1. Submit a compact replication package that reproduces every corrected table and figure from raw data and code. At ICLR, the paper’s credibility would rise substantially if reviewers could independently verify the refutations end-to-end.

2. Convert the critique into a general evaluation protocol for sampling papers: matched tuning budgets, direct pairwise comparisons, and mandatory uncertainty reporting. That would make the contribution more than a single case study.

3. Run a blinded re-annotation of the human-evaluation free-text responses with two independent annotators and report inter-annotator agreement. This would test whether the qualitative rebuttal is stable rather than a single annotator’s interpretation.

4. Test whether the same failure modes appear in other recent sampling papers that make comparable “quality plus diversity” claims. If the blueprint is meant to generalize, the paper should show that these problems recur beyond min-p.

# Final Consolidated Review
## Summary
This paper is a critique/re-analysis of Nguyen et al. (2024)’s min-p sampling paper. It argues that the original paper’s central claim—that min-p consistently improves quality and diversity—is not supported once one inspects the full human-evaluation data, accounts for multiple comparisons, normalizes for hyperparameter-tuning volume, and scrutinizes the LLM-as-a-judge and community-adoption evidence.

## Strengths
- The paper surfaces several concrete, substantive problems in the original work, especially the omission of one-third of the human-evaluation data and the inconsistent / selective reporting in the LLM-as-a-judge section. These are not stylistic quibbles; if accurate, they directly undermine the credibility of the original claims.
- The hyperparameter-volume analysis is the most interesting methodological contribution: it highlights a real confound in empirical ML comparisons, namely that a method can appear better simply because it was searched more extensively. This point is broadly relevant beyond min-p.

## Weaknesses
- The paper repeatedly states stronger conclusions than the evidence supports. The analyses often show that the original paper’s evidence is insufficient or ambiguous, but the manuscript slides into categorical claims that min-p is invalidated or offers no advantage in general. That overstatement matters because several of the authors’ own analyses hinge on nontrivial methodological choices.
- The “fair comparison by hyperparameter volume” argument is not fully justified. Equalizing by number of swept hyperparameters is plausible, but it is not obviously the right fairness criterion for all samplers or all tasks, and the paper does not adequately explore alternative reasonable normalizations. This leaves the key benchmark critique more persuasive than decisive.
- The new human-evaluation study is difficult to interpret cleanly because too many factors changed at once: sampler implementation, participant pool, hyperparameters, reading time, text, and rubric. The paper acknowledges this, but then still draws fairly strong conclusions from the follow-up study. That weakens the force of the reanalysis.
- The LLM-as-a-judge section is methodologically under-specified in the original paper, but the critique here remains somewhat indirect. The paper correctly flags ambiguity and selective reporting, yet it does not provide a clean direct reproduction of the protocol that would make the refutation maximally compelling.

## Nice-to-Haves
- Provide a clearer separation between confirmed re-analysis results, exploratory analyses, and interpretive judgments. This would make it easier to see exactly which claims are hard evidence versus inference.
- Package a fully self-contained replication artifact with frozen data snapshots and scripts for every table and figure. The current paper relies heavily on external links and public discussions, which makes verification unnecessarily cumbersome.

## Novel Insights
The most novel and valuable insight is that the apparent superiority of a sampling method can be an artifact of search budget rather than intrinsic quality: if one method is tuned much more heavily than others, “best-of-sweep” comparisons can be misleading even when they look rigorous on the surface. The paper also usefully illustrates how multiple small reporting issues—missing data, pooling across conditions, under-specified judge evaluations, and selective reporting—can compound into a convincing but fragile narrative of superiority.

## Potentially Missed Related Work
- Freitag et al. 2021; Belz et al. 2021; Thomson et al. 2024 — relevant for common failure modes in human evaluation.
- Van der Lee et al. 2019; Khashabi et al. 2022 — relevant best practices for human evaluation of generated text.
- Zheng et al. 2023; Xu et al. 2025 — relevant to LLM-as-a-judge methodology and non-transitivity issues.
- Stiennon et al. 2020; Nakano et al. 2021 — relevant to best-of-N style selection and search-budget effects.
- Agarwal et al. 2021 — relevant for statistical caution in empirical comparisons.

## Suggestions
- Rephrase the central conclusion more carefully: emphasize that the original evidence does not support the strong superiority claim, rather than claiming definitive falsification of min-p as a method.
- Add a sensitivity analysis for the core conclusions under reasonable alternatives: two-sided tests, FDR vs Bonferroni, stratified vs pooled analyses, and alternative hyperparameter-budget normalizations.
- For the benchmark section, report a direct budget-matched pairwise comparison table alongside the Best-of-N plots, so the fairness criterion is transparent rather than implicit.
- For the human-evaluation sections, clearly show the full dataset and a provenance table listing each disputed claim, the original statement, the corrected analysis, and the resulting conclusion.


# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0]
Average score: 2.7
Binary outcome: Reject
