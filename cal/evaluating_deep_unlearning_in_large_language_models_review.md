=== CALIBRATION EXAMPLE 9 ===

# Final Consolidated Review
## Summary

This paper introduces **deep unlearning**, a formalization of LLM fact unlearning that accounts for logical deducibility: a target fact is truly unlearned only if it cannot be reconstructed by applying a known rule set to the facts retained in the model. The authors propose recall and accuracy metrics grounded in the notion of minimal deep unlearning sets, construct a controlled synthetic benchmark (EDU-RELAT) of family relationships and biographies with 48 logical rules, and empirically demonstrate that four standard unlearning methods all fail to achieve high recall without severe collateral unlearning across four LLMs of varying scale.

---

## Strengths

- **Formally isolating a real vulnerability in prior unlearning work.** The paper cleanly demonstrates that superficial fact unlearning leaves a trivially exploitable attack surface: an adversary who extracts retained premises and applies known rules externally can reconstruct the "unlearned" target. This vulnerability is not merely theoretical—Figure 1 and Figure 5 together show that GA achieves near-perfect superficial unlearning while still failing deep unlearning, making the gap concrete and quantifiable rather than speculative.

- **EDU-RELAT dataset design is methodologically principled.** The choice of a synthetic benchmark is not laziness—the paper provides specific, well-reasoned arguments for why real-world knowledge bases fail here: (1) partial observation can falsely signal success, and (2) differing underlying knowledge bases across LLMs prevent cross-model conclusions. The synthetic construction mirrors realistic constraints (birth-year alignment, name inheritance, geographically realistic birthplace distributions), reducing the gap to real-world settings within the controlled domain.

- **The negative result is surprisingly strong.** Even with a metric that is (as noted below) arguably optimistic in the method's favor, no method achieves recall ≥ 0.8 and accuracy ≥ 0.8 simultaneously when unlearning a *single* fact. The paper notes this is remarkable because "accuracy of 0.8, i.e., dropping 0.2 from 1, is actually a high cost, as this is the cost of unlearning only single fact." This emphatic failure is directly actionable: current methods are simply not designed for this setting.

- **The finding that larger LLMs perform better at deep unlearning is non-trivial and hypothesis-generating.** The ~0.1–0.2 improvement in Acc@Recall≥0.8 for Llama2-7b and Llama3-8b over the smaller models is one of the few cross-cutting signals in the experiments. The hypothesis—that larger models encode stronger inter-fact correlations, which helps unlearning methods propagate edits—is plausible and opens a concrete empirical research question for future work.

- **The mechanistic analysis of WHP's plateau is specific and informative.** The explanation (lines 354–355) that certain token probability dimensions are invariant to α due to the max(·, 0) operator provides a method-level insight that would be directly useful to practitioners or method designers working on WHP-style approaches.

---

## Weaknesses

### Fatal

None. The paper's main empirical claim—that existing methods fail at deep unlearning—is robust to the identified concerns and even strengthened by them.

### Major

- **The max-based recall metric is conceptually misaligned with the adversarial threat model.** The threat model is that an adversary can reconstruct the target fact if *any* intact deductive path exists (i.e., any minimal deep unlearning set is left unbroken). Deep unlearning (Definition 2) requires that the target fact is not in the deductive closure of retained facts—this fails if *any* minimal set is untouched. Yet Equation 1 takes the *maximum* overlap across all minimal sets. A method that completely eliminates one deductive path while leaving another fully intact would receive recall = 1.0 despite failing at deep unlearning. A min-based or coverage-based metric would more faithfully measure whether every path has been disrupted. Crucially, the paper's conclusion that all methods fail is *not* undermined by this—it is actually strengthened, since the true failure is worse than reported. However, the metric misaligns with the paper's own formal definition and should be addressed in future iterations or explicitly acknowledged as a conservative lower bound on failure.

- **No oracle upper bound experiment.** There is no baseline that provides the method with the true minimal deep unlearning set (from Algorithm 3) and then executes unlearning on those specific facts. This would directly answer whether the failure is in *identifying* which facts to unlearn (the algorithm's diagnostic problem) or in *executing* targeted unlearning once the set is known (a transformer architecture limitation). Without this, the paper cannot distinguish between two fundamentally different failure modes, which is important for guiding future method design.

- **Approximation quality of Algorithm 3 is unvalidated.** The paper relies on Algorithm 3 to approximate the full set of minimal deep unlearning sets, which drives both the recall and accuracy metrics. The qualitative endorsement—"more than half of the facts have 6–17 different sets found"—demonstrates diversity but does not bound how well the sampled collection approximates the complete set. For facts with dense deduction graphs, the true number of minimal sets could be much larger. A small-scale comparison against exact enumeration (even on simplified subgraphs where it is tractable) would substantially strengthen confidence in the metric values reported throughout the paper.

### Minor

- **Single question template per fact introduces prompt-sensitivity risk.** Table 1 shows one fixed question–answer pair per fact. LLM knowledge elicitation is known to be sensitive to phrasing: a model may fail to answer "Who is X to Y?" but correctly answer a paraphrase, leading to false positives in "forgetting" assessments. Even a small-scale paraphrase experiment would calibrate how much this affects the results.

- **Distribution of minimal deep unlearning set sizes is not reported.** The paper states that deep unlearning a single fact can require unlearning "more than 10 facts," which is striking, but no distribution over the 55 target facts is provided. A histogram would allow readers to understand whether difficulty is concentrated in a few hard cases or uniformly distributed, and would help contextualize what a recall score of, say, 0.65 actually means across easy and hard instances.

- **Rule set construction and completeness are unexplained.** The paper states that the rule set has 48 rules (Table 2 shows a subset) but does not describe how they were derived—whether manually curated, from an ontology, or generated. Missing rules would lead to underestimated minimal deep unlearning sets, making the benchmark easier than intended. A brief statement on the construction process and whether the rules are claimed to be exhaustive for the family domain would clarify this.

- **The 0.8 threshold is not justified.** Acc@Recall≥0.8 and Recall@Acc≥0.8 use an unexplained threshold. The full Acc-Recall curves in Figure 4 partially address this, but the primary aggregated numbers in Figure 3 and Table 3 are all tied to this one value.

### Tiny

- **WHP is the only method with a mechanistic explanation**; the others are left as empirical observations. Even brief hypotheses for why GA/NPO/TV behave as they do would make the discussion more instructive.

---

## Nice-to-Haves

- **Include model-editing methods (e.g., ROME, MEMIT) as additional points of comparison.** These methods are explicitly designed for localized fact modification in LLMs and are the closest methodological alternative to targeted fact unlearning. The paper acknowledges them in related work (Section 6) and they are outside the paper's stated scope of *unlearning* methods—but including them would strengthen the empirical claim that the problem is hard across all current fact-targeting approaches.

- **Deduction-chain-depth ablation.** Analyzing whether unlearning difficulty scales with 1-hop vs. 2-hop vs. 3-hop deduction chains would distinguish whether the failure is driven by logical complexity or simply by the blunt nature of the unlearning methods' gradient signals.

- **Discussion of how findings translate to non-conjunctive or probabilistic rule settings.** The framework is restricted to conjunctive Horn rules, which excludes negation, existential quantification, and probabilistic entailments common in real-world knowledge graphs. A brief discussion of what extensions to the framework would be needed for these cases would help contextualize the scope of the contribution.

- **Expand the number of evaluation facts** from 55 to a larger set to permit more fine-grained statistical comparisons, especially for cross-method differences that currently sit within standard deviation ranges.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"formally proposes" overstates mathematical originality** (Harsh Critic): This is a style nitpick. The novelty is in applying deductive-closure machinery to the unlearning setting, which is clearly novel in context.

- **ROME/MEMIT exclusion as fatal gap** (Harsh Critic): The paper's scope is LLM *unlearning* methods. ROME and MEMIT are model-editing methods, and their exclusion is appropriate given the stated scope. This is a nice-to-have, not a weakness.

- **Fine-tuning vs. pretraining knowledge regime concern as a major flaw** (Harsh Critic): The paper explicitly acknowledges this in Section 7—"Current unlearning methods would likely have numerically higher accuracy on real data than on our benchmark"—and argues that the real-world task is actually *harder*, not easier. The synthetic fine-tuning protocol is a deliberate design choice for controlled evaluation, and the discussion adequately contextualizes it.

- **Assumption of perfect logical reasoning undermines the premise** (Review 2): The paper's threat model does not require the LLM to perform multi-hop internal reasoning. The adversary extracts atomic facts from the LLM and applies rules *externally*. The evaluation protocol tests whether individual atomic facts are retained, not whether the LLM deduces conclusions. This concern is based on a misreading of the threat model.

- **Unfair comparison / need for statistical significance testing beyond error bars** (Harsh Critic): Error bars over 55 facts across multiple LLMs are standard for this type of evaluation. Demanding formal hypothesis testing here is not the norm in LLM unlearning evaluation literature.

---

## Novel Insights

The most substantive novel observation, which cuts across all three reviews, is that the max-based recall metric—while conceptually misaligned with the adversarial threat model (where any intact deductive path suffices to reconstruct the target)—actually makes the paper's negative conclusion *more conservative*, not less. If a min-based or coverage-based metric were used, every reported failure would look worse. This asymmetry is underappreciated in the paper itself: the authors frame the max as a reasonable measure, when they could instead frame it as a deliberately lenient metric to avoid any possible objection that they are unfairly penalizing methods—and note that even this lenient evaluation reveals total failure. Reframing this would strengthen both the theoretical presentation and the rhetorical impact of the empirical results.

---

## Suggestions

1. **Add a min-coverage metric alongside the existing max-recall**, or reframe the current max-recall as a deliberate lower bound on failure (i.e., if even this lenient metric shows failure, things are only worse). Either choice makes the theoretical framing more coherent with Definition 2.

2. **Run an oracle baseline**: take the minimal deep unlearning sets from Algorithm 3 and explicitly run gradient ascent (or another method) on those exact facts. Report its recall and accuracy. This single experiment would answer the most important open question the paper raises.

3. **Validate Algorithm 3 on small subgraphs** where exact enumeration of all minimal vertex cuts is tractable. Even 5–10 examples where the approximation is compared to the ground truth would provide quantitative confidence in the metric.

4. **Report the distribution of minimal deep unlearning set sizes** across the 55 target facts (e.g., a histogram or box plot). This is a low-cost addition that substantially aids interpretation of all reported recall numbers.

5. **Include at least one paraphrase robustness check** for the probing methodology—test a small subset of "forgotten" facts with alternative question phrasings to calibrate false-positive rates in forgetting assessment.

---

**Evaluation summary:**

- **Novelty**: High — the deep unlearning formalization is a genuine conceptual advance over all prior fact unlearning work.
- **Technical soundness**: Moderate-to-high — the formalism is clean and the algorithms are principled, but the recall metric has a meaningful conceptual gap that the paper does not acknowledge.
- **Empirical support**: Moderate — the evidence for failure is compelling and consistent across methods and models, but the reliance on a single synthetic domain, 55 evaluation facts, and one question template per fact limits the strength of the empirical claims.
- **Significance**: High — the paper exposes a specific, actionable failure mode in a problem (fact unlearning) that is directly relevant to privacy regulation compliance, and does so with sufficient rigor to guide future method development.
- **Clarity**: Good — the formalism is accessible, the running example is consistently maintained, and the key finding is unambiguous.

# Actual Human Scores
Individual reviewer scores: [5.0, 5.0, 6.0]
Average score: 5.3
Binary outcome: Reject
