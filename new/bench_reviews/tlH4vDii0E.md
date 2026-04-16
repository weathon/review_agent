## Summary
This paper proposes Causal Transfer Learning (CTL), a method for improving OOD robustness of fine-tuned PLMs in single-domain settings by combining two ideas: using pretrained vs. fine-tuned representations as paired “environments” to extract an invariant feature \(C\), and using token-level local features \(\Phi\) in a front-door-style adjustment to estimate \(P(Y\mid do(X))\). Empirically, CTL improves over vanilla fine-tuning and a few robustness-oriented baselines on controlled sentiment-classification shifts with injected spurious correlations.

## Strengths
- **Interesting problem and framing.** The paper targets a relevant setting—single-domain OOD robustness for fine-tuned PLMs—where multi-domain DG assumptions are often unavailable. The causal motivation for preferring \(P(Y\mid do(X))\) over \(P(Y\mid X)\) is clearly laid out in Sections 3–4.
- **Creative use of PLMs as paired views.** The idea in Assumption 2—treating pretrained and fine-tuned representations of the same input as paired representations from different environments—is novel and practically appealing, since it does not require collecting extra domains.
- **Useful ablations that probe the method structure.** The inclusion of CTL, CTL-N, CTL-C, and \(\Phi\)-only variants is genuinely informative. In particular, the poor OOD behavior of CTL-\(\Phi\) and the stronger performance of CTL/CTL-C support the claim that the method is doing more than just exploiting the injected shortcut.
- **Consistent gains on the paper’s controlled benchmarks.** On both semi-synthetic setups and the dataset-built-from-real-reviews setting, CTL generally improves over SFT and often over SWA/WISE as the shortcut correlation flips more aggressively. The gains at severe shift are nontrivial, e.g. Amazon OOD 10%: 58.40 (CTL) vs 49.24 (SFT), and the analogous trend also appears in Table 2.
- **Clarity of high-level presentation.** Despite some theoretical gaps, the paper is easy to follow at a conceptual level, with explicit assumptions, algorithm sketches, and a coherent story from motivation to method.

## Weaknesses

###: Fatal
- **The central identification claim for the “causal” predictor is not established convincingly.**  
  This is the most serious issue because the paper’s main contribution is not just a robustness heuristic; it is explicitly framed as identifying and estimating \(P(Y\mid do(X))\) via a front-door construction. However, Theorem 2’s proof does not justify the stated formula. In particular, the step
  \[
  P(y\mid do(c)) = \sum_{\Phi'} P(y\mid \Phi', c)P(\Phi')
  \]
  is asserted as “Frontdoor Criterion & Assumption 3 and 4,” but the paper does not show the graphical conditions needed for a valid front-door adjustment in the graph of Fig. 1(c), nor explain why conditioning on \(c\) together with mediation through \(\Phi\) yields this expression. Assumption 4 states that fixing \(\Phi\) gives no extra information once \(C\) is fixed:
  \[
  P(Y\mid do(\Phi),do(c)) = P(Y\mid do(c)),
  \]
  which if anything makes \(\Phi\) look redundant given \(C\), not like a standard front-door mediator supporting Eq. (1). Since the paper’s “causal estimator” interpretation depends on Theorem 2, this is a core soundness issue rather than a minor proof omission.

### Major:
- **There is a substantial gap between the theorem and the implemented algorithm.**  
  Eq. (1) marginalizes over \(x'\) through \(P(\Phi' \mid x')P(x')\), but Algorithms 1–2 approximate this by shuffling \(\Phi\) within a minibatch and averaging across such shuffles. The paper does not provide assumptions under which minibatch shuffling is a valid estimator of the stated quantity, nor analyze dependence on batch composition. As written, a test example’s prediction depends on what other samples happen to share its minibatch, which is an unusual and consequential property for a purported interventional predictor. This weakens the claim that CTL is an implementation of the theorem rather than a heuristic inspired by it.
- **The “real-world” evaluation does not actually test natural real-world distribution shift.**  
  Section 6.2 is described as a real-world experiment, but the paper explicitly says the spurious signal is injected by appending markers such as “amazon.xxx” and “yelp.yyy” into the text. That is still a controlled synthetic artifact, just on data derived from real reviews. This supports claims about robustness to injected shortcuts, but not the broader framing in the abstract and introduction about practical real-world OOD generalization under naturally arising spurious correlations.
- **Assumption 2 is strong and insufficiently validated.**  
  The method relies heavily on the claim that pretrained and fine-tuned representations of the same text preserve the same causal factor \(C\) while differing in spurious factors \(S\). But fine-tuning can alter both useful and spurious content. The paper presents this as a key assumption motivated by prior theory, but does not empirically validate that the PLM/SFT pair really satisfies the required premises in this NLP setting. Since Theorem 1 depends on this paired-view interpretation, the lack of evidence here matters.
- **The empirical support for superiority over existing approaches is narrower than the paper claims.**  
  The baselines used are SFT0, SFT, SWA, and WISE. These are reasonable baselines, but they are not enough to support the broad claim in the abstract of “superior generalizability ... compared to existing approaches,” especially given the paper’s framing around causal/domain-generalization methods for single-domain robustness. This is an evidential limitation on the comparative claim, even though the chosen baselines themselves are not inappropriate.

### Minor
- **The practical value of the front-door component over the simpler CTL-C variant is not fully clarified.**  
  CTL usually beats CTL-C, especially in Table 2 at stronger shift, so it is not fair to say the front-door part does nothing. But on some semi-synthetic settings the margins are very small, suggesting that much of the gain may already come from the invariant feature \(C\). The paper would be stronger if it explained more clearly when the extra front-door-style adjustment materially helps.
- **The experimental scope is limited.**  
  All experiments are binary sentiment classification. The paper notes the method could extend to other NLP tasks, but this is not demonstrated. This does not invalidate the current experiments, but it does limit the breadth of the claims.
- **Uncertainty reporting in the main tables is light.**  
  The paper reports mean F1 over 5 runs and includes box plots, which is better than single-run reporting. Still, standard deviations or confidence intervals in Tables 1–2 would make the smaller gaps easier to assess directly.

### Trivial
- **Some notation/presentation remains confusing around Theorem 2 and Eq. (1).**  
  For example, the role of \(\hat{\Phi}'\) versus \(\Phi'\) in Eq. (1) is not clearly explained in the main text. This is not just style, because the notation contributes to ambiguity in the estimator definition.

## Nice-to-Haves
- Add a sensitivity study for batch size, number of shuffle samples \(K\), and the choice of 10 token patches, since these directly affect the practical approximation of Eq. (1).
- Include qualitative or probing analysis of what \(C\) and \(\Phi\) capture, to test whether the intended causal/spurious decomposition is reflected in practice.
- Evaluate on at least one naturally shifted NLP benchmark and one additional task/model family to support broader generalization claims.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Complaints about code/model availability or release status.** The paper cites the relevant models/tools and states code is in supplementary / to be released upon acceptance; per instruction, existence/release-status criticisms are removed.
- **Pure formatting/parser issues.** For example, the duplicated “SFT” row name in Table 2 may well be extraction noise from the PDF text dump; this is not reliable enough to treat as a substantive weakness.
- **Overstated baseline-demand phrasing that assumes external methods must be included.** It is fair to say the current baselines are insufficient to support the paper’s broad comparative claim; it is not fair, without external verification, to insist that specific untested methods necessarily would dominate or that omission alone invalidates the paper.

## Novel Insights
The paper appears strongest not as a validated causal-identification result, but as an interesting robustness heuristic built from two ingredients: cross-state alignment between pretrained and fine-tuned representations, and randomized local-feature averaging at inference. The empirical evidence suggests the first ingredient may already do much of the work, while the second adds some extra robustness under stronger shortcut shifts. That reframing matters: the work likely has real practical signal, but the current presentation over-attributes that signal to a front-door identification theorem that is not yet adequately justified.

## Suggestions
- Rework Theorem 2 carefully: either provide a correct identification argument with explicit graphical conditions, or scale back the claim and present CTL as a causally motivated approximation rather than an identified estimator of \(P(Y\mid do(X))\).
- Justify the minibatch-shuffling estimator formally or empirically; at minimum, analyze sensitivity to batch size/composition and explain why predictions depending on batch peers is acceptable.
- Temper the “real-world” and “existing approaches” claims in the abstract/introduction unless additional evaluations are added.
- Add an empirical test of Assumption 2, e.g. probing or alignment analyses showing what information is preserved/changed from \(R_0\) to \(R_1\).
- Expand evaluation to at least one natural-shift benchmark and one additional task beyond binary sentiment classification.
- Report standard deviations (or similar uncertainty measures) in the main result tables.

## Score and Decision
**Originality:** moderate-to-good. The PLM-as-paired-environment idea is interesting.  
**Importance of the question:** high. Single-domain OOD robustness for fine-tuned PLMs is a meaningful problem.  
**Support for claims:** mixed to weak for the main causal-identification claim; moderate for the narrower claim that the method improves robustness on injected-shortcut benchmarks.  
**Experimental soundness:** decent within the paper’s controlled setup, but limited by the synthetic nature of the shifts and the narrow baseline/task scope.  
**Clarity:** fairly good at a high level, though the core theorem/estimator details are unclear.  
**Value to the community:** potentially useful as a promising direction or heuristic, but not yet solid enough in theory/evidence for acceptance in current form.

**Calibration.** I compared this paper against:
- **OatZMyMuIo** (“Causal Representation Learning and Inference for Generalizable Cross-Domain Predictions,” scores 5/5/3/3, Reject): similar pattern of interesting causal-DG idea plus serious invariance/identification concerns. The current paper is in a similar bucket, with somewhat cleaner empirical trends but still a central theorem problem.
- **fHZ04oyEed** (“Representation Learning from Interventional Data,” scores 5/3/3/3, Reject): similar weakness pattern of limited comparative evidence and a gap between motivating theory and realistic evaluation. The current paper is somewhat better organized and more empirically consistent, but still below acceptance.
- **wFf9m4v7oC** (“Causal Inference with Conditional Front-Door Adjustment...,” scores 6/6/6/5, Accept): this is a useful high-side anchor because it also concerns front-door adjustment. That paper was accepted despite some theory-to-practice gap, but reviewers there felt the core front-door theorem itself was mathematically credible. Here, the corresponding identification claim is the primary unresolved issue, so this paper should score materially lower.

Given those anchors, this paper lands below the accepted front-door paper and closer to the rejected causal-representation papers with core invariance/identification issues. The empirical results are promising enough to avoid a very low score, but the main theoretical claim is too shaky for acceptance.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>