## Summary
This paper proposes GEFA, a black-box feature attribution framework that operates in a continuous proxy space over feature-presence probabilities, allowing a path-integral style explainer to be applied to both discrete and continuous inputs. The main claimed contributions are a proof that the resulting attributions equal Shapley values, a conceptual link to Integrated Gradients, and a simple control variate intended to reduce Monte Carlo variance.

## Strengths
- **Clear conceptual extension beyond prior gradient-estimation explainers.** The proxy-variable construction in Section 4.1 is a meaningful idea: by defining explanations over Bernoulli feature-presence parameters rather than the original input space, the framework can in principle handle discrete inputs such as text without requiring access to internal embeddings.
- **Strong and coherent theoretical positioning.** The paper states a compelling sequence of results: axiomatic properties in Theorem 1, Shapley equivalence in Theorem 2, unbiasedness under the control variate in Theorem 3, and a connection to IG in Theorem 4. Even without the appendix, the main text presents a coherent narrative tying together gradient estimation, path methods, and cooperative-game-style attribution.
- **The text/image pairing is a reasonable first demonstration of cross-modality applicability.** The experiments do at least show the method on one discrete-input setting (BERT sentiment) and one continuous-input setting (ImageNet/InceptionV3), which is appropriate evidence for the claim that the proxy-space formalism is broader than GEEX.
- **The paper is generally well motivated and the explanation of why discrete inputs are challenging for path methods is good.** The discussion in the introduction and Section 4 makes the scope and intended advantage of the method easy to understand.
- **The control-variate idea is simple and practically motivated.** Section 4.3 gives a concrete variance-reduction mechanism rather than only a theoretical method, and the experiments show some empirical gain from it.

## Weaknesses

###: Fatal

### Major:
- **The empirical claim that GEFA improves explanation quality over competing black-box/Shapley-based methods is only partially supported, especially in text, because the evaluation is entangled with different feature-absence semantics.** In Section 5.2, GEFA uses *token removal* as a natural absence notion for text, while IG/VG necessarily rely on zero embeddings; the paper itself argues that this semantic difference is the main reason GEFA can outperform IG under token removal. That makes the strongest text result in Table 1 difficult to interpret as evidence that GEFA is the better attribution estimator rather than that token deletion is a harsher or more faithful intervention. This does **not** invalidate the method, but it does weaken the headline empirical claim of superior explanation quality.
- **The evaluation is too narrow for the paper’s broad “general / for all / regardless of input type” framing.** The paper evaluates one text classifier and one image classifier, both in standard classification settings. That is enough to show promise across two modalities, but not enough to substantiate claims like “generally applicable to arbitrary black-box models” or “for all.” In particular, there is no evidence on mixed-type tabular data, structured/categorical inputs beyond token sequences, non-classification outputs, or settings where absence semantics are especially problematic.
- **The experimental section is too thin to substantiate the paper’s practical efficiency/variance claims.** GEFA’s practical argument rests on lower information waste and improved variance, yet the experiments report only single nAOPC values at one query budget per domain (500 for text, 5000 for images), with no query-budget ablation, convergence curves, run-to-run variability, or confidence intervals. Because the method is explicitly Monte Carlo and Section 4.3 introduces a variance-reduction contribution, this missing analysis materially limits how convincing the practical claims are.
- **The paper relies almost entirely on deletion-based nAOPC, which is an imperfect and potentially method-favoring evaluation for a masking-based explainer.** The authors do acknowledge concerns about deletion metrics and refer to Appendix B for additional discussion, so this criticism should be weakened rather than overstated. Still, in the main paper there is no complementary insertion-style, synthetic-ground-truth, or exact-small-game evaluation to show that the rankings are not an artifact of the chosen perturbation metric. Given that GEFA itself is built around feature masking/removal, stronger triangulation would have significantly improved confidence.

### Minor
- **Equation (8) is not presented as clearly as it should be for a central estimator.** The notation  
  \[
  \frac{1}{n}\sum_{\gamma \sim \mathcal U[0,1]} \sum_{\epsilon \in \pi(\gamma \mathbf{1}_p)} \cdots
  \]
  is ambiguous about whether one mask is sampled per \(\gamma\) or whether the inner term denotes an expectation. Algorithm 1 strongly suggests one \((\gamma,\epsilon)\) sample per iteration, but the equation reads less clearly. This is a presentation issue, but it matters because it affects reproducibility and interpretation of the Monte Carlo variance.
- **The control-variate assumption is plausible but under-characterized.** Assumption 1 requires nonzero correlation between model output and the number of present features. The paper does discuss why this may hold less strongly for language due to negation/irony, which is a reasonable acknowledgement, but it does not empirically measure when the assumption holds or fails. Since the control variate is a named contribution, a bit more characterization would help.
- **The image experiments are informative but still narrower than the claims they support.** The ImageNet result is fairly strong within the chosen setup, but it mainly shows that GEFA works well for a pixelwise Bernoulli-masking game on one CNN. It is weaker evidence for broad superiority over Shapley estimators in general.
- **The paper’s practical discussion of computational tradeoffs is limited.** Query budgets are reported, but there is no runtime comparison or analysis of how estimation quality scales with dimensionality and budget. For a black-box method intended for high-dimensional settings, this would be valuable.

### Trivial
- **Some central claims in the abstract/conclusion are stronger than the evidence shown.** The broadest wording should be toned down to match what is actually demonstrated experimentally.

## Nice-to-Haves
- Add query-budget/convergence plots against PSHAP and GEEX to directly substantiate the “information waste” and variance-reduction claims.
- Include at least one additional evaluation axis beyond deletion nAOPC, e.g., insertion, exact Shapley recovery on small synthetic tasks, or a controlled faithfulness benchmark.
- Test a mixed-feature tabular dataset, which would be a particularly natural setting for the paper’s “general feature type” claim.
- Empirically report the correlation underlying Assumption 1 and analyze when the control variate helps or fails.
- Provide runtime or query-efficiency comparisons, not just final quality scores.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The theorem cannot be assessed because the appendix is unavailable / proof deferred, therefore the claim is an evidential gap.”**  
  Removed because this is largely a review-context limitation rather than a paper flaw. The paper explicitly states that the proofs are in the appendix, which is standard practice for theorem-heavy submissions. It is still fair to say the main text could better present estimator details, but not fair to criticize the authors because the appendix was not included in the extracted prompt.
- **Criticisms doubting the existence/availability of cited models, benchmarks, datasets, or code.**  
  Removed per instruction.
- **Complaints about missing related work not verifiable from the paper text.**  
  Removed per instruction.
- **Pure formatting/style/parser artifacts.**  
  Removed per instruction.
- **Claims that the paper is unfair because KernelSHAP was excluded on ImageNet.**  
  Removed as a main weakness. The paper explicitly explains that “solving the linear regression requires a query budget matching the dimensionality of the input feature space,” which is a reasonable practical scope choice rather than a methodological flaw. The more defensible criticism is that the resulting evidence supports a narrower claim than the broad framing, which is already captured above.
- **Strong novelty attack claiming the Shapley connection is merely known/reformulated.**  
  Removed as too speculative without external verification. The paper’s framing can still be judged for overclaiming practical generality, but not for undocumented novelty disputes.

## Novel Insights
The most interesting synthesis here is that the paper’s theory and experiments support **different levels of generality**. The theoretical construction is genuinely broad: defining explanations in proxy space neatly unifies discrete and continuous inputs and gives a principled bridge between gradient-estimation methods and Shapley-style attribution. But the experimental evidence is much narrower and, in text especially, partially confounded by differing absence semantics across methods. So the paper is best viewed not as having fully established a universally better explainer, but as having introduced a promising and theoretically elegant **new computational route to baseline-conditioned Shapley-style attribution under black-box access**, with practical superiority shown only in limited settings.

## Suggestions
- Narrow the headline claims to match the evidence: emphasize “a general framework in principle, demonstrated on text and image classification” rather than “for all / arbitrary black-box models.”
- Strengthen the empirical section with query-budget ablations, repeated runs, and confidence intervals; this is especially important because the paper explicitly claims variance reduction.
- Separate “better attribution algorithm” from “better absence semantics” in the text experiments. A cleaner comparison would evaluate methods under as matched an intervention as possible, then separately discuss the benefits of token removal as an absence model.
- Add one complementary evaluation beyond deletion nAOPC, ideally one that does not so closely mirror the masking mechanism used by GEFA.
- Clarify Equation (8) to align unambiguously with Algorithm 1.
- Provide at least one experiment on mixed-feature tabular data or another non-image/non-text domain if the broad generality claim is to be retained.

## Score and Decision
**Assessment across axes:**  
- **Originality:** Moderate to high. The proxy-space path formulation is interesting and appears meaningfully distinct from prior continuous-input gradient-estimation explainers.  
- **Importance:** Good problem choice; black-box feature attribution across discrete/continuous domains is important.  
- **Support for claims:** Mixed. The theory is ambitious and coherent, but the empirical support for the strongest practical claims is incomplete.  
- **Experimental soundness:** Moderate. Results are promising, but the evaluation is narrow and lacks variance/query-efficiency analysis.  
- **Clarity:** Generally good, with some ambiguity in the estimator presentation.  
- **Value to the community:** Positive, especially for researchers interested in the bridge between Shapley values, IG, and black-box gradient estimation.

**Calibration papers used:**  
- **/home/wg25r/review_agent/human_reviews/Fq25rH3ytL.md** (Reject; scores 3,3,3,5): another black-box gradient-estimation explanation paper that was viewed as too weak in novelty/evaluation. The current paper is **stronger** than this anchor because its theory is more substantial and the discrete-input extension is more compelling.  
- **/home/wg25r/review_agent/human_reviews/CNZmaInj9n.md** (Reject; scores 3,6,6): a Shapley estimation paper with some promise but concerns about novelty and empirical support. The current paper is in a **similar band**, perhaps slightly stronger theoretically, but still limited by thin empirical substantiation.  
- **/home/wg25r/review_agent/human_reviews/rGP2jbWt0l.md** (Accept Poster; scores 6,6,6,3): an attribution paper that initially faced concerns about optimizing/evaluating the same metric but still had enough empirical strength to land around acceptance. The current paper is **a bit weaker overall** because its broad claims are less fully supported and the variance-efficiency contribution is under-evaluated.  
- **/home/wg25r/review_agent/human_reviews/wg3rBImn3O.md** (Accept Spotlight; scores 8,6,8): a strong Shapley estimation paper with clear theory plus strong empirical validation. The current paper is **well below** this bar because it lacks comparable empirical depth and practical calibration.

Putting these together, this submission looks better than clearly weak rejects in the area, but below the level of a confident accept. It feels like a **borderline reject / weak reject**: a paper with real ideas and real contributions, but not yet enough experimental substantiation for its strongest claims.

**Score: 4.5 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>