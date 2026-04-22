## Summary

The paper studies how prediction probabilities of image classifiers change under different localized perturbations, and what this implies for perturbation-based fidelity metrics used to evaluate saliency maps. It formalizes two desiderata on probability drops and ranking stability (P1/P2), proposes conformity scores DROP and PSim to quantify them, and empirically shows that typical models and perturbations substantially violate these desiderata across architectures, datasets, and perturbation schemes.

## Strengths

- Clear, simple formalization of two intuitive perturbation-behavior conditions (P1: probability drop on perturbation, P2: ranking stability across perturbations) in Sec. 2.1, connected to PIR-based evaluation procedures.
- Introduction of DROP and PSim (Sec. 2.2–2.3, Alg. 1) as model-agnostic diagnostics; they can be computed using only model predictions under perturbations and provide compact summaries of directionality and rank stability.
- Broad empirical coverage: 3 standard CNN architectures plus 2 adversarially trained variants, 3 standard image datasets, 9 perturbation types, and both pixel-wise and segment-wise schemes (Sec. 4, Tables 1–2), with tens of millions of forward passes.
- Consistent empirical finding that DROP ≈ 0.5–0.6 and PSim ≈ 0.3–0.6 instead of near 1 (Sec. 5.1–5.3, Tables 1–2, Figs. 2–5), showing that these desiderata fail widely rather than in isolated cases.
- Identification of relatively more “stable” perturbation families (Gaussian blur, and Telea vs. Navier-Stokes inpainting) via both distributions and KDE-based probabilities (Secs. 5.2–5.3, Figs. 2–5), providing constructive guidance.

## Weaknesses

### Fatal

None.

### Major

- **Overstated and somewhat strawman characterization of fidelity metrics’ assumptions.**  
  The paper repeatedly frames P1 (“there is a drop in the output probability when a pixel is perturbed”, instantiated as \(p_0 > p_i^\phi\ \forall i,\phi\), Eq. (2)) and P2 (rank invariance across perturbations, Eq. (5)) as assumptions that “perturbation based fidelity metrics should conform to” (Sec. 2.1, line 88; Sec. 5.1, 204), and concludes that “fidelity metrics… do not always hold, making them inconsistent and unreliable” (abstract) and “would be rendered unreliable and fail the sanity checks” (Sec. 6). In the original literature, however, metrics like AOPC, AD/IC/W, or “faithfulness” are defined relative to a fixed perturbation scheme and rely on relative drops along an explanation-induced ordering, not on monotone decrease for all pixels or invariance of PIR across qualitatively different perturbation operators. The paper cites Tomsett et al. (2020) for inconsistencies but does not show that those works (or others) require P1/P2 in the strong form used here. As a result, the empirical violation of P1/P2 establishes that these particular desiderata rarely hold, but does not by itself justify the strong headline that existing fidelity metrics are fundamentally “inconsistent and unreliable.”

- **No direct evaluation of actual fidelity metrics or explanation methods.**  
  While the abstract and introduction emphasize “inconsistencies in existing fidelity metrics” and the need for caution in their use, all experiments operate directly on model probabilities and PIR rankings, without ever computing AOPC, AD/IC/W, or “faithfulness” scores for saliency methods under different perturbations. There is no experiment showing, for example, that varying perturbation type reorders a set of saliency methods, or that low DROP/PSim corresponds to metrics mis-ranking obvious baselines. This makes the link from “PIR variability” to “metric unreliability” largely speculative.

- **Logical gap between DROP/PSim findings and the central claim about metric reliability.**  
  The main argument chain (Sec. 5.1–5.3 and Sec. 6) is: DROP and PSim are far from 1 → P1/P2 are violated → PIR is high-variance → “metrics that implicitly rely on the invariance of PIR… would be rendered unreliable” (Sec. 6). Even granting the first two steps, the paper does not establish that the fidelity metrics in question, as actually used, require such strong PIR invariance. Nor does it measure how much observed PIR variability affects downstream metric conclusions. Without a demonstrated connection between low DROP/PSim and concrete failures of fidelity metrics, the conclusion that those metrics are “inconsistent and unreliable” overstates what the evidence supports.

### Minor

- **Assumptions P1/P2 are not clearly motivated as necessary conditions rather than heuristic ideals.**  
  P1 requires decrease of \(p_0\) on essentially every local perturbation; P2 requires near-identity of PIR across distinct perturbation operators. In realistic models, some perturbations can legitimately increase \(p_0\), and different perturbations can emphasize different structures. The paper notes that DROP and PSim have “ideal value 1” and treats “higher is better” (Secs. 2.2–2.3), but does not analyze when deviations from 1 actually undermine the intended use of perturbation-based evaluation, versus being benign.

- **Gaussian blur recommendation is weakly grounded.**  
  The paper concludes that “Gaussian Blur was relatively consistent… [and] should be considered” (Sec. 5.2, 5.3, 6) because it yields higher DROP and PSim. However, it does not validate that blur-based perturbations lead to fidelity metrics that better correlate with any external notion of correctness (e.g., ground-truth masks, human annotations) or with task performance. At present, “more internally stable under DROP/PSim” is not clearly shown to mean “better for evaluating faithfulness.”

- **Scope of experiments relative to claims about adversarial training.**  
  Table 2 shows that the two adversarially trained ResNet50 variants still have low DROP/PSim. The conclusion carefully avoids overclaiming (“we refrain from making conclusive remarks”, line 206), but the abstract and introduction could more clearly frame adversarial experiments as exploratory rather than as evidence about fidelity metrics in robust models more generally.

### Trivial

- Some minor notational/presentation inconsistencies (e.g., Algorithm 1 using \(\delta\mathcal{P}\) both as list of deltas and then appending a count, and the mismatch between comments and equations at lines 241–256) slightly reduce clarity but do not affect the core methodology.

## Nice-to-Haves

- Evaluate at least one or two standard perturbation-based fidelity metrics (e.g., AOPC, AD/IC/W, faithfulness) across multiple saliency methods and perturbation types, and correlate their stability or ranking changes with DROP/PSim. This would empirically ground the argument that conformity scores are useful “preconditions” for metric use.
- Provide qualitative case studies showing images where perturbation pairs yield substantially different PIR, and then how fidelity metric scores (for several saliency maps) differ under these perturbations.
- Analyze which parts of the ranked list drive low PSim: are disagreements concentrated in low-importance pixels or do they affect the very top of the rankings, which matter most for many metrics?

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Claim that the paper mischaracterizes PIR as the metric itself.**  
  The harsh review notes that PIR is “not the metric” but a ground-truth proxy. The paper does acknowledge this, stating that PIR “serves as a proxy for ground truth, enabling the estimation of the fidelity score for saliency maps” (lines 29–29). The criticism correctly points out that AOPC/AD/etc. are functions of explanation rankings under a chosen perturbation, but since the paper’s analysis is explicitly about the PIR construction those metrics rely on, the charge of conflation is overstated.
- **Assertion that P1/P2 are claimed to be universally assumed by prior work.**  
  The reviewer suggests the paper attributes P1/P2 directly to prior metrics. In fact, Sec. 2 says “The fidelity metrics are based on the PIR which assume…” (line 47) and then posits P1/P2 as “based on two aspects” rather than citing a specific prior formalization. The criticism that the connection to prior literature could be better aligned is valid and kept (Minor), but the stronger claim that the authors state existing metrics literally assume Eq. (2) and Eq. (5) is not textually accurate, so that stronger form is removed.
- **Critique that the algorithm “explicitly does not involve any saliency method” is a fundamental flaw.**  
  Algorithm 1 indeed operates directly on prediction changes, but the paper positions DROP/PSim as model-level conformity checks “before using fidelity metrics” (abstract, line 37–43, Sec. 3.1, 120–123), not as replacements for saliency-based metrics. The lack of integration with saliency methods is already captured in the major weakness on missing metric experiments; framing this as an inconsistency in the method itself is unnecessary duplication.

## Novel Insights

The most substantive insight is that the paper’s empirical methodology is strong and broadly executed, but its conceptual framing overshoots: DROP and PSim provide useful, model-centric diagnostics of how sensitive prediction probabilities and PIR are to perturbation choices, yet the current text treats violations of their idealized conditions as if they directly invalidate existing fidelity metrics. A more compelling contribution would explicitly limit the scope to characterizing model–perturbation behavior and then, in future work, overlay standard fidelity metrics on top of this diagnostic layer.

## Suggestions

- Reframe the paper’s core claim: present DROP/PSim primarily as tools to characterize model sensitivity and PIR stability across perturbations, and soften the language about existing metrics being “inconsistent and unreliable”; emphasize instead that metric outcomes are perturbation-dependent and that conformity scores can help interpret or design those perturbations.
- Add a focused experiment where you (i) select a small set of standard saliency methods, (ii) compute a common perturbation-based metric (e.g., AOPC or AD/IC/W) under different perturbation types, and (iii) show how metric rankings change and how this correlates with DROP/PSim. This would directly test the asserted connection between low conformity and metric instability.
- Provide more nuanced discussion of P1/P2: explicitly acknowledge that they are stronger than what classical metrics strictly require, argue why high conformity is desirable (if you believe it is), and clarify that moderate deviations do not automatically render metrics unusable.
- When recommending Gaussian blur, either (a) present it clearly as a hypothesis based on internal stability (“blur appears more stable under our diagnostics; validating whether this improves metric faithfulness is future work”) or (b) add external validation (e.g., correlation with mask-based ground truth) to support the recommendation.
- Clean up minor notation inconsistencies in Algorithm 1 and cross-references between equations and narrative to improve clarity for readers who may try to re-implement DROP/PSim.

### Overall Evaluation (Originality, Importance, Support, Soundness, Clarity, Value)

- **Originality:** Moderately original. DROP/PSim are simple but new operationalizations of commonly-discussed issues (perturbation sensitivity, PIR variance).
- **Importance of question:** Medium. Understanding how perturbation choices affect explanation evaluation is important in XAI, though this paper does not yet close the loop to metric behavior.
- **Support for claims:** Mixed. Claims about model–perturbation variability are well supported; claims that existing fidelity metrics are “inconsistent and unreliable” are not.
- **Experimental soundness:** Strong for what is measured (probabilities, PIR, conformity scores), but missing the crucial layer of actual metrics/explanations.
- **Clarity:** Generally clear in methods and experiments; less precise in positioning relative to existing metrics and in justifying P1/P2 as necessary assumptions.
- **Value to community:** With a reframed, more modest claim, this would be a solid diagnostic study and potentially a useful reference; as written, the overclaiming weakens its suitability for acceptance.

## Score and Decision

### Calibration Anchors

- **High-scoring anchors (>7):**
  - `/home/wg25r/review_agent/human_reviews/PBjCTeDL6o.md` (avg 8.0, Accept Oral): Strong, technically deep method plus extensive experiments directly evaluating faithfulness metrics (e.g., MuFidelity) and robustness. Compared to this, the current paper is less ambitious methodologically and especially weaker in connecting diagnostics to evaluation outcomes.
  - `/home/wg25r/review_agent/human_reviews/GjfIZan5jN.md` (avg 7.33, Accept Spotlight): Proposes a new interpretability metric (IIS) with thorough experiments and clear applications. This is similar in “metric” flavor but better grounded in how the metric relates to useful properties and applications.

- **Medium-scoring anchors (4–6):**
  - `/home/wg25r/review_agent/human_reviews/L7jtdGhWzT.md` (avg 4.67, Reject): Critiques existing perturbation-based faithfulness metrics and proposes a method (FEI) with some quantitative improvements but limited baselines and clarity. This has conceptual and experimental gaps similar in severity to the overclaim vs. evidence gap here.

- **Low-scoring anchors (<3):**
  - `/home/wg25r/review_agent/human_reviews/wJVZkUOUjh.md` (avg 2.0, Reject): Addresses explanation disagreement (EXAGREE) with substantial conceptual confusion and misaligned problem framing. Compared to this, the current paper is substantially clearer and more coherent; its main issue is overclaiming, not fundamental misunderstanding.

Relative to these anchors, the present paper is clearly stronger than the low band (EXAGREE) in clarity and methodological execution, but weaker than the high band (UNI, IIS) in aligning claims with evidence and in closing the loop from diagnostics to practically used metrics. Its overall profile is closest to the medium band (faithfulness-guided FEI), perhaps slightly stronger empirically but with a more problematic overstatement of its main conclusion.

I therefore place it in the mid-range, slightly below a clear borderline accept.

**MY FINAL SCORE: <pineapple>5.0</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>**