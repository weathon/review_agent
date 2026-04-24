## Summary

This paper proposes CEMA, a black-box adversarial attack for heterogeneous multi-task text systems. Its key idea is to cluster a small set of auxiliary inputs together with their black-box outputs into two groups, train lightweight binary substitute classifiers on these pseudo-labels, and then transfer standard text-classification adversarial attacks to the victim models. The authors evaluate the method on open-source classification and translation models as well as commercial translation APIs, reporting high attack success rates and low BLEU scores with a small auxiliary budget.

## Strengths

- **Realistic and understudied threat model.** The paper explicitly targets black-box, multi-model, heterogeneous-task settings (e.g., joint classification and translation) and evaluates against both open-source Hugging Face models and proprietary APIs (Baidu Translate, Ali Translate). This practical focus is a genuine departure from prior white-box, shared-parameter multi-task attack assumptions. *Evidence:* Sections 1 and 3, Tables 1–2.
- **Creative plug-and-play reduction.** The cluster-oriented substitute training (Algorithm 1, Sections 4.1–4.3) is a conceptually novel way to unify incompatible task outputs into a single surrogate, enabling off-the-shelf classification attacks to be reused in a multi-task context.
- **Robustness to design choices.** Ablations show that attack performance is not brittle to the choice of clustering algorithm or embedding model. *Evidence:* Table 4 and Table 5 compare Spectral, K-means, BIRCH, mT5, XLM-R, and one-hot vectorization.
- **Viability under distribution mismatch.** CEMA remains effective when the 100 auxiliary texts are drawn from a different dataset than the victim data. *Evidence:* Table 6 (zero-shot attack).

## Weaknesses

### Fatal
None.

### Major
- **Misleading query accounting and structurally uneven baseline comparison.** CEMA reports “~0.05 queries per text” (Table 1) by amortizing a fixed upfront cost of 100 victim queries (to label auxiliary data) across thousands of test examples. The baselines, however, are restricted to 30 victim queries *per target text* during attack generation and are not provided with the same auxiliary-data budget. This conflates a data-acquisition budget with a per-example attack budget: for an attacker targeting only a small number of texts, CEMA’s fixed 100-query overhead makes it *more* expensive, not less. A fair comparison must hold constant either the auxiliary-data assumption or the total query budget for a fixed-size target set. Because query efficiency is a central claim, this accounting significantly distorts the empirical comparison.  
- **No validated link between cluster-label flips and actual victim-task failure.** The paper asserts that “if an adversarial attack on the substitute model … successfully changes the cluster label … the label $y_i^A$ shifts accordingly, indicating a successful attack on task $A$” (Introduction / Section 4.3), yet provides no empirical or theoretical validation of this leap. There is no analysis showing how often a substitute cluster flip coincides with (a) a classification label change on the victim or (b) a BLEU drop on the victim. For 5-way or 6-way classification, a binary surrogate is not synonymous with multi-class misclassification, and for translation the mechanism connecting a binary cluster-ID flip to seq2seq degradation is entirely unexplained. Because the method optimizes a surrogate objective whose alignment with the true attack goal is untested, the scientific rationale for the central mechanism remains unsubstantiated.

### Minor
- **Theoretical lower bound adds limited insight.** Equations (2)–(5) in Section 4.4 derive a standard union bound on independent Bernoulli trials to argue that generating more candidates increases success probability. The authors acknowledge in a Remark that the independence assumption is violated, which blunts the practical value of the bound. The derivation is elementary and does not illuminate why the specific substitute models or attack methods work.  
- **Translation results lack qualitative support.** The paper reports BLEU scores as low as 0.14–0.15 on commercial translation APIs without showing any example inputs, adversarial perturbations, or model outputs. Given such extreme scores, qualitative case studies are needed to assess whether the adversarial text is fluent and semantically preserved (as required by the USE > 0.85 constraint) or whether the outputs are simply collapsed/broken, which would affect the interpretability of the cross-task claim.

### Trivial
None.

## Nice-to-Have
- A direct correlation analysis (e.g., substitute cluster-flip rate vs. victim label-change rate and BLEU delta) on held-out auxiliary data to validate the surrogate objective.
- A fair baseline comparison that grants all methods the same total query budget for a fixed target-set size, or that reports CEMA’s true per-text cost including the fixed overhead.
- A single-task substitute ablation (training on individual task outputs rather than clustered multi-task outputs) to isolate the value of the clustering step.
- Qualitative adversarial examples for translation to complement the quantitative BLEU scores.

## Removed Points
These points are flagged to be removed; treat them with caution.
- **“Victims are not true multi-task systems.”** The paper explicitly frames its threat model as *multi-model* multi-task learning (Sections 2.2 and 3) and cites Aoki et al. (2022). Attacking an ensemble of independent black-box services is a valid and stated scope; criticizing the absence of shared-parameter architectures is scope creep. The novelty claim would be stronger with such a baseline, but the evaluated setting is not a misrepresentation.
- **“Translation results are implausible.”** Without evidence of data fabrication, questioning the plausibility of reported numbers is speculative. The proper criticism is the *lack of supporting evidence* (qualitative examples, controls), which is listed above as a minor weakness.
- **“Section 5.6 undermines representation learning.”** The insensitivity of performance to vectorization method is presented by the authors as robustness. Interpreting this as a negative is unjustified without evidence that representation quality should matter more.
- **Elementary theory as contribution.** The paper does not grossly overstate the union-bound derivation; it is framed as a justification for ensembling. While thin, it is not a fatal overclaim.
- **Missing appendix proofs / missing related works.** Per the hard rules, these are not valid criticisms.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

1. **Fix query reporting.** Report total query budgets transparently: show the fixed 100-query auxiliary cost plus any evaluation queries, and compare against baselines under equal total budgets for attacking a fixed-size target set (e.g., 100 or 1,000 texts).  
2. **Validate the surrogate objective.** Add an analysis that measures, on auxiliary or validation data, the correlation between flipping the substitute’s cluster label and the victim’s actual task-specific failure (label change for classification, BLEU drop for translation). This would substantiate the core claim that cluster discrimination translates to multi-task attack success.  
3. **Provide qualitative translation examples.** Show original inputs, adversarial perturbations, and both original and adversarial translations to help readers interpret near-zero BLEU scores.

## Score and Decision

**Calibration anchors consulted:**
- `/home/wg25r/review_agent/human_reviews/htX7AoHyln.md` (GSBA$^K$, avg 6.5, Accept): Stronger theoretical grounding and clearer experimental setup than the current paper; our paper falls below this due to unvalidated mechanism and misleading query accounting.
- `/home/wg25r/review_agent/human_reviews/mJzOHRSpSa.md` (RLS, avg 5.33, Reject): Shares query-accounting and baseline-comparison issues; our paper has a more creative core idea but similar structural flaws in evaluation.
- `/home/wg25r/review_agent/human_reviews/LO4MEPoqrG.md` (ReG-QA, avg 5.0, Accept Poster): Has narrower scope and missing comparisons, but its core experimental claims are not undermined by accounting artifacts; our paper’s issues are more fundamental to its central “few-shot” and “cross-task” claims.
- `/home/wg25r/review_agent/human_reviews/x9gCQC3rVA.md` (AdvWeb, avg 4.4, Reject): Strong empirical numbers paired with methodological concerns about threat-model practicality and baseline fairness; comparable quality band, though our paper’s idea is more novel.
- `/home/wg25r/review_agent/human_reviews/BXMoS69LLR.md` (Blind Baselines MI, avg 4.5, Reject): Flawed comparison assumptions; our paper is slightly above this because its empirical scope is broader and the clustering heuristic is genuinely creative.

The paper proposes an intriguing heuristic and delivers broad empirical coverage, but its two major issues—misleading query accounting that structurally favors the proposed method, and an unvalidated central assumption linking cluster flips to victim failure—are severe enough that the experimental evidence does not fully support the headline claims. Relative to the anchor cluster, the paper sits between the rejected 4.5-band papers and the borderline 5.0 posters, leaning toward the lower end because the flaws strike at the core empirical claims. A score of **5.0** reflects a borderline submission with real creativity but insufficient methodological rigor for acceptance without major revisions.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>