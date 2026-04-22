## Summary

The paper analyzes two-layer neural networks trained with a single gradient step on the first layer under a structured Gaussian mixture data model with low-rank-plus-identity covariances. It proves that, in a proportional asymptotic regime and under explicit scaling conditions, the network’s training and generalization errors are equivalent (in probability) to those of (i) a conditional Gaussian feature model and (ii) a finite-degree Hermite/polynomial activation model, and it supports these claims with simulations including GAN-based Fashion-MNIST experiments.

## Strengths

- **Nontrivial extension of feature-learning theory to structured Gaussian mixtures.**  
  Sections 2 and 4 define a Gaussian mixture with per-component finite-rank-plus-identity covariance (Assumption A.4, Eq. (9)) and a proportional scaling regime (A.1–A.2). Lemmas 1–2 and Theorems 3–4 provide an asymptotic characterization of training and generalization errors in this richer setting, moving beyond isotropic or simple spiked-covariance models common in prior one-step analyses.

- **Clear spike+bulk and structure+bulk decompositions enabling conditional universality.**  
  Lemma 1 decomposes the gradient matrix as \(G = u v^\top + \Delta\) with explicit norm scalings in terms of \(\alpha,\beta\) (Eq. (11)), and Lemma 2 decomposes \(\tilde F x\) into a bulk term \(F^\perp z^\perp\) and a structured term \(a_{|\kappa_c}\) (Eq. (12)). These steps underpin Theorem 3’s conditional Gaussian-equivalent feature map (Eqs. (13)-(15)), giving a crisp conceptual story for what one gradient step learns under mixture structure.

- **Hermite-model equivalence linked to joint data–learning-rate scaling.**  
  Theorem 4 defines a truncated Hermite activation \(\hat\sigma_l\) with an added Gaussian residual (Eq. (16)) and shows that, when \((l-2)/(l-1)<\beta<(l-1)/l\), replacing \(\sigma\) by \(\hat\sigma_l\) yields the same training and generalization errors in probability (Eq. (16), lines 185–192). This connects the required polynomial degree directly to the strength parameter \(\beta=\log(\eta\|\Sigma\|)/\log n\), generalizing isotropic results to the mixture setting.

- **Empirical section that closely mirrors the theory and explores data structure.**  
  Figures 1–2 systematically vary \(k/m\), \(\alpha\), \(\beta\), mixture imbalance, alignment, and covariance rank, consistently showing that NN and Hermite model generalization errors are very close (e.g., Fig. 1 caption; Fig. 2 caption). The Fashion-MNIST experiments (Fig. 3) demonstrate near-equivalence between the NN and a degree-5 Hermite model on GAN-generated mixture-like data. These experiments are well designed to probe the theoretical quantities.

## Weaknesses

### Fatal

None.

### Major

- **Ambiguity and over-strong narrative around the notion of “equivalence.”**  
  Theorems 3 and 4 correctly state that training and generalization errors with the original and equivalent models “converge in probability to the same value” (Theorem 3, lines 176–179; Theorem 4, lines 191–192). However, the paper frequently describes these models as “equivalent” and claims they can be “effectively substituted … without impacting training and generalization errors” (Theorem 3 discussion, lines 183–184; Theorem 4 discussion, lines 213–217), and speaks of “equivalence class of activation functions” (line 217) without clearly emphasizing that the equivalence is only asymptotic, in probability, under the stated proportional limits and assumptions (including A.9 for generalization). The mode of convergence (in probability vs almost sure, quenched vs annealed) is not clarified beyond the brief theorem statements, yet the narrative sometimes sounds universal and parameter-free. This looseness between the precise asymptotic statements and the broader interpretive language overstates the generality of the results.

- **Hermite truncation only rigorously covers \(\beta\) away from 1, but interpretation extrapolates to \(\beta\approx 1\).**  
  Theorem 4 relies on the regime \(\frac{l-2}{l-1}<\beta<\frac{l-1}{l}\), and the text notes explicitly that “Theorem 4 does not address the maximal value for the strength parameter (\(\beta = 1\)), in contrast to Theorem 3. While \(\beta \rightarrow 1\) implies \(l \rightarrow \infty\)” (line 213). Nevertheless, the experiments and narrative repeatedly highlight behavior for \(\beta\approx 1\) and for settings like \(\|\Sigma\|=n,\eta=1\) (Fig. 3, line 247) and state that “a finite \(l\) value is enough to achieve the equivalence of generalization errors in our numerical simulations for \(\beta\approx 1\)” (line 213). The strong message that “a finite-degree polynomial model serves as an equivalent performance model” (contributions, line 35; conclusion, lines 255–257) thus extends beyond the formal coverage of Theorem 4, especially in the practically interesting high-\(\beta\) regime. There is an evidential gap: the theorem does not guarantee equivalence there, and the finite-sample experiments are suggestive but not sufficient to claim asymptotic equivalence.

- **External validity of the Fashion-MNIST experiments is overstated.**  
  The “real data” experiments use data generated from a conditional GAN trained on Fashion-MNIST and then preprocessed (demeaned, rescaled, and noise-added) “such that assumptions (A.2)–(A.4) are satisfied” (Fig. 3 caption, lines 247–248). This means the evaluation is effectively on synthetic Gaussian-mixture-like data constructed to match the theory, not on the original Fashion-MNIST distribution. Yet the abstract and conclusions state that these experiments “indicat[e] that our findings can translate to realistic data” (abstract, line 16; Section 6, line 253; conclusion, lines 255–257). Without any results on raw Fashion-MNIST or diagnostics quantifying how well the GAN+preprocessing matches the assumed mixture model, this narrative overstates realism; the experiments principally validate the theory on data that is by design within the model class.

### Minor

- **Single-index teacher and its justification are somewhat under-discussed.**  
  The label model assumes a single-index teacher \(y=\sigma_*(\xi^\top x, c)\) (Eq. (2), line 50–56), justified by the statement that the NN with one gradient step “can only learn one direction about the labels (Lemma 1)” (line 56). Lemma 1, however, provides a rank-one spike in parameter space for the gradient (Eq. (11)), not an explicit guarantee that all label-relevant structure in data is effectively one-dimensional in input space. Single-index teachers are a standard simplification in this literature, but for a paper emphasizing realistic mixture structure and low intrinsic dimension, a short discussion of how multi-index targets might change the picture (even at a heuristic level) would better contextualize this limitation.

- **Strong structural restrictions on the mixture model are acknowledged but undercut some “realistic data” claims.**  
  Assumption (A.3) enforces \(\mu_c=0\) and equal trace across components, and (A.4) restricts each \(\Sigma_c\) to finite-rank-plus-identity (lines 116–122). The discussion notes that zero-mean can be relaxed in Appendix F, but it does not clarify whether equal-trace or the precise low-rank-plus-identity form are essential for Theorems 3–4 (beyond “simplify our analysis,” line 136). Many realistic mixtures have heterogeneous scales across components, which could affect the Hermite expansion and spike–bulk decomposition. This does not undermine the core mathematical results in the stated setting, but it means claims like “our data model captures both the mixture nature and intrinsic low-dimensionality of real-world datasets” (line 30) and “theoretical insights on realistic data” (line 253) should be slightly more qualified.

- **Opacity of Assumption (A.9) and its role in generalization equivalence.**  
  Both Theorems 3 and 4 require an additional assumption (A.9) from Appendix B to extend equivalence from training to generalization errors (lines 179–180 and 191–192), but this assumption is not summarized at all in the main text. As a result, the reader cannot judge how mild or restrictive it is when reading the core theorems. This is a clarity issue rather than a correctness problem.

- **Interpretation of Figure 2 where the Hermite model outperforms the neural network.**  
  The caption for Figure 2 states that “The Hermite Model consistently achieves lower generalization errors than the Neural Network” (lines 225–227), even though Theorem 4 predicts equality in the asymptotic limit. The main text does not comment on why the Hermite model might be slightly better in finite-sample simulations (regularization, variance reduction, implementation details) or whether the implemented Hermite model exactly matches the theoretical \(\hat\sigma_l\). Clarifying this discrepancy would strengthen the empirical narrative and make clear that the theorem claims equality, not systematic improvement.

### Trivial

- Minor phrasing overstatements in the introduction and conclusion, such as “comprehensive understanding” of how structured data and feature learning influence generalization (line 255), which is stronger than warranted given the specific setting (one-step training, proportional limit, Gaussian mixtures with equal-trace finite-rank-plus-identity covariances and single-index teacher).

## Nice-to-Haves

- A more explicit discussion of modes of convergence and randomness in Theorems 3–4 (e.g., whether convergence is conditional on the training sample, over fresh test points, over network initialization, or jointly) and how this connects to the equivalence narrative.

- Systematic experiments varying \(\beta\) and the Hermite degree \(l\) to empirically map where finite-degree truncation works or fails, particularly as \(\beta\) approaches 1, and plots of NN–Hermite error differences versus dimension \(n\) to visually support convergence.

- Additional simulations that relax some structural assumptions (e.g., unequal traces across mixture components, more general covariance spectra) to test robustness of the qualitative equivalence beyond the precise model analyzed.

- An exploratory experiment on raw Fashion-MNIST (or minimally processed data) using the one-step network and an approximate Hermite model, even without theory, to give a transparent picture of performance when assumptions are violated.

## Removed Points

These points are flagged to be removed, treat them with caution.

- Any claim that the paper fails to define what kind of mathematical convergence underlies “equivalence” would be inaccurate: both Theorem 3 and 4 explicitly state convergence in probability of the training and generalization errors (lines 176–179, 191–192). The issue is not absence of a definition but limited emphasis and somewhat stronger surrounding prose.

- Complaints that the Hermite activation \(\hat\sigma_l\) is misrepresented as “polynomial” because of the added Gaussian noise term could be overstated. The authors do define \(\hat\sigma_l\) with this noise (Eq. (16)) and later clarify that the Hermite model includes a Gaussian term “that accounts for residuals” (lines 215–217); so, while “polynomial activation” is slightly informal, it is not a serious methodological flaw.

- Concerns that the paper hides or omits Appendix content (e.g., the statement of A.9, proof details, or additional experiments) cannot be blamed on the authors; the submission format here strips appendices by design.

## Novel Insights

None beyond the paper’s own contributions; the reviews primarily refined the scope and interpretation of the equivalence claims rather than uncovering qualitatively new phenomena.

## Suggestions

- **Clarify and slightly temper the equivalence narrative.** Emphasize early and repeatedly that “equivalence” means equality of training and generalization errors in probability in the proportional limit, under assumptions (A.1)–(A.8) and (A.9), and that finite-sample results are approximate. Consider replacing phrases like “defines an equivalence class of activation functions” with wording that explicitly refers to asymptotic risk equivalence under the same scaling regime.

- **Make the \(\beta\)–\(l\) limitation more central.** In the abstract, contributions, and conclusion, note that the Hermite/polynomial equivalence is proved for \(\beta\) satisfying \((l-2)/(l-1)<\beta<(l-1)/l\) and does not rigorously cover \(\beta=1\); treat the \(\beta\approx 1\) experiments as empirical evidence rather than as direct consequences of Theorem 4.

- **Summarize Assumption (A.9) in the main text.** Add a short description near Theorems 3–4 of what (A.9) requires (e.g., moment or spectral regularity conditions) and why it is expected to hold in the intended settings, so that readers can assess the scope of the generalization-equivalence results.

- **Discuss modeling choices and realism more carefully.** In Section 2 and the conclusion, explicitly list the main structural assumptions (equal-trace, low-rank-plus-identity covariances, single-index teacher, one-step training) as part of the scope; for the Fashion-MNIST experiments, state clearly that the data is GAN-generated and then processed to satisfy these assumptions, and rephrase claims about “real data” accordingly.

- **Comment on Figure 2’s Hermite–NN gap.** Add a brief discussion on why the Hermite model sometimes yields slightly lower generalization error in finite-sample simulations and whether this is expected from the theoretical construction or due to implementation/regularization effects.

## Score and Decision

### Calibration Anchors

- **High-scoring anchors (avg > 7):**
  - `/home/wg25r/review_agent/human_reviews/dEypApI1MZ.md` (avg 7.20, Accept Spotlight): strong single-step/feature-learning theory with very tight alignment between theory and experiments and carefully scoped claims. Compared to this, the current paper is somewhat narrower (one-step only), has solid but slightly less polished handling of assumptions and equivalence notions, and somewhat overstates realism; thus it feels slightly weaker overall.
  - `/home/wg25r/review_agent/human_reviews/is4nCVkSFA.md` (avg 7.50, Accept Oral): single-index model analysis with clean theorems and careful empirical validation. The current paper is somewhat comparable in technical depth but has more restrictive data assumptions and a looser empirical story on “real data,” so likely merits a lower score.
  - `/home/wg25r/review_agent/human_reviews/ze7DOLi394.md` (avg 7.50, Accept Oral): strong joint model–data–feature interaction study. Again, the present paper is technically solid but less broadly impactful and with some narrative overreach.

- **Medium-scoring anchors (avg 4–6):**
  - `/home/wg25r/review_agent/human_reviews/tJDlRzQh7x.md` (avg 4.33, Reject): theoretical universality work with interesting ideas but significant gaps or overclaims; weaker empirical support or clarity. The current paper’s technical core seems sounder and better tied to experiments than this anchor, suggesting a higher score.
  - `/home/wg25r/review_agent/human_reviews/QY52D9BeJo.md` (avg 6.00, Reject): Hermite/information-exponent style theory with decent contributions but some issues that kept it out; the present paper feels in a similar quality band but slightly more cohesive and relevant to current feature-learning literature, closer to a borderline accept.
  - `/home/wg25r/review_agent/human_reviews/wOSYMHfENq.md` (avg 6.00, Accept Poster): solid universal-approximation theory, clear but somewhat niche. The current paper is comparably strong: focused theoretical development, clean main results, and good simulations, but some overclaim.

- **Low-scoring anchors (avg < 3):**
  - `/home/wg25r/review_agent/human_reviews/2NwHLAffZZ.md` (avg 2.33, Reject): theoretical analysis of linearization and generalization with substantial methodological or correctness problems. The current paper is clearly stronger: its assumptions are explicit, results align with experiments, and no fatal flaws have been identified.
  - `/home/wg25r/review_agent/human_reviews/KNQJtoPZmz.md` (avg 3.00, Reject): generalization explanation via simplicity bias with overreach and insufficient rigor. Again, the present submission is significantly more rigorous.

Positioning relative to these anchors, this paper is much better than the low-score group, comparable to mid-high “poster-level” theory papers but a bit below the strongest “oral/spotlight” feature-learning works because of its somewhat restrictive setting and over-strong narrative around equivalence and realism. That suggests a calibrated score around 6.5.

**Final score:** 6.5  
**Decision:** Reject (borderline; solid theory but not quite strong enough, and claims need tightening).

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>