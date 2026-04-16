## Summary

The paper proposes NAMMD, a “norm-adaptive” modification of the maximum mean discrepancy (MMD) for kernel-based distribution closeness testing. NAMMD rescales the squared MMD by a function of the RKHS norms of the mean embeddings, with the goal of making distances more informative when comparing multiple distribution pairs and improving the power of associated hypothesis tests, both in two-sample and generalized closeness-testing settings. The authors provide asymptotic distribution results, sample-complexity upper bounds, power comparison theorems versus MMD, and a range of empirical studies on synthetic data, benchmark datasets, and ImageNet-based case studies.

## Strengths

- **Clear, simple statistic with preserved basic properties.**  
  NAMMD is defined as
  \[
  \mathrm{NAMMD}(\mathbb{P},\mathbb{Q},\kappa)=\frac{\|\mu_\mathbb{P}-\mu_\mathbb{Q}\|_{\mathcal{H}_\kappa}^2}{4K-\|\mu_\mathbb{P}\|_{\mathcal{H}_\kappa}^2-\|\mu_\mathbb{Q}\|_{\mathcal{H}_\kappa}^2},
  \]
  for kernels bounded by \(K\). It is easy to implement from the standard unbiased MMD U‑statistic and yields a bounded \([0,1]\) distance that is still zero iff \(\mathbb{P}=\mathbb{Q}\) for characteristic kernels (Lemma 9). This makes it directly usable as a drop‑in replacement in existing kernel testing pipelines (e.g., MMDFuse, permutation-based tests).

- **Technically solid asymptotic and concentration analysis.**  
  The paper develops asymptotic distributions for the empirical NAMMD statistic in both the \(\epsilon>0\) and \(\epsilon=0\) regimes (Theorem 2), a variance estimator with controlled bias (Lemma 4), finite-sample concentration (Lemmas 6–7), and sample-complexity upper bounds (Theorem 8). These mirror the well-known MMD theory and show the adaptation is mathematically coherent.

- **Consistent empirical competitiveness with MMD.**  
  Across multiple kernels (Gaussian, Laplace, Mahalanobis, Deep) and five datasets (blob, higgs, hdgm, mnist, cifar10), NAMMD-based tests match or slightly exceed the power of MMD-based tests when Type-I error is controlled via permutation (Table 1). In the multi-kernel two-sample setting, replacing MMD by NAMMD inside MMDFuse (NAMMDFuse) yields test-power curves that are at least competitive with, and usually above, state-of-the-art baselines (Fig. 2).

- **Extension of distribution closeness testing to complex data.**  
  Classical DCT work has almost exclusively used total variation on finite discrete supports. By formulating a kernel-based closeness test that can be applied to high-dimensional continuous data (including images), this paper helps bridge DCT ideas to mainstream representation-learning settings. The ImageNet/domain-shift and adversarial experiments (Figs. 3–5) illustrate how a closeness threshold calibrated on one reference pair can be reused to compare other unlabeled distribution pairs.

## Weaknesses

### Fatal

_None appear: the paper is a well-formed theoretical/statistical contribution with correct basic constructions. The concerns below are about overclaiming and limited practical impact rather than invalidating errors._

### Major

- **Overstated core motivation: “MMD is less informative” as a closeness measure is not fully justified.**  
  The paper’s main conceptual story is that MMD is “less informative” across multiple distribution pairs because different pairs can share the same MMD yet have different RKHS norms (and thus different “closeness levels”), and that NAMMD fixes this. The work takes as implicit that, for a fixed MMD, pairs with larger \(\|\mu_\mathbb{P}\|^2+\|\mu_\mathbb{Q}\|^2\) should be regarded as less close. This is a modeling choice rather than a clearly argued desideratum: nothing in standard discrepancy theory says that “more concentrated marginals at the same mean-embedding separation must be considered further apart”. The main evidence is Fig. 1, showing that as norms increase (holding MMD fixed), the variance of the empirical MMD shrinks so p‑values decrease; NAMMD then correlates better with these p‑values. But that is about the *test statistic’s sampling distribution* at a fixed effect size, not about an intrinsic notion of distance. MMD remains a metric on distributions; the paper shows NAMMD behaves more like a variance-normalized effect size but does not really establish a principled target notion of “closeness across distribution pairs” that MMD fails to reflect. As a result, the central problem formulation (“MMD is less informative; we must replace it”) feels overstated relative to what is actually shown (“we introduce a normalized MMD variant that often yields slightly larger test power”).

- **Power comparisons to MMD are conditional and empirically modest, yet presented as broadly superior.**  
  Theorems 10 and 12 are used to claim that NAMMD “achieves higher test power compared to the MMD test.” In detail:
  - Theorem 10 (two-sample testing) states that under \(H_1\), with high probability, whenever a thresholded MMD test rejects, the corresponding thresholded NAMMD test also rejects, and with probability \(\varsigma\ge 1/65\) there exist situations where NAMMD rejects while MMD does not, provided \(m\ge C'\) (with \(C'\) depending on \(\mathbb{P},\mathbb{Q}\)). However, the text immediately notes that for fixed \(\mathbb{P},\mathbb{Q}\), NAMMD is asymptotically just a constant multiple of MMD, so the standard test-power estimator is identical. Thus any advantage must stem from finite-sample behavior and thresholding rules, not an inherent asymptotic dominance. The theorem’s conditions (\(C'\), \(\varsigma\)) are opaque and do not translate into a clear characterization of when practitioners should expect notable gains.
  - Theorem 12 (distribution closeness testing) is even more conditional. It requires (i) that both MMD- and NAMMD-based tests are in their respective alternative regimes and (ii) the additional norm inequality \(\|\mu_{\mathbb{P}_1}\|+\|\mu_{\mathbb{Q}_1}\|<\|\mu_{\mathbb{P}_2}\|+\|\mu_{\mathbb{Q}_2}\|\). Under these, whenever the MMD-based DCT rejects, the NAMMD-based DCT also rejects, and sometimes NAMMD rejects when MMD does not. The condition is only heuristically justified (“often met in practice”), with no theoretical or empirical quantification of its prevalence.
  - Empirically, Table 1 shows average power improvements in the 0.002–0.01 range, generally within one standard deviation. The paper does not report paired significance tests or confidence intervals to show these differences are statistically meaningful. Figures 3–5 show curves where NAMMD lies above MMD, but differences are often small and not analyzed. Hence the strong repeated claim that NAMMD “achieves higher test power” is only weakly supported: what is convincingly true is that NAMMD is at least as good and sometimes slightly better, not that it robustly dominates MMD in a practically significant way.

- **Comparison with TV-based DCT (Canonne’s test) is not carefully controlled.**  
  In Section 5.2 (Table 2), the paper compares NAMMD-based DCT with the total-variation–based tester of Canonne et al. on discrete distributions over 50 points. While NAMMD indeed often shows higher power, the comparison mixes substantially different regimes and design goals:
  - NAMMD uses a Mahalanobis kernel and full kernel matrices; Canonne’s method uses counts and is originally designed for sub-linear sample complexity and worst-case guarantees. Computational cost and sample-complexity regimes are not controlled for or discussed.
  - The kernel choice is somewhat arbitrary for these purely discrete domains; with an appropriate “identity” or discrete kernel, an MMD-based DCT might behave differently. As implemented, the experiment mainly shows that a flexible kernel smoother can beat a naive TV estimator at small sample sizes—not specifically that the *norm-adaptive* part of NAMMD is superior to traditional DCT measures. There is no comparison to an MMD-based DCT on the same discrete supports.
  This weakens the paper’s claim to advance DCT “beyond total variation”: the experiment does not separate the contributions of using kernels vs. using NAMMD in particular.

- **Domain adaptation / ImageNet case studies are illustrative rather than rigorously evaluated.**  
  The ImageNet variants and confidence/adversarial margin case studies (Figs. 3–5) are appealing but largely qualitative:
  - For ImageNet vs. its variants, the paper states NAMMD “effectively reflects the closeness relationships indicated by accuracy margin” but does not quantify this (e.g., via rank correlation between NAMMD distances and empirical accuracy gaps, or predictive performance on held-out variants). The plotted test-power curves show NAMMD above MMD, but do not establish that NAMMD is a strong predictor of whether model adaptation is needed.
  - The confidence margin experiments explicitly group classes by margin value (computed using labels) and then show NAMMD’s rejection rate tracks these groups. This is more a consistency check than an out-of-sample prediction. Similarly for adversarial perturbations, the mapping perturbation size → test-rejection pattern is plausible but baseline-free.
  - No other baselines beyond MMD (e.g., classifier-based distribution tests or domain similarity metrics tailored for performance prediction) are considered. Thus the ambitious motivation (“decide if we really need to adapt a model”) remains suggestive, not demonstrated.

### Minor

- **Ad hoc nature of the denominator choice and limited discussion of alternatives.**  
  The NAMMD denominator \(4K-\|\mu_\mathbb{P}\|^2-\|\mu_\mathbb{Q}\|^2\) is chosen to ensure a [0,1] range and an increasing dependence on norms. The remark explains the heuristic but there is no argument that this is in any sense optimal (e.g., maximizing signal-to-noise ratio, minimizing asymptotic variance, or aligning with some information-theoretic bound). The paper also does not compare against simpler normalized variants such as \(\|\mu_\mathbb{P}-\mu_\mathbb{Q}\|^2 / (1+\|\mu_\mathbb{P}\|^2+\|\mu_\mathbb{Q}\|^2)\) or a version using a square root in the denominator; without that, it is uncertain that this specific form is essential.

- **Kernel-class restriction and interpretation not fully clarified in the main text.**  
  Many arguments (e.g., the “variance” identity \(\mathrm{Var}(\mathbb{P},\kappa)=1-\|\mu_\mathbb{P}\|^2\)) rely on kernels bounded in \([0,K]\) with \(\kappa(x,x)=K\) and of the special form \(\Psi(x-x')\). The main exposition sometimes reads as if the method and its interpretive story (norms ↔ concentration) hold for any characteristic kernel. The experiments say “including Deep kernels” under this form, but how deep kernels are constrained to satisfy these assumptions is not described in detail. This is primarily a clarity/scope issue but affects how broadly one should interpret the main intuitions.

- **Finite-sample stability of the ratio statistic is not discussed.**  
  When \(\|\mu_\mathbb{P}\|^2+\|\mu_\mathbb{Q}\|^2\) is close to \(4K\) (highly concentrated distributions), the denominator becomes small; conversely, when norms are tiny, NAMMD reduces almost to scaled MMD. The paper gives asymptotic variance and a CLT, but does not comment on potential finite-sample instability or numerical issues in high-norm regimes where the denominator is small. In practice this may or may not be a concern, but a short discussion would help.

- **Experimental design for two-sample SOTA comparison could be more balanced.**  
  In Fig. 2, NAMMDFuse, MMDFuse, MMDAgg, ACTT use all test samples (with no training phase), while methods that require kernel learning (MMD-D, AutoTST, MEMabid) effectively have fewer samples available at test time due to a split between training and testing. The paper mentions this but does not investigate sensitivity to such splits. That makes the comparison somewhat conservative for NAMMDFuse’s competitors but does not invalidate the qualitative takeaway that NAMMDFuse is competitive.

### Trivial

- Notational minor issue in Theorem 2: \(\sigma_{\mathbb{P},\mathbb{Q}}^2\) is written with a square root in the numerator; as stated it looks more like a standard deviation, though it is called a variance. This is likely a typographical/notation slip; it does not affect the main ideas but could be cleaned up.

## Nice-to-Haves

- A direct quantitative evaluation of how well NAMMD distances (or closeness test outcomes) predict downstream model performance, e.g., correlation between NAMMD and accuracy gaps across many domain shifts or perturbation levels.
- Experiments explicitly varying the mean-embedding norms while controlling MMD (as in Fig. 1) on real data, to show how the power difference between NAMMD and MMD grows with norm, rather than just reporting aggregates.
- Comparisons to at least one additional DCT-style kernel or classifier-based baseline on the ImageNet/domain-shift experiments to contextualize NAMMD beyond MMD.

## Removed Points

These points are flagged to be removed from the main set of weaknesses; treat them with caution if you revisit them.

- **“MMD is less informative is not a problem” because MMD is a metric and thus sufficient.**  
  The harsh review implied that having multiple pairs share the same MMD can never be problematic for closeness testing. This goes too far: the paper’s use case—comparing levels of separation across multiple pairs under a fixed kernel—is reasonable. It is valid to propose a normalization that reflects additional structure (norms) even if MMD is a perfectly good metric. The issue is overclaiming a deficiency, not the existence of the concern.
- **Concerns that deep kernels cannot fit the assumed kernel class and thus invalidate experiments.**  
  The paper states explicitly that in experiments they “use the selected characteristic kernels of the form \(\kappa(x,x')=\Psi(x-x')\in(0,K]\) with \Psi(0)=K, including Gaussian, Laplace, Mahalanobis and Deep kernels.” Without the appendix, we cannot definitively verify the deep-kernel construction, but we should not assume the authors failed to enforce the boundedness/normalization. Hence detailed criticisms about nonconformant deep kernels are speculative and not kept as core weaknesses.
- **Claims that NAMMD cannot control Type‑I error because of variance estimation / learned kernels.**  
  The paper explicitly proves Type‑I error control under the stated testing procedures (Theorem 5) and uses permutation tests in the \(\epsilon=0\) case, which is standard. Without contrary evidence, generic doubts about Type‑I error are too speculative to be central criticisms here.

## Novel Insights

The most intellectually interesting aspect of the paper is the reframing of cross-pair comparisons in kernel testing: NAMMD takes MMD, which is fundamentally a distance, and normalizes it by a function of the norms of the mean embeddings to approximate a variance-normalized effect size that is more comparable across distribution pairs under a fixed kernel. The normalization is grounded in the RKHS geometry and the boundedness of the kernel, allowing the statistic to live in [0,1] with an interpretable dependence on “concentration” via \(\|\mu_\mathbb{P}\|^2\) and \(\|\mu_\mathbb{Q}\|^2\). While the current theory and experiments only partially realize the promise (and the power gains are modest), the idea of explicitly integrating norm information into kernel discrepancy measures to facilitate cross-pair comparisons and threshold sharing is a useful conceptual contribution that future work could generalize or refine.

If judged narrowly, the paper’s primary novelty is this particular normalization plus the associated asymptotics; there are no broader novel insights beyond the paper’s own contributions.

## Suggestions

- **Reframe claims more modestly and sharpen the problem statement.**  
  Rather than claiming MMD is “less informative” or fundamentally inadequate, frame NAMMD as a normalized version of MMD designed to (i) map into [0,1], (ii) encode concentration information via mean-embedding norms, and (iii) yield more comparable effect sizes across distribution pairs. This would align better with what is actually demonstrated.

- **Strengthen and better qualify the power-comparison story.**  
  - Clarify up front that asymptotically, for a fixed pair \((\mathbb{P},\mathbb{Q})\), NAMMD and MMD have the same Pitman power; the improvements are finite-sample and thresholding dependent.  
  - Either provide more interpretable conditions in Theorems 10 and 12 (e.g., explicit dependence on MMD, norms, and sample size) or temper the claims to “under the specific conditions in these theorems, NAMMD is at least as powerful as MMD and sometimes more so.”  
  - Run paired statistical tests (e.g., paired t‑tests over repetitions) on Table 1 and report when differences are significant.

- **Clarify and experimentally explore the norm-condition in Theorem 12.**  
  - Give intuitive examples where \(\|\mu_{P_1}\|+\|\mu_{Q_1}\|<\|\mu_{P_2}\|+\|\mu_{Q_2}\|\) and where it fails, and discuss expected behavior of NAMMD vs MMD in both cases.  
  - In the experiments, measure how often this inequality holds for the constructed reference and target pairs, and whether observed power differences align with theory.

- **Improve DCT baseline comparisons.**  
  - On the discrete synthetic experiments, add at least one MMD-based DCT baseline (e.g., closeness test using standard MMD with the same kernel) to isolate the value of norm-adaptation versus simply using a kernel.  
  - Discuss the distinct design goals of TV-based DCT (sub-linear sample complexity, worst-case guarantees) versus kernel methods, and present your results as complementary rather than as “NAMMD is better than TV DCT.”

- **Quantify alignment with downstream performance in the case studies.**  
  For ImageNet/domain-shift and adversarial settings, go beyond power curves: report rank correlations between NAMMD (and MMD) distances and accuracy margins or perturbation strengths; if feasible, evaluate predictive performance in a label-free “forecasting” scenario (choose a reference, compute NAMMD to new domains, then compare to later observed accuracy).

- **Discuss finite-sample and computational aspects explicitly.**  
  Briefly address: (i) potential instability when the denominator is small and how you guard against it numerically (e.g., floors, regularization), and (ii) computational cost relative to MMD (both are U‑statistic–based, but NAMMD’s denominator and variance estimator add overhead). Even a short complexity analysis or a timing table would strengthen the practical message.

- **Clarify kernel assumptions in the main text.**  
  Move the key kernel-form limitation (bounded, \(\kappa(x,x)=K\), \(\Psi(x-x')\)) from the appendix into the main method section, and explain how deep kernels in the experiments are constructed to satisfy these constraints.

## Score and Decision

### Calibration Process

I consulted several human-reviewed papers with similar themes:

- **WnqD3EiylC (Representation Jensen-Shannon Divergence)** – Reject, scores mostly 5–6 with one 3. This paper proposed a new RKHS-based divergence and showed small, somewhat unclear empirical gains over MMD; reviewers emphasized “rather small performance gain” and questioned practical impact.  
- **GPcSYm89wK (Practical Kernel Learning for CI Test)** – Reject, scores 3–5. Strong idea, but theoretical guarantees and practical conditions were not fully aligned; reviewers worried about Type-I error control and applicability of assumptions.  
- **GZ6AcZwA8r (MMD Graph Kernel)** – Accept Spotlight, scores 6–8, with clear empirical wins and stronger significance/impact.  
- **3fEKavFsnv (MMD-MP for malware detection)** – Accept Poster, scores 6–8; solid empirical improvements in a clear application, though theoretically less deep.  
- **z9j7wctoGV (Deep Kernel Relative Test)** – Accept Poster, scores around 6; good empirical performance and reasonable theory with a new kernel-testing variant.

Comparing:

- This NAMMD paper is **stronger theoretically** than the more application-focused MMD-MP or Deep Kernel Relative Test, with detailed asymptotics and sample complexity, and sits conceptually closer to RJSD.  
- However, like RJSD, its **empirical gains over MMD are small**, and the main narrative somewhat overclaims the impact (“MMD is less informative”, “higher power”) relative to what is actually, robustly demonstrated. There is also some mismatch between theoretical conditions and how they are interpreted in the main text, reminiscent of GPcSYm89wK’s issues.  
- Unlike GZ6AcZwA8r, it does not convincingly show large, practically meaningful improvements or a compelling real-world win where the new measure clearly outperforms existing approaches.

On this basis, I would place it above a borderline-weak theory paper that is clearly flawed, but below solid poster-level acceptances that either show larger gains or are more careful in claims. The contribution is genuine but currently feels like a modest normalization plus solid but relatively incremental theory, with somewhat overstated empirical takeaways.

A reasonable calibrated score is **5.0**: a borderline paper with interesting ideas and good theory, but insufficiently compelling empirical evidence and somewhat overstated motivation for acceptance in a competitive venue.

MY FINAL SCORE: <pineapple>5.0</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>