## Human Reviewer 1

### Summary
The paper studies the “counterintuitive likelihood” phenomenon in deep generative models, where anomalies sometimes receive higher likelihoods than normal data. While this issue is common in image domains, the authors show it is rare in tabular settings. They formalize a domain-agnostic definition of the phenomenon and evaluate normalizing flows and baselines across 47 tabular and 10 embedding datasets from ADBench. Results indicate that normalizing flow with Simple Likelihood Test is reliable for tabular anomaly detection. The authors further analyzes why the issue is less prominent in tabular data, highlighting the roles of dimensionality and feature-correlation differences.

### Strengths
- The empirical study is comprehensive and the analysis is reasonable.

- The finding that normalizing flow with Simple Likelihood Test outperforms other complicated methods is meaningful.

### Weaknesses
- The conclusion that normalizing flow with Simple Likelihood Test performs well on tabular data is indeed not surprising. As suggested by Kirichenko et al. (2020) and Schirrmeister et al. (2020), the counterintuitive phenomenon in the image domain is mainly caused by the strong local correlations in images, which is clealry not the case for tabluar data.

- The analysis from the perspective of intrinsic dimensionality and feature correlation is not novel. The impact of feature correlation is studied by Kirichenko et al. (2020) and Schirrmeister et al. (2020), and a deeper analysis of intrinsic dimensionality is provided in [1].

The present paper extends these arguments to a tabular benchmark but does not dramatically advance the theoretical understanding beyond existing literature.

[1] Kamkari, Hamidreza, et al. "A Geometric Explanation of the Likelihood OOD Detection Paradox." International Conference on Machine Learning. PMLR, 2024.

### Questions
How sensitive are the results to the particular choice of model architecture (e.g., flow-based vs autoregressive)?

### Soundness
3

### Presentation
2

### Contribution
1

### Rating
2

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper investigates why the counterintuitive phenomenon observed in image-domain anomaly detection—where deep generative models assign higher likelihoods to out-of-distribution data than to in-distribution data—rarely occurs in tabular data. The authors propose a formal definition of this phenomenon and conduct extensive experiments on 47 tabular datasets from ADBench using normalizing flows with simple likelihood tests (NF-SLT). They provide both theoretical and empirical explanations focusing on two factors: (1) lower dimensionality, and (2) lower feature correlation in tabular data. The results show that NF-SLT achieves competitive performance among 13 baselines without suffering from the counterintuitive phenomenon.

### Strengths
1. The theoretical analysis connecting dimensionality to likelihood gap and AUROC bounds is mathematically rigorous, the d Ratio analysis connecting intrinsic dimension to feature correlation provides valuable insights.

2. A comprehensive set of experiments was conducted, encompassing synthetic Gaussian experiments, dimensionality reduction studies, intrinsic dimension analysis, and performance evaluation on real-world datasets, ensuring the reliability and robustness of the findings.

3. The paper addresses a genuine gap in understanding why likelihood-based anomaly detection behaves differently across domains.

### Weaknesses
1. The authors conduct only simple likelihood testing with normalizing flows but provide no improvement strategies for scenarios where their analysis predicts poor performance. Through theoretical (Theorem 5.4, Corollary 5.6) and empirical analyses (Section 5.2, Table 3), the paper establishes that high dimensionality and strong feature correlation lead to SLT failure. However, no solutions are proposed or discussed for such cases.

2. The paper's contribution is weakened by defining a problem specifically to prove its absence. The authors construct Definition 3.3 based on observed characteristics of image-domain failures, then demonstrate that tabular data rarely exhibits this specific pattern. Definition 3.3's thresholds (β, γ) are empirically calibrated rather than theory-derived, and disconnected from Theorem 5.4's absolute failure conditions (entropy/dimension bounds). The paper proves tabular data lacks image-specific failure patterns, not that SLT genuinely succeeds. This reduces to showing "tabular ≠ images" rather than establishing reliability, limiting research contribution.

3. Lack of Failure Case Analysis: The paper briefly mentions "yeast" dataset failure but doesn't deeply analyze when and why NF-SLT fails.

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper asks whether the likelihood paradox that is well known from image out of distribution settings truly holds for tabular anomaly detection. The authors propose a clear operational definition of a counterintuitive effect based on the relative area under the receiver operating curve gaps against a pool of comparison methods. They study a normalizing flow model with a simple likelihood test on a large collection of tabular datasets and on several vision and language embedding datasets, and they compare against many classical and modern baselines. The main empirical message is that for ordinary tabular anomaly detection, the paradox is rare and that a straightforward flow plus likelihood test is frequently competitive and often strong. To explain why images behave differently, the paper develops a theoretical account in which increasing dimensionality and strong feature correlation tend to shrink the likelihood gap that separates in distribution and out of distribution. Additionally, the analysis is complemented with synthetic studies and an intrinsic dimension survey that suggests that common tabular datasets have an effective dimension closer to the ambient one than typical image data.

I find the overall direction valuable. The study addresses a common confusion in practice. The experimental sweep across many datasets is helpful for practitioners who need a dependable baseline.

At the same time, some choices reduce my confidence in the strength of the conclusions. The formal definition of a counterintuitive effect depends on fixed thresholds and on the exact competitor pool, yet there is no sensitivity analysis to show that the conclusion does not hinge on these choices. Several baselines appear under tuned or restricted, for example one class support vector machines only with a linear kernel, and some stronger recent tabular methods are not included. The paper repeats runs many times but does not report confidence intervals or statistical tests, and for heavily imbalanced problems I would also like to see area under the precision recall curve with uncertainty. The theoretical results rely on independence or near independence assumptions that are not directly checked on real data, while the empirical link through intrinsic dimension uses estimators that are known to be noisy and sensitive to scaling and sample size. The cross domain comparison also leans on experiments with embeddings for images rather than raw pixels in the main table, which weakens the bridge to the classical image paradox. With stronger baseline fairness, sensitivity checks, and uncertainty reporting, the message would be much firmer.

### Strengths
The paper addresses a real and timely question and does so with a broad empirical study. The message that tabular anomaly detection does not inherit the image likelihood paradox by default is useful and somewhat surprising to many readers. The definition of what counts as a counterintuitive phenomenon makes the discussion concrete and allows consistent counting across datasets. The theoretical story is appealing and connects intuition with mathematics. Higher dimension and stronger correlation often make pure likelihood less discriminative, and the synthetic ablations support this picture. The writing is mostly clear, tables are compact, and the appendices contain many implementation details that will help reproduction. As a practical takeaway, the flow plus likelihood recipe emerges as a credible line in the sand for tabular anomaly detection.

### Weaknesses
The credibility of the main claims would benefit from additional controls. The decision rule that declares a counterintuitive effect uses thresholds and a chosen competitor set. Without a sensitivity study, it is difficult to determine the stability of the conclusion under reasonable alternatives. Baseline coverage and tuning are uneven. Some classical methods are limited to narrow settings, and several stronger tabular anomaly approaches are missing. Hyperparameter search spaces differ across methods, which can favor the proposed approach. 

Although the authors repeat experiments multiple times, providing more information on confidence intervals, significance tests, and the area under the precision-recall curve could help assess the general quality, especially for imbalanced datasets. 

The theoretical analysis assumes independence or weak dependence for parts of the argument, yet there is no diagnostic that checks these assumptions on the real datasets. The bridge from theory to practice relies on intrinsic dimension estimators that are sensitive to preprocessing.

The work standardizes features but does not report how scaling choices and one hot encodings influence the intrinsic dimension statistics. I am missing a discussion of contamination or near duplicate risks in the benchmark pool, which is relevant for widely reused repositories.

### Questions
1) Can you provide a short sensitivity analysis of the counterintuitive definition with several reasonable choices of thresholds and with alternative competitor pools, and report how often the decision flips?

2) Can you include at least one strong one-class support vector machine with a radial basis kernel under a time budget, and add one or two recent tabular anomaly detectors that you excluded, so that the baseline set reflects the current state of the art?

3) Could you report diagnostics that test the independence or weak dependence assumptions on the real datasets, for example, correlation summaries or partial correlation sparsity, and discuss how violations would affect the theory.

4) For the intrinsic dimension link, can you show robustness to different scalings, to sub-sampling, and to an estimator beyond the two nearest neighbor and maximum likelihood ones, and clarify the handling of categorical variables and one-hot encodings

5) Please move at least a small pixel-level image comparison into the main text or summarize it more clearly, so that the reader can judge the cross-domain claim without digging through the appendix.

6) Did you run a near duplicate or contamination check across the dataset pool, and if not, can you comment on this risk

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
4

---

## Human Reviewer 4

### Summary
The present work investigates the *counterintuitive phenomenon* of likelihood in the context of structured tabular data. In the weakly supervised anomaly detection (AD) setup, where one disposes of a labeled dataset of normal and anomaly samples, one can train a generative model solely on the normal samples. The counterintuitive phenomenon describes the situation where a generative model, that provides an estimate of the likelihood of a sample given the learned **normal** distribution, e.g., Normalizing Flow, gives on average a higher likelihood to anomaly samples than to normal samples. This phenomenon has been mostly observed in the CV field and has yet to be investigated in the tabular domain.

The authors provide a formal definition of this counterintuitive phenomenon and investigate whether it arises in the tabular domain. They do so by comparing a Likelihood-based anomaly detection method to competing AD methods. They posit that if (i) likelihood as estimated by their model is sometimes higher for anomaly samples than for normal samples, but (ii) the other AD methods display an equivalent behavior with the anomaly score being higher for normal samples than for anomalies, this may be a sign that a different underlying phenomenon might be the cause.

They provide extensive experiments on a widely used AD benchmark, ADBench, and demonstrate that this counterintuitive phenomenon almost never happens when it comes to tabular data. The authors provide empirical and theoretical explanations as to why they observe these results.

### Strengths
**S1:** Experiments are rigorous and extensive. Authors rely on a widely used benchmark in the AD literature on tabular data.

**S2:** The results are **fully reproducible** as the authors provide the code to run their experiments.

**S3:** The research question is interesting and is worth investigating. The approach chosen by the authors is relevant and well-motivated.

### Weaknesses
**W1:** Overall the paper is **very hard to follow**. The structure is somewhat disorganized, and the formulations occasionally make it difficult to understand the points made by the author. Similarly, the paper contains **many typos**, we invite the authors to more thoroughly proofread their manuscript before submitting it.

**W2**: The authors should **avoid citing preprints (e.g., from arXiv) when the papers have been published**. A few examples found in the reference section with arXiv citations:
- A geometric explanation of the likelihood OOD detection paradox (Kamkari et al. 2024) has been accepted to ICML 2024.
- Density estimation using real nvp. (Dinh et al., 2017) has been accepted to ICLR 2017.
- On the universality of volume-preserving and coupling-based normalizing flows (Draxler et al., 2024) has been accepted to ICML 2024.
- Beyond individual input for deep anomaly detection on tabular data (Thimonier et al., 2024) has been accepted to ICML 2024.

**W3**: The method NF-SLT put forward by the authors is not explained in the main part of the paper.

### Questions
**Q1**: As mentioned in **W1** the paper should be **reorganized significantly** as it suffers from (i) unclear turns of phrases, (ii) many typos, (iii) unnecessary formalization. A few non-exhaustive examples:
- lines 54 to 68 are very hard to follow and can be confusing.
- line 114: "it follows a simple distribution $p_{\mathbf{z}}$ that tipically selected standard Gaussian".
- line 116: "$p_{\mathbf{x}}$ can be expressed as a formula expressed in terms of $p_{\mathbf{z}}$. The authors then proceed to use the term 'expressed' twice again in the following sentence.
- lines 141-143: the authors mention performance improvement twice, but never truly explain to what performance they are referring.
- line 148: "overcame disadvantage"
- Section 3: The beginning of Section 3 repeats more or less what is already being said in the introduction.
- Definition 3.3 appears as an overcomplication to formalize a simplistic problem that can be easily explained in plain words (as done in lines 204 to 208). Moreover, I do not think this can be considered a definition. 
- The acronym NF-SLT is used on several occurrences throughout the manuscript but is never defined (e.g., in the abstract).
- line 296: 'we extend the expression which expresses the expected'
- On several occasions, the authors mention "the d" to refer to the feature dimension. This is an odd turn of phrase.
- lines 340 to 361 are tough to follow and would benefit from smaller sentences.

**Q2**: As observed in Table 1 (lower part) and mentioned in section 4, it appears that **on the CV and NLP datasets, the *counterintuitive phenomenon* is not observed as NF-SLT outperforms all competing methods**. Since the premise is that this phenomenon is widely observed in the CV field, how do you account for the fact that it disappears when using the embedding representations rather than the original pixel representation? These embeddings do not suffer from the heterogeneity of features that tabular data displays. This should thus push this counterintuitive phenomenon according to section 5.2.

**Q3**: Could you provide **in the main text** the hyperparameter setting and overall training settings? While I acknowledge that some information is given in section 4, I would recommend providing more details (on architecture for example).

I lean towards reject due to the listed weaknesses. While I believe that this work might prove valuable in the future, it still needs significant improvement to enhance its clarity.

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
2

### Confidence
4