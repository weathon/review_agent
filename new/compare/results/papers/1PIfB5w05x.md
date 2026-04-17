# Price Of Quality: Sufficient Conditions For Sparse Recovery Using Mixed-Quality Data

Youssef Chaabouni Operations Research Center Massachusetts Institute of Technology Cambridge, MA 02139, USA youss404@mit.edu David Gamarnik Operations Research Center Massachusetts Institute of Technology Cambridge, MA 02139, USA gamarnik@mit.edu

## Abstract

We study sparse recovery when observations come from mixed-quality sources: a small collection of high-quality measurements with small noise variance and a larger collection of lower-quality measurements with higher variance. For this heterogeneous-noise setting, we establish sample-size conditions for informationtheoretic and algorithmic recovery. On the information-theoretic side, we show that it is sufficient for (n1, n2) to satisfy a linear trade-off defining the Price of Quality: the number of low-quality samples needed to replace one high-quality sample. In the agnostic setting, where the decoder is completely agnostic to the quality of the data, it is uniformly bounded, and in particular one high-quality sample is never worth more than two low-quality samples for this sufficient condition to hold. In the informed setting, where the decoder is informed of persample variances, the price of quality can grow arbitrarily large. On the algorithmic side, we analyze the LASSO in the agnostic setting and show that the recovery threshold matches the homogeneous-noise case and only depends on the average noise level, revealing a striking robustness of computational recovery to data heterogeneity. Together, these results give the first conditions for sparse recovery with mixed-quality data and expose a fundamental difference between how the information-theoretic and algorithmic thresholds adapt to changes in data quality.

## 1 Introduction 1.1 Overview And Previous Work 1.1.1 Sparse Recovery

Sparse recovery is a central problem in high-dimensional statistics and machine learning. Its applications include compressive sensing (Foucart et al., 2013; Candes et al., 2006; Donoho, 2006), ` signal denoising (Chen et al., 2001), sparse regression (Miller, 2002), data-stream algorithms (Cormode & Hadjieleftheriou, 2009; Indyk, 2007; Muthukrishnan et al., 2005), and combinatorial group testing (Du & Hwang, 1999). Other applications range from medical imaging to communications and compression (Foucart et al., 2013, Chap. 1). We formulate the problem as follows. A high-dimensional *signal* β
⋆ ∈ R
p(also called model or ground truth), unknown but a-priori s-sparse, is transmitted through a noisy channel that projects it onto a collection of n random vectors {xi}i∈[n]
in R
p. This is expressed as:
Y := Xβ⋆ + Z, (1)
where X = (x1*, . . . , x*n)
Tis called measurements, design or features; Y observations, annotations or *labels*; and Z *noise*. Specifically, we consider the setting of additive Gaussian noise, which is standard in the compressive sensing and sparse linear regression literatures. On the other end of the channel, a decoder who observes (*X, Y* ) is interested in recovering the support of the original signal β
⋆, i.e. the subset S
⋆:= {i ∈ [p]: β
⋆
i
̸= 0} ⊆ [p], known a-priori to be of cardinality s. How many observations n (as a function of p and s) does the decoder need to recover the support of the signal as the dimension of the problem grows to infinity? Previous works have shown that the sparse recovery problem exhibits two phase transitions at two thresholds, one *information-theoretic* and one *algorithmic*:

$$n_{\mathrm{INF}}={\frac{2s\log\left(p/s\right)}{\log s}}\quad{\mathrm{and}}\quad n_{\mathrm{ALG}}=2s\log\left(p-s\right)+s+1,$$

leading to three regimes:
- *n < n*INF: Signal support recovery is information-theoretically impossible. (Reeves et al., 2019). - nINF *< n < n*ALG: The maximum likelihood estimator (MLE) recovers S
⋆. However, it is believed that no algorithm can do it in polynomial time since the problem exhibits an Overlap Gap Property (OGP) (Gamarnik & Zadik, 2022), a structural property of the solution space known to often imply the failure of tractable algorithms to find optimal solutions.

- *n > n*ALG: The ℓ1-regularized least-squares estimator (also known as the LASSO (Tibshirani, 1996)) recovers S
⋆(Wainwright, 2009).

Of particular interest is the signal-to-noise ratio (SNR), known to be an important quantity for characterizing the difficulty of sparse recovery problems (Wang et al., 2010; Reeves et al., 2019; Chaabouni & Gamarnik). It's defined as follows:

$$\left(2\right)$$

$$\mathbf{SNR}:={\frac{\mathbb{E}\|X\beta^{\star}\|_{2}^{2}}{\mathbb{E}\|Z\|_{2}^{2}}}.$$

$$(3)$$
. (3)

## 1.1.2 Mixed Quality Data

A recent body of work has explored how low-quality data, e.g. labeled by an LLM or weak annotator (Ratner et al., 2017; Frenay & Verleysen, 2013), should be combined with fewer but higher-quality ´ data, e.g. labeled by humans or experts, for prediction and inference tasks (Gligoric et al., 2024; Li ´ et al., 2023; Zhang et al., 2023; Egami et al., 2023). In this paper, we formalize the mixed-quality data setting for sparse signal recovery: the decoder has access to n1 noisy projections of the signal β
⋆ with a small noise level σ 2 1 > 0 that we denote
{(yi, xi)}
n1 i=1 and call *high-quality* data. In addition, the decoder also observes a larger set of n2 >
n1 noisy projections of the same signal β
⋆, but with a higher noise level σ 2 2 > σ21, that we denote
{(yi, xi)}
n2 i=n1+1 and call *low-quality* data. We distinguish two settings:
- **Agnostic setting**: The decoder lacks access to observation-level noise variances and treats all measurements as if drawn from a single homogeneous model. This occurs when heterogeneous data sources lose provenance: for example in web-scale text corpora (Ratner et al., 2017; Frenay & ´ Verleysen, 2013) or citizen-science campaigns lacking sensor calibration (Silvertown, 2009). The decoder simply applies standard sparse-recovery methods without noise estimation or reweighting.

- **Informed setting**: where the decoder has access to the per-sample noise variance of the data. This regime captures situations where provenance information accompanies each observation, so the decoder knows which measurements are high- or low-quality. Examples include multi-site clinical trials or sensor networks that log calibration statistics (Loh & Wainwright, 2011; Delaigle et al., 2008), and medical-imaging datasets with per-rater confidence scores (Rajpurkar et al., 2018).

## 1.2 Our Work

In this paper, we consider the sparse recovery problem described above (1). Specifically, we study the setting where the measurements are drawn i.i.d. from a standard normal Gaussian distribution, and the noise is unbiased and drawn independently from Gaussian distributions of variance σ 2 1for the high-quality samples and σ 2 2 > σ21for the low-quality ones:

$$\{X_{ij}\}_{i\in[n],j\in[p]}\stackrel{{\text{i.i.d.}}}{{\sim}}\mathcal{N}\left(0,1\right)\text{and}Z=\Sigma W;\text{where}\Sigma=\begin{pmatrix}\sigma_{1}I_{n_{1}}&0\\ 0&\sigma_{2}I_{n_{2}}\end{pmatrix},W\sim\mathcal{N}\left(0,I_{n}\right).\tag{4}$$
Although much of the literature on sparse recovery in the homogeneous noise setting assumes constant noise level σ
2, we don't assume in this work that σ
2
1and σ
22are constant. In fact, the reason
$\eqref{eq:walpha}$. 
previous work can assume constant noise variance without loss of generality is that the model (1) could be scaled down by σ when the noise is homogeneous with variance σ 2to make it constant.

However, it is not the case anymore when the noise is heterogeneous. While our results naturally extend to sub-Gaussian errors, the sufficient conditions derived herein are not universal for general additive noise distributions. The assumptions we impose (Gaussian design, exact sparsity, and additive noise) are standard in the sparse recovery literature (e.g. Wainwright (2009); Reeves et al. (2019); Gamarnik & Zadik (2022)), and are adopted here to isolate the effect of heterogeneous noise while retaining the canonical structure of the recovery problem. Our analysis allows for arbitrary scalings of σ 21and σ 2 2 with respect to p and s. Since data come from two different sources with different scalings, we define in addition to (3) two signal-to-noise ratios: SNR1 for high-quality observations and SNR2 corresponding to low-quality observations. We are interested in the two following questions:
- **Sampling complexity of sparse recovery:** How large do the sample sizes (n1, n2) need to be for the decoder to be able, information-theoretically, to recover the support of the signal?

- **Algorithmic recovery:** How large do the sample sizes (n1, n2) need to be for the decoder to be able to recover the support of the signal using a polynomial-time algorithm?

We summarize below our findings on each of these questions in the agnostic and informed settings.

## 1.2.1 Sampling Complexity Of Sparse Recovery

In the first part of our work (section 3), we focus on the question of sampling complexity. For simplicity, we assume the signal is binary, i.e. β
⋆ ∈ {0, 1}
p. Note that in this case, recovering the support is equivalent to recovering the signal. This assumption is very common in the literature (Aeron et al., 2010; Reeves et al., 2019; Gamarnik & Zadik, 2022; Chaabouni & Gamarnik). Intuitively, detecting a component of size 1 is at least as hard as detecting a stronger component, so the resulting thresholds are representative of signals with non-zero entries bounded away from zero. We discuss this assumption in more detail in Remark 3.1. Our main results, Theorem 1 for the agnostic setting and Theorem 2 for the informed one, each provide a sufficient condition (9, 16) on the sample sizes (n1, n2) for support recovery. In both results, the condition has the form α1n1 + α2n2 > n⋆, for some coefficients α1, α2 > 0 depending on σ 2 1, σ22and s, and having different expressions in the agnostic and informed settings. In particular, we note that if (n1, n2) verify this condition (i.e. are together large enough), then so do (n1 − 1, n2 + α1/α2). In this sense, we say that one unit of high-quality data is worth:

$$\gamma\left(s,\sigma_{1}^{2},\sigma_{2}^{2}\right):=\frac{\alpha_{1}}{\alpha_{2}}$$
$$({\boldsymbol{5}})$$

units of low-quality data for the sufficient condition to hold. We label γ the *Price of Quality* and study its behavior in the agnostic and informed settings and for different regimes of SNR1 and SNR2. In the agnostic setting, it is uniformly bounded. In particular, under our sufficient condition, one high-quality sample is never worth more than two low-quality samples (13, 14) for the sufficient condition. In the informed setting, where the decoder is informed of per-sample variances, the price of quality goes to infinity in the low SNR2 & high SNR1 regime (20), and can be arbitrarily large in both low and high SNR regimes (19, 21).

## 1.2.2 Algorithmic Recovery

In the second part of our work (section 4), we focus on the question of algorithmic recovery. Unlike for sampling complexity, we don't assume that the signal is binary, but still require non-zero components to be bounded away from zero, i.e. there exists ρ > 0 such that mini∈S⋆ |β
⋆ i | ≥ ρ. This is standard in the literature (Aeron et al., 2010; Ndaoud & Tsybakov, 2020; Wang et al., 2010) since we can't hope to detect non-zero signal components if they can have arbitrarily small amplitude. Specifically, we study the question of *signed support* recovery, that is, recovering not only the indices of the non-zero components of the signal but also their sign (+ or −). This is usual in the algorithmic sparse recovery literature (Wainwright, 2009; Wang et al., 2010; Omidiran & Wainwright, 2008), as it follows naturally from the standard proof techniques.

Our main result, Theorem 3, provides necessary and sufficient conditions for the ℓ1-regularized leastsquares estimator (known as the LASSO) to recover the signed support of β
⋆in the agnostic setting.

Our result reveals that the problem behaves like the homogeneous-noise setting (Wainwright, 2009) with a homogeneous noise level equal to the average noise level of Z:

$$\sigma_{\mathrm{avg}}^{2}:={\frac{n_{1}\sigma_{1}^{2}+n_{2}\sigma_{2}^{2}}{n}}.$$
$$(6)$$

n. (6)
$$(7)$$
4
In particular, the sample size conditions (26, 27) do not depend on the noise levels σ 21, σ22. The condition on the LASSO regularization parameter (28) only depends on σ 21and σ 22through σ 2 avg and is the same as the one for homogeneous noise σ 2 avg (see equation (28) in Wainwright (2009)). We further provide a necessary and sufficient condition on noise scaling (Proposition 4.1). This shows that, unlike in sampling complexity, high-quality and low-quality data contribute equally to the sample size condition under which the LASSO recovers the support of the signal. Although we don't address algorithmic recovery in the informed setting, we briefly discuss it in Remark 4.2, where we discuss why the proof of Theorem 3 cannot be easily extended to the informed case.

## 1.3 Contributions, Outline And Notations

To the best of our knowledge, this paper is the first to: 1. Provide a sufficient condition for sparse recovery in the heterogeneous noise case, and quantify the trade-off between high-quality and low-quality data in the agnostic and informed settings.

2. Extend necessary and sufficient conditions for LASSO sparse recovery to the heterogeneousnoise, agnostic setting and show that high-quality and low-quality data contribute equally to reaching the algorithmic threshold.

We organize the rest of the paper as follows. Section 2 introduces the problem setup. Section 3 studies the sampling complexity of sparse recovery under heterogeneous noise. Section 4 investigates algorithmic recovery using the LASSO. Section 5 concludes and outlines directions for future work. Throughout this document, we will use the following notations:
- We say that f (x) ≃ g (x) as x → a ∈ R *∪ {−∞*, +∞} if and only if f (x) = g (x) (1 + o (1)).

- We denote by h (·) the binary entropy: h (x) = −x log x − (1 − x) log (1 − x), x ∈ (0, 1).

- We call ℓ0*-norm* the number of non-zero coordinates of x ∈ R
d, that is ∥x∥0
:=Pd i=1 1 (xi ̸= 0).

- We use uppercase letters (e.g. *X, Y, Z*) to indicate random quantities, and lowercase letters (e.g.

β) to denote deterministic parameters.

## 2 Preliminaries

The problem of sparse signal recovery is defined above (1). The decoder a-priori knows that β
⋆is s-sparse and belongs to a known set A ⊆ R
p. The design and noise are random with
(Xij )i∈[n],j∈[p]
i.i.d. ∼ N (0, 1) and Z := ΣW with Σ and W defined as in (4) and n1 + n2 = n.

We define Z
1:= (Z1*, . . . , Z*n1)
Tand Z
2:= (Zn1+1, . . . , Zn)
T, so that Z =
Z
1 Z
2
. The signalto-noise ratio (3) writes:

$$\mathbf{SNR}:={\frac{\|X\beta\|_{2}^{2}}{\|Z\|_{2}^{2}}}={\frac{n s}{n_{1}\sigma_{1}^{2}+n_{2}\sigma_{2}^{2}}}={\frac{s}{\sigma_{\mathrm{avg}}^{2}}},$$
, (7)
where σ 2 avg denotes the average noise level (6). In addition, we define the *high-quality SNR* and the low-quality SNR respectively by:

$$\text{SNR}_{1}:=\frac{\mathbb{E}\big{\|}\big{[}y_{i}-x_{i}^{T}\beta^{*}\big{]}_{i=1}^{n_{1}}\big{\|}_{2}^{2}}{\mathbb{E}\|Z^{2}\|_{2}^{2}}=\frac{s}{\sigma_{1}^{2}}\,\ \text{SNR}_{2}=\frac{\mathbb{E}\big{\|}\big{[}y_{i}-x_{i}^{T}\beta^{*}\big{]}_{i=n_{1}+1}^{n_{2}}\big{\|}_{2}^{2}}{\mathbb{E}\|Z^{2}\|_{2}^{2}}=\frac{s}{\sigma_{2}^{2}}.$$  In particular, we always have $\text{SNR}_{2}<\text{SNR}_{1}$, which reveals three regimes of interest:

- High SNR: when SNR1, SNR2 → +∞, or equivalently σ 22 = o (s).

- Low SNR2, High SNR1: SNR2 → 0, SNR1 → +∞ or equivalently σ 2 2 = ω (s), σ 21 = o (s).

- Low SNR: when SNR1, SNR2 → 0, or equivalently σ 2 1 = ω (s).

## 3 Sampling Complexity Of Sparse Recovery

In this section, we are interested in determining whether it is possible, information-theoretically, to recover the support of the signal, depending on the sample size n. We assume that β
⋆is binary and a priori s-sparse, that is: A := Bp,s = {β ∈ {0, 1}
p: ∥β∥0 = s} .

Remark 3.1 (Binary-signal assumption). Our results for sparse recovery can be viewed as applying to signals whose non-zero components are at least 1 in magnitude, i.e. β
⋆ ∈ Cp,s (1) := 
β ∈ R
d: mini∈Supp(β)|βi| ≥ 1	. Assuming that the non-zero entries are exactly equal to 1 serves only to simplify computations. Intuitively, detecting a component of magnitude 1 is at least as hard as detecting a stronger component, so stronger signals can only make recovery easier. Conversely, detecting a signal in Cp,s (1) is at least as hard as detecting a binary signal, since {0, 1}
p ⊆ Cp,s (1).

More generally, recovering any signal whose non-zero entries are bounded below by some ρ > 0 can be reduced to the case of Cp,s (1) by rescaling the model (1) by ρ.

Let A △ B *:= (*A ∪ B) \ (A ∩ B) denote the symmetric difference between any two finite sets A
and B, and Supp (β) := {i ∈ [p]: βi ̸= 0} denote the support of any vector β ∈ R
p. Let δ ∈ (0, 1).

We say that βˆ ∈ Bp,s *recovers the support of* β
⋆ *up to error* δ if
Supp βˆ△ Supp (β
⋆)
 < 2δs.

## 3.1 Agnostic Setting

In the agnostic setting where the decoder ignores the quality of each observation, the sample sizes
(n1, n2) and the noise levels σ
21, σ22
. Motivated by the maximum likelihood estimator in the
homoscedastic setting (see Gamarnik & Zadik (2022)), we define an estimator such that:
$$\hat{\beta}\in\operatorname*{arg\,min}_{\beta\in{\mathcal{B}}_{p,x}}\|Y-X\beta\|_{2}^{2}\,.$$
. (8)
Theorem 1 (Sufficient condition for support recovery in the agnostic setting). 1. Assume s = o (p) and s → +∞ as p → +∞*. Then let* n
⋆:= 2s log (p/s).

2. Assume s = αp for some constant α ∈ (0, 1)*. Then let* n
⋆:= 2h (α) p.

In both settings described above, if there exists ε > 0 *such that:*

$$n_{1}\log\left(1+\frac{\delta\left(2\sigma_{2}^{2}-\sigma_{1}^{2}\right)s}{2\sigma_{2}^{4}}\right)+n_{2}\log\left(1+\frac{\delta s}{2\sigma_{2}^{2}}\right)\geq\left(1+\varepsilon\right)n^{\star},$$
$$({\mathfrak{s}})$$
$$(9)$$
$$\mathbf{\Pi}_{,}^{r}\mathbf{100}$$
_then $\hat{\beta}$ recovers the support of $\beta^{*}$ up to error $\delta$ w.h.p.:_  $$\mathbb{P}\left(\left|\text{Supp}(\beta^{*})\triangle\text{Supp}\left(\hat{\beta}\right)\right|<2\delta s\right)\geq1-\exp\left\{-\left(\varepsilon+o\left(1\right)\right)n^{*}/2\right\}\stackrel{{p\to+\infty}}{{\longrightarrow}}1.$$

−→ 1. (10)
Proof Sketch. The proof of Theorem 1 is in appendix A and uses standard techniques. We control the probability that a high-error support attains a lower objective value in (8) than the ground truth and then take a union bound over such supports. For any β, we have:

$$\left\|Y-X\beta\right\|_{2}^{2}-\left\|Y-X\beta^{*}\right\|_{2}^{2}=\sum_{i=1}^{n}\left\{\left\langle X_{i},\beta^{*}-\beta\right\rangle^{2}+2Z_{i}\langle X_{i},\beta^{*}-\beta\rangle\right\}.$$
$$(11)$$

Applying a Chernoff bound to the RHS above and analyzing the MGF of the summands yields an exponent that factorizes across two blocks (see Proposition A.1). We conclude using a union bound over supports S with |S △ S
⋆| ≥ 2δs (there are at most ps of them).

We interpret Theorem 1 as follows. - **Sample Complexity.** In our setup, the decoder knows that β
⋆is exactly s-sparse. When s = 0 or s = p, the support is fully determined and there is no ambiguity, making recovery trivial and requiring no samples. For intermediate values of s, the decoder must distinguish among many candidate supports, whose cardinality is ps
. This combinatorial ambiguity renders the recovery problem non-trivial and leads to the sample complexity characterized by n
⋆in Theorem 1.

- **Price of Quality.** The sufficient condition for recovery (9) is equivalent to a linear combination of the sample size n1 and n2 being larger than the threshold n
⋆. The coefficients of the sample sizes reveal that one unit of high-quality data is worth:

$$\gamma:=\frac{\log\left(1+\delta\left(2\sigma_{2}^{2}-\sigma_{1}^{2}\right)s/\left(2\sigma_{2}^{4}\right)\right)}{\log\left(1+\delta s/\left(2\sigma_{2}^{2}\right)\right)}>1$$
$$(12)$$

units of low-quality data for the sufficient condition to hold. We call γ the *Price of Quality*. In fact, one unit of high-quality data can be replaced by γ units of low-quality data: that is, if (n1, n2)
are large enough for the sufficient condition to hold (and are hence sufficient for recovering β
⋆),
then so are (n1 − 1, n2 + γ).

- High SNR2 **regime.** Assume s = ωσ 22
. The price of quality (12) writes:

$$=\omega\;(\omega_{2})$$
$$\gamma\simeq\frac{\log\left(\delta s/\left(2\sigma_{2}^{2}\right)\right)+\log\left(2-\sigma_{1}^{2}/\sigma_{2}^{2}\right)}{\log\left(\delta s/\left(2\sigma_{2}^{2}\right)\right)}\simeq1,$$

which means that when σ 2 1, σ22 = o (s), the high-quality and low-quality data contribute equally to the recovery condition (9).

- Low SNR2 **regime.** Assume s = oσ 22
. The price of quality (12) writes:

$$(13)$$
$$\gamma\;(1\angle2)\;\;\mathrm{with}$$
$=\;\angle J\;=\;\frac{\pi}{2}$
$$\gamma\simeq\frac{\delta\left(2\sigma_{2}^{2}-\sigma_{1}^{2}\right)s/\left(2\sigma_{4}^{2}\right)}{\delta s/\left(2\sigma_{2}^{2}\right)}\simeq2-\frac{\sigma_{1}^{2}}{\sigma_{2}^{2}}.$$
$$(14)$$

. (14)
Note that γ < 2 for any σ 2 1
, σ2 2. We conclude, in the low SNR regime, that under our sufficient condition, one high-quality sample can be replaced by up to two low-quality samples.

## Remark 3.2 (Limitations).

- The condition in Theorem 1 is sufficient and is not expected to be information-theoretically sharp.

The potential looseness arises from a relaxation in the Chernoff bound used to control the probability of support misidentification. In the heterogeneous-noise setting, optimizing the Chernoff exponent leads to a cubic equation (see (37)), whose exact solution yields a tighter but less tractable condition. To retain a closed-form and interpretable sufficient condition, we rely on a relaxation of this equation. In the homogeneous-noise case, solving the analogous equation is known to recover the sharp threshold, and we expect that optimizing (37) would similarly lead to a tighter characterization here, though we do not pursue this direction in the present work.

- Even under the assumption that the decoder is agnostic to the quality of the data, the estimator βˆ
(8), might not constitute the best approach to recover the support of β
⋆. For instance, especially in the low SNR regime, the decoder might re-weight the loss of each observation by the magnitude of its observed label, i.e.:

$$\operatorname*{arg\,min}_{\beta\in{\mathcal{B}}_{p,s}}\sum_{i=1}^{n}{\frac{1}{Y_{i}^{2}}}\Big(Y_{i}-\langle x_{i},\beta\rangle\Big)^{2},$$

as an attempt to rescale each row of data by its noise level. In fact, in the low SNR regime we have EY
2 i ≃ σ 2 i where σ 2 idenotes the noise level corresponding to the i th observation, which motivates the use of Y
2 ias a proxy for σ 2 i when the noise levels are unknown.

- Classical approaches to heteroscedastic regression either assume known noise levels or explicitly acknowledge heteroscedasticity as part of the statistical modeling assumptions (see, e.g. Buja et al. (2019)). Extending such considerations to sparse support recovery in the mixed-quality setting introduces an additional layer of difficulty, since one must control both the accuracy of variance-related modeling and its impact on support identification. While variance-aware procedures may improve performance when the noise levels differ significantly, a rigorous analysis of such methods is beyond the scope of this work and constitutes an interesting direction for future research.

## 3.2 Informed Setting

In this section, we assume that the decoder knows the distribution of each noise entry: N0, σ21 or

N0, σ22
. Recall the distributions of Z and W from (4). The MLE is define by (see appendix B for
a proof):
a proof).  $$\hat{\beta}_{\text{MLE}}\in\arg\min_{\beta\in B_{p,*}}\left\|\Sigma^{-1}\left(Y-X\beta\right)\right\|_{2}^{2}.$$  **Theorem 2** (Sufficient condition for support recovery in the informed setting).  
. (15)
1. Assume s = o (p) and s → +∞ as p → +∞*. Then let* n
⋆:= 2s log (p/s).

2. Assume s = αp for some constant α ∈ (0, 1)*. Then let* n
⋆:= 2h (α) p.

In both settings described above, if there exists ε > 0 *such that:*

$$(15)$$
$$n_{1}\log\left(1+\frac{\delta s}{2\sigma_{1}^{2}}\right)+n_{2}\log\left(1+\frac{\delta s}{2\sigma_{2}^{2}}\right)\geq\left(1+\varepsilon\right)n^{\star},\tag{16}$$  _then $\hat{\beta}_{\textit{MLE}}$ recovers the support of $\beta^{\star}$ up to error $\delta$ w.h.p.:_  $$\mathbb{P}\left(\left|\textit{Supp}\left(\beta^{\star}\right)\triangle\textit{Supp}\left(\hat{\beta}_{\textit{MLE}}\right)\right|<2\delta s\right)\geq1-\exp\left\{-\left(\varepsilon+o\left(1\right)\right)n^{\star}/2\right\}^{p\to+\infty}1.\tag{17}$$

Proof Sketch. The proof of Theorem 2 is given in appendix C and follows a similar argument as Theorem 1. Here, the rescaled loss in (15) leads to a Chernoff bound that can be optimized in closed-form, yielding a sharp convergence rate.

We interpret Theorem 2 as follows.

- **Price of Quality.** In the informed setting, the expression of the price of quality is different from the one in the agnostic case (12). It writes:

$$\gamma=\log\left(1+\frac{\delta s}{2\sigma_{1}^{2}}\right)\Big/\log\left(1+\frac{\delta s}{2\sigma_{2}^{2}}\right).$$

- **Low SNR regime.** Assume σ 21 = ω (s). Then:

$$\gamma\simeq\sigma_{2}^{2}/\sigma_{1}^{2}.$$
1. (19)
* **Low SNR${}_{2}$, High SNR${}_{1}$ regime.** Assume $\sigma_{2}^{2}=\omega\left(s\right)$ and $\sigma_{1}^{2}=o\left(s\right)$. Then: $$\gamma=\Theta\left(\frac{\log\left(s/\sigma_{1}^{2}\right)}{s/\sigma_{2}^{2}}\right)=\Theta\left(\frac{\log\text{SNR}_{1}}{\text{SNR}_{2}}\right)\stackrel{{p\rightarrow+\infty}}{{\longrightarrow}}+\infty.$$
$$(18)$$
$$(19)$$
$$(20)$$

- **High SNR regime.** Assume σ

$${\mathrm{assume~}}\sigma_{2}^{2}=o\left(s\right).{\mathrm{~Then:~}}$$
$$\gamma\simeq\log\left(s/\sigma_{1}^{2}\right)/\log\left(s/\sigma_{2}^{2}\right)=\log\mathrm{SNR}_{1}/\log\mathrm{SNR}_{2}.$$
= log SNR1/ log SNR2. (21)
Remark 3.3. - Compared to the agnostic setting (Theorem 1), the appropriate rescaling of the loss in the MLE
(15) constitutes a better use of the high-quality data, in the sense that it leads to a higher price of quality γ. In particular, γ is infinite in the low SNR2 & high SNR1 setting (20) and can be arbitrarily large in both low and high SNR regimes (19, 21).

$$(21)$$

- The sufficient condition (16) is obtained by optimizing the Chernoff exponent exactly (see (39)
and (42)). In homogeneous-noise settings, analogous optimizations are known to yield necessary and sufficient thresholds (Gamarnik & Zadik, 2022; Wang et al., 2010; Chaabouni & Gamarnik). Establishing full necessity in the heterogeneous setting remains an interesting direction for future work.

Remark 3.4 (Generalizations of Theorem 1 and Theorem 2). - Generalization to *signed* **support recovery.** The large-deviation bound in the proofs of Theorem 1 and Theorem 2 suggests a potential extension of the sufficient conditions, respectively (9)
and (16), to the setting where the non-zero components of the signal β
⋆are not necessarily +1 but rather in {−1, +1}, and the decoder is interested in the *signed* support recovery, where they recover not only the indices of the non-zero components of β
⋆, but also their sign. In this setting, the error measure expressed by the symmetric difference of supports in (10) and (17) extends to the number of 'wrong' components in βˆ, given by ∥βˆ − β
⋆∥0. The threshold n
⋆ ≃ log ps would increase by an additive factor of s log 2 to account for the increase in the size of the search space since β
⋆ ∈ {β *∈ {−*1, 0, 1}
p: ∥β∥0 = s}, which has cardinality 2 sps
. Asymptotically, this does not change the leading-order scaling in the sub-linear regime when s = o (p), and adds an extra α log 2 term to the h (α) factor in the linear regime when s = αp.

- Generalization to *arbitrary noise* **structures.** Theorem 1 and Theorem 2 are stated in the simple setting where the data comes from two sources, one *good* and one bad, motivated by the high- and low- quality data problem. The proof strategy suggests that these results extend to non-singular noise. In fact, if we only assume that Σ is invertible, but not necessarily having the form in (4),
then the sufficient condition (9) in Theorem 1 extends to:

$$\sum_{i=1}^{n}\log\left(1+\frac{\delta\left(2\sigma_{\max}\left(\Sigma\right)^{2}-\sigma_{i}\left(\Sigma\right)^{2}\right)s}{2\sigma_{\max}\left(\Sigma\right)^{4}}\right)\geq\left(1+\varepsilon\right)n^{\star},\tag{22}$$
$$(23)$$

where {σi (Σ)}
i=n i=1 denote the σ-values of Σ, σmax (Σ) := maxi=1*,...,n* σi (Σ) and σmin (Σ) :=
mini=1*,...,n* σi (Σ). Similarly, the sufficient condition (16) in Theorem 2 extends to:

$$\sum_{i=1}^{n}\log\left(1+{\frac{\delta s}{2\sigma_{i}\left(\Sigma\right)^{2}}}\right)\geq\left(1+\varepsilon\right)n^{\star}.$$
⋆. (23)

## 4 Algorithmic Recovery

In this section, we are interested in the existence of a tractable algorithm to recover the support of the underlying signal. We assume that the components of the signal β
⋆take real values and are bounded away from zero: that is, A := Cp,s (ρ) = β ∈ R
p: ∥β∥0 = s, mini∈Supp(β)|βi| ≥ ρ	, for some ρ ∈ R+. We say that βˆ ∈ R
precovers the signed support of β
⋆if sign(βˆ) = sign(β
⋆), where the sign: R *−→ {−*1, 0, 1} function is defined by sign (0) = 0 and sign (x) = x/ |x| for all x ̸= 0, and is applied coordinate-wise. A common approach to recovering the signed support of the signal is using the solution to the following ℓ1-constrained quadratic program, also known as the LASSO:

$${\mathcal{B}}_{\mathrm{Lasso}}:=\operatorname*{arg\,min}_{\beta\in\mathbb{R}^{p}}\left\{{\frac{1}{2n}}\left\|Y-X\beta\right\|_{2}^{2}+\lambda_{p}\left\|\beta\right\|_{1}\right\},$$

where λp ≥ 0 denotes a sequence of regularization parameters converging to 0 as p → +∞. We are interested in characterizing the regime where the LASSO recovers the signed support of the true signal. Specifically, we call "recovery" the event:

$${\mathcal{R}}\left(X,\beta^{\star},Z,\lambda_{p}\right):=\left\{\,\exists\,\,{\hat{\beta}}\in{\mathcal{B}}_{\mathrm{Lasso}}\,\colon\,\,\mathrm{sign}\left({\hat{\beta}}\right)=\mathrm{sign}\left(\beta^{\star}\right)\right\}.$$

In the homogeneous noise setting, Wainwright (2009) showed that the performance of the LASSO in estimating the signed support of β
⋆exhibits a phase transition with respect to the sample size. In fact, there exists a threshold nALG such that: - If *n > n*ALG: then the LASSO correctly recovers the signed support of β
⋆.

$$(24)$$

$$(25)$$

## - If N < Nalg: Then The Lasso Fails To Recover The Signed Support Of Β ⋆.

In addition, it is widely believed that no algorithm can recover the support of β
⋆in polynomial time when *n < n*ALG. Indeed, Gamarnik & Zadik (2022) showed that the problem exhibits an OGP. This motivates the use of (24) to estimate β
⋆in the agnostic setting where the decoder treats the data impartially. Our main result of this section, Theorem 3, extends the result mentioned above on the LASSO threshold (by Wainwright (2009)) to the heterogeneous, agnostic noise setting.

Theorem 3 (Lasso recovery phase transition). Assume that, as p → +∞; s goes to infinity, s =
o (p) and n1, n2 = ω (s). Let nALG := 2s log (p − s) + s + 1.

_._ 2. _If there exists $\varepsilon>0$ such that:_ $$n<(1-\varepsilon)\,n_{\text{\tiny{MLG}}},$$ (26) _then, for any sequence $\lambda_{p}>0$ such that $\frac{n_{1}\sigma_{1}^{2}+n_{2}\sigma_{2}^{2}}{\lambda_{p}^{2}n^{2}}$ has a limit in $\mathbb{R}_{\geq0}\cup\{+\infty\}$, we have_ $\mathbb{P}_{X,Z}\Big{(}\mathcal{R}\left(X,\beta^{*},Z,\lambda_{p}\right)\Big{)}\to0$_._
$$(30)$$
$$(31)$$
9
ii. If there exists ε > 0 *such that:*
n > (1 + ε) nALG, (27)
_and $\left(\lambda_{p}\right)_{p\geq1}\to0$ is chosen such that:_  $$\frac{n\lambda_{p}^{2}}{\sigma_{\alpha x}^{2}\log\left(p-s\right)}\to+\infty,\quad\text{and}\quad\frac{1}{\rho}\left[\lambda_{p}\sqrt{s}+\sqrt{\frac{\sigma_{\alpha x}^{2}\log s}{n}}\right]\to0,$$  _then $\mathbb{P}_{X,Z}\Big{(}\mathcal{R}\left(X,\beta^{*},Z,\lambda_{p}\right)\Big{)}\to1$._
$$(27)$$
$$n>\left(1+\varepsilon\right)n_{A L G},$$
$$(28)$$
The full proof of Theorem 3 is given in Appendix D and follows the core LASSO threshold argument of Wainwright (2009). We use the same argument but generalize it to the heterogeneous-noise setting, where the presence of the matrix Σ, no longer a scalar multiple of the identity, causes key steps of the classical proof to fail. We overcome this by applying a Gram–Schmidt (QR) decomposition of XS (49) and analyzing the resulting orthogonal matrix using properties of the Haar measure on the orthogonal group (e.g. see Lemma D.6). The monograph of Meckes (2019) on Haar-distributed matrices was particularly valuable in understanding this component from random-matrix theory.

Proof Sketch of Theorem 3. We express the recovery property (25) via the first-order optimality conditions of the LASSO (24):  $$\mathcal{R}\left(X,\beta^{*},\Sigma w,\lambda_{p}\right)\Longleftrightarrow\left\{\begin{aligned}&\left|\left(\frac{1}{n}X_{S}^{T}X_{S}\right)^{-1}\left(\frac{1}{n}X_{S}^{T}\Sigma w-\lambda_{p}\text{sign}\left(\beta_{S}^{*}\right)\right)\right|<\left|\beta_{S}^{*}\right|\\ &X_{S}^{T}X_{S}\left(X_{S}^{T}X_{S}\right)^{-1}\left(\frac{1}{n}X_{S}^{T}\Sigma w-\lambda_{p}\text{sign}\left(\beta_{S}^{*}\right)\right)-\frac{1}{n}X_{S}^{T}\cdot\Sigma w\right|\leq\lambda_{p}\\ \end{aligned}\right.\tag{29}$$
where absolute values and inequalities are taken component-wise. This well-known result (Wainwright, 2009; Fuchs, 2004; Meinshausen & Buhlmann, 2006; Tropp, 2006; Zhao & Yu, 2006) is ¨ stated in Proposition D.1. When (27) and (28) hold, the random variables inside the absolute values on the RHS of (29) concentrate below their respective upper bounds, establishing sufficiency. When (26) holds, the second absolute value in (29) concentrates above λp, showing necessity. Although Theorem 3 does not explicitly state any condition on the scaling on the noise, the existence of λp → 0 such that (28) holds requires that the noise does not scale arbitrarily large. The next result explicitly states this condition.

Proposition 4.1 (Necessary and sufficient condition on noise scaling). *If there exists* (λp)p≥1 → 0 such that (28) holds, then:

$$\sigma_{\alpha\mathrm{{avg}}}^{2}=o\left({\frac{n}{(1+s/\rho^{2})\log{(p-s)}}}\right).$$

Conversely, if (30) holds, let:

$$\lambda_{p}:=\left(\frac{\sigma_{\mathrm{avg}}^{2}\log\left(p-s\right)}{\left(1+s/\rho^{2}\right)n}\right)^{1/4}.$$

Then λp → 0 and (28) holds.

Proof. See appendix E.

Remark 4.1 (Correlated features). Theorem 3 is stated for independent features (i.e. xi ∼ N (0, Ip) for all i ∈ [n]). In the homogeneous-noise setting, analogous results for correlated designs under suitable regularity conditions on the covariance matrix were established by Wainwright (2009). Extending the heterogeneous-noise analysis to correlated designs requires additional tools and is left for future work. In this paper, we therefore focus on the independent-feature case.

Remark 4.2 (Informed setting). Although we establish the phase transition for the LASSO only in the agnostic setting, a natural extension in the informed setting is the rescaled estimator defined by minimizingΣ
−1(Y − Xβ)
2 2 instead of ∥Y − Xβ∥
2 2 in (24). Extending the proof of Theorem 3 and Wainwright (2009) to this setting is nontrivial, as the presence of Σ
−1factors alongside the design matrix in (29) destroys the Wishart structure XT
S XS ∼ W(Is, n) used to control the moments of its inverse via classical inverse-Wishart arguments (Anderson et al., 1958; Siskind, 1972). An analysis would therefore require controlling the moments of (XT
S Σ
−2XS)
−1, which remains an interesting direction for future work.

## 5 Conclusion And Future Work

We study the problem of sparse recovery when observations come from mixed-quality sources. We establish sufficient conditions on the sample sizes (n1, n2) for both information-theoretic and algorithmic recovery purposes and in two settings, one when the decoder is completely agnostic to noise and one where they are informed of the per-sample noise variance. At the level of the information-theoretic threshold, we study the trade-off between high-quality and low-quality samples, and label the number of low-quality samples required to replace one highquality sample when our sufficient condition holds *the Price of Quality*. In the agnostic setting, we reveal that this entity is quite low: in particular, under our sufficient condition, one high-quality sample is never worth more than two low-quality samples. However, in the informed setting, the price of quality can grow arbitrarily large depending on the noise variances and the signal-to-noise regime. This highlights a key practical implication of our results: whenever possible, quantify uncertainty in the annotations and rescale the loss accordingly. At the algorithmic threshold, we show in the agnostic setting that the classical LASSO recovery results from the homogeneous setting remain valid in the heterogeneous case and depend only on the total sample size n1 + n2. First, the threshold itself is independent of the individual noise levels. Second, the sufficient condition on the penalization coefficient involves the noise only through its average, exactly as if all observations had that average noise. Consequently, high-quality and low-quality samples contribute *equally* to the sample-size requirement for LASSO recovery. This reveals an unexpected difference in the effect of data heterogeneity on the information-theoretic and algorithmic thresholds for recovery. Within the Gaussian design framework considered here, the informed information-theoretic threshold and the LASSO threshold are sharp, whereas the agnostic information-theoretic condition is sufficient but not proven tight. In a broader discussion on how the information-theoretic and algorithmic thresholds interact across different problem settings, our result further emphasizes that the algorithmic threshold seems to be more "robust" to changes in the traditional problem settings (Gamarnik & Zadik, 2022; Wainwright, 2009). In fact, Wang et al. (2010) and Chaabouni & Gamarnik observed that when the noise is homogeneous but the design is sparse (i.e. Xij set to 0 uniformly at random) the informationtheoretic threshold increases, while Omidiran & Wainwright (2008) showed that the algorithmic threshold remains the same and is unaffected by changes in the sparsity level of the data (although this was shown only for the sufficient condition, with no corresponding result on necessity). Although we do not study LASSO recovery in the informed setting, this remains a promising direction for future work. It would be interesting to study the price of quality there, and compare it to LASSO recovery in the agnostic setting on one hand, and to the price of quality of informationtheoretic recovery on the other.

## Acknowledgments

This work was supported by the National Science Foundation (NSF) under grant CISE-2233897.

Youssef Chaabouni thanks Mehdi Makni, Marouane Nejjar, Panos Tsimpos, Malo Lahogue, and Alexandre Misrahi for insightful discussions and valuable feedback.

## References

Shuchin Aeron, Venkatesh Saligrama, and Manqi Zhao. Information theoretic bounds for compressed sensing. *IEEE Transactions on Information Theory*, 56(10):5111–5130, 2010. doi: 10.1109/TIT.2010.2059891.

Theodore Wilbur Anderson, Theodore Wilbur Anderson, Theodore Wilbur Anderson, Theodore Wilbur Anderson, and Etats-Unis Mathematicien. ´ An introduction to multivariate statistical analysis, volume 2. Wiley New York, 1958.

Andreas Buja, Lawrence Brown, Richard Berk, Edward George, Emil Pitkin, Mikhail Traskin, Kai Zhang, and Linda Zhao. Models as approximations i. *Statistical Science*, 34(4):523–544, 2019.

Emmanuel J Candes, Justin Romberg, and Terence Tao. Robust uncertainty principles: Exact signal `
reconstruction from highly incomplete frequency information. *IEEE Transactions on information* theory, 52(2):489–509, 2006.

Youssef Chaabouni and David Gamarnik. The price of sparsity: Sufficient conditions for sparse recovery using sparse and sparsified measurements. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Scott Shaobing Chen, David L Donoho, and Michael A Saunders. Atomic decomposition by basis pursuit. *SIAM review*, 43(1):129–159, 2001.

Graham Cormode and Marios Hadjieleftheriou. Finding the frequent items in streams of data. Communications of the ACM, 52(10):97–105, 2009.

Laurens De Haan and Ana Ferreira. *Extreme value theory: an introduction*. Springer, 2006. Aurore Delaigle, Peter Hall, and Alexander Meister. On deconvolution with repeated measurements.

2008.

David L Donoho. Compressed sensing. *IEEE Transactions on information theory*, 52(4):1289–
1306, 2006.

Ding-Zhu Du and Frank Kwang-ming Hwang. *Combinatorial group testing and its applications*,
volume 12. World Scientific, 1999.

Naoki Egami, Musashi Hinck, Brandon Stewart, and Hanying Wei. Using imperfect surrogates for downstream inference: Design-based supervised learning for social science applications of large language models. *Advances in Neural Information Processing Systems*, 36:68589–68601, 2023.

Simon Foucart, Holger Rauhut, Simon Foucart, and Holger Rauhut. An invitation to compressive sensing. Springer, 2013.

Benoˆıt Frenay and Michel Verleysen. Classification in the presence of label noise: a survey. ´ *IEEE*
transactions on neural networks and learning systems, 25(5):845–869, 2013.

J-J Fuchs. Recovery of exact sparse representations in the presence of noise. In 2004 IEEE International Conference on Acoustics, Speech, and Signal Processing, volume 2, pp. ii–533. IEEE, 2004.

David Gamarnik and Ilias Zadik. Sparse high-dimensional linear regression. estimating squared error and a phase transition. *The Annals of Statistics*, 50(2):880–903, 2022.

Kristina Gligoric, Tijana Zrnic, Cinoo Lee, Emmanuel J Cand ´ es, and Dan Jurafsky. Can unconfident `
llm annotations be used for confident conclusions? *arXiv preprint arXiv:2408.15204*, 2024.

Yinzheng Gu. Moments of random matrices and. 2013. Piotr Indyk. Sketching, streaming and sublinear-space algorithms. Graduate course notes, available at, 33:617, 2007.

Michel Ledoux. *The concentration of measure phenomenon*. Number 89. American Mathematical Soc., 2001.

Minzhi Li, Taiwei Shi, Caleb Ziems, Min-Yen Kan, Nancy F Chen, Zhengyuan Liu, and Diyi Yang.

Coannotating: Uncertainty-guided work allocation between human and large language models for data annotation. *arXiv preprint arXiv:2310.15638*, 2023.

Po-Ling Loh and Martin J Wainwright. High-dimensional regression with noisy and missing data:
Provable guarantees with non-convexity. *Advances in neural information processing systems*, 24, 2011.

Pascal Massart. Concentration inequalities and model selection: Ecole d'Ete de Probabilit ´ es de ´
Saint-Flour XXXIII-2003. Springer, 2007.

Elizabeth S Meckes. *The random matrix theory of the classical compact groups*, volume 218. Cambridge University Press, 2019.

Nicolai Meinshausen and Peter Buhlmann. High-dimensional graphs and variable selection with the ¨
lasso. 2006.

Alan Miller. *Subset selection in regression*. chapman and hall/CRC, 2002.

Shanmugavelayutham Muthukrishnan et al. Data streams: Algorithms and applications. *Foundations and Trends® in Theoretical Computer Science*, 1(2):117–236, 2005.

Mohamed Ndaoud and Alexandre B Tsybakov. Optimal variable selection and adaptive noisy compressed sensing. *IEEE Transactions on Information Theory*, 66(4):2517–2532, 2020.

Dapo Omidiran and Martin J Wainwright. High-dimensional subset recovery in noise: Sparsified measurements without loss of statistical efficiency. *arXiv preprint arXiv:0805.3005*, 2008.

Pranav Rajpurkar, Jeremy Irvin, Robyn L Ball, Kaylie Zhu, Brandon Yang, Hershel Mehta, Tony Duan, Daisy Ding, Aarti Bagul, Curtis P Langlotz, et al. Deep learning for chest radiograph diagnosis: A retrospective comparison of the chexnext algorithm to practicing radiologists. *PLoS*
medicine, 15(11):e1002686, 2018.

Alexander Ratner, Stephen H Bach, Henry Ehrenberg, Jason Fries, Sen Wu, and Christopher Re.´
Snorkel: Rapid training data creation with weak supervision. In Proceedings of the VLDB endowment. International conference on very large data bases, volume 11, pp. 269, 2017.

Galen Reeves, Jiaming Xu, and Ilias Zadik. The all-or-nothing phenomenon in sparse linear regression. In *Conference on Learning Theory*, pp. 2652–2663. PMLR, 2019.

Alvin C Rencher and G Bruce Schaalje. *Linear models in statistics*. John Wiley & Sons, 2008.

Jonathan Silvertown. A new dawn for citizen science. *Trends in ecology & evolution*, 24(9):467–
471, 2009.

Victor Siskind. Second moments of inverse wishart-matrix elements. *Biometrika*, 59(3):690–691, 1972.

Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology, 58(1):267–288, 1996.

Joel A Tropp. Just relax: Convex programming methods for identifying sparse signals in noise.

IEEE transactions on information theory, 52(3):1030–1051, 2006.

Martin J Wainwright. Sharp thresholds for high-dimensional and noisy sparsity recovery using ℓ1-constrained quadratic programming (lasso). *IEEE transactions on information theory*, 55(5):
2183–2202, 2009.