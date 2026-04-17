# Sample-efficient Multiclass Calibration under $\ell_{p}$ Error

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Calibrating a multiclass predictor, that outputs a distribution over labels, is particularly challenging due to the exponential number of possible prediction values. In this work, we propose a new definition of calibration error that interpolates between two established calibration error notions, one with known exponential sample complexity and one with polynomial sample complexity for calibrating a given predictor. Our algorithm can calibrate any given predictor for the entire range of interpolation, except for one endpoint, using only a polynomial number of samples. At the other endpoint, we achieve nearly optimal dependence on the error parameter, improving upon previous work. A key technical contribution is a novel application of adaptive data analysis with high adaptivity but only logarithmic overhead in the sample complexity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper is theoretically focused. It introduces a new definition of multiclass calibration error that unifies and interpolates between existing notions. Additionally, it provides an algorithm that achieves polynomial sample complexity, which is near-optimal in some cases. 

The main body of the paper presents key theorems, lemmas, and proofs, while detailed derivations are deferred to the appendix. The work also includes a discussion of related literature in calibration and adaptive data analysis, highlighting how the proposed framework unifies and extends existing results. There are no experiments or empirical illustrations, as the contribution is mainly theoretical.

### Strengths
The paper is original in formulating the new $\ell_p$ calibration error, which unifies and interpolates between existing calibration notions while removing the exponential dependence on the number of classes found in prior work. The theoretical analysis appears rigorous and internally consistent, introducing an interesting adaptive-data-analysis argument that achieves near-optimal sample complexity. The paper is well structured for a theoretical contribution. Overall, it seems to provide a meaningful and technically solid advance in the theory of multiclass calibration, but I did not verify the proofs.

### Weaknesses
1. While the theoretical contribution appears strong, the work does not include an empirical analysis that could help clarify the behavior of the proposed algorithm. The practicality of the method is not discussed; although the algorithm is polynomial, the constants may be large, and it is unclear how it would scale in realistic settings. It is not clear how tight the upper bounds are in practice, this could have been investigated empirically on synthetic datasets.

2. The paper is highly technical and difficult to follow: 
* There are no illustrative diagrams or figures that could help explain the algorithm. For example, for 3 classes it would be possible to visualise the simplex, the initial discretization corresponding to $\lambda$, and the following merging steps in G and M. In particular, I would have appreciated more intuitive descriptions about the interplay of G and M, both in the general case and for some concrete toy example with 3 classes.
* The three events are introduced without an explanation about why these particular events are good to consider. 
* I think that pushing some lemmas to the appendix and making more space for illustrations and empirical analysis would have made sense.

3. Some minor weaknesses are reflected in the questions below.

### Questions
1. In the introduction (lines 25–28), could the authors clarify how “trustworthiness” and “interpretability” relate to calibration? The connection between these ideas could be made more explicit.  
2. Line 103 mentions that “for interpretability reasons, the outputs of our calibrated predictor should be probability distributions.” Could the authors elaborate on this? Shouldn’t the predictor’s outputs be probability distributions regardless of interpretability concerns, e.g. for automatic cost-sensitive decision-making purposes?  
3. It might be better to define the asymptotic notation with a tilde ($\tilde{O}$) explicitly, as it appears in Theorems 1 and 6 but is never formally introduced.  
4. How practical would it be to implement the proposed algorithm for different values of $k$ and $\varepsilon$? It would be helpful if the authors could provide an estimate or discussion of the expected runtime or sample requirements in practical settings to better understand whether the method could be applied in practice or remains primarily of theoretical interest. 
5. Minor (Line 621): There are missing spaces in the proof of Lemma 2, which slightly affects readability.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper studies efficient calibration of existing predictors while controlling the accuracy loss. It generalizes (interpolate) the existing sample error definitions of $\ell_1$ and $\ell_\infty$ to $\ell_p$ and preserves polynomial sample complexity.

### Strengths
Originality:
The use of adaptive data analysis can be considered creative. 

Quality:
Some intermediate results are technically correct.

Clarity:
General findings are understandable.

Significance:
The theoretical novel findings are the extension of the existing calibration error definitions in the form $\ell_p$ to $p\in(1,\infty)$ and applying suitable calibrations with efficient sampling.

### Weaknesses
I argue for rejection. Below is my reason.

I understood the general direction of the paper. However, most of the details are not apparent to me. The paper has major clarity issues. I am unsure about the meaning and definitions of many of the notations, so it is extremely difficult for me to verify (even coarsely) many of the intermediary claims and understand the explanations in between. For example, when you say $O(\log(|B|))$ collections of bins, how are these collections separated? Possibly, the paper is too condensed. Even so, I believe the paper needs to be revised at least to become more reader-friendly.

### Questions
Questions:

Page 1 Line 53: weaker definition is unclear. What is it exactly?

Page 2 Line 84: why $1/\epsilon^2$?

Page 3 Line 112: intuition regarding $1/\epsilon^3$ is unclear. Can you explain more?

Page 5 Line 266: why do they never split?

Page 6 Line 323: using $O(...)$ what exactly?

Page 8 Line 379: why are the sizes powers of $2$?


Suggestions:

Page 5 Definition 5: the error definition seems unconventional since the probabilities are also taken to the power of $p$. The effect of this in the final findings needs to be addressed.

Page 6 Line 290: this paragraph is too convoluted and does not help digestion.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new Lp calibration error definition for multiclass classification that interpolates between expected calibration error (p=1) and maximum calibration error (p=∞). The authors present an algorithm that calibrates any given k-class predictor using polynomial samples for all p > 1 and achieves near-optimal dependence on the error parameter for p=∞.

### Strengths
- The polynomial sample complexity for all p > 1 is a significant achievement
- The near-optimal Õ(1/ε²) bound for p=∞ substantially improves over prior exponential-sample methods.
- The proposed discretization scheme using cleverly reduces the prediction space from exponential to polynomial by enforcing the simplex constraint.
- The algorithm preserves the accuracy of the given predictor (up to small discretization error)

### Weaknesses
**Conflicting definition of Lp calibration error**

The paper introduces "Lp calibration error" as a novel contribution, but this term is already established in the literature with a different definition. 

- https://arxiv.org/pdf/1909.10155 Eq (1) and paragraph below
- https://proceedings.neurips.cc/paper_files/paper/2022/file/33d6e648ee4fb24acec3a4bbcd4f001e-Paper-Conference.pdf Eq (1)
- https://proceedings.neurips.cc/paper_files/paper/2022/file/3915a87ddac8e8c2f23dbabbcee6eec9-Paper-Conference.pdf Eq (2)

Reusing established terminology for a different concept creates confusion and makes it difficult to compare with related work.

**Complete absence of experiments**

The paper is purely theoretical with zero empirical validation. No experiments on synthetic or real datasets demonstrating:
- Practical sample complexity vs. theoretical bounds
- Computational runtime
- Comparison to existing calibration methods 
- Behavior across different values of p
- Quality of calibrated predictions
- Density of theoretical presentation

**The paper presents a very dense sequence of theoretical results**

 While mathematical rigor is commendable, the presentation feels overwhelming and difficult to parse. The paper would benefit from 1) providing more intuitive explanations before formal statements 2)  including proof sketches or key ideas for the most important lemmas in the main text 3) potentially consolidating some lemmas or moving technical details to supplementary material to improve readability

**Limited discussion of p=1 case**

The ECE (p=1) case is mentioned but explicitly excluded from the algorithm. This seems like an important practical case that practitioners care about.

Minor:
- Typo L381: there are can be

### Questions
How does the proposed definition relate to established concepts in calibration: top-label, marginal and canonical calibration?

How should practitioners choose between your definition and these alternatives? A formal hierarchy or comparison of calibration definitions would clarify the paper's positioning.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This manuscript addresses the challenge of achieving sample-efficient multiclass calibration, particularly in scenarios where the prediction space grows exponentially with the number of classes $k$. The authors introduce a new definition of calibration error: $\ell_p$ calibration error. This definition is proposed to interpolate between definitions requiring exponential samples and those requiring polynomial samples. The main result Theorem 6 demonstrates an algorithm that can calibrate any given predictor $f$ to achieve low $\ell_p$ calibration error ($\leq \epsilon$) for all $p > 1$, using only a sample complexity that is polynomial in $k$ and $1/\epsilon$. Furthermore, for the case $p=\infty$, the sample complexity achieves a near-optimal dependence on the error parameter $\epsilon$, scaling as $O(1/\epsilon^2)$ up to logarithmic factors. The algorithm is designed to preserve the accuracy of the original predictor $f$, keeping its squared error within an additive term $\tilde{O}(\epsilon^{p/(p-1)} / 2^{1/(p-1)})$. An important technical contribution lies in the analysis of adaptive data analysis for the iterative calibration process which yields an overhead in sample complexity that is only logarithmic in $1/\epsilon$ and improves upon prior polynomial overheads. The algorithm relies on identifying high-probability bins and adaptively merging groups of bins by utilizing a unique discretization scheme that is significantly smaller than previous approaches.

### Strengths
1. The $\ell_p$ calibration error definition naturally addresses the issue of the exponential complexity associated with strong error notions in the multiclass setting.

2. The algorithm achieves polynomial sample complexity in the number of classes $k$ for the entire range $p>1$ which provides a powerful tool for multiclass calibration where previous efforts either achieved weak guarantees or required exponential samples.

3. The method successfully calibrates an existing predictor while bounding the resultant decrease in accuracy by a small additive term.

4. The technical result bounding the sample complexity overhead due to adaptive data analysis to $O(\log(1/\epsilon))$ is a significant improvement over standard differential privacy composition techniques that yield polynomial or square-root overheads.

5. The used prediction space partition provides a polynomial bound on the number of bins, contrasting sharply with the exponential size of the space used in prior work.

### Weaknesses
1. The work is purely theoretical. While the theoretical contribution seems strong, empirical evidence is required to validate the practical feasibility and tightness of the bounds. This especially true regarding the complexity constants hidden in the $\tilde{O}$ notation and the effect of the adaptive merging process.

2. The calibration procedure involves correcting prediction vectors and projecting them onto the probability simplex. As noted by the authors, this projection can alter previously calibrated coordinates and needs careful selection to preserve accuracy. While the analysis addresses this especially in Lemma 10, 11, 12; the practical stability and computational efficiency of these projection steps might be a concern.

3. The bounds explicitly depend on $p$, particularly the factor $2^{1/(p-1)}$ and $\epsilon^{p/(p-1)}$. When $p$ is close to 1, this factor becomes large. This reflects the increased difficulty when approaching the ECE ($p=1$) definition since the algorithm cannot handle at that endpoint. Further discussions on practical guidance for selecting $p$ would enhance clarity.

### Questions
1.  The case $p=1$ corresponds to the expected calibration error (ECE). Could you eloborate on why the algorithm fails for $p=1$? Is there an intrinsic limitation preventing the extension of the current adaptive analysis or error bounding techniques to this critical case? Is there any intuitive reason?

2.  The definition of the $\ell_p$ calibration error incorporates the probability mass of the bin into the $p$-exponent. How does this definition practically compare to other proposed $\ell_p$-style measures in the literature (e.g., Kumar et al., 2019; Popordanoska et al., 2022) in terms of interpretability or robustness beyond the theoretical advantage of testability?

3.  The algorithm's efficiency relies on the adaptive data analysis achieving logarithmic overhead, which leverages the disjointness property of groups in $M$. Does the implementation of the core adaptive mechanism require the strong composition property of differential privacy in practice, or is this primarily an analytical tool to derive the sample complexity bound? Clarifying the practical requirements for achieving this logarithmic overhead would be valuable.

4.  The final predictor $h(x)$ uses the nearest valid probability vector $\rho(R(f(x)))$ if $f(x)$ falls into a low-probability bin (outside $B$). Since $B$ contains bins with estimated probability mass $\mû_v \geq \beta/6$, can the authors quantify how large the calibration error might be for these low-probability bins that are not explicitly modified or tracked by the iterative process?

### Soundness
3

### Presentation
2

### Contribution
3
