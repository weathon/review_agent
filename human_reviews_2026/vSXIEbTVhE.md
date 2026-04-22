# Censoring with Plausible Deniability: Asymmetric Local Privacy for Multi-Category CDF Estimation

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
We introduce a new mechanism within the Utility-Optimized Local Differential Privacy (ULDP) framework that enables censoring with plausible deniability when collecting and analyzing sensitive data. Our approach addresses scenarios where certain values—such as large numerical responses—are more privacy-sensitive than others, while accompanying categorical information may not be private on its own but could still be identifying. The mechanism selectively withholds identifying details when a response might indicate sensitive content, offering asymmetric privacy protection. Unlike previous methods, it avoids the need to predefine which values are sensitive, making it more adaptable and practical. Although the mechanism is designed for ULDP, it can also be applied under symmetric LDP settings, where it still benefits from censoring and reduced privacy cost. We provide theoretical guarantees, including uniform consistency and pointwise weak convergence results. Extensive numerical experiments demonstrate the validity of developed methodologies.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes a mechanism for estimating conditional distribution functions
$P(X \leq x \mid Y=k) \mid k \in \\{1,\dots, K\\}$ from $I$ users' features $(x_i, y_i) \in [0,1] \times \\{1,\dots, K\\}$ in a locally private manner.  
Unlike in standard local differential privacy, which requires privacy uniformly for all values, this work seeks a form of asymmetric privacy in which only large values of $x_i$ are considered more sensitive.
An additional challenge this works seeks to address is that it may be hard to define a threshold $t_n$ that divides sensitive and non-sensitive values.

The proposed mechanism largely works as follows:
First, a threshold $t_i$ is randomly sampled from $G = \mathrm{Uniform}(0,1)$ for each user $i$.
Second, their features are mapped into $\mathcal{X} = \\{0,1\\} \times \\{1,\dots, K\\}$ via the corresponding indicator function.
This transformed feature space is thus considered divided into a sensitive region $\mathcal{X}_N = \\{1\\} \times \\{1,\dots, K\\}$, which corresponds to large $x_i$, and a safe region (its complement).
To guarantee privacy for these transformed features, the proposed mechanism (I will forego the indicator function notation here) deterministically maps all elements from
$\mathcal{X}_N$ to $1$ (censoring the categorical feature) and randomly maps elements $(0, k)$ from $X \setminus \mathcal{X}_N$ to either $(0, k)$ or $1$.

This mechanism is shown to be privacy-preserving in the following sense ("ULDP" from prior work):
For sensitive output $1$, it is hard (in the DP sense) to determine the original transformed feature from $\mathcal{X}_N$. No privacy is required for the non-sensitive outputs
in $\\{0\\} \times \\{1,\dots, K\\}$.

Using the post-processing property of DP, the authors then present some MLE-based method for recovering 
They further derive error bounds, which characterize the expected error in estimating $P(X \leq x \mid Y=k)$, averaged over all $k$, when randomly sampling $x$ from the threshold-generating distribution $G$.

Finally, the method is numerically evaluated on its ability to recover some simple hand-crafted CDFs.

### Strengths
* The mechanism for privately mapping from transformed discrete features to CDFs is well-founded and improves upon prior work
* The theoretical evaluation of error bounds / asymptotic properties is extensive, and makes for a much better contribution than pure empirical evaluation of the privacy--utility trade-off
* The considered problem: Only protecting sensitive values, without knowing exactly which range of values is sensitive, is novel and well-motivated
* The outline in Section 1.2 helps significantly in following the paper
* Limitations are openly and extensively discussed

### Weaknesses
## Main Weakness
The main issue I see with the paper is that it **does not address the actual problem of performing ULDP with unknown sensitive ranges** and it is **not clear what form of privacy guarantee is made w.r.t. the original continuous features**.
The underlying cause is that the privacy analysis only starts *after* we have decided on thresholds $t_i$, instead of analyzing the entire mechanism.

Generally speaking, when analyzing the privacy of a composite mechanism $M = M_1 \circ M_2$, it is not sufficient to only analyze the privacy of $M_1$ in the co-domain of $M_2$.

More concretely, this work essentially assumes that whatever thresholds we sample do in fact accurately separate sensitive and non-sensitive value, and provides its guarantees under this assumption.
But there being no fixed "correct" thresholds is precisely what motivated the research in the first place.   
To me, it seems like the work is trying to model some prior belief about the correct thresholds for each user (here: an uninformative, uniform prior). But it does not model at all what happens when there is a mismatch between the unknown thresholds $t_i^*$ and the sampled threshold $t_i$. For example, we can probably say that being in the top $0.1\\%$ of earners is sensitive. But with the proposed mechanisms, there is a
slightly smaller than $0.1\\%$ **chance that this sensitive information is leaked without any privacy protection at all**.  
It appears like actually solving the considered problem requires defining some $(\epsilon,\delta)$-style relaxed ULDP notion that models the inherent uncertainty of the sensitive range and resulting failures of providing ULDP.

## Other Weaknesses
Presentation:
* The abstract highlights "extensive numerical experiments" as a core contribution. However, these results only appear in the appendix (and are not particularly extensive either). It seems like the authors did not put enough care in actually writing their work for the conference format
* While the overall storyline (see Outline in Section 1.2) is coherent, the individual sections 3-5 are poorly structured. They are essentially just a stream of prose, interspersed with equations. This makes it hard to deduce the main arguments the authors are trying to make. The reading flow could be much improved by including more Proposition/Definition/Remark etc. environments.
* The algorithm for computing CDFs from the mechanism's output is not sufficiently discussed. The authors just point at some existing paper, which puts an unnecessary burden on the reader.

Other:
* The method is only evaluated on a handful of hand-crafted CDFs instead of real-world data. One could compare the private CDF with the empirical CDF to get better insights into the real-world utility of the mechanism.
* While I understand that pure DP work is regularly published at ML conferences, the authors should at least make an effort to explain how their work is relevant for machine learning purposes
* It's somewhat problematic that the main utility metric (expected error in CDF) is directly coupled with the threshold distribution $G$ of the mechanism. Example: Consider threshold distribution $G = \delta_{0.5}$. Then the metric would only measure how well the true CDF is approximated at $x=0.5$, completely ignoring that it is an extremely poor match (a heaviside step function) everywhere else.
* Appendix F only shows that the mechanism improves upon randomized response, but not that it improves upon the "utility-optimized randomized response mechanism" mentioned in Section 3.2.

## Minor comments

* The subscripts $N$, $I$, $S$, $P$ in Definition 3 are somewhat confusing. Wouldn't using $S$ ("sensitive") and $N$ ("non-sensitive") for both the domain and co-domain be sufficient.
* Definition 1 speaks of "randomized algorithms" whereas Definition 2 speaks of "randomized mechanisms". 
* In Definition 1, it is not clear that datasets $S$ are composed of elements of $X$, which (for a reader without a DP background) would make it hard to tell the difference to Definition 2
* "Sensitivity" already has a well-defined meaning in the differential privacy context. I would suggest using another word for "being sensitive".
* "while allowing exact outputs" in (ll. 160) is somewhat imprecise. ULDP can also be fulfilled if we do not return the exact value, but some deterministic/invertible transformation of it, right?
* Using $\mathcal{Y}$ for the co-domain of the mechanism and $Y$ for the categorical part of the domain is slightly confusing.
* Introducing $\Delta_i$ from Section 3.3 earlier would have made the one-hot construction in Section 3.2 much easier to follow



## Conclusion
If this work were to only operate in the usual ULDP framework, it would be a relatively straight-forward work that proposes a somewhat improved mechanism with nice theoretical analysis but somewhat lacking presentation and experimental evaluation. This would put it somewhere in borderline territory.

However, the authors open up a much more interesting problem: How to adapt ULDP mechanisms when the true separation into "sensitive" and "safe" values is unknown. A solution to this problem could be much more impactful and be of broader interest for the privacy commmunity. 
Unfortunately, their proposed method fails to adequately address this problem and does not provide sound privacy guarantees (see "Main Weaknesses" above).
I would encourage the authors to revise their manuscript, and maybe try to explicitly model the interaction between uncertain "sensitive ranges" and the randomly sampled thresholds, for a sound and much stronger submission.

### Questions
/

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new mechanism for Utility-Optimized Local Differential Privacy (ULDP) that enables asymmetric privacy protection—censoring potentially sensitive responses while maintaining plausible deniability. Instead of predefining a sensitive region, the mechanism relies on a direction of sensitivity (e.g., higher income or debt values are more sensitive). It combines deterministic thresholding with randomized response to protect sensitive values while retaining useful information about nonsensitive ones. The authors define the Asymmetrically Censored Randomized Response (ACRR) mechanism. Then they develop a maximum likelihood estimator (MLE) for CDF estimation under ULDP and prove $L^2$ and uniform consistency (rates $n^{-1/3}$).

### Strengths
- Novel conceptual contribution: The idea of asymmetric censoring with plausible deniability is both original and practically motivated. It moves beyond uniform privacy budgets by allowing directional sensitivity—aligning privacy protection more closely with real-world data semantics (e.g., “high debt” being sensitive).
- Solid theoretical foundation: The asymptotic analysis (Theorems 1–3) is rigorous and builds on advanced shape-constrained inference and current-status data literature. The proof techniques are carefully adapted to handle ULDP’s non-differentiabilities.
- Strong connection between privacy and statistics: The paper effectively bridges differential privacy and nonparametric inference, showing that local privacy mechanisms can be interpreted through censoring frameworks common in survival analysis.
- Practical mechanism design: The proposed ACRR is interpretable, easy to implement, and avoids arbitrary sensitive-region definitions—a major usability improvement over earlier ULDP schemes.

### Weaknesses
- Please use \cite, \citep, and \citet correctly. Note that the form of \cite can vary between different Latex template. 
- Line 114: The symbol $K$ appears for the first time here but is not defined. Please define $K$ at its first occurrence.
- About the definition of ULDP: 
	- The initial presentation of the ULDP definition is somewhat difficult to follow and would benefit from a more intuitive explanation or an illustrative example. Since ULDP is not yet a widely adopted concept, I strongly suggest adding a short, concrete example (e.g., a toy scenario) to help readers quickly grasp what the mechanism is doing.
	- The definition itself is quite interesting. Taking the survey protocol in Appendix A as an example, the plausible deniability comes from some low-debt participants occasionally reporting “Yes,” thereby concealing the true high-debt participants. However, from the perspective of high-debt individuals, the only valid output is “Yes.” This asymmetry could lead to survey refusal or misreporting if respondents are not well informed about the mechanism. My question is: if such refusal occurs, can the current framework naturally accommodate it—for example, by allowing some users (possibly those with high debt, i.e., data-dependent) or a predefined subset of users to instead follow a standard LDP mechanism? This issue has been discussed in the literature in the context of heterogeneous privacy requirements, where different data records may demand different protection levels. See, for example [1,2].

[1] Versatile Differentially Private Learning for General Loss Functions.

[2] Locally Private Estimation with Public features.

### Questions
See weaknesses. I would be happy to raise my score if the questions were discussed in depth.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a new mechanism in the Utility-Optimized Local Differential Privacy framework that enables censoring with plausible deniability for sensitive data collection. It targets settings with asymmetric sensitivity (e.g., large numeric responses are more private than small ones, and categorical context can be identifying) by selectively withholding identifying details when a response likely signals sensitive content. Unlike prior approaches, the mechanism does not require pre-specifying which values are sensitive, improving practicality and adaptability; it also applies under standard symmetric LDP, where censoring can still lower privacy cost. The authors provide theoretical guarantees, including uniform consistency and pointwise weak convergence, and corroborate the approach with extensive numerical experiments demonstrating favorable utility–privacy trade-offs.

### Strengths
The paper is technically sound and the proofs appear correct. The paper provides both theoretical guarantees and empirical results.

### Weaknesses
**Major concerns**

- Missing baselines: The paper lacks comparisons against baseline methods. 

- Utility bounds and privacy dependence: The stated utility guarantees (Theorems 1 and 2) do not appear to depend on the privacy parameter $ \varepsilon $. It would strengthen the work to provide utility bounds for the private algorithm that make the $ \varepsilon $-dependence explicit.

- Absent privacy proofs: I could not find a formal proof of the mechanisms’ privacy guarantees (eg. ACRR in defn. 4). Please include precise statements (with assumptions and neighboring relations) and complete proofs or clear references.

- Meaning of “adaptive” censoring: The claim “a flexible ULDP mechanism that *adaptively* censors potentially sensitive responses without requiring a predefined sensitive region” needs clarification. What quantities are adapted, based on which statistics or at what granularity?

**Minor issues**

- The paper never defines “censoring.” A formal definition would be helpful.

### Questions
Please see the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel asymmetric local differential privacy mechanism under the Utility-Optimized Local Differential Privacy (ULDP) framework.

The key contribution is a censoring mechanism with plausible deniability, designed to protect users whose responses may implicitly reveal sensitive information (e.g., large numerical values). Unlike previous approaches, the method does not require predefined sensitive regions, instead using a direction of sensitivity (e.g., “larger means more sensitive”).

The work represents the first asymptotic analysis of CDF estimation under ULDP with multi-category prediction, bridging privacy-preserving data collection and nonparametric estimation theory.

### Strengths
1.  **Originality**: Introduces asymmetric local privacy with plausible deniability—protecting sensitive values adaptively without predefining a sensitive region; Bridges ULDP and classical CDF estimation theory, leveraging Chernoff-type asymptotics.
2.  **Quality**: Rigorous proofs of consistency and weak convergence and nontrivial extension of LDP results (Liu et al., 2024) from single to multi-category settings.

### Weaknesses
1.  The estimator’s runtime grows superlinearly with sample size and linearly with the number of categories K. 
2.  The “capping at 1” rule for CDF may distort tail behavior; a quantitative analysis of this effect would strengthen the paper.
3.  The $O_p(n^{-1/3})$ rate is slower than the parametric $O_p(n^{-1/2})$ rate; discussion of possible improvements (e.g., semi-parametric approaches) would be valuable.

### Questions
1.  Can the mechanism handle non-monotonic sensitivity, where both small and large values are sensitive?
2.  Could jointly estimating sub-CDFs under a sum-to-one constraint mitigate boundary bias?
3.  How would the estimator generalize to multivariate continuous data $X \in \mathbb{R}^d$ ?
4.  Can the mechanism be adapted to streaming or online ULDP settings?

### Soundness
3

### Presentation
2

### Contribution
2
