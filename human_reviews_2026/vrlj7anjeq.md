# Rao Differential Privacy

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2, 4

## Abstract
Differential privacy (DP) has recently emerged as a definition of privacy to release private estimates. DP calibrates noise to be on the order of an individual's contribution. Due to this calibration, a private estimate obscures any individual while preserving the utility of the estimate. Since the original definition, many alternate definitions have been proposed. These alternates have been proposed for various reasons including improvements on composition results, relaxations, and formalizations. Nevertheless, thus far nearly all definitions of privacy have used a divergence of densities as the basis of the definition. In this paper we take an information geometry perspective towards differential privacy. Specifically, rather than define privacy via a divergence, we define privacy via the Rao distance. We show that our proposed definition of privacy shares the interpretation of previous definitions of privacy while improving on sequential composition.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new variant of Differential Privacy (DP) called “Rao Differential Privacy” (Rao DP). Instead of defining DP via a divergence of probability densities (as most other definitions do, with the exception of GDP), Rao DP defines privacy via the Rao distance of probability densities. The authors demonstrate that Rao DP supports tight (sequential) composition, and give Rao DP guarantees for the Laplace, Gaussian and Generalized Gaussian mechanisms.

### Strengths
1. Defining DP via distance between densities rather than a divergence makes sense.
2. The paper is well-written in a pleasant style that makes it easy to follow.
3. I think it can be argued that the paper fits the scope of ICLR.

### Weaknesses
1. A paper whose main contribution is to propose a new DP variant needs to, in my opinion, argue convincingly for why the new definition is needed. Besides being of theoretical interest, I do not see what need Rao DP addresses.
2. The existence of Gaussian DP weakens the case for Rao DP. While the definitions are different, one capturing distance over densities and the other hardness of distinguishing two densities via hypothesis testing, they appear to behave similarly. Both definitions exactly characterize the Gaussian mechanism (it is $\mu$-GDP and $\theta$-Rao DP), exhibit the same composition behavior, and both avoid the use of divergences in their definition.
3. I am not convinced that Rao DP as a metric for privacy is easy to convey. Due to the interpretation via hypothesis testing, 1-GDP has a clear meaning independently of the specifics of the use-case. It is not clear to me what guarantee I am afforded by 1-Rao DP.

### Questions
The paper is well-written and pleasant to read, and its proposed privacy definition of Rao DP appears technically correct. That said, I am not fully convinced that Rao DP as a concept has a clear use case, therefore I lean slightly towards rejection as things stand. I am willing to change my stance if the authors or other reviewers bring up good arguments in support of the definition's utility.

I ask the following questions to get a better idea of the paper’s contribution.
1. What is the use case for Rao DP? Is it a promise of tighter composition in general? If so, the paper would benefit from an example where it improves over e.g., GDP.
2. There are no conversion theorems in the paper from Rao DP to other DP variants (or vice versa). Is this endemic to how Rao DP is defined (the avoidance of divergence), or is it possible to give meaningful conversion theorems? In the case of Gaussian DP there are conversion results from Gaussian DP to zCDP and approximate DP.
3. You analyzed the Laplace and Gaussian mechanism, for which you could derive the Rao DP guarantee cleanly. Do you expect the guarantee for other DP primitives such as randomized response, sparse vector technique, report noisy max, exponential mechanism etc., to follow as cleanly? Generally speaking, is proving Rao DP for an algorithm easier or harder compared to e.g., approximate DP?
4. Conversely, is there any structure in the definition of Rao DP that make it amenable to showing strong lower bounds?
5. How would you explain the guarantee afforded by 1-Rao DP, without discussing the output distribution of the algorithm satisfying it?
6. On page 3, you have a footnote saying you only consider sequential composition, and not e.g., parallel composition. Does Rao DP satisfy parallel composition? I would assume it does, and if so I would add a proof/remark somewhere in the paper, or just add to the footnote "[...] which Rao DP also supports".

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Rao Differential Privacy (Rao-DP), which bounds the Fisher–Rao (Rao) geodesic distance between output distributions of a mechanism on adjacent datasets using a privacy budget θ. It establishes sequential composition, post-processing invariance, and calibration for Laplace, Gaussian, and Generalized Gaussian mechanisms. The authors also relate Rao-DP to Gaussian DP (GDP).

### Strengths
S1: This paper introduces a true distance metric (Fisher–Rao) for privacy, linking DP to information geometry.

S2: The paper proves post-processing invariance using the square-root density embedding.

S4: It derives calibration formulas for standard mechanisms (Laplace, Gaussian).

S5: The paper extends to generalized Gaussian mechanisms beyond standard distributions.

### Weaknesses
W1: The definition of sensitivity in the paper is presented abstractly through a general distance function $d_\eta$ between datasets, but there is limited discussion on how this metric should be chosen or computed for real-world queries. In standard differential privacy, sensitivity is explicitly tied to the query function (for example, using L1 or L2 norms for numerical queries), which enables straightforward calibration of noise. While the Rao-DP formulation leaves $d_\eta$ largely symbolic. It would be more interesting to make it clear how to apply the framework to practical tasks such as mean estimation, histogram queries, or model parameter updates.

W2: The current explanation of $\delta$ as “the probability that pure-DP holds with probability 1 − $\delta$ and is violated with probability $\delta$” is somewhat misleading. $\delta$ does not literally represent the probability that privacy fails on a random draw. Rather, $\delta$ bounds the total probability mass of outcomes for which the $\epsilon$-DP inequality may not hold. It would be clearer to phrase $\delta$ as a worst-case slack or tail-probability bound instead of a “failure probability.”

W3: There are many typos and grammar errors:
Definition 2.1: “is said to by $\epsilon$-differentially private” → should be “is said to be $\epsilon$-differentially private.”
Definition 2.2: “datesets” → should be “datasets.” 
“a pre-specified parameters” → should be “pre-specified parameters.” (remove “a”).

Page 3: "One requires the defintion of" -> "One requires the definition of"
Page 9: "definition an be extended" -> "definition can be extended"

### Questions
Q1: Could the authors clarify how the distance function $d_\eta$ is chosen or computed in practice? For instance, should it measure differences in the data domain, the mechanism’s output space, or the Fisher–Rao parameter space? 

Q2: Could the authors clarify their interpretation of the $\delta$ parameter in Definition 2.2? Specifically, do they intend $\delta$ to represent a literal probability of privacy failure, or a bound on the probability mass where the $\epsilon$-DP inequality may be violated?

Q3: Does the √(Σ θᵢ²) composition bound rely on independence between mechanisms? If the mechanisms are dependent or adaptive, does the bound still hold, and under what conditions?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Rao Differential Privacy (Rao DP), a new definition of differential privacy based on the Rao distance rather than divergence measures such as KL or Rényi divergence. The authors take an information geometry perspective, proposing that measuring the dissimilarity between mechanisms via a proper distance metric offers a more natural and interpretable notion of privacy. They show that Rao DP satisfies key properties of DP, i.e., post-processing immunity and sequential composition, and derive the corresponding privacy parameters for Laplace, Gaussian, and Generalized Gaussian mechanisms. The composition rule under Rao DP yields a tighter bound (Euclidean-like addition of privacy budgets) than traditional DP.

### Strengths
* Introduces a novel, geometry-based definition of differential privacy using the Rao distance.

* Demonstrates that Rao DP satisfies composition and post-processing properties fundamental to DP.

* Provides closed-form derivations for Laplace, Gaussian, and Generalized Gaussian mechanisms.

### Weaknesses
* Limited empirical or practical validation

The paper is entirely theoretical; it does not provide any numerical illustration or simulation to demonstrate the implications of Rao DP in realistic DP tasks (e.g., trade-offs between privacy and utility). Including such an example would make the contribution more compelling.

* Comparative discussion lacks depth

While the paper claims tighter composition and better interpretability, it does not quantitatively compare the bounds of Rao DP with those of existing relaxations (e.g., Rényi DP or zCDP) on a common task. Without this, it is hard to assess how substantial the improvement is in practice.

* Connection to Gaussian DP is underdeveloped

The paper notes that Rao DP and Gaussian DP (GDP) share similar composition behavior and even the same parameter mapping for the Gaussian mechanism. However, the conceptual difference between the two frameworks remains somewhat vague. The discussion in Appendix C.1 hints at a deeper connection but stops short of establishing whether Rao DP subsumes or generalizes GDP.

* Scope limitations

The definition is demonstrated only for continuous, one-parameter mechanisms. The extension to multivariate or discrete domains is acknowledged but not developed, which limits the generality of the proposal.

### Questions
1. The paper claims tighter composition, but could the authors provide a quantitative example comparing the total privacy loss between Rao DP and Rényi DP or zCDP for the same mechanism and parameters?

2. How does Rao DP relate to hypothesis-testing interpretations of privacy? Can the geometric interpretation yield new privacy–utility trade-offs beyond composition?

3. Since GDP and Rao DP coincide for the Gaussian mechanism, is Rao DP effectively a geometric reinterpretation of GDP or does it generalize it to other mechanisms?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a new notion of differential privacy namely Rao DP

### Strengths
See detailed comments below

### Weaknesses
See detailed comments below

### Questions
This paper introduces a new definition of privacy based on the Rao distance for densities proposed in [1].

Major comments

1) The Gaussian and Laplace mechanisms are computed for the 1 dimensional manifold of distribution (with parameter fixed) with results from [1]. However it is not properly cited i.e., it is not apparent that the equations given in Sec 2.3 Example 1 have been derived in [1] when it is the case. Please refer to them like "from the result in Section 4.2 of [1]". The same is the case in Sec 3.1 (the equation just above Lemma 3.2) where a result from [1] is directly written as well as applied in Lemma 3.2. Wherever direct equations from a source paper are re-written, it is expected to cite the source clearly.

2) Apart from the post-processing proof Sec 3.2, the paper lacks novelty of theoretical contributions, although the idea of using the Rao distance for DP is interesting. The theoretical analyses could have been extended for a 2-dimensional manifold, as most of the results for 1-dimensional case has been derived in previous works already as pointed out above

3) Line 453 seems to be a misleading statement and seems to indicate a strong advantage for the Rao DP framework which is incorrect. Gaussian noise mechanisms will lead to non-zero privacy leakage since the privacy loss RV is unbounded. By changing  the notion of DP to the Rao DP framework one cannot  claim the following statement "As opposed to approximate DP, we do not have a second
parameter such as $\delta$. This is a strength of our framework since, as noted previously, this $\delta$ is the
probability of privacy leakage. Our formulation does not allow such a case" (line 453-455 of the paper). In fact most works in DP usually give a conversion from their notion of DP to the standard $(\epsilon, \delta)$ notion of DP since that is the widely accepted operational notion of DP. Gaussian mechanism will lead to non-zero privacy leakage and the fact that Rao DP does not reveal this is actually worrisome.

4) The impact of the derived results is not clear since  here differential privacy (ie Rao DP) is defined in
terms of a distance metric for densities while other notions are divergence based, hence, how will one compare across these different notions of DP?  To get a better perspective some empirical results would help. It would be useful to see an application where this notion of Rao DP helps 


minor comments
1) At certain places, the writing style of the paper is not formal. For example, in the paragraph just below Definition 2.3, the sentence "More on this shortly." is written, not referring to the exact section or subsection to point the reader towards.
2) I found typos/grammatical errors in few following places: 4th line, 2nd paragraph of Sec 1; 7th line, 3rd paragraph of Sec 1; 1st line of Definition 2.1 in Sec 2.1Please do a proof check for grammatical errors.

3) In 3rd paragraph of Sec 2.3, shouldn't the "not belongs to" symbol be used for p_1 + p_2 ?

[1] Burbea, J., Rao, C.R.: Entropy differential metric, unified approach. Journal of Multivariate Analysis 12, 575–596 (1982)

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a new definition of differential privacy (DP) from an information geometry perspective. All existing definitions of differential privacy (like standard $\epsilon$-DP, Rényi DP, etc.) are based on measuring the "difference" between a mechanism's output on adjacent datasets using a divergence (like the Kullback-Leibler (KL) divergence), which are often asymmetric. The paper proposes a new definition that measures this difference using a proper distance metric (which is symmetric). Specifically, it uses the Rao distance. The core contribution is to introduce Rao DP - a novel DP definition that uses the Rao distance as its measure of similarity. It proves RDP has crucial DP properties: post-processing privacy preservation and sequential composition of privacy budgets. Finally, it analyzes classic mechanisms: The paper determines the privacy parameters (the "privacy cost") for the two most common mechanisms, the Gaussian mechanism and the Laplace mechanism, under this new RDP framework.

### Strengths
This paper has conceptual novelty. To some extent it challenges the existing

### Weaknesses
Despite the novelty I'm not fully convinced that we should bring such a new definition into the already rich collection of potentially meaningful DP definitions. The paper correctly points out that the previous definitions do not really use metrics, but I don't think that is crucial if we aren't looking for purely rigorous mathematic definitions. All these definitions are still "symmetric" in the sense that the same constraints imposed by privacy need to hold even if we swap $D$ and $D'$. It's also unclear how the Rao metric compares to other metrics. Does it guarantee closeness of the distributions everywhere? Is it more (or less) strict when compared to previous definitions? The paper gave some discussion but it lacks depth. Finally there is also the question of what we can gain from this new proposal. The paper used it to sort of paraphrase the privacy guarantees of established privacy mechanisms. The results are not surprising. It's unclear whether it can be used to solve real problems.

### Questions
See Weakness. In addition:
1. Have you considered using other distance metrics between distributions as well? Rao metric is not the only one that makes sense from that perspective.
2. Does the new definition bring about new mechanisms designs, understanding of previously existing problems, etc.?

### Soundness
2

### Presentation
3

### Contribution
2
