# Multidimensional Uncertainty Quantification via Optimal Transport

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 2, 2

## Abstract
Most uncertainty quantification (UQ) approaches provide a single scalar value as a measure of model reliability. However, different uncertainty measures could provide complementary information on the prediction confidence. Even measures targeting the same type of uncertainty (e.g., ensemble-based and density-based measures of epistemic uncertainty) may capture different failure modes.
  We take a multidimensional view on UQ by stacking complementary UQ measures into a vector. Such vectors are assigned with Monge-Kantorovich ranks produced by an optimal-transport-based ordering method. The prediction is then deemed more uncertain than the other if it has a higher rank.
  The resulting \emph{VecUQ-OT} algorithm uses entropy-regularized optimal transport. The transport map is learned on vectors of scores from in-distribution data and, by design, applies to unseen inputs, including out-of-distribution cases, without retraining.
  Our framework supports flexible non-additive uncertainty fusion (including aleatoric and epistemic components). It yields a robust ordering for downstream tasks such as selective prediction, misclassification detection, out-of-distribution detection, and selective generation. Across synthetic, image, and text data, \emph{VecUQ-OT} shows high efficiency even when individual measures fail.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present a new ingenious way of combining different uncertainty metrics that involves writing them in a vector form, and then uncertainty is fused using  entropy-regularized optimal transport (OT).

### Strengths
The idea is interesting and new, and the experiments seem to support it.

### Weaknesses
While very creative, I find the paper to be not ready for publication yet. Naturally this is a personal opinion, and if the majority of the referees are in favor of it, I will abide.

What the paper seems to imply is that if we consider $m$ different positive uncertainty quantification measures, they give rise to an $m$-dimensional vector space, a subset of $\mathbb{R}^m_+$. This is a very profound claim that is not spelled out, nor justified in the paper. While I understand heuristically why this is appealing, still it puzzles me quite a bit: For example, could it not be the case that, since one or more uncertainty measures may be correlated, we need less than $m$ dimensions?

Other things that make me lean towards not accepting it in its current form are the writing, that is too fragmented and could be improved, and the fact that the authors say that $\mu$ is the empirical distribution of the $\mathbf{s}_i$'s. But surely this would give rise to a discrete uniform more often than not, right? This is because two uncertainty measures vectors need to be exactly equal in each component to be counted twice. Wouldn't this be a problem?

Finally, I would like to invite the authors to compare their method to imprecise probabilistic machine learning, as studied by Sale, Hüllermeier, Munadet, Chau, Caprio, Javanmardi, Desterke, Cuzzolin, and more, where the concept of "meta-uncertainty", heuristically similar to the one presented by the authors in the form of a vector of uncertainty measures, is ubiquitous. In particular, even though for different reasons, in that literature OT for uncertainty quantification has been already proposed https://arxiv.org/abs/2410.03267, published in the proceedings of ISIPTA 2025.

Overall, in my opinion this reads like a good paper that needs a few more tweaks and explanations to be ready for publication.

### Questions
See Weaknesses

### Soundness
3

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
This paper introduces VecUQ-QT, an uncertainty quantification (UQ) method that uses multiple UQ metrics to more holistically / robustly evaluate uncertainty. Given a stack of uncertainty measures, VecUQ-QT learns a mapping between an isotropic distribution (either Beta or exponential) and uncertainty vectors on in-distribution (ID) calibration data. Once the mapping is obtained, the scalar uncertainty rank can be determined and used for downstream tasks like out-of-distribution detection.

### Strengths
1. Conceptual novelty. Many existing / prior works incorporate various sources of uncertainty (e.g., epistemic, aleatoric) to more holistically capture total uncertainty. However, the general framing / idea of the proposed approach (i.e., Monge-Kantorovich ranks) for learning optimal transport between the ID data and the target distribution is quite novel and interesting.
2. Experimental validation includes a good range of datasets and downstream tasks, which adequately supports the author's theoretical claims.
3. Simplicity. The proposed method is quite simple in its methodology. It only requires a small calibration set.

### Weaknesses
1. Theoretical analysis. Authors address the convex-hull limitation by adding outer support anchors using an empirically determined scaling factor. Current analysis / discussion surrounding this is mostly heuristic and would definite for more rigorous theoretical investigation on the trade-offs of this workaround (e.g., are there any formal guarantees on OOD inputs on the extended support?)
2.  Scalability and design of uncertainty vector. The proposed approach counter-balances the weaknesses of certain metrics (e.g., 1-MSP) by leveraging the strengths of other metrics (e.g., Mahalanobis distance). However, there is little discussion regarding strategies for choosing metrics to include. This will presumably have a large impact on the quality of VecUQ-QT. Furthermore, the inclusion of additional metrics will increase the computational load of the overall pipeline---there is a trade-off there that warrants further discussion / investigation.
3. Notation. The equations and notation in Section 2 are a little dense and would greatly benefit from additional context / explanation.

### Questions
1. How sensitive is the proposed method to the size of the calibration set? How large does the ID dataset need to be for robust performance?
2. Is the method sensitive or negatively impacted in any way by including highly correlated metrics?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes stacking multiple uncertainty measures into a vector and ordering these vectors via entropy-regularized optimal transport (OT) to a simple reference distribution. The final scalar is the distance from the reference center. The method is evaluated across several tasks, including misclassification detection, OOD detection, selective prediction, and selective generation, and often stays strong even when one component is weak.

### Strengths
+ The idea of a multidimensional view on uncertainty quantification is relatively new and interesting.
+ The empirical results are consistently competitive across various tasks.

### Weaknesses
- The paper lacks a clear motivation and solid theoretical analysis. It is unclear why OT is considered and why the radial order to a reference can be used as an uncertainty score. No guarantee that the radial order preserves or improves any proper scoring rule. And there is no principled criterion for the choice of reference distribution.
- The paper claims 'our multidimensional framework supports non-additive aggregation of AU and EU', but the experiments did not explicitly support this claim. It is unclear how AU and EU are operationally measured, and non-additively aggregated here.
- The computational cost of stacking multiple uncertainty measures and the ranking algorithm is hand-waved.

### Questions
See weaknesses.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new method for uncertainty quantification using vectors of uncertainty values. These vectors can be constructed by computing the uncertainty of an input using different uncertainty measures. The resulting vector is then used to generate uncertainty values by using a rank map fit using entropic optimal transport with calibration data. Practically, this procedure is expanded by additional “outer anchor” points to enforce OOD inputs, which should have uncertainty scores outside of the calibration range, to map to targets points on the boundary. The approach is evaluated in synthetic, image, and text settings across multiple uncertainty tasks in both a qualitative and quantitive manner.

### Strengths
- The paper covers the relevant topic of uncertainty quantification.
- The proposed approach is novel and raises interesting discussion.
- The paper features a wide range of experiments.

### Weaknesses
- The paper mentions that validation data is split into calibration data and test data, with the calibration data being used to compute the transport map. However, the models used for the baseline uncertainty measures do not get access to this calibration data. This means that the comparison is not fair.
- The proposed method should be evaluated against the (simple) baseline of summing the respective uncertainty measures that are used to construct the vectors. 
- There should be a discussion of what it means to rank inputs based on such a vector of uncertainty values.

### Questions
- How do you select which uncertainty measures to use to construct the vector? It would be good to add some guidelines about when you should choose which uncertainty measures.
- Let’s say one uses vectors of AU, and EU based on the log loss / entropy, which according to an additive decomposition would sum to TU. Would the ranking of the [AU, EU] vector induced by your approach be the same as the ranking induced by using just TU?
- Is there an (intuitive) meaning behind the uncertainty values that are obtained from a vector of uncertainty values of different uncertainty measures?

### Soundness
1

### Presentation
2

### Contribution
3
