# Analysis of High-order Interactions in Shapley value for Model Interpretation

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
The Shapley value is a fundamental game-theoretic framework for allocating a utility function’s output among participating players, and is commonly interpreted as the expected marginal contribution under random coalitions. However, when applied to complex functions such as deep neural networks, this expected marginal contribution implicitly aggregates higher-order interaction effects, which can obscure the true contribution of features. In this study, we derive a generalized decomposition of the Shapley value that expresses it as a sum of interaction terms of arbitrary order, making explicit how higher-order interactions are incorporated within marginal contributions. We also provide an unbiased estimator for our representation via permutation sampling, enabling practical computation. We further show that when interaction effects vary substantially across contexts, these embedded higher-order terms can lead to misleading attributions for model interpretation. Our theoretical analysis and empirical evaluations demonstrate that variance in lower-order interactions reliably signals the presence of hidden higher-order structure, providing a principled criterion for when such interactions should be explored. This interaction-based perspective clarifies when the Shapley value becomes unreliable and offers new guidance for interpreting model behavior.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an interaction representation of the Shapley value and the associated cooperative game $\nu$. This interaction representation relies on discrete derivatives $\Delta_S\nu(T)$ with $\vert S \vert \leq k$ for a chosen maximum interaction order $k$. The representation is an extension of the second-order ($k=2$) representation for the Shapley value. (Chang et al., 2025) to higher order $k>2$. The paper then observes that this representation can be viewed as an expectation over random permutations, similar to permutation sampling for the Shapley value. For each sampled permutation, the corresponding discrete derivatives are computed and then averaged. Conequently, within the computation, discrete derivatives are evaluated for multiple $T$, where the variance between these values is viewed as an indicator for higher-order interactions, e.g. the variance of the marginal contributions $\mathbb{V}_T[\Delta_i\nu(T)]$ is an indicator for interactions, since it quantifies the non-additivity of the game.
The authors then validate their approach empirically on synthetic functions, and investigate interactions in deep neural networks.

### Strengths
- The paper is well structured and the theoretical results seem sound
- Analyzing higher-order interactions to understand model decision is very important
- Decomposing the Shapley value into higher-order interactions is an interesting approach that extends the initial approach by Chang et al. (2025), and provides a useful tool for analysis.
- Investigating the variance of discrete derivatives / marginal contributions as a measure of interaction is an interesting and novel idea
- Efficient computation of Shapley interactions via extended permutation sampling is an interesting research direction.
- Analyzing average discrete derivatives as a measure of interaction is a well-known and useful technique

### Weaknesses
- This paper is very poorly embedded in the current feature interaction literature, where several important papers were not discussed:
  - The Shapley interaction index as the first measure of interaction in cooperative games extends the Shapley value to higher-order interactions. More importantly, the Shapley interaction index is an average of discrete derivatives (over T), very related to the second sum in Eq. (10) (see questions). Moreover, the Shapley interaction index also satisfies a recursive property that relates higher-order interactions to lower-order interactions, e.g. the Shapley value.
  - All interaction indices [1-5] can be viewed as average (over T) discrete derivatives, similar to the measure that appears as the second part of the sum in Eq. (10). It is crucial to understand the differences between these measures, e.g. it was shown that they all summarize higher-order Harsanyi dividends differently [Proof of Theorem 8, 5]. It seems that this measure of interaction used here could be very well related to existing interaction measures. An additional analysis in terms of the Möbius transform would be very helpful.
  - In [4] there were already two permutation sampling approaches for interaction measures proposed. In addition to the theoretical comparison of the average discrete derivatives, it seems that these approaches use similar ideas.
 - Besides sampling permutations, the method requires evaluating several discrete derivatives. An analysis of the complexity of these calculations is missing.
- Depending on the connection to existing Shapley interaction indices, the approximation method could be compared against existing methods to compute Shapley interactions [4,6]. If this interaction measure is indeed novel, then heuristics could be used to evaluate the approximation quality.

Overall, in its current state, it is difficult to judge how this approach is related to ongoing efforts of analyzing interactions. It would be great, if the authors could clarify this.

### Questions
1. What is the connection of the average discrete derivatives used in the second part of the sum in Eq. (10) as an interaction measure compared to:
- The Shapley interaction index [1]?
- The Shapley Taylor interaction index [2]?
- The n-Shapley values [3,5]? Specifically, the n-Shapley values are constructed such that the second-order interactions can be added to the first-order values to obtain the Shapley values.
- The Faithful Shapley interaction index [4]?


2. How does your expectation over discrete derivatives summarize the Harsanyi dividends?
3. How does your permutation sampling approach relate to the permutation sampling extensions used as baselines in [4]

**Literature**

[1] Grabisch, Michel, and Marc Roubens. "An axiomatic approach to the concept of interaction among players in cooperative games." International Journal of game theory 28.4 (1999): 547-565.

[2] Sundararajan, Mukund, Kedar Dhamdhere, and Ashish Agarwal. "The shapley taylor interaction index." International conference on machine learning. PMLR, 2020.

[3] Lundberg, Scott M., et al. "From local explanations to global understanding with explainable AI for trees." Nature machine intelligence 2.1 (2020): 56-67.

[4] Tsai, Che-Ping, Chih-Kuan Yeh, and Pradeep Ravikumar. "Faith-shap: The faithful shapley interaction index." Journal of Machine Learning Research 24.94 (2023): 1-42.

[5] Bordt, Sebastian, and Ulrike von Luxburg. "From Shapley values to generalized additive models and back." International Conference on Artificial Intelligence and Statistics. PMLR, 2023.

[6] Fumagalli, Fabian, et al. "SHAP-IQ: Unified approximation of any-order shapley interactions." Advances in Neural Information Processing Systems 36 (2023): 11515-11551.

[7] Muschalik, Maximilian, et al. "shapiq: Shapley interactions for machine learning." Advances in Neural Information Processing Systems 37 (2024): 130324-130357.

### Soundness
4

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents a novel way of identifying higher-order interactions to summarize model behavior and improve model explanations such as feature attributions. Therein, the paper analyzes the Shapley value and its characteristic functions (value function) and proposes a novel representation of the Shapley value and the value function in terms of higher-order interactions of a fixed order $k$. The paper then uses this representation to propose an estimator for these interactions and a scheme to systematically search for interesting coalitions/interactions. 

---

Overall I think the contribution of the work is **solid and worthy of acceptance**, once the problems with the related work and therefore rather weak empirical evaluation is addressed.

 ---
### References
The following references are used throughout the review
- [1] SPEX: https://openreview.net/pdf?id=UQpYmaBGwB
- [2] ProxySPEX: https://arxiv.org/abs/2505.17495
- [3] shapiq: https://proceedings.neurips.cc/paper_files/paper/2024/file/eb3a9313405e2d4175a5a3cfcd49999b-Paper-Datasets_and_Benchmarks_Track.pdf
- [4] FaithSHAP: https://jmlr.org/papers/v24/22-0202.html
- [5] Shapley Taylor: https://proceedings.mlr.press/v119/sundararajan20a/sundararajan20a.pdf
- [6] Efficient Shapley Interaction Index: https://proceedings.mlr.press/v206/bordt23a/bordt23a.pdf
- [7] SHAP-IQ: https://proceedings.neurips.cc/paper_files/paper/2023/file/264f2e10479c9370972847e96107db7f-Paper-Conference.pdf
- [8] SVARM-IQ https://proceedings.mlr.press/v238/kolpaczki24a/kolpaczki24a.pdf

### Strengths
- **Good Contribution:** All in all I really like the contribution of the paper! I think it's clearly presented, well based on the fundamental works and *complements* the current research stream of interaction quantification based on Shapley Interactions. However, I think this work's contribution would greatly benefit from a proper discussion and comparison of the related work, which is currently missing.
---
- **Well Written and Presented:** The writing and presentation of the paper is of a very high quality. The paper flows well and shows its results clearly. While I needed some time to wrap my head around Figure 1 and Lemma 1 the main part of the paper is very clear.
---
- **Good Case Studies:** I think the current empirical evaluation is already showing why the method is useful and can be applied to identify interesting interactions. While I think that a better comparison to the current literature on interaction quantification is needed, the quality of what's already there is quite high.

### Weaknesses
- **Poor Related Work:** The paper **basically disregards** the last three to four years of Shapley interaction research and does not acknowledge or reference to it. While I agree with the authors when they write *"Note that the term ‘interaction’ in this study indicates causal interaction to understand implicit interaction effects behind the Shapley value, not interaction index in game theory literature, which provides a generalized allocation framework for a subset of players."* that the contribution is different (see strengths) the related work on computing Shapley Interactions still cannot be neglected. The following contains a non-exhaustive list of works surprisingly missing from this submission where a comparison and or delineation from would be very important:
  - Recently the line of research around SPEX [1] (followed-up by ProxySPEX [2] ) also identifies *sparse interactions* (the paper often refers to sparse interactions) based on the Möbius representation of the value function. Those sparse Möbius interactions are then transformed into the well-known Shapley Interactions. The paper often refers to those sparse interactions and how to find them. SPEX does so too.
   - The line of research on model-agnostic estimation of Shapley Interactions summarized in the `shapiq` [3] library. In the last years a couple of methods to estimate Shapley Interactions have been proposed that sample coalitions of players, evaluate the value function on these coalitions and based on this construct estimates of different Shapley interaction indices. Missing estimators that like this paper a based on the representation described in Fujimoto et al. (2006) are SHAP-IQ [7] and SVARM-IQ [8]. Notably, the work [4] presenting the faithful interaction index presents two estimation procedures as well, one related to KernelSHAP for the faithful interactions and one permutation estimator for the Shapley interaction index (similar to your permutation-based estimator). SPEX [1] and ProxySPEX [2] are also model-agnostic estimators. Most of the estimation methods presented here are also available for direct use in the shapiq library and it seems like they even integrated the new ProxySPEX method there recently.
   - Lastly, even if it is not the focus the different interaction indices should also be referenced and delineated from this work accordingly. The Shapley interaction index is already present in the paper but not properly compared to. Its efficient generalization [6] and the Shapley Taylor index [5] is missing. Lastly the faithful interaction index [4] is also not present, which is probably the most interesting angle for this work.
---
- **Weak Empirical Evidence:** As it stands currently the empirical side of this work is rather limited and leaves much to be desired. The main experiments are rather small case studies with a very limited sample size. Given that the work did not adequately compare itself to the most important related work it is also clear that the empirical evaluation of the method is rather limited in its current form. Of course it would be important to compare the interactions identified by this work with interactions derived from the different interaction indices and their estimations. A good selection of methods would be desirable given that most of the methods seem to be readily available in the shapiq library. 
- **No Ablations:** The paper does not discuss the methods hyperparmaters nor does it show a proper analysis of its performance. I am unsure how efficient the method is in computing the interactions and how noisy its estimates are. A proper evaluation is needed. Questions that should be addressed: How does the method behave when the player size increase? What are the failure modes of the methods? What happens with no higher-order interactions and little higher order interactions (the literature [3,4,7,8] uses Sum of Unanymity Games for something like this)? What is the estimation quality of the estimator (when does it converge or not)?
- **No Code:** It's sad to see another paper not submitting the source code of their work for peer review. This greatly limits the reviewing quality and ultimately my view on this work.
---
- **Minor**: I had issues understanding Figure 1 and the central theorems for some time. I do not think Figure 1 in its current form helps all too much in explaining the core concepts since I am missing a lot of information in the infographic. I do not know how to easily improve this but maybe start of with giving the sets (R and T a good name and include some arrows what goes where or color code the formulas...)
- **Typos:** While reading I spotted some typos in the manuscript and small grammatical errors, but nothing major.

### Questions
- Q1: How does your interaction computation scheme scale? Is it quick? Is it slow? What are its direct failure modes?

### Soundness
4

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
3

### Summary
This paper proves that the Shapley value can be expressed as a weighted sum of its Shapley interaction terms, formalized through the Harsanyi dividends in game theory. The authors introduce a permutation-sampling strategy to estimate these interaction terms and provide a proof of its correctness. They further argue that features with high variance tend to contribute more through higher-order interactions, and propose using variance as a heuristic to select informative feature subsets for computing these interactions. In experiments on both vision and language models, they demonstrate that this filtering heuristic is effective: high-variance features tend to correspond to semantically meaningful regions, such as the central objects in images.

### Strengths
1. The paper is clearly written and well-structured, with experiments spanning a broad range of both vision and language models.

2. It introduces an alternative perspective on Shapley values by expressing them as a sum over interaction terms.

3. The authors propose a thoughtful feature filtering strategy based on variance, avoiding arbitrary feature selection when computing interaction terms.

4. This enables a practical “model auditing” approach, allowing users to explore and interpret predictions by focusing on high-variance feature subsets.

### Weaknesses
1. While I am familiar with SHAP and have some background in higher-order SHAP interaction terms, I am not an expert on this specific area of high-order interaction terms. One of the main limitations of the paper is the lack of a thorough discussion of related work on interaction-based SHAP explanations, which makes it difficult to assess the novelty of the proposed approach. In particular, it is unclear how the interaction terms that are analyzed here are related or differ mathematically from prior formulations such as (1) Shapley-Taylor interactions, (2) the SHAP-IQ framework, and (3) Shapley residuals. Given that there is a substantial body of work on SHAP interaction terms, the paper should better clarify how its proposed analysis relates or differs to these many existing approaches.


2. In addition to a thorough mathematical and formal comparison with existing interaction definitions, it would also be helpful to include an empirical comparison of these interaction terms to other interaction definitions to better assess the novelty and practical significance of the proposed approach.

3. The idea that Shapley values can be decomposed into a sum of higher-order interaction terms is not novel and has been previously established, for example, in Shapley-Taylor interactions derived from a Taylor series expansion.

4. It is unclear how much of the theoretical contribution is truly novel relative to existing results on Harsanyi dividends in the game theory literature, and the proofs provided appear relatively straightforward.

5. While using variance to filter important feature subsets is a reasonable strategy, the motivation behind what could be the use of computing these interactions remains somewhat unclear. A reader may still wonder: “so what?” Why is it meaningful to compute high-order interactions among high-variance features, and what practical insight does this provide? Although the results show that these subsets tend to align with the main objects in images, it is not evident what actionable or interpretive value this offers. Have you uncovered any interesting model behaviors or insights through this analysis that justify its usefulness?

6. It is unclear how the proposed estimation approach would scale to higher-order interactions beyond those evaluated in the experiments.

### Questions
1. How does your mathematical analysis differ from prior formulations of higher-order SHAP interactions, such as Shapley-Taylor, SHAP-IQ, and Shapley residuals, as well as other related notions?

2. What aspects of the connection between Harsanyi dividends and Shapley values are genuinely new? Was the decomposition of Shapley values via Harsanyi dividends already been established in the game theory literature? If this connection is already known, how does your work provide new theoretical insight?

3. In what way does the proposed mathematical formulation justify prioritizing features with high variance as a first step before computing higher-order interactions?

4. What is the practical benefit of the proposed pipeline: filtering high-variance features and then computing higher-order SHAP interactions? What downstream insights or applications does this enable?

5. How does the proposed estimation method scale to higher-order interactions? Are there guarantees or empirical evidence regarding computational efficiency or feasibility for large feature sets?

### Soundness
3

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
This paper provides a novel theoretical reinterpretation of the Shapley value as not only a measure of individual contributions but also a structured decomposition of higher-order interactions among players (features). The authors derive a generalized formulation showing that computing the Shapley value is equivalent to decomposing the characteristic function into interaction terms of arbitrary order and evenly distributing each among involved players. They further propose an unbiased permutation-based estimator for practical computation and demonstrate that the variance of low-order interactions can serve as an indicator of hidden higher-order structures. Experiments on deep neural networks (e.g., ViT, VGG, BERT) validate the theoretical claims and show how this framework enhances the interpretability of black-box models.

### Strengths
1. Novel Shapley Value Decomposition: The paper introduces a method to decompose Shapley values into higher-order interaction terms, offering a deeper understanding of feature interactions, especially in deep learning models.

2. Theoretical and Practical Insights: The paper bridges theory and practice by offering a solid foundation for higher-order interactions in Shapley value attribution, particularly for deep neural networks.

3. Empirical Validation: The method is empirically validated on real-world models (VGG, ViT), showing how analyzing higher-order interactions improves model interpretability in complex decision-making tasks.

### Weaknesses
1. Some theoretical aspects lack empirical analysis. For example, Theorem 3 proposes an estimation method for Shapley values, but does not provide a rigorous analysis of its estimation error.

2. There is a lack of comparison with previous methods (such as Harsanyi dividend). Why is the method proposed in this paper more effective at revealing higher-order interaction effects? In what situations does it offer advantages?

3. The paper mentions using variance analysis to identify higher-order interaction effects but does not explore in-depth how these results match with the specific interaction terms in the theoretical analysis.

4. The experimental section of the paper does not clearly reveal how higher-order interaction terms directly affect the inference process of neural networks. The variance alone does not clearly explain the specific role of higher-order interactions in model inference, and the paper lacks a quantitative analysis of these effects during the inference process.

### Questions
1. How does the method proposed in Theorem 3 for estimating Shapley values perform in terms of estimation error, and could the paper include a more rigorous empirical analysis to evaluate its accuracy?

2. How does the proposed method compare with previous methods like Harsanyi dividend in revealing higher-order interaction effects, and under what conditions does it offer a clear advantage?

3. Can the paper further explore how the variance analysis results match with the specific higher-order interaction terms identified in the theoretical analysis?

4. Could the paper provide a more detailed quantitative analysis of how higher-order interaction effects directly impact the neural network's inference process, beyond just variance analysis?

### Soundness
3

### Presentation
2

### Contribution
2
