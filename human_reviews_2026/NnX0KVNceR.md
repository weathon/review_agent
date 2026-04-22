# Generalized Pref-SHAP to Explain Preference Functions

- Avg Score: 2.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 0, 6, 0

## Abstract
We address the problem of feature attribution for skew-symmetric preference functions in dueling data settings, using the cooperative game-theoretic concept of \textit{Shapley values}. Building on Pref-SHAP[\cite{hu2022explaining}], we propose \textit{Generalized Pref-SHAP}, a framework that extends its applicability to a broader class of preference functions. Our method leverages a simple neural network to model arbitrary feature mappings while exploiting the canonical block structure inherent to skew-symmetric functions, enabling more meaningful explanations. Additionally, we explore foundational questions about Pref-SHAP, including its relationship with the block decomposition structure of skew-symmetric generalized preference function (GPM)[\cite{hu2022explaining}]. We perform experiments on a range of synthetic datasets to demonstrate the effectiveness and efficiency of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Generalized Pref-SHAP (GPref-SHAP), an extension of the Pref-SHAP framework designed to explain skew-symmetric preference function. The authors identify a key limitation in the original Pref-SHAP: it fails to respect the inherent "block structure" of certain preference models when input features are statistically correlated. This can cause feature attributions to "leak" across conceptually separate feature blocks, potentially producing misleading explanations.

To address this, GPref-SHAP proposes an explanation-aware modeling approach. The method uses a neural network to explicitly learn a feature mapping, $\phi$, from the original inputs. This learned representation is then fed into a fixed architecture that computes the preference score as a sum of decomposed, block-wise interactions

By enforcing this functional decomposition, the model is claimed to become interpretable by design, ensuring the resulting explanations align with its structure. Experiments on synthetic data claim that GPref-SHAP recovers ground-truth feature importance more accurately than the original Pref-SHAP and performs better on sanity checks involving inactive features.

### Strengths
The paper makes a good point on the independence assumption Pref-SHAP makes and how certain pathologies arise during computation of Shapley values when this is violated. The arguments for this is somewhat well presented.

### Weaknesses
1. The proposed method immediately proposes neural network and the additional integrated gradient approach to get interpretable features from the block format. What are the computational overhead for this, Shapley values are generally expensive to compute so adding complexity and overhead must be done with great care. It would have been more convincing to introduce an example where Pref-SHAP explicitly breaks due to this correlation, and to quantify at what degree of multicollinearity Pref-SHAP breaks. 
2. In one of the experiments the proposed method seems to hamper predictive performance, how does improved explanations trade-off against predictive performance - i.e. when is it better to have slightly wrong explanations for a model that predicts very well or accurate explanations for a model that predicts wrongly. 
3. The exposition of the paper needs work, the plots are not very well formatted and unnecessarily big. 
4. Some more real-life experiments would strengthen the paper. 
5. In the Pokemon experiment, Pref-SHAP also finds speed to be the most important feature similar to GPref-SHAP. Is there case here where GPref-SHAP finds insights Pref-SHAP is unable to derive?

### Questions
1. When exactly does Pref-SHAP break? Can you provide a minimal empirical example that demonstrates using data exhibiting multicollinearity? 
2. Do we really need a neural network to satisfy the block requirement? Can't we just do 2-SLS or remove highly correlated features and see if things improve?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper investigates an extension to PrefSHAP by considering a more general class of preference functions. The authors provide some results on the (extended) value function in this setting (Proposition 1), which uses feature independence or a "block structure". The authors then provide an example, and study theoretical properties in a setting with two features. Moreover, an algorithm is presented, and some empirical results are presented.

### Strengths
- Investigating more general forms of preference learning with Shapley values could possibly be interesting. However, an efficient computation of such Shapley values would be desirable by exploiting certain properties of this novel value function. I am doubtful, if this extensions will yield such insights.

### Weaknesses
In my view, this paper, in its current form, should not have been submitted at any conference, and is far from most scientific standards: There are obvious formatting issues, e.g. typos, citations, exceeded margins, no clear structure, figures and descriptions are chaotic. Moreover, the contribution is very unclear. Block structures are not introduced well, the purpose of the algorithm is not clear at all. I did not understand any central part of the contribution, e.g. Section 3 discusses the Block Pattern, what should that be? I did not understand the example and its purpose. The properties examined in Section 4 with two features are very artificial, and I still did not understand the insight. The experiments use a single synthetic dataset with 4 (!) features. There might be some interesting insights in this method, but as it is being presented now, it is not understandable, and clearly not ready for being accepted at this conference.

### Questions
I do not think my questions can be sufficiently addressed by the authors in the rebuttal, but some were stated under "weaknesses".

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Generalized Pref‑SHAP, an extension of Pref‑SHAP for explaining pairwise preference functions with nonlinear feature mappings. It learns structured feature representations via neural networks while preserving the canonical skew‑symmetric block structure. The method improves interpretability, decomposability, and consistency across correlated features in preference learning models.

### Strengths
- The authors successfully preserve block‑wise interpretability in skew‑symmetric preference models.
- This paper demonstrates that the framework supports nonlinear and learned feature mappings, making it more generalizable.
- The proposed method achieves higher attribution accuracy and robustness compared with existing approaches.

### Weaknesses
- The proposed method requires higher computational cost due to neural network training and multiple KRR models.
- This paper relies on independence assumptions for certain theoretical guarantees, limiting universality.
- The authors provide limited real‑world evaluation, focusing primarily on synthetic datasets.

### Questions
- The proposed method requires higher computational cost due to neural network training and multiple KRR models.
- This paper relies on independence assumptions for certain theoretical guarantees, limiting universality.
- The authors provide limited real‑world evaluation, focusing primarily on synthetic datasets.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper proposes extending Pref-SHAP (Hu et al., NeurIPS 2022), a method for explaining preference learning using Shapley values. Theoretical analysis gives a single proposition on the block decomposition of conditional Pref-SHAP under feature independence. Experiments with two synthetic tabular datasets compare the proposed Generalized Pref-SHAP to Pref-SHAP.

### Strengths
Unfortunately, it is challenging to find any.

### Weaknesses
This work resembles more a preliminary workshop contribution rather than a complete conference publication:
1. The motivation and significance of this research are weak. There are no impactful applications for the method. This is evident from the fact that experiments are conducted primarily with synthetic data. A single "real-world" example with a "Pokémon" dataset is shown in Appendix E, Figure 6. Furthermore, the paper does not reference any emerging applications.
2. Discussion of related literature is limited to only 13 references. Why is this research important? The introduction provides no context for studying such a method.
3. Presentation is subpar (see feedback below).

### Questions
1. Why is Appendix G empty?
2. Can this research be motivated by reinforcement learning from human feedback and preference learning for LLMs?

Feedback: 
- All figures can be much smaller, saving space for actual research content. You should not write "Section 3.1: This discussion and the related literature are included in the appendix due to space constraint."
- The critical example with "real-world" data should be included in the main text.
- Figure 1 should be a Table. It is also too large in width.
- Equation (11) shouldn't exceed the margin.
- Why are citations written without parentheses like "In the Pref-SHAP framework Hu et al. (2022)," instead of "In the Pref-SHAP framework (Hu et al., 2022),"?
- L315: typo in "kernelChau et al. (2022a)"
- Figure 6 should list feature names next to bars, not in the caption.

In general, you can mimic the Pref-SHAP (Hu et al., NeurIPS 2022) paper to directly improve the presentation of this submission.

### Soundness
1

### Presentation
1

### Contribution
1
