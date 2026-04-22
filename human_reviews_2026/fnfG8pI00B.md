# When Machine Learning Gets Personal: Evaluating Prediction and Explanation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 2, 6

## Abstract
In high-stakes domains like healthcare, users often expect that sharing personal information with machine learning systems will yield tangible benefits, such as more accurate diagnoses and clearer explanations of contributing factors. However, the validity of this assumption remains largely unexplored. We propose a unified framework to quantify how personalizing a model influences both prediction and explanation. We show that its impacts on prediction and explanation can diverge: a model may become more or less explainable even when prediction is unchanged. For practical settings, we study a standard hypothesis test for detecting personalization effects on demographic groups. We derive a finite-sample lower bound on its probability of error as a function of group sizes, number of personal attributes, and desired benefit from personalization. This provides actionable insights, such as which dataset characteristics are necessary to test an effect, or the maximum effect that can be tested given a dataset. We apply our framework to real-world tabular datasets using feature-attribution methods, uncovering scenarios where effects are fundamentally untestable due to the dataset statistics. Our results highlight the need for joint evaluation of prediction and explanation in personalized models and the importance of designing models and datasets with sufficient information for such evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a unified framework to evaluate how *personalization* in machine learning affects both **prediction accuracy** and **explanation quality** across groups.  
It extends the *Benefit of Personalization (BoP)* from binary classification to regression and explanation metrics (sufficiency, incomprehensiveness), proving that personalization can improve or harm interpretability even when accuracy is unchanged.  
The authors derive finite-sample limits on the reliability of hypothesis tests for personalization and apply their framework to healthcare datasets (MIMIC-III, UCI Heart), showing that personalization effects are often untestable given dataset size and attribute structure.

### Strengths
- Extends BoP theory to continuous and explanation-based settings, bridging personalization, fairness, and explainability.  
- Offers clear theoretical results (Thms. 4.1–4.3) showing prediction and explanation gains can diverge; provides a rare link between them under additive models.  
- Derives practical finite-sample error bounds that guide when personalization tests are statistically valid.  
- Empirical study (MIMIC-III) effectively illustrates the theoretical limits, highlighting cases where high empirical gains are statistically meaningless.  
- Well-motivated and relevant to fairness, interpretability, and trustworthy AI.

### Weaknesses
- The explanation method is left vague in the main text; it becomes clear only in Appendix A. Stating early that the framework is demonstrated with Integrated Gradients, DeepLIFT, and SHAP would help.  
- Heavy notation and scattered definitions (Tables 1 & 3) slow reading; concise examples or visual summaries could clarify.  
- Some theorems (e.g., 4.3) claim necessity (“always”) rather than possibility (“can”)—the text should match this (compare lines 232 vs 242)
- Too much of the intuition (e.g., Fig. 6) is tucked into appendices, reducing accessibility. You have some nice plots!
- Limited discussion of connections to fairness of recourse (implying/not implying fairness of prediction) literature, despite thematic overlap.

### Questions
1. Does the framework apply beyond feature-attribution explainers (e.g., to counterfactual or surrogate explanations)?  
2. Could the additive-model alignment in Thm. 4.3 generalize to linear or kernel settings?  
3. How robust are the lower-bound results to distributional misspecification (e.g., Laplace vs Gaussian)?  
4. Can the framework test *absence of harm* rather than *presence of benefit*?


### Comment on Table 3 (Appendix B):
Table 3 could be made more consistent with Table 1, particularly regarding the inclusion of s∖Js_{\setminus J}s∖J​ in the definition of incomprehensiveness. Clarifying this alignment would improve coherence across the theoretical and empirical sections.
It would also help readers if the caption or surrounding text explicitly stated that higher BoP values in each row indicate greater benefit for subgroup sss, reinforcing the interpretability of the metrics.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper assesses the impact of personal information on explanation fidelity, fairness and model performance claiming that impact on predictions and explanations can be different.

### Strengths
* XAI is a hot topic with many open questions and investigating personalization and faithfulness is an interesting angle
* The paper aims at a strong mathematical underpinning paired with real-world examples

### Weaknesses
* The paper should early on state that they rely on importance scores, which covers only a subset of XAI methods though arguably the most important one (as the authors claim). Also the emphasis is on tabular data, which is a further narrowing down. Both should be clear from the abstract.

* The setting is not fully clear. On the one hand, the title should contain fairness rather than prediction (pointing towards overall prediction accuracy) as fairness seems to be the focus and not prediction accuary as also stated "We propose a framework to evaluate whether this weaker fairness criterion is satisfied, both theoretically and empirically, rather than proposing corrective algorithms."   . In turn, also some of the abstract and introduction might require rephrasing. On the other hand, H1 points towards an improvement of all groups. It is better to make one clear case on either of the two.

* There is no passage in the appendix (or main part) for theorm 5.1. called like "proof of theorem 5.1" ( it might have been called proof for theorem D.3 / 4.3 (which exist) or so), but I could not find the proof. While this might be a minor issue, it essentially makes the work non-verifyable.

* Given the inaccessibility of the proof (prior comment), I focused on the intution provided, which could be expanded. Essentially, the paper appears to say that if there is colinearity among features (i.e., the personalization ones and others) , faithfulness can be jeopardized.  While this sounds reasonable, it is not clear, why that must be really the case. It seems not really specific to personalization but a rather general issue with colinearity. Do you agree?

* The case-study style evaluation is not appreciated. It seems to jeopardize generalization and also makes an assessment clear given by all the practioners advice - a reference to an Appendix is of little help here, but some kind of convincing overview would be better. The promise of providing clear datasets characteristics (mentioned in abstract) to see when personalization hurts seems to be missing - or where is it? 

Detail: use faithfulness/fidelity in the main part of the paper, not just in the Appendix, as it is a well-known term and clarifies some of our complex arguign, ie.  the sentence "this is by design:...." is obsolete if you clearly speak of fidelity.

### Questions
see above.

* Why do you need your framework and cannot just assess fairness using one of the many existing ways, ie.:
1) Add personalization features -> compute fairness metrics
2) Remove personalization features -> compute fairness metrics
3) Compare

### Soundness
3

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
The paper aims to measure how personalisation in machine learning impacts model performance and explainability. The authors show that, for the Bayes optimal classifier, accuracy and explainability are not necessarily aligned. However, when the data distribution follows an additive model, they show that there is a strong relationship between accuracy and explainability. The paper presents a statistical test to determine whether personlisation leads to improvements in accuracy and/or explainability.

### Strengths
- Paper is well written and clear for the most part
- Figure 3 and example scenario in Section 6 may be helpful for practitioners who might use the framework

### Weaknesses
**Weak Problem Statement**

There are two facets here:
1. Does this actually occur in practice?
2. If yes, what are the ramifications?

The authors don't really address these questions. I'd imagine the most salient case here is when we see a benefit in predictive accuracy but not in explainability (which kind of seems to be the case in Table 2?). For example, Monteiro Paes et al. (2022), demonstrate that personalisation may lead to harm on the second page (Table 1). Even still, we need to **infer** why a decrease in incomprehensiveness (or sufficiency) is undesirable. The authors should explicitly discuss the harm here.

I know that the authors state that the BoP framework is metric agnostic, but they should note that they have chosen these metrics to use throughout the paper, including in theoretical results.

The main motivation in this paper is:
> A common intuition is that if personalization improves prediction, it must also improve explanations (Del Giudice, 2024) (lines 189-190).

However, this doesn't seem to be the point of Del Giudice (2024). "Explanation quality" in this paper is different. Del Giudice is talking about the **model's** ability to represent real-world phenomena:
> The cost is that maximizing the predictive accuracy of a model tends to sacrifice its ability to represent the underlying phenomenon in an accurate and interpretable fashion (page 24).

This differs from the explanation metrics one might use in XAI, which aims to measure whether explanations faithfully explain **the model** (not nature).

**Results**
> Insight: The choice of improvement threshold $\varepsilon$ is key.

The paragraph here is just repeating the theoretical result in words; its quite trivial. The authors need to elaborate on this further.

Moreover, I can't seem to find the results that accompany the first and last insight.

**Overall Takeaway**

The authors conclude that there are limits to testing whether there is a benefit to personlisation. However, Monteiro Paes et al. have already demonstrated this in their work. Instead, if the authors have shown a tighter bound, they should frame the takeaway in light of those results.

### Questions
- Is it easy to construct distributions that show Theorem 4.1 and 4.2? I would be interested in how often this arises in practice.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper discusses personalization and explainability of machine learning models and prove the relationship between these two. The paper then discusses hypothesis testing, lower bound, practical aspects of the theory and concludes with experiments.

### Strengths
I am not really an expert in this field so just offering my best evaluation here. 

The paper is well written and provides important insights on the personalization and explainability. The sections on their relationship are well-thought and the theoretical aspects are sound.

### Weaknesses
I am not familiar with the relevant literature but the paper is quite original and contains insightful theories.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3
