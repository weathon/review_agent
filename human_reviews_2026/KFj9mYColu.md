# Explaining the Inconsistency of Perturbation-Based Fidelity Metrics

- Decision: Reject
- Scores: 2, 4, 2, 4, 4

## Abstract
Saliency maps are one of the most widely used post-hoc approaches for interpreting the behavior of Deep Learning models. Yet, assessing their fidelity is difficult in the absence of ground-truth explanations. To address this, numerous fidelity metrics have been introduced. Previous studies have shown that fidelity metrics can behave inconsistently under different perturbations, and a recent work has attempted to estimate the extent of this inconsistency. However, the underlying reasons behind these observations have not been systematically explained. In this work, we revisit this problem and analyze why such inconsistencies arise. We examine several representative fidelity metrics, apply them across diverse models and datasets, and compare their behavior under multiple perturbation types. To formalize this analysis, we introduce two conformity measures that test the assumptions implicit in existing metrics. Our results show that these assumptions often break down, explaining the observed inconsistencies and calling into question the reliability of current practices. We therefore recommend careful consideration of both metric choice and perturbation design when employing fidelity evaluations in eXplainable Artificial Intelligence (XAI).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the underlying reasons behind the inconsistency of perturbation-based fidelity metrics used for evaluating saliency maps in deep learning models.  The authors formalize two core assumptions underlying existing fidelity metrics and introduce two new conformity measures, DROP(Drop in Prediction Probability) and PSim(Pixel Rank Similarity), to test those assumptions directly. Experiments are conducted using three pretrained and two adversarially trained models (InceptionV3, Xception, ResNet50) on three datasets (Imagenette, Oxford-IIIT Pets, PASCAL VOC 2007) across nine perturbation types and two perturbation schemes.

### Strengths
1. The paper clearly identifies a gap: prior work observed inconsistency or predicted it, but this work seeks to explain it. This shift from "what" to "why" is a valuable contribution.

2. This paper introduces two novel conformity measures, DROP and PSim.  DROP (Eq. 6) quantifies how often perturbations lead to output probability decreases, an interpretable operationalization of P1. PSim (Eq. 7) measures rank invariance across perturbations, directly assessing P2. Both measures provide lightweight diagnostics compared to prior supervised approaches such as FRIES.

3. This paper has an extensive experimental evaluation. It evaluates three pretrained and two adversarially trained models across three datasets and tests nine perturbation types, including inpainting and Gaussian Blur with multiple kernels.  Large-scale computation (≈75 million model predictions) demonstrates strong experimental effort and implementation reproducibility.

### Weaknesses
1. Insufficient Justification for Assumptions:
- The paper states that fidelity metrics *assume* "[P1] There is a drop in the output probability when a pixel is perturbed" (Sec 2, line 234) and formalizes this as $p_0 \ge p_i^{\phi}$ (Eq 2). This assumption seems overly strong and is not well-defended. It is plausible that perturbing (e.g., masking) a pixel that provides *negative* or *distracting* evidence for the correct class could *increase* the output probability. 
- There is no evidence to support these two assumptions. This paper also does not claim the relationship between these two assumptions and metric faithfulness. It never establishes a theoretical or empirical link between satisfying these assumptions and a fidelity metric’s ability to measure true faithfulness. 

2. The approach setting:
- In Algorithm 1, this paper calculates the metric DROP and PSim via perturbing pixel by pixel. In my opinion, it tests the robustness of the pretrained model rather than measuring the faithfulness. For one picture, only one or two pixels removed or added to the noise, the predictions are supposed to have a minor change. 
- In Eq. 5, this paper claims that the ideally Rank Biased Overlap should be one. In Eq. 7, this paper introduces different perturbations. However, previous papers demonstrate the out-of-distribution problem should be considered. For example, when a segmentation was removed in this picture, it is hard to tell if the prediction is affected by OOD problem or the faithfulness.
- This paper uses CNN-based architectures. It is better to introduce other architecures, such as transformer, LSTM, etc.

3. The presentation:
- The citations are not consistent. In line 35,36, the citation is the same as the main body, making it hard to read. In line 61, 63, 70, two other style citations are introduced. 
- Inconsistent terms. For example, in line 178, 189, there are two style **DROP**. In line 195,203, there are two style **PSim**. 
- The algorithm's readability can be improved. 
- The notation in Equation (1) ($\mathfrak{R}=\{a_{1},a_{2},a_{3},a_{4},...a_{i}\}$) and its description ("$a_{1}\rightarrow a_{i}$ are pixels sorted... a greater i denotes greater importance") is confusing. It seems to mix indices ($i$) with pixel identifiers ($a_i$). It is unclear if $i$ is the total number of pixels or an index.   

4. Limited reference:
- The references used in this paper were mainly published before 2021. In the related works, only two papers were mentioned. Some important papers are missing, such as ROAR[1], ROAD[2], F-Fidelity[3].

Reference:
- [1].  A benchmark for interpretability methods in deep neural networks.NIPS 2019.
- [2].  A Consistent and Efficient Evaluation Strategy for Feature Attributions. ICML 2022
- [3]. F-Fidelity: A Robust Framework for Faithfulness Evaluation of Explainable AI. ICLR 2025

### Questions
- Why is it assumed that perturbing *any* pixel should *drop* the output probability? Couldn't perturbing (e.g., masking) a pixel that provides *negative* evidence for the correct class *increase* the probability?
-  How would these results differ for transformer-based or self-attention models?
- Are there cases where low conformity does *not* imply low fidelity reliability? 
- How generalizable are the conclusions to non-vision modalities or structured data?
- Could conformity measures be incorporated directly into the training objective to improve interpretability?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates why perturbation-based fidelity metrics used for evaluating saliency maps, such as AOPC, Average Drop, and Faithfulness, often yield inconsistent results across perturbation types. 
Rather than proposing a new metric, the authors aim to explain the causes of these inconsistencies by formalizing the implicit assumptions behind such metrics. 
They introduce two measures: DROP (Drop in Prediction Probability) and PSim (Pixel Rank Similarity), which test whether model outputs conform to the expected behavior under perturbation. 
Experiments across multiple architectures (InceptionV3, Xception, ResNet50, and adversarial variants) and datasets (Imagenette, Oxford-IIIT Pets, PASCAL VOC 2007) reveal widespread violations of these assumptions. 
The authors conclude that fidelity evaluations should always report the perturbation type used and test conformity beforehand.

### Strengths
- The paper addresses an underexplored question: why fidelity metrics fail rather than how much they fail. This conceptual shift from measurement to explanation is valuable for the XAI community.

- The proposed DROP and PSim scores provide lightweight, interpretable diagnostics that can be computed without training new models, offering a practical tool for assessing metric reliability.

- The experiments span several CNN architectures, adversarial training, and diverse perturbation types, demonstrating the pervasiveness of inconsistency. The large-scale empirical study adds credibility to the conclusions.

### Weaknesses
- The formalization of assumptions (P1, P2) is clear but descriptive. The paper does not deeply analyze why certain perturbations, such as Gaussian blur, produce higher conformity, or what model characteristics drive inconsistency.

- While DROP and PSim are useful, they remain diagnostic tools rather than conceptual breakthroughs. Related studies, such as Schulz et al. (Restricting the Flow: Information Bottlenecks for Attribution, ICLR) and Šimić et al. (Perturbation Effect, CIKM 2022), have already examined the robustness and validation of attribution metrics under perturbations, suggesting that this work extends rather than advances the discussion.

- The study involves tens of millions of forward passes but largely confirms intuitive expectations that fidelity metrics depend on perturbation choice, with limited new explanatory mechanisms offered.

- The work is restricted to image-based saliency metrics, whereas recent studies have explored perturbation effects and attribution consistency more extensively in time-series data. This limits generalizability and leaves open whether the same inconsistencies hold in sequential or temporal domains.

### Questions
- Why is Gaussian blur the most consistent perturbation type? Does this reflect properties of convolutional inductive biases or feature smoothness? A more theoretical explanation would be valuable.

- How do DROP/PSim scores correlate with human-annotated explanation quality, if at all? Even a small user study or qualitative check could contextualize the practical meaning of “consistency.”

- Would the same inconsistency patterns appear in non-visual modalities (e.g., text models or tabular classifiers)? Extending beyond image classification could greatly increase impact.

- Given the massive computational load, can DROP/PSim be approximated efficiently (e.g., via sampling strategies or low-rank perturbation models)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Common "fidelity" metrics in explainable AI for saliency maps are inconsistent, i.e., the metrics differ based on perturbation type.
The paper proposes two simple metrics to measure properties of a model: roughly, whether the probabilities drop under perturbation (a type of monotonicity property) and whether the pixel rankings are consistent across perturbation types.
DROP metric merely determines if the probability goes down or not. 
PSIM metric simply averages pixel importance over multiple perturbation types.
The paper than examines multiple perturbation types across a range of different models and concludes that most models do not satisfy the proposed properties.

### Strengths
- The paper poses an interesting question: Are explanation methods based on Pixel Importance Rankings actually a good measure of fidelty?
- The metrics are simple to compute.

### Weaknesses
- The paper does not justify that [P1] and [P2] are required. While these seem relatively intuitive, it is not clear why these are required. Is this true just based on the definition of Pixel Importance Rank (PIR) and where is the original definition of PIR? Does the definition of PIR assume these are these "implicit" assumptions that the authors feel are appropriate? If it is not in the definition of PIR, why must these be true? If it is in the definition, please provide the best justification for them from the original paper(s).
  - Regarding PSim, it is claimed that these should have the same ranks IF "the model is consistent". What is meant by a model being consistent here? Is this formally proved? I'm not sure that a model that fails this test is inherently "inconsistent", it would just have different rankings based on different perturbation models. Again, not sure this is required to evaluate model explanations and/or the model.

- Overall, the story of the paper seems incremental. Other prior works have observed that perturbation type is important when defining PIR metrics. Also, it is unsurprising that the rankings of pixel are not the same using different perturbation types. What is perhaps a more interesting question is whether the explanation method's ranking differs significantly depending on the perturbation type. However, the paper does not answer this more important practical question.

- The paper lacks any significant theoretical or methodological advances. It proposes some simple measures for monotonicity and differences across perturbations but there is nothing particularly insightful about these metrics and the basis for [P1] and [P2] is not well justified. It is unclear if those points are actually required for the metrics to be reasonable.

- (Formatting) It looks like you used "\epsilon" instead of "\in" for summation subscripts, e.g., Eqn 6 $\phi \epsilon \mathcal{N}$.

### Questions
- How does this relate to [Wang & Wang, 2024]? In this paper, they explicitly increase the probability of the class in a deletion metric. As in, the probability doesn't go down. The perturbation is deletion and it is cumulative I think.

[Wang & Wang, 2024] Wang, Y. &amp; Wang, X.. (2024). Benchmarking Deletion Metrics with the Principled Explanations. <i>Proceedings of the 41st International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 235:51569-51595 Available from https://proceedings.mlr.press/v235/wang24br.html.

- How do you justify the set of perturbation types in DROP? This seems somewhat arbitrary. Is there a theoretically best version of this?

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
4

### Summary
This paper investigates the underlying reasons for the widely reported inconsistency of perturbation-based fidelity metrics in XAI. The authors posit that these inconsistencies stem from the violation of two fundamental assumptions inherent to such aspects: “[P1] There is a drop in the output probability when a pixel is perturbed; [P2] The magnitude of drop in output probability is proportional to the relevance of the pixel”. To formalize and test this hypothesis, the paper introduces two novel conformity measures: DROP and PSim. The authors conduct a large-scale empirical study involving five DL three datasets, nine distinct perturbation types, and two perturbation schemes (pixel-wise and segment-wise). The results demonstrate that both assumptions are frequently and significantly violated across most model-perturbation settings. And this is why perturbation-based fidelity metrics are inconsistent.

--soundness--
The paper posits that inconsistent fidelity metrics may stem from violations of assumptions [P1] and [P2] However, the proposed PSim metric and the experimental design are insufficient to substantiate these claims. 
--contribution--
•	Importance of the Question: The reliability of evaluation metrics is a critical, foundational issue in XAI. Saliency methods are widely used, but the community lacks consensus on how to evaluate them. 
•	Originality: The key originality lies in the conceptual reframing of the problem. Instead of treating inconsistency as a property of the metric, the authors identify it as a failure at the more fundamental model-perturbation interaction level.

### Strengths
•	Conceptual Insight and Originality: The paper's primary strength is its novel conceptualization of the problem. By shifting the focus from the fidelity metrics themselves to the underlying assumptions about model behavior, it provides a deeper and more fundamental explanation for the observed inconsistencies. This is a significant step forward from prior work.
•	Clarity and Presentation: The paper is exceptionally clear in its writing, structure, and presentation of results. The core ideas are communicated effectively, making the work accessible and its implications easy to grasp.

### Weaknesses
•	Random sampling: The practical implementation of the random sampling methodology introduces significant vulnerabilities that challenge the validity of the experimental conclusions. Although the underlying premise—that relative importance rankings are preserved in a subset—is correct, the approach is undermined by two fundamental limitations. First, the chosen sample size (e.g., 50 pixels out of 299 ×299, 224 ×224 or 600 ×600 pixels) is exceptionally small, raising concerns about the statistical power and significance of the findings. Second, and more critically, the method fails to address the problem of sample representativeness. An unbiased random draw from a typical image will, with high probability, yield a sample dominated by low-importance pixels with little variance in their contribution to the model's decision. This lack of diversity renders the rank-based evaluation metric unstable and incapable of meaningfully differentiating between the performance of various explanation techniques. Therefore, the reliability of the results becomes contingent on the fortuitous acquisition of a well-distributed sample, and a typical, uninformative sample would render the subsequent analysis and comparisons empirically weak.

•	The PSim score is a relative measure; it only reveals how similar the pixel rankings of two perturbation are to each other. A high PSim score between two methods does not prove that [P2] The magnitude of drop in output probability is proportional to the relevance of the pixel.

•	The experiment results may be caused by out-of-distribution data.

### Questions
•	Why conducting the analysis of adversarially trained models?
•	See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper addresses a critical issue in Explainable AI (XAI) - saliency maps (like CAM) are widely used to interpret deep learning models, but assessing their quality through fidelity metrics is unreliable. Different fidelity metrics behave inconsistently under different perturbations, making it unclear which explanations are actually trustworthy.

### Strengths
1. The introduction of conformity measures (DROP and PSim) that directly test metric assumptions is genuinely novel. This shifts from empirical observation to theoretical diagnosis.

2. Distinctive from FRIES: While using similar "foundational primitives," the authors cleverly repurpose them for diagnostic rather than predictive purposes; a supervised model is needed. This is an elegant conceptual distinction.

### Weaknesses
Unclear decision boundaries: Tables 1 and 2 show DROP and PSim scores with means and standard deviations, but there's no clear guidance on what constitutes acceptable conformity. For example, is DROP=0.504±0.131 good or bad? When should practitioners be concerned?

KDE-based cutoffs are ad-hoc: The paper mentions using KDE with cutoffs at 0.80, 0.85, 0.90, and 0.95, but the rationale for these specific thresholds is unclear. Why these values? How sensitive are conclusions to threshold choice?

Probabilistic interpretation is vague: Lines 358-365 mention that "estimated probabilities for DROP scores to be above the cutoffs...were low, and the variants of Gaussian Blur showed relatively higher probabilities than other perturbations." This is descriptive but doesn't translate to actionable guidance: Should I avoid fidelity metrics entirely? Switch perturbations? Use different metrics?

### Questions
like weakness

### Soundness
3

### Presentation
3

### Contribution
3
