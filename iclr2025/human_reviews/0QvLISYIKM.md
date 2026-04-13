## Human Reviewer 1

### Summary
This paper explores the use of point-wise information measures as notions of prediction confidence for neural networks. They propose three existing information measures (PMI, PSI, PVI) with associated estimation methods, and state their theoretical properties in terms of invariance to transformations, geometric properties w.r.t. decision boundary, and convergence rates of the estimators to the true measures. The results are motivated as useful or intuitive for uncertainty quantification or confidence estimation. Then two experiments on misclassification detection and selective prediction are performed, where the measures are compared to a few simple alternative notions of model confidence. Finally, their calibration property in terms of ECE is examined. The authors suggest that their point-wise information measures provide accurate and robust measures of model confidence.

### Strengths
- Approaching confidence estimation from an information-theoretic perspective provides an interesting angle, and the suggested information measures seem relevant and somewhat practical. 
- The paper clearly outlines multiple factors of motivation for the work, which help put the approach into a broader context.
- The information measures are closely examined and various theoretical properties are studied. This is also obvious from the substantial appendix which lists many properties of the information measures and possible estimation methods.

### Weaknesses
My main concerns are with respect to the claims made by the paper, obtained insights, and the experimental design and connection to confidence estimation. I will list my points of concern under each of these categories, albeit they are connected.

Claims
- The proposed measures are motivated as a post-hoc approach to confidence estimation. All the estimation methods for PMI/PSI/PVI require custom neural network modeling and training. How is this supposed to be post-hoc? For example, how am I supposed to apply this to an existing, pre-trained neural network that I treat as a black-box and do not want to fine-tune? I do not think this qualifies as post-hoc.
- The proposed measures are motivated by a “Relationship to Probabilistic Causation” yet this relationship is never mentioned again or examined, and only briefly mentioned in the limitations/future work section. In that section it is then also claimed that "the PI measures are the optimal choice of explainability" but there are no proper experiments on model explainability or causality, so this claim is not backed up in any way.
- The proposed measures are motivated by their “Direct Probability Computation”, but given their value ranges the only way to obtain probabilities is to pass them through a squashing function such as softmax, which is precisely done in the experiments. So how does the interpretation of obtained probabilities, and their associated reliability, differ in any way from just a regular softmax on logits? The associated claims on "robustness" are not examined or backed up in any way.
- The proposed measures are motivated by the need for “uncertainty quantification” and prevalent miscalibration of neural networks. Firstly, recent research has shown that modern neural network architectures such as transformers (not considered here) can in fact be quite well calibrated [1,2], and even if this was not the case, their proposals do not address a way to remedy model miscalibration but rather just suggest another confidence measure. Their claim on having “better calibrated” confidence measures is unconvincing to me due to their experimental design (see below), and thus their very strong claim on "outperforming all existing baselines for post-hoc confidence estimation" is poorly backed up. Finally, the relationship between proposed measures and meaningful uncertainty interpretations is very speculative and does not reach beyond a few broad and high-level arguments in their remarks (see below). So overall, this angle of motivation is also lacking based on their strong claims.

Insights
- I have a key question: By the invariance properties in sec 3.1 we have that PMI is best, by the geometric and convergence properties in sec 3.2. and 3.3 we have that PSI is best, but in the performed experiments we find that PVI is best. How can you reconcile this and claim that theory and experiments are in line with each other?
- To re-iterate on the connection to uncertainty quantification: it is repeatedly stated that there is a high relevance for “model uncertainty”, which equates to a notion of epistemic uncertainty. But then, Remark 1 motivates that uncertainty should be invariant to data transformations, which now relates to notions of data (aleatoric) uncertainty. Yet overall, the quantity of interest is in fact p(y|x) which is simply predictive uncertainty of the model given an input. So, it does not seem to me like there is a principled association between the information measures and actual notions of uncertainty, and the authors are not clear about what kind of uncertainties we are trying to address. Overall, the connections to uncertainty are mainly contained in the motivating introduction, and in Remark 1 and Remark 2, and are all very high-level and speculative.
- Regarding Remark 1: the quantity of interest is p(y|x), whereas data transformations are applied to features X. Since we then have that $g(X) \neq X$, I don't necessarily see an issue if $p(y| g(X)) \neq p(y|X)$ because we are conditioning on a different quantity.
- Regarding Remark 2: The provided interpretation for confidence estimation does not take into account data atypicality or OOD'ness [3]. These samples may lay far away from the decision boundary/margin but also in the tail of the data support, and thus should ideally exhibit low confidence. Also, the interpretation on confidence correlating with margin distance is only desirable for overlapping supports. If e.g. $P(X|Y=0)$ and $P(X|Y=1)$ are clearly separated (as e.g. used in Prop. 4) then we would desired maximum confidence everywhere. So, it is unclear to me how Remark 2 follows from the stated results and how directly applicable/useful these results are.
- Regarding sec 3.2: The section seems to borrow different tools from different papers to show results for the different information measures. However, the assumptions (which seem very strong), conditions of validity, and form of results are all very different, and no interpretations are given. How do they relate to each other in terms of strength and the individual components influencing them? For example, what is Prop. 4 for PMI useful for and why do we not have a similar result in regards to "sample-wise margin" as for PSI and PVI? Similar questions also apply to sec 3.3 on convergence results.
- Regarding L228: based on the wording it is unclear what the “sample-wise margin” is supposed to be. A mathematical definition seems necessary here.
- Regarding the margin correlation experiment: The results are confusing to me because in Table 1 it seems like the results between different measures are quite different (e.g., PMI has lower and more volatile correlations than PSI), yet the UMAPs virtually all look the same. How should I understand that?

Experiments
- My main concern is about the fact that their information measures are passed through a softmax function and *calibrated with temperature scaling* before benchmarking against other methods. This seems like a very biased comparison, since we observe in App D.1 that these operations significantly alter the distributions of the information measures, and improve upon the considered performance metrics. It seems unreasonable to me to *scale and calibrate* your measures beforehand, and then claim afterwards that they provide "direct probabilities" and "well-calibrated" confidence estimates. How is this a fair comparison to any baselines that are not subject to the same transformations, e.g. ML, LM? In that context, do you also apply temperature scaling to any of the other baselines such as softmax (MSP, SM)? I feel like any performance claims should rather be reported for the raw information measures instead, since it otherwise becomes unclear where any benefits stem from.
- In the experiments on confidence calibration only two simplistic baselines are considered, and they are marginally outperformed. To claim that they are "outperforming all existing baselines" seems like a very strong claim in that light. It would be more meaningful to consider other baselines for confidence estimation, including those that have been subjected to a similar approach of re-calibration as they do for their own measures (using temperature scaling), e.g. isotonic regression [4], regularization [5], or other uncertainty methods like models with variance predictor [6] etc.
- Relatedly, the results in Table 3 are often within each other's margin of error, so there are some questions on the reliability or significance of “outperforming”.
- In L401-403, what is the intuition for only working with features from the last layers? This is not clearly explained.
- Are there any clear principles guiding the choices of estimation methods for the information measures, and associated transformations (i.e., softmax or temperature scaling)? Based on the appendix it seems like purely based on hold-out performance. If so, why not consider other squashing functions or re-calibration procedures? The choices are not well documented and seem primarily motivated for their simplicity.
- I am personally missing a more detailed and meaningful interpretation of the results beyond stating what can be seen in the provided results tables.

Summary
- In conclusion, I find that the paper makes overly strong claims and motivates the work from multiple angles which are then left unexplored or never properly analyzed. The experimental design raises some questions and combined with the marginal improvements in experiments casts doubts on the practicality and usefulness of the approach. I am struggling to see the real novelty of the paper. The proposed information measures and their estimation methods are all taken from existing papers, and many of the theoretical results rely strongly on these papers as well. Is the theoretical analysis novel? Personally, it is hard for me to say since I am unfamiliar with this research domain. Is the novelty then in its application/use for confidence estimation? The experiments are unconvincing, and the connections to uncertainty do not go beyond some high-level arguments. In addition, their theoretical insights and empirical results seem somewhat contradictory on what the best information measure is supposed to be. For example, they conclude that "This superior performance is likely due to PVI being the most well-rounded metric, particularly in terms of its invariance and margin sensitivity”, even though Remarks 1, 2 and 3 on these properties rank PVI lowly. While the paper explores some interesting information-theoretic tools, their use for robust and reliable confidence estimation is substantially lacking in my opinion.

References

[1] Minderer, Matthias, et al. "Revisiting the calibration of modern neural networks." Advances in Neural Information Processing Systems 34 (2021): 15682-15694.

[2] Wang, Deng-Bao, Lei Feng, and Min-Ling Zhang. "Rethinking calibration of deep neural networks: Do not be afraid of overconfidence." Advances in Neural Information Processing Systems 34 (2021): 11809-11820.

[3] Yuksekgonul, Mert, et al. "Beyond confidence: Reliable models should also consider atypicality." Advances in Neural Information Processing Systems 36 (2024).

[4] Naeini, Mahdi Pakdaman, and Gregory F. Cooper. "Binary classifier calibration using an ensemble of near isotonic regression models." 2016 IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016.

[5] Mukhoti, Jishnu, et al. "Calibrating deep neural networks using focal loss." Advances in Neural Information Processing Systems 33 (2020): 15288-15299.

[6] Maddox, Wesley J., et al. "A simple baseline for bayesian uncertainty in deep learning." Advances in neural information processing systems 32 (2019).

### Questions
Please observe and address my questions and comments in the weakness section, such as on the strength of claims, usefulness of theoretical results for confidence estimation, experimental design, and others.

### Soundness
1

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper studies the impact of three pointwise information (PI) measures on the uncertainty quantification quality with Deep Neural Network (DNN). Through the lens of Information Theory (IT), the authors provide rigorous theoretical properties regarding the invariance, geometric, and convergence rates. Extensive experimental results confirm the theoretical arguments, and, the benchmarking suggests the pointwise V-information (PVI) outperforms the mutual information (PMI) and the sliced mutual information (PSI) in failure prediction and confidence calibration.

### Strengths
- This paper is very well-written and is clear to understand the important aspects of the algorithm.
- I like the new direction of using IT tools (e.g., mutual information, conditional entropy, etc.) to improve the reliability of DNN. I believe the benchmark results of three recent PI measures are useful for communities.
- The theoretical results are solid with clear mathematical notations, clear statements, and proof of the invariance, geometric properties, and convergence rate per each PI measure.
- The theory is also confirmed by experimental evidence, e.g., Fig.1 with the experiment on correlation to margin results in geometric properties, Table 2 with the convergence rate.
- The experimental results are extensive with several settings across different modern DNN architectures on standard benchmark datasets.

### Weaknesses
- The novelty regarding a proposed method is weak since PMI, PSI [1], and PSI [2] have been proposed before.
- The novelty in theoretical analysis is quite weak. Specifically, the invariance properties have been mentioned in Section 3 of [1] and the convergence rate has been analyzed in Section 3 of [1] and Section 4 of [2].
- The connection between theoretical results and model uncertainty is unclear to me. Details are in Question 2.
- Three PI measures are less computationally efficient than other baselines (e.g., standard Softmax) by requiring additional models and computes either $pmi(x;y)$, $psi(x;y)$, or $pvi(x\rightarrow y)$. 
- The experimental results also lack some measurements such as sharpness and predictive entropy to assess the uncertainty quality performance.

### Questions
1. What do the authors mean by a “post-hoc manner” in L-45? Is this post-hoc recalibration technique with additional hold-out calibration data to fine-tune some learnable parameters?
2. Remark 1 is vague to me. Firstly, what kinds of uncertainty are you talking about (aleatoric or epistemic)? It would be great if the authors could explain this kind of uncertainty through the lens of IT (see Eq.1 in [3]). Secondly, why when a classifier is uncertain on X, the uncertainty about $g(X)$ should ideally be the same? Can you formally explain this argument and give some examples about this?
3. In proof of Prop.6, while [4] provides estimation error bounds on the sample marginal distribution $P(X)$, why the authors can trivially apply their results on the conditional $P(X|Y)$? 
4. Could the authors please compare methods with the sharpness score [5] in Section 4.2? I think this is important because lower ECE is only a necessary condition, it is not a sufficient condition to evaluate a good uncertainty estimation with DNN.
5. L-172 mentioned that PVI uses temperature scaling, is this the main reason PVI achieves the lowest ECE in Tab.4?
6. Can PI measures extend to other kind of dataset such as text, audio, video, etc.? Is there any challenge with this extension?

References:

[1] Goldfeld et al., Sliced mutual information: A scalable measure of statistical dependence, NeurIPS, 2021.

[2] Xu et al., A theory of usable information under computational constraints, ICLR, 2020.

[3] Mukhoti et al., Deep deterministic uncertainty: A new simple baseline, CVPR, 2023.

[4] Jiang et al., Uniform Convergence Rates for Kernel Density Estimation, ICML, 2017.

[5] Kuleshov et al., Calibrated and sharp uncertainties in deep learning via density estimation, ICML, 2022.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
Uncertainty estimation becomes essential for ensuring safety AI deployment. To estimate this, various measures, commonly based on softmax probs, are employed, but they are often poorly calibrated. The authors handle this issue by utilizing pointwise information measures - PMI, PSI, PVI. They analyze several properties of measures to validate its reliability, i.e., invariance and sensitivity to margin, and conduct empirical evaluations to support its effectiveness on several datasets. Experimental results provide some findings regarding the superiority and scalability among the measures.

### Strengths
- It provides theoretical analysis regarding the properties of several PI measures, which are supported by empirical observations.
- Effectiveness of the measures are validated via two types of experiments.

### Weaknesses
- More comprehensive analysis with other uncertainty metrics are needed, such as MC Dropout, MCMC, or Laplace approximation. It also needs to compare with non-pointwise information measures, such as MINE [1].
- Empirical results are based only on small-scale datasets, such as MNIST or CIFAR-10, although this paper aims to address scalability.
- To better understanding, it would be helpful to add some visualizations, such as saliency map (with more curated examples as well as Fig.5), ROC curve, or ECE diagram.

[1] Belghazi et al., Mutual Information Neural Estimation, ICML 2018

### Questions
- In L500, why convergence rate is a crucial factor for confidence calibration?
- Why PVI is favorable for complicated dataset than other PI measures?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
The following paper conducts theoretical and experimental analysis on different pointwise information measures that serve as a metric denoting the confidence of neural network predictions. It considers three pointwise information measures: (1) pointwise mutual information (PMI), (2) pointwise $\mathcal{V}$-information (PVI), and (3) pointwise sliced mutual information (PSI). Initially, the paper introduces the formal definition of each information measure and its pointwise version, followed by analyses of each pointwise measure on (1) invariance properties, (2) geometric properties, and (3) convergence properties. There are also experiments on failure prediction and confidence calibration tasks to measure each pointwise information measure in terms of confidence estimation.

### Strengths
1. The paper is well-written and highly enjoyable to read.
2. The explanation regarding experiments in this paper goes in-depth, and I highly appreciate the authors for making Jupyter Notebook and source code available.
3. While this paper does not propose any new methods, it provides novel insights on utilizing various pointwise information measures to estimate the confidence of DNNs.

### Weaknesses
1. Experiments available in Section 4 of the paper are done on relatively small datasets and architectures. Is it possible to scale up the dataset (TinyImageNet, ImageNet) and architecture (ViT, DeIT), just like the experiments conducted by Jaeger et al., 2023 (https://arxiv.org/abs/2211.15259)? I am asking because benchmark methods used for comparison in Section 4 evaluate their method on a relatively larger scale in terms of data and architecture within their original paper.
2. More on the Questions section.

### Questions
1. In Table 2, why do $n$ for PMI and PSI differ in value? I fail to see the results to be compatible with each other if the $n$ values are not comparable.
2. I might not get 100% on the correlation between the properties of each pointwise information measure outlined in Section 3 with the results in Section 4. For example, how would you correlate the invariance property, in which PMI theoretically has an edge, with the result obtained in Section 4 for failure prediction in Section 4.1 and confidence calibration in Section 4.2?
3. For someone who is not that familiar with the following line of work, with regards to the Convergence Rate part detailed in Section 3.3, are there any possible ways to model $\mathcal{V}$ for PVI in a way such that its estimation error are comparable to $|\mathrm{pmi}(x;y) - \hat{\mathrm{pmi}}_n|$ as in PMI case?
4. Typos:
- In point 1 of \textbf{Contributions} , there should be a period between estimation and We ("estimation. We" instead of "estimation We". [Section 1]
- "from in" -> "from" [Section 4]

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 5

### Summary
This paper investigates the use of information-theoretic measures for confidence estimation in deep neural networks in a post-hoc manner. It specifically compares three measures from the prior works: pointwise mutual information (PMI), pointwise $\mathcal{V}$-information (PVI), and pointwise sliced mutual information (PSI), on their effectiveness as tools for confidence estimation. The study examines the theoretical properties of these measures in terms of invariance, correlation with margin, and convergence rate. Empirical evaluations are conducted on tasks such as misclassification detection, selective prediction, and calibration error analysis, using image classification tasks. These evaluations compare the three measures against baseline methods including softmax, margin, max logit, and negative entropy. The results indicate that PVI outperforms both PMI and PSI in terms of effectiveness as a confidence estimation tool.

### Strengths
* The paper provides a comprehensive theoretical and empirical analysis of three pointwise information measures (PMI, PVI, and PSI) on their effectiveness to be used as confidence estimation tools. It demonstrates that these measures can be applied in a post-hoc manner and do not require modifying the model architecture or retraining the network.

* Empirical evaluation covers broader scope including misclassification detection, selective prediction and calibration error analysis. These aspects are crucial for thoroughly analyzing the reliability of the confidence measures in supporting model predictions.

### Weaknesses
* Despite the use of pointwise information measures requiring the training of additional models in a post-hoc manner, both PMI and PSI perform worse than the baseline softmax measure, as evidenced by the results in Table 3 and Table 4. Additionally, the PVI estimator employs temperature scaling, a post-hoc confidence calibration method, which raises concerns about whether the improved performance of PVI is due to the PVI measure itself or the temperature scaling. The paper would benefit from further evaluation of additional benchmark methods to provide clarity on this issue, specifically: (1) PVI estimator without the temperature scaling, and (2) Softmax (SM) with temperature scaling [Guo et. al. 2017]

* Given the focus and motivation of the paper on exploring post-hoc confidence estimation tools, the empirical evaluation does not include comparisons with established post-hoc confidence calibration methods. To address this gap, it would be beneficial to compare the proposed measures with well-known methods such as Temperature Scaling (TS) [Guo et. al. 2017] and Ensemble Temperature Scaling (ETS) [Zhang et. al. 2020].


[Guo et. al. 2017] Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. "On calibration of modern neural networks." In International conference on machine learning, pp. 1321-1330. PMLR, 2017.

[Zhang et. al. 2020] Zhang, Jize, Bhavya Kailkhura, and T. Yong-Jin Han. "Mix-n-match: Ensemble and compositional methods for uncertainty calibration in deep learning." In International conference on machine learning, pp. 11117-11128. PMLR, 2020.

### Questions
* Can you please address the concerns about whether the improved performance of PVI is due to the PVI measure itself or the temperature scaling in the PVI estimator?

* What is the reason for the contradictory findings where PSI is shown to be a better confidence estimator than PVI in experiments on correlation to Margin, while PVI outperforms PSI in experiments on misclassification detection, selective prediction, and calibration analysis?

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
3