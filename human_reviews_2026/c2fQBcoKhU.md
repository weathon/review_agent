# Diagnosing Generalization Failures from Representational Geometry Markers

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 10

## Abstract
Generalization—the ability to perform well beyond the training context—is a hallmark of biological and artificial intelligence, yet anticipating unseen failures remains a central challenge. Conventional approaches often take a "bottom-up" mechanistic route by reverse-engineering interpretable features or circuits to build explanatory models. While insightful, these methods often struggle to provide the high-level, predictive signals for anticipating failure in real-world deployment. Here, we propose using a "top-down" approach to studying generalization failures inspired by medical biomarkers: identifying system-level measurements that serve as robust indicators of a model’s future performance. Rather than mapping out detailed internal mechanisms, we systematically design and test network markers to probe structure–function links, identify prognostic indicators, and validate predictions in real-world settings. In image classification, we find that task-relevant geometric properties of in-distribution (ID) object manifolds consistently forecast poor out-of-distribution (OOD) generalization. In particular, reductions in two geometric measures—effective manifold dimensionality and utility—predict weaker OOD performance across diverse architectures, optimizers, and datasets. We apply this finding to transfer learning with ImageNet-pretrained models. We consistently find that the same geometric patterns predict OOD transfer performance more reliably than ID accuracy. This work demonstrates that representational geometry can expose hidden vulnerabilities, offering more robust guidance for model selection and AI interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes to predict when models will fail to generalize especially under distribution shifts, even before deploying them. A diagnostic framework is presented that uses measurable properties of the model's internal representations. Through experiments on CIFAR-10/100 and ImageNet, the authors show that simple geometric indicators of in-distribution (ID) features, such as effective manifold dimension (Deff) and effective utility (Psi_eff), can serve as biomarkers for out-of-distribution (OOD) robustness. Applied to ImageNet-pretrained architectures, these metrics are able to identify which model weights will transfer better to new datasets like Flowers, Cars, and Places.

### Strengths
The paper addresses a real gap: predicting generalization failure before deployment.

The results hold across several architectures, optimizers, and datasets.

Although dense, the paper is well-written and the figures are quite helpful.

### Weaknesses
While the methods are well explained, the paper could present a step-by-step practical guide (an actual "diagnotic kit", as the authors suggest themselves). This would make the findings more easy for practitioners to apply when designing new architectures.

It is unclear how this framework generalizes to language or multimodal models. A short discussion about this or a pilot result would considerably strengthen the relevance of these results.

### Questions
How computationally expensive are the GLUE metrics used in this paper?

What hyperparameters (if any) that must be set for computing the GLUE metrics?

In practice, are their any thresholds of when Deff or Psi_eff would indicate "good" vs. "bad" geometry?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the question of predicting OOD performance of an image classification model having only the access to the in-distribution data. The authors draw parallels between the methodology applied in medicine to find meaningful biomarkers with significant predictive power about some phenomenon of interest. The parallel is exploited to find a set of analogous quantities for neural networks and their OOD performance based on the geometric structure of the neural network representations and the previously introduced GLUE framework. The paper argues for the approach using several experimental results suggesting the potential of the presented approach.

### Strengths
1. The topic is of great importance -- being able to reliably predict a model's OOD performance without having access to the OOD dataset is of great importance.
2. The presentation of the whole work is clear and interesting, with some room for improvement regarding the technical details (which I'll discuss later). 
3. I particularly appreciate authors' debating current trends which overly focus on studying models using tools developed in mathematics or physics. The idea to draw more inspiration from other fields, e.g. medicine is an important one to keep a healthy level of exploration within the community. That being said, I don't feel that the analogy is particularly helpful in this case (I'll elaborate on this in the weaknesses section).
4. The claims made by the authors are mostly clearly supported by the provided and clearly presented experiments; I didn't have any feeling of the authors overclaiming their contributions.

### Weaknesses
1. The main issue I see with this work is the lack of comparison to previous works studying the problem. While the authors do reference several papers in related works, they seem to miss the core works that study the same questions. For instance, [1] introduced the Tunnel Effect Hypothesis, showing that the drop of OOD performance is strongly correlated with the numerical rank of representations. Further [2] refined the Tunnel Hypothesis, showing how the Tunnel Effect (and thus OOD performance) depends on multiple factors (e.g. image resolution, augmentation strength, architecture). While in this work, the authors study only 4 learning rates and 4 weight decays. Next [3, 4] study how one can influence the model's OOD performance by regularising for or against Neural Collapse, in particular [3] shows that OOD generalisation works against OOD detection. Finally [5] further confirms the findings from [1, 2, 3] and explains the mechanisms for increased or degraded OOD performance (something that the authors of this work purposefully set aside as they advocate for finding reliable markers first and trying to explain their mechanisms later).

 2. The work is heavily based on the GLUE framework, which is only briefly introduced in the main paper, and crucial aspects such as explanation of the effective dimension, effective radius,and  participation rate, are sent to the appendix -- these are the crucial objects 
studied in this work and require a proper introduction (and justification) within the main text.

3. Limited scope of the experiments, while the experiments on image classfication are well executed I would expect more experiments focusing on different domains or different architectures especially given the fact that the experiments require training only linear classifier on top of representations, which can be easily collected from pretrained models which are freely available on repositories like HuggingFace or PyTorch.

4. The analogy with medicine "preclinical" studies is exaggerated. Testing the validity of the approach on smaller, simpler datasets and applying it on bigger, more complex datasets is what a typical workflow in ML experiments looks like, I don't feel we need to justify it with the medicine-based approach. 

5. Overly used footnotes, please use them sparingly it makes it harder to follow the text when one has to jump to footnoote too often.


[1] The Tunnel Effect: Building Data Representations in Deep Neural Networks, https://arxiv.org/abs/2305.19753

[2] What Variables Affect Out-of-Distribution Generalization in Pretrained Models?, https://arxiv.org/abs/2405.15018

[3] Controlling Neural Collapse Enhances Out-of-Distribution Detection and Transfer Learning, https://arxiv.org/pdf/2502.10691

[4] NECO: NEURAL COLLAPSE BASED OUT-OF-DISTRIBUTION DETECTION, https://arxiv.org/pdf/2310.06823 

[5] Unpacking Softmax: How Temperature Drives Representation Collapse, Compression, and Generalization, https://arxiv.org/abs/2506.01562

### Questions
1. How does the method compare to previously observed link between representations rank and ood performance? 
2. What is the complexity of the method? How it depends on the number of samples used for the estimation of the quantities? How sensitive this method is to noise in both data and labels?
3. What are the weak points of the method? When can we expect this method to be a reliable source of information and when should we expect to break? How do these areas compare to using numerical rank?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a top down, systems-level framework for understanding neural network failures inspired by similar approaches in biology and medicine. The framework is broken down into three stages: 1) identifying "biomarkers" or task-relevant measures of performance, 2) using biomarkers as prognostic tools in "clinical experiments" (i.e., to predict how markers identify failures to generalize), and 3) apply prognostic tools in a real-world application such as selecting the best pre-trained model for transfer learning. The "biomarkers" used in this paper lean heavily on prior work in representational geometry, specifically GLUE. They do a comprehensive empirical study to show that two measures from GLUE, the effective dimension and effective utility, reliably predict OOD performance better than less discriminative measures such as test accuracy. Finally, they apply their framework to the transfer learning setting they introduce earlier in the paper and show that geometric measures are more useful for predicting the best model to select.

### Strengths
1. The paper is extremely well-written, clear, and argues its central claims well. Despite relying heavily on prior work in GLUE, and therefore having little space to go over the theory, the authors do a good job of providing the necessary intuition for various concepts.
2. The empirical evaluation is comprehensive, covering many model architectures, datasets, and hyper-parameter configurations. Not only does this go a long way towards bolstering the authors claims, I believe the existence of this evaluation is useful to the community in and of itself.
3. Framing neural network interpretability work in the context of medical methodology is novel and potentially insightful.

### Weaknesses
1. While I appreciate the novelty of the medical framing, ultimately, none of the technical aspects of this framing translate into the framework. Instead of providing new insight, this perspective only seemed to confuse me. For example, the connection to "biomarkers" is far less important than emphasizing that measures of performance need to be task-relevant *as well as* descriptive of underlying mechanisms. The framing is not a deal-breaker, but I believe the paper would be stronger if it spent more time explaining what is unique about their measures w.r.t. predicting OOD generalization.
2. Though there is strong evidence in the paper for the GLUE-based framework under image classification, there is little evidence or discussion of how to extend the framework to other settings or measures. Overall, it feels as though this should have been a paper about the applications of GLUE for OOD.

### Questions
The paper's claim's can be broken down into:
1. A framework for neural network failure diagnosis based on top-down approaches.
2. A specific instance of (1) for image classification based on GLUE.

Q1. For Claim (1), little time is spent discussing how the framework should apply to settings other than image classification. Figure 1 is vague-enough that it could describe nearly any top down approach. And this is not the only top down approach for OOD performance prediction (see [1-5]). Can you explain what this framework proposes specifically to do in general?

Q2. For Claim (2), the paper focuses on GLUE, and it does a good job of explaining why the measures make sense for OOD prediction. But it isn't clear if this is the only measure that could work, or what properties of this measure make it effective. Based on the text, it seems that being task-relevant and discriminative are important, but then could [2] work? Are there alternatives?

Q3. Related to Q2, stronger evidence for Claim (1) would be to show something outside of GLUE that works and captures the essential properties of an effective biomarker.

Q4. The evaluation in Table 2 is limited to a set of models that where v1 and v2 differ in how much they train on the ID data. Could you test of a more diverse set of pre-trained models where it is less expected that the over-trained version performance worse OOD but better ID?

Q5: [6] seems to have noted the relationship between OOD performance and the measures in this paper; it is probably worth citing as related work.

[1] : https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Distribution_Shift_Inversion_for_Out-of-Distribution_Prediction_CVPR_2023_paper.html

[2] : https://proceedings.mlr.press/v162/yu22i.html

[3] : https://openaccess.thecvf.com/content/ICCV2021/html/Guillory_Predicting_With_Confidence_on_Unseen_Distributions_ICCV_2021_paper.html

[4] : https://aclanthology.org/D19-1222/

[5] : https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html

[6] : https://openreview.net/pdf?id=1ae108kHk2

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The manuscript attempts to identify markers — statistics of the learned representations — that would help predict OOD failures. Comparing multiple marker candidates across multiple datasets, it concludes that object manifold dimension and utility are prime candidates for this. In a follow-up experiment, it is demonstrated that indeed they serve this purpose well.

### Strengths
* Fantastic scientific exposition of the idea, the design, and the results. 
* The found candidates for OOD failure markers are interesting and non-trivial. Thus, it may trigger future research on the mechanism beyond their contribution to identifying OOD failures, leading to a breakthrough in our understanding of the issue.

### Weaknesses
* The manuscript is performing what is called in statistical literature "a fishing expedition" for markers. For a fixed set of datasets, had the author tested thousands of markers, they could have reported selectively on the most promising ones, thus finding markers that succeed by chance and do not generalize to other datasets. I don't suspect the authors' ethics -- but they need to take measures against such a mistake. The authors can safeguard against random markers by calculating the number of markers for which the achieved result would have happened by change, showing they are well below it (e.g. "this would have happened by change in probability 1/200, and we only tested 20 markers").
 * The main result, that a reduced manifold dimension is associated with poor generalization, is somewhat paradoxical. Previous literature cited in "related work" reports a reduction in manifold dimension across layers of a deep network, which is associated with improved classification performance. Indeed, if an object were to collapse to a point, ID performance would peak, and OOD performance would diminish. On the other hand, an uninitialized deep network where the manifold dimension is very high would perform equally bad in ID and OOD tasks. The authors should acknowledge and address this weakness openly.
 * The authors' explanation for why these specific markers are best (section 3.3) seems post-hoc and premature.

### Questions
* Can you suggest what the optimal levels of the candidate markers are? Or an educated procedure on how to operationalize their use in practice?

### Soundness
3

### Presentation
4

### Contribution
4
