# Learning Predictive Checklists with Probabilistic Logic Programming

- Avg Score: 5.67
- Decision: Reject
- Scores: 8, 3, 6

## Abstract
Checklists have been widely recognized as effective tools for completing complex tasks in a systematic manner. Although originally intended for use in procedural tasks, their interpretability and ease of use have led to their adoption for predictive tasks as well, including in clinical settings. However, designing checklists can be challenging, often requiring expert knowledge and manual rule design based on available data. Recent work has attempted to address this issue by using machine learning to automatically generate predictive checklists from data, although these approaches have been limited to Boolean data. We propose a novel method for learning predictive checklists from diverse data modalities, such as images, time series, and text, by combining the power of dedicated deep learning architectures with the interpretability and conciseness of checklists. Our approach relies on probabilistic logic programming, a learning paradigm that enables matching the discrete nature of a checklist with continuous-valued data. We propose a regularization technique to tradeoff between the information captured in discrete concepts of continuous data and permit a tunable level of interpretability for the learned checklist concepts. We demonstrate that our method outperforms various explainable machine learning techniques on prediction tasks involving image sequences, clinical notes, and time series.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose a new framework for learning predictive checklists. The method is able to process time series, images, tabular features, etc. The use of techniques enabling sparse representations of the inputs and the use of a fairness metric lead to checklists that would have interpretable features and that promote fairness toward sensible variables.

### Strengths
-The « related works »  analysis is thorough and seems up-to-date.
-I found the experiment subsection 5.1 truly compelling. Many metrics are reported, which I think is not done often enough.
-The approach is well-explained and flexible.

### Weaknesses
**Major**

1 – Most how the points I would like to raise concern interpretability. (See also the points in the Question section on that matter.)

1.1 - Interpretability is directly impacted by the complexity of the model itself, but the fact that the algorithm is in itself a black box makes it such that understanding why the model is what it is is unreachable.

1.2 – As discussed in [1], p.17, when it comes to logical rules, the more digits there are to take into account, the less the rule is interpretable. One could argue that the checks from Figure 4 aren’t that interpretable. When it comes to the features themselves: what does it really mean to have an « sd » of FiO2 above 0.035? It is normal? Is it higher than the average? Is it higher than a certain minimum threshold? Such questions arrises with every check.

1.3 – The interpretation made of the checks from MNIST Images task is questionable. It’s been known for a long time now that saliency maps, especially the vanilla approach of looking at the gradient map, can easily lead to false conclusions, especially when those conclusions match what a person is seeking [2]. Interpretability is needed in the context where explainability (e.g. saliency maps) is not trustworthy.

1.4 – Finally, the interpretation of the features is made in an example where the relationship between the inputs of a problem and the labels is known. The procedure given in order to make sense of the feature extract wouldn’t work if this knowledge was unknown.

2.1 – It is shown that the use of the fairness regularizer works in order to minimize both FNR and FPR, but it is not discussed whether or not the constraint impacts the performances of the checklist, so there is no way to truly understand if its usage is really beneficial.

2.2 – The second contribution states « We investigate the impact of different schemes for improving the interpretability of the concepts learnt as the basis of the checklist. We employ regularization techniques to encourage the concepts to be distinct, so they can span the entire input vector and be specialized, i.e. ignore the noise in the signal and learn sparse representations. We also investigate the impact of incorporating fairness constraints into our architecture. » But since (as discussed in 2.1) there lacks evidence of the soundness of the fairness regularizer, combined with the fact that there is no evidence demonstrating that « regularization techniques encourage the concepts to be distinct, so they can span the entire input vector and be specialized » (as discussed in 1.4), or at least that the « different schemes for improving the interpretability of the concepts learned » truly are responsible for such observations, the soundness of all Contribution 2 can be questioned.

[1] : Rudin, C., Chen, C., Chen, Z., Huang, H., Semenova, L., & Zhong, C. (2021). Interpretable Machine Learning: Fundamental Principles and 10 Grand Challenges. ArXiv. /abs/2103.11251

[2] : Julius Adebayo, Justin Gilmer, Michael Muelly, Ian J. Goodfellow, Moritz Hardt, and Been Kim. 2018. Sanity Checks for Saliency Maps. In NeurIPS. 9525–9536.

**Minor**

1 – Typos. There are several of them... 

1.1 - « Figure 1: Example checklist **learnt** by our architecture. Three **of** more checks […]. »

1.2 - « Clinical practice is **an** highly stressful [...] »

1.3 - « […] programming and thus exhibits much faster **?** times and [...] » (a word is missing; « computation », « training »?)

1.4 - « […] we can write the **probabality** of query q as follows. »

1.5 - « We **additional** introduce a regularization [...] »

1.6 - « We investigate the performance **?** ProbChecklist along [...] »

1.7 - « We create **a We** briefly describe the MNIST [...] »

1.8 - « focus on the image’s upper half and **centre** »

1.9 - « we visualize **? learnt** by ProbChecklist in one of the experiments »

1.10 - « Detailed complexity analysis can be found in the **?** B. »

1.11 - « […] interpretable such as decision trees) **)** and posthoc [...] »

2 - « ProbChecklist » is named one time (the first time) before being properly introduced (the second time it is mentioned).

3 – The first citation « Learning Predictive and Interpretable Timeseries Summaries from ICU Data, volume 1, 2021. » doesn’t respect the template. It should be something like « Johnson N, Parbhoo S, Ross AS, Doshi-Velez F. Learning Predictive and Interpretable Timeseries Summaries from ICU Data. AMIA Annu Symp Proc. 2022 Feb 21;2021:581-590. PMID: 35309006; PMCID: PMC8861716. »

4 – The fourth chapter’s title should be isolated from the previous paragraph.

5 – Using a single letter (with the same calligraphy) for two different usages (‘d’, both overall dimension and error criterion) is not desirable.

6 – Constraints are not respected concerning the configuration of the table (Table 1): « number and title always appear before the table » (ICLR24 template and instructions).

7 – In Table 1: why is there no number bolded for some dataset / metric? Why are there two bolded results for MIMIC III – Accuracy?

### Questions
1 - Interpretability is not inherent to a family of models. For example, a checklist whose features aren’t interpretable or a checklist with too many « checks » to look at (as with linear models) isn’t interpretable either, for the simple knowledge is drowned in the quantity of information to manipulate. Therefore: how is it made sure that the model, concerning those two criteria, remains interpretable?

2 – It is argued that decision trees could be of lesser interest when it comes to medical applications. But, the interpretation of a model is part of how the features interact with each other in order to generate a given response. When it comes to checklists, no interaction is presented whatsoever; in the case of decision trees, it is inherent what features need to be looked at carefully given the value of some other feature. Wouldn’t that be more appropriate in the context of medical tasks?

3 – It has been briefly discussed that there is an exponential memory complexity intrinsic to the model. Was that a limitation to the experiments that have been run?  

4 - How did it impact the training time? Was that training time similar to the compared approaches? How many hyperpameters are there in total, and when compared to the baselines?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a novel method based on probabilistic logic programming to learn predictive checklists from diverse data modalities including images and time series. The proposed approach was validated using several public benchmark datasets.

### Strengths
- Originality: The paper demonstrates originality in creative combinations of existing ideas and approaches to the target problem
- Clarity: Problem formulations and related works are clearly described and cited. The paper is well-organized with most components including limitations.
- Significance: Classification performances are reported with multiple metrics and confidence intervals

### Weaknesses
1. One weakness is the results discussion using MNIST data only, which is not so intuitive in the checklist concept motivated by healthcare examples in the introduction part. And the paper has results from clinical data of PhysioNet and MIMIC III in the supplementary materials, which should be much better than the MNIST story. The necessity of using checklist, instead of other benchmark methods, on the experiment data tasks (especially non-healthcare MNIST data) is another question not explained.  
2. Model comparison in Table 1 would be better to also be illustrated in graph and plots for easy visualization.
3. Concepts learned from images seem not human-interpretable if looking at the example in Figure 3. The two concepts might still look like visual patterns that could be only differentiable by machines or algorithms. It will be hard in practice to create human-understandable checklist out of the concepts illustrated, especially in clinical domain. 
4. Concepts learned from other data modality is not illustrated in the main paper, especially time series and text, which weakens the claim of interpretation utility of the proposed algorithm in different modality.
5. Several typos in the paper, e.g. (683, 2021) on page 6, not sure whether it's citation or time series specification; and also "We create a We briefly" in line #2 on page 7. The paper needs some proofreading.

### Questions
1. Same in weakness. If the checklist concept learned from the data is not easy for human to understand and annotate, what's the potential utility of the proposed method?
2. How does the proposed method compared to other benchmark method without using checklist? a.k.a. Why using checklist to identify MNIST or predict sepsis or mortality? Is the performance better than other methods in the literature?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a method for learning checklist models for diverse data modalities, such as images, time series, and text. Checklists are a type of interpretable models that are widely used in clinical settings. A checklist model consists of a set of concepts, each of which is assigned an integer weight (always +1 in this paper). The prediction is made by summing the weights of the concepts that are present in the input and comparing the sum to a threshold $T$. Existing methods for learning checklist models are limited to tabular data. To learn checklists from these raw data modalities, the authors propose to train neural networks using a probabilistic logic programming (PLP) framework. Basically, the neural network maps the input signals to a fixed number of logits, each of which is regarded as the log probability of the presence of a concept. One can then use the logits to compute the likelihood of the positive/negative label based on the definition of the checklist model. The likelihood of the positive label is the probability of the event that at least $T$ concepts are present in the input. The model is then trained with the cross-entropy loss. The authors also propose to add several regularization terms to encourage interpretability and fairness. In the experiments, the proposed model is compared to integer programming and deep learning baselines.

### Strengths
- Originality: This work extends checklist learning to data modalities other than tabular data and combines the power of deep learning with the interpretability of checklist models. The proposed method is interesting.
- Clarity & Quality: The background and methodology are clearly presented. The paper is easy to follow.
- Significance: The proposed method seems to be a practical solution to the problem of learning checklist models from raw data modalities. Such models, if learned successfully, may be used in many real-world applications, such as clinical decision support.

### Weaknesses
- There are too many hyperparameters in the proposed method, including the weight of the regularization terms, the number of concepts, and the threshold $T$, in addition to the architecture details of the neural networks. The authors should provide some guidance on how to choose these hyperparameters.
- The learned "concept"s are hard to interpret from my point of view. The authors suggest that the concepts can be sensed by using post hoc attribution methods. However, it is well-known that the attribution methods are not perfect and may not be reliable.
- Missing related work: I believe this work should be connected to the literature on concept-based explanation and learning, such as [1], [2], and the references therein. The authors should discuss the connections and differences.
- The computational cost of the proposed loss function scales exponentially with the number of concepts.
- Typos: "LSTMS" -> "LSTMs", "TANGOS" -> "TANGOs"

[1] Amirata Ghorbani, et al. Towards Automatic Concept-based Explanations. NeurIPS 2019
[2] Pang Wei Koh, et al. Concept Bottleneck Models. ICML 2020

### Questions
- Is it possible to extend the proposed method to learn checklist models with integer weights that are not necessarily +1? This may be useful in many applications.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
