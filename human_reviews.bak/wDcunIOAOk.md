# Intrinsic User-Centric Interpretability through Global Mixture of Experts

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
In human-centric settings like education or healthcare, model accuracy and model explainability are key factors for user adoption. Towards these two goals, intrinsically interpretable deep learning models have gained popularity, focusing on accurate predictions alongside faithful explanations. However, there exists a gap in the human-centeredness of these approaches, which often produce nuanced and complex explanations that are not easily actionable for downstream users. We present InterpretCC (interpretable conditional computation), a family of intrinsically interpretable neural networks at a unique point in the design space that optimizes for ease of human understanding and explanation faithfulness, while maintaining comparable performance to state-of-the-art models. InterpretCC achieves this through adaptive sparse activation of features before prediction, allowing the model to use a different, minimal set of features for each instance. We extend this idea into an interpretable, global mixture-of-experts (MoE) model that allows users to specify topics of interest, discretely separates the feature space for each data point into topical subnetworks, and adaptively and sparsely activates these topical subnetworks for prediction. We apply InterpretCC for text, time series and tabular data across several real-world datasets, demonstrating comparable performance with non-interpretable baselines and outperforming intrinsically interpretable baselines. Through a user study involving 56 teachers, InterpretCC explanations are found to have higher actionability and usefulness over other intrinsically interpretable approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduce a set of intrinsically (by design) interpretable machine learning methods, motivated by a global mixture of experts method. The goal of these methods (InterpretCC) focus on accurate predictions and aims to provide faithful explanations. The authors conduct experiments on a variety of tasks, including some user studies, and demonstrated the effectiveness of their approach.

### Strengths
1. the motivation of this work is clear and solid, and I believe intrinsic explainable machine learning should definitely be an important direction of research with great importance
2. good amount of experiments are conducted (including time series, text and tabular inputs) and the experimental results seems to be reasonable
3. the proposed methods seems to be reasonable novel (based on a global mixture of experts, with non-overlapping group/ set selections)

### Weaknesses
1. there exist evidence of over-claiming (such as Table 1, method comparison; after reading the paper, I am not convinced that the proposed methods actually achieve "faithfulness", and maybe somewhat allow full "coverage") and definition of these terms are also not clear / or there is no solid definition (although briefly mentioned in background -- Interpretability Foundations)

2. I am fully understand that explanation methods like InterpretCC are build based on expert knowledge or domain expertise (based on l231-296), are other intrinsic baseline methods adopt the same concepts/expert knowledge? would this really be a fair comparison otherwise? Also, for explainable machine learning, there is a trade-off between injecting how much domain knowledge v.s. black box methods (such as LIME or SHARP), could you add a discussion on if such domain knowledge does not exist (as this can often be the case in real-life scenarios)?

### Questions
Please refers to my questions in the "weakness section". The general landscape of explainable machine learning requires a stronger and more scientific taxonomy / definitions of these terms, the authors are thus encourage to discuss / elaborate on this with respect to their proposed methods.

I think this paper also lack a bit of theoretical analysis/justification (or at least tried to explain why the global mixture of experts method make sense and why it works for all these tasks). Could the authors try to discuss this?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposed the intrinsic explanation framework called InterpretCC to combine feature gating and group routing based on the design space by considering user-centric features to enable actionable interpretation without sacrificing prediction performance compared with traditional prediction models. It conducted user evaluation and extended intrinsic explanation models to other less popular modalities, such as time series, tabular and text.

### Strengths
The unique strength of this paper is to put user-centric design into intrinsic explanation models, aiming to address the applicability of developed models in general domains. It is nice to see that efforts of pushing model design into user-centric design.

It extends the intrinsic explanation models from visions to other modalities, especially time series, text and tabular datasets. 

It has conducted systematic evaluation of four domains with their baseline different explanation models and measured the interpretation based on several well-defined metrics. 

The user evaluation of recruiting 56 teachers to judge the usefulness of the interpretation is a great plus.

### Weaknesses
- In real world scenarios, explanations are not just a set of features, rather than the interactions of a pair of features. Do you consider to identify the interactions of features in your interpretCC framemwork
- In your user evaluation, as your method is providing local interpretation, as mentioned that interpretCC can recommend interpretation like "“This student was predicted to pass the course because and only because of the student’s regularity and video watching behavior". How can you prove such if and only if situation, because all prediction methods in interpretCC are association based, not causal based.
- can you please provide more details on how to deal with the grouping of features (it is great to see that you are using LLM to group features) for MoE, but grouping features itself can be challenge as some features might belong to two different groups, which could lead to errors in feature gating, and MoE.

### Questions
- can you think about using graph model to extend the explanability from single feature based to feature interaction based.
- in your evaluation, can you provide more details about features in EDU example, in Figure 10 in appendix, many features are similar, how these teachers could look at these features to measure whether "“This student was predicted to pass the course because and only because of the student’s regularity and video watching behavior".
- can you please provide more details on how the user predefined features (or user preferred features) are considered in feature gating and group routing.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose two new neural network architectures that are designed to be intrinsically interpretable. The first uses a gating mechanism to select a sparse set of features, and the second uses a mixture-of-experts approach to select a sparse set of interpretable feature groups. In experiments on five datasets spanning three modalities, the authors show that their method achieves comparable (or better) performance compared to black-box, non-interpretable models and two intrinsically interpretable baselines. The authors also conduct a user study involving teachers and the task of predicting student performance, and they find that the explanations produced with their method are preferred compared to baselines.

### Strengths
1. **Simple, intuitive, and novel idea for the design of intrinsically interpretable neural network architectures.** The authors present a novel idea for the design of neural network-based models that are intrinsically interpretable yet retain strong performance.
2. **Thorough experimental analysis.** The authors evaluate their method on five datasets, spanning several different domains and modalities. They analyze both the performance of the model and the quality of the explanations it produces in terms of faithfulness, sparsity, and human satisfaction.
3. **Results show that the proposed method produces useful explanations without sacrificing performance.** Across the five datasets, the proposed method performs comparably (or better) to black-box models with the same architectures in terms of predictive accuracy. The authors also show that the explanations produced by the proposed method tend to be sparse in terms of the percentage of features that are activated. The explanations also achieve high faithfulness scores on the synthetic data. Finally, the authors conduct a user study with 56 teachers, where they apply their method to the task of predicting student performance. They find that the study participants prefer the explanations produced by their method compared to baselines in terms of almost all of the criteria examined (e.g, usefulness, trustworthiness) as well as overall.
4. **Writing clarity.** The paper is clearly written and easy to follow.

### Weaknesses
1. **Missing comparison/discussion of prior work on explain-then-predict / extractive rationale methods.** There is a substantial amount of existing work on intrinsically interpretable models that involve the same basic two steps proposed in this work: (1) select a subset of the input as the “explanation”/”rationale” and (2) use a model that sees only this explanation to make the final prediction. A lot of this has been done in the NLP space; see the discussion in Section 4.5.2 in [1], and the specific methods in [2]-[5]. Since these works take the same basic approach to producing explanations, I think they should be included as baselines in the evaluation. At the very least, the authors should mention this work in the related work section and justify why their work is sufficiently different such that an experimental comparison is not needed. As a related point, the authors say in their intro that prior work on intrinsically explainable models is “rare” for “text modalities”, and they say that one of their contributions is extending intrinsic interpretability methods to “modalities and domains” that are less common for this area, such as text. I’m not entirely convinced by this point, especially since the authors did not mention any existing intrinsically interpretable approaches for text data (e.g., [2]-[5]) in their related work section.
2. **Primarily applicable to cases in which model inputs are composed of interpretable features.** The explanations produced by the proposed method take the form of a subset of model inputs (and in the group routing version, a subset with group labels). While this is human-understandable in the case in which model inputs are human-understandable (e.g., time-series features or words in a document), it is not clear that the explanations would be useful in cases where the model inputs are less structured/interpretable (e.g., pixels in an image, raw time-series data, text tokens). In many applications, the most performant models use raw/complex data as inputs as opposed to handcrafted features. Therefore, this seems to be a major limitation of the method. And it is not discussed in the paper. In addition, all experiments in the paper involve model inputs that consist of interpretable features (i.e., words, handcrafted times-series features, or image features). I would like to understand to what extent the method can be applied when the inputs are images, raw time-series, speech, text tokens, etc.
3. **Doesn’t address the possibility that explanations produced with their method are misleading/unfaithful.** Although the authors claim that their method is guaranteed to be faithful, I don’t think this is actually the case. As pointed out in prior work (e.g., [6], [7]), “select-then-predict” methods of this nature can produce misleading explanations. For example, it could be the case that the predictive model looks for superficial patterns in the selected feature set (e.g., how many features are selected) rather than uses the features as a human would expect. The authors do not address this risk in their paper.
4. **Limited analysis of impact of sparsity threshold.** In Section 6, the authors state that tuning the feature selection threshold “was key to achieving strong results.” I think the paper would be stronger if the authors included analysis of the impact of the threshold in the main text. There is some analysis in the appendix, but it appears that the experiments were only run with a single seed (there is no variance). In addition, it would be interesting to see the tradeoff between feature sparsity and performance (and how this is impacted by the choice of the threshold parameter).
5. **User study has some limitations.** Overall, the user study appears well-executed and provides evidence of the utility of the authors proposed method. However, it does have some notable limitations. The most glaring is that the authors conducted the study on only four test samples from a single dataset. This sample is small and the task is specific, so it’s hard to understand how the findings would generalize beyond the specific cases examined. Further, as the authors acknowledge, it seems like the author’s decisions around how to visualize the explanations produced by each method could impact the results. 

[1] Lyu, Qing, Marianna Apidianaki, and Chris Callison-Burch. "Towards faithful model explanation in nlp: A survey." Computational Linguistics (2024): 1-67.

[2] Jain, Sarthak, et al. "Learning to faithfully rationalize by construction." arXiv preprint arXiv:2005.00115 (2020).

[3] Bastings, Jasmijn, Wilker Aziz, and Ivan Titov. "Interpretable neural predictions with differentiable binary variables." arXiv preprint arXiv:1905.08160 (2019).

[4] Yu, Mo, et al. "Rethinking cooperative rationalization: Introspective extraction and complement control." arXiv preprint arXiv:1910.13294 (2019).

[5] Lei, Tao, Regina Barzilay, and Tommi Jaakkola. "Rationalizing neural predictions." arXiv preprint arXiv:1606.04155 (2016).

[6] Jacovi, Alon, and Yoav Goldberg. "Aligning faithful interpretations with their social attribution." Transactions of the Association for Computational Linguistics 9 (2021): 294-310.

[7] Zheng, Yiming, et al. "The irrationality of neural rationale models." arXiv preprint arXiv:2110.07550 (2021).

### Questions
My primary concern with this paper is the lack of mention and comparison to existing work on interpretability methods that follow a "select" then "predict" pipeline. Why aren't these mentioned in the paper and why aren't they included as baselines?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes InterpretCC (ICC), an intrinsic interpretable framework for prediction.

InterpretCC has two modes:

- InterpretCC Feature gating (ICC-FG): here the models rely on a sparse set of different features to make the decision for each instance.

- InterpretCC Group routing (ICC-GR): The feature space is divided into subgroups, subnetworks are trained for each group and interpretCC activates different subnetworks for a given sample.

## Model Architecture:

As described in figure 1, features (for feature gating) or group of features (group routing) are selected via a discriminator network (i.e the discriminator network predicts a mask) the masking is done using Gumbel softmax trick to keep it differentiable.

- For the feature gating: The mask is multiplied by the features and passed to a predictive model to make the final output.

- For the Group routing: The mask is used to select the group of features; each group has a subnetwork, and the final prediction is the weighted sum from the mask and the predictions of the subgroups. This can be viewed as a mixture of expert model where each subnetwork is an expert. Soft masking is used during training to efficiently train the subgroups while hard masking is used during inference. The paper investigates selecting groups in different ways, including handcrafted (user-defined) patterns and using LLMs for grouping.

## Experiments

### Data

The paper showed there approach on Time Series, Text, and Tabular data. For Tabular data the inputs are the features, for text the tokens are the features and for multivariate time series each input across a period of time is a feature i.e they apply the same mask across all time steps. They focused on classification problems.

### Results

- **Accuracy**  ICC was  compared with black box models and interpretable models like SENN and NAM across 8 datasets. Overall, InterpretCC Feature gating seems to outperform interpretable baselines. For non-text datasets ICC can also outperform DNN baselines. For breast cancer dataset, group routing appears to be extremely helpful. They also report that grouping using GPT is, on average, better than other methods, suggesting that using automated grouping methods does not mean compromising performance.

- **Explanations**

  - The paper used the synthetic dataset OpenXAI to evaluate the quality of explanations in comparison to ground-truth, all interpretable methods seem to align to groundtruth explanations in term of faithfulness, while ICC FG ranking of feature importance is higher than others.
  - The paper shows that on all datasets, ICC has high feature sparsity, which makes it more user-friendly.

- **User study**

  -  Four samples were randomly selected (i.e., four students) for prediction from each model, 56 teachers were recruited, the teachers were showed them each model’s prediction of the student’s success or failure along with its explanation. Participants were asked to compare these explanations according to five criteria: usefulness, trustworthiness, actionability, completeness and conciseness. Here SENN, NAM, ICC FR and ICC GR were compared.

  - Overall, ICC models are favored over baselines in 4 out of 5 criteria and in terms of global satisfaction from the domain experts.

### Strengths
## Originality -- high
- ICC group routing is original.
- The paper showed that grouping could be done by LLM, and it is as good as the handcrafted ones, showing that this approach can be used on a large scale without requiring extra labor.

##  Quality -- high
- The paper showed ICC can be used on different data types as they tested ICC  on 3 different data domains.
- ICC was benchmarked against strong DNN baselines and good interpretable baselines across 8 datasets.
- For interpretability, OpenXAI was used to show that the explanations produced by ICC match ground truth explanations (although since  ICC produces a mask used to select the input to the model, so this was expected).
- The paper showed that explanations produced by ICC were sparse and, therefore, user-friendly.
- Paper performed a user study that showed domain experts (here teachers) preferred explanations produced by ICC over other interpretable models.
- Overall, the experiment section and the results are very thorough.

## Clarity   -- high
- The paper is well-written and easy to follow.

### Weaknesses
## Significance -- Medium 
- ICC feature gating is almost the same as SENN feature except with a sparse mask.
- ICC group routing might be inefficient when the number of groups significantly increases, since the model complexity will increase as well.
- ICC sparsity depends on the temperature of the Gumbel Softmax, but its effect was not investigated in the paper.

### Questions
-For the group routing, is the input to the discriminator all features (from all groups) and the output just the number of groups? i.e., the discriminator doesn't exactly know which group the features belong to, correct? While each subnetwork only ever sees the features assigned to that subgroup? Please clarify...
- In line 204, you mentioned the following "feature j is activated (the associated value in the mask is non-zero) if the Gumbel Softmax output exceeds a threshold τ , a hyperparameter. This allows the model to adaptively select the number of
features based on each instance, using fewer features for simpler cases and more for complex ones" how was $\tau$ selected?, What is the effect of the changing $\tau$ on the accuracy metrics?
- How are SENN features different from InterpetCC feature gating? Is the only difference that InterpetCC has a Gumbel Softmax while SENN doesn't?

### Soundness
3

### Presentation
3

### Contribution
3
