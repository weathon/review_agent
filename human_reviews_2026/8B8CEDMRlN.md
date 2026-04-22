# Rethinking Cross-Lingual Gaps From A Statistical Viewpoint

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Any piece of knowledge is usually expressed in one or a handful of natural languages on the web or in any large corpus. Large Language Models (LLMs) act as a bridge by acquiring knowledge from a source language and making it accessible when queried from target languages. Prior research has pointed to a cross-lingual gap, viz., a drop in accuracy when the knowledge is queried in a target language compared to when the query is in the source language. Existing research has rationalized divergence in latent representations in source and target languages as the source of cross-lingual gap. In this work, we take an alternative view and hypothesize that the variance of responses in the target language is the main cause of this gap. For the first time, we formalize the cross-lingual gap in terms of bias-variance decomposition. We present extensive experimental evidence which support proposed formulation and hypothesis. We then reinforce our hypothesis through multiple inference-time interventions that control the variance and reduce the cross-lingual performance gap.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The study aims to understand the cause for poor cross-lingual transfer performance from a bias-variance perspective. The study claims that variance in the responses in the target language causes the cross-lingual transfer gap. Formalize bias and variance in cross-lingual transfer from source and target language responses and provide experimental results using inference-time method to reduce variance in the target language.

### Strengths
- identifying the cause for the cross-lingual transfer gap is very relevant and interesting
- the study views the problem from a variance-bias perspective, which is novel to the best of my knowledge 
- the proposed inference-time approaches exposing the model to multilingual inputs (and asking them to explicitly translate inputs) show promising results in cross-lingual transfer performance

### Weaknesses
- The manuscript is not well structured and hard to follow. Experimental setup and evaluation methodologies are presented in the results section. The discussion of the results is very short and relevant parts of the discussion were moved to the appendix.
- The plots in Figure 6 seem misleading: the x-axis are inconsistent, as they should all range from 0.0 to 1.0. Also, there is no description in the caption as to how the displayed variance functions were computed. While there seems to be a trend, it is not as clear as the figures may suggest in (a) and the sample size in (b) is very low
- Inference-time methods TrEn-k and TTA are not well motivated and embedded in the rest of the manuscript and discussions. 
- The derived conclusions, i.e. reducing variance in the source language reduces variance in the target language, does not hold following the conclusions in Appendix I.

### Questions
- PCA plot in figure 1, b is underspecified: what representations are used to compute this pca plot?
- Why would you expect responses in different target languages to exhibit similar variance as responses in a single source language?
- To test the hypothesis that target language variance is proportional to source language variance, I'd suggest to analyze languages pairwise, instead of the aggregated figures and values presented in table 1 and figure 6.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Overall, this paper solved the problem of understanding the causes behind cross-lingual performance gaps in multilingual LLMs. This paper proposed a statistical framework based on bias–variance decomposition, hypothesizing that cross-lingual gaps arise primarily from increased variance in target languages, rather than biases. Through formal analysis and experiments on multiple benchmarks (ECLeKTic, MMLU with mixup), the authors show that ensembling and inference-time variance control significantly reduce these gaps, suggesting that knowledge itself transfers well but confidence does not.

### Strengths
1) Presents a novel and rigorous statistical perspective (bias–variance decomposition) on the cross-lingual gap problem.

2) Strong empirical validation using multiple benchmarks and large models (Gemini, GPT-5, DeepSeek).

3) Well-designed experiments with both response and input ensembling to test the hypothesis.

4) Clear contributions and discussion showing practical mitigation strategies that are inference-time only.

### Weaknesses
1) This work focuses on inference-level variance, I'm curious if this variance/bias stil exists during training.

2) The assumption that representation bias is negligible may oversimplify real-world multilingual disparities, especially for low-resource languages.

3) Experimental scope is limited to well-represented languages; findings may not generalize to unseen or underrepresented ones.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors investigate gaps in knowledge (and therefore performance) on tasks where the only difference is the language in which the query is posed in and adopt a bias-variance framework to explain the gap. 

# Method

Logits from source language modeled as RVs from a Gaussian distribution and target logit drawn from bias and variance components, linearly weighed. 
From Prop 1 and 2: 

"Reducing the radii (√variance) will make the average responses from source and
target agree more often only when there are no biases"

From Prop 3.:
"
1.When the source confidence is high, i.e., (μ0 −μ1)/pσ2s + 2 ≫ 1 then the target confidence must
also be high based on Proposition 3.
2.Since source and target confidence are related, we should see increasing agreement (or suppressed
cross-lingual gap) as confidence in source increases.
"
# Results

The authors test on ECLeKTic and MMLU (w/ language mixup)
 
To show the gap is due to variance: embedding the L2 distance between source and target languages reduces with increasing ensemble size. Similarly, authors show decreasing Chi-square distance over multiple-choice with increasing ensemble size. Similalry, the authors do a variant of ensembling i) multiple translations presented at once and ii) translated then answer; only i) improves performance.

On ECLeKTic, authors show "High confidence in source leads to high confidence in target".

### Strengths
1. The authors take a principled approach to investigate the variance hypothesis and show the results on frontier performance. 
2. Section 4.2 and Prop 3. results are strong and align with the presentation and inference around it.

### Weaknesses
1. The authors skip a lot of existing strategies (train and inference time) that people have looked at [1-3, and many others]. Discussion of the proposed framework and analyzing results from these works seems critical
2. In-line with the above comment, Section 4.1 results, especially 4.1.2, the results are a) not novel; b) novelty aside, TTA-results are extremely surprising, especially with the bigger models and attributing it failure to follow instructions doesn't seem satisfactory (unless all prompt optimizations were conducted and the models still fail to do so, which is not detailed anywhere and unlikely that is happening). 
3. There is more to the results in 4.1 - there is everything from in-context learning, few-shot examples, impact of language similarity (intermediate languages), etc., all considered in previous works have to be investigated, and similar results /observations seem important. 
4. Another missing analysis is correlation to language maps, where the distance between languages is calculated [4]. Previous works have shown these strategies to work, and this framework can help explain some of those empirical observations.
5. Analysis around scale of models - deeper analysis across model families should strengthen the findings.


[1] Kumar, Somnath, et al. "Bridging the Language Gap: Dynamic Learning Strategies for Improving Multilingual Performance in LLMs." Proceedings of the 31st International Conference on Computational Linguistics. 2025.
[2] Agrawal, Ashish Sunil, Barah Fazili, and Preethi Jyothi. "Translation errors significantly impact low-resource languages in cross-lingual learning." arXiv preprint arXiv:2402.02080 (2024).
[3] Wang, Weixuan, et al. "Bridging the language gaps in large language models with inference-time cross-lingual intervention." arXiv preprint arXiv:2410.12462 (2024).
[4] Littell, Patrick, et al. "URIEL and lang2vec: Representing languages as typological, geographical, and phylogenetic vectors." Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers. 2017.

### Questions
Please look at the weaknesses above.

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
3

### Summary
This paper studies the cross-lingual transfer by examining the output distribution. The authors argue that the source and target languages in cross-lingual transfer share the response space, and the variance in the source language transfers to the target language. As a result, if the variance in the source language is low, the cross-lingual transfer is strong.

### Strengths
1.	Cross-lingual transfer is a key feature of multilingual LMs, but it is not fully understood. Studies of this feature help the community design multilingual LMs to support more languages. 

2.	The authors study the cross-lingual transfer in a black box and connect the sampling process with cross-lingual transfer, which is interesting. 

3.	Presentation is clear.

### Weaknesses
1.	While the idea is interesting, I have some general concerns or questions:

- The language bias or the language modeling performance is not identical for all languages. This might be a confounding factor in the study as the entropy or the output variance is different for all languages. One actionable suggestion here, consider studying a high-resource language to another high-resource language, high-resource to low-resource,  low-resource to high-resource, and low-resource to high-resource.
- The experimental design for ECLeKTic is not clear. Throughout all the paper, I assume the authors attempt to analyze the variance in the logits. However, for ECLeKTic, the authors consider the embedding distance via an external embedding model. This is confusing as it does not support the main claim of this paper.

2.	The prompts in the experiments are not intuitive and clear.  My understanding here (via multiple reading rounds; correct me if necessary ),  you prompt ten times for each language and compute the variance across languages. What is the intuition of using TrEn-k and TTA baselines as baselines?

### Questions
Please refer to Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
