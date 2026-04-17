# Adaptive Logit Adjustment for Debiasing Multimodal Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Vision-Language Models (VLMs) and Large Multimodal Models (LMMs) have significantly advanced image-to-text generation tasks such as image captioning and visual question answering (VQA). 
However, these models often exhibit biases, including attribute misalignment between the generated text and the input image, or the reinforcement of harmful stereotypes.
Existing debiasing techniques primarily focus on modifying representations at the encoder or decoder level, which can degrade model performance and may be susceptible to bias reintroduction from external sources. In this work, we propose **Adaptive Logit Adjustment (ALA) for Bias Alignment and Neutralization**, a post-hoc debiasing method that operates directly on logits during autoregressive text generation. Unlike prior approaches that modify internal representations, ALA selectively adjusts token probabilities to mitigate biases without distorting essential model outputs. Our approach leverages external classifiers to measure bias misalignment between image and text, applies gradient-based importance analysis to identify bias-inducing tokens, and dynamically refines token probabilities to reduce undesired biases. 
We evaluate ALA on image captioning and various VQA tasks, demonstrating its effectiveness in mitigating bias while maintaining contextual accuracy. Notably, our approach is applicable to various multimodal architectures in a model-agnostic manner, including VLMs and LMMs, across different tasks that involve autoregressive text generation. Our results show that logit-based debiasing offers a flexible and efficient alternative to existing encoder- and embedding-centric approaches, providing a more practical solution for building fairer multimodal AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Adaptive Logit Adjustment (ALA), a post-hoc debiasing method for VLMs and LMMs. Instead of modifying encoder or decoder representations, ALA directly adjusts token-level logits during autoregressive generation. The method leverages external classifiers (for image and text) to measure bias misalignment (e.g., gender, toxicity) and performs logit correction guided by gradient-based importance scores. Experiments across image captioning and multiple VQA tasks show that ALA effectively mitigates bias while preserving model utility.

### Strengths
1. This paper suggest post-hoc debiasing method for VLMs and LMMs, which does not require additional retraining or affect internal representation which may lead to large degradation.
2. Proposed idea is simple but effectively mitigate the bias of VLMs and LMMs.
3. The paper demonstrates consistent improvements in fairness scores with minimal degradation of utility across diverse datasets.

### Weaknesses
1. Results that show the “prompt” baseline has bad fairness score is somewhat interesting. Including experiments with VLM backbones which have stronger instruction following capability, such as Qwen-2.5-VL (or Qwen-3-VL), could strengthen the proposed method.
2. While the paper claims generality to “large multimodal models,”, the evaluated VQA tasks are limited to the captioning or keyword tagging. I suggest that the authors include experiments on more diverse VQA tasks or moderate their claim.
3. Minor weaknesses about presentation
    1. Several core components are deferred to the appendix (e.g., Algorithm 1, analysis about limitation regarding classifier, or definition of evaluation metric D_mean). A brief summary of these in the main text would aid comprehension.
    2. “Baseline” in figure 4 is ambiguous. It seems to denote the original model. Explicitly stating this would help readers interpret the plots.

### Questions
Please refer to weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a method for adaptive logit adjustment such that the output generation of an LMM is less biased in the direction of protected attributes. Specifically, for each attribute, a pair of classifiers (one for image and text respectively) are trained to predict the amount of "bias" in the input image as well as the current generated text. Given a mismatch in bias between the image and text, a corrective factor is applied to the logits, and the given text is modified such that the generation aligns with the image. 

This method is evaluated on LMMs like LLava and PaliGemma w.r.t multiple datasets including COCO-captions, FACET, and SocialCounterfactuals -- which are also used in previous relevant work.  

Results are shown in terms of fairness-utility tradeoffs, i.e. "does the model retain its capabilities". Ablation studies are performed on the adjustment strength hyperparameter.

### Strengths
* This work is well written and easy to follow

* I believe this method is a sensible step in the context of previous work. All debiasing methods have an inherent tradeoff with the method they use to localize the bias (internal or logits). As long as the classifier training is reliable, this method seems reasonable. 

* I appreciated the comparison of resources needed to run each method examined here, this is important for an inference time method. 

* The datasets and models used are appropriate, but I would have appreciated more models for a method that only augments the logits e.g. qwen-vl and/or llama-3.2

### Weaknesses
* My main concern with this method is the reliance on an external classifier. Other previous inference-time methods, specifically model steering are able to more easily compute an adjustment to the target model's behavior no matter what the target attribute is. The need to train a binary classifier may be difficult depending on the deployment setting as well as the acquiring data about the attribute itself. These kinds of inference-time debiasing methods exist primarily because fine-tuning is prohibitively expensive to do well, so the requirement to train a classifier seems like a regression. 

* I think absolute counts or base rates are important here in the context of debiasing. Especially given that some models may have very different rates of producing biased text, its useful to know how much biased text is generated vs caught.

### Questions
1) How could this approach scale to multiple attributes? Presumably we may want to debias the LMM away from a large set of protected attributes, are there any issues for this method?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a debiasing technique for VLMs and LMMs called Adaptive Logit Adjustment (ALA) which adjusts token probabilities rather than operating directly on internal model states. Bias misalignment between vision and text is evaluated using classifiers, with a gradient-based method used to identify and adjust probabilities for the most relevant bias-inducing tokens. The proposed ALA method is evaluated across three VQA tasks as well as image captioning. An ablation study on the effect of the hyperparamter introduced by ALA is also conducted.

### Strengths
1. Overall the paper is clear, well-written, and easy to follow
2. To the best of my knowledge, the proposed ALA method is a novel approach to debiasing VLMs and MLLMs
3. The approach is well-motivated in that it aims to address the pitfalls of existing debiasing techniques (i.e., general performance degradation) by avoiding manipulation of internal model representations.
4. The experiments cover a decent range of tasks (image captioning + 3 VQA tasks) and datasets (MS-COCO, FACET, SocialCounterfactuals).

### Weaknesses
1. The experimental results are somewhat limited by the fact that only two models are evaluated for each task
2. Some of the experimental results seem odd and unintuitive. Why does DeAR lead to such a large increase in bias for image captioning? Why does ALA lead to large debiasing effects relative to the baseline for CLIP-CAP but not for BLIP? Additional explanation of these results would be helpful.
3. VQA-Task-3 aims to measure "core utility" of the model by asking directly for identification of gender. It seems like this task should also cover the identification of other attributes such as race, particularly because they are often described with words that have multiple ambiguous meanings (e.g., "black", "white"). It is important to ensure that probabilities for these words are not being lowered in other contexts where they do not refer to the social attribute.

### Questions
1. Lines 327-329 state that the goal of VQA-task-2 is to ensure non-toxicity across all attributes. Shouldn't the goal rather be to ensure there are not differences in the level of toxicity across groups? A model can be toxic but not biased if it is equally toxic for all groups.

### Soundness
2

### Presentation
3

### Contribution
3
