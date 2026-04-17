# Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs

- Decision: Reject
- Scores: 4, 2, 8, 0

## Abstract
Model diffing, the process of comparing models' internal representations to identify their differences, is a promising approach for uncovering safety-critical behaviors in new models. However, its application has so far been primarily focused on comparing a base model with its finetune. Since new LLM releases are often novel architectures, cross-architecture methods are essential to make model diffing widely applicable. Crosscoders are one solution capable of cross-architecture model diffing but have only ever been applied to base vs finetune comparisons. We provide the first application of crosscoders to cross-architecture model diffing and introduce Dedicated Feature Crosscoders (DFCs), an architectural modification designed to better isolate features unique to one model. Using this technique, we find in an unsupervised fashion features including Chinese Communist Party alignment in Qwen3-8B and Deepseek-R1-0528-Qwen3-8B, American exceptionalism in Llama3.1-8B-Instruct, and copyright refusal in GPT-OSS-20B. Together, our results work towards establishing cross-architecture crosscoder model diffing as an effective method for identifying meaningful behavioral differences between AI models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents the first application of crosscoders for model diffing between different model architectures, highlighting interesting ideological features which differ between models in different families. The authors introduce Dedicated Feature Crosscoders, which partition the feature dictionary to enforce model-exclusive features rather than identifying these post-hoc based on feature decoder norms. They find that DFCs are more effective at identifying model-exclusive features in both a toy model, where the ground truth features are known, and in real LLMs. The exclusivity of features is validated using model-stitching to assess feature transferability between models. 

The application of crosscoders between architectures is a useful demonstration of their potential for auditing behavioural differences between models, and this work presents some helpful methods, such as the window expansion algorithm for activation alignment, to enable this. The DFC method also enables improved identification of model-exclusive features, which is quite well validated with the toy and real model setups and has applications for both cross-architecture and single-architecture crosscoders. This is generally an interesting and novel piece of work. However, the analysis is weak in several places, which causes me to weakly recommend rejection. Particularly, the 'exclusivity score' which underlies most of the results should be better validated, and the feature analysis should be more extensive beyond the cherry-picked ideological feature examples.

### Strengths
This work presents a useful case study of a novel application of crosscoders (cross-architecture model diffing). It also presents a simple and novel modification to improve their performance in identifying model-exclusive features more broadly. 

The use of the toy model with a ground truth is a compelling validation of the DFCs improved ability to extract model-exclusive features. Including results for 2 different model pairs improves validity, though more models across different sizes would be valuable (including potentially diffing models of different sizes in the same family).

The paper is generally well presented and clear, though several key aspects of the method are only included in the Appendices (see weaknesses).

### Weaknesses
The exclusivity score is an important part of the validation for DFCs, and some more details of how this is calculated should be given in the main text. Similarly, for the toy model, I would recommend including some additional information regarding how the uniform sampling of the ground truth concepts in the main text to improve reader understanding.

The exclusivity score seems poorly validated. Did you manually validate a good number of the LLM judged examples, and can you include some examples of steered behaviours and their similarity scores in the Appendices? Also, how did you select the maximum steering magnitudes and how did you ensure that the range tested was sufficient to elicit behavioural changes? (For example, you could have steered to a magnitude where 80% of responses were still coherent, and compared behaviours in these responses, rather than instructing the judge to ignore coherence degradation).

The false positive rate ‘recovering shared concepts’ seems extremely high (Figure 3, right). When applying DFCs for behavioural diffing in safety auditing, identifying shared features as model-exclusive does not necessarily seem to be a favourable trade-off. It would be useful to have some discussion of how this trade-off arises and whether the issue could be mitigated.

The analysis focuses heavily on the ideological model-exclusive features, which are clearly heavily cherry-picked. I appreciate the analysis of LLM identified important features in Appendices C.5 and C.6, but this seems weak given the claims that this technique can be generally useful for model-auditing.

Minor comment - check all of your quotation mark formatting (e.g. line 40)!

### Questions
Is there a reason why the DSF crosscoder scores are not included in Figure 4? This would be useful to see.

In Section 2.2 you describe allocating 1-5% of total features to the model-exclusive partitions, but seem to mainly include results for the 5% DFCs with only minimal results from 1% DFCs in A C.4.  Can you include some more details of how this affected results, e.g. with 1% DFC versions of Figure 4? If the 5% version gave the best results, did you try higher allocations?
Regarding the analysis of model-exclusive features - what prompt was used to ask Claude to flag important features? (How was ‘relevance’ defined?)

You say that the  “vast majority of features flagged as important relate to general concerns” and may be false positives. This seems to undermine any reliable auditing application. Have you tested any methods for identifying these (the approaches in Section 3.2.1 don't seem to address the 'recovering shared concepts' issue)? Otherwise could you include a more detailed discussion of how this method could still be useful when the features are unreliable?

To avoid focusing on cherry-picked results, can you include a selection of feature explanations for randomly selected model-exclusive features for each model, along with their exclusivity scores?

See also the questions in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a technique for model diffing using an architecture (named DFC) focusing model-exclusive concepts. The evaluation results indicate that the proposed technique may help identify model-exclusive concepts in terms of dictionary sizes. Specifically, this paper also provides empirical results of diffing two pairs of models.

### Strengths
This paper presents an alternative architecture for model diiffing.

### Weaknesses
1. The study reported in this paper is not well-motivated. It is unclear why we need to diff models of different architectures. Diffing base models with their fine-tunes may help us understand the impacts of the fine-tuning. But this need is obviously not applicable for cross-model diffing. 
2. The main part of this paper focuses on identifying model-exclusive features. First, I think identifying model-exclusive features is not specific to cross-model diffing, but also applicable to diffing base models and their fine-tunes. Second, whether the diffing should identify model-exclusive features or shared features may depend on the purpose of diffing. That is to say, we may not claim that identifying model-exclusive features is superior to identifying shared features.
3. The technical contribution is unclear. The DFC architecture is proposed in Section 2.2 (page 3) with its illustration (i.e., Figure 2) on the top of the same page. This part is too concise for readers to understand the real differences of the DFc architecture.
4. The evaluation is unsatisfactory. In my opinion, to demonstrate the superiority of DFC, we need to show that DFC achieves better results than standard crosscoders when performance same tasks. The reported evaluation focuses on only model-exclusive features, making the whole evaluation unconvincing.

### Questions
1. Why do we need to diff models of different architectures?
2. Why are model-exclusive features more important than shared features? 
3. Is DFC more useful than standard crosscoders in any real tasks?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors apply crosscoders to cross-architecture model diffing and develop a modification of crosscoders (Dedicated Feature Crosscoders) that help find features that are exclusive to one model as opposed shared features.

### Strengths
- Clear motivation: model diffing to discover unknown behaviors not covered by evaluation suites
- Novel approach, very timely and original
- Clear explanations of shortcomings of previous methods

### Weaknesses
- Section 3.2.1. includes too much detail
- The motivation for the design change of DFC's is not clear enough
- A feature is not the same as a (propensity for a certain) behavior. How do you make sure that there are no other semantically different features that the method does not catch - features and concepts are not a 1:1 match
- Section 3.3 is the most interesting but also entirely qualitative; are there any quantitative metrics you might report as well?

### Questions
- Why is the false-positive rate expected to be increased? Is this generalizable?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper studies model diffing—identifying how two trained models differ by comparing internal representations. The authors adapt Crosscoders to a new variant, Dedicated Feature Crosscoder (DFC), intended to better isolate features unique to one model when architectures differ. Crosscoders learns a shared feature space between different models.

### Strengths
Important problem: systematic methods for model diffing across different architectures are valuable.

### Weaknesses
Limited novelty w.r.t. Crosscoders, amplified with negligible gains. The method appears to be a modest modification of existing Crosscoders to encourage exclusivity. The paper does not convincingly argue why vanilla Crosscoders are fundamentally unable to isolate exclusive features, nor provide theory or diagnostics showing the failure mode that DFC fixes. Empirical gains (e.g., Fig. 3) are very small, questioning the modification further. Finally, as experiments show, existing Crosscoders work well in this setting where models have different architectures. 

Clarity and presentation issues. The exposition is not self-contained. Section 2.1 interleaves prior Crosscoder material with new contributions, obscuring what’s novel. Lines 139–142 are particularly difficult to parse. Key assumptions and the exact DFC objective are underspecified.

Inadequate baselines and analysis. The paper omits a substantial body of prior work that targets the same goal. In particular, it should compare against explanation-based techniques explicitly designed to identify differences between models with disparate architectures (e.g., [jia2022a]). Without these baselines, it’s unclear whether the proposed method offers any advantage over established approaches. 

@inproceedings{
jia2022a,
title={A Zest of {LIME}: Towards Architecture-Independent Model Distances},
author={Hengrui Jia and Hongyu Chen and Jonas Guan and Ali Shahin Shamsabadi and Nicolas Papernot},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=OUz_9TiTv9j}
}

### Questions
What is the exact gain/contribution/novelty of the proposed modification to existing Crosscoders?

Why is Crosscoders the right solution for you to choose? Why not adopt or extend explanation-based techniques explicitly designed for cross-architecture comparisons (e.g., [jia2022a])? Can you provide a principled justification (failure modes, assumptions) and empirical evidence (head-to-head baselines, ablations) supporting this choice?

Can you provide a clear, formal definition of “model-exclusive feature” and a quantitative metric used throughout?

### Soundness
1

### Presentation
1

### Contribution
1
