# Gazelle: A Multimodal Learning System Robust to Missing Modalities

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 3

## Abstract
Typical multimodal classification systems exhibit deteriorated performance if one or more modalities are missing at test time. In this work, we propose a robust multimodal classification system, Gazelle, which is less susceptible to missing modalities.  It consists of a single-branch network sharing weights across multiple modalities to learn intermodal representations.  It introduces a novel training scheme featuring a modality switch mechanism over input embeddings extracted using modality-specific networks to maximise performance as well as robustness to missing modalities. Extensive experiments are performed on four challenging datasets including textual-visual (UPMC Food-$101$, Hateful Memes, Ferramenta) and audio-visual modalities (VoxCeleb$1$).   Gazelle achieved superior performance when all modalities are present  as well as in the case of missing modalities compared to the existing state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a multi-modal training method to enhance the robustness of multi-modal models to modality missing. Furthermore, since this paper utilizes CLIP embeddings, its absolute performance is significantly superior to previous methods in some datasets.

### Strengths
1, Using CLIP embeddings has led to a significant improvement in the overall performance of the multi-modal model.

2, The proposed training method indeed makes the multi-modal model more robust in cases of missing modalities compared to conventional models.

However, I believe that many of the experiments in this paper are, in fact, unfair in their comparisons. I have provided a detailed explanation of this in the "Weaknesses" section.

### Weaknesses
1, The reason this multi-modal model can achieve SOTA results on several datasets is fundamentally due to the use of embeddings from pre-trained models (such as CLIP embeddings), rather than the inherent superiority of the proposed training method itself. If you want to demonstrate how good your proposed training method is, different training methods should be applied with the same backbones. For the reasons mentioned above, I find the significance of Tables 2, 3, 4, and 5 to be quite limited because the performance improvement is not a result of your paper's new method but rather the utilization of pre-trained models from previous works.

2, In Table 6, when comparing the proposed method with Ma et al., I believe there is a significant misconception here. You used the CLIP model pre-trained on a large-scale text-image dataset by OpenAI, while Ma et al. used the ViLT backbone. The absolute performance of the model in this paper is better than Ma et al., which may be due to the superiority of CLIP over ViLT, rather than the training method proposed in this paper is better than Ma et al.'s method. **A more accurate comparison should be based on the proportion of performance degradation.**   Specifically, when 10% of the text is missing, Gazelle shows a decrease of (94.6-93.2)/(94.6-81.7)=10.85%, while Ma et al. exhibits a decrease of (92.0-90.5)/(92.0-71.5)=7.32%. From this perspective, when 10% of the text is absent, Ma et al. experience a relatively smaller proportion of decrease. Your higher absolute performance is simply due to the use of stronger pre-trained model embeddings, not because your proposed method is superior.

3, The results in Table 6 for Hateful meme, where having 50% text performs better than having 70% text, and where 0% text and 10% text yield the same performance, are indeed puzzling. This could suggest that the method proposed in this paper may not make optimal use of the available text data.

4, The method proposed in this paper requires that the sizes of features from different modalities remain consistent, which actually limits the flexibility of the entire model. For example, it may prevent the combination of BERT-Large and ViT-B.

### Questions
See weaknesses.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes a new method for multimodal learning while dealing with missing modalities. The proposed method uses a single-branch network and a modality switching mechanism that shares weights for multiple modalities.

### Strengths
The paper tackles the interesting and important problem of learning multimodal data while being able to deal with missing modalities.

### Weaknesses
There are a number of shortcomings in the paper:

- The writing is generally ok, but a bit concise imo. Starting off the introduction with "social media users" is a bit strange, given that multimodal data have far wider uses other than social media.

- The method section is unclear and not well-written. First, it states "...sequential fashion. It is achieved by introducing a modality switching mechanism that determines the order in which embeddings are input to the single-branch network." What are the theoretical foundations for this? why is this used? what is the motivation and intuition behind it? Next, the paper states that they have three possible strategies: 1- randomly switching, 2- swishing between multimodal and unimodal 50-50, 3- going only with unimodal. Yet, no details are provided. Which of these are the proposed method? Is the paper simply exploring three options? Are there no other options? why not set the ratio as a hyperparameter and optimize it?

- The entire method is basically explained in a single paragraph, making it almost impossible to understand the details, fundamental theories and motivations behind things, etc.

- The methods used for comparison in Tables 2 through 5 have many important papers missing.

- Especially for the missing modality experiments, only 1 comparison is done (against Ma et al., 2022). Unfortunately, this is not enough, even if the method was sound and explained properly. Further experiments are required to validate the method.

### Questions
Please see my comments under weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a robust multimodal classification system, which is less susceptible to missing modalities. This system leverages a single-branch network to share weights across multiple modalities, and introduces a novel training scheme for modality switch over input embeddings. Extensive experiments demonstrate the effectiveness of the proposed system.

### Strengths
1. The paper is clearly written and contains sufficient details and thorough descriptions of the experimental design.
2. Extensive experiments are conducted to verify the effectiveness of the proposed method.

### Weaknesses
1. While ViLT is a good baseline, it is not a "SOTA" method as there are many more advanced models in recent years. Choosing ViLT as the baseline makes the comparison less convincing. Especially, the proposed system uses pre-extracted embeddings (e.g., CLIP).

2. For the table 2-5, the choices of baselines are a little bit out-of-date. The improvements are marginal while the proposed model uses better features with a lot of heuristic designs.

### Questions
See the weakness

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents Gazelle, a simple yet robust multimodal classification model for handling incomplete modalities. The key idea of the model is to use a modality switching mechanism to sequence the embedding streams of single-branch networks. While the experiments demonstrate Gazelle's superior performance in dealing with missing modalities, the paper could benefit from improvements in presentation clarity, additional theoretical analysis, and more robust experimental results.

### Strengths
1. The paper introduces a simple yet robust method for handling missing modalities. It is presented in an easy-to-follow manner.
2. The method demonstrates superior robustness when compared to existing state-of-the-art methods.

### Weaknesses
1. Incomplete modality/view learning is an important topic in machine learning community, which has achieved great progress in recent years. The authors need to provide a more comprehensive review of the topic.
2. What is the intuition of presenting the modality switching mechanism? A clearer motivation is needed.
3. The proposed method seems to be treated as a training trick. As a general framework, it would be better to provide a theoretical analysis for Gazelle. 
4. The readers would be confused with the presentation of Figure 2. For example, what is the mean of each column in S-1, -2, and -3?
5. Can the proposed method handle missing modality in the training stage? How does the method fuse different modalities?
6. The experiment part could be improved by providing a more in-depth analysis. For example, trying to explain why the proposed modality switching strategy is helpful, and whether existing multimodal learning methods benefit from the strategy.


1. In the field of incomplete modality/view learning, it is imperative to provide a comprehensive review of recent advancements within the machine learning community.
2. It would greatly benefit the paper to clarify the intuition behind presenting the modality switching mechanism. A clearer motivation for its inclusion is necessary.
3. The proposed modality switching mechanism can be treated as a training trick. It would be better to provide a theoretical analysis for it. 
4. Clarifications should be provided for the presentation of Figure 2, particularly regarding the meanings of each column in S-1, -2, and -3 to avoid confusion for readers.
5. Further details regarding the capability of the proposed method to handle missing modalities during the training stage and insights into how it effectively fuses different modalities are needed for clarity.
6. The experiment part could be improved by providing a more in-depth analysis. For example, explain how the proposed modality switching strategy improves robustness, and whether existing multimodal learning methods benefit from the strategy.

### Questions
please see the weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
