# Angle-optimized Text Embeddings

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 6

## Abstract
High-quality text embedding is pivotal in improving semantic textual similarity (STS) tasks, which are crucial components in Large Language Model (LLM) applications.  However, a common challenge existing text embedding models face is the problem of vanishing gradients, primarily due to their reliance on the cosine function in the optimization objective, which has saturation zones.  To address this issue, this paper proposes a novel angle-optimized text embedding model called AnglE. The core idea of AnglE is to introduce angle optimization in a complex space. This novel approach effectively mitigates the adverse effects of the saturation zone in the cosine function, which can impede gradient and hinder optimization processes.  To set up a comprehensive STS evaluation, we experimented on existing short-text STS datasets and a newly collected long-text STS dataset from GitHub Issues.  Furthermore, we examine domain-specific STS scenarios with limited labeled data and explore how AnglE works with LLM-annotated data. Extensive experiments were conducted on various tasks including short-text STS, long-text STS, and domain-specific STS tasks. The results show that AnglE outperforms the state-of-the-art (SOTA) STS models that ignore the cosine saturation zone.   These findings demonstrate the ability of AnglE to generate high-quality text embeddings and the usefulness of angle optimization in STS.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper notices that existing text embedding models mainly use cosine function as a part of the objective function, but cosine function has a saturation zone, which may cause gradient vanishing problem and influence the quality of text embeddings. To mitigate this problem, this paper proposes to evaluate the angle difference between two text embeddings for optimization. Experiments on variable lengths of text datasets, including a newly introduced long-text dataset, are conducted to evaluate the performance of the proposed model.

### Strengths
1. This paper identifies an interesting research question, the gradiant vanishing problem appearing at the saturation zone of cosine function influences the quality of text embeddings.

2. The proposed solution of using angle difference for optimization is orginal and novel.

3. Experiments on semantic textual similarity task are sufficiently conducted.

### Weaknesses
Despite an appealing motivation and an interesting solution, I still have the following concerns:

1. From my point of view, the only technical contribution of this paper is to design how to evaluate angle difference. This contribution is indeed interesting, but is a bit superficial and insufficient for a long research paper of ICLR standard. I expect authors to propose more __insightful__ designs to better solve the gradient vanishing problem.

2. The explanation of why saturation zone in cosine function influences text embedding learning is not clearly written at the Introduction section. Authors are suggested to explain more about the meaning of saturation zone and why it causes gradient vanishing problems.

3. Usually we encourage authors to conduct the same experiment multiple times and report both mean and standard deviation, in order to verify that the proposed model indeed significantly outperforms baselines. However, I see mean but not standard deviation in the paper.

### Questions
1. Authors use absolute value at Eq. 6. But absolute function in pytorch or tensorflow is not differentiable, how do authors deal with error backpropagation for absolute function?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
To overcome the negative impact of vanishing gradients caused by the cosine optimization function, this paper proposed a novel angle-optimized target to improve the quality of text embeddings. Moreover, this paper conducted extensive experiments to prove the effectiveness of the proposed method. Meanwhile, this paper also developed a novel long-text STS dataset to support the community.

### Strengths
1.	This paper proposed a novel angle-optimized target to enhance the learning ability of contrastive learning-based representation learning models, which tried to alleviate the problem of vanishing gradients. 
2.	This paper developed a novel long-text STS dataset to better evaluate the performance of representation learning models. 
3.	This paper also explored LLM-based supervised data generation and contrastive learning, which is very interesting.

### Weaknesses
1.	First of all, the authors argued that gradient vanishing problem is caused by the saturation zones in cosine functions in the optimization target. However, as far as I know, the gradient vanishing problem is mainly due to the deep structure. The saturation zones can be used to prove the high similarity between sentences. Therefore, the motivation of this paper is not so convincing. More explanations are needed. 
2.	Second, the authors focused on contrastive learning target, which limits the application range of the proposed method. The authors should provide more evidence to demonstrate the effectiveness of their method since their main contribution is adding an additional target in contrastive loss. 
3.	Third, the related work in this paper is not sufficient enough. More content should be cited, such as different contrastive loss designs, sentence similarity measurement designs, etc.

### Questions
N/A

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a novel angle-optimized text embedding model to improve the semantic textual similarity (STS) tasks, by mitigating the vanishing gradients of cos similarity. Specifically, the authors employ a contrastive learning objective and introduce optimization in a complex space to address the saturation zone in the cosine function. Extensive experiments are conducted to show the effectiveness of the proposed method on various tasks including short-text STS, long-text STS, and domain-specific STS.

### Strengths
1. The proposed method of calculating similarity looks novel to me.

2. The impact of the method has the potential to be significant in many fields.

### Weaknesses
1. According to the paper, the motivation for introducing a complex space is to deal with the vanishing gradient of cos. In this sense, it would be great if techniques like gradient clipping and gradient normalization could be compared. 

2. The writing can be improved. E.g., section 3.4 is a bit confusing to me. See my questions below.

3. I am also worried about the empirical significance. In table 2, the proposed method only improves the performance marginally (<1%) compared to SimCSE-BERT. I appreciate the effort that the p-value is reported and yet the p-value is smaller than 0.05 according to the caption of table 2.

### Questions
1. In section 3.4, X is decomposed into real part Xre and imaginary part Xim, both of which have dimension 1. However, in the context of contrastive learning / the use of cos similarity, X is often high dimensional. How do you decompose X? If I am not mistaken, this part is missing in the paper.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a new method called AnglE to address the vanishing gradient problem of optimizing cosine similarity in text embedding learning models. AnglE uses an angle-based optimization method to learn text embeddings in a complex space. The method is demonstrated to outperform state-of-the-art models on various semantic textual similarity (STS) tasks, including short-text STS, long-text STS, and domain-specific STS. Additionally, AnglE can be used with limited labeled data and LLM-annotated data, and it achieves competitive performance in these settings.

### Strengths
* The paper addresses an important issue in optimizing the cosine similarity of learning text embeddings, and the proposed method is interesting and novel.
* It introduces the GitHub Issues Similarity Dataset as a testbed for evaluating model performance on long-text STS tasks.
* The proposed method achieves promising results on a wide range of STS tasks.

### Weaknesses
* Some technical details are not clearly explained. For example, while the angle objective optimizes the text representations in a complex space, it's unclear how these complex vectors are obtained as the representations from language models are real vectors.
* The paper seems to have missed discussions with a few important related studies. For example, [1] addresses the gradient vanishing issue by incorporating cosine distance in learning text embeddings, [2] designs angular softmax objectives to learn visual representations. The LLM-supervised learning procedure largely follows the prompt-based training data generation paradigm in [3,4,5]. While this part is not the major contribution of the paper, it's better to reference these related works as well.

References:  
- [1] “Spherical Text Embedding.” NeurIPS (2019).
- [2] “SphereFace: Deep Hypersphere Embedding for Face Recognition.” CVPR (2017).
- [3] “Generating Datasets with Pretrained Language Models.” EMNLP (2021).
- [4] “Generating Training Data with Language Models: Towards Zero-Shot Language Understanding.” NeurIPS (2022).
- [5] “ZeroGen: Efficient Zero-shot Learning via Dataset Generation.” EMNLP (2022).

### Questions
* Could you explain how the complex vectors are obtained exactly from the language models?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
