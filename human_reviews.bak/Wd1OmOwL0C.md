# Adjusting Pretrained Backbones for Performativity

- Decision: Reject
- Scores: 5, 3, 6, 6

## Abstract
With the widespread deployment of deep learning models, they influence their environment in various ways. The induced distribution shifts can lead to unexpected performance degradation in deployed models. Existing methods to anticipate performativity typically incorporate information about the deployed model into the feature vector when predicting future outcomes. While enjoying appealing theoretical properties, modifying the input dimension of the prediction task is often not practical. To address this, we propose a novel technique to adjust pretrained backbones for performativity in a modular way, achieving better sample efficiency and enabling the reuse of existing deep learning assets. Focusing on performative label shift, the key idea is to train a shallow adapter module to perform a \emph{Bayes-optimal} label shift correction to the backbone's logits given a sufficient statistic of the model to be deployed. As such, our framework decouples the construction of input-specific feature embeddings from the mechanism governing performativity. Motivated by dynamic benchmarking as a use-case, we evaluate our approach under adversarial sampling, for vision and language tasks. We show how it leads to smaller loss along the retraining trajectory and enables us to effectively select among candidate models to anticipate performance degradations. More broadly, our work provides a first baseline for addressing performativity in deep learning.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper considers a scenario where the input data distribution shifts based on the model's predictions once the trained model is deployed. The authors specifically assume a shift in class marginal distributions and propose a method to learn a shallow adapter module that performs Bayes-optimal label shift correction on the pre-trained classifier. This module adjusts the classifier's logits using a given sufficient statistic. Experiments on CIFAR-100, ImageNet100, Amazon, and AGNews datasets demonstrate that the proposed method can create robust models against tested performative label shifts in both vision and language domains.

### Strengths
1. The proposed method is lightweight and easy to implement.
2. The proposed method can create robust models against tested performative label shifts in both vision and language domains.

### Weaknesses
1. The assumed performative label shift model in the experiments (Eq. 6) is too simple. It seems possible to model a function that corrects the classifier's logits without using a neural network module. What exactly is the neural network module learning in the proposed method?
2. In real-world scenarios, what situations would the performative label shift tested in the experiments apply to?
3. What is the most common type of performative shift in real-world applications?

### Questions
(Copied from Weaknesses)
1. The assumed performative label shift model in the experiments (Eq. 6) is too simple. It seems possible to model a function that corrects the classifier's logits without using a neural network module. What exactly is the neural network module learning in the proposed method?
2. In real-world scenarios, what situations would the performative label shift tested in the experiments apply to?
3. What is the most common type of performative shift in real-world applications?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposed PaP to adjust pretrained backbones for performativity in a modular way, enabling the reuse of existing deep learning assets.
The  key idea is to train a shallow adapter module to perform a Bayes-optimal label shift correction to the backbone's logits given a sufficient statistic of the model to be deployed.

### Strengths
The paper idea is simple and reasonable. The overall framework is efficient and easy to deploy.
Compared to without adapdation, the model significantly improved the performance.

### Weaknesses
1. Some of the definitions in the paper is clear, which increased the difficulty to understand the paper.
2 The benchmark dataset setting is not clearly stated.

### Questions
1. What is the clear definition of distribution shift? There is no clear definitions for this term throughout the paper. I think in this paper you only study the label shift. You should clearly define it instead of mentioning distribution shift everywhere.
2. What is the definition of label marginals, how do we calculate it? This is also not defined but used everywhere.
3. What is the memory M in the algorithm 1? I still did not get why we need M. I did not see any definitions of it in the paper. Also, considering it is a memory buffer, I did not get the point how did you get the gradient to update T.
4.  In Table 1, PaP estimated performance is reported here. How did you calculate it? I still did not see any related descriptions in the paper. I would guess it is sufficient statistic S in your paper, but that is also never clearly defined for the related calculations.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the use of a novel adapter module with pretrained vision/language backbones to address the issues with performative label shift. These label shifts are usually a result of deployed deep learning models. The authors validated their proposed approach with simulated label shifts on vision and language tasks with adversarial sampling setting etc..

### Strengths
This paper addressed an important problem of performativity for deployed deep learning models given post deployment distribution shifts, and proposed a plug-and-play adapter module to be used with pre-trained backbones. The authors have formulated this interesting problem and conducted various experiments demonstrating the effectiveness of their method.

### Weaknesses
1. It's a bit hard to understand the proposed method given the writing, which needs to be improved particularly at places critical for elaborating the core module and results. For example, reading through section 1.1 and section 3, it's unclear what "sufficient statistic" refers to, which is an important inputs to the proposed adapter module. It's not until in later sections the authors gave examples for statistic such as class accuracies. Another example, reading through p. 7 lines 324-328, I still have trouble understanding what to take from Table 1 and how to interpret the ranking.

2. While the author claimed that none of the prior work addressing distribution shifts is designed to address them proactively. The proposed method also relies on data with label shifts to train the adapter. This method similarly requires distribution shifted data at train time. The author should clarify what is the key advantage of the adapter since it does not work proactively either.

3. It will help understand the effectiveness of the method if more experiment details are given. E.g., how does the parameter "round T" affect the final adapter from the training stage? Given that in Algorithm 1 the output is one single adapter module at time T, is this adapter used to test throughout the 100 rounds shown in Figure 2?

4. Are there some conjectures / explanations for the drastic drop in Oracle finetuning at Round 1 from Figure 2 and 3? It's unclear if this is an issue with the particular label shift or finetuning stage setup.

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the challenge of performative label shift, where traditional methods assume access to performativity-augmented datasets, but training from scratch for each shift can be costly. The proposed method introduces an adaptation module that predicts label marginals and applies a Bayes-optimal correction to model logits, effectively reducing performance degradation due to distributional shifts. This approach is the first to adapt deep learning models for performative shifts, offering the potential for zero-shot transfer and anticipating model brittleness across shifts. Experimental results support the method’s effectiveness in mitigating performance loss with limited performativity-augmented data.

### Strengths
Originality: The paper introduces two novel contributions: a practical baseline for handling performative label shift and a module to predict model performativity. These aspects offer a fresh approach in this field.

Quality: The paper is well-executed, with extensive experiments across several datasets. Baselines are clearly presented, and the rationale behind the choices is well-explained, making the contribution and results robust and grounded.

Significance: Although I am not familiar with this field, the framework appears practical, easily reproducible, and offers potential for further development. While I cannot fully assess the technical impact, the framework shows promising potential for future improvements.

### Weaknesses
I must admit I'm having difficulty understanding certain aspects of this article. Firstly, regarding the evaluation metric in Lines 362-363: what exactly is the 'retraining trajectory'? Is this term defined within this article, or does it originate from previous work? Additionally, how is this definition of 'Acc' distinct from 'Acc in S'? This difference significantly impacts my interpretation of the results in Figures 2 and 3.

Similarly, in Algorithm 1, does 'Deploy' imply training on the sample, and does 'Observe sample' indicate the same? I'm finding it challenging to follow the process outlined in this algorithm diagram. Furthermore, in Section 3.2, what is the purpose of the discussions on 'Dynamic benchmarking' and 'self-selection'? I haven't found results related to these methods in the experimental section.

Some details require further clarification. First, what are the results for other datasets in Table 1? This information is essential to more comprehensively support PaP's ability to predict model performativity. Second, which pre-training datasets were used, and how might the distribution of the pre-training data impact the results

### Questions
Please see the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3
