# Generating Model Parameters for Controlling: Parameter Diffusion for Controllable  Multi-Task Recommendation

- Decision: Reject
- Scores: 5, 3, 6, 6

## Abstract
Commercial recommender systems face the challenge that task requirements from platforms or users often change dynamically (e.g., varying preferences for accuracy or diversity). Ideally, the model should be re-trained after resetting a new objective function, adapting to these changes in task requirements. However, in practice, the high computational costs associated with retraining make this process impractical for models already deployed to online environments. 
This raises a new challenging problem: how to efficiently adapt the learning model to different task requirements by controlling model parameters after deployment, without the need for retraining.
To address this issue, we propose a novel controllable learning approach via Parameter Diffusion for controllable multi-task Recommendation (PaDiRec), which allows the customization and adaptation of recommendation model parameters to new task requirements without retraining. 
Specifically, we first obtain the optimized model parameters through adapter tunning based on the feasible task requirements. Then, we utilize the diffusion model as a parameter generator, employing classifier-free guidance in conditional training to learn the distribution of optimized model parameters under various task requirements. Finally, the diffusion model is applied to effectively generate model parameters in a test-time adaptation manner given task requirements. As a model-agnostic approach, PaDiRec can leverage existing recommendation models as backbones to enhance their controllability. Extensive experiments on public datasets and a dataset from a commercial app, indicate that PaDiRec can effectively enhance controllability through efficient model parameter generation. The code is released at https://anonymous.4open.science/r/PaDiRec-DD13e.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces PaDiRec, a neural network diffusion-based approach for generating adaptable model parameters for controllable multi-task recommendation.

### Strengths
1. This paper presents an intriguing application of diffusion models for multi-task recommendation.

2. The paper is well-written and organized.

3. Code availability enhances the paper’s reproducibility.

### Weaknesses
Here are the polished points based on the content of the paper:

1. This paper primarily represents an application of the paper of "Neural Network Diffusion"  [1] in recommender systems, specifically building on the framework from the referenced neural network diffusion work. While the authors cite this paper, they do not provide a thorough enough introduction to or comparison with this prior work in the related work section, which does not fully credit the original contribution. It is suggested that this work be submitted to an application-focused conference.

2. This paper emphasizes practical applications rather than algorithmic development, making online testing essential. However, all experiments are conducted on offline datasets, raising concerns about the real-world usability of the proposed approach.

3. The backbone models used, such as SASRec, are not state-of-the-art in recommendation systems, which could limit the effectiveness and generalizability of the proposed approach when compared with modern backbone models in recent two years such as [2].

4. The paper’s focus is limited to accuracy and diversity, yet recommendation systems are often evaluated by a broader set of metrics, including serendipity, fairness, and more. Expanding the testing metrics could more robustly validate the method's effectiveness across a range of evaluation criteria.

[1] Wang, Kai, et al. "Neural network diffusion." *arXiv preprint arXiv:2402.13144* (2024).

[2] Yue, Zhenrui, et al. "Linear recurrent units for sequential recommendation." *Proceedings of the 17th ACM International Conference on Web Search and Data Mining*. 2024.

### Questions
1. Are there references supporting the use of Pearson r-d, or is this application new in this paper? If it’s new, could other established methods be used for evaluation instead?

2. In calculating the reported efficiency, is the diffusion training time included into the results? Additionally, is the diffusion model designed to require only a single training session that allows for permanent deployment, or does it need periodic retraining?

### Soundness
2

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
This paper highlights the dynamic preferences to be considered when deploying recommender systems in practice. Traditional models, including multi-task learning methods, cannot effectively handle dynamic user preferences. Although retraining is a common approach, it is very time-consuming and resource-intensive. Therefore, they proposed a parameter generation method for controllable multi-task recommendation, which effectively generates task-specific model parameters using a generative model. Specifically, they proposed PaDiRec by formulating an objective function consistent with task-specific preference weights and then using adapter tuning to fine-tune model parameters. They trained a diffusion model to learn the conditional distribution of these optimized adapter parameters. Finally, they evaluated PaDiRec using different sequential recommendation backbones to produce recommendations.

### Strengths
(1) This paper studies an interesting research problem that is common in the field of recommender systems.
(2) The authors proposed a series of methods, including adapter modules, accuracy + diversity objective functions, adapter tuning, etc.
(3) The authors proved their point through a large number of experiments

### Weaknesses
(1) The evaluation presented in this paper raises some concerns. First, the authors assessed the sequential recommendation task using the MovieLens dataset. I would strongly advise caution, as the user behaviors reflected in the MovieLens data are not genuine viewing behaviors but rather rating behaviors. Many users rate movies they have not actually watched, often based on prior knowledge. A closer examination of the dataset reveals that some users can rate five or even ten movies within just a few minutes.  The rating time in MovieLens is not the watching time! Also see  [1] by Sun et.
 
(2) The hyperparameters may not have been fine-tuned carefully. While the MovieLens dataset does exhibit high sequential patterns, it is important to note that, as mentioned, the user behaviors are not reflective of actual viewing. The sequential patter of recommendation systems employed by MovieLens can be easily detected using models like SASRec. I observed that the authors used an embedding size of 64; to my knowledge, SASRec's optimal embedding size is significantly larger than this in MovieLens. Utilizing a non-optimal embedding size for baseline models undermines the claim that the proposed model is superior. It’s essential to recognize that in deep learning, simply comparing models with the same embedding size is not a fair approach. Some deep learning models can effectively utilize larger embedding sizes, while others may reach saturation with smaller embedding sizes.

(3) Multi-task learning is often applied in the ranking stage, while sequential recommendation typically functions as a recall algorithm. I do not think it makes much sense to achieve better performance in the sequential recommendation task.

(4) In fact, having a more powerful sequential recommendation model may not be that important, as such improvements generally do not bring any benefits in the ranking stage or in online systems. Many sequential patterns learned by deep Transformer models are more closely related to recommendation exposure patterns than to actual user behavior patterns. Therefore, I do not think that the solution proposed in this paper has a particular impact on the current recommender system landscape.

[1] Our Model Achieves Excellent Performance on MovieLens: What Does It Mean? TOIS2024

### Questions
no

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces PaDiRec, Parameter Diffusion for Controllable Multi-Task Recommendation, a framework designed to adapt recommender systems to changing task requirements without retraining. Traditional recommendation models often struggle to adjust dynamically. PaDiRec addresses this issue using a parameter diffusion model that can modify model parameters at test time, making it a model-agnostic solution adaptable to various backbone structures. Experiments on public datasets and an industrial dataset show that PaDiRec achieves high controllability and performance.

### Strengths
PaDiRec enables real-time, on-demand changes in task preferences without requiring expensive retraining, making it ideal for industry. The use of diffusion models as parameter generators is also novel in the context of recommender systems, offering robust parameter generation that captures task-specific nuances.

The paper provides well-designed experiments and several datasets. According to the results, PaDiRec integrates well with diverse backbone models, proving its applicability across different recommendation systems and settings as well as the capability to achieve near real-time responses with significantly reduced latency compared to traditional retraining.

### Weaknesses
The approach relies heavily on predefined utilities which may not fully generalize to tasks with complex, non-linear objectives. This dependency could limit its flexibility in handling multi-faceted objectives beyond accuracy and diversity.

While PaDiRec performs faster than retraining, the diffusion process may still be computationally intensive, especially in environments with limited processing power. More details on efficiency optimization could enhance its feasibility for larger-scale systems.

### Questions
How should practitioners choose between the various conditioning strategies (e.g., Pre&Post, Ada-Norm) when applying PaDiRec to new recommendation environments? Are there guidelines or heuristics to assist in selecting the optimal strategy?

How would PaDiRec handle non-standard or non-linear preference metrics? Would additional diffusion model tuning be required?

### Soundness
3

### Presentation
3

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
This paper proposes a new learning method called "parameter diffusion" for controllable multi-task recommendation, which allows customization and adaptation of recommendation model parameters to new task requirements without retraining. The proposed method uses existing optimized model parameters to generate new ones that are adapted to different task requirements through parameter diffusion. The main contribution of this paper is to provide a new learning method that enhances the controllability and flexibility of multi-task recommendation systems.

### Strengths
•	Tackles the practical issue of dynamic task requirements
•	Allows customization and adaptation of recommendation model parameters to new task requirements without retraining
•	Novel combination of diffusion models with adapter tuning

### Weaknesses
•	High inference complexity of diffusion models may not meet real-time recommendation requirements. No detailed analysis of computational overhead during inference

•	Parameter Optimization Strategy: 
o	Relies solely on single-task optimized parameters, ignores potential benefits of multi-task joint optimization

•	Limited Generalization and Flexibility: 
o	The method is primarily tested on only two specific tasks (accuracy and diversity); Unclear scalability to more utilities/tasks.
o	Predefined preference weights limit the model's adaptability to unexpected scenarios

### Questions
Are there any standard or guiding principles that can help us choose appropriate task requirements (preference weights)?

### Soundness
3

### Presentation
2

### Contribution
3
