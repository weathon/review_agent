# Exploring Weak-to-Strong Generalization for CLIP-based Classification

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3, 3, 3, 3

## Abstract
Aligning large-scale commercial models with user intent is crucial to preventing harmful outputs. Current methods rely on human supervision but become impractical as model complexity increases. When models surpass human knowledge, providing accurate feedback becomes challenging and inefficient.
A novel solution proposed recently is using a weaker model to supervise a stronger model. This concept leverages the ability of weaker models to perform evaluations, thereby reducing the workload on human supervisors. 
Previous work has shown the effectiveness of weak-to-strong generalization in the context of language-only models. Extending this concept to vision-language models leverages these insights, adapting the proven benefits to a multi-modal context.
In our study, we explore weak-to-strong generalization for CLIP-based classification. We propose a method, \emph{class prototype learning} (CPL), which aims to enhance the classification capabilities of the CLIP model, by learning more representative prototypes for each category.
Our findings indicate that despite the simple loss function under weak supervision, CPL yields robust results.
Our experiments are conducted on challenging datasets to evaluate our method. Extensive experiments show that our method is effective, achieving a 3.67\% improvement over baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper aims to explore the weak-to-strong generalization framework in image classification tasks. The paper proposes to learn a classifier for a strong model guided by a pre-trained weak model.

### Strengths
1. The paper explores an interesting weak-to-strong generalization problem in image classification tasks, which is still underexplored.
2. The proposed approach is simple and easy to implement, which only needs to learn a set of class prototypes in downstream tasks.
3. Experiments show that the proposed weak-to-strong generalization method outperforms several baselines.

### Weaknesses
1. Regarding the problem setup, the reviewer finds it unclear why labeled data cannot be directly used to fine-tune a strong model, especially since weak models in the experiments are trained with labeled data. In my opinion, the weak-to-strong generalization framework is more reasonable if the pre-trained CLIP serves as a weak learner to guide a strong model with more learnable parameters.

2. Regarding the novelty of the method, the proposed class prototype learning is closely related to training cosine classifiers initialized with CLIP text prompts, a strategy already explored in previous work [1]; Additionally, knowledge distillation using unlabeled data has been extensively studied in previous semi-supervised learning research [2].

3. The paper may overclaim its contribution to extending the weak-to-strong generalization to multi-modal tasks because the paper only uses the CLIP model and does not experiment on multi-modal datasets.

4. Implementation details are missing because the appendix of the paper is not accessible.

[1] Long-Tail Learning with Foundation Model: Heavy Fine-Tuning Hurts. ICML 2024.

[2] Big Self-Supervised Models are Strong Semi-Supervised Learners. NeurIPS 2020.

### Questions
1.	The implementation details mention that the experiments were conducted on a single V100 GPU with 40GB of memory. However, does the V100 GPU actually have 40GB of memory?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper focuses on knowledge distillation of visual language models - CLIP. Specifically, this paper proposes a simple method called CPL, which considers extracting knowledge from additional weak models and additional unlabeled datasets to enhance a strong model.

### Strengths
The proposed method is quite simple and easy-to-follow.

### Weaknesses
The reviewer may lack familiarity with recent work on *weak-to-strong generalization* and will therefore wait additional perspectives from other reviewers to assess the technical novelty of this paper. Based on the my expertise in representation learning and knowledge distillation, I have the following concerns and questions:

1. **Methodology**: The approach proposed in this paper issomewhat straightforward, essentially using the original KD (Knowledge Distillation) loss. This may detract from the novelty of the method. Does the author have any distinctive insights tailored to the characteristics of vision-language models?

2. **Experiments**: The experimental comparisons presented in the paper lack clarity. For instance:
   - *Strong Ceiling*: What fine-tuning method is applied here?
   - *KD+LP*: What does this refer to, and how does it differ from CPL? From my understanding, CPL is exactly KD between two models.
   - Most importantly, how do you utilize \(D_{hold}\) and \(D_{train}\) in various settings?

3. **Motivation**: The authors also need to clarify the relationship between their problem setup and other weakly supervised learning setups, particularly semi-supervised learning. If the Strong Ceiling (directly fine-tuning a strong model with \(D_{hold}\)) achieves optimal performance (as shown in Table 2), what motivates us to first fine-tune a weak model with \(D_{hold}\) and then use that weak model to supervise the strong model? In both cases, \(D_{hold}\) is assumed to be accessible, and the computational costs is similar  (if not higher). *Conversely, following general knowledge distillation principles, I would expect the two-step approach to achieve better results than Strong Ceiling. This should ideally be independent of whether the teacher model is weak or strong, as even a simple label smoothing can sometimes yield improved performance.*

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper investigates the concept of weak-to-strong generalization within the context of vision-language models (VLMs), specifically focusing on the CLIP model. The authors proposed class prototype learning (CPL), which performs logit distillation between a supervised learning (weak) model and the class prototypes of a CLIP model (strong). The effectiveness of CPL is demonstrated through experiments on the DomainNet dataset, reporting a 3.67% improvement over baseline methods.

### Strengths
- This paper explores weak-to-strong generalization -- how to train models when models surpass human knowledge. Unlike previous works that consider LLM, it works on CLIP, a VLM.
- Experiments show that the proposed method shows improvements over other CLIP tuning methods.

### Weaknesses
- The paper starts with an ambitious story (weak-to-strong generalization) but remains unclear how the setting and solution can benefit human-surpassing models. The scope of the paper narrows down to a specific application in CLIP classification, which is disconnected from the overarching goals of weak-to-strong generalization. If the major novelty is considering a VLM, why not consider LLaVA, BLIP2, or similar models?
- The setting and proposed method appear to be somewhat ad-hoc focused on improving CLIP classification through distillation from a supervised learning model. This raises questions about the generalizability of the findings beyond the specific context of classification and CLIP. For one thing, can the method help CLIP recognize unseen classes or unseen concepts? For another, can the method improve generative VLMs?
- The method itself is naive and lacks novelty. Can the authors clarify more on the novel contributions compared with prior methods?
- The writing lacks clarity and coherence, making it difficult for readers to grasp the main contributions and findings. The paper spends excessive time on background information and settings, which detracts from the core message. Also, the setting of data splits (L315-L323) is confusing and hard to understand.

### Questions
N/A

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes Class Prototype Learning (CPL), a novel method which aims to improve the classification performance of CLIP models , via learning more representative prototypes for each category under weak supervision. Despite using a simple loss function, CPL achieves obvious improvement over baselines on DomainNet dataset (including 6 different styles: Clipart, Infograph, Painting, Quickdraw, Real, and Sketch)
This work highlights the potential of weak-to-strong generalization for image-text alignment, providing a new direction for future fine-tuning research on VLMs.

### Strengths
1.As claimed by the authors, their CPL method archives SOTA on many kinds of weak models(including resnet and cvt), when testing on six distinct domains of DomainNet dataset.

2.Unlike traditional knowledge distillation, weak-to-strong generalization uses weaker model as the teacher, and this work firstly introduces this new knowledge distillation method to VLM. In fact, the research is quite interesting and meaningful.

### Weaknesses
1.More related experiments are needed to increase confidence of your method, as experimental result comparisons with counterparts on many important datasets are missing.

2.There are some imprecise statements in the paper, and the CPL method lacks visual demonstration of results. It is necessary to add the relevant content in the appendix.

Please see the questions section for more details.

### Questions
1.Why do you conduct experiments on DomainNet dataset only?As you aim to improve the classification performance of CLIP models, testing on ImageNet(including the comprehensive Stanford version and -V2/ A / R/ Sketch etc.) and other smaller but typical datasets(Flower102, DTD, Pets, StanfordCars, UCF101, Caltech101, Food101, SUN397, Aircraft, EuroSAT etc.). Conducting experiments on those datasets can significantly enhances the credibility of your CPL method. 

2.Some statements in your paper are not rigorous and lack clarity. For example, in lines 242-243, 'students ' among 'Here, the weaker models act as students guiding the stronger model' should be 'teachers', if you mean the knowledge distillation in weak-to-strong generalization. Also, in line 489, ‘RLHF’ is mistakenly written as ‘RLFH’. 

3.The paper lacks an introduction to phototype learning, which not only reduces the readability of the article but also violates the academic requirements for referring previous work. Additionally, It's better to give references for the terminology like RLHF and AdaptConf etc., even though the terminology is well-known.

4.The paper should specify the basic conditions and configurations of the experiments. For example, you should clarify what image encoder and text encoder are used in the strong model (CLIP), and state that all strong models referred to in this paper are configured with this specific CLIP setup. This avoids confusion for readers and the need to repeatedly check the context for confirmation.

5.Other popular VLMs (such as Llama and LLaVA) could be introduced as strong models and compared against CLIP. That can further enhance the generalizability and universality of your paper.

In fact, I have no idea why your work is so rough. Maybe lack of time for the submission deadline? If so, you can supplement the relevant information during the subsequent rebuttal phase. I hope you can carefully check the references and mathematical derivations in the paper again, to ensure the accuracy and standardization of the article. Once the requirements are met, I will reset my review score.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper investigates weak-to-strong generalization in CLIP-based classification. The author introduces a class prototype learning (CPL) method to enhance the classification capabilities of a stronger CLIP model with assistance from a weaker model. CPL achieves this by developing more representative prototypes for each category. Empirical results demonstrate that CPL consistently outperforms baseline methods across various weak models.

### Strengths
1. This paper is the first to explore weak-to-strong generalization in CLIP-based classification.
2. This paper proposes a straightforward yet effective method for achieving good performance.

### Weaknesses
1. The aim of weak-to-strong generalization is to mitigate harmful outputs while enhancing model performance. However, in your experimental results, while an improvement in the strong model's performance is evident, the aspect of protection against harmful outputs is not sufficiently demonstrated.
2. You have only validated your method on the DomainNet dataset. We recommend testing the effectiveness of your approach on additional datasets.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper aims to enhance the classification capability of the CLIP model given a weak model and an unlabeled training set. It proposes a new method called class prototype learning (CPL). Specifically, for a classification task with k categories, CPL first initializes k prototypes using the text features, and calculates the strong logits by matching the image features and the prototypes. After that, it utilizes a weak model to calculate the weak logits, and then uses the weak logits to teach the strong logits by optimizing the class prototypes. In the inference stage, the model is able to make predictions by matching the image features and the learned prototypes. The authors also conduct experiments to support the proposed method.

### Strengths
- The idea of weak-to-strong generalization is somewhat meaningful.
- The efficiency is improved by removing the text encoder in the inference stage.

### Weaknesses
- I am quite confused with the 'unlabeled data' setting. Although CPL does not use labeled data, the weak model is trained with ground truth labels. Maybe you just want to verify the idea of 'weak-to-strong'. However, in practice, you will not have a weak model just right having k candidate categories (particularly when the weak model is a vision-only model), thus it is infeasible to optimize the CPL loss to teach the k prototypes.
- The ideas of using the text encoder for initialization and dropping the text encoder during the inference stage have been proposed by multiple previous works [1-3].
- To demonstrate the efficiency of CPL, it is better to include the complexity analyses.
- The related works regarding unsupervised learning with CLIP are not discussed [4-5].

[1] Masked Unsupervised Self-training for Label-free Image Classification.

[2] Parameter-Efficient Long-Tailed Recognition.

[3] Exploring Vision-Language Models for Imbalanced Learning.

[4] Unsupervised Prompt Learning for Vision-Language Models.

[5] PromptKD: Unsupervised Prompt Distillation for Vision-Language Models.

### Questions
See weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1
