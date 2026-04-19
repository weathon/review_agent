# Improving Prompt-based Continual Learning with Key-Query Orthogonal Projection and Prototype-based One-Versus-All

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 5

## Abstract
Drawing inspiration from prompt tuning techniques applied to Large Language Models, recent methods based on pre-trained ViT networks have achieved remarkable results in the field of Continual Learning. Specifically, these approaches propose to maintain a set of prompts and allocate a subset of them to learn each task using a key-query matching strategy. However, they may encounter limitations when lacking control over the shift of features in the latent space and the relative separation of latent vectors learned in independent tasks. In this work, we introduce a novel key-query learning strategy based on orthogonal projection, inspired by model-agnostic meta-learning, to enhance prompt matching efficiency and address the challenge of shifting features. Furthermore, to harness the benefits of reduced feature shifting, we introduce a One-Versus-All (OVA) prototype-based component that enhances the performance of the classification head. Experimental results on benchmark datasets demonstrate that our method empowers the model to achieve results surpassing those of current state-of-the-art approaches by a large margin of up to 20%.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper performed an in-depth analysis of the state-of-the-art CODA-Prompt for continual learning of pre-trained ViT. The authors attributed the problem as the mismatch in prompt representation between training and testing and feature shifting during inference. The authors then proposed Key-Query orthogonal projection to reduce dependence of old task queries on new task keys and introduced a prototype-based OVA loss to complement the Key-Query orthogonality. The proposed method achieves a surprisingly high performance,  even much higher than the joint training.

### Strengths
1. The analysis of CODA-Prompt is very extensive, and the identified issues are reasonable. In particular, I appreciate the discussion about the different effects of training samples and test samples.

2. The proposed method has strong motivation, which directly targets the identified issues of CODA-Prompt.

3. The proposed method achieves a surprisingly high performance over widely used benchmarks.

### Weaknesses
1. My major concern is the surprisingly high performance of KOPPA, which is even significantly higher than the joint training performance by more than 10%. Since the joint training usually serves as the upper bound of continual learning, I would suggest the authors to provide an in-depth analysis and explanation of this abnormal phenomenon.

2. From Table 3, the outstanding performance of KOPPA seems to be largely due to the OVA. The contribution of Key-Query orthogonal projection in Sec. 3.3 seems to be marginal. 

3. From Table 5, the performance improvements of KOPPA rely heavily on the number of prototypes. Although the authors have evaluated the effect of CE and OVA in Table 4, how about using the preserved prototypes to compute CE rather than OVA (i.e., identical to regular feature replay for continual learning)?

-------------------------------------------------------------------------------

After reading all reviewers' comments and the authors' rebuttal, I think this is a borderline paper with both pros and cons. The pros include a clear analysis of SOTA prompt-based methods and the remarkably high performance. The cons include technical contributions (the OVA dominates the improved performance but seems borrowed from another paper) and additional storage cost.

In fact, I'm very surprised that the use of OVA (and even a naive classifier trained with prototypes, i.e., CE † + CE in rebuttal) to rectify the final prediction can improve the performance by such a huge margin. Then I carefully check the provided code. I think the implementation of OVA might suffer from an **information leakage issue**: At test time, the OVA score is calculated as the average of a test batch, and then used to rectify the final prediction of each test data (see default.py line 475-505, 536-546). I find the test_loader has batch size = 128 and shuffle=False (see trainer.py line 139 and configs), which means the OVA uses the average of a large batch of the same label as the prediction. In practice, this implementation can largely increase the "prediction accuracy", but is not reasonable.

An empirical validation of this potential issue is that, setting batch size = 1 and/or shuffle = True in the test_loader to run the implementation code. I have run some experiments with the provided code. When setting batch size = 1 of the test_loader, the performance of KOPPA declines from 97.82% to **86.64%** on Split CIFAR-100, which is comparable to its target CODA-Prompt (86.25%). I think this is a strong evidence that the improvement of OVA is from the **information leakage issue**. Considering the limited improvement of the other design (i.e., the orthogonal projection) as suggested by other reviewers, the technical contributions of this paper seem to be less than significant. Therefore, I decrease my score to 3.

### Questions
Please refer to the weakness. 

Besides, I would encourage the authors to discuss and compare with more advanced prompt-based baselines, such as [1] and [2]. While it is not required, such a discussion could bring this paper to a more advanced position, especially when the results of this paper seem to be far superior to [1] and [2].

[1] Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality, NeurIPS 2023.

[2] RanPAC: Random Projections and Pre-trained Models for Continual Learning, NeurIPS 2023.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper claimed that current prompt selection methods suffer from the mismatch in prompt representation between training and testing and feature shifting during inference. To this end, the authors proposed a MAML-inspired method to ensure an almost certain perpendicular constraint between future keys and past task queries, effectively eliminating feature shifts. The proposed method also contains a prototype-based One-Versus-All (OVA) component to boost the task classification head distinction. The proposed method shows accuracy much higher than joint training (upper bound of continual learning, using all the data for training).

### Strengths
- The problem that this paper is trying to solve is well-motivated
- The overall flow of the paper is easy to follow.
- The proposed method shows very good empirical results.

### Weaknesses
- Some parts of the paper are unclear and confusing to me. Please refer to the Question section. 
- There could be mistakes in some calculations.  Please refer to the Question section. 
- The authors did not mention the limitations of their method and potential future work. The paper does not explore or discuss potential failure cases of the proposed methods. Understanding when and why the methods might fail is crucial.
- typos
   - "alpha if corresponding weight vector" "if" should be "is" right?
   - "It also mitigates the chance that the query q(x′) is a past task uses the prompt" two verbs in one sentence

### Questions
- It's unclear how S_i is obtained and how Q^t is update from Q^{t-1}. Could the authors elaborate on them? 
- What's the size of Q. Do we keep one Q for all tasks or one Q per task?
- "It also mitigates the chance that the query q(x′) is a past task uses the prompts of the current/future tasks, hence the prompts P^t
for the task t have more contribution to the prompts P_x of example x in this t". This is about testing or training? if it’s about training, we don’t see old task sample x’; if it’s about testing, P^t’s contribution to x only depends on q(x) and K^t right? Chance of q(x') use P^t does not impact P^t contribution to x?
- It seems OVA is the key to the performance boost. In Table 3, the authors provided CODA + OVA. Is it possible for the authors to provide the performance of other baselines + OVA?
- "might trigger wrong task classification heads". CIL only has one prediction head right? 
- the prototype sizes N × T × d (100 × 20 × 768)  should be multiplied by 4bytes (float), thus the size is around 6.1MB and image net image size is 224x224x3x 1bytes (uint8). So, it should be 40 images instead of 10, as stated in the paper, right? ACIFAR image is 32x32x3x1 = 3kb. For 10 tasks, the storage of the prototypes is the same as the storage of 1k images right?
- The proposed method KOPPA outperforms JOINT by a large margin. JOINT is supposed to be the upper bound of CL. What's the main reason that KOPPA surpasses JOINT by such a large margin?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper extends the groundwork established by CODA-Prompt, aiming to resolve two primary concerns identified within it: (1) the mismatch in prompt representation between training and testing examples, and (2) the erroneous activation of the task classification head. To address these challenges, the authors introduce a look-ahead orthogonal projection optimization process for the former and employ a one-versus-all loss function for the latter. However, a significant drawback of this paper lies in the absence of important details. Furthermore, the working mechanism of the proposed methods seems not aligned entirely with the claims made by the authors.

### Strengths
- The research focus on continual learning for pre-trained models holds significant significance.
- The analyses delving into the issues concerning CODA-Prompt are particularly intriguing.

### Weaknesses
- Several vital details are missing, preventing a comprehensive evaluation of the proposed method.
  * Specifically, how is $\mathcal{Q}^t$ calculated? What is the specific sample size used for this calculation?
  * How is $g_\phi$ implemented? Is it similar to a cosine distance between the current prototype and previous prototypes?
  * What are the hyperparameters employed, such as the prompt length and the prompt pool size? How does your method compare with CODA when these hyperparameters are adjusted?"
- The operational mechanism of the proposed method appears to diverge from the authors' assertions.
  * Upon juxtaposing the results from Table 3 and Table 4, it becomes evident that the orthogonal projection component is minimally effective; the sole operational aspect is the One-Versus-All (OVA) loss, previously introduced by (Saito & Saenko, 2021). This observation is reasonable, given that the orthogonal constraint has already been applied in CODA, albeit between keys. It seems that the first identified issue has not been adequately addressed by the proposed method.
  * Regarding the OVA loss, I have two hypotheses:
    - The true effective component might be the prototypes of previous tasks, as indicated in Table 5. In this scenario, an additional ablation study, replacing $h_\theta$ with a prototype-based classifier, is necessary. If successful, the unique contribution of this work, the OVA loss, may be rendered not valid any more.
    - Its effectiveness could stem from the similarity between testing and training tasks, which allows for the identification of the most closely related and well-trained training task, thereby resolving the challenge. To validate this, additional experiments involving diverse data splits or datasets are essential to assess the practicality of the proposed methods beyond the current benchmarks."
- The writing is disorganized with a lot of symbols randomly used, quite challenging to follow.

### Questions
See the first weakness listed above

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper aims to address key challenges in Prompt Learning based Continual Learning (CIL) methods in class incremental setting. The authors highlight two potential issues in prior prompt learning based CIL methods: (1) Mismatch between per task final prompt representation during training and inference phase as prompt keys of one task could correlate with other tasks due to no explicit constraint and (2) The triggering of wrong classifier head due to the shift in the sample features of same task between training and inference. 

To ensure that prompt representations of one task remains consistent during the training of prompts for upcoming tasks, the authors propose to enforce the orthogonality constraint between prompt keys of current task with the subspace of previous task. To address the issue of effective classifier head distinction, prototype based method is used which keeps prototypes from all tasks and eventually refine the classification task score. 

The model is finetuned in class-incremental setting with the combination of above techniques and shows improvement against previous methods.

### Strengths
Strengths:

(1) This paper has identified relevant issues and key challenges in prompt learning based CIL methods such as mismatch the between final prompt representation per task during training and testing. . Methods to improve these limitations can greatly enhance the resulting performance and advance the progress in prompt learning based CIL methods.

(2) The idea of imposing orthogonality between prompt keys and previous task sub-keys is motivating. This aims to reduce the impact of prompt keys of new task to calculate prompt representations of previous task, resulting in correct prompt key activation specially during testing. 

(3) The method shows impressive results against previous methods and the proposed technique is motivated with fair ablation studies.

### Weaknesses
Weaknesses:

My main concern for this work lies in potential violation of rehearsal-free CIL experimental rules. 
1) In the first proposed module, the authors keep subspace Qt for upcoming new tasks to ensure the orthogonality constraint. This subspace potentially include sample information from previous tasks till t-1, which means that for the current task, information about the previous task samples is explicitly utilized and this possibly violates the rehearsal-free CIL setting where no information about previous task examples is known.

2) Similarly, for the second proposed OVA technique, prototypes are stored from each task which are feature representations of training examples from each task. Therefore, the authors are indirectly utilizing a buffer in the feature space where the task ID of each task is completely known. I am finding it difficult to understand how this method does not belong to rehearsal-based CIL setting. 


3) The baseline CODA performs additional evaluation on DomainNet which is missing in this comparisons.

### Questions
Please refer to weaknesses section for questions.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
