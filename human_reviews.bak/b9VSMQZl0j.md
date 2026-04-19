# General Skeleton Semantics Learning with Probabilistic Masked Context Reconstruction for Skeleton-Based Person Re-Identification

- Decision: Reject
- Scores: 8, 8, 6, 5, 6

## Abstract
Person re-identification (re-ID) via skeleton data is an emerging topic with immense potential for safety-critical applications. Existing methods usually utilize spatial or temporal skeleton semantics learning (SSL) tasks to facilitate skeleton representation learning, while most SSL tasks are *model-dependent* and lack the ability to capture general fine-grained (*e.g.*, joint-level) spatial-temporal skeleton patterns under different model architectures. To delve into multi-faceted generality of SSL tasks, we first propose an SSL generality assessment framework termed **SCUT** that identifies four key SSL properties: **S**patial-temporal effectiveness, **C**o-training compatibility, **U**nsupervised trainability, and **T**ask transformability. By formulating systematic evaluation criteria for each property, SCUT enables both qualitative and quantitative analysis of SSL generality under varying models and scenarios. Motivated by SCUT to fully harness skeleton context for semantics learning, we further devise a generic **Pro**babilistic **M**asked S**p**atial-**T**emporal cont**e**xt **R**econstruction (**Prompter**) task to enhance performance of skeleton-based person re-ID models. Specifically, Prompter first probabilistically and independently masks joints' structural locations to generate *spatial context*, and then randomly conceal their motion trajectories to form *temporal context*. 
Through combining both spatial and temporal skeleton context representations to jointly reconstruct and infer skeleton sequences, Prompter encourages the model to capture general valuable spatial-temporal skeleton patterns for person re-ID. Empirical evaluations on SCUT and five benchmark datasets demonstrate the superiority of Prompter to most state-of-the-art SSL tasks. We further validate its general effectiveness in different skeleton modeling, RGB-estimated or cross-domain scenarios

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a new generality assessment framework and a probabilistic masked spatial-temporal reconstruction method. Specifically, the assessment framework encompasses four properties (CTC, STE, UT, TT) from quantitative and qualitative perspectives to evaluate the generality of methods. Then, the probabilistic masked strategies are utilized in the skeleton at the spatial and temporal levels and reconstructed to learn task-agnostic skeleton representation. Extensive experiments show the method achieves superior results.

### Strengths
1. The idea of this paper is exciting, especially in the assessment framework aspect.
2. The proposed assessment framework should be appreciated, as it can be helpful in the skeleton representation learning field.
3. The experimental results achieve the sota.

### Weaknesses
1. The indicator used in the assessment framework lacks a reasonable explanation.
2. The proposed assessment framework is not evaluated in the supervised/self-supervised skeleton representation learning methods.

### Questions
1. I have a doubt concerning the quantitative properties. The CTC and STE utilize the Rank-1 accuracy as the performance indicator. However, the skeleton semantic representation should be evaluated at the feature level, not the logits of the output. Why did the authors choose to use this indicator? I think some feature cluster metrics, such as feature silhouette sore and FMI score, may be more reasonable. 
2. The authors mentioned that their method has unsupervised trainability. Therefore,  the authors should evaluate the zero-shot capability of their method as they have described in line 235 that the UT can guarantee the SSL task applies to unknown classes.
3. The mask-reconstruction pipeline has been employed in many studies; what motivates using probabilistic masking in this method rather than fixed ratio mask strategies such as MAE? Meanwhile, what is necessary to leverage them into spatial and temporal levels, respectively, should be described clearly.
4. Does this assessment framework only suit the task of person re-identification? The authors should add the experiments to evaluate them in the supervised/self-supervised skeleton representation learning methods.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper addresses the task of Skeleton Semantics Learning (SSL) with two contributions. First, it proposes a framework to assess the generality of a SSL model with 4 criteria: Spatial-temporal effectiveness, Co- training compatibility, Unsupervised trainability, and Task transformability. Second, it proposes Prompter, a novel idea that can be integrated into any SSL model to enhance the discriminative power of extracted skeleton features, where spatial and temporal context are masked and then reconstructed and inferred.

### Strengths
- The proposed framework for assessing the generality of a SSL model is a great contribution. The 4 suggested criteria are reasonable and can be applied to many other tasks as well.
- The proposed idea of probabilistically and independently mask the spatial and temporal context of skeletons and then reconstruct and infer the sequence, though sounds simple, is reasonable and shows great improvement when being coupled with previous SSL methods.
- Valuable aspects of the framework are discussed and analyzed, including Loss Visualization, Cross-domain evaluation, ...
- Writing is clear and technical details are correctly presented.

### Weaknesses
- References are not adequate. Literature in Skeleton-Based Person Re-Identification is quite richer than what has been covered in the paper. For example, see the papers below.
- The chosen datasets are specifically designed for SSL, while the paper claims the paper contributes significantly to Person Re-ID, thus, experiments on Re-ID datasets should be conducted to support the claim and make the paper more convincing.
- Results comparison are not comprehensive enough, where barely any direct Re-ID method is taken into consideration. 
- SSL should be widely applied in Gait Recognition task but not limited to Person Re-Identification. Thus, results comparison or results of integration of Prompter into Gait Recognition method would be necessary. 

[1] Wang et al., Human skeleton mutual learning for person re-identification. In Neurcomputing, 2021.
[2] Lu et al., Exploring high-order spatio–temporal correlations from skeleton for person Re-identification. In TIP, 2023.
[3] Nguyen et al., Attention-based shape and gait representations learning for video-based cloth-changing person re-identification. In VISIGRAPP, 2024.

### Questions
Please see the weaknesses, which also include my main questions/concerns.

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
3

### Summary
The paper addresses person re-identification (re-ID) using skeleton-based methods, which offer advantages like smaller data input and better privacy protection. Traditional methods rely on visual features, while skeleton-based approaches focus on body structure and motion semantics. The authors introduce a systematic generality assessment framework called SCUT to evaluate self-supervised learning (SSL) tasks. They propose a novel SSL task, Prompter, which uses probabilistic spatial and temporal context masking to enhance skeleton sequence reconstruction. The study highlights the importance of spatial-temporal effectiveness and model compatibility in SSL, demonstrating Prompter’s effectiveness across various datasets and models.

### Strengths
1. The authors identify four general assessment methods for evaluating skeleton semantic learning, which brings new insights into this topic.
2. The extensive experiments demonstrate that the proposed Prompter achieves better results over different model architectures and benchmarks.

### Weaknesses
1. Since skeleton-based person re-ID is fundamentally similar to skeleton-based gait recognition, the authors should compare their work with more recent gait studies in terms of method design and accuracy, such as SkeletonGait [1], GPGait [2], and CAG [3]. This comparison is crucial to assess the superiority of their approach.

2. The proposed Prompter technique, which involves spatial-temporal masking for skeleton sequences, has been extensively explored in previous works like SkeletonMAE [4] and MS2L [5]. These methods have studied masking skeleton joints in spatial and temporal dimensions. Consequently, the novelty of this masking technique appears limited.

[1] Fan, Chao, et al. "SkeletonGait: Gait Recognition Using Skeleton Maps." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 2. 2024.

[2] Fu, Yang, et al. "Gpgait: Generalized pose-based gait recognition." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[3] Huang, Xiaohu, et al. "Condition-adaptive graph convolution learning for skeleton-based gait recognition." IEEE Transactions on Image Processing (2023).

[4] Wu, Wenhan, et al. "Skeletonmae: Spatial-temporal masked autoencoders for self-supervised skeleton action recognition." 2023 IEEE international conference on multimedia and expo workshops (ICMEW). IEEE, 2023.

[5] Lin, Lilang, et al. "Ms2l: Multi-task self-supervised learning for skeleton based action recognition." Proceedings of the 28th ACM international conference on multimedia. 2020.

### Questions
1. The recent skeleton-based gait recognition papers should be cited and compared, as they are strongly related to this work (see weaknesses).

2. The proposed masking method has limited novelty since some relevant papers have studied this topic extensively (see weaknesses).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper addresses person re-identification (re-ID) via skeleton data, aiming to improve generalizability in safety-critical applications. The authors introduce SCUT, an SSL generality assessment framework that identifies four key properties: Spatial-temporal effectiveness, Co-training compatibility, Unsupervised trainability, and Task transformability.  Furthermore, they introduced Prompter task to probabilistically masks joint positions and motion to capture robust spatial-temporal patterns.

### Strengths
1.This paper is well organized. Writing is clear and easy to follow.
2.The experiments are extensive and thorough, providing convincing evidence of the proposed method's effectiveness. The algorithm consistently outperforms baselines across various settings, demonstrating its potential and robustness in skeleton-based person re-ID.

### Weaknesses
The motivation for reconstructing individual joint locations or trajectories in joint-level is unclear. While part-level or body-level dynamics provide intuitive insights into human movement, joint-level information can be noisy. In datasets like BIWI-W, where the state-of-the-art Rank-1 accuracy is only 34.6% across 50 individuals, human poses can be ambiguous, suggesting limitations in relying on fine-grained joint-level details for re-ID tasks.

### Questions
1. Could you clarify the motivation for using Prompter? Specifically, is there empirical evidence that compares the effectiveness of joint-level reconstruction against part-level or body-level reconstruction approaches?
2. Are there any qualitative examples or insights into how Prompter’s performance generalizes across datasets with different characteristics or collection methods? For instance, while IAS-A and IAS-B achieve MAP scores of 34.1 and 43.8 and Rank-1 scores of 39.5 and 60.4, respectively, when trained individually, these metrics decrease to MAP scores of 19.1 and 18.5 and Rank-1 scores of 35.8 and 34.9 when transferred from BIWI. Could you clarify which features Prompter learns that transfer well across datasets and which are specifically learned within each dataset?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors propose the SCUT framework that identifies four key properties (STE, CTC, UT, TT) to assess the generality of skeleton semantics learning (SSL) tasks across different models and scenarios. 

Based on SCUT, they further devise a generic SSL task, i.e., Prompter, to probabilistically and independently mask the spatial context of structural locations and temporal context of motion trajectories, which are exploited to reconstruct and infer complete skeleton sequences to capture general effective spatial-temporal skeleton semantics for person re-ID.

### Strengths
+ The insight of this work is novel and distinctive.

+ The writing of this paper is satisfactory.

+ The proposed SCUT is the first SSL generality assessment framework, which may be useful for the community.

+ The proposed Prompter is effective and adaptive generally for person Re-ID methods as shown in the experimental results.

### Weaknesses
- The proposed Co-Training Compatibility (CTC), Spatial-Temporal Effectiveness (STE), Task Transformability (TT), and Unsupervised Trainability (UT) are reasonable for SSL assessment. My question is whether they are necessary or complete for it.

- The proposed method Prompter is straightforward, which is a classical mask-and-reconstruct strategy.

- The Qualitative Attributes including UT and TT are also very simple to set as Yes or No. 

- The different ranges and calculation methods of the 4 terms in SCUT make the parameter determination among these four terms is not easy.

- The paper layout can be further improved. For example, Tables 1 and 2 are far from their descriptions.

### Questions
As shown in Eqs. (4) and (5), we can see that G_ST has included G_C. This way, whether such two terms are both required?

### Soundness
3

### Presentation
3

### Contribution
3
