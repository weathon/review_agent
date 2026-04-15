# X-Fi: A Modality-Invariant Foundation Model for Multimodal Human Sensing

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6, 6

## Abstract
Human sensing, which employs various sensors and advanced deep learning technologies to accurately capture and interpret human body information, has significantly impacted fields like public security and robotics. However, current human sensing primarily depends on modalities such as cameras and LiDAR, each of which has its own strengths and limitations. Furthermore, existing multimodal fusion solutions are typically designed for fixed modality combinations, requiring extensive retraining when modalities are added or removed for diverse scenarios. In this paper, we propose a modality-invariant foundation model for all modalities, X-Fi, to address these issues. X-Fi enables the independent or combinatory use of sensor modalities without additional training by utilizing a transformer structure to accommodate variable input sizes and incorporating a novel "X-fusion" mechanism to preserve modality-specific features during multimodal integration. This approach not only enhances adaptability but also facilitates the learning of complementary features across modalities. Extensive experiments conducted on the MM-Fi and XRF55 datasets, employing six distinct modalities, demonstrate that X-Fi achieves state-of-the-art performance in human pose estimation (HPE) and human activity recognition (HAR) tasks. The findings indicate that our proposed model can efficiently support a wide range of human sensing applications, ultimately contributing to the evolution of scalable, multimodal sensing technologies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a multi-modal foundation model that integrates six distinct modalities to enhance human sensing capabilities. The proposed model demonstrates the scalability of various modality combinations and incorporates an X-Fusion block to fuse multi-modal features while preserving modality-specific information. The authors conduct extensive experiments on two multi-modal datasets to evaluate the model's performance in human pose estimation and human action recognition tasks.

### Strengths
1. The foundation model of different human sensing modalities is a promising approach, and the author made a meaningful attempt. The training schedule of this model is interesting

### Weaknesses
1. The section of problem definition is missing. This paper addresses two different tasks, which generally require different input formats of modalities. For instance, HAR requires video clips, but HPE could be video clips or frames. The clear details about the inputs and outputs are crucial for reader to understand.
2. There are some undefined notations, such as  in line 172 and  in Eq. 4.
3. The motivation regarding the module in subsection 3.2.2, which is to pay equal attention to each modality, is unconvincing. Different modalities offer varying degrees of human sensing information, and only valuable modality or information are important. The model paying more attention to dominant modalities is more intuitive. Equal attention could lead to the inclusion of irrelevant information and the potential omission of key details.
4. The cross-attention module is intended to accentuate modality-specific information. Nevertheless, the absence of ablation studies on this module precludes a conclusive evaluation of its efficacy, and there is no empirical evidence to substantiate the modality-specific features extracted into the K-V pairs. Some additional constraints are necessary. For instance, SimMMDG [1] leverages distance loss between modality-specific and modality-shared features and the orthogonal constraints is adopted in [2].
5. No ablation study on the components of proposed method.
6. The experimental results of Lidar in MM-Fi reveal inferior performance relative to mmWave in Tab. 1. This is in contrast to the results in MM-Fi under same experimental setup and backbones.  And it is also not intuitive as Lidar data, characterized by its higher point density, offers a more precise and detailed representation of the human compared to mmWave.
7. To ensure a fair comparison in Tab. 4, the number of parameters in the transformer and X-Fusion block should be equalized.

8.You employ multiple layers of cross-modal transformers to obtain cross-modal embeddings. How does the number of these iterative layers，N, impact the experimental results? Additionally, how might varying the number of transformer layers for different modality counts influence the outcomes?

9. The proposed method in the paper enforces a balance across all modalities. Could this forced balancing lead to a negative impact in tasks that primarily rely on a specific modality, especially if some modalities contain less informative data?

10. Performing computations across all modalities in each iteration could result in high computational complexity, particularly in multi-modal processing and long-sequence modeling. This increased demand could pose challenges for real-time applications, making practical deployment of the method more difficult.


[1]Dong H, Nejjar I, Sun H, et al. SimMMDG: A simple and effective framework for multi-modal domain generalization[J]. Advances in Neural Information Processing Systems, 2023, 36: 78674-78695.
[2]Jiang Q, Chen C, Zhao H, et al. Understanding and constructing latent modality structures in multi-modal representation learning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 7661-7671.

### Questions
Please see the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a modality-invariant foundation model that allows for the integration of various sensor modalities for human sensing. The proposed X-Fi model employs a transformer architecture with a "X-fusion" mechanism to preserve modality-specific features in the multimodal integration process. The proposed model is designed to support any combination of sensor modalities with flexibility of the input dimension without requiring additional retraining. They evaluated the proposed method on two benchmark datasets, MM-Fi and XRF55, covering six distinct modalities, with two downstream tasks: human pose estimation and human activity recognition. The experimental results showed that the proposed X-Fi model improved the performance of HPE and HAR.

### Strengths
+ This paper addresses the important problem of modality-invariant human sensing by designing a foundational model with flexiblity without retraining. 
+ The paper is clearly written and well-structured, effectively communicating the motivation, methodology, and results. Concepts such as modality-specific feature extraction, cross-attention fusion, and the role of the X-fusion block are clearly explained, The transformer-based architecture handles multimodal input data and X-fusion block is used for cross-modal feature fusion by using cross-attention.

### Weaknesses
- The paper lacks a thorough investigation of related cross-modal / modality-invariant foundational models, such as CLIP [1], PaLM-E [2], Meta-Transformer [3], ImageBind [4], IMU2CLIP [5], and zero-shot learning for HAR [6]. The authors need to investigate recent cross-modal foundational models.
- While the paper proposes the X-fusion block for multimodal fusion, the core approach of using transformers and cross-attention for feature fusion is not new. Similar attention-based fusion methods are widely used in the literature for feature-level integration. The paper could strengthen the novel contributions and clarify the novel aspects of X-fusion compared to existing methods.
- The evaluation experiments focus on comparing the proposed method with basic feature and decision-level fusion approaches. The proposed foundational model may be compared with other cross-modal foundational models.
- The evaluation was conducted on MM-Fi and XRF55 datasets, which primarily include specific modalities like cameras, WiFi, LiDAR. However, human activity recognition often involves wearable sensors like IMUs. It is better to add one or two more benchmark datasets with IMU sensors.


[1] Radford, Alec, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry et al. "Learning transferable visual models from natural language supervision." In International conference on machine learning, pp. 8748-8763. PMLR, 2021.
[2] Driess, Danny, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid et al. "PaLM-E: An Embodied Multimodal Language Model." In International Conference on Machine Learning, pp. 8469-8488. PMLR, 2023.
[3] Zhang, Yiyuan, Kaixiong Gong, Kaipeng Zhang, Hongsheng Li, Yu Qiao, Wanli Ouyang, and Xiangyu Yue. "Meta-transformer: A unified framework for multimodal learning." arXiv preprint arXiv:2307.10802 (2023).
[4] Girdhar, Rohit, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. "Imagebind: One embedding space to bind them all." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15180-15190. 2023.
[5] Moon, Seungwhan, Andrea Madotto, Zhaojiang Lin, Aparajita Saraf, Amy Bearman, and Babak Damavandi. "IMU2CLIP: Language-grounded Motion Sensor Translation with Multimodal Contrastive Learning." In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 13246-13253. 2023.
[6] Tong, Catherine, Jinchen Ge, and Nicholas D. Lane. "Zero-shot learning for imu-based activity recognition using video embeddings." Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 5, no. 4 (2021): 1-23.

### Questions
1. Can the authors provide additional details on the distinguished contribution points of the X-fusion block compared to existing cross-attention fusion approaches? 
2. Would the authors consider conducting further experiments with other cross-modal foundational models (e.g., Meta-Transformer, ImageBind) as baselines to provide a more comprehensive comparison?
3. Is it possible to include additional benchmark datasets with IMU data for HAR task?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors propose a multi-modal modality invariant foundation model to tackle the problem in the current human sensing techniques. They introduced X-Fi block to combine the feature representation from different modalities.  The authors utilized two datasets to verify the effectiveness of the proposed model on human action recognition and pose estimation. The proposed model outperformed the other methods.

### Strengths
The paper presents a simple yet effective framework for multi-modal fusion from heterogenous modalities.
Two large datasets were adopted in the experiments.  
The authors conducting experiments on  different modality combinations.

### Weaknesses
1- The paper needs polishing and revision. For example in the first page (line 043) "Alternative modalities like LiDAR, mmWave radar,
and WiFi have be introduced to overcome these challenges" --> it should been not be. Additionally, the authors are mixing Fig with Figure. Or capital letters within the sentence--> Line 233 "Notably, The multi-head" and Line 259.
2- The authors mentioned in the contribution and abstract the proposed model surpassed the SOTA. I did not find a table of comparison against the SOTA especially the only comparison is against baseline.
3- The authors did not provide the experimental setting. For example, MM-Fi [1] has three settings and it is not clear which setting is adopted. I recommend to adopt the three settings.


[1]Yang, Jianfei, He Huang, Yunjiao Zhou, Xinyan Chen, Yuecong Xu, Shenghai Yuan, Han Zou, Chris Xiaoxuan Lu, and Lihua Xie. "Mm-fi: Multi-modal non-intrusive 4d human dataset for versatile wireless sensing." Advances in Neural Information Processing Systems 36 (2024).

### Questions
1- I suggest to have a joint loss ( L= L_I+ L_R+L_W+....L_N). In other words, to have loss for each modality and the X-Fusion output as well. It should boost the performance.  
2-   I suggest an ablation study on the number of  stacked blocks in X-Fi  block. 
3- I was expecting an experiment which gradually add new modality each time. For example the model is trained first on RGB then WiFI is  added by fixing the parameters and finetuning the model after adding the new modality.

### Soundness
2

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
This paper introduces a model to perform human sensing on various input modalities and output tasks. The method utilizes transformer-based attention to input a varying number of modalities and learn complementary features. The method can perform during test time with any combination of input modalities better than a fusion architecture trained with those modalities, which indicates a step towards developing a multi-sensor foundation model for human motion understanding.


This review follows the sections of the paper.

### Strengths
Introduction:

1. The need for a flexible multimodal human sensing model is well-motivated. Particularly, the second paragraph outlining specific modalities and how their strengths and weaknesses can compensate for each other is strong.

Modality Invariant Foundation Model for Human Sensing:

2. The motivation for the cross-attention multimodal fusion (X-Fusion) is good. Attempting to combat the domination of certain modalities by incorporating modality-specific information is good. How did the authors know that the attention mechanism was weighing some modalities more than others? Did they view the transformers' attention weights and notice a trend? Did previous works report that happening? It would be great to see these preliminary results somewhere.
3. I appreciate the thought and structure that went into randomizing the modalities in "3.3 Modality-invariant training." Table 3 is good. How did you determine the final probabilities? How did you know it was necessary to include more LiDAR samples than the other modalities?

Experiments:

4. Just to clarify, in tables 1 and 2 X-Fi was only trained once and tested on various input modalities (each unique row), but the baselines were retrained on each modality combination separately. Is this true (line 365)? If this is the case, I think this should be noted in the caption and is a strength for your method.
5. I like Figure 4's clusterability scores. The visualization makes it hard to distinguish the difference between R and W+R, but I think the numbers emphasize the point that multimodal is better.

### Weaknesses
Introduction:

1. Figure 1: I am not sure what the purpose of this figure is. What are you trying to show? X-Fi can replace a bunch of modality-specific models? But within X-Fi we have all the modality-specific encoders anyways. Are you trying to indicate that you don't need to retrain models for each combination of modalities? Nothing in this figure indicates retraining. I would suggest brainstorming a more informative diagram.
2. Figure 2 is good, but could use some clarifications. Why are the last 3 modalities with dotted lines? Can you label the dark gray background boxes (which module is the X-Fusion part)? Also the text is quite small and hard to read. What does N stand for in the arrow loop?
3. I think the term "foundation model" is starting to be used loosely quite frequently, and I would appreciate if the authors could clarify in a sentence or two. Maybe define a foundation model in this context and how their method is such before referring to their method as a foundation model. This could also be done in related works, comparing your method to any related foundation models.

Related Works:

4. I think a discussion of multimodal missing modalities literature is missing. There are many works that perform multi-modal HAR in the face of missing modalities during training/testing which is a similar setup to X-Fi. Also, there is a lot of literature on transformer based fusion that could be mentioned here.
Here are some potentially relevant works:
- Sangmin Woo, Sumin Lee, Yeonju Park, Muhammad Adi Nugroho, and Changick Kim. Towards
Zihui Xue, Zhengqi Gao, Sucheng Ren, and Hang Zhao. The modality focusing hypothesis: Towards understanding cross modal knowledge distillation. arXiv preprint arXiv:2206.06487, 2022.
- Quan Kong, Ziming Wu, Ziwei Deng, Martin Klinkigt, Bin Tong, and Tomokazu Murakami. Mmact: A large-scale dataset for cross modal human action understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8658–8667, 2019.
- Qi Wang, Liang Zhan, Paul Thompson, and Jiayu Zhou. Multimodal learning with incomplete modalities by knowledge distillation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 1828–1838, 2020.
- XB Bruce, Yan Liu, and Keith CC Chan. Multimodal fusion via teacher-student network for indoor action recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35(4), pp. 3199–3207, 2021.

5. In the second sentence under Modality invariant methods. I would disagree that decision-level approaches are restricted to fixed modality combinations, and both citations provided are feature level fusion. Decision level methods may average the outputs of separate modality modals, or perform some sort of voting, so it would be easy to extend to more modalities or drop the outputs of some modalities. In fact I think that's what your baseilne1 does to my understanding.
6. I don't think the discussion in the last paragraph citing LLM methods and a visual odometry method in robotics is very relevant.

Modality Invariant Foundation Model for Human Sensing:

7. A lot of notation is introduced and it is hard to follow. Consider maybe labeling the arrows and boxes in figure 2 with the notation ($G_{ca}, G_{cm}, Emb_{cm}$ etc.). Also maybe the dimensions of the cubicle blocks so we can see what n_f and d_f is referring to. Is $\theta_{Enc}$ Just the concatenation of $\theta_{E}$ and $\theta_{LP}$? Is positional encoding shown in Equation (1)?
8. The key value generators and cross-attention transformers is unclear. It would be helpful to provide a diagram (at least in the appendix) for them. Also, when using the term transformer, I assume the input is one set of tokens and that's projected into keys, queries and values, but "in cross-attention transformer," the input is already K,Q,V's. I would suggest referring to it as just a cross-attention module. The original Transformer in Vaswani et al (the only cited Transformer paper in 3.2.1) includes the KQV projection as well as an encoder with self-attention and a decoder with self-attention/cross attention and multiple heads. Is that what your "Transformer" is doing?
9. In line 272, I don't understand why Emb_{cm} is being added back to O_i ? Again a diagram here might help.
10. I would suggest an ablation without the K,V generators. It is just two MLP's but it happens right after the modality-specific linear projections, so maybe the modality-specific projections can capture that information.

Experiments:

11. In Tables 1 and 2 I would suggest labeling the reported metrics either in the column header or caption (it sounds like MSE for HPE and Accuracy for HAR?).
12. HPE Qualitative results Figure 3. It is not apparent to me that RGB captures better leg movements and depth is better for hands. Can you circle on the images where you are observing this? Also, any thoughts on why this would be the case? Maybe even providing numbers for the errors on hand joints vs leg joints could make this result more convincing!
13. The results could be strengthened by comparing to other multimodal methods. Are there other works that perform multimodal HPE or HAR on the MM_Fi and XRF55 datasets that you can compare to?

### Questions
I presented many questions and suggestions in the sections above. I think the main issue of the paper was a lack of clarity in describing the method and its implementation.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents X-FI, a foundation model for multi-modal human sensing. X-FI allows inference with versatile combinations of sensor modalities, and supports multiple tasks including pose estimation and activity recognition. Key to X-FI is a multi-modal feature encoding module that builds on cross-attention. X-FI was evaluated on two public datasets, demonstrating some encouraging results.

### Strengths
The paper addresses an interesting and important problem of developing foundation models for multi-modal human sensing. 

There are some interesting ideas in the model design, e.g., using cross-attention to iteratively refine the average pooled multimodal features.

### Weaknesses
I am not convinced by the claim on “modality invariant human sensing”, one of the key contributions of the paper. Conceptually, there are several approaches for modality invariant processing. For example, the model in Memmel et al., 2023 (a Transformer that takes tokens from any combination of modalities) can in theory be adapted for human sensing. Technically, this invariance is achieved by dropping out modalities during training, which is again discussed in Memmel et al., 2023 and also mentioned Chen et al., 2023.

The experiments miss important baselines. There is hardly a comparison to prior methods in the paper. At least, two sets of baselines should have been considered: (a) latest methods using a single modality (to demonstrate the benefit of multi-modal fusion); and (b) previous methods that also considers multimodal fusion such as Chen et al., 2023 (to highlight the benefit of the proposed method). The current baselines are rather straw man.  

The presentation of the technical components lacks clarity. Some of the designs are not well justified. 
* Eq 2: The use of average pooling is not justified. In fact, with different numbers of input modalities, each bin in average pooling may span (1) tokens from a single modality; or (2) tokens from more than one modality. It is not clear why average pooling works here. One possibility is learning a fixed number of queries and using cross-attention to project the input multi-modal feature, similar to the design of Percerver IO.
* Eq 4: It is unclear how to determine the number of iterations for the X-fusion block. I also could not find the exact number used in the experiments. 
* L229: I assume n_f is the number of tokens per modality, but it was not described in the paper.

================================

Post-rebuttal update: In the rebuttal, the authors have addressed my previous concerns, included substantial updated results, and majorly improved the paper. While I still have some concerns over the technical innovation and the use of average pooling, I feel like the paper is now beyond the bar. Therefore, I have increased my rating.

### Questions
Can you clarify the number of iterations needed for the X-fusion block?

It will be interesting to compare average pooling vs. cross-attention with learned queries.

Comparison to prior methods (see weaknesses) should be included.

### Soundness
3

### Presentation
2

### Contribution
2
