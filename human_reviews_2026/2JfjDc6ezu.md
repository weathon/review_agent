# Exploring Training Time Modality Incompleteness and Learning from Diverse Modalities

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Multimodal learning benefits from the complementary signals across different data sources, but real-world scenarios often encounter missing modalities, particularly during training. Existing approaches focus on addressing this issue at test time and typically rely on fully co-occurring multimodal data, which can be difficult and costly to collect. We propose a two-stage framework designed to address training-time modality incompleteness without requiring co-occurring samples. The first stage, Data Fusing with Label-guided Mapping (DFLM), constructs a pseudo-multimodal dataset by aligning user data across modalities using supervised contrastive learning guided by shared labels. The second stage, Cooperative Cross-attention Multimodal Transformer (CCAMT), learns from the constructed dataset using a cross-attention mechanism that supports both modality-specific learning and cross-modal interaction with drastically different modalities. An extensive evaluation on three popular datasets (Multimodal Twitter, Multimodal Reddit, and StudentLife) demonstrates that CCAMT significantly outperforms the best-published baselines across all metrics. CCAMT achieves an impressive 96.5% accuracy, significantly outperforming single-modal baselines by up to 10.5% in accuracy. The physical activity data increases the model's accuracy by 2.8%. It also significantly outperforms the state-of-the-art time2vec multimodal transformer by 3% in accuracy, 2.9% in F1 score, 0.9% in precision, and 2.8% in recall. It outperforms other strong multimodal baselines by up to a 7.7% increase in accuracy and a 6.8% improvement in F1 score. Our robustness analysis with imbalanced data evaluation shows that CCAMT can achieve 74.2% accuracy with only 10% of data, significantly outperforming time2vec Transformer (at 47.3%) and SetTransformer (at 50.2%). The Edge deployment evaluation also shows that CCAMT's encoder configuration is up to 83.04% faster than other configurations on an Nvidia Jetson device.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes 1) Data Fusing with Label-guided Mapping (DFLM) that aims to address missing modality during training, and 2) Cooperative Cross-attention Multimodal Transformer (CCAMT) that combines cross-attention and self-attention in learning both modality-specific and cross-modal interaction with drastically different modalities. CCAMT is extensively evaluated on 3 multimodal datasets that have both text, visual, and physical activity data. CCAMT is widely compared to several baselines to demonstrate its effectiveness. Ablation analysis also shows the robustness and efficiency in edge deployment of CCAMT.

### Strengths
- The paper is well-motivated by the research questions of addressing incomplete modality and multimodal learning with drastically different modalities, which are core research problems in multimodal learning with the later (multimodal learning with common modalities such as text, vision, and low-resource modalities such as physical activity data) less frequently explored in multimodal research;
- Methods and experiments are well-documented;
- The proposed method is extensively evaluated on 3 real-world tasks and compared with various baselines and SOTA methods, with reasonable margins that corroborate the effectiveness of the method;
- Ablation analysis on robustness and efficiency demonstrate the advantage and practical utility in real-world deployment;

### Weaknesses
The reviewer has 2 major concerns and is willing to raise scores if the authors can properly address them in the rebuttal:
- One major concern about evaluation is that given the paper proposes 2 stages, DFLM and CCAMT, while only CCAMT (or the entire pipeline) is evaluated in section 4, **the DFLM stage is not evaluated and therefore its effectiveness remains unclear**. Note there are many existing methods aiming to address the challenge of multimodal learning with missing modalities. The paper should at least include a study that ablates the effect of DFLM on the overall performance of the pipeline and compare DFLM with other techniques used to address incomplete modalities. The reviewer also recommends to name the proposed method differently from CCAMT (e.g. DFLM+CCAMT or something else), so the evaluation in section 4 is not confused with evaluation with only the CCAMT stage;
- The formulation of CCAMT resembles many existing notions of cross-modal attention such as in [1, 2, 3], so this is not a new concept and should be clearly differentiate from the existing works by highlighting the extra contribution of CCAMT on top of the existing formulation, with (empirical) evidence supporting those contribution claims;
- Writing: the writing is overall clear and easy-to-follow, but the introduction is a little too lengthy (as the method section only starts at the bottom of page 3). The introduction does not need to contain all the results, instead it should just give a high-level overview of the contributions and interesting findings.

[1] Wang, Meifang, and Zhange Liang. “Cross-modal self-attention mechanism for controlling robot volleyball motion.” Frontiers in neurorobotics vol. 17 1288463. 10 Nov. 2023, doi:10.3389/fnbot.2023.1288463

[2] Tsai, Jia-Hua, and Wei-Ta Chu. “Multimodal Fusion with Cross-Modal Attention for Action Recognition in Still Images.” Proceedings of the 4th ACM International Conference on Multimedia in Asia (MMAsia ’22), Association for Computing Machinery, 2022, Article 31, 5 pp.

[3] X. Wei, T. Zhang, Y. Li, Y. Zhang and F. Wu, "Multi-Modality Cross Attention Network for Image and Sentence Matching," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 2020, pp. 10938-10947, doi: 10.1109/CVPR42600.2020.01095.

### Questions
- Weakness 1: what is the effect of DFLM? How is its effectiveness compared to other existing approaches in addressing multimodal learning with missing modalities?
- Weakness 2: how is CCAMT different from existing formulation of cross-modal attention?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the challenge of modality incompleteness during training in multimodal learning through a two-stage framework. In the first stage, **D**ata **F**using with **L**abel **G**uided **M**apping (**DFLM**), the model constructs pseudo-multimodal datasets by pairing users from separate single-modal datasets (online activity and physical activity) using supervised contrastive learning based on shared depression labels. In the second stage, **C**ooperative **C**ross **A**ttention **M**ultimodal **T**ransformer (**CCAMT**), it integrates cross-attention for inter-modal interaction and self-attention for intra-modal representation learning. Experimental results demonstrate consistent improvements over baselines and show the feasibility of the proposed approach for edge deployment.

### Strengths
- The paper provides comprehensive experimental results and analysis supporting its main contributions
- The proposed approach is evaluated on a variety of tasks, including general applications (e.g., classification) as well as deployment on edge devices, demonstrating its broad applicability.

### Weaknesses
1. The main contribution of this paper appears somewhat incremental. Specifically, there already exist several prior works that incorporate both cross-attention and self-attention mechanisms during training to effectively leverage intra- and inter-modal information [1][2][3].
2. The comparisons in Table 1 and Table 2 seem potentially unfair due to differences in the number of modalities. Moreover, the proposed model shows slightly degraded performance under equivalent settings. For instance, on the Twitter dataset, Time2VecTransformer very slightly outperforms CCAMT w/o physical (T+I modalities) in terms of F1, Precision, and Accuracy. 
3. In my understanding, the paper lacks clear justification and consistency in its analysis settings. For example:
    - The robustness analysis in Section 4.2 (particularly Fig.2e) focuses on class imbalance between depressed and non-depressed categories. However, in recent works, robustness typically refers to Out-of-Distribution (OOD) generalization or adversarial settings. This may lead to confusion regarding the interpretation of “robustness.”
   - Fig.2f presents the degradation of performance under label shuffling but does not clearly demonstrate whether CCAMT is robust to label noise. The message conveyed is thus unclear.
   - The edge-device analysis in Section 4.3 appears to hinder the paper’s main contribution, as it lacks comparative evaluations against other architectures to substantiate whether CCAMT achieves superior inference efficiency.

[1] Wei, Xi, et al. "Multi-modality cross attention network for image and sentence matching." CVPR 2020\
[2] Quan, Weize, et al. "TCAN: Text-oriented cross attention network for multimodal sentiment analysis." Preprint 2025\
[3] Zhang, Zhicheng, et al. "Moda: Modular duplex attention for multimodal perception, cognition, and emotion understanding." ICML 2025

**Minor Concern** (not affect the rating)
- The combination of Online Activity datasets (Twitter/Reddit) and Physical Activity datasets (StudentLife) is unconventional and not directly comparable to such general multimodal benchmarks (e.g., CREMA-D, Kinetics-Sounds, CMU-MOSEI). In my point of view, this makes it slightly difficult to assess how CCAMT would perform on widely-used evaluation protocols or to compare against the broader multimodal learning.
- Typos
  - line 198:  online embedding $\ z_i^{\text{online}}\$} $\rightarrow$ {$\ z_i^{\text{online}}\$}
  - Eq.(2), Eq.(3): The font size appears inconsistent and should be standardized.

### Questions
- **Clarification on Weaknesses 1**: How does the proposed method differ fundamentally from existing attention-based multimodal frameworks?
- **Clarification on Weaknesses 2**: How was fairness ensured in the comparisons, particularly regarding the number of modalities used? Does this imply that the improvements are primarily due to access to additional modality data rather than the effectiveness of the proposed approach itself?
- **Clarification on Weaknesses 3**: How do the authors define “robustness” in this context, and how does it relate to standard OOD or adversarial evaluations?
- Have the authors considered validating CCAMT on other multimodal learning tasks beyond classification, such as retrieval or visual question answering (VQA)?
- Additional experiments could further substantiate the robustness claims. For instance, incorporating adversarial attack evaluations would strengthen the paper that CCAMT is a robust model [1][2][3]

[1] Yin, Ziyi, et al. "Vlattack: Multimodal adversarial attacks on vision-language tasks via pre-trained models." NeurIPS 2023\
[2] Dou, Zhihao, et al. "Adversarial attacks to multi-modal models." ACM Workshop LAMPS 2024\
[3] Cui, Xuanming, et al. "On the robustness of large multimodal models against image adversarial attacks." CVPR 2024\

=======================================================

**Note**: I acknowledge that I may have partially misunderstood certain aspects of the paper. Therefore, I am willing to raise my rating score if these questions and concerns are adequately addressed.

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
3

### Summary
In this work, the authors proposed a novel pipeline that allows incorporation of drastically different data from another modality without co-occurrence to improve multimodal classification performance. Specifically, the work explored incorporating physical data into online data for depression detection. The proposed method first uses DFLM to pseudo-align each online user with some physical data, then devised a new multimodal fusion architecture called CCAMT that has both cross-modal attention and self attention within each modality. The proposed method was evaluated on two depression datasets with online content, and used one physical activity-depression dataset as auxiliary physical modality, and the paper showed that the proposed method outperforms all baselines, and that incorporating physical data improves performance. Additional analysis was performed to demonstrate the proposed method's robustness to fewer training data and incomplete modalities.

### Strengths
1. The paper explored a very interesting direction of incorporating un-aligned data from a completely different modality (but about the same task) to improve model performance on the task. The idea is innovative, and the experiments demonstrated its validity.

2. The proposed fusion module, CCAMT, although simple, achieved strong performance against baseline fusion models.

3. There is extensive analysis on the proposed method's robustness under partial training data or modality imbalance, and it is also nice to have the encoder-choice latency study to help future researchers select their encoders.

### Weaknesses
1. The proposed method is only designed for and evaluated on one particular domain/task: depression detection. Thus, it is completely unclear whether this idea of incorporating a very different modality on the same task without co-occurrence can be applied to any other multimodal domains or tasks at all, and the application generalizability of this method may be rather limited.

2. It is not very clear to me why the contrastive approach for DFLM works. It seems like the contrastive objective is trained over randomly paired data within each label group over the same set of data as the one we try to pair with similarity later, so if the projection modules are trained to convergence with the InfoNCE objective, wouldn't the projection layers just memorize the random pairings, and the similarity-based mapping would just be identical to the random pairings?

### Questions
1. In line 174, all 3 X have different sequence lengths, but after passing them through pre-trained encoders, all of a sudden all 3 modalities have sequence length T. How does this happen? I don't think the arbitrarily selected pre-trained encoders (like the ones in the experiments) can guarantee that all 3 modalities have the same encoded sequence length. Do you perform the 1D interpolations here instead of later (as in lines 192-193)?

2. Following the question above, if the sequence lengths are already matched to T in line 174 for the three E, then the z in line 190-191 will always have matched sequence length. Is the 1D linear interpolations (on line 192-193) still necessary?

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
2

### Summary
This paper proposes a two-stage framework to address the highly practical challenge of training-time modality incompleteness, where real-world modalities (like social media and wearable data) are often collected from different, non-overlapping user cohorts. To address this, the authors propose to perform Data Fusing with Label-guided Mapping (DFLM), which uses shared labels (e.g., "depressed") and Supervised Contrastive Learning to map disparate modal embeddings into a shared latent space. To effective learn from the generated pseudo multimodal datasets, a cooperative cross-attention transformer is presented. Comprehensive evaluations across diverse datasets and deployment settings, demonstrating the effectiveness and robustness of our framework.

### Strengths
- The paper is easy to follow and the proposed solution is technically sound;
- Implementation details, hyperparameters, and encoder combinations are clearly reported;
- The combination of label-guided contrastive alignment and cross-attention fusion  is conceptually coherent.

### Weaknesses
Weakness & Questions:
- Cross-attention fusion. It is common to use Cross-attention for multimodal fusion, and CCAMT’s architecture is a bit similar to standard cross-attention Transformers with bidirectional connections (Multimodal Transformer). Are there any specific designs in CCAMT?
- There’s no analysis of failure cases. For example, when modalities are weakly correlated or when label noise disrupts alignment.
- For better verify the effectiveness of the proposed approach, please conclude more baselines for comparison, e.g., CycleGAN-style modality transfer, CMMD (contrastive-based);
- Running efficiency comparison. Please also include more quantitative evaluation of inference & running efficiency.

### Questions
Please see weakness.

### Soundness
2

### Presentation
2

### Contribution
2
