## Human Reviewer 1

### Summary
This work proposed G-SFormer, which introduces GNN and Transformer to facilitate 3D human pose estimation. G-SFormer consists of three modules: Spatial Graph Encoder for part-based structural learning within each frame, Skipped Transformer Encoder, and Decoder for hierarchical extraction and aggregation of temporal features. The authors also propose effective data completion methods, which are parameter-free and easy to implement. Experiments are conducted on 3 widely used datasets.

### Strengths
1. The method is clearly described
2. Based on the evaluation, the overall quality of the results seems to be satisfactory.

### Weaknesses
### 1.Unfair comparison and overclaim. 
This is my main concern. The authors claim that the proposed G-SFormer outperforms the previous methods with around 1% computational cost. However, this statement is totally wrong. Given a 2D pose sequence, **the G-SFormer only estimated the 3D pose of the center frame,** which is the seq2frame pipeline. In contrast, **seq2seq methods estimated 3D pose sequence rather than only the center frame** (MixSTE[1], STCFormer[2], MotionBERT[3], MotionAGFormer[4], KTPFormer[5]). To ensure a fair comparison, the authors should evaluate the average computational cost per frame for each method (e.g. **FLOPs/Frames**). As shown in the table, the **seq2seq methods are more efficient than the proposed G-SFormer.** The computation of FLOPs/Frames (or MACs/Frames) is a common evaluation metric in monocular 3D human pose estimation, as demonstrated by previous works such as MotionAGFormer[4], PoseMamba[11] and PoseMagic[12].

|Method|Pipeline|FLOPs (M)|FLOPs/Frames (M)|MPJPE|
|----|----|----|----|----|
|MixSTE[1]|seq2seq|278076|1144|40.9|
|STCFormer[2]|seq2seq|156392 |643|40.5|
|MotionBERT[3]|seq2seq|349434|1438|38.2|
|MotionAGFormer[4]|seq2seq|156492|644|38.4|
|KTPFormer[5]|seq2seq|278119|1144|40.1|
|G-SFormer|seq2frame|2366|2366|40.5|

Moreover, when estimating poses for 243 consecutive frames, **seq2seq methods only require these 243 frames, while seq2frame methods additionally need 121 frames before the left boundary and 121 frames after the right boundary.**

###  2.Limited novelty.
MotionAGFormer[4] combines the GNN and Transformer, achieving better results. While authors have discussed the difference between their G-SFormer and MotionAGFormer (lines 89-90), the MotionAGFormer outperforms the G-SFormer in terms of accuracy and efficiency. We can not simply judge a method as "novel" just because it differs from previous ones; it should also outperform previous methods to be considered truly novel. There are also many works[7-9] that leverage the body parts. In addition, the authors state that "Furthermore, none of them cut to the biggest computational overhead – the Self Attention calculation which is quadratic to the number of tokens. (lines 78-80)" However, HoT[6] has addressed this problem. 

### 3.Minor problem.
For optimal image clarity, especially when magnified, it is recommended to use PDF format for all images within the article. PDF format can preserve image quality at various zoom levels.

It is recommended to compare the attention maps with the latest methods instead of P-STMO, which is somewhat outdated. (Appendix A.4)






**Reference**

[1] MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video. CVPR'22

[2] 3D Human Pose Estimation with Spatio-Temporal Criss-cross Attention. CVPR'23

[3] MotionBERT: A unified perspective on learning human motion representations. ICCV'23

[4] MotionAGFormer: Enhancing 3D Human Pose Estimation with a Transformer-GCNFormer Network WACV'24

[5] KTPFormer: Kinematics and Trajectory Prior Knowledge-Enhanced Transformer for 3D Human Pose Estimation CVPR'24

[6] Hourglass Tokenizer for Efficient Transformer-Based 3D Human Pose Estimation CVPR'24

[7] Uncertainty-Aware Human Mesh Recovery from Video by Learning Part-Based 3D Dynamics ICCV'21

[8] Towards Part-aware Monocular 3D Human Pose Estimation: An Architecture Search Approach ECCV'20

[9] Limb Pose Aware Networks for Monocular 3D Pose Estimation IEEE TIP'21

[10] PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation CVPR'23

[11] PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Model arXiv'24

[12] Pose Magic: Efficient and Temporally Consistent Human Pose Estimation with a Hybrid Mamba-GCN Network arXiv'24

[13] P-STMO: Pre-Trained Spatial Temporal Many-to-One Model for 3D Human Pose Estimation ECCV'22

### Questions
My main concern is the contribution of this work, as discussed in the weakness. 

The claim in the title of "..... for efficient and robust 3D pose estimation" is not fully supported by the current manuscript. While the method may offer some advantages, its efficiency is lower than existing seq2seq methods, and the evidence for its robustness is limited. Therefore, I argue that this is not enough to fully justify the core contribution of a research paper in a top-tier conference like ICLR.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes a novel method, G-SFormer, which leverages both Transformer and Graph Neural Network (GNN) to improve 3D human pose estimation. The model is composed of three key modules: a Spatial Graph Encoder for part-based structural learning within per frame, and a Skipped Transformer Encoder and Decoder that concurrently establish long-range dynamics for temporal feature extraction across multiple frames. Additionally, the authors introduce parameter-free data completion strategies for 2D pose inputs. The method's effectiveness is demonstrated through extensive experiments on three widely recognized datasets.

### Strengths
1. The method’s effectiveness is supported by extensive experiments conducted on multiple widely recognized datasets.
2. The paper introduces a effective approach that addresses key challenges in the field, offering a practical solution and advancing the current state of research.

### Weaknesses
1. Unfair comparison. 
This is a significant concern. The authors emphasize in the title "… for efficient and robust 3D pose estimation" and assert in the abstract that G-SFormer outperforms previous methods while requiring only "around 1% computational cost". However, this claim is inaccurate. G-SFormer follows a seq2frame approach, estimating the 3D pose only for the center frame in a 2D pose sequence. In contrast, many seq2seq methods, such as MixSTE[1], MotionBERT[2], and KTPFormer[3], estimate the entire 3D pose sequence rather than focusing solely on the center frame. For a fair comparison, the authors should report the average computational cost per frame, such as FLOPs/Frame, a widely used evaluation metric in monocular 3D human pose estimation, as demonstrated by previous works like MotionAGFormer[4].

2. Lack of Innovation in the Proposed Method.
While G-SFormer combines GNN and Transformer to improve 3D human pose estimation, it is not the first method to do so. Previous approaches, such as MotionAGFormer[4], have demonstrated better results, outperforming G-SFormer in both accuracy and computational efficiency. Additionally, the approach of part-based structural learning has already been explored in prior works, such as [5], making it less of a novel contribution in this context. Lastly, the claimed improvement in reducing the computational cost of Transformer-based models is not unique to this paper, as HoT[6] has already addressed this challenge effectively.

Reference
[1] MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video. CVPR'22
[2] MotionBERT: A unified perspective on learning human motion representations. ICCV'23
[3] KTPFormer: Kinematics and Trajectory Prior Knowledge-Enhanced Transformer for 3D Human Pose Estimation CVPR'24
[4] MotionAGFormer: Enhancing 3D Human Pose Estimation with a Transformer-GCNFormer Network WACV'24
[5] Uncertainty-Aware Human Mesh Recovery from Video by Learning Part-Based 3D Dynamics ICCV'21
[6] Hourglass Tokenizer for Efficient Transformer-Based 3D Human Pose Estimation CVPR'24

### Questions
Please see more questions in the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper proposes a simple yet effective Graph and Skipped Transformer (G-SFormer) for 3D human pose estimation. It leverages a Part-based Adaptive GNN and a Frameset-based Skipped Transformer to capture both detailed pose representations and multi-perspective dynamic representations. Experimental results show that G-SFormer achieves competitive performance on benchmark datasets, including Human3.6M, MPI-INF-3DHP, and HumanEva.

### Strengths
The paper is well-structured and clearly presented. 
The idea is simple, intuitive yet effective.

### Weaknesses
- The proposed method does not compare with the state-of-the-art method [1,2]. For example, MotionAGFormer achives 38.4 mm in MPJPE on Human3.6M, which achive better performance than the proposed method. 
- Although the authors claim that their method can address joint noise, no clear evidence or quantitative results are provided to substantiate this claim. It is recommended that the authors include experimental comparisons and present evidence demonstrating why the proposed method effectively addresses joint noise to strengthen their argument.
- Given the paper’s emphasis on efficiency, a time-comparison table (inference time) should be included, as a FLOPs comparison alone is insufficient. Furthermore, since G-SFormer only regresses the 3D pose of the center frame, it appears to achieve a much lower speed compared to seq2seq-based methods [1,2,3]. 
- The paper lacks a comparison with other efficient methods, while I think it is necessary. Such as flash-attention [4] and state-space models [5]. 

[1] MotionAGFormer: Enhancing 3D Human Pose Estimation with a Transformer-GCNFormer Network

[2] MotionBERT: Unified Pretraining for Human Motion Analysis

[3] MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video

[4] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

[5] Mamba: Linear-Time Sequence Modeling with Selective State Spaces

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
5

---

## Human Reviewer 4

### Summary
This paper mainly focuses on accelerating the temporal 2D-3D human pose lifting task. The authors propose a Part-based Adaptive Graph Neural Network (GNN) to dynamically model joints more robustly and efficiently. Additionally, they introduce a skipped transformer to handle temporal redundancy more effectively. Comparisons with other state-of-the-art methods demonstrate its computational efficiency while maintaining good performance.

### Strengths
- The writing of this paper is well-organized and easy to understand.

### Weaknesses
- Vector graphics should be used.
- In Figure 1(a), the number of frames used by each model should be annotated at each point for a clearer and fairer comparison.
- Line 78 states, "none of them cut to the biggest computational overhead – the Self Attention calculation which is quadratic to the number of tokens." The improvements to attention here primarily involve modifications to the input length; Deciwatch [1] also makes similar improvements and should be compared.
- The related work section should also introduce efficient human pose lifting.
- The Part-based Adaptive GNN dynamically aggregates part embedding after partitioning the human body into parts, which is similar to the MTF-Transformer [2]. The innovation needs to be clarified, and a simple comparative experiment should be conducted.
- Line 93-94 states, "without relying on pre-defined skeletal topology as priors," but the partitioning of different body parts inherently includes priors about the human body. Difference between the weights of constructed human part graphs and human skeletal structure does not support this point. If feasible, randomly partition the body to evaluate the model in the absence of any human priors.
- The method of partitioning the body is not well explored in the paper. For example, how would the results change if each joint is treated as an independent part? How would the results change if there are overlapping joints between the parts?
- Table 5 lacks experiments for w/o SPE + w/o DR and comparisons with raw data expanding (Figure 4(c)).
- Table 6 lacks comparisons for m=1 regarding the effects of Spatial-MLP and joint-wise GCN, as the influence of skip attention should be excluded. Additionally, including experiments with a pure transformer to model joints for comparison would be beneficial to comprehensive comparison.

[1] DeciWatch: A Simple Baseline for 10× Efficient 2D and 3D Pose Estimation

[2] Adaptive Multi-View and Temporal Fusing  Transformer for 3D Human Pose Estimation

### Questions
- How exactly is AMASS utilized? The paper does not explain this in detail. AMASS is a significantly larger dataset than Human 3.6M, but the improvement after pre-training with AMASS is generally moderate; this section lacks analysis.
- Regarding data padding, should at most T/2 frames be padded for a sequence of T frames?
- The paper uses SPE for PE(position encoding). What advantages do SPE have over the commonly used learnable PE? Compare visualizations of the two position encodings, showing cosine similarity between positions, would be helpful. The benefits of encoding relative positional information for temporal or joint modeling are not addressed. If this is important, why not use the rotary PE commonly used in LLMs?

### Soundness
2

### Presentation
3

### Contribution
1

### Rating
3

### Confidence
4

---

## Human Reviewer 5

### Summary
This work proposes a method for 2D-to-3D human pose sequence lifting. The focus is on efficiency. The method fits into a long line of research using spatiotemporal transformers that aggregate information both across time and space (meaning here the human body parts).
The novelty lies in using a "Frameset-based Skipped Transformer" for the temporal aspect, and a Part-based Adaptive Graph Neural Network for the spatial aspect. The former achieves significant improvements in efficiency by not computing attention between all pairs of frames. The latter allows learning graph connections from data, instead of hand-specifying them, e.g. based on the usual skeletal structure. Furthermore a new strategy for padding input poses at the edges of the sequence is also proposed.
The method achieves or approaches the state-of-the-art on Human3.6M and MPI-INF-3DHP, while having much lower computational cost as measured in FLOPs. Ablations on Human3.6M show that the proposed techniques are effective.

### Strengths
* Temporal human pose lifting from 2D to 3D is a topic with a lot of research interest in recent years and research into making these models more efficient will have positive impact in this particular community.
* The method achieves substantially lower computational cost compared to prior works at similar levels of joint error.
* Models of different sizes are proposed for tradeoff selection between speed and accuracy.
* The ablations verify that the skipped transformer and the adaptive GNN are bringing benefits.
* The presentation of the related works is comprehensive and comparisons are done with the latest works in this area.

### Weaknesses
* Only studio datasets are used. Performing experiments on datasets such as 3DPW or EMDB would strengthen the claims.
* The proposed data completion methods, as well as the sinusoidal positional encodings, bring tiny improvements only, which might be due to noise or might not generalize to other datasets. (Table 3.). 
* Since the focus is on efficiency, actuall wall-clock inference time comparison would also be important, not only FLOPs and parameter counts. See e.g. [1].
* The presentation could be improved. Font sizes in the figures are unreadably small. Similarly with Tables 3 and 4.
* The contribution could be called incremental, since reduced-cost attention variants are a very-well researched area. The impact is therefore narrower and limited to the 2D-to-3D pose lifting community, for which a better venue might be CVPR/ICCV.

[1] Dehghani et al. The Efficiency Misnomer. ICLR 2022

### Questions
Minor suggestions:

Eq. 4. would be clearer with explicitly saying "concat" or "cat" or using || as an infix operator.

L102 typo "sate of the art"
L262 ".:"
L350: matrics -> metrics
It is typically better to place all tables and figures to the top of the page (LaTeX [t])

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
5

### Confidence
3

---

## Human Reviewer 6

### Summary
This paper proposes G-SFormer, a combination of a part-based GNN and a skipped transformer for 3D human pose estimation. The proposed approach achieves competitive results on three datasets while being efficient due to the frame-skipping strategy. Comprehensive experiments and ablation studies are presented to verify the effectiveness of the approach.

### Strengths
1. This paper shows comprehensive experiment results and ablation studies.
2. The proposed approach, G-SFormer, achieves competitive performance on standard benchmarks while being efficient compared to state-of-the-art methods.

### Weaknesses
1. The proposed method combines a part-based GNN for frame-feature extraction with a skipped transformer for efficient temporal processing, which seems incremental.
2. While it's easy to understand that the proposed model architecture is efficient due to the lower cost of temporal self-attention, it remains unclear to me why the learned representation is robust against noisy 2D keypoints. The model architecture does not appear to be well-motivated with respect to this goal. Can authors provide insights regarding this? In addition to the qualitative results shown in the paper, I suggest the authors also present quantitative results like Fig. 6 of PoseFormerV2 [1], where Gaussian noise is added to the input key points to investigate performance drop. The proposed method is expected to show stable performance under such perturbation.
3. The proposed data completion strategy shows a marginal effect in Tab. 5.
4. Authors should discuss related work regarding efficient 3D human pose estimation [1, 2, 3] in Related Work.

[1] Zhao et al. PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation, 2023.

[2] Li et al. Exploiting Temporal Contexts with Strided Transformer for 3D Human Pose Estimation, 2022.

[3] Einfalt et al. Uplift and Upsample: Efficient 3D Human Pose Estimation with Uplifting Transformers, 2022.

### Questions
1. I'm confused about the statement in L78-80: "Furthermore, none of them cut to the biggest computational overhead – the Self Attention calculation which is quadratic to the number of tokens". The proposed method is similar to PoseFormerV2 [1] (and also other methods [2][3]) in terms of improving efficiency via a reduction in token number for self-attention (or frame number reduction). 
2. How is the generalization ability of the proposed skipped transformer (how does it work if I replace the naive transformers in previous methods with skipped transformer)?

[1] Zhao et al. PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation, 2023.

[2] Li et al. Exploiting Temporal Contexts with Strided Transformer for 3D Human Pose Estimation, 2022.

[3] Einfalt et al. Uplift and Upsample: Efficient 3D Human Pose Estimation with Uplifting Transformers, 2022.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4