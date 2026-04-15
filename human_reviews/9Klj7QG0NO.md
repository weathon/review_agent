# ONE-PEACE: Exploring One General Representation Model Toward Unlimited Modalities

- Decision: Reject
- Scores: 6, 8, 5

## Abstract
In this work, we propose ONE-PEACE, a highly extensible model with 4B parameters that seamlessly aligns and integrates representations across vision, audio, and language modalities. The ONE-PEACE architecture consists of shared self-attention layers, modality adapters and FFNs. This design allows for multi-modal fusion through self-attention layers, while also providing the flexibility to easily incorporate new modalities. Two modality-agnostic pretraining tasks, cross-modal aligning contrast and intra-modal denoising contrast, are developed to align the semantic space of different modalities and capture fine-grained details within each modality simultaneously. With the scaling-friendly architecture and tasks, ONE-PEACE has the potential to expand to unlimited modalities. Without utilizing any vision or language pretrained model for initialization, ONE-PEACE achieves new SOTAs across a wide range of uni-modal and cross-modal tasks. Furthermore, we show that ONE-PEACE possesses a strong emergent retrieval capability, enabling it to align modalities that are not paired in the training data.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposed a extensible multi-modal model named ONE-PEACE. The architecture of ONE-PEACE consists of multiple modality adapters which extract unified features from different raw signals, and a modality fusion encoder which facilitate information extraction between and within different modalities. To pretrain ONE-PEACE, this work used cross-modal contrastive learning and intra-modal denoising contrastive learning. The experimental results on different tasks across various modalities shows the advantages of the model.

### Strengths
+ The experiments in this work are very comprehensive, including extensive experiments on downstream tasks, ablation experiments, and visual results.
+ The experimental results in the paper unquestionably demonstrate the superior performance of the model. The excellent fine-tuning and zero-shot performance across various downstream tasks in the visual, language, and audio modalities makes this model an outstanding three-modal universal model.
+ The model has a relatively straightforward overall architecture. The functions of each module are easy to comprehend.

### Weaknesses
- As an engineering project, this work is exceptional, with the proposed model demonstrating superior performance and good reproducibility. However, as an academic research, this work does not bring interesting findings or questions. It appears more like a fusion of various well-established and effective techniques, like hMLP,  Sub-LayerNorm and LayerScale. The contribution of this work should be reconsidered.
- Experiments solely on vision, language and audio modalities cannot prove that the model can generalize to "unlimited" modalities. Many heterogeneous modalities are hard to collect paired data and align with a existing modality, such as sensors, tables or even proprioception [a]. An experiment on a more heterogeneous modality like IMU should be conducted at least.
- I also wonder why the AVQA dataset is merely used for AQA task? The model is trained on paired data of two modalities, thus the performance of the model on a task with all three modalities is important. This experiment should be conducted.


[a] P. P. Liang, Y. Lyu, X. Fan, J. Tsaw, Y. Liu, S. Mo, D. Yogatama, L.-P. Morency, and R. Salakhutdinov, “High-modality multimodal transformer: Quantifying modality & interaction heterogeneity for high-modality representation learning,” Transactions on Machine Learning Research, 2022.

### Questions
The authors should provide a better exposition of the contributions of this work, especially the problems that the model addresses, rather than solely emphasizing its superior performance. The above weaknesses should be concerned.

Figure. 1: Adaptor -> Adapter

### Soundness
4 excellent

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduce ONE-PEACE, a simple but effective model for tri-modality representation learning. The proposed model use two stage training to align the visual acoustic and linguistic representation and it generalize well to downstream tasks. The paper is well written and the proposed method is reproduciable.

### Strengths
- the paper is intuitive, straight-forward and working as expected.
- The paper is well written and easy to follow. The paper provides enough details to reproduce the results.
- The results on downstream tasks are solid and convincing. Although not the SOTA as for now, but still strong enough.

### Weaknesses
1. The proposed method use the two stage training method, the idea behind it is to align the visual and acoustic information with linguistic representation. This is a practical way to pre-train the model but may lead to representation mis-alignment between visual and acoustic modalities. Consider to add more results to backup the visual-acoustic feature alignment quality.
2. The experimental results section is sufficient with different downstream results, but lacks the insights on the comparison against other LMMs, especially the ones with different designs.

### Questions
1. Please further discuss if the two stage training is a compromise of dataset and data quality, the training resources or it is designed intentionally.
2. I actually have hands on experience with ONE-PEACE. seems the visual-acoustic alignment is on and off, is this because of the dataset and data quality or the model design?

### Soundness
3 good

### Presentation
3 good

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
This paper proposes a model with 4B parameters that aligns and integrates representations across vision, audio, and language modalities. Two pertaining tasks, cross-modal aligning contrast and intra-modal denoising contrast are developed to align the semantic space of different modalities.

### Strengths
the paper is well-written, and the experiments are thorough. The problem of unifying representations from multiple modalities is significant and the proposed approaches showed some potential in this direction.

### Weaknesses
The paper presents an ambitious effort to amalgamate multiple modalities into a singular embedding space, a concept previously explored in works such as ImageBindm (encompassing images, language, audio, depth, thermal, and IMU modalities), CLAP (audio and language), ULIP (3D, image, and language), and Chatbridge (audio, video, image, and language), but seems not thoroughly discussed and compared. Notably, this study posits the advantage of a scaling-friendly architecture, purportedly capable of incorporating an unlimited array of modalities. While this is a compelling proposition, the reviewer suggests that the paper could better substantiate this claim by integrating and examining a broader range of modalities. Such an expansion would more robustly demonstrate the architecture's potential and scalability, thereby providing a more comprehensive understanding of its capabilities in handling diverse and complex multimodal datasets.

### Questions
see the weakness

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
