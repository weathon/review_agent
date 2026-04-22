# A Separable Self-attention Inspired by the State Space Model for Computer Vision

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 4, 2, 2

## Abstract
Separable self-attention is an early attention mechanism with linear complexity. When parameters and FLOPs are comparable, lightweight networks built upon separable self-attention and its variants underperform the recent Vision Mamba (ViM). By analyzing the strengths and weaknesses of separable self-attention, we distill four design principles and, inspired by the State Space Model (SSM) serving as the core of ViM, propose a novel separable self-attention termed Vision Mamba Inspired Separable self-Attention (VMI-SA). Notably, VMI-SA does not incorporate any SSM blocks, and its attention computation process differs from all existing attention mechanisms to the best of our knowledge. We introduce proof-of-concept networks, VMINet and VMIFormer, enabling fair comparisons with ViMs through deliberate control of parameters, FLOPs, and encoder numbers. Compared to state-of-the-art Transformers, CNNs, and ViMs, VMINet and VMIFormer achieve competitive results in image classification and high-resolution dense prediction tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper analyzes the strengths and weaknesses of separable self-attention, and based on which, proposes a new family of separable attention based model, namely VMI-SA. Comparing with previous methods, the main innovations of VMI-SA include:

1. The attention blocks in VMI-SA apply element-wise multiplication to replace the traditional matrix multiplication

2. Context vectors are introduced to replace attention matrices. 

3. Depth-wise conv is used to introduce local spatial correlations before the element-wise multiplication operation. 

Based on the ideas above, VMINet and VMIFormer are designed. Experimental results on image classification and object detection tasks show that the proposed method has comparable or better performance comparing with several recent proposed CNNs and Transformers.

### Strengths
1. The theoretical basis of the proposed method is carefully proposed. 

2. The process of the designing of VMI-SA is well-presented. Each main component, such as element-wise multiplication, context vector, and MAMBA inspired attention, are analyzed detailedly. 

3. The experiments cover many main-stream ViT and CNN models, thus prove the advantages of the method.

### Weaknesses
1. Some arguments need to be clarified. For instance, in 3.2, the authors firstly mentioned that "The higher the
rank of the attention matrix, the more attention information it contains, and the richer the feature diversity." Which implies that an attention matrix with higher rank may provide some benefits on feature extraction. After that, Eq.7 shows that the rank of context vector is less or equal with min{L, D}. Then the authors argued that the attention information in softmax(Q)⊙K is not only less abundant but also severely homogenized. Here, it seems like one benefit of context vector is to lower the rank of attention matrices. This is a conflict with the previous context. Moreover, in 3.3.2, the authors again mentioned that we need to enhance the rank of the attention matrix in the proposed method. 

2. Some tiny problems that may be improved. For instance, in Figure 2, it is better to mark some important features of the model, such as Q, K, and context vector.

### Questions
My questions are proposed in the part "Weaknesses".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a variant of  separable self-attention method to incorporate correlation between tokens, which a basin SSA lacks. The authors incorporates three components: SSA, depthwise convolutio, and mask matrix to enhance the rank. This paper shows somewhat strong performance.

### Strengths
The evaluation is quite convicing. The comparison with ViM models shows that VMiNet and VMIFormer achieve superior performance over ViM variants.

Also, the ablation study of mask in Appendix demonstrates the importance of mask operation.

### Weaknesses
It is not clear what the authors really adopt from SSM to this proposed model. The explation between Eq. 9 and Eq. 10 in not clear.
Also, the efficiency analysis is too limited. Efficient VMamba shows the least FLOPS with longer latency and the explatnion is "nsufficient GPU utilization in EfficientVMamba’s SSM module during shorter sequence processing." Does it mean the results would be different on longer sequences?
Also, the comparison does not include Flatten Transformer.

### Questions
1. Please clarify what exactly the inspiration from SSM is and the logic behind Eq. 9 and Eq. 10.
2. Please add comparison with Flatten Transformer.
3. Please include Top-5 accuracy.

* please go over equations. For example, in Eq.5 == and != should be $-$ and $\neq$.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This manuscript proposes a novel linear-complexity separable self-attention mechanism called Vision Mamba Inspired Separable self-Attention (VMI-SA), which draws inspiration from the SSM/Mamba while avoiding integrating any SSM blocks and featuring an attention computation process distinct from existing mechanisms. It first distill four design principles by analyzing the strengths and weaknesses of separable self-attention, then design a recurrent formulation of VMI-SA and a matrix formulation to enhance token dependency modeling and computational efficiency. Based on VMI-SA, they construct two proof-of-concept networks, VMINet and VMIFormer, and conduct fair comparisons with state-of-the-art Transformers, CNNs, and ViMs by controlling parameters, FLOPs, and encoder counts. Experimental results show that VMINet and VMIFormer achieve competitive performance in ImageNet-1K image classification, MSCOCO object detection, and ADE20K semantic segmentation, demonstrating VMI-SA’s effectiveness in balancing performance and efficiency.

### Strengths
1. This paper introduces an interesting model, which incorporates separate self-attention modules into the Mamba marco design.
2. The experiment are conducted on competitive benchmarks, e.g., ImageNet, COCO and ADE20K.
3. The final model have linear complexity, which is a very promising research topic to explore.

### Weaknesses
1. The novelty is limited. This paper incorporated the minor design in separable self-attention into the Mamba marco design, titled Mamba Inspired Separable self-Attention. It is very similar to MLLA (Mamba-Inspired Linear Attention)[1] , which incroporate the Mamba minor design into the vision transformer marco design.
2. The paper also lacks the method comparsion and performance comparsion with MLLA[1].
3. Although the authors claim this is a linear model, the performance when the token length varies is missing.
4. There is nearly no ablation in the submission. Only ablation in mask type in Tab 5. However, how each minor design affects the final result is unclear.

[1] Han et al, Demystify Mamba in Vision: A Linear Attention Perspective, in NeurIPS 2024.

### Questions
1. How does each detailed structure affect the final performance?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces Vision Mamba Inspired Separable Self-Attention (VMI-SA), a new separable self-attention mechanism drawing design principles from State Space Models, particularly Mamba. The authors propose VMINet—a prototype vision backbone built purely from stacking VMI-SA blocks and downsampling layers. Through extensive experimentation across image classification, detection, and segmentation tasks, VMINet is shown to outperform state-space-based Vim models and be competitive with strong baselines in lightweight settings.

### Strengths
This paper analyzes different designs principles of self-attention, vision mamba, separable self-attention, and conclude the results into four rules to guide the design of the vision models.

### Weaknesses
1. The involvement of causal mask does not make sense for most of the vision tasks since there are no causal hypotheses in the spatial dimension of the images and videos. That is why Mamba models [1,2,3] in vision need to define one or several complicated scanning sequence to ensure the visual signals are correctly modeled. In VMI-SA, the authors use two set of learnable gating parameters  $\alpha$s and  $\beta$s to control the proportion between the causal contexts and the direct contexts. It is of vital importance to carefully analyze how and go for different inputs and in different layers, which can provide meaningful insights on how these two types of context affect the model on vision tasks. Another important work [4] points out that the causal modeling in vision mamba models could be regarded as a forced local modeling pattern, which is also helpful. However, these analyses are missing in current submission, which fade the technical depth of the paper.

2. For the discussion part of Effectiveness of VMI-SA, the authors replace the VMI-SA block with an FC layer. However, this design choice does not resemble ConvNeXt block since the normalization layers are not the same. Current drop of the accuracy cannot support the assertion. The authors could adopt the MetaFormer [5] archictecture equipped with VMI-SA, Pooling, and Self-attention, respectively to verify the impact of the spatial modeling module.

3. The experiment results do not report the model variants in larger sizes, e.g, GFLOPs for inputs on the ImageNet-1K datasets, and models with longer sequence inputs, e.g., input resolutions. It is hard to distinguish the proposed VMINet out of the baselines such as ViM [1].

[1] Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. "Vision mamba: Efficient visual representation learning with bidirectional state space model.", ICML 2024

[2] Liu, Yue, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, Jianbin Jiao, and Yunfan Liu. "Vmamba: Visual state space model.", NeurIPS 2024

[3] Yang, Chenhongyi, Zehui Chen, Miguel Espinosa, Linus Ericsson, Zhenyu Wang, Jiaming Liu, and Elliot J. Crowley. "Plainmamba: Improving non-hierarchical mamba in visual recognition.", BMVC 2024

[4] Han, Dongchen, Ziyi Wang, Zhuofan Xia, Yizeng Han, Yifan Pu, Chunjiang Ge, Jun Song, Shiji Song, Bo Zheng, and Gao Huang. "Demystify mamba in vision: A linear attention perspective.", NeurIPS 2024

[5] Yu, Weihao, Chenyang Si, Pan Zhou, Mi Luo, Yichen Zhou, Jiashi Feng, Shuicheng Yan, and Xinchao Wang. "Metaformer baselines for vision.", IEEE TPAMI

### Questions
1. The format of the paper is a little messy with large blanks and unaligned equations.

2. The "Related Works" section is missing, making it confusing to position this paper in some lines of research and show its unique advantages.

### Soundness
2

### Presentation
1

### Contribution
2
