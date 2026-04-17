# Compression of Vision Transformer by Reduction of Kernel Complexity

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Self-attention and transformer architectures have become foundational components in modern deep learning. Recent efforts have integrated transformer blocks into compact neural architectures for computer vision, giving rise to various efficient vision transformers. In this work, we introduce Transformer with Kernel Complexity Reduction, or KCR-Transformer, a compact transformer block equipped with differentiable channel selection, guided by a novel and sharp theoretical generalization bound. To reduce the substantial computational cost of the MLP layers, the KCR-Transformer performs channel selection on the outputs of its self-attention layer.
Furthermore, we provide a rigorous theoretical analysis establishing a tight generalization bound for networks equipped with KCR-Transformer blocks. Leveraging such strong theoretical results, the channel pruning by KCR-Transformer is conducted in a generalization-aware manner, ensuring that the resulting network retains a provably small generalization error.
Our KCR-Transformer is compatible with many popular and compact transformer networks, such as ViT and Swin, and it reduces the FLOPs of the vision transformers while maintaining or even improving the prediction accuracy. In the experiments, we replace all the transformer blocks in the vision transformers with KCR-Transformer blocks, leading to KCR-Transformer networks with different backbones. The resulting KCR-Transformers achieve superior performance on various computer vision tasks, achieving even better performance than the original models with even less FLOPs and parameters. The code of the KCR-Transformer is available at \url{https://anonymous.4open.science/status/KCR-Transformer}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, a compression method for ViTs, namely KCR-Transformer,  is proposed. KCR-Transformer is a channel pruning method, which focuses on remove some of output channels of self-attentions to reduce the computation costs of MLP layers. To select the removeable channels, a decision mask g is defined, and according to the paper, g can be trained using gradient descent algorithm.  Furthermore, this paper includes theorietical proof to show that the KCR-Transformer can result in a tight upper and lower bound of the expected risk of DNN if the kernel complexity is small enough. This serves as a basis of the introducing of KCR blocks. Experimental results on variety of image-related tasks show that KCR-Transformers have lower computation costs but comparable or better performances comparing with their counterpart.

### Strengths
1. The proposed method find a suitable way to remove part of output channels of attention module. In this way, the computation costs of ViTs are reduced, but the network performances can be maintained. 

2. This paper provides theorietical basis of the method, and shows that reducing KC is important for the final performance. 

3. The experiments cover many image-related tasks, and the results can support the main arguements of the paper. Moreover, the ablation studies support theorem 3.1 well.

### Weaknesses
1. Some details of the method need to be included in the main paper. For instance: 

(1) How to use gradient descent to train the decision mask g, and what the values of hyperparameters of g are used in each experiments. 

(2) Why the decision mask g should multiply with both input and output features of MLP layers?  What is the difference between this method and multiplying the pruned input of MLP with a smaller weight matrix to get a smaller output?

(3) Some details, such as experiment configurations, should be move into the main paper.

### Questions
My questions are included in the part of Weaknesses.

### Soundness
3

### Presentation
2

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
The paper demonstrates their research on transformer architecture design that both benefits from channel selection in the MLP layers for attention outputs with Gumbel Softmax and generalization-aware pruning with the proposed KC metric. The authors report that their model achieves improved performance on classification, object detection, semantic segmentation, and VQA tasks, all while reducing computational overhead.

### Strengths
1.	The authors provide a solid and detailed proof for their main theory, Theorem 3.1, and comprehensively used the theorem guiding them to train their model.  
2.	Excessive experiments on 4 different types of tasks are conducted. Promising results in all fields are reported.

### Weaknesses
1.	The practical benefit of the theorem 3.1 and the according adapted KCR loss function (2) using KC lacks further evidence. The contribution of including the KC as regularization term is not proven in the ablation study in section 4.3. The section merely provides the effectiveness of approximate TNN on regularizing the KC metric but not explicitly improving the accuracy. Experimental results without KCR term might help in explaining this. 
2.	The claim of generalization-awareness in Section 3.2 is purely theoretical and lacks empirical backing, for instance, providing zero-shot experiments. 
3.	The paper’s explainability would be significantly improved with more figures, such as heat maps of the Gumbel-Softmax channel selection or figures that explicitly visualize the generalization performance. 
4.	The contributions appear to be incremental, as the work builds on existing ones on Gumbel-Softmax channel selection and neural architecture searching techniques. The paper's main theoretical contribution (KCR) lacks the necessary empirical validation to prove its effectiveness.

### Questions
1.	How does KCR help in terms of accuracy, efficiency and generalization? Could the authors provide more direct experimental evidence (e.g., targeted ablations)? 
2.	How exactly is the KCR involved in the whole model and training process? (Only the loss function part according to my understanding)

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a two-stage training framework for network pruning, aiming to balance model efficiency and performance. In the Search Stage, the weights are optimized with cross-entropy loss and Kernel Regularization, while channel gates are updated with an additional computation cost constraint, and the gate values are set to 0/1 via temperature annealing to finalize the pruned structure. Then, in the Retraining Stage, the fixed pruned structure is fine-tuned to achieve efficient model compression while preserving original model’s performance.

### Strengths
- The overall pipeline is clearly presented with a detailed formula and explanation.
- The method is well motivated, followed by a two-stage design to help find the efficient structure of the model and recover the model performance with fine-tuning.
- Experiments across different backbones consistently show better performance compared to the baseline method and other model compression methods, with fewer parameters and FLOPS.

### Weaknesses
- For Theorem 3.1 and its proof, it is unclear to me what $r^*$ represents. (Lines 859 and 860)
- In section 3.2, Line 226 states that $F \in R^{n\times d}$, but line 228 further claims $K = F^T F \in R^{n\times n}$. Is there a contradiction here?
- It would be better for the authors to add an ablation study to verify whether the observed improvement is attributed to the designed two-stage training method or the kernel function acting as a regularizer.

### Questions
- It is common practice in network pruning to employ a fixed decision mask for sample selection, yet the paper adopts a less conventional approach: optimizing the sampling parameter $\alpha$ via gradient descent. This choice lacks explicit justification, for instance, why is gradient-based optimization more effective, as it may introduce training instability or overfitting.
- Algorithm 1 specifies that only 30% of samples are used to update $\alpha$ (with 70% for weight updates), but the rationale behind this 30% ratio remains unclear. It would be valuable to conduct an ablation study to verify how varying this sample split ratio impacts $\alpha$’s ability to select meaningful channels, as well as to further explain why this specific ratio was chosen over alternatives (e.g., 20%, 40%) or adaptive splitting strategies.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the KCR-Transformer, a method that compresses Vision Transformers by using a Kernel Complexity (KC) generalization bound to guide the differentiable channel pruning of its MLP layers.

### Strengths
KCR-Transformer is shown to effectively reduce FLOPs, parameters, and the KC metric, while achieving strong empirical results that often surpass the original, uncompressed models in accuracy.

### Weaknesses
Unclear Training Cost and Scalability:
The proposed KCR-Transformer introduces several additional computational components, including Gumbel-Softmax-based channel selection, the TNN regularization term, and a complex two-stage training process (search + retraining). These modules inevitably increase the total training-time computation and memory usage compared to baseline models. While the paper provides a Minutes/Epoch comparison for the retraining phase in Appendix D.5 (Table 9), this analysis is incomplete. It fails to quantify the total training cost, most notably the significant overhead from the entire search phase. 

Lack of Fine-Grained Ablation on the KC-Performance:
The critical ablation is absent: comparing models of identical compressed size while demonstrating performance differences solely by varying the corresponding KC values is necessary, otherwise the claim that KC drives performance remains unproven.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
2
