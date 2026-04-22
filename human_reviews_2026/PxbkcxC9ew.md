# Federated Learning with Binary Neural Networks: Competitive Accuracy at a Fraction of the Cost

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 6, 2, 0

## Abstract
Federated Learning (FL) preserves privacy by distributing training across devices. However, using DNNs is computationally demanding for the low-powered edge at inference. Edge deployment demands models that simultaneously optimize memory footprint and computational efficiency, a dilemma where conventional DNNs fail by exceeding resource limits. Traditional post-training binarization reduces model size but suffers from severe accuracy loss due to quantization errors. To address these challenges, we propose FedBNN, a rotation-aware binary neural network framework that learns binary representations directly during local training. By encoding each weight as a single bit $\{+1, -1\}$ instead of a $32$-bit float, FedBNN shrinks the model footprint, significantly reducing runtime (during inference) FLOPs and memory requirements in comparison to federated methods using real models. Evaluations on multiple benchmark datasets demonstrate that FedBNN reduces resource consumption greatly while performing similarly to existing federated methods using real-valued models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes FedBNN, a Rotation-Aware Binary Neural Network framework for Federated Learning (FL). It aims to address the significant computational and storage burdens associated with training and inference on edge devices within the FL paradigm. FedBNN directly learns binary weights on the client side. To mitigate angular bias induced by binarization, it introduces trainable rotation matrices and an adaptive mixing weight strategy. Furthermore, it enhances gradient propagation during backpropagation through a training-aware approximation function. Experiments conducted on datasets including FMNIST, SVHN, and CIFAR-10 demonstrate that FedBNN achieves within 10% of its real-valued counterparts while reducing runtime FLOPs by approximately 40–58 times and memory footprint by 32 times. Ablation studies are also presented to analyze the impact of rotation matrix aggregation and initialization strategies.

### Strengths
1.Federated Extension of Rotational Binary Neural Networks: Adapting rotational binary neural networks to the federated learning framework, this work extends the centralized rotational optimization proposed by Lin et al. (2020) to federated learning scenarios. By introducing trainable rotation matrices and a federation-aware weighting mechanism, the angular deviation caused by binarization is effectively mitigated, balancing local optimization with global consistency.

2.Significant Computational Cost Reduction: Achieves 40–58 times lower runtime FLOPs and 32 times reduced memory usage, validating edge deployment potential.

3.Comprehensive Experiments: Evaluated on FMNIST, SVHN, CIFAR-10 under IID/non-IID scenarios, with comparisons to FedAvg/FedBAT/FedMud. in computational load, storage cost, and accuracy.

### Weaknesses
1.Limited theoretical analysis: Convergence proof of FedBNN in federated scenarios is not provided (compared to the theoretical analysis of FedAvg). The impact of rotational optimization on convergence speed also needs to be quantified (e.g., upper bound of convergence rounds).

2.Insufficient dataset complexity: Validation is only conducted on small-to-medium-scale datasets (FMNIST/SVHN/CIFAR-10), lacking verification on more complex or high-dimensional datasets (e.g., CIFAR-100 or ImageNet).

3.Inadequate model complexity: FedBNN is experimentally validated only on ResNet-10 and a small CNN, with unknown optimization effects on deeper models.

4.Lack of quantitative analysis on communication cost: Although the paper claims significant savings in inference and storage, transmitting rotation matrices may introduce additional communication overhead. The experiment also fails to compare the communication cost of FedBNN with FedAvg/FedBAT/FedMud.

### Questions
1.Convergence analysis: It is recommended to supplement the theoretical convergence analysis of FedBNN in federated scenarios. For example, by comparing with the existing theoretical framework of FedAvg, further quantifying the impact of rotational optimization on convergence speed (e.g., convergence round boundaries) to enhance the theoretical support of the method.

2.Communication overhead: Transmitting rotation matrices may introduce additional communication costs. It is suggested to supplement the quantitative analysis of this overhead and compare it with the communication costs of methods like FedAvg and FedBAT to comprehensively evaluate overall performance.

3.Scalability validation: Current experiments are mainly based on small-to-medium-scale datasets and shallow models. It is recommended to further validate on more complex datasets (e.g., CIFAR-100, ImageNet) and deeper networks (e.g., ResNet-18, MobileNetV3) to demonstrate the method’s generality.

4.Practical deployment testing: It is suggested to supplement deployment tests in real edge device environments to verify the feasibility and efficiency of the method under actual hardware conditions.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces **FedBNN**, a rotation-aware **Binary Neural Network (BNN)** framework for **Federated Learning (FL)** that achieves competitive accuracy with dramatically reduced computation and memory costs. Unlike conventional FL methods that rely on full-precision models. To mitigate quantization error, FedBNN applies trainable rotation matrices before binarization and introduces **federated-aware fusion** and **adaptive rotation adjustment** mechanisms that align local and global model directions under heterogeneous data. A training-aware approximation function further stabilizes gradient flow during optimization. Experiments show that FedBNN maintains accuracy within 5–10% of real-valued models while achieving superior efficiency, outperforming existing binarization-based FL methods such as FedBAT.

### Strengths
The paper demonstrates strong **originality** by being the first to integrate rotation-aware binary neural networks—originally designed for centralized settings—into federated learning, introducing a novel client-side fusion of local and global weights before rotation and adaptive correction terms (α, β) that jointly minimize quantization-induced angular bias and align with the server model. Its **technical quality** is high, featuring a well-motivated method (FedBNN), rigorous experiments across three datasets under IID and two levels of Non-IID heterogeneity, and comprehensive comparisons against state-of-the-art baselines like FedBAT and FedMud.

### Weaknesses
A key weakness of the paper is the lack of a fair comparison under **equal computational budgets**. The authors claim FedBNN achieves "competitive accuracy" by comparing its binary model directly against full-precision baselines like FedAvg, but on the performance, there is a significant gap. As a result, a smaller model with the same FLOPs may still achieve the same performance as FedBNN. Without evaluating a smaller real-valued model matched to FedBNN’s computational cost—such as a pruned or shallower CNN—the claimed accuracy trade-off remains misleading; the performance gap may stem more from model capacity disparity than binarization itself. Furthermore, the efficiency gains (FLOPs and memory) are **purely theoretical** and not validated on real edge hardware. Since XNOR+bitcount acceleration depends heavily on hardware support (e.g., ARM NEON or specialized accelerators), the absence of end-to-end latency, energy consumption, or inference speed measurements on actual devices leaves the practical deployability of FedBNN unsubstantiated. Including both equal-budget comparisons and real-device benchmarks would significantly strengthen the paper’s claims.

### Questions
1. **Fairness of Accuracy Comparison**:  
   The paper compares FedBNN’s accuracy against full-precision FedAvg while highlighting massive FLOPs/memory savings. However, this comparison conflates *model capacity* with *quantization effects*. Could the authors provide results for a **smaller real-valued model** (e.g., a pruned or width-scaled CNN/ResNet) that matches FedBNN’s FLOPs or memory footprint? This would clarify whether the accuracy gap is due to binarization itself or simply reduced representational capacity.

2. **Real-Device Efficiency Validation**:  
   All efficiency claims (FLOPs, memory) are theoretical. Could the authors provide **real-world latency or energy measurements** on an edge device (e.g., Raspberry Pi, smartphone) comparing FedBNN against a real-valued baseline? Without this, the practical deployability of the claimed 40–58× speedup remains speculative.

Addressing these points would significantly bolster the paper’s technical rigor and practical relevance.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies federated learning on heterogeneous, resource-limited devices, focusing on reducing communication, computation, and memory overhead. The core idea is to train binarized versions of the model on devices while maintaining competitive accuracy. However, the contribution appears limited, as the proposed approach targets only CNN architectures, the performance analysis is restricted to relatively small models, and there is not proof of convergence.

### Strengths
The problem is important and has been studied extensively in recent years; efforts to reduce the resource footprint of federated learning are valuable.

### Weaknesses
1. The method is presented only for simple CNN models, with no discussion of whether it can extend to more complex architectures (e.g., LLMs).

2. The evaluation is restricted to relatively small models: four-layer CNNs for FEMNIST and SVHN, and a ResNet-10 for CIFAR-10.

3. There is no convergence proof; thus, convergence is assessed only empirically—and those experiments are limited to very simple models.

### Questions
1. Can the proposed method be extended to more complex architectures, such as LLMs or ResNet-50?

2. Is there any guarantee of training convergence? The current evaluation is limited to very simple models, so such a conclusion cannot be drawn from these results.

3. Could the authors provide more details on the convergence behavior of FedBNN? Are the results in Tables 1 and 2 reported for the same number of rounds, or does FedBNN require more rounds to converge?

4. Why are the baselines limited to this specific set? What about other state-of-the-art methods, such as FedProx?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes FedBNN, a federated learning framework that trains rotation-aware binary neural networks directly on clients. However, the work is not well motivated and appears to be a relatively straightforward extension of RBNN with limited adaptation to the federated setting. The paper overlooks the significant computational and memory costs during local training, failing to address the core resource constraints that motivate efficiency in federated learning. In addition, several methodological details are unclear or incorrect, and some settings lack explanation or justification, which further weakens the overall contribution.

### Strengths
1. Binary deployment is essential for  cross-device federated learning 
2. FedBNN significantly improves the inference efficiency.

### Weaknesses
I carefully read both this paper and the referred RBNN [1] and found that the main contribution of this work is to apply RBNN to federated learning with an additional server weight fusion mechanism using learnable fusion factors. However, this extension appears rather trivial and insufficiently motivated. The paper does not clearly explain why fusing the out-of-date server weights ($w_{server}$) would benefit training instead of potentially interfering with it. Likewise, it remains unclear why the fusion factor must be trainable rather than treated as a hyperparameter, as done in most related works. No ablation study or empirical analysis is provided to justify these design choices, making it difficult to attribute any improvement specifically to the proposed modifications. As a result, the novelty seems limited, with most of the technical substance originating from RBNN rather than new contributions.

Moreover, in federated learning, resource-constrained clients are a critical concern. Directly applying RBNN training mode in such settings can substantially increase computational and memory overhead during training, yet this issue is not addressed or discussed in the paper.

In addition,  the most equations in this paper are same RBNN, only with discrepancies in several equations. For instance, in Eq. (14), RBNN defines the condition as $|x| < \sqrt{2}t x$, while this paper uses $|x| < \sqrt{2t} x$; and in Eq. (15), RBNN uses $10^{T_{\min} + \frac{\text{current epochs}}{\text{Total Epochs}} (T_{\max} - T_{\min})}$, whereas this paper writes $10T_{\min} + \frac{\text{current epochs}}{\text{Total Epochs}} (T_{\max} - T_{\min})$. However, the authors do not clarify these differences. So I do not know whether these are intentional modifications or typo errors, which raises concerns about correctness and reproducibility.

From a presentation standpoint, several symbols are used without prior definition, such as $N_s$ in Equation (1) and $\bar{W}$ in Equation (9), reducing readability.

Finally, the experimental section is weak: there is no ablation study for the fusion, no sensitivity analysis for key hyperparameters, and no overhead analysis. Both the models and benchmarks are out-of-date. This makes it hard to assess the robustness and significance of the reported results.

[1] Lin, Mingbao, et al. "Rotated binary neural network." Advances in neural information processing systems 33 (2020): 7474-7485.

### Questions
In Eq. (14), RBNN defines the condition as $|x| < \sqrt{2}t x$, while this paper uses $|x| < \sqrt{2t} x$;

 in Eq. (15), RBNN uses $10^{T_{\min} + \frac{\text{current epochs}}{\text{Total Epochs}} (T_{\max} - T_{\min})}$, whereas this paper writes $10T_{\min} + \frac{\text{current epochs}}{\text{Total Epochs}} (T_{\max} - T_{\min})$. 

Are these intentional modifications or typo errors?

### Soundness
2

### Presentation
1

### Contribution
1
