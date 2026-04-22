# Calibrated Information Bottleneck for Trusted Multi-modal Clustering

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Information Bottleneck (IB) Theory is renowned for its ability to learn simple, compact, and effective data representations. In multi-modal clustering, IB theory effectively eliminates interfering redundancy and noise from multi-modal data, while maximally preserving the discriminative information. Existing IB-based multi-modal clustering methods suffer from low-quality pseudo-labels and over-reliance on accurate Mutual Information (MI) estimation, which is known to be challenging. Moreover, unreliable or noisy pseudo-labels may lead to an overconfident clustering outcome. To address these challenges, this paper proposes a novel CaLibrated Information Bottleneck (CLIB) framework designed to learn a clustering that is both accurate and trustworthy. We build a parallel multi-head network architecture—incorporating 
one primary cluster head and several modality-specific calibration heads—which achieves three key goals: namely, calibrating for the distortions introduced by biased MI estimation thus improving the stability of IB, constructing reliable target variables for IB from multiple modalities and producing a trustworthy clustering result. Notably, we design a dynamic pseudo-label selection strategy based on information redundancy theory to extract high-quality pseudo-labels, thereby enhancing training stability. Experimental results demonstrate that our model not only achieves competitive clustering accuracy on multiple benchmark datasets but also exhibits excellent performance on the expected calibration error metric. Code is available at \textcolor{red}{https://shizhehu.github.io/}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes CLIB, a calibrated information bottleneck framework for multi-modal clustering. It introduces a multi-head architecture with dedicated calibration heads to mitigate the impact of noisy pseudo-labels and biased mutual information estimation—key issues in existing IB-based methods. A dynamic label selection strategy further improves training stability. Experiments show CLIB achieves state-of-the-art clustering accuracy and superior calibration on multiple benchmarks.

### Strengths
-Well-motivated problem: The focus on improving trustworthiness and calibration in multi-modal clustering is timely and important.
-Innovative architecture: The multi-head design with dedicated calibration heads is a clever way to decouple representation learning from label refinement.

### Weaknesses
-Limited discussion on calibration in clustering: While ECE is adapted for clustering, the paper does not fully address the challenge of defining "correctness" without ground-truth labels during calibration evaluation.
-Lack of computational analysis: No comparison of training time or model complexity is provided, making it hard to assess practical efficiency.
-Hyperparameter sensitivity: The balance between IB objectives and calibration is controlled by hyperparameters; their robustness is not thoroughly analyzed.

### Questions
The method relies on pseudo-labels for both clustering and calibration, yet these labels are inherently noisy and evolve during training. How does the proposed calibration mechanism avoid reinforcing or amplifying incorrect pseudo-labels in the early or unstable stages of training? A brief analysis or design justification on the robustness of calibration to label noise would significantly strengthen the paper's claim of producing "trusted" clustering.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Information bottleneck theory effectively removes redundancy or noise in multi-modal clustering while preserving discriminative information, but existing IB-based methods face challenges such as low-quality pseudo-labels, over-reliance on accurate mutual information estimation and unreliable clustering outcomes. This work proposes a calibrated information bottleneck framework featured a parallel multi-head network to reach three typical goals, calibrating biased MI estimation to enhance IB stability, building reliable IB targets from multi-modal data samples and getting trusted results. A dynamic pseudo-label selection strategy grounded in information redundancy theory improves training stability by filtering high-confidence labels. Experiments show the method achieves promising results across many benchmarks.

### Strengths
1.The proposed CLIB framework is innovative. Its parallel multi-head architecture successfully decouples the calibration objective from the final clustering objective.

2.Theorem 1 given in the paper is insightful. It theoretically connects the difficult problem of MI estimation bias with the pseudo-label screening mechanism via a clear logical chain.

3.The experimental section is solid. On five widely-used benchmark datasets, CLIB achieves state-of-the-art performance on all three metrics. Particularly, the significant reduction in ECE robustly demonstrates CLIB's effectiveness in mitigating model overconfidence.

### Weaknesses
1.One of the core motivations of the paper is that MI estimation is difficult and biased. However, in the actual implementation, different parts of the model use three different MI estimation strategies. The authors do not explicitly explain the rationale for choosing this.

2.The implementation environments of the code are not given in the experiments, which may influence the reproductivity. A complete experimental details is usually needed in the experimental setup subsection.

### Questions
1. In the adaptive fusion, could you provide the specific weights w_m learned during the experiments? This would help verify whether the model is truly adapting the importance of different modalities.

2. The ablation study for L_con shows a potential trade-off between ACC and ECE.  In a practical application, if one must prioritize one over the other, how would you advise them to adjust the model?

3. Is this the first work addressing the trusted multi-modal clustering problem with calibration? If yes, please explicitly describe it in the paper. If not, please deeply discuss the differences with the related works.

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
5

### Summary
This paper introduces an information bottleneck-based multi-modal clustering method, which addresses the issue of overconfident clustering results caused by low-quality or unreliable pseudo-labels in multi-modal clustering. The parallel multi-head architecture is proposed to correct distortions in mutual information estimation. Experiments demonstrate that the proposed method achieves superior performance compared with existing baselines across multiple datasets.

### Strengths
1. This work addresses the reliance on reliable target variables and the overconfidence caused by noisy pseudo-labels, which is a commonly encountered problem in clustering.
2. The proposed dynamic pseudo-label screening strategy based on information redundancy offers a promising alternative to existing probability-based thresholding.
3. The proposed framework takes the inherent difficulty of precise MI estimation into consideration, alleviating the negative impacts of such estimation biases.

### Weaknesses
1. The proposed method consists of M+1 heads, a two-stage training process, and multiple loss terms, which may increase computational overhead compared to baseline methods. The authors are encouraged to discuss the model’s computational complexity to better assess its practical applicability.
2. In the current implementation, gradients are blocked from the calibration heads while allowing backpropagation from the cluster head to the IB to avoid contradictory objectives. It would be helpful if the authors clarify what specific contradictions were observed and whether the fusion-based cluster head’s objective is consistently better aligned with the IB optimization than the single-modality calibration heads.
3. The pseudo-label screening excludes high-entropy, uncertain samples during training. It would be interesting to investigate the final clustering performance on these “difficult” samples after convergence. Would they yield much better performance than before?
4. There are some other clustering methods (such as Twin Contrastive Learning for Online Clustering, IJCV 2022) that also use pseudo labels to boost clustering performance. The authors could clarify the differences between this work and previous ones.

### Questions
I expect the authors to address my concerns in the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
By seeing the challenging problems faced by existing information bottleneck-based multi-modal clustering methods, in this work a novel calibrated information bottleneck is proposed for trusted multi-modal clustering to learn more accurate and trustworthy clustering outcome. It mainly presents a parallel multi-head network architecture containing clustering and calibration heads for outputting high-quality data assignments. Lots of different kinds of experiments have illustrated the superiority of the method on multiple benchmark datasets with metrics of clustering accuracy and calibration error.

### Strengths
1) The paper is well-written with clear logic. The flow from problem introduction and method description to experimental analysis is fluent. Figure 1 provides a clear and understandable illustration of the whole framework, helping readers quickly grasp the model's workflow.

2) The control over the gradient flow is very fine-grained, it blocks the gradient from the calibration heads but allows it to back-propagate from the main cluster head. This design ensures that the feature learning of IB serves the final clustering task while avoiding potentially contradictory signals from the calibration objectives of different modalities.

3) The parameter sensitivity analysis shows that the model maintains stable performance over a wide range of choices for hyper-parameters. This indicates that the method does not require excessive tuning and possesses good practical value.

### Weaknesses
1)  Eq. 3 of the paper relies on the NT-Xent loss. The effectiveness of NT-Xent is highly dependent on the Batch Size (N), as its MI lower bound is log(N)-L_(NT-Xent). If N is too small, this bound becomes loose, leading to poor alignment. The paper does not mention the Batch Size used in the experiments or its impact.

2) The paper does not detail the specific network architectures used for feature extraction and the various heads, where they are also important in reproducing the results for readers.

3) The paper relies entirely on quantitative metrics. For a clustering task, providing t-SNE visualizations of the feature space would be highly persuasive. A comparison of the feature distributions after the warm-up and after calibration could visually demonstrate how the framework improves inter-class separation and intra-class compactness.

### Questions
1) The method requires the number of clusters, C, to be specified in advance. How sensitive is the model to the choice of C? If C is set incorrectly, how much are the model's performance and calibration affected?

2) Could you provide the specific network architectures for the Backbones and the CalHead/CluHead? For instance, what kind of encoders were used for the image and text modalities, respectively?

3) What are the limitations that the proposed method still exist in calibrating the multi-modal clustering results? Could you provide some future insights in this area for readers?

### Soundness
3

### Presentation
3

### Contribution
3
