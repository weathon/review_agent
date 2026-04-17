# Pushing Toward the Simplex Vertices: A Simple Remedy for Code Collapse in Smoothed Vector Quantization

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Vector quantization, which discretizes a continuous vector space into a finite set of representative vectors (a *codebook*), has been widely adopted in modern machine learning. Despite its effectiveness, vector quantization poses a fundamental challenge: the non-differentiable quantization step blocks gradient backpropagation.

*Smoothed* vector quantization addresses this issue by relaxing the hard assignment of a codebook vector into a weighted combination of codebook entries, represented as the matrix product of a simplex vector and the codebook. Effective smoothing requires **two properties**:

1. smoothed quantizers should remain close to a onehot vector, ensuring tight approximation, and
2. all codebook entries should be utilized, preventing *code collapse*.

Existing methods typically address these desiderata separately. By contrast, the present study introduces **a simple and intuitive regularization that promotes both simultaneously** by minimizing the distance between each simplex vertex and its $K$-nearest smoothed quantizers. Experiments on representative benchmarks&mdash;including discrete image autoencoding and contrastive speech representation learning&mdash;demonstrate that the proposed method achieves more reliable codebook utilization and improves performance compared to prior approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the non-differentiability of hard quantization by smoothing assignments while also aiming to avoid code collapse. It reformulates VQ as smoothing of one-hot vectors and adds a simple KNN regularization that pulls smoothed assignments toward all simplex vertices (using L2 or cross-entropy), compatible with softmax or Gumbel-softmax. On ImageNet autoencoding and wav2vec2 pretraining, the method shows reasonable performance against the reported baselines.

### Strengths
- The paper is well written. The technical parts are clear and detailed.
- Smoothing the VQ process is an interesting idea. It provides new insight and motivates a simple KNN regularization to reduce code collapse. The loss is simple and effective.
- Experiments confirm the effectiveness of the proposed method across tasks.

### Weaknesses
- First, the experimental scale is limited, and the paper mostly compares its own variants. There are few comparisons to recent methods, and the related work leans on pre‑2023 citations, with the rotation trick as the only recent item. As a result, the basis feels thin for an active area like VQ codebook learning.
- Second, more experiments are needed to study how batch size influences the KNN regularization. For example, compute the loss using groups within the same batch and analyze the trends. This would show whether the method scales to larger settings.
- Third, include an ablation of the Gumbel‑Softmax temperature to make the study more complete and to clarify its interaction with the proposed loss.

### Questions
- Please refer to the Weaknesses section. Overall I appreciate the technical contribution, but because the experiments appear insufficient and the discussion of recent related work is limited, I will keep the current score.

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
3

### Summary
The paper addresses the issue of code collapse in smoothed vector quantization by proposing a simple K-nearest neighbor (KNN) regularization that simultaneously enforces tight smoothing and uniform code utilization. Empirical evaluations on discrete image autoencoding and contrastive speech representation learning demonstrate that the proposed approach achieves nearly complete codebook utilization and competitive or superior reconstruction quality compared to baselines including straight-through estimation, Gumbel-softmax, and perplexity-based regularization.

### Strengths
1. The illustration and explanation of smoothed vector quantization from a simplex perspective provide a clear geometric intuition that unifies and clarifies connections with prior approaches. 
2. The proposed KNN regularization is an intuitive idea that offers strong empirical benefits compared to prior entropy or perplexity-based regularization methods.

### Weaknesses
1. While the method is designed to improve both smoothing tightness and codebook utilization, the paper does not provide quantitative analysis or visualization to support the claim about improved smoothing tightness. 
2. The KNN loss is acknowledged to be memory-inefficient, but the paper does not include any experiments or analysis to quantify the computational overhead or explore mitigation strategies.

### Questions
1. How does the proposed regularization scale to very large codebooks (e.g., M = 16k ~ 64k)? Have you explored memory-efficient approximations such as partial vertex sampling or approximate KNN search? 
2. How sensitive are the results to the choice of K (number of nearest neighbors)? Is there a principled way to select K relative to the codebook size or batch size? 
3. While the paper demonstrates improved code utilization and reconstruction performance, does this lead to improvements in downstream tasks (e.g., image generation or text-to-speech synthesis)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new regularization method for smoothed vector quantization that simultaneously promotes near one-hot assignments and prevents codebook collapse. Their method primarily varies from prior methods in its choice of anchors and neighbours. More specifically, they use codebook entries as anchors and data as neighbours. Using this, the KNN-based regularizer encourages samples to cluster near all vertices, ensuring full codebook utilization and tight smoothing. They test their method on two VQ tasks (auto-encoding and contrastive learning) and show that their method performs on par with or better than the baseline methods in different codebook settings.

### Strengths
1. Their method naturally incorporates all the entries of the codebook, which ensures maximum codebook utilization by design.
2. They achieve 100% codebook utilization among all the tasks they train, while performing on par or better than baseline methods
3. The regularization loss they propose is quite flexible and can be combined with other backbones, such as Gumbel softmax, L2 or cross entropy losses, etc.

### Weaknesses
1. Missing comparison with other baseline methods such as Group VQ, SimVQ etc.,
2. Since they use the codebook entries as anchors, some of the datapoints might not contribute to the regularization loss
3. Their method does not perform the best in some settings like in table 1, 64x64x3 setting, where the STE method performs the best

### Questions
1. The paper might improve by adding more baseline methods to show how their method compares with current state-of-the-art methods
2. How does the method perform on Image generation (diffusion, flow-based, or autoregressive) or language models?
3. How sensitive is the performance of the method in terms of K
4. A dedicated ablation study and an explanation for the choice of hyperparameters can help in reproducibility.

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
3

### Summary
- This paper proposes a simple yet effective KNN-based regularization method to stabilize smoothed vector quantization. A well-behaved quantizer should produce one-hot–like assignments while maintaining balanced codebook usage. The proposed approach achieves both by penalizing the distance between each simplex vertex and its K nearest soft assignments. This encourages all codebook entries to be actively used while keeping each assignment sharply concentrated near a single vertex.


- As a result, the method mitigates the train-test mismatch that occurs in perplexity-based regularization, where training uses soft assignments but inference requires hard quantization.


- Experiments on discrete image autoencoding and contrastive speech representation learning demonstrate that the proposed regularizer effectively prevents code collapse, achieves near-complete code utilization, and delivers improved reconstruction and representation quality compared with other baselines.

### Strengths
- Empirical results support that KNN is superior to Perplexity regularizer, as it effectively reduce the mismatch between soft / hard assignment when training and testing.

- Experiment results in representation learning highlights the robustness of the proposed method when adopted with cross-entropy (KNN-CE), which achieves 100% usage in both single and dual codebooks, where other methods fail.

- The paper is generally clear. The Related Work section provides not only a concise overview of recent advances in vector quantization but also a coherent narrative that situates the proposed method within the broader literature. It effectively summarizes the fundamental components of VQ, such as the quantization step, gradient estimation, and smoothing strategies, making it accessible and informative even to readers new to the topic.

### Weaknesses
- The paper lacks an ablation study on the number of neighbors (K), which is a key hyperparameter of the proposed regularizer.


- There is limited qualitative results in the experiment section. Beyond the simplex visualizations, it would be valuable to include additional qualitative examples such as reconstructed images or assignment-map visualizations to illustrate the impact of KNN-based over Perplexity regularization.


- Lack of experimental settings: details such as the exact loss components (reconstruction, codebook, commitment) and weighting used for baselines like STE, RE, and Perplexity regularization should be declared.

### Questions
- The experimental results show that the proposed method combined with L2 distance performs noticeably worse than KNN + CE, particularly on the contrastive learning task. My main concern is about the method’s robustness to the choice of distance metric:

Is the proposed regularizer sensitive to the form of the distance function?

Could you provide theoretical intuition or gradient-level analysis explaining why CE behaves better than L2 in contrastive settings?

Additional experiments or ablation studies comparing both distances under controlled settings would strengthen the claim.

- Please provide further analysis on the influence of number of neighbors (K):

When K is small, can the method still mitigate the soft/hard assignment mismatch observed in Perplexity regularization?

Please include an ablation study on varying K and compare it with other baselines to measure effectiveness of the proposed method.

- Because the value of K directly affects GPU memory usage, please report experiments for both baselines and the proposed method under similar GPU resource.

### Soundness
2

### Presentation
3

### Contribution
2
