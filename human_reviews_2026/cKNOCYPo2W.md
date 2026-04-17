# Conditioned Initialization for Attention

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Transformers are a dominant architecture in modern machine learning, powering applications across vision, language, and beyond. At the core of their success lies the attention layer, where the query, key, and value matrices determine how token dependencies are captured. While considerable work has focused on scaling and optimizing Transformers, comparatively little attention has been paid to how the weights of the queries, keys and values are initialized. Common practice relies on random initialization or alternatives such as mimetic initialization, which imitates weight patterns from converged models, and weight selection, which transfers weights from a teacher model. In this paper, we argue that initialization can introduce an optimization bias that fundamentally shapes training dynamics. We propose **conditioned initialization**, a principled scheme that initializes attention weights to improve the spectral properties of the attention layer. Theoretically, we show that conditioned initialization can potentially reduce the condition number of the attention Jacobian, leading to more stable optimization. Empirically, it accelerates convergence and improves generalization across diverse applications, highlighting conditioning as a critical yet underexplored area for advancing Transformer performance. Importantly, conditioned initialization is simple to apply and integrates seamlessly into a wide range of Transformer architectures.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the initialization of transformers, specifically the initialization of the attention mechanism. Specifically, arguing that for better optimization of the network, the initialization should bound the condition number of the Jacobian matrix of the attention computation.

The paper suggests a practical implementation for this by initializing the $\mathbf{W}_q$ and $\mathbf{W}_k$ matrices as orthogonal matrices, and $\mathbf{W}_v$ matrix as an identity one.

Through a series of experiments across language and vision tasks, the initialization is shown to achieve better performance, while achieving the same accuracy as baselines with significantly less compute, highlighting its efficiency.

### Strengths
S1: The paper is clear and can be followed easily.

S2: The practical implementation of the initialization is easy to use, making it conducive to wide adoption.

S3: The method shows consistent performance gains compared to the baselines across different tasks and modalities (language and vision).

S4: The suggested initialization method is highly efficient, achieving the same performance as baselines with much less compute. This is an advantage that could be highlighted further by the authors, perhaps through showing detailed training loss curves of the different models and adding more figures like Figure 2 for all tasks.

### Weaknesses
W1: The main weakness is that while the paper demonstrates the proposed initialization bounds the condition number of the Jacobian matrix of the attention computation, it does not sufficiently explain the theoretical basis for why this is a desired property for better optimization. Although better performance is shown in practice, a proof or a more robust theoretical explanation for this optimization benefit is expected, especially since the paper outlines how to achieve the bound.

W2: This is a minor point, but using the notation $\mathbf{A}(\mathbf{X})$ for the output of the attention computation is potentially confusing, as this symbol is often reserved for the attention matrix itself. A symbol change is suggested for clarity.

### Questions
See W1

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes to initialize the query and key weights of self-attention layers as semi-orthogonal matrices, and the value weights as rectangular identity matrices. This scheme is based on an analysis of the condition number of the Jacobian of self-attention layers, supported by previous research suggesting that bounding this condition number leads to smoother/faster convergence.

### Strengths
- Comprehensive empirical validation on a variety of vision tasks, showing clear positive benefit for large and small scale image classification, detection and segmentation
- (Somewhat less) comprehensive empirical validation on language tasks, showing improved performance on LRA and on 100m-param-scale language modeling
- Demonstrated benefit over mimetic initialization
- Addresses the ViT small-scale-data issue as well or better than mimetic initialization while being similarly simple
- Theoretically grounded
- Works for a variety of attention mechanisms

### Weaknesses
- Relatively small-scale language model evaluation (e.g., experiments requiring one GPU while training ViT-B presumably took multiple GPUs). But I still find the results convincing and promising. I know it's hard to do this on an academic budget. And the vision results alone justify my score.

### Questions
Did you try anything at the intersection of conditioned and mimetic initializations? For example, you could maybe pick a close orthogonal pair of matrices. Or you could use conditioned init for W_Q, W_K and mimetic init for W_V, W_O. 

Did you try any ablations? (Like only initializing W_Q, W_K or only initializing W_V?)

Did you consider the output projection at all? It seems like it would be closely tied to W_V.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes conditioned initialization, an initialization scheme for attention weights that explicitly targets the conditioning of the attention Jacobian. The authors present a theoretical analysis showing that the condition number of Jacobian can be upper-bounded in terms of the condition numbers of the query, key, and value matrices. Making these matrices well-conditioned at initialization is expected to stabilize optimization. Extensive experiments on various downstream tasks demonstrate that this initialization improves final accuracy and accelerates convergence compared to baseline methods on small models. The approach is simple to apply and does not require changes to existing training pipelines.

### Strengths
1. The paper provides a clear theoretical motivation and analysis, deriving an explicit upper bound related to attention optimization stability. This is both novel and well justified.
2. The paper is well written and easy to follow. The proposed method is straightforward to implement and architecture-agnostic.
3. Experimental results across various downstream tasks are promising. The method consistently improves performance and accelerates convergence.

### Weaknesses
1. The theoretical analysis optimizes an upper bound on the condition number of the Jacobian rather than the condition number itself. Although empirical results support the approach, the gap between the bound and the actual condition number is not fully characterized. It remains unclear whether a tighter bound would yield further improvements.
2. All experiments are conducted on relatively small models. It is unclear whether the benefits of conditioned initialization extend to large-scale models.
3. There is a typo on line 869: “The implementation of the ViTs were” should be “The implementation of the ViTs was.”

### Questions
1. Does conditioned initialization continue to provide performance gains and faster convergence when hyperparameters such as learning rate, warmup steps, and weight decay are re-tuned for each baseline? Could baseline methods catch up with light hyperparameter tuning?
2. Is the proposed method still effective for large-scale models?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a spectrum inspired approach to initialize the attention matrices to reduce the conditioning number of the attention Jacobian matrices with partial theoretical analysis. Extensive experiments demonstrate that using the presented initialization approach leads to consistently superior performance over the baselines.

### Strengths
+ The paper proposes a simple but effective intialization approach, which might have a better condition number of the Jacobian of the attention matrix with respect to the parameter matrices $W_Q$, $W_K$ and $Q_V$.   
+ Extensive experiments demonstrate consistent improvements over the two baseline methods.

### Weaknesses
- The results in Lemma 3.1 and Theorem 3.1 cannot be listed as the (theoretical) contributions of the paper. These results are merely the formal expression of the gradients of a matrix function with respect to the parameter matrices via the Kronecker product. Similar results can be found in prior work, e.g., the appendix in [a]. 

- The results for Jacobian matrices developed in the paper might not complete due to ignorance of the stablization structure, e.g., Layer Norm (LN), RMSNorm, QKNorm. Thus, it is also questionable whether the upper bound of the condition number makes any sense in practical. If either LN, or RMSNorm, or QKNorm is introduced, can the proposed approach still yield improved performance comparing to the counterpart baseline methods? 

- While the condition number of the proposed initialization strategy is reduced, it is merely a heuristic way to form the initialization for the attention matrices. Does it enable the training process stable? How about the effects of using the conditioned initialization on the learning curves?  In practice, when one of LN, or RMSNorm, or QKNorm or some combination of them is introduced, what aobut the learning curves? 

- In previous work, there are many attempts to design stablized optimization algorithm for training Transformer. It would be more interesting if some evaluations to connect the proposed initialization strategy with the stablized algorithms. The reviewer is curious about that whether or not the proposed initialization still works when stablized optimization algorithms (or stablized structure, e.g., LN, RMSN, QKNorm, etc.) used.


[a] Taming Transformer Without Using Learning Rate Warmup, ICLR'25. 

[b] Learning deep transformer models for machine translation. arXiv preprint arXiv:1906.01787, 2019.

[c] Query-key normalization for transformers. EMNLP 2020

[d] Scaling vision transformers to 22 billion parameters. ICML 2023.

[e] Root mean square layer normalization. NeurIPS 2019.

### Questions
- Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
