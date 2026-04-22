# TokenCLIP: Token-wise Prompt Learning for Zero-shot Anomaly Detection

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Adapting CLIP for anomaly detection on unseen objects has shown strong potential in a zero-shot manner. However, existing methods typically rely on a single textual space to align with visual semantics across diverse objects and domains. The indiscriminate alignment hinders the model from accurately capturing varied anomaly semantics. We propose TokenCLIP, a token-wise adaptation framework that enables dynamic alignment for fine-grained anomaly learning. Rather than mapping all visual tokens to a single, token-agnostic textual space, TokenCLIP aligns each token with a customized textual subspace that represents its visual characteristics. Explicitly assigning a unique learnable textual space to each token is computationally intractable and prone to insufficient optimization. We instead expand the token-agnostic textual space into a set of orthogonal subspaces, and then dynamically assign each token to a subspace combination guided by semantic affinity, which jointly supports customized and efficient token-wise adaptation. To this end, we formulate dynamic alignment as an optimal transport problem, where all visual tokens in an image are transported to textual subspaces under the cross-modal similarity cost matrix. The marginal constraint and minimal cost objective of OT ensure sufficient optimization across subspaces and encourage them to focus on different semantics. Solving the problem yields a transport plan that adaptively assigns each token to semantically relevant subspaces. A top-k masking is applied to further sparsify the plan and specialize subspaces for distinct visual regions. Extensive experiments demonstrate the superiority of TokenCLIP.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel zero-shot anomaly detection method, TokenCLIP, which consists of four key components: base semantics learning, decoupled local and global text prompts, and orthogonal regularization. The proposed approach demonstrates superior performance on both industrial and medical datasets, achieving high detection accuracy while maintaining fast inference speed.

### Strengths
1.The motivation of the paper is clear. To address the problem of indiscriminate alignment in existing methods, the authors introduce multiple orthogonal textual subspaces to dynamically align each visual patch token according to its visual semantics.
2.The experiments are comprehensive. The proposed method is validated on both industrial and medical datasets, achieving state-of-the-art results in anomaly detection and localization. Furthermore, the comparison of inference efficiency convincingly shows the method’s high effectiveness.
3.The ablation study is well-designed and thoroughly demonstrates the contribution of each component.
4.The overall structure of the paper is clear and logically organized.

### Weaknesses
1.The paper proposes multiple orthogonal textual subspaces to address the issue of indiscriminate alignment, which is an interesting and promising idea. However, the specific implementation details and the underlying rationale for this design are not clearly explained in the current version of the paper.
2.The paper lacks sufficient discussion on the selection and sensitivity of key hyperparameters involved in this process.
3.There are several minor typographical issues in the manuscript. For instance, in the first paragraph of the Introduction, the sentence explore zero-shot capabilities by adapting FMs ((Pang et al., 2021; Zhou et al., 2022; Khattak et al., 2023; Jeong et al., 2023; Zhou et al., 2024a; ?) contains an extra question mark. Similarly, another question mark appears in These methods typically project either learnable (Zhou et al., 2024a; ?) in the second paragraph. These should be corrected for clarity.

### Questions
1.It is not entirely clear how the orthogonality of the textual subspaces obtained via multi-head projection is verified. It would be helpful to provide a more intuitive and concrete illustration—such as a visualization (e.g., t-SNE plot)—to support this claim.
2.The paper states that TokenCLIP benefits most from an appropriate k that supports subspace specialization while avoiding semantic over-coupling. As shown in Table 6, model performance first improves and then declines as k increases. Given this, how should the hyperparameter k and the subspace number Q be selected in practical applications?
3.From Table 4, the improvement brought by semantics learning is much more significant on the MVTec dataset than on the Visa dataset. Could the authors elaborate on the reasons behind this large performance discrepancy?

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
4

### Summary
This paper presents TokenCLIP, a novel framework for ZSAD that addresses the limitations of existing CLIP-based methods. The authors argue that the standard approach of aligning all visual tokens with a single, indiscriminate textual embedding compromises the model's ability to capture diverse anomaly semantics. To overcome this, TokenCLIP introduces a dynamic, token-level alignment mechanism. The core idea is to project a base textual space into multiple orthogonal subspaces and then use an Optimal Transport formulation to assign each visual token to a weighted combination of these subspaces. This approach allows for more fine-grained supervision, enabling the model to learn specialized representations for different semantic patterns. The paper demonstrates through extensive experiments on industrial and medical datasets that TokenCLIP achieves state-of-the-art performance.

### Strengths
1. The paper identifies a clear and significant limitation in current ZSAD methods and proposes a well-motivated and elegant solution.
2. The application of Optimal Transport to dynamically align visual tokens with a set of learned subspaces is a novel contribution to the field of anomaly detection. This formulation provides a principled way to achieve fine-grained, many-to-many correspondence.
3. The experimental results are strong and comprehensive. The method shows consistent and significant performance gains over strong baselines across a wide variety of 15 different datasets, demonstrating its effectiveness and generalizability.

### Weaknesses
1. The mechanism that drives the semantic specialization of the subspaces could be explained more clearly. The paper attributes this to the minimal cost objective of OT. However, OT's primary role is to find the most efficient matching between two fixed distributions. The specialization itself may heavily rely on the orthogonality regularization term $L_{reg}$, which explicitly forces the subspaces to be distinct. The paper would be stronger if it disentangled the contribution of OT's cost minimization from that of the explicit regularization in achieving this specialization.
2. While the authors provide an analysis of computational overhead, the discussion is brief. The introduction of an iterative OT solver for every batch during training can be computationally demanding. A more detailed analysis comparing the training time and convergence speed against baselines like AnomalyCLIP would be beneficial for practitioners assessing the method's trade-offs.

### Questions
1. The motivation for the method hinges on using multiple "textual subspaces." Have the authors considered whether these subspaces are fundamentally different from a set of randomly initialized, learnable vectors? In other words, if the "base textual space" was replaced with a generic learnable parameter, could the model still achieve comparable performance through the OT and regularization framework? This would help clarify the importance of starting from a text-based embedding.
2. How does the OT formulation specifically encourage different subspaces to focus on different semantics (e.g., one on foreground objects, others on background textures)? The paper suggests this is a result of the minimal cost objective. Could the authors elaborate on the intuition behind this? Is it possible that without the orthogonality constraint L_reg, a single "winner-take-all" subspace could still dominate the alignment for most tokens, thus preventing specialization?

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
5

### Summary
This paper addresses the task of ZSAD using CLIP-based models. The authors identify a limitation in current methods, which typically align all visual tokens from an image to a single, shared textual embedding space. They propose a method, TokenCLIP, which instead performs a token-wise alignment. The proposed framework expands a base textual space into a set of orthogonal subspaces using a multi-head projection. The core of the method is the formulation of a dynamic alignment mechanism as an Optimal Transport problem, which maps each visual token to a combination of these textual subspaces based on a cross-modal similarity cost. The resulting transport plan is sparsified using a top-K masking approach to produce the final assignment weights. The model is trained end-to-end with a combination of losses, including a global image-level loss, a base local loss from an initial indiscriminate alignment, and a dynamic alignment loss derived from the OT plan. The effectiveness of TokenCLIP is evaluated on 15 industrial and medical datasets for both image-level anomaly detection and pixel-level anomaly segmentation.

### Strengths
1.The proposed method demonstrates SOTA performance across multiple datasets.

2.The central idea of employing textual subspaces and formulating the alignment as an Optimal Transport problem is novel for this task.

### Weaknesses
1.The claim that the learned spaces are "textual subspaces" is not sufficiently justified with theoretical or experimental evidence. CLIP-based zero-shot anomaly detection operates on the premise that CLIP has aligned visual features with semantic textual features during pre-training. Detection is achieved by comparing image features against textual embeddings that explicitly represent concepts like "normal" and "abnormal." However, in this work, the embeddings within the so-called "textual subspaces" are derived from initial text prompts but are then optimized via a multi-head projection and end-to-end training. It becomes unclear whether these final embeddings still retain their original textual semantics. Are they still representative of textual information, or have they become latent vectors that are simply trained to have high similarity with normal or abnormal visual features? This ambiguity raises questions about the fundamental working principle of the method. It is possible that the model is learning a direct mapping from visual features to classifiable embeddings, rather than leveraging the rich, generalizable text-image alignment that is the cornerstone of CLIP's zero-shot capabilities.

2.While the paper visualizes the spatial assignments of subspaces, the semantic roles of these subspaces are not deeply analyzed. The interpretation that one subspace captures "object-centric semantics" while others capture "background" is based on visual inspection. A more quantitative analysis linking subspaces to specific anomaly categories, object parts, or textual concepts would provide stronger evidence for semantic specialization.

3.There appears to be missing citations in the introduction (line 037). The citation list reads "...Zhou et al., 2024a; ?", which should be corrected.

### Questions
1.In Table 5, the case where Q=1 seems conceptually similar to the baseline AnomalyCLIP, as both use a single textual space for alignment. However, the results for Q=1 show a significant performance improvement over the reported AnomalyCLIP baseline. Could the authors clarify the architectural or training differences that lead to this performance gain? Is the Q=1 model not exactly equivalent to AnomalyCLIP?

2.According to Figure 3(c), on the MVTec AD dataset, the TokenCLIP-Van variant (which uses greedy local matching instead of OT) results in a 4.2 point drop in Image-level AUROC compared to the full TokenCLIP model. This performance drop would place TokenCLIP-Van significantly below the AnomalyCLIP baseline, which uses only a single textual space. This result is counterintuitive, as one might expect that ensembling multiple textual features, even with a simple greedy assignment, would not perform worse than using a single feature space. What is the authors' explanation for this phenomenon? Why does the introduction of multiple subspaces without the global optimization of OT lead to a performance degradation compared to the simpler single-subspace baseline?

### Soundness
2

### Presentation
2

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
The paper proposes TokenCLIP, a fine-grained alignment framework that adaptively assigns a weighted combination of textual subspaces to each visual token. It reformulates dynamic alignment between tokens and orthogonal textual subspaces as an optimal transport (OT) problem, which helps ensure sufficient optimization and encourages semantic specialization across subspaces. A top-K masking is applied to further sparsify the transport plan and specialize subspaces for distinct visual regions. Extensive experiments validate the method across multiple datasets.

### Strengths
1. The method is well designed, with a clear and coherent architecture

2. The writing is fluent and clear.

3. The experimental coverage is fairly comprehensive, spanning multiple datasets.

### Weaknesses
1. No localization visualizations are provided, only numerical results, which weakens the credibility of the experiments.

2. There are many hyperparameters; does performance require per-dataset tuning?

3. Some implementation details are missing, e.g., OT marginals and weight settings are insufficiently specified.

4. There are several minor errors, such as multiple citations in the introduction rendered as “?”.

### Questions
See weaknesses:

1. No localization visualizations are provided, only numerical results, which weakens the credibility of the experiments.

2. There are many hyperparameters; does performance require per-dataset tuning?

3. Some implementation details are missing, e.g., OT marginals and weight settings are insufficiently specified.

4. There are several minor errors, such as multiple citations in the introduction rendered as “?”.

### Soundness
3

### Presentation
3

### Contribution
2
