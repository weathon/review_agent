# Low-Rank Few-Shot Node Classification by Node-Level Graph Diffusion

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
In this paper, we propose a novel node-level graph diffusion method with low-rank feature learning for few-shot node classification (FSNC), termed Low-Rank Few-Shot Graph Diffusion Model or LR-FGDM. LR-FGDM first employs a novel Few-Shot Graph Diffusion Model (FGDM) as a node-level graph generative method to generate an augmented graph with an enlarged support set, then performs low-rank transductive classification to obtain the few-shot node classification results. Our graph diffusion model, FGDM, comprises two components, the Hierarchical Graph Autoencoder (HGAE) with an efficient hierarchical edge reconstruction method and a new prototypical regularization, and the Latent Diffusion Model (LDM). The low-rank regularization is robust to the noise inherently introduced by the diffusion model and empirically inspired by the Low Frequency Property. We also provide a strong theoretical guarantee justifying the low-rank regularization for the transductive classification in few-shot learning. To further enhance the performance of LR-FGDM, we introduce LRA-LR-FGDM with a novel efficient LR-Attention layer, or the LRA layer, which applies self-attention to the output of the LR-FGDM encoder. The LRA layer further reduces the kernel complexity of LR-FGDM and contributes to a tighter generalization bound, leading to improved performance. Extensive experimental results evidence the effectiveness of LR-FGDM for few-shot node classification, which outperforms the current state-of-the-art. The code of the LR-FGDM is available at \url{https://github.com/Statistical-Deep-Learning/LR-FGDM}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on few-shot node classification (FSNC) tasks with generative model-based data augmentation method. Specifically, the proposed LR-FGDM method adopts the diffusion model-based few-shot learning paradigm and serves as a plug-in module to enhance existing FSNC frameworks. The FGDM model utilizes a hierachical graph autoencoder to obatin label prototypes and synthesize graph attributes for original graph augmentation. A low-rank few-shot classification module is designed to reduce inductive noise from diffusion generation process. Experiments and ablation studies verify the effectiveness of LR-FGDM.

### Strengths
1. The proposed LR-FGDM is a plug-in-and-play framework that can be integrated into multiple FSNC models.
2. It is a reasonable idea to apply low-frequency filters to avoid additional noise from generative model.
3. Comprehensive experimental studies on the model's FSNC performance.

### Weaknesses
1. The overall contribution appears somewhat incremental, as the proposed framework mainly combines elements from prior works on diffusion on graphs (DoG) and graph augmentation–based few-shot learning.
2. Although the authors claim that the low-rank regularization ensures an upper bound on the Frobenius norm of the predicted labels, the theoretical bound presented in Eq 3 is not illustrative enough to reflect its convergence characteristics.
3. While the execution time for training and generation in FGDM is reported, the comparative training efficiency against baseline models remains unexplored.
4. The paper would benefit from a more detailed introduction to the overall FSNC algorithm and a broader discussion on the range of FSNC frameworks to which LR-FGDM can be applied.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This isn't a paper in my field, but it appears to be a very solid piece of work. It includes derivations, well-conducted experiments, comparisons with recent baselines, and open-sourced code, a good paper.

The paper proposes LR-FGDM, a novel and well-motivated framework for few-shot node classification that combines node-level graph diffusion with low-rank transductive classification. Its key strengths include: (1) a prototype-conditioned graph diffusion model (FGDM) that generates both synthetic nodes and realistic edges without relying on unavailable class labels during test time; (2) an efficient hierarchical edge reconstruction mechanism that reduces decoding complexity; (3) a theoretically grounded low-rank regularization based on the Low Frequency Property, supported by a generalization bound; and (4) comprehensive experiments across diverse datasets—including heterophilic graphs—showing consistent improvements over state-of-the-art methods, along with new evaluation metrics (FND/FED) to assess generation fidelity.

### Strengths
The paper proposes LR-FGDM, a novel and well-motivated framework for few-shot node classification that combines node-level graph diffusion with low-rank transductive classification. Its key strengths include: (1) a prototype-conditioned graph diffusion model (FGDM) that generates both synthetic nodes and realistic edges without relying on unavailable class labels during test time; (2) an efficient hierarchical edge reconstruction mechanism that reduces decoding complexity; (3) a theoretically grounded low-rank regularization based on the Low Frequency Property, supported by a generalization bound; and (4) comprehensive experiments across diverse datasets—including heterophilic graphs—showing consistent improvements over state-of-the-art methods, along with new evaluation metrics (FND/FED) to assess generation fidelity.

### Weaknesses
Despite its strengths, the work has several limitations. First, the comparison with DoG is potentially unfair, as DoG is adapted using pseudo-labels that may not reflect its optimal use in few-shot settings. Second, the computational cost of the truncated nuclear norm (TNN) regularization—particularly the need for SVD—is not adequately addressed, raising scalability concerns. Third, while FND/FED quantify distributional similarity, the semantic correctness of generated nodes lacks intuitive validation (e.g., via embedding visualization). Fourth, the reason for strong performance on heterophilic graphs like Roman-Empire is not deeply analyzed. Finally, certain implementation details (e.g., SVD approximation, prototype initialization) remain unclear, which could affect reproducibility.

### Questions
see above.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a framework for few-shot node classification that uses a node-level graph diffusion model to generate synthetic support nodes and edges for augmenting the support set, and introduces low-rank regularization to mitigate diffusion-induced noise, with theoretical guarantees for transductive classification; extensive experiments on 8 datasets show LR-FGDM outperforms sota FSNC methods.

### Strengths
- LR-FGDM solves FSNC data scarcity by generating synthetic support nodes and edges.
- The low-rank regularization in LR-FGDM fixes diffusion-induced noise, makeing the quality of synthetic data more reliable for model training.
- The paper provides theoretical proof for the low-rank regularization, supporting the method’s validity with mathematical generalization bounds.
- The proposed model outperforms state-of-the-art FSNC methods consistently, showing real effectiveness across 8 benchmark datasets.

### Weaknesses
- For the hierarchical edge reconstruction, \(d_i\) (number of connected clusters per node) is bounded by node degree, but what if nodes have extremely high degrees (e.g., hub nodes in scale-free graphs)?
- The low-rank regularization relies on truncated nuclear norm (\(\|K\|_{r_0}\)), but \(r_0\) is selected via cross-validation—why not use adaptive rank selection (e.g., based on eigenvalue decay rate of K) to avoid manual tuning?
- When generating synthetic edges, the HGAE decodes intra-cluster neighbor maps, but how to ensure synthetic edges follow real graph structural properties (e.g., small-world, power-law)?
- The LDM is conditioned on prototypes instead of labels, but prototypes are learned from base classes, does prototype shift (between base and novel classes) reduce synthetic node quality for novel classes, and how to mitigate this?

### Questions
See Weaknesses

### Soundness
4

### Presentation
3

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
This paper proposes Low-Rank Few-Shot Graph Diffusion Model (LR-FGDM) for few-shot node classification (FSNC). LR-FGDM augments the limited support set by generating realistic synthetic nodes and edges through a node-level graph diffusion generator (FGDM), which combines a Hierarchical Graph Autoencoder (HGAE) with prototypical regularization and a prototype-conditioned Latent Diffusion Model (LDM). To mitigate noise from diffusion generation, a low-rank regularization inspired by the Low-Frequency Property (LFP) is applied to the transductive classifier, with theoretical guarantees on reduced kernel complexity and generalization error. Extensive experiments across multiple graph benchmarks demonstrate that LR-FGDM consistently outperforms state-of-the-art FSNC methods in both accuracy and structural fidelity.

### Strengths
- The paper introduces a novel and well-motivated framework for few-shot node classification that combines prototype-conditioned diffusion-based augmentation with low-rank regularization. 
- The idea of performing node-level diffusion generation, as opposed to graph-level synthesis, is original and provides a meaningful contribution to generative graph learning. 
- The theoretical analysis grounding the low-rank transductive classifier adds rigor and clarity, while the empirical results demonstrate consistent improvements over strong baselines across multiple benchmarks.

### Weaknesses
- The pipeline figure could be substantially improved. It is visually cluttered and does not clearly separate the HGAE, diffusion, and low-rank classification stages, making it difficult to follow the data flow.

- The length allocation of the paper is also unbalanced: the main text devotes extensive space to model formulation while leaving only two pages for experiments, which contain no figures and limited analysis. Important aspects such as efficiency comparisons and runtime scalability should be included in the main text rather than deferred to the appendix.

- Table 2 suggests that the performance gain from the model’s architectural design may be relatively small, and Table 6 shows that the p-values for COSMIC (LR-FGDM) remain relatively high, raising concerns about the statistical robustness of the reported improvements.

- The paper lacks an ablation study for the diffusion (LDM) component, making it difficult to assess how much of the improvement originates from the proposed generative mechanism rather than from low-rank or prototypical regularization.

### Questions
1. On the Theoretical Scope of Theorem 3.1.
Could the authors explicitly state the assumptions underlying Theorem 3.1 and clarify under what conditions the generalization bound holds? Specifically, does the theorem rely on boundedness, Lipschitz continuity, or i.i.d. sampling assumptions, and is it theoretically valid for all graph structures or only for homophilic/community-structured graphs?

2. On the Missing Diffusion Ablation and Efficiency Evidence.
While Table 2 ablates the low-rank and prototypical regularizations, the LDM module itself is never directly examined. Could the authors include a variant without the diffusion process (e.g., using only the VAE) or with a generic diffusion without the VAE, and report runtime and memory comparisons with COSMIC and DoG to clarify whether the generative component justifies its computational overhead?

### Soundness
3

### Presentation
2

### Contribution
2
