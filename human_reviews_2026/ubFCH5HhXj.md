# DSTGN: Decoupled SpatioTemporal Graph Network for Multi-scenario Dynamics Learning

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Machine Learning (ML) methods have played a pivotal role in a wide range of dynamics tasks, such as physical simulation, multi-modal interaction, and real-time prediction. Among them, Graph Neural Networks (GNNs) have emerged as effective surrogates for numerical solvers, thanks to their ability to model pairwise interactions between nodes and their neighboring edges. However, the inherent tendency of GNNs to perform local aggregation limits their expressiveness in capturing intricate global interactions. To address this limitation, we propose a general framework, Decoupled SpatioTemporal Graph Network (DSTGN), to enhance prediction performance across various downstream tasks. DSTGN enhances latent feature representations by decoupling spatial connectivity in both the physical and latent spaces. In parallel, it introduces a learnable temporal integration mechanism that decouples inter-step dynamics at each latent layer, effectively mitigating error accumulation during autoregressive inference. Extensive experiments on multiple types of benchmarks demonstrate that DSTGN consistently outperforms existing baselines in both accuracy and generalization, across regular and irregular domains, as well as static and dynamic meshes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Decoupled SpatioTemporal Graph Network (DSTGN), a framework that improves GNN-based dynamic prediction by addressing the limitations of local aggregation. DSTGN decouples spatial connectivity in both physical and latent spaces to better capture global interactions and introduces a learnable temporal integration mechanism to reduce error accumulation. Experiments show that DSTGN consistently outperforms existing methods in accuracy and generalization across diverse simulation domains.

### Strengths
- The paper is clearly written and well-structured, making it easy to follow the motivation, methodology, and results. 
- The paper proposes a learnable temporal integration strategy, inspired by predictor–corrector schemes, introduces a novel and theoretically grounded perspective that distinguishes this work from standard message-passing approaches.
- The experimental evaluation is extensive, covering diverse settings including 2D/3D PDE systems, real-world oceanographic data, and dynamic UAV simulations. The inclusion of detailed ablation studies further strengthens the empirical validation of each architectural component.

### Weaknesses
- The decoupling in the physical space, which separates the graph into nodes and edges for message passing, is conceptually aligned with the design used in MeshGraphNets and most existing GNN-based simulators. As such, this component does not introduce a clear methodological novelty.
- The proposed S2 and T1 blocks essentially perform additional forms of message aggregation and transformation within the latent space and temporal domain. While they provide performance gains, the overall architecture still closely resembles MeshGraphNets with auxiliary feature fusion. 
- The current predictor–corrector temporal strategy is restricted to a second-order formulation. Although effective for moderate systems, it may not remain stable under chaotic or stiff dynamics. Extending the framework to higher-order or adaptive integration schemes could further enhance long-term accuracy and robustness.

### Questions
1. In table 2, the S1 configuration appears to correspond directly to the MeshGraphNets baseline, while the variants (S1 + S2, S1 + T1, S2 + T1, and the full model) add additional blocks and, therefore, a larger number of parameters. Are the improvements due to design or model size? Please clarify whether parameter counts were controlled for fair comparison.
2. The term “Order” in the first column of Table 3 is not clearly defined. Please elaborate on what each “order” (e.g., S → T, T → S, S ∥ T) represents, how these learning patterns differ in the data flow or training process.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Summary: GNNs are not expressive in capturing global interactions. To mitigate this, the authors propose decoupled spatio-temporal GN to enhance prediction. Additionally, introduce learnable time integration to decouple inter-step dynamics. They introduce a predictor corrector framework for the time-stepper.

### Strengths
The authors propose a learnable time-integration strategy that leverages a predictor-corrector setup. 
The authors have conducted rigorous experimentation to showcase the improved performance of their architecture

### Weaknesses
1. Several grammatical errors. Needs to be improved. 
2. Section 3.1.1 jumps into the architecture directly, without providing any introduction. The virtual intermediate variable for instance is a term that is not defined prior to 3.1.1. 
3. “ there is a unknown” – poor writing. 
4. “exploiting more virtual intermediate variables leads to a denser distribution,” – not clear what this means. 
5. “a intractable issue arises as the dimensionality increases” - dimensionality of what? 
6. “Thus, we retain only the first two components in Eq. 1” – as opposed to what? What are the remaining components? Why are they not retained?

### Questions
1. Could you explain how the model facilitates global modeling? There does not seem to be anything in section 3.1 that indicates that there is some form of global information aggregation being performed here. I would recommend re-writing all of 3.1. 
2. The following claims are made in Table S4 but many of the terms used in the table are not mentioned elsewhere in the paper - Global modeling, temporal strategy, spatial strategy (What are these referring to? )
3. Overall the writing is very unclear and it is hard to differentiate the difference between the proposed architecture and UPT, with the exception of predictor-corrector within the time-stepper module.

### Soundness
1

### Presentation
1

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
This paper focuses on learning dynamic systems and proposes a decoupled method that process spatial features and temporal update steps separately. The spatial encoder adapts an edge-based GNN and the temporal block includes the predictor-corrector sampling into the temporal update. The decoupled design enables the proposed DSTGN to achieve more precise temporal sampling process and mitigate the error accumulation issue.

### Strengths
1. It is a novel and valuable idea to introduce predictor-corrector updates for learning dynamic systems.
2. The decouple of spatial and temporal representations provides insights for future PDE-based models.
3. The experimental results validate the generalizability of DSTGN on a variety of dynamic systems.

### Weaknesses
1. While the spatial block of the proposed DSTGN is based on the message passing over node and edge embeddings, it does not fully exploit the high-order simplex as the authors have mentioned.
2. Most of the chosen baseline methods are neural operator models. Since the learning objective and architecture of DSTGN are more similar to mesh-based models, including more recent and relevant baselines should be considered to make a comprehensive comparison.
3. The learnable PC update module is proposed as a solution to error accumulation. However, its effectiveness is not theoretically guaranteed.

### Questions
1. Have you investigated any theoretical properties (e.g., stability, convergence) of the proposed temporal update rule?

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
This paper introduces a general framework named Decoupled SpatioTemporal Graph Network (DSTGN) for learning multi-scenario dynamics. DSTGN decouples spatial connectivity in both physical and latent spaces through S-blocks and introduces a learnable time integration mechanism via T-blocks to mitigate error accumulation in autoregressive inference. Experiments on multiple datasets demonstrate superior performance and generalization across static/dynamic and regular/irregular meshes.

### Strengths
- The paper is well-written and easy to follow.
- The spatial blocks consider both physical space and latent space interaction.
- Extensive experiments demonstrate the superior performance of the proposed method over the baselines.

### Weaknesses
- My major concern is novelty. Although the paper introduces the integration of decoupled spatial and temporal learning, most architectural components (e.g., encoder–processor–decoder, message passing, predictor–corrector) are adapted from existing works.
- The paper would benefit from a clear comparison of parameter counts across different models to better demonstrate computational efficiency and fairness in evaluation.
- In Table 3, several symbols and abbreviations are insufficiently explained, making it difficult for readers to fully understand.

### Questions
- Line 205: The paper mentions “channel cluster rather than node cluster.” Could the authors clarify what this means in practice and how the grouping operation is implemented?
- In Table 2, there is a notable performance drop when using the S2 + T1 configuration. Could the authors provide an explanation for why this combination leads to substantially weaker results?
- In Table 7, the reported running time of DSTGN appears faster than some baselines such as MGN. Could the authors elaborate on the factors contributing to this efficiency advantage?

### Soundness
3

### Presentation
3

### Contribution
2
