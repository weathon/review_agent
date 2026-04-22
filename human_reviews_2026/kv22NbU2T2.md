# Subspace Kernel Learning on Tensor Sequences

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4

## Abstract
Learning from structured multi-way data, represented as higher-order tensors, requires capturing complex interactions across tensor modes while remaining computationally efficient. We introduce Uncertainty-driven Kernel Tensor Learning (UKTL), a novel kernel framework for $M$-mode tensors that compares mode-wise subspaces derived from tensor unfoldings, enabling expressive and robust similarity measures. To handle large-scale tensor data, we propose a scalable Nystr\"{o}m kernel linearization with dynamically learned pivot tensors obtained via soft $k$-means clustering. A key innovation of UKTL is its uncertainty-aware subspace weighting, which adaptively down-weights unreliable mode components based on estimated confidence, improving robustness and interpretability in comparisons between input and pivot tensors. Our framework is fully end-to-end trainable and naturally incorporates both multi-way and multi-mode interactions through structured kernel compositions. Extensive evaluations on action recognition benchmarks (NTU-60, NTU-120, Kinetics-Skeleton) show that UKTL achieves state-of-the-art performance, superior generalization, and meaningful mode-wise insights. This work establishes a principled, scalable, and interpretable kernel learning paradigm for structured multi-way and multi-modal tensor sequences.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present a tensor-based learning approach with uncertainty modeling.

### Strengths
The problem of tensor learning is very interesting.

### Weaknesses
The boilerplate goes until page 4 using space that seems to be missing for details later. I find the description of the uncertainty vectors and the other material in Sections 3 and 4 not easy to follow. Many details are missing and the description is too brief to be able to follow the discussion.

### Questions
- I would like a more detailed description of the HoT layers. The hyperedges are undefined on the same page.
- What is the index H in equation (1)?
- What are uncertainty vectors? 
- Is the method also applicable for different kernels, such as the Dusk kernel? (DuSK: A Dual Structure-preserving Kernel for Supervised Tensor Learning with Applications to Neuroimages Lifang He, Xiangnan Kong, Philip S. Yu, Ann B. Ragin, Zhifeng Hao, Xiaowei Yang)

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Uncertainty-driven Kernel Tensor Learning (UKTL), a framework for learning from high-order tensor data such as skeletal action sequences. Unlike traditional tensor or kernel methods that flatten data or ignore mode-specific variability, UKTL compares tensors through mode-wise subspaces obtained from tensor unfoldings, preserving spatial, temporal, and semantic structure.

The method builds a kernel that captures both independent and joint correlations across tensor modes. To scale to large datasets, the authors use a Nyström approximation with dynamically learns pivot tensors—cluster centers that represent “local modes” of the data distribution. Anothr module estimates uncertainty for each mode, weighting subspaces based on their reliability. 

Experiments on NTU-60, NTU-120, and Kinetics-Skeleton show that UKTL surpasses graph, hypergraph, and transformer baselines while maintaining efficiency.

### Strengths
- The paper presents a novel approach for learning from structured, high-order data with original compnents such as uncertainty-driven subspace weighting and sum–product Grassmann kernels.

- The theoretical formulation is sound and is a valuable contribution bridging kernel theory,and uncertainty modeling.

- The paper is clearly written and  well-organized

### Weaknesses
1. Evaluation is limited to skeleton data. This is a key limitation which hinders the generality of the method. Although these datasets are standard and challenging, they represent a very narrow class of structured motion data. Demonstrating the approach on a different tensor domains (uch as video features, fMRI signals, or audio visual data) would better support claims of generality and broad applicability. Even a small-scale study in a non-skeletal domain would reinforce the method’s versatility.

2. The use of the term 'mode' should be better clarified from the very beginning, since multi-modality is usually more often used to imply multiple sensor modalities  (also 'mode' is used in yet another fashion in the paper when the authors introduce 'local modes' which as I understand are 'modes within modes' ).  Maybe, clarifying the distinction between tensor-mode structure and cross-modal fusion would avoid overstated claims. This brings to the next point:

3. As already mentioned, while the paper repeatedly refers to “multi-modal” learning, the experiments involve only skeleton data, where “modes” correspond to tensor dimensions (spatial, temporal, coordinate). I believe the framework is also well-suited to genuine multimodal sensor fusion (e.g., RGB + depth + skeleton, avaialble in the NTU datasets by the way). However, such potential is not empirically demonstrated. This point relates also to point n. 1.

4. The full pipeline (MLP + Higher-order Transformer + SVD decomposition + uncertainty module + Nyström kernel) is intricate. While each component is (mostly) motivated, a complexity analysis comparing runtime and memory with strong deep baselines (e.g., CTR-GCN or transformer variants) is missing and would help quantify scalability.

5. The initial MLP: I would appreciate some more details on this module, which at first glance looks like a standard "flattening" operation for the input tensors.

### Questions
Please refer to the weakness above, which highlight what I believe are the main limitations to be addressed.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Uncertainty-driven Kernel Tensor Learning (UKTL), a scalable kernel framework for learning from structured tensor sequences, particularly applied to skeleton-based action recognition.
The central idea is to represent each tensor by its mode-wise subspaces obtained via unfolding and SVD, and then define product and sum Grassmann kernels to measure similarity across these subspaces.
To handle scalability, the authors employ a Nyström kernel linearization with dynamically learned pivot tensors via soft k-means clustering, and further propose uncertainty-aware subspace weighting (Multi-mode SigmaNet) that adaptively down-weights unreliable tensor modes.

The model is trained end-to-end, integrating a lightweight encoder (MLP + Higher-order Transformer), the proposed kernel layer, and a classifier.
Experiments on three large-scale benchmarks (NTU-60, NTU-120, and Kinetics-Skeleton) show that UKTL achieves state-of-the-art performance, outperforming strong graph, hypergraph, and transformer-based baselines. The paper also provides ablation studies on kernel composition, pivot selection, and subspace order, as well as visualizations of Tucker components for interpretability.

### Strengths
- The principled formulation bridging tensor subspaces and kernels is well-formulated. The idea of defining kernels over mode-wise Grassmann subspaces is conceptually elegant and mathematically well-grounded, combining multilinear structure preservation with nonlinear kernel flexibility.

- The introduction of the Multi-mode SigmaNet to learn mode-wise uncertainty is a thoughtful innovation that improves robustness and interpretability, addressing a long-standing issue in tensor learning where all modes are treated equally.

- The experiments are extensive and convincing. UKTL consistently surpasses graph and transformer baselines across three challenging datasets. The ablation results are well-structured and demonstrate the individual benefit of each component (subspace order, kernel choice, uncertainty weighting, pivot count).

### Weaknesses
- The paper combines multiple ideas—Grassmann kernels, Nyström approximation, HoT encoders, and uncertainty modeling—making it dense and difficult to parse. The derivations are mathematically sound but the overall design is somehow overcomplex. What's more, as each component (tensor subspace kernel, Nyström approximation, uncertainty weighting) has precedent, the paper’s contribution lies in their combination and engineering coherence rather than a fundamental breakthrough. 

- The related work section omits a substantial body of research on temporal tensor learning and tensor time-series models， which  have also been extensively studied in ML community, focusing mainly on sparse reconstruction and forecasting tasks through low-rank tensor factorization or probabilistic temporal models like[1][2][3][4]. While UKTL focuses on action recognition, more discussion on these studies would help situate the contribution more clearly within the broader literature on tensor-based temporal representation learning, and highlight the discriminative, kernelized nature of the proposed approach.

- Despite its potential generality, all experiments are in one domain of skeleton action recognition. Without tests on other structured tensor data (e.g., video or neuroimaging), the claimed “multi-modal generality” remains unproven.


[1]: Fang, et al. "Bayesian continuous-time Tucker decomposition." ICML 2022.
[2]: Tao, et al.. "Undirected probabilistic model for tensor decomposition." NeurIPS 2023
[3]: Wang et al. "Dynamic tensor decomposition via neural diffusion-reaction processes." NeurIPS 2024
[3]: Chen, et al. "Functional Complexity-adaptive Temporal Tensor Decomposition." NeurIPS 2025

### Questions
- Could the authors provide a formal analysis (in big-O notation) of the computational complexity per iteration, especially how Nyström pivot count C and subspace dimension p affect training and inference?

- How sensitive is performance to the mixture weight between sum and product kernels? Is μ learned or manually tuned?

### Soundness
2

### Presentation
3

### Contribution
2
