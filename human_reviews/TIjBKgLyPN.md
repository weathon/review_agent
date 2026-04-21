# Adaptive Sparse Allocation with Mutual Choice \& Feature Choice Sparse Autoencoders

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 5, 6, 5

## Abstract
Sparse autoencoders (SAEs) are a promising approach to extracting features from neural networks, enabling model interpretability as well as causal interventions on model internals. SAEs generate sparse feature representations using a sparsifying activation function that implicitly defines a set of token-feature matches. We frame the token-feature matching as a resource allocation problem constrained by a total sparsity upper bound. For example, TopK SAEs solve this allocation problem with the additional constraint that each token matches with at most k features. In TopK SAEs, the k active features per token constraint is the same across tokens, despite some tokens being more difficult to reconstruct than others. To address this limitation, we propose two novel SAE variants, Feature Choice SAEs and Mutual Choice SAEs, which each allow for a variable number of active features per token. Feature Choice SAEs solve the sparsity allocation problem under the additional constraint that each feature matches with at most m tokens. Mutual Choice SAEs solve the unrestricted allocation problem where the total sparsity budget can be allocated freely between tokens and features. Additionally, we introduce a new auxiliary loss function, aux_zipf_loss, which generalises the aux_k_loss to mitigate dead and underutilised features. Our methods result in SAEs with fewer dead features and improved reconstruction loss at equivalent sparsity levels as a result of the inherent adaptive computation. More accurate and scalable feature extraction methods provide a path towards better understanding and more precise control of foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses challenges in sparse autoencoders (SAEs) by introducing Feature Choice SAEs and Mutual Choice SAEs, designed to improve adaptive computation by tackling dead features and feature sparsity. These SAEs are framed as resource allocation problems via the Monotonic Importance Heuristic for adaptive feature selection, where features are adaptively assigned to tokens. Empirical validation of the proposed SAEs demonstrated that they achieve fewer dead features and improved reconstruction loss at similar sparsity levels, attributed to their intrinsic adaptive computation mechanism. Key contributions include new SAE architectures, a heuristic theoretical foundation for adaptive feature allocation, and valuable insights into efficient sparse neural models.

### Strengths
This work has potential for applications needing efficient feature allocation and reduced model complexity, like sparse representation learning. The adaptive computation strategy to reduce dead features could improve SAE design, making it more efficient for high-dimensional settings. Though incremental, the contributions offer valuable insights into current SAE limitations. The paper provides a basic experimental setup with empirical validation on real-world datasets. The originality of this work lies in framing SAEs as resource allocation problems and introducing the Feature Choice SAE and Mutual Choice SAE variants, offering a new approach to adaptive computation.

### Weaknesses
- **Lack of Reproducible Code**: The paper does not include reproducible code for the numerical experimental results, making it difficult to verify the empirical findings. Consequently, my comments are based solely on the results as presented in the paper.

- **Monotonic Importance Heuristic:** The theoretical framing of models such as TopK SAEs, Feature Choice SAEs, and Mutal Choice SAEs as resource allocation problems is coherent. However, it is important to verify the mathemtical rigor in their proofs, even though the authors offer an informal theoretical motivation in Appendix G.

- **Lack of Extensive and Comprehensive Experimental Setup**: Although the comparisons with standard SAEs aim to demonstrate improvements in dead features and reconstruction loss, the evaluation lacks thoroughness and would benefit from a more comprehensive approach (see additional details in Questions).

- **ICLR 2025 Citation Style**: The paper should adhere to the ICLR 2025 citation style: when the authors or publication are part of the sentence, the citation should appear without parentheses using \citet{} (e.g., “See Hinton et al. (2006) for more information.”). Otherwise, the citation should be placed in parentheses using \citep{} (e.g., “Deep learning shows promise to advance AI (Bengio \& LeCun, 2007).”) The authors should revise all citations to ensure they adhere to this standard format.

- **Inconsistent Mathematical Notations**: The notation for multivariate variables in equations (1), (2), (3), and beyond is inconsistent, alternating between bold and regular font. Following a consistent notation style, as recommended by the ICLR guidelines, would enhance clarity.

- **Missing Information on Citations**: There are several instances of missing conference or journal names in the references. The authors should address these omissions to ensure completeness. For example, on lines 648-650, the correct citation format should be: "William Fedus, Barret Zoph, and Noam Shazeer. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity, 2022. *Journal of Machine Learning Research*, 23.120 (2022): 1-39."

- **Typos:**   **Line 206**: The mask $S$ should be written in bold as $\mathbf{S}$. **Line 266:** aux\_$k_l$oss should be aux\_k\_loss.  **Line 903:** "Algorithm 1 algorithm 1" should be "Algorithm 1". **Line 452:** "fig. 6" should be "Fig. 6".

### Questions
- **Lack Comparison with Concurrent Work:** In addition to the baseline SAE types listed in Table 2, the authors should include a comparison with Bussmann et al. (2024) [1], which introduced BatchTopK-a method that also offers adaptive computation and could serve as a useful benchmark agaist Mutual Choice SAEs.

References:

[1] Bart Bussmann, Patrick Leask, and Neel Nanda. BatchTopK: A Simple Improvement for TopK-SAEs. July 2024. https://www.alignmentforum.org/posts/Nkx6yWZNbAsfvic98/batchtopk-a-simple-improvement-for-topk-saes

- **Line 195**: Could the authors clarify the definition of the pre-sparsifying activation feature magnitudes $\mathbf{z}'$? I assume this refers to the token-feature affinities, but how is this value determined for real applications?

- **Line 258-260**: The authors should provide a more detailed definition of "dying features" to improve clarity.

- **Table 1**: Could the authors clarify the term "w/ resampling"? Is it a fair comparison to evaluate results with and without resampling? Additionally, in practical applications, which approach yields a lower percentage of Dead Features?

- **Line 855-857:** The authors mentioned the failure mode "dictionary Collapse", which I personally think it is analogous the the collapsing problem in sparse mixture of experts models. The authors should demonstrate this via visualization or quantitative empirical results.

- **Line 855-857**: The authors mention the failure mode "dictionary collapse," which I believe is analogous to the collapsing problem seen in sparse mixture of experts models [1,2]. To strengthen this point, the authors should consider demonstrating it through visualizations or quantitative empirical results as in [1,2].

References:

[1] Chi, Z., Dong, L., Huang, S., Dai, D., Ma, S., Patra, B., Singhal, S., Bajaj, P., Song, X., Mao, X.L. and Huang, H., 2022. On the representation collapse of sparse mixture of experts. Advances in Neural Information Processing Systems, 35, pp.34600-34613.

[2] Do, T.G., Pham, Q., Nguyen, T., Doan, T.N., Nguyen, B.T., Liu, C., Ramasamy, S., Li, X. and Steven, H.O.I., HyperRouter: Towards Efficient Training and Inference of Sparse Mixture of Experts. In The 2023 Conference on Empirical Methods in Natural Language Processing.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces two novel variants of Sparse Autoencoders (SAEs) - Feature Choice SAEs and Mutual Choice SAEs - for extracting interpretable features from neural networks. The key innovation is framing the sparsifying activation function as a resource allocation problem, where a fixed sparsity budget must be distributed across token-feature matches. The authors also introduce aux_zipf_loss, which addresses underutilized features by generalizing the existing aux_k_loss. The proposed methods show improvements in reconstruction accuracy and feature utilization compared to existing approaches like TopK SAEs, particularly in handling dead features.

### Strengths
1. The resource allocation perspective on sparsifying activation functions provides an intuitive framework for understanding and improving SAEs.
2. The Feature Choice and Mutual Choice variants offer clear improvements over existing methods. For example, the Feature Choice SAEs reliably achieve zero dead features even at large scale and the scaling results are good that is able to handle up to 34M dimension models.
3. The discussion on the Zipf distribution in feature usage is also thoughtful.

### Weaknesses
1. One of my main concerns is about the presentation. Some technical details are scattered throughout different sections and the Figures and Tables are not clear or elegant which are beyond the width of the paper. The figures could also be better explained. The paper would benefit from clearer organization and improved writing.

2. The experiment results are somewhat not sufficient. Experiments primarily focus on GPT-2 small. More diverse model architectures would strengthen claims.

3. The contribution of this work is somehow incremental with limited significance and impact. The core problem about dead features in SAEs handled by this paper has limited impact in the interpretability community.

### Questions
1. The paper claims improved reconstruction accuracy, but how does this translate to downstream task performance beyond the provided loss metrics?
2. Could the authors elaborate on the computational overhead of Feature Choice and Mutual Choice SAEs compared to TopK SAEs?
3. How well do these methods generalize to other types of neural networks beyond language models?
4. What are the practical implications of these improvements for real-world applications?

### Soundness
3

### Presentation
2

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
The paper proposes two new variants of sparse autoencoders (SAEs), namely Feature Choice SAEs and Mutual Choice SAEs, to address the limitations of existing TopK SAEs. A key improvement is that the proposed models adaptively allocate the total sparsity budget between tokens and features, resulting in improved reconstruction performance (Figure 4). Additionally, the paper introduces a new auxiliary loss function, aux_zipf_loss, to mitigate underutilized features based on the observation that the Zipf distribution generally fits most of the feature densities, except at the lower end. The proposed aux_zipf_loss enhances feature utilization and reduces dead features (Table 1).

### Strengths
Clear Visualization: Figure 1 effectively visualizes the design of the sparsifying activation functions and the differences between the proposed methods and existing approaches.

Adaptive Computation: Feature Choice SAE and Mutual Choice SAEs demonstrate improved performance through adaptive computation, effectively allocating more resources to more complex tokens.

Mitigating Dead Features: The introduction of the aux_zipf_loss and the proposed definition of dying features successfully mitigate the occurrence of dead and underutilized features, leading to better overall model performance.

### Weaknesses
Organization of Sections: Sections 3 and 4 appear somewhat disorganized, with some concepts in Section 3 that would be better suited for the Methods section. This restructuring could improve the overall flow and clarity of the paper.

Limited Evaluation Scope: The analysis of the Zipf distribution fit (Figure 2) is an interesting discovery, but it appears that the authors only conducted this analysis on GPT-2. Broader validation on different models or datasets would strengthen the claims.

### Questions
1. In Line 344, the authors mentioned that they evaluated the interpretability of the SAEs using Juang et al. (2024)'s automated interpretability (AutoInterp) process. Could the authors provide results of this evaluation?

2. Could the authors provide more details about the experimental setup, including the datasets and the exact procedures used, to ensure the reproducibility of the results? Ideally, releasing the codebase would greatly benefit the community.

3. Have the authors consistently observed the discrepancy between the feature density and the Zipf curve in different settings, or is this observation limited to the GPT-2 experiments presented here?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper proposed Feature Choice and Mutual Choice Sparse Autoencoder (SAE). The feature choice version enforces sparsity by restricting each feature to match with at most $m$ tokens; the mutual choice version allows any choice of token-feature matching. zipf distribution is used to set the constraints for feature choice, and an additional auxiliary loss is used to prevent dead features. Numerical experiments show the proposed methods outperform existing SAE methods in terms of reconstruction at different sparsity levels and percentage of dead features

### Strengths
The proposed methods effectively reduce dead features and allow for flexible feature allocations, which allows the model to utilize features better and potentially more interpretable.

### Weaknesses
1. The experiments are rather limited, only one setting with GPT2 activations is performed. 
2. To demonstrate the significance of the results, I think it would be better to include the average and standard deviation of the evaluated metrics across different runs
3. SAE is described as a promising approach for extracting sparse, meaningful, and interpretable features from neural networks. But under the current experiment setup, it is hard to tell how the extracted features are meaningful and interpretable.
4. The problem is framed as a resource allocator in section 3, but I think it is not very clear how the edge weights are defined. If I understand correctly, it is somehow defined as related to the magnitude of feature activation, e.g. in line 483. 
5. There are some typos, e.g. in lines 210 and 221 $S$ is defined the same way, but shouldn't it be different?  in line 266, l is written in as a subscript in aux_k_loss

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
2
