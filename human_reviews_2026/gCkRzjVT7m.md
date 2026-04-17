# Parameter-Efficient Fine-Tuning of LLMs with Mixture of Space Experts

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Large language models (LLMs) have achieved remarkable progress, with Parameter-Efficient Fine-Tuning (PEFT) emerging as a key technique for downstream task adaptation. However, existing PEFT methods mainly operate in Euclidean space, fundamentally limiting their capacity to capture complex geometric structures inherent in data.  While alternative geometric spaces, such as hyperbolic geometries for hierarchical data and spherical manifolds for circular patterns, offer theoretical advantages, constraining representations to single manifold types fundamentally limits expressiveness, even with learnable curvature parameters.

To address this, we propose \textbf{MoS} (Mixture of Space), a unified framework that leverages multiple geometric spaces simultaneously to learn richer, curvature-aware representations. Building on this scheme, we develop \textbf{MoSELoRA}, which extends Low-Rank Adaptation (LoRA) with heterogeneous geometric experts, enabling models to adaptively select or combine appropriate geometric spaces based on input. 
Besides, to address the computational overhead of frequent manifold mapping, we develop a lightweight projection mechanism. Moreover,
We provide empirical insights into how curvature optimization impacts training stability and model performance.
Our experiments across diverse benchmarks demonstrate that MoSELoRA consistently outperforms strong baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors propose an MoSELoRA framework for parameter-efficient fine-tuning (PEFT) of large language models (LLMs). This framework integrates multiple geometric spaces such as Euclidean, hyperbolic, and spherical, into a unified architecture. The key idea is that these different linguistic structures are better represented in different geometric manifolds. In this regard, the MoSELoRA extends Low-Rank Adaptation (LoRA) by incorporating a Mixture of Space (MoS) approach, where tokens are dynamically routed to geometric experts based on input context. The authors introduce a lightweight routing mechanism and curvature-aware optimization to reduce computational overhead and improve training stability. Experiments on natural language understanding and mathematical reasoning benchmarks demonstrate consistent improvements over strong baselines.

### Strengths
1. The integration of multiple constant-curvature spaces into PEFT is original and well-motivated by linguistic and structural properties of language.
2. The use of stereographic projection and Lorentz model for efficient manifold transitions is elegant and avoids expensive exp/log mappings.
3. The paper includes curvature dynamics analysis, optimizer comparisons, and single-space expert variants to validate design choices.
4. The proposed approach demonstrates speedups and reduced parameter activation compared to other MoE-based methods.
5. MoSELoRA outperforms state-of-the-art PEFT methods across multiple benchmarks, especially in mathematical reasoning tasks.

### Weaknesses
1. The paper is focused entirely on LLMs and NLP benchmarks. It is unclear how this approach will be applied to multimodal LLMs unless extended to vision-language models or geometric representation learning in vision.
2. The routing mechanism that assigns tokens to geometric experts is described as “lightweight” and based on token-level projections, but the mathematical formulation lacks clarity. There is no theoretical guarantee that the routing leads to optimal space selection. 
3. The paper does not analyze the stability or convergence properties of curvature learning. For instance, how does curvature interact with gradient flow in high-dimensional manifolds? Also, does the routing converge to a stable assignment? Is it differentiable and robust to noise?
4. There is no in-depth analysis of why certain tasks (e.g., math reasoning) benefit from hyperbolic embeddings beyond empirical observation. A theoretical framework linking task structure to manifold geometry would strengthen the claims.
5. While the method generalizes well to unseen math problems, its performance on other domains (e.g., commonsense reasoning, multi-hop QA) is not explored.

### Questions
1. Can MoSELoRA be extended to vision-language models? Have the authors considered applications in geometric scene understanding? The current evaluation is limited to LLMs and NLP benchmarks, with no experiments on multi-modal reasoning or vision-language grounding.
2. Is there any theoretical guarantee that the routing leads to optimal space selection? Could the authors provide visualizations of token embeddings across different spaces to support interpretability?
3. Does the routing converge to a stable assignment, and is it robust to noise? Are curvature parameters sensitive to initialization, and would curvature regularization improve stability?
4. Can the authors provide a theoretical framework linking task structure to manifold geometry?
5. The authors do not provide the source code. Can they clarify implementation details or release the code to ensure reproducibility?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MoSELoRA, a new fine-tuning method that generalizes the Mixture of LoRA Experts (linear mapping in the Euclidean space) to Mixture of Space Experts (non-Euclidean geometric mappings, including hyperbolic and spherical spaces). The authors proposed a unified representation of the mapping for all three considered spaces. Numerical experiments have been reported on several standard fine-tuning benchmarks, demonstrating the effectiveness.

### Strengths
1. The idea of the mixture of spaces is interesting.
2. The development of the lightweight token routing mechanism and the unified mapping for three spaces is interesting. 
3. The proposed simplification achieves an acceleration of the computation for the geometric mapping.

### Weaknesses
1. The overall scope of the experiments is somewhat limited in terms of the evaluated models, model sizes, datasets, and tasks. Expanding the experimental coverage would strengthen the empirical claims.
2. The paper remains somewhat vague in several important aspects. The clarity and presentation could be improved by providing more details, such as:
- how each expert is selected during the forward pass
- how the routing value is computed for each token and expert
- what the auxiliary loss is
- what is the resulting full fine-tuning method
- which rank is used for each method in the Table in the experiments
- what are the hyperparameters that are used for all other baseline methods and are they optimally tuned, 
- what is the total number of trainable parameters for all the other baseline methods. 

3. The advantage of the proposed method over using only the hyperbolic space appears relatively small for the current setup. It might be more interesting to see the following experiments: 

 - The paper notes that different geometric spaces may be better suited to different datasets. Suppose dataset A aligns best with Euclidean geometry. Would the trained MoSELoRA model, after being trained on a large and diverse dataset (as in the current setup), tend to select the Euclidean space more frequently during inference when being evaluated on dataset A?

- Conversely, if MoSELoRA were trained only on dataset A (which is well-suited to the Euclidean space), would it again favor the Euclidean space more often during training and inference?

### Questions
1. It is a bit surprising that LoRA with a higher rank and DoRA underperform the base LoRA.
2. Are there any reasons why the target modules do not include q,k, and v projections?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a novel and compelling approach to Parameter-Efficient Fine-Tuning (PEFT) by integrating Low-Rank Adaptation (LoRA) with a Mixture-of-Experts (MoE) framework across heterogeneous geometric spaces (Euclidean, Hyperbolic, Spherical). The proposed MoS and MoSELoRA methods dynamically route tokens to the most suitable space, enhancing the model's ability to capture diverse semantic structures.

### Strengths
1) Mixture of Space (MoS) considers more kinds of constant curvature spaces with learnable Gaussian curvature.

2) Ablation experiments make sense.

3) A comprehensive introduction to the references is given.

### Weaknesses
Quality: 
1) The proposed method has a serious flaw in its geometric interpretation, manifested as structural inversion. Specifically, linear layer transformations should be performed in the embedding space, and vector additions should be carried out in the full space to present a clear geometric meaning, not the other way around. Further, no experiment verifies the advantages of learnable curvature families over fixed curvature families, and no theoretical evidence indicates any advantage of introducing non-Euclidean geometric for LoRA fine-tuning. Besides, there is a lack of experimental comparison with HypLoRA [1] and HELM [2].

Clarity: 
1) The fine structure of Figure 1 is unclear (projection misleading, no reflection of learnable curvature, no evident basis for routing selection).
2) The actual method associated with Figure 3 is unclear and there is no comparison with traditional MoE methods.
3) The preliminary knowledge is poorly organized. In fact, the discussion about the exponential map is not beneficial for understanding the proposed method in this paper.

Significance: 
1) The performance improvement shown in Table 1 is not very significant.

[1] Hyperbolic fine tuning for large language models. arXiv preprint arXiv:2410.04010, 2024.

[2] Hyperbolic large language models via mixture-of-curvature experts. arXiv preprint arXiv:2505.24722, 2025.

### Questions
see the Weaknesses.

### Soundness
2

### Presentation
2

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
The paper introduces MoSELoRA (Mixture of Space Experts LoRA), a parameter-efficient fine-tuning (PEFT) framework that integrates heterogeneous constant-curvature spaces (hyperbolic, spherical, and Euclidean) within a unified Mixture of Space (MoS) formulation. Unlike prior LoRA variants that operate purely in Euclidean geometry, MoSELoRA dynamically assigns token representations to curvature-specific subspaces through a lightweight routing mechanism, thereby adapting the fine-tuning process to the intrinsic geometry of language data. The method learns curvature parameters end-to-end, stabilizes training via separate optimizers for curvature and LoRA weights, and ensures gradient boundedness across all geometric spaces. Empirical results demonstrate consistent improvements on benchmarks.

### Strengths
1. The paper presents a well-motivated observation that existing PEFT methods assume flat Euclidean geometry, which may be suboptimal for hierarchical or cyclic semantic structures. MoSELoRA bridges this gap by modeling curvature diversity through a mixture of geometric experts.
2. The lightweight routing and space-mapping method achieves over 4x speedup over standard exp–log scheme
3. The curvature evolution analysis demonstrates interpretable geometry adaptation (Fig. 2): lower transformer layers remain near-Euclidean, while higher layers evolve toward hyperbolic or spherical curvature depending on task semantics

### Weaknesses
1. Experiments are limited to the Qwen2-1.5B model. There is no analysis on how this method scales for larger models or different architecture 
2. Efficiency is only reported in terms of runtime, it is unclear if there is any memory overhead or advantages  
3. The motivation for employing the curvature spaces (hyperbolic, spherical, Euclidean) is well supported by prior literature. However, the paper does not empirically justify the necessity of using all three simultaneously. Table 3 compares single-space variants against the full mixture but omits two-expert combinations (e.g., hyperbolic + Euclidean), leaving it unclear whether the full tri-expert setup is essential or if similar benefits could be achieved with fewer experts and reduced complexity.

### Questions
Please refer to the weaknesses. 
* Would integrating MoSELoRA with quantization-based (e.g., QLoRA [1]) or gradient-efficient PEFT methods (e.g., GaLore [2]) yield similar performance and efficiency gains?

### Soundness
3

### Presentation
3

### Contribution
3
