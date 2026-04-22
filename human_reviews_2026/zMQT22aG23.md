# MSRS: Adaptive Multi-Subspace Representation Steering for Attribute Alignment in Large Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Activation steering offers a promising approach to controlling the behavior of Large Language Models by directly manipulating their internal activations. However, most existing methods struggle to jointly steer multiple attributes, often resulting in interference and undesirable trade-offs. To address this challenge, we propose Multi-Subspace Representation Steering (MSRS), a novel framework for effective multi-attribute steering via subspace representation fine-tuning. MSRS reduces inter-attribute interference by allocating orthogonal subspaces to each attribute, isolating their influence within the model’s representation space. MSRS also incorporates a hybrid subspace composition strategy: it combines attribute-specific subspaces for unique steering directions with a shared subspace for common steering directions. A dynamic weighting function learns to efficiently integrate these components for precise control. During inference, MSRS introduces a token-level steering mechanism that dynamically identifies and intervenes on the most semantically relevant tokens, enabling fine-grained behavioral modulation. Experimental results show that MSRS significantly reduces attribute conflicts, surpasses existing methods across a range of attributes, and generalizes effectively to diverse downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Multi-Subspace Representation Steering (MSRS), a novel framework designed to address the challenge of simultaneously controlling multiple attributes in LLMs without the typical interference and performance trade-offs. Existing methods often struggle when steering for conflicting objectives like truthfulness and bias. MSRS overcomes this by decomposing the model's internal representation space into orthogonal subspaces: a shared subspace to capture common steering directions and multiple attribute-specific subspaces to isolate control. A key contribution is the use of SVD to adaptively determine the dimensionality of each subspace based on its expressive needs, rather than using fixed-size partitions. Furthermore, MSRS introduces a dynamic token intervention mechanism that identifies and applies steering to the most semantically relevant tokens for each attribute, enabling more precise, fine-grained control. Through comprehensive experiments, the authors demonstrate that MSRS surpasses strong baselines, effectively mitigating attribute conflicts and achieving superior performance across diverse models and tasks while preserving the model's general capabilities.

### Strengths
- Introduces a novel hybrid architecture that disentangles steering vectors into a shared subspace for common features and orthogonal, attribute-specific subspaces. 
- The porposed method use of SVD to adaptively allocate subspace dimensionality based on the captured energy of activation differences. This is an advance over earlier approaches that relied on equal-sized partitions.
- The method demonstrates ability to navigate attribute trade-offs, such as concurrently improving scores on TruthfulQA and BBQ, a task where baselines often sacrifice performance on one metric to gain on the other.

### Weaknesses
- If I understand it correctly, the scalability of the MSRS framework is not tested beyond attribute pairs. With a fixed total rank R, it is unclear if the method can effectively manage the trade-offs when steering a larger number of attributes, as the capacity of each subspace would necessarily shrink.
- The 60% energy threshold for defining the shared subspace is presented as a fixed hyperparameter without a sensitivity analysis. This value is critical for balancing shared and specific control, and its robustness across different models and attribute combinations is not validated.
- The paper lacks a qualitative analysis to provide insight into the semantic nature of the learned subspaces. While demonstrating good quantitative results, it misses what features are being isolated.

### Questions
1. How does MSRS perform when steering more attributes simultaneously (e.g., truthfulness, harmlessness, and verbosity)? Given a fixed total rank R, does the decreasing capacity available for each attribute-specific subspace lead to a significant performance drop-off, and if so, at what point?
2. Could you provide a direct comparison of the inference latency of MSRS against the fixed-position baseline, ReFT? This would clarify the practical computational cost introduced by the dynamic token selection mechanism.
3. For steering baseline, have you considered recent works like DoLA [1] and SEA [2]? You should at least mention these works in your related work. 


[1] Chuang et al. (2023) DoLa: Decoding by contrasting layers improves factuality in large language models

[2] Qiu et al. (2024) Spectral Editing of Activations for Large Language Model Alignment

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
The paper proposes MSRS (Multi-Subspace Representation Steering) for multi-attribute control of LLMs. Building on representation fine-tuning (ReFT), MSRS: (i) splits a low-rank representation space into attribute-specific and a shared subspace; (ii) uses SVD of attribute activation differences to size/initialize subspaces; (iii) adds alignment losses to keep the learned subspaces close to the SVD priors; and (iv) introduces dynamic token-position selection based on similarity to the learned subspace for inference-time interventions. Experiments across LLaMA-2/3-8B, Qwen2-7B, and Mistral-7B show consistent gains on TruthfulQA, BBQ, Alpaca, Refusal/Sorry-Bench, HelpSteer, and GLUE, with ablations on rank, shared-vs-private ratio, layer choice, and token selection.

### Strengths
1. The method is novel, well motivated and modular.
2. Dynamic token routing is simple and interesting, consistent gains over last-token steering.
3. Comprehensive experiments across several base models and tasks, statistically reported and supported by ablations.

### Weaknesses
1. As I understand, the size of each attribute subspace is decided by the number of top singular vectors. However, in line 202, $H_{res} = \tau_i - B_{shared}^{\top} B_{shared} \tau_i$ which should have the same dimension as $\tau_i$, i.e, $\mathbb{R}^d$. The authors's performing SVD on $H_{res}$ is not making sense to me since this is just a single vector. Therefore, no adaptive size of subspace can be deduced.
2. Though not clearly stated, I am assumming that $r = r_s + \sum_{i=1}^nr_i$, where $r$ is the size of $R$. How did the authors ensure this equality when constructing $B_i$'s and $B_{shared}$? Moreover, if my assumption is true, does that mean $r > n$ no matter how large $n$ is?
3. In line 242, the authors introduce a alignment loss:
$$
\mathcal{L}\_{align} = 1 - \dfrac{\langle R, S_{align} \rangle}{1- \Vert R\Vert\_2 \Vert S\_{align} \Vert\_2}
$$

how is the matrix inner product computed?

4. MSRS adds projection(s), mask MLP, and token selection. You reference compute cost in Appendix F; please summarize throughput/latency overhead and VRAM vs. ReFT and CAA at inference for typical sequence lengths.
5. Multi-attribute sets are evaluated in pairs or small sets. How does MSRS scale when n (attributes) grows in terms of rank budgeting, mask sparsity, and interference? A scaling-law-style study would strengthen claims.

### Questions
Refer to the weakness

### Soundness
2

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
3

### Summary
The paper proposes Multi-Subspace Representation Steering (MSRS), a novel framework for multi-attribute control in large language models by allocating orthogonal subspaces for each attribute and a shared subspace for common steering directions. This design mitigates interference between attributes and enables more precise behavior modulation. MSRS also introduces a dynamic token-level steering mechanism during inference, selecting semantically relevant tokens for intervention. Experiments across multiple models (LLaMA2, LLaMA3, Qwen2, Mistral) and tasks (multiple-choice, open-ended generation) show that MSRS significantly reduces attribute conflicts and outperforms existing steering methods, while also generalizing effectively to standard NLP benchmarks such as HellaSwag and GLUE.

### Strengths
- Addresses a practical challenge in activation steering: controlling multiple attributes with minimal cross-attribute interference.
- Introduces a novel decomposition into a shared subspace and multiple attribute-specific subspaces.
- Covers an extensive set of datasets, metrics, models, and baseline methods.
- Demonstrates strong results across multiple benchmarks and model families.
- Includes robustness evaluation showing that general capabilities remain unaffected under steering.
- Provides ablation studies validating the role of the shared subspace in integrating multi-attribute features.
- Clearly written, with intuitive visualizations and easy-to-follow presentation.

### Weaknesses
See Questions

### Questions
- Line 158: "ReFT assumes a single attribute per input"—but since R is r-dimensional, shouldn't it be capable of representing multiple attributes?
- Lines 195–199: Why does selecting more top vectors for the shared directions allow adaptive "subspace sizes for each attribute based on its expressive needs"? Wouldn't allocating more space to one attribute reduce the shared subspace size?
- Lines 201–204: Since $\tau_i$ is a vector, shouldn't $H^{(i)}_\text{res}$ also be a vector?
- Equation 2: The formulation of $B_i$ implies that $B_i$ and $B_j$ can overlap via shared directions. Wouldn't this bias the alignment of $R$ toward shared directions in $S_\text{align}$?
- The concept of dynamic intervention has been explored in prior work (e.g., [1, 2]). The authors should discuss connections and distinctions.
- Line 425–427: Does "last token" mean steering is applied only to the last token of the input? Please clarify.

[1] Programming Refusal with Conditional Activation Steering  
[2] Angular Steering: Behavior Control via Rotation in Activation Space

### Soundness
2

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
4

### Summary
This paper proposes Multi-Subspace Representation Steering (MSRS), a novel framework for multi-attribute control in LLMs through subspace representation fine-tuning. The method addresses attribute interference by allocating orthogonal subspaces to each attribute while maintaining a shared subspace for common steering directions, combined with dynamic token-level interventions during inference. Experimental results demonstrate improvements over existing methods across multiple attributes and downstream tasks.

### Strengths
- This paper addresses a practical problem of multi-attribute steering in LLMs, where existing methods struggle with attribute interference and trade-offs.

- The proposed approach is well-motivated and principled, combining orthogonal subspace decomposition with SVD-based adaptive dimensionality allocation and a shared subspace for capturing common steering directions.

- This paper is clear and easy to understand.

### Weaknesses
- **Training Data Specifications and Potential Overlap.** A critical detail appears to be missing or unclear in the paper: what specific training data is used to fine-tune the steering representations? Section 5.1 describes the evaluation datasets (TruthfulQA, BBQ, Alpaca, Refusal, HelpSteer) and mentions using these for training steering functions, but the exact training splits, data sizes, and construction procedures are not sufficiently detailed. More importantly, there is concern about potential overlap between the training data used for learning steering subspaces and the benchmark test sets used for evaluation. If the steering functions are trained on samples from the same distributions as the test benchmarks, this could lead to inflated performance estimates and raise questions about generalization. The paper should explicitly clarify: (1) the exact datasets and splits used for training each attribute's steering function, (2) whether any samples overlap with evaluation benchmarks, and (3) how data contamination is prevented or controlled.

- **Scalability to More Attributes.** While the paper demonstrates effective steering for pairs of attributes (truthfulness-bias, instruction-following-refusal, helpfulness-coherence-verbosity), the scalability limits remain unclear. What is the maximum number of attributes that can be jointly controlled before performance degrades? The current experiments focus on relatively compatible attributes within similar semantic spaces. Can MSRS handle fundamentally different attribute types simultaneously, such as jointly steering safety, factuality, style, and reasoning capabilities? As the number of attributes increases, the total subspace rank R must grow, which may eventually exceed practical limits given the model's hidden dimension. Additionally, attributes like safety and harmlessness present qualitatively different challenges compared to truthfulness or bias, they may require intervention at different layers or positions, potentially conflicting with the current single-layer intervention design. The paper would benefit from experiments explicitly testing the upper bounds of attribute scalability and demonstrating effectiveness on more diverse, potentially conflicting attribute combinations including critical safety-related attributes.

- **Insufficient Evaluation.**
The experimental validation relies primarily on earlier LLMs (Llama2-7B, Llama3-8B-Instruct, Qwen2-7B-Instruct, Mistral-7B-v0.3), while more recent model families with substantially different architectures and capabilities have emerged, including Qwen2.5, the Qwen3 series, and more large reasoning models like Deepseek-r1, QwQ, Qwen3-thinking-mode. These newer models may have fundamentally different internal representations and steering dynamics, particularly reasoning models that employ chain-of-thought or other structured reasoning mechanisms. It remains unclear whether MSRS's subspace decomposition and dynamic token selection strategies, which were developed and optimized on earlier model generations, will transfer effectively to these more advanced architectures. Moreover, the paper does not compare against more latest SOTA methods, like AlphaEdit.

### Questions
The questions are listed in the weaknesses, and if authors could address them, I will raise my rating.

### Soundness
3

### Presentation
2

### Contribution
2
