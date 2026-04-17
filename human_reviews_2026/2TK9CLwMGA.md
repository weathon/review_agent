# PREP: Pre-inference Guided Token Pruning for Efficient Vision-Language Models

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Recent Visual-Language Models (VLMs) have demonstrated strong fine-grained perception capabilities across a wide range of Visual Question Answering (VQA) tasks. However, this advantage comes at the cost of a rapidly increasing number of visual tokens, leading to substantial computational and memory overhead. Existing training-free methods adopt fixed-layer or layer-by-layer pruning, which disrupts modality fusion before alignment and leads to significant performance degradation under high pruning ratios. In this study, we observe that after the early stage of modal fusion, cross-modal attention not only accurately identifies regions of interest but also demonstrates less sensitive to pruning. Building on this, we propose \textbf{PREP}, a training-free method that identifies optimal pruning layer via patch-level pre-inference, thereby avoiding the loss of fine-grained details under stepwise pruning. Specifically, PREP identifies the the layer with accurate cross-modal alignment using an \textbf{E}ntropy--\textbf{KL} divergence (EKL) score derived from the Information Bottleneck principle, and then retains tokens at this layer that are critical for visual integrity and semantic alignment during full inference. Experiments on LLaVA-1.5-7B show that with only \textbf{9} visual tokens and half of the layers used in pre-inference, PREP preserves \textbf{96.2\%} of the original performance while retaining just \textbf{16} visual tokens (\textbf{3\%}), leading to a \textbf{67\%} reduction in KV-cache usage and a \textbf{1.66$\times$} acceleration in inference speed. We have presented our code in the supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes PREP, a training-free pruning framework for efficient inference in Visual-Language Models (VLMs). The method leverages patch-level pre-inference and an Entropy-KL divergence (EKL) score to identify an optimal pruning layer. At this selected layer, PREP retains tokens based on a multi-modal importance score derived from both visual-visual and visual-prompt attention matrices. PREP effectively reduces visual tokens and KV-cache usage while accelerating inference speed with minimal performance degradation across various VQA benchmarks and VLM backbones.

### Strengths
1. The problem of reducing computational and memory overhead in VLMs is important.
2. The paper presents extensive experimental validation across nine VQA benchmarks and multiple VLM backbones.

### Weaknesses
1. The introduction of $Q^k$ in line 169 lacks a high-level explanation. While the formula is provided, the conceptual motivation and its role in reflecting cross-modal alignment are not clearly articulated beforehand. This makes it difficult for readers to grasp its significance.
2. In line 205, the paper states that the IoR range is calculated for LLaVA-1.5-7B to identify layers with stable alignment. The concern is whether this specific range for LLaVA-1.5-7B is universally applicable or if it needs to be re-computed for every new VLM architecture?
3. Table 6 shows that a group size of 64 tokens outperforms larger group sizes. This result is counter-intuitive, as one might expect larger group sizes to retain more fine-grained information and thus lead to better performance.

### Questions
please refer to weakness

### Soundness
3

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
3

### Summary
As video sequences grow increasingly longer, Vision-Language Models (VLMs) face significant computational and memory overhead when processing these extended inputs. Although prior work has explored token pruning to alleviate unnecessary computation, these methods often require additional training or struggle to balance pruning efficiency against performance degradation. To tackle these challenges, the paper propose PREP, a training-free approach for efficient token pruning in VLMs. Specifically, PREP identifies the optimal pruning layer via patch-level pre-inference using an Entropy-KL divergence (EKL) score derived from the Information Bottleneck principle. Subsequently, PREP leverages intra-visual attention and visual-text attention scores to selectively prune visual tokens, thereby significantly reducing computational overhead during inference. Experimental results demonstrate that PREP not only achieves substantial computational savings but also preserves model performance.

### Strengths
- 1. The authors propose PREP, a novel and training-free token pruning framework. PREP creatively introduces a two-stage strategy: identifying the optimal pruning layer via EKL scores during the pre-inference stage and performing selective token pruning based on attention scores during inference. This design not only significantly enhances pruning efficiency, but also provides valuable inspiration for future research in this area.

- 2. The authors thoroughly analyze the property that VLMs fuse visual and textual information in intermediate layers and observe that initiating token pruning from these layers effectively reduces computational costs without sacrificing performance. This insight offers theoretical and practical guidance for researchers working on VLM token pruning.

- 3. The paper formalizes the theoretical foundation of pruning layer selection with clear mathematical derivations, which strengthens the credibility of the proposed approach.

- 4. Extensive experiments across multiple datasets are conducted to comprehensively validate the effectiveness of PREP in improving computational efficiency and preserving model performance.

### Weaknesses
- 1. Lack of design details. One of the core contributions of the paper is the EKL score, which is used to dynamically select the optimal pruning layer k. However, the selection mechanism is not clearly or rigorously explained. The authors observe stable performance in layers 6-15 of the LLaVA model via the IoR metric, but this is merely an empirical observation and does not constitute an actionable selection criterion. For a new model, readers cannot determine how to select candidate intermediate layers for pruning.
- 2. Lack of theoretical justification. The EKL score, as a core metric, lacks a strict and transparent derivation from the Information Bottleneck (IB) principle to the final formula EKLk = H(Qk) + DKL(Qk || Qk-1). The authors assume that H(Qk|Y) is constant in intermediate layers and thus simplify maximizing I(Qk;Y) to maximizing H(Qk), which is a strong assumption. Although IoR experiments are provided as supporting evidence, the generality and theoretical rigor of this assumption need further substantiation. Additionally, the approximation of minimizing I(Qk; Qk-1) by minimizing DKL(Qk || Qk-1) also lacks necessary derivation steps. The paper should provide more detailed theoretical analysis or ablation studies to justify why this particular EKL formulation is optimal.
- 3. Insufficient evaluation. In the efficiency evaluation, the authors only present the total inference latency of their method under different compression rates, but do not compare the latency changes with other SOTA methods. Although Section 4.2 compares the theoretical FLOPS reduction against SOTA methods, FLOPS only reflects theoretical computational complexity and does not directly demonstrate the advantages in actual inference latency.
- 4. Limited generalizability of key insight. The authors propose a key insight that visual and textual information is fused in intermediate layers, making these layers suitable for token pruning. However, this phenomenon is only verified on Llava-1.5-7B and InternVL3. It remains unclear whether this insight generalizes to all models, especially larger models, or those with different architectures. This limits the universality of the conclusion.

### Questions
- 1. Could authors elaborate on the EKL score’s selection criteria for pruning layers across different models and tasks? Is there a general method for selecting candidate layers in new models?
- 2. Could authors supplement the theoretical derivation of the EKL score, or provide ablation studies to validate the effectiveness of its components, thereby strengthening both theoretical and experimental support for the metric?
- 3. Could authors provide additional experiments comparing the actual inference latency of their method with current SOTA methods?
- 4. Does the property of “visual and textual information fusion in intermediate layers” exist in all VLM models, including larger-scale VLMs?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
### Summary

This paper proposes PREP, a novel token pruning strategy that utilizes a \textbf{Pre-inference} metric derived from a lightweight MLP classifier to selectively drop redundant visual tokens. The method effectively maintains performance under high pruning ratios by preserving tokens deemed crucial for cross-modal fusion in later layers.

### Strengths
### Strengths

1.  **S1: Targeted and Layer-Aware Pruning Strategy.**
    PREP successfully addresses the main flaw of many existing training-free token pruning methods by not relying on fixed layers or uniform pruning. The strategy is layer-aware, adapting the pruning ratio based on the observed redundancy dynamics, which preserves the crucial cross-modal fusion necessary for high performance in later Transformer blocks.

2.  **S2: Demonstrable Efficiency Gains Across Key Architectures.**
    The method demonstrates its effectiveness in achieving significant computational savings (FLOPs reduction) with minimal performance degradation on major VLM tasks. The validation on modern generative architectures like LLaVA and InternVL showcases its practical applicability as a tool for deploying complex large vision-language models more efficiently.

### Weaknesses
## Weaknesses

1.  **W1: Insufficient Novelty in Core Motivation and Finding.**
    The paper's key observation—that visual tokens are highly redundant in early layers but critical for fusion in later layers—is highly overlapping with existing consensus and prior works on layered token pruning.
    * Evidence is lacking to demonstrate that PREP's Pre-inference metric captures any signal of importance that is fundamentally different from simply relying on depth or existing token-level attention scores. This weakens the claim of innovation in the core pruning insight.

2.  **W2: Contradiction Between Efficiency Claims and Pre-inference Overhead.**
    PREP relies on a lightweight MLP classifier for token importance, which necessitates: a) extra training of the MLP on the target dataset; and b) a full pre-inference pass over the entire dataset.
    * This data-dependent pre-computation overhead is computationally heavier than pure attention-based or fixed-layer pruning methods.
    * This creates a significant contradiction with the claim of being a general, efficient, and "training-free" pruning tool, as it introduces substantial setup and data dependence.

3.  **W3: Sufficiency of the Metric in Complex Generative Tasks.**
    PREP assesses a token's importance solely based on its local discriminability towards the final decision via an MLP.
    * For Large Generative VLMs (like LLaVA or InternVL), a token's true importance lies more in its informational complementarity and its precise contribution within the highly complex, multi-layered cross-attention blocks.
    * Relying solely on a token's discriminability risks misclassifying crucial cross-modal fusion tokens, especially at high pruning ratios, which are essential for maintaining generative coherence.

4.  **W4: Insufficient Cross-Task Generalization Validation.**
    Although tested on LLaVA/InternVL, PREP's core mechanism relies on the target dataset and task to train its MLP metric.
    * This implies that whenever the model is deployed to a new downstream task or domain, the MLP must be retrained and validated on new data.
    * This strong task- and data-dependency severely limits PREP's utility as a general, one-time optimization VLM deployment tool, which contradicts the expected plug-and-play nature of efficient inference techniques.

5.  **W5: Lack of Evaluation on Diverse Model Families and Multiturn Dialogue.**
    The paper focuses mainly on the LLaVA and InternVL families, lacking validation on \textbf{other major VLM families} (e.g., BLIP-2, Qwen-VL) to prove PREP's universality.
    * Furthermore, all evaluations focus on single-turn VQA, failing to demonstrate pruning robustness on the more challenging and practical scenarios of multiturn conversation datasets or tasks.
    * The lack of validation in contexts with cumulative redundancy and complex context dependency limits PREP's practical relevance for real-world applications.

### Questions
## Questions

1.  **Q1: The Fundamental Computational Burden of "Pre-inference."**
    The paper claims the MLP metric is "Pre-inference" yet requires a \textbf{full pass} over the target dataset and \textbf{additional training} of the MLP. What is the fundamental difference in \textbf{actual computational burden} between this and traditional \textbf{Training-Aware} token importance scoring (e.g., using gradients or fine-tuned attention weights)? If the total overhead (MLP training + data pass) exceeds the cost of fine-tuning a simpler pruning strategy, does the claim of being \textbf{"training-free"} or \textbf{efficient} hold up?

2.  **Q2: Metric Disconnect from Cross-Modal Fusion and Complementarity.**
    PREP's metric assesses a token's importance based on its local discriminability towards the final decision. However, in generative VLMs, a visual token's core role is often to provide \textbf{complementary information} within the cross-attention blocks.
    * A token might have \textbf{low discriminability} (poor ability to predict the answer alone) but be \textbf{crucial for guiding the LLM's text generation}.
    * How do the authors prove that basing importance solely on \textbf{token discriminability} is sufficient to gauge its functional necessity for \textbf{complex cross-modal information complementarity}?

3.  **Q3: Robustness Challenge in Multiturn Dialogue and Generalization Contradiction.**
    The paper's evaluation is limited to single-turn VQA. In \textbf{Multiturn Conversation}, the importance of visual tokens changes dynamically with the \textbf{accumulation of conversational context}.
    * Can PREP's \textbf{single-pass pre-inference metric} accurately capture and handle this \textbf{dynamic, context-sensitive redundancy}?
    * Furthermore, given the strong \textbf{dependency on the downstream dataset} to train the MLP, how does this \textbf{task-dependency} reconcile with PREP's positioning as a \textbf{general VLM inference optimization tool}?

4.  **Q4: Lack of Ablation to Prove Unique Metric Value.**
    The paper claims PREP's metric is superior to attention scores. However, have the authors performed an \textbf{ablation study} to demonstrate that the importance signal captured by PREP is fundamentally different from the signal produced by combining \textbf{model depth} (prune more in early layers, less in late layers) and \textbf{average token-level attention scores}? If the metric's \textbf{unique, non-redundant value} is not proven, is PREP's complex pre-computation step merely unnecessary overhead?

5.  **Q5: Evaluation Gap in Diverse and State-of-the-Art Architectures.**
    The paper's universality is limited by its focus on LLaVA and InternVL families. To strengthen the claim of being a general pruning guide, can the authors demonstrate the effectiveness and efficiency of PREP on a more diverse and state-of-the-art generative architecture, specifically \textbf{Qwen-VL 2.5}? This validation is crucial for proving the method's applicability beyond the current scope.

### Soundness
2

### Presentation
3

### Contribution
2
