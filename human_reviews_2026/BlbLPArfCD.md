# MoST: Mixing Speech and Text with Modality-Aware Mixture of Experts

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
We present MoST (Mixture of Speech and Text), a novel multimodal large language  model that seamlessly integrates speech and text processing through our proposed Modality-Aware Mixture of Experts (MAMoE) architecture. While current multimodal models typically process diverse modality representations with identical parameters—disregarding their inherent representational differences, we introduce specialized routing pathways that direct tokens to modality-appropriate experts based on input type. MAMoE simultaneously enhances modality-specific learning and cross-modal understanding through two complementary components: modality-specific expert groups that capture domain-specific patterns and shared experts that facilitate information transfer between modalities. Building on this architecture, we develop an efficient transformation pipeline that adapts the pretrained MoE language model through strategic post-training on ASR and TTS datasets, followed by fine-tuning with a carefully curated speech-text instruction dataset. A key feature of this pipeline is that it relies exclusively on fully accessible, open-source datasets to achieve strong performance and data efficiency. Comprehensive evaluations across ASR, TTS, audio language modeling, and spoken question answering benchmarks show that MoST consistently outperforms existing models of comparable parameter counts. Our ablation studies confirm that the modality-specific routing mechanism and shared experts design significantly contribute to performance gains across all tested domains. To our knowledge, MoST represents the first fully open-source speech-text LLM built on a Mixture of Experts architecture.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MoST (Mixture of Speech and Text), a multimodal large language model that unifies speech and text processing via a novel Modality-Aware Mixture of Experts (MAMoE) architecture. MAMoE employs modality-specific routing to assign tokens to appropriate experts. The authors also propose an efficient adaptation pipeline that fine-tunes a pretrained MoE language model on open-source ASR, TTS, and speech-text instruction datasets.

### Strengths
- This paper introduces a novel Modality-Aware Mixture of Experts (MAMoE) architecture and a highly data-efficient Text-Speech Transformation Pipeline, which skillfully adapts a pretrained LLM into a powerful speech-text model through targeted post-training and instruction tuning.
- The paper's most illustrations are commendable for their clarity.
- The commitment to fully open-sourcing the work provides a valuable asset to the research community.

### Weaknesses
- The modality-aware routing relies on deterministic, tag-based assignment to expert groups, which closely resembles a two-tower architecture and may underutilize MoE's core strength of dynamic, content-based routing.
- Based on the results presented in Table 1 and Table 2, while the proposed MoST model consistently outperforms the baselines across several benchmarks, the margin of improvement appears to be somewhat limited on certain metrics.

### Questions
- Given that the modality-aware routing mechanism relies on tag-based assignment rather than dynamic, token-level routing, could the authors elaborate on how the model leverages the full flexibility and adaptive potential of the Mixture-of-Experts framework?
- The paper currently lacks comparisons with other models built on the Mixture-of-Experts (MoE) architecture. Including such baselines would help better contextualize the proposed method’s contributions and highlight its relative advantages. Adding such comparisons or clearly justifying their absence would enhance the paper’s completeness.

### Soundness
2

### Presentation
3

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
The authors attempt to use MoE to address interference in speech-text modality representations. They build the MoST layer, which employs a modality-aware router to select modality-specific experts, along with a shared expert to facilitate information exchange. They also develop a data generation pipeline to provide training data for large models. The model is evaluated on multiple tasks, including ASR, TTS, and QA, and achieves superior results on certain metrics compared to some open-source models.

### Strengths
Using MoE to construct a large speech-text model is an interesting approach.

### Weaknesses
1. The motivation for using a modality-aware router is unclear, as modality representations are generally easy to distinguish. The necessity of MoE in this context is not well justified.

2. The comparisons of data and models in the paper are unclear. The description of the initialized large model is insufficient, and evaluations on Llama Question S2T and Web Question are missing. Evaluations of text-based foundational models are also lacking.

3. Additionally, the model weights are not open-sourced, and the specifics of data usage are unclear. Comparisons with recent works such as MinMo[1] and LLama-Omni2[2] are missing, which undermines the validity of performance claims, such as "MoST achieves state-of-the-art or competitive performance compared to existing models with similar parameter counts."

  [1] Minmo: A multimodal large language model for seamless voice interaction. Chen et al., 2025.
  [2] LLaMA-Omni2: LLM-based Real-time Spoken Chatbot with Autoregressive Streaming Speech Synthesis. Fang el al., 2025.

### Questions
How does MoE impact the original text foundation model?

### Soundness
2

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
The paper introduces MoST, a speech-text large language model built on a Modality-Aware Mixture of Experts (MAMoE) architecture. The main idea is to extend a pretrained MoE language model with modality-aware routing, using modality-specific experts for speech and text tokens, along with shared experts for cross-modal interactions. The model is post-trained through a two-stage procedure of Cross-Modal ASR/TTS Post-Training and Mixed Instruction Fine-Tuning. The training data are based on open-source or derived from open-source datasets. Experiments cover ASR, TTS, audio language modeling, and spoken QA. Across these tasks, MoST achieves competitive or superior results compared with strong baselines while maintaining computational efficiency. The paper includes detailed ablation studies and commits to releasing code, model weights, and data for reproducibility.

### Strengths
1. The modality-aware mixture of experts (MAMoE) provides a clean and intuitive way with modality-specific expert groups and a parallel shared expert for training a speech language model.



2. The experimental validation is solid and rigorous, including ablations on initialization with non-MoE LLM (Llama3.2 3B), and ablations without modality-specific experts or shared experts.

### Weaknesses
1. The partition of 50% of the initial text expert capacity to $\mathcal{E}_{audio}$ is a major structural change, but the division is simply based on index without any reliable partition mechanism. The hard 50% partition of experts may introduce a risk of losing valuable text knowledge. 

2. The paper lacks text-only evaluations.



3. The paper frequently claims "efficiency" and "data efficiency" without direct, quantifiable metrics and experimentation. It feels like the efficiency claim is only an architectural inheritance from MoE.

### Questions
1. Could you explicitly name the specific pretrained MoE LLM that served as the starting point for MoST? The paper mentions the “stronger initialization (DeepSeek-V2 Lite)” in the ablation section 6. Is this the base model that you used in your main experiments?

2. Could we see some text-only evaluations to understand the quantitative impact on core LLM tasks?

3. Could you provide the Llama Q ($S \to T$) result in addition to Llama Q (S→S) as well?

4. Minor issues: In Section 6.1, the inline citation should be citep instead. Adding a clearer setup description for MoST-Style Upcycling would be helpful for clarity. In Figure 5, shouldn’t that be labeled as ‘MoST-NoShared’ instead of ‘MAMoE w/ Shared Expert’?

### Soundness
3

### Presentation
3

### Contribution
3
