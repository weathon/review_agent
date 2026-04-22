# Sparse Shortcuts: Facilitating Efficient Fusion in Multimodal Large Language Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
With the remarkable success of large language models (LLMs) in natural language understanding and generation, multimodal large language models (MLLMs) have rapidly advanced in their ability to process data across multiple modalities.
While most existing efforts focus on scaling up language models or constructing higher-quality training data, limited attention has been paid to effectively integrating cross-modal knowledge into the language space.
In vision-language models, for instance, aligning modalities using only high-level visual features often discards the rich semantic information present in mid- and low-level features, limiting the model’s ability of cross-modality understanding.
To address this issue, we propose SparseCut, a general cross-modal fusion architecture for MLLMs, introducing sparse shortcut connections between the cross-modal encoder and the LLM. These shortcut connections enable the efficient and hierarchical integration of visual features at multiple levels, facilitating richer semantic fusion without increasing computational overhead.
We further introduce an efficient multi-grained feature fusion module, which performs the fusion of visual features before routing them through the shortcuts.
This preserves the original language context and does not increase the overall input length, thereby avoiding an increase in computational complexity for the LLM.
We systematically evaluate the performance of various shortcut patterns and demontrate that SparseCut can enhance the performance of MLLMs across various multimodal benchmarks with high training stability. It is also compatible with different base LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SparseCut, a cross-modal fusion architecture for multimodal large language models (MLLMs). The key idea is to introduce sparse shortcut connections between layers of the vision encoder (ViT) and the language model (LLM). SparseCut proposes an efficient multi-grained fusion module, which merges low- and high-resolution image features via cross-attention within each shortcut before they are fed into the LLM. This allows multi-resolution integration without increasing the token sequence length, thereby avoiding quadratic attention cost. Experiments on several benchmarks (VQAv2, GQA, VizWiz, SciQA-IMG, MMBench, SEEDBench, etc.) show consistent improvements over LLaVA-1.5 and DeepStack, both on Vicuna-7B and Vicuna-13B backbones.

### Strengths
- The paper’s motivation and contributions are explicitly stated and consistently reflected in the methodology and experiments.
- The multi-grained fusion approach seems to address the inefficiency of concatenating high-resolution tokens, offering a simple architectural solution.

### Weaknesses
- The baseline (DeepStack [1]) shown in the tables of the manuscript appears to be relatively outdated, dating back to June 2024. Could the authors provide comparisons with more recent models or benchmarks?
- Some baselines (e.g., Qwen2.5-VL, VITRON, LLaVA-NeXT) are missing, making the claimed SOTA improvement less convincing.
- It is unclear whether this mechanism can be well scaled to video or higher resolution inputs, which are more important for MLLM processing long sequences.

[1] Meng, Lingchen, et al. "Deepstack: Deeply stacking visual tokens is surprisingly simple and effective for lmms." Advances in Neural Information Processing Systems 37 (2024): 23464-23487.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SparseCut, a general multimodal fusion framework for MLLMs, which enhances cross-modal understanding by establishing sparse shortcut connections between multiple layers of the vision encoder and the language model. These shortcuts allow hierarchical and multi-grained visual information to be integrated efficiently without extending the LLM’s input length. By fusing high- and low-resolution visual features through cross-attention, SparseCut preserves rich semantics while maintaining computational efficiency. Experiments on various benchmarks show consistent improvements over LLaVA and DeepStack, achieving better performance with minimal additional cost and strong scalability across different base LLMs.

### Strengths
1. This paper proposes a method that can efficiently integrates multi-level and multi-resolution visual features without increasing computational cost.
2. Through shortcut connections, SparseCut effectively incorporates multi-granularity visual features into the LLM while preserving its original context length and computational efficiency.
3. The experimental results demonstrate strong generalization and scalability across different base LLMs.

### Weaknesses
1. The choice of shortcut pattern (density, distribution) may require manual tuning.
2. The method relies on a frozen vision encoder, potentially limiting deeper cross-modal alignment.

### Questions
1. The choice of shortcut pattern (density, distribution) may require manual tuning.
2. The method relies on a frozen vision encoder, potentially limiting deeper cross-modal alignment.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work presents SparseCut, a cross-modal fusion approach that introduces sparse shortcut pathways to efficiently inject multi-level visual information into LLMs. The proposed design aims to enhance semantic fusion while maintaining low computational overhead for the LLM. Experimental results are reported to show notable performance gains across several benchmarks, suggesting promising generalization.

### Strengths
1. The overall idea is conceptually clear and well motivated.
2. The evaluation covers a reasonably broad set of benchmarks, demonstrating the generalization capabilities of the approach.
3. The manuscript is clearly written and easy to read.

### Weaknesses
1. The experiments are confined to the Vicuna-based LLaVA framework. To support the claim of wide applicability, additional validation on more diverse and up-to-date LLM backbones (e.g., Qwen2.5 series) and multimodal architectures (e.g., Qwen2.5-VL, InternVL-2.5) would be essential.
2. The paper emphasizes efficiency on the language side but neglects the additional cost incurred by processing higher-resolution images through the vision encoder. Reporting overall metrics such as end-to-end FLOPs or inference FPS would provide a more accurate assessment of the actual computational burden.
3. The baseline (LLaVA-1.5) is evaluated at a lower image resolution, whereas the proposed method adopts a much higher resolution. Since increased resolution itself can yield substantial performance improvement, this discrepancy makes it difficult to isolate the contribution of the SparseCut design. A fair comparison under identical resolution settings is needed.

### Questions
See Weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes SparseCut, a general cross-modal fusion architecture for Multimodal Large Language Models (MLLMs) to address limitations in existing MLLMs—specifically the neglect of mid/low-level visual semantics and high computational costs from multi-grained feature integration.
Experiments validate SparseCut’s effectiveness across multiple benchmarks. Ablation studies confirm that sparse/uniform shortcuts and multi-grained fusion contribute to performance gains

### Strengths
1.	Addresses two critical pain points of existing MLLMs—loss of mid/low-level visual semantics (by leveraging multi-level vision encoder layers) and high computation from multi-resolution features (by fusing features before shortcut injection)—filling gaps in current cross-modal fusion designs.
2.	SparseCut is compatible with diverse base LLMs (Vicuna, Phi-3) and scales across model sizes (3.5B–13B). The shortcut pattern (order/distribution/density) is configurable, making it a flexible framework rather than a task-specific solution.
3.	By avoiding input context length expansion (a common issue with multi-grained fusion), SparseCut maintains low computational complexity while enhancing performance—critical for practical deployment of large MLLMs.

### Weaknesses
1.	While the paper tests sparse/uniform, dense/bottom patterns, it lacks a systematic exploration of why the U-shaped order is optimal (e.g., no comparison to linear/ random connection orders) or how to dynamically adjust shortcut density/distribution for different tasks (e.g., fine-grained recognition vs. coarse visual reasoning).

2.	The paper mentions freezing the vision encoder during training but provides no analysis of training stability (e.g., whether sparse shortcuts mitigate overfitting) or convergence speed compared to baselines. It also does not explore the impact of pretraining/fine-tuning data size on SparseCut’s performance.
		
3.	Quantitative benchmarks dominate the evaluation, but there is no qualitative analysis (e.g., case studies of VizWiz unanswerable questions or MMBench reasoning) to illustrate how multi-level/multi-grained fusion specifically improves cross-modal understanding (e.g., reducing hallucinations).

### Questions
Besides the weakness, I have extra questions:

Compared to UNet format connections, what’s the performance of the method that fuses shallow features of ViT with deeper features of LLM?

### Soundness
3

### Presentation
2

### Contribution
3
