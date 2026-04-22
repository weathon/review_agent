# Prune Redundancy, Preserve Essence: Vision Token Compression in VLMs via Synergistic Importance-Diversity

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Vision-language models (VLMs) face significant computational inefficiencies caused by excessive generation of visual tokens. While prior work shows that a large fraction of visual tokens are redundant, existing compression methods struggle to balance \textit{importance preservation} and \textit{information diversity}. To address this, we propose $\textbf{PruneSID}$, a training-free Synergistic Importance-Diversity approach featuring a two-stage pipeline: (1) Principle Semantic Components Analysis (PSCA) for clustering tokens into semantically coherent groups, ensuring comprehensive concept coverage, and (2) Intra-group Non-Maximum Suppression (NMS) for pruning redundant tokens while preserving key representative tokens within each group. Additionally, $\textbf{PruneSID}$ incorporates an information-aware dynamic compression ratio mechanism that optimizes token compression rates based on image complexity, enabling more effective average information preservation across diverse scenes. Extensive experiments demonstrate state-of-the-art performance, achieving $\textbf{96.3}$% accuracy on LLaVA-1.5 with only $\textbf{11.1}$% token retention, and $\textbf{92.8}$% accuracy at extreme compression rates ($\textbf{5.6}$%) on LLaVA-NeXT, outperforming prior methods by $\textbf{2.5}$% with $\textbf{7.8}$x faster prefilling speed compared to the original model. Our framework generalizes across diverse VLMs and both image and video modalities, showcasing strong cross-modal versatility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes PRUNESID, a training-free approach for efficient vision–language model inference that balances token importance and diversity. PRUNESID uses a two-stage pipeline: (1) Principal Semantic Components Analysis (PSCA) to cluster tokens into semantically coherent groups, and (2) Intra-group Non-Maximum Suppression (NMS) to prune redundant tokens while preserving key representatives. An information-aware dynamic compression ratio adapts token retention based on image complexity. Experiments show state-of-the-art performance. PRUNESID generalizes across different VLMs and modalities, including images and videos.

### Strengths
1. The paper is clear and practical, proposing PRUNESID to balance token importance in vision–language models. The manuscript is well-structured and presents the method and results in a coherent and accessible manner.

2. I personally find it very interesting to apply NMS from object detection to token pruning.

3. Experimental validation is sufficient. The authors conduct comprehensive experiments on various tasks and show improvements, to validate the effectiveness of the method. Moreover, the ablation study is detailed, particularly in the efficiency analysis section.

### Weaknesses
1. Compared with VisionZip, PRUNESID performs notably better under the 64-token setting. However, its advantage diminishes under the 128- and 192-token settings. I suggest evaluating PRUNESID on more challenging benchmarks, such as MMStar and MathVista, to better highlight its strengths.

2. I would also like the authors to provide results on Qwen2.5-VL, especially on high-resolution benchmarks.

### Questions
1. I am quite curious because NMS is typically very time-consuming, especially with many groups. How does PRUNESID achieve such low latency?

2. In Table 5’s ablation study of groups, did the authors try any alternative clustering methods?

### Soundness
3

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
4

### Summary
This paper introduced PRUNESID, a training-free token pruning framework that balances importance and diversity. It first clusters visual tokens via Principal Semantic Components Analysis (PSCA) to cover key concepts, then applies intra-group NMS to remove redundancy; a data-driven dynamic compression ratio adapts to image complexity. On LLaVA-1.5, PRUNESID retains only 11.1% tokens while preserving 96.3% accuracy.

### Strengths
1. The proposed method is training-free and can be seamlessly integrated into existing vision–language models. This makes it practical and easily deployable in real-time inference scenarios.

2. The two-stage design—PSCA for semantic grouping and intra-group NMS for redundancy suppression—offers a clean and interpretable way to capture both salient and diverse information. The structured distinct from previous one-sided importance- or diversity-only methods.

3. The paper is well-organized, with intuitive figures and clear experimental tables, making the method easy to follow and reproducible.

### Weaknesses
1. Generalization yet to be verified: The paper lacks experiments on different models and numbers of parameters; effectiveness on other architectures (e.g., LLaVA-OV[1], InstructBLIP[2], Qwen-VL[3]) and other numbers of parameters(e.g., LLaVA-13B) remains to be validated.

2. Baseline selection is not enough: The comparison with existing methods is not entirely up-to-date. Therefore, more methods should be compared, for example, the VisPruner[4], CDPruner[5], and so on.


[1] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. ArXiv, 2024a.

[2] Wenliang Dai and Junnan Li and Dongxu Li and Anthony Meng Huat Tiong and Junqi Zhao and Weisheng Wang and Boyang Li and Pascale Fung and Steven Hoi. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. ArXiv, 2023a.

[3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.

[4] Zhang, Q., Cheng, A., Lu, M., Zhuo, Z., Wang, M., Cao, J., ... & Zhang, S. (2024). [CLS] Attention is All You Need for Training-Free Visual Token Pruning: Make VLM Inference Faster. ICCV, 2025.

[5] Zhang, Q., Liu, M., Li, L., Lu, M., Zhang, Y., Pan, J., ... & Zhang, S. (2025). Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs. NeurIPS, 2025.

### Questions
1. Could you please give more visualization of the images in the different benchmarks, to see whether the PSCA and NMS method truly select different semantic tokens?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
PRUNESID is a training-free vision token compression framework for VLMs that balances semantic importance and information diversity via two stages: PSCA for semantic grouping and intra-group NMS for redundancy pruning, plus a dynamic compression ratio. It achieves SOTA efficiency—retaining only ~5–11% tokens while maintaining over 92–96% accuracy across image and video tasks.

### Strengths
1. The writing is fluent and overall clear.
2. The performance is well validated on both **LLaVA-1.5** and **LLaVA-NeXT**.

### Weaknesses
1. The performance on LLaVA-1.5 and LLaVA-NeXT appears relatively weak; please provide comparisons on Qwen2.5-VL instead.

2. Your method performs compression mainly after the vision encoder. I’m curious about how it would behave when combined with compression techniques applied during the LLM stage, such as PyramidDrop, which already adopts a multi-stage framework. Could such a combination achieve a more extreme level of compression by eliminating redundancy more thoroughly at each stage? Furthermore, after applying your compression, do the observations from PDrop, for example, that “almost all visual tokens become redundant after the 24th layer”, still hold true?

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents PRUNESID, a training-free visual token compression framework for VLMs that combines semantic grouping with intra-group NMS pruning. The approach is simple yet effective, achieving strong accuracy retention at extreme compression rates and substantial inference speedups. Overall, it’s a well-motivated and technically solid contribution with clear practical impact.

### Strengths
This paper makes a solid and well-executed contribution to efficient vision-language modeling. Its originality lies in the creative combination of semantic grouping and redundancy pruning within a training-free framework, offering a fresh perspective on token compression. The experimental work is thorough and convincing, with strong empirical support across multiple models and benchmarks. The writing is generally clear and the presentation effective, though some technical sections are dense. Overall, the work is of high quality and significant for improving the scalability and practicality of multimodal systems

### Weaknesses
While the paper is strong overall, several weaknesses limit its impact and generality. First, the conceptual novelty is somewhat incremental. The theoretical grounding is relatively weak — PRUNESID is motivated intuitively, but lacks a formal analysis of why PSCA and NMS together should optimally balance semantic importance and diversity.

### Questions
Analyze qualitative and failure cases: Can the authors provide visual examples showing when PRUNESID succeeds or fails to preserve key visual semantics?
Would incorporating attention-based or saliency-based weighting improve the interpretability of importance?

### Soundness
3

### Presentation
4

### Contribution
3
