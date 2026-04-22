# Plug-and-Fold: Weight-Preserving Structured Compression for Large Language Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Large Language Models (LLMs) have achieved remarkable performance across a wide range of tasks, but their growing size poses significant challenges for deployment and efficiency. Among existing model compression methods, structured pruning has emerged as a popular approach for reducing model size. However, pruning removes structural components such as layers, heads, or channels, which can disrupt pre-trained weights and lead to fragile recovery fine-tuning process. In this work, we propose Plug-and-Fold (PnF), a weight-preserving yet structurally effective compression method. Rather than removing weights or modifying the model architecture, PnF introduces lightweight, learnable adapter modules into the projection layers of attention and feed-forward networks. These adapters are trained while keeping the original weights frozen, and are later folded into the base weights via simple matrix multiplications. This process yields a compressed model that is structurally identical to the original and incurs no additional runtime overhead. We evaluate PnF across a variety of benchmarks and model scales, demonstrating consistent improvements over recent state-of-the-art structured compression baselines. Our results highlight that preserving the integrity of pretrained weights not only simplifies the compression pipeline, but also improves generalization and performance recovery in compressed LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents  a structured compression method for LLMs that mitigates the performance loss of pruning. PnF inserts lightweight, low-rank adapters into frozen linear layers of a pre-trained model. These adapters are trained using a KL-divergence distillation loss and a Group-wise Sequential Training (GWST) scheme, then folded into the base weights by simple matrix multiplication. The result is a dimension-reduced model with the same architecture and zero runtime overhead. Experiments show consistent gains over strong baselines, though the approach relies on manual compression schedules and lacks full efficiency analysis.

### Strengths
+ The method provides a clean separation between the knowledge locked in the pre-trained weights (frozen $W$) and the structural compression process (learning $P$). This is an elegant theoretical step forward from disruptive structured pruning techniques.

+ The post-training folding step is a crucial practical advantage. By fully merging the adapters into the base weights, the method avoids introducing any new operations or latency during inference, ensuring the compressed model is truly efficient and deployment-friendly.

+ The performance gains over strong compression baselines (SliceGPT, LaCo, etc.) are notable and consistent across different Qwen and OPT model scales. This suggests the core principle of PnF is highly effective at preserving model fidelity during dimensionality reduction.

### Weaknesses
+ PnF’s uses a hand-crafted per-layer reduction schedule, with no automatic or data-driven metric, thereby limiting reproducibility and generalization.

+ Although framed as an efficiency approach, the paper reports only parameter counts. No FLOPs or latency data are provided.

+ Recovering near-baseline performance requires large amount of distillation samples

+ No Comparison to Analytical Low-Rank Methods:

### Questions
+ Could you provide quantitative evidence (e.g., activation similarity or representation distance metrics) showing how freezing W
contributes to knowledge retention compared to jointly training W and P? 

+ Have you examined whether the folded matrices W^Comp =W⋅P introduce non-Gaussian outliers or distributional shifts that could complicate low-bit quantization (e.g., 4-bit) compared to quantizing the original 
W?

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
This paper introduces Plug-and-Fold (PnF), a structured compression method for Large Language Models (LLMs) that preserves the original pre-trained weights and model architecture. Instead of removing components like neurons or layers, PnF inserts lightweight, learnable adapter modules into the projection layers of the attention and feed-forward networks. During a multi-stage training process, only these adapters are trained while the original weights remain frozen. After training, the adapters are folded back into the base weights via simple matrix multiplication, resulting in a compressed model that is structurally identical to the original and incurs no inference overhead. Extensive experiments across various model scales and compression rates demonstrate that PnF consistently outperforms state-of-the-art structured pruning baselines, particularly on knowledge-intensive tasks, highlighting the benefit of retaining the integrity of pre-trained weights.

### Strengths
1.The core innovation is a weight-preserving approach. By keeping pre-trained weights frozen and only training small adapters, the method retains the knowledge and expressive power of the original model, leading to better performance recovery after compression.

2.The method is rigorously evaluated against strong baselines on multiple models and benchmarks. It demonstrates consistent and often significant performance improvements, especially at higher compression rates, establishing a new state-of-the-art for structured compression.

3.The proposed training pipeline—comprising compression planning, group-wise sequential training, and KL-divergence distillation—is well-designed to stabilize training, prevent covariate shift, and effectively align the compressed model's output with the original model's.

### Weaknesses
1.While group-wise sequential training enhances stability, it also introduces a more complex and potentially longer training procedure compared to normal fine-tuning methods.

2.The use of varying compression rates across different layers results in an irregular model structure, which can introduce challenges and inconvenience for practical deployment.

3.It should be noted that the approach of compressing original weight matrices through adapter modules is not novel in the literature. Previous works such as WID[1] and SP3[2] have explored similar methodologies, where trainable compression matrices are inserted before and/or after the original weight matrices. These compression matrices are subsequently merged back into the original weights through matrix multiplication after training to achieve the final compressed form. However, the authors did not cite these two relevant papers in their work.

[1]Wu, Taiqiang, et al. "Weight-inherited distillation for task-agnostic bert compression." arXiv preprint arXiv:2305.09098 (2023).

[2]Hu, Yuxuan, et al. "$\rm SP^ 3$: Enhancing Structured Pruning via PCA Projection." arXiv preprint arXiv:2308.16475 (2023).

### Questions
Did the author know about the two highly similar papers, WID and SP3? The method proposed in this paper compresses by directly introducing a low-dimensional compression matrix, whereas WID and SP3 introduce higher-dimensional compression matrices and compress them during training. In my opinion, there is not much fundamental difference between these two approaches.

### Soundness
3

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
Plug-and-Fold (PnF) is a new structured compression framework for large language models (LLMs). It keeps the pretrained backbone intact while inserting small, trainable adapters into attention and feed-forward layers. Once trained with frozen base weights, the adapters are folded into the model through simple matrix multiplications, producing a compressed variant structurally identical to the original and free of runtime overhead. We validate PnF on multiple LLMs, including Qwen and OPT families.

### Strengths
1. The paper is clearly organized, and the figures effectively illustrate the folding pipeline.

2. The paper includes group-wise and data-size ablations that show trade-offs among compression rate, accuracy, and recovery set size, providing useful insights into design choices.

### Weaknesses
1. PnF is conceptually close to AdaLoRA [4], which also trains low-rank adapters and merges them into base weights, but differs mainly in emphasizing structural identity after folding. The analogy to LLM-Pruner [5] is less precise—PnF does not perform pruning but instead adds and folds adapters. Overall, the method introduces limited theoretical or algorithmic innovation beyond existing adapter-based compression frameworks.

2. Evaluations partially miss recent baselines. While SliceGPT [1], LaCo [2], and ShortGPT [3] are included, stronger or newer methods such as LLM-Pruner [5] and DISP-LLM [6] are absent under matched compression ratios and datasets. Their inclusion would strengthen the empirical comparison.

3. The paper claims “deployment efficiency” but provides no FLOP, latency, or energy analyses. Such quantitative metrics are essential to substantiate statements like “no runtime overhead” and to contextualize benefits relative to SliceGPT [1] or LLM-Pruner [5].

4. Experiments cover only the OPT and Qwen series, which limits generality. Additional results on diverse architectures, such as LLaMA-3, T5, or Mixtral, would better demonstrate robustness and broader applicability.

4. The manual grouping schedule is empirical and could be automated. The paper itself acknowledges that this stage depends on practitioner heuristics. Replacing it with learned or adaptive strategies, such as sensitivity-based or reinforcement-learning layer selection as in DISP-LLM [6], would enhance reproducibility.

5. The “weight-preserving” claim is supported only by performance metrics. Analyses of representational similarity (e.g., centered kernel alignment (CKA) [8]) or Jacobian similarity would substantiate that the folded model genuinely retains the representational structure of the original weights.

6. Section 3.2 would benefit from concise pseudocode summarizing the plug-and-fold pipeline. Presenting the compression planning, group-wise training, and folding steps in algorithmic form would improve reproducibility and clarify the workflow.

[1] S. Ashkboos et al. SliceGPT: Compress Large Language Models by Deleting Rows and Columns. 2024.

[2] Y. Yang et al. LaCo: Large Language Model Pruning via Layer Collapse. 2024.

[3] X. Men et al. ShortGPT: Layers in Large Language Models Are More Redundant Than You Expect. 2024.

[4] Q. Zhang et al. AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning. 2023.

[5] X. Ma et al. LLM-Pruner: On the Structural Pruning of Large Language Models. 2023.

[6] S. Gao et al. DISP-LLM: Dimension-Independent Structural Pruning for Large Language Models. 2024.

[7] A. Gromov et al. The Unreasonable Ineffectiveness of the Deeper Layers. 2024.

[8] S. Kornblith et al. Similarity of Neural Network Representations Revisited. 2019.

### Questions
1. How does PnF differ algorithmically from existing merging or folding approaches such as AdaLoRA [4] and LLM-Pruner [5], which also integrate low-rank or pruned components into base weights?

2. Could the authors include evaluations against recent structured compression baselines, such as SliceGPT [1], LaCo [2], ShortGPT [3], LLM-Pruner [5], and DISP-LLM [6], under identical compression ratios and datasets for fair comparison?

3. What are the quantitative FLOP, latency, and energy reductions of PnF compared with SliceGPT [1] and LLM-Pruner [5], and how do these relate to the claimed deployment efficiency?

4. Have the authors tested PnF on architectures beyond OPT and Qwen (e.g., LLaMA-3, T5, Mixtral) to demonstrate cross-architecture robustness and folding consistency?

5. Could the authors consider replacing the manual grouping plan with an adaptive or automated selection strategy?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces Plug-and-Fold (PnF), a structured compression method for LLMs that achieves parameter efficiency without altering pretrained weights. Rather than pruning neurons, heads, or layers, PnF augments attention and feed-forward projections with lightweight, trainable low-rank adapters. These adapters are optimized with the backbone frozen and later folded into the original weights via simple matrix operations, resulting in a model identical in structure and runtime cost to the original. Evaluations on several transformer backbones show that PnF delivers competitive performance compared to state-of-the-art structured pruning and compression baselines.

### Strengths
1. The paper is clearly written and well-organized, with intuitive visualizations (e.g., Fig. 1) illustrating the plug-and-fold pipeline and training procedure.

2. The proposed method provides a practically deployable compression mechanism, maintaining the original model architecture and avoiding inference-time modifications.

3. The weight-preserving philosophy aligns with recent trends emphasizing knowledge retention during LLM compression, and the authors demonstrate partial empirical validation of this premise.

### Weaknesses
1. The contribution of PnF appears incremental relative to prior work leveraging activations or low-rank decompositions for pruning and compression. Methods such as SparseGPT [1], Wanda [2], and SlimGPT [3] compute parameter importance via activation or sensitivity metrics, while SoBP [4] and CALDERA [5] integrate structured or low-rank regularization. The manuscript does not clearly articulate a conceptual or algorithmic distinction beyond the “folding” of adapters into pretrained weights.

2. The claim that adapter folding better preserves pretrained knowledge than conventional low-rank compression lacks theoretical grounding. In contrast, SVD-based approaches such as SVD-LLM [6] provide explicit reconstruction-error bounds. The paper would benefit from either a formal argument or a detailed empirical comparison of reconstruction fidelity.

3. The multi-stage training of adapters and the group-wise sequential procedure imply substantial recovery cost. However, the paper omits quantitative comparisons of wall-clock time, GPU hours, or FLOPs against one-shot pruning methods (e.g., SparseGPT [1]) and lightweight low-rank alternatives (e.g., CALDERA [5]). Without such data, the practical efficiency of PnF remains unclear.

4. Experiments primarily target knowledge and commonsense benchmarks (e.g., PIQA, HellaSwag, MMLU) but omit instruction-following, reasoning, or multilingual tasks that test deeper semantic retention. Moreover, comparisons exclude competitive structured compression baselines such as SVD-LLM [6] and quantization/distillation hybrids like BitDistiller [7], weakening the generality of the claims.

5. The compression plan design remains heuristic, requiring manual per-layer rate selection. This approach limits reproducibility and scalability. Automated sensitivity or reinforcement-based planning, as explored in recent structured pruning works [4], could improve generality and reduce manual tuning.

References:

[1] Frantar & Alistarh, SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot, ICML 2023.

[2] Sun et al., Wanda: A Simple and Effective Pruning Approach for Large Language Models, ICLR 2023.

[3] Liu et al., SlimGPT: Layer-wise Structured Pruning for Large Language Models, 2024.

[4] Zhang et al., SoBP: Structured Optimal Brain Pruning for Large Language Models, 2024.

[5] Li et al., CALDERA: Compressing LLMs Using Low-Rank and Low-Precision Decomposition, 2024.

[6] Wang et al., SVD-LLM: Structured Low-Rank Compression for Large Language Models, 2025.

[7] Zhao et al., BitDistiller: Distilling LLMs into Binary and Low-Bit Networks, 2024.

### Questions
1. How does PnF differ algorithmically from prior activation- or sensitivity-based pruning methods (e.g., SparseGPT [1], Wanda [2], SlimGPT [3]) beyond adapter insertion and folding?

2. Can the authors provide theoretical or empirical evidence, such as layerwise reconstruction error or spectral analysis, showing that folded adapters preserve semantic function more effectively than SVD-based compression [6]?

3. What is the additional fine-tuning cost introduced by group-wise adapter training in GPU hours or wall-clock time, compared with one-shot baselines [1]?

4. Will PnF be evaluated on instruction-following and reasoning benchmarks (e.g., GSM8K, MMLU-Pro) and compared to structured low-rank baselines [6] [7]?

5. Could the empirical compression plan be automated via sensitivity analysis or other heuristic search methods, and if so, how consistent are the resulting compression ratios across backbones?

### Soundness
2

### Presentation
2

### Contribution
2
