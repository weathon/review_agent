# GPTailor: Large Language Model Pruning Through Layer Cutting and Stitching

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Large language models (LLMs) have shown remarkable capabilities in language understanding and generation. However, such impressive capability typically comes with a substantial model size, which presents significant challenges in deployment and inference. While structured pruning of model parameters offers a promising way to reduce computational costs at deployment time, current methods primarily focus on single model pruning. In this work, we develop a novel strategy to compress models by strategically combining or merging layers from finetuned model variants, which preserves the original model's abilities by aggregating capabilities accentuated in different finetunes. We pose the optimal tailoring of these LLMs as a zero-order optimization problem, adopting a search space that supports three different operations: (1) Layer removal, (2) Layer selection from different candidate models, and (3) Layer merging. Our experiments demonstrate that this approach leads to competitive model pruning, for example, for the Llama2-13B model families, our compressed models maintain approximately 97.3\% of the original performance while removing ～25\% of parameters, significantly outperforming previous state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel and interesting approach to structured pruning by leveraging multiple fine-tuned models for layer selection, removal, and merging. It breaks away from the traditional single-model pruning framework and attempts to construct a smaller but more comprehensive model by stitching the expertise of different models. The paper conduct extensive experiments on the Llama2-7B and 13B models and demonstrate its effectiveness.

### Strengths
- The idea of ​​compressing a model by strategically combining or merging layers from fine-tuned model variants is novel.
- The ablation experiment is very thorough

### Weaknesses
- The proposed method is not tested on a larger 70B model, so its generalization cannot be demonstrated.
- The algorithm part is very vague.
- Lack of theoretical support, why can better results be achieved by stitching multiple models?
- There are too few baselines for comparison. Please compare more baselines, such as [1-3].
- Lack of ablation experiments on Calibration Data.
- As your mention in sec3.2, If $\sum^K_{j=1} c_{i,j} = 0$, we retain the layer from the base model instead. If $\sum^K_{j=1} c_{i,j} = 0$, why not remove this layer?
- The author only gives a general description of how to optimize the search space to obtain the final pruned model. This part seems more like the result of manual attempts, because the search space is large when using a search algorithm, and how to ensure that the model you searched is the global optimal one. I hope the author can give a detailed and clear answer.

[1] Reassessing layer pruning in llms: New insights and methods. arXiv preprint arXiv:2411.15558

[2] Streamlining redundant layers to compress large language models. arXiv preprint arXiv:2403.19135.

[3] Finercut: Finer-grained interpretable layer pruning for large language models. arXiv preprint arXiv:2405.18218.

### Questions
- If the performance of the original model you choose is too poor, will it affect the merged or spliced ​​pruned model?
- How much impact will switching to other merge methods have on the performance of the pruned model?

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
GPTailor is a structured pruning framework that compresses LLMs by cutting, selecting, and stitching layers from multiple fine-tuned variants. Using zero-order (Bayesian/SMAC) search, it finds layer combinations that balance accuracy and size. On LLaMA-2-13B models, GPTailor retains ~97% accuracy with ~25% fewer parameters, outperforming prior structured pruning baselines.

### Strengths
1. Introduces a creative, structured pruning paradigm that assembles a smaller model by reusing layers from multiple fine-tuned variants.

2. Frames pruning as a meta-optimization problem, offering a search-based alternative to rule-based or heuristic layer removal.

3. The method is conceptually simple and potentially compatible with existing LLM fine-tuning workflows.

### Weaknesses
1. Since GPTailor builds its pruned model by mixing layers from several fine-tuned versions, it’s hard to tell whether the reported improvements actually come from the proposed search and stitching method, or simply from combining strong models together. The paper would be more convincing if it included comparisons with random or heuristic layer combinations to separate the effect of its algorithm from that of model diversity.

2. The approach assumes that we already have multiple fine-tuned versions of the same base model lying around, which isn’t realistic in many deployment settings. Keeping and using several large checkpoints also adds heavy storage and computation overhead, making the method less appealing for practical use.

3. While GPTailor reduces parameter count, the paper doesn’t show whether this actually leads to faster inference, lower FLOPs, or any hardware-level acceleration. Without latency or wall-clock measurements, it’s difficult to judge whether the pruning translates into real performance benefits in practice.

### Questions
1. How do you ensure that the reported performance improvements are due to the search/stitching algorithm and not simply from  combining already-strong fine-tuned variants?

2. What is the total computational and memory overhead of the search procedure (number of model evaluations, GPU hours, etc.)?

3. Do you observe actual inference acceleration (throughput or latency) on GPU hardware after pruning?

4. How does GPTailor behave if only a single fine-tuned model is available?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
GPTailor attempts to compress LLMs not by pruning a single network, but rather by cutting and stitching layers taken from different fine-tuned model variants. In their technique, for every layer it can either drop it, borrow it from another candidate, or merge layers from multiple candidate models. GPTailor uses a zero-order, multi-objective search over the space to keep task performance high. On Llama-2-7B and 13B it removes about a quarter of the parameters while still keeping most of the original performance, clearly ahead of structured-pruning baselines like LLM-Pruner, SliceGPT, LaCo, and ShortGPT under similar ratios. The ablations indicate that cross-model layer merging is what really recovers performance, and that increasing the size of the candidate pool usually helps while staying stable even with weaker models. Overall, it’s a solid argument that multi-model, search-based layer tailoring is significantly more effective than single-model, metric-based pruning.

### Strengths
- Reframes pruning as cross-model layer selection/merging as opposed to single-model trimming, which is solid approach. 
- Flexible search space (drop / pick / merge per layer).
- Strong empirical results on Llama-2-7B/13B: ~25% layers removed while retaining most of the performance performance, outperforming structured-pruning baselines by a significant margin given the same candidates. 
- Useful ablations showing that merging across variants is one of the main source of quality recovery, not just cross-model picking. 
- Zero-order search makes it automatable and task-agnostic if a calibration set is defined.

### Weaknesses
- No latency/throughput inference comparison with LLM-Pruner, SliceGPT, LaCo, or ShortGPT, so the practical speedup of the 25% layer drop is unclear.
- The search itself is heavy (500 SMAC trials, multi-fidelity), which somewhat offsets the advantages and could be hard to reproduce for bigger pools of models. 
- Assumes access to task-specialized variants of the same base model, which doesn't always match deployment settings.

### Questions
- It would be useful to report latency/throughput to show that the ~25% layer removal/merging actually yields practical speedups over baselines like LLM-Pruner, SliceGPT, LaCo, and ShortGPT.
- How does the search cost scale in practice for bigger model pools (more than the three presented in Table 16), and is there a lighter configuration that could be used users with limited compute?

I'd be happy to reconsider my score if these questions are addressed!

### Soundness
3

### Presentation
4

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
The paper introduces a new structured pruning approach that constructs smaller LLMs by selectively removing, merging, and reusing layers from multiple fine-tuned variants of a base model. Instead of pruning a single network, it formulates pruning as a zero-order optimization problem that searches across combinations of layers from task-specialized models to maximize performance under a sparsity constraint. Experiments on the LLaMA-2 and LLaMA-3 families show that GPTailor maintains up to 97% of the original performance while removing about 25% of parameters, outperforming existing pruning methods and demonstrating strong generalization to newer model architectures

### Strengths
1. The idea of this paper is both interesting and innovative. By compressing and merging task-specific fine-tuned models, it achieves impressive compression performance without requiring additional fine-tuning.
2. The experiments are solid, demonstrating the method’s effectiveness across four categories and fourteen datasets.
3. The paper is clearly written and easy to follow.

### Weaknesses
1. Theoretical analysis is weak, and although the experimental results of the method are significant, there is a lack of theoretical explanation or visual analysis (such as layer representation similarity) on why cross-model stitching is better than single model pruning.
2. Insufficient interpretability, although a structural diagram of the pruned model is provided, there is a lack of in-depth analysis on why certain layers are retained or merged.

### Questions
There is a lack of in-depth analysis on why certain layers are retained or merged.

### Soundness
3

### Presentation
3

### Contribution
3
