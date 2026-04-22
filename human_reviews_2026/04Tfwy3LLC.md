# Reassessing Layer Pruning in LLMs: New Insights and Methods

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 8

## Abstract
Although large language models (LLMs) have achieved remarkable success across various domains, their considerable scale necessitates substantial computational resources, posing significant challenges for deployment in resource-constrained environments. Layer pruning, as a simple yet effective compression method, removes layers of a model directly, reducing computational overhead. However, what are the best practices for layer pruning in LLMs? Are sophisticated layer selection metrics truly effective? Does the LoRA (Low-Rank Approximation) family, widely regarded as a leading method for pruned model fine-tuning, truly meet expectations when applied to post-pruning fine-tuning? To answer these questions, we dedicate thousands of GPU hours to benchmarking layer pruning in LLMs and gaining insights across multiple dimensions. Our results demonstrate that a simple approach, i.e., pruning the final layers followed by fine-tuning the lm\_head and the remaining last three layers, yields remarkably strong performance. These pruning strategies are further supported by theoretical analyses based on the gradient flow. Following this guide, our method surpasses existing state-of-the-art pruning methods by $5.62\%$–$17.27\%$ on Llama-3.1-8B-It, by $2.36\%$–$19.45\%$ on Llama-3-8B and by $4.34\%$–$9.59\%$ on Llama-3-70B. The code is available at at https://github.com/yaolu-zjut/Navigation_LLM_layer_pruning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper relates to the pruning of LLM layers. The paper consists of three main parts:
1. Discussion of criteria for identifying prunable layers
2. Comparison between LoRA and partial fine-tuning methods for recovering accuracy after pruning
3. Theoretical analysis of gradient flow in the presence Pre-Layer Normalization, and how this affects layers by depth

The main observation in the paper is the relative unimportance of deep layers, and the fact that pruning the last layers is a more useful heuristic than other more elaborate importance estimators (c.f. Magnitude, Taylor, PPL, BI).
This claim is supported by Table 1, which shows superior results for the "reverse order" method, at a 20% pruning ratio, for Qwen1.5-7B, Llama-3.1-8B-It and Vicuna-7B-v1.5

A parallel finding is the fact that partial fine-tuning of the last one or two layers yields a greater accuracy recovery than full LoRA fine-tuning.
This claim is supported by Table 2.

In the last paragraph of the main body of the paper, the theoretical analysis of gradient flow and show that Pre-LN architectures inherently weaken the gradients and contributions of deeper layers due to the normalization step scaling them down.

### Strengths
The paper is nicely written, and very well laid out, making it an enjoyable read.

### Weaknesses
The paper focuses on depth pruning, however there is abundant evidence in the literature that layer-wise pruning is not as efficient as width pruning. For example, this claim is made in Muralidharan et al. (2024), which is cited in this paper.

In the main body of the paper, results are shown for a pruning ratio of 25%. We need to read the appendix to see results for 50% pruning ratio in Table G, and these results seem to contradict the main finding of the paper, since the "reverse order" method yields inferior results there. The PPL method appears to dominate at 50% pruning ratio, and the "random" method even wins the benchmark for Qwen1.5-7B, which raises questions about the relevance of the results.

The the LoRA vs. partial fine-tuning experiments, the study is limited to partial fine-tuning of the last few layers. Table 2 shows that fine-tuning the last three layers is better than fine-tuning the last two layers, which is better than fine-tuning the last layer. Thus, why stop at three layers? It would seem like if the trend follows, fine-tuning all layers would be optimal?

The theoretical analysis builds upon prior analyses of Pre-LN vs. Post-LN Transformers (e.g., Xiong et al., 2020; Liu et al., 2020). It's known that Pre-LN helps with training stability by damping gradients as depth increases, avoiding explosions near the output (which Post-LN can cause without warmups). However the theoretical analysis falls short of proving the optimality of fine-tuning just the last three layers.

I could not access any of the files behind the URL (https://anonymous.4open.science/r/Navigation-LLM-layer-pruning-DEB7/README.md) due to "The requested file is not found".

### Questions
"reverse order" wins the benchmark at 25% pruning ratio, but does not perform well at 50% pruning ratio, would you be able to do a comprehensive sweep of pruning ratios in order to collect more data points? For example, from 5% to 75%, by increments of 5%.

Can you repeat the experiments in Table 2 with partial-to-full fine-tuning, so we can see which setting is optimal in experimental results?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper re-evaluates layer pruning methods for Large Language Models (LLMs), addressing whether complex metrics are needed to identify redundant layers and if LoRA is the optimal fine-tuning choice after pruning. Through extensive experiments across various metrics, LLMs, and fine-tuning methods, the paper reveals that a simple "backward pruning" (removing the last few layers directly) often outperforms more complex indicators. Furthermore, "partial layer fine-tuning" (tuning only the last few layers and the output layer) is found to be more effective and faster than LoRA for performance recovery. This paper provide a theoretical framework based on gradient flow to explain why deeper layers in Pre-LN Transformers contribute less, validating their approach. Pruned models based on these findings significantly surpass existing methods across benchmarks.

### Strengths
1.Comprehensive experimental design covering diverse pruning metrics, fine-tuning methods, and models.

2.The proposed "backward pruning + partial layer fine-tuning" strategy is simple yet effective.

3.Theoretical analysis using gradient flow provides a rationale for the method's efficacy.

4.Achieves significant performance gains across multiple models, outperforming other methods.

### Weaknesses
1.Inconsistent calibration datasets and data volumes were used for different pruning metrics, which could affect experimental fairness.

2.The performance of the pre-pruned models should be included in the results tables.

### Questions
1.Could you show the results of different pruning metrics without any subsequent training?

2.Have you compared pruning using other metrics (e.g., cosine similarity, perplexity) followed by fine-tuning only the layers immediately surrounding the pruned sections?

3.Recent work suggests deeper LLM layers are crucial for reasoning[1]. Does direct pruning of the final layers impact reasoning capabilities? It would be beneficial to evaluate this method on mathematical and code-related tasks to assess its performance in reasoning.

[1] Song, Xinyuan, et al. "Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning." arXiv preprint arXiv:2510.02091 (2025).

If the author's response addresses my questions, I will consider increasing my rating.

### Soundness
3

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
4

### Summary
This paper re-evaluates layer pruning for Pre-LN LLMs and shows that a simple strategy that prunes layers in reverse order and then fine-tune only the LM head plus the last 1-3 layers consistently matches or even outperforms more complicated pruning methods on a few standard benchmarks (PIQA, HellaSwag, WinoGrande, ARC-e/c, OBQA, MMLU, CMMLU). The empirical study is broad (several LLaMA and Qwen-style models) and scales up to LLaMA-3-70B. The authors give gradient-flow explanation for why deeper layers in Pre-LN are matter less, and they also find that this approach can beat the usual "prune + LoRA" recovery. This makes the paper especially useful for users who just want a reliable pruning recipe without complex per-layer scoring.

### Strengths
- Clear recipe to prune layers in reverse order and fine-tune only the LM head alongside the last 1–3 layers.
- Reasonable empirical baking, tested on several LLaMA-3 and Qwen-style models at several pruning ratios, and several standard benchmarks, and it still works at 70B scale.
- Practical impact, simple post-pruning FT outperforms the common "prune + LoRA" setup.
- Plausible architectural explanation, the Pre-LN gradient-flow analysis motivates why late layers are safer to drop.

### Weaknesses
- They don’t evaluate on generation or reasoning datasets (e.g. GSM8K), so the conclusions are validated only on specific LM-harness-style multiple-choice tasks.
- Prior work shows that layer importance depends on the nature of the task. Without generation tasks, the paper assumes task-invariance of the "prune-from-the-top" rule. Later layers tend to be more critical for perplexity, so pruning them first might hurt exactly the tasks they didn’t test.
- As a result, the current recipe is a strong default for classification-style LLM evals, but its generality to generation remains unproven.

### Questions
- Can you add some generation/reasoning benchmarks (e.g. GSM8K) to verify that reverse-order pruning still holds outside multiple choice tasks?
- Please also report the perplexity on some perplexity based data sets e.g. wikitext to see how that varies across varies different techniques.
- Do the results apply for different model architectures as well? (it would be interesting to see the results on some models mixture of experts models e.g. Mixtral 8×7B)

I'd be happy to increase my score if these experiments are included!

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper is about empirical benchmarking and methodological clarification for layer pruning.

Benchmarks 7 layer-selection metrics and 6 fine-tuning methods across Vicuna-7B, Qwen-7B, and Llama-3.x models.

Finds that reverse-order pruning (dropping last layers) consistently outperforms complex importance metrics.

Shows partial-layer fine-tuning (LM head + last 1–3 layers) surpasses LoRA/QLoRA for accuracy and training cost.

Extends tests to Llama-3-70B.

Reports 2-19 pp improvement over prior layer-pruning baselines.

Adds a gradient-flow derivation explaining why deep layers matter less.

Notes that iterative prune–tune cycles provide no benefit over one-shot pruning.

### Strengths
Comprehensive and reproducible experimental design.

Honest ablations revealing when complexity adds no value.

Simple, clearly-defined recipe that practitioners can reproduce in hours.

Really primarily illustrates a weakness in all the other papers on layer pruning: they ought to have used final layer pruning as the obvious control experiment and have failed to do so. Providing this missing baseline is probably important within the narrow domain of layer pruning.

Experimentally verifies a fact that is part of the design of LLM architectures and their understanding as unrolling, and has also been examined theoretically and by other experimental methods before.

### Weaknesses
Scope: confined to layer pruning; ignores dominant GPU-friendly methods (structured width pruning, 2:4 sparsity, quantization).

Novelty: theoretical component re-derives known results; empirical finding is mainly that others’ metrics fail.

Practical relevance: minimal for most users in practice. For people training from scratch, incremental deepening is probably preferable. For people trying to squeeze a large model into a slightly smaller GPU, quantization and GPU-friendly sparsity are probably preferable even if smaller models aren't just available. The primary use case is where a user has an unusual model, cannot control training, but quickly wants to squeeze it into an existing GPU with somewhat more limited space.

The paper really ought to have compared final layer pruning with models of the same final size trained from scratch, since they are architecturally identical. This would indicate whether final layer pruning could be a useful shortcut for generating a simple multi-depth collection of models

### Questions
Ought to address:

- Clarify that the theoretical contribution is an application of prior analyses, not new theory.
- Discuss why reverse-order pruning should be a baseline control for future pruning papers.

I think the following is really future work:

- Benchmark versus quantized models at equal memory budgets.
- Investigate interaction of quantization and depth--does aggressive quantization change which layers are dispensable?
- Compare against GPU-usable sparsity (2:4) and width pruning for completeness.
- Compare with models trained from scratch at the final depth, as well as models incrementally grown.

### Soundness
4

### Presentation
3

### Contribution
2
