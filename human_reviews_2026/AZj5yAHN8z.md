# MIN-MERGING: MERGE THE IMPORTANT NEURONS FOR MODEL MERGING

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Recent advances in deep learning have led to a surge of open-source models across diverse domains. While model merging offers a promising way to combine their strengths, existing approaches often suffer from parameter conflicts that degrade performance on domain-specific tasks. We propose MIN-Merging, a router-based framework that selectively merges the most important neurons to reduce such conflicts.Extensive experiments on Computer Vision and Natural Language Processing benchmarks show that MIN-Merging achieves consistent gains on in-domain tasks while preserving the generalization ability of pretrained models on out-of-domain tasks. These results highlight its effectiveness as a practical solution to the parameter conflict problem in model merging.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work presents a model merging pipeline that features:

1. Finetune with LoRA and prune layer-wise.
2. A learned router that picks which LoRA branch to activate. 
3. This router permits dynamic merging at inference time.

The authors then compare this pipeline with several existing methods on common benchmarks.

### Strengths
The writing and organization of this work are mostly clear and easy to follow. 

The experiments are clearly explained and provide convincing evaluations of the proposed methods.

### Weaknesses
I fail to understand the novelty and the effectiveness of the proposed method.

On the method:

1. Finetuning with LoRA for merging is not new. For example, many works on this line are compared in [1]
[1] Tang, Dennis, et al. "LoRA Merging with SVD: Understanding Interference and Preserving Performance." ICML 2025 Workshop on Reliable and Responsible Foundation Models.

2. Merging weights after filtering is also not new, for example in [2]. The only difference seem to be entry-wise filtering vs layer-wise filtering. 
[2]Yadav, Prateek, et al. "Ties-merging: Resolving interference when merging models." Advances in Neural Information Processing Systems 36 (2023): 7093-7115.

3. Introducing a router that enables dynamic merging. As the authors pointed out, this resembles Twin-Merging. It is also similar to [3]

[3] Wu, Xun, Shaohan Huang, and Furu Wei. "Mixture of lora experts." arXiv preprint arXiv:2404.13628 (2024).

Although this work proposes a particular combination of known techniques, I would appreciate the contribution, if:
1. New SOTA can be achieved with statistical significance. 
2. Or new understanding / insights are made. 

However, there is little new insights. Further the results presented seem to be worse by a significant margin than what was known (for example in the CV exp, see [5][4] )

[4] Marczak, Daniel, et al. "No task left behind: Isotropic model merging with common and task-specific subspaces." arXiv preprint arXiv:2502.04959 (2025).
[5] Yang, Enneng, et al. "Representation surgery for multi-task model merging." arXiv preprint arXiv:2402.02705 (2024).

Consequently, I fail to see the contribution of this submission.

Minor question:
In 3.3, why is the merging considered layer-wise? Eq(10) seems to be identical for all layers. Are experts chosen differently in each layer?

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the problem of merging multiple fine‑tuned models that may contain conflicting parameter updates and cross‑task interference. To mitigate these issues, the authors propose MIN‑Merging, a neuron‑level framework that identifies and selectively merges only the most important components across expert models. The method operates in three coordinated stages: (1) Expert Enhancement, where redundant LoRA layers or neurons are pruned to retain each expert’s most informative parameters; (2) Router Training, where a lightweight MLP router learns to assign input‑dependent weights and select the top‑k relevant experts; and (3) Dynamic Layer‑wise Merging, where the merged model combines the core layers of the most relevant expert with redundant layers from others to balance specialization and generalization.

Across GLUE‑style NLP and a suite of vision datasets, MIN‑Merging: (i) outperforms weight averaging, task arithmetic, TIES‑Merging, and Twin‑Merging; (ii) sometimes exceeds single‑task fine‑tuning; (iii) scales to 7B models; (iv) maintains OOD performance on MMLU.

### Strengths
- The paper’s operational combination of (a) neuron/layer pruning to sharpen per‑task experts and (b) a top‑k router to drive input‑conditional merging is a tidy assembly of known ideas. The hierarchical “core vs. redundant” routing within merging is a mildly novel twist.
- Cross‑domain scope (NLP + CV) and an attempt at ablations (removing filtering / hierarchical / router) shows some attention to component contribution.
- If credible and reproducible, an input‑conditional merging recipe that’s lighter than MoE at inference (single merged model) could be impactful for multi‑task deployment on limited hardware.

### Weaknesses
- The paper never gives a concrete, reproducible criterion for selecting “core” vs. “redundant” layers/neurons beyond descriptive language and a hand‑wavy SNR analogy.
- Equations (4–5, 12) invoke SNR, mutual information, and entropy but there is no concrete estimation procedure or empirical SNR plots.
- "Performance" values >100% appear in Tables 10–18; if they refer to accuracies, this is of course impossible. Please clarify this issue.
- Competing methods (TIES‑Merging, Twin‑Merging, DARE, AdaMerging) are sensitive to hyperparameters and sparsity cutoffs; no tuning budgets or fairness criteria are detailed.

### Questions
1. What algorithm selects core vs. redundant units/layers? Provide the formula, thresholds, and a reference implementation.
2. What features does the router ingest (raw inputs, embeddings, intermediate activations)? How is leakage avoided (e.g., label leakage via task‑specific preprocessing)?
3. The Limitations section admits uncertainty across heterogeneous backbones. Can MIN‑Merging operate when experts differ in depth/width or when only partial layer alignment exists?
4. How does the method behave when experts are antagonistic (e.g., mutually exclusive label spaces or heavily conflicting features)?

### Soundness
1

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
4

### Summary
This paper proposes MIN-Merging, a router-based framework for model merging. Unlike prior weight-averaging or task-vector approaches, MIN-Merging selectively merges only the most important neurons from each expert model, guided by neuron importance estimation and a dynamic routing mechanism. The framework consists of three stages: expert enhancement, router training, and dynamic layer-wise merging. Through these steps, the model dynamically combines the LoRA adaptors (sparse linear combination) via router.

Experiments on NLP (GLUE with Qwen2.5-0.5B/7B) and CV (ViT-Base on 10 datasets) show that MIN-Merging outperforms prior merging methods, and in some cases even exceeds task-specific fine-tuned models. It also demonstrates scalability to larger models, robustness to out-of-domain tasks (MMLU), and efficiency gains in memory and inference time.

### Strengths
1. **Novel framing of neuron-level merging**. The paper frames a new approach that dynamically merges fine-tuned adapters by emphasizing neuron importance and task-aware routing. While not conceptually deep, the integration of pruning, routing, and dynamic weighting is novel within the LoRA merging context.
2. **Broad and strong empirical evaluation**. The experiments span both NLP (GLUE, MMLU) and CV (ViT-based) tasks, including small and large model scales. The results consistently show improvements over prior model-merging baselines such as Task Arithmetic and Twin-Merging.
3. **Potential practical utility**. The router-based linear composition of adapters could be a useful engineering strategy for quickly combining fine-tuned LoRA modules without retraining or maintaining separate models.

### Weaknesses
1. **Misleading framing as "model merging"**. The method does not produce a single unified model; it keeps all LoRA adapters and linearly combines them at inference. This is conceptually closer to adapter ensembling than to true model merging. The paper should make this distinction explicit and mention that the method benefits from small size of LoRA adaptors. Because all expert adapters are retained, memory usage and inference cost scale with the number of tasks. This contradicts the claim of improved parameter efficiency and undermines the notion of "merging into one compact model."

2. **Weak or confusing presentation**. Figures 1 and 2 fail to clarify the core mechanics. Figure 1 is generic, while Figure 2 suggests an abstract router-based fusion without showing how LoRA adapters are combined. Moreover, the text obfuscates the central role of LoRA, giving the misleading impression of a general model-merging algorithm.

3. **Lack of true theoretical or mechanistic insight**. The core novelty (selecting important neurons and weighting experts) is largely heuristic, with limited analysis of why it outperforms simpler ensembling or gating mechanisms.

### Questions
1. Since the method heavily relies on LoRA adapters for scalability, could the authors discuss or provide evidence of how the adapter size (rank) affects performance and memory efficiency? Would the approach remain feasible under full fine-tuning or larger adapter ranks?
2. Figures 1(a) and 2 could be improved to more clearly convey the core idea (how neuron importance and routing interact). In addition, Figures 3 and 4 would be easier to interpret with standard readability enhancements (e.g., grid lines, consistent axes, clearer legends).
3. In Table 2 (CV results), the PLANTS column shows that Task-Arithmetic and Ties-Merging achieve the best or nearly best performance.

### Soundness
3

### Presentation
1

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
The paper proposes MIN-Merging, a novel framework to merge multiple LoRA-based expert models by mitigating parameter conflicts. It uses a three-stage process: "Expert Enhancement" to partition layers into core/redundant sets, a "Router" to select top-k experts, and "Dynamic Merging" to combine them. Experiments on NLP and CV tasks are conducted to show the method outperforms baselines and even individual fine-tuned models.

### Strengths
1. The paper is well-structured and clearly written. The motivation for solving parameter conflicts is well-established, and the proposed three-stage solution (Enhancement, Routing, Merging) follows a logical and intuitive progression, making the overall argument easy to follow.

2. The paper adopts an intuitive and relatively simple approach to solve the foreseeable problem of parameter conflicts. The description of the router-based dynamic merging mechanism is clear, and the high-level idea of partitioning layers into 'core' and 'redundant' sets is an elegant conceptual contribution.

3.The observation of redundant layer is quite inspiring and worth further investigation.

### Weaknesses
1. The authors claimed that the problem of task conflict is solved by the method which seems kind of over-claimed. From my point of view, the methods only mitigate the problem by layer-wise reduction instead of solving it.

2. Figure 3 demonstrates that the fine-tuned model could sometimes perform better with certain layers dropped. This is a very interesting observation and need further explanation. 

3. Some minor mistakes should be corrected. For example, in Figure 1, two images are identical at bottom. At line 248, ‘he expert …’ should be ‘the expert …’. The paper needs to be polished.

### Questions
1. I wonder if the final performance outcome source from the observation in Figure 3, could you include performance of results of finetuned models combined by MoE method with the redundant layer reduced to see if the fusing method causes performance drop?

2. To evaluate the performance of the proposed method on task conflict issue during fusing, the author should evaluate more multitask methods instead of the most original one.

3. Could you please show some results under the case that the redundant layer are randomly assigned to demonstrate the effect of pruning

### Soundness
2

### Presentation
3

### Contribution
3
