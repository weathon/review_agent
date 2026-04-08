## Human Reviewer 1

### Summary
This paper introduces DarwinLM, a method for structured pruning of large language models that uses evolutionary search to find optimal non-uniform compression patterns. The core idea is to generate multiple "offspring" models through mutations that shift sparsity between layers, then select the best candidates using a multi-step training-aware process. The method builds a database of pre-pruned layers at different sparsity levels using second-order information, then searches over combinations of these levels while incorporating lightweight fine-tuning to predict which models will perform best after full training. The authors test on several models (Llama-2-7B, Llama-3.1-8B, Qwen-2.5-14B) and show improvements over uniform pruning and competing methods like ShearedLlama and ZipLM. They claim to achieve comparable or better accuracy while using 5x less training data than ShearedLlama. The method is also extended to MoE architectures, which is claimed as a first for structured pruning.

### Strengths
1. The multi-step selection strategy that progressively increases fine-tuning data (10K→50K→100K→200K tokens) is both intuitive and empirically validated. By showing in Figure 2 that small-scale training predicts larger-scale performance, the paper introduces a practical and well-motivated solution to a long-standing inefficiency in pruning and neural architecture search.

2. The method is evaluated across diverse model families and sizes—up to 70B parameters—with generally fair baselines and detailed ablations (offspring count, sparsity levels, fitness metrics). The results consistently demonstrate superiority over baselines like ShearedLlama and ZipLM.

### Weaknesses
See Questions.

### Questions
1. Evolutionary search feels overcomplicated. The core is just trying different per-layer sparsity combinations. The mutation operator is trivial (swap sparsity between two layers).  Do simpler search methods like beam search also work? It would be better to have comparison with random search given the same budget.

2. Search cost are unclear. 200 generations × 16 offspring = 3,200 evaluations, each requiring model stitching and training. How does total cost compare to ZipLM's dynamic programming or ShearedLlama's approach?

3. How sensitive is DarwinLM to the choice of training-aware selection token budgets (10K–200K)? Could different scaling change the final selection outcome?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 2

### Summary
DarwinLM proposes training-aware, non-uniform structured pruning for LLMs. It first builds a per-layer “sparsity level” database via second-order one-shot pruning, then uses an evolutionary search with “level-switch” mutations under a target size/speed constraint. Fitness is measured via KL divergence to the dense model. Selection is multi-step and training-aware, followed by a final 10B-token post-compression finetune. The method reports one-shot and post-train results on Llama-2-7B, Llama-3.1-8B, Qwen-2.5-14B, and an MoE case (Qwen-3-30B-A3B).

### Strengths
* Clear pipeline: second-order one-shot pruning, sparsity-level database, evolutionary search with speed/size constraints, short finetune. 

* Training-aware selection nicely predicts which offspring recover best after longer finetunes; ablation supports the idea.

### Weaknesses
* In the main table (Table 1), the performance gaps shrink notably after finetuning. It’s unclear whether DarwinLM’s one-shot advantage persists under longer training or on performance after training converges. The pruned+finetuned models may still underperform public small dense models (e.g., pruned-llama 3.1 8B vs llama 3.2 3B model). While this is reasonable as only 10B tokens are used for recovering performance, this still raise the need for further fine-tuning the models to be actually used. It would be helpful to include results under longer training or distillation to see if the initialization gain is still there.
 
* The search can yield highly non-uniform (unbalanced) per-layer sparsity. It would be beneficial to show how the final architecture looks and whether unbalanced shapes can harm optimization or stability during continued training. Comparing the final shape with Shearedllama can provide more insights.

* The paper states that "this is the first work to explore structured pruning in MoE architectures" in the abstract, which is not accurate and overstated. Some works have worked on structured pruning of MoE [1,2].

* It's not very accurate to claim "Orthogonal to Minitron/Flextron”. Those pipelines couple structured pruning with KD and depth/width choices; DarwinLM’s search over non-uniform width allocations overlaps in spirit.

* The non-uniform architecture can also affect the inference efficiency. Irregular shapes can reduce kernel efficiency and complicate inference optimization. Some discussion or experiments comparing uniform and non-uniform will be helpful.


[1] SlimMoE: Structured Compression of Large MoE Models via Expert Slimming and Distillation

[2] Demystifying the Compression of Mixture-of-Experts Through a Unified Framework

### Questions
Included above.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 3

### Summary
DarwinLM claims to improve structured pruning of LLMs using a hybrid of second-order saliency and evolutionary search over sparsity allocations. Each generation mutates layer-wise sparsity ratios, fine-tunes offspring, and keeps the fittest models. The authors report strong perplexity and downstream accuracy on LLaMA-2, LLaMA-3.1, and Qwen models, achieving up to 50–60% parameter reduction.

### Strengths
1. Targets structured pruning, which is the only kind of sparsity actually exploitable by modern inference frameworks.

2. Attempts to automate non-uniform sparsity allocation rather than relying on fixed heuristics.

3. The proposed approach could be extend to MoE-based model pruning which is a nice bonus.

### Weaknesses
1. The paper does not report latency, throughput, or kernel utilization results. I only found the authors roughly mentioned it in a small table (Table 4) Without demonstrating that the pruned models actually run faster on common inference engines such as TensorRT-LLM or vLLM, it remains unclear whether the proposed structured pruning translates into practical efficiency gains.

2. The evolutionary search involves training many fine-tuned offspring models, which likely requires substantial compute. However, the paper does not provide GPU-hour or data-usage accounting, making it difficult to assess whether the approach offers a favorable cost-benefit trade-off compared to simpler pruning baselines.

3. Combining second-order importance estimation with an evolutionary search procedure builds on several existing frameworks (e.g., AutoCompress, MetaPruning, ShearedLLaMA). The overall idea is coherent but does not introduce a clearly new optimization insight or theoretical advancement beyond prior art.

4. (minor) The Darwinian terminology somewhat overemphasizes the novelty of the approach. In essence, the method performs parameter mutation and selection within a standard hyper-parameter search loop rather than a biologically inspired or algorithmically distinct evolutionary process.

### Questions
1. Can you provide concrete latency numbers (e.g., ms/token) before and after pruning on e.g., A100/H100 GPUs under vLLM or TensorRT? Also, can you include more comparison methods instead of only the dense model as baselines in Table 4?

2. What is the full compute cost (GPU hours) of the evolutionary search versus the achieved inference savings?

3. Do your irregular sparsity patterns map cleanly to N:M kernels, or do they require custom CUDA kernels?

4. How does this differ, concretely, from FlexPrune or ShearedLLaMA beyond empirical tuning?

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
4

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper introduces DarwinLM, a training-aware structured pruning framework that treats model compression as an evolutionary process, where candidate submodels are iteratively generated, fine-tuned on small datasets, and selected based on fitness metrics. By integrating lightweight fine-tuning into the evolutionary search, DarwinLM effectively predicts long-term recovery potential and optimizes sparsity allocation across model layers and modules, including both dense and mixture-of-experts architectures. Experiments on Llama and Qwen models demonstrate that DarwinLM achieves state-of-the-art pruning efficiency, recovering over 90% of accuracy with only one-fifth of the training data required by prior methods while nearly doubling inference speed.

### Strengths
1. This paper is innovative in reducing the retraining cost of pruned models through evolutionary search, and it demonstrates strong practical value for real-world applications.

2. It is among the few works that conduct structured pruning on MoE models, which can inspire future research on lightweight optimization of MoE architectures.

3. The experiments are solid and comprehensive, demonstrating the effectiveness of the proposed method across several recent large language models.

4. The paper is clearly written and easy to follow.

### Weaknesses
The non-uniform sparsity allocation may affect the deployment efficiency of the pruned model. A more thorough discussion and analysis of this effect would strengthen the paper.

### Questions
Does the non-uniform sparsity allocation affect the deployment efficiency of the pruned model？

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
4