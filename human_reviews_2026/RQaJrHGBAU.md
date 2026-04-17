# PATCH: Learnable Tile-level Hybrid Sparsity for LLMs

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Large language models (LLMs) deliver impressive performance but incur prohibitive memory and compute costs at deployment. Model pruning is an effective way to reduce these overheads, yet existing approaches face challenges: unstructured sparsity, where nonzeros can appear anywhere, preserves accuracy but yields irregular access patterns that prevent GPU acceleration, while semi-structured 2:4 sparsity is hardware-friendly but enforces a rigid 50% pattern that degrades model quality. To bridge this gap, we introduce PATCH, a hybrid sparsity framework that enables a continuous sparsity ratio between 0% and 50%. PATCH partitions weight matrices into tiles, assigning each tile to be either dense or 2:4 sparse via a learnable mask selection mechanism. This design provides fine-grained control over accuracy–acceleration tradeoffs and supports non-uniform sparsity across layers, leading to superior overall quality. Across models from 0.5B to 8B parameters, PATCH consistently narrows the gap to dense accuracy while delivering practical speedups. For instance, on LLaMA-2 7B with an A6000 GPU, PATCH achieves 1.18×–1.38× end-to-end speedup over dense baselines while improving accuracy by 0.37%–2.96% compared to the state-of-the-art 2:4 pruning method, MaskLLM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a "hybrid sparse" framework PATCH. Instead of simply mixing unstructured and structured sparseness, it divides the weight matrix into "tiles" and dynamically decides whether to keep each tile dense or adopt hardware-friendly 2:4 sparseness through a learnable mask. The effectiveness of this method has been verified on models with parameters ranging from 0.5B to 8B. However, since no experiments are conducted on larger models (e.g., 70B), the generalization of the proposed method cannot be verified.

### Strengths
- The proposed "hybrid sparse" framework is not a simple mixture of unstructured and structured sparse.
- The paper is easy to follow.
- The ablation experiment is very thorough.

### Weaknesses
- This paper does not report the computational cost of the masking process, which will significantly affect the effectiveness of the pruning algorithm.
- No experiments are conducted on larger models (e.g., llama3-70B), which resulted in a lack of generalization of the algorithm.
- Can the proportion of weight pruning be increased to 80-90%? What is the performance of the proposed method in this case?
- Table 3 shows the impact of different tile sizes on the performance of the pruned model. So how do different tile sizes affect the performance acceleration of the pruned model?
- If the authors use other datasets to train the mask, will the effect be different?

### Questions
Compared with existing channel pruning and layer pruning methods, which one has better hardware acceleration effect?

### Soundness
2

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
5

### Summary
This paper proposes to learn a heterogeneous sparsity for weight tensors which combines dense and 2:4 sparse tiles. The introduction of dense tiles enables the model sparsity to range from 0-50% and vary between layers. Gumbel-Softmax reparameterization is used to learn soft masks during training which converge to hard masks by the end of training as the distribution hyperparameters, $\tau$ & $k$, are modified accordingly. The proposed method, PATCH, at 45% sparsity exceeds the accuracy of 2:4 one-shot pruning methods and MaskLLM on loglikelihood-based question answering language benchmarks. The method is benchmarked using Triton kernels from the STOICC library and demonstrates up to 1.38x acceleration at 45% sparsity on A6000 GPUs.

### Strengths
* Heterogeneous sparsity within a single weight tensor is an under-explored setting for exploiting sparsity for efficiency gains. The proposed method is hardware friendly for commodity hardware. 
* The layer-wise sparsity distribution is learned rather than imposed via heuristic. The paper includes some intriguing details on which layers are generally more sparse, such as the FFN layers, whereas the key/value projections remain more dense. 
* A number of ablations are provided, including: joint mask and tile vs. tile-only search, tile size, layerwise sparsity allocation.
* The observed acceleration on A6000s at 45% acceleration approaches the expected benefit of pure 2:4 sparsity.

### Weaknesses
Overall, my main concerns relate to gauging the accuracy and throughput of PATCH in comparison with simpler approaches. 

# Major concerns
The following represent key weaknesses that must be addressed to increase the rating:
* Heterogeneous intra vs. inter layer sparsity: The sparsity distributions suggest that PATCH predominantly prunes the FFN layers. A robust baseline may be to simply prune only the FFN linear layers to 2:4 sparsity. For instance, with Llama-3.1-8B-Instruct, excluding layernorm parameters, ~81% of the parameters associated with each decoder block are contained in the FFN module. Pruning just the FFN modules to 2:4 will yield a global sparsity of 40%. For direct comparison with PATCH at 35%, 4 decoder layers can be left dense to yield a global sparsity of 35.3%, for instance, the last 4 decoder blocks. 
* Sparse fine-tuning: PATCH requires significant training resources to learn the optimal patch assignments and specific 2:4 masks. Given a fixed compute budget, simply fine-tuning the sparse weights from a one-shot pruning method may exceed both the accuracy and throughput of PATCH. 
* 2:4 benchmarks: What is the throughput of global 2:4 pruned models under the same settings that PATCH was benchmarked? What is the throughput when only the FFN layers are pruned to 2:4?
* Sparsity-matched accuracy comparisons: Direct comparison of 45% PATCH and 2:4 sparse models from other compression methods is challenging. How do select baselines such as SparseGPT pruned to 3:4 compare with the 25% sparse PATCH models? 

# Minor concerns
The following are minor concerns, typos, etc. which would improve the work but do not affect the final rating: 
* Lack of generative evaluations: Loglikelihood-based QA benchmarks and PPL can be misleading when evaluating compressed models. Including a benchmark that assesses the generative outputs of the model, such as EvalPlus, would help establish how practical PATCH is for real-world tasks. 
* Hopper benchmarks: Generally the efficiency gains from 2:4 sparsity have been less impressive on Hopper. It would be interesting to see benchmarks on the current generation of NVIDIA GPUs to assess the benefits of PATCH.
* L092: “while freezing the initial 2:4 mask fixed” -> “while freezing the initial 2:4 mask”.
* Section 5.3 quantization section: This paragraph is lacking details. Including further discussion on the compatibility of quantization with PATCH would be beneficial. 
* Layer-wise sparsity distribution comparisons: Prior methods for determining optimal layer-wise sparsity ratios have been proposed. Comparisons between the distribution learned by PATCH and [1-3] may be interesting. 

[1] L. Li et al., “Discovering Sparsity Allocation for Layer-wise Pruning of Large Language Models,” presented at the The Thirty-eighth Annual Conference on Neural Information Processing Systems, Nov. 2024. Available: https://openreview.net/forum?id=rgtrYVC9n4

[2] H. Lu, Y. Zhou, S. Liu, Z. Wang, M. W. Mahoney, and Y. Yang, “AlphaPruning: Using Heavy-Tailed Self Regularization Theory for Improved Layer-wise Pruning of Large Language Models,” Oct. 14, 2024, arXiv: arXiv:2410.10912. doi: 10.48550/arXiv.2410.10912.

[3] L. Yin et al., “Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity,” Oct. 08, 2023, arXiv: arXiv:2310.05175. doi: 10.48550/arXiv.2310.05175.

### Questions
See major concerns in the Weaknesses section above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work aims at solving the efficiency challenge of deploying LLMs by introducing PATCH, a new hybrid pruning framework that mixes dense weights and structured sparse weights at a tile level within each weight matrix. Vanilla pruning methods either use unstructured sparsity which is not hardware-efficient, or semi-structured 2:4 sparsity which is hardware-friendly but often hurts accuracy. PATCH bridges this gap by allowing a continuous sparsity ratio of 0-50% across the model. It partitions weight matrix into small tiles and uses a learnable binary tile mask (trained with Gumbel-Softmax relaxation to allow differentiable sampling) to assign each tile with either dense or 2:4 sparse, allowing important regions to remain fully dense while still exploiting 2:4 sparsity elsewhere. The authors also introduce a further memory-efficient variant of PATCH. This variant further reduces the number of learnable parameters by fixing the fine-grained 2:4 mask (e.g., initialized by a one-shot method like MaskLLM or SparseGPT) and only learning the tile assignments.

### Strengths
1. This work addresses a critical quality v.s. hardware efficiency tradeoff with hybrid apporach. By allowing a mixture of dense and 2:4-sparse tiles within weight matrices, PATCH achieves pruning ratios that were previously impossible with hardware-friendly methods.
2. PATCH shows SOTA results in terms of the aforementioned trade-off for pruned LLMs. Experiment results consistently show that PATCH-pruned models have higher accuracy / lower perplexity than those pruned by other methods at comparable sparsity level. Also this work demonstrates practical efficiency gains with up to 1.38x end-to-end speedup. Many prior pruning works struggled to translate sparsity into actual speed due to irregular patterns. PATCH overcomes that by construction (using a hardware-supported pattern and grouping it in tiles). Showing a non-uniform sparsity method yielding clock speed improvements is a commendable achievement.
3. Evaluation coverage is thorough -- multiple LLM model families (Llama, Qwen, Gemma) across model sizes from 500M up to 8B params on 8 diverse and repersentative zero-shot tasks (MMLU, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, OpenBookQA, RACE, HellaSwag) plus WikiText2 perplexity
4. Using the Gumbel-Softmax for differentiable mask is a smart choice that leverages known methods in a new context

### Weaknesses
1. PATCH’s benefits rely on specialized support with native 2:4 sparse op (this is not a weakness of this work per-se but inherited from 2:4 sparsity for a sparsity-accuracy tradeoff). It uses the STOICC compiler to handle the irregular pattern of dense v.s. sparse tiles which might limit adoption in the broader community in the short term.
2. Pruned models still do not fully match the accuracy of the dense models. For example, at 45% sparsity, Llama-2 7B under PATCH reaches ~49% average accuracy compared to 54.6% in dense; Even at 25% sparsity (the least aggressive sparsity PATCH experimented with), it gets ~51.6% vs 54.6% dense. This gap is much smaller than with previous structured pruning (MaskLLM had ~6 point gap at 50%), but it’s still a performance loss.

### Questions
1. Have the authors considered or attempted applying PATCH to models larger than 8B, such as 13B or even 70B LLMs? If not in practice, a discussion on expected behavior would be valuable. Do the authors anticipate any new challenges at that scale?
2. Have the author evaluated integrating a low-rank adaptation during mask training (e.g., fine-tuning a small low-rank subset of weights to compensate for pruning)?
3. Could the authors compare PATCH’s learned layer-wise sparsity distribution to a heuristic allocation? Would a combined approach further benefit accuracy? Prior work (Yin et al. 2025) suggests not all layers should be pruned equally. It might strengthen the work to discuss or even experiment with a strawman baseline
4. How stable are the results across different random seeds or runs? Did the masks learned converge to similar patterns each time, or was there variance? -- Gumbel-Softmax introduces stochasticity so some note on this front would be beneficial
5. After having the final mask, have the authors considered doing a brief fine-tuning of the remaining weights to further recover accuracy?

### Soundness
3

### Presentation
3

### Contribution
3
