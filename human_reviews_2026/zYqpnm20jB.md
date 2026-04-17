# E$^3$-Pruner: Towards Efficient, Economical, and Effective Layer Pruning for Large Language Models

- Decision: Reject
- Scores: 4, 4, 6, 2, 6

## Abstract
With the increasing size of large language models, layer pruning has gained increased attention as a hardware-friendly approach for model compression. However, existing layer pruning methods struggle to simultaneously address key practical deployment challenges, including performance degradation, high training costs, and limited acceleration. To overcome these limitations, we propose \name, a task-\underline{E}ffective, training-\underline{E}conomical and inference-\underline{E}fficient layer pruning framework. \namespace introduces two key innovations: (1) a differentiable mask optimization method using a Gumbel-TopK sampler, enabling efficient and precise pruning mask search; and (2) an entropy-aware adaptive knowledge distillation strategy that enhances task performance. Extensive experiments over  diverse model architectures and benchmarks demonstrate the superiority of our method over state-of-the-art approaches. Notably, \namespace achieves 96\% accuracy, a mere 0.8\% drop from the original model (96.8\%) on MATH-500 when pruning 25\% layers of Qwen3-32B, outperforming existing SOTA (95\%), with a 1.33$\times$ inference speedup by consuming merely 0.5B tokens (0.5\% of the post-training data volume).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a layer pruning framework utilizing differentiable Gumbel-TopK sampler for and entropy-weighted knowledge distillation (KD) strategy for accuracy recovery. Experiments across different models show empirically better performance retention over baselines.

### Strengths
Efficient LLM pruning while addressing all three axes (accuracy, cost, speed) is of high importance. The experiments were well-designed in general and the results look promising.

### Weaknesses
+ The core technical contributions are straightforward heuristics adapted from existing differentiable pruning methods (Gumbel-Softmax) and weighting schemes. They offer negligible conceptual advance and are presented without necessary theoretical justification.

+ Lacking analysis of convergence and other key characteristics.

+ No wall-clock latency analysis.

+ The comparison restricts competitive baselines like DarwinLM.

The model optimization is presented as a black box, without any interpretability analysis of the resulting layer redundancy profile.

### Questions
+ Provide a rigorous hardware-level latency analysis on a standard GPU. Quantify the real-world inference throughput after pruning.

+ Explain why the Gumbel-TopK sampler should converge to a better mask than a simpler method, and show the sensitivity of the final mask composition to the annealing temperature (τ) schedule. Would a random search for layer indices yield a mask that performs comparably after distillation?

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
$E^{3}$-PRUNER is a layer pruning framework for large language models. This approach combines a differentiable search strategy using a Gumbel-TopK sampler with adaptive knowledge distillation techniques to find and train the optimal layer pruning configuration in an end-to-end manner. Extensive experiments on multiple models demonstrate the superiority of it over state-of-the-art approaches.

### Strengths
- The verification experiment is done quite thoroughly, spanning from 7B to 671B.
- The paper is easy to follow.
- The Gumbel-TopK sampler is introduced to replace the discrete TopK selection, achieving end-to-end differentiable optimization of layer pruning masks.

### Weaknesses
- As shown in Table 1, the model accuracy remains above 80% after pruning from 6.7B to 2.7B, which is extremely unusual in the field of LLM layer pruning. I am skeptical of this result and hope that the authors can provide a reproducible model checkpoint. In addition, since the proposed method can achieve such a high compression rate for LLaMA-2-7B, why is the compression rate so low for Qwen3-32B and DeepSeek-R1? In theory, the more model parameters there are, the more parts that can be compressed.
- Regarding the performance recovery of the pruned model, this paper uses adaptive knowledge distillation. Is it to fine-tune the entire model or only some parameters?
- Compared with traditional LoRA fine-tuning, how much performance improvement does the proposed adaptive knowledge distillation have, and how much difference is there in computational cost between these two?
- How much performance improvement would occur if other pruning methods were used with Adaptive Knowledge Distillation?
- No comparison with more advanced layer pruning methods.

### Questions
Given that you used knowledge distillation to restore the pruned DeepSeek-R1 model, how many GPUs do you use for the experiment?

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
4

### Summary
This paper proposes E3-PRUNER, which is a layer-pruning framework for large language models that learns which layers to keep or remove through a differentiable mask optimized with a Gumbel-TopK sampler. It further uses entropy-aware adaptive knowledge distillation to retain performance. Experiments show it achieves up to 2.18× speedup with minimal accuracy loss, offering an efficient, economical, and effective solution for LLM compression.

### Strengths
1. This paper proposes a differentiable mask learning framework (Gumbel-TopK) for layer pruning, enabling efficient gradient-based layer selection.
2. The method achieves a good pruned model performance, outperforming prior pruning methods.
3. The paper further introduces entropy-aware adaptive knowledge distillation, effectively preserving key reasoning tokens.
4. The experiments demonstrate consistent and superior results across multiple LLMs with minimal accuracy loss.

### Weaknesses
1. The paper provides a limited theoretical explanation of why the Gumbel-TopK mask search is able to identify the optimal layers.
2. The paper does not clarify whether the performance gain comes from the layer pruning method or the Adaptive Knowledge Distillation. It would be better to compare the zero-shot performance of the pruned model without fine-tuning or apply Adaptive KD to baseline pruning methods to evaluate their relative effects.

### Questions
This is the first academic paper I have seen that fine-tunes the 671B DeepSeek-R1 model. How many GPUs do you use? This information could be included in the Settings section of the paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes $E^3$-Pruner, a layer pruning framework for large language models that integrates a differentiable Gumbel-TopK sampler for efficient mask optimization and an entropy-aware adaptive knowledge distillation strategy to address the limitations of existing methods in performance preservation, training cost, and inference acceleration.

### Strengths
1. This paper combines a differentiable Gumbel-TopK sampler, for efficient and accurate pruning mask search with an entropy-aware adaptive knowledge distillation strategy, for enhanced knowledge transfer with reduced computational cost.
2. This paper executes extensive experiments on across diverse LLMs with different sizes and architectures, and evaluates on multiple benchmarks, demonstrating the generalization and practicality of the proposed $E^3$-Pruner framework.

### Weaknesses
1. **Limited novelty:** Gumbel-TopK sampling for pruning has been explored in prior works [1-2] for model compression, and the progressive layer pruning strategy is a common approach (e.g., SLEB [3]) with no significant innovation here.
2. **Incomplete baselines:** Heuristic layer pruning methods like SLEB [3] and Shortened LLaMA [4] should be added as baselines to enable more comprehensive comparison and better highlight the proposed method’s advantages.
3. **Unfair comparison design:** The paper fails to clarify whether baseline methods update parameters (for performance recovery after pruning). Comparing parameter-updating and non-updating methods (e.g., training-free ShortGPT’s pruned model vs. $E^3$-Pruner’s fine-tuned model) is inappropriate; training-free baselines should be compared with $E^3$-Pruner’s post-search model.
4. **Insufficient evidence for training economy:** Only training token counts are reported. Actual training time and computational resources (e.g., GPU hours) during pruning should be provided to substantiate the claim of training efficiency.
5. **Lack of ablation study on layer importance initialization:** Comparing different initialization methods (e.g., random initialization, ShortGPT’s layer importance metric) would clarify how initialization impacts final performance.

[1] Gonzalez-Carabarin, et al. Dynamic Probabilistic Pruning: A General Framework for Hardware-Constrained Pruning at Different Granularities. TNNLS 2022.

[2] Tan, et al. Mutually-aware Sub-Graphs Differentiable Architecture Search. arXiv 2021.

[3] Song, et al. SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks. ICML 2024.

[4] Kim, et al. Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods. arXiv 2024.

### Questions
1. Minitron proposed iterative prune+distill for model pruning. How many prune+distill iterations were conducted in your experiments? If multiple, please provide the number of iterations and detailed settings; if only one, explain the rationale for choosing single over multiple iterations.
2. How does the performance of ShortGPT-pruned models fine-tuned with $E^3$-Pruner’s adaptive knowledge distillation compare to pruned models after $E^3$-Pruner's two stages process? This comparison would help isolate the effects of the pruning method and distillation strategy.
3. Could you provide qualitative examples of outputs from the original and pruned models on specific tasks (e.g. MMLU using Qwen2.5-14B-Instruct)? This would illustrate the practical impact of pruning on model performance.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposed E³-PRUNER, a two-stage layer-wise pruning framework tailored for LLMs. In the mask search stage, it learns a binary layer mask with a Gumbel-TopK sampler and STE backward, warmed up by KL-based layer-importance initialization and a curriculum schedule that gradually increases the pruning ratio. In the recovery stage, it applies adaptive knowledge distillation, using offline teacher Top-K logits and token-wise entropy weighting on the per-token loss. In the experiments, the method preserves accuracy while reducing depth and improving latency under modest training budgets across models and tasks.

### Strengths
1. The problem statement and challenge are clear and significant. Section 2.2 provides a detailed analysis of why existing training-free, differentiable, and NAS-based approaches each fall short (e.g., accuracy drop, high token budgets, irregular speedups), setting up a precise target for improvement.
2. This paper is fairly written. The narrative is well-structured and easy to follow: motivation, formulation, differentiable mask search, and recovery via adaptive KD. Figures and algorithms (Fig. 3; Algs. 1–2; Eqs. (4)–(6)) make the procedure executable and reduce ambiguity about forward/backward behavior and training schedules.
3. Clear real-world deployment evidence supports the author’s claim on effectiveness and inference efficiency gain. Beyond accuracy tables, the paper reports wall-clock improvements for large-scale models

### Weaknesses
1. The behavioral consistency metric is weakly defined. The paper measures consistency as average accuracy on a small mixed set rather than teacher–student agreement (e.g., output match rate, output distributions, or log-prob correlations). This undermines the claim that KD better preserves behavior.

2. Storage/IO for offline KD is unquantified. The method relies on offline Top-K logits and asserts “minor storage,” using Top-10 in all configs, but provides no concrete footprint or bandwidth numbers under long contexts and 0.5B-token training budgets. This limits reproducibility and deployability assessments.

3. Fairness of the comparison. ShortGPT is training-free, so its lower accuracy vs. trained methods is expected; using it as a headline accuracy comparator can be misleading. A fair alternative is ShortGPT with the same recovery budget (SFT/KD) to normalize token use.

4. Missing comparisons to recent depth-pruning baselines. The paper does not compare to other layer/depth pruners that replace blocks via KD (e.g., LLM-Streamline), which target similar deployment goals.

### Questions
1. Authors assert “minor storage,” set Top-10 in all configs, but provide no concrete footprint/bandwidth numbers. To better understand the practicality of the Top-10–logits setup, could you share results, such as the approximate on-disk size per token when storing Top-10 logits and the total footprint for 0.5B tokens under typical context lengths?

2. Table 3 shows the ablation of the searching budget. Any clue why adding the search budget would even decrease performance?

3. How does E³-PRUNER compare with other LLM depth pruners? For example, LLM-Streamline[1], which identifies less important blocks and substitutes them with a light-weight block obtained via KD.

[1]Streamlining Redundant Layers to Compress Large Language Models, iCLR'25

### Soundness
3

### Presentation
3

### Contribution
2
