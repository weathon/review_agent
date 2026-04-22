# PEML: Parameter-efficient Multi-Task Learning with Optimized Continuous Prompts

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Parameter-Efficient Fine-Tuning (PEFT) is critical for adapting Large Language Models (LLMs) for various tasks. 
Recently, there has been an increasing demands for fine-tuning LLMs for multiple tasks because it requires overall less data for fine-tuning thanks to the common features shared among tasks. More importantly, LLMs are resource demanding and deploying a single model for multiple tasks facilitates resource consolidation and consumes significantly less resources compared to deploying individual large model for each task. Existing PEFT methods like LoRA and Prefix Tuning are designed to adapt to a specific task. LoRA and its variation focus on aligning the model itself for tasks, overlooking the importance of prompt tuning in multi-task learning while Prefix Tuning only adopts a simple architecture to optimize prompts, which limits the adaption capabilities for multi-task. To enable efficient fine-tuning for multi-task learning, it is important to co-optimize prompt optimization and model adaptation. In this work, we propose a Parameter-Efficient Multi-task Learning (PEML), which employs a neural architecture engineering method for optimizing the continuous prompts while also performing low-rank adaption for model weights. We prototype PEML by creating an automated framework for optimizing the continuous prompts and adapting model weights. We compare against state-of-the-arts MTL-LoRA, MultiLoRa, C-Poly, and MoE, and results on the GLUE, SuperGLUE, Massive Multitask Language Understanding and commonsense reasoning benchmarks. The evaluation results presents an average accuracy improvement of up to 6.67%, with individual tasks showing peak gains of up to 10.75%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel framework named PEML (Parameter-Efficient Multi-task Learning)
to address the limitations of existing Parameter-Efficient Fine-Tuning (PEFT) methods in multi-task
learning (MTL) scenarios. PEML integrates low-rank adaptation (LoRA) with a Neural Architecture
Search (NAS) method called PrefixNAS to collaboratively enhance both model weights and the
structure of continuous prompts. The authors conduct extensive experiments across multiple
benchmarks, which demonstrates that PEML outperforms current state-of-the-art multi-task PEFT
methods in both performance and computational efficiency.

### Strengths
This study presents an innovative application of Neural Architecture Search (NAS) to optimize
continuous prompt structures for multi-task learning. The authors show consistent (though
sometimes marginal) improvements over several LoRA-based MTL baselines, demonstrating that
the idea is practically plausible. The paper also includes valuable efficiency analyses comparing
VRAM, throughput, and inference latency, which are crucial for PEFT methods.

### Weaknesses
1. The paper lacks a rigorous explanation that directly links the optimality of the discovered
architecture to the performance gain. The authors compare PEML (LoRA + PrefixNAS) only
against other LoRA-only baselines. The most critical ablation is missing: LoRA + standard
Prefix-Tuning. Without this baseline, it is impossible to determine if the performance gains
stem from the sophisticated PrefixNAS search or merely from the addition of any prompttuning
module. This is a significant omission that needs to be addressed.

2. The paper positions PEML as a "parameter-efficient" and resource-conscious method. While
it is efficient in terms of trainable parameters and VRAM usage during training, this narrative
conveniently ignores the massive upfront computational cost of the NAS search. A fair
comparison must account for this search cost, which is entirely absent from the baseline
comparisons.

### Questions
1. Is there a way to gain insights that using PrefixNAS is a better choice than simply combining
LoRA with a standard, off-the-shelf Prefix-Tuning module?

2. Can the authors provide a rigorous comparison against a baseline of LoRA + standard Prefix-
Tuning? This seems essential to justify the complexity and cost of the entire PrefixNAS
framework.

3. Are there attributes of PEML other than the "optimized" architecture that may contribute to
its performance? How can the authors justify the NAS cost-benefit ratio when the gains are
marginal?

4. Could you provide a quantitative comparison of the total computational cost (Search Time +
Training Time) for PEML versus the (Training Time)-only cost of baselines like MTL-LORA?

5. I noticed in Table 6, PEML significantly underperforms MTL-LORA and DORA on HellaSwag.
Do you have a hypothesis for this failure case?

### Soundness
3

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
3

### Summary
This paper proposes PEML, a method that combines LoRA with PrefixNAS to enhance multi-task learning in LLMs. Unlike existing PEFT methods that focus solely on model weight adaptation, PEML jointly optimizes both prompt alignment through PrefixNAS and model adaptation through LoRA. The authors construct paired training data from FLAN and evaluate on GLUE, SuperGLUE, MMLU, and commonsense reasoning benchmarks using T5-Large, FLAN-T5-Large, LLaMA-7B, and LLaMA2-7B. Results show average accuracy improvements up to 6.67% over baselines including MTL-LoRA, MultiLoRA, C-Poly, and MoE.

### Strengths
Well-motivated problem: The paper clearly identifies limitations of existing multi-task PEFT methods—adapter switching overhead, lack of prompt optimization, and VRAM inefficiency in methods like MultiLoRA.

Comprehensive evaluation: Testing across four major benchmarks (GLUE, SuperGLUE, MMLU, commonsense reasoning) with multiple model families (T5, LLaMA) demonstrates breadth.

Thorough ablation studies: Section 5 and Appendix 7.4 provide good analysis of design choices including layer count, optimization order (parallel vs. sequential), and search space operations.

Practical considerations: The paper addresses real deployment concerns like VRAM usage (Section 7.5) and inference latency (Section 7.7), showing PEML avoids the linear VRAM growth of MultiLoRA.

### Weaknesses
1. Computational cost not properly accounted:
	○ NAS search requires 2 hours on 8×A100 GPUs (16 GPU-hours) per benchmark, representing significant upfront cost.

	○ This one-time cost is dismissed too lightly—for new task combinations, the search must be repeated.

	○ Fair comparison should include baseline hyperparameter tuning time or report total wall-clock time including search.

	○ The claim of "efficiency" is misleading when ignoring NAS computational budget.

2. Incomplete analysis and missing experiments:

	○ No analysis of which tasks benefit most from prompt optimization vs. weight adaptation.

	○ Missing comparison to simpler alternatives: What if we just use LoRA with larger rank? What about manually designed prefix architectures?
	○ Generalization not tested: Does a PrefixNAS architecture found on GLUE transfer to SuperGLUE? This would test if the search truly finds universal structures.

3. Theoretical analysis limitations:

	○ Section 7.1 convergence analysis assumes convex optimization properties (β-smoothness, bounded gradients) that may not hold for neural architecture search.

	○ The analysis does not account for the discrete architecture selection after continuous relaxation.

	○ No analysis of how architecture search affects the joint optimization landscape.

	○ Gap between theory (assuming smooth optimization) and practice (discrete architecture decisions).

4. Limited scope of "multi-task":

	○ All tasks are still within NLU—no evaluation on truly diverse tasks like generation, translation, code, reasoning.

	○ "Multi-task" means multiple NLU benchmarks, not fundamentally different task types.

	○ Unclear if approach would work for more heterogeneous task mixtures.

### Questions
1. Cost-benefit analysis: Can you provide a comprehensive comparison including the NAS search time? 

2. Architecture transferability: If you find an optimal prefix architecture on GLUE, does it transfer to SuperGLUE or MMLU without re-searching? This would validate whether PrefixNAS discovers general structures vs. overfitting to each benchmark.

3. Simpler alternatives: Have you compared against LoRA with rank=96 (matching your total parameter budget)? Table 5 shows LoRA_r=96 achieves 75.2% while PEML gets 80.3%—but how much of this is from NAS search vs. just the combination?

4. Per-task analysis: Which types of tasks benefit most from prompt optimization? Are there tasks where LoRA alone is sufficient? This would provide insights into when PrefixNAS is worth the cost.

5. Failure analysis: Table 6 shows HellaSwag performance drops significantly (77.4% vs. 93.1%). Can you explain why PEML underperforms on this task? What characteristics cause failures?

### Soundness
3

### Presentation
2

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
The paper proposes a parameter-efficient multi-task framework that jointly optimizes LoRA weight updates with a differentiable PrefixNAS module. During training, LoRA and PrefixNAS are optimized in parallel; after training, LoRA is merged into the base model and only the learned prefix architecture is kept for inference, avoiding adapter switching. Evaluations on GLUE, SuperGLUE, MMLU, and commonsense benchmarks report average gains up to 6.67% (peaks up to 10.75%) over strong PEFT baselines (LoRA, AdaLoRA, MultiLoRA, C-Poly, MoE).

### Strengths
* Clear unified design: concurrent LoRA + PrefixNAS with a concrete training algorithm (Alg. 1). 
* Deployment efficiency: LoRA is merged; inference uses one prefix, reducing switching/VRAM overhead. 
* Broad empirical coverage with consistent improvements across multiple benchmarks.

### Weaknesses
* Differentiable–discrete gap: architecture is relaxed via soft weights but finalized with argmax selection (Eq. 7), lacking analysis of search-time gradient bias or stability after discretization. 

* Search cost & fairness: PrefixNAS + TPE requires non-trivial compute (e.g., 8×A100, hours per benchmark), raising risks of validation overfitting and unequal hyperparameter budgets vs. baselines. 

* Task sensitivity: results note variability (e.g., WSC), suggesting remaining brittleness in cross-task generalization despite average gains.

### Questions
* The PrefixNAS module relies on a continuous relaxation during search but finalizes the architecture via a discrete argmax operation (Eq. 7).How stable is the gradient-based optimization when transitioning from continuous to discrete architectures, and could a smoother relaxation (e.g., Gumbel-Softmax or straight-through estimators) yield more consistent convergence and better generalization?

* PEML’s joint LoRA + PrefixNAS optimization requires multi-GPU resources (up to 8 × A100 for each benchmark).How can the framework ensure fair comparison with lightweight PEFT baselines like LoRA or AdaLoRA, and could a two-stage or surrogate-based NAS reduce cost without sacrificing accuracy?

### Soundness
2

### Presentation
2

### Contribution
2
