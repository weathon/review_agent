## Human Reviewer 1

### Summary
This paper introduces PLoRA, a system designed to accelerate the hyperparameter tuning process for LoRA by addressing hardware underutilization. The authors first provide an empirical study demonstrating the necessity of LoRA hyperparameter tuning and identifying that typical tuning jobs, which often use small batch sizes, lead to inefficient GPU usage. To solve this, PLoRA proposes packing multiple LoRA configurations into a single fine-tuning job. The system comprises an offline packing planner, which uses an optimization algorithm to schedule jobs, and an online execution engine equipped with custom GPU kernels for efficient computation of packed adapters. Experimental results show significant reductions in the overall tuning time (makespan) and improvements in training throughput.

### Strengths
1. The paper begins with a thorough empirical study (Section 2) that clearly establishes the problem's significance. By demonstrating that optimal LoRA hyperparameters are task- and model-specific and that fine-tuning often leads to hardware underutilization, the authors provide a compelling justification for their work.

2. PLoRA is a well-designed, end-to-end system that combines high-level scheduling with low-level kernel optimizations. The two-stage approach of an offline planner and an online execution engine is practical, and the development of custom packed LoRA kernels directly addresses the core performance bottleneck. The reported improvements in makespan and throughput are substantial and demonstrate the effectiveness of the proposed approach.

### Weaknesses
1. The proposed packing strategy assumes that all configurations within a job are trained for the same duration. This is incompatible with adaptive HPO algorithms like HyperBand or Asynchronous Successive Halving (ASHA), which rely on early termination of unpromising trials to improve efficiency.

2. The scheduler appears to assume that all LoRA configurations in the search space require the same number of training steps. In practice, different configurations (e.g., with different learning rates) may converge at different speeds, or a user might want to train them for different numbers of epochs.

3. The paper compares against "Min GPU" and "Max GPU" baselines, which represent simple, manual strategies. While reasonable, they do not represent more sophisticated scheduling heuristics. The contribution of the ILP-based planner could be better isolated by comparing it against a simpler, greedy packing algorithm.

4. The planner relies on a recursive algorithm (DTM) that calls an ILP solver. While the paper states the offline planning time is negligible for 120 configurations, this approach may not scale to scenarios with thousands of configurations, which are common in large-scale HPO.

5. The paper claims applicability to other parallelism strategies like FSDP and provides a formulation in the appendix. However, all experiments are conducted with Tensor Parallelism (TP). FSDP, particularly ZeRO-3, has fundamentally different memory and communication patterns that might complicate the packing of heterogeneous LoRA adapters.

6. The planner's decisions are based on a cost model that estimates throughput from the first few training iterations.

7. The custom CUDA kernels are tuned for specific GPU architectures (Ampere). This is a common practice for high-performance systems but limits portability and may require significant engineering effort for new hardware.

8. The main text presents a simplified throughput maximization problem (Eq. 1), while the appendix details a full makespan minimization MILP (Eq. 8). The connection and transition between these two formulations could be made clearer.

### Questions
1. How can the PLoRA framework accommodate adaptive HPO strategies that require early stopping? Would it be possible to dynamically dissolve a packed job or stop gradient updates for certain adapters within a pack if they are identified as unpromising?

2. How does the planner handle a search space where configurations have heterogeneous training-length requirements? Does the current model assume a fixed number of steps for all trials, and what would be the implications of relaxing this assumption?

3. The paper compares against "Min GPU" and "Max GPU" baselines, which represent simple, manual strategies. While reasonable, they do not represent more sophisticated scheduling heuristics. The contribution of the ILP-based planner could be better isolated by comparing it against a simpler, greedy packing algorithm.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper presents PLORA, a system designed to drastically reduce the time and computational cost of hyperparameter tuning for RA fine-tuning. Its core innovation is the "packing" of multiple, heterogeneous LoRA configurations into a single training job, thereby improving hardware utilization. The work also demonstrates the advantage of using small batch sizes in LoRA fine-tuning, which makes the motivation more plausible.

### Strengths
1. The approach that packs LoRA configurations to improve hardware utility in hyperparameter tuning makes sense to me.
2. The empirical speedup in training time is impressive.

### Weaknesses
1. Although the authors have claimed through several observations that LoRA fine-tuning sometimes benefits from small batch sizes or configurations, I believe the most straightforward way to prove PLoRA's efficiency is to compare with baselines such as using larger ranks or batch sizes to improve hardware utilization. A lack of direct comparison with baselines like this somehow makes me unsure about whether the proposed techniques are essential.
2. I am not clear about the total efficiency gain of PLoRA. It seems that hyperparameter tuning usually costs a small fraction of time for the total training procedure. Taking the complete training cost after hyperparameter selection into consideration, I'm afraid PLoRA's efficiency gain in hyperparameter selection phase can be ignored. Maybe I'm wrong, but I expect to see clearer explanations on this issue.

### Questions
1. Can the authors give more explanations on the overall gain of PLoRA when considering the total training procedure, including both hyperparameter tuning and normal training?
2. There has been a lot of LoRA variants in the literature that can have better fine-tuning performance. I wonder if the core mechanism of PLoRA can also be extended to some successful LoRA variants?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 3

### Summary
The paper presents PLoRA, a system for efficient hyperparameter tuning of Low-Rank Adaptation (LoRA) for large language models (LLMs). Unlike prior work focusing on multi-LoRA inference serving, PLORA tackles inefficiencies in LoRA training, especially during hyperparameter sweeps. The key idea is to pack multiple LoRA adapters with distinct hyperparameter settings into a single fine-tuning job, leveraging shared frozen base models to improve GPU utilization.

### Strengths
1. Novel problem framing.
The work identifies an under-explored inefficiency: hyperparameter tuning for LoRA adapters, which has received little systems-level attention compared to LoRA inference. The idea of intra-run concurrent training for multiple LoRAs is well-motivated.

2. Solid systems contribution.
PLORA provides a principled optimization formulation (NP-complete but approximated with an ILP-based algorithm) and a modular architecture with a packing planner, execution engine, and GPU kernels.

3. Substantial empirical gains.
The results are clear and significant: up to 7× shorter tuning time and 12× throughput improvement without quality loss across diverse models and tasks (MRPC, CoLA, GSM8K, WNLI). The experiments are thorough, covering hardware setups, baselines (Min/Max GPU), and sensitivity analyses.

4. Insightful empirical study.
Before introducing PLORA, the authors systematically show that LoRA hyperparameters (rank, α, batch size, LR) have strong and task-dependent effects, justifying the need for efficient tuning

5. Clarity and completeness.
The system design is clearly illustrated with well-labeled figures and pseudocode. The appendices further include a detailed cost model and memory constraints.

### Weaknesses
1. Limited comparison to existing hyperparameter tuning frameworks.
Although the authors position PLORA as orthogonal to Bayesian optimization and other search strategies, some empirical comparison could be helpful.

2. Fairness of baselines.
The “Min GPU” and “Max GPU” baselines are simple but may not represent the state of the art in distributed hyperparameter tuning. Showing how PLORA interacts with these would strengthen the systems claim.

3. Evaluation scope.
The experiments, though extensive, focus only on medium-scale models (up to 32B). It remains unclear how PLORA scales to >70B or multi-node settings, especially given communication overheads in packed LoRA kernels.

4. Theoretical clarity.
While the optimization formulation is detailed, the paper would benefit from a concise intuitive explanation of the trade-offs (e.g., how packing degree affects convergence noise or training interference). Currently, it’s heavy on equations but light on intuition.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3