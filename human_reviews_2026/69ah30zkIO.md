# Adaptive Budget Allocation for Orthogonal-Subspace Adapter Tuning in LLMs Continual Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Large language models (LLMs) often suffer from catastrophic forgetting in continual learning (CL) scenarios, where performance on previously learned tasks degrades severely while training on sequentially arriving tasks. 
Although pioneering CL approaches using orthogonal subspaces can mitigate task interference, they typically employ fixed budget allocation, neglecting the varying complexity across tasks and layers. 
Besides, recent budget-adaptive tuning methods for LLMs often adopt multi-stage paradigms that decouple optimization and budget allocation. Such decoupling results in potential misalignment, which hinders those approaches' practical application in CL scenarios. 
To address these limitations, we propose OA-Adapter, a novel parameter-efficient approach for continual learning in LLMs that unifies dynamic budget adaptation with orthogonal subspace learning in an end-to-end training stage. 
Specifically, OA-Adapter introduces a dynamic bottleneck dimension adaptation mechanism that simultaneously allocates an efficient parameter budget and optimizes task objectives without misalignment.
To effectively preserve previously acquired knowledge while coordinating with the dynamic budget allocation, orthogonal constraints are applied specifically between the parameter subspace of the current task and the dynamically allocated parameter subspaces of historical tasks. 
Experimental results on continual learning benchmarks demonstrate that OA-Adapter outperforms state-of-the-art methods in both accuracy and parameter efficiency. OA-Adapter achieves higher average accuracy while using $58.5\\%$ fewer parameters on the standard CL benchmark, and maintains its advantages on two larger benchmarks comprising 15 tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes OA-Adapter, a CL method for LLMs that unifies dynamic budget allocation and orthogonal-subspace constraints across tasks in a single end-to-end training stage. Concretely, each adapter uses $W_2\,\Gamma\,W_1$ with $\gamma_i=\mathrm{soft}(g_i;\tau)$, enabling bidirectional activation of latent dimensions during training. Inter-task orthogonality is enforced on the column space of $W_2$ with respect to the activated dimensions from prior tasks, reducing interference without task IDs. Empirically, on three CL benchmarks with T5 and LLaMA-7B backbones, OA-Adapter improves average accuracy over strong baselines while using 46–59% fewer parameters. Ablations indicate both orthogonality and adaptive budget contribute, and analysis shows heterogeneous budget needs across layers and tasks.

### Strengths
1. OA-Adapter proposes a joint budget adaptation and orthogonal subspace learning scheme, eliminating the need for multi-stage or decoupled optimization seen in prior adaptive PEFT methods. This unified treatment minimizes misalignment between objective optimization and parameter budget allocation.
2. Single-stage joint optimization avoids misalignment of multi-stage budget tuning. Learnable $\tau$ is a simple, effective control knob.
3. OA-Adapter demonstrates consistent improvements in AA, BWT, and FWT on multiple benchmarks. The method uses up to 58.5% fewer parameters than O-LoRA while achieving higher accuracy.

### Weaknesses
1. While O-LoRA, ProgPrompt, L2P, LFPT5, Replay are covered, budget-adaptive PEFT baselines such as AdaLoRA, ALoRA, ElaLoRA, DiffoRA are not compared in a CL setting.
2. $\lambda_{\text{orth}}$, $r_{\max}$ and initial $\tau$ per layer might be suboptimal versus per-head or per-row thresholds; sensitivity analysis is limited.
3. While parameter count and GPU memory (Appendix A.8) are reported, the per-step overhead imposed by the orthogonality term is nontrivial. The implications for scaling to longer task sequences or larger models are not deeply discussed. Similarly, there is no analysis of the practical selection of bottleneck max dimension or how threshold dynamics interact on long sequence benchmark.
4. The paper argues multi-stage methods are costly, but does not provide the actual elapsed time or FLOPs memory comparisons vs baselines.
5. Many tables report averages across task orders but lack standard deviations, confidence intervals, or significance testing, which matters given CL variance.

### Questions
1. The orthogonality constraint is only imposed on up-projection columns corresponding to activated dimensions. Could this leave the method vulnerable to information leakage across temporarily deactivated subspace slots, or suboptimal alignment when dimensions are reactivated?
2. Since $\partial \gamma_i/\partial g_i=0$ when $|g_i|\le\tau$, learning relies on $\partial L/\partial \tau$. Did you observe dead dimensions that never re-activate?
3. Although Figures 2–5 visualize dimension adaptation, a more detailed analysis would strengthen the argument: how often do deactivated dimensions reactivate? Are there oscillations? How are learned thresholds distributed?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes OA-Adapter, a parameter-efficient continual learning (CL) method for large language models that jointly learns task-specific orthogonal subspaces and dynamically allocates parameter budgets in an end-to-end single-stage training loop.
A soft-thresholding mechanism on a learnable mask controls the effective rank of each layer’s adapter, allowing the model to (i) suppress interference across tasks, (ii) spend more capacity on difficult tasks/layers, and (iii) avoid the multi-stage optimization/budget-misalignment issues of prior budget-adaptive PEFT methods. Extensive experiments on standard 8-task and larger 15-task CL benchmarks show higher average accuracy than state-of-the-art baselines while using ≈58% fewer adapter parameters.

### Strengths
This is the first work that unifies learnable budget allocation and orthogonal-subspace updates inside a single training stage for LLM continual learning. The soft-threshold mask with a learnable $τ$ is technically simple yet novel, enabling bidirectional (expand/shrink) capacity adjustment that prior multi-stage methods cannot perform.

### Weaknesses
1.  Orthogonality is enforced via a simple cosine-penalty between current and previous subspace bases; no discussion of how this relates to classical projection-based CL guarantees or how the soft mask interacts with the orthogonality constraint from an optimization-theory viewpoint.
2. All experiments stop at 15 tasks. Because the sum of previously frozen orthogonal subspaces implicitly reduces the available rank, it is unclear whether the method will collapse when T≫50. A curve showing accuracy vs. number of tasks (up to the point where ractive→0) is missing.
3. For every layer and every task the full left/right bases must be kept to enforce orthogonality. For 15 tasks and 1 B-param LLMs this is still small, but memory footprint analysis (GB vs. #tasks) is not provided; the claim of “parameter efficiency” counts only trainable parameters, not total storage.
4. While Figures 4–5 show final dimensions, no quantitative analysis links task difficulty metrics (gradient norm, Fisher, forgetting score) to the learned rank, leaving the intuition that “hard tasks get more parameters” largely qualitative.

### Questions
1. How does the soft-threshold mask behave when the remaining rank approaches zero? Could τ become negative or get stuck, and do you have any gradient-tricks to revive dead dimensions?  
2. What is the total storage cost (bases + masks) after 15 tasks for T5-XL in GB? How does it scale with T?  
3. Have you tried non-orthogonal but still decomposed adapters (e.g., with spectral norm regularization) to see whether orthogonality is the key, or whether the dynamic budget alone suffices?  
4. Can the method handle task-free or time-varying domains where task boundaries are unavailable? If not, what would be the minimal change (e.g., sliding window, change-detection) to make it work?  
5. Why not compare with LoRA-GA/DoRA-CL which also adapt rank in a single stage? Please provide numbers or explain the mismatch.

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
5

### Summary
To address misalignment caused by budget-adaptive optimization in continual learning, this paper proposes OA-Adapter, a parameter-efficient method, to unify dynamic budget allocation with orthogonal subspace learning. OA-Adapter utilizes a dynamic bottleneck dimension adaptation to simultaneously allocate budgets and optimize objectives without misalignment. Experimental results show that OA-Adapter effectively reduces the budgets as well as achieves better performance.

### Strengths
1. OA-Adapter addresses dynamic budget allocation in continual learning, not to constrain all layers to use the fixed rank.

2. The paper is well written.

### Weaknesses
1. The paper does not discuss the motivation of applying the parameter-efficient method both after feed-forward and multi-head attention, which is different from other LoRA methods, which are only applied in multi-head attention. But there is no clear discussion of this difference.

2. The paper lacks a strong theoretical analysis to reflect how orthogonal parameter subspace constraints affect the final loss. For example, the paper can analyze the training loss reduction or forgetting error in continual learning.

3. Experiments are not sufficient. For example, it lacks recent LoRA-based baselines in continual learning, such as InfLoRA [1]. Also, comparison of performance with other baselines, such as O-LoRA, is not under the same parameter budget setting.

[1] InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning (CVPR2024)

### Questions
1. OA-Adapter is constructed after the feed-forward module and multi-head attention, but other LoRA methods are constructed in the multi-head attention. Can authors explain the benefits of this design? If removing the OA-Adapter after feed-forward, will the performance be dropped?

2. Why is the performance in Table 1 of SuperNI benchmark across baselines much lower than the results in SAPT [1]? Also, why do not compare the performance of SAPT and InfLoRA [2] in the experiments? Can authors compare OA-Adapter with these two works in experiments and analyze the difference?

[1] SAPT: A Shared Attention Framework for Parameter-Efficient Continual Learning of Large Language Models (ACL2024)

[2] InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning (CVPR2024)

3. What models are used in Table 2 and Table 3? Why do authors compare OA-Adapter only with O-LoRA? Can authors explain this reason? Also, the results of Table 3 come from which task order?

4. Table 3 shows the performance of O-LoRA drops when decreasing rank, however, in the original paper of O-LoRA, it shows that there is no linear relationship between rank and accuracy. Can authors explain the difference? Also, for comparing the performance between O-LoRA and OA-Adapter, they should be under the same budget to compare. For example, the initial rank of 16 for OA-Adapter would drop to 9.95, and use a fixed rank of 9 or 10 for O-LoRA to compare rather than using the rank of 16.

5. What’s the experimental setting for Figure 3? It seems like authors only show the forgetting issue and mitigation under normal sequential training rather than using OA-Adapter. Can authors show the comparison without using OA-Adapter and with using OA-Adapter for showing the occurrence and mitigation in Figure 3?

6. Table 1 shows that OA-Adapter has poor performance in FWT metric. Can authors explain the reason? The results seem to show that OA-Adapter does not help knowledge transfer from previous tasks to new tasks.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes OA-Adapter (Orthogonal-subspace Adapter with Adaptive Budget), a parameter-efficient framework that unifies dynamic budget allocation and orthogonal subspace learning in an end-to-end training stage. Experiments on T5-Large and LLaMA-7B show OA-Adapter outperforms baselines.

### Strengths
1. The learnable threshold (τ) allows bidirectional dimension adjustment (activation/deactivation), adapting to task complexity and layer needs.
 
2. Orthogonal constraints, applied to dynamically allocated subspaces, effectively reduce cross-task interference.

### Weaknesses
1. The learnable threshold (τ) is critical for budget adaptation, but the paper provides no analysis of how its initial value or update dynamics affect performance. 

2. While OA-Adapter is parameter-efficient, the orthogonality regularization introduces computational overhead that grows linearly with task count.  

3. Only evaluates classification tasks (sentiment, topic, NLI, QA). It does not test complex tasks like open-ended generation (summarization, dialogue) or reasoning (math, logic), where token-level frequency cues may behave differently.

4. Only tested on T5 and LLaMA-7B. It is unknown if OA-Adapter works for newer models (e.g., LLaMA3-8B, Qwen3-8B).

### Questions
1. How does the initial value of the learnable threshold (τ) affect budget adaptation? 

2. For very long task sequences (50+ tasks), how does the linear growth of orthogonality regularization overhead impact training time and memory? 

3. How does OA-Adapter perform when tasks have high similarity (e.g., multiple NLI tasks)? Does its dynamic budget allocation automatically reduce orthogonality (to enable more transfer) or maintain strict constraints (wasting transfer potential), and can the orthogonality regularization be modulated by task similarity?

### Soundness
3

### Presentation
2

### Contribution
3
