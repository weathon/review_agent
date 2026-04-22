# The Unreasonable Effectiveness of Scaling Agents for Computer Use

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Computer-use agents (CUAs) hold promise for automating everyday digital tasks, but their unreliability and high variance hinder their application to long-horizon, complex tasks. We introduce Behavior Best-of-N (bBoN), a method that scales over agents by generating multiple rollouts and selecting among them using behavior narratives that describe the agents' rollout. It enables both wide exploration and principled trajectory selection, substantially improving robustness and success rates. On OSWorld, our bBoN scaling method establishes a new state of the art (SoTA) at 69.9\%, significantly outperforming prior methods and approaching 72\% human-level performance, with comprehensive ablations validating key design choices. 
We further demonstrate strong generalization results to different operating systems on WindowsAgentArena and AndroidWorld.
Crucially, our results highlight the unreasonable effectiveness of scaling CUAs, when you do it right: effective scaling requires structured trajectory understanding and selection, and bBoN provides a practical framework to achieve this.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces bBoN, a “wide‑scaling” paradigm for computer‑use agents that generates multiple trajectories in parallel and selects the best via narrative comparison, a concept well‑framed and clearly presented in the abstract. However, it provides little discussion of the substantial computational cost such scaling entails.

### Strengths
The paper tackles a central issue in computer‑use agents—how to scale behavioral selection effectively—which is timely and relevant as larger models are increasingly limited by interaction efficiency.

The proposed bBoN framework is clearly formulated, with intuitive motivation and a well‑explained pipeline from trajectory generation to narrative‑based comparison.

The experiments, especially the comparison in Figure 5, provide concrete evidence that bBoN outperforms both average‑rollout and simple WebJudge baselines, illustrating the benefit of cross‑trajectory evaluation.

### Weaknesses
Although bBoN achieves better performance through multiple rollouts and narrative evaluations, this improvement comes with a potentially huge computational overhead. The paper does not quantitatively analyze cost, efficiency, or fairness under equal‑budget conditions, leaving readers uncertain about the real‑world practicality of the method.

The main contribution—running many rollouts in parallel and selecting via pairwise narrative evaluation—follows a relatively straightforward ensemble logic: performance generally improves when sampling more trajectories and filtering them with a learned judge. While well‑executed, this approach feels incremental rather than conceptually innovative, relying on existing ideas of scaling and comparative evaluation.

The method relies on aggregating multiple rollouts into a single narrative summary for comparison. As the number of rollouts or the length of each trajectory increases, both token‑level and computational costs grow rapidly. Moreover, very long context windows may dilute key information and amplify model attention errors, leading to less reliable judgments. The paper does not analyze how accuracy or reasoning quality scales with context length, which raises concerns about robustness in large‑scale or long‑horizon tasks.

### Questions
How does the framework handle potential side effects or environment resets when multiple rollouts are executed repeatedly? Do many resets cost more?

Does the narrative summarization stage require multimodal input (e.g., multiple images per rollout)? If so, how is this handled given that some current LLM/VLM APIs only support single‑image inputs?

What is the additional inference or token cost introduced by the narrative‑comparison stage?

In Section 3.2, the paper describes the Behavior Best‑of‑N Judge as performing a multiple‑choice (MCQ) comparative evaluation over N candidate trajectories. However, the corresponding system prompt (Appendix) only supports pair‑wise comparison (“which one better completes the user request”), without any structure for handling N > 2 options. This discrepancy suggests that the actual implementation is Best‑of‑2 rather than the claimed Best‑of‑N？

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
4

### Summary
The paper introduces a test-time scaling framework for computer-use agents called Behavior Best-of-N (bBoN): it runs multiple full task trajectories in parallel and then selects the best one at the end, greatly improving reliability on long, brittle workflows. To make different candidates directly comparable, it compresses each step into a concise Behavior Narrative that records the action and resulting screen change, and uses a multi-way bBoN judge to pick the strongest trajectory rather than scoring runs in isolation. The authors also streamline the underlying agent (Agent S3) by removing heavy hierarchical planning and integrating a coding agent, so it can generate stronger candidates faster. Together, these components yield higher success rates and better cross-platform generalization (e.g., to Windows and Android) under practical step budgets.

### Strengths
- Running multiple full trajectories in parallel and selecting the best (bBoN) reduces single-run brittleness in long-horizon tasks and provides predictable gains as the number of candidates increases.
- Behavior Narratives distill each step into factual “action → screen change” summaries, enabling a multiway judge to compare candidates directly, more reliable and scalable than independent scoring or pairwise elimination.
- The streamlined base agent (Agent S3) produces stronger rollouts with lower overhead, and the full stack shows consistent success-rate improvements and positive cross-platform transfer (e.g., Windows/Android) under realistic step budgets.

### Weaknesses
- Missing implementation details for parallel rollouts：The paper claims Best-of-N by running multiple full trajectories concurrently but omits a reproducible execution recipe: no specification of multi-machine vs. single-host isolation, environment cloning/reset procedures (images/snapshots/templates), seeding and cross-run isolation (caches, network, account state), or resource-parity policies (CPU/GPU/IO). This gap undermines reproducibility and comparability.
- Best-of-N increases cost roughly with N, yet the paper does not report wall-clock time, CPU/GPU hours, energy, or cost-normalized success. There is no equal-compute baseline comparison (e.g., single/multi-sample methods under the same total budget), so observed gains may primarily reflect more sampling rather than a more efficient method.
- Gains may hinge on heterogeneous candidate pools (different backbones, prompts, temperatures). Without isolating diversity vs. pure sampling, it’s unclear whether bBoN helps when all candidates come from the same model/config.
- The method is demonstrated on tasks with clear success signals. It remains unclear how the judge handles open-ended goals, partial credit, or ambiguous outcomes where “best” is not binary.
- The work does not characterize marginal gains as N grows or provide guidance to select N under a fixed budget, making it hard to tune for real-time constraints.

### Questions
- When generating N parallel trajectories, what infrastructure do you use (multi-machine vs. single host with multiple VMs/containers), how are environments cloned/reset, and how are seeds and resource quotas set to ensure isolation and fairness across candidates? Could you release scripts/configurations to reproduce the same parallel conditions?
- Under equal total compute budgets, how do bBoN results compare to single-run or few-sample baselines, and what wall-clock/CPU-GPU hours and energy are reported?
- Do gains depend on heterogeneous candidate pools (different models/prompts/temperatures), or do they persist when all candidates come from the same model and config?
- Are there early-exit policies to stop clearly bad candidates mid-run, or adaptive N strategies by task difficulty to control latency and cost at test time?
- How does bBoN perform under UI/layout drift, app updates, ads/pop-ups, and network jitter, particularly when initial snapshots cannot perfectly normalize the environment?

### Soundness
3

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
This work proposes a method for improving the performance of computer-use agents by generating multiple rollouts (wide scaling), summarizing their behavior by generating facts per action using a VLM (Behavior Narrative Generator), and selecting the best one through comparative evaluation using a VLM (Behavior Best-of-N Judge). The proposed framework achieves state-of-the-art performance on OSWorld, a standard benchmark for computer-use agents, in success rate for long-horizon tasks.

### Strengths
- Experimental setup: Detailed experimental evaluation, including ablation studies, baselines, and datasets.

- Strong empirical performance: Achieves state-of-the-art success rate on OSWorld.

- Clear presentation: The method is well-described and motivated.

### Weaknesses
- Inefficiency: Naive wide scaling means running N times more rollouts, with no mechanism to prune unpromising rollouts early or summarize fewer behaviors.

- No learning or adaptation: The method relies on the performance of pre-trained LLMs and VLMs in a zero-shot setting.

- VLM dependency: The approach depends on pre-trained VLMs both for summarizing behavior and for selecting the best trajectory, which may compound errors.

### Questions
- Could you add in Table 2 the runtime for different BoN values? Is it expected to be exactly N× that of Agent S3?

- How many steps does an optimal agent typically need to solve OSWorld tasks? Why did you choose 50 and 100 steps? Could you report the statistics (e.g., mean, median, variance) for the number of steps of an optimal agent?

- What is the impact of visual augmentations for pointer interactions on the performance of the Behavior Narrative Generator? Could you provide further details on the visual augmentation method?

-  Have you tried generating facts for the entire behavior with a single prompt (i.e., a trajectory-level summary) instead of per-action summaries?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a Behavior Best-of-N (bBoN) method, which scales over agents by generating multiple rollouts and selecting the best trajectory among them for execution.

### Strengths
The paper explores the scaling over agents by a simple but effective best-of-N method, which largely improves the performance in benchmarks. 

The paper proposes an effective Behavior Judge method that could support the scaling, which is better than the WebJudge.

### Weaknesses
1.  The proposed method is not practical. It requires a simulator or virtual machine to perform multiple rollouts before real execution, which limits its applicability and can incur excessive computational costs during testing, especially when the N is set to a large number.


2. This is also based on a strong assumption that the environment's transition dynamics are not static and that the simulator could be exactly the same as the real environment. This is difficult to guarantee in the real world, as websites are not stationary and can always have pop-ups or ads [1]. Thus, although the proposed method can achieve better performance in benchmarks, it is actually overfitting to the benchmark, which cannot represent real-world performance. 

3. As one can access the simulator, it would be better to run the baseline the same N times and use the task success indicator to select the successful trial among the N trials. This allows for calculating the success rate for each task, which can serve as the performance upper bound of the best-of-N method.


[1] DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning.

### Questions
See the weakness.

### Soundness
1

### Presentation
2

### Contribution
2
