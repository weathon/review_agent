# LightAgent: Lightweight and Cost-Efficient Mobile Agents

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
With the advancement of multimodal large language models (MLLMs), building GUI agent systems has become an increasingly promising direction—especially for mobile platforms, given their rich app ecosystems and intuitive touch interactions. Yet mobile GUI agents face a critical dilemma: truly on-device models (4B or smaller) lack sufficient performance, while capable models (starting from 7B) are either too large for mobile deployment or prohibitively costly (e.g., cloud-only closed-source MLLMs). To resolve this, we propose LightAgent, a mobile GUI agent system that leverages device-cloud collaboration to tap the cost-efficiency of on-device models and the high capability of cloud models, while avoiding their drawbacks. Specifically, LightAgent enhances Qwen2.5-VL-3B via two-stage SFT→GRPO training on synthetic GUI data for strong decision-making, integrates an efficient long-reasoning mechanism to utilize historical interactions under tight resources, and defaults to on-device execution—only escalating challenging subtasks to the cloud via real-time complexity assessment. Experiments on the online AndroidLab benchmark and diverse apps show LightAgent matches or nears larger models, with a significant reduction in cloud costs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LightAgent, a lightweight yet capable mobile GUI agent that uses device–cloud collaboration to balance performance and computational cost. The authors enhance a Qwen2.5-VL-3B model via a two-stage SFT and GRPO training pipeline on synthetic GUI reasoning data. They propose an efficient reasoning template with history summarization and introduce a dynamic switching mechanism that delivers complex subtasks to a cloud LLM only when needed. Experiments on the AndroidLab benchmark and real Android apps show that LightAgent achieves competitive performance while significantly reducing cloud API usage and latency.

### Strengths
1. Practical problem relevance. The paper addresses an important and rapidly growing problem: building mobile GUI agents that operate within strict computing budgets. Despite the paper’s limitations, the motivation and problem framing are valid and meaningful to the agent community.

2. Non-trivial engineering effort with a complete pipeline. This paper implements an end-to-end device-cloud collaborative agent system, including task complexity assessment and a dynamic switching mechanism. The evaluation on the AndroidLab benchmark is conducted in an online manner, making execution closer to real-world deployment.

### Weaknesses
1. Limited novelty; mainly a system integration effort. The paper primarily combines existing components including synthetic data generation, GRPO training, CoT-style reasoning templates, and device–cloud fallback. The claimed “device–cloud collaboration framework” is a system-level architecture rather than a methodological innovation, and similar fallback or hybrid execution has been explored before [1].
2. Heavy reliance on the cloud despite “edge-first” positioning. Despite being presented as an edge-prioritized system, LightAgent still requires cloud offloading for roughly 65% of steps according to its own analysis. This suggests that the 3B base model is still insufficiently capable as an autonomous agent and depends strongly on the cloud model to complete tasks. The paper does not analyze failure cases in offline or no-cloud settings. 
3. Lack of experimental clarity in early figures. Figure 4 reports a comparative performance plot between on-device and cloud models but provides no information regarding the dataset. It is unclear whether results are based on AndroidLab or another benchmark.
4. Experimental evaluation is narrow and insufficient. The paper only evaluates on AndroidLab and a small set of four Android apps, despite the existence of established GUI agent benchmarks. It ignores offline navigation benchmarks such as GUI-Odyssey and AndroidControl, and does not compare on the standard online benchmark AndroidWorld. This limited evaluation scope makes it unclear whether the method generalizes beyond the narrow AndroidLab setting, and misses critical comparisons that would strengthen claims regarding generality and effectiveness in both offline and online environments.
5. Limited real-device evaluation weakens “mobile agent” claim. All experiments are performed on GPUs (NVIDIA RTX 3090) rather than real mobile hardware. The paper does not provide inference latency, peak memory usage, or thermal behavior on actual smartphones. Without demonstrating real on-device deployment, it is premature to claim that the model is suitable for mobile usage.

[1] Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks.

### Questions
1. What is the size and nature of the synthetic dataset? The dataset design is claimed as a main contribution, but the paper does not report the dataset scale or other details. 
2. The system is evaluated only on Android GUI agents. Do the authors expect their approach to generalize to other platforms like Web, Windows desktop workflows, or iOS? If so, what components are platform-specific?

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
This paper introduces light agent for on-device GUI agent to help real-world deployment. To improve the performance, they introduce the on-device and cloud-based agent to collaborate. They introduce two-stage training for the on-device agent and a collaborate control framework to decide if switch to cloud-based agent. They carry out experiment on AndroidLab.

This paper tries to address an important problem for on-device deployment GUI agent, my concerns are mainly in (details see questions and weakness):
A.	The novelty: SFT and GRPO training of GUI agent, as well as their design of implementing these two modules, are not new.
B.	Experiment only on one dataset and lack important comparisons with other 3B-4B models. 
C.	The efficiency lack important evidence, especially the additional cost of cloud LLMs in monitoring the on-device agent .

### Strengths
1, The paper is clearly and logically structured, making the content highly accessible and easy to follow.
2, The integration of on-device and cloud LLMs represents a promising approach that could significantly accelerate the industrial deployment of GUI agents.
3, The author provided sufficient implementation details, including the specific prompts, which greatly facilitates the understanding of the individual modules.
4, The breakdown of step percentages across the on-device and cloud models is valuable, as it effectively quantifies the source of efficiency improvements.

### Weaknesses
1, Lack of evidence to demonstrate the necessity of finetuning on-device agent with the proposed two step strategies. A direct evidence would be comparing with other 3B models like Qwen2.5-VL-3B under w and w/o Cloud LLM setups.
2, Lack of discussion of the accuracy of the two modules in collaborative control frameworks. From A.4.2 and A.4.3, it seems priors on task complexity are provided in the prompts, but where do these priors come from and how to collect for other apps/tasks are not clear.
3, Additional Cloud-based agents are involved before the model is switched to cloud models in deciding/ monitoring the on-device agent.  But their cost seems not counted as part of the cloud cost, especially in Figure 6.
4, Experiment only carried out on one AndroidLab dataset, while others such as Android World[a] and offline datasets such as Android control[b] is not compared.
[a]AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents
[b] On the Effects of Data Scale on Computer Control Agents

### Questions
1, Could the author provide the additional cloud-agent costs associated with the two modules: task complexity determination and model monitoring? What is the average number of calls to the cloud agent within these two modules? This information would help in understanding the true cost by integrating the cloud-agent cost into Figure 6.
2, Could the author discuss the motivation for employing the two-step training process described in Section 2.1, particularly in comparison to other methods? Reasoning or history understanding is a shared strategy in current works (e.g., M3A[a], UI-Tars).
3, Could the authors provide the average number of steps required to complete tasks under the on-device only, cloud LLM only, and on-device with cloud LLM configurations/settings? The average number of steps is an important criterion for evaluating efficiency. How many cloud LLM calls are made in each setup?
4. Would the cloud LLMs over write the previous progress by some actions such as Home() and the effort of the on-device agent is ignored?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper designs a device-cloud collaboration framework to solve mobile
interaction tasks efficiently and effectively. The exhibited results are
promising. However, the paper is not perfectly polished, holds obvious symbol
inconsistency and typos.

### Strengths
1. This paper studies a valuable problem, device-cloud collaborative GUI agent,
  balancing the invocation cost and overhead with the execution performance.
2. The exhibited results look promising.

### Weaknesses
1. In the proposed execution flow, the active model will not switch back to the
   on-device model after switching to the cloud model. Why isn't a switch-back
   mechnism integrated?

### Questions
1. The monitoring starting step is denoted by $\tau$ on Line 211, but $\gamma$
   in Algorithm 1.
2. Symbols tau in Algorithm 1 is not used. Symbol T in Algorithm 1 is not
   introduced.
3. What model is used for task complexity assessment and dynamic orchestration
   policy?
4. What is $R_{acc}$ and $R_{fmt}$ in Equation 4?
5. How is $k$ computed in Equation 4?
6. The letter cases in Table 1 are not consistent.
7. The device-cloud model combinations in Table 1 and Figure 6(a) are not
   consistent? Why does this occur? Are the success rates of combinations like
   ours+GLM-4.5-V not satisfactory enough to demonstrate the validity of the
   proposed method? Is the step percentage of Gemini-2.5-Pro too high to be
   shown in Figure 6(a)?
8. What's the meaning of SN in Table 2?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper targets a very real bottleneck in mobile GUI agents: small on-device MLLM/VLMs (≈3B) can run locally but are too weak to finish real Android tasks, while cloud LLM/VLM calls are accurate but expensive and latency-sensitive. The goal is to push a 3B open model to be “good enough” for most steps, and only escalate to cloud on hard steps, so that mobile agents become practically deployable.

### Strengths
- Clear decomposition of cost: SFT+GRPO makes the 3B model more reliable; the scheduler makes cloud use predictable; the switch makes it robust.
- Data generation pipeline: using stronger models to auto-generate GUI episodes with CoT and tool-calls is sensible for this domain.

### Weaknesses
- No direct comparison to other 3B GUI-R1 / GUI-G1–style models, e.g., GUI-R1, "GUI-R1 : A Generalist R1-Style Vision-Language Action Model For GUI Agents"; GUI-G1, "GUI-G1: Understanding R1-Zero-Like Training for Visual Grounding in GUI Agents": the authors compare to larger/stronger or clouded agents on AndroidLab, but not to concurrent 3B R1-like GUI agents, so the “3B is competitive after GRPO” claim is only relative to the chosen baselines. This is the biggest missing experiment.
- Scheduler is rule/LLM–driven: good engineering, but not theoretically grounded; if app distributions shift, $\gamma/\omega$ may need re-tuning.
- Reliance on AndroidLab: results are shown on one environment; it would be stronger to show that the device–cloud policy transfers to more dynamic benchmarks (e.g., SPA-Bench).

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
1
