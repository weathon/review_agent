# SWIRL: A Staged Workflow for Interleaved Reinforcement Learning in Mobile GUI Control

- Avg Score: 3.60
- Decision: Reject
- Scores: 2, 6, 2, 4, 4

## Abstract
The rapid advancement of large vision language models (LVLMs) and agent systems has heightened interest in mobile GUI agents that can reliably translate natural language into interface operations. Existing single-agent approaches, however, remain limited by structural constraints. Although multi-agent systems naturally decouple different competencies, recent progress in multi-agent reinforcement learning (MARL) has often been hindered by inefficiency and remains incompatible with current LVLM architectures. To address these challenges, we introduce SWIRL, a staged workflow for interleaved reinforcement learning designed for multi-agent systems. SWIRL instantiates a round-level training scheme that treats MARL as a sequence of single-agent reinforcement learning subproblems, updating one agent at a time while keeping the others fixed. This formulation enables stable training and promotes efficient coordination across agents. Theoretically, we provide a stepwise safety bound, a cross-round monotonic improvement theorem, and convergence guarantees on return, ensuring robust and principled optimization. In application to mobile GUI control, SWIRL instantiates a Navigator that converts language and screen context into structured plans, and an Interactor that grounds these plans into executable atomic actions. Extensive experiments demonstrate superior performance on both high-level and low-level GUI benchmarks. Beyond GUI tasks, SWIRL also demonstrates strong capability in multi-agent mathematical reasoning, underscoring its potential as a general framework for developing efficient and robust multi-agent systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SWIRL, a novel RL framework for multi-agent systems.
Specifically, SWIRL conducts training of every single agent sequentially, thus stabilising the entire training process of the multi-agent system.
The authors provide a theoretical analysis for step-wise safety, monotonic improvement, and convergence of SWIRL.
Empirical results on GUI tasks and mathematical reasoning tasks demonstrate the superior performance of SWIRL.

### Strengths
1. Different from step-level alternation [1], SWIRL present a novel round-level alternation, sequentializing single-agent updates while maintaining multi-agent coordination.
2. Presenting theoretical results based on [1].
3. Extensive experiments on different benchmarks, including GUI tasks and math reasoning tasks.



>Reference
>>[1] Heterogeneous-agent reinforcement learning.

### Weaknesses
1. First concern is the unclear structure, presentation and writing, especially in Section 2. The main contribution is the round-level alternation, while section 2.1 (Lines 59-200) is unclear in illustrating the underlying motivation. Additionally, the theoretical results are based on previous work [1]; it is important to illustrate clearly how to adapt from the previous one to address the challenges in this work (e.g., from step-level to round-level).
2. For GUI tasks, a bunch of related works are missing, including [2-7].
3. Limited novelty. The navigator-interactor pipeline is a common approach in the literature. For example, existing work [6] leverages VLM as the subgoal-generator to provide low-level instructions for the low-level agent.


>Reference
>>
>>[1] Heterogeneous-agent reinforcement learning
>>
>>[2] You only look at screens: Multimodal chain-of-action agents
>>
>>[3] Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning
>>
>>[4] Digi-q: Learning q-value functions for training device-control agents
>>
>>[5] Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning
>>
>>[6] Vsc-rl: Advancing autonomous vision-language agents with variational subgoal-conditioned reinforcement learning
>>
>>[7] Distrl: An asynchronous distributed reinforcement learning framework for on-device control agents

### Questions
See Weaknesses.
Additionally, it would be great if more GUI experiments, comparing with existing baselines [1-6] on the AitW benchmark [7].


>Reference
>>
>>[1] You only look at screens: Multimodal chain-of-action agents
>>
>>[2] Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning
>>
>>[3] Digi-q: Learning q-value functions for training device-control agents
>>
>>[4] Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning
>>
>>[5] Vsc-rl: Advancing autonomous vision-language agents with variational subgoal-conditioned reinforcement learning
>>
>>[6] Distrl: An asynchronous distributed reinforcement learning framework for on-device control agents
>>
>>[7] A Large-Scale Dataset for Android Device Control

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SWIRL, a staged, interleaved multi-agent reinforcement learning (MARL) workflow that decomposes multi-agent training into sequential single-agent updates. It presents formal convergence and safety proofs for interleaved updates, empirical validation showing superior zero-shot performance and robustness, and transferability to mathematical reasoning tasks.

### Strengths
1. This paper gives a clever recombination of MARL, alternating optimization, and LVLM integration, executed with clear theoretical rigor and practical benefit.
2. Experiments are diverse, spanning GUI control (high-level and low-level) and math reasoning. Baselines include major prior systems (GUI-R1, MARFT, OS-Atlas, ReMA), with clear evidence of superior performance and resource efficiency (O(1) memory scaling).

### Weaknesses
1. As a critical stability factor in MARL, interleaving frequency determines how quickly agents co-adapt. This paper does not empirically investigate how interleaving frequency or synchronization cadence affects performance, convergence, or stability.
2. Although SWIRL is compared against OS-Atlas, GUI-R1, and MARFT, it omits recent multi-agent finetuning frameworks such as MultiAgent Debate (Du et al., 2023), AutoGen (Wu et al., 2024a), and Rema (Wan et al., 2025) in full experimental benchmarking. Comparing against them would help position SWIRL within the broader multi-agent LVLM training ecosystem.
3. The paper emphasizes efficiency (O(1) actor memory), but does not quantify training sample efficiency or scaling behavior. The total training set (3,500 samples) is small, yet the authors do not clarify if this is sufficient for convergence or merely a constraint of resources.
4. SWIRL’s performance may be conflated with the capabilities of the underlying Qwen2.5-VL-3B models. The absence of experiments with smaller or weaker backbones makes it difficult to isolate what proportion of improvement derives from the interleaved RL algorithm itself.
5. The mathematical reasoning experiments (Section 3.4) show numerical improvements but do not explain why interleaving helps reasoning tasks or provide qualitative examples of improvement.

### Questions
See the Weaknesses Section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SWIRL, a method that frames a MARL problem but leverages single-agent algorithms through sequential freezing and interleaving of agents during training. Agents included are Navigator and Interactor, dealing with UI screenshots and user requests as well as low level instructions from the Navigator and UI screenshots to derive actions (Interactor). Authors claim for this setup to outperform SOTA models such as GPT5, R1 etc.

### Strengths
- Interesting idea
- Good experiments on replacing the Navigator with SOTA models such as GPT5 etc. to get  Interactor performance

### Weaknesses
- It is not clear to me why this is formulated as a multi-agent problem, as the setup is stated as a "...sequential single-agent... enabling reuse of standard RL..."

- I believe the main weakness of the paper is the experiment design to answer the question above. In the experiment, I would have expected the following experiments:
1. End-to-end RL training of Navigator and Interactor, potentially sharing rewards
2. Isolated training of the Navigator until convergence and then isolated training of the Interactor (with frozen, well-performing Navigator)
3. Interleafed Navigator / Interactor (SWIRL, yours) training

This is in combination with my main concern: The Indicator is clearly dependent on the performance of the Navigator. If the Navigator performs badly, the Interactor has no chance of performing well.

- I don't see clear conclusion and discussion why the interleafed setup performs better than all other setups.

### Questions
Why are you reporting performance on pre- and post-training language benchmarks (GSM8K, MATH500 etc.) when the paper is looking at vision-language models?

I would be interested in the isolated performance of the Navigator and the Interactor. Why did you decide not to show that?

### Soundness
3

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
2

### Summary
This paper introduces SWIRL, a framework for reformulating MARL as a sequence of single-agent tasks, therefore sidestepping many of the known challenges, such as stability. The authors provide theoretical guarantees including per-step safety bounds, monotonic improvement across rounds, and convergence properties. In particular, this paper focuses on improving the capabilities of mobile GUI agents, while also testing on mathematical benchmarks. The paper achieves very strong empirical results.

### Strengths
**Impressive empirical results** - SWIRL outperforms or achieves near state-of-the-art in a variety of benchmarks. This is a major strength of the paper.

**Problem is important** - Mobile GUI control is a useful area of research, so I believe the paper is well-motivated and will be of use to the community.

**Transfer beyond GUI** - I appreciate that this paper extends its results to more than one benchmark, namely mathematical benchmarks. Validity of the work is improved by showing results are not just closely tied to one benchmark.

**Source Code** - The source code being included in the supplementary material improves the validity and reproducibility of this work.

### Weaknesses
- **Support for $>2$ agents** - While the proposed SWIRL framework is reported to support N different agents, all experiments focus on settings with only two agents. Do you have any results demonstrating a setting with more than 2 agents?

- **Walltime** - Currently, the paper does not disclose any information about the wall time of this method. I would imagine that the alternating nature of the approach may increase this substantially. 

- **Limitations Section?** - There appears to be no limitations section in this paper. This is a considerable red flag, as this is almost always an important section of research papers.

- **Wording of Claims** - From the wording of this paper (such as line 20), it seems as though the idea of interleaved training with fixed agents is novel. While the authors acknowledge that this has been done before (via citing numerous prior works), I still feel that the authors should be careful not to overreach in terms of their claims.

### Questions
- How does this approach work when scaling to more than 2 agents?

- What impact does this approach have on walltime?

- Where is the limitations section of this paper? What are the limitations?

I will cautiously provide a score of 4, given that this is not my primary area of expertise; however, I will raise my score if my concerns are addressed.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
SWIRL splits a GUI agent into a Navigator (plans) and an Interactor (executes) and trains them alternately, freeze one, optimize the other, so multi-agent training reduces to standard trust-region single-agent updates. This staged setup targets stability/efficiency and shows gains on mobile GUI control, with spillover benefits to math reasoning.

### Strengths
Clear role decomposition (auditable plan to act chain). Practical pipeline that reuses mature single-agent RL tooling. Theoretical footing (safety bound, round-wise monotonic improvement). Solid empirical wins for low-level execution and planner replacement. Transferable beyond GUI with modest data.

### Weaknesses
Training always freezes one agent (Navigator or Interactor) and optimizes the other, effectively decomposing the multi-agent problem into a sequence of single-agent GRPO/PPO/TRPO steps. This is closer to modular alternating optimization than classic concurrent cooperative MARL. As a result, non-stationarity and coordination that arise when both agents evolve simultaneously are not directly learned.

The GUI evaluation relies on offline single-step accuracy (Type/GR/SR)—e.g., a click landing in the box counts as correct, without reporting episode success rate, number of clicks, latency, or recovery/rollback behavior. High offline SR does not guarantee closed-loop, end-to-end performance.

The paper removes certain QA-style samples from GUI-Act-Web and evaluates on an older GUIOdyssey Test-Random split due to version changes. While this sharpens focus on manipulation skills, it alters the test distribution and complicates apples-to-apples comparison with prior work.

The reported stability stems from “alternating updates + online reweighting,” and ablations (e.g., removing reweighting) show clear degradation. This indicates robustness depends heavily on sample filtering and the training schedule, rather than fundamentally resolving non-stationarity under concurrent cooperative learning.

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3
