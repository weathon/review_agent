# Learning on the Job: An Experience-Driven Self-Evolving Agent for Long-Horizon Tasks

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Large Language Models have demonstrated remarkable capabilities across diverse domains, yet significant challenges persist when deploying them as AI agents for real-world long-horizon tasks. Existing LLM agents suffer from a critical limitation: they are test-time static and cannot learn from experience, lacking the ability to accumulate knowledge and continuously improve on the job. To address this challenge, we propose MUSE, a novel agent framework that introduces an experience-driven, self-evolving system centered around a hierarchical Memory Module. MUSE organizes diverse levels of experience and leverages them to plan and execute long-horizon tasks across multiple applications. After each sub-task execution, the agent autonomously reflects on its trajectory, converting the raw trajectory into structured experience and integrating it back into the Memory Module. This mechanism enables the agent to evolve beyond its static pretrained parameters, fostering continuous learning and self-evolution. We evaluate MUSE on the long-horizon productivity benchmark TAC. It achieves new SOTA performance by a significant margin using only a lightweight Gemini-2.5 Flash model. Sufficient Experiments demonstrate that as the agent autonomously accumulates experience, it exhibits increasingly superior task completion capabilities, as well as robust continuous learning and self-evolution capabilities. Moreover, the accumulated experience from MUSE exhibits strong generalization properties, enabling zero-shot improvement on new tasks. MUSE establishes a new paradigm for AI agents capable of real-world productivity task automation.
Demo videos can be found in our supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose MUSE, a novel self-evolving AI agent framework that enables LLMs to learn continuously from experience on real-world long-horizon tasks. It features a hierarchical memory module where the agent autonomously reflects on trajectories to create and integrate structured experience. This mechanism drives self-evolution, leading to new state-of-the-art performance on the TAC benchmark and demonstrating strong generalization to new tasks.

### Strengths
1) The experimental results demonstrate strong performance, achieving SOTA results compared to recent LLMs.
2) The work is well-motivated by the need for LLM agents to continuously learn from experience and overcome their static nature.

### Weaknesses
1) I wonder whether the performance gains are primarily derived from leveraging previous successful solutions. It is crucial to explore the framework's ability to learn from and generalize over failures. How about provide more harder questions in the initial and then easier questions, and will the model benifit from initial hard failure samples?
2) A primary concern for the framework is the computational overhead during inference. Can the authors provide inference time computation on the questions compared to without memory, and without MUSE?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

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
MUSE proposes an agent framework with 3 modules : strategic, procedural, tool steps which refine knowledge in a predefined operation standard ( plan-execute-reflect-memorize ) scaffolding loop. When using MUSE on TheAgentCompany, it improves by over 20% from prior SOTA using Gemini-2.5-flash model.

### Strengths
MUSE is the first method to exceed 50% on TAC with a test time continuous learning module. The 3-tier structure (Strategic/Procedural/Tool) is intuitive and well-motivated, providing different levels of abstraction.

### Weaknesses
The paper doesn't adequately explain the details of the scaffolding specifically : How memory is deduplicated and pruned at scale? What happens when memories conflict? Computational cost of memory retrieval as size grows? ( These could be resolved with the inclusion of supplementary materials but its not provided )

The experiment setting is a bit weak with only one single benchmark ( the agent company ) and lacks the commonly used benchmarks such as AppWorld, OSWorld or even simpler task such as SWE-bench.

But more critically, the framework is not different enough from similar works which learns from trajectories ( ExpeL, Memp, Agent Workflow Memory), making the contributions too weak to justify its novelty.

### Questions
Major:

1. Would MUSE be applicable to SWE-bench, OSWorld, WebArena or AFLWorld as well?

2. Could you provide measurements of latency, token costs, or context growth over time? Given memory updates after every sub-task, computational overhead could be prohibitive

3. How sensitive is performance to the quality of the Reflect Agent? What if it extracts poor memories ( by using smaller models on reflect agent)?

4. What’re the error failure case where MUSE still fails to perform successfully? For example in the original TAC benchmark, the author identifies lack of social skills, browsing, self-confusion are one of the 3 major causes of error.

Minor:

5. How long does it take to craft the prompts for each of the modules? Does the time needed to craft each of the module prompts to adapt to new LLMs or tasks hinders the adoption of MUSE?

6. Is there a solid reason why the supplementary material is not provided? Without the code it is impossible to review if the evaluation does not contain any information leak from ground truth during the iterative improvement process.

7. In the paper, it claims MUSE is model-agnostic memory, but were the memories actually tested when transferred to significantly different model families ( deepseek-3.1 -> gemini 2.5 flash )?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work creates an framework that allows a model to learn from experience for long-horizon tasks using a memory module. They achieve state-of-the-art performance on the TAC benchmark.

### Strengths
- Addresses timely concern on models being static and not learning as they perform the task
- Achieves new SOTA on TAC benchmark
- Demonstrates that memory module is memory-agnostic and can then be plugged into different models

### Weaknesses
- Only use of one domain; while i understand that they need a difficult domain with a long horizon, it would be important to see what how this agent would perform on other problems
- The questionable choice of baselines, they all use different base models and none of the works in the related works were experimented with
- It is important to discuss the time completing tasks and the time it takes to train

### Questions
Figure 3 is hard to read

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes MUSE, an experience-driven agent framework with a hierarchical memory module for self-evolution in long-horizon tasks.

### Strengths
MUSE achieves a new SOTA on the TAC benchmark using a lightweight model, demonstrating effective continuous learning and generalization.

### Weaknesses
1. The limitations of existing methods or the motivation of this paper are not entirely accurate. The limitations of existing methods mentioned in the abstract—"they are test-time static and cannot learn from experience, lacking the ability to accumulate knowledge and continuously improve on the job"—can be addressed by an RL-based LLM. 
2. The experimental comparisons are incomplete and unfair.

### Questions
1. The limitations of existing methods mentioned in the abstract—"they are test-time static and cannot learn from experience, lacking the ability to accumulate knowledge and continuously improve on the job"—can be addressed by LLM based on RL. However, the limitations of existing methods or the motivation in this paper are not entirely accurate.
2. For experience-driven self-evolving agents, how large is the amount of accumulated experience data? What is the scale of structured experience? How can we ensure that this experience is not forgotten during task switching?
3. Many self-evolving agents have not been compared. You can search for surveys of Self-Evolving Agents.
4. Why was only a subset of 18 tasks used for continuous learning experiments, and how does this selection bias affect the claim of general self-evolution capability?
5. How does the hierarchical memory structure ensure that retrieved experiences are contextually relevant and do not introduce noise or outdated strategies?
6. Can you justify the fairness of comparing MUSE using Gemini-2.5 Flash against stronger models in baseline methods without ablation on model capacity?

### Soundness
2

### Presentation
2

### Contribution
2
