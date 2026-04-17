# OpenApps: Simulating Environment Variations to Measure UI Agent Reliability

- Decision: Accept (Oral)
- Scores: 6, 6, 8, 6

## Abstract
Reliability is key to realizing the promise of autonomous UI-agents, multimodal agents that directly interact with the apps humans use, as users must be able to trust an agent to complete a given task. Current evaluations rely on fixed environments---often clones of existing apps--- which are limited in that they can only shed light on whether or how often an agent can complete a task within a specific environment. When deployed however, agents are likely to encounter variations in app design and content that can affect an agent’s ability to complete a task. To address this blind spot of measuring agent reliability across app variations, we develop OpenApps, a light-weight open-source ecosystem with six apps (messenger, calendar, maps, etc.) that are configurable in appearance and content. OpenApps requires just a single CPU to run, enabling easy generation and deployment of thousands of versions of each app. Specifically, we run more than 10,000 independent evaluations to study reliability across seven leading multimodal agents. We find that while standard reliability within a fixed app is relatively stable, reliability can vary drastically when measured across app variations. Task success rates for many agents can fluctuate by more than 50\% across app variations. For example, Kimi-VL-3B's average success across all tasks fluctuates from 63\% to just 4\% across app versions. We also find agent behaviors such as looping or hallucinating actions can differ drastically depending on the environment configuration. These initial findings highlight the importance of measuring reliability along this new dimension of app variations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents a light-weight version of a benchmark for GUI agents, which presents high reproducibility and easy/cost-effective evaluations. Especially, the authors focused on evaluating the reliability of the agents across UI variants. Across six applications that are popularly employed in many routines, authors configure various combinations of UI diversities, constituting more than 10,000 independent cases. Based on the cases, the vulnerability of agents is revealed, and their qualitative analysis is presented.

### Strengths
On top of all, I believe that this work is highly notable because a light-weight benchmark for GUI agents is a missing yet demanding block in this community. I describe several other strengths of this work.
1. Important problem: the reliability of agents in practice is a highly crucial problem. I believe that this work represents an important step forward for the community.
2. Robust success detection logic: the authors put considerable effort into creating success detectors that can work around diverse UI variations. I believe that the efforts that the authors put into this part should be recognized, as this is known to be a challenging problem.
3. Extensive experiments: the volume of the experiments is notable. I believe that such a massive numbers of experiments make the results presented more reliable.
4. Interesting observations: the authors also present a detailed behavioral analysis. These observations highlight the shortcomings of current agents, enabling the design of future solutions.

### Weaknesses
I discuss the weaknesses of this paper, including questions and suggestions.
1. Mobile device control benchmarks: while the paper presents fruitful discussion on web benchmarks, discussion on mobile device control benchmarks [1,2,3,4] (which also overlaps with OS-level benchmarks) is absent. Notably, B-MoCA [2] and AndroidWorld [3] both tackle the robustness of the agents, where the former one demonstrates the feature of UI variations and degradation of agents’ performance with respect to the variations, similar to this work. Yet, I do think that this work presents unique and distinct features from prior works (e.g., focus on easy reproduction), which I hope to read in the revision. 
2. HTML diversity: while the use of FastHTML is desirable, I question if it would have negative effects in terms of HTML diversity. To elaborate, I think evaluating the agents with diverse formats can be more appealing if the agents take text input.
3. Intentionally misleading description: I worry this is out of scope in this work. There are many works tackling the robustness of the agents in an adversarial manner [5]. However, from my understanding, the UI variations are a ‘natural’ perturbation challenging the agent's robustness rather than ‘intentional’ (line 188-189). Such features should be handled differently, in my opinion.
4. Simplicity of tasks: the proposed test suite suffers from both (1) a lack of diversity and (2) a lack of challenges. There is not enough headroom for improvements in this benchmark for the state-of-the-art agent (i.e., GPT-4o), which I assume would be a bigger problem with the recent agent (e.g., GPT-5). I suggest diversifying the tasks in terms of both volumes (i.e., more than 15 task templates) and difficulties.

I do believe that this work has a strong potential to be a highly noteworthy work that can function as a standardized benchmark.

--- 

References:

[1] Rawles et al., “Android in the Wild: A Large-Scale Dataset for Android Device Control” (2023).

[2] Lee et al., “Benchmarking Mobile Device Control Agents across Diverse Configurations” (2024).

[3] Rawles et al., “AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents” (2024).

[4] Zhang et al., “LlamaTouch: A Faithful and Scalable Testbed for Mobile UI Task Automation” (2024).

[5] Wu et al., “Dissecting Adversarial Robustness of Multimodal LM Agents” (2025).

### Questions
For brevity, I included questions and suggestions in the section above.

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces OpenApps, a novel, lightweight, open-source ecosystem designed to measure a new dimension of reliability for autonomous UI-agents: performance fluctuations across app variations. Current evaluations rely on fixed environments, failing to capture how changes in app design, appearance, or content affect agent success. OPENAPPS provides six common apps (calendar, maps, messenger, etc.) that are highly configurable via simple Python/YAML files, enabling large-scale, reproducible experiments (over 10,000 trials conducted). The study finds that while reliability within a fixed app is stable, reliability across app variations fluctuates drastically (e.g., Kimi-VL-3B's success rate varied from 63% to 4%), and that failure modes like looping and hallucination are highly environment-dependent. This highlights app variation as a critical, overlooked axis of agent reliability.

### Strengths
1. The paper correctly identifies the critical gap between testing on fixed clones and real-world deployment, where app style, content density, and language constantly change. This significantly advances the utility of reliability metrics.

2. OpenApps is designed for parallel, large-scale study (single CPU, < 10MB memory, pure Python). This low overhead is a crucial enabler for large-scale RL training and comprehensive testing that previous VM-based or complex web-clone environments could not afford.

3. Beyond average success rates, the study provides valuable diagnostics on how environment changes induce specific failure modes.

### Weaknesses
1. The paper focuses exclusively on 15 simple, short-horizon tasks (e.g., adding a to-do item). While justified to isolate reliability, the performance of agents on complex, multi-app workflows (which is the ultimate promise of UI-agents) remains unevaluated. Future work needs to extend the task set to long-horizon, multi-step tasks to form a comprehensive reliability benchmark.
2. The study primarily varies each app factor (appearance or content) independently. The authors acknowledge that interactions between multiple variations (e.g., dark theme + German language + dense content) could expose novel failure modes. A brief initial exploration of combined variations would strengthen the argument.
3. The paper provides excellent diagnostics but offers limited concrete proposals for agent development (e.g., does VLM fine-tuning on synthetic variants solve the problem? Should contrast-boosting pre-processing be used for dark themes?). While the goal is diagnosis, suggesting a simple architecture or training technique that could leverage the synthetic data would enhance the paper's prescriptive value.

### Questions
1. The flexibility of OpenApps allows generating training data across thousands of versions. Did the authors explore whether fine-tuning a model (e.g., UI-TARS) on a synthetic dataset generated by OpenApps could significantly improve its overall deviation (Figure 5) compared to training only on a single fixed version?

2. The paper notes GPT-4o performs highly when given simplified AX tree representations along with the screenshot. Which specific content variations (e.g., German language, adversarial text) affected the reliability of AX tree parsing more than the visual recognition, or was the failure primarily a model-level reasoning issue?

3. The reward function is deterministic. For complex, open-ended tasks, a continuous partial reward is often more useful. Could the authors confirm that the Python logic of OpenApps could support easily pluggable, continuous reward functions (e.g., L2 distance between the current state vector and the target state vector) for future RL training efforts?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces OpenApps, a lightweight and scalable benchmark designed to evaluate the **reliability of multimodal UI agents under app-environment variations**. It exposes an overlooked weakness in current evaluation protocols—agents’ fragility to minor visual or structural changes in otherwise identical tasks. The framework is technically sound, reproducible, and empirically comprehensive across seven models. Results show dramatic reliability drops (up to 50% across app versions) and reveal new failure behaviors such as looping and hallucination. The contribution is both **timely and impactful**, offering a reproducible infrastructure that can underpin future robustness research.

While methodologically solid, the work would benefit from a **clearer theoretical framing of “reliability,” richer task diversity, and improved presentation**. The current tasks are simple and limited to single-app workflows; including long-horizon tasks would strengthen generality. Despite these limitations, the paper’s novelty, execution quality, and potential community value are strong.

### Strengths
1. The paper identifies a key blind spot in existing agent benchmarks: reliability across app variations. Unlike prior environments that focus on fixed app clones, OpenApps systematically quantifies how changes in design and content affect UI-agent performance. This is a genuine conceptual contribution to the field of multimodal agent evaluation.
2. OpenApps is implemented in pure Python and runs on a single CPU, removing the heavy dependencies of prior environments (e.g., emulators, Docker containers, or large memory requirements). This design choice makes it widely accessible and reproducible.
3. The authors conduct 10,000+ trials across seven state-of-the-art agents, including GPT-4o, Claude Sonnet, Kimi-VL, Qwen-VL, and UI-TARS. The scale and comprehensiveness of these experiments convincingly demonstrate the practical significance of app variation as a factor in agent reliability.
4. Results are striking: task success can fluctuate by more than 50% across app versions, and specific models show massive degradation (e.g., Kimi-VL drops from 63% to 4%). The paper also documents behavioral shifts such as looping and hallucination, revealing that environmental variability induces new failure modes.
5. OpenApps can be extended for safe training, adversarial robustness testing, or sim2real transfer studies. The discussion section articulates a credible roadmap for future research directions, showing awareness of the broader ecosystem.

### Weaknesses
1. **Task Simplicity and Limited Scope**
    - The current experiments focus on **15 simple tasks** (e.g., adding a to-do item). While the paper demonstrates large variability even on these, such simplicity limits conclusions about generalization to *complex or long-horizon* tasks seen in real-world apps.
2. **Insufficient Theoretical Framing of “Reliability”**
    - While empirical results are strong, the paper lacks a deeper theoretical formalization of *reliability across app variations*—for example, framing it as an expected reward stability problem under distributional shifts could strengthen the conceptual rigor.

### Questions
+ How are app variations generated — random parameter changes, manually curated modifications, or rule-based templates? Could the process introduce artificial correlations that models might exploit?
+ How do you ensure that a task instance in version A is semantically equivalent to the same task in version B (e.g., identical goal, only UI difference)?
+ Did you observe any model types that adapt better to variations (e.g., multimodal LLMs vs. RLVR UI-trained agents)? If so, what might explain this?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the question of reliability of UI agents when the underlying environments change (without affecting overall functionality). The key idea is to create multiple appearance and content variations of a set of UI apps and measure agent performance across these variations. The authors find that all models have significant variance in performance, with larger models being relatively more stable compared to smaller models. The analysis further illustrates that certain specific types of variations in the environment lead to consistently larger drops in agent performance (e.g. using darker themes for apps leads to consistent failures).

### Strengths
- The question of reliability is an important one, especially when considering how dynamic real world apps often are. Assessing agent performance under these variations is quite useful.

- The analysis shows that multimodal models have large variability in performance, raising important questions about their suitability in practical settings, where the underlying environments change. 

- This benchmark could be a useful additional evaluation for multimodal agent solutions in addition to the existing benchmarks.

### Weaknesses
The main issue I have with this work is that the motivations for the variations is not adequately justified and the choices for curation are not well explained. 
  - Content variations seem somewhat arbitrary. If one were to include misleading descriptions and adversarial perturbations performance is going to drop. What is the point of this exercise? There is not much in terms of motivation for why these were done and what specific ways in which these were created. Why use German translations? Why not others? 
  - Similarly the choice of stylistic variations also appears somewhat arbitrary and adversarial. It would have been much more natural to target the most frequently used variances alongside some of the more rarely used variations.

### Questions
- I appreciate the state based evaluation and the rigorous evaluation. However, it seems like partial progress towards the task is not accounted for in the current analyses. Is there a way to extend the analysis to partial progress? 

- How are the within app fluctuations observed? Are these deviations computed across all queries? Or averages of standard deviations over multiple attempts for each query?

### Soundness
3

### Presentation
4

### Contribution
3
