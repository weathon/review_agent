# WorldGUI: An Interactive Benchmark for Desktop GUI Automation from Any Starting Point

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4, 2

## Abstract
GUI agents have achieved outstanding performance in GUI element grounding. However, planning remains highly challenging, especially due to the sensitivity to the initial state of the environment. Specifically, slight differences in the initial state-such as the target software not being open or the interface not being in its default state, often lead to planning errors. This issue is widespread in real application scenarios, but existing benchmarks fail to evaluate it. To address this gap, we introduce WorldGUI, a comprehensive GUI benchmark containing tasks across ten widely used desktop and web applications (e.g., PowerPoint, VSCode, Acrobat), each instantiated with diverse initial states to simulate authentic human–computer interactions. Complementing this, we propose GUI-Thinker, a universal framework that unifies three core modules: Planner-Critic for high-level plan refinement, Step-Check for intermediate verification, and Actor-Critic for action-level optimization to proactively detect and correct errors. Experimental evaluation shows that WorldGUI-Agent outperforms the outstanding existing model (Claude-3.5-Sonnet Claude Computer Use) by 12.4% in success rate on WorldGUI, and achieves a 31.2% overall success rate on WindowsAgentArena, surpassing the prior state-of-the-art by 11.7%. Our analysis further reveals that dynamic augmentation tasks and desktop environments pose substantial hurdles, underscoring the necessity of adaptive planning and feedback-driven execution for advancing real-world GUI automation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes WorldGUI, a new benchmark designed to evaluate GUI automation agents under varied initial conditions. The key insight is that slight differences in an application’s starting state can derail existing GUI agents, yet prior benchmarks did not test this sensitivity. WorldGUI addresses this by providing 611 tasks across 10 popular desktop and web applications (PowerPoint, VSCode, Acrobat), each task instantiated with multiple diverse initial states to mimic real user scenarios. Alongside the benchmark, the authors introduce WorldGUI-Agent, a unified agent framework emphasizing “critical thinking” at three levels: a Planner-Critic module that critiques and refines the high-level action plan, a Step-Check module that verifies intermediate steps against the current GUI state, and an Actor-Critic module that checks each executed action and corrects it if needed.

### Strengths
The paper presents a timely and well-motivated contribution by introducing the first GUI benchmark that emphasizes diverse and non-default initial states, addressing a key gap in existing evaluations. The proposed WorldGUI-Agent integrates three critical reasoning modules (planning critique, step validation, and action correction) into a coherent and practical framework that significantly outperforms prior baselines. The benchmark is comprehensive, covering realistic desktop and web environments, and the experimental results are thorough, with clear evidence from ablations and comparative studies. Overall, the work is well-executed, clearly written, and offers both a strong dataset and an effective agent design that can benefit future research in GUI automation.

### Weaknesses
1. The framework heavily relies on instructional videos to guide planning. However, the paper does not clarify how the quality, consistency, or completeness of these videos is ensured. Since the video is only used during the initial planning phase and is processed via Whisper into subtitles, it's unclear why raw video is needed at all—could equivalent textual guidance achieve the same effect? A discussion on this design choice is missing.

2. While the proposed “WorldGUI-Agent: Thinking Before Doing” architecture emphasizes reflective decision-making, the general idea of reactive execution with verification has appeared in prior work. The modular design introduces multiple sequential LLM calls (planning, critique, checking, correction), which likely contributes to higher latency and execution overhead. However, the paper does not analyze or discuss the computational cost tradeoffs involved.

3. The paper lacks qualitative or statistical analysis of failure cases. Given that even the best-performing agent still solves only 36% of tasks, a deeper understanding of where and why the system fails would be valuable—for example, issues related to visual grounding, ambiguous UI states, or suboptimal plan corrections.

### Questions
1. What is the rationale for using instructional videos rather than equivalent textual descriptions, especially given that the video is only transcribed and used during planning? How do you ensure the quality and consistency of these videos?

2. Since the proposed agent includes multiple sequential reasoning modules, did you evaluate or analyze the runtime cost and latency of the system compared to simpler baselines? How practical is it for real-time or interactive use?

3. Could you provide an analysis of common failure cases (e.g., types of errors, failure modes)? This would help clarify where the agent struggles most and whether failures arise from perception, planning, or action execution.

4. test with newer or stronger foundation models, such as GPT-5 or Claude 4.5.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
WorldGUI introduces a 611-task benchmark spanning ten desktop/web applications with deliberately varied initial states to better reflect real human–computer interaction dynamics. The paper also proposes a modular agent, WorldGUI-Agent, incorporating three “critical-thinking” checkpoints (Planner-Critic, Step-Check, Actor-Critic) that verify and correct plans/actions on the fly. Experiments show that WorldGUI-Agent improves success rates over strong baselines on both the new benchmark and WindowsAgentArena.

### Strengths
1. Existing GUI benchmarks assume default initial states; this work addresses that gap with systematic state diversification

2. 611 tasks across 10 apps, each with five augmentations; annotation and data-construction pipeline is described.

### Weaknesses
1. Benchmark novelty relative to AssistGUI/OSWorld is mostly the “pre-action” augmentation; more rigorous quantitative analysis of state-diversity (e.g., edit distance between GT plans, UI tree variance) is missing.

2. Limited discussion of annotation quality: inter-annotator agreement and error rates for GT plans/pre-actions are not reported.

3. recent UIExplorer (2025) and GUI-World (2025) datasets/agents are not compared or cited, though they target dynamic GUI exploration.

4. Planner uses GPT-4o subtitles but no quantitative ablation on subtitle quality or Whisper errors; unclear how video actually helps beyond the textual query (Table 5 only shows “w/o Inst. Video” for five apps).

5. Actor module’s low-level action grammar (PyAutoGUI) may cause brittle coordinates; no analysis of failure due to resolution or scaling.

6. Compute cost is high (Table 14: ~2k tokens/step, 23 steps ⇒ >40k tokens per task) but little optimisation discussion; reproducibility for researchers without API budgets is challenging.

7. Mathematical formulation (§2.1) introduces POMDP but later modules do not exploit it formally; no learning—only prompting—so the RL framing feels decorative.

### Questions
N/A

### Soundness
2

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
This paper introduces WorldGUI, a desktop and web GUI benchmark that evaluates agents from varied starting states created by pre actions, rather than only the default initial screen. The dataset covers ten common applications with 611 tasks, each paired with a user query, an instructional video, and a project file to ensure reproducibility. The authors also propose WorldGUI Agent, a framework with three critical thinking modules called Planner Critic, Step Check, and Actor Critic, designed to verify and correct plans and actions during execution. Experiments report higher success rates than prior systems on the new benchmark and on WindowsAgentArena, with detailed ablations that show the importance of each module.

### Strengths
1. The main novelty is to evaluate GUI agents from many non default initial states. In real use, a user often calls an assistant in the middle of a workflow or with the app in a non default configuration. Existing benchmarks usually fix the initial state and therefore miss this difficulty. WorldGUI fills this gap with explicit pre actions that alter the context or put the task in an intermediate state.
2. The proposed Planner Critic, Step Check, and Actor Critic form a simple verify then correct loop that is easy to reason about and implement.
3. The ablations quantify how each module matters. Removing Actor Critic cuts the overall success sharply, which is convincing evidence that online verification is critical for GUI control.

### Weaknesses
1. Verify and correct loops and plan self-critique have already appeared in recent agents for GUI and web, for example, Agent S, Agent S2, and also earlier reflection-style methods. WorldGUI Agent packages these ideas well, but the technical novelty beyond a careful engineering of prompts and modules is modest. It would help to show larger gaps against recent native models like UI TARS and ShowUI under matched conditions.
2. The abstract claims an improvement of 1.7% on WindowsAgentArena. Table 6 on page 9 lists an overall success of 31.2% and includes Agent S2 at 29.8%, which makes the margin smaller. Please reconcile the exact baseline and evaluation settings, and release full prompts and seeds for that run.
3. The framework uses a GUI parser and OCR and runs several checks per step. The Limitations section explains the cost and the tradeoff between performance and running time. Practitioners may face long wall clock time, especially compared with end-to-end native agents. A quantitative time and cost comparison against Agent S2 and UI TARS on the same hardware would make the value clearer.

### Questions
See weaknesses and the following questions:
1. On WindowsAgentArena, what exact task list, VM image, model version, and prompt settings were used for the 31.2% result in Table 6. How does this compare to Agent S2 and UI TARS when you keep the same setup?
2. How do you generate and validate pre actions?
3. How sensitive is the framework to the GUI parser? Can you report an ablation that removes the parser and relies only on visual grounding, and another that swaps in a different grounding model, such as OS Atlas or ShowUI?

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
3

### Summary
The author's main proposition is that when agents perform a task, slight differences in the initial state can significantly impact their performance. To test the robustness of agents for this, the authors introduce a benchmark, WorldGUI, where they create tasks that include the initial states of the tasks augmented in various ways. The authors also introduce WorldGUI-Agent, which incorporates reflection modules at various stages of planning and action execution to correct the agent's mistakes at different points. The authors show the effectiveness of their model by evaluating it in WorldGUI and WindowsAgent Arena.

### Strengths
1. The authors introduce an interesting problem of varying the initial state of the task to determine how the agent would perform. Aside from how thoroughly it is tested, the problem itself is interesting.
2. The authors introduce methods to vary the initial state of the task and benchmark models to show that performance drops. This is a surprising result. Further, the authors show that their method outperforms Plan and Act, highlighting the importance of reflection during planning and action execution.

### Weaknesses
1. The authors could have addressed the question of varying initial states by taking existing benchmarks like OsWorld or Windows Agent Arena and running their augmentation methods on these benchmarks. It is not clear what the advantage of introducing a new benchmark is. The videos, as well as ground truth plans, could be obtained from the successful completion of tasks from exisiting benchamrks. This would make it much easier to understand the core question of how varying initial states affect model performance, as these benchmarks have standard evaluation and several models evaluate on these, and hence changes in initial state could be studied more effectively. Have the authors thought of this? And if so, why was a new benchmark created?
2. From my understanding, it feels like the WorldGUI-Agent uses ground truth videos to calibrate its plans, and Table 5 shows a significant drop in performance when video is not used. Since videos are not available in real-world tasks, this is limiting.
3. It is not very clear why agents fail on augmented tasks. Given that this is the core problem of the paper, it is not properly discussed. Particularly, what type of augmentations do the agents fail on mostly, and how do the agents behave?
4. The evaluation setup needs various clarifications (please see weaknesses). The lack of virtualenvs and Docker, similar to OSWorld and WindowsAgent Arena, introduces a limitation regarding ease of use and standardization of the benchmark.
5. For the Windows Agent Arena, it looks like Agent S2 achieves comparable or better performance on all domains except Web. Can the authors provide some insights into this? This result indicates that WorldGUI-Agent's elaborate reflection process might not provide explicit benefits beyond the task setting of WorldGUI. For the Windows category, Agent S2 has a better number than WorldGUI-Agent. This needs to be corrected.

### Questions
1. The evaluation setup is not clear. The authors indicate that they perform execution based evaluation, but also state that there are experts who evaluate whether the task has been performed correctly. Is this to do a human eval on task completion beyond execution based evaluation? Also, if there is no Docker or virtual env where the agent is acting, how exactly is the evaluation done? Also, are these evaluators the authors of the paper? If so, that could lead to biases if human eval is the source of reporting numbers.
2. How do the annotators select videos to be used to create the benchmark?
3. How are the videos and ground truth plan used in WorldGUI Agent? If it is used in the main results, that might be limiting in realistic scenarios, as ground truth plans and videos are not usually available in realistic scenarios (as mentioned in the weakness).
4. Could the authors provide some examples where agents perform well on the actual tasks and fail on meta tasks, and why these happen? This is a surprising result, as most agents are not explicitly optimised for a particular starting point, to the best of my knowledge.
5. Why is the agent only evaluated in WindowsAgent Arena and not OS-World?
6. Why do the authors only provide the current state of the screenshot and metadata and not the history of past actions or previous (k) screenshots?

Please see the weakness for more questions

### Soundness
2

### Presentation
2

### Contribution
2
