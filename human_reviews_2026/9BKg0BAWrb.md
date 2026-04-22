# K²-Agent: Co-Evolving Know-What and Know-How for Hierarchical Mobile Device Control

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Existing mobile device control agents often perform poorly when solving complex tasks requiring long-horizon planning and precise operations, typically due to a lack of relevant task experience or unfamiliarity with skill execution. We propose $\textbf{K²-Agent}$, a hierarchical framework that models human-like cognition by separating and co-evolving declarative ("knowing what") and procedural ("knowing how") knowledge for planning and execution. K²-Agent’s high level reasoner is bootstrapped from a single demonstration per task and runs a Summarize–Reflect–Locate–Revise (SRLR) loop to distill and iteratively refine task-level declarative knowledge through self-evolution. The low-level executor is trained with our curriculum-guided Group Relative Policy Optimization (C-GRPO), which (i) constructs a balanced sample pool using decoupled reward signals and (ii) employs dynamic demonstration injection to guide the model in autonomously generating successful trajectories for training. On the challenging AndroidWorld benchmark, K$^2$-Agent achieves a new $\textbf{state of the art}$ with $\textbf{76.1\% success rate}$, ranking $\textbf{1st}$ among all methods $\textbf{using only raw screenshots and open-source backbones}$. Furthermore, K²-Agent shows powerful dual generalization: its high-level declarative knowledge transfers across diverse base models, while its low-level procedural skills achieve competitive performance on unseen tasks in ScreenSpot-v2 and Android-in-the-Wild (AitW).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper **“K²-Agent: Co-Evolving Know-What and Know-How for Hierarchical Mobile Device Control”** introduces a cognitively inspired hierarchical agent that explicitly separates declarative (“know-what”) and procedural (“know-how”) knowledge to improve performance on mobile device control tasks. The system combines a **symbolic planner** that refines task knowledge through the **SRLR (Summarize–Reflect–Locate–Revise)** self-evolution loop, and a **reinforcement learning executor** trained with **C-GRPO (Curriculum-Guided Group Relative Policy Optimization)**. This dual-loop architecture enables co-evolution: the planner learns better strategies from execution feedback, while the executor gains structured guidance from the planner’s task decomposition. Experiments on the AndroidWorld benchmark demonstrate that K²-Agent achieves a **76.7% success rate**, surpassing prior open-source baselines while requiring minimal supervision (one demonstration per task). The work bridges symbolic reasoning and embodied learning in a cognitively grounded way, suggesting potential for broader applications in generalist AI systems.

### Strengths
- **Novel cognitive framework:** The clear distinction and co-evolution of declarative and procedural knowledge mirror human learning mechanisms, offering both conceptual clarity and practical benefits.  
- **Methodological innovation:** The SRLR loop provides a systematic mechanism for reflective reasoning and self-correction, while C-GRPO elegantly handles data imbalance and sparse rewards through curriculum design.  
- **Strong empirical results:** The system achieves state-of-the-art performance on AndroidWorld with extremely low data requirements, showing both efficiency and robustness.  
- **Interpretable learning process:** The explicit task knowledge base and revision history improve transparency and allow fine-grained analysis of reasoning and execution errors.  
- **Broader relevance:** The cognitive analogy to human dual-memory systems and the hierarchical design are relevant to multiple domains beyond mobile control, such as robotics and multimodal planning.

### Weaknesses
- **Limited evaluation diversity:** The experiments focus primarily on AndroidWorld; results on other benchmarks (e.g., WebArena, mobile navigation, or real-device tasks) would strengthen claims of generalization.  
- **Scalability concerns:** The SRLR loop may involve significant overhead as task complexity grows, especially if symbolic reasoning steps become large or interdependent.  
- **Ambiguity in error localization:** The “Locate” phase of SRLR appears manually defined or heuristic-based; it is unclear how reliably the system identifies and generalizes error points without human intervention.  
- **Comparative analysis:** The work could benefit from deeper comparisons with alternative hybrid systems that combine reasoning and RL (e.g., ReAct, Voyager, or RPA-style architectures).  
- **Reproducibility limitations:** Key implementation details—such as how knowledge base edits are represented or parameterized—are under-specified, making replication challenging for other researchers.

### Questions
1. How scalable is the SRLR loop when applied to tasks requiring multiple interdependent app contexts (e.g., cross-app workflows)?  
2. Could C-GRPO be adapted for domains beyond visual mobile control—such as web navigation or embodied robotics—and if so, what adjustments would be required?  
3. Does the declarative knowledge base support compositional reuse, allowing the agent to combine skills learned in separate tasks into new ones?  
4. How sensitive is the system to the initial demonstration quality—can it recover from a poorly demonstrated “seed” trajectory?  
5. How are symbolic edits validated to prevent catastrophic forgetting or the accumulation of contradictory task rules over long SRLR iterations?

### Soundness
2

### Presentation
3

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
This work tackles long‑horizon mobile device control and argues that agents should separate two kinds of knowledge for accurate planning and execution: (i) declarative “knowing what” for task planning, and (ii) procedural “knowing how” for precise UI actions. The proposed K²‑Agent is a two‑layer planner–executor:

High‑level planner (know‑what): a training‑free VLM that bootstraps from one demonstration per task and maintains a textual task knowledge base (KG) updated by a Summarize Reflect Locate Revise (SRLR) loop. SRLR distills an initial plan from a demo, analyzes failures, identifies the wrong decision points, and applies atomic edits to the knowledge base.

Low‑level executor (know‑how): a trainable VLM optimized with Curriculum‑guided GRPO (C‑GRPO). C‑GRPO (1) builds error‑decoupled replay pools for type vs. parameter errors, and (2) uses dynamic demonstration injection to prepend variable‑length expert prefixes for difficult samples; the policy is updated with GRPO.

### Strengths
* The experimental results are strong. Especially, the KG graph being transferred to different VLMs, and the low-level executor being transferred to AiTW and ScreenSpot-v2 is interesting.
* Proposing to train the low-level executor with reinforcement learning while refining the planner’s knowledge base through training-free self-evolution (SRLR) offers an efficient division between low-level actions and declarative knowledge planning & refinement.

### Weaknesses
* The overall flow of section 4 itself was easy to understand, but specifically what each component is and how they operate is unclear in several parts. Specifically, (1) what 'atomic' edits are and how they are applied to the KG was unclear until looking at the appendix. (2) In dynamic demonstration injection, what 'variable length expert prefix' is unclear. (3) What the single demonstration that is provided is, and whether this is identical or different from the AndroidWorld benchmark is unclear
* If the benchmark provides single expert demonstrations per task category, this may be providing more information than other results

### Questions
* What is the motivation of applying the dynamic length prefix? Why does this improve performance?
* What is the dynamic length expert prefix and how is the length dynamically applied? For example,  if the expert prefix something like 'Be sure to answer in the format of ...', how do you dynamically change the length of this prefix?
* Regarding the single demonstration provided, is the provided task excluded from the benchmark? Also, if a successful task is provided per category of the AndroidWorld benchmark, it seems not fair to say that this method utilizes less information than other methods that only use screenshots or A11y trees. 
* How is the KG constructed from the high-level planner utilized by the low-level executor? The reverse is clear to me, the demonstrations of the low-level executor fails is provided to the planner.

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
2

### Summary
The paper proposes $K^2$-Agent, a hierarchical framework for mobile device control that explicitly decouples declarative “know-what” (a high-level planner distilled via a Summarize–Reflect–Locate–Revise, SRLR, loop) from procedural “know-how” (a low-level executor trained with a curriculum-guided C-GRPO objective). On AndroidWorld (116 tasks across 20 apps), the authors claim state-of-the-art 76.7% success “ranking 1st … among methods using only raw screenshots and open-source backbones” and show transfer to ScreenSpot-v2 and Android-in-the-Wild (AitW) without fine-tuning. The intended scope is screenshot-only agents on Android emulator tasks, plus cross-benchmark skill transfer.

### Strengths
* Clear hierarchical decomposition (SRLR + C-GRPO) with reasonable mathematical formalization (PPO-style clipping; explicit content/format rewards).
* The paper’s split between a high-level SRLR planner (“knowing what”) and a low-level C-GRPO executor (“knowing how”) isn’t just conceptual; you show the actual SRLR prompts and output formats. This makes the approach inspectable and reuse-friendly.
* Even with screenshot-only I/O, the result on AndroidWorld is pretty strong as shown in Table 1.
* The executor’s reward is specified precisely (format + content, Eq. (5) and total reward definition) and the action space is enumerated (Table 6), which helps reproduction and critique.

### Weaknesses
* The authors mention that “Human experts achieve about 80% average success” on AndroidWorld without details (line 373): annotator count, expertise, task sampling, time limits, or inter-rater protocol.
* The AndroidWorld results do not report of seeds, dispersion, or confidence intervals (CIs). Without multi-seed evaluation, the “SOTA” claim is statistically fragile.
* The executor is trained on 606 single-step samples derived from 116 demos (Appendix B.2.2). Even with the stated seed-based split controls, the paper doesn’t quantify app- or template-level overlap between training demonstrations and evaluation task distributions; leakage through UI idiosyncrasies remains plausible without stricter controls or OOD app tests.

### Questions
* How many seeds were used for Table 1 and Figure 1? Please report mean ± std over ≥3 seeds and add 95% CIs to plots. 
* Right now, it is hard to tell whether the gains come from C-GRPO itself or. from extra compute or curriculum variations. You should compare GRPO vs C-GRPO with everything else identical: model init, data, rollout length, batch size, optimizer, KL/clip settings, and exactly the same number of environment interactions (and similar wall-clock within ±5%). Also please report the learning curves of these runs.
* Vary the key curriculum knob(s) you introduce (e.g., mixing ratios, scheduling temperature T if applicable) across a small grid (e.g., 3–5 values). For which hyperparameters is your method brittle or robust?

### Soundness
2

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
4

### Summary
This work proposed a new mobile control agents built on (V)LLMs with a hierarchical design. Specifically, the authors proposed a high-level planner that is capable of reasoning with the task goal, storing knowledge from execution and analysing failure patterns to better solve the problem. A new C-GRPO method is proposed to fine-tune the low-level execution policy for better performance in long horizon tasks.

### Strengths
This paper is well written and easy to follow.

The proposed method of C-GRPO is new.

Experimental results on benchmark environment show positive gains.

Ablation study shows the effectiveness of some high-level designs proposed in this work.

### Weaknesses
The overall novel in the developed technique is not strong. In the domain of AI planning with LLM/VLM, approaches like task decomposition, building memory and knowledge graph or reflective planning have been widely used in different applications. This work largely follows these patterns as well.

Some technical design lacks sufficient motivations or analysis. It is unclear in some technical sections what the key problem targeted to address is and why the process is designed in the current way. Please see the following several detailed points:
- Section 4.2.1: it is proposed to summarize the knowledge into rules or checklist, but there is no analysis on how robust this representation is. In general, what type of knowledge can the summarize induce? What are the unknow knowledge or failure cases that the summarize cannot induce? Will the summarizer induce wrong knowledge that harms the following steps?
- Section 4.2.2 - Task-level: It would be better to illustrate what kind of root-cause of the failure can be captured by the reflection from the episode trajectory, and what cannot. Are there any cases where the VLM fails to identify the reason of the failure? For example, where the failure happens underlying in the operation system that cannot be directly observed from screenshots.
- Section 4.2.3:
  - Prepending a variable length expert prefix to the model input is used to gradually improve the training. However, the reason why the policy on original query got improved through training with query and several hints is less well explained. It would be better to show the performance like the reward of the original query with training on query + hints during the curriculum training process. This can improve the understanding of generalizing from query + hints to query alone.
  - Another existing similar approach in RL to guide exploration is hindsight experience reply. It would be better to have a study on this approach also.
Overall, the proposed system as whole seems provide gains in training and application, but the developed approach did not provide strong technical insights on why individually proposed component bring improved performance.

Figure 5, there is no standard deviation on the reward in figure b.

### Questions
Please see the weakness part.

### Soundness
3

### Presentation
3

### Contribution
2
