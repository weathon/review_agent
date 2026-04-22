# EduVerse: A User-Defined Multi-Agent Simulation Space for Education Scenario

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 4, 4, 2, 2

## Abstract
Reproducing cognitive development, group interaction, and long-term evolution in virtual classrooms remains a core challenge for educational AI, as real classrooms integrate open-ended cognition, dynamic social interaction, affective factors, and multi-session development rarely captured together. Existing approaches mostly focus on short-term or single-agent settings, limiting systematic study of classroom complexity and cross-task reuse.
\textcolor{blue}{We present \textbf{EduVerse}, \textcolor{blue}{one of the first} \emph{user-defined} multi-agent classroom simulator supporting customizable environment, customizable agents, and multi-session evolution.} A distinctive human-in-the-loop interface further allows real users to join the space. Built on a layered \textbf{CIE} (\textbf{C}ognition–\textbf{I}nteraction–\textbf{E}volution) architecture, EduVerse ensures individual consistency, authentic interaction, and longitudinal adaptation in cognition, emotion, and behavior—reproducing realistic classroom dynamics with seamless human–agent integration.
We validate EduVerse in middle-school Chinese classes across three text genres, environments, and multiple sessions. Results show: \textbf{(i) Instructional alignment}: simulated \textcolor{blue}{Initiate-Response-Feedback (IRF)} rates ($0.34$--$0.55$) closely match real classrooms ($0.37$--$0.49$), indicating pedagogical realism; \textbf{(ii) Group interaction and role differentiation}: network density ($0.27$–$0.40$) with about one-third of peer links realized, while human–agent tasks indicate a balance between individual variability and instructional stability; \textbf{(iii) Cross-session evolution}: the positive transition rate $R^{+}$ increase by 11.7\% on average, capturing longitudinal shifts in behavior, emotion, and cognition and revealing structured learning trajectories; 
\textcolor{blue}{\textbf{(iv) Cross-disciplinary generalization}: without any additional tuning, IRF rates and peer-interaction topologies naturally adapt to the discourse characteristics of history instruction while preserving the core instructional structure, demonstrating robust cross-disciplinary transfer.}
Overall, EduVerse balances realism, reproducibility, and interpretability, providing a scalable platform for educational AI. The system will be open-sourced to foster cross-disciplinary research.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes an LLM-based multi-agent simulation designed for educational settings. Specifically, they try to capture the dynamics of cognitive development and classroom interactions over time.

### Strengths
- Educational psychology theories back up the design of the simulation components
- The simulation captures a lot of factors that go into classroom dynamics, such as seating arrangements, varying personalities, emotions, etc.
- They performed rigorous experiments validating different aspects of the model. 
- The results are quite promising. The interaction dynamics resemble behaviors observed in real classroom settings.

### Weaknesses
- Not sure if the authors can claim to be the first multi-agent simulation space for education since I found some existing papers that use multi-agent simulations in the education domain [a, b], aside from those already cited in the more comprehensive related works in the appendix. Granted that they are doing different things, "multi-agent simulation space for education" is broad enough to encompass their works as well.

[a] Xu, S., Wen, H. N., Pan, H., Dominguez, D., Hu, D., & Zhang, X. (2025, April). Classroom Simulacra: Building Contextual Student Generative Agents in Online Education for Learning Behavioral Simulation. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems (pp. 1-26).

[b] Arana, J. M., Carandang, K. A. M., Casin, E. R., Alis, C., Tan, D. S., Legara, E. F., & Monterola, C. (2025, July). Foundations of PEERS: Assessing LLM Role Performance in Educational Simulations. In ACL 2025 Student Research Workshop.

- Evaluations on the temporal dynamics / trajectories are a bit weak in my opinion. They are not backed up by any data but only rather vague statements like "clear individual differences", "sustained positive affect", etc. 

- The memory management and knowledge progression is also not quite clear. The authors mention that they are adjusted based on behavioral signals such as bloom level and response type. However, there does not seem to be very convincing validation of this design.

### Questions
- Regarding the temporal dynamics experiments, it is not quite clear what we expect the curves to be. What is a valid trajectory and what is not? Does any curve that exhibit positive transitions / shifts valid? How well does this match reality?
- How do you manage the memory? How do you decide what gets stored and what gets forgotten? How well does this match realistic human student memory recall?
- How are the emotions being probed? Is it just through direct prompting, or do you also ask the agents to answer some kind of questionnaire similar to what we would give a human participant?
- Out of curiosity, are you able to capture problematic student behaviors? It would be very interesting to simulate interventions or management strategies for them.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents EduVerse, a user-defined multi-agent simulation framework for educational scenarios. It introduces a Cognition–Interaction–Evolution (CIE) architecture to simulate realistic classroom dynamics among virtual students and teachers. EduVerse enables customization of agents, environments, and interactions, while supporting human-in-the-loop participation. The authors evaluate the system through simulated and real classroom experiments in Chinese language teaching, showing that EduVerse can reproduce authentic teaching dynamics and capture long-term learning evolution. The platform demonstrates promising potential for educational research, intelligent tutoring, and social learning analysis.

### Strengths
Solid theoretical foundation – The CIE framework is conceptually well-motivated and systematically designed, combining cognitive modeling, social interaction, and evolution mechanisms.

Rich and diverse experiments – The authors conduct multiple experiments across different educational aspects (cognitive alignment, group interaction, long-term evolution), providing strong empirical support.

### Weaknesses
Unclear system details – The description of the system’s user interface and real-user interaction mechanism (how students and teachers use EduVerse) is vague and underdeveloped.

Limited explanation of real-world experiments – Although the paper claims real classroom validation, the implementation details of these experiments (e.g., how data were collected, how participants interacted) are not clearly stated..

Scalability and generalization – The experiments are confined to a specific subject (Chinese language classes), and the system’s adaptability to other domains remains untested.

### Questions
Questions:
1. What does "IRF" mean? This abbreviation appears in abstract without providing any full name before.
2. For the LLM, you mentioned that you use "InternVL" and "GPT-4" (In line 216-219), do you fine-tuning the LLMs via education data to get better results?
3. You mention that "EduVerse provides a human-in-the-loop interface that admits real students or teachers alongside virtual agents" (Line 281-282), how can students and teachers in the real world interact with the system? Does the user interface (UI) is something like the UI of ChatGPT?
4. Do the names appear in the experiment part like "Zhang Jie", "Liu Li" are the names of your simulated student agent or real student name in the real world?
5. Do you have more information about experiments conducted in real world classrooms? It seems that all the experiments in the experiment part are conducted in simulators.

Suggestions:
1. As the author provide so much appendix, I recommend add a table of contents before appendix part.

Typos:
1. In Fig1, ②: Mr. Zhuvividly => Mr. Zhu vividly
2. In Fig1, circle a: Cognition Engin => Cognition Engine

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
This paper presents a framework, EduVerse, for user-defined multi-agent simulation in the context of AI in education. The authors deployed  EduVerse in middle school Chinese language classes with diverse educational 
tasks, rich emotional expression, and complex interaction structures. The authors also conducted empirical experiment with existing frameworks.

### Strengths
- Timely topic focusing on the AI in education and LLM 
- In-depth analysis of related work 
- The proposed framework  combines the cognitive, interactive, and 
evolutionary dynamics of developmental agents in the context of AI in education
- Deployed in classrooms showcases the practical impact
- Human-in-the-loop interface allows real teachers and students  to enabling simulation, causal testing, and validation

### Weaknesses
For designing an intelligent tutoring system, it is crucial to take into account the subject domain. For example, students cognitive, help seeking behavior, and peer discussion vary widely across math vs writing an essay in literature vs introductory programming. 

The prior work by other researchers cited by the authors are also domain specific. Would the authors say how to incorporate the framework for a specific subject domain with different question difficulties and knowledge base?

### Questions
Please see weakness

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors introduce EduVerse, a “user‑defined multi‑agent simulation space” for virtual classrooms built around a Cognition–Interaction–Evolution (CIE) architecture layered over a Perception–Cognition–Action loop. Users can customize the seat graph/layouts, teacher/student agents,, and sessions (multi‑lesson trajectories). A human‑in‑the‑loop interface lets real users join a simulated class. Figure 1 lays out the three components, user‑defined environment, CIE agent modeling, and interaction/evolution experiments. 
The authors’ core claim is the simulated instructional realism of a typical classroom (measured by IRF rates).

### Strengths
I think this is a very interesting idea with a good approach but I have alot of reservations about the claims made by the authors.

Focusing on the positives, I think the work done itself is good. There are plenty of good uses for a simulator of this type, especially ones that involve a human in the loop. 

I do like the modular CIE breakdown, the explicit teacher pacing controller, and that the tasks are already implemented. The range of evaluation criteria is good, even if I have some concerns about them. IRF alignment, B/E/C distributions, small‑graph network summaries, ablations, human–agent tasks, and a cross‑session measure. 

My favorite part is probably the CIE-based agent modeling. I think there’s alot of potential in the ideas that the authors outlined here with how the process of teacher-led group discussion can play out.

### Weaknesses
While there’s alot to like about this paper, I think there are some pretty severe issues with the main claim:

- The authors position EduVerse as the “first” user‑defined multi‑agent classroom simulator. But they even acknowledge other pre-existing multi-agent class room simulators in their own related work, and other general agent set ups (ie, AgentVerse) that already support role‑based, IRF‑style interactions. 

- The Abstract and Table 1 frame IRF rates as “close” to real classes, but Table A4 shows sizeable divergences (e.g., Argumentative Essay, Lecture: 0.639 vs. 0.417 real). ESPECIALLY with a signal as noisy as teacher-led discussion in classes, I feel like it's hard to take any purely quantitative analyses at face value without some kind of qualitative evidence to back it up. 
- There’s a lack of details about how many classes/schools were used as the comparison baseline and, again, who annotated the logs who could provide qualitative evidence as backup. 
- The system labels its own cognition (Bloom) and emotion during the Monitor step, then reuses these labels for evaluation (BEC distributions). If im not misunderstanding, this is basically just the model asking itself if it's correct, which doesnt seem super reliable.
- The authors  fine‑tune VLM backbones (InternVL/LLaVA/Qwen‑VL/MiniCPM) for text‑only style, trained on ~6k utterances. Why VLMs for text style control? The authors also report InternVL “achieved the highest scenario‑grounded performance,” but the metric and protocol aren’t shown. 
- Not a huge negative but a heads up, for Figure 1, the middle section has “Cognition Engin” instead of “Cognition Engine”

### Questions
- Did the authors inspect the generated EduVerse logs vs the conversation logs of a real classroom?
- Were all experiments/baselines drawn from the same classroom or different classrooms?
- What was the motivation for using VLM backbones for what seems, to me, a largely text-based scenario?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on reproducing realistic classroom dynamics. To achieve this, the authors present EduVerse, a novel user-defined multi-agent simulation platform that introduces a Cognition–Interaction–Evolution (CIE) architecture. This architecture models the long-term cognitive, emotional, and behavioral development of virtual agents within customizable classroom environments.

### Strengths
Human–Agent Interaction provides valuable insights through experimental studies.

### Weaknesses
- The work attempts to address multiple aspects, including individual modeling, role-differentiated social interaction, and longitudinal instructional adaptation. But does not clearly explain them.

- The evaluation is vague.
  - Please include the key metrics in the main paper instead of the appendix. This would improve both readers’ understanding and reviewers’ efficiency.
  - Table 1 does not show how the simulation aligns with real classroom data. For example, IRF_rate on Lyrical Prose (0.336 vs. 0.486) contradicts the claim of only minor genre-specific variations.
  - Figure 5 lacks a clear caption about the ablation study, making it difficult to follow the analysis and interpret the bar chart.
  - Much of the analysis focuses on individual cases, while the main focus should be at the class level.

- The work is limited to Chinese language classes. Cross-domain or cross-linguistic experiments would strengthen the generalization of this work.

- Figure 4 is too small and hard to review.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
