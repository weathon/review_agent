# PersonaAgent: When Large Language Model Agents Meet Personalization at Test Time

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Large Language Model (LLM) empowered agents have recently emerged as advanced paradigms that exhibit impressive capabilities in a wide range of domains and tasks. Despite their potential, current LLM agents often adopt a one-size-fits-all approach, lacking the flexibility to respond to users’ varying needs and preferences. This limitation motivates us to develop PersonaAgent, the first personalized LLM agent framework designed to address versatile personalization tasks. Specifically, PersonaAgent integrates two complementary components: a personalized memory module that includes episodic and semantic memory mechanisms; a personalized action module that enables the agent to perform tool actions tailored to the user. At the core, the persona (defined as unique system prompt for each user) functions as an intermediary: it leverages insights from personalized memory to control agent actions, while the outcomes of these actions in turn refine the memory. Based on the framework, we propose a test-time user-preference alignment strategy that simulate the latest $n$ interactions to optimize the persona prompt, ensuring real-time user preference alignment through textual loss feedback between simulated and ground-truth responses. Experimental evaluations demonstrate that PersonaAgent significantly outperforms other baseline methods by not only personalizing the action space effectively but also scaling during test-time real-world applications. These results underscore the feasibility and potential of our approach in delivering tailored, dynamic user experiences.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper focuses on personalized LLM agents. To address the one-size-fits-all limitations of existing approaches, it proposes PersonaAgent. The core of this framework is the persona, which acts as an intermediary between the personalized memory module (including semantic memory mechanisms) and personalized actions. Based on this framework, the paper proposes a test-time user-preference alignment strategy to optimize the persona by simulating recent interactions. Finally, the effectiveness of the proposed framework is demonstrated on the LaMP benchmark.

### Strengths
1. Using the persona as an intermediary between memory and action is intuitive and reasonable.
2. A complete personalized agent framework, PersonaAgent, is proposed.
3. Experimental results demonstrate the effectiveness of the proposed strategy, and detailed ablation experiments are performed.

### Weaknesses
1. The paper is poorly written, the method is obscure, and lacks necessary details. For example, the input to $f_{enc}$ is a tuple. How is the tuple encoded into an embedding? How does the resulting $\mathcal{R}^u(q^*)$ work? What is the observation? How are personas and observations combined to perform personalization? What is the textual loss function? These are unclear.
2. The motivation for the proposed module is unclear, making this paper less like a technical report.
3. Are the baseline methods adapted to the dataset used, or do they use their generalized form? Furthermore, do they also use the tools used in this paper?
4. There is a lack of discussion on the space and time complexity of the algorithm. Each user needs to maintain a large amount of information, and test time alignment may introduce significant inference delays.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
1

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
This paper proposes PersonaAgent, a personalized LLM agent framework for conversational AI. PersonaAgent integrates two memory types—episodic and semantic memory—and a personalized action module, all coordinated via a dynamically optimized user persona (system prompt). The framework introduces a test-time user-preference alignment strategy that updates the persona prompt based on recent user interactions. Experiments on the LaMP benchmark demonstrate improved performance over non-personalized, workflow-based, and agentic baselines.

### Strengths
* Proposes a unified memory-action framework for personalization, generalizable across tasks.

* Introduces a test-time persona optimization mechanism, enabling real-time adaptation to user preferences.

* Provides comprehensive experiments and ablation studies, showing the necessity of each component.

### Weaknesses
* Evaluation relies on machine metrics (accuracy, F1, ROUGE) not fully convincing; would be better to include personalization metrics (e.g., Persona-F1, faithfulness).

* The computational cost and scalability of test-time alignment are not thoroughly discussed.

### Questions
* How does the test-time alignment impact inference latency and scalability?

* How does the method perform for users with limited interaction history?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PersonaAgent, the first personalized LLM agent framework that adapts to individual user preferences through a dynamic persona. It combines personalized memory (episodic and semantic) and action modules, with the persona acting as an intermediary that evolves via user interactions. Experiments show it outperforms baselines in personalization and scales effectively in real-world test-time settings.

### Strengths
1.	PersonaAgent is the first LLM agent framework for dynamic user-level personalization, combining episodic/semantic memory and persona-driven actions for continuous adaptation.
2.	The test-time alignment method optimizes the persona via simulated interactions and textual loss, enabling real-time, scalable personalization without retraining.
3.	The work rigorously validates its approach across four diverse personalization tasks, ablation studies, and scaling analyses.

### Weaknesses
1.	The evaluation relies primarily on LaMP, which focuses on text classification and generation tasks that do not adequately capture instruction-following ability in real interactive dialogues—would the framework still excel other benchmarks?
2.	The action module only uses Wikipedia search and personal data retrieval; given that Wikipedia search may dominate performance gains, does the personalization component (i.e., personal data retrieval alone) meaningfully contribute to the agent’s effectiveness?
3.	Although persona case studies are included, the full agent execution process is not illustrated—could a detailed step-by-step example better demonstrate how personalization operates in practice?
4. The paper lacks runtime analysis—how long does the agent actually take to execute?

### Questions
See the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
