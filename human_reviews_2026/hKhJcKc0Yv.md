# Automotive-ENV: Benchmarking Multimodal Agents in Vehicle Interface Systems

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Multimodal agents have demonstrated strong performance in general GUI interactions, but their application in automotive systems has been largely unexplored. In-vehicle GUIs present distinct challenges: drivers' limited attention, strict safety requirements, and complex location-based interaction patterns. To address these challenges, we introduce Automotive-ENV, the first high-fidelity benchmark and interaction environment tailored for vehicle GUIs. This platform defines 185 parameterized tasks spanning explicit control, implicit intent understanding, and safety-aware tasks, and provides structured multimodal observations with precise programmatic checks for reproducible evaluation. Building on this benchmark, we propose ASURADA, a geo-aware multimodal agent that integrates GPS-informed context to dynamically adjust actions based on location, environmental conditions, and regional driving norms. Experiments show that geo-aware information significantly improves success on safety-aware tasks, highlighting the importance of location-based context in automotive environments. We will release Automotive-ENV, complete with all tasks and benchmarking tools, to further the development of safe and adaptive in-vehicle agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a benchmark for vehicle interface systems called Automotive-ENV; which consists of 185 tasks and 8 apps for benchmarking agents that controls GUI inside the vehicle. To operate this kind of environment, this paper presents a framework called ASURADA that use multimodal inputs including vehicle status, screen shot and GPS information.

### Strengths
1. Important problem: vehicle interface systems needs special attention for agent research.

Vehicle interface systems have special challenges that 1) it heavily rely on active GPU position of the vehicle and the driving status; 2) it can potentially cause safety issues for the driver. Therefore, it worth a set of special evaluations and benchmarks to ensure its safety and usefulness when controlled by the LLM agents. This paper presents a valuable contribution in this direction.

2. Interesting benchmark: Automotive-ENV can be an interesting benchmark platform for in-vehicle GUI control.

The proposed benchmark contains many potentially interesting tasks like AC control for specific location, speed alert given geo location and local regulation, using fog-removal system under foggy whether. All these are quite practical and interesting applications in the real world. Furthermore, I think the developed AutomotiveOS can be an interesting prototype platform for research in this direction.

### Weaknesses
1. Lack of important information regarding the benchmark.

I cannot find many important information in the paper or supplementary material:
- What are the specifically 185 tasks and 8 apps?
- How is the geolocation data generated?
- How does the internet connection API work?
- Is the benchmark assume agent will always run in the backend or just run given user prompt?
- Is the vehicle assumed to be parked or driving? How to generate realistic vehicle status?

This set of information is necessary for one to judge the usefulness of the proposed benchmark. With them missing it's hard to see how useful the benchmark would be for the community.

2. Lack of important information regarding the experiment.

Many experiment details are lacking in the paper. For example:
- how many instances are performed for each task to compute the success rate? 
- why almost all scores for Safety-Aware metrics are rounded to integer? does this mean the number of instances performed is very limited?
- what's the specific system prompt for each type of the method?
- why human perform the best for safety-aware tasks without GPS input? does this imply human operators know the geolocation information in other ways?

Lack of these information make it hard to adjust how meaningful the experiment results are in this paper.

3. Method contribution is not significant.

The proposed method, to my understanding, is simply adding GPS information to the previous prompts. This does not seem very novel as GPS information can be seen as another kind of vehicle status and can be well represented with text; therefore I don't think it really deserve to be called another modality.

4. Further analysis about the safety issues would be important.

A important aspect of vehicle interaction system is that it's highly safety-critical: some of the error could lead to very bad outcome or safety issue. A carefully designed benchmark should capture carefully the safety concerns of false operation or errors. For example, it should not only check whether a certain operation is success or not; but also measure if this operation will lead to safety issue (for example, close the beam on a night highway should receive severe penalty in the benchmark).

### Questions
Please refer to the weakness section.

### Soundness
3

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
3

### Summary
This paper introduces Automotive-ENV, a large-scale and high-fidelity benchmark for evaluating multimodal agents operating in automotive graphical user interfaces (GUIs). The environment is implemented on top of a simulated Automotive Operating System (Automotive OS), which reproduces real in-vehicle control modules such as HVAC, navigation, safety alerts, and infotainment.
The system provides a structured observation space.
Vehicle state and system API access for reward checking.
Agents interact through standard GUI actions (click, swipe, text input) and specialized automotive safety APIs. The benchmark defines 185 parameterized tasks categorized into explicit control, implicit intent understanding, and safety-aware tasks, each validated through low-level system-state checks for deterministic evaluation.
On top of this environment, the authors propose ASURADA, a geo-aware multimodal agent architecture. ASURADA extends a ReAct-style reasoning agent with two modules:
A GPS-informed reasoning module that integrates real-time location, weather, and local regulations into the decision-making process;
A multimodal planning pipeline that sequentially performs perception (screen + a11y tree parsing), reasoning (text + GPS grounding), and execution (GUI/API actions), followed by reflective memory updates.
Experiments across GPT-4o-Mini and Gemini series models show that incorporating GPS context improves performance—particularly in safety-critical tasks such as “driving alignment” and “environment alerts”—though overall success rates (≈65%) remain below human-level baselines.
In summary, the paper contributes (1) a reproducible and geographically parameterized benchmark for automotive GUI agents, and (2) a geo-aware multimodal agent framework (ASURADA) that adapts reasoning to dynamic driving contexts.

### Strengths
1.This work considers the safety-critical nature and geographic dependence of in-vehicle systems—factors often overlooked in prior studies—and integrates them into the evaluation and reasoning process. This represents a degree of innovation, and the paper introduces the first benchmark specifically designed for multimodal agents in automotive graphical user interface (GUI) environments.
2. The evaluation methodology is reliable, supported by a well-structured simulation pipeline that effectively replicates a real automotive operating system. The framework ensures reproducibility through deterministic state resets, consistent system images, and programmatic reward checking based on actual system APIs rather than visual matching.
3. The writing quality is adequate. The paper is clearly structured and easy to follow, with a logical flow from motivation to experimental results.
4.Given the current significant performance gap between intelligent agents and human operators, the research direction holds promising potential and long-term significance.

### Weaknesses
1.Although the work fills a gap in the study of automotive GUI systems by introducing two additional input modalities, its conceptual foundation is highly similar to existing benchmarks such as WebArena, AndroidWorld, and OSWorld. The contribution mainly lies in extending general GUI evaluation frameworks to a new application domain, rather than proposing fundamentally new technical mechanisms.
2.While the engineering quality of the benchmark environment is acceptable, the paper demonstrates limited technical depth in model design and experimental analysis. The ASURADA agent is presented conceptually, but lacks detailed architectural discussion or rigorous methodological evaluation.
3.Some experimental results—such as the reasoning efficiency comparison in Figure 4—lack sufficient quantitative explanation, making it difficult to determine the actual source of performance improvement.
4.The experimental results and code are not publicly available, which prevents proper assessment of the system’s reproducibility, openness, and practical usability.

### Questions
How is the GPS signal encoded and how is it integrated with other features?

Beyond the conceptual description, the paper should also explain why the addition of multimodal information leads to improved accuracy—whether the gain primarily results from the semantic understanding capability of the underlying LLM, or from the enhanced architectural design of the proposed module. Such clarification is needed to demonstrate that the improvement originates from the model architecture itself. 

For the autonomous driving task, not only the memory usage is required, but also the issue of response delay needs to be addressed.

### Soundness
2

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
1

### Summary
This paper provides a multi-modal agent benchmark to evaluate Automotive graphic user interface. It is powered by GPT based model and it supports reproduicible benchmark over auto UI.

### Strengths
1. Multi-modal agent sounds novel for auotmotive ui benchmark.

### Weaknesses
1. I am not really not an expert in this field. But it occurs to me this paper does not suit for ICLR venue. Better venue might be ITSC or IV related driving related conferences or journals.

### Questions
n/a

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
This paper introduces Automotive-ENV, a benchmark and simulation environment for evaluating multimodal agents in automotive graphical user interface (GUI) systems. The platform comprises 185 parameterized tasks across general and safety-aware categories, featuring structured multimodal observations and programmatic reward signals based on low-level system states. The authors also propose ASURADA, a geo-aware multimodal agent that integrates GPS data to inform its actions. Experimental results demonstrate that incorporating geographic context significantly improves performance on safety-critical tasks, though a substantial gap to human performance remains.

This work addresses an important and underexplored domain for autonomous agents, providing a dedicated benchmark for in-vehicle GUI interactions. The benchmark is clear, featuring a high-fidelity environment, a large number of parameterized tasks, and a clear taxonomy. The proposed agent, ASURADA, incorporates a novel geo-contextual reasoning module, which is shown to substantially enhance performance on safety-aware tasks. However, the experimental comparison to baseline agents could be more extensive, and details regarding the reproducibility of the environment and agent implementation are limited.

### Strengths
[1] Novel Domain and Benchmark: The paper identifies a significant gap in agent evaluation by focusing on the automotive domain, which presents unique challenges like safety-criticality and dynamic context. Automotive-ENV is the first dedicated benchmark for this setting, as confirmed by the comparison table.

[2] Clear Task Design: The benchmark includes a large and diverse set of 185 tasks, categorized into explicit control, implicit intent, driving alignment, and environment alerts, enabling a multifaceted evaluation of agent capabilities.

[3] Geo-Aware Agent Design: The proposed ASURADA agent introduces a meaningful innovation by integrating real-time GPS data to query external context, enabling location-aware decision-making.

### Weaknesses
[1] Limited Baseline Comparisons: The primary baselines (T3A, M3A) are adapted from a single prior work (Rawles et al., 2024). A comparison with other recent, powerful GUI agents (e.g., CogAgent, AppAgent) is absent, making it difficult to fully situate the performance of ASURADA within the current state-of-the-art.

[2] Insufficient Implementation and Reproducibility Details: Key details for replicating the environment or the agent are missing. The manuscript does not specify the source of the "production-grade" Automotive OS image, the emulation setup, or the specific APIs used for state checks and actions.

[3] Limited Analysis of Failure Modes: While overall success rates are reported, there is little discussion or analysis of the specific types of errors made by the agents, particularly for the challenging Implicit Intent tasks where performance is low.

[4] Uncertain Statistical Significance: The number of evaluation episodes or random seeds used per task/model is not reported, making it difficult to assess the statistical significance of the reported performance differences.

[5] Ambiguity in Geo-Context Integration: The precise mechanism for how the GPS signal triggers network queries and how the retrieved context is formatted and integrated into the agent's reasoning process (e.g., prompt construction) is not described in detail.

### Questions
1. Expand the baseline comparisons by evaluating against a wider range of recent, open-source GUI agents (e.g., CogAgent, AppAgent) to provide a more comprehensive performance landscape.

[1] Hong, W., Wang, W., Lv, Q., Xu, J., Yu, W., Ji, J., ... & Tang, J. (2024). Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 14281-14290).

[2] Zhang, C., Yang, Z., Liu, J., Li, Y., Han, Y., Chen, X., ... & Yu, G. (2025, April). Appagent: Multimodal agents as smartphone users. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems (pp. 1-20).

[3] Li, Y., Zhang, C., Jiang, W., Yang, W., Fu, B., Cheng, P., ... & Wei, Y. (2024). Appagent v2: Advanced agent for flexible mobile interactions. arXiv preprint arXiv:2408.11824.

2. Include a dedicated section or appendix detailing the environment setup, including the emulated Automotive OS specifications, the list of accessible APIs, and the exact procedure for state checking. Committing to a public release of the code and environment would significantly strengthen the paper's impact.

3. Conduct a qualitative error analysis on a subset of failed tasks, particularly for Implicit Intent and low-performing Safety-Aware tasks, to provide insights into common failure modes and directions for future model improvement.

4. Report the number of independent runs or evaluation episodes for each agent configuration and consider reporting confidence intervals or conducting statistical significance tests for the key results in Table 3. 

5.Provide a more detailed description or a schematic of the GPS context integration pipeline within ASURADA, clarifying the query formulation, information source, and context injection mechanism into the agent's reasoning loop.

6.Consider including an ablation study on the components of ASURADA (e.g., the contribution of the reflection mechanism versus the geo-context module) to better isolate the source of its performance gains.

### Soundness
2

### Presentation
2

### Contribution
1
