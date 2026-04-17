# COMPASS: A Multi-Turn Benchmark for Tool-Mediated Planning & Preference Optimization

- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
Real-world tasks like travel planning require LLM agents to satisfy hard constraints (dates, budget) while optimizing user's utility preferences (cheapest hotel, most convenient flights). We formalize this as *constrained preference optimization*, where agents strategically use tools to gather information and compare options to optimize user's preferences. We introduce **COMPASS** (**C**onstrained **O**ptimization through **M**ulti-turn **P**lanning **a**nd **S**trategic **S**olutions), a benchmark evaluating agents through realistic travel planning. We build a travel database covering transportation, accommodation, and ticketing for 20 U.S. National Parks, plus a tool ecosystem mirroring commercial booking platforms. Evaluating state-of-the-art models reveals a significant **acceptable–optimal gap**: models achieve 85-95% constraint satisfaction but only 60-70% preference optimization, settling for feasible rather than optimal solutions. Performance degrades sharply on multi-service coordination tasks. Our tool-use analysis shows task success strongly correlates with information gathering—insufficient exploration is the primary bottleneck, though future models should prioritize efficient over exhaustive search. *COMPASS* provides a rigorous benchmark for diagnosing core challenges in constrained preference optimization and guiding development of user-aligned agents.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper propose the Compass benchmark that focus on multi-turn preference optimization with rich and complex environments and tools.
They adopt hard constraints and soft preferences and the latter is expected to be optimized through multi-turn interactions.
They also design different levels of tasks to evaluate model performance.
The tools included are diverse and comprehensive, mirroring the real-world environment.
The results show that despite some models can perform well on acceptable rate, they fall short in optimal rate.

### Strengths
1. This paper provides a complex environment that mirrors real-world and messy noises rather than a cleaned and simplified simulation.
2. It focuses on the optimal solution rather than an acceptable solution that evaluates how models can best serve as an applicable agent.
3. The multi-turn interactions with the simulated user brings more complications into this problem.

### Weaknesses
1. The simulated user query could be deployed to attain more diverse interactions beyond the 241 tasks.

### Questions
1. The paper find that agents have higher acceptable rate than the optimal rate. Do you have a hypothesis for why this happens? Is it a failure of insufficient working memory, lack of exploration or sticking to the original solution without searching further?
2. Can the user simulators be deployed to create more diverse and complex test cases?

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
5

### Summary
The paper introduces COMPASS, a benchmark for evaluating LLM agents’ constrained preference optimization in travel planning, addressing gaps in existing tool-use and planning benchmarks. However, critical limitations in novelty prevent it from meeting ICLR’s standards for publication.

### Strengths
1. The benchmark design integrates realistic travel data (20 U.S. National Parks), a multi-turn user simulator, and a comprehensive tool ecosystem, which aligns with real-world agent deployment scenarios.

2. The identification of "acceptable–optimal gap" and "plan-coordination gap" provides insights for future agentic travel planning.

### Weaknesses
1. The primary issue is that the core idea of integrating constrained satisfaction with preference optimization is built upon existing research without introducing substantive innovations. For instance, TravelPlanner initially addressed hard constraint satisfaction, while subsequent works such as ChinaTravel [1] have emphasized the synergy between hard and soft constraints, and TripTailor [2] has also explored aspects of personalization. However, this paper entirely overlooks these aspects, thereby significantly undermining its claimed contributions.

2. Although the authors focus on constraint satisfaction problems, the Related Work section predominantly discusses tool-use benchmarks, while largely ignoring established benchmarks in travel planning—a domain that has seen extensive research. This omission represents a notable gap in the literature review.

3. The proposed level classification lacks intuitive justification. Why not use the number of constraints as a basis for level categorization, which would offer a more straightforward and interpretable framework?

4. The concept of an "acceptable–optimal gap" raises questions about its practical relevance. Under hard constraints, single-objective preference optimization admits a deterministic solution. While achieving this optimum may be challenging for models, it is debatable whether reaching the true optimum is necessary in real-world scenarios. What, then, is the substantive significance of this gap?

5. The relationship between the "plan-coordination gap" and constrained optimization remains unclear and appears conceptually disjointed. The connection between the two proposed gaps and how LLMs address constraint satisfaction problems is neither well-motivated nor clearly articulated.

6. The evaluation methodology for hard and soft constraints is inadequately defined. Using the set of feasible solutions as a metric is problematic: when the number of constraints is small and the database is large, the feasible set can become excessively large—or even impossible to enumerate—rendering evaluation infeasible. While such a measure may be meaningful for preference evaluation, insisting on optimality for all preferences is arguably unreasonable. This touches upon multi-objective optimization (MOO), where Pareto optimality should be considered. Why not employ model-based ranking for preference evaluation instead?


7. The experiment completely ignored Agent-based algorithms, such as ReAct and LLM-module. Why?



[1] ChinaTravel: An Open-Ended Benchmark for Language Agents in Chinese Travel Planning. 2024.

[2]  TripTailor: A Real-World Benchmark for Personalized Travel Planning. 2025.

### Questions
see weaknesses

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- Proposes COMPASS, a multi-turn benchmark for tool-mediated travel planning and preference optimization.

- Builds an interactive evaluation framework with tools, databases, and a user simulator reflecting dynamic user behaviors.

- Evaluates several LLM agents, revealing gaps in preference optimization and multi-tool coordination performance.

### Strengths
I find this paper’s contribution significant in its dataset and evaluation framework.

- **Dataset**
    - The dataset itself is highly valuable. While most prior works rely on synthetic data, this paper collects a *real-world* dataset through APIs, which makes it much more realistic and potentially very useful for other researchers in this domain.
- **Systematized Evaluation Framework**
    - The authors integrate various evaluation aspects from prior travel-planning studies into a systemized framework.
    - To my knowledge, this is the first systematic attempt to evaluate preference optimization, which is an essential dimension in this domain.
    - **Answer construction**: The authors consider all possible combinations of factors when constructing reference answers, and then define metrics accordingly. This goes beyond simple “pass/fail” checks for each constraint and represents a thoughtful attempt to capture composite preferences.

### Weaknesses
While the dataset and framework contributions are strong, some parts, especially the experimental section, leave room for improvement.

**w1. Dataset description insufficiency**

  * The paper lacks a detailed explanation of how the dataset was collected via APIs.
  * While collecting *real-world* data is commendable, the authors should provide a short case study illustrating **what aspects of real-world interactions are captured that synthetic datasets fail to model**.

 **w2. Missing citations / related work**

  * There exist prior works on **multi-turn travel planning** (e.g., [1]) that address similar interactive planning settings, which is similar to one of their proposed user settings (*progressive constraint revelation*).

 **w3. Framework design clarity**

  * *User Simulator Design* (around L250) introduces within-conversation dynamics and persona diversity, which are interesting ideas.
  * However, it remains unclear **whether these behaviors are grounded in the real-world dataset** or are purely heuristic / author-defined.
  * The rationale and distribution (e.g., “how often users exhibit such behaviors”) should be explicitly stated.

 **w4. Framework validation**
  * While authors provide validation of framework in Table 2, It would be helpful if the validation were reported separately for each simulation type.

 **w5. Limited evaluation depth**

  * The results are not sufficiently analyzed across different **user simulation types**, though model behavior likely varies under each setting.
  * Although tool-calling is presented as a key feature of the benchmark, the paper does not analyze **how often models succeed or fail at tool calls**, nor how such success/failure impacts final outcomes.
  * (Minor) The discussion in L411–413 about open-source model tendencies seems weak, since only the Qwen models were evaluated, limiting generalizability.

**w6. Limited novelty/insight in findings**

  * The contribution of collecting a real-world dataset is valuable, but some findings derived from it struggle to present clear takeaways beyond the dataset itself.
  * For instance:

    * Section 5.1 on *constraint conflicts* somewhat overlaps with the findings of [1]. 
    * Section 5.2 on *conversation efficiency* reproduces patterns already observed in prior reasoning tasks ([2]).

**w7. Presentation and display issues (minor)**

  * Tables and figures are placed far from where they are discussed, disrupting readability (e.g., Figure 2 on page 2).
---
Reference

[1] Flex-TravelPlanner: A Benchmark for Flexible Planning with Language Agents, Oh et al, 2025

[2] MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback, Wang et al, 2023

### Questions
Already covered in Weaknesses.

### Soundness
3

### Presentation
2

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
This paper introduces a multi-turn benchmark for evaluating Large Language Model (LLM) agents on tool-mediated planning and constrained preference optimization within a travel planning environment. The benchmark's evaluation reveals that current state-of-the-art LLMs can satisfy basic constraints but consistently fall short in finding truly optimal, preference-aligned solutions, particularly as task complexity increases.

### Strengths
1. The paper is original in how it formalizes multi-turn travel planning as constrained preference optimization, explicitly separating hard feasibility constraints from soft preference objectives. This framing enables measurable evaluation of both “acceptable” solutions and optimization quality.   
2. The benchmark design is reasonable. It couples a realistic tool ecosystem (hotels, flights, permits) with a dynamic, persona-driven user simulator and provides ground-truth optima via exhaustive enumeration over feasible combinations. Metrics are interpretable and complementary, and the analysis is thorough across plan-coordination levels, constraint count, search complexity, and conversation efficiency.

### Weaknesses
1. External validity is constrained by the LLM-based user simulator. Despite some human audits, the simulator’s coverage of adversarial, ambiguous, or inconsistent user behaviors is limited, and there is no direct comparison with real users.  
2. The domain scope is narrow, which can even be considered an extension of the TravelPlanner benchmark. Triptailor[1] is also a multi-turn travel planning benchmark, which involves more detailed itineraries and personalized requirements. Treating this work as an interface design and analyzing the challenges of the task on different benchmarks would make a greater contribution to the community. 
3. The analysis of tool usage is crucial in travel planning, as discussed in section 5.3. However, it's unfortunate that only a case study is provided there. Is there a better mechanism for quantitative analysis in this area? 
4. The paper's title emphasizes "Tool-Mediated Planning and Preference Optimization," yet a substantial portion of the content is dedicated to addressing constraints of varying difficulty levels. This creates a certain degree of misalignment between the stated focus and the actual technical emphasis. 

[1] Triptailor: A real-world benchmark for personalized travel planning

### Questions
1. Simulator coverage: How diverse are personas and scripts relative to the full task space? Can you provide coverage analyses and persona fidelity checks, including adversarial or underspecified behaviors?  
2. In an agent that uses multiple rounds of tool calls to ultimately achieve planning, how should the agent consider calling the tools, how should the tools be called, and whether the call is successful? What methods are available for analyzing the capabilities of an LLM in this regard?

### Soundness
2

### Presentation
3

### Contribution
2
