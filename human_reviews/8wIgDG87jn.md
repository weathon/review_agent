# MorphAgent: Empowering Agents through Self-Evolving Profiles and Decentralized Collaboration

- Avg Score: 5.25
- Decision: Reject
- Scores: 6, 5, 5, 5

## Abstract
Large Language Model (LLM) based multi-agent systems (MAS) have shown promise in tackling complex tasks, but often rely on predefined roles and centralized coordination, limiting their adaptability to evolving challenges. This paper introduces $MorphAgent$, a novel framework for $\textit{decentralized}$ multi-agent collaboration that enables agents to $\textit{dynamically evolve their roles and capabilities}$. Our approach employs self-evolving agent profiles, optimized through three key metrics, guiding agents in refining their individual expertise while maintaining complementary team dynamics. $MorphAgent$ implements a two-phase process: a warm-up phase for initial profile optimization, followed by a task execution phase where agents continuously adapt their roles based on task feedback. Our experimental results show that $MorphAgent$ outperforms traditional static-role MAS in terms of task performance and adaptability to changing requirements, paving the way for more robust and versatile multi-agent collaborative systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
MORPHAGENT is a fully decentralized multi-agent system that enables agents to autonomously adapt their roles and capabilities through self-evolving profiles optimized using three key metrics: Role Clarity Score, Role Differentiation Score, and Task-Role Alignment Score. The framework employs a two-phase process—a warm-up phase for initial profile optimization and a task execution phase where agents iteratively update their profiles based on task feedback—enhancing the system's adaptability and robustness in dynamic environments without relying on predefined roles or centralized coordination. Experimental results demonstrate that MORPHAGENT outperforms traditional static-role multi-agent systems in task performance and adaptability, effectively handling domain shifts and node failures.

### Strengths
I will preface this review by saying that this is not my area of expertise, therefore I might be unfamiliar with crucial work in the state of the art, making it difficult for me to fairly asses the contribution.

1. **Experimental results**: The experimental results are strong and demonstrate the advantages of using MORPHAGENT for tasks that require coordination especially when centralization might lead to issues (due to failure of important nodes) or there is domain switch.

2. **Motivation**: Decentralized systems are particularly useful in real-world scenarios where failure of specific nodes might cause the entire system to fail, therefore MORPHAGENT stands out as a promising approach for complex environments. 

3. **Novelty**: The paper addresses an under-explored problem and proposes a very unique solution that is demonstrated to work in the evaluation scenarios.

### Weaknesses
1. **Computational overhead**: My main issue with this paper is that even though the computational overheads are aknowledged in the limitations section, they are not directly stated. In particular how much more computation is being used in wall-clock time v.s. the baselines? Without it, it is difficult to asses how applicable and practical the method really is.

2. **Clarity**: The writing of the paper is not super clear, it took me a long time to understand some of the metrics because fundamental definitions and terms are missing. In particular in dependency score, the definition of "subtree" is missing and since there are no references to Dependency Parsing, it was hard to infer that subtree referred to the dependency subtree. Similarly terms like "skill prototype" and "potential skill tokens" are used for metric definitions but not defined. More importantly, there is no intution on why the metrics are chosen, making some of them seem arbitrary in the context of role ambiguity (e.g. why is the dependency score correlated to the specificity of the profile).

3. **Fairness of the baseline comparisson**: This is a relatively minor issue, but GPTSwarm is evaluated in the GAIA Benchmark, so why not use GAIA here as well? The lack of this comparisson makes it difficult for me to assess wether the strength of MORPHAGENT is dependent on dataset specifics.

### Questions
1. How does MORPHAGENT handle communication between agents?
2. How did you determine the weighting coefficients $(\beta_1, \beta_2, \beta_3)$ in the Role Clarity Score? Are these weights task-specific, or did you find a set of weights that work well across different tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces MorphAgent, a framework for decentralized multi-agent LLM collaboration that enables agents to dynamically evolve their roles and capabilities. Unlike existing approaches that rely on predefined roles or centralized coordination, MORPHAGENT employs self-evolving agent profiles optimized through three metrics. The framework implements a two-phase process: a warm-up phase for initial profile optimization, followed by a task execution phase where agents continuously adapt their roles based on task feedback. Through evaluations on various benchmarks, the authors demonstrate that MorphAgent outperforms traditional static-role systems and shows better adaptability to domain shifts and robustness to node failures.

### Strengths
1. The paper effectively communicates its ideas through clear visualization - Figure 1 illustrates the key challenges with concrete examples, while Figure 2 provides a comprehensive overview of the framework's workflow. 
2. The experimental results seem good, showing MorphAgent's consistent performance gain across different benchmarks. 
3. Analyses of their framework's advantages is presented.

### Weaknesses
1. The implementation details and methodology are severely unclear and poorly explained:
   - The profile updating process is vaguely described, with crucial details buried in figures and appendix
   - The three metrics are defined with numerous undefined notations and unexplained components (e.g., *skill prototype* and *potential skill tokens* in Definition 3.1, and *vector representations* in Definition 3.3)
   - The design choices lack justification, such as using dependency trees in RCS
   - The auxiliary agent is only mentioned in Section 3.1. Why is it necessary? What's the disadvantage of letting autonomous agent directly interact with the environment?
   - Experimental settings in Sections 4.2 and 4.3 are incomprehensible - the domain shift setup and node failure mechanism are not properly explained. I can't even know how these two experiments are conducted.
   - There are too many things that are not clearly explained. I've tried to list them, but there is definitely something else missing for a reader to fully understand the framework.

2. The experimental results presentation has some issues:
   - Table 1 is poorly presented with unexplained notations. I don't know what are the two numbers represent in each cell.
   - The reported improvement on MATH dataset with MorphAgent (over 30 points!) with GPT-3.5-turbo is suspiciously large and lacks explanation. It is nearly impossible for me that multi-agent debate can lead to such a significant improvement.
   - The explanation of the level in the caption of Table 1 is inconsistent with the text content.
   - The analysis of results is superficial, lacking a detailed discussion of why the method works

3. The paper lacks concrete examples and case studies:
   - No examples showing how agent profiles evolve through iterations
   - No comparison of actual responses between MorphAgent and baselines

4. The evaluation methodology is questionable:
   - The node failure experiments lack clear description of failure mechanisms. How did you incur the node failure? What does node failure mean?
   - Domain shift experiments don't clearly specify whether it's transfer learning or continuous adaptation. Is it that a multi-agent team obtained through optimization on one task is transferred to another task?

Overall, while the paper presents an interesting idea, the poor explanation of implementation details, questionable result presentation, and lack of concrete examples make it difficult to assess the true value and reproducibility of the work.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Motivated by current challenges in multi-agent systems (MAS), this paper proposes a decentralized and dynamic framework that enhances system robustness and adaptability. 
By introducing a fully decentralized collaboration mechanism, agents can autonomously coordinate without reliance on any critical node, ensuring resilience in the face of failures. 
Additionally, the adaptive role optimization mechanism allows agents to dynamically adjust and improve their roles based on task requirements, resulting in a more flexible and robust system. 
Comprehensive experiments validate this approach, demonstrating improvements in task performance and adaptability.

### Strengths
1. This paper identifies key challenges in multi-agent systems (MAS) and addresses them through decentralized and adaptive paradigms, with experiments demonstrating the effectiveness of this approach.

2. It introduces agent profiles as dynamic representations of evolving capabilities and responsibilities, using three quantitative metrics to evaluate and guide profile improvement.

3. Extensive experiments validate the proposed method, confirming its effectiveness and robustness.

### Weaknesses
1. While some algorithms are mentioned in the appendix, key details regarding their implementation and operation are not sufficiently clear. 
   
2. The experiments are conducted only on two closed large language models (LLMs), which limits the generalizability of the findings. The exclusion of open-source models prevents a broader evaluation of the proposed method's effectiveness across diverse models. 
   
3. This paper primarily considers agent profiles as dynamic representations of evolving capabilities. While this focus is valuable, it may constrain the system's overall ability to adapt and improve.

### Questions
1. How do the autonomous agents collaborate to solve tasks? Is this collaboration sequential, or is there another coordination strategy involved? Additionally, how and where do auxiliary agents contribute? I couldn’t find any difference between autonomous agents and auxiliary agents in the algorithm in appendix A.

2. You propose three metrics for profile evaluation and optimization. Could you clarify how these numerical metrics, as optimization objectives, directly guide profile optimization? Is there a curve or trend showing the progression of these metrics through iterations of profile improvements? 

3. You mentioned that during the warm-up phase, profile initialization and iterative optimization are performed. Why is this phase necessary? How do profile updates during the warm-up phase differ from those during task execution?

4. In Section 3.2, within the definition of **SKILL**, what does \[s\] represent? It’s described as a "skill prototype," but this term is unclear. How do you obtain the set of potential skill tokens, \[PS(p)\]? Could you provide some examples for clarification? And regarding the definition of **TRAS**, how are \[v_{complex}\], \[v_{simple}\], and \[v_{capable}\] determined? Are these values pre-defined representations or are they calculated dynamically?

5. In Experiment 4.1, you compare your method with three baselines, and in Experiment 4.3, you compare it with Agentverse. However, Agentverse is not included in your main experiments. I would like to know why this is the case.

6. In Experiment 4.2, you evaluate performance on domain shift. Each dataset consists of 50 sequences, with each sequence representing a shift between different domains. In Table 1, two numbers are provided for each paradigm: the first likely represents accuracy before the domain shift, while the second represents accuracy after the shift. How did you obtain these two accuracy results? Do they represent results from different sequences, or are they overall results from the mixed dataset? I would like to know which specific data were used to obtain these two results.

7. In Experiment 4.3, you evaluate performance on robustness. How do you simulate potential node failures? Are these simulated through handcrafted methods or other approaches?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces MORPHAGENT, a novel framework for decentralized multi-agent collaboration that enhances problem-solving capabilities in complex tasks through self-evolving profiles and decentralized collaboration. By defining three metrics, MORPHAGENT allows agents to dynamically adjust their roles in response to dynamic task requirements and team composition changes.

### Strengths
- MORPHAGENT moves from predeﬁned roles and centralized coordination to adaptive, fully decentralized coordination.
- It defines three metrics to measure the guide the agent profile design.
- Experiments on three benchmarks and ablation studies demonstrates improvements.

### Weaknesses
Frankly speraking, the paper's core contribution lies in the definition of three key metrics—Role Clarity Score (RCS), Role Differentiation Score (RDS), and Task-Role Alignment Score (TRAS)—to optimize agent profiles within a decentralized multi-agent system. 
I feel that this contribution is more like a prompting engieering technique, not enough to be an innovative point in an ICLR paper.

### Questions
- Can you provide more details on how those three metrics are used to optimize the profiles, as this seems to be unclear from the current manuscript?
- Why choose CodeBench, BigBenchHard, MATH? I feel that HumanEval[1] and MBPP [2] are also worth testing. Please justify your choice of benchmarks and explain why you believe these are sufficient or most appropriate for evaluating their method.
- The paper mentioned that the method rely on predeﬁned roles and centralized coordination, e.g. AgentVerse[3], MetaGPT[4], would fail in dynamic, unpredictable environments, but those methods were not selected as the baselines. Although AgentVerse was selected in the robustness comparison, I would like to see the full comparison in Figure 3.

[1] Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H.P.D.O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G. and Ray, A., 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.
[2] Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q. and Sutton, C., 2021. Program synthesis with large language models. arXiv preprint arXiv:2108.07732.
[] Hong, S., Zheng, X., Chen, J., Cheng, Y., Wang, J., Zhang, C., Wang, Z., Yau, S.K.S., Lin, Z., Zhou, L. and Ran, C., 2023. Metagpt: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352.

### Soundness
2

### Presentation
3

### Contribution
2
