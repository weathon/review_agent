# PCEval: A Benchmark for Evaluating Physical Computing Capabilities of Large Language Models

- Avg Score: 5.50
- Decision: Reject
- Scores: 2, 4, 8, 8

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains, including software development, education, and technical assistance. Among these, software development is one of the key areas where LLMs are increasingly adopted. However, when hardware constraints are considered—for instance, in physical computing, where software must interact with and control physical hardware —their effectiveness has not been fully explored. To address this gap, we introduce PCEVAL (Physical Computing Evaluation), the first benchmark in physical computing that enables a fully automatic evaluation of the capabilities of LLM in both the logical and physical aspects of the projects, without requiring human assessment. Our evaluation framework assesses LLMs in generating circuits and producing compatible code across varying levels of project complexity. Through comprehensive testing of 13 leading models, PCEVAL provides the first reproducible and automatically validated empirical assessment of LLMs’ ability to reason about fundamental hardware implementation constraints within a simulation environment. Our findings reveal that while LLMs perform well in code generation and logical circuit design, they struggle significantly with physical breadboard layout creation, particularly in managing proper pin connections and avoiding circuit errors. PCEVAL advances our understanding of AI assistance in hardware-dependent computing environments and establishes a foundation for developing more effective tools to support physical computing education.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces PCEVAL, a benchmark for evaluating large language models (LLMs) in physical computing tasks—where software must interact with and control hardware. Unlike prior benchmarks focusing only on logical or coding abilities, PCEVAL enables fully automated and reproducible evaluation of LLMs in both logical and physical aspects of hardware-related projects, eliminating the need for human assessment.

### Strengths
The paper presents a benchmark for evaluating LLM in physical computing tasks.

### Weaknesses
1) This work focuses on physical computing but primarily adds components related to breadboard and logic design. The breadboard design tasks are relatively simple and not strongly connected to embedded system design. Moreover, even in cases where breadboard design is necessary, such tasks could likely be automated using traditional algorithms, and it remains unclear why an LLM-based approach is required here.

2) Regarding the logic design aspect, LLM-based methods in this area have already been extensively explored, with numerous existing benchmarks available. The contribution of this work therefore appears to be largely a combination of prior benchmarks. The key distinctions between this benchmark and existing logic design benchmarks should be clearly articulated.

### Questions
Please see the weakness.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a benchmark for evaluation the physical computing capabilities of large language models. Specifically different from prior work, the benchmark also includes new tasks such as logical circuit generation and physical circuit generation (physical circuit layout breadboard implementations). and automatic evaluation protocal. The authors also evaulated on 13 leading models.

### Strengths
(1) Novel and well-motivated benchmark addressing previously unexamined domain. Clear task decomposition and fully automated evaluation framework ensuring reproducibility

### Weaknesses
(1) Evaluations lack variance measures, such as pass@k metrics etc. LLMs results could be noisy under temperature sampling.

(2) The benchmarks scope is largely Arduino-centric. This raises question about generality and difficulty of the task on real-world example use cases (other than for educational purposes).

(3) It seems the LLM constently make mistakes on problems easily to correct (i.e. physical contraint violation). Would incorporating such feedback in a agentic framework improve the results?

### Questions
(1) Some aspects of the automated validation pipeline are insufficiently detailed for replication. Will the benchmark be open-sourced?

(2) Can the authors present more metrics on evaluation such as pass@k etc.?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors propose PCEval, the first reproducible and verifiable assessment suite to test LLMs in the context of designing physical computing for educational purpose. Empirical evidence suggests that large language models are still insufficient in relevant tasks. It means that we have to use LLMs with caution in physical computing, and more attention should be devoted to this area.

### Strengths
1. The investigated question is novel and interesting: how good LLMs are in physical computing for educational purposes. It is well motivated. With the development of LLMs in many fundamental tasks such as reasoning, it is important to understand how useful they are in real-life tasks.

2. The study design is carefully constructed. It starts by interviewing multiple CS educators about what problems are critical in their context, making sure that the investigated problems are relevant to real-world deployment.

3. The proposed framework is scalable and verifiable. This is important in the future impact of this work in relevant and broader areas.

4. The presentation is very clear and easy to follow and understand.

### Weaknesses
Mitigation methods can be more thoroughly discussed. For instance, why CoT works on some models but not others? Is there any method to improve model performance instead of simply prompting?

### Questions
Please refer to weaknesses above.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces PCEval, the first benchmark for evaluating the capabilities of large language models in physical computing-that is, reasoning and generating both logical and physical circuits, as well as the code that operates them.
The benchmark breaks down physical-computing reasoning into four tasks:
1. Logical circuit generation (D,C--L), 2. Physical circuit generation (D,C--P), 3. Code generation from logical circuits (D,L--C), 4. Code generation from physical circuits (D,P--C). 
PCEval includes 50 Arduino-based projects spanning four complexity levels that can be fully executed in simulation using the Wokwi environment.  Each project includes a test procedure for automated validation, which eliminates subjective expert judgment common in previous works such as MICRO25 or EmbedTask. 
Results from 13 leading models show that while LLMs perform reasonably on code generation (60-70% success), they struggle dramatically with physical layout generation (<10% success for most models).

### Strengths
- Novel Evaluation Dimension
The paper precisely identifies the missing capability in current LLM evaluation: the ability to reason about and execute tasks that require physical computing.

- Reproducible Evaluation Pipeline
The benchmark's use of fully automated simulation (Wokwi) ensures objective, quantitative validation of the generated circuits and code. This contrasts with previous work (e.g., EmbedTask, MICRO-25), which relied on subjective human grading or partial execution tests. The inclusion of physical-level error metrics (pin conflicts, bypass errors, and isolated components) provides an unusual level of granularity and makes the results understandable to engineering reseachers.

- Comprehensive Model Coverage and Systematic Analysis
The evaluation covers 13 prominent LLMs, including both closed-source (GPT-4o, o3-mini, Gemini-2.5-Pro, Claude 3.7) and open-source (Mistral-Large, Qwen-VL-Max, Llama-3-70B) systems.  Consistent prompting and multi-trial runs produce a reliable cross-model comparison. The analysis goes beyond raw accuracy to reveal failure modes by reasoning stage (for example, layout versus logic consistency), which is extremely useful for model diagnosis.

- Educational and Practical Grounding
Interviews with eight computer science educators guide the benchmark's development, ensuring that the tasks chosen reflect realistic student exercises (sensor integration, actuation control, sequential logic).  This foundation strengthens the benchmark's authenticity and demonstrates its potential educational impact( for example, as a diagnostic or formative assessment tool in engineering courses).

-Exploration of Self-Corrective Techniques
The inclusion of self-improvement and chain-of-thought prompting (resulting in a +10-18% improvement) shows that the benchmark can directly stimulate methodological research rather than serving solely as a static evaluation dataset.

### Weaknesses
- Dataset's Scale and Diversity Limitations
With only 50 projects, PCEval's coverage is limited when compared to large-scale code or reasoning benchmarks. The tasks are primarily for introductory Arduino applications (LEDs, sensors, and servos), with less emphasis on other topics such as real-time signal processing.

-Limited Comparison
Lack of comparisons with classical or hybrid design-automation systems (e.g., symbolic circuit solvers, search-based algorithms). This makes it hard for the audience to contextualize LLM weaknesses in relation to domain-specific benchmarks. Such comparisons could help determine whether failures are due to reasoning limitations or a lack of embedded knowledge.

-Evaluation Fairness  
Details about prompt standardization (temperature, token limit, and input modality) are not fully specified. Such parameters have a significant impact on model rankings, particularly when comparing multimodal and text-only systems.

### Questions
Dataset Balance & Expansion - Given 50 projects, how do you ensure adequate representation of diverse physical computing paradigms (sensing vs actuation control)? Are there plans to scale beyond Arduino while preserving automated validation?

Evaluation Fairness and Reproducibility - Were all LLMs prompted with the same temperature, context length, and output format constraints? How sensitive are results to prompt reformulation (particularly in D,P→C tasks)?

Self-improvement and COT Prompts - How was the iterative refinement protocol developed (failure-log feedback schema)? Did any models show overfitting or oscillatory corrections during the multi-turn refinement?

In the focus-group study, were educators asked to rank task readability or error traceability? Could the authors release annotated examples labeled by usability to facilitate future "AI-pedagogy alignment" studies?

### Soundness
2

### Presentation
2

### Contribution
3
