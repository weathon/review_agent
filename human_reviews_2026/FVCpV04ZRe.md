# From EduVisBench to EduVisAgent: A Benchmark and Multi-Agent Framework for Reasoning-Driven Pedagogical Visualization

- Avg Score: 4.40
- Decision: Accept (Poster)
- Scores: 6, 2, 2, 6, 6

## Abstract
While foundation models (FMs), such as diffusion models and large vision-language models (LVLMs), have been widely applied in educational contexts, their ability to generate pedagogically effective visual explanations remains limited. Most existing approaches focus primarily on textual reasoning, overlooking the critical role of structured and interpretable visualizations in supporting conceptual understanding. To better assess the visual reasoning capabilities of FMs in educational settings, we introduce EduVisBench, a multi-domain, multi-level benchmark. EduVisBench features diverse STEM problem sets requiring visually grounded solutions, along with a fine-grained evaluation rubric informed by pedagogical theory. Our empirical analysis reveals that existing models frequently struggle with the inherent challenge of decomposing complex reasoning and translating it into visual representations aligned with human cognitive processes. To address these limitations, we propose EduVisAgent, a multi-agent collaborative framework that coordinates specialized agents for instructional planning, reasoning decomposition, metacognitive prompting, and visualization design. Experimental results show that EduVisAgent substantially outperforms all baselines, achieving a 40.2% improvement and delivering more educationally aligned visualizations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces EduVisBench, a benchmark designed to systematically evaluate the pedagogical visualization capabilities of foundation models such as diffusion models and LVLMs. The study reveals that existing models struggle with visual reasoning, semantic alignment, and text–graphic coherence. To address these issues, the authors propose EduVisAgent, a multi-agent collaborative framework comprising agents for task planning, conceptual mapping, reasoning decomposition, metacognitive review, and visualization design. Experimental results show a 40.2% improvement over state-of-the-art baselines, demonstrating superior pedagogical coherence, logical structuring, and interactivity.

### Strengths
1. The paper identifies a genuine research gap in the pedagogical visualization ability of foundation models.

2. Results across multiple STEM domains convincingly demonstrate performance gains with detailed quantitative metrics.

3. The five-dimension rubric provides a reproducible and extensible evaluation standard.

### Weaknesses
1. Limited real-world validation: While grounded in educational theory, no classroom-level or human-teacher evaluation supports pedagogical impact.

2. Incomplete interpretability of agent collaboration: The internal coordination among agents lacks empirical or ablation-based justification.

3. Evaluation bias risk: Heavy reliance on GPT-based automated scoring may introduce bias or circular reasoning.

4. Limited domain generalization: The benchmark focuses on STEM subjects; extension to other domains remains unclear.

5. Density and readability: The paper is information-heavy, which may reduce accessibility for general AI researchers.

### Questions
1. Are there any conflicts or redundancies among the agents? Have inter-agent dependencies been empirically analyzed or ablated?

2. Has the pedagogical efficacy been validated with human teachers or learners?

3. Could the GPT-4o-based evaluation introduce model familiarity bias toward similar architectures?

4. Will the authors release full source code and interactive visualization generation modules?

5. Can EduVisBench be extended to non-STEM disciplines or open-ended educational reasoning tasks?

### Soundness
3

### Presentation
2

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
The paper introduces EduVisBench, a new benchmark to evaluate how well AI models generate pedagogically sound, step-by-step visual explanations, and proposes EduVisAgent, a multi-agent framework that significantly outperforms existing models by coordinating specialized agents for planning, reasoning, and visualization to create more effective and interactive learning tools. However, the paper lacks completeness in two core aspects: the composition of the benchmark and the implementation details of the multi-agent system. Please refer to the Weakness section for more details.

### Strengths
1. This paper introduces the first benchmark for visualized instruction, which is one of its key contributions.
2. This paper is well written, with a clear structure.

### Weaknesses
1. The benchmark samples presented are not sufficiently representative. For example, the visualization of mathematical instruction is based on Math 500, a dataset with the difficulty level of high school math competitions. However, the left panel of Figure 3 only shows a simple addition problem (7 + 9), which is not adequate or representative.
2. The paper lacks a concrete description of the process for converting text problems into image problems, for example how prompts are designed and configured.
3. Regarding the proposed multi-agent method, although it is stated to consist of multiple agents, the paper does not detail the prompt design for each agent or which models were used. A careful check of the appendix revealed no such information—this is a significant omission in terms of completeness.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
- The authors propose a methodology to address the limitations of existing generative models in creating effective visual explanations.  
- They introduce EduVisBench, a metric for evaluating the educational score of visual materials.  
- They successfully utilize the EduVisAgent multi agent framework to generate high quality educational visualization data.  
- EduVisAgent achieved a significant performance improvement over existing models through the collaboration of agents with instructional strategies.

### Strengths
- Developed EduVisBench, a benchmark with richer information compared to datasets from existing generative models.  
  - Successfully validated the reliability of the LLM-based automatic evaluation system using human assessments shown in Table 2.  
- Utilized five specialized multi agents to implement strategies.  
- Tested broad generalization capabilities across three major academic domains: mathematics, physics, and chemistry.

### Weaknesses
-  A detailed explanation of the dataset utilized for evaluation is required, as the content presented in Figure 3 is unclear.  
- The input prompts used for the LLM evaluation in Table 1 were not disclosed, making the verification of fairness difficult.  
- The description of each EduVisAgent is too simple, leaving the method of theory implementation unclear.  
- The description of how the theory was implemented in the system is lacking.  
Openness regarding the benchmark and the educational data is important for validating the reproducibility and reliability of the research.

### Questions
1. Are there plans to provide a public repository link for the EduVisBench dataset, and will high resolution versions of figures, such as Figure 3, be supplemented?  
2. Can you disclose the specific generation prompts used to evaluate the baseline models in Table 1 to allow for the verification of reproducibility?  
3. Is it correct that all agents, except the visualization agent, are composed of LLMs? If correct, are the authors willing to clearly disclose the specific details of the structured prompts used to implement the education theories within the LLM-based agents?  
4. Are there plans to provide an additional analysis on the mechanism by which each theoretical implementation maximizes the education effectiveness of the generated output?  
5. The baseline comparison was made against simple LLMs, is it a comparison possible against multi-agent systems or recent prompt engineering techniques?  
6. What was the primary reason for constructing a multi-agent system? Given that many recent LLMs have significantly larger input token limits, what is the performance difference when all prompts are aggregated and input to a single large LLM versus the proposed multi-agent approach?  
7. Although "six specialized expert agents" are mentioned on line 86, Section 3 EduVisAgent only has five bolded agents. What does the remaining one refer to?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces EduVisBench, a multi-domain and multi-level benchmark designed to evaluate the visual reasoning capabilities of foundation models in educational contexts. The benchmark includes diverse STEM problem sets and a fine-grained evaluation rubric grounded in pedagogical theory. The authors also propose EduVisAgent, a multi-agent framework that coordinates specialized agents for instructional planning, reasoning decomposition, and visualization design.

### Strengths
(1) The formulation of a multi-agent system specifically tailored for pedagogical visualization seems novel.

(2) The paper is well-executed, with a rigorous experimental setup involving multiple model families.

(3) The writing is clear and well-structured.

### Weaknesses
(1) While the use of GPT-4o as an automated judge is validated, it remains a single-model evaluator. Including more diverse evaluators (e.g., human teachers, multiple LVLMs) could strengthen the reliability of the scoring system.

(2) The paper does not include an ablation study to analyze the contribution of each agent in EduVisAgent. Understanding which components are most critical would help future researchers prioritize agent design.

(3) The multi-agent system is computationally intensive. A discussion of inference time, resource requirements, or potential optimizations would be useful for real-world deployment.

### Questions
(1) Could the authors provide an ablation study to show the individual contribution of each agent (e.g., removing the metacognitive reviewer or reasoning decomposition agent) to the overall performance?

(2) While automated scoring is efficient, have the authors considered a more extensive human evaluation with actual educators or students to assess the pedagogical effectiveness of the generated visualizations?

(3) What are the main practical challenges in deploying EduVisAgent in real educational settings (e.g., latency, integration with LMS, adaptability to different curricula)?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the critical and underexplored challenge of generating pedagogically effective visual explanations using foundation models. The authors argue that existing models, while proficient in textual reasoning, fail to create structured, interpretable visualizations that support conceptual understanding in educational contexts.
To address this, the paper presents two primary contributions:

1. EduVisBench: A comprehensive benchmark for evaluating the pedagogical visualization capabilities of FMs. It consists of 1,154 STEM problems across Mathematics, Physics, and Chemistry, organized by difficulty. Crucially, it introduces a fine-grained, five-dimensional evaluation rubric  grounded in pedagogical principles.

2. EduVisAgent: A novel multi-agent collaborative framework designed to excel at this task. Inspired by expert instructional design, the framework coordinates five specialized agents to systematically decompose a problem, structure the reasoning process, and generate a coherent, interactive, and visually grounded solution.

Through extensive experiments on EduVisBench, the authors demonstrate that existing state-of-the-art FMs and LVLMs perform poorly. In contrast, their proposed EduVisAgent achieves an average score of 81.6%, representing a substantial 40.2% relative improvement over the best-performing baseline, validating the effectiveness of their structured, multi-agent approach.

### Strengths
1. The paper tackles a timely and important problem. As AI becomes more integrated into education, the ability to generate not just correct answers but effective teaching materials is paramount. The focus on pedagogical visualization as a distinct capability gap in FMs is novel and well-motivated.

2. The design of EduVisAgent is not an ad-hoc collection of agents but is thoughtfully grounded in pedagogical theory, mimicking the division of labor in instructional design. The performance improvement is not marginal; a 40.2% relative gain over the strongest baseline is substantial.

### Weaknesses
1. The EduVisAgent framework consists of five distinct agents. While the overall system is highly effective, the paper lacks an ablation study to analyze the individual contribution of each agent. For example, how critical is the Metacognitive Reviewer or the Conceptual Mapping Agent to the final score? Understanding the impact of each component would provide deeper insight into the architecture and help identify the most critical elements for pedagogical visualization.

2. The benchmark and agent are designed for STEM problems that typically have a clear, decomposable reasoning path. It is unclear how this framework would generalize to more qualitative or open-ended domains, such as literature, history, or social sciences, where visualization might serve to illustrate arguments, relationships, or timelines rather than step-by-step problem-solving.

### Questions
1. Could you provide more insight into the necessity of each of the five agents?

2. How do you envision the EduVisAgent framework being adapted for educational domains outside of STEM that rely on more narrative or conceptual reasoning?

### Soundness
3

### Presentation
3

### Contribution
3
