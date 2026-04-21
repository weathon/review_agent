# Agent-as-a-Judge: Evaluating Agents with Agents

- Avg Score: 5.67
- Decision: Reject
- Scores: 6, 6, 5

## Abstract
Contemporary evaluation techniques are inadequate for agentic systems. These approaches either focus exclusively on final outcomes---ignoring the step-by-step nature of the thinking done by agentic systems---or require excessive manual labour. To address this, we introduce the **Agent-as-a-Judge** framework, wherein agentic systems are used to evaluate agentic systems. This is a natural extension of the LLM-as-a-Judge framework, incorporating agentic features that enable intermediate feedback for the entire task-solving processes for more precise evaluations. We apply the Agent-as-a-Judge framework to the task of code generation. To overcome issues with existing benchmarks and provide a proof-of-concept testbed for Agent-as-a-Judge, we present **DevAI**, a new benchmark of 55 realistic AI code generation tasks. DevAI includes rich manual annotations, like a total of 366 hierarchical solution requirements, which make it particularly suitable for an agentic evaluator. We benchmark three of the top code-generating agentic systems using Agent-as-a-Judge and find that our framework dramatically outperforms LLM-as-a-Judge and is as reliable as our human evaluation baseline. Altogether, we believe that this work represents a concrete step towards enabling vastly more sophisticated agentic systems. To help that, our dataset and the full implementation of Agent-as-a-Judge will be publically available at [REDACTED]

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces "Agent-as-a-Judge," a framework that uses AI agents to evaluate other AI agents' code generation capabilities, extending the existing LLM-as-a-Judge paradigm. The authors also present DevAI, a new benchmark dataset containing 55 AI development tasks with hierarchical requirements, designed to test code-generating AI systems. They evaluate three popular open-source code generation agents (MetaGPT, GPT-Pilot, and OpenDevin) using both human evaluators and their Agent-as-a-Judge system. Their results show that Agent-as-a-Judge achieves comparable performance to human evaluators (90% alignment with human consensus) while being significantly more cost-effective, reducing evaluation time by 97.72% and costs by 97.64% compared to human evaluation. The paper demonstrates that automated evaluation of AI agents is feasible and can potentially scale up the development of more sophisticated AI systems.

### Strengths
1. Using agent to evaluate the agent's performance is very interesting and important to advance the development of the foundation agent. This paper addresses a critical and growing challenge in AI: how to effectively evaluate increasingly complex AI agents. 

2. Thorough experimental design with multiple levels of evaluation, including comprehensive ablation studies to understand component contributions, careful analysis of human evaluation biases and errors and clear comparison between human evaluators, LLM-as-a-Judge, and Agent-as-a-Judge

3. The proposed DevAI is a good benchmark, which has carefully designed hierarchical requirements, includes explicit dependencies between requirements, incorporates both mandatory requirements and optional preferences, and focuses on realistic AI development tasks

### Weaknesses
My main concerns of the paper are:

1. The limited technical contributions. This agent-as-a-judge is a natural extension of LLM-as-a-judge, and this is more like an engineering project. 

2. The proposed DevAI is small scale with only 55 tasks, which is insufficient for a comprehensive benchmark. Though low cost is a good point, this would also over-simplify the tasks with limited task complexity and diversity.

### Questions
Here are some questions:

1. Could you clarify the technical contributions of this agent-as-a-judge? How this method differ from LLM-as-a-judge? 

2. Could you clarify about the succeffiency of the tasks in DevAI? More tasks and evaluations should be evaluated. 

3. Does this method generalize to novel tasks, which is out of the considered benchmark tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper "Agent-as-a-Judge: Evaluating Agents with Agents" addresses the inadequacy of traditional evaluation techniques for assessing agentic systems, which require more sophisticated, step-by-step feedback mechanisms. The authors propose the Agent-as-a-Judge framework, a novel approach that employs agentic systems to evaluate other agentic systems, integrating capabilities to provide intermediate feedback throughout the task-solving process for more precise evaluations.

The paper introduces a new benchmark, DevAI, to demonstrate and validate the proposed framework. DevAI comprises 55 realistic AI code generation tasks, complete with detailed manual annotations, making it ideal for agentic evaluators. The authors benchmark three leading open-source code-generating systems—MetaGPT, GPT-Pilot, and OpenDevin—using the proposed framework. Their experiments reveal that the Agent-as-a-Judge framework outperforms the LLM-as-a-Judge method and matches the reliability of a human evaluation baseline.

The primary contributions of this work include:
1. The release of the DevAI benchmark, which consists of 55 comprehensive AI development tasks with accompanying tags, hierarchical requirements, and preferences, designed to enhance the evaluation of agentic systems.
2. The introduction of the Agent-as-a-Judge framework, an innovative method for evaluating agentic systems using other agentic systems, providing rich, intermediate feedback for more accurate evaluations that align closely with human evaluators while significantly reducing time and cost.

### Strengths
**Originality:**
The paper introduces the novel benchmark DevAI, which comprises 55 realistic AI code generation tasks with comprehensive manual annotations and hierarchical solution requirements. This benchmark addresses the gap in existing evaluation methods and is a significant contribution to the AI development community. The introduction of the Agent-as-a-Judge framework is another innovative aspect, proposing a novel method for evaluating agentic systems using other agentic systems. This extends the existing LLM-as-a-Judge framework by incorporating capabilities to provide intermediate feedback, thereby enhancing evaluation precision.

**Quality:**
The quality of the research is evident in the thorough experimental setup and robust analysis. The authors have benchmarked three leading open-source code-generating systems—MetaGPT, GPT-Pilot, and OpenDevin—using the proposed framework. They conducted experiments across various settings, including black-box and gray-box scenarios, as well as independent and task-dependent evaluations. These comprehensive experiments validate the effectiveness of the Agent-as-a-Judge framework and underscore its superiority over traditional methods. The detailed statistical analysis and alignment with human evaluators further reinforce the reliability of the proposed method.

**Clarity:**
The paper is well-structured and clearly presents its ideas and contributions. The introduction effectively outlines the motivation and objectives, while subsequent sections provide a detailed explanation of the DevAI benchmark and the Agent-as-a-Judge framework. The methodologies and experiments are described with clarity, making it easy for readers to follow and understand the research. Additionally, the inclusion of figures and tables enhances the presentation by providing visual insights into the experimental results.

**Significance:**
The significance of the paper lies in its potential to transform the evaluation of agentic systems. By introducing a novel and possibly more sound method—Agent-as-a-Judge—the paper addresses the limitations of traditional evaluation techniques that focus solely on final outcomes or require excessive manual labor. The DevAI benchmark, coupled with the Agent-as-a-Judge framework, provides a comprehensive and efficient approach for assessing the performance of agentic systems. This work lays the groundwork for more precise, intermediate evaluations, which can significantly accelerate progress in the development and deployment of sophisticated agentic systems.

### Weaknesses
**Human Evaluation Methodology :**
The use of only three human experts as judges raises concerns about the robustness and reliability of the evaluation process. Given that these experts are authors of the paper, there is a potential for bias, and the large disagreements observed among them further question the dependability of the evaluation. To improve this aspect, the study could benefit from recruiting a larger and more diverse pool of external evaluators. Even applying a subset of tasks to a larger group of judges would offer more statistically reliable results and help validate the labels in the DevAI benchmark. This would ensure that the benchmark has a more concrete "ground truth", thus making the subsequent results more robust and credible.

**Inconsistent Statements on Human Evaluation Reliability :**
There are conflicting statements regarding the reliability of human evaluations. In Section 4 (Line 366), it is mentioned that "Human evaluation, while somewhat reliable," which seems inconsistent with the prior conclusions in Section 3.4 (Line 354) "Human judgment errors are inevitable" and the observations in Lines 319-327 about the large disagreements among human evaluators. The paper would benefit from clearly articulating the relative reliability of various evaluation methods. If I understand correctly, the paper is expressing the relationship of reliability as LLM-as-a-Judge < Single-Human-as-a-Judge < Agent-as-a-Judge < Ensemble of Human Judges. The paper would benefit from stating this clearly.

**Clarity in Figures :**
Figure 2(1) appears unprofessional and is highly unreadable due to the mixed vertical and horizontal text as well as the small font size. Enhancing the readability and professional presentation of this figure is essential.  the authors might consider using a different type of diagram to display the word frequency more effectively.

**Citation Issues :**
There are several inconsistencies and inaccuracies in the citations. For instance, the citation of SWE-Bench in the introduction leads to SWE-Agent, which is incorrect. Moreover, some papers that have been accepted to conferences are cited in their arXiv versions (e.g., SWE-Bench, DSPy, HumanEval, AgentBench). Ensuring that all citations are accurate and up-to-date will enhance the paper's credibility. Each referenced work should be properly cited in its correct and most formal publication form when available.

### Questions
- I'm not exactly sure of the meaning of independent tasks (I) and tasks considering task dependencies (D). Does R0 belong to (I) and R1, R2 belong to (D) in Figure 1? Maybe I missed something, could you please further clarify this?
- In line 486, how is Table 3 able to demonstrate the usefulness of the "retrieve" module?
- What is your backbone model for the LLM-as-a-Judge and Agent-as-a-Judge frameworks? Does using a different backbone affect your main results?

Please also pay attention to the questions and suggestions mentioned in the "Weaknesses" section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces the Agent-as-a-Judge framework, an innovative approach for evaluating agentic systems by employing other agentic systems as evaluators. Building on prior methods like LLM-as-a-Judge, this framework provides intermediate feedback throughout the problem-solving process, moving beyond final outcome evaluation. The paper applies this approach to code generation tasks, introducing the DevAI dataset, consisting of 55 realistic development tasks with hierarchical solution requirements. The results show that Agent-as-a-Judge aligns closely with human evaluation and outperforms LLM-as-a-Judge in task accuracy and cost-effectiveness.

### Strengths
1. The introduction of the Agent-as-a-Judge framework addresses a critical gap in evaluating agentic systems by enabling feedback at each stage of task completion, making it especially suited for complex, multistep tasks.
2. The DevAI dataset provides a practical and comprehensive testbed for agentic systems, encompassing various AI development tasks that closely mirror real-world demands, which enhances the relevance of the evaluation.
3. The study compares Agent-as-a-Judge with both human evaluators and LLM-as-a-Judge, demonstrating superior alignment with human consensus while significantly reducing costs and evaluation time.
4. The paper includes extensive ablation studies, cost analysis, and performance metrics, providing insights into the efficacy of different components of the framework and supporting reproducibility.

### Weaknesses
1. The overall writing is inconsistent and confusing: in the abstract and introduction,  the Agent-as-a-Judge (AAAJ) framework is proposed with its novelty and effectiveness being emphasized. But in Section 2 and 3, the paper suddenly turns to the DevAI benchmark, introducing the motivation and technical details of this particular benchmark. The AAAJ benchmark is not comprehensively introduced until Section 4.  I feel like these two contributions, namely the AAAJ framework and the DevAI benchmark, should have been separated into two papers and discussed in more details respectively. Otherwise, it may be better to start from the DevAI benchmark, with agentic evaluation framework being one of its useful features.
2. Among the four principal contributions, the top two seem to be deviated from the main idea of this paper.
3. In Section 4, AAAJ is only discussed as a proof-of-concept, with little technical details. 
4. Code generation is only one of the many applications of agentic systems. There are plenty of other domains where LLM agents may help. It is therefore not enough to consider the task of code generation alone. The agentic evaluation framework also seems to be ad hoc to code generation.

### Questions
1. While the AAAJ framework achieves higher alignment rate than other baseline algorithms, chances are that AAAJ makes incorrect evaluations. Did you conduct any studies on the failure cases to see whether there are consistent patterns that may be exploited by agentic systems?
2. Can the AAAJ framework be evaluated in benchmarks other than code generation?

### Soundness
2

### Presentation
1

### Contribution
3
