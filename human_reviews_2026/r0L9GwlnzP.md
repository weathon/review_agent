# Do LLM Agents Know How to Ground,  Recover, and Assess? Evaluating Epistemic Competence in Information-Seeking Agents

- Avg Score: 4.40
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4, 4

## Abstract
Recent work has explored training Large Language Model (LLM) search agents with reinforcement learning (RL) for open-domain question answering. However, most evaluations focus solely on final answer accuracy, overlooking how these agents reason with and act on external evidence.
We introduce **SeekBench**, the first process-level evaluation framework for LLM search agents that operationalize *epistemic competence* through metrics derived from an annotation schema.
We develop and validate our annotation schema using an expert-annotated dataset of 190 traces (over 1,800 steps). 
To evaluate at scale, we introduce an LLM-as-judge pipeline.
Our framework provides granular analysis of whether agents demonstrate: (1) **groundedness**, by generating reasoning steps supported by observed evidence; (2) **recovery**, by adaptively reformulating searches to recover from low-quality results; and (3) **calibration**, by correctly assessing whether current evidence is sufficient to provide an answer. 
By applying our evaluation framework to state-of-the-art search agents tuned on Qwen2.5-7B, we uncover critical behavioral gaps that answer-only metrics miss, as well as specialized skills such as Search-R1's synthesis abilities. These analyses highlight distinct epistemic competencies, offering actionable insights for the development of more capable and trustworthy agents. Code is available at
https://github.com/SHAO-Jiaqi757/SeekBench.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SeekBench, a benchmark to evaluate LLM search agents on epistemic competence of three dimensions/metrics: reasoning groundedness, search recovery, and answer calibration.

### Strengths
1. The proposed benchmark is well-designed, and the methodology is well-elaborated.
2. The evaluation is extensive on seven QA datasets, using multiple LLM search agents.
3. Further analysis provides insights into the differences between LLM agents regarding the proposed metrics/capabilities.

### Weaknesses
- Some concerns regarding methodology, experiments, and analysis. Please refer to "Questions" below.

### Questions
1. What are the raw data source(s) and domain(s) for constructing SeekBench?
2. What does each annotated feature in Table 1 mean?
3. To clarify, the human/LLM annotators will label the correctness of each trace i, as well as the clarity and quality of the retrieved evidence at each turn t in each trace, right? What exactly is "quality" defined here?
4. Section 3.3.3 Definition 3.5 (Calibration Error (CE)): Why should the ideal policy answer if and only if the evidence is sufficient? Whether the explicit evidence is sufficient or not also depends on the internal knowledge of the policy, so it would also be acceptable if the policy answers correctly without much retrieval, especially for easy questions.
5. Section 4 "We evaluate the agents on a diverse set of seven question-answering benchmarks": Does this mean the original QA datasets are transformed as SeekBench-style benchmarks using the methodology (annotation process) described in Section 3? If so, as many QA datasets do not provide reasoning traces, where do the original traces and steps come from? Why not evaluate the agents on the proposed SeekBench ("SeekBench comprises 190 expert-annotated traces with over 1,800 response steps")?
6. Section 4 analyzes the differences between LLM agents regarding the proposed metrics/capabilities, but what causes such differences? Is it because of the training schema or other aspects of different LLM agents? What do the comparisons reveal? What are the insights into improving the current LLM agents based on the findings of SeekBench?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SeekBench, an evaluation framework for analyzing the intermediate reasoning and search process of search-based LLM agents. The authors construct a framework centered on 3 competencies (groundedness, recovery, and calibration), proposing various cases of each competencies and associated metrics.

### Strengths
- Evaluation beyond outcome is an important topic: Validating that the processes search agents take in finding the final answer is necessary for reliability, transparency, and can be a valuable feedback signal for improving future agents.
- For search-based agents, the creation of the competencies, annotated features, and metrics seems reasonable, involving three rounds of review with humans and LLM-as-judges. This results in an evaluation schema that is both principled and scalable, with automatic evaluators exhibiting high levels of aggregate-agreement with humans.
- The analysis identified clear shortcomings of RL-trained search agents: outcome-based RL training leads to better calibrated agents (i.e., agents tend to produce answers when the evidence state is both clear and sufficient to answer the question). However, the intermediate reasoning steps produced by agents are not necessarily grounded, reflecting the outcome-driven nature of RL training.

### Weaknesses
- In my opinion, many parts of the abstract and introductory sections are insufficiently clear, and largely have to do with the repeated statement that SeekBench is a process-level benchmark consisting of 190 traces with 1800+ annotated steps (i.e., in abstract, L88, L97). This framing lead me to believe that SeekBench was a benchmark for evaluating process-level evaluators, like those present in math domains, e.g., ProcessBench. **In reality, an annotation schema is derived from said traces, and used to validate automatic evaluation approaches, like LLM-as-judges**. This should be made clearer to the reader, especially as the evaluation process in Section 4 does not really directly use these annotations.
- The paper seems to only analyze search agents that can use a single web-search tool. This may not be representative of state-of-the-art search agents, e.g., [1,2], which are capable of multiple tools, such as using a diverse array of tools like code interpreters or more complex forms of search like web browsing. It’s unclear how SeekBench analysis holds when agents are given toolkits more representative of real-world use-cases. I don’t think states with other tools can easily be folded into existing competencies, nor do I think the existing framework can be applied directly to other tools.
- Evaluated baselines: 
  - It seems that only agents of 7B model size are evaluated, even though many baselines have larger, more capable model variants. For example, ASearcher has 14B and 32B variants, trained from Qwen2.5-14B and QwQ-32B, respectively. 
  - Furthermore, newer, more capable tool-calling baselines, such as WebSailor [2] (and v2 variant) are missing. These missing baselines may be a result of the above weakness, as they are typically trained to use multiple tools.
  - Lastly, API models with search tool-calling ability (e.g., OpenAI models) are not analyzed.
  - This weakness is critical in my opinion. Because this paper mainly performs analysis on the behavior of search-based agents, it should aim to cover as many search agents are reasonably possible. But a critical dimension, agent model size, is missing! How many of the unearthed agent shortcomings can be mitigated or solved simply by scaling up the agent size?


[1] https://arxiv.org/abs/2509.06283

[2] https://arxiv.org/abs/2507.02592

### Questions
- LLM-as-judge agreement with human annotations is presented at an aggregate level, without per-metric breakdowns. What is the per-metric breakdown? Are judges more reliable for certain metrics? If yes, can we still reliably trust judges for **all** metrics?
- What is approximate cost of annotating one search agent in terms of LLM-as-judge API calls? It seems that each intermediate step in an agent requires multiple API calls.
- The authors have a unique opportunity to more thoroughly benchmark LLM-as-a-judge models on the human annotations to determine the best model based on cost-performance tradeoffs. Why were GPT-4.1, 4.1-mini, and GPT-5 chosen? Why not evaluate GPT-5-mini,nano, or models like Claude Sonnet, Gemini 2.5 Pro, or more capable local models that are “free” (in terms of API cost)? Further, what reasoning effort was GPT-5 evaluated with? To inform practitioners, how does this performance vary with reasoning effort setting?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the SeekBench benchmark: the first process-level benchmark for LLM search agents, decomposing agent traces into four steps (reasoning, searching, evidence, and answering). It establishes an operational framework with metrics, formalizing three core cognitive abilities—"evidence-based reasoning," "adaptive evidence recovery," and "evidence-aligned calibration"—into measurable indices and designs precise, interpretable metrics to quantify these capabilities. Based on these metrics, the evaluation is conducted across seven QA benchmarks (28,493 traces), revealing that reinforcement learning (RL) agents excel in evidence gathering but exhibit weak reasoning. Additionally, the study highlights that standard accuracy metrics fail to distinguish specific agent strengths (e.g., Search-R1's synthesis capability vs. base models' reasoning ability), while combining agents with complementary strengths improves overall performance.

### Strengths
1. Addressing the limitation of "answer-only" agent evaluation, this work focuses on cognitive behaviors (e.g., evidence usage, uncertainty decision-making) during retrieval and reasoning, offering a more comprehensive assessment aligned with real-world agent capability requirements.
2. It introduces theoretical concepts like "evidence states" alongside concrete, quantifiable metrics. Large-scale evaluations yield actionable findings (e.g., combining agent strengths enhances performance), providing practical guidance for future research and applications.

### Weaknesses
1. Lack of dataset details: The experiments mention 28,493 traces from seven QA benchmarks, and the appendix briefly outlines the dataset selection process. However, the main text (and appendix) lacks detailed sample descriptions or a dedicated section for statistical analysis of the data.
2. Insufficient validation of core metrics: The three proposed metrics—Reasoning Quality Index (RQI), Evidence Recovery Function (ERF), and Calibration Error (CE)—are inadequately justified, undermining their validity and innovation. Key gaps include:

    1）Weak motivation: Only definitions are provided without explaining the limitations of existing metrics (e.g., accuracy, F1) in assessing cognitive behaviors or comparing them to related work. Theoretical foundations (e.g., cognitive science frameworks) are absent, weakening the design rationale.

    2）Missing case studies: A case analysis in the introduction could clarify why the three-stage division better evaluates agent capabilities.

    3）No superiority demonstration: While proposing a process-level framework, the paper fails to prove its advantages over task-bound evaluation methods—lacking dimensional comparisons (e.g., comprehensiveness, interpretability) or experiments revealing overlooked flaws. The claim that "three-stage evaluation is superior" lacks validation.
3. Underdeveloped agent combination insight: The finding that combining agent strengths improves performance lacks deeper exploration—no discussion of viable strategies, applicable scenarios, or optimization potential.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SeekBench, the first benchmark for evaluating the epistemic competence of Large Language Model (LLM) search agents through step-level analysis of their reasoning traces. SeekBench comprises 190 expert-annotated agent traces generated by LLM agents on open-domain QA tasks, and formalize three metrics (Groundedness, Recovery, Calibration).

### Strengths
The paper obtains some findings based on SeekBench such as 1) RL-trained agents achieve higher answer accuracy but exhibit lower reasoning groundedness than few-shot or base models. 2) All agents struggle with Plan Formation and State Assessment, though they perform better at Information Synthesis.

### Weaknesses
- The current epistemic metrics (Groundedness, Recovery, Calibration) rely on binary judgments (e.g., grounded vs. not grounded), which may overlook nuanced or intermediate reasoning states—such as partially supported claims or ambiguous evidence.

- As illustrated in the paper, RL typically optimizes only for the correctness of the final answer, while ignoring the goals at the process level. If adding supervision signals during the RL training process can solve this problem, how can the cognitive abilities proposed in the paper be integrated into the training process?

- The current trace schema focuses on textual search and reasoning steps but does not account for richer agent behaviors—such as invoking code interpreters, calling external APIs, browsing web pages, or analyzing structured data (e.g., tables, JSON). it is important to enhance its applicability of SeekBench to real-world  search agentic systems.

- While the expert-annotated subset (190 traces) ensures high-quality grounding, its limited size may hinder coverage across diverse question types and failure modes. Moreover, scaling annotations via LLM-as-judge—despite strong inter-annotator agreement—risks propagating systematic biases or blind spots inherent in the judge model itself, especially on adversarial examples.

### Questions
What is the specific token consumption for LLM-as-judge?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces SeekBench, a benchmark for evaluating the epistemic competence of LLM-based search agents. Rather than judging agents solely by final-answer accuracy, SeekBench analyzes step-level response traces—reasoning, search, evidence, and answer—to assess how agents acquire, evaluate, and act on knowledge. The benchmark enables fine-grained diagnostics of three core competencies: (1) evidence-grounded reasoning, (2) recovery, and (3) calibrated answering. Using this schema, the authors design interpretable metrics and show that widely used accuracy-only evaluations can mask important behavioral differences. SeekBench thus provides a principled, scalable framework, supported by high human agreement and aligned LLM judges.

### Strengths
1. Novelty of the research question. I do think the step-by-step debugging of the retrieval-intensive search agents is an important yet unexplored direction in current answer-only evaluation frameworks.

2. Clarity of the proposed framework. The framework seems a bit complicated (since it contains many evaluation aspects), but well documented, making it easy to understand the overall process and the details of each evaluation component.

3. Effective visual communication. Figures and plots are of high quality and well chosen. These improve interpretability and aid readers’ comprehension of the empirical findings.

### Weaknesses
1.  Rationale behind the metric

I get the rationale behind the metric and do think they are well-designed to capture groundness, recovery, and calibration. However, a few questions remain. 

First, why do we have to measure the groundness of an insufficient evidence state (E=0)? In other words, in Eqn 4, the RQI is decomposed w.r.t. evidence states E while accounting for E=0. Isn't it more reasonable to only consider E=1 and E=2, since groundness on insufficient or incorrect evidence is not meaningful? 

Second, in Eqn 8 to define ERF, why not account for the length of the traces T_i? In my point of view, adding 1/T_i term seems more natural if the goal is to measure "how fast the agent can escape from poor evidence", since it then becomes independent of the trace length. For example, if there are two gold traces, A with 3 steps and B with 10 steps, having both ERF(3), the meaning can be very different since it can mean faster escape in trace B than in trace A.


2. Unclear points on identifying failure patterns & Lack of human annotators

In Section 3.1, the authors seem to identify failure patterns in agents' traces to develop the benchmark evaluation scheme. However, which models & agents have been observed is unclear. Are the models in different sizes & different families (Llama, GPT, Qwen ... ) accounted for, or is only Qwen used (the ones used in the experiments)? Furthermore, three human annoataors seems insufficient to support human aggreement with LLM judges.


3. Lack of experimental baselines


The paper only evaluates two Qwen models, which limits the effectiveness of experimental results to support the claims. Also, while the paper compares "base", "few-shot", and "RL-trained" methods, prompting methods such as CoT and ReAct should be included.


4. Insufficient cost analysis

While using LLM-as-Judge provides a scalable approach (and high human agreements according to the paper's claim), it may incur a significant cost. The paper should include a cost analysis on time and API costs (in case of using proprietary models like GPT-4 or GPT-5 as used in the paper) and discuss potential limitations.

### Questions
1. Can you provide the examples of poor/partial/good evidences? I checked the actual prompts in Appendix, but I think it can be very subjective (especially for partial evidences) even for humans. I assume there can be some gray areas to divide these categories, which should be discussed in th paper.

### Soundness
3

### Presentation
3

### Contribution
2
