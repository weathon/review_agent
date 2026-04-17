# MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science

- Decision: Accept (Oral)
- Scores: 6, 4, 8, 8

## Abstract
We introduce MedAgentGym, a scalable and interactive training environment designed to enhance coding-based biomedical reasoning capabilities in large language model (LLM) agents. MedAgentGym comprises 72,413 task instances across 129 categories derived from 12 authentic real-world biomedical scenarios. Tasks are encapsulated within executable sandbox environments, each featuring detailed task specifications, interactive feedback mechanisms, verifiable ground truth annotations, and scalable training trajectory generation. Extensive benchmarking of 29 LLMs reveals substantial performance disparities in biomedical data science between commercial and open-source LLMs. Leveraging efficient multi-threaded and multi-turn trajectory sampling in MedAgentGym, Med-Copilot achieves performance gains of +43.02% and +45.28% from offline and online reinforcement learning, respectively, demonstrating MedAgentGym as an effective training ground while establishing itself as a cost-effective, privacy-preserving alternative competitive with proprietary LLMs (gpt-4o). By offering a unified execution environment with a comprehensive benchmark and accessible, extensible training resources, MedAgentGym delivers an integrated platform to develop LLM-based coding assistants for advanced biomedical data science.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduced MedAgentGym, a training environment for coding-based biomedical agents. It consists of three folds of contributions: (1) MedAgentGym involves 72,413 task instances across 129 categories derived from 12 biomedical scenarios. (2) This training platform allows for efficient deployment and scalable evaluation, benchmarking 29 LLMs. (3) The training data collected from MedAgentBench leads to the powerful Med-Copilot-7B/14B, which produce comparable results as the much larger proprietary LLMs.

### Strengths
S1: The preparation of MedAgentGym requires a significant amount of effort, and is a non-trivial contribution to the open-source community.  
S2: The setup in MedAgentGym is comprehensive, and the evaluation of existing LLMs on coding-based medical reasoning tasks is thorough.  
S3: Med-Copilot-7B and -14B are also very useful open-source models for medical reasoning tasks, and their training and testing setups are solid.

### Weaknesses
W1: How are the tasks studied in this paper fundamentally different from the general-purpose coding tasks? To what extent is the medical knowledge essential here? If an agent excels in general-purpose coding tasks, does it still perform well here? How do the rankings differ?  
W2: Similarly, the related work lacks a discussion of existing general-purpose coding benchmarks.  
W3: The data construction step in Section 3.2 is unclear, and particularly, it does not show the difference between the contributed benchmark and the constituent datasets from existing work. Based on Table 2, is MedAgentGym simply an ensemble of all the existing benchmark datasets? Is this paper overclaimed?  
W4: Can you provide some example instances to illustrate MedAgentGym qualitatively?

### Questions
Q1: In Line 358, how exactly do you prepare the online pairs for DPO? Isn't DPO an offline algorithm?  
Q2: What do you mean by "accurately selects successful trajectories" in Line 417? Can you explain the difference between Pass@K and Best@K? Are they metrics for the agent or for the verifier? And why does a small gap between the two metrics indicate that "the verifier can effectively identify successful trajectories"?  
Q3: What do you mean by "repeat this DPO step iteratively" in Line 443, and what do you mean by "DPO using eight new rollouts per task"? Can you explain the setup of iDPO? And why do you need "eight new rollouts per task" in addition to the 4,298 pairs?  
Q4: Can you compare the results from the self-improvement in Section 5.3 and the results using the setup in Section 5.1?  
Q5: How exactly is Figure 10 computed? Based on what features did you calculate the cosine similarity, and what does the "in/out-of-distribution" in "inter-distribution" mean? How did you determine if a task should belong to in- or out-of-distribution? What's the average number of turns and other statistics of those tasks?

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
This paper presents MedAgentGym, an extensible agentic environment and benchmark designed to systematically advance code-centric medical reasoning in LLM-based agents. It encapsulates 72,413 executable medical data science tasks spanning 129 categories derived from 12 scenarios, featuring isolated Docker-based sandboxes, ground-truth verifiers, multi-turn feedback, and scalable trajectory collection. Through comprehensive empirical evaluation, MedAgentGym is used to benchmark 29 LLMs, highlighting persistent deficits for medical code generation, and demonstrating substantial improvements via agentic reinforcement learning fine-tuning.

### Strengths
1. MedAgentGym aggregates an exceptionally broad and diverse set of medical code-centric tasks.
2. The environment is built around reproducible, interactive Docker sandboxes, allowing code execution, error handling, debugging, and dynamic dependency install, addressing reproducibility and privacy.
3. Med-Copilot exhibit strong improvements over baselines, with RL strategies yielding notable boosts, and ablation studies clarifying contributions.

### Weaknesses
1. MedAgentGym is constructed by integrating 12 existing datasets. Although it provides a division between training and test sets, the model is exposed to the task types and data patterns from these datasets during training. Therefore, its strong performance on the internal test set may partially result from memorization of specific task patterns or overfitting, rather than genuinely acquiring a universal biomedical code reasoning capability.
2. While integrating these components into a large-scale, biomedical-oriented environment represents a significant engineering contribution, the core technical concepts underlying the environment—such as the use of Docker sandboxes, provision of interactive debugging feedback, and trajectory collection for reinforcement learning—have precedents in the AI agent domain.
3. The medical-specific models evaluated in the paper, such as HuatuoGPT-o1-7B and MedReason-8B, are relatively small, with only 7B/8B parameters. Attributing their suboptimal performance solely to the limitations of medical specialization, while overlooking the substantial differences in model size, is logically flawed. A more equitable comparison would be to assess a large-scale, medically optimized model against a general-purpose large model.

### Questions
1. Can the authors clarify how different error types (see Figure 7) are incorporated into the RL reward signal? Are there distinct penalties for, say, 'stuck in the loop' vs. compile/runtime/IO errors? How sensitive is final agent performance to this reward model?

### Soundness
2

### Presentation
2

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
The paper presents a released training environment MedAgentGym for the purposes of benchmarking and training LLMs for the use of biomedical data science coding tasks. Benchmarking against many propriatary and (varying sized) open-source LLMs demonstrates the state of the field in this task. Training demonstrates impressive gains in task performance of OS LLMs, comparable to gpt-4o. The authors conduct extensive benchmarking and experiments to demonstrate the utility of their training environment.

### Strengths
* The paper is written to a good standard and certainly looks publication-ready
* The consideration of open-source LLMs for the papers setting of biomedical data science is important, as a large amount of data will be under stringent data privacy rules. Many alternative similar papers operating in this field do not consider this
* Good contribution over prior literature, encapsulating the majority of tasks that I believe would be applicable for biomedical data science
* Extensive benchmarking of many existing open-source and proprietary LLMs gives important information regarding the capabilities of such models
* Thorough experiments for model Med-Copilot. Results look promising

### Weaknesses
* I have a feeling that the the title and naming given to the training environment is slightly overstepping and too generelized. Perhaps 'BioMedAgentGym' is more suitable. 
* Since the paper is releasing a training environment for the practical real-world use of biomedical data science, I would like to see some discussion on the implications of this and reccomendations to users (please see questions below) 
* I cannot see many weakensses, though I am not familiar with the field of biomedical research nor such benchmarking papers

### Questions
* Table 3 demonstrates results of LLMs on MedAgentGym. Some "best avg. scores" (for a given LLM size) are relatively quite low. For example, the OSS <10B has Qwen3-8B at a success rate of 30.83. Given that some practitioners may only have the compute for such models, are you able to give reccomendations (complementing the writing in §4.2) regarding this? For example, what do you deem a sufficiently good performance on benchmarking for real-world deployment of a LLM?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
MedAgentGym introduces an interactive training environment for code-centric biomedical reasoning. There are numerous scenarios, with different LLMs evaluated in the environment showing gaps between especially closed vs open models. They also introduce Med-copilot trained on the env which is very strong.

### Strengths
1. Solving an important problem & provides a comprehensive benchmark on real-world medical tasks. 

2. Very rigorous evals across many tasks over many models 

3. Strong Performance Gains of Med-Copilot on the env

### Weaknesses
- Only execution evals, no assessment of intermediate reasoning and steps which is vital in medicine. Where the trajectory matters as much as the solution

- Big OOD drops unexplained on external dataset (more validation and digging). Maybe things are just overfit?

### Questions
- Can you assess some trajectories for sound reasoning vs just only correctness

- Can we know that none of the LLMs have already trained on the benchmark datasets? Maybe need some private data or new data

- To the OOD drop please can you dig in and understand why?

- Please can you add variance not just mean to understand overlap of models

### Soundness
4

### Presentation
4

### Contribution
4
