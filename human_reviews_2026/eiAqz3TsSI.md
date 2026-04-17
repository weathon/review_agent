# O-Reseacher: An Open Ended Deep Research Model via Multi-Agent Distillation and Agentic RL

- Decision: Reject
- Scores: 6, 2, 4, 2

## Abstract
The performance gap between closed-source and open-source large language models (LLMs) is largely attributed to disparities in access to high-quality training data. To bridge this gap, we introduce a novel framework for the automated synthesis of sophisticated, research-grade instructional data. Our approach centers on a multi-agent workflow where collaborative AI agents simulate complex tool-integrated reasoning to generate diverse and high-fidelity data end-to-end. Leveraging this synthesized data, we develop a two-stage training strategy that integrates supervised fine-tuning with a novel reinforcement learning method, designed to maximize model alignment and capability. Extensive experiments demonstrate that our framework empowers open-source models across multiple scales, enabling them to achieve new state-of-the-art performance on the major deep research benchmark. This work provides a scalable and effective pathway for advancing open-source LLMs without relying on proprietary data or models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes O-RESEACHER, a framework to tackle the "deep research" performance gap between open-source and closed-source LLMs. It introduces a novel, multi-agent "divide-and-conquer" workflow to synthetically generate training data by decomposing complex queries and collecting agent reasoning trajectories. This data is used to train an open-source model via a two-stage strategy: Supervised Fine-Tuning (SFT) to learn research structure, followed by Reinforcement Learning from AI Feedback (RLAIF) with PPO. Experimental results confirm the effectiveness of O-Researcher.

### Strengths
1.	The paper tackles the highly significant challenge of democratizing the development of powerful AI research agents, directly addressing the data-access bottleneck that hinders the open-source community. 

2.	The manuscript is well-written. The design of the proposed framework is explained clearly, which makes the paper easy to understand.

3.	The "divide-and-conquer" multi-agent workflow is an innovative and scalable solution for generating the complex, multi-step trajectory data required for training capable agents. This method is far more efficient and scalable than manual human annotation. 

4.	Experimental validation demonstrates that the proposed method can outperform many baseline methods on the selected benchmarks.

### Weaknesses
1.	The paper claims a "novel reinforcement learning method," but the description points to a standard application of PPO with a task-specific reward function. This overstates the novelty of the technical contribution. 

2.	Based on my understanding, the results in Table 1 are exclusive to the Qwen2.5-32B model. While the results are strong on this architecture, this narrow focus leaves the framework's generalizability as an open question. The claim to provide a "pathway for advancing open-source LLMs" would be substantially strengthened by demonstrating its effectiveness on other popular open-source models, such as Llama 3 or the newer Qwen3.

### Questions
1.	How does the paper's method technically differ from a standard PPO application beyond the task-specific reward function?

2.	Can the authors provide results from any experiments conducted to demonstrate the method's effectiveness on other popular open-source models, such as Llama 3, Llama 3.1, and Qwen3

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
4

### Summary
This paper introduces O-Researcher, an open-ended deep research model designed to enhance the research and reasoning capabilities of open-source LLMs. The framework centers on a multi-agent distillation process that automatically synthesizes high-quality, research-grade instruction data by simulating tool-integrated reasoning workflows. It proposes a two-stage training pipeline including SFT and RL that optimizes model alignment and tool use efficiency through a composite reward function. Experiments on the Deep-Research-Bench dataset show that O-Researcher significantly closes the performance gap between open-source and closed-source LLMs.

### Strengths
1. Addressing the data and performance gap between open- and closed-source LLMs is an important and high-impact problem, particularly in the era of agentic and research-oriented AI systems.

2. The proposed two-stage pipeline is conceptually coherent and well-aligned with practical open-source development needs.

### Weaknesses
1. The novelty primarily lies in system engineering and workflow design, not in algorithmic innovation. The reinforcement learning method is essentially PPO with an LLM-based reward, which has been explored extensively in prior literature.

2. While the overall pipeline is described, many details remain vague. For example, the reinforcement learning implementation lacks sufficient transparency for replication. Similarly, the mechanism of agent collaboration and the exact roles of sub-agents are under-specified.

3. The paper heavily depends on LLM-generated trajectories, but provides little quantitative or qualitative evaluation of their quality. There is no human verification or analysis of data fidelity, bias, or diversity.

4. The benchmark evaluation is limited to a single dataset (Deep-Research-Bench) without tests on more diverse reasoning or real-world tasks. There is also no comparison with existing multi-agent or synthetic data generation frameworks such as DeepResearcher, AutoGLM-Research, or OpenResearcher in equivalent settings.

5. The presentation is poor: (1) Several references are missing; (2) The paper contains numerous typographical errors; (3) The case studies section is disproportionately long, occupying two full pages of the main text.

6. As presented, the work is difficult to reproduce due to missing details on training duration, computational resources, model initialization, and evaluation protocol.

### Questions
I suggest highlighting the best results in each table to make the comparisons clearer.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The O-Reseacher framework aims to close the performance gap between proprietary and open-source LLMs by automating the synthesis of high-fidelity, research-grade instructional data. This data is generated through a collaborative multi-agent system that simulates complex, tool-integrated reasoning (divide-and-conquer strategy) . The resulting structured trajectories are used in a two-stage training process—supervised fine-tuning followed by agentic reinforcement learning from AI feedback—to align the student model's policy and tool-use capabilities.

### Strengths
1. The framework successfully fine-tunes open-source models, enabling them to achieve new state-of-the-art results on deep research benchmarks by forcing the student model to strictly internalize a structured, multi-step research process using XML-like tags.

2. The RLAIF stage incorporates a novel adaptive tool-use score into its reward function. This specialized component explicitly balances report quality with computational efficiency by penalizing tool invocation that is disproportionate to the query's intrinsic difficulty, which is a significant engineering advance for scalable LLM agents.

3. The multi-agent workflow is a robust engineering solution for producing highly structured, end-to-end research trajectories, ensuring the student model learns hierarchical planning, tool utilization, and sequential information aggregation.

### Weaknesses
1. The claim of "Multi-Agent Distillation" is challenged. The method uses fine-tuning on sequential trajectories, which is less sophisticated than established techniques like MAGDi. MAGDi employs architectural components, such as a GNN and contrastive loss, to rigorously distill the structural, interactive knowledge between agents. O-Reseacher is more accurately defined as Multi-Agent-Aided Trajectory Fine-Tuning.

2. The primary quality signal for the RLAIF stage is derived entirely from an LLM-as-a-Judge. This reliance risks inheriting biases or focusing the optimization on superficial stylistic consistency rather than external, verifiable scientific accuracy or novelty, leading to potential non-human-aligned errors.

### Questions
What specific, measurable, and transferable methodology is employed to reliably quantify the "intrinsic difficulty" of a new research query before the model has attempted to solve it?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces O-Researcher, an open-ended deep research model aimed at closing the performance gap between open-source and closed-source large language models (LLMs) in research-oriented reasoning tasks. It focuses on automating research-grade data generation and training open-source LLMs through multi-agent collaboration and agentic reinforcement learning (RL).
The authors develop a two-stage training strategy: (1) supervised fine-tuning on the multi-agent-generated trajectories (the synthetic data) and (2) reinforcement Learning from AI Feedback to further align models by measuring quality and tool-use rewards.

### Strengths
- Generating synthetic data for training LLMs for deep research agents and combining multi-agent distillation with agentic reinforcement learning is meaningful. 
- The performance of the resulting model looks promising.

### Weaknesses
- The submitted draft is simply incomplete and it seems like the write-up was never finished before submission. The authors have simply submitted an incomplete paper in my opinion. The explanation of methods lacks details. For example, in section 3.1, "These reports are then aggregated by a dedicated model as the final report.", the authors did not explain clearly how are the reports aggregated, which will make the readers confused and question the reliability.
- The paper seems to be missing the results from Gemini-2.5 Pro Research from the DeepResearch Bench paper. Gemini-2.5 Pro Research outperforms both GPT and O-Researcher based on the numbers. The results are minimal and basic. Some results are referred to in the text but not actually present in the paper (e.g., results for specialized research API). 
- Unclear what the performance of Qwen2.5 models was before any kind of specialized SFT and RLAIF.
- Prompts used are not actually mentioned anywhere.
- The Reward Function is not particularly novel and so the novelty of the overall RLAIF is rather low in my opinion. The paper doesn't show any kind of ablation study on the specific reward function chosen. Besides, the design of RLAIF only takes consideration two aspects: report quality and efficient tool utilization. There should be more aspects to consider, such as scale, granularity, temporal dynamics, etc.
- The evaluation is rather weak, and most ablation studies that might provide some more insight into the design choices are missing. Overall, the experiments are not strong and comprehensive, they cover too few aspects, including model types, benchmarks, and metrics.
- There is no clear explanation for the case study output presented in section 5.1 "Case study"; it is hard to understand.
- There is "multi-agent distillation" in the paper title, but there is no content mentioning "distillation" in the paper. I can understand that it is explained in section 3.1 "Supervised Fine-Tuning (SFT)", but the authors should clearly state that and explain.
- The storytelling in the paper could be significantly improved. Seems like this paper was submitted in an incomplete state. There are a couple of citations that are missing in the text. Figure 1 is never fully explained in the text, and all the stages of the bottom part of the figure. Table 2 is present but not referenced in text; case studies have no description whatsoever. The appendix is almost empty when there is a lot of information that should be present in there e.g. the prompts used, the prompt for the tests done with raw models. 
- The idea has potential, but as this paper currently stands, the contribution is poor because the description of the system, the results, and the analysis is incomplete.

### Questions
- How much it cost to create the human-annotated examples for SFT that the authors refer to in the paper
- What was the size of the dataset for SFT and what was the amount of data for RLAIF?
- What happened to the Qwen2.5-72B model? Since it was mentioned as the one of the two base models used but then it is never referred again in the rest of the paper
- Were there more rules used in rejective sampling than the one mentioned in Figure 1?
- Why can this method be claimed specially designed for deep research data and models? The divide-and-conquer workflow is not much different from the existing general agent workflows.
- In section 3.1, for the reward function, how are $w_1$ and $w_2$, and how are $R_{base}$ and $R_{tool}$ calculated?
- Inside table 1, are there only results for O-Researcher-SFT and O-Researcher-RL? The paper proposes the method that combine SFT and RL, but I did not see the performance results of the combined methods. Also, where are the results for the models that were mentioned in section 4.3 "Deep research frameworks, particularly OpenAI DR and Perplexity"?
- What is the point of presenting table 3 "The comparison between gpt-5 with and without divide-and-conquer workflows"? The proposed divide-and-conquer workflow is not much different from the existing methods, thus there is not much meaning to compare the performance with or without this workflow. This should be replaced with experiments covering more important aspects and insights.

### Soundness
1

### Presentation
1

### Contribution
1
