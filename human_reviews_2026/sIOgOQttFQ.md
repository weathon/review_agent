# MegaFlow: Large-Scale Distributed Orchestration System for the Agentic Era

- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
The rapid development of interactive and autonomous AI systems signals our entry into the agentic era. Training and evaluating agents on complex agentic tasks such as *software engineering* and *computer use* requires not only efficient model computation but also sophisticated infrastructure capable of coordinating vast agent-environment interactions. However, no open-source infrastructure can effectively support large-scale training and evaluation on such complex agentic tasks. To address this challenge, we present **MegaFlow**, a large-scale distributed orchestration system that enables efficient scheduling, resource allocation, and fine-grained task management for agent-environment workloads. MegaFlow abstracts agent training infrastructure into three independent services (*Model Service*, *Agent Service*, and *Environment Service*) that interact through unified interfaces, enabling independent scaling and flexible resource allocation across diverse agent-environment configurations. In our agent training deployments, MegaFlow successfully orchestrates tens of thousands of concurrent agent tasks while maintaining high system stability and achieving efficient resource utilization. By enabling such large-scale agent training, MegaFlow addresses a critical infrastructure gap in the emerging agentic AI landscape.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper introduces MegaFlow, which is a distributed orchestration system designed for training and evaluating agentic workloads. The key idea is a three-service decomposition: Environment, Agent, and Model. The authors demonstrate large-scale deployments with notable improvements.

### Strengths
I am not an expert in large-scale systems. My evaluation can only focus on clarity and methodological soundness. The three-service decomposition sounds clean and well-motivated to me, although I am not sure about the main literature in this field. The experimental results seem convincing, but I am not familiar with the setups. Since I am less familiar with this field, I recommend seeking opinions mainly from other reviewers with stronger expertise.

### Weaknesses
Since I am not an expert in this field, I found it difficult to grasp the key contributions of the paper. As a result, I was unable to provide meaningful weaknesses. I recommend seeking opinions from other reviewers with stronger expertise.

### Questions
Same as above.

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
2

### Summary
This paper presents MegaFlow, a large-scale distributed orchestration system tailored for training and evaluating autonomous agents in complex, interactive environments. The system decouples agent training into three modular services, with unified APIs and elastic cloud-based resource allocation. MegaFlow aims to overcome practical bottlenecks in agent-environment training pipelines, including security and isolation constraints, storage overhead from task-specific containerized environments, and throughput limitations of centralized high-spec clusters.

### Strengths
1. The paper articulates key system-level challenges in scaling interactive agent training, differentiating this setting from traditional large-model training workloads.
2. The three-service architecture is well-structured and clearly explained.

### Weaknesses
I am not an expert in agent orchestration, so please correct my mistakes in my questions:

1. Comparisons seem to be mainly against high-spec centralized machines rather than alternative distributed or hybrid systems.
2. While the modular three-service abstraction is intuitive, the paper would benefit from clearer articulation of which components introduce fundamentally new design ideas versus mature cloud-native practices adapted to the agent training context.
3. While the system enables large-scale rollouts, can authors offer analyses about whether this orchestration translates into better learning outcomes (e.g., improved agent capabilities, faster convergence) beyond execution efficiency?

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces MegaFlow, a distributed system for efficiently connecting environments, agent scaffolds, and LLMs for training. This system solves many of the practical issues around agent training, like security constraints, storage space, and the need for powerful machines to run the environments, resulting in a significant cost reduction.

### Strengths
- This work addresses a significant challenge regarding scaling up data collection for agentic LLM training.
- The results and analysis are comprehensive from a systems perspective.
- The text is well-written and easy to follow.

### Weaknesses
- There are no results regarding the downstream utility of MegaFlow in the context of LLM training. I recognize that this is more of an infrastructure/systems paper, but any downstream results would've been appreciated.
- The CPU utilization and memory utilization is consistent but still low.

### Questions
- Does MegaFlow support conducting multiple concurrent tasks on each (8-core, 16 GB) system? Would this help increase utilization (at the cost of increasing latency)?
- MegaFlow seems to support a lot of existing LLM agent infrastructure, so I was wondering approximately how difficult would it be for a new user to start using MegaFlow (in terms of lines of boilerplate code or general user time)?

### Soundness
3

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
2

### Summary
The manuscript introduces MegaFlow, a distributed orchestration system for large-scale agent training, and claims cost and scaling advantages over centralized baselines.  Nevertheless, the decomposition into model/agent/environment services is straightforward, and the empirical comparison is confined to an internal cloud deployment without reproducible artifacts.

### Strengths
1. The three-service modularization yields a clean separation of concerns that simplifies independent scaling and maintenance.  
2. The evaluation dataset is substantial (though I have no idea what the dataset essentially is), providing a degree of empirical credibility rarely seen in infrastructure proposals.

### Weaknesses
1. Figure 1 contains limited information; I suggest either reducing its space allocation or adding more explanatory details to enhance clarity.
2. In Line 215, what exactly are the “complex resource monitoring and allocation algorithms”? Likewise, what does the “standardized compute instance” implemented by the authors refer to? In Line 246, more details are needed regarding the document database—specifically, the structure of the operational metadata, its storage format, and how it is managed and retrieved. Without these details, it is difficult to discern the novel contribution of the proposed framework.
3. The evaluation setup is unclear. In Line 302, the authors mention “30,000 ephemeral execution tasks and over 2 million persistent execution tasks.” What exactly do these tasks represent, and how were they generated? Are they derived from specific training datasets?
4. Are there truly no comparable baselines? Could approaches such as VERL agent training or frameworks like AReal serve as baselines?
5. The authors claim this is an agent training framework, yet the experimental section lacks details on what LLM was trained, what data were used, and what hyperparameter configurations were applied, which makes the setup somewhat confusing.

### Questions
See Weakness

### Soundness
2

### Presentation
2

### Contribution
2
