# Go-Browse: Training Web Agents with Structured Exploration

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
One of the fundamental problems in digital agents is their lack of understanding of their environment. For instance, a web browsing agent may get lost in unfamiliar websites, uncertain what pages must be visited to achieve its goals. To address this, we propose Go-Browse, a method for automatically collecting diverse and realistic web agent data at scale through structured exploration of web environments. Go-Browse achieves efficient exploration by framing data collection as a graph search, enabling reuse of information across exploration episodes. We instantiate our method on the WebArena benchmark, collecting a dataset of 10K successful task-solving trajectories and 40K interaction steps across 100 URLs. Fine-tuning a 7B parameter language model on this dataset achieves a success rate of 21.7% on the WebArena benchmark, beating GPT-4o mini by 2.4% and exceeding current state-of-the-art results for sub-10B parameter models by 2.9%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Go-Browse, a structured exploration framework for collecting high-quality interaction data for web agents. By treating the web as a graph and using an outer-loop frontier expansion with an inner-loop task discovery process, the method enables more efficient coverage of websites and deeper navigation compared to prior unsupervised approaches. Fine-tuning a 7B model on the collected dataset yields new SOTA results among sub-10B models on WebArena.

### Strengths
- Clear and intuitive idea: The graph-based formulation cleanly explains how and why structured exploration improves data efficiency, making the approach easy to understand and reproduce.

- Strong motivation with practical impact: Training web agents without human demonstrations is important and timely, and the proposed method directly contributes toward scalable data generation.

- Consistent empirical evidence: The paper presents solid improvements in both success rates and depth of exploration, demonstrating that the design choices (e.g., prefixed sampling) meaningfully contribute to better real-world web agent capabilities.

### Weaknesses
- Limited evaluation of generalization capability and lack of out-of-domain (OOD) analysis: In the Introduction (line 48-49), the authors argue that “agents are likely to be more successful if they learn directly from environments they will encounter.” While this claim is straightforward and somewhat obvious, it is impractical to train on all possible websites, and agents trained this way may be vulnerable to website updates. Therefore, generalization capability is crucial for web agents. However, the current paper’s analysis primarily focuses on in-domain performance, with insufficient analysis on OOD (Mind2Web) settings. It would be valuable to add additional results and analysis on OOD setups, such as (1) direct comparison with the baseline model and other methods, and (2) success and failure case analysis based on the similarity between in-domain training websites and OOD test websites

- Limited applicability of the proposed algorithm for capturing real-world web usage patterns: While the proposed graph-based algorithm improves coverage by systematically exploring web environments, it may not align well with how users interact with the web in practice. Specifically, the framework assumes a navigation-first, leaf-level task completion structure, where the agent traverses through a hierarchy of pages before executing an action. However, many realistic tasks (e.g., flight booking) require frequent alternation between navigation and value input across multiple intermediate pages. This discrepancy reflects a deeper human–agent misalignment, as the method prioritizes coverage over modeling the natural, goal-driven behavior of real users, potentially limiting its ability to capture the tasks that matter most in real-world web interactions.

- Lack of Analysis and Justification for VLM as a Judge:
The FeasibilityChecker component and the dataset filtering process seem to heavily rely on VLM as a judge, yet there is no analysis provided (e.g., robustness or correlation with human judges). Moreover, since the VLM judge evaluates success or failure based on screenshots, a justification is needed for whether this observation-based environment allows for accurate evaluations.

### Questions
See the weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Achieving effective exploration of web navigation agents can be challenging with existing approaches for the following reasons. Instruction-first approaches, which generate tasks (or task instructions) first and then navigate the web to solve it, are dependent on and limited by the prior information and knowledge about those web pages. On the other hand, interaction-first approaches, which navigate the web pages first and then post-process the collected trajectories, may be inefficient and less cost-effective. This work proposes a hybrid approch named Go-Browse. The authors view web navigation as expanding the graph of visited web pages and combine two types of explorations: global (outer loop) and local (inner loop) explorations. Local explorations are in-page explorations that happen in "frontier" nodes in the graph. With the use of vision-language models as judges, the tasks are used to gather trajectories and filtered. For scaling of the trajectory dataset, the authors perform additional sampling of trajectories on the filtered tasks by employing cheaper models to solve the tasks from the current web page (prefixed) or initial web page (unprefixed). Using the trajectory data collected in the WebArena environment, the authors fine-tune Qwen 2.5 7B and compare it with the baselines on Online-Mind2Web and WebArena to show the improved empirical performance.

### Strengths
1. Motivations  
The work and proposed approach are reasonably motivated. As described by the authors, exploration in web environments is one of the challenges for web navigation data collection. Especially, the shortcomings of interaction-first and instruction-first approaches mentioned by the authors could be a bottleneck for scalable trajectory data collection on the web.

2. Presentation  
The manuscript provides comprehensive information. It contains figures and pseudo-codes that make it easy to follow the content of the paper. Also, there are many statistics and analyses included, which can suggest insights about the empirical results from different angles. Additionally, the authors also care about reproducibility. The manuscript presents the prompts used for different components of the proposed approach. They also share a link to the source codes for their experiments. Overall, this can encourage the adoption and extension of this work.

### Weaknesses
1. Scalability of the proposed exploration approach  
The authors use WebArena as the testbed for their exploration algorithm. However, the WebArena environment consists of concept/mockup websites and is limited in multiple aspects compared to real-world websites. Regardless of the number of unique web pages it provides, the structure of its websites and thus the possible patterns of navigation may not be diverse enough to test the scalability of the proposed approach. This is especially important, as exploration is more helpful and needed when there is more complexity in the environment.

2. Insufficient set of baselines  
The authors compare the proposed method (Go-Browse) primarily with NNetNav, an interaction-first exploration method. On the other hand, while the authors mention instruction-first approaches as a relevant line of research, they do not perform an empirical comparison with such methods. Adding comparison with more baselines could make the submission stronger.

3. Comparison with NNetNav  
While NNetNav is the only exploration approach that is being compared with Go-Browse, it does not look like an apples-to-apples comparison to me. There are some components that are reasonably fair for both: the same base model (Qwen2.5-7B-Instruct), data from the same WebArena environment, and comparable numbers of samples (around 39k steps for Go-Browse and around 45k steps for NNetNav). However, I am concerned about other components. Importantly, the authors make use of Claude Sonnet 3.7 and GPT-4o for different purposes. Figure 3 also states that the trajectory data from Claude Sonnet 3.7 constitute 33.9% of the final (successful) trajectory data. On the other hand, for NNetNav, Llama 3.1 70B Instruct was used. Therefore, the strength of the employed models could have meaningfully contributed to the current success rate wins vs. NNetNav on WebArena (2.9%) and Online-Mind2Web (1.33%).

### Questions
1. Is there empirical evidence of the applicability of Go-Browse to real websites with fair complexities? I believe this would be an important question for web navigation exploration approaches.

### Soundness
1

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
5

### Summary
This paper proposes a pipeline for generating web trajectories via structured exploration that maintains a graph of discovered pages and resets to promising frontier nodes. A feasibility checker filters proposed tasks by attempting them with a strong model and using a VLM as a judge. Solvers then sample extra trajectories (prefixed/unprefixed) with cheaper models to scale data. The system is then instantiated on WebArena for collection and evaluation. Also evaluated on Online-Mind2web.

### Strengths
1. Treating websites as a URL graph with a maintained frontier reduces redundant exploration across episodes and helps reach deeper states that matter for task completion. The outer loop/frontier mechanism is well-motivated and is validated by broader site coverage and deeper success trajectories.
2. The dataset of ~10K trajectories is a valuable resource to the community to train web agents.

### Weaknesses
1. The evaluation results on Online-M2W are weak. While the authors say it is due to a different domain, it does not help sell their synthetic trajectory generation approach. The primary purpose of synthetic data generation for web agents is to improve their performance on real-world websites in the wild.
2. This proposed pipeline may not work as well on the real-world websites, as these are dynamic and the graph can change during the course of exploration.

### Questions
Is there a reason for choosing WebArena rather than the real-world web for trajectory synthesis? I would like to see results on Online-M2W in-domain after trajectory synthesis on those websites.

### Soundness
4

### Presentation
3

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
This paper proposes Go-Browse, a go-explore inspired method for gathering training trajectories for web agents. Go-Browse leverages an outer loop that selects a previously discovered webpage, then performs an inner loop that uses modules to propose navigation and local tasks within a page, then uses verifier and task solver to gather trajectories for feasible tasks. Experimental results show that Go-Browse improves over the comparable state-of-the-art method for trajectory generation, Nnetnav, and analyses show that the proposed method results in deeper and wider exploration of the web environments.

### Strengths
- The proposed method is well motivated, and the adaptation of Go-Explore to exploring from a frontier of discovered webpages is intuitive and novel. 
- The proposed method shows clear effectiveness over a comparable state-of-the-art method.
- The experiments and analyses are thorough, and all details as well as prompts are provided, enhancing reproducibility.
- The paper is well-written and easy to follow.

### Weaknesses
- The method leverages claude-3.7-sonnet for trajectory gathering, and it is unclear whether this may be a significant advantage of the proposed approach over NnetNav.
- I'm not sure I understand the purpose of the experiments on Online Mind2web, as the results seem to be evaluating WebArena-trained models on Online Mind2Web. However, my understanding is that the proposed method is more effective at exploring a given environment such as the websites in WebArena, while Online Mind2Web consists of entirely different websites.

### Questions
- Can you clarify "explore 20 different URLs for each of the five domains" (L269)? Does this mean that you use 20 starting URLs?
- What is the overall number of trajectories used for finetuning with Nnetnav (vs Go-Browse)?
- For finetuning, are both prefixed and unprefixed trajectories used together?
- What is the accuracy of the LLM judge? Is there any concern of label noise in the dataset from judge errors?

### Soundness
3

### Presentation
3

### Contribution
3
