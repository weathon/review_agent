# SmartRAG: Jointly Learn RAG-Related Tasks From the Environment Feedback

- Decision: Accept (Poster)
- Scores: 6, 5, 8, 5

## Abstract
RAG systems consist of multiple modules to work together. However, these modules are usually separately trained. We argue that a system like RAG that incorporates multiple modules should be jointly optimized to achieve optimal performance. To demonstrate this, we design a specific pipeline called SmartRAG that includes a policy network and a retriever. The policy network can serve as 1) a decision maker that decides when to retrieve, 2) a query rewriter to generate a query most suited to the retriever and 3) an answer generator that produces the final response with/without the observations. We then propose to jointly optimize the whole system using a reinforcement learning algorithm, with the reward designed to encourage the system to achieve the highest performance with minimal retrieval cost. When jointly optimized, each module can be aware of how other modules are working and thus find the best way to work together as a complete system. Empirical results demonstrate that the jointly optimized system can achieve better performance than separately optimized counterparts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a jointly optimization method called SmartRAG which combines the retriever, the query rewriter and the answer generator. The authors mainly use a policy network which is powered by a LLM to decide whether to retrieve or answer directly. They set up a quota limitation, the policy network can only answer the query when the quota is reached. A supervised warm-up with constructed dataset which contains direct answer, retrieved answer and observed answer is firstly applied to the policy network. This step ensures the policy network with the ability to output a [retrieve] or [answer] action. Then they design a reward at time step t of either answer or retrieve and use PPO to optimize the policy network. SmartRAG can decide the action and generate answers with/without retrieved observations.

### Strengths
1. I like the idea of this paper, it consider a real problem LLM applications where the pre-trained language models and downstream RAG applications are built and optimized separately.
2. The paper is well-written, most ideas are clearly illustrated.
3. SmartRAG uses reinforcement learning methods to optimize its policy network, this is reasonable since states, actions and rewards observed in previous time steps can be used as environmental feedback instead of human feedback.

### Weaknesses
1. In the methodology part, the authors should address why there is a pre-specified quota for retrieval. RAG technicals are designed to address the lack of specific domain knowledge in the training data of a pre-trained LLM. It doesn’t make sense for the LLM-driven policy network being forced to generate the [answer] action instead of [retrieve] action when the retrieve times met the quota. I’ve never seen such a setting in industrial RAG applications. Retrieval seems necessary and provides solid answers, thus shouldn’t cause penalty even the retrieval cost of similarity search in a knowledge base do exist.
2. The optimization, however, seems not fully end-to-end. The knowledge base, which serves as a fundamental role in a RAG system, is ignored in the the design (Algorithm1 SmartRAG pipeline) and rarely mentioned in the paper. If the knowledge base is involved in the process, in methodology part, the optimization should not be tailored for specific query datasets, but remains effective even for unseen queries.

### Questions
1. In equation (1) where os is the concatenation of all the historical observations returned from the retriever, how do you solve the problem of LLM token limit?
2. Why [retrieve] action suffers from a 100% penalty in equation (5) of the reward design? If retrieval cost contributes to a negative reward, how about the correct answer generated from retrieved observations? Will that lead to an increase in the reward?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes an integrated framework that includes multiple components of RAG. The intuition is that jointly training these components can yield optimal performance. The experiments are conducted on four benchmarks with several baselines and two backbone models. The author also conducts ablation studies to investigate SmartRAG's behavior in terms of when to retrieve, what to retrieve, and how to answer.

### Strengths
1. The proposed SmartRAG is clear and easy to understand. Some of the designs are intuitive.
2. The paper writing is relatively smooth and easy to follow.

### Weaknesses
1. Part of the setups of SmartRAG are not well-justified. Please take a look at the questions.
2. Certain experimental observations are not elaborated clearly. Please take a look at the questions.
3. The RAG baselines seem not extensive, compared with references on this topic.

### Questions
1. Why does the policy network serve three roles in the model? What's the motivation behind it? Won't these three roles conflict with each other?
2. The optimization of the whole system is dependent on the feedback from the RL framework. How to distinguish whether a low reward is caused by a wrong policy or inaccurate retrieval?
3. Why is the retrieval percentage of SKR in Figure 2 fixed at 0.8?
4. In Section 4.1, the author states that *OpenBookQA, MedQA-cn, and ARC-c datasets* have nothing to retrieve from the Bing search engine. If that's the case, why does SFT RAG yield the best performance in 2 out of 3 cases with the highest retrieval ratio?

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
3

### Summary
This paper proposes a new Smart RAG framework that consists of two parts. (1)Policy Network and (2) Retriever. The policy network decides whether to retrieve or not based on the input state (question + initial observation), and generates an appropriate query to obtain the required information form the retriever. The system is trained in an end-end method using supervised fine tuning and PPO.

### Strengths
1. The main contribution of this work is the end to end trining of the RAG framework. The authors do a good job in analyzing the proposed framework (Section 4.1-4.4) 

2. The paper is generally well written.

### Weaknesses
The proposed method is not generalizable. 

1. The proposed method is very domain specific and requires the training of the entire module for each dataset.
2. The generalizability of the approach is further curtailed by the availability of datasets with true final answers. (Which is needed for SFT and  PPO) 
3. The end to end framework requires complete retraining if any part of the framework is changed.

### Questions
1. Clarification on the training dataset. 
    1. Was all of the training data used for SFT and PPO?
    2. On which dataset was the EM and F1 evaluations done? Was this used as a part of the SFT or PPO process? 
2. Is SFT RAG the performance of $\pi_{\theta^*_0}$ in the setting described in Figure 1? (Was SFT RAG evaluated in the same setting as SmartRAG? where the policy network was allowed one retrieval? )
3. Assuming the availability of a powerful model (GPT-3.5) for generating the dataset for SFT is not always true. Have the authors tried using the base LLM to generate the SFT dataset?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces SmartRAG, a retrieval-augmented generation (RAG) framework designed to enhance performance through joint optimization of its modules. Unlike traditional RAG approaches, SmartRAG uses a policy network that dynamically decides when to retrieve information, what queries to use, and how to synthesize answers based on retrieved observations. Reinforcement learning with environment feedback is employed to jointly train the system, improving coordination between modules and reducing retrieval costs. Experimental results on multiple QA datasets show that SmartRAG outperforms separately optimized systems.

### Strengths
a) The paper introduces reinforcement learning to jointly optimize RAG components, integrating retrieval decisions, query rewriting, and answer generation as policy-driven actions, representing progress in RAG architectures.

b) The introduction of reinforcement learning further demonstrates the flexibility of RAG and explores the potential of integrating RAG with other AI technologies.

c) Evaluation across several datasets demonstrates the effectiveness of the approach.

### Weaknesses
a) There exists many works focused on the question of "When to retrieval" in RAG, but in Section 4.1, only one baseline was selected for comparison. Incorporating a set of baseline examples into these studies would enhance comparative and analytical insights. 

b) The current baselines primarily apply process-wide optimizations, such as fine-tuning the generator, without optimizing individual modules (e.g., rewriter, generator, and decision-maker) separately. This setup limits the ability to demonstrate how joint optimization in SmartRAG resolves potential conflicts and enhances coordination across modules.
 
Detailed comments:
a) A dedicated section on the limitations and assumptions underlying SmartRAG, particularly regarding the potential latency issues in real-world applications, would be beneficial. 

b) While effective, the joint optimization approach may introduce complexity that could limit the model’s scalability, especially when applied to very large or real-time systems. Additional discussion on computational overhead and training time would strengthen the paper. 

c) Considering the potential for optimizing PPO's efficiency or exploring alternative RL algorithms to reduce training costs, it could be beneficial to further discuss these aspects, which may enhance the practicality of the framework.

### Questions
a) There exists many works focused on the question of "When to retrieval" in RAG, but in Section 4.1, only one baseline was selected for comparison. Incorporating a set of baseline examples into these studies would enhance comparative and analytical insights. 

b) The current baselines primarily apply process-wide optimizations, such as fine-tuning the generator, without optimizing individual modules (e.g., rewriter, generator, and decision-maker) separately. This setup limits the ability to demonstrate how joint optimization in SmartRAG resolves potential conflicts and enhances coordination across modules.
 
Detailed comments:
a) A dedicated section on the limitations and assumptions underlying SmartRAG, particularly regarding the potential latency issues in real-world applications, would be beneficial. 

b) While effective, the joint optimization approach may introduce complexity that could limit the model’s scalability, especially when applied to very large or real-time systems. Additional discussion on computational overhead and training time would strengthen the paper. 

c) Considering the potential for optimizing PPO's efficiency or exploring alternative RL algorithms to reduce training costs, it could be beneficial to further discuss these aspects, which may enhance the practicality of the framework.

### Soundness
2

### Presentation
3

### Contribution
2
