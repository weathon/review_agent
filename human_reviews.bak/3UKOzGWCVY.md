# Learn-by-interact: A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Autonomous agents powered by large language models (LLMs) have the potential to enhance human capabilities, assisting with digital tasks from sending emails to performing data analysis.   The abilities of existing LLMs at such tasks are often hindered by the lack of high-quality agent data from the corresponding environments they interact with.  We propose LEARN-BY-INTERACT, a data-centric framework to adapt LLM agents to any given environments without human annotations.   LEARN-BY-INTERACT synthesizes trajectories of agent-environment interactions based on documentations, and constructs instructions by summarizing or abstracting the interaction histories, a process called backward construction. We assess the quality of our synthetic data by using them in both training-based scenarios and training-free in-context learning (ICL), where we craft innovative retrieval approaches optimized for agents. Extensive experiments on SWE-bench, WebArena, OSWorld, and Spider2-V spanning across realistic coding, web, and desktop environments show the effectiveness of LEARN-BY-INTERACT in various downstream agentic tasks — baseline results are improved up to 11.1% for ICL with Claude-3.5 and 23.1% for training with Codestral-22B. We further demonstrate the critical role of backward construction, which provides up to 10.6% improvement for training.  Our ablation studies demonstrate the efficiency provided by our synthesized data in ICL and the superiority of our retrieval pipeline over alternative approaches like conventional retrieval-augmented generation (RAG). We expect that LEARN-BY-INTERACT will serve as a foundation for agent data synthesis as LLMs are increasingly deployed at real-world environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a data synthesis method named “LEARN-BY-INTERACT” that uses LLM to generate instructions and trajectories based on given environmental documents, and the synthesized data can be used in ICL and training scenarios to improve the performance of agents. Experiments conducted on 4 agent datasets validate the effectiveness of LEARN-BY-INTERACT.

### Strengths
1. To the best of my knowledge, the backward construction mechanism is novel.
2. The paper is well written.

### Weaknesses
1. Although the authors claim that the proposed  LEARN-BY-INTERACT can adapt LLM agents to any given environments without human annotations in both abstract and conclusion sections, but I think its application may not be very wide, it needs the environment to have corresponding documentation, and the LLM used to synthesize the data should be familiar with the environment, otherwise it is difficult for the LLM to synthesize valid instruction and trajectory. More discussion about the limitations of the methodology would make this paper better.

2. There are many works that focus on using more powerful LLMs to synthesize data to improve agent performance, such as AgentGen [1] and AgentTuning [2], but this paper does not discuss or compare them.

3. This method requires a lot of LLM calls to generate and filter the data, especially the backward construction phase, which seems costly. 

[1] AGENTGEN: Enhancing Planning Abilities for Large Language Model based Agent via Environment and Task Generation
[2] Agenttuning: Enabling generalized agent abilities for llms.

### Questions
Refer to the concerns in “Weaknesses”

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a novel data synthesis framework to enhance agent performance. Contrary to the conventional forward data construction, the proposed backward construction generates instructions from interaction trajectories with the environment. The synthesized data can be used for training the generator or in-context learning. Experiments across four environments demonstrate the potential of this method.

### Strengths
In contrast to conventional data synthesis approaches, the proposed backward construction leverages unsuccessful trajectories, thereby improving data collection efficiency. This idea bears a high-level resemblance to the renowned reinforcement learning algorithm Hindsight Experience Replay, which is elegant and proven effective. The paper also provides comprehensive experiments covering performance, efficiency, and scalability.

### Weaknesses
However, as shown in Algorithm 1 (lines 16-21), the proposed backward construction has quadratic complexity concerning trajectory length, $O(\text{len}(T)^2)$. This raises concerns regarding data collection efficiency and potentially higher computational costs than conventional forward construction. I am open to raising my score if the authors address the following concerns listed in the Questions section.

### Questions
- As mentioned above, the proposed backward construction may have quadratic complexity. I note the relevant discussion in Figure 2, but it is unclear whether this figure applies to inference only or the entire training-inference pipeline.

- On page 3, Algorithm 1 appears to lack a definition of the L() function presented in line 11. Does this function rely on the same LLM backbone as the other function, LLM(), mentioned above?

- On page 5, Table 1, the drop rate is relatively high for OSWorld and Spider2-V, particularly for the latter, where fewer than 20% of synthesized samples are retained. This appears inefficient. Could the authors provide more discussion on this matter?

- Following Question 3, could the authors assess the potential impact of this filtering rate on final performance? For example, if a less strict filtering rule is applied, retaining more samples, how would this affect overall performance?

- In Algorithm 2 on page 4, what is the difference between the append() operation (line 10) and the += operator (line 19)?

- (Minor) It seems that all evaluations lack the percentage unit (%) for accuracy.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes learn-by-interact, which generates task-specific exemplars which are curated using backward construction which annotates the trajectory with an aligned objective instruction, and filtering using a committee of LLMs.
The results are evaluated on a wide array of benchmarks, SWE-bench, WebArena, OSWorld and Spider2-V, showing the effectiveness over strong baselines.

### Strengths
- The proposed approach for generating exemplars for ICL is novel and effective across several agentic scenarios.
- The paper is well written and easy to follow
- The experiments and ablations are very thorough, and validate the components of the proposed method well

### Weaknesses
- The discussion/details on filitering of synthesized trajectories could be improved.

### Questions
- Can you show examples of what trajectories get filtered out?
- Was LATS implemented by the authors for the benchmarks tested? As far as I'm aware the original LATS didn't evaluate on the benchmarks tested

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to address the critical problem of data collection for training language agents. Annotating trajectory-level data in various environments can be quite expensive. To deal with this, this paper instead proposes a data synthesis pipeline that leverages the documentation available on the internet to generate high quality task instructions, and use the LLM to compose the corresponding trajectory for each instruction. Specifically, the error rate of directly generating the trajectory using LLM can be quite high. As a result, this paper proposes a novel scheme called backward construction to summarize the trajectory and refine the original instruction to make it align better with the generated trajectory. In addition, they also use LLMs to filter out low-quality data points. After obtaining the synthetic data, they use them for both fine-tuning and ICL in multiple different domains, including code agent, OS agent, and web agent. Experimental results show the effectiveness of their synthetic data.

### Strengths
1. The problem addressed in this paper is highly significant and of great interest to the language agent community. Due to the scarcity of process/trajectory data available online, the community is eager to find efficient and scalable methods to obtain more trajectory-level data for training. The dataset collected in this paper might be a valuable resource to the community.
2. This paper covers multiple domains and demonstrates promising results on all of them, which shows the effectiveness of the data collection process.
3. This paper conducts comprehensive analyses, including scaling law of the training data and the effect of trajectory length.

### Weaknesses
1. A key issue with the data synthesis pipeline proposed in this paper is its generalizability. Specifically, the pipeline relies on a set of source documentation to generate meaningful task instructions, serving as the starting point of the entire process. However, the assumption that in-domain documentation will always be available may not hold in all cases.
2. Related to the first point, the reliance on in-domain documentation might also set a ceiling for the size of the dataset. Also, the scaling law in this paper (i.e., Fig 3) suggests that achieving continuous gains becomes challenging once around 100k data samples have been collected.

### Questions
Generalizability of this method, for example, for the web domain, it's kinda cheating to use these sources of documentation? WebArena is essentially built on open-source alternatives of these sources. It might be interesting to explore removing the reliance on documentation, as it may not be strictly necessary; maybe you can just ask the LLM to propose task instructions based on the given environment?

In your dataset, is it possible for one data sample to be a sub trajectory of another sample?

On WebArena, why do you choose Step as your baseline method rather than more direct baseline used in the original paper of WebArena?

Typos:
line 44: desktop computing etc. -> desktop computing, etc.
Alg 1: initilize interaction trajectory -> initialize interaction trajectory
Alg 2: Langueg Model -> Language Model

### Soundness
3

### Presentation
2

### Contribution
2
