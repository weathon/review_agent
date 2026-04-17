# ContrastGen: A Multi-agent Contrastive Framework for Hard Retrieval Data Generation

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
The embedding model vectorizes queries and passages separately and uses the distance between the two resulting vectors as the basis for retrieval matching. It serves as a core component in retrieval tasks. However, since training datasets often consist predominantly of simple queries, the embedding model is usually unable to develop the capability to handle complex, hard queries. This leads to a serious performance bottleneck and an upper limit on its effectiveness. To address the challenge of handling hard queries, existing methods propose new training strategies tailored for embedding models or simplification mechanisms during the query inference phase. In contrast and orthogonal to these approaches, this paper focuses on tackling the problem from the data level, aiming to improve the performance of the embedding model by generating high-quality hard query training data. More specifically, inspired by the ability of agents to closely simulate human behavior, and with the goal of generating queries that retain semantics and logical knowledge similar to those of human-generated queries, this paper proposes a multi-agent framework to generate hard queries, thereby enhancing the training performance of the embedding model. The core idea involves first using a generation agent to create new queries, followed by specialized agents—such as those focused on logical reasoning and semantic understanding—to filter and identify truly hard queries. Experimental results on different embedding models and datasets demonstrate that our method outperforms existing approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors employ a multi-agent framework for hard query generation and use this for contrastive learning for retrieval tasks. The multi-agent framework consists of two steps - a consenus between a code agent and a CoT agent step , a multi-agent group discussion phase (if the first did not lead to a consensus). Theire results show their model improves upon other approaches.

### Strengths
1. The paper tackles an important problem. 
2. Their agentic solution seems intuitive and makes sense for applications. 
3. The need for hard queries and the challenges in building them is present both in research and industry and particularly important for domain heavy topics like medicine, science, engineering, telecom etc.
4. their results are numerically superior.

### Weaknesses
The authors measure the performance of their model by precision,recall, NDCG. But the core aspect to be measured is are the queries harder ? Where is this being measured? For example in Table 1, the original data has a lower precision, recall than the model - how do we know that the questions are harder? One possible option is measure on generated queries on pre-trained models and compare with provided data - harder queries should lead to poorer performance. 

Overall the paper is interesting but it is hard to understand if the benefits are actually significant or statistical noise and how much of it have the queries improved (become harder)?

Specific comments below

1. What models are used in the different agents. How does the agent choice (size, model family) affect the results has not been analyzed. 
2. Numbers in tables need statisical significance testing - is the ContrastGen model statistically better than the second best model? In table 1 for example most improvements are less than 1 percent - this needs statisical validation.
3. Figure 3 exagerrates the differences and no statisical testing is done. For example on top row left (recall@10) all numbers are 52.xx% but the image tends to give an impression of much higher variation. Without statistically testing how many are equal. Authors can report test results from say ANOVA or pairwise t-tests. 
4. In table 4, what does the 0 label mean? Do the datasets map "6 quart crackpot" to the text? More likely this is drawn from a negative. if so, is it a negative with high cosine similarity. Without clarity on these aspects it is hard to understand how to interpret this especially the label 0 scenario. 
5. Results from ablation study also inform little without statisical testing. 
6. The problem of generating hard queries is an important one both in research and in practice (industry). However, what is not clear how this can be extended to domain heavy scenarios. Also are the costs of generating a query of 102k tokens on average prohibhitive for large dataset generations? 
7. The datasets are public datasets and hence the LLMs would have seen them. An analysis on propietary data would be interesting here if possible (or future work)


Minor comments:-

1. Lines 91-104 are largely repetitive of content from Section 1. 
2. What is $\mathbf{w}$ in $E(.;\mathbf{w})$ is not very clear. Embedding models do not generally have any parameters other than dimension?
3. Typo in table 1 - should be BGE-M4 not BEG-M3.

### Questions
Please respond to the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper ContrastGen: A Multi-Agent Contrastive Framework for Hard Retrieval Data Generation introduces a multi-agent approach to generate challenging query–passage pairs for improving embedding-based retrieval models. The framework has two stages: first, a query generation agent rewrites queries while two contrasting evaluators, the Code Agent using rule-based logic and the CoT Agent using semantic reasoning, assess their match with the passage. When the two agents disagree, the sample is identified as a hard example. These hard cases are then refined through a multi-agent group discussion, where several expert agents debate and vote to decide the final label. This process produces training data that captures nuanced reasoning and complex semantics, enhancing the model’s ability to distinguish relevant from irrelevant passages. Experiments on Shopping Queries, arXiv, and MS MARCO datasets show consistent improvements in retrieval metrics, and ablation studies confirm that both the dual-agent contrast and the discussion mechanism are essential. The main contribution is reframing retrieval data generation as a multi-agent contrastive process that uses agent disagreement and collaboration to create high-quality hard samples that improve model robustness.

### Strengths
1) Instead of following the conventional path of improving model architectures, the authors take a data-centric approach by generating high-quality hard samples to enhance the discriminative capability of retrieval models.

2) The idea of defining sample difficulty through the disagreement between a rule-based Code Agent and a reasoning-based CoT Agent is both intuitive and interesting.

3) The authors validate the framework across multiple public datasets and diverse embedding models, consistently showing performance gains.

### Weaknesses
1) The system’s performance is tightly coupled to specific prompts, role descriptions, and reasoning styles. The lack of a systematic sensitivity study leaves open whether the method would remain stable under different model versions, prompting templates, or domain shifts. Besides, the final label aggregation relies on an ad-hoc greedy rule that mixes majority votes with the first CoT decision. This design lacks theoretical grounding and could bias results toward early, possibly noisy agent judgments.

2) The observed performance peak at a moderate amount of generated data is not theoretically or empirically explained. A deeper analysis of data diversity, redundancy, or quality filtering would clarify why excessive generation degrades results.

3) The reported improvements lack confidence intervals or significance tests, and the authors do not provide the code, making it difficult to assess robustness or reproducibility.

### Questions
Q1: How does the proposed pipeline scale when generating millions of samples?

Q2: What is the average time and cost per generated example, and how does that compare to human annotation or standard augmentation methods?

Q3: Did you test other aggregation mechanisms, such as weighted confidence or probabilistic fusion? And how do you handle ties or inconsistent reasoning paths across discussion rounds?

Q4: What are the most common failure modes for the Code Agent and CoT Agent?

Q5: Did you observe any systematic biases in the group discussion outcomes (e.g., majority bias or echoing effects)?

### Soundness
2

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
4

### Summary
While I appreciate the clear problem formulation and insightful process-level metrics, I believe the paper requires several revisions. My primary concerns are the questionable novelty of the pipeline (Sec.3) and the contradiction between the claim of a "long" benchmark and its actual length. Additionally, crucial methodological details regarding data filtering, utility calculation, and the justification for your similarity metrics are insufficiently explained.

### Strengths
Clear Problem Formulation. The authors clearly identify a critical gap in existing research: the lack of benchmarks for evaluating long-chain, structured, and agentic multimodal reasoning. Current benchmarks are often limited to 1-2 hop retrievals, which is insufficient for testing advanced agentic capabilities.

Comprehensive Evaluation and Metrics: The paper moves beyond simple final-answer accuracy by introducing insightful process-level metrics like Hit per Step (HPS) and Rollout Deviation (RD).

### Weaknesses
1. Novelty of Sec.3.1. Could the authors elaborate on the novelty of "Agentic MM-RAG Pipeline" compared to [OmniSearch (Sec.4.2)](https://arxiv.org/pdf/2411.02937)?
2. The authors claim to introduce the "first benchmark with **long**, step-wise annotated instructions.(Line101)" However, the provided average length of 3.7 steps seems to contradict the descriptor "long."
3. Line180, Equation 2. The calculation of $Util(t)$ will become expensive if $T$ is large or the base model is large. (Did I miss some clever way to efficiently calculate $Util(t)$?)
4. Line197: The logic of marking a sample as *redundant* seems not rigorous enough. E.g., if a redundant and useless entity appears twice in a trajectory, which is likely when models make a mistake, the data will still be marked *not redundant* because $Nav(t)$ equals 1.
5. Need more elaboration on Data Filtering (Line175). What happens if a step is marked as redundant? Is it removed directly? How do you handle the connection of context of this removed step? Is direct concatenation likely to cause incoherent logic?

### Questions
1. Eq.4 How do you define the equality of $\hat{r}_{t'}$ and $r_{t}$? (E.g., exact match?) The evidence could be sequences of tokens.

2. Line233 The paper describes each reasoning graph as "a sequence of retrieval-augmented reasoning steps indexed by t." This phrasing suggests a strictly linear, sequential structure, essentially a reasoning chain. However, the term graph typically implies the possibility of branching, merging, or more general topological structures beyond a simple sequence.
    - Could the authors clarify whether the "reasoning graphs" in this work are always linear chains (i.e., sequences), or whether there are scenarios in which they exhibit genuine graph-like structures (e.g., multiple predecessors/successors for a step, parallel reasoning paths, or dynamic step dependencies)?
    - If all reasoning graphs are indeed sequences, would it be more precise to refer to them as reasoning chains to avoid potential confusion? Conversely, if non-linear graph structures do arise, please elaborate on (1) under what conditions such structures occur, and (2) how the model's inference procedure generates them.

3. Could you please elaborate on why applying *maximum-weight bipartite matching* is a good enough way to model the similarity of two trajectory graph.

### Soundness
2

### Presentation
2

### Contribution
2
