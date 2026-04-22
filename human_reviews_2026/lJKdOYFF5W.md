# Unraveling the Complexity of Memory in RL Agents: an Approach for Classification and Evaluation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
The incorporation of memory into agents is essential for numerous tasks within the domain of Reinforcement Learning (RL). In particular, memory is paramount for tasks that require the use of past information, adaptation to novel environments, and improved sample efficiency. However, the term ``memory'' encompasses a wide range of concepts, which, coupled with the lack of a unified methodology for validating an agent's memory, leads to erroneous judgments about agents' memory capabilities and prevents objective comparison with other memory-enhanced agents. This paper aims to streamline the concept of memory in RL by providing practical precise definitions of agent memory types, such as long-term vs. short-term memory and declarative vs. procedural memory, inspired by cognitive science. Using these definitions, we categorize different classes of agent memory, propose a robust experimental methodology for evaluating the memory capabilities of RL agents, and standardize evaluations. Furthermore, we empirically demonstrate the importance of adhering to the proposed methodology when evaluating different types of agent memory by conducting experiments with different RL agents and what its violation leads to.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The overcome the limitation that the term “memory” in RL remains inconsistently defined and often misaligned with what experiments actually test, this paper introduces a unified taxonomy of memory in RL based on temporal dependencies and task structure. More specifically, they give a formal definition for the Memory Decision-Making with declarative long-term and short-term memory and Meta-RL with procedural memory. Then an experimental algorithm is proposed to test agent memory types based on context length and context memory border. Experiments on T-Maze, MiniGrid, and POPGym show that the proposed algorithm is efficient at distinguishing memory types.

### Strengths
1.	This paper is well-written and well-organized. It provides a clear taxonomy of memory in RL with clear definitions.
2.	The classification of long-term and short-term memory with context length and context memory border is reasonable.
3.	The experiments in Section 5.1 show that the variable correlation horizon could hide long-term memory’s deficits.

### Weaknesses
1.	The definition of memory in RL is not a novel thing. For example, long-term and short-term memory are already well-studied in the domain of RL. And skill transfer in Meta-RL is also a popular concept.
2.	Many works in memory RL such as [1,2] are not classified using the proposed experimental classification algorithm (Algorithm 1).
3.	There are some typos such as “According to Theorem Theorem 2,”.

References

[1] Beck, Jacob, et al. "Amrl: Aggregated memory for reinforcement learning." International Conference on Learning Representations. 2020.

[2] Loynd, Ricky, et al. "Working memory graphs." International conference on machine learning. PMLR, 2020.

### Questions
1.	Could the authors classify more classical works in the domain of memory RL?
2.	Could this method be applied to agentic RL with LLMs?
3.	Are there any future research directions based on this work?

### Soundness
2

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
2

### Summary
This paper proposes a standard "K-$\xi$" definition of different kinds of "memory" in RL Agents. The majority of the paper is dedicated to establishing and defining the proposed classification system. Afterward, it explains how to use this classification system to evaluate LTM and STM separately and shows examples of the misleading conclusions of combined evaluations.

### Strengths
1. They standardized the definitions of different types of memory, thereby providing a framework for fair evaluation of subsequent models.
2. The definitions and formalizations are careful and rigorous.

### Weaknesses
The paper spends too much time on the basic theoretical and general analyses, while the experiments and analyses conducted under this theoretical framework are rather limited. Although the authors include several validation experiments demonstrating the necessity and importance of defining and distinguishing different types of memory, they do not provide any insightful conclusions or analyses derived from evaluations based on this distinction. In other words, what I expected to see was either a newly designed evaluation benchmark or tasks built upon this theoretical distinction of memory, or a comparative analysis of how the latest models or methods perform across different types of memory. But neither is included in the paper.

Frankly speaking, I am not very familiar with the expected workload or contribution standards for theoretical analyses in the RL field. Therefore, I have set my confidence score relatively low. Overall, what I expected to see here was the proposal of a new theory accompanied by analyses of the limitations of current research or guidance for the design of new models and methods. However, this paper seems to have achieved only the first part. In addition, since it does not introduce any new concepts but rather provides standardized definitions for existing ones, I feel that the overall workload and contribution of this work are somewhat limited. Again, as I am not fully familiar with the typical expectations for such work in RL, I am open to discussion and would be willing to raise both the rating and confidence scores if convinced.

### Questions
1. The order of the figures and tables is disorganized.
2. You claim that LSTM is better than Transformers in terms of LTM. What's your opinion on the following facts?
    * The performance of LSTM falls low when the training sequence is larger than 300. Its low performance under both seen validation samples and longer, unseen validation samples seems to indicate that the LSTM's capacity to perform both LTM and STM tasks falls short when the training sequence becomes too long. This precisely reflects the gradient vanishing problem of LSTM and seems to contradict your claim that "BC-LSTM retains LTM despite challenges like vanishing gradient."
    * Current studies have designed many methods to introduce LTM into transformers, such as maintaining an external memory bank and summarizing previous steps. How well do these methods perform in mitigating the LTM issues inherent in Transformer models?

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
2

### Summary
This paper proposes a clear, systematic, and practically meaningful framework for classifying and evaluating “memory” capabilities in reinforcement learning (RL) agents. Drawing inspiration from cognitive science—particularly classic memory typologies such as short-term vs. long-term memory and declarative vs. procedural memory—the authors formalize these concepts into operational definitions suitable for RL settings. The paper further introduces key notions such as “memory-intensive environments,” “correlation horizon (ξ),” and “context memory border (K),” and builds upon them a rigorous experimental methodology.

### Strengths
**Conceptual Clarification and Formalization**
The paper successfully translates ambiguous cognitive science terms (e.g., STM/LTM, declarative/procedural memory) into precise, quantifiable, and verifiable definitions within RL (see Definitions 4.4–4.6). This formalization fills a significant void in current RL literature, where “memory” is often used loosely or inconsistently.

**Proposal of a Unified Evaluation Framework**
The distinction between Memory Decision-Making (Memory DM) and Meta-RL is clearly articulated. The operational definitions of declarative memory (nenvs × neps = 1) and procedural memory (nenvs × neps > 1) provide a practical and measurable criterion for categorizing memory types.

### Weaknesses
**Abstract Treatment of Memory Mechanisms**
While Definition 4.7 defines a memory mechanism as a mapping from base context K to effective context K_eff, it does not differentiate between implementation strategies (e.g., external memory, world models, state-space models). A deeper discussion of how different architectures realize µ(K) would strengthen the framework.

**Limited Coverage of Other Memory Types**
The paper focuses primarily on declarative memory along the temporal axis. Although the authors explicitly state they do not aim to replicate all aspects of human memory, a brief discussion of how “working memory” or “episodic memory” might fit into this taxonomy would enhance its extensibility.

The paper reads more like a taxonomy of memory types and seems to lack sufficient novelty or creativity.

### Questions
NO

### Soundness
2

### Presentation
2

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
This paper highlights the lack of a unified methodology for validating “memory’ for RL agents and presents a robust experimental methodology and standardize evaluations. In particular, this paper provides formal definitions to clarify the widely adopted concepts of memory in cognitive science.

### Strengths
This paper is generally well-written and well-motivated. This paper studies the relatively underexplored subject of “memory in RL agents” and presents formal definitions of various memory concepts, highlighting the importance of appropriate experimental configurations for evaluating them. Pictorial illustrations help clarify the concepts.

### Weaknesses
Some of the definitions require more explanation to fully understand the concept. Paper formatting could be improved. Please see the questions and comments.

### Questions
**Questions**

Q1. Is it possible to present the outer loop in Figure 3?

Q2. Could the authors provide a more detailed explanation of $n$ in $\{\zeta_n\}_n$? Do we also have a set of $n$ here? This point is not very clear. $n$ should be properly defined first.

Q3. Please use consistent notation for $\min_n \Theta$ and $\max_n \Theta$, instead of mixing $\min_n \Theta$ and $\min \Theta$.

Q4. Some illustrations depicting cases that validate LTM and STM for different $K$ values (as explained in the lower part of Page 6) would help clarify the general characteristics.

**Comments**

C1. Please check the citation format in “skills across tasks Team et al. (2023)” on Page 2.

C2. Figure 4 could be properly defined as a table rather than a figure. The term 0-shot seems a bit unnatural. Please use a more common term, such as 'zero-shot'.

C3. Please check the citation format for Theorem 2 on Page 5 and for Algorithm 1 on Page 7.

C4. $L$ is used before being properly defined, and its definition is placed within the script for Table 1. The reviewer strongly recommends defining the values before use.

### Soundness
3

### Presentation
2

### Contribution
3
