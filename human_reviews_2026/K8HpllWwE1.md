# Iterative Amortized Inference: Unifying In-Context Learning and Learned Optimizers

- Avg Score: 6.00
- Decision: Reject
- Scores: 8, 4, 8, 4

## Abstract
Modern learning systems increasingly rely on amortized learning — the idea of reusing computation or inductive biases shared across tasks to enable rapid generalization to novel problems. This principle spans a range of approaches, including meta-learning, in-context learning, prompt tuning, learned optimizers and more. While motivated by similar goals, these approaches differ in how they encode and leverage task-specific information, often provided as in-context examples. In this work, we propose a unified framework which describes how such methods differ primarily in the aspects of learning they amortize — such as initializations, learned updates, or predictive mappings — and how they incorporate task data at inference. We introduce a taxonomy that categorizes amortized models into parametric, implicit, and explicit regimes, based on whether task adaptation is externalized, internalized, or jointly modeled. Building on this view, we identify a key limitation in current approaches: most methods struggle to scale to large datasets because their capacity to process task data at inference (e.g., context length) is often limited. To address this, we propose iterative amortized inference, a class of models that refine solutions step-by-step over mini-batches, drawing inspiration from stochastic optimization. Our formulation bridges optimization-based meta-learning with forward-pass amortization in models like LLMs, offering a scalable and extensible foundation for general-purpose task adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper unifies meta-learning, learned optimizers, in-context learning, etc. into a common framework of amortized learning, i.e. learning how to reuse shared information across tasks to improve performance. In this framework, methods vary along two axes, how meta-testing data is converted into task-specific parameters, and how those parameters are combined with queries to obtain predictions. Those two processes are either fixed or learnable, leading to three categories of methods. Based on the categorization, the paper then proposes two general methods that iteratively refines task-specific parameters using mini-batches of meta-testing data (which subsumes some existing algorithms). Experiments are done on regression, classification, and generation datasets showing the power of the framework.

### Strengths
This paper provides a framework that unifies three disparate ideas, meta-learning, learned optimizers, and in-context learning. The connection of in-context learning to the others is especially insightful. I believe that this kind of work that combines different topics and provides a bird's-eye view is valuable for the community, as it has the potential to bring to light new avenues of research.

The framework itself is mathematically rigorous and clearly defined, and it encompasses many previous algorithms. Experiments are done on a variety of tasks, including classification, regression, and generation. Results provide practical lessons, e.g. parametric models outperform explicit models.

### Weaknesses
The experiments are incomplete in some respects. First, only one network architecture is considered - it would strengthen the work to see how the method scales. Second, the parametric and explicit models only consider regression and classification tasks, but they can also be used for generation.

Moreover, the paper can be unclear at times. The main idea is abstract, so more explanation would be helpful, for example by:
- Adding an explanation to the caption of figure 1
- Explaining what likelihood is being referred to in line 206
- In line 322, explaining what is the causal structure being masked
- At the end of section 4, comparing and contrasting the proposed methods to existing algorithms

### Questions
The errors in table 2 are quite large. Is there any intuition why?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a unified framework for amortized learning, aiming to connect and clarify the relationship between methods like meta-learning, in-context learning (ICL), and learned optimizers. The authors propose that these diverse approaches can be understood through a common functional decomposition involving a task-adaptation function and a prediction function. Building on this framework, they introduce a taxonomy that categorizes models into three regimes: parametric, implicit, and explicit amortization, based on how they encode shared inductive biases and adapt to new tasks.
The paper also identifies a key weakness in existing methods: their limited ability to scale to large datasets during inference due to constraints like context length. To overcome this, the authors introduce Iterative Amortized Inference, a novel technique that refines a solution iteratively over mini-batches of task data.

### Strengths
1. The conceptual framework and taxonomy is nice and reasonable. The f(x, g(D_T)) formulation provides a clear and powerful lens for understanding the underlying mechanics of ICL, learned optimizers, and meta-learners, highlighting their shared principles.
The proposed taxonomy of parametric, implicit, and explicit amortization is intuitive and effectively organizes the space of amortized learning models.

2. Novel and well-motivated method. Iterative Amortized Inference is a novel solution to the critical problem of scaling in-context adaptation to large datasets. The idea of processing data in mini-batches at inference time is a natural and scalable extension of existing paradigms, merging the strengths of forward-pass and optimization-based approaches.

### Weaknesses
1. The unification of ICL and learned optimizers, and the taxonomy, are not novel. ICL has been viewed as a form of meta-learners in literature [1,2,4]. 
And the f(x, g(D_T)) framework also has exists in literature for a not short time [1,3].
Especially, [1] has unified ICL and meta-learners explicitly and theoretically, under the same f(x, g(D_T)) framework. As learned optimizer is a subset of meta-learners, this paper exactly falls into the existing paradigm. However,  [1] has not even been referred, as a very closely related work.

2. Limitation of proposed method:
the baseline comparison and application scenario discussion are insufficient. Note that most experiments follows meta-learning setting, where numerous meta-learning methods are feasible, but this paper has only compared with 1-step amortization with transformer (e.g., standard ICL). Even restricting in this scope, it seems that training set size per task would play a critical role in the effect of iterating over one-step, which should be ablation studied and discussed. Existing results are far from to be able to see when and what advantage IAI would bring comparing with feasible baselines in practice.

3. Some technical design of IAI is questioned. Please refer to Questions 1.

4. The final implementation of IAI has hardly distinct with existing auto-regressive training, which is a common practice in ICL with transformer. Please refer to Question 2.

[1] Why In-Context Learning Models are Good Few-Shot Learners?. ICLR2025

[2] General-purpose in-context learning by meta-learning transformers. arXiv:2212.04458

[3] Conditional neural processes. ICML 2018. and Neural Processes. ICML 2018 workshop

[4] Learning to Learn with Contrastive Meta-Objective. arXiv:2410.05975

### Questions
1. For parametric and explicit setup, the authors treat $\theta$ as a token to be input to transformer, along with other tokens which represent context. Considering the possible very large discrepancy in dimensions (e.g.,$ W_T \in R^{784\*100} $) while context units is about 8*8 in MNIST), how to feed $\theta$ as an input token to transformer? And how to unify different modalities of input tokens (parameter, data, gradient)?
The reviewer think the traditional way to modulate $\theta$ by some functions using the output of $h_\psi$ [5], rather than feeding $\theta$ into transformer, might be more reasonable and practical.

2. The principle idea of introducing iteration into inference, and the implementation, seems to be similar to auto-regressive training that has been commonly applied in ICL with transformer. What is the difference? 

3. The usage and meaning of  "amortization" seems different from literature [6]. I am curious where this term originates from, and why this paper use it in this way.

[5] Fast and flexible multi-task classification using conditional neural adaptive processes. NIPS2019

[6] Memory efficient meta-learning with large images. NIPS2021

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Iterative Amortized Inference (IAI), a unified framework that generalizes amortized learning approaches (meta-learning, ICL, prompt tuning and learned optimizers) under a common mathematical framework. Furthermore, the authors provide a categorization of amortized models into parametric, implicit, and explicit regimes. Finally, they also propose a mechanism inspired by stochastic optimization, which processes mini-batches of task data incrementally.

### Strengths
The unified perspective proposed in the paper is valuable, as it clearly articulates how different amortized learning methods relate to one another within a common framework. The taxonomy provides a useful conceptual structure that can guide future research in this area. Moreover, the proposed IAI approach effectively addresses the scalability limitations of previous methods. The empirical results are convincing, demonstrating consistent improvements over existing baselines in both in-distribution and out-of-distribution settings.

### Weaknesses
- The paper could benefit from more general reference to the meta-learning literature. Including references to survey papers in the main text could help the reader to better understanding the meta-learning problem, which is only detailed in the appendix.
- While Section 5 discusses memory usage and computational cost, the analysis lacks details on the runtime required to train a k-step iterative model.
- It would be valuable to discuss theoretical guarantees (e.g., convergence bounds) for IAI.
- Minor issue: it would be helpful to illustrate $g_\psi$ in Figure 1.

### Questions
- The paper mentions improved scalability through iterative amortization. Could the authors clarify whether this scalability refers primarily to model size, dataset size, task complexity, or another dimension? A more detailed evaluation along the chosen dimension would strengthen the claim.
- How would recent works on meta-learning with in-context learning (e.g., [1,2]) fit into the proposed taxonomy?
- The experimental tables report error rates rather than average accuracy. Is there a specific reason for this choice? Reporting accuracy would make it easier to compare with previous studies.
- How is the number of steps for IAI selected? Do additional steps continue to yield improvements, or is there a point of diminishing returns?
- Given the iterative refinement process, IAI appears well suited for continual or lifelong learning setups. Have the authors considered this application, or do they foresee challenges in extending IAI to such scenarios?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors present a unified framework that encompasses various amortized learning paradigms, including meta-learning, context learning and learned optimization. Within this framework, a learned function $ f_{\gamma}$ operates in conjunction with another function $ g_{\phi}$. Specifically, $f_{\gamma}$ performs inference based on the knowledge that $g_{\phi}$ has acquired from a dataset $ D_{t}$. This formulation integrates the key characteristics of different amortized methods, allowing each individual approach to be viewed as a special case}within the broader framework. Furthermore, the authors categorize tasks into three classes, parametric, implicit, and explicit, which help distinguish among different amortized models according to whether $f_{\gamma}$ and $g_{\phi}$  are trainable or fixed. In addition, they propose a scalable sequential inference mechanism, where the output from each step serves as the input to the next, effectively modeling the process as a sequence-based system.

### Strengths
1: This paper investigates the similarities among different amortized methods, which are typically treated as distinct research areas. The conclusions drawn from this unified analysis are meaningful, as they help reveal the common underlying principles shared across these approaches.

2: The authors categorize various amortized methods based on whether the functions f and g are trainable or fixed. This categorization offers an interesting and novel perspective for understanding the relationships between different amortized learning paradigms.

3: The authors provide sufficient theoretical context and analysis to support the effectiveness of the proposed framework, thereby strengthening its conceptual soundness and empirical credibility.

### Weaknesses
1: The presentation of the paper is somewhat confusing. It lacks a clear clarification of the unified framework or an algorithmic summary. The purely descriptive explanations make it difficult to understand how the proposed unified system actually works in practice.

2: The explanation of the iterative amortized inference process is unclear. The distinction between the proposed method and existing approaches such as meta-learning is vague, which makes it challenging to identify the novel contributions of this work.

3: Since the framework aims to unify different amortized methods, the authors should provide a more thorough comparison with existing approaches across multiple baselines. This would help ensure that the unification does not lead to a significant performance drop relative to specialized methods.

### Questions
1: It would be helpful to include a concise summary of the methods and an algorithmic description, especially for the iterative amortized inference component. Because the framework spans multiple cases, a step-by-step outline (inputs, updates, outputs) would make the overall system easier to follow.

2: A more thorough comparison against baselines from multiple categories (e.g., meta-learning, context learning, learned optimizers) would better demonstrate the framework’s effectiveness. Reporting results across diverse, representative baselines would clarify where the unified approach helps and where specialized methods still have an edge.

### Soundness
3

### Presentation
2

### Contribution
2
