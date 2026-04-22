# Reframing attention as a reinforcement learning problem for causal discovery

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
Most neural models of causality assume static causal graphs, failing to capture the dynamic and sparse nature of physical interactions where causal relationships emerge and dissolve over time. We introduce the Causal Process Framework and its neural implementation, Causal Process Models (CPMs), for learning sparse, time-varying causal graphs from visual observations. Unlike traditional approaches that maintain dense connectivity, our model explicitly constructs causal edges only when objects actively interact, dramatically improving both interpretability and computational efficiency. We achieve this by formulating causal discovery as a multi-agent reinforcement learning problem, where specialized agents sequentially decide which objects are causally connected at each timestep. Our key innovation is a structured representation that factorizes object and force vectors along three learned dimensions (mutability, causal relevance, and control relevance), enabling the automatic discovery of semantically meaningful encodings. We demonstrate that a CPM significantly outperforms dense graph baselines on physical prediction tasks, particularly for longer horizons and varying object counts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the Causal Process Framework (CPF) and its neural implementation, the Causal Process Model (CPM), which reinterprets the attention mechanism of Transformer networks as a reinforcement learning problem for causal discovery. Instead of soft attention weights, the model employs two RL agents to decide which causal edges to instantiate between objects and forces over time. This allows the system to construct sparse, time-varying causal graphs that reflect active physical interactions rather than dense potential dependencies. Experiments in a synthetic physics environment show that CPM outperforms Graph Neural Networks (GNNs), Transformers, Recurrent Independent Mechanisms (RIMs), and modular networks in prediction accuracy, long-horizon generalization, and downstream RL performance.

### Strengths
- **Novel conceptual link between attention and RL:** The paper offers a fresh theoretical view by reframing attention as a decision-making problem. This perspective connects causal discovery, reinforcement learning, and neural attention in an elegant way.  
  - **Dynamic causal modeling:** The proposed Causal Process Framework explicitly models causal graphs that evolve over time, addressing a key limitation of static Structural Causal Models when applied to dynamic physical systems.  
  - **Interpretability and sparsity:** The all-or-nothing edge construction naturally yields interpretable causal graphs, where connections correspond to actual interactions (e.g., collisions) rather than dense message passing.  
  - **Strong empirical results:** The CPM demonstrates clear improvements over baselines in multi-object physical environments and provides consistent advantages in both observed and unobserved generalization settings.

### Weaknesses
- **Clarity and notation:** Section 3.1 is particularly dense and difficult to follow. Some key symbols (e.g., $J^t$) are used before being defined, and sets $N$ and $M$ are not introduced at all. The abundance of indices and nested distributions makes it hard to parse the formalism without additional diagrams or examples.  
  - **Imprecise language:** The paper frequently uses vague phrasing that leaves important concepts underdefined. For example, the phrase “defining the causal chain of object-to-force and force-to-object connections” lacks a precise mathematical meaning and forces the reader to infer the intended interpretation.  
  - **Limited evaluation diversity:** While the synthetic physics environment provides proof of concept, it remains relatively simple. There are no experiments on more complex, real-world settings or comparisons to recent structured causal transformers beyond Melnychuk et al. (2022).  
  - **Reward learning unclear:** While Figure 5 reports mean reward values across object counts, it remains unclear how these rewards are linked to the CPM’s core training objective. The paper does not specify whether maximizing these rewards directly improves causal graph accuracy, predictive performance, or both. Moreover, the overall optimization landscape is ambiguous, what exactly constitutes the optimum? Is it a state where the RL agents select edges yielding minimal prediction loss, or where the learned reward MLPs stabilize under inverse RL? Without this connection between the agent-level rewards and the CPM loss, it’s difficult to interpret the learning dynamics or judge convergence.
  - **Missing ablation or analysis of learned structure:** The discussion section mentions plans to analyze semantic sub-vectors (mutable, causal, controllable), but such analysis would have strengthened the current submission by demonstrating interpretability concretely.

### Questions
- Can the authors clarify how the learned rewards $R_O$ and $R_{O<->F}$ correspond to meaningful causal evaluation criteria?  
  - Are the interaction-scope and effect-attribution agents trained jointly or alternately, and how stable is this process?  
  - How sensitive are results to the inductive biases (e.g., pairwise force-object constraints)?  
  - Could the authors show qualitative examples of inferred causal graphs during different physical interactions to support interpretability claims?  
  - How does CPM scale with larger numbers of objects or higher-dimensional state representations?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The authors consider the problem of modeling predictive dynamics of the world from the perspective of causal modeling where they highlight the limitations of modeling static causal graphs and then further extend it to the temporal domain. In particular they consider a graph comprising of objects and forces and break the dependency structure through interactions from objects to forces and then from forces to objects. The models are trained through a mix of contrastive learning and reinforcement learning, where the latter is used to train the policies that govern interactions (creation of the temporal causal graph) over time. Evaluation of the proposed methodology is done on synthetic domains with unknown underlying causal graphs, akin to Causal Structured World Models (C-SWM), where the authors demonstrate improved performance of their proposed approach.

### Strengths
- The work provides nice insights into the limitations of modeling a static causal graph and provides a viable alternative for modeling temporal data with potentially symmetric relationships (eg. collision of two objects).
- The proposed method (CPM) improves performance over relevant baselines (RIMs, Transformers, etc.) in both long-horizon predictive tasks as well as downstream RL.

### Weaknesses
- It seems that Equations (1) and (2) form the crux of the method but unfortunately they are quite verbose and not clear. It would be beneficial to walk through these equations with a concrete but simple example. From my naive understanding, what I get is that $\rho^t_{\mathcal{O}}$ denotes a distribution over interactions from objects to forces, while $\rho^t_{\mathcal{O}\leftrightarrow\mathcal{F}}$ denotes distributions over interactions from force to objects. However, the comment on different conditionings for both is lost on me.
- Similar to the above point, Section 4.1 introduces a lot of notation without any motivation into why this notation is needed (mutability, causal relevance and control relevance) or even what these terms stand for.
- The authors model all objects to be of the same type and all forces to be of the same type as well which is quite restrictive. In general, different kinds of forces can affect different kinds of objects and sharing the type / class is a big limitation.

Overall, my biggest concerns revolve around presentation and clarity of the proposed approach. The manuscript introduces a lot of notation and terminology without clearly outlining the need or motivation for it which makes it really hard to understand the whole method as well as the training pipeline. It also seems to be lacking in generality and more heavily engineered for the tasks the authors consider in evaluation, but I may be wrong in this assessment since I will admit I was not able to fully grasp the proposed approach from the draft.

### Questions
- The authors make an assumption that at any given time, the force only affects one of the objects that it takes as input. How is this a reasonable assumption, since collision of two objects leads to forces that impact both the objects and not just one?
- Could the authors also clarify the assumption that if a force affects an object at $t+1$, then that object must also affect the force at time $t$? Maybe from the lens of collision? 
- It is unclear how the parameters of the policy $\pi$ impact equation (8) since the authors use a fully connected causal graph in learning the CPM module.

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
3

### Summary
The paper proposes an RL approach to modeling dynamical systems in an environment using causal graphs. The method first encodes objects in input images as latent vector representations, then iteratively builds a causal graph representing the forces applied on them between a timestep $t$ and a timestep $t+1$. The learned policies use attention-based mechanisms to this end. This approach achieves better performance than baselines on a dynamical system learning dataset.

### Strengths
The paper tackles a challenging problem in causal representation learning for dynamical systems and proposes an interesting and original approach. I find particularly relevant the division between objects and forces that enforce regularization constraints over the causal graph (e.g. sparsity induced by the limit of affected objects).

### Weaknesses
1. The framing of the paper is curious and overstates the actual contributions. In that regard, the title is also misleading for the reader as the purpose of the paper is not to reframe attention but to propose a causal modeling approach sparsifying and disentangling dependencies between object dynamics. From my understanding, attention is only marginally used in the paper. The paper would greatly benefit by changing its framing to better reflect its actual contributions.
2. The paper lacks comparison with related work, notably on causal methods for disentanglement in vision or dynamical systems, e.g. [1-3].
3. Some design choices are not well justified, notably the harcoding of mutability, causal and control relevance features and the asymmetry between objects contributing to a force (two) and the affected objects (one), which seem specific to the current environment and may not generalize properly to new settings.
4. The experiments are made on a single environment. As some of the inductive biases described above seem tailored for the current environment, there can be reasonable doubts about the generalization of the proposed approach to new environments.
5. The baselines do not include more modern versions of the architectures used. For instance, GNN baselines could include GCN [4], GIN [5] or GAT [6]. A great variety of transformer models also exist. For a fair comparison, the number of parameters of each baseline model could help highlight the benefit of the proposed architecture compared to the amount of compute required. Baselines could also include the ones from [7].




[1] Yang, Mengyue, et al. "Causalvae: Disentangled representation learning via neural structural causal models." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

[2] Lei, Anson, Bernhard Schölkopf, and Ingmar Posner. "Causal discovery for modular world models." NeurIPS 2022 Workshop on Neuro Causal and Symbolic AI (nCSI). 2022.

[3] Wang, Zizhao, et al. "Causal dynamics learning for task-independent state abstraction." arXiv preprint arXiv:2206.13452 (2022).

[4] Kipf, T. N. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016).

[5] Xu, Keyulu, et al. "How powerful are graph neural networks?." arXiv preprint arXiv:1810.00826 (2018).

[6] Veličković, Petar, et al. "Graph attention networks." arXiv preprint arXiv:1710.10903 (2017).

[7] Ke, Nan Rosemary, et al. "Systematic evaluation of causal discovery in visual model based reinforcement learning." arXiv preprint arXiv:2107.00848 (2021).

### Questions
1. The two interaction scope controllers ensure that only one object is affected by a force. Why this choice as opposite forces are typically applied to pairs of objects? Are you representing the force on the second object with a second relationship? If so, do you ensure that it is equal to the first one? If not, could the proposed approach generalize beyond the used environment (where a simplifying assumption assumes that small objects do not affect big ojects)?
2. Does Figure 2 correspond to the transition function of $f_F(\dots)$ or $f_O(\dots)$?
3. What is the rationale behind the hardcoding of the mutability, causal and control relevance features? Why not letting the model learn them? Could this choice hurt generalization to new environments?
4. Could you elaborate why performance increases when the number of objects in the environment increases? Unless I misunderstood the meaning of the axis, intuitively performance should decrease as the environment gains in complexity.
5. Why is the average across top 8 out of 10 seeds shown? Can outliers affect the performance of the method?

### Soundness
3

### Presentation
2

### Contribution
2
