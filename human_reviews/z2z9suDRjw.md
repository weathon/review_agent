# GOAL: A Generalist Combinatorial Optimization Agent Learner

- Avg Score: 6.25
- Decision: Accept (Poster)
- Scores: 5, 6, 8, 6

## Abstract
Machine Learning-based heuristics have recently shown impressive performance in solving a variety of hard combinatorial optimization problems (COPs). However they generally rely on a separate neural model, specialized and trained for each single problem. Any variation of a problem requires adjustment of its model and re-training from scratch. In this paper, we propose GOAL (for Generalist combinatorial Optimization Agent Learner), a generalist model capable of efficiently solving multiple COPs and which can be fine-tuned to solve new COPs. GOAL consists of a single backbone plus light-weight problem-specific adapters for input and output processing. The backbone is based on a new form of mixed-attention blocks which allows to handle problems defined on graphs with arbitrary combinations of node, edge and instance-level features. Additionally, problems which involve heterogeneous types of nodes or edges are handled through a novel multi-type transformer architecture, where the attention blocks are duplicated to attend the meaningful combinations of types while relying on the same shared parameters. We train GOAL on a set of routing, scheduling and classic graph problems and show that it is only slightly inferior to the specialized baselines while being the first multi-task model that solves a wide range of COPs. Finally we showcase the strong transfer learning capacity of GOAL by fine-tuning it on several new problems. Our code is available at https://github.com/naver/goal-co .

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper introduces GOAL, a generalist machine learning model for solving various combinatorial optimization problems. It uses a single backbone model with lightweight problem-specific adapters and shows strong performance in solving multiple optimization tasks. The model also exhibits efficient transfer learning capabilities.

### Strengths
Writing is easy to follow. Motivation makes sense. Experiments are sufficient. 

The paper gives an interesting trial on the multi-task neural CO solver with a common encoder and task-specific adaptor.

### Weaknesses
## Contributions

- Contributions can be over claimed. Authors claim that it is ``the first model that solves such a variety of CO problems.'' As I know, [1] proposes a multi-task solver where encoders for different tasks share a common part, just like GOAL. So GOAL might not be the first one. Many other works e.g. [2] also have a similar idea. So, the idea of a multi-task solver may not be that new.

[1] Efficient training of multi-task combinarotial neural solver with multi-armed bandits. arXiv preprint arXiv:2305.06361, 2023.

[2] Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization

- The contribution to the foundational neural model for CO appears to be somewhat limited, as it seems to utilize the same neural network architecture (mixed attention) as MatNet. 

## Experiments 

- The running efficiency is not competent compared with other methods, as shown by Table 1 and 2. In ATSP and CVPR, it may take several more times of the running time compared with other methods (AM, MatNet, MDAM, etc.)

- There is still a significant gap between GOAL and the optimal solutions as shown in Table 1 and 2. Though it is understandable that some heuristics can be simultaneously fast and effective, the gap between GOAL and the best solver may make GOAL useless in solving real-world problems. Such a gap may also suggest that the time of ``general solvers'' has not yet arrived, and more efforts should be paid on task-specific models.

- Since GAOL adopts a similar neural networks as MatNet, the experimental results may not be that convincing. MatNet is trained on i.i.d. ATSP data, while GOAL is trained on multi-problem data. Given the same (or close) model capacity, how and why does it happen that MatNet performs worse than GOAL? Is it because that MatNet is trained with less epochs and does not converge completely? It looks quite unusual and requires a more detailed analysis and explanation.

## Theoretical analysis

- The neural network of mixed attention has a fatal flaw. The dimension of the node feature $x$ is $ N \times F_t$. When there are no explicit node features, $x$ is initialized as a one-hot vector. That is to say, in this case, the scale of the problem $N$ that the model can handle cannot exceed $F_t$, otherwise there would be a reuse of one-hot vectors. It appears that the model's generalization is severely compromised when $ N > F_t$, and the authors have not analyzed this situation. 

- Though the method empirically works, there is still a lack of necessary theoretical explanations that why the method has the ability to generalize to instances of a larger scale while other baselines do not have such an ability. Also, there lacks comprehensive analysis over the method's multi-problem solving ability and how the model balances the weights during the learning process for different problems simultaneously to make the model performs best on average.

### Questions
- The mixed attention works similar to MatNet. So why MatNet cannot be run on ATSP500 or larger, but GOAL can? Another question, why GOAL outperforms MatNet, which parts of GOAL contribute to the improvement?

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
4

### Summary
The paper studies the problem of training a joint multi-task model on multiple combinatorial optimization tasks (COP). 
To this end, the GOAL architecture is proposed. This (graph) transformer architecture learns a shared transformer backbone that is shared across COPs. It further maintains COP-specific adaptors that are applied before and after the backbone.
The experiments show that the model trained on multiple tasks is competitive and yields slightly worse than single-task models of the same architecture. It is further shown that fine-tuning a pretrained multi-task model to new tasks yields significantly faster convergence than training a single-task model from scratch.

### Strengths
* The problem of multi-task pretraining on multiple CO problems is novel and interesting.
* The main experiments are fairly comprehensive and consider a wide range of problems.
* The reported results are promising and suggest that pretraining can significantly improve convergence speed when fine-tuning for a new problem.

### Weaknesses
I am skeptical about the architecture design for multi-type problems, which is why I currently rate this work as a borderline reject. If I understand correctly, two changes are made when working on a multi-type problem:
1. Multiple adaptors are learned, one for each node type (and edge type), respectively.
2. The same backbone model is applied separately to each type and each node type now applies the attention module twice: once to self-attend to other nodes of the same type and a second time to cross-attend to nodes of other types.

The first modification seems useful to map the features of each type to different positions in the latent space. However, it is not clear to me why the second change is helpful. Is this superior to applying the single-type backbone to embeddings produced by heterogeneous adaptors? If so, why? The ablation study seems to combine both changes at once, so the individual contribution of either modification is not demonstrated.

I also think the illustration of the multi-type mechanism in Figure 1 is a bit misleading as it suggests that the output of the self-attention is passed as input to the cross-attention. However, the source code applies both operations in parallel and sums the results.

### Questions
* How exactly does the proposed multi-type architecture improve performance? (see Weakness section)
* Considering that the multi-task architecture performs slightly worse than the single-task models, have the authors considered using some multi-task learning techniques (GradNorm, etc...) to potentially close this gap? 
* For the fine-tuning experiments, how different would the performance be if one freezes the backbone and only trains the adaptors?

### Soundness
2

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
This paper presents a new transformer-like architecture, named GOAL, for general types of combinatorial optimization training. GOAL is an auto-regressive model with a shared architecture across different problem types with specialized input and output layers for each problem. This architecture learns a generalizable rule that solves different COs and can generalize to new problems by fine-tuning. Experiment result shows that GOAL could be comparative or outperform problem-specific state-of-the-arts (greedy version).

### Strengths
* A generalized combinatorial optimization solver is favored by the research community.
* The proposed GOAL transformer architecture seems interesting and promising, especially given the fact that it outperforms other neural networks when trained for a specific problem.
* The generalized training and fine-tuning result seems sound and promising. The greedy version of GOAL is comparative to other greddy peer methods.

### Weaknesses
Some important details are missing in this paper:
* What are the implementation details of the "codebook"? Please specify
* Definitions of  BQ-MDP and tail-recursive are needed in the main text to make this paper self-contained.

Misc:
* The first paragraph is too long
* There are multiple misuses of \citep and \citet in this paper. For example, In L112, please use \citet for Khalil et al., 2017. In L116, please use \citep for Kool et al. (2019). Please proofread and fix all the misusages
* If the oracle solver is not an optimal solver, the results should not be claimed as "optimal gap" in Table 1

### Questions
The following questions should not be considered as "weaknesses", but I believe are worth discussing:
* What is the parameter size of GOAL and how does it compare to other peer methods?
* How to implement the post-processing step (such as MCTS, 2-opt search) that has been proven to be prominent for neural network CO solvers?
* For problems where there could be different ways of defining nodes and edges, how will different node and edge definitions affect the solver's performance?
* To what extent does the current version of GOAL scale up to larger-sized problems?

### Soundness
4

### Presentation
3

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
This paper introduces a generalist model designed to address various combinatorial optimization problems. Unlike traditional machine learning approaches, which require a specialized and separately trained model for each problem, this method utilizes a shared backbone network with lightweight, problem-specific adapters for input and output processing. The backbone incorporates mixed-attention blocks that accommodate different combinations of node, edge, and instance-level features, while a multi-type Transformer architecture handles heterogeneous node and edge types. Experiments show that this method performs nearly as well as specialized models in a multi-task setting across diverse problems and demonstrates strong transfer learning capabilities, adapting effectively to new problems through fine-tuning.

### Strengths
（1）The paper is novel , as it designs a multi-task learning approach to solve various combinatorial optimization problems through an end-to-end model. The authors developed a mixed-attention block to effectively achieve this objective.

（2）The paper is well-organized, concisely written, and has good readability.

（3）This paper demonstrates substantial work, conducting experiments on various combinatorial optimization problems and showcasing the effectiveness of the proposed method in terms of solution quality and speed.

### Weaknesses
（1）The description of dimension transformations and the learning process of the model is not very illustrative. It is recommended to add figure and text to enhance the explanation.

（2）In Table 1, only one problem size is tested, and it is relatively small. It is recommended to include experiments with larger problem sizes.

（3）The paper lacks a theoretical analysis of the method's effectiveness, and it is recommended to include this section.

（4）There are few effective baselines in Table 1. For ATSP, CVRPTW, OP, KP, MVC, and JSSP, there is only one or two baselines, and UMSP lacks an NCO baseline, which weakens the convincing power. It is recommended to add more baselines.

### Questions
（1）Please address the issues raised in the weaknesses section.

（2）The paper assumes that solving strategies for various combinatorial optimization problems share common knowledge. How can it be proven that such knowledge exists, and how can the overlap of this knowledge across different combinatorial optimization problems be demonstrated?

### Soundness
3

### Presentation
3

### Contribution
3
