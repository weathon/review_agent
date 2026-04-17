# Timed Dynamic Expansion for Continual Learning

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Catastrophic forgetting remains a core challenge in continual learning, where parameter updates for new tasks interfere destructively with knowledge acquired from previous tasks. Dynamic expansion methods mitigate forgetting by inserting task-specific adapters, but rigid growth schedules often expand capacity unnecessarily on stable tasks while failing to protect against interference that arises later within a task. We propose Timed Dynamic Expansion (TIDE), a method that stabilises expansion by creating adapters only at the moments they are needed, preventing both wasted growth and destructive forgetting. This strategy improves training stability by limiting redundant modules, reduces memory overhead by avoiding expansion on tasks that do not conflict with prior knowledge, and ensures protection when forgetting arises unpredictably. At inference, TIDE combines adapter outputs through a Fisher Information–weighted gating mechanism to route information through the adapters most critical for retention. Experiments on standard CL benchmarks demonstrate that TIDE reduces forgetting, improves long-term retention, and achieves these gains with lower parameter growth than existing expansion methods. Code is available at https://anonymous.4open.science/r/TIDE-23B3/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose an adaptive model expansion based continual learning method that creates adapters upon detecting forgetting. They also introduce a forgetting metric based on Fisher information and evaluate the approach on several benchmarks.

### Strengths
1. The authors improve expansion based methods by inserting adapters only when the current task exhibits forgetting, which prevents unnecessary growth.
2. The authors conduct experiments on several benchmarks and report improvements over competing methods.

### Weaknesses
The mathematical expressions are not rigorous, and many discussions rely heavily on strong assumptions:
1. It is unclear which dataset the loss in the equation at line 186 is evaluated on, $D_A$ or $D_t$? This must be clarified, because if the loss is evaluated on $D_t$, the definition of forgetting in Eq.(1) would be incorrect.
2. The authors assume small parameter perturbations, meaning the updates on new tasks are small, which naturally implies little or no forgetting. Moreover, if $||\Delta\theta||$ is small, the second-order term should be smaller than the first-order term. On what basis do you conclude that “the leading-order change in loss is driven by the second derivative”?
3. The assumption in lines 211–215 that both the first and second derivatives are zero is not generally observed in practice.
4. The statements of Theorems 1 and 2 are puzzling and do not appear to align with the claims the authors intend to make.

### Questions
Besides the weaknesses, we have the following questions:
1. In line 244, what does the notation F_hist represent?
2. In the experiments, why not report the model’s performance on mitigating forgetting using standard metrics, such as Backward Transfer, or the forgetting metric defined by the authors?

### Soundness
2

### Presentation
1

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
Continual learning is challenged by catastrophic forgetting: training on later tasks overwrites information learned from earlier tasks. To mitigate, existing methods expand the network for each new task, leading to a large growth in parameter counts. Authors propose a flexible expansion method that install small adapter modules into a pre-trained backbone when forgetting for an old task is measured. To measure forgetting, authors weight the change in parameter p to its reference value after training task t with the Fisher information matrix. If forgetting is statistically significant, a new adapter is integrated. At inference, adapter outputs are weighted by their FIM. This method is extensively evaluated in the class-incremental setting on 5 image and two mixed-domain datasets, where it outperforms existing CIL methods.

### Strengths
- Novelty of the idea: tying parameter expansion to fisher based forgetting
- Extensive set of experiments on also large scale ImageNet-scale datasets (with 25+ repetitions per dataset to average numbers)
- Several ablation studies integrated that analyse the impact of different componens (e.g., adapter $\alpha$)
- Presentation of the paper easy to follow (I enjoyed reading the paper)

### Weaknesses
- I kindly disagree with the notion that the adapters are inserted for an old task. Rather, I think that the adapters are inserted for the current task, to prevent it from overwriting the important parameters of the old task.

- Its unclear to me how/if adapters are trained: once an adapter is integrated for old task A, how will task A know of its existence? I mean, the newly-installed adapter was not there when the model was trained on task A? Then, if the adapter is trained (? line15 in algorithm 1 suggests so), how is it trained to not interfere with A? Shouldn't the old task A completely ignore the newly-installed adapter (because, again, it was not there during original trainin on task A)? Please clarify the role of

### Questions
- Is the backbone even trained? Figure 1 and Algorithm 1 seem to suggest that only the adapters are trained, and the rest is frozen? But, then, how can an adapter be trained for task 1, because in task 1 there is no forgetting (because its the first task, no historical performance is available yet), and hence no adapter needs to be installed for task 1?
- Whats the difference between $F^{A}\_{ij}$ and $F^{A}\_{i}$ (Why do you switch in equation 1?)
- line 315: is this really a prediction? or rather the hidden state, computed as the weighted average? Or, do you refer to the actual label prediction, but where the respective adapters inside the models have been weighted accordingly?
- line 306: why use the historical importance? Won't this become outdated as the adapter is trained?

(Most of my questions and concerns revolve around the way the adapters are used. Please clarify this in the discussion period)

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes TIDE, a continual learning framework that dynamically expands adapter modules only when statistically significant forgetting is detected. Instead of expanding at fixed task boundaries, TIDE continuously monitors Fisher-weighted forgetting scores and triggers adapter creation as needed. A Fisher-informed gating mechanism then aggregates multiple adapters per task during inference. The method aims to balance stability and plasticity by coupling capacity growth directly to observed forgetting, achieving strong performance across several CIL benchmarks and a MLLM setting with lower parameter overhead.

### Strengths
1. The paper is well-written, with clear illustrations and easy to follow
2. Introduces a novel forgetting-triggered expansion mechanism that adapts the capacity of the model during continual learning.
3. Achieves state-of-the-art or comparable accuracy on multiple benchmarks with fewer trainable parameters.
4. Extends beyond image classification to MLLM settings, showing method versatility.

### Weaknesses
1. Thanks for the efforts in reproducing all previous rehearsal-free methods using a replay buffer. While all compared methods are rehearsal-free, TIDE still relies on memory buffers for forgetting detection, raising concerns about its feasibility in a strictly rehearsal-free setting. 

2. The approach closely resembles prior self-expansion or dynamic adapter methods. 

3. The Fisher-based forgetting metric, based on trace approximation and small replay buffers, may be noisy. Its robustness to estimation errors is unclear, and reducing the memory buffer significantly hurts performance (Fig. 9).

4. Continual evaluation of forgetting for all past tasks could become costly as task numbers grow.

5. Visualizations showing when adapters expand during the whole CIL task sequences would clarify TIDE’s dynamics.

6. In Table 3, parameter counts for CORe50 between Task Expansion and TIDE do not differ much, weakening the claim of lower growth.

7. Since adapters are added when forgetting occurs, per-task forgetting curves before and after expansion are essential to demonstrate effectiveness.

### Questions
see weakness

### Soundness
3

### Presentation
4

### Contribution
3
