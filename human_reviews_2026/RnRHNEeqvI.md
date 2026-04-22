# Generalization in LLM Problem Solving: The Case of the Shortest Path

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 2

## Abstract
Whether language models can systematically generalize remains actively debated. 
Yet empirical performance is jointly shaped by multiple factors such as training data, training paradigms, and inference-time strategies, making failures difficult to interpret.
We introduce a controlled synthetic environment based on shortest-path planning, a canonical composable sequential optimization problem. 
The setup enables clean separation of these factors and supports two orthogonal axes of generalization: ***spatial transfer*** to unseen maps and ***length scaling*** to longer-horizon problems.
We find that models exhibit strong spatial transfer but consistently fail under length scaling due to recursive instability.
We further analyze how distinct stages of the learning pipeline influence systematic problem-solving:
for example, data coverage sets capability limits; reinforcement learning improves training stability but does not expand those limits; and inference-time scaling enhances performance but cannot rescue length-scaling failures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a map navigation benchmark based on 2D sparse grid navigation tasks, and uses it to evaluate transformer models with LLaMA-style architecture,
The authors analyze model performance on this benchmark using two dimensions: 
- Transfer (systematicity): solving the same class of problems in new environments, and 
- Scaling (productivity): solving harder problems after mastering simpler ones. 
They consider two model training paradigms: (1) Supervised fine-tuning, with paths encoded as movement directions; model predicts continuation given start/end prompt. (2) Reinforcement learning, giving binary reward for valid shortest paths. They find that for supervised fine-tuning distinct questions drive transfer more than multiple solutions per question.
To determine which types of questions best support spatial transfer they focus on coverage (fraction of unique nodes in the training map included in training) vs diversity (average number of distinct endpoints each start node is paired with.) of training questions.

### Strengths
The authors focus on defined generalization dimensions (transfer and scaling) and the experimental design appears to be sound, separating coverage vs diversity.

Overall, this work clearly *intended as* a methodologically solid framework with careful experiments, however poor presentation makes this conceptual design very hard to understand and evaluate.

### Weaknesses
Presentation is very poor, making it hard to follow this paper. 

Beginning as early as the abstract, the writing is often superficially coherent, but conceptually incoherent - reflecting perhaps the lack of conceptual depth, or perhaps a last minute writing of work that is intended as well thought out, but obscured by poor presentation. For instance:

"_Someone who learns to walk shortest paths in New York can, upon receiving a map of Paris, immediately apply the same rule to navigate, despite never practicing there._"

Same rule (as in, the same heuristic) or same algorithm (as in, Djkstra's algorithm)? 
Much of the field would not agree with this statement, as New York has a grid-like North American design, and Paris is an older city - meaning that different rules/heuristics and approximations will be optimal.

The next sentence then jumps topic to combining rules - "_This ability to recombine known rules to solve novel problems exemplifies compositional generalization_.." 
There is no sense how this is related to the previous sentence, or the remainder of the abstract.

I am unable to evaluate the paper, because I can not unearth the content from the ineffective presentation. I feel that this work is not yet ready for an archival publication, but after a careful revision it might be.

### Questions
N/A

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors focus on the understanding of the generalization and problem solving abilities of state of the art machine learning models. The authors use the simple example of token based map navigation to study these questions. To this end, they explore a series of highly constrained evaluations in order to disentangle the effects of data coverage, diversity, and scaling.In addition, they look at the effects of reinforcement learning. A number of key findings are uncovered. Coverage is very important for spatial transfe, with diversity generally less important. Reinforcement improves optimization but does not improve transfer.

### Strengths
1. Generalization, and the understanding of what leads to generalization, is of fundamental importance in machine learning. 

2. Paper is well written and easy to understand.

3. Although the problem and domain is quite narrow (token based navigation), evaluation is quite detailed and thorough, and the results are of interest to the community.

### Weaknesses
1. The domain used is a toy problem (token based navigation) and very narrow. It is unclear if the results in the paper hold in other domains or problems in machine learning.

2. Qualitative results or a visualization of navigation capabilities (or failure cases) would strengthen the paper.

### Questions
1. Could the authors please elaborate on how the results could apply to other domains? The test-bed of navigation is quite narrow.

2. Would it be possible to include any qualitative examples or failure cases to understand the difference in performance between various regimes?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a controlled “map worlds” testbed to dissect extrapolative problem solving in Transformers along two axes of compositional generalization: spatial transfer (applying rules to disjoint environments) and length scaling (solving longer instances). Using 8-layer LLaMA-style models pretrained on random walks and fine-tuned (or RL-trained) to generate shortest paths, the authors show: (1) under fixed budget, allocating data to many distinct questions (start–end pairs) beats collecting many solutions per question; (2) coverage of primitives sets the transfer ceiling, unlocked by modest diversity, with a sharp SR inflection around 20–25% coverage; (3) diversity yields log-linear gains at moderate/high coverage but can hurt at low coverage; (4) length scaling generally fails unless training includes a small curriculum of neighboring-but-longer examples (+2–4 steps), while shorter or much longer examples help little or harm; and (5) RL (Dr.GRPO) stabilizes training and avoids overfitting but never exceeds the best SFT performance. The work offers practical data-design guidance (prioritize coverage and distinct questions; minimal solution diversity) and clarifies RL’s role as stabilizer rather than capability expander in this setting.

### Strengths
- 1. Clear decomposition of generalization: The paper cleanly separates spatial transfer from length scaling in a controlled “map worlds” testbed, enabling precise causal insights that are hard to obtain in natural language settings.

- 2. Controllable data design: The graph-based framework enables precise control over dataset factors, including node coverage (fraction of unique nodes appearing in training questions), pairing diversity (average number of endpoints per start node), and path-length regimes (via sampling constraints). This controllability supports principled ablations on how dataset composition, independent of total record count, shapes generalization.

- 3.The paper systematically varies training budget as a percentage of the possible start–end records and studies trade-offs between the number of distinct questions and the number of solutions per question under approximately fixed total records. These ablations quantify data-efficiency regimes and clarify which allocations (e.g., higher question diversity vs. multiple solutions per question) most improve spatial transfer, though matching total optimization steps (compute) is not explicitly controlled.

### Weaknesses
- Synthetic scope and small models: Results are demonstrated on small LLaMA-style models and synthetic sparse grids. External validity to larger models, richer graph families, or open-domain language tasks is uncertain.
- RL ceiling bounded by SFT without diagnosis: RL consistently fails to surpass the best SFT model, but the paper provides limited analysis of why (e.g., exploration limits, reward sparsity, credit assignment), and does not test alternative rewards (e.g., feasible-path rewards) or curricula beyond near-boundary lengths.

### Questions
- 1. **Pre-training**: Recent work suggests[1,2] that continued or mid-stage pretraining can substantially improve downstream RL gains, so it’s reasonable to worry that the findings here may be shaped by an underpowered pretraining phase. The paper pretrains on long random walks to impart “map semantics” and confirms this doesn’t leak shortest-path skills, but it does not report scale, steps, or convergence, nor ablate the pretraining budget. Without those controls, we can’t tell whether stronger pretraining would raise spatial-transfer ceilings, improve length scaling, or allow RL to surpass SFT. 

- 2. **Compute**: As mentioned in the paper that the budget is controlled by the percentage of possible records. Are experiments controlled with same amount of training compute? i.e. are experiments with fewer training data training on more epochs to match equal optimization steps?

- 3. **Length statistic details**: The study carefully separates spatial transfer (within training-length regime) from length scaling (strictly longer than training), but it does not provide the actual path-length distributions for training or evaluation, nor how those distributions shift as coverage increases. Reporting histograms (x: path length, y: sample count) for each training setup and test split, along with Lmax per setup and the fraction of samples near Lmax, would reveal whether performance changes stem from coverage/diversity design or from incidental shifts in length exposure.

- 4. **Long-to-short generalization**: The paper analyzes “short-to-long” length scaling but does not test the reverse direction. A simple control would train on a length band with relatively high minimum length and evaluate on shorter paths to see whether a long-to-short generalization effect exists.

- 5. **Non-optimal navigation evaluation**: In some cases of real-world navigation tasks, people may only care about finding a feasible path rather than shortest. Is there evaluation results for this setting? Will generalization happen when training on navigating shortest paths while evaluating feasible paths?

[1] Wang, Zengzhi, Fan Zhou, Xuefeng Li, and Pengfei Liu. "Octothinker: Mid-training incentivizes reinforcement learning scaling." arXiv preprint arXiv:2506.20512 (2025).

[2] Cen, Zhepeng, Yihang Yao, William Han, Zuxin Liu, and Ding Zhao. "Behavior Injection: Preparing Language Models for Reinforcement Learning." arXiv preprint arXiv:2505.18917 (2025).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates compositional generalization in neural networks by introducing a controlled map-navigation testbed that cleanly separates two fundamental dimensions: spatial transfer (applying learned rules to entirely new environments) and length scaling (solving longer problems than seen during training). Using shortest-path navigation on 2D grid maps, the authors train small transformer models and systematically study how data selection and training paradigms affect extrapolative problem-solving. Their key findings show that spatial transfer is primarily enabled by maximizing the number of distinct training questions with high primitive coverage and only modest diversity, rather than collecting multiple solutions per question. In contrast, length scaling requires exposure to neighboring-but-longer examples and cannot be achieved through spatial transfer alone. Comparing supervised fine-tuning (SFT) and reinforcement learning (RL), they find that while RL provides training stability and prevents overfitting, it does not unlock capabilities beyond what the best SFT model can achieve—the performance ceiling is always set by SFT when data is sufficient and high-quality. These results provide principled insights into data-efficient training strategies for extrapolative problem-solving in language models.

### Strengths
The map-navigation testbed is cleverly constructed to provide truly disjoint test domains while maintaining control over primitives and rules, enabling causal analysis of data properties that is nearly impossible in realistic settings. The systematic experiments on coverage versus diversity provide actionable insights for data collection, with clear resource-efficiency guidelines showing that high coverage with modest diversity outperforms the reverse. The comparison between SFT and RL across both generalization dimensions is thorough and well-controlled, with multiple rollout configurations and warm-start conditions. The writing is clear, findings are well-supported by comprehensive ablations, and the practical takeaways are concrete.

### Weaknesses
The main limitation is generalizability. All conclusions are drawn from small (8-layer) transformers on synthetic shortest-path tasks, raising significant questions about whether these insights transfer to billion-parameter LLMs solving complex mathematical or coding problems. The testbed, while controlled, may oversimplify compositional reasoning by reducing it to spatial navigation, potentially missing critical aspects like nested recursion, abstract rule composition, or semantic understanding that characterize real problem-solving. The finding that RL doesn't surpass SFT contradicts recent work showing RL enables extrapolation in mathematical reasoning, though the authors acknowledge this may reflect differences in data quality rather than fundamental capabilities. The paper also lacks theoretical analysis explaining why coverage dominates diversity or why length scaling requires neighboring examples, relying primarily on empirical observations.
Also, I want you to compare offline planning line of works that also tries to enhance performance on maze tasks, and position how your work is better than existing works on offline planning in advance.
Here are some example papers:
Yang et al, Chain of Thought Imitation with Procedure Cloning
Kim et al, How language models extrapolate outside the training data: A case study in Textualized Gridworld

### Questions
How sensitive are these findings to model scale? Would coverage versus diversity trade-offs shift dramatically with GPT-4-scale models that have vastly more capacity? Can the neighboring-but-longer insight be formalized into a curriculum learning strategy applicable beyond shortest paths? How do these findings relate to recent work on test-time compute scaling and chain-of-thought reasoning, where models might internally generate longer reasoning traces? Finally, would outcome-based RL (as tested here) versus process-based RL that rewards intermediate steps show different results for length scaling, given that shortest-path generation resembles step-by-step reasoning?

### Soundness
3

### Presentation
3

### Contribution
2
