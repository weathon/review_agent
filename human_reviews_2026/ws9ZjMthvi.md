# Quantifying Memory Use in Reinforcement Learning with Temporal Range

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
How much does a trained RL policy actually use its past observations? We propose
*Temporal Range*, a model-agnostic metric that treats first-order
sensitivities of multiple vector outputs across a temporal
window to the input sequence as a temporal influence profile and summarizes it
by the magnitude-weighted average lag. Temporal Range is computed via
reverse-mode automatic differentiation from the Jacobian blocks
$\partial y_s/\partial x_t\in\mathbb{R}^{c\times d}$ averaged
over final timesteps $s\in\{t+1,\dots,T\}$ and is well-characterized in the
linear setting by a small set of natural axioms. Across diagnostic and control
tasks (POPGym; flicker/occlusion; Copy-$k$) and architectures (MLPs, RNNs,
SSMs), Temporal Range (i) remains small in fully observed control, (ii) scales
with the task's ground-truth lag in Copy-$k$, and (iii) aligns with the minimum
history window required for near-optimal return as confirmed by window
ablations. We also report Temporal Range for a compact Long Expressive Memory
(LEM) policy trained on the task, using it as a proxy readout of task-level
memory. Our axiomatic treatment draws on recent work on range measures,
specialized here to temporal lag and extended to vector-valued outputs in the RL
setting. Temporal Range thus offers a practical per-sequence readout of memory
dependence for comparing agents and environments and for selecting the shortest
sufficient context.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces "Temporal Range," a novel, model-agnostic metric designed to directly quantify how much a trained reinforcement learning (RL) policy utilizes its history of past observations. This metric addresses the common problem of only indirectly inferring memory usage from model architecture or task design. Temporal Range is calculated as a magnitude-weighted average look-back time, derived from the Jacobians (sensitivities) of the policy's final output with respect to the sequence of past inputs, and is justified by a set of intuitive axioms. The paper's key contributions are the formalization of this metric, its empirical validation showing that it correctly scales with known memory requirements in diagnostic tasks (e.g., Copy-k) and aligns with the minimum history needed for near-optimal returns in control tasks, and a practical method for applying it to non-differentiable or black-box policies using a Long Expressive Memory (LEM) proxy.

### Strengths
Originality: The paper's central idea is very original and the experiments are well designed to prove its correctness
Significance: The paper does a good job of justifying the utility of the temporal range metric that they introduce. Although I was initially unsure of it being useful, I was convinced by the end of the paper.

### Weaknesses
- The citation style should be changed to have the author's names in parentheses
- Figure captions are too short: they should not only describe what is shown in the figure, but also the main message the figure is trying to show in relation to the story of the paper
- The presentation of the figures could be improved a lot. For example, if one of the points being made is that \ro_t matches the minimum history window required for near-optimal return, then the \ro_t for each task should be shown as a dotted vertical line on the plots in figure 1 showing the return vs the history window size
- I think some of the interpretations/analyses of the results are false. For example, the paper states in multiple places that temporal range matches minimum history window required for near-optimal return as confirmed by window ablations. However, we see for:
Stateless cartpole -> min window = 8, temporal range = 18.81
Copy k = 1 -> min window = 2, temporal range = 6.95
Copy k = 3 -> min window = 4, temporal range = 9.52
Similar for copy k = 5 and copy k = 10
The results don't seem to back up the conclusion of the paper, which also makes the paper's proposed application of the metric invalid.
If the paper is trying to say that their metric is proportional to the minimum window, then the authors should make that clearer and remove any excessive claims - however in that case I don't think a metric proportional to the minimum window is a significant enough contribution because it then can't be used in the way proposed by the authors.

It is possible I am misunderstanding the claims of the paper, but unless this is cleared up I don't think this paper is suitable for publication in its current form.

### Questions
Questions and suggestions are in "weaknesses" section

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The work proposes temporal range, a metric that allows to determine the attribution of temporal inputs to a sequence model's final output. The metric can be computed using the Jacobian of the final model output with respect to the inputs across time. Beyond deriving this metric, the work conducts experiments in the POPGym benchmark that evaluates memory abilities of algorithms in partially observable environments. Experiments consider MLPs, different types of RNNs and SSM models. Experiments show that the temporal range metric increases with increases in memory requirements. Lastly, long expressive memory (LEM) models are proposed to train a model of non-differentiable policies that allows for stable differentiation across long sequences.

### Strengths
Overall, I find the work to be clearly presented and original. I have some concerns about the significance of the proposed metric that I will discuss below in weaknesses, but first focus on the strengths of the work.

The proposed metric of temporal range is clearly defined and introduced. I was not familiar with most of the related theoretical prior work but could follow the definitions and axioms as defined. The metric itself appears rather elegant and simple which I appreciate.

I also appreciate that the work combines its theoretical metric with practical experiments that showcase the merits of their metric. Experiments are conducted in fairly simple tasks but these are well selected to focus on tasks with clear memory components that fit the focus of the metric.

### Weaknesses
Below, I highlight any weaknesses that I see as critical / major with (**Major**). I'd expect these to be addressed for this work to be considered for acceptance. 
## Understanding the Usefulness of Temporal Range
1. **Major:** My main concern about this work is the significance of the proposed temporal range metric. While Section 3.5 and 7 argue that the temporal range can be used to identify whether a shorter or longer context should be chosen for the current model, it appears difficult to identify what context length would be ideal given the temporal range. For example, when comparing Table 1 with results in Figure 1 for different window lengths, I see that GRUs prefer a context length of ~8 in both Cartpole and Stateless Cartpole. However, it is unclear how that information could have been derived from temporal range values in Table 1. I would appreciate if the authors could shed more light on how exactly they believe their metric to be used for tuning architectures in these experiments.

Related to this major concern, I would also like to better understand the assumptions made for the temporal range metric.

2. **Major:** Temporal range identifies the importance of prior inputs on the policy's final output vector but it is unclear why only the final output vector is considered. Imagine a task of 100 steps in which the agent early in the episode observes a code that it needs to enter into a keypad 90 steps away to reach the goal shortly after. Significant memory might be required to generate the action at timestep 90 when the agent reaches the keypad to remember the code, but hardly any memory might be needed for the final action when the agent is already in the last room in front of the goal. Am I missing something, or does the temporal range assume that the final output of the model is somehow the output that is most indicative of the memory required for a task?
3. **Major:** As I understand the temporal range metric, a trained model with a sensible input window length will be needed in order to compute informative temporal range (otherwise no sensible $y_T$ will exist to compute the Jacobian for). If a model is trained with barely any context, then the temporal range metric would not be able to identify much temporal influence of inputs prior to the model context. This appears to be a chicken-and-egg problem. The temporal range is supposed to help practitioners choose a sensible model configuration, but a sensible model configuration is needed in order to obtain an informative metric. It would be helpful if the authors could shed some light on this problem, connected to concerns raised in weakness 1.
## Experiments
4. I find the results shown in Figure 1 and Table 2 quite surprising. I would have expected GRUs and LSTMs to solve Cartpole close to optimally without issue but final performance is stated to be 0.817 and 0.185 for GRU and LSTMs, respectively. I believe these results raise some questions.
	1. **Major:** Do you have any explanation as to why in particular the LSTM performance is so poor in Cartpole?
	2. **Major:** How did you tune/ obtain hyperparameters reported in Table 5 of the Appendix? Based on a single set of hyperparameters being stated, it appears that no specific tuning has been done for any architecture.
	3. For results in Figure 1, do I understand correctly that all algorithms have been trained with full episodes for temporal context and the Figure shows performance for each architecture when the model is only given historical context of limited length during evaluation? If so, I find it also rather surprising how the performance appears to degrade quite sharply when the context is allowed to build up for 64 time steps given that during training these models have presumably typically received longer sequences than 64 time steps. Do you have any explanation as to why performance appears to drop very significantly in most cases for longer context lengths?
5. Figure 2 shows temporal influence profiles for LEM. How do these profiles look like for other architectures such as LSTMs or GRUs?
6. Do you have an explanation as to why the profile of LEM for Copy $k=10$ looks very different from Copy with $k=1, 3, 5$? For smaller $k$, there appears to be a consistent increase in temporal influence over the time sequence with a peak around $k$ steps before the final step which makes sense given the task. I would have expected a similar graph for $k=10$ but instead there appears to be a declining impact from early to later time steps with a small spike ~$k=10$ time steps before the end.

### Questions
1. I would appreciate if the authors could shed more light on how exactly they believe their metric to be used for tuning architectures for RL practitioners. (Weakness 1)
2. Does temporal range assume that the final output of the model in a task is most indicative of the memory required for a task? How would the temporal range be affected in a task that requires memory up to a particular time step some steps before the final step? (Weakness 2)
3. Do you have any explanation as to why in particular the LSTM performance is so poor in Cartpole according to Table 2? (Weakness 4.1)
4. How did you tune/ obtain hyperparameters reported in Table 5 of the Appendix? (Weakness 4.2)
5. Figure 2 shows temporal influence profiles for LEM. How do these profiles look like for other architectures such as LSTMs or GRUs? (Weakness 5)
6. Do you have an explanation as to why the profile of LEM for the Copy $k=10$ task looks very different from Copy with $k=1, 3, 5$? (Weakness 6)

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes temporal range, a method to measure how policies use past observations. The temporal range is defined as the norm of the Jacobian at each timestep with respect to the input $x$

$$ \lVert J_{T, t} \rVert = \lVert \frac{\partial y_T }{\partial {x_t}} \rVert $$

scaled by the distance in time

$$ \sum_{t=1}^T (\lVert J_{T, t} \rVert \cdot (T - t))$$

and then normalized over the sequence of Jacobians 

$$ \frac{\sum_{t=1}^T (\lVert J_{T, t} \rVert \cdot (T - t))}{\sum_{t=1}^T \lVert J_{T, t} \rVert } . $$

It determines how far some function looks back on average. The temporal range is scale invariant due to the normalization. The paper evaluates temporal range on tasks with known optimal lookback, finding that temporal range often tracks observability.

### Strengths
1. A lookback metric is useful in RL POMDP contexts
2. The proposed metric does what it advertises on the tested tasks
3. The paper format is nice and the paper is easy to read

### Weaknesses
1. The contributions of this paper are a bit on the lighter side, comprising a single metric and evaluation on toy experiments.
2. The metric itself appears to be a weighted average. Tasks like CartPole and Copy have strong temporal correlations, looking at the last few timesteps or precisely $k$ timesteps back respectively. Tasks like 3D navigation are likely to have multimodal lookbacks that could provide misleading results with this metric.
3. The performance on some tasks in Table 2 seems underwhelming. For example, none of the learned policies appear to solve cartpole, assuming 0.865 corresponds to surviving for 86.5% of the total timesteps. It is surprising that the LSTM completely fails at cartpole.

### Questions
- What happens for more realistic tasks that could have multimodal lookback? Consider a navigation task where the agent should remember its starting location $t=1$ and its current location $t=T$.
- Can you explain why the scores in table 2 look lower than expected? Would this have an effect on the results presented in figures 1 and 2?

### Soundness
2

### Presentation
3

### Contribution
2
