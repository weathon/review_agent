# Theoretical foundations of curriculum learning in linear RNNs

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Pretraining models with a curriculum of simpler tasks is a common approach to speed up training. However, it is unclear what aspects of task structure drive learning speed, and how to practically choose the curriculum based on theoretical principles. Using recent advances in the analysis of learning trajectories in linear RNNs (Proca et al., 2025), we study a simple but informative example of performing two integration tasks in sequence, and ask what aspects of their task structure lead to faster overall learning of the second ``target'' task. We show both analytically and through simulations that even for tasks that are similar in their geometry, sequencing them based on the strength and scale of the input-to-target correlations can provably enhance learning speed. A surprising result from our theory that goes against conventional wisdom is that training intermediate tasks to suboptimal accuracies can be more beneficial to learning speed, rather than training them to convergence. These results provide foundational insight into how task similarity forms both a theoretical and practical basis for curriculum learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the training time of linear RNNs in two scenarios: (i) learning a single "target" task; (ii) first learning an "easier" task and then learning the "target" task when initialized from the first task. By simplifying the problem via an alignment assumption between tasks, the authors derive closed-form solutions for the time required for both scenarios. These times depend on the task via the covariances of the data. They analyze when (ii) is faster than (i) by considering training the first task to an $\epsilon$ error. They validate their analysis in linear RNNs and provide further evidence from nonlinear RNNs.

### Strengths
The paper has a refreshing style based on training dynamics analysis for curriculum learning. It provides sharp and testable predictions. As far as I'm aware, this is novel. The theory is also presented clearly.

### Weaknesses
1. The main issue with the paper is the assumption that the training time of a gradient flow dynamics quantifies the hardness of the task. The speed of learning is sensitive to the learning rate. Based on the authors' definition, one can adjust the learning rate to change the hardness of the problem, which does not make sense. This is very much reflected in the results of the paper. Using stronger singular values in the first task is a way of increasing the learning rate.

2. Related to the point above, the hardness of a task should incorporate something about the statistical hardness of the task. It is not at all clear that the notion the authors proposed aligns with the statistical difficulties of problems. For example, consider a regression problem where the data is scaled by some constant. This is the same problem and the difficulty should be the same. Therefore, the results need to take into account some form of normalization, which is beyond the learning rate discussion. Ideal results with training dynamics would look like this: the first task allows an efficient recovery of the domain of the target task and then the second task is learned much more efficiently.

### Questions
1. Can you comment on the example I have given regarding scaled regression? Is it true that your model considers this as a simpler task? Can't you derive the benefits of curriculum with simple learning rate scheduling then? 

2. What is the difference between linear networks and linear RNNs when we just focus on prediction at the last token? I see that we get more complicated covariances but aren't they the same up to some transformations in the data?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the effect of curriculum in a linear RNN setting with fixed recurrent weights and trainable input-to-hidden and hidden-to-output weights. It identifies task statistics that yield faster learning from a curriculum as opposed to direct training. The results are verified in simulation and qualitatively hold in ReLU RNNs.

### Strengths
The paper presents new exact solutions to the learning dynamics of a linear RNN with fixed recurrent weights.

The results identify the key dataset properties in this setting which enable curriculum learning to outpace direct training.

The paper is clear and the figures are to a high standard.

### Weaknesses
The studied model is a particularly simple form of RNN in which the recurrent weights are not trainable. The paper could be strengthened by investigating whether qualitatively similar results hold when recurrent weights are trained as well.

The tasks studied are versions of learning to integrate an input. While this is an interesting task, it would be useful to understand the limits of the theory for other types of tasks.

The paper could benefit from discussing other theoretical work on curriculum, for instance the work of Stefano Sarao Mannelli.

### Questions
Does a qualitatively similar picture hold when recurrent weights are trained?

Does a feedforward network yield similar optimal curricula or does recurrence make a different set of curricula beneficial?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper develops a theoretical account of curriculum learning (CL) in linear RNNs and validates key predictions in simulations (with a small nonlinear check). Using recent analyses of learning trajectories in linear RNNs (Proca et al., 2025), the authors study training two related tasks in sequence and ask when pretraining on task T1 accelerates learning of target task T2. The main results are: (i) sequencing by task covariance matters—pretraining that increases input–target covariance strength and scale can provably reduce training time for T2; (ii) counterintuitively, stopping T1 early (sub-optimal accuracy) can yield faster T2 learning; and (iii) the theory gives explicit time-to-convergence formulas and phase-plane predictions, corroborated numerically.

### Strengths
1. The abstract and discussion cleanly articulate what structural aspects of the two tasks (covariance strength/temporal structure) drive speedups, with the “stop T1 early” prediction highlighted as surprising.

2. The appendix lays out the model and derivation path (extending recent work), including the linear RNN equations and loss setup.

3. The paper motivates CL/pretraining’s mixed empirical record and positions the analysis within that context.

4. Simulations, including a nonlinear-RNN sanity check (Fig. 4), exhibit the same qualitative dependencies predicted by the theory across different task-covariance regimes.

### Weaknesses
1. Results are derived for pairs of “similar” tasks with shared geometry, and training focuses on input/output weights with predefined recurrence—limiting generality.

2. The main emphasis is learning speed; robustness/generalization/noise sensitivity are left to future work.

3. The nonlinear experiments are positioned as qualitative trend checks rather than tight quantitative tests of the theory.

4. The paper does not cite some relevant work, eg papers that analyzed curriculum sequencing, representational transfer, and gradient-alignment mechanisms in RNNs, like Kepple, Engelken, Rajan, ICLR, among others. Those works anticipated aspects of the current analysis—particularly the role of inter-task geometry in shaping convergence—so their absence weakens contextualization. The related-work section should explicitly discuss how this study’s theoretical framework extends or differs from that earlier line of curriculum-learning theory.

### Questions
1. Can you add quantitative error bars comparing predicted vs. observed time-to-convergence in nonlinear RNNs across the covariance sweeps in Fig. 4?

2. What breaks first if T1 and T2 do not share task geometry (e.g., rotations/compositional changes)? Any preliminary results on the “rotation to factorized regime” you outline?

3. How would updating Wh alter the phase-plane and convergence-time analysis? You note it as feasible for long-timescale computations—any tractable subcase?

4. Given the “stop early” result, could you propose a curriculum-selection heuristic (e.g., proxy measures of input–target covariance strength/scale) and a stopping rule for T1? (Pointers exist but a recipe would help.)

5. Any insight on whether the same covariance principles predict robustness/generalization improvements, not just speed?

### Soundness
3

### Presentation
3

### Contribution
3
