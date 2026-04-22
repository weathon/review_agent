# Decision-Focused Learning:  Learning to Rank Based on Sample Average Approximation

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Decision-Focused Learning (DFL) improves prediction models by directly optimizing the decision quality of downstream optimization problems, where Learning to Rank (LTR) approaches treat the solution set as a ranking set and design surrogate losses based on the objective function. The DFL-LTR framework exhibits strong applicability; however, it lacks a specially designed subset construction method, which limits its performance. To address this issue, tailored to the intrinsic characteristics of the DFL-LTR framework, we first articulate two fundamental bottlenecks that any ranking subset construction must resolve: ($i$) the infeasibility of fully enumerating the solution space; ($ii$) the resulting upper bound on the loss function family that remains unbroken. To eliminate these limitations, we introduce a ranking subset construction paradigm driven by Sample Average Approximation (SAA). By performing stochastic optimization over minibatches, the proposed method yields an equivalent approximation of the complete solution set, suppresses loss variance to stabilise gradients, and consequently elevates the performance upper bound of the entire framework. Our LTR-SAA subset construction module is fully plug-and-play: it introduces no extra hyperparameters, incurs zero additional time complexity, and remains compatible with the entire family of LTR loss functions. In the latest open-source benchmark (comprising 7 optimization problems), our proposed method achieves SOTA decision quality on 5 of these problems. Compared with other DFL and 2-stage methods, it demonstrates significant performance advantages and generality. Code is available at https://anonymous.4open.science/r/SAA-LTR-33B0.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a modification to learning-to-rank approaches for decision focused learning. The main idea is to change the way that solutions are added to the set that the ranking loss is evaluated over, so that instead of adding based on the optimizers of a single-step perturbed cost, instead optimizers of a running-average cost are included.

### Strengths
Learning to rank is a promising family of approaches to this problem. The proposed approach is very scalable and easily implementable in existing pipelines.

### Weaknesses
My main concern is that the approach is attempting to optimize an internal component of the overall training procedure but it's not clear how well this translates into better results overall. In the results, the relevant comparison seems to be between each Org-* algorithm and the corresponding SAA-*, since this isolates the impact of making the authors' suggested change to the LTR approach. However, it looks like the performance difference from swapping in the new subset construction is very uneven; performance gets worse about as often as it gets better. 

Moreoever, claiming that the new method achieves SOTA on 5/7 tasks is based on taking the best across each of the three LTR approaches, which effectively rewards increasing the variance in the performance of the base methods regardless of whether the average/typical performance improves. In order for this to be truly valid, the best of the three LTR losses should be selected on a validation set, not the test set. It would also need to be counted as a computational cost that the "approach" the authors are implicitly proposing is to run all three base losses and select the best one, and the appropriate baselines would then include competitors that take similar ensembling approaches.  

Alternately, if the three variants are to be assessed individually, there should be more information about their individual performance levels (e.g., average performance & rank across tasks) since it's not clear whether each of them would outperform their original counterpart in an overall sense (ie the metric right now is implicitly "number of tasks with SOTA result" which is interesting but also leaves out a lot of information compared to "average/typical performance across tasks")

### Questions
Comments how we should think about the evaluation (best of three vs evaluating each individually) would be helpful.

### Soundness
3

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
4

### Summary
The paper presents an incremental improvement to the Decision-Focused Learning (DFL) framework, specifically addressing the ranking subset construction issue in Learning to Rank (LTR) with a Sample Average Approximation (SAA) approach. While the proposed method demonstrates improvements in decision quality and experimental results, it largely remains an extension of existing work, rather than introducing a truly novel concept.

### Strengths
- The paper effectively identifies a gap in the existing Decision-Focused Learning (DFL) framework, particularly regarding the subset construction problem within Learning to Rank (LTR).
- The authors provide detailed experimental setups, including datasets, hyperparameters, and code, ensuring that their results are reproducible.

### Weaknesses
- While the paper presents a novel way to address the ranking subset construction problem, it can be seen as an incremental improvement rather than a breakthrough in DFL or LTR. The concept of improving ranking subsets has been explored before, and the novelty of using SAA for this purpose does not represent a major conceptual shift in the field.
- The work primarily focuses on **solving a specific subproblem within the broader DFL-LTR framework**. Although this problem is important, the contribution seems more incremental than revolutionary, and it does not fundamentally change the landscape of decision-focused learning methods.
- The paper does not sufficiently explore how the method scales to larger, more complex, or real-world optimization problems. There is no discussion on how the method performs when applied to dynamic systems or problems with uncertainty.
- The paper could benefit from a more detailed sensitivity analysis. A more comprehensive sensitivity analysis would help readers understand the robustness of the method and how it behaves under different conditions.

### Questions
- While the theoretical analysis provides performance bounds and regret minimization guarantees, how do these theoretical results generalize to more complex real-world problems? Are there any assumptions in the theoretical model that might limit its applicability to a wider range of problems?
- You claim that the method introduces no additional time complexity and is compatible with existing LTR loss functions. However, could there be any practical limitations when applying this method to larger datasets or problems with more complex decision spaces? How does the method handle outliers or noisy data in real-world settings?
- In the experiments, how does the size of the ranking subset impact performance? Does increasing the size of the subset always improve the results, or is there a point where further increasing the subset size leads to diminishing returns?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work addresses the limitations of decision-focused learning with rank-based objectives. The authors apply a sample-based averaging mechanism as an approximation to the complete solution set and derive performance upper bounds for the entire framework. Experimental results demonstrate improvements over existing baselines on test open-source benchmarks in terms of both training time and solution quality.

### Strengths
1. The work addresses a practical and specific limitation of the DFL framework with rank-based objectives.
2. The paper presents a comprehensive theoretical analysis for the sampling-based learning approach, including PAC regret bounds.

### Weaknesses
1. The methodological contribution appears incremental: the work primarily applies well-established stochastic approximation techniques to a specific decision-focused learning problem with rank-based objectives. It would be helpful if the authors could clarify: (a) what novel theoretical insights or algorithmic innovations are introduced beyond standard stochastic approximation, and (b) whether stochastic approximation is already a common approach in learning-to-rank research, and if so, how this work differs from or extends those methods.
2. The overall framework is limited to a specific problem setting, and it is unclear how generalizable this method is to other decision-focused learning scenarios or different types of objectives beyond rank-based ones. A discussion of potential extensions or the fundamental barriers to generalization would strengthen the paper.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a novel framework for Decision-Focused Learning (DFL) using Learning to Rank (LTR) methods and introduces an innovative Sample Average Approximation (SAA)-based ranking subset construction. The primary objective is to enhance decision-making quality in downstream combinatorial optimization problems by refining the selection process of subsets of solutions during training. Specifically, this paper addresses a significant issue in existing DFL-LTR approaches, where the solution space cannot be fully enumerated, leading to inefficiencies in decision quality. By utilizing stochastic optimization over minibatches with the SAA method, the authors propose a plug-and-play subset construction method that does not add extra time complexity or hyperparameters and is compatible with various ranking loss functions.

### Strengths
1. This paper addresses a significant gap in decision-focused learning by proposing a dedicated method for ranking subset construction within the LTR framework. This is a crucial contribution for improving model performance in combinatorial optimization problems.

2. The use of the Sample Average Approximation for subset construction is an insightful and effective way to reduce variance and stabilize the training process. The SAA-LTR method avoids the pitfalls of greedy algorithms that rely on potentially noisy or fluctuating predictions.

3. The proposed method consistently outperforms previous DFL methods, achieving state-of-the-art decision quality in 5 out of 7 optimization problems in the benchmark tests, without introducing the cost of additional time complexity.

### Weaknesses
1. While Lemma 3 proves the variance of ranking scores decays, the experimental results in Figure 3(a) show that this reduction is incremental and contributes little to performance gains. It is essential to clarify why the primary performance driver appears to be the better approximation of $S^*$ rather than the variance reduction, which was initially considered a key advantage.

2. The use of a cumulative cost $c_{rec}$ over the entire training history is a central but potentially confusing aspect. It is treated as an unbiased estimate of the population cost, but computed from a non-stationary sequence as the model parameters $\theta$ are updated. A more detailed discussion on the justification for this in a non-i.i.d. online learning setting would strengthen the theoretical argument.

### Questions
How would the SAA method be adapted for a batch_size > 1?

### Soundness
3

### Presentation
3

### Contribution
3
