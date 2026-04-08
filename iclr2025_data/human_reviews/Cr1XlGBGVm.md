## Human Reviewer 1

### Summary
This work attempts to explain the performance improvement of the Offline Meta RL (ORML) optimization framework. It identifies that the variation of task representation learned through the optimization process if often ignored. Such issue violates the condition of monotonic performance improvements. Thus, addressing task representation shift with carefully designed encoder updates is necessary. Experimental results across two different benchmarks with 3 different types of training objective and data qualities have been presented.

### Strengths
1. This work considers a nuanced issue in traditional OMRL optimization framework. It critically investigates the components and show that variation in task representation is fundamental to monotonic performance improvement. It introduces the phenomena called "task representation shift".
2. Theoretical justification for the claims are well presented and it outlined a complete algorithmic framework. 
3. It presents rigorous experimental validation using several environments from two different benchmarks with 3 types of objective functions and 3 types of data sets.

### Weaknesses
1. The task variation used in the experiments could be improved by including more distinct tasks. 
2. Lack of results with more statistically significant metrics such as interquartile mean with confidence interval. Such comparison would help the reader as the general standard deviation highly overlaps.

### Questions
Can you elaborate more on how different batch sizes induce different task level contexts?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper addresses the overlooked issue of task representation shift in offline meta-reinforcement learning (OMRL), which can prevent consistent performance improvement. 

The authors provide theoretical insights and practical strategies to control this shift, demonstrating that such control enables more stable and higher asymptotic performance. They propose adjustments in training, such as tuning batch sizes and accumulation steps, to manage task representation effectively. Experimental results across benchmarks validate that these adjustments lead to performance gains, highlighting the importance of considering task representation dynamics in OMRL.

Overall, this is a very interesting paper with great potential to inspire future work. I would be willing to increase the score if the authors could address my following concerns.

### Strengths
1. The paper is well-structured and clear.
2. The authors identified a unique challenge existing in offline meta-RL, task representation shift, which is highly novel.
3. The proof provided is detailed and logically rigorous, highlighting a flaw overlooked by previous work: it ignores the variation of task representation in the optimization process. I believe this is the most significant contribution of this paper.
4. The paper concludes with some interesting discussions, which have the potential to motivate future research.

### Weaknesses
1. The algorithm box does not clearly explain how $k$ and $acc$ are utilized. Adding a brief explanation in the red-highlighted part of the algorithm box about how these are calculated would make the algorithm more understandable.
2. The experiments are limited and need improvement; it would be beneficial to verify the impact of task representation shift in more diverse testing scenarios.
3. The results largely depend on the settings of hyper-parameters, such as $k$ and $acc$, which seem to be unstable. The paper lacks an analysis of the experimental effects caused by adjusting these two parameters.

### Questions
1. The setup in Section 4.3 is a bit confusing. Could the authors clarify what "accumulation steps of task representation shift" refers to? Also, does setting $k = 2 \times bs$ refer to the initial value for training?
2. It would be better to explain in the appendix how the cross-entropy-based loss can replace the loss in FOCAL, preferably by providing the corresponding expression.
3. The motivation for using the cross-entropy-based algorithm is somewhat unclear. Could you explain why it replaces the distance metric learning loss?
4. From the experimental results, the improvement seems marginal. Could you analyze the reasons behind this?
5. The authors state in section 6.2, "We recognize that the visualization result can be seen as an auxiliary metric to assist in determining the task representation." Why is the visualization result insufficient to fully represent the true task representation? The visualized convergence results being imperfect yet leading to better outcomes seem counterintuitive.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper explains the reasons for the effectiveness of previous context-based offline meta reinforcement learning methods and introduces a task representation shift problem. The authors further demonstrate both theoretically and empirically that improving the update process of the context encoder for this problem can significantly enhance the performance of the original COMRL algorithm.

### Strengths
1.Provides a new perspective for learning better task representations in COMRL.  
2.The paper includes a detailed mathematical derivation process.  
3.The authors provide their code implementation in the supplementary materials.

### Weaknesses
Experimental Section:
1. The experimental results are obtained only within the improved FOCAL framework, lacking experiments on important baselines such as CORRO [1], CSRO [2]. I believe conducting experiments in just one framework is insufficient to verify effectiveness. It would be more convincing if significant results could be demonstrated across multiple baselines. 
2. The authors only consider four environments, while in CSRO and UNICORN, each algorithm considers six environments under Mujoco.

### Questions
1. The authors' description of the task representation shift is unclear. Can it be understood as the phenomenon where the task representation deteriorates after updating the encoder? A specific example would be helpful for better understanding this point. 
2. The authors introduce a large number of assumptions in Section 4. Has the reasonableness of these assumptions been justified? 
3. When choosing the Ant environment for experiments under Mujoco, was it randomly selected, or was it only the environment that showed the best results? 
4. How are the hyperparameters selected for different environments? It seems from the results in Figure 2 that the experimental outcomes are quite sensitive to the choice of k/acc. Is it possible to analyze the results of different hyperparameters in relation to the characteristics of the environments themselves?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper focuses on task representation shift in context encoder updates when the context encoder is updated in offline meta reinforcement learning (OMRL). In particular, the paper analyzes the relationship between performance improvement in policy updates and the amount of data used for context encoder updates.

The paper claims two contributions:

(1) Identifying task representation shift as a major issue in context based offline meta reinforcement learning. The task representation shift makes it harder to satisfy the condition necessary for monotonic performance improvements. For the "major issue" claim, the paper claims to provide empirical evidence.

(2) Proof of monotonic performance improvement taking into account task representation shift and number of samples used to improve the context encoder. This information can be used in practice, for example, such that an algorithm updates the context encoder only when the improvement due to policy updates is sufficiently large such that the number of samples used for the context encoder updates is sufficient.

### Strengths
According to my knowledge the theoretical analysis of task representation shift in context based offline meta reinforcement learning is original and may be of interest to researchers working in this domain and potentially others where context encoders are used.

For claim (1), the theoretical analysis is a solid contribution in case the authors can clarify the questions/comments further below. The insight that task representation shift influences the requirements for monotonic performance improvements is valuable.

For claim (2), mathematical proofs under the made assumptions appear correct.

The paper is overall well written and understandable.

### Weaknesses
For claim (1), regarding the claim that task representation shift is a "major issue", the experimental results do not at the moment provide evidence for this as claimed in the paper. Statistical significance analysis of the results should provide further information.

The Assumptions 4.7, 4.8, and 4.9 need to be motivated in more detail. In particular, in Assumption 4.9, assuming that fitting error decreases inversely proportional to the number of samples is questionable.

vol(Z) needs to be explained.

Other technical details, discussed below, need to be described in more detail.

DETAILS:

In more detail, the paper says "As shown by Figure 2, even with minor changes to the algorithms, the performance improvements are substantial." but that the performance improvements are "substantial" is not at all obvious but rather a result of random chance.
Statistical significance testing is needed to draw conclusions about the experimental results. This applies to all the results.

In "vol(Z) as the volume of the task representation coverage simplex.", please define "coverage simplex". What exactly is it? Furthermore, on lines 860 - 863, Lemma 8.2 is used. However, this Lemma applies in the case of discrete values and uses the number of values '|A|' in '2^|A|' but here '|A|' is replaced with vol(Z) resulting in 2^vol(Z). Why can this be done? This needs detailed explanation.

In Definition 4.5, I do not understand what "among the expectation of tasks before update of the context encoder and the policy" and "among the expectation of tasks after update of the context encoder and the policy." mean. Maybe the word "among" is here confusing. Can you provide a more explicit detailed definition? Just provide the equations for J^1(\theta_1) and J^2(\theta_2)?

On Line 281, the paper refers to Eq. (32) which has not been introduced yet.

Notation: using multiple characters to denote quantities such as "bs", "acc" etc. is not a good way. If this kind of textual description for variables is desired, one way is to use something like N_{\text{acc}}.

The notation is slightly confusing. Now \theta denotes policy parameters but it would be good to add also a symbol to denote the context encoder parameters to distinguish clearly in the equations what is being optimized.


RELATED WORK:

In "While ContrBAR (Choshen & Tamar, 2023) also benefits from the performance improvement guarantee in the OMRL setting, it is specifically served for its own framework", what does "it is specifically served for its own framework" mean? Since the proofs for performance improvement guarantee are one of the claimed contributions in this paper it is important to describe this in sufficient detail and discuss the differences.


LANGUAGE/PRESENTATION:

In "weakens the condition necessary for monotonic performance improvements, and may therefore violate the monotonicity.", rephrasing may be needed. "weakens" is slightly misleading since the sufficient conditions are actually stricter, not weaker, that is, when taking the task distribution shift into account larger policy improvements are needed to satisty monotonicity according to the analysis in this paper.

The sentence "However, ours considers the variation of task representation ignored by the previous training framework by imposing the extra condition to determine whether needs to update the context encoder." is missing some words.

On line 261: "As shown in Corollary 4.4, the monotonic performance improvement can be guaranteed with only better approximation to Z^*(·|x).": I maybe understand the intention of the text here but would be good to describe this more explicitly, that is, that Z(·|x) should be close to Z^*(·|x) such that the lower bound is small enough for finding a policy that improves on the old policy?

"we need to find the positive C to improve the performance monotonically." ->
"we need to find a positive C to improve the performance monotonically."

"Center around this motivation" -> "Centering around this motivation"

"an policy learning algorithm" -> "a policy learning algorithm"

### Questions
Update: I am happy with most of the authors' answers and recommending accepting the paper. I raised the rating to a strong 6.

Old questions:

1) Statistical significance analysis should be run in all experiments or claims on experimental results changed significantly, 2) Please answer all technical questions and comments.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4