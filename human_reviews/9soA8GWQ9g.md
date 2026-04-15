# Beyond the Boundaries of Proximal Policy Optimization

- Decision: Reject
- Scores: 5, 3, 6, 3

## Abstract
Proximal policy optimization (PPO) is a widely-used algorithm for on-policy reinforcement learning. This work offers an alternative perspective of PPO, in which it is decomposed into the inner-loop estimation of update vectors, and the outer-loop application of updates using gradient ascent with unity learning rate. Using this insight we propose outer proximal policy optimization (outer-PPO); a framework wherein these update vectors are applied using an arbitrary gradient-based optimizer. The decoupling of update estimation and update application enabled by outer-PPO highlights several implicit design choices in PPO that we challenge through empirical investigation. In particular we consider non-unity learning rates and momentum applied to the outer loop, and a momentum-bias applied to the inner estimation loop. Methods are evaluated against an aggressively tuned PPO baseline on Brax, Jumaji and MinAtar environments; non-unity learning rates and momentum both achieve statistically significant improvement on Brax and Jumaji, given the same hyperparameter tuning budget.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper introduces outer-PPO, a novel perspective of proximal policy optimization that applies arbitrary gradient-based optimizers to the outer loop of PPO.

### Strengths
1. The idea of the proposed method (outer-PPO) is interesting.
2. This paper provides a comprehensive evaluation of the proposed method.

### Weaknesses
1. In section 5.1, it would make this paper stronger if the authors could explain metrics in detail. 
2. The analysis in section 5.1 could dive deeper into why, in some environments, the methods proposed in this paper work well, while in other environments, these methods are not better than the baseline PPO. For example, the authors could provide some hypotheses and use some experiments to support these hypotheses. 
3. In section 5.2, hyperparameter sensitivity, it would make this paper stronger if the authors could compare the hyperparameter sensitivity of their methods with baseline PPO.
4. Given Figure 8 and Figure 9 in the appendix, it is unclear if the final converged performance of the proposed methods would be better than baseline PPO since it seems all algorithms still haven’t converged.

Suggestions:
1. In the experiments that the authors conducted, it seems the proposed methods do not show better performance than baseline PPO, especially since it is unclear which implementation version of baseline PPO the authors used. Currently, there are many different implementations of PPO, and different implementations of PPO can significantly affect the final performance [1]. For research focus, it is acceptable that the proposed method does not perform better than PPO in general environments. However, it would make the paper stronger if the authors could provide evidence in which situations (such as environments with high dimensional input or output), the proposed method is better than PPO.
2. It would make this paper stronger if the authors could provide the details of the training time of each algorithm with the same number of timesteps.
3. Page 4, Line 195, appendix ??.

[1] The 37 Implementation Details of Proximal Policy Optimization

### Questions
Check the weaknesses section above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The typical PPO algorithm performs an inner loop optimization at each policy update step. With a fixed trajectory data and current policy parameter $\theta$, it optimizes an update to the policy parameters $\theta’$, then applies the new parameters to get more trajectory data. The current paper proposes to modify this update rule by instead considering $\Delta = \theta’ - \theta$ as a replacement to the gradient in any typical gradient-based optimizer. For example, one can consider the update $\sigma \Delta$. In this case $\sigma=1$ will correspond to the standard PPO, but by changing $\sigma$ it generalizes the algorithm. As the original PPO is included in this class of updates, tuning the parameter is guaranteed to at least not decrease the performance compared to PPO. They compare performance on Brax continuous control tasks, MinAtar and Jumanji, and through an extensive hyperparameter tuning procedure claim 5-10% improved performance for the same number of hyperaprameter optimization trials (there were some differences in the hyperparameter optimization procedure between outer-PPO and regular PPO).

### Strengths
- The paper is clear.
- There do a lot of experiments, and the results are reported faithfully.

### Weaknesses
- The performance improvements are at best small (5-10%)
- The performance change could also be due to changes in hyperparameter tuning procedures.
Currently, the authors tune PPO by optimizing 11 hyperparameters for 600 trials (each trial averaged across 4 seeds) using a Tree Structured Parzen method, and then doing a final evaluation at the best parameters for 64 seeds (different from the initial 4 seeds). And the outer-PPO methods are hyperparameter tuned by first doing 500 trials of PPO hyperparameter optimization, then doing an additional 100 trials by a hyperparameter sweep over the outer-PPO parameters. These two optimization methods are inconsistent, as outer-PPO includes a direct sweep over the hyperparameters. Such an inconsistency in tuning may lead to an improved performance for outer-PPO. There is no guarantee that, for example, the standard PPO may also not perform better if it first does 500 trials of optimization with the Tree Structured Parzen, and then does a 100 trial sweep over some chosen important hyperparameters. In general, while I appreciate that the authors tried to tune the parameters exhaustively, 11 hyperparameters are a lot to tune, and with the evaluation noise due to using only 4 seeds it is difficult to create a fully convincing evaluation method where it would be clear that small improvements like 5-10% are due to an improvement in the algorithm, and not some small details in the tuning procedure. Perhaps a simpler experimental protocol would be more convincing. For example, tune PPO  on all tasks, then employ a single fixed outer learning rate (the same one across all tasks). If such a simple procedure lead to an improved performance, it would be more convincing.

- The optimal outer-PPO hyperparameters are different per task.
- No strong justification is given why we should expect the new method to lead to large improvements in performance (whilst it is clear that the performance should not decrease under proper tuning, there is no indication that large improvements are expected.)
- The outer learning rate sensitivity plots show that the peak performance is not much different compared to $\sigma=1$, which corresponds to standard PPO

### Questions
For the outer-PPO hyperparameter tuning, did you also first tune using 4 seeds, and then run a final evaluation using 64 new seeds, just like for the standard PPO experiments?

In general, I did not find the paper interesting, and I would expect much larger improvements in performance (2x improvement or something around that) for me to recommend the work for publication. There is no strong justification for why the proposed method solves a fundamental limitation in PPO or why it would lead to a fundamental improvement in performance. I do not expect any practitioner to adopt the proposed algorithm. For this kind of work, I would expect large empirical performance gains for me to recommend it for publication. Therefore, I do not foresee myself changing my score unless such results are provided in the rebuttal. I would recommend aiming to submit the work to a venue that de-emphasizes significance, and merely looks at correctness of the claims.

Typos:
“Given it’s” → “Given its”
Line 195: “appendix ??” the reference link is missing.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an improvement of Proximal Policy Optimization (PPO), one of the most well-known online policy gradient algorithms.

PPO works by following two nested optimization stages: the inner optimization optimizes a "clipped surrogate objective" that can be seen as a constrained optimization problem, where the optimal solution optimizes the expected advantage while keeping the target policy close to the data; the outer optimization stage simply loops simply updates the target policy and collects new data. 

In PPO, the inner objective is optimized by gradient ascent, while the outer objective is seen simply as a loop.

The authors' idea is to see the inner optimization as a gradient estimation procedure and the outer loop as a gradient ascent procedure. By leveraging this idea, the authors explore different learning rates (which in PPO is inherently set to 1), the application of momentum, and the possibility of "biasing" the initialization of the inner loop by leveraging on the outer gradient ascent.

### Strengths
Originality
--------------

The idea presented in the paper is, as far as I know, novel.

Quality and Clarity
-------------------------

The paper presents the idea in a clear way, highlighting the main research questions well and providing a robust empirical analysis of the proposed algorithm (with very detailed ablation studies). The algorithm is sound and coherent with the investigation objective.

Significance
----------------

I am unsure about the significance of the proposed idea. The paper introduces new hyperparameters that make PPO more complicated rather than more simple. I am unsure whether the complexity introduced is a payoff of the little gains in terms of performance. However, the takeaway message of seeing the outer loop of PPO as a gradient ascent procedure is interesting.

### Weaknesses
As I have mentioned above, I think a weakness of the proposed method is the introduction of new hyperparameters (i.e., outer learning rate and momentum) - with what seems to be little payoff.

Furthermore, I am unsure whether the learning rate is necessary: the hyperparameter $\epsilon$ of PPO already provides a mechanism to control "how aggressive" the policy updates are. While I can acknowledge that the learning rate and the $\epsilon$ are two different terms (i.e., the learning rage $\sigma$ acts in the parameter space, while the hyperparameter $\epsilon$ acts in the "policy" space), I can't see what is the advantage of using $\sigma$ in place of modifying $\epsilon$.

Perhaps, as highlighted in the "Strength" section, the main weak point of the paper is the significance.

### Questions
I ask the authors to clarify what is the advantage of introducing the learning rate instead of modifying $\epsilon$.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
The paper designs a novel framework, called outer-PPO, to further modify PPO’s trust region gradients through an outer loop. PPO conducts several gradient updates using each set of collected data. The proposed framework computes these gradients in the inner loop without updating and combines these gradients into one update with extra stepsize and momentum designs in the outer loop. The proposed algorithms are evaluated on Brax, Jumaji, and MinAtar environments, showing statistically significant improvement on Brax and Jumaji and comparable performance on MinAtar.

### Strengths
The algorithms are clearly introduced with the help of figure representations. The paper makes claims on its empirical performances, which are well supported by the empirical results.

### Weaknesses
The novelty of the algorithm can be explained more. Empirically speaking, outer-PPO with a non-unity learning rate performs best on Brax and Jumaji and functions as the main contribution. However, what is the difference between this proposed algorithm and rescaling the learning rate of the original PPO? Is the empirical result suggesting that stochastic gradient descent can be a better optimizer than the commonly used Adam?

### Questions
1. Line 37: why is “exactly coupled” emphasized? Some newly defined terms, like the behaviour parameters, can be emphasized instead.

2. What is the intuition behind the biased initialization? Trust region is used to reduce the off-policy distribution shift, but biased initialization worsens the situation.

3. Why were Brax and Jumanji chosen instead of MuJoCo or the DeepMind suite?

4. Line 303, during the hyperparameter tuning, does each trial represent a choice of the hyperparameters, and does each agent represent a random seed? Can learning curves for the baseline be provided? How is the final performance of the baseline? Could you help me read Figure 10? What are the points and meanings of the x-axis and y-axis?

5. In Figure 3, how is the metric, optimality gap, defined?

6. The result in Figure 4 is very straightforward. How is this probability metric computed? From Figure 3, there is no significant difference between algorithms. However, the probability of the proposed algorithms being better than the baseline is larger than 0.5. What causes the probability measure to lean towards proposed algorithms?

7. Line 313, “normalizing with the min/max return found for each task across all trained agents (including sweep agents).” What are sweep agents? How is the normalization conducted with the min/max return? In line 470, why does a 0.9 peak normalized return imply less variance compared to 0.7 or 0.8?

8. For the outer loop, is the gradient update applied without using any optimizers?

9. Are these outer loop hyperparameters task-dependent? Could they differ a lot for each task?

10. What does a smaller stepsize on a policy improvement direction suggest? One hypothesis can be the PPO’s gradient direction is not toward policy improvement. However, the algorithm won’t learn in this case, and this hypothesis does not seem to hold.

### Soundness
3

### Presentation
3

### Contribution
1
