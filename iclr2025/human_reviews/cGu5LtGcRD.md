## Human Reviewer 1

### Summary
This paper proposed to strengthen the HL-Gaussian method by having dynamic interval on which the value function is projected into a categorical distribution. The existing works use a pre-specified fixed-length interval $[v_\text{min}, v_\text{max}]$ which can be hard to estimate. The authors conducted theoretical analysis and revealed that this projection incurs two losses:  projection loss and truncation loss. Based on the analysis, the authors proposed a learnable dynamic interval scheme in an attempt to minimize truncation error and projection error. The method was plugged into DQN and SAC and tested on standard benchmark environments.

### Strengths
The paper is overall well-written and sufficiently motivated. The preliminary section introduced the background and related work in an accessible way, paving the way for later analysis and proposal. Theoretical analyses are novel and intuitive, indicating an adaptive interval could provide an edge in minimizing the truncation and projection error. The proposed learnable scheme was combined with popular base algorithms to show quantitative improvements.

### Weaknesses
My concern for the paper is that its scope is limited and might not be interesting to the community, considering that the adaptive HL-Gaussian is an incremental improvement on the HL-Gaussian, and no significant higher scores can be seen from Figure 4-6. Maybe because the learning horizon was not suffiicently long, considering the original C51 algorithm ran 200m frames.

Another concern is the results seem to be limited as well. Central to the proposal, it would be more convincing to show how the dynamic interval $\zeta$ would change as learning proceeds, e.g. $\zeta$ can shrink drastically in some cases but being wide in the other. Consider those environments in figure 5 and 6 where no substantial difference was observed, was it because of $\zeta$ converged/flucturating around a number? And how was the loss of learning $\zeta$?  A question related to this is: if $\zeta$ did not change drastically but rather stayed around a fixed value, could we simply normalize the reward into $[0, 1]$, and replace the interval $[v_\text{min}, v_\text{max}]$ with $[0, \frac{1}{1-\gamma}]$? This reward normalization trick is frequently used and we do not need a separate learning scheme then.

### Questions
Please refer to the above for my questions.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper introduces a task-agnostic, value-function-aware mechanism for adaptively adjusting the intervals of HL-Gaussian, an existing approach that replaces the mean squared error loss function with cross-entropy loss in reinforcement learning. To achieve this, the paper provides a theoretical analysis of the errors arising from discretization and truncation in HL-Gaussian, guiding the development of a solution to minimize these errors. Experiments on Atari 2600 games and Gym Mujoco demonstrate the advantages of the proposed approach.

### Strengths
The paper addresses a novel problem: adaptively setting intervals for HL-Gaussian, distinguishing it from existing approaches that rely on fixed, predetermined intervals. This adaptive approach is motivated by a theoretical analysis of the discretization and truncation errors in HL-Gaussian.

### Weaknesses
The paper's major weaknesses include unclear notation and an insufficient connection between the theoretical analysis and the proposed solution.

Regarding notation issues, it may be partly due to my background, but the paper's notations are quite confusing, making it challenging to verify the accuracy of the theoretical analysis. Some specific examples include:

1. Equation (1) says $Q\^\*(s\_t,a\_t) = (T\^\* Q\^\*)(s\_t,a\_t)$. It seems to be incorrect as it means $T\^*$ is an identity function. The same issue is in Equation (2).

2. In lines 129–130, it appears that $z$ is treated as a random variable, yet the paper describes it as a distribution. Additionally, this distribution is said to be taken from a set of discrete locations, which seems unusual.

3. In Equation (3), $\delta_{z_i}$ is undefined. If it is the Dirac delta function, then is $Z(s_t,a_t)$ a probability? But if $Z(s_t,a_t)$ a probability, then then $Q(s_t,a_t)$ in Equation (3) is clearly incorrect.

4. In lines 138-139, why does $p_i(s_t,a_t)$ exist if the true Q value function does not belong to the interval $[v_{min}, v_{\max}]$?

5. Why does $p_i(s_t,a_t)$ in Equation (6) satisfy the condition in line 139?

6. In the proof of Proposition 3.1: What is "the distributional value function Q"? Why does $h(x)$ is supported in the range $[v_{min}, v_{max}]$? Why is $p_\mu$ independent of the learned variable $x:=(s,a)$ even though $p_{\mu} = [p_1(s,a),\dots,p_m(s,a)]$? There are also errors in lines 704, 711.

7. In line 198, $\delta$ is defined when there is a bin containing $\mu$. However, in line 212, it is used even for the case when $\mu$ is not in the interval (no bin containing $\mu$).
   
8. In line 212, $\mathcal{E}_{discretization} \le 0$, what does this mean?

9. In the theoretical analysis, the paper uses the same notation $Q$ and $\hat{T}$ for both the action-value functions computed using a continuous range and those computed using a truncated discrete range. However, these two procedures are different, and the error between them accumulates over the Markov chain in the Bellman equation. It appears that the paper does not address this error accumulation. 

10. Line 777 is not acceptable in stating that a certain term is constant and that its value is $o(1)$ by "empirically verified". Additionally, the meaning of $o(1)$ in this context is unclear.

Regarding the solution, it appears to be largely disconnected from the theoretical analysis. Specifically, there is no assessment of the algorithm’s projection error. Furthermore, while this error depends on both the number of bins and the interval size, the proposed solution optimizes only the interval size while keeping the number of bins fixed. More importantly, the solution does not incorporate any findings from the theoretical analysis.

### Questions
Please see the above weaknesses.

### Soundness
1

### Presentation
1

### Contribution
2

### Rating
3

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper introduces Adaptive HL-Gaussian (AHL-Gaussian), an approach to learn categorical distributions for value functions in reinforcement learning. AHL-Gaussian improves upon the previous HL-Gaussian approach, which uses a fixed interval for the categorical distribution's support. The primary limitation of HL-Gaussian is that a poorly chosen interval can restrict the value function to an inaccurate range, leading to significant truncation and discretization errors.

To address this, the authors conduct a theoretical analysis of how interval selection affects error rates, clarifying the failure cases of the standard HL-Gaussian. Based on these insights, they propose AHL-Gaussian, which dynamically adjusts the support interval to align with the evolving value function. Experimental results demonstrate that AHL-Gaussian outperforms the static approach, advancing state-of-the-art performance.

### Strengths
The main strength of the paper is the theoretical analysis of the dependency between projection error and the support interval. The derivations are easy to follow and are supported by explanatory figures. The section concludes with a very clear and concise summary of the main findings.

The experimental section is also well-structured and the experiments' goals are clearly explained. The authors use Atari and Mujoco and apply AHL-Gaussian to DQN, TD3, and SAC.

### Weaknesses
In my eyes, the experiments are not fully convincing. I.e., the improvement over other methods seems to be environment dependent and pretty minor overall. Besides that, I doubt that the comparison is fair, as all baselines use fixed interval [-10, 10] for all environments.

- It is unclear why the interval of [-10, 10] is used for all non-adaptive methods for all environments. It looks like an unfair comparison to me. I would want to see how AHL-Gaussian compares to baselines with more adequate intervals.

- In Section 4.3, auhtors compare their method against non-learning approaches for updating the interval. As $1.1 * \text{Max Target}$ performs better than $1 * \text{Max Target}$, I would expect to see experiments with larger values. Besides, it seems natural to test other non-learning approaches. e.g., setting interval as $[\frac{r_{min}}{1-\gamma},\frac{r_{max}}{1-\gamma}]$, where $r_{min}, r_{max}$ are min/max rewards observed during training.

### Questions
My main question, as explained, is related to practical usability of the method. For which problems do the authors recommend using AHL-Gaussian as opposed to treating the support interval as a tunable hyperparameter?

Typos:
Line 486: Typo ``TTo’’.

Line 711: Missing $\mathbb{E}$

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper introduces a novel adaptive method for value based RL using cross entropy loss with discretized value space. Prior works have shown that using cross entropy loss instead of mean squared error improves the performance of RL algorithms but it also leads to suboptimal performance if value space is poorly discretized. The authors show that static discretization in HL-Gaussian leads to a projection error which increases with increase in interval size. The then propose an adaptive method that dynamically adjusts the support interval of value function during training by minimizing the projection error. They conduct extensive experiments to demonstrate the efficacy of their method over other baselines on various benchmark environments.

### Strengths
1. The paper theoretically characterizes the deficiencies of the standard HL Gaussian method based on projection error.
    
2. It proposes a novel adaptive algorithm that dynamically adapts the value range and corresponding discretization by minimizing the projection error.
    
3. Their method can be integrated with various RL baselines like DQN, SAC, TD3 and it shows nearly consistent improvement in performance when compared to standard RL training.

### Weaknesses
1. The proposed algorithm lacks theoretical insights on why quantization should help in training. There are no theoretical guarantee on performance of the algorithm.
2. The experiments are mostly conducted on simple RL environments and it is not yet clear if the method would work well on more complex real world environments like Dota, Minecraft etc.

### Questions
1. How does the choice of bin quantization(number and varying size) affect the performance of the algorithm? Why not adaptively optimize them as well?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3