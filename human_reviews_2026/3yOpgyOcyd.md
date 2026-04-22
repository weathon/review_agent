# PitStop: Physics-Informed Training with Gradient Stopping

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Physics-informed learning offers a powerful approach for modeling physical systems by enforcing governing equations directly within the training process. However, optimizing such models remains inherently challenging, especially for large systems, due to the ill-conditioned nature of these residual-based loss functions. In this paper, we critically examine the limitations of classical optimization techniques by developing a comprehensive theoretical framework for physics-informed setups, including insights on convergence guarantees, convergence speed and fixed points. Next, we introduce PitStop, a novel optimization method for physics-informed training based on gradient stopping, which overcomes the limitations of classical methods by backpropagating feedback differently from the standard chain rule of calculus. The method is motivated and mathematically analyzed in our theoretical framework, incurs no additional computational cost compared to standard gradients, and achieves superior results in our experiments. Our work paves the way for more scalable and reliable physics-informed model training by fundamentally rethinking optimization paradigms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PitStop to optimize physics-informed loss that modifies gradient backpropagation by stopping certain gradient flows, thereby deviating from the standard chain rule. The authors argue that classical optimization methods are ill-suited for physics-informed tasks because the residual-based loss functions are ill-conditioned and their approximate minima often fail to align with the supervised objective. PitStop is theoretically grounded using linear fixed-point theory and is shown to ensure convergence under mild conditions. The method requires no extra computational cost compared to standard gradients and exhibits improved convergence rates and stability. Extensive experiments on linear and nonlinear systems, including Burgers’ and Navier–Stokes equations, demonstrate faster convergence and better supervised accuracy compared to standard optimizers.

### Strengths
1.The paper provides a comprehensive framework that connects optimization theory and physics-informed learning, especially through non-symmetric fixed-point operators and gradient stopping.

2.Theorems on convergence, rate, and fixed points are well-structured and backed by detailed proofs.

3.Reinterpreting backpropagation itself, rather than tuning weights or loss terms, is an innovative contribution to the field.

4.The method maintains computational efficiency, making it suitable for large-scale physics-informed neural network (PINN) problems.

### Weaknesses
1.The analysis is confined primarily to linearized systems and time/space-discretized settings, raising uncertainty about applicability to fully nonlinear PDEs or continuous-time PINNs.

2.The method requires explicit temporal discretization, potentially limiting use in frameworks that rely on automatic differentiation.

3. While the proofs seems rigorous, the implementation details and intuition behind gradient stopping, as well as some symbols in the proofs could be clarified further for non-theoretical audiences.

4. The effect of hyperparameters (e.g., $\alpha$ , learning rate $\eta$) on convergence robustness could be elaborated.

### Questions
1.Can the convergence guarantees extend to nonlinear PDE operators or models with data-driven mixed losses?  

2.How sensitive is the method to the choice of α (initial condition weight) and step size η in practice?

3.The physics-informed loss (6) is presented in a discretized form, i.e. in the settings the temporal and spatial domains are setting to equidistant points. This is too restrict. As we know, one of PINN’s advantage is its meshless property, but (6) is the form of finite difference scheme loss, reduced the advantage of the continuous physics-informed loss. In this setting we can solve (6) by solving linear algebraic equations like in the finite difference method, so why we need another optimization method to solve (6)? How to extend the method to nonregular domain? Can the method extend to continuous time/space physics-informed loss?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides an interesting study on the effect of gradient stopping in the optimization of time-dependent PINN loss. The authors propose to stop gradient propagation during autoregressive residual calculation. The experimental results show the benefit of this approach.

### Strengths
This work performs systematic study of the effect of gradient stopping in training autoregressive PDE solvers, a popular trick that hasn't been properly investigated before. The authors analyze the effect of gradient stopping in linearized fix point setting, which is simple but illustrative perspective. The experimental results shows the effectiveness of this approach comparing to default optimizers.

### Weaknesses
I have a few main concerns:

1. The problem setting of PINN is non-standard. The authors say "We restrict ourselves to a setup where the time derivative is discretized directly through a numerical time-stepping scheme, rather than being computed via automatic differentiation". For most PINNs, there's no time-stepping and time derivatives are calculated very efficiently using AD. Even for PINOs, time derivatives can be evaluated using higher-order scheme.

2. The comparison to standard PINNs is missing. It is likely that the mismatch of GD is due to the way time derivatives are calculated.

### Questions
1. Is this PINN setting general enough?

2. The experiments lack basic setting. What is the input/output of the network? If it's a standard MLP with time input, how is the time-stepping done? If not, how many roll out steps are done for both methods?

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
4

### Summary
This paper proposes new optimization solutions for the key challenges of PINN optimization. The problem solved is very meaningful, and the method is also novel, but the insufficient evaluation of the results has affected the performance of the method. If the author solves my concerns and demonstrates that the proposed method has significant advantages, I will increase the rating.

### Strengths
1. A new physics-informed loss optimization method has been derived through theoretical derivation, which has theoretical basis.
2. Verified on 4 PDEs.
3. The problem being solved is urgent and important.

### Weaknesses
1. In Line 97, why can such an assumption be made? This approach yields different results from automatic differentiation, and different discretization methods can also lead to different outcomes.
2. A typo is in Line 131.
3. The experimental section only compares one curve. If experiments can be conducted under different boundary conditions and initial conditions, and statistical analysis can be added, it will improve the credibility of the paper.
4. PINN will use second-order optimizers such as LBFGS after Adam, but this baseline is not mentioned in the paper.
5. The paper lacks an introduction to the baseline. Is the calculation of PDE loss done using discrete methods or automatic differentiation in time derivatives?
6. Can you draw an error graph for Figure 4, as it appears that gradient descent is more similar to solution.
7. The paragraph starting from line 474 appears to have no relation to the previous part, and there is no relevant evidence before it. This paragraph makes the paper unscientific.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents an optimization framework named PitStop for optimizing physics-informed objective functions. In the current paradigm, most work optimizes Physics-informes loss, which corresponds to the supervision loss of the temporal derivative of the governing physics model. By optimizing this derivative-based loss, the existing methods aim at attaining the optimal Supervision loss. However, this work points out that the optimal points of these two different loss functions are not always the same, and based on this observation, proposes PitStop as an alternative for the existing classical gradient-based optimization methods like Gradient Descent and Gauss-Newton methods. To analyze the properties of the PitStop, the authors use the lens of linear fixed-point iterations. With this interpretation, the authors claim that while PitStop could converge to the worse fixed point than the Gradient Descent in terms of the physics-informed loss, it eventually converges to the better Supervision loss faster than these methods. The authors provide experimental results on one toy example, harmonic oscillator, Burger's equation, and Navier-Stokes equation, and show that PitStop converges faster to the better solution than the conventional methods.

### Strengths
- The motivation is good, and authors provided a thorough theoretical analysis of their approach. 
- With simple experiments, the authors effectively show that the current approach to minimize the Physics-informed loss does not always align with the final goal to minimize the Supervision loss. They also sho

### Weaknesses
- The overall description was hard to follow. More intuitive explanation about why PitStop works would be appreciated.
- The overall description was hard to follow, making it challenge to reproduce the results
- Even though the authors gave detailed definitions and analysis of their approach, it is unclear how we can implement it. Seeing the equation 7 and 8, I feel like we can reproduce the results by only cutting the gradient flow across different time steps, but I'm not sure if I understand it correctly. It would be helpful if the authors provide an explicit algorithm (or pseudo code).
- The experimental results are not convincing that this approach is an overall better approach than existing methods.  For example,  in   Figure 4, I believe (d) GD gives better result than PitStop.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
3
