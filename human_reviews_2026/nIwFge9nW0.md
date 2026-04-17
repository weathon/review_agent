# DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Recent GRPO-based approaches built on flow matching models have shown remarkable improvements in human preference alignment for text-to-image generation.
Nevertheless, they still suffer from the sparse reward problem: the terminal reward of the entire denoising trajectory is applied to all intermediate steps, resulting in a mismatch between the global feedback signals and the exact fine-grained contributions at intermediate denoising steps.
To address this issue, we introduce \textbf{DenseGRPO}, a novel framework that aligns human preference with dense rewards, which evaluates the fine-grained contribution of each denoising step.
Specifically, our approach includes two key components: (1) we propose to predict the step-wise reward gain as dense reward of each denoising step, which applies a reward model on the intermediate clean images via an ODE-based approach. This manner ensures an alignment between feedback signals and the contributions of individual steps, facilitating effective training;
and (2) based on the estimated dense rewards, a mismatch drawback between the uniform exploration setting and the time-varying noise intensity in existing GRPO-based methods is revealed, leading to an inappropriate exploration space.
Thus, we propose a reward-aware scheme to calibrate the exploration space by adaptively adjusting a timestep-specific stochasticity injection in the SDE sampler, ensuring a suitable exploration space at all timesteps.
Extensive experiments on multiple standard benchmarks demonstrate the effectiveness of the proposed DenseGRPO and highlight the critical role of the valid dense rewards in flow matching model alignment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Recent GRPO-based flow matching models improve human preference alignment in text-to-image generation but still suffer from sparse rewards, where a single terminal reward is applied to all denoising steps. DenseGRPO addresses this by introducing dense, step-wise rewards that evaluate each denoising step via an ODE-based reward model. Furthermore, it adaptively calibrates the sampler’s exploration based on reward feedback, ensuring appropriate stochasticity across timesteps and yielding stronger alignment performance.

### Strengths
- This paper proposes a step-wise reward for RL finetuning.
- This paper designs an adaptive noise scale.

### Weaknesses
- There is no 'return' in Algorithm 1.
- There are many metrics in Table 1, and they have different directions for better performance. It will be clearer to add an arrow after them to denote which direction (up/down) is better.
- Only results of $a=0.7$  are provided in the ablation 'Effect of Exploration Space Calibration'. What about other $a$? Is $a=0.7$ the best hyperparameter?
- The return defined in reinforcement learning is $R_{t_0}=\sum_{t=t_0}^\infty \gamma^{t-t_0} r$, so the reward is not $R_{t+1}-R_t$. 
- There is no code repository and reproducibility statement.

### Questions
- Why does the noise scale aim to guide the distribution of dense rewards to be balanced? Why not guide the distribution to fall in the positive reward region?
- In the ablation study 'Effect of Different ODE Denoising Steps', what about using $n=t/C$, where $C$ is a constant? This is proportional to $t$, while faster than $n=t$.

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
3

### Summary
This paper introduces DenseGRPO, a novel reinforcement learning framework for aligning flow matching models with human preferences in text-to-image generation. The key innovation addresses the "sparse reward problem" in existing GRPO-based approaches, where a single terminal reward for the entire denoising trajectory is naively applied to optimize all intermediate steps, creating a mismatch between global feedback and fine-grained step-wise contributions.

DenseGRPO proposes two main components: (1) **Step-wise Dense Reward Estimation**: Instead of using a single terminal reward, the method predicts reward gains at each denoising step by applying a reward model to intermediate clean images obtained through ODE-based denoising. This approach assigns $\Delta R_t^i = R_{t-1}^i - R_t^i$ as the dense reward for each step, aligning feedback signals with actual step contributions. (2) **Reward-Aware Exploration Space Calibration**: Based on the estimated dense rewards, the authors identify that existing uniform noise injection in SDE samplers creates inappropriate exploration spaces. They propose an adaptive scheme that adjusts timestep-specific stochasticity  to balance dense reward distributions across all timesteps.

Experiments on three benchmarks (compositional image generation, visual text rendering, and human preference alignment) demonstrate superior performance over Flow-GRPO and CoCA baselines. The method achieves notable improvements, particularly +1.01 PickScore on human preference alignment, while maintaining good generalization across multiple evaluation metrics.

### Strengths
**Originality:** The paper presents a genuinely novel perspective on the sparse reward problem in flow matching model alignment. While dense rewards have been explored in text generation and diffusion models, the specific ODE-based approach for estimating step-wise rewards without additional specialized models is creative and practical. The identification of exploration space mismatch through dense reward analysis is an insightful observation that leads naturally to the second contribution. The paper distinguishes itself from CoCA by actually training with step-wise rewards rather than just using trajectory-wise signals proportionally weighted.

**Quality:** The experimental validation is comprehensive, covering three distinct tasks with multiple evaluation metrics. The ablation studies systematically validate each design choice: dense vs. sparse rewards (Figure 6a), exploration space calibration (Figure 6b), and ODE denoising steps (Figure 6c). The visualization of dense reward distributions (Figure 3) provides compelling evidence for the exploration space problem. The method demonstrates consistent improvements across all tasks, with particularly strong results on human preference alignment. The authors appropriately acknowledge discrepancies in reproducing baseline results with UnifiedReward and provide their own evaluations for fair comparison.

**Clarity:** The paper is generally well-written with clear motivation. Figure 1 effectively illustrates the core problem and proposed solution. The mathematical formulation is precise, and the algorithm pseudocode (Algorithm 1) provides implementation details. The progression from problem identification to solution is logical and easy to follow.

**Significance:** The work addresses a fundamental limitation in current GRPO-based alignment methods for flow matching models. The ODE-based dense reward estimation is elegant and can integrate with any existing reward model, making it broadly applicable. The reward-aware exploration calibration provides insights that could benefit other RL-based generative model training approaches. The consistent improvements across diverse tasks suggest practical value for the community.

### Weaknesses
**1. Limited Analysis of Reward Hacking:** While Table 1 shows some reward hacking (e.g., Aesthetic Score degradation in compositional generation), the discussion is superficial. The paper doesn't analyze: (a) why certain metrics degrade while others improve, (b) whether dense rewards exacerbate or mitigate reward hacking compared to sparse rewards, (c) qualitative failure cases where the model exploits the reward function, or (d) potential solutions. Given that reward hacking is a critical concern in RL-based alignment, this deserves deeper investigation.

**2. Generalization Questions:** Several aspects of generalization remain unexplored: (a) Does the method work with different base models beyond SD3.5-M? (b) How does it scale to higher resolution generation or longer sampling trajectories? (c) Does it apply to other generative models (e.g., diffusion models, consistency models)? (d) How does it perform on out-of-distribution prompts or edge cases?

**3. Writing and Presentation Issues:** Minor clarity issues: (a) Notation inconsistency: sometimes $x_t^i$ and sometimes $x^i_t$, (b) Figure 2 could be clearer about which components run during training vs. inference, (c) The connection between ODE denoising steps n and timestep t is confusing ("n may be any integer in [1,t]" but then "we set n=t"), (d) Some experimental details are missing (e.g., learning rates, number of training iterations to convergence, batch sizes).

### Questions
1. How many iterations does Algorithm 1 require for $\psi(t)$ to converge? What are the sensitivity and recommended values for hyperparameters $\epsilon_1$ and $\epsilon_2$?
2. Is exploration space calibration performed once as preprocessing or continuously during training? If continuous, how frequently?
3. How does the accuracy of dense reward estimates degrade as you move further back in timesteps (larger t)? Have you analyzed the reward estimation error at different timesteps?

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
3

### Summary
This paper proposes DenseGRPO, a reinforcement learning framework for aligning flow-matching models (e.g., text-to-image generation) with human preferences.
While previous GRPO-based approaches such as Flow-GRPO and DanceGRPO rely on a sparse reward that assigns a single trajectory-level reward to all denoising steps, DenseGRPO introduces a step-wise dense reward estimation method. Specifically, it leverages ODE-based denoising to obtain intermediate clean images and applies a reward model to compute per-step reward gains. In addition, the paper identifies a mismatch between uniform noise injection and the time-varying denoising process in existing methods, and proposes a reward-aware exploration space calibration scheme that adaptively adjusts timestep-specific stochasticity.
Empirical results on multiple text-to-image benchmarks show consistent improvements over Flow-GRPO and Flow-GRPO+CoCA.

### Strengths
- The paper addresses the well-known issue of sparse reward assignment in GRPO-based flow matching models and provides a conceptually simple fix.

- DenseGRPO demonstrates noticeable improvements across several text-to-image benchmarks and alignment metrics (e.g., +1.0 PickScore gain).

- Figures and algorithm descriptions are intuitive, especially the visualization of dense reward distributions and the adaptive exploration scheme.

### Weaknesses
- Conceptual Incrementality without a Deeper Credit Assignment Analysis

While the shift from sparse to dense reward seems natural, the paper does not provide a principled analysis of why ODE-based reward estimation captures per-step contribution more faithfully. The method assumes that ODE rollouts preserve semantic consistency, yet no theoretical or empirical verification supports this. Without a formal treatment of credit assignment or causality, the approach reads more as a heuristic refinement than a conceptual advance.

- Limited Perspective Beyond Flow Models

The method is tightly coupled with ODE-based denoising, making it unclear whether the same principle generalizes to other generative families (e.g., DDPM, consistency models). Without this, the claimed “general framework for dense reward” seems overstated.

- Unclear Attribution of Performance Gains

The performance boost could stem from either (a) the dense reward, (b) better noise scheduling, or (c) hyperparameter tuning. The ablation is not disentangled enough to isolate these effects. For example, the “uniform a=0.7” baseline in Fig.6(b) still benefits from DenseGRPO, which suggests confounding factors that aren’t fully analyzed.

### Questions
- Since dense rewards are computed using ODE rollouts while policy trajectories come from SDE sampling, how do the authors justify that the resulting reward estimates remain unbiased with respect to the actual policy distribution?
- Theoretical framing: Could the authors connect their dense reward to existing temporal credit-assignment frameworks (e.g., potential-based shaping or advantage decomposition)?
- Would DenseGRPO still hold if the base model were trained with a consistency loss or direct velocity regression instead of flow matching? Is the dense reward specific to ODE semantics?
- Is the reward gain $\Delta R_t = R_{t-1}-R_t$ guaranteed to reflect incremental improvement rather than numerical noise from the reward model? Has the variance of this difference been analyzed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DenseGRPO, which is a framework that enhances flow-matching based GRPO by introducing dense, step-wise rewards to better align human preferences with intermediate denoising contributions. It further calibrates the exploration space through a reward-aware, timestep-adaptive scheme, achieving more efficient and consistent alignment across benchmarks.

### Strengths
* The paper highlights the importance of **dense rewards** for effective **credit assignment** in reinforcement learning tasks. This is indeed a crucial factor that can significantly improve policy optimization stability and efficiency.
* Figure 6(c) provides informative results. It clearly demonstrates that the most straightforward approach of predicting the final $x_0$ in a single step to compute delta rewards performs even worse than the original Flow-GRPO, underscoring the necessity of proper dense reward design.

### Weaknesses
*  The proposed Dense-GRPO algorithm introduces many additional ODE denoising steps, which inevitably increase computation time. However, the paper lacks a comparison with Flow-GRPO under the same training-time horizontal axis to show efficiency differences.

*  Since the modified algorithm may alter KL-consumption behavior, it would be helpful to visualize how the KL loss evolves during training for both Flow-GRPO and Dense-GRPO.

*  The proposed Exploration Space Calibration module seems designed specifically for dense rewards. It would be valuable to clarify whether this technique can be directly applied to Flow-GRPO.

### Questions
What is the difference between the sparse reward setting in Figure 6(a) and the original Flow-GRPO, and why does it perform significantly better than the original version?

### Soundness
3

### Presentation
3

### Contribution
3
