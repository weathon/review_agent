# GoldenStart: Q-Guided Priors and Entropy Control for Distilling Flow Policies

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 2, 4, 8, 8

## Abstract
Flow-matching policies hold great promise for reinforcement learning (RL) by capturing complex, multi-modal action distributions. However, their practical application is often hindered by prohibitive inference latency and ineffective online exploration.  Although recent works have employed one-step distillation for fast inference, the structure of the initial noise distribution remains an overlooked factor that presents significant untapped potential. This overlooked factor, along with the challenge of controlling policy stochasticity, constitutes two critical areas for advancing distilled flow-matching policies. To overcome these limitations, we propose GoldenStart (GS-flow), a policy distillation method with Q-guided priors and explicit entropy control. Instead of initializing generation from uninformed noise, we introduce a Q-guided prior modeled by a conditional VAE. This state-conditioned prior repositions the starting points of the one-step generation process into high-Q regions, effectively providing a ``golden start'' that shortcuts the policy to promising actions. Furthermore, for effective online exploration, we enable our distilled actor to output a stochastic distribution instead of a deterministic point. This is governed by entropy regularization, allowing the policy to shift from pure exploitation to principled exploration. Our integrated framework demonstrates that by designing the generative startpoint and explicitly controlling policy entropy, it is possible to achieve efficient and exploratory policies, bridging the generative models and the practical actor-critic methods. We conduct extensive experiments on offline and online continuous control benchmarks, where our method significantly outperforms prior state-of-the-art approaches.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce a golden start method for policy distillation with Q-guided priors. In addition, they allow the distiller to output a stochastic distribution based on entropy regularization, that allows the policy to explore beyond what has been learnt from offline data. The authors claim that they provide a new state-of-the-art.

### Strengths
The paper is nicely written and easy to follow. There are many comparisons with baselines (but see below). Combining a golden start with entropy regularization seems to be an important move forward.

### Weaknesses
It looks like there is an important typo in Table 1. The results claimed are not as good as I initially thought: In the 4 first tasks, the new method outperforms the previous ones only in 1 case. But a closer look reveals that this is a typo, as ReBRAC has a performance of 91, above the new method. Therefore, the new method does not show any benefit in the locomotion tasks. 

Further, a closer look at the D4RL environments reveals that the improvement in performance is only marginal with respect to FQL, despite having a higher computational cost. 

Second, the mathematics presented in the appendix are kind of trivial. Showing that under an updated policy that considers the highest values of a learned Q value there is going to be an improvement in reward is a trivial fact. I am not sure what the scope of Theorem 1 is beyond showing an expected result, which is so by construction. 

Finally, the introduction of the entropy bonus term follows standard approaches in previous RL literature (Haarnoja, icml, 2018, quoted by the authors). From this addition it is totally expected that exploration would be improved. However, no comparison with continuous optimal controllers, such as SAC, are provided.

### Questions
What does the math in the Appendix reveal that is novel?

What would be the performance of the new distillation method against SAC, let us say, in locomotion and maze tasks?

Typos: Line 366 “achieves”, and Line 483 “establishes”.

### Soundness
2

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
3

### Summary
The main contribution of this paper is a new method to learn a “golden prior” noise distribution for flow matching, that produces higher-valued actions than the initial Gaussian distribution. This prior is learned by sampling many noise vectors, mapping them to actions through flow matching, and then training a conditional VAE to generate only the highest-valued of the noise vectors. This transformed distribution is then used to seed the flow-matching policy for action selection. The starting point for the broader algorithm is FQL, and this modification, along with adding stochasticity and entropy-regularization to the policy, improve the performance of the final algorithm.

### Strengths
The idea of Golden priors is an interesting one, and there are not many instantiations of it. As this paper points out, directly optimizing flow-matching policies using RL has many challenges that this method avoids. Advantage Noise Selection is a reasonable method for creating higher-valued noise targets. The experimental results are thorough, and the crescent experiment is a very good demonstration of the method doing what it claims. The method achieves quite good performance on the extensive benchmark suite.

### Weaknesses
I would describe the instantiation of Advantage Noise Selection as amortizing the rejection-sampling process. I think a stronger argument could have been made for, and more time spent on, why we expect an advantage of this method over rejection sampling (IFQL is the included reference for this) — why does this lead to better performance when it is doing something pretty similar? The 2x speedup I think is not strength enough alone given the added complexity.

The entropy regularization is empirically helpful, and good to have tested, but I would argue not very surprising.

I think the theory section has some weaknesses. First, it relies on the prior actually producing better initializations, and Lemma 2 is not quite rigorous. And second, is that implicit in Theorem 1 is that sampling a higher-Q-estimate action is always than sampling a lower-Q-estimate action. When this in fact depends on whether your Q function can be trusted, which in off-policy RL is one of the main difficulties, and why we encourage sticking to the prior. So I don’t think this proof is wrong, but also doesn’t provide much supporting evidence for the model’s effectiveness. As a minor thing, I think the theory should be mentioned in the main text, as I was surprised to see it when I scrolled down.

Another minor thing, L_distill is confusingly included twice meaning two different things, Eq 10 and Eq 2.

### Questions
Mainly, I would like to better understand the comparison of IFQL and other rejection-sampling methods with yours. First, what “N” was used for rejection sampling in IFQL? Second, should we expect Advantage Noise Estimation to be better than rejection sampling, or just faster?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces GoldenStart, an improvement of previous existing methods for offline and offline-to-online reinforcement learning tasks, that requires to model and learn complex distributions. 

Method:

This work builds on Flow Q-learning (FQL), a method based on the actor-critic structure. FQL is based on flow matching: the goal is to train a distilled (one-step) state-conditional flow student network to output actions that (i) maximize the value function (ii) while minimizing a distillation loss against a flow teacher. The flow teacher is trained to do behavioral cloning on an offline dataset D using multiple timesteps (unlike the student), akin to the standard flow matching method. (i) encourages exploitation and (ii) encourages staying close to the collected high reward dataset D.
On top of this, GoldenStart introduces 2 main contributions: 
-  First, instead of using gaussian noise as prior,  they learn a Q value guided prior model as a state-conditional VAE. 
     Such a conditional VAE is trained in 2 steps: 
          - First, they generate "advantage noises". Given a state and N_cand initial gaussian noises, they generate actions using the student flow network, and they keep the action with the highest value function,
           - Second, they train a state-conditional VAE that learns to generate the advantage the distribution of advatange noises from a gaussian noise.
- Second, they train the student flow to predict a mean and a variance. Why? Because in addition to the value maximization term and the distillation term of FQL, that were already present in FQL, they add an entropy term. This encourages exploration in addition to exploitation.

Experiments: 

They introduce a new toy task called crescent task,  and they do ablation studies to illustrate the added value of each of their contributions in terms of exptected return. 
They also use the same experimental setups as in FQL (i.e. OGBenchmark, AntMaze, Visual envs ), and the average expected return (over different subtasks) is higher with GoldenStart than with FQL and other baseline methods .

### Strengths
- The clarity of the paper, the explanations, the contributions and evidence supporting them
- The introduction of the crescent toy task with the 2D visualisations, as well as the ablation studies in the experiments for crescent for each of the contributions (prior / entropy), which helps in understanding,
- The simplicity of the introduced changes
- The exhaustive set of experiments (as in FQL)

### Weaknesses
- The additional computational cost for learning the VAE prior distribution

### Questions
- The teacher flow network trained on the offline dataset using gaussian noise as a starting point, but in the text you do inference from it using the distribution of x_advantage as a starting distribution. Could you explain how the teacher adapts to the distributional shift of this initial distribution? 
- Correct me if I am wrong, but it seems to me that the conditional VAE is learns a "higher return actions distribution" than the student flow network (because of the argmax in the Q value over N_candidates, and the absence of the entropy regularization term) ? If yes, could you show the expected return over some tasks in the OGBench using the advantage noise selection module only? 
- Why do you need to predefine an entropy target H_{target}? How do you choose it in your exps? Does it depend on the task?
- Do you have some results that could show that GoldenStart is doing principled exploration that FQL is not doing? E.g. discovering some actions that FQL does not discover, and that are not in the offline dataset D ? (It is difficult to infer from the expected return only that GoldenStart allows for better exploration compared to FQL).

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes GoldenStart (GS-flow), a novel framework for distilling flow-matching policies in reinforcement learning. The authors identify two key limitations of existing one-step distilled flow policies: 

(1) the uninformed Gaussian prior that ignores the value landscape, and (2) the deterministic mapping that prevents controllable exploration.

To address these issues, GS-flow introduces two main innovations:

(1) Q-Guided Prior Learning, where a conditional VAE learns a state-conditioned prior that biases the initial noise toward high-Q regions, providing a “golden start” for generation; and
(2) Entropy-Regularized Distillation, which replaces deterministic action generation with a stochastic actor trained under an entropy-regularized loss, enabling adaptive exploration during online fine-tuning.
Extensive experiments on OGBench, D4RL AntMaze, and Visual RL benchmarks demonstrate that GS-flow consistently outperforms prior state-of-the-art methods (e.g., FQL, IFQL), particularly in multi-modal and exploration-heavy tasks such as Cube Double and Puzzle-4×4. The method preserves single-step inference efficiency while achieving significant gains in both offline and online performance.

### Strengths
The paper precisely identifies two previously overlooked weaknesses in existing one-step distilled flow policies — the use of an uninformed Gaussian prior and the absence of controllable stochasticity — and builds upon them to formulate a coherent research question and motivation.

The work introduces a conditional VAE that learns a state-conditioned high-Q prior, transforming the random starting point in generative inference into a structured, value-aligned initialization, thereby providing an effective “shortcut” for policy optimization.

The authors reformulate the student policy from a deterministic mapping into a distributional generator with entropy control, achieving a principled balance between exploration and exploitation and significantly enhancing online adaptability.

Across benchmarks such as OGBench, D4RL, and Visual RL, GS-flow consistently achieves state-of-the-art performance, demonstrating superior sample efficiency and exploration capability while maintaining single-step inference speed. Moreover, the ablation studies clearly disentangle and validate the independent contributions of the prior learning and entropy regularization modules.

### Weaknesses
1. The Advantage Noise Selection module relies on a hard argmax over Q-values to identify the optimal noise per state. While simple, this approach may amplify critic bias and reduce diversity in the learned prior. More robust alternatives such as soft advantage weighting or top-k filtering could potentially mitigate this brittleness.

2. While GS-flow demonstrates impressive efficiency within the family of flow-matching methods (notably compared to FQL and IFQL), it remains unclear whether this efficiency translates to practical advantages over lightweight Gaussian actor-critic approaches such as SAC or IQL. Given the additional complexity of the teacher–student architecture and prior learning modules, the overall time–performance trade-off might still be less favorable in real-world applications.

### Questions
1. The Multi-Crescent environment is an interesting design to expose Q-value overestimation, but given its continuous nonconvex reward surface, there might be potential for spurious high-Q regions or reward hacking when the critic is imperfect. Could the authors examine whether GS-flow’s Q-guided prior might amplify such artifacts? For instance, visualizing true reward vs. predicted Q over the state-action space or adding a calibration/consistency check (e.g., reward clipping or Q–r alignment) would strengthen the claim that the method is robust to reward misestimation.

2. The actor is trained with a composite objective combining distillation, Q-value maximization, and entropy regularization. Given the potentially conflicting gradients among these components, could the authors clarify whether they observed optimization instability or gradient interference during training?

3. Have the authors explored combining GS-flow with language- or vision-conditioned policies (e.g., VLA settings) to test generality beyond low-dimensional control?

4. Could the authors provide empirical insights into the interaction between entropy temperature α₂ and Q-guided prior learning — are they synergistic or competing during online fine-tuning?

### Soundness
3

### Presentation
3

### Contribution
3
