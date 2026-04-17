# Efficient Morphology-Control Co-Design via Stackelberg Proximal Policy Optimization

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6, 6

## Abstract
Morphology-control co-design concerns the coupled optimization of an agent’s body structure and control policy. This problem exhibits a bi-level structure, where the control dynamically adapts to the morphology to maximize performance. Existing methods typically neglect the control’s adaptation dynamics by adopting a single-level formulation that treats the control policy as fixed when optimizing morphology. This can lead to inefficient optimization, as morphology updates may be misaligned with control adaptation. In this paper, we revisit the co-design problem from a game-theoretic perspective,  modeling the intrinsic coupling between morphology and control as a novel variant of a Stackelberg game. We propose *Stackelberg Proximal Policy Optimization (Stackelberg PPO)*, which explicitly incorporates the control’s adaptation dynamics into morphology optimization. By modeling this intrinsic coupling, our method aligns morphology updates with control adaptation, thereby stabilizing training and improving learning efficiency. Experiments across diverse co-design tasks demonstrate that Stackelberg PPO outperforms standard PPO in both stability and final performance, opening the way for dramatically more efficient robotics designs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel reinforcement learning-based brain-body co-design framework, named Stackelberg Proximal Policy Optimization (Stackelberg PPO). The core innovation lies in formally framing the co-design process as a Phase-Separated Stackelberg Markov Game, where a "leader" agent sequentially constructs the robot morphology, and a "follower" agent learns the control policy for the resulting body. The authors derive a novel Stackelberg policy gradient using a likelihood-ratio trick, which allows the leader to anticipate the follower's adaptation without requiring backpropagation through the non-differentiable simulator. Integrated with PPO's clipping mechanism for stability, the method demonstrates significant improvements in final performance but is kind of computationally expensive.

### Strengths
1. The paper contributes a rigorous reformulation, which formulates the brain-body co-design problem as a Stackelberg game. 

2. The experiments are comprehensive and convincing.

3. The paper includes thorough ablation studies that validate key design choices, such as the regularization parameter.

### Weaknesses
1. The proposed method introduces very significant algorithmic complexity. The requirement to compute or approximate inverse Hessians via conjugate gradient, even with efficient tricks, adds computational overhead and implementation hurdles compared to vanilla PPO. Furthermore, the Stackelberg gradient relies on estimating second-order terms. As the complexity of the morphology and control spaces increases, how does the computational cost of this estimator scale?

2. The paper claims that "Transform2act enables joint training of structure and control without retraining per design. However, the interface
between structure and control remains non-differentiable: the two modules are optimized independently under a shared reward, with no explicit gradient flow between them". I don't quite understand what this sentence means. I think it is differentiable via policy gradient modeling.

3. The authors are encouraged to clarify and analyze differences between their work and other competion-based co-design methods, such as CompetEvo and "The Body is Not a Given".

4. The proposed approach seems to be too complex, relies on a lot of tricks, and can cause a lot of trouble for engineering implementation. It's more like a new PPO algorithm than a specific algorithm for co-design problem. Is it possible to use this algorithm in the neural architecture search problem (which also has a bi-level structure)?

### Questions
Please refer to the "Weakness" Section.

### Soundness
3

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
This paper makes a step on further improving embodiment co-design based on a recent interesting work BodyGen[1]. How to balance the bi-level optimization of co-design, is indeed a critical point for this area. This paper introduces Stackelberg Game, making this paper a promising solution for this target. The authors provide sufficient experiment results, and achieve some advantages over previous works.

[1] Lu, Haofei, et al. "Bodygen: Advancing towards efficient embodiment co-design." arXiv preprint arXiv:2503.00533 (2025).

### Strengths
The experiments cover multiple challenging co-design benchmarks, including both 2D and 3D locomotion as well as complex terrain settings. The comparisons against strong and diverse baselines demonstrate the robustness and generality of the approach. The method is supported by principled mathematical analysis with explicit gradient derivations under the Stackelberg formulation. The theoretical insights connect naturally with the empirical benefits in sample efficiency and stability.

### Weaknesses
I hope the author could provide anonymous code, which is strongly encouraged by ICRL. For the remaining weakness, please refer to questions.

### Questions
Personally, I have several questions about this paper:
1) As far as I know, the bi-level problem introduced in the previous work, e.g. BodyGen and Transform2Act, their formulation is $max_{morph} (max_{control, morph})$, which already based on that the inner control policy is the best control policy. They just flatten the formulation with a universal controller, in order not to re-train the entire controller from the scratch. However, in the formulation of this paper, aka Stackelberg Game, I don't find any core differences with the old formulation.
2) Following 1), for practical training using modern neural network, I believe that, if you use an end2end training using PPO like BodyGen, the inner universal control policy is always the "best one".
3) You emphasise the adaptiveness of the inner controller according to the outer morph-designer, can you provide any experiments, demonstrating the evolving (especially topology differences) during training, e.g. 1/5, 2/5 of total iterations, what the morphology is like.
4) I am curious about why you do not use all the tasks provided by your baseline, BodyGen & Transform2Act, since you build your entire pipeline on the top of BodyGen. Instead, you add some your personal task for evaluation.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper reframes morphology–control co-design as a phase-separated Stackelberg game and derives a likelihood-ratio–based Stackelberg PPO that captures follower adaptation without back-propagating through a non-differentiable interface. The method claims better sample efficiency and final reward across six MuJoCo tasks versus PPO-based and evolutionary baselines, with theoretical support (local equivalence surrogates) and stabilizing design choices (PPO clipping, Fisher metric).

### Strengths
1. **Clear problem framing for non-differentiable interfaces.** The phase-separated SMG formalization precisely matches co-design realities (discrete edits then control), addressing why prior Stackelberg methods fail here. Treating co-design as leader–follower nicely explains why joint training can wobble or fail.

2. **Technically neat gradient path.** The trajectory likelihood-ratio surrogate for the cross-derivative (Theorem 1) avoids differentiating through morphology transitions and yields local equivalence to true Stackelberg gradients. This is practical for implementation, easy for adoption with downstream ideas.

3. **Stability machinery.** PPO-style clipping and Fisher (natural-gradient) Hessian approximation are well-motivated for practical computation. The $\lambda$ sweep and Fisher vs. analytic Hessian give me a bit of insight into stability.

### Weaknesses
1. There are concerns about the experiment outlines:
- Prior Stackelberg RL (e.g., Stackelberg actor-critic / policy-gradient) is acknowledged, but there’s no controlled ablation that swaps in those estimators under the same phase-separated setting to isolate what PPO-clipping contributes vs. the SID term itself.
- Only four seeds are used “due to cost”; yet the authors claim “+22.1% average, +31.9% on 3D”, which needs more statistical significance. These environments are not that complicated to run at least 10 seeds.
- The paper argues for efficiency, but training still takes ~30 hours per model on an A100 + 10 CPU cores. There should be implementation optimization, or the absence of wall-clock comparisons to ES baselines (which can parallelize) leaves practical efficiency unclear.
- Most environments reward forward velocity (with optional small effort penalty), so improvements might partially reflect optimization for locomotion speed rather than broader co-design quality (e.g., robustness, energy, material constraints).

2. On the theory side, surrogates are locally equivalent; the practical bias/variance of the coupled gradient estimator (esp. through long leader horizons) is not deeply characterized.

### Questions
1. Please address the concerns in the Weakness section.

2. Estimator properties. What are the empirical bias/variance characteristics of the follower-implicit gradient term (eqs. 6–9) as the leader horizon grows? Any variance-reduction beyond standard advantage baselines?

3. Local equivalence radius. How sensitive are results to step size, i.e., when does local surrogate equivalence break, and how does PPO clipping mitigate that in practice? Provide KL traces or constraint violations.

4. Reward shaping sensitivity. How do results change if you (i) increase control-effort penalties, (ii) impose morphology complexity/material costs as leader rewards, or (iii) evaluate robustness (disturbances/terrain noise)?

5. Generality. Does Stackelberg PPO extend beyond locomotion (e.g., manipulation, multi-objective design)? Any results on unseen tasks or morphology priors to show transfer?

### Soundness
2

### Presentation
3

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
This work proposes efficient morphology control co-design via Stackelberg PPO. It formulates morphology–control co-design as a phase-separated Stackelberg Markov Game, where the leader is the morphology designer and the follower the policy controller. The Stackelberg PPO is a policy-gradient method that uses an implicit-differentiation surrogate (likelihood–ratio–based) across a non-differentiable leader–follower interface, plus Fisher-information preconditioning and PPO-style gradient clipping. On MuJoCo co-design benchmarks (Crawler, Cheetah, Glider, TerrainCrosser) and two new stair tasks, the method reports better stability and final reward than PPO-based co-design and evolutionary baselines (e.g., +22% on average, ~+32% on harder 3D tasks).

### Strengths
- This paper proposes a novel formulation of the morphology co-design problem, where the gradient from the morphology design phase is not differentiable. The Stackelberg game formulation is interesting and well-motivated. This avoids ad-hoc joint updates and gives a clean bilevel control–morphology structure.
    - The derivation for a Phase-Separated Stackelberg Markov Game’s **likelihood-ratio surrogate** with a **Fisher preconditioner** creates a tractable gradient signal for the leader without assuming differentiability of morphology parameters or unrolling large follower optimization loops. This is a **meaningful algorithmic advance** over evolutionary search or unrolled gradient-based bilevel RL, which are expensive and brittle.
- The multi-step morphology evolution, compared to prior art (e.g. Transform2Act) is a good insight and brings in notable improvement.
- Training efficiency improvement is notably higher than prior art as the morphology editors are now more informed with controller training results.

### Weaknesses
- Most evaluations reduce to forward-velocity rewards on stylized locomotion creatures; even the new stair tasks keep the same simple progress reward. This makes it hard to assess whether Stackelberg coupling helps with *real* co-design constraints (payload, torque limits, manufacturability, sensor placement, power, robustness).
- The λ/Fisher ablations are informative, but I’d like visibility into **data efficiency**: how many follower steps are actually saved vs PPO co-design? Since the thesis is “less re-optimization,” show sample-complexity curves for controller updates per viable morphology. (There is a follower-sampling hyperparameter, but not a direct “control-retraining cost” metric.)

### Questions
- How sensitive are results to morphology editing horizon T?
- Does the follower avoid full re-training when morphologies change significantly, or does the advantage mostly come from better leader gradients? Any evidence of faster controller adaptation?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper formulates morphology–control co-design as a phase-separated Stackelberg Markov Game (SMG) where a leader edits morphology via discrete actions and a follower optimizes control on the resulting body. Because the leader–follower interface is non-differentiable, the authors derive a trajectory likelihood-ratio loss for the Stackelberg cross-derivative and use the natural gradient approximation as a stable inverse-Hessian proxy. Combining everything gives the Stackelberg PPO. The experiments on various mujoco-based morphology control tasks demonstrate the competitiveness of the proposed method.

### Strengths
1. The formulation of morphology–control co-design as a phase-separated Stackelberg game is intuitive and matches the causal structure between design and control.

2. The paper gives a clear algorithmic pipeline that can be implemented with existing PPO infrastructure.

3. Experimental results are consistent across several benchmarks and ablations are reasonably detailed.

### Weaknesses
1. The technical novelty is modest, since the likelihood-ratio surrogate, natural gradient, and PPO clipping are all existing techniques; the contribution lies mostly in integrating them coherently.

2. the cross-derivative estimator in eq. (6) seems to have very high variance. 

3. The baselines do not include prior Stackelberg RL algorithms (e.g., Stackelberg Actor-Critic), so it is unclear whether the improvement arises from the Stackelberg formulation itself.

### Questions
The natural policy gradient is more computationally expensive than regular PPO. How does the computation efficiency compare with the other baselines, given the experiments are all in simulation?

### Soundness
3

### Presentation
3

### Contribution
2
