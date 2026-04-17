# Abstracting Robot Manipulation Skills via Mixture-of-Experts Diffusion Policies

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 8

## Abstract
Diffusion-based policies have recently shown strong results in robot manipulation, but their extension to multi-task scenarios is hindered by the high cost of scaling model size and demonstrations. We introduce Skill Mixture-of-Experts Policy (SMP), a diffusion-based mixture-of-experts policy that learns a compact orthogonal skill basis and uses sticky routing to compose actions from a small, task-relevant subset of experts at each step. A variational training objective supports this design, and adaptive expert activation at inference yields fast sampling without oversized backbones. We validate SMP in simulation and on a real dual-arm platform with multi-task learning and transfer learning tasks, where SMP achieves higher success rates and markedly lower inference cost than large diffusion baselines. These results indicate a practical path toward scalable, transferable multi-task manipulation: learn reusable skills once, activate only what is needed, and adapt quickly when tasks change.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SMP, a diffusion-based Mixture-of-Experts (MoE) policy designed to improve efficiency and generalization in multi-task manipulation. Unlike the existing MoE approaches, where the experts are entangled, the proposed method learns to route a subset of expert skills for each task, thereby reducing inference cost and enabling transferable skill learning across tasks. The authors provide theoretical analysis supporting the framework and conduct extensive experiments in both simulated and real-world environments to demonstrate its effectiveness. Overall, the paper contributes a novel integration of diffusion models and expert routing mechanisms for scalable and generalizable robotic control.

### Strengths
- The method is theoretically well-grounded, using a principled variational objective to learn a state-adaptive orthogonal skill basis.
- The paper's core innovation is combining this orthogonal basis with "sticky" routing dynamics. This is a novel approach that directly addresses prior limitations of skill entanglement and unstable, "chattering" gates.
- The approach shows significant empirical success, achieving higher multi-task and transfer learning performance than strong baselines.
- The method demonstrates markedly lower inference cost and latency by adaptively activating only a small subset of experts, highlighting its practical value for real-time control.

### Weaknesses
- The paper claims that a fixed basis "may fail to capture" action variability, which is the main justification for using a more complex state-adaptive basis. It would be better to include an ablation study comparing the proposed method to a version with a fixed basis. This would provide stronger evidence for this key design choice.
- The paper introduces some advanced mathematical concepts, like "thin-QR retraction with sign stabilization" and "sticky Dirichlet Markov dynamics". These might be a little unfamiliar to some readers. It would be helpful to add a brief intuitive explanation or a few more citations in the main text to help people understand the paper more smoothly.

### Questions
- For the inference time results in Table 2, your approach, SMP, uses the fewest active parameters during inference (80.2M), but its inference time (107.3 ms) is not the smallest; the ACT baseline is faster (94.8 ms). Do you have any insight on this? Is this slight overhead caused by other parts of your model, such as the router, the state-adaptive basis generation, or the final action composition step?
- The authors stated that the DP, DP3, and ACT baselines "underfit the multimodal distributions". In Table 2, the total number of parameters for these baselines (83.9M - 132.5M) is much smaller than the total parameters for your SMP method (258.9M). Is it possible that this "underfitting" is simply because the baseline models are under-parameterized in training space? What would you expect the success rate to be if these baselines were scaled up to have a similar total parameter count as SMP?

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
4

### Summary
This paper proposes Skill Mixture of Expert Policy (SMP), which learns a compact orthogonal skill basis, trains a mixture-of-experts diffusion policy on each basis, and employs a sticky routing mechanism. The main idea lies in decomposing the action space into orthogonal bases using differentiable QR factorizations to learn such bases, and establishing Dirichlet–Markov dynamics to control the gating. The results surpass the baselines and demonstrate a clear skill routing phase.

### Strengths
1. This paper proposes learning a lightweight basis in the action space and introduces a novel method for learning the gating mechanism. The experiments demonstrate the effectiveness of this approach in terms of success rate and inference computation (activated parameters). The paper also presents explainable and stable routing phase transitions, such as rotation and translation.
2. This paper exhibits several appealing features resulting from its decoupled structure (action basis, routing, and coefficients on the basis), including flexible inference sampling strategies and skill composition capabilities.

### Weaknesses
1. The paper claims that “a fixed basis may fail to capture such variability.” However, there is no comparison provided between a fixed basis and a learned basis. Moreover, from Figures 2 and 3, the action bases appear to be fixed and simple—such as left/right translation and rotation—suggesting that a fixed basis might be sufficient. How does the learned basis actually change with state?
2. In Equation (3), there are three hyperparameters. How do these parameters influence the final results?
3. Section 4.1 lacks sufficient explanation. For example, the equation is correct only under certain conditions, and the variable B does not appear in the right hand side. It would be better to clearly state the underlying assumptions and derivations.
4. Section 4.2 mentions top-k or coverage selection. What is the comparison between these two approaches? As the number of active experts increases, does the performance consistently improve?
5. The paper introduces several mathematical notations—such as simplex weights and Dirichlet–Markov dynamics—but does not explain them clearly. Including more mathematical details or visual illustrations would help improve clarity.
6. The paper would benefit from clearer writing and more detailed mathematical explanations.

### Questions
1. In Appendix B.3, the vision feature is concatenated with the robot state and projected into a shared feature o. Do the gating and basis modules also use this shared feature, or only the robot state? Additionally, could the authors provide details on the number of parameters and inference time for each component—namely, the vision encoder, gating module, basis, and diffusion experts?
2. Could the authors visualize the learned orthogonal basis throughout task execution? Figure 3 shows the basis directions, but they appear consistent within the same task execution, suggesting they may not vary with state.
3. It is quite common to apply PCA to extract principal components in dexterous manipulation and vehicle trajectory prediction, where PCA also learns an orthogonal basis. What advantages does learning a state-dependent basis provide compared to PCA or other fixed orthogonal decompositions?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a diffusion-based MoE framework designed for reusable motion primitives across multiple tasks. It features in orthonormal skill basis and sticky routing that reduce gate switches and activated experts. Abundant experiments have been conducted both in simulation and real world with ablation studies supporting the design choices. Overall I think this is a solid submission for ICLR.

### Strengths
1. The paper is well-organized and easy to follow.
2. The idea of orthogonal skill basis seems very attractive, based on the ablation studies it does provide good cross-task skill reuse with few switches.
3. The ablation studies are abundant and helpful in supporting the claimed contributions.
4. Real-world experiments looks good.

### Weaknesses
1. I am curious about the training cost of learning good orthonormal skill basis, more details on this part will be appreciated.
2. In Figure 3 task "Put bread into skillet" and "Lift tray with block in it", there are still considerable portion of the trajectories with multiple experts activated at the same time with similar gate values, does it mean in these cases the experts are still overlapping?

### Questions
See weakness

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a mixture of experts framework for diffusion-based policy learning in multi-task training scenario. The main difference from a "standard MoE model for score function" approach is that the authors introduce a state-dependent orthonormal action basis, where each vector in the basis represents a skill. The weights per skill are predicted by skill-specific diffusion experts. A gating function is used to activate a set of experts and ensure that the gating weights do not change significantly for consecutive inferences. Strong results are shown on multi-task bimanual manipulation scenarios.

### Strengths
The paper presents an intuitive method for learning skills implicitly for multi-task scenarios. The engineering in choosing the suitable method to construct the orthonormal basis, training targets for skill-specific diffusion experts, and sticky gating function is well executed.

### Weaknesses
Overall the major concern I have is that the paper do not provide any ablation results for the specific design choices like:

1. How does the results change without the sticky gating function? Since this is one of the major contributions, it will be worth looking at how this imapcts simpler MoEs like Sparse DP.
2. How does the results change with k or the mass threshold? Linear mass vs quadratice mass?
3. The practical implementation subsection in the appendix suggests that the proposed method requires significant tuning to avoid collapse.

### Questions
See weaknesses above.

### Soundness
4

### Presentation
4

### Contribution
4
