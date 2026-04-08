## Human Reviewer 1

### Summary
The paper proposes TD-JEPA, a latent-predictive representation learning method for zero-shot reinforcement learning (RL) that uses temporal difference (TD) learning to capture long-term multi-policy dynamics from offline, reward-free data. By training separate state/task encoders, policy-conditioned predictors, and parameterized policies in latent space, TD-JEPA enables zero-shot optimization of arbitrary reward functions—even for challenging pixel-based inputs. Theoretical guarantees and extensive experiments across 13 datasets (ExoRL/OGBench) validate its effectiveness.

### Strengths
1. Sound Motivation and Novel Method: The work addresses a key limitation of existing latent-predictive methods (single-task/one-step prediction, on-policy dependence) by leveraging TD learning for offline, multi-step, multi-policy dynamics modeling. Its design—separating state (low-level dynamics) and task (high-level context) encoders, plus TD-based losses and regularization—balances innovation and practicality, avoiding representation collapse while enabling offline training.
2. Comprehensive and Convincing Experiments: TD-JEPA is evaluated across 65 tasks (locomotion, navigation, manipulation) with proprioceptive/pixel inputs. It matches or outperforms SOTA baselines. Ablations confirm the value of multi-step prediction and separate encoders, while fast adaptation results show pre-trained representations boost sample efficiency for fine-tuning.
3. Rigorous Theoretical Foundations:
The paper provides solid theoretical support.

### Weaknesses
TD-JEPA relies on FlowQ-like behavioral cloning regularization to handle low-coverage datasets in OGBench, but its performance under more extreme data scarcity (e.g., critical action gaps, sparse trajectories) is not fully validated. It would help to add analysis on how TD-JEPA’s performance decays as data coverage decreases, and compare it to methods specifically designed for low-quality offline data. This would better demonstrate its practical applicability to real-world scenarios where data is often incomplete.

### Questions
In terms of methodology, could you further compare the approach proposed in this paper with that proposed by Motivo [1]?

[1] Andrea Tirinzoni, Ahmed Touati, Jesse Farebrother, Mateusz Guzek, Anssi Kanervisto, Yingchen Xu, Alessandro Lazaric, and Matteo Pirotta. Zero-shot whole-body humanoid control via behavioral foundation models. ICLR, 2025.

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
TD-JEPA targets learning latent policy dynamics models through successor features from a collection of reward-free off-policy data. Unlike prior work often limited to one-step predictions, this TD loss enables the model to learn representations predictive of long-term, policy-conditioned latent dynamics. The system trains four components directly in latent space: a state encoder ($\phi$), a task encoder ($\psi$), a policy-conditioned multi-step predictor ($T_\phi$), and a set of policies ($\pi_z$). The predictor learns to approximate successor features, which allows the agent to perform zero-shot optimization of any new reward function at test time.

### Strengths
TD-JEPA introduces a method for learning long-term transition dynamics rather than single-step dynamics. There is extensive experimentation in simulated benchmarks across DMC and OGBench showing general improvement over other zero-shot RL baselines.

### Weaknesses
Perhaps it is because I am not in the immediate area, but it was challenging for me to determine what the real-world impact of this method is. It certainly makes sense in relation to recent latent prediction models, but the introduction may benefit from some framing that takes a step back. Is there more application-facing work that has called for learning from large reward-free multi-task offline datasets that could be cited? And for zero-shot RL? If this doesn’t exist, why not yet?

### Questions
In the abstract, it is written “(TD) learning enables learning representations predictive of long-term latent dynamics across multiple policies…” . Would it be more accurate to say “long-term latent task/policy dynamics” instead?

For me, the notational assignments were difficult to maintain as I was reading through the relatively terse section 3. I wonder if there is a more distinguishing notation between \psi and \phi?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The authors propose TD-JEPA, a method for learning policy-dependent representations of long-term dynamics that can be used for zero-shot RL. They learn the representations by combining temporal difference methods with the self-predictive framework of JEPA. Across a number of experimental benchmarks, they show that TD-JEPA matches or outperforms existing baselines, especially on pixel-based environments. The authors also discuss a number of theoretical properties of the proposed algorithm.

### Strengths
The paper is well written and clearly motivates the proposed algorithm. The experiments are comprehensive and show convincing performance. The proofs justify the algorithm theoretically and build on techniques used in previous works.

### Weaknesses
I do not have any major weaknesses for this work, but I point out things I found interesting which could benefit from more detailed discussion (although I understand some of these points might be outside of the scope of this work).

1. What is the computational cost of the different proposed methods? I know many of the considered methods train multiple function approximators, but it would be nice to have at least a brief discussion of the training speed for each method.
2. I found the comparison between BYOL and TD-JEPA interesting. In particular, the contrast between modeling expert policies vs. policy conditional measures. I wonder if there are other simple analyses/visualizations that can pinpoint this down further, beyond performance on benchmarks. For example, is one representation more robust to noise / generalizes better (perhaps since TD-JEPA is better on pixel environments, this would be the case)?  
3. For pixel-based environments, how do the visual features compare from using a purely visual pre-training strategy (MAE for example) to learning the encoders with TD-JEPA (or another RL method)?

In general, there seem to be a number of different "representation learning" methods that are useful for RL in the community. The authors already briefly do this in the discussion section, but continued conversation about the long-term desiderata for a representation, beyond performance on zero-shot benchmarks would be interesting for the community.

### Questions
Please see above.

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper proposes a new state representation learning method called TD-JEPA. The main idea of TD-JEPA is to use TD learning to learn BYOL/JEPA-like self-predictive representations, where positives are sampled from geometrically distributed future states (in the MC case). Importantly, instead of learning representations w.r.t. the behavioral policy (like BYOL-$\gamma$), the authors simultaneously learn a latent task embedding $z$ and train a $z$-conditioned policy and the corresponding representations, somewhat similarly to FB representations. The authors show that TD-JEPA representations enable zero-shot RL, outperforming previous zero-shot RL approaches on ExORL and OGBench. They also demonstrate that these representations can be effectively fine-tuned for offline-to-online RL.

### Strengths
This is a well-written paper with solid theoretical and empirical results. The proposed method is (to my knowledge) novel, even though there are a number of closely related (but different) works, such as FB and BYOL-$\gamma$. The authors compare their method with previous methods across diverse categories, and convincingly demonstrate its effectiveness on a wide array of tasks and settings.

Another strength is that the authors provide a solid theoretical analysis of their method. They theoretically show that their method (more or less) low-rank approximates successor measures. With this connection, they are able to relate downstream performance to their loss. While I didn't exhaustively check the correctness of the proofs in Appendix, they appear to be solid and have some new, intriguing aspects of their own.

### Weaknesses
I don't see any major weaknesses in this work. While I believe the current results are already enough for an ICLR publication, I think they could be further strengthened by demonstrating the performance on even more complex, "new" environments that go beyond the standard benchmarks used in previous work (Motivo is a good example of this, and I think this could also be done in a separate follow-up work). Another nitpick is that the related work section is placed in Appendix. Especially given that this area is (relatively) dense, I believe discussions about related work are essential to understanding this method, and would like to encourage the authors to move this section to the main paper in the final version.

### Questions
- In Table 4, why do some of the settings have zero hidden layers?
- While I understood that the TD-JEPA representation approximates the successor measure of $\pi^z$ in a low-rank manner ($F_z^\top B$) for a **fixed** $z$, can the authors explain what kind of behavior **set** this method would learn? In other words, are there any explicit descriptions of learned skills? For example, for HILP, one can describe its skills as those that maximally span the isometric latent embedding space. Do the authors have a similar intuitive (yet "correct") description of skills learned by TD-JEPA? (I guess this question directly applies to FB representations as well.)

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4