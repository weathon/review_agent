# Primary-Fine Decoupling for Action Generation in Robotic Imitation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
Multi-modal distribution in robotic manipulation action sequences poses critical challenges for imitation learning. 
To this end, existing approaches often model the action space as either a discrete set of tokens or a continuous, latent-variable distribution.
However, both approaches present trade-offs: some methods discretize actions into tokens and therefore lose fine-grained action variations, while others generate continuous actions in a single stage tend to produce unstable mode transitions.
To address these limitations, we propose Primary-Fine Decoupling for Action Generation (PF-DAG), a two-stage framework that decouples coarse action consistency from fine-grained variations. First, we compress action chunks into a small set of discrete modes, enabling a lightweight policy to select consistent coarse modes and avoid mode bouncing. Second, a mode conditioned MeanFlow policy is learned to generate high-fidelity continuous actions.
Theoretically, we prove PF-DAG’s two-stage design achieves a strictly lower MSE bound than single-stage generative policies. 
Empirically, PF-DAG outperforms state-of-the-art baselines across 56 tasks from Adroit, DexArt, and MetaWorld benchmarks. It further generalizes to real-world tactile dexterous manipulation tasks. Our work demonstrates that explicit mode-level decoupling enables both robust multi-modal modeling and reactive closed-loop control for robotic manipulation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces PF-DAG (Primary-Fine Decoupling for Action Generation), a two-stage imitation learning framework that separates coarse discrete mode selection from fine-grained continuous action generation in robotic manipulation. The method first compresses action chunks into discrete primary modes using a VQ-VAE and predicts these modes through a lightweight classifier π₁ (Sec. 3.3). Conditioned on the selected mode, a one-step MeanFlow policy π₂ produces continuous residual actions (Sec. 3.4). The authors show theoretically (Eq. 6) that this two-stage design achieves a lower optimal MSE bound than single-stage generative models by removing inter-mode variance. Experiments on 56 manipulation tasks across Adroit, DexArt, and MetaWorld (Table 1, Table 5) demonstrate significant improvements over diffusion and flow-based baselines such as DP3 and FlowPolicy. Real-world evaluations on xArm and XHand setups (Table 2) further support PF-DAG’s efficiency and robustness, and ablations (Table 3, Table 7) examine the contributions of each component, the number of discrete modes, and tokenization schemes.

### Strengths
1. The paper presents a clean, interpretable two-stage structure that maps well onto the inherent hierarchy of manipulation behaviors. The distinction between discrete primary modes (coarse subgoals) and continuous fine variations (residual control) is intuitive and empirically justified by improved temporal stability and reduced “mode bouncing” (Fig. 3, Fig. 5c).

2. PF-DAG is evaluated on 56 simulated and several real-world tasks, with consistent and sizable gains over strong diffusion and flow baselines (Table 1, Table 5). The inclusion of tactile sensing and dexterous-hand setups makes the evaluation unusually comprehensive for imitation learning studies.

3. The ablation studies (Table 3) systematically isolate the contributions of the Primary Mode Policy and MeanFlow decoder, demonstrating clear complementary effects. Additional analyses on the number of discrete modes (Table 7) and solver sensitivity (Table 4) reveal robustness to hyperparameters and architecture choices.

4. The MeanFlow one-step generation (Sec. 3.4) avoids the high computational overhead of multi-step diffusion or normalizing flow inference, achieving inference speeds comparable to 1-NFE diffusion with substantially higher success (Fig. 5b). This efficiency supports PF-DAG’s practicality for real-time control.

### Weaknesses
1. The MSE bound proof in Sec. 3.5 assumes oracle access to correct mode assignments, ignoring errors from the learned π₁. In practice, misclassification reintroduces inter-mode variance, invalidating the “strictly lower bound” claim except in the idealized setting of perfect mode prediction.

2. The VQ-VAE learns discrete modes via unsupervised L₂ reconstruction (Eq. 3), but this can merge semantically distinct actions or distort temporal consistency, particularly in high-DOF settings (DexArt). The resulting quantization bias contradicts the clean variance decomposition used in the theoretical section.

3. The MeanFlow decoder (Eq. 4, 16) predicts deterministic residuals rather than sampling from a conditional distribution, meaning PF-DAG cannot represent intra-mode uncertainty. This weakens the paper’s claim of “multi-modal modeling” within each mode.

4. Although the paper highlights interpretability benefits (Fig. 5a, Fig. 7), these are shown only through qualitative cluster plots. No quantitative metrics (e.g., mode purity, task-level consistency) are provided, making the claim of interpretable sub-behaviors unsupported.

### Questions
please address the concerns above

### Soundness
3

### Presentation
3

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
The paper proposes a hierarchical method for imitation learning by firstly decomposing action chunks from demonstrations into discrete high-level coarse skills which are used to condition a residual policy that is trained using MeanFlow, a one-step generative modelling technique, to then produce the final action to be executed on the robot. At the high-level, a Vector Quantized VAE is used to extract a codebook of distinct modes, wherein each discrete mode represents a coarse action. To make up for the errors between a coarse and continuous action arising from a discretization and subsequently to improve the continuous predictions, a second fine-grained policy is used that is conditioned on the predicted high-level skill which adds a residual action to the coarse action.

### Strengths
The proposed approach shows promising results on a variety of tasks both in simulation and in the real world, not just in terms of accuracy but also in capturing the multimodality effectively. The hierarchical disambiguation helps prevent "mode bouncing" which allows for a conditional generation of the fine-grained actions. The use of MeanFlow as compared to Flow Matching also shows a good improvement which adds some novelty. The paper is written well and clear to follow.

### Weaknesses
While the proposed approach shows promise, this approach of hierarchical policy structures for robot learning has been well studied in previous approaches, yet there is no mention or comparison against previous approaches. Only some discussion on Imitation Learning and Action space discretization is presented. There is no comparison against any hierarchical approach, but rather against standard end-to-end visuomotor policy learning approaches like using Diffusion or Flow Matching. Many previous approaches have shown how such a hierarchical structure can be use breakdown policy learning into a high-level skill decomposition module that is then used to condition a low-level policy, even with multisensory inputs or to train a residual policy as the authors propose. The proposed approach is not really a novel idea that is being proposed, but rather a variant of this hierarchical family of approaches that uses a residual policy using MeanFlow, instead of directly conditioning the low-level policy.

Some related works on hierarchical decomposition:
- Rana, Krishan, et al. "Residual skill policies: Learning an adaptable skill-based action space for reinforcement learning for robotics." Conference on Robot Learning. PMLR, 2023.
- Cui, Te, et al. "Hierarchical Autoregressive Modeling With Multi-Scale Refinement for Robot Policy Learning." IEEE Robotics and Automation Letters (2025).
- Chen, Lili, Shikhar Bahl, and Deepak Pathak. "Playfusion: Skill acquisition via diffusion from language-annotated play." Conference on Robot Learning. PMLR, 2023.
- Mete, Atharva, et al. "Quest: Self-supervised skill abstractions for learning continuous control." Advances in Neural Information Processing Systems 37 (2024): 4062-4089.
- Wan, Weikang, et al. "Lotus: Continual imitation learning for robot manipulation through unsupervised skill discovery." 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024.
- Kujanpää, Kalle, Joni Pajarinen, and Alexander Ilin. "Hierarchical imitation learning with vector quantized models." International Conference on Machine Learning. PMLR, 2023.
- Zhao, Tianxiang, et al. "Skill disentanglement for imitation learning from suboptimal demonstrations." Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023.
- Tanneberg, Daniel, et al. "Skid raw: Skill discovery from raw trajectories." IEEE robotics and automation letters 6.3 (2021): 4696-4703.
- Ju, Zhaoxun, et al. "Rethinking mutual information for language conditioned skill discovery on imitation learning." Proceedings of the International Conference on Automated Planning and Scheduling. Vol. 34. 2024.
- Liang, Zhixuan, et al. "Skilldiffuser: Interpretable hierarchical planning via skill abstractions in diffusion-based task execution." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
- Hao, Ce, et al. "Skill-critic: Refining learned skills for hierarchical reinforcement learning." IEEE Robotics and Automation Letters 9.4 (2024): 3625-3632.
- Feng, Ruoxuan, et al. "Play to the Score: Stage-Guided Dynamic Multi-Sensory Fusion for Robotic Manipulation." Conference on Robot Learning. PMLR, 2025.

### Questions
- What are the key contributions that make the paper stand out compared to other hierarchical policy learning approaches? How does this necessarily translate into improved performance gains?
- Are there any architectural differences between the proposed approach and the baselines? Are the architechtures from the baselines used as is without any modification? Is it the same observation feature extraction followed by the same ($\pi_2$) network used for the meanflow residual policy without conditioning? If not, then it may not be a fair comparison in my opinion. 
- How is the multisensory information effectively used? Or is it just encoded directly and used to train the policy end-to-end? For example using the high-level to potentially weight the information from the different sensors.
- The network architechtures for the VQ-VAE and MeanFlow residual network is missing in section A.4. The implementation details (Lines 233-237) do not make this clear either.

### Soundness
3

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
3

### Summary
This paper introduces a two-stage imitation learning framework PF-DAG for robot manipulation that explicitly separates coarse discrete decisions from fine-grained continuous action generation. The key idea is that many manipulation behaviors can be decomposed into a small number of interpretable primary modes plus within-mode variations. In the first stage, PF-DAG uses a VQ-VAE to encode action chunks into discrete primary modes and trains a lightweight policy to predict these modes from robot observations. In the second stage, a mode-conditioned MeanFlow decoder generates continuous actions conditioned on both the selected mode and the observation, improving action fidelity and stability. Empirically, PF-DAG outperforms behavioral cloning, diffusion, and flow-based baselines across 56 simulated manipulation tasks (Adroit, DexArt, MetaWorld) and transfers successfully to real-world tactile dexterous manipulation.

### Strengths
The paper tackles a key challenge in imitation learning, modeling multi-modal action distributions, with a clear two-stage formulation.

The Primary-Fine Decoupling design (discrete mode selection + continuous residual generation) is simple yet effective, and leads to more stable, temporally consistent control compared to other diffusion or flow-based baselines.

The paper is well-supported by both theoretical justification and empirical evidence, including a variance-based MSE bound analysis, as well as comprehensive ablation studies and quantitative experiments. The proposed method shows strong results across diverse benchmarks (Adroit, DexArt, MetaWorld) and have a real-world experiment.

### Weaknesses
While the paper attributes instability to "mode bouncing", it does not empirically compare against simpler baselines that include observation or action history. In practice, conditioning on temporal history or predicting waypoints (as done in prior works) can also effectively reduce mode switching. Also, some other related approaches such as "Behavior Generation with Latent Actions" (ICML 2024) and "Hierarchical Diffusion Policy: Manipulation Trajectory Generation via Contact Guidance" (T-RO) should be discussed or may be compared.

Although some real-robot experiments are reported, the hardware evaluation is relatively limited. It lacks more detailed comparisons, visualizations, and failure analyses, making it difficult to assess robustness and reproducibility.

In Table 3, the performance gap between K-means and VQ-VAE quantization is minimal, raising questions about whether the added complexity of VQ-VAE is necessary or provides substantial benefits beyond a simpler clustering baseline.

### Questions
Figure 6 provides an interesting visualization of the mode probability distribution and the corresponding policy behaviors in simulation. Would it be possible for the authors to include videos illustrating this? And if possible, also show the results for K-means side-by-side? Given that Table 3 reports very close performance between VQ-VAE and K-means, I am particularly curious whether the two produce noticeably different qualitative behaviors.

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
3

### Summary
The paper proposes PF-DAG, a two-stage framework that decouples discrete “primary” action modes from continuous, fine-grained residuals. Stage 1 quantizes action chunks with a small VQ-VAE codebook and trains a lightweight classifier pi_1 to select a primary mode from observations. Stage 2 conditions a one-step MeanFlow decoder pi_2 on the selected mode and observations to output high-fidelity continuous actions. A simple variance decomposition argument claims a strictly lower optimal MSE bound than single-stage generative policies. Experiments across 56 tasks (Adroit, DexArt, MetaWorld) and four real-world tasks report higher success and improved speed/stability versus diffusion and flow baselines. The work targets multi-modal action generation and mode-bouncing in closed-loop manipulation and argues significance via better robustness and real-time control.

### Strengths
- The writing is good and the paper is easy to follow.
- The two-stage design that separates coarse mode selection from fine residual generation is clear and practical. The one-step MeanFlow decoder offers compelling speed advantages while maintaining accuracy. 
- Empirical results are broad and consistently favorable across many simulated tasks, and the real-world demos indicate potential practical relevance. The ablations on K and tokenization provide preliminary guidance on design choices.

### Weaknesses
- The greedy mode selection without temporal smoothing can still make mode-bouncing happen and it highly relies on a carefully selected #mode hyperparam
-  Several design choice (VQ-VAE, MeanFlow) seems only provide marginal improvement based on the ablation studies.
- Clustering-based coarse-to-fine/mode-selection mechanism has been explored in other robotics domain such as motion forecasting in autonomous driving, the novelty of this paper is somewhat limited.

### Questions
- The authors mentioned the mode bouncing issue, which is also raised in RNR-DP which is cited in the related work in this paper. However it is not served as a baseline. I'm curious whether how it performs in the authors' setting since it tried to tackle similar problem.
- Would be great if the authors could report stability related metrics (mode-switch rate, jerk, pose variance etc.) to justify the claim.

### Soundness
3

### Presentation
3

### Contribution
2
