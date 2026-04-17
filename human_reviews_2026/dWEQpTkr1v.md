# RL4SBDD: Reinforcement Learning for Preference Alignment in Structure-based Drug Design

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Generative models have been proven effective in designing novel ligands conditioned on protein target structures, which is a fundamental task in drug discovery. Recent research has explored aligning generative models with practical requirements for certain molecular properties through gradient guidance or direct preference optimization. In this paper, we introduce RL4SBDD, a reinforcement learning framework that unifies existing alignment methods and analyzes their limitations, including biased value estimation, inefficient data utilization, and off-policy distribution shift. Building on our framework, we propose RL4SBDD-M, a novel approach to address these problems and facilitate preference alignment for arbitrary verifiable rewards in structure-based drug design. We introduce a policy model conditioned on rewards to avoid value estimations and fully leverage training data, and perform iterative reinforcement learning to mitigate off-policy distribution shift. Moreover, we introduce a sampling process with mixed guidance and noise reduction to improve generation quality and efficiency. By jointly optimizing binding affinity and synthesizability, RL4SBDD-M achieves a state-of-the-art success rate of 58.1% on the CrossDocked2020 benchmark, which is further boosted to 93.0% within 50 generations with oracle evaluations. Experiment analysis on a broader combination of rewards further confirms its efficacy in balancing multiple objectives, underscoring its potential for practical drug design.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
While my primary expertise is not in the specific generative methodologies employed (diffusion models, Bayesian flow networks), this review is based on extensive experience in the domain of AI-driven drug discovery and molecular design, coupled with a deep understanding of reinforcement learning for molecular optimization.

This paper introduces RL4SBDD, a framework utilizing advanced reinforcement learning (RL) to shift the sample distribution, p(x∣target), of distribution learning-based structure-based drug design (SBDD) generative models. The central aim is to guide the generation process toward molecules that optimize a given reward function, such as binding affinity, thereby exploring chemical space beyond the distribution of the original training data.

A noteworthy aspect of this work is its claim to improve docking scores while maintaining molecular sizes comparable to those of reference compounds. This approach commendably attempts to avoid the common pitfall of artificially inflating atom counts to exploit scoring functions like AutoDock Vina, representing a more principled approach to validation.

Nevertheless, the all generation examples raise significant doubts about the practical utility of the proposed method and reliability of evaluation. **The high prevalence of aromatic groups in the generated molecules** indicates a form of reward hacking, **reminiscent to how early RL methods achieved SOTA on optimization benchmarks by generating simple carbon chains**. This concern is substantiated by the fact that the reported chemical diversity of the generated molecules is remarkably lower than that of the baseline methods, casting doubt on their effectiveness and real-world applicability.

Methodologically, the paper combines existing RL techniques to a new domain rather than proposing a fundamentally new algorithm. In the context of AI4Science, such an application can be a valuable contribution. However, in these cases, I believe the evaluation criteria should shift from pure algorithmic novelty to how effectively the method addresses critical requirements defined by domain experts. From this standpoint, the validation presented in the manuscript does not sufficiently demonstrate that proposed method produces molecules with the desirable properties for a practical drug discovery campaign.

---
**Usage of LLM**: I wrote the entire review myself and only used the LLM to correct the grammar and improve readability.

### Strengths
1. RL4SBDD achieves the noticeable performance improvement in benchmark table by combining advanced RL methods.
2. The authors conduct a detailed analysis of how various parameters (guidance and noise level, etc) affect performance.
3. The motivation and key differences from existing methodologies are well visualized.

### Weaknesses
**1. Insufficient Baseline Comparisons**

While RL4SBDD conducts online training, its performance was not compared to pure RL-based SBDD methods with online training, such as TacoGFN [1] and 3DSynthFlow [2]. In particular, 3DSynthFlow has demonstrated state-of-the-art (SOTA) Vina docking scores and QED ensuring high synthesizability, while maintaining diversity comparable to distribution-learning based models and generating molecules 5x faster than MolCRAFT.

**2. Low Chemical Diversity**

The reported diversity of generated molecules is markedly low. Given that non-flexible docking is known to have a high false-positive rate [3,4], generating a chemically diverse set of candidates is essential for increasing the probability of finding viable hit compounds [5,6].
Though authors show that it is possible to control diversity and potency of generated molecules, they do not report the performance of high-diversity setting in Benchmark Table 1.

**3. Concern about reward hacking**

Figures 5 and 15 reveals that the protein-ligand interaction patterns of generated molecules from RL4SBDD-M differ substantially from the reference ligands, indicating the high probability to be false-positive.
**Specifically, the generated molecules exhibit a high prevalence of aromatic rings, even for hydrophilic pockets (e.g., 4RV4).**
This is in stark contrast to MolCRAFT, which report a balanced number of H-Bond donors/acceptors and hydrophobic groups, even though RL4SBDD-M shares the same architecture to MolCRAFT.

This observation leads to three critical concerns:
- 3.1: Reward hacking by increasing aromatic rings.
- 3.2: If the model over-samples aromatic rings, which are structurally simple, the RMSD < 2Å metric becomes less meaningful. It is much easier to predict the structure of planar benzene ring than other druggable functional groups.
- 3.3: Highly hydrophobic molecules with numerous aromatic rings often have poor ADME properties, limiting their real-world application.

To substantiate the model's performance and allay concerns of reward hacking, please consider the following additional analyses:
- A comparative analysis of physicochemical property distributions (e.g., logP, TPSA, number of HBD/HBA/aromatic rings) between the generated molecules, reference molecules, and a drug-like molecular database like ChEMBL.
- An analysis of protein-ligand interaction profiles using a tool like PoseCheck [7], with a comparison to other baselines.
- Calculation of the Delta Score [8] to verify that the model is optimizing for genuine protein-ligand interactions rather than hacking Vina scoring function.

**4. Ambiguity in Multi-Objective Optimization**

Figure 2 presents three molecular properties (Vina score, QED, SA score), yet the reward function defined in Equation 34 incorporates only the Vina and SA scores. This discrepancy is potentially misleading.

In drug discovery, the ability of an RL method to balance multiple objectives is a critical determinant of its utility. The authors are encouraged to report performance using a reward function that simultaneously optimizes all three properties, following previous optimization methods like TacoGFN [1] and MOOD [9].

---
**References**
1. Shen, Tony, et al. "TacoGFN: target-conditioned GFlowNet for structure-based drug design." TMLR (2024).
2. Shen, Tony, et al. "Compositional Flows for 3D Molecule and Synthesis Pathway Co-design." ICML (2025).
3. Passaro, Saro, et al. "Boltz-2: Towards accurate and efficient binding affinity prediction." BioRxiv (2025): 2025-06.
4. Furui, Kairi, and Masahito Ohue. "Boltzina: Efficient and Accurate Virtual Screening via Docking-Guided Binding Prediction with Boltz-2." arXiv preprint arXiv:2508.17555 (2025).
5. Lyu, Jiankun, et al. "Ultra-large library docking for discovering new chemotypes." Nature 566.7743 (2019): 224-229.
6. Bengio, Emmanuel, et al. "Flow network based generative models for non-iterative diverse candidate generation." Advances in neural information processing systems 34 (2021): 27381-27394.
7. Harris, Charles, et al. "Benchmarking generated poses: How rational is structure-based drug design with generative models?." arXiv preprint arXiv:2308.07413 (2023).
8. Gao, Bowen, et al. "Rethinking specificity in sbdd: Leveraging delta score and energy-guided diffusion." arXiv preprint arXiv:2403.12987 (2024).
9. Lee, Seul, Jaehyeong Jo, and Sung Ju Hwang. "Exploring chemical space with score-based out-of-distribution generation." ICML (2023)

### Questions
1. In the abstract, the authors use the term "desirable molecules." Could the authors please clarify the specific criteria used to define "desirability" in this context? In drug discovery, a molecule is typically considered desirable only if it satisfies a complex profile of properties, including potency, selectivity, and favorable physicochemical characteristics (ADME). The manuscript currently seems to equate desirability primarily with the **success rate**, and a clarification would be beneficial for the reader.

2. The paper describes performing two iterative RL steps. To better illustrate the learning process and the efficacy of this iterative approach, would it be possible to include a plot showing the evolution of key performance metrics (e.g., average reward, Vina score) across the RL iterations? This would provide valuable insight into the online training.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes RL4SBDD, a unifying reinforcement learning view of preference alignment for structure-based drug design, and introduces RL4SBDD-M, a practical method that conditions the policy on continuous reward targets, updates the dataset with iterative RL, and uses a mixed guidance schedule with reduced noise during sampling. The authors argue that popular approaches such as gradient guidance and DPO are optimizing a KL-constrained reward objective in soft RL and diagnose three weaknesses in current practice: biased value estimates early in the trajectory, inefficient use of pairwise data in DPO, and off-policy distribution shift between training and inference. On CrossDocked2020, RL4SBDD-M reports strong improvements, including a median Vina score of −8.70 kcal/mol and a 58.1% success rate with a 4.6× sampling speedup over a recent state of the art, and further reaches 93% success by selecting the best of 50 generations under oracle screening.

### Strengths
The work’s strengths are the conceptual clarity of the RL framing, which helps reconcile guidance-based and preference-based methods and yields concrete propositions that tie guidance to optimal policies under soft RL assumptions. The reward-conditioned policy is a simple, scalable idea that sidesteps noisy value learning early in sampling and exploits supervision from all training examples rather than only best–worst pairs, while the iterative RL loop squarely targets distribution shift. The mixed guidance schedule is well motivated by analyses showing when gradient guidance is unreliable or redundant, and the reduced-noise corrector connects neatly to a predictor–corrector interpretation of the sampler. Empirically, the paper compares against diverse baselines with a consistent evaluation suite and reports both quality and efficiency gains, complemented by ablations that isolate the contributions of each ingredient and a test-time scaling study that is relevant for practical lead finding. The paper is clearly written, the figures are helpful, and the inclusion of implementation details and a reproducibility statement is appreciated.

### Weaknesses
The main weaknesses concern scope, assumptions, and external validity. Several claims of unification and improvement rest on idealized conditions such as the Dirac-prior assumption in the linkage between guided transitions and the optimal policy; more discussion of how deviations from these assumptions affect guarantees would strengthen the theoretical framing. The empirical validation is heavily anchored to docking and easy-to-compute cheminformatic metrics, which are known to be imperfect surrogates for true binding and developability; without broader property coverage or prospective wet-lab feedback there remains risk of score hacking and limited translational value. Diversity decreases across RL iterations, suggesting potential mode concentration that may impair downstream medicinal chemistry; additional safeguards or multi-objective controls could help. The critique of DPO’s pair construction would be more convincing with a direct comparison to BT-consistent training that uses all pairs under identical compute. The test-time scaling results rely on oracle screening and pocket-specific atom-count priors, so the practicality and fairness of the reported compute budgets relative to search-based baselines deserve closer scrutiny. Finally, sensitivity analyses for the guidance switch point, guidance weight, and noise scale are informative but point to nontrivial tuning; reporting statistical significance across seeds and clarifying hardware parity for timing would improve confidence.    

---

For rebuttal and discussion:

1. Theoretical claims hinge on idealized assumptions such as a Dirac-prior link between guidance and the optimal policy.
2. Heavy reliance on docking and simple cheminformatic metrics risks score hacking and limits biological validity.
3. Diversity declines across RL iterations, suggesting potential mode concentration that could hinder downstream use.
4. Critique of DPO efficiency would be stronger with compute-matched, all-pairs BT-consistent comparisons.
5. Test-time scaling depends on oracle screening and pocket-specific priors, raising fairness and practicality questions.
6. Performance appears sensitive to guidance weights, switch points, and noise scales, with limited variance reporting across seeds and hardware.

### Questions
-

### Soundness
3

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
3

### Summary
This paper introduces RL4SBDD, a reinforcement learning framework for structure-based drug design that unifies gradient guidance and direct preference optimization (DPO) under a soft policy optimization perspective. The authors identify challenges in existing SBDD methods, including biased value estimation, inconsistent reward interpretation, and off-policy training instability. They propose a reward-conditioned policy that reconstructs molecules based on their rewards, together with iterative RL training and guided sampling, to improve alignment with affinity and synthetic accessibility objectives. Experiments on CrossDocked2020 show improved docking affinity, generative quality, and inference efficiency compared to prior approaches.

### Strengths
- The paper presents a clear and unified theoretical framework that connects gradient-guided diffusion, direct preference optimization, and KL-regularized reinforcement learning in the context of structure-based drug design.
- The proposed reward-conditioned generative policy and iterative RL refinement strategy are implemented thoughtfully, with strong engineering execution.
- Experimental results demonstrate consistent performance gains and significant inference-time speedup compared to state-of-the-art diffusion-based molecular design models, indicating that the RL perspective not only clarifies the theory but also yields practical benefits.

### Weaknesses
- The three proposed components, reward conditioning, iterative RL, and mixed guidance, are all adaptations of existing techniques from diffusion modeling and RL literature rather than fundamentally new contributions. The theoretical unification, while neat, largely reuses soft-RL derivations known from Haarnoja et al. (2017) and Schulman et al. (2017).
- The method is only evaluated on one benchmark (CrossDocked2020). There is no demonstration of generalization to other protein–ligand datasets (e.g., PDBBind), which weakens claims of broad applicability.
- Although the RL formulation is mathematically sound, it doesn’t yield deeper theoretical insights or new algorithmic properties. The propositions largely restate standard equivalences between KL-regularized RL and reward-weighted likelihoods.
- While the reported 6.8% success-rate gain is notable, it is relatively modest given the methodological complexity. It remains unclear whether the improvement primarily stems from the reward-conditioning trick or the iterative data expansion.

### Questions
- The paper adopts normalized Vina and SA scores as independent reward dimensions. How does the proposed method trade off between binding affinity and synthesizability when the objectives conflict, and is there an explicit mechanism to avoid dominance of one reward over the other?
- The work emphasizes bypassing value-function estimation by reward conditioning and iterative dataset expansion. Could the authors provide empirical evidence showing that reward conditioning yields superior credit assignment compared to learned value models, beyond the early-stage bias argument?
- The experiments are conducted primarily on CrossDocked2020. Have you tested generalization to other SBDD datasets (e.g., PDBBind)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes RL4SBDD, a unified reinforcement-learning framework for preference alignment in structure-based drug design, which bridges gradient-based and gradient-free methods under a single KL-regularized objective, and emphasize the limitation including biased value estimation, insufficient use of training data, and off-policy distribution shift of current preference alignment methods. The authors introduce a reward-conditioned policy model $\Phi_{P}$ with value model $\Phi_{V}$, and a mixed-guidance sampling strategy that combines gradient-based and gradient-free methods for better performance. The method achieves state-of-the-art results on CrossDocked2020 with improved efficiency.

### Strengths
- Provides a clear theoretical unification of preference alignment methods under a principled RL objective.
- Offers solid theoretical derivations and clean algorithmic formulation
- Introduces a reward-conditioned generative policy, novel in SBDD and well-grounded
- Comprehensive experiments with consistent performance improvements and ablations

### Weaknesses
Please see questions below.

### Questions
1. Lines 205–206 and 209–210 state that current DPO data selection leads to suboptimal results. However, this appears not to be an intrinsic limitation of DPO itself, but rather a data efficiency issue, since DPO must iterate over all possible preference pairs to fully utilize training data.
2. The rationale for the noise reduction design requires further justification. When noise level decreases from 1 to 0.2, the diversity drops sharply to 0.53, while the success rate increases only marginally. This suggests that the current setting may lead to over-regularization.
3. It remains unclear to me whether the reward-conditioned policy model alone suffices for generation. What is the performance when sampling directly from $\Phi_{P}$ with high-reward conditioning only? Why is additional mixing with the reference model $\Phi_{ref}$ necessary?
4. Is there any quantitative analysis of the reward model $\Phi_{V}$? Its predictive accuracy and stability would help clarify how much it contributes to the overall performance.
5. The authors only use the normalized Vina Score and SA score as reward signals, neglecting QED. Given that the improvement in QED is relatively small, it would be helpful to explain this choice and whether the framework generalizes to other reward functions.

### Soundness
3

### Presentation
3

### Contribution
3
