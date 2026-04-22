# Binding Mode Matters: Residue-Guided Drug Discovery via Explorative Preferences

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
The discovery of novel hit or lead molecules requires navigating a vast chemical space to identify compounds with optimal binding modes, which are typically unknown beforehand. Despite various generative approaches, they have predominantly relied on optimizing a monolithic scalar docking score to guide generation, masking the distinct contributions of key binding determinants.
In this work, we introduce a paradigm shift by formulating target-based drug design as a multi-objective exploration task, where each objective explicitly corresponds to enhancing interactions with a specific key residue. To this end, we introduce **BindMol**, a novel generative framework that integrates a fragment-based generator with a customized multi-objective reinforcement learning algorithm. By incorporating explorative preferences during training, our approach efficiently uncovers molecules with distinct and desirable binding profiles.
Empirical evaluations demonstrate that **BindMol** facilitates the discovery of structurally novel, high-affinity compounds across five protein targets and establishes new state-of-the-art records on the multi-property optimization tasks in GuacaMol benchmarks, thereby providing a versatile paradigm for goal-directed drug discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Through this paper, the authors propose BindMol, a fragment-based molecule generation framework using a multi-objective RL algorithm. Specifically, the authors newly propose incorporating explorative preferences during training. The experiments show that BindMol can discover molecules with good binding profiles.

### Strengths
- The authors provided the codebase.
- The proposed formulation that views target-based drug discovery as a multi-objective optimization problem, with each objective corresponding to a key residue within the binding pocket, is interesting and reasonable.
- The proposed BindMol shows good performance on various benchmarks.

### Weaknesses
Weaknesses
I will combine the *Weaknesses* section and the *Questions* section. My concerns are as follows:
- The main weakness of this paper is its weak novelty. As the authors mentioned in lines 56~58, the BindMol framework consists of three key components: action space, reward function, and the RL algorithm. For the action space, BindMol adopts FREED [1]. Section 4.1 actually should actually be placed in Preliminaries section and is not an invention of this work. BindMol's reward function design (Section 4.3) is a heuristic that relies on the PLIP tool and docking scores, and it cannot be considered a significant contribution from an ML perspective. The only main contribution of this work is Envelope SAC, a multi-objective RL algorithm that defines a preference-aware Bellman operator and a vectorized Q-function (Section 4.2). However, the central idea is to integrate the envelope-based update mechanism is very similar to MORL [2]. Overall, I am not convinced that this work provides a new approach compared to previous methods in the domain.
- In the GuacaMol MPO experiment (Section 5.2, Table 4), SOTA molecular optimization baselines such as GenMol [3] and Genetic GFN [4] are missing. Comparisons with these baselines are necessary for the results to be considered meaningful.

---

**References:**

[1] Yang et al., Hit and lead discovery with explorative rl and fragment-based molecule generation, NeurIPS, 2021.

[2] Yang et al., A generalized algorithm for multi-objective reinforcement learning and policy adaptation, NeurIPS, 2019.

[3] Lee et al., GenMol: A Drug Discovery Generalist with Discrete Diffusion, ICML 2025.

[4] Kim et al., Genetic-guided GFlowNets for Sample Efficient Molecular Optimization, NeurIPS 2024.

### Questions
Please see the *Weaknesses* section for my main concerns.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes BindMol, a fragment based molecular generation framework for structure guided drug design that treats binding as a multi objective reinforcement learning problem. Instead of optimizing a single docking score, the method defines residue level rewards that capture interactions with individual protein residues and trains a policy using an Envelope Soft Actor Critic algorithm to explore diverse binding preferences. The approach constructs molecules by sequentially attaching fragments and leverages a dynamic vocabulary to expand chemical space during training. Experiments on five protein targets and multi property GuacaMol benchmarks show improved hit rates, docking scores, and chemical diversity compared to strong baselines, indicating that residue guided rewards can promote diverse binding modes and high affinity candidates

### Strengths
- The paper introduces a biologically motivated multi objective formulation for structure based molecular generation, where residue level interaction signals replace a single scalar docking score, providing a more interpretable and controllable reward structure.
- The Envelope SAC algorithm with preference based exploration is a technically novel component that extends entropy regularized RL to vector valued rewards and empirically improves Pareto frontier coverage in binding mode space.
- Experimental evaluation is broad and competitive, covering five protein binding tasks and seven GuacaMol multi property benchmarks, and demonstrating consistent state of the art performance in affinity, novelty, and diversity metrics.

### Weaknesses
- The connection between the scalar reward definition in Section 4.3 and the multi-objective formulation in Section 4.2 remains ambiguous. It is not clearly explained how the final scalar reward integrates into the Envelope SAC optimization process.
- The residue-based reward design merely counts the number of interactions without reflecting residue-specific importance or interaction strength. This simplification may limit the model’s ability to capture nuanced biochemical factors in binding.
- Several mathematical symbols and operators (e.g., ω*, Hα, Qθ) are introduced without full contextual definition or consistent usage, which may hinder the theoretical clarity and reproducibility of the proposed method.
- Although experiments support that multi-residue optimization improves diversity, the paper does not clearly justify why interacting with more residues should yield better ligand quality or affinity. The underlying biochemical rationale remains underexplored.
- The Envelope SAC algorithm involves solving an additional optimization (arg max over ω) during training, but the paper does not quantify the resulting computational overhead or its impact on convergence time.
- In the experimental section, it is not described how the preference vector ω is set during inference, leaving unclear how the model’s multi-objective nature is actually utilized when generating final molecules.

### Questions
- The reward function counts PLIP-detected interactions equally across residues. Do the authors observe cases where increasing the number of weak or geometrically marginal interactions leads to inflated reward? Have they considered weighting interaction types by estimated energetic contribution?
- The model relies on static docking poses to compute residue-level interactions. How do the authors mitigate the risk that suboptimal or strained docking poses lead to spurious interactions being rewarded?
- The dynamic fragment vocabulary is introduced as a novelty. To isolate its contribution, could the authors provide ablation results comparing a fixed-vocabulary variant with identical reward shaping?
- Multi-objective RL methods often face challenges with instability as the number of objectives grows. The paper cites settings with five to thirty residues. Could the authors report performance as a function of the number of objectives, or provide guidance on stability and hyperparameter sensitivity?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this work, the authors introduce BindMol, a novel framework for multi-objective RL in structure-based drug design. Unlike traditional generative models that optimize a single scalar docking score, BindMol decomposes the reward into residue-level objectives, encouraging the generation of molecules with diverse binding modes to the same protein target. The model uses a fragment-based molecular generator and a new RL algorithm called Envelope SAC, which learns a convex envelope over vectorized Q-values to balance multiple interaction objectives efficiently.

Contributions:

Reformulates target-based molecular design as a multi-objective optimization problem focused on residue-level interactions.

Proposes Envelope SAC, a preference-aware RL algorithm for optimizing multiple objectives jointly.

Integrates PLIP-based residue-level rewards with docking scores for richer feedback.

Demonstrates good performance across five protein targets and seven GuacaMol benchmark tasks, with improved hit rates, diversity, and binding affinity compared to prior RL and generative models.

### Strengths
Originality:
The paper introduces a conceptually novel formulation of target-based drug design as a multi-objective optimization problem over residue-level interactions rather than a single docking score, and a novel Envelope SAC algorithm.

Quality:
The methodology is technically sound and well-motivated. The empirical validation is in alignment with other studies in molecular generation domain. 

Clarity:
The paper is clearly written and logically structured and generally easy to follow.

### Weaknesses
The technical novelty of the proposed Envelope SAC algorithm, from the perspective of the ICLR machine learning audience, appears somewhat limited. A more thorough discussion of prior work in multi-objective reinforcement learning including established formulations and optimization strategies would help contextualize the contribution. Furthermore, benchmarking against existing multi-objective RL methods would strengthen the paper’s empirical claims and clarify the specific advantages introduced by the proposed operator. Finally, it would be valuable to know whether the authors have compared their hit rates (Tables 1–3) with recent results such as Pandey et al., “Pretraining Generative Flow Networks with Inexpensive Rewards for Molecular Graph Generation,” arXiv:2503.06337 (2025), which reports strong performance on molecular design tasks for the same targets as considered by the authors. Such comparisons could provide a clearer assessment of BindMol’s relative progress over recent generative paradigms.

### Questions
Please see weaknesses

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
4

### Summary
This paper presents BindMol, a new reinforcement learning framework for structure-based drug design that reformulates molecule generation as a multi-objective optimization problem. Instead of optimizing a single scalar docking score, BindMol defines residue-level rewards based on protein–ligand interaction counts (via PLIP) and introduces Envelope Soft Actor-Critic (Envelope SAC) to explore trade-offs among multiple objectives (binding residues). Experiments on five protein targets and the GuacaMol multi-property benchmark demonstrate that BindMol achieves state-of-the-art performance, generating novel and diverse compounds with improved docking scores and Pareto coverage.

### Strengths
- The paper clearly identifies the limitation of scalar docking-based RL and reframes drug discovery as a multi-objective exploration problem, aligning with the biological reality of residue-specific interactions.
- BindMol consistently outperforms a wide range of strong baselines across multiple targets, with substantial gains in both novel hit ratio and binding diversity.
- The use of explorative preferences effectively improves the coverage of chemical space and encourages multiple binding modes, as supported by hypervolume and case study visualizations.

### Weaknesses
- The integration with fragment-based generation and residue-level rewards, while well motivated, combines existing ideas rather than introducing a clearly novel modeling mechanism.
- The maximization over ω′ in Equation (8) requires either dense sampling or approximation; the paper does not explain how ω⋆ is computed efficiently or whether it introduces bias.
- The dynamic fragment vocabulary update may leak information from test targets if not strictly separated

### Questions
- How does the computational efficiency of BindMol compare to other baseline methods, particularly in terms of training time and docking evaluation cost?
- Since per-residue interaction rewards may incentivize generating larger molecules to form more contacts, how does the molecular weight distribution of samples produced by BindMol compare to those of the baselines?

### Soundness
3

### Presentation
3

### Contribution
3
