# Learning Task-Agnostic Motifs to Capture the Continuous Nature of Animal Behavior

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Animals flexibly recombine a finite set of core motor motifs to meet diverse task demands, but existing behavior segmentation methods oversimplify this process by imposing discrete syllables under restrictive generative assumptions. To better capture the continuous structure of behavior generation, we introduce motif-based continuous dynamics (MCD) discovery, a framework that (1) uncovers interpretable motif sets as latent basis functions of behavior by leveraging representations of behavioral transition structure, and (2) models behavioral dynamics as continuously evolving mixtures of these motifs. We validate MCD on a multi-task gridworld, a labyrinth navigation task, and freely moving animal behavior. Across settings, it identifies reusable motif components, captures continuous compositional dynamics, and generates realistic trajectories beyond the capabilities of traditional discrete segmentation models. By providing a generative account of how complex animal behaviors emerge from dynamic combinations of fundamental motor motifs, our approach advances the quantitative study of natural behavior.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a task-agnostic motif framework that models animal behavior as a continuous, compositional mixture rather than discrete syllables. The method learns low-dimensional motif bases and expresses reward, value, and policy on the same basis, enabling soft blending of behaviors. For continuous spaces, transitions are modeled with an energy-based approach and trained via noise-contrastive estimation, including a learned mapping from internal features to motifs. In the experiments, synthetic gridworld (recovery of known rewards), real maze navigation (agreement with Ke et al.’s reward maps used as a proxy ground truth), and free behavior (AUC-based discrimination and qualitative visualizations) were examined, where the proposed method outperforms Keypoint-MoSeq, SemiSeg, and OPAL on the reported metric in the last experiment.

### Strengths
Originality lies in shifting from discrete segmentation to a continuous, compositional description that unifies behavior interpretation and policy learning. The technical design is coherent across settings, with clear explanations and visualizations that make the idea of blending motifs intuitive. The evaluation spans synthetic, controlled, and naturalistic data, showing consistent advantages on the chosen metric. The concept of a reusable, low-dimensional basis that both explains behavior and supports control is potentially impactful for neuroscience and IL/RL applications.

### Weaknesses
The manuscript may lack sufficient methodological detail about the NCE negative-sample distribution (ρ), and it may not clearly position the motif idea relative to related robotics primitives or recent RL/IL behavioral-modeling work.  Evaluation may be also limited in the second and third experiments. The details are given in the following questions.

### Questions
1. The concrete definition and implementation of the noise distribution ρ (i.e., from which distribution you sample the negatives) are not fully clear to me. For robustness and reproducibility, could you specify exactly how ρ is chosen and implemented in each experiment? In the future, providing an ablation showing how different choices of ρ (and the number of negatives) affect training stability and final performance would be better.

2. This study’s motifs concept appears very close to robotics literature on primitives/skills with mixing or blending. While some citations are given, I feel a more systematic positioning in Related Work would be helpful. For example, is it possible to compare (a) the basis relation (e.g., linear combination of probabilistic movement primitives vs. linear reward/Q on motifs in this study, including latent-space structure and regularization), (b) learning regimes (supervised / IL / RL / offline RL), and (c) treatment of normalization (need for EBM/NCE)?

3. Recent works that explain animal behavior via RL/IL are cited, but the background positioning is also not fully clear to me. To make your novelty explicit, is it possible to clarify: (a) the objective functions (MaxEnt, IRL/IL variants), (b) learning signals used (behavior-only vs. joint optimization with neural data), and (c) whether and how your motif space can be related or mapped to neural representations?

4. In the second experiment, When using the Ke et al. reward maps as a valid ground truth, is it possible to provide cross-validation of external validity? For example, agreement with other algorithms or with human annotations (e.g., correlation, map distance metrics, overlap of high-reward zones)? 

5. In the third experiment, in addition to AUC, is it possible to evaluate long-horizon rollouts from the learned policy against statistics of real data, and/or a blinded human discrimination test (human judges comparing generated vs. real trajectories)? Either would provide complementary evidence for generative quality and interpretability.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces Motif-based Continuous Dynamics (MCD) for modeling animal behavior as continuous compositions of a finite set of reusable motor motifs. Methodologically, MCD (i) learns interpretable motif “bases” by exploiting representations of behavioral transition structure, and (ii) models time-varying behavior as smoothly evolving mixtures of those motifs. Experiments span a multi-task gridworld, a labyrinth navigation task, and freely moving animal behavior, showing reusable components, realistic trajectory generation, and advantages over discrete-segmentation approaches.

### Strengths
1. Recasts behavior modeling as continuous mixtures of interpretable motifs, moving beyond discrete syllables.
2. Multi-environment evaluation suggests motif reusability and improved generative realism compared to discrete segmentation.
3. Offers an interpretable generative account of behavior that could aid cross-task generalization, analysis of natural behavior, and links to neural data.

### Weaknesses
1. Compare not only to discrete segmentation (HDP-HMM/AR-HMM, SLDS, MoSeq-style) but also modern continuous SSMs (e.g., Neural SSM/RSSM, N-SLDS) that capture smooth trajectories without explicit motifs.
2. Demonstrate cross-task/subject transfer: learn motifs on subset A, evaluate reuse and performance on held-out tasks/animals (quantify similarity up to permutation/rotation).
3. Beyond visuals, report held-out log-likelihood, forecasting error, and realism/coverage metrics; if annotations exist, quantify alignment (e.g., NMI/ARI, MI).

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
A method for extracting "motifs" (a set of elemental behaviors) from animal behavior trajectories is proposed. The method is based on a modal decomposition of the transition kernel of the environment and a reward function modeled using such modes. In the paper, they show that the maximum entropy policy based on the action-value function can also be written in terms of the modes. They then suggest a method to estimate such modes (and their activations) using the noise contrastive estimation.

### Strengths
The method is principled. It is nicely derived based on the MDP with minimal assumptions.

The analyses of the experimental results are in-depth and detailed.

### Weaknesses
Although the current results are nicely analyzed, the applicability to various real-world data is not very clear.

Discussion on the method's utility from the ethological or neuroscience points of view is missing. We may be able to read and somehow "interpret" the results, but it is not clear how these are scientifically meaningful. This lack of a domain expert's analysis might make the argument sound slightly arbitrary.

### Questions
**(1)**
The method is still not completely clear. Is there any constraint on the scales of the estimated quantities or functions, $\psi$, $\nu$, $f$, $u$? If they are not constrained, some of them seem to be unidentifiable.

**(2)**
The authors contrast the proposed method with "dynamics-based" approaches. I am not quite sure of the intention here. To me the proposed method looks a kind of dynamics-based too, because it is based on the estimation of the quantities related to the dynamics (the policy can also be seen as a part of the dynamics $s \to s'$). This is a question just out of curiosity and is not a fatal factor in my evaluation though.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Motif-based Continuous Dynamics (MCD), a novel framework for segmenting and analyzing animal behavior. The method reframes behavior analysis within an imitation learning (RL) context, using spectral decomposition of the transition dynamics to learn a set of task-agnostic "motor motifs." These motifs are then used as a basis to construct continuous, compositional policies that aim to explain observed behavioral trajectories. The authors validate their approach on simulated and real-world datasets, arguing that it offers a more flexible and interpretable alternative to traditional discrete segmentation methods.

### Strengths
1.  **Novel Conceptual Framework:** The primary strength of the paper is its innovative application of an RL and imitation learning framework to behavior segmentation. Viewing behavior as the output of a policy optimizing an internal reward is a powerful paradigm shift that promises deeper insights than purely kinematic models.
2.  **Continuous and Compositional Representation:** The method's ability to model behavior as a continuous mixture of motifs is a clear advantage over methods that force a discrete "syllable" at each time point. This naturally handles behavioral transitions and, more importantly, the co-occurrence of different motor programs.

### Weaknesses
1. **Missing Architectures:** Key neural network architectures for  $\nu $ and the mapping $f$ in the continuous version are not specified in the main text or the appendix.
    *   **Hyperparameter Sensitivity:** The framework appears to have a large number of hyperparameters (e.g., motif dimensions, learning rates for multiple components, NCE negative samples, GRW priors) that would require significant tuning. The lack of detailed training protocols and ablation studies makes it hard to assess the robustness of the method.

2.  **Fairness of Experimental Comparison:** The quantitative comparison in Section 4.3 using the AUC metric is concerning. MCD's motifs are high-dimensional, continuous functions that explicitly encode the *strength* of each motif's expression ($u(t)$). In contrast, the baseline methods (Keypoint-MoSeq, SemiSeg, OPAL) are primarily designed to output a discrete behavioral label or syllable. By design, MCD's richer, continuous representation has more capacity to fit the nuances of the data. Therefore, outperforming on a likelihood-based metric like AUC seems like an expected outcome of this representational advantage, rather than a clear demonstration of a superior underlying model.

3.  **Limited and Primarily Qualitative Real-World Validation:** While the idea is compelling, the practical advantages on real-world data are not convincingly demonstrated. The analysis is confined to a single freely-moving mouse dataset, and the comparison to state-of-the-art baselines is almost entirely qualitative. While the visualizations are insightful, they are not a substitute for rigorous quantitative comparisons on established benchmarks.

4.  **Lack of Time-Scale Interpretation:** Classical spectral methods on transition matrices provide a natural interpretation of motifs/eigenmodes in terms of the time-scales of the dynamics. It is unclear if the EBM+NCE framework used for the continuous case retains any of this powerful property. The paper does not discuss how the learned motif $\phi$ relate to the long-term or multi-scale structure of the behavior, a feature that methods like MotionMapper explicitly address via wavelet.

### Questions
1.  **On Model Architectures and Dimensions:**
    *   **Question 1a:** For Experiment 4.1 (discrete gridworld), you state the number of motifs is 64. Does this mean the motif feature dimension is `d=64`, and the learned `φ` matrix is of size `(9 states * 4 actions) x 64`? Please confirm the relationship between "number of motifs" and the feature dimension `d`.
    *   **Question 1b:** For the continuous version (Sec. 3.2), what are the specific neural network architectures (number of layers, hidden sizes, activation functions) used for `ν(s')` and the mapping `f(ψ)`? This information is essential for reproducibility and is currently missing.

2.  **On Novelty and Relation to Prior Work:**
    *   **Question 2:** How does the motif discovery method for the discrete case differ from the spectral decomposition representation proposed in SPEDER (Ren et al., 2023)? A direct comparison would help clarify the novelty of this part of your contribution.

3.  **On Theoretical Justifications:**
    *   **Question 3a (Time-Scales):** Do the motifs learned via your continuous EBM+NCE framework have an interpretable connection to the time-scales of the behavioral dynamics, analogous to the eigenvalues in classical spectral analysis or the scales in wavelet transforms? If not, does the framework lose the ability to explicitly model multi-scale behavioral organization?
    *   **Question 3b (Single-Step vs. Long-Horizon):** Could you please provide a theoretical argument or an intuitive explanation for why features (`ψ`, `ν`) learned via the single-step NCE objective (Eq. 7) are suitable as a basis for the long-horizon Q-function? Is there an implicit connection that guarantees these locally-optimized features capture global dynamic properties?

4.  **On Experimental Design:**
    *   **Question 4:** Given that MCD uses a continuous, weighted representation while baselines like Keypoint-MoSeq use discrete states, do you agree that a direct comparison on a likelihood-based metric like AUC may inherently favor MCD? Could you propose or discuss an alternative evaluation protocol that might offer a fairer comparison, perhaps by evaluating the quality of the segmentation itself or the downstream utility of the learned representations?

**Minor Comment:**
*   In the paragraph starting at line 36, you state there are "three major limitations" but then proceed to list four distinct points.

### Soundness
3

### Presentation
2

### Contribution
2
