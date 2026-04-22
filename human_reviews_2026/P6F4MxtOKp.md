# Spectral-guided Physical Dynamics Distillation

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 4

## Abstract
The problem of physical dynamics, which involves predicting the 3D trajectories of particles, is a fundamental task with wide-ranging applications across science and engineering. However, accurately forecasting long-horizon trajectories from initial states remains challenging, due to complex particle interactions and entangled multi-scale dynamics involving both low- and high-frequency components. 
To address this, we propose a novel knowledge-distillation-based framework, SGDD (Spectral-Guided Dynamics Distillation), which integrates a spectral-guided enhancement to adaptively prioritize key frequency components within a unified spatio-temporal representation. Through knowledge distillation, SGDD leverages future trajectories as privileged information during training, guiding a teacher encoder to generate comprehensive dynamics representations while a student encoder approximates them using only the initial state. This enables the student to generate effective dynamics representations at inference, even without privileged information, thereby enabling accurate long-horizon trajectory prediction.
Experimental results on molecule, protein, and human motion datasets demonstrate that our method achieves more accurate and stable long-term predictions than previous physical dynamics models, successfully capturing the complex spatio-temporal structures of real-world systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes Spectral-Guided Dynamics Distillation (SGDD) for long-horizon trajectory forecasting from only initial states. SGDD leverages full future trajectories as privileged information for a teacher encoder and distills that knowledge into a student encoder that must construct the full trajectory solely from the initial state. A spectral-guided enhancement module reweights joint spatiotemporal frequency components, enabling capture key low-frequency patterns.

### Strengths
- The paper is clearly written, well structured, and easy to follow.
- The joint handling of spatial and temporal information, particularly the emphasis on capturing overall low-frequency patterns, is insightful.
- The distillation strategy offers a practical way to train forecasting models that operate from initial conditions alone.
- Aligning the student with the teacher both in representation space and in the spectral domain is conceptually sound.

### Weaknesses
- Forecasting long-horizon time series **solely** from initial states is often unrealistic. In many real-world settings, future trajectories depend on external inputs, or may be influenced by events. For example, predicting a patient’s blood glucose purely from the current state ignores forthcoming meals (e.g., salad vs. burger). Because SGDD predicts future trajectories from only the initial state, it appears unable to handle such exogenously driven series. This is the main reason I lean toward a **weak reject**.
- As I understand it (see Question 2), the method relies on a spatiotemporal joint basis $B_{K} \in \mathbb{R}^{NT \times K}$. While this enables joint spatiotemporal modeling, it introduces two concerns:
  - Because the shape of $B_K$ explicitly depends on the number of time steps $T$, SGDD may only forecast a **fixed** number of steps at inference, limiting practical applicability.
  - For very high-dimensional systems or very long sequences, storing $B_{K}$ could become memory intensive.
- The primary experiments are in scientific domains, yet the method does not enforce physical constraints; the physical consistency of the forecasts is therefore uncertain.
- Experimental results should report mean ± standard deviation across multiple random seeds.
- If possible, evaluating SGDD with different encoders and decoders would better demonstrate the method’s robustness and generality.

### Questions
- I am a bit confused about the “spatial graph.” Each temporal graph $G_t$ already encodes the physical connections among particles. Why is an additional spatial graph $G_s$ mentioned?
- Do the authors build the joint basis $B_{K} \in \mathbb{R}^{NT \times K}$ from the training set and then **store** it for subsequent training and inference?  
  - If **yes**, the storage and fixed-horizon issues in the Weakness section become significant.  
  - If **no**, please clarify how $B_{K}$ is computed/updated and used in practice.
- Are the frequency-specific weights $w$ **different** for the teacher and student encoders? My understanding is **yes**.

### Soundness
3

### Presentation
2

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
The paper proposes SGDD, a “spectral-guided dynamics distillation” framework for long-horizon particle-dynamics forecasting that trains a teacher encoder on full future trajectories as privileged information and distills its spatio-temporal representation into a student that only sees the initial state at test time. Concretely, both teacher and student representations are mapped into a truncated joint spatio-temporal spectral basis built from temporal and spatial Laplacians. A learnable frequency-weighting module emphasizes low-frequency modes while preserving residual components, and training aligns the student and teacher both in representation space and in the spectral coefficients.

### Strengths
1. The joint spatio-temporal spectral basis with learnable per-mode weights provides a principled way to emphasize low-frequency global trends while preserving high-frequency residuals, which directly targets long-horizon stability.  
2. The dual alignment together with gradient detaching for the teacher is a clean instantiation of privileged-information distillation that avoids trivial position-level imitation and back-door leakage during training.
3. On MD17, the method demonstrates large, interpretable gains specifically where baseline errors concentrate in low frequencies.
4. Ablations indicate that both spectral- and feature-space alignments contribute, and that the truncation K trades off emphasis breadth against noise, which is informative for practitioners.

### Weaknesses
1. Statistical rigor is uneven, with several results reported as single numbers or with “±0.0” and no confidence intervals, and with seeds unspecified for some tables, which limits conclusions about robustness.
2. Fairness and tuning budgets are not fully documented. This is important because SGDD wraps baseline decoders; strong gains might partly arise from extra representation capacity or training length unless search spaces and early-stopping criteria are matched and reported. 
3. The method’s dependence on decoder capacity is acknowledged but not quantified.

### Questions
1. Could you replicate all baselines within your codebase with matched search to rule out confounds from external reported numbers, and add a “decoder-capacity-matched” comparison where the decoder architecture and training schedule are identical with and without SGDD?
2. The frequency weighting advocates prioritizing low-frequency modes. Can you provide frequency-conditioned error profiles across datasets (as in Fig. 4 for Benzene) and show at least one regime where high-frequency content is crucial? Do gains persist there, or does performance degrade as K grows?
3. Beyond qualitative statements, it would be good to quantify decoder dependence by swapping in a weaker and a stronger decoder and plotting accuracy vs. decoder capacity to show that SGDD’s benefit is not merely a proxy for more parameters.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the challenging problem of long-horizon trajectory prediction for physical dynamics, where systems exhibit a complex interplay of low- and high-frequency components. The authors propose a novel framework, SGDD (Spectral-Guided Dynamics Distillation), which leverages privileged information (future trajectories) within a knowledge distillation setup. The core of SGDD is a spectral-guided enhancement module that operates on a unified spatio-temporal graph representation.

### Strengths
- Addresses a Core Challenge: The paper directly tackles the critical challenge of disentangling and prioritizing multi-scale dynamics, which is a primary reason why long-horizon prediction is so difficult. The results on low-frequency error reduction are particularly impressive.
- Strong and Consistent Performance: SGDD consistently outperforms very strong, recent baselines (EGNO, GFNode) across multiple diverse and challenging datasets (molecules, human motion, proteins).
- Generality: The framework is designed to be modular. It can be instantiated with different equivariant GNNs as decoders (as shown with EGNO and GFNode), highlighting its flexibility and potential for broad impact.

### Weaknesses
- Complexity of the Framework: The overall system is quite complex, involving multiple encoders, a custom spectral module, and a staged training process. This might pose a challenge for reproducibility and for other researchers seeking to adopt the method.
- Choice of Teacher/Student Architecture: The paper uses sophisticated models (STSGNN, GAT) for the teacher and student encoders. A deeper justification for these specific choices, and an analysis of how the performance depends on the capacity of these encoders, would be beneficial.
- Dependence on a Good Basis: The method's success hinges on the quality of the spatio-temporal graph basis derived from the Laplacian. While this is standard, a discussion on potential limitations for systems where the graph structure is dynamic or poorly defined would be valuable.

### Questions
1.  The spectral-guided enhancement relies on a truncated basis of the *K* lowest-frequency modes. The ablation in Figure 6 shows that performance is sensitive to *K*. How should one choose an optimal *K* in practice for a new dataset or system? Is there a more adaptive way to select the basis rather than a fixed low-pass filter?
2.  The dual-level alignment (in both spatio-temporal and spectral domains) is shown to be crucial. Could you provide some intuition as to why aligning in both domains is superior to aligning in just one? Does one domain contribute more to the final performance than the other?
3.  The paper focuses on distilling a representation. Have you considered also distilling the final output distribution (i.e., the predicted trajectory itself), as is common in classic knowledge distillation? Would this provide complementary benefits?

### Soundness
2

### Presentation
2

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
This paper proposes a novel framework, Spectral-Guided Dynamics Distillation (SGDD), for long-horizon prediction of 3D particle trajectories. The method is designed to tackle the challenge of modeling complex spatio-temporal dynamics that involve a mix of low- and high-frequency components. The core of SGDD is a knowledge distillation setup where a "teacher" encoder has access to the full future trajectory (privileged information) during training, while a "student" encoder only sees the initial state. Both encoders produce a spatio-temporal representation, which is then refined by a "spectral-guided enhancement" module. This module projects the representation onto a joint spatio-temporal basis (derived from graph Laplacians), applies learnable weights to different frequency components, and reconstructs the representation. The student is trained to mimic the teacher's enhanced representation, effectively distilling the knowledge of future dynamics into a model that operates only on the initial state at inference time. The distilled representation is then fed to a decoder (e.g., EGNO or GFNode) to generate the trajectory. The authors demonstrate state-of-the-art performance on molecular dynamics, protein dynamics, and human motion datasets.

### Strengths
1.  **Strong Empirical Performance:** The SGDD framework consistently achieves state-of-the-art results across multiple diverse and challenging benchmarks, demonstrating its effectiveness in practice.
2.  **Effective Use of Privileged Information:** The knowledge distillation setup provides an elegant way to incorporate information from future trajectories during training to improve the representation learned from only the initial state.
3.  **Addresses a Key Challenge:** The paper tackles the important and difficult problem of long-horizon forecasting in systems with complex, multi-scale dynamics.

### Weaknesses
1.  **Methodological Ambiguity:** The construction of the spatio-temporal graph and, more importantly, the process of projecting the initial-state representation onto the spatio-temporal basis are not clearly explained. This is a critical flaw that hinders understanding and reproducibility.
2.  **Insufficient Ablation Studies:** The ablation studies do not adequately isolate the contribution of the core novelty—the spectral-guided enhancement. A baseline that uses the same distillation framework but omits the spectral module is essential to validate the paper's central claim.
3.  **Limited Novelty of Components:** The constituent ideas (knowledge distillation with future-as-PK, graph spectral analysis) are not new. The contribution is in their combination, but the significance of this combination is not fully substantiated due to the aforementioned weaknesses.
4.  **Complexity:** The proposed framework is highly complex, involving two separate encoders, a custom spatio-temporal basis construction, spectral projection, re-weighting, reconstruction, and a two-stage training process. This complexity makes it difficult to dissect the reasons for its success and may limit its practical adoption.

### Questions
1.  Could you please provide a precise mathematical definition of the temporal graph `G_t` and its Laplacian `L_t`? How are the temporal connections established?
2.  Please provide a detailed explanation of how the representation `z_init`, derived from the single initial graph `G_0`, is prepared for projection onto the spatio-temporal basis `B_K`. What are the exact "projection and expansion" steps mentioned in lines 202-204?
3.  Would it be possible to provide results for an ablation study where knowledge distillation is performed directly on the spatio-temporal representations (`z_dyn` and `z_init`) without the spectral-guided enhancement module? This seems crucial for proving that the spectral component is the key to your method's success.
4.  The truncation parameter `K` is a key hyperparameter. The ablation in Figure 6 shows its impact, but how was `K` chosen for the main experiments on the MD17 and Protein datasets? Is it a fixed value, or was it tuned per dataset? How sensitive are the results to this choice?

### Soundness
2

### Presentation
3

### Contribution
2
