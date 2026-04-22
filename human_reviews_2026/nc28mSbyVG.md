# Swap-guided Preference Learning for Personalized Reinforcement Learning from Human Feedback

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Reinforcement Learning from Human Feedback (RLHF) is a widely used approach to align large-scale AI systems with human values. However, RLHF typically assumes a single, universal reward, which overlooks diverse preferences and limits personalization. Variational Preference Learning (VPL) seeks to address this by introducing user-specific latent variables. Despite its promise, we found that VPL suffers from posterior collapse. While this phenomenon is well known in VAEs, it has not previously been identified in preference learning frameworks. Under sparse preference data and with overly expressive decoders, VPL may cause latent variables to be ignored, reverting to a single-reward model. To overcome this limitation, we propose Swap-guided Preference Learning (SPL). The key idea is to construct fictitious swap annotators and use the mirroring property of their preferences to guide the encoder. SPL introduces three components: (1) swap-guided base regularization, (2) Preferential Inverse Autoregressive Flow (P-IAF), and (3) adaptive latent conditioning. Experiments show that SPL mitigates collapse, enriches user-specific latents, and improves preference prediction. Our code and data are available at https://github.com/cobang0111/SPL

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a new method for personalized RLHF. It includes an experimental evaluation on two Llama variants (3B, 8B), with the proposed method comparing favorably to the baselines.

### Strengths
The paper is well-written and easy to understand. It addresses an important and underserved niche of LLM personalization through diverse preferences. The method is well-justified and includes solid theoretical foundations.

### Weaknesses
1. In Section 3, the datasets are not yet introduced in any way, which makes it quite confusing to read.
2. No confidence intervals in Table 1.
3. Evaluations are based on completely synthetic preferences, not actual individual human preferences.
4. The paper does not provide the Pets dataset in meaningful detail, it also seems to be missing from the code release.
5. The performance improvement is relatively modest, which might be worth it depending on the computational cost trade-off.

### Questions
1. Can you run additional experiments to obtain tighter confidence bounds in Table 2? Some of the values are quite close to one another, making it hard to draw conclusions.
2. What is the computational overhead of using this method, over the baselines?
3. How exactly was the Pets dataset constructed?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the problem of posterior collapse in Variational Preference Learning for personalized reinforcement learning from human feedback, where user-specific latent variables often become uninformative and default back to a single reward model. 
This paper propose Swap-guided Preference Learning (SPL), which leverages the mirroring property of swapped preferences and introduces three components: swap-guided base regularization, P-IAF, and adaptive latent conditioning. 
Experiments show that SPL avoids collapse, stabilizes latent representations, and achieves higher preference-prediction accuracy than baselines.

### Strengths
1. The paper clearly identifies posterior collapse in personalized preference learning and introduces the intuitive idea of swap-guided mirroring, where swapping preferences flips the latent mean but keeps variance invariant, offering a novel and insightful diagnostic lens.
2. SPL integrates swap-guided base regularization, P-IAF, and adaptive latent conditioning into a coherent framework, directly addressing collapse while preserving user-specific information. The design is principled, interpretable, and builds effectively on established variational methods.

### Weaknesses
1. SPL introduces many additional hyper parameters, like $\beta, \gamma, \eta$, but does not analyze how robust these hyperparameters are or how much tuning would cost.

### Questions
1. How would you use the personalized reward model in policy model training?

### Soundness
3

### Presentation
4

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
The paper tackles an important problem of personalizing RLHF to diverse users. It builds on existing paradigm that uses variational inference to model different users. The main focus of this paper is on the widely known “posterior collapse” problem with VAEs. The authors introduce swap annotators to the preference dataset i.e. effectively data augmentation by flipping the context labels and the target user preference order to create more data that voids posterior collapse. Further, they introduce a more expressive prior with a normalizing flow based model, and run experiments on preference modelling across two datasets. The experiments show that the final method is robust to posterior collapse and achieve a mean higher accuracy on preference modelling over the baselines.

### Strengths
- The paper is well written, with the motivation and background work sufficiently laid out.
- The authors clearly expose the posterior collapse problem with experimental evidence. This is a strong motivation towards explaining the issues with priors works and motivating the solutions introduced.
- The data augmentation technique for regularisation seems to be very effective and low-overhead, which potentially makes it very efficient.
- The P-IAF architecture introduced ensures that the regularisation and the mirror property hold after the transformations. Also present a theoretical justification for the approximation with the swap-invariant and swap-depedent variables.
- The authors show the validity of the method on multiple datasets againts multiple baselines.

### Weaknesses
- An issue with the method is that it assumes that the user-context is provided via binary preference labels. This doesnt seem to be scalable as more recent works [1], have focused on expanding user context to muti-turn dialogue. It would be interesting if the authors could discuss the applicability of the introduced regularisation to other forms of context.
- The swap based data augmentation seems to be a very interesting contribution. If the authors could include a baseline that trains a VPL based model with only the additional mirrored data, it could further show the benefit of the additional contributions beyond the swap guided pairs.
- In Table 3:, The adaptive latent conditioning and base regularisation provide negligible improvement to the modelling accuracy, which makes it hard to justify their contribution to the overall performance.
- Overall, the paper introduces interesting and know-techniques to improve preference modelling under diverse users. While the problem setting and the motivation of the solution is completely justified, the individual components introduced and the ablations over them provide relatively weak signals. If the authors are able to answer my questions, and provide additional ablations or experiments to strengthen their claims I would be happy to increase my score.

[1] Enhancing Personalized Multi-Turn Dialogue with Curiosity Reward. Yanming Wan, Jiaxing Wu, Marwa Abdulhai, Lior Shani, Natasha Jaques

### Questions
- In Figure 3b is the non-collapsed posterior trained via VPL or SPL? I.e does the sign reversal happen under the original dataset or the augmented dataset?
- The swap based regularisation assumes that the dataset contains context and preference pairs where the preference order is always dependent on the user context. But this would introduce wrong labels for preference pairs that are independent of the context i.e. same label regardless of the context. How do the authors resolve this issue?

### Soundness
3

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
The paper shows that a leading personalization method, Variational Preference Learning, suffers from posterior collapse, causing user latents to be ignored. It proposes Swap-guided Preference Learning (SPL), which exploits a mirroring property of preference pairs by constructing fictitious “swap” annotators and enforcing sign-reversal of posterior means while keeping variances invariant. SPL combines (1) swap-guided base regularization, (2) a Preferential Inverse Autoregressive Flow (P-IAF) to disentangle swap-reversal and swap-invariant signals, and (3) adaptive latent conditioning for the reward decoder. Experiments demonstrate that SPL reliably prevents collapse across KL weights and improves preference-prediction accuracy over strong baselines.

### Strengths
1. The paper surfaces posterior collapse in preference learning (especially for VPL), provides detailed diagnostics of why the user latent is ignored, and introduces a swap-guided remedy centered on the mirrored “swap” property to keep the latent informative.
2. SPL is derived from a clear ELBO with a swap-guidance regularizer. Analyses show how base regularization and the proposed P-IAF reduce swap-mismatch and prevent cross-context leakage, which explains why the posterior should not collapse.
3. Across KL weights, SPL maintains substantially higher Active Units than VPL (particularly on UF-P-4), indicating non-collapsed, informative user latents.

### Weaknesses
1. Some recent methods reported results on UF-P and are discussed by the authors but not included in experiments (e.g., Nam et al., 2025), making it hard to situate SPL’s gains against the newest alternatives. Adding these would strengthen claims.
2. The approach adds flow-based inference (P-IAF) and swap-guidance terms, which plausibly increase compute and memory, but the paper does not report training time, inference latency, or budget-constrained comparisons. Reporting these costs would clarify practicality.

### Questions
1. What are SPL’s training and inference compute overheads relative to VPL/VPL-IAF?
2. Is SPL robust to noisy labels (e.g., label-flip noise) which is common in real-world settings?
3. How well does SPL generalize to unseen but in-distribution user contexts (i.e., train/test contexts are disjoint but user classes remain fixed)?
4. How does SPL compare to more recent personalized-reward baselines on UF-P (e.g., Nam et al., 2025)?
5. How were key hyperparameters chosen, and how did their values influence Active Units and accuracy?

### Soundness
3

### Presentation
3

### Contribution
3
