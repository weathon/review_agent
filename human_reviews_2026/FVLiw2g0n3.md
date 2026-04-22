# Sample Efficient Offline RL via T-Symmetry Enforced Latent State-Stitching

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4, 6

## Abstract
Offline reinforcement learning (RL) has achieved notable progress in recent years. However, most existing offline RL methods require a large amount of training data to achieve reasonable performance and offer limited out-of-distribution (OOD) generalization capability due to conservative data-related regularizations. This seriously hinders the usability of offline RL in solving many real-world applications, where the available data are often limited. In this study, we introduce TELS, a highly sample-efficient offline RL algorithm that enables state-stitching in a compact latent space regulated by the fundamental time-reversal symmetry (T-symmetry) of dynamical systems. Specifically, we introduce a T-symmetry enforced inverse dynamics model (TS-IDM) to derive well-regulated latent state representations that greatly facilitate OOD generalization. A guide-policy can then be learned entirely in the latent space to optimize for the reward-maximizing next state, bypassing the conservative action-level behavioral regularization adopted in most offline RL methods. Finally, the optimized action can be extracted using the learned TS-IDM, together with the optimized latent next state from the guide-policy. We conducted comprehensive experiments on both the D4RL benchmark tasks and a real-world industrial control test environment, TELS achieves superior sample efficiency and OOD generalization performance, significantly outperforming existing offline RL methods in a wide range of challenging small-sample tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a sample-efficient offline RL method, TELS. The method builds on a T-symmetry-enforced inverse dynamics model (TS-IDM) for learning a latent space and offline policy optimization in the latent space. A guide policy selects a target for the next state, and the final action is extracted from the latent inverse dynamics module. The method is benchmarked on D4RL in a small-sample setting and on a real-world industrial control environment, and the method is found to outperform existing offline RL methods on these benchmarks.

### Strengths
- The method with the latent ODE and symmetry design is well-motivated and connected to prior work. The method appears to be novel and sound.
- The method shows significant sample efficiency improvements over the chosen baselines.
- The ablations in 4.2 are highly interesting, and they clearly illustrate the value of the ODE property enforcement and the T-symmetry consistency loss. The OOD experiments in Figure 3 are likewise impressive, and the method seems to be fast to train.
- The fact that the learned representation also benefits IQL and TD3+BC supports the claim that the latent-space learning is valuable and generalizable.
- Even though the overall dynamic model learning objective (Eq. 7) consists of multiple terms, equal weighting of all the loss terms is nice and speaks to the robustness of the method.

### Weaknesses
- No code made available.
- You could consider citing and comparing to existing trajectory stitching methods in offline RL like DiffStitch [1].
- The real-world experiment (Table 2) is very cool, but its significance is somewhat hard to assess. Given that the experiments were run once (as I understand the paper), there are no error bars, and the differences in results could simply be due to, e.g., how well the initial hyperparameter guess worked out.
- The paper claims to "completely bypass the conservatism issue caused by the action-level regularization". However, the deterministic policy loss (Eq. 9) simply replaces the TD3+BC-like action-regularization with a next state-regularization term. The stochastic variant uses AWR over the next states. It is still somewhat unclear if this next state-regularization leads to different behavior and/or learning dynamics than action-level regularization in practice. The empirical results indicate that the difference-maker is the latent space and the advantage-weighed regression for the maze tasks.
- The weight of \alpha (Table 10) differs greatly between environments, raising questions about hyperparameter sensitivity.
- As acknowledged by the authors, the ODE and T-symmetry regularizations can limit the model's expressive power, making this method specifically useful for small datasets.
- Minor: Some issues with presentation & typos, for instance: Equations 9, 10 use h_{ivs} instead of h_{inv}, Appendix E "Border Impact", Zhan et al. 2025a and 2025b is the same paper cited twice, comparision (Table 11 caption), exhibt on L126, Training Perparameters (Table 9). 

[1] Li, G., Shan, Y., Zhu, Z., Long, T., & Zhang, W. (2024). Diffstitch: Boosting offline reinforcement learning with diffusion-based trajectory stitching. ICML.

### Questions
- Eq3: Do you use stop-gradient at all, e.g. for z_s, z_{s'} in the second term?
- Would it be possible to ablate the second term of the deterministic policy loss by replacing it with a BC-like term in the action space?
- Would you expect this method to be suitable for offline-to-online adaptation? I interpret that the l_{T-sym} term in the policy loss is essentially preventing the policy from going where the dynamics model is inaccurate, and if you fine-tune the encoder, you might run into non-stationarity issues. Is my understanding correct?
- Could you discuss the potential trade-offs of using the T-symmetry prior? It could be a problem in manipulation tasks or other tasks with impacts. On the other hand, based on Table 1, the method works reasonably well in Adroit. Evaluating also on *-cloned and/or *-expert would strengthen the case.
- Would the computation of the Jacobians be a bottleneck in the case of, for example, visual offline RL tasks, where the networks would benefit from being scaled up? Probably manageable, if you use a frozen backbone? If you have time and resources, V-D4RL experiments, for instance, could add value to the paper.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes TELS, a sample-efficient offline reinforcement learning (RL) algorithm that leverages time-reversal symmetry (T-symmetry) to learn latent state and action representations. 
By enforcing T-symmetry in an inverse dynamics model and optimising a guide policy entirely in latent space, the method improves out-of-distribution generalisation and sample efficiency compared to existing offline RL approaches. 
The paper is clearly written, well-motivated, and supported by thorough experiments on both D4RL benchmarks and a real-world industrial control task, demonstrating strong empirical gains in small-sample regimes.

### Strengths
- Clarity and presentation:

    The paper is clearly written and well-organised. The introduction and preliminaries effectively motivate the work, set the stage for the proposed approach, and position it well in relation to prior research. The method and experimental sections are easy to follow, and the paper maintains a strong logical flow throughout.

- Methodological contribution:

    Although the proposed method is a relatively straightforward extension of existing offline RL techniques, the authors demonstrate that this extension—enforcing time-reversal symmetry in the latent dynamics model—has a substantial positive impact on performance and generalisation.

- Empirical evaluation:

    The experiments are comprehensive and well-designed. The results show strong performance, particularly in data-limited offline RL settings, and provide convincing evidence of the method’s effectiveness and sample efficiency. The inclusion of both D4RL benchmarks and a real-world industrial control environment strengthens the paper’s practical relevance.

### Weaknesses
- Captions lacking detail:

    In general, table and figure captions do not provide enough information.

    - Figure 1: The caption should explain how to interpret the figure.

    - Table 1: The caption should clarify what “±” represents (e.g., standard deviation?), what the bolded values indicate (best method overall or under a statistical test such as a paired t-test?), and where readers can find details about the reduced-size datasets.

    - Table 2: In the third row, two values are bolded—please clarify what bolding signifies in the caption.

    - Table 3: The caption should explain what each row represents, as this is not clear on first reading.

- Missing normalisation details:

    The paper does not specify how the normalised scores are computed. The authors should include this information, ideally in the captions of tables or figures where normalised scores are reported.

- Figure 2 presentation:

    Figure 2 could be improved by aggregating results across both environments (and potentially more). The authors may find the rliable package useful for this purpose. I recommend reporting the interquartile mean along with 95% stratified bootstrap confidence intervals.

- Clarification on thermal safety violations (Line 358):

    The paragraph beginning on Line 358 does not make it clear what causes thermal safety violations or what they reveal about the algorithm’s behaviour. Please add a brief sentence providing intuition.

Minor issues:
- Line 78: When introducing “latent representations,” clarify that these refer to latent state and action representations.
- Line 102: The citation should be presented textually (i.e., integrated into the sentence rather than in parentheses).
- Line 311: Clarify the meaning of the tilde (~) symbol. Should this instead read “5k–100k”?

### Questions
1. One thing that I think has been overlooked is that this method won't work if the environment's transition dynamics are stochastic. Please can the authors comment on this?
2. What is $h\_{ivs}$ in Equation 9?

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
1

### Summary
This work introduces a novel T-symmetry enforced latent state-stitching (TELS) algorithm designed to improve the sample efficiency and OOD generalization in offline RL. While existing offline RL methods requires massive dataset to learn reliable policies, and perform poorly when an action has not been observed in training data, TELS addresses those by focusing on maximizing the utility of limited dataset. 

The main technical contribution is the TELS mechanism, which learns a low-dimensional latent state representation. It can help creates a robust set of imagined trajectories, which serves as an implicit form of data augmentation and leads to a more reliable policy.

### Strengths
1 The TELS demonstrates superior performance, such as D4RL's medium replay datasets, showing its effectiveness in data-limited settings.

2 The TELS policy is trained to navigate a wider and more reliably state-action space, reducing the conservativeness required by typical OOD regularization techiniques.

3 The ability to stitch data allows the algorithm to handle disparate parts of the state space, making it highly suitable for real-world applications where data is collected sporadically.

### Weaknesses
1 TELS algorithms requires a complex model, including both RL components and particularly a separate high-quality latent state model with symmetry regularization. 

2 The performance of TELS depends heavily on the learned latent space being an accurate and consistent representation. 

3 While latent stitching is meant to improve generalization, if the environment dynamics violate the T-symmetry assumption, the learned policy may rely on imagined transitions that are impossible in practice, which may ultimately japardize its performance.

### Questions
1 See Weaknesses.

2 Regarding overall learning objective (7) and Table 7: Selecting same weight $\beta$ for $\ell_{dyn}$ and $\ell_{ode}$ is natural while selecting the same weight for the last term $\ell_{T-sym}$ is not intuitive. a) It seems Table 7 is on Hopper-me rather than 10k Hopper-m (Check Table 6 performance score). b) Provide further evidence for weights $(1,1,5)$, $5,5,5$, $(0.5,0.5,1)$ and $(0.5,0.5,0.5)$ could be helpful to show the rationale for selecting a shared $\beta$. A better illustration would be a performance score figure where x axis is weight of $\ell_{dyn},\ell_{ode}$ , y axis is $\ell_{T-sym}$ , and z axis is the score.

3 Can the latent state encoder learned by TELS transferable? What is the performance by re-using the trained encoder from related tasks?

4 It would be great to demonstrate the idea of state-stitching by toy examples to show successful (or unsuccessful) stitched trajectories projected back into the original observation space. This would help the reader to understand the concept of state-stitching.

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
4

### Summary
This paper introduces a T-symmetry enforced inverse dynamics model (TS-IDM) that can learn well-behaved and OOD generalizable latent representations, and facilitate action inference. The proposed method outperforme existing offline RL algorithms on small datasets.

### Strengths
The paper is clearly written and logically structured. 
The proposed method outperforme some offline RL algorithms on small datasets.

### Weaknesses
1.	The paper appears to be a straightforward combination of the POR and TSRL.
2.	The compared baselines are relatively outdated and do not include comparisons with some recent and important baseline.

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes TELS, an offline RL algorithm addressing poor sample efficiency and limited OOD generalization in existing methods. It leverages time-reversal symmetry (T-symmetry) and latent state-stitching via a T-symmetry Enforced Inverse Dynamics Model (TS-IDM), which learns generalizable latent representations by enforcing ODE properties. TELS optimizes a latent guide-policy to avoid action-level conservatism and extracts actions via TS-IDM. Experiments on reduced-size D4RL benchmarks and a real-world data center testbed confirm its SOTA small-sample performance.

### Strengths
1. Novel and Principled Design: Integrating T-symmetry (a fundamental physical property) and latent state-stitching is clever. The ODE perspective—regularizing both encoders and decoders (unlike TSRL’s encoder-only focus)—ensures distribution-agnostic dynamics capture, while separating latent guide-policy optimization from action extraction avoids over-conservatism, making the method theoretically sound.
2. Comprehensive Experiments: TELS is rigorously tested on 5k–100k sample D4RL tasks (0.5–10% of original sizes) and outperforms 11 baselines. It also works in real-world data center cooling (20.17% ACLF, no thermal violations). Ablations (TS-IDM components, T-symmetry regularization) and OOD tests (Antmaze critical sample deletion) validate robustness.
3. Clear Computational Efficiency: The appendix addresses ODE-related overhead concerns, showing TELS (2-layer MLPs, compact latent space) is efficient—JAX implementation trains in 20 minutes on 10k samples, outpacing baselines like TSRL (160 mins) and CQL (780 mins).

### Weaknesses
1. Unaddressed Dynamics Assumptions: TELS assumes invertible, deterministic environment dynamics (required for T-symmetry and inverse modeling), but real-world settings often have non-invertible (e.g., irreversible heat dissipation) or non-deterministic (e.g., noisy sensors) dynamics. No discussion of how TELS performs here limits its applicability.
2. Limited Latent Interpretability: While TS-IDM learns “well-behaved” latent representations, there is no qualitative analysis (e.g., dimensionality reduction, correlation with physical variables) to show what these representations capture, weakening claims about their structure driving generalization.

### Questions
TELS uses reverse/backward dynamics for T-symmetry and representation learning, similar to PlayVirtual (Yu et al., NeurIPS 2021), which leverages backward dynamics for RL generalization.  Could the authors clarify the similarities and differences between TELS and PlayVirtual?

### Soundness
3

### Presentation
3

### Contribution
3
