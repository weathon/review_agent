# From Many Imperfect to One Trusted: Imitation Learning from Heterogeneous Demonstrators with Unknown Expertise

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Imitation learning (IL) typically depends on large-scale demonstrations collected from multiple human or algorithmic demonstrators. Yet, most existing methods assume these demonstrators are either homogeneous or near-optimal---a convenient but unrealistic assumption in many real-world settings. In this work, we tackle a more practical and challenging setting: IL from heterogeneous demonstrators with unknown and widely varying expertise levels. Instead of assuming expert dominance, we model each demonstrator's behavior as a flexible mixture of optimal and suboptimal policies, and propose a novel IL framework that jointly learns (a) a state-action optimality scoring model and (b) the latent expertise level of each demonstrator, using only a handful of human queries.  The learned scoring model is then integrated into an policy optimization procedure, where it is fine-tuned with offline demonstrations, on-policy rollouts, and a fine-grained mixup regularizer to produce informative rewards.  The agent is trained to maximize these learned rewards in an iterative fashion. Experiments on continuous-control benchmarks show that our approach consistently outperforms baseline methods. Even when all demonstrators are highly suboptimal, each exhibiting only 5-15% optimality, our method achieves performance comparable to a baseline trained on purely optimal demonstrations, despite our lack of optimality labels.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The submission studies imitation learning from demonstrators with different levels of expertise, assigning each a scalar expertise value and treating trajectories as mixtures of optimal and suboptimal behavior. It introduces a two-stage pipeline: an EM-style algorithm that learns an optimality score function and the expertise levels, followed by policy training that fine-tunes the policy with stopping, mixup, and an agent-matching penalty. Stage 1 has a convergence guarantee. On MuJoCo tasks across general and very low expertise regimes, the method outperforms GAIL, RIL, and WGAIL, with ablation showing the effectiveness of relabeling.

### Strengths
- The problem formulation is clear, and the approach is reasonable.
- A standard convergence analysis of the EM algorithm is provided for the first stage of the proposed method.
The proposed method is tested on MuJoCo and Gymnasium tasks.
- An ablation study on relabeling, early stopping, and top-k selection.
- Two sets of experiments—the “general expertise test” and “low expertise test”—demonstrate the effectiveness of the proposed method.

### Weaknesses
- The model for the expertise level is too simple compared to (https://arxiv.org/pdf/2202.01288) mentioned by the paper. The current formulation may miss demonstrators who are experts in some regions and poor in others.
- The criteria (line 253 and line 264) discussed after Theorem 1 make the approach ad hoc and require high quality supervision (albeit a small amount).
- ILEED is missing from the experiment.

### Questions
- What is the usage of alpha-hat_i in Algorithm 2?
- Why is it fine to choose alpha’ = 0.5 (line 209)? What could a misspecified outcome be? How can the misspecified case be handled?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles imitation learning (IL) when demonstrations come from heterogeneous demonstrators of unknown and widely varying expertise. The authors model each demonstrator as a mixture policy under a latent optimality label on state–action pairs. Stage 1 learns (i) a state–action optimality scorer via surrogate demonstrator classification, and (ii) demonstrator expertise levels by an EM‑style alternating procedure. Stage 2 uses a surrogate reward to train a policy with SAC, while iteratively refining using on‑policy rollouts, a negative (agent‑matching) term, and a mixup regularizer, plus a top‑k pseudo‑labeling scheme with an early‑stopping heuristic. On MuJoCo control tasks, the method outperforms GAIL, RIL, WGAIL and approaches performance of an oracle GAIL trained on hand‑selected optimal subsets, including regimes where all demonstrators are highly suboptimal.

### Strengths
- The paper addresses a realistic data regime, i.e., many imperfect demonstrators with unknown quality, and proposes a concrete way to mine signal without explicit optimality labels.
- Leveraging surrogate set classification (SSC) to recover P(z=1∣x) from multi‑set membership is a smart reduction that connects demonstrator‑ID prediction to optimality scoring with a known transform.
- The refinement loop (on‑policy rollouts, an agent‑matching penalty, and mixup) directly targets covariate shift and over‑confidence issues that often hurt IL, and the ablations help isolate these effects.

### Weaknesses
- The EM‑style analysis is potentially incorrect in places. In Theorem 1, the “M‑step” is written with a minimization over $\(\phi, \alpha\)$ that still plugs in $\phi_t$, and the “E‑step” uses a hard 0.5 threshold instead of the expected posterior E[z∣s,a]. More importantly, identifiability of the expertise priors $\{\alpha_i\}$ and the optimality scorer via SSC requires strong conditions (e.g., class‑conditional distributions invariant across sets and “mutual irreducibility” assumptions); these are not clearly stated in the main text, yet they are central to SSC’s guarantees. Without them, $\alpha$ and $f_\phi$ can be non‑identifiable or flip‑ambiguous. It it necessary to make the assumptions explicit and align the proof to standard EM or variational lower‑bound updates.
- Eq. (1) implicitly assumes a single shared suboptimal policy for all demonstrators. In real data, different novices commit different types of errors and visit different states. SSC’s reduction typically assumes shared class‑conditionals p(x∣z) across sets; the paper’s occupancy‑measure formulation includes demonstrator‑dependent state visitation, which can violate these assumptions and bias the recovered scorer. What if each demonstrator has a different structured suboptimality with domain shift?
- Here are places where the exposition is inconsistent or confusing: the top‑k selection text conflicts with the $f_\phi(s, a) > 0.5$ rule (lowest vs. highest scores), Algorithm 1’s steps don’t align with the proof, and Theorem 1’s objective/updates are misstated.
- Closely related works are not cited or are under‑discussed:
  - AIRL for learning disentangled, portable rewards from demos [1].
  - T‑REX/D‑REX for better‑than‑demonstrator performance from suboptimal data via ranking [2–3].
  - VPIL for vague feedback over demos [4].

## References

[1] Learning Robust Rewards with Adversarial Inverse Reinforcement Learning.

[2] Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations.

[3] Better-than-Demonstrator Imitation Learning via Automatically-Ranked Demonstrations.

[4] Imitation Learning from Vague Feedback.

### Questions
Please refer to the weaknesses.

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
This paper considers the imitation learning problem from heterogeneous demonstrators of unknown expertise, proposing a two-stage EM-style framework that jointly estimates demonstrator expertise and learns a state–action optimality scoring model, which is then used as a for policy optimization. The paper is an extension of ILEED (Beliaev et al., ICML 2022) which considered unsupervised expertise estimation for heterogeneous demonstrators. In the authors' formulation, instead of modeling state-dependent embeddings, it models demonstrator expertise as a global scalar mixture coefficient to have a more structured suboptimality.

### Strengths
- The problem is important as heterogeneous and imperfect demonstrations are the norm in large-scale IL.
- The paper presents a clean EM formulation, with a clear separation between expertise estimation and policy learning.

### Weaknesses
- The novelty is quite incremental relative to ILEED. The new formulation simplifies expertise modeling from state-dependent embeddings to demonstrator-level mixture coefficients and reframes the joint estimation as a classification-based EM procedure. 
- The paper should have direct numerical comparison with ILEED as well as other baselines specifically designed for suboptimal demonstrations. The current basedlines mostly cover standard IL methods.
- Evaluations are conducted on synthetic MuJoCo environments where suboptimality is simulated by mixing optimal and degraded SAC policies. The paper will benefit from real human demonstrations like the robomimic dataset.
- The theoretical contribution (EM convergence) is modest and to my best knowledge a well-known proof.

### Questions
Can you provide quantitative comparison with ILEED and stronger evidence of scalability or generalization using real human demonstrations?

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
3

### Summary
This paper addresses imitation learning (IL) from heterogeneous demonstrators with unknown and varying expertise levels. The authors propose a two-stage framework: (1) jointly learning demonstrator expertise levels and an optimality scoring model through an EM-style iterative procedure, and (2) using this scoring model as a surrogate reward function for policy learning with progressive refinement. The method is evaluated on MuJoCo continuous control tasks under two challenging scenarios - general expertise (0.1-0.9) and low expertise (0.05-0.15) settings. The approach claims to achieve performance comparable to oracle methods trained on purely optimal demonstrations.

### Strengths
- The paper addresses a realistic scenario where demonstrations come from multiple sources with unknown, heterogeneous expertise levels - a common real-world challenge.
- Provides convergence guarantee for the EM-style optimization (Theorem 1), giving the approach theoretical grounding.
- Thorough evaluation across multiple environments, expertise settings, and ablations. The low-expertise test (0.05-0.15) is particularly challenging and demonstrates robustness.

### Weaknesses
- Theorem 1 only guarantees convergence to a stationary point, not optimality
- No sample complexity analysis or bounds on expertise estimation error
- The connection between surrogate classification accuracy and IL performance isn't theoretically characterized
- The "Optimality Alignment Criterion" requires human queries, making the approach not fully unsupervised
- The EM procedure requires multiple random initializations with variance-based selection, which could be computationally expensive

### Questions
- How sensitive is the method to the number of human queries? The paper uses only 5 queries but doesn't provide ablation on this critical parameter.
- Can you provide theoretical analysis on the sample complexity? How many demonstrations are needed for reliable expertise estimation?
- Why not compare with ILEED directly? The paper mentions it but doesn't include it in experiments despite addressing the same problem.
- What's the computational overhead of the multiple random initializations? How many initializations are typically needed in practice?

### Soundness
3

### Presentation
3

### Contribution
3
