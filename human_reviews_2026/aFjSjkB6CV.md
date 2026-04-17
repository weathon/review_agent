# EXPO: Stable Reinforcement Learning with Expressive Policies

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 8

## Abstract
We study the problem of training and fine-tuning expressive policies with online reinforcement learning (RL) given an offline dataset. Training expressive policy classes with online RL present a unique challenge of stable value maximization. Unlike simpler Gaussian policies commonly used in online RL, expressive policies like diffusion and flow-matching policies are parameterized by a long denoising chain, which hinders stable gradient propagation from actions to policy parameters when optimizing against some value function. Our key insight is that we can address stable value maximization by avoiding direct optimization over value with the expressive policy and instead construct an on-the-fly RL policy to maximize Q-value. We propose Expressive Policy Optimization (EXPO), a sample-efficient online RL algorithm that utilizes an on-the-fly policy to maximize value with two parameterized policies -- a larger expressive base policy trained with a stable imitation learning objective and a light-weight Gaussian edit policy that edits the actions sampled from the base policy toward a higher value distribution. The on-the-fly policy optimizes the actions from the base policy with the learned edit policy and chooses the value maximizing action from the base and edited actions for both sampling and temporal-difference (TD) backup. Our approach yields up to 2-3x improvement in sample efficiency on average over prior methods both in the setting of fine-tuning a pretrained policy given offline data and in leveraging offline data to train online.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The manuscript introduces EXpressive Policy Optimization (EXPO), a reinforcement learning (RL) algorithm designed to enable stable and sample-efficient online fine-tuning of expressive policy classes (e.g., diffusion or flow-matching policies) using both offline data and online
interaction. The key innovation is to avoid direct value maximization through the expressive policy, instead employing two parameterized components: (1) a large base expressive policy trained with an imitation learning (IL) objective, and (2) a small Gaussian edit policy that locally adjusts actions toward higher Q-values. These are combined via an on-the-fly (OTF) policy that selects the highest-value action among base and edited samples for both execution and temporal-difference (TD) backup (Sec. 4.2; Eq. (3); Fig. 2). EXPO demonstrates up to 2–3× higher sample efficiency than prior methods on 12 sparse-reward tasks across Antmaze, Adroit, Robomimic, and MimicGen domains (Fig. 3–4; Sec. 5.1–5.4).

The paper is well-motivated and tackles a clear problem—stability in RL fine-tuning of expressive policy classes—with a simple yet effective hybrid approach combining imitation learning and Q-value editing (Sec. 4.1–4.2; Fig. 1–2). Empirical coverage is broad and
convincing, with comprehensive ablations and strong baselines (Sec. 5; Fig. 3–7). The presentation is clear, though several mathematical formulations and assumptions could be better formalized (e.g., edit distance constraint, entropy-regularized variants). Moreover,
computational efficiency and theoretical analysis of stability remain underdeveloped—currently argued qualitatively (Sec. 6). Nonetheless, EXPO offers a promising direction for expressive policy fine-tuning that balances innovation with empirical rigor.

### Strengths
Clear motivation and problem framing:
1. The paper explicitly identifies the gradient instability of expressive policy classes (diffusion/flow policies) when directly optimizing Q-values (Sec. 1; p. 2–3). Thisconnects to recent findings on denoising-step chains (Ding & Jin 2024; Park et al.2025), establishing a solid motivation.
2. The problem is formalized as stable value maximization given a pre-trained expressive policy and dataset (Sec. 3; Eq. (1)), aligning well with practical robotic fine-tuning setups.
3. This grounding strengthens technical soundness and real-world relevance.

Elegant decomposition of learning objectives:
1. The split between base (IL-trained) and edit (Q-optimized) policies isolates instability sources and allows targeted optimization (Sec. 4.1; Eq. (2); Fig. 2).
2. The OTF mechanism (Sec. 4.2; Eq. (3)) performs implicit value maximization without direct gradient propagation through the expressive base, offering conceptual simplicity and implementation clarity.
3. This design contributes to both training stability and policy generality, as EXPO is agnostic to the expressive policy parameterization (Sec. 4.4).

Comprehensive and reproducible experimental validation:
1. Twelve challenging continuous-control tasks across four domains (Sec. 5.1; Fig. 3–4) provide strong empirical grounding.
2. Baselines include IDQL, Cal-QL, RLPD, and DAC—covering both expressive and nonexpressive methods (Sec. 5.2; App. C).
3. EXPO consistently achieves higher sample efficiency and avoids offline-to-online degradation (Fig. 4; Sec. 5.4), directly supporting the main claim.
4. Reproducibility is strengthened by hyperparameter tables (App. B; Table 1), dataset details (App. B; Table 2), and explicit evaluation protocols (App. B, p. 15).

Insightful ablations and diagnostics: 
1. Ablations quantify contributions of on-the-fly TD backup (Fig. 5), edit policy (Fig. 6), and entropy backup (Sec. 5.5; Fig. 7), illustrating component necessity and interpretability.
2. Correlation between offline dataset quality and fine-tuning performance (Fig. 7) provides useful empirical evidence for practitioners.
3. The study in Appendix A demonstrates robustness even without retaining offline data, reinforcing adaptability.

Clarity and connection to prior work:
1. Related works are extensive and well-situated (Sec. 2; p. 2–4).
2. The manuscript differentiates from diffusion-based fine-tuning (DAC, QSM) and residual-policy methods (Ankile et al. 2024), improving conceptual clarity.
3. Figures 1–2 concisely visualize the EXPO workflow, enhancing accessibility.

### Weaknesses
Limited theoretical justification for stability and convergence:
1. The claim that separating imitation and Q-optimization yields “stable value maximization” (Sec. 4; 6) is intuitively argued but lacks formal analysis or empirical stability metrics (e.g., gradient variance, TD error oscillations).
2. No theoretical bound or convergence guarantee is provided for the coupled base-edit policy updates (Eq. (2)–(5)).
3. The assumption that local Gaussian edits suffice to capture high-value modes is unproven; sensitivity to β (edit radius) is only empirically illustrated (App. B; Table 1).
4. This limits the technical depth relative to recent theoretical RL papers.

Computational efficiency and scaling not analyzed:
1. The OTF policy requires sampling and Q-evaluation for multiple action candidates per step (Eq. (3); Sec. 4.2), which increases computational cost; yet wall-clock or FLOP comparisons are not reported (No direct evidence found in the manuscript).
2. The discussion acknowledges sampling overhead (Sec. 6 p. 9) but does not quantify trade-offs or propose scaling solutions.
3. Without resource profiling, claims of “sample efficiency” may not directly translate to runtime efficiency.

Mathematical clarity and notation consistency:
1. The use of “β” for both edit magnitude and softmax scaling (Sec. 4.1 vs. 4.3) can cause confusion; explicit notation conventions are missing.
2. The edit constraint “â ∈ [−β, β]” (Eq. (1); Sec. 4.1) lacks formal definition for vectorvalued actions—element-wise, norm-bounded, or learned scaling?
3. Entropy backup formulations (Eq. (4)–(5)) omit proofs of equivalence to standard SAC objectives under mixed sampling distributions.
These minor ambiguities hinder full reproducibility.

Empirical scope and generalization:
1. Experiments focus exclusively on robotics control benchmarks; no results on nonrobotic or discrete-action tasks (Sec. 5).
2. Although the method is claimed to be “agnostic to policy parameterization” (Sec. 6), no experiments validate this with non-diffusion expressive models.
3. Generalization to large observation spaces (e.g., visual RL) or transfer scenarios remains unexplored.

Assumption dependence and hyperparameter sensitivity
1. Performance depends heavily on the presence of “reasonable priors” via offline datasets (Sec. 6; p. 9), but quantitative thresholds for dataset adequacy are not defined.
2. The β and N hyperparameters control exploration and action sampling; only limited tuning guidelines are provided (App. B).
3. Lack of systematic sensitivity analysis may limit practical deployment.

### Questions
Provide theoretical and empirical stability analyses:
1. Include diagnostics of gradient norms, TD-error variance, or critic loss oscillations comparing EXPO to direct value-maximization baselines (Sec. 4.1–4.2).
2. Formalize stability claims via convergence theorems or bounded-update assumptions for coupled base/edit policies (No direct evidence found in the manuscript).
3. Report quantitative ablation of β radius vs. policy divergence to support the “local edit” assumption.

Report computational cost and scaling results:
1. Add per-step runtime, GPU hours, or FLOPs for EXPO vs. IDQL/RLPD across domains (Sec. 5.1–5.2).
2. Analyze the impact of N (number of action samples) on both performance and compute (App. B Table 1).
3. Explore efficient approximations (e.g., importance-weighted sampling or learned Qsampler) to mitigate the OTF sampling cost (Sec. 6).

Clarify mathematical notation and definitions:
1. Introduce a global notation table clarifying β (edit radius vs. temperature), πOTFentropy definitions, and vector norm conventions (Sec. 4).
2. Explicitly specify whether constraints are applied element-wise or via ℓ₂-norm bounds (Eq. (1); Sec. 4.1).
3. Expand Sec. 4.3 to show derivation of Eq. (5) from SAC objective, ensuring reproducibility.

Broaden evaluation and generalization claims:
1. Include at least one non-robotic or discrete-action task (e.g., Atari or tabular MDP) to substantiate generality beyond continuous control (Sec. 5).
2. Demonstrate compatibility with another expressive policy (e.g., flow-matching or autoregressive transformer) to validate policy-agnostic claims.
3. Provide limited visual or textual policy diagnostics (e.g., action entropy or diversity plots) for interpretability.

Quantify assumption and hyperparameter robustness:
1. Define empirical metrics for “reasonable prior” dataset quality (e.g., imitation policy success rate threshold) and report how EXPO behaves below it (Sec. 5.5; Fig. 7).
2. Conduct systematic sensitivity analysis on β and N, plotting performance variance vs. hyperparameter scaling (App. B Table 1).
3. Offer practical default ranges and scaling rules-of-thumb for new user

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Expressive Policy Optimization (EXPO), an approach for fine-tuning large pre-trained policies (like diffusion models) using a lightweight "edit policy" layered on top of the base policy. The key idea is to freeze or avoid altering the complex base policy and instead train a small edit policy that makes local adjustments to the base policy’s output actions to maximize rewards. The edit policy is updated with a standard policy-gradient algorithm augmented with entropy regularization (similar to SAC) to encourage exploration, while a critic (Q-function) is learned via a typical off-policy TD loss. Because the base policy is an expressive model (e.g. a diffusion model) for which computing entropy or direct gradients is difficult, the authors approximate the entropy term by sampling multiple actions from the base policy and fitting a simpler surrogate distribution for the edit policy’s updates (adding some engineering overhead but keeping the base policy’s parameters fixed). Overall, this technique defers the reward optimization to the constrained edit policy rather than directly modifying the pre-trained expressive policy, thereby avoiding the unstable gradients that would result from naive end-to-end fine-tuning of the large model. Empirically, EXPO is evaluated in an offline-to-online reinforcement learning setting (starting from offline pre-training and then fine-tuning online), and it shows significantly improved policy performance and stability.

### Strengths
Decoupling the reward optimization from the large expressive policy by introducing a small edit Gaussian policy is a practical idea that mitigates unstable gradients when fine-tuning complex policies. By avoiding direct backpropagation through the diffusion-based base policy and instead using a separate editable layer, the approach maintains stability in training. This design helps prevent the large pre-trained model from diverging due to high-variance gradient updates.

### Weaknesses
The core contribution of EXPO feels incremental, essentially adding a small trainable "edit head" on top of a fixed base policy and applying well-known fine-tuning strategies. Similar architectures have appeared in prior work – for example, learning a residual policy to refine the actions of a pre-trained policy is not a new concept. The authors combine standard elements (behavior cloning for the base policy, an actor-critic with entropy regularization for the edit policy, Q-learning with TD loss, etc.), so the method comes across as a straightforward amalgamation of known techniques rather than a fundamentally new algorithm or theoretical insight.

Given that the idea is not new, a deeper understanding of the idea in diffusion RL would be useful. The authors show that it is difficult to compute entropy in the diffusion setting and propose a solution. Yet this is also new a new trick to me. Besides this, the proposed idea mainly relies on intuitive arguments without rigorous verification. As a result, the contribution may be viewed as more of an engineering solution than a novel research insight.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Expressive Policy Optimization (EXPO), an online RL algorithm that enables sample-efficient training and finetuning of expressive policies such as diffusion policies. Instead of directly updating the expressive policy to maximize the Q-value, the authors propose to train a separate Gaussian edit policy with RL, while the expressive policy itself is trained via stable imitation learning. In addition, best-of-n sampling is used in both action sampling and TD backup for faster learning. Extensive experiments demonstrate strong performance in online training and finetuning with a given offline dataset.

### Strengths
1. The proposed method is novel to the field, and the analysis is well-supported. By using a lightweight Gaussian policy for policy improvement and using imitation learning to train the expressive policy, the method obtains a good balance between policy expressiveness and training efficiency. 
2. The experiment results are solid and well analyzed. The ablation studies focus on a few of the most important components such as on-the-fly policy extraction and action edits, which demonstrate the effect of the proposed contributions well.
3. The paper is well-written and easy to follow.

### Weaknesses
1. The experiments only cover online reinforcement learning with a given offline dataset, while no experiments are conducted in a purely online RL setting without a pre-collected dataset.
2. The experiment does not include several strong online diffusion-policy baselines, such as DACER [1], DIPO [2], and DPMD [3], which have been shown to perform better than or comparably to QSM on various tasks.

[1] Wang Y, Wang L, Jiang Y, et al. Diffusion actor-critic with entropy regulator[J]. Advances in Neural Information Processing Systems, 2024, 37: 54183-54204.

[2] Yang L, Huang Z, Lei F, et al. Policy representation via diffusion probability model for reinforcement learning[J]. arXiv preprint arXiv:2305.13122, 2023.

[3] Ma H, Chen T, Wang K, et al. Efficient Online Reinforcement Learning for Diffusion Policy[C]//Forty-second International Conference on Machine Learning.

### Questions
1. Instead of scaling the edit action to lie within the range $[-\beta, \beta]$, would a more natural approach such as enforcing a KL-divergence between the edited policy and the base policy also be applicable for keeping the edited action close to the original action samples?
2. How does the proposed method perform in purely online scenario without a given offline dataset?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper considers a scenario where one pre-trains an expressive policy on offline data via imitation learning and would like to fine tune it online. Instead of directly modifying the expressive policy—which can be rather difficulty due to architectural complexity and poor gradient propagation—they learn a Gaussian "edit" policy which is optimized to produce action residuals that better maximize value over the base, expressive policy. The final policy, denoted the on-the-fly policy, then samples the edit policy multiple times, and chooses the edit which produces the largest action-value. They then update the action-values with an approximate Q-learning update, where the max operation of the bootstrap target is approximated by the aforementioned on-the-fly procedure. Across a wide variety of domains, they evaluate their approach (EXPO) empirically in both online and offline-to-online setups.

### Strengths
* The use of an edit policy is elegant in that the approach is agnostic to the form of the base policy (e.g., compatible with non-parametric policies).

* The empirical performance seemed relatively consistent, both in terms of variability and how it faired relative to baseline methods.

* The paper performed a variety of ablations to justify the role of various choices in their algorithm (e.g., whether or not to re-run the on-the-fly policy for the TD target, the role of the edit policy, etc.)

### Weaknesses
* The evaluation only considered 3 seeds. This is reconciled by the breadth of environments and the results from taking it all together, but can limit the significance of individual plots. Further, there doesn't seem to be too much variability in EXPO's performance, perhaps due to an already really good base expressive policy?

* The appendix states that it is presenting the max and min, which measures variability and not statistical confidence. Providing a measure of confidence is crucial for making claims about performance differences between methods here.

### Questions
* In the ablation over action edits, the performance in "square" seemed better than many of the baselines. Is this suggesting that the offline-to-online procedure of the baselines is actively making the base policy worse here?

* Can the authors comment on the computational complexity of EXPO relative to the baselines? Running the OTF policy to compute the TD target for every sample in a mini-batch sounds considerably expensive if the action-values have to compute a forward pass for every sampled edit.

* Have the authors considered alternate forms of the OTF policy? e.g., directly performing gradient-ascent in Q(s,.) starting from the base policy's action or base policy + mean edit, or conditional cross-entropy (Lim et al., 2018), etc. There have been various proposed methods in the literature for performing this approximate max operation for continuous-action Q-learning, that—given the interpretations/explanations for the benefit of the OTF policy—I'd like to hear whether there are any thoughts or intuitions around how this might perform within EXPO?

* In some of the plots (e.g., all throughout Figure 3), the curve can be completely outside of the shaded regions. This seems like a bug, if the shaded regions supposedly represent max and min—what's going on here?

### Soundness
3

### Presentation
3

### Contribution
3
