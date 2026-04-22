# Pretrain Value, Not Reward: Decoupled Value Policy Optimization

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 2, 6

## Abstract
In this paper, we explore how directly pretraining a value model simplifies and stabilizes reinforcement learning from human feedback (RLHF). 
In reinforcement learning, value estimation is the key to policy optimization, distinct from reward supervision. 
The value function predicts the \emph{return-to-go} of a partial answer, that is, how promising the partial answer is if it were continued to completion. 
In RLHF, however, the standard pipeline first pretrains a reward model and then learns a value function online, even though no new reward signals are available once preference data is collected. 
This makes critic learning redundant, as the process of training a reward model and then deriving a value model is informationally equivalent to directly pretraining a value model.
Importantly, this requires no additional supervision, and our value model is trained on exactly the same data used for reward modeling. 
Building on this insight, we introduce \emph{Decoupled Value Policy Optimization} (DVPO), a framework that pretrains a \emph{Global Value Model} (GVM) offline and freezes it as a universal critic for policy learning. 
The GVM provides stable, fine-grained credit assignment without critic drift or trajectory sampling. 
Experiments across MT-Bench, Alpaca-Eval, and Arena-Hard demonstrate that DVPO matches or surpasses state-of-the-art RLHF methods. 
These results highlight RLHF can be reframed as policy-only optimization guided by a single pretrained value model. The implementation code for our method is available in \url{https://github.com/microsoft/DKI_LLM/tree/main/dvpo}

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents Decoupled Value Policy Optimization, a framework for reinforcement learning from human feedback that replaces the typical online-learned critic or reward model with a single, offline-pretrained global value model. The central idea is that pretraining a value model on the same dataset used for reward modeling is informationally equivalent to the joint reward-critic learning process, but with advantages in stability and efficiency. DVPO demonstrates strong empirical results on standard RLHF benchmarks (MT-Bench, Arena-Hard, Alpaca-Eval), surpassing or matching state-of-the-art methods, while significantly reducing memory consumption and training time.

### Strengths
1. The paper is well-motivated and the equivalence between reward-then-critic pipelines and direct value pretraining is well-argued and convincingly formalized, highlighting under-recognized redundancy in the RLHF paradigm.

2. The framework is particularly interesting to the community as DVPO simplifies RLHF engineering by eliminating online critic training, reducing GPU memory use (40%) and time (35%), thus enabling larger models or faster iteration with fewer resources.

3. Experiments, ablations, and case studies clearly show that DVPO attains or improves upon baseline methods in terms of various metrics.

### Weaknesses
1. The framework assumes the offline preference or reward dataset is broad enough for effective generalization.

2. In scenarios where additional human or environmental feedback can be injected mid-training, DVPO might not benefit from it.

### Questions
1. What mechanisms exist to detect or mitigate GVM misgeneralization due to dataset bias or coverage gaps?

2. Can the model identify when its value predictions become unreliable or out-of-distribution?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel approach to improve PPO-based RLHF by avoiding explicit reward modeling and instead learning a human preference-consistent critic/value model. The authors transform the traditional two-stage RLHF pipeline into an offline critic training plus online value optimization framework, termed DVPO. Through theoretical analysis, the paper establishes an equivalence relationship between reward models and value models under fixed feedback conditions, providing theoretical foundations for training the Global Value Model (GVM) and deriving convergence guarantees for DVPO. Experiments are conducted on the UltraFeedback dataset using models ranging from 3B to 8B parameters, both with and without instruction tuning. Results demonstrate DVPO's effectiveness across settings on evaluation benchmarks including MT-Bench, AlpacaEval 2.0, and Arena-Hard.

### Strengths
1. **Clear presentation and theoretical rigor**: The paper is well-structured with clear insights in the introduction, particularly regarding the equivalence between training reward models and critic models based on fixed feedback. The visualization effectively depicts algorithmic differences between PPO and DVPO, and the theoretical analysis is well-organized.
2. **Novel algorithmic contribution**: The proposed alignment algorithm offers a compelling alternative to the conventional two-stage RLHF pipeline (reward modeling + preference optimization), contributing a simplified RLHF framework that effectively reduces variance in online critic learning and improves training stability.
3. **Comprehensive empirical evaluation**: The experimental studies demonstrate thoroughness in base model selection, baseline method comparisons, and evaluation benchmark choices.

### Weaknesses
1. **Unclear mechanism of performance gains**: The underlying reasons for DVPO outperforming baselines remain insufficiently explained. The value model trained on pre-collected preference feedback effectively learns $V^{\pi_B}$ (assuming responses are collected from some behavior policy $\pi_B$), which contradicts the paper's claim that this value model is a "global" one (definitely not $V^*$). Additionally, the paper attributes superiority over sentence-level reward-based methods to GVM providing more fine-grained feedback. However, since human preferences are provided at the sentence level, the results and analysis do not adequately support this claim.
2. **Non-standard hyperparameter settings**: The experimental setup uses a batch size of 8, whereas current standard practice in the field uses batch sizes of 32 to 128. The KL coefficient is crucial for algorithm performance in both DVPO and baseline methods (including PPO and GRPO), and providing the search range would enhance experimental credibility.
3. **Insufficient justification for explicit value modeling**: The significance of explicitly fitting a value model needs stronger empirical backing. Given that GRPO and RLOO use Monte Carlo-based methods to estimate values, the authors should provide deeper insights into how separately training a value model enables DVPO to outperform these baselines. Furthermore, the comparison in Table 6 is unfair to GRPO regarding response generation time—Monte Carlo-based methods use an n-shot sampling strategy, making generation time equivalent to DVPO given the same training batch size.

### Questions
1. Have you considered incorporating Bradley-Terry reward modeling in the pre-training of value models? This is the most standard approach to reward modeling, as human feedback typically comes in pairwise comparisons. Simply assigning -1 and +1 rewards does not reflect the actual reward modeling procedure.

### Soundness
2

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
The paper proposes Decoupled Value Policy Optimization (DVPO) for RLHF: instead of the standard reward-model → online critic → policy pipeline, it pretrains a single Global Value Model (GVM) on the same offline preference data and freezes it as a universal critic to guide policy learning. The problem being solved is the instability and compute overhead of actor–critic RLHF (critic drift, high variance from sequence-level rewards, and the need to run multiple models in training).

Key claims:
(Theoretical) With fixed feedback (no new rewards), reward-model pretraining + value estimation is informationally equivalent to direct value pretraining; policy gradients differ by at most a bounded term κ that vanishes as approximation errors go to 0 (Lemma 3.1). A corollary states DVPO preserves PPO-style local convergence under KL regularization.
(Algorithmic) DVPO eliminates online critic training, using a frozen, token-level value signal (GVM) for advantages, thereby avoiding critic drift and reducing variance while retaining on-policy updates.
(Empirical) On MT-Bench, Arena-Hard, and Alpaca-Eval, DVPO matches or surpasses PPO/GRPO/ReMax/DPO across Llama-3B/8B and Mistral-7B settings (Tables 1–2), and shows improved compute efficiency (≈40–50% lower memory, ≈35% faster training in the base setting; Table 7).

### Strengths
Clear, impactful simplification: Recasting RLHF as policy-only optimization with a pretrained value is elegant and practically meaningful.
Solid theory–practice bridge: The equivalence lemma and convergence corollary directly justify the algorithmic design.
Consistent empirical gains & breadth: Improvements across models/benchmarks in both base and instruction settings; strong compute-efficiency results.
Token-level credit & interpretability: Concrete examples demonstrate fine-grained attributions absent in reward-only methods.
Reusability/generalization: A policy-agnostic critic simplifies reuse across models and appears robust vs. actor–critic critics on distribution shift.

### Weaknesses
Fixed-feedback assumption (no new rewards). DVPO’s theory and setup hinge on no additional reward during training; if limited online human feedback arrives, the frozen GVM cannot adapt.
Q: If limited online feedback becomes available, could you support periodic GVM refresh (semi-online DVPO) while retaining the stability guarantee, and what parts of Lemma 3.1 / the corollary would need to change?

Scope of evaluations. Benchmarks are mainstream chat-style; tougher code/maths/long-horizon tasks may stress token-value fidelity.
Q: Any results (or planned) on reasoning-heavy suites to test structure-aware value estimation?

Stability evidence presentation. Paper claims reduced critic drift and smoother training but lacks side-by-side variance/curve plots quantifying it.
Q: Can you add curves (PPO vs DVPO) and a critic-policy disagreement metric (e.g., advantage-distribution KL) to concretely show drift reduction?

### Questions
seen in the weakness section. 

Table 1 (Base setting). Please include #prompts and #rollouts per prompt used during RL for each method (PPO/GRPO/ReMax/DVPO), and whether generation lengths and KL targets were matched across methods (important for fairness).

Compute tables (Table 6 vs Table 7). The narrative states ~40% memory and ~35% time reductions, while Table 7 gives concrete GB/seconds for specific setups. Please reconcile the headline percentages with the numbers in Table 7

### Soundness
3

### Presentation
3

### Contribution
3
