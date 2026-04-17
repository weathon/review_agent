# TOGA: Trigger Optimization for Clean Data Ordering Backdoor Attack

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Recent work has shown that backdoors can be learned in neural networks purely through the malicious reordering of clean training data, without modifying labels or inputs. These data ordering attacks rely on gradient alignment, ordering clean samples to approximate the gradients of an adversarial task.
However, the effectiveness of such attacks depends greatly on the choice of the backdoor trigger, which determines how closely clean gradients align with the backdoor gradients. In this work, we introduce the first framework (TOGA - Trigger Optimization for Gradient Alignment) for optimizing triggers specifically for data ordering attacks. Our method significantly improves attack success rates by an average of 46\% over prior methods across benchmark datasets (CIFAR-10, CelebA, and ImageNet) and sensitive application domains (ISIC 2018 for dermatology and UCI Credit-g for credit scoring), without compromising clean performance. We further show that optimized triggers can be adapted to create subpopulation-specific backdoors, selectively targeting vulnerable subpopulations. Finally, we show our method is robust against purification and gradient-similarity defenses. Our findings reveal new security and fairness risks for high-stakes domains, underscoring the need for broader defenses against data ordering attacks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an approach for optimizing backdoor poisoning by data-ordering attack.
The attack aims to perturb the gradient of a data-batch during deep learning (back-propagation).

### Strengths
The authors propose an interesting optimization framework with several penalty terms indicating that they understand all of the issues in play.

### Weaknesses
First note that data-ordering attacks require the adversary to be an insider of the training process to affect the batch gradients used for deep learning and to have access to the complete training dataset (Eq. (3)). Though the authors argue that their attack requires less assumptions than previously proposed data-ordering attacks (see line 173-), their attack nevertheless requires much stronger assumptions than passive data-poisoning (esp. prior to the formation of the training dataset) as considered in most backdoor papers. Moreover, a benign training authority (if present) could simply randomly re-select from the training data to form batches to defeat any data-ordering attack. So, it's not clear to me how practically important such attacks are. Though it's interesting how data ordering could bias a model, prior papers have already established this.

The training of models described in 208 and optimizations such as (3) raise additional concerns regarding the adversary's work-factor relative to an obvious and simple defense (or to simple passive data-poisoning attack).

On line 191, I think the authors should point out that the ground truth class of x_i is _not_ y^{adv}, which is confused by writing y_i^{adv}=y^{adv}. 

The description of the method on p. 5-6 is pretty terse.

Lines 244, 259 and 260 say the same thing.
The fonts on the figures are too small.
:

### Questions
See above.

### Soundness
3

### Presentation
2

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
This paper introduces TOGA, a novel framework for optimizing backdoor triggers specifically tailored to clean data ordering attacks. Rather than modifying training data directly, TOGA identifies triggers whose adversarial gradient trajectories are well-aligned with those reachable by merely reordering clean training samples. The approach is empirically evaluated across diverse datasets (CIFAR-10, CelebA, ImageNet, ISIC 2018, UCI Credit-g), demonstrating higher attack success rates than prior baselines while maintaining clean performance. The framework is also extended to subpopulation-specific backdoors and tested against recent defenses.

### Strengths
+ Advancement over Prior Work. TOGA addresses a missing piece in data ordering attacks, by proposing a concrete, optimization-based trigger-selection framework. This idea is substantive, as previous data ordering techniques largely assumed manually chosen triggers, limiting attack effectiveness
+ Extensive Evaluation. The experimental results are comprehensive, spanning image and tabular domains, and including sensitive applications.
+ Extension to Subpopulation Attacks. The method is extended to design subpopulation-targeted triggers, providing a new threat axis toward vulnerable demographic groups.
+ Reproducibility. Supplied materials code is attached for reproducibility.

### Weaknesses
A substantive assessment of the weaknesses of the paper. Focus on constructive and actionable insights on how the work could improve towards its stated goals. Be specific, avoid generic remarks. For example, if you believe the contribution lacks novelty, provide references and an explanation as evidence; if you believe experiments are insufficient, explain why and exactly what is missing, etc.

+ Incremental Advancement Over Data Ordering Pipelines. While TOGA provides a principled mechanism for trigger optimization within data ordering attacks, the core pipeline remains largely based on Shumailov et al.’s gradient alignment framework [1]. The paper would benefit from a clearer articulation of conceptual novelty beyond trigger search, and ablations that isolate how much of the reported gains stem specifically from the proposed optimization rather than broader design choices or hyperparameter tuning.
+ Ambiguity and Potential Flaws in Mathematical Formulation. In Section 3.3, several aspects are under-specified or ambiguous:
	+ In the definition of $\mathcal{L}_{\text{match}}$, the variable $D$ appears in the normalization term but is not explicitly defined in the local context (e.g., batch size vs. dataset cardinality). Clarifying this would improve reproducibility and reader understanding.
	+ In the subpopulation-aware objective $\mathcal{L}_{\text{subpop}}$, the use of $z_i \cdot y^{\mathrm{adv}}$ as a target encoding is somewhat unconventional. A concise explanation of how this interacts with standard label representations (e.g., masking vs. mixing) would prevent potential misinterpretation.
	+ For the spillover penalty $\mathcal{L}_{\text{spillover}}$, penalizing the target logit directly rather than the full loss may not necessarily correlate with reduced misclassification risk. An ablation comparing loss-based versus logit-based regularization would strengthen confidence in this design choice.
+ Limited Analysis of Clean-Benign Tradeoffs. Although the paper reports a “<5%” drop in clean accuracy, a more granular analysis would be helpful. In particular, it is unclear whether certain fine-grained classes, minority attributes, or distributional subgroups are disproportionately affected. Rare catastrophic degradations could matter significantly in security-sensitive deployments.
+ Assumptions on Adversarial Capabilities. The threat model assumes an adversary with substantial visibility into training loss trajectories. While plausible in compromised supply-chain scenarios, a more explicit discussion of when such access is feasible would contextualize applicability. Additionally, comparison to alternative stealth vectors (e.g., weight manipulation under constrained visibility) would strengthen the realism argument.
+ Evaluation Model Specification and Training Details. The manuscript does not clearly specify the exact architectures and training hyperparameters used for each dataset.


[1] Ilia Shumailov, Zakhar Shumaylov, Dmitry Kazhdan, Yiren Zhao, Nicolas Papernot, Murat A. Erdogdu, and Ross Anderson. 2021. Manipulating SGD with data ordering attacks. In Proceedings of the 35th International Conference on Neural Information Processing Systems (NIPS '21)

### Questions
1.Ablation on Optimization Components

Could the authors provide ablations isolating contributions from (i) trigger optimization, (ii) spillover penalty, and (iii) subpopulation-aware loss? Without this, it is difficult to attribute gains to the proposed components.

2.Sensitivity to Optimizer Scheduling

Ordering attacks are known to depend on training dynamics. How sensitive is TOGA to changes in learning rate schedules or batch sizes? Do the gains persist under alternative training regimes?

3.Definition and Role of $D$ in $\mathcal{L}_{\mathrm{match}}$

Please clarify whether $D$ denotes batch size, dataset cardinality, or feature dimension. How does varying $D$ affect gradients?

4.Interpretation of $z_i \cdot y^{\mathrm{adv}}$

Is this equivalent to masking? Would one-hot mixtures or other encodings change the behavior? A short explanation or visualization would improve clarity.

5.Distributional Impact on Clean Accuracy

Do certain demographic groups or minority classes suffer higher degradation, especially in CelebA or tabular datasets? Aggregated accuracy may obscure subgroup harm.

6.Trigger Robustness Across Initialization Seeds

How consistent is trigger quality across random seeds? Are there significant variances in attack success rate?

7.Stealth Assumptions Under Practical Limitations

In real-world training pipelines, access to gradient information may be restricted. Can the authors quantify performance degradation if gradient signals are noisy or discretized?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses data ordering attacks, a recently explored type of backdoor threat where the adversary manipulates the sequence of training samples rather than their content or labels. The authors propose an optimization-based framework for trigger design under this restricted setting. The proposed algorithm aims to generate adaptive triggers that can selectively target specific subgroups of data, achieving effective backdoor injection even when only sample ordering is controllable. Experimental results demonstrate the attack’s potential impact and illustrate the selective behavior of optimized triggers.

### Strengths
1.The paper proposes an effective algorithm for optimizing triggers specifically under the data ordering attack setting. This contributes a valuable methodological perspective to a relatively new attack paradigm.
2.The optimized triggers can selectively affect particular data subgroups, enabling a more controllable and interpretable attack mechanism. This feature highlights the flexibility of the proposed optimization formulation.
3.The mathematical formulation of the trigger optimization  is clearly described, making the approach easy to follow.

### Weaknesses
1.Unclear problem motivation and missing analysis of prior methods:
The paper claims that existing trigger optimization methods fail under the ordering-attack constraint, emphasizing that manually chosen triggers may not achieve sufficient gradient alignment (as noted in Fig. 1). However, it does not explain why prior optimization-based trigger methods (e.g., Doan et al., 2021b; Saha et al., 2020) are ineffective when only the sample order can be modified. The technical difficulty of optimizing triggers in this restricted setting is therefore not well articulated. The authors should clarify this from a unified optimization viewpoint—why exactly optimization for ordering attacks is nontrivial, and provide theoretical or empirical evidence demonstrating where existing approaches fail. Simply contrasting against manually designed triggers risks overstating the contribution.
2.Lack of comparative evaluation with existing optimized triggers:
The experiments mainly compare with manually chosen triggers (Flag/Class) and a single Silent Killers trigger variant, but omit a fair end-to-end comparison against existing optimization-based trigger generation methods within the same “ordering-only” pipeline. Without this, it is unclear whether those methods truly fail or merely underperform under the new constraint. Such comparisons are crucial to substantiate the claimed novelty.
3.Limited experimental comprehensiveness:
The current experimental section lacks sufficient depth. The dataset coverage and ablation analyses are limited, and the empirical evidence does not fully establish the generality or robustness of the proposed algorithm.

### Questions
1.Can you provide either a theoretical argument or empirical demonstration showing why existing trigger optimization methods fail under the ordering-attack constraint?

2.Could you include direct comparisons by adapting prior optimization-based trigger methods to the same “ordering-only” setting?

### Soundness
2

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
This paper presents TOGA (Trigger Optimization for Gradient Alignment), a new framework for optimizing triggers in data ordering backdoor attacks—a stealthy class of attacks that reorder clean training data to implant backdoors without data or label modifications. The authors argue that prior works use manually chosen triggers that limits the attack success rate (ASR). TOGA formulates a trigger optimization objective that jointly minimizes gradient-matching loss, adversarial loss, and a penalty loss to align clean and adversarial gradients. The method significantly improves attack success rate (ASR) across benchmarks (CIFAR-10, CelebA, ImageNet, ISIC, Credit-g), achieving up to 99% ASR with < 5% benign accuracy drop on CIFAR10, compared to state-of-the-art baseline Flag 25% and , and can further craft subpopulation-specific backdoors. The paper also evaluates robustness against purification and gradient-based defenses.

### Strengths
1. Novel Threat Model: First to propose trigger optimization specifically for data ordering backdoors.
2. Solid Empirical Results: Significant attack success rate improvement on Extensive experiments on five diverse datasets including sensitive domains (healthcare + finance).

### Weaknesses
1. Heuristic Methodology: The paper doesn't clarify why using the cosine similarity as the penalty, why not L2 norm or any other regularizers. Furthermore, it doesn't show that the proposed methdo is a good proxy for the origianl bi-level optimization objective.
2. Many Hyperparameters might Cause Unstable Optimization: The loss functions contain more than 3 weighting hyperparameters $\lambda$
. However, the paper doesn't cleaify the effect of choosing hyperparameters. It might causes the optimization become sensitive the hyperparameter choosing.
3. Lack of Methodological Innovation: New threat model made with old optimization tricks.

### Questions
1. Can you elaborate how does the hyperparameters $\lambda$ affect the stealth and attack success rate? What's the hyperparameters used in the experiments?
2.  Can you show how does the loss function as equation (1) and (2) reduce to the bi-level optimization objective mentioned in 3.2? It's hard to connect 3 of them.

### Soundness
3

### Presentation
3

### Contribution
2
