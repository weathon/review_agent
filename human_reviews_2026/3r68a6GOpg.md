# ALM-MTA: Front-Door Causal Multi-Touch Attribution Method for Creator-Ecosystem Optimization

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 4

## Abstract
Consumption‑Drives‑Production (CDP) on social platforms aims to deliver interpretable incentive signals for creator‑ecosystem building and resource utilization improvement, which strongly relies on attributions. In large-scale and complex recommendation system, the absence of accurate labels together with unobserved confounding renders backdoor adjustments alone insufficient for reliable attribution. To address these problems, we propose Adversarial Learning Mediator based Multi‑Touch-Attribution (ALM-MTA), an extensible causal framework that leverages front-door identification with an adversarially learned mediator: a proxy trained to distillate outcome information to strengthen causal pathway from treatment to outcome and eliminate shortcut leakage. Then, we introduce contrastive learning that conditions front door marginalization on high match consumption upload pairs for ensuring positivity in large treatment spaces. To assess causality from non‑RCT logs, we also incorporate a non‑personalized bucketed protocol, estimating grouped uplift and computing AUUC over treatment clusters. Finally, we evaluate ALM-MTA performance using a real-world recommendation system with 400 million DAU (daily active users) and 30 billion samples. ALM-MTA has increased DAU with 0.04% and 0.6% of the daily active creators, with unit exposure efficiency increased by 670%. On causal utility, ALM-MTA achieves higher grouped AUUC than the SOTA in every propensity bucket, with a maximum gain of 0.070. In terms of accuracy, ALM-MTA improves upload AUC by 40% compared to SOTA. These results demonstrate that front -door deconfounding with adversarial mediator learning provides accurate, personalized and operationally efficient attribution for creator ecosystem optimization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper abstracts the recommendation system into a multi-touch attribution problem. The authors argue that due to potential confounding factors, relying solely on backdoor adjustments is insufficient for reliable attribution. To achieve reliable attribution, the authors propose Adversarial Learning Mediator-based Multi-Touch Attribution (ALM-ATA). ALM-ATA leverages frontdoor principles and adversarial learning to refine outcome information and strengthen the causal path from treatment to outcome.

### Strengths
1. Using the front-door criterion to mitigate the impact of potential confounding factors
2. Designing an Area Under the Uplift Curve evaluation scheme based on Shapley Value sampling, providing a new method for evaluating attribution quality on observational data
3. Conducting experiments in real-world recommender systems

### Weaknesses
1. The authors need to further clarify their innovation. The authors claim to have solved the unobserved system-level confounding factors in the recommendation ecosystem and overcome a core limitation of previous attribution methods. However, in the field of debiased recommendation systems, several works have proposed solutions to potential confounding factors [1][2][3], including the application of the front-door criterion [2].
2. How do the authors distinguish between covariates X and potential confounders W in practice, and how do they ensure that the model does not regard X as a confounder? From Figure 2, we can see that X can also be regarded as a confounder and is observable.
3. The authors regard the observed results of Y as Y' and use it as a proxy variable for M to guide the generation of M. In my opinion, Y' is almost equivalent to Y, and there is a risk of data leakage. Although the authors claim to use adversarial learning to suppress data leakage and provide relevant experimental evidence in the "Ablation Studies and Evaluation" section, I still have concerns about this. I think the current experimental data is still the result of data leakage, but the degree of leakage has been suppressed.
4. In Section 4, the authors did not describe the specific implementation details, but only provided a formal description. For example, the authors introduce adversarial learning but don't provide a detailed implementation. Similarly, the paper doesn't explain how the mediating variable M is generated. This raises concerns about whether ALM-ATA can be replicated.

I am willing to increase my score if the author can address my doubts.

[1] Z Huang, H Yuxuan, D Cheng, et al. Multi-Cause Deconfounding for Recommender Systems with Latent Confounders[J]. Knowledge-Based Systems, 2025. https://doi.org/10.1016/j.knosys.2025.114345
[2] Xu S, Tan J, Heinecke S, et al. Deconfounded causal collaborative filtering[J]. ACM Transactions on Recommender Systems, 2023, 1(4): 1-25.
[3] Xu H, Xu Y, Li C, et al. Causal structure representation learning of unobserved confounders in latent space for recommendation[J]. ACM Transactions on Information Systems, 2025.

### Questions
See weakness

### Soundness
2

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
3

### Summary
=

The paper targets multi-touch attribution (MTA) in creator/content ecosystems where there exist **unobserved confounders** and the standard back-door/front-door criteria are hard to apply directly. The authors propose **ALM-MTA**, a framework that combines front-door–style causal reasoning with **proxy variables, adversarial learning, IPW, and contrastive learning** to estimate causal contributions of different touchpoints. The key idea is: the true mediator in a front-door graph is not observed, so the model builds a *noisy proxy* for it and uses **adversarial training** to strip out components tied to the unobserved confounder, thereby recovering (approximately) the causal path from exposure/treatment to outcome. The method is evaluated with a causal-style ranking metric (gAUUC) and also with **online A/B tests**, which show business gains, making the work practically relevant.

---

### 2. Strengths

#### 2.1 Conceptual and modeling contribution

* The paper addresses a realistic variant of **front-door identification with an unobserved mediator**: in production systems we often can’t see the mediator that carries the effect, but we can see a *correlated, noisy signal*. Turning that into a front-door–like procedure via adversarial learning is clever and novel in the MTA context.
* The adversarial branch is used to **force the main prediction network to drop information that helps predict the proxy** (which is assumed to be tied to the unobserved confounder). This is a reasonable deep-learning instantiation of “recovering the clean causal path.”

#### 2.2 End-to-end, deployment-oriented design

* The framework simultaneously considers **observed confounding** (handled via IPW), **unobserved confounding** (handled via adversarial proxy learning), **high-cardinality and sparse features** (mitigated via contrastive learning), and **training stability**. This kind of “full-stack” design is what real industrial MTA systems need, and is not always seen in academic MTA papers.
* The paper clearly recognizes practical issues in creator ecosystems: heterogeneous touchpoints, weak overlap across users/items, and very large action spaces.

#### 2.3 Evaluation and metrics

* The paper proposes **gAUUC** to better align model ranking with *causal* uplift rather than predictive accuracy. This is more appropriate than plain AUC in a causal attribution setting.
* Ablation studies show that removing the adversarial part or treating the problem as pure prediction leads to worse causal ranking, which supports the core claim.
* **Online A/B** results (on DAU/WAU/creator WAU) make the contribution significantly stronger: the method is not only theoretically motivated but also deployable.

---

### 3. Weaknesses and Limitations

#### 3.1 Complexity and reproducibility

* The training pipeline optimizes multiple objectives at once (main prediction, proxy branch, adversarial loss, IPW-related parts, contrastive loss). This makes the system **heavy and hard to reproduce** for teams without mature ML/causal infra.
* Because of this complexity, the method is somewhat **black-box**: it is not straightforward to tell which part of unobserved bias was actually removed.

#### 3.2 Strong assumptions

* The approach still relies—implicitly—on a **front-door–like identification story**: after conditioning on treatment and observed covariates, the unobserved confounder’s effect through the proxy/mediator is supposed to be “filterable.” In real platforms, this can be violated easily.
* The whole adversarial cleaning step hinges on the **quality of the proxy**: if the available behavioral/log signal is only weakly correlated with the true mediator, the causal benefit may shrink.
* The final attribution seems to assume **approximately additive contributions** across touchpoints, but in creator ecosystems **order and interaction effects** are common, so this assumption is somewhat fragile.

#### 3.3 Metric self-referentiality

* gAUUC, as defined, partially depends on model-based or constructed pseudo–ground truths. This can introduce a mild circularity (“we use a model to evaluate a model”). The paper acknowledges AUC is not ideal, but gAUUC is not yet a universally accepted gold standard.

---

Some suggestions

1. **Make the causal graph explicit.** Present a minimal DAG, list the front-door conditions, and pinpoint which one is relaxed and how the proxy + adversarial block recovers it. This will help readers from the causal community.
2. **Add proxy-quality sensitivity.** Vary the correlation between the proxy and the latent mediator, and show how performance (especially gAUUC and online KPIs) decays. This will make the method look more robust and honest.
3. **Discuss non-additive / order effects.** Even a short section on how to extend to interaction-aware aggregation (e.g. attention over touch sequences) would ease concerns about the additivity assumption.
4. **Provide a training recipe.** Since there are multiple losses, offer recommended weights, pretraining/warmup order, and common failure modes (overpowerful adversary, unstable IPW, etc.), so others can reimplement it.
5. **Relate gAUUC to business KPIs.** Show which buckets / segments show the strongest monotonic relationship between gAUUC lifts and real online lifts, to justify the metric choice.

### Strengths
see above

### Weaknesses
see above

### Questions
see above

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
5

### Summary
This paper presents a practically impactful and conceptually novel solution to causal multi-touch attribution in creator-driven recommendation systems. It addresses a critical gap where existing heuristic or back-door–based approaches fail under latent confounding, making attribution unreliable for optimizing creator ecosystems. The proposed ALM-MTA framework operationalizes a front-door causal design by introducing an adversarially learned proxy mediator, enabling identification of touchpoint-level causal effects even when the true mediator is unobserved. The method further integrates IPW debiasing and contrastive learning to ensure stability and overlap in large treatment spaces.

The contribution is notable for its theoretical grounding as well as its large-scale real-world impact, demonstrating measurable online gains on a 400M-DAU platform. This work stands out as one of the few that successfully bridges causal identification with production-level recommender optimization, making it relevant and valuable for both the causality and RS communities.

### Strengths
1. A major strength lies in the novel design of an adversarially learned proxy-mediated front-door architecture. The paper introduces a proxy variable $Y'$ along with an adversarial learning mechanism to approximate the unobservable mediator required for front-door identification. This is a clever and original solution that operationalizes causal mediation where the true mediator is latent, helping to mitigate outcome leakage while preserving the causal pathway. The integration of IPW for debiasing and contrastive learning to maintain overlap in high-cardinality treatment spaces reflects a comprehensive and well-engineered approach to real-world constraints.

2. The empirical evaluation is compelling. The model demonstrates consistent and substantial improvements over strong baselines in terms of AUC, log-loss, and gAUC. Particularly noteworthy is the inclusion of large-scale online A/B testing on a production system with 400M DAU and 30B training samples—showing statistically meaningful gains in creator activity, exposure efficiency, and platform health. Such deployment results significantly enhance the practical value and credibility of the work.

### Weaknesses
1. The use of the proxy variable $Y'$ as a surrogate for the mediator $M$ is insufficiently justified. The paper does not provide rigorous theoretical or empirical evidence establishing the validity of $Y'$ and it remains unclear whether $Y'$ meets the necessary proxy conditions for front-door identification. More importantly, the paper does not prove that $Y'$ does not introduce a new path to the outcome $Y$, which could bias the causal effect estimation.

2. the applicability of the front-door criterion is assumed rather than validated. The key identification assumption—that all causal influence of touchpoints on uploads must operate exclusively through the latent mediator $M$ appears overly strong and may not hold in real-world CDP environments. In practice, certain types of touchpoints can directly trigger uploads without being reflected in the proxy-mediated “inspiration” pathway captured by $Y'$. For example, other creators get more rewards / exposure today can directly motivate uploads through platform incentives rather than creative inspiration, or some simple notifications (e.g., “Your followers are waiting—post an update now!”) can directly activate upload behavior and bypass the inspiration pathway altogether.

### Questions
Based on the the weaknesses, I have the following questions:

1. Could the authors provide empirical or theoretical evidence that the front-door conditions truly hold in this CDP setting? Specifically, how do you justify the assumption that all causal influence of consumed touchpoints on uploads flows exclusively through the mediator $M$?

2. How do the authors account for touchpoints that may directly trigger uploads without going through inspiration (such as the examples I established in weakness 2)? Would such direct causal paths violate front-door identification?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes using front-door identification to handle latent confounders in recommendation systems and innovative adversarial proxy mediator design. This method reduces variance in uplift estimation by conditioning on the proxy. It is claimed that the method is designed to be scalable with real-world performances. The paper presented business results on DAU, daily active creators, and an upload AUC of 0.907 (40% increase on sota). The paper performs an analysis on the result, presenting a propensity-stratified grouped AUUC protocol using Shapley value sampling, and the algorithm is able to combine multiple sophisticated techniques with ablation studies.

### Strengths
- The paper describes a novel application of the front-door identification causal framework to handle latent confounders in recommender systems. 
- The methods are experimented on a real-world online recommendation system, showing significant improvement. 
-The paper combines multiple complex methods together and shows ablation on each factor.

### Weaknesses
- It is less clear in the writing how the proposed methods align, compare, or improve upon methods in prior art 
- Compared with prior methods, other than Table 2, it could be useful to compare on more than one dataset or task, and perform experiments to show how the proposed methods work better and the specific reason why (e.g., due to front-door ID, adversarial mediator design, or scalability. 
- Presentation: Figure 5 and Figure 6 are a bit hard to read 
- it's a bit unclear whether the production or real-world data is made public for future work to compare to this paper

### Questions
- For comparison on other datasets and tasks, could more experiments be validated or provided to compare the proposed method against prior art? 
- Could the presentation be clearer on the figures? 
- Is the real-world data used made public for research comparisons?

### Soundness
2

### Presentation
3

### Contribution
2
