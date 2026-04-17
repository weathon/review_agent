# CoFact: Conformal Factuality Guarantees for Language Models under Covariate Shift

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Large Language Models (LLMs) excel in natural language processing (NLP) tasks but often generate false or misleading information, known as hallucinations, raising reliability concerns in high-stakes applications. To provide statistical guarantees on the factuality of LLM outputs, conformal prediction based techniques have been proposed.  Despite their strong theoretical guarantees, they rely heavily on the exchangeability assumption between calibration and test data, which is frequently violated in real-world scenarios with dynamic covariate shifts.  To overcome this limitation, we introduce \textbf{CoFact}, a conformal prediction framework that uses online density ratio estimation to adaptively reweigh calibration data, ensuring alignment with evolving test distributions.  With this approach, CoFact bypasses the exchangeability requirement and provides robust factuality guarantees under non-stationary conditions. To theoretically justify CoFact, we establish an upper bound on the gap between the actual hallucination rate and the target level $\alpha$, demonstrating that the bound asymptotically approaches zero as the number of rounds and calibration samples increase.
Empirically, CoFact is evaluated on MedLFQA, WikiData, and the newly introduced \textbf{WildChat+} dataset, which captures real-world covariate shifts through user-generated prompts. Results demonstrate that CoFact consistently outperforms existing methods, maintaining reliability even under dynamic conditions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces CoFact, a novel conformal prediction framework designed to ensure factuality guarantees for large language models (LLMs) under dynamic, real-world distribution shifts. Specifically, CoFact incorporates online density ratio estimation (DRE), enabling adaptive reweighting of calibration data to align with the changing test distribution. CoFact is validated through a theoretical analysis and empirical experiments.

### Strengths
1. The paper addresses a critical limitation of existing conformal prediction methods by introducing online density ratio estimation, which allows for adaptation to dynamic and non-stationary distributions.

2. The authors provide a solid mathematical foundation for CoFact by establishing a theoretical upper bound on the hallucination rate under shifting distributions.

3. A new dataset, WildChat+, is proposed to evaluate the approach, featuring real-world user-generated prompts that effectively capture distribution shifts.

4. CoFact is rigorously evaluated across multiple experimental settings, demonstrating its robustness and effectiveness.

### Weaknesses
1. The writing in the paper is difficult to follow, as many key terms are introduced without proper explanation. For instance, the concept of calibration is presented without clarifying its meaning or role within the framework, making it harder for readers to grasp its significance.

2. The framework heavily depends on accurate online density ratio estimation, which can be computationally intensive and challenging to implement efficiently, particularly for high-dimensional or complex data distributions.

3. While the inclusion of WildChat+ is a notable contribution, the paper could benefit from exploring additional domain-specific real-world applications, such as legal, financial, or healthcare contexts, to further demonstrate its practical impact in high-stake tasks.

4. CoFact's emphasis on factuality overlooks other crucial response qualities, such as informativeness. For example, the framework might encourage the model to produce overly short or simplistic responses that sacrifice depth or detail in order to meet factuality requirements. This potential trade-off warrants further investigation.

5. The paper lacks an extensive discussion on distribution shifts, such as distinguishing between in-distribution and out-of-distribution user prompts. Providing examples or a detailed analysis of these distinctions would help clarify how the framework handles varying prompt distributions.

### Questions
Including case studies and error analysis can significantly enhance the clarity and impact of the paper. 

1. Case studies can illustrate how the proposed method performs under distribution shifts, offering concrete examples of its effectiveness. 

2. Meanwhile, an in-depth error analysis can help readers understand why and how CoFact outperforms previous approaches, shedding light on its strengths and limitations in handling challenging real-world scenarios.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the challenge of maintaining the factuality of large language model (LLM) outputs in real-world scenarios where the distribution of user prompts evolves. The authors correctly point out that existing conformal prediction (CP) methods for providing factuality guarantees fall short in this setting. This is because they depend on the exchangeability assumption, which holds when the calibration and test data come from the same distribution. However, this assumption no longer holds when the prompt distribution changes.

To address this limitation, the paper proposes CoFact. This is a new conformal prediction framework designed to maintain factual reliability under distribution shift. The key idea behind CoFact is to employ online density ratio estimation to adaptively reweight the static calibration data. This way, it aligns with the evolving test distribution at each time step. This enables the computation of an adaptive conformal threshold that effectively filters out hallucinated claims. Moreover, the system no longer relies on the exchangeability assumption or the unrealistic requirement of having ground-truth labels for test instances.

### Strengths
This paper addresses the problem of LLM factuality guarantees under the realistic condition of distribution shift. Moreover, the experimental validation is comprehensive. A key strength is the inclusion of the new challenging real-world benchmark (WildChat+). The results show that CoFact outperforms baselines and achieves its stated goal.

### Weaknesses
The CoFact methodology relies on an online ensemble framework where each expert is updated using an Online Newton Step (ONS). I wonder if this method is computationally more expensive. If it is more costly, this is a potential limitation for real-world, low-latency deployment.

### Questions
The CoFact methodology relies on an online ensemble framework where each expert is updated using an Online Newton Step (ONS). I wonder if this method is computationally more expensive. If it is more costly, this is a potential limitation for real-world, low-latency deployment.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
To theoretically control the rate of hallucination of large language models (LLMs) for their trustworthy use on safety-critical applications under dynamic environments where the test distribution may be different from the calibration data for training, authors propose the online confomal prediction method using an online density estimation technique.

Specifically, they assume that the test data distribution continuously changes over time, not in a too extreme manner (Assumption 1).

Besides, for the definition of hallucination of the generated answer on a given prompt, following Mohri et al. (2024), authors define the term as whether a filtered answer contain any hallucinated facts.

In terms of the technical contribution when compared to existing works on online conformal prediction under the distribution shift scenario, authors assume more challenging problem set-ups.

(1) Specifically, the authors assume continuous distribution shift scenario on the test data distribution, unlike a single distribution shift scenario (Tibshirani et al., 2019).
(2) Furthermore, they assume the correctness labels on the test data (i.e., whether a hallucinated sub-claim is contained is the filtered sub-claims) remain unrevealed even after the online evaluation.

This is a more challenging scenario compared to existing online conformal prediction methods (Gibbs and Candes, 2021; Gibbs and Candes, 2024; Areces et al., 2025) in which correct labels are revealed, enabling them to be used for training in subsequent time steps. 

In addition to the theoretical guarantee on the upper bound of the gap between the target coverage level and the average hallucination rate, empirical results on existing and newly proposed benchmarks show that CoFact controls the hallucination rate to the desired degree.

### Strengths
- It is a well-written paper which is easy-to-follow.

- They tackle the online conformal prediction problem for the theoretical control of hallucination of LLMs for its trustworthy use in safety-critical applications, which is one of primary interests from LLM users these days. 
Specifically, they propose the method which is valid under the more challenging set-up compared to existing works, which most resembles the dynamic real world problems.

### Weaknesses
[Weakness 1] Generalizability to online batch setting: I have understood that you are assuming a problem set-up, where single sample is provided for each time step for simplicity (Line 185-186). Then, how the Eq. (8) would look like if you consider a general setting, where a batch of samples may be provided for each time step? If not, the proposed threshold estimator may not be used in an online batch learning set-up. 

[Weakness 2] While existing online conformal prediction methods require additional information in terms of training which is not possible in the current problem set-up, it would be more informative to provide results of these methods equipped with the necessary information for training, since baselines in the Experiment section all assume i.i.d. data generating process under the batch learning set-up. As an concrete example, you may consider a problem set-up with (1) a single distribution shift scenario as Tibshirani et al. (2019) and (2) an accessibility to the ground truth labels in an online manner. While existing methods are expected to show more superior performance comapred to CoFact since they utilize additional information in terms of online threshold selection, I think it would be much more informative than just comparing with baselines assuming i.i.d. data generating process under the batch learning set-up.

### Questions
[Question 1] Shouldn't the "T" in Eq. (14) be substituted to "t"? Additionally, the term \theta_t^\ast is not formally defined.

[Question 2] How the Eq. (8) would look like if you consider a general setting, where a batch of samples may be provided for each time step? If you can propose one, does it also enjoy the theoretical guarantee that Eq. (8) has? Please refer to [Weakness 1] in Weaknesses Section for detail.

[Question 3] Could you compare CoFact with existing online conformal prediction methods under the scenario where existing methods assume? Please refer to [Weakness 2] in Weaknesses Section for detail.

[Queston 4]  The followings are typos to be addressed.

(Line 35) Despite of => Despite

(Line 77) CoFact bypass => CoFact bypasses

(Line 106-107) The following expression seems awkward to me: "... that transforms the outputs of a black-box predictor into prediction sets..."

(Line 115-118) \alpha => \beta or \beta => \alpha

(Line 314-315) \leq => \geq

(Line 269, Line 328) The same divergence function \psi is defined differently.

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
This paper addresses the critical challenge of providing statistical guarantees on the factuality of Large Language Model (LLM) outputs under distribution shift. The authors propose CoFact, a conformal prediction framework that employs online density ratio estimation (DRE) to adaptively reweigh calibration data, thereby maintaining factuality guarantees even when the exchangeability assumption is violated.  key contributions include:
A novel framework combining conformal prediction with online DRE to handle continuous distribution shifts; Theoretical analysis establishing an upper bound on the gap between actual and target hallucination rates; WildChat+, a new dataset capturing real-world distribution shifts; Empirical validation on MedLFQA, WikiData, and WildChat+ demonstrating superior performance over baseline methods

### Strengths
The paper tackles a significant limitation of existing conformal prediction methods for LLMs, they rely on the exchangeability assumption, which is frequently violated in real-world applications. The integration of online DRE with conformal prediction is creative and well-motivated. The use of an ensemble of experts with geometric lifetimes to track evolving distributions is technically sound. THeoretical part is solid: Theorem 1 provides a rigorous bound showing that the hallucination rate gap converges to zero as O(max{T^{-2/3}V_T^{2/3}, T^{-1/2}} + 1/n). It also has detailed experiments on the evaluation in simulated shifts (4 types) and real-world shifts (WildChat+).

### Weaknesses
Assumption 1 requires that the conditional distribution P(W|Z) remains unchanged while only the marginal P(Z) shifts. This is quite strong and may not hold in many real scenarios (eg, if model quality degrades over time or if certain types of prompts systematically elicit more hallucinations). The paper only compares against SCP and CondCP. What about other methods for handling covariate shift in conformal prediction. Figure 2, Legend is difficult to read. The paper doesn't discuss how to set T in advance or what happens when the time horizon is unknown.

### Questions
How many calibration samples n are needed in practice to achieve reasonable performance? The method requires know about feature representations $\phi(z)$. How should these be chosen for different applications? Statistical significance testing would strengthen the claims on table 2, 3. There seems no ablation on key design choices (e.g., number of experts, expert lifetime schedule, choice of divergence function. How sensitive is performance to the feature representation?

### Soundness
4

### Presentation
3

### Contribution
3
