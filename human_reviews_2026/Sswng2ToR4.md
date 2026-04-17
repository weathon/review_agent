# Downgrade to Upgrade: Optimizer Simplification Enhances Robustness in LLM Unlearning

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Large language model (LLM) unlearning aims to surgically remove the influence of undesired data or knowledge from an existing model while preserving its utility on unrelated tasks. This paradigm has shown promise in addressing privacy and safety concerns. However, recent findings reveal that unlearning effects are often *fragile*: post-unlearning manipulations such as weight quantization or fine-tuning can quickly neutralize the intended forgetting. Prior efforts to improve robustness primarily reformulate unlearning objectives by explicitly assuming the role of vulnerability sources.  In this work, we take a different perspective by investigating the role of the *optimizer*, independent of unlearning objectives and formulations, in shaping unlearning robustness. We show that the "*grade*" of the optimizer, defined by the level of information it exploits, ranging from zeroth-order (gradient-free) to first-order (gradient-based) to second-order (Hessian-based), is tightly linked to the resilience of unlearning. Surprisingly, we find that downgrading the optimizer, such as using zeroth-order methods or compressed-gradient variants (*e.g.,* gradient sign-based optimizers), often leads to stronger robustness. While these optimizers produce noisier and less precise updates, they encourage convergence to harder-to-disturb basins in the loss landscape, thereby resisting post-training perturbations. By connecting zeroth-order methods with randomized smoothing, we further highlight their natural advantage for robust unlearning.  Motivated by these insights, we propose a *hybrid optimizer* that combines first-order and zeroth-order updates, preserving unlearning efficacy while enhancing robustness. Extensive experiments on the MUSE and WMDP benchmarks, across multiple LLM unlearning algorithms, validate that our approach achieves more resilient forgetting without sacrificing unlearning quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates how optimizer design affects the robustness of LLM unlearning, introducing the concept of optimizer grade to compare high-order and downgraded optimizers. It finds that downgraded optimizers (e.g., signSGD, zeroth-order methods) lead models to flatter and more stable minima, improving post-unlearning robustness. A hybrid optimizer is further proposed, achieving superior resistance to relearning and quantization attacks across multiple benchmarks.

### Strengths
1. Introduces a novel perspective by analyzing optimizer grade as a key factor in unlearning robustness.

2. Provides strong empirical evidence across diverse benchmarks and unlearning methods.

3. Proposes a simple yet effective Hybrid FO–ZO optimizer with clear performance gains.

### Weaknesses
1. The paper explains the robustness gains mainly through intuition about flatter minima but lacks a deeper theoretical or quantitative analysis to support this claim.

2. All experiments are limited to the unlearning setup, so it remains unclear whether the proposed idea generalizes to other fine-tuning or robustness tasks.

3. The study does not provide details on the computational overhead of zeroth-order or hybrid optimization, leaving questions about practical scalability.

4. The Hybrid FO–ZO optimizer design appears largely empirical; the choice of alternation schedule is not well-motivated or systematically analyzed.

### Questions
See weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a new perspective on robust large language model (LLM) unlearning—the process of selectively erasing specific knowledge from models while maintaining their general capabilities. Prior studies have largely focused on modifying the unlearning objectives or introducing explicit robustness regularizers (e.g., sharpness-aware or meta-learning methods). In contrast, this work investigates the role of the optimizer itself in shaping the robustness of unlearning outcomes, asking whether the optimizer’s complexity—or “grade”—affects how resilient forgetting remains under post-unlearning perturbations such as relearning attacks and weight quantization.

### Strengths
The authors are obviously the experts in the field. The draft is well structured, insightful, and easy to read. Although the current draft has some minor limitations, I will definitely improve my score when they are solved. 

This paper Introduces “optimizer grade” as a new dimension for understanding unlearning dynamics, shifting focus from objective design to optimizer behavior.

This paper demonstrates that weaker or noisier optimizers (e.g., sign-based, ZO) can improve robustness. It is an impactful finding that challenges conventional assumptions about optimization fidelity.

The proposed FO–ZO hybrid method is straightforward to implement, and can be integrated with existing unlearning systems without redesigning objectives.

### Weaknesses
Some previous works, such as those combined with IRM and SAM, can be viewed as using higher order gradients, which are shown to be more robust to attacks. They somehow conflict the opinion of this paper, instead showing that lower order methods are better. Could the authors explain further about these different conclusions?

Due to the use of momentum, Adam is not strictly a first-order approach but between first and second orders. From my view, only vanilla SGD is first ordered. So, maybe SGD should also be adopted for comparison. 

A weird thing is that, the second order approaches are mentioned in the abstract, but not involved in the main context. A remember some papers have claimed the use of natural gradient for LLM unlearning by assumption diagonal Hessian. I am not sure if they have released their codes, but as a paper that specifically focuses on optimizers, they should be involved in this paper. 

It is good to see the authors define unlearning as erasing specific data under the condition in preserving the overall performance. Under this condition, comparing unlearning strength across optimization method is reasonable and fair to me. However, in practice, strict preservation is hard. The results in Figs 1-6 only report unlearning metrics, but I do not know how the retention performance is affected. So, the results are not convincing to me. A simple counterexample to show the results are not reliable is that, when we set an extreme large learning rate for GA, the model will erase all its knowledge. In this case, the model cannot recover any knowledge to be unlearned. However, in this case, we cannot say using large learning rate will make the model more robust. On the other side, it is hard to tune for the equal level of retention, so I suggest you to further report the results after UWC [1], which uses model mixing to align retention performance. 

[1] Unlearning with Control: Assessing Real-world Utility for Large Language Model Unlearning

In Fig 1, did the authors adopt the same learning rate across methods, or using the best hyper parameters across trials? Also, it is interesting to discuss what ensure a fair comparison across methods, e.g., the same learning rate or the same gradient magnitude. 

How to ensure the conclusion “a downgraded optimizer can in fact lead to upgraded unlearning robustness”, drawn from NPO, is general across a wide range of methods?

The related works can benefit from more recent and milestone works, such as [2-8].

[2] Rethinking Unlearning for Large Reasoning Models

[3] BLUR: A Bi-Level Optimization Approach for LLM Unlearning

[4] DRAGoN: Guard LLM Unlearning in Context via Negative Detection and Reasoning

[5] GRU: Mitigating the Trade-off between Unlearning and Retention for Large Language Models

[6] Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond

[7] Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning

[8] Rethinking machine unlearning for large language models

### Questions
Kindly please see the drawbacks above.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper explores an underexplored dimension of LLM unlearning: the impact of the optimizer on the robustness of forgetting. While most prior work has focused on modifying unlearning objectives, this study reveals that the "grade" of the optimizer from zeroth-order (gradient-free) to second-order (Hessian-based) influences the resilience of unlearning to post-hoc operations like quantization or fine-tuning. The authors find that lower-grade optimizers (e.g., zeroth-order or compressed-gradient optimizers) improve robustness by converging to flatter, harder-to-disturb basins in the loss landscape. Zeroth-order methods are naturally linked to randomized smoothing, which enhances resilience against perturbations. Based on these insights, the authors propose a hybrid optimizer combining first- and zeroth-order updates to strike a balance between unlearning efficacy and robustness.

### Strengths
1. This paper focuses on robust unlearning in the optimization process, showing a fresh research insight.

2. The proposed hybrid strategy effectively balances precise convergence (first-order) with robust basin discovery (zeroth-order).

### Weaknesses
1. The assessment of utility preservation is limited. Benchmarks such as MMLU should be included to evaluate general knowledge retention, and for the ToFU dataset, real-author and world-fact evaluations should be considered to better reflect practical unlearning scenarios.

2. The paper does not examine how varying the number of unlearning steps affects performance. Analyzing different unlearning durations would clarify the stability and convergence behavior of the proposed method.

3. No experiments are provided to study how different step counts between the two optimization methods influence results. Moreover, there is no theoretical justification for using the same number of steps (N) for both optimizers. A rationale or ablation study would strengthen the methodological soundness.

4. The proposed approach primarily combines two existing optimizers without introducing substantial conceptual innovation.

5. The robustness claim is insufficiently supported. To substantiate it, the method should be evaluated under membership inference and adversarial attack settings, which are standard in assessing unlearning and privacy resilience.

### Questions
Please refer to the Weaknesses section for a detailed discussion.

### Soundness
2

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
The paper investigates how optimizers of different “grades”—namely 0th-order, 1st-order, and 2nd-order—affect the process of machine unlearning. The authors observe that lower-grade optimizers, while less effective at unlearning, are more robust to relearning. Building on this insight, they propose a hybrid optimizer that combines 0th- and 1st-order methods to achieve a balance between unlearning effectiveness and robustness. Experiments conducted on MUSE and WMDP support the findings.

### Strengths
* A timely and relevant exploration of the relationship between optimizer order and unlearning dynamics.
* The finding that lower-grade optimizers are more robust to relearning is intriguing and adds nuance to our understanding of unlearning mechanisms.
* The paper examines multiple optimizer variants, including SignedSGD, SignedAdam, and low-bit quantization, providing a richer empirical picture.

### Weaknesses
* While the observation that lower-grade optimizers are robust to relearning is interesting, the underlying cause may not be adequately analyzed. The discussion already hints that the robustness could stem from higher noise levels in these optimizers, which also naturally explain their weaker unlearning effects. If so, the “grade” of the optimizer might be a proxy variable rather than the true causal factor—making noise the key driver of robustness.
* The discussion of computational cost is largely absent. If noise is indeed the mechanism behind robustness, it could impose significant efficiency costs on the unlearning process. Consequently, the proposed hybrid approach may also entail higher cost. Since cost–performance trade-offs are crucial for practical deployment, this omission weakens the empirical contribution.
* The evaluation is limited to only three datasets, which feels insufficient for a paper with primarily empirical claims. Broader validation would help establish generality and robustness of conclusions.

### Questions
See Weaknesses, and also
1. Lines 177–180 appear to be repeated
1. Line 366: Should $k$ be even?

### Soundness
2

### Presentation
2

### Contribution
2
