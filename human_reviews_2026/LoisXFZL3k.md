# Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Intervention-based model steering offers a lightweight and interpretable alternative to prompting and fine-tuning. However, by adapting strong optimization objectives from fine-tuning, current methods are susceptible to overfitting and often underperform, sometimes generating unnatural outputs.
We hypothesize that this is because effective steering requires the faithful identification of internal model mechanisms, not the enforcement of external preferences.
To this end, we build on the principles of *distributed alignment search (DAS)*, the standard for causal variable localization, to propose a new steering method: **Concept DAS (CDAS)**.
While we adopt the core mechanism of DAS, *distributed interchange intervention (DII)*, we introduce a novel distribution matching objective tailored for the steering task by aligning intervened output distributions with counterfactual distributions.
CDAS differs from prior work in two main ways:
first, it learns interventions via weak-supervised distribution matching rather than probability maximization;
second, it uses DIIs that naturally enable bi-directional steering and allow steering factors to be derived from data, reducing the effort required for hyperparameter tuning and resulting in more faithful and stable control.
On AxBench, a large-scale model steering benchmark, we show that CDAS does not always outperform preference-optimization methods but may benefit more from increased model scale.
In two safety-related case studies, overriding refusal behaviors of safety-aligned models and neutralizing a chain-of-thought backdoor, CDAS achieves systematic steering while maintaining general model utility.
These results indicate that CDAS is complementary to preference-optimization approaches and conditionally constitutes a robust approach to intervention-based model steering.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This works presents a new activation steering methods building up on the idea of distributed alignment search (DAS). Therefore, the authors introduce a novel distribution matching objective tailored for the steering task by aligning intervened output distributions with counterfactual distributions. Authors verify the proposed method on AxBench, two cases studies and also ablate the performance tradeoff of steering.

### Strengths
- The paper is **clearly written** and is easy-to-follow despite a lot of technical details
- **Appealing formulation** with distribution matching to the counterfactual distribution. Only caveat is that it introduces another axis of data augmentations requirements to satisfy the training data need.
- **Exhaustive experiments** over multiple benchmarks and various ablation studies, analyzing the benefits and limitations of CDAS. Overall, CDAS shows promising performance, especially under the fact that it seems to lead to less off-target effects than RePS (as confirmed in table 3) which seems to dominate performance-wise.
- **Steering / Performance tradeoff experiment in table 3** — this aspect is heavily ignored in many papers and I appreciate this analysis. However I am not sure if overriding refusals is the most convincing steering axis to test in this setting, as refusals may happen rarely on these benchmarks. For instance, repeating this experiment on a safety relevant task would further strengthen the evidence.

### Weaknesses
- **Noise in Experiments:** Some results seem quite noisy (e.g. Figure 1) — I would strongly recommend adding std. or confidence intervals to the remaining experiments.
- **Tuned Factor Dependence:** Can you provide more intuition on why there is such a big difference between the unit factor and the tuned factor?
- **Benefits with Scale:** The authors raise multiple time the argument that CDAS benefits with scale. Do you have an explanation for this? Especially as you present better performance scores for the 7B instead of the 80B model (in Table 2)

### Questions
- Why is DAS missing in the AxBench experiments?
- Why do you choose the tuning factor differently for DAS/CDAS (based on Alpaca) compared to RePs? Is this the source of the performance discrepancy? 
- Figure 1C: Variance is huge and it seems to work only on one tested layer. Do you have an explanation for this? 
- Table 4: Is the fact that CDAS improves performance on tinyMMLU not an indicating that results might be very noisy? 
- Do trends between methods change in other data regimes (less or more data)
- Do you match the amount of training data? All pairwise approaches basically use the double amount of data (when treating every prompt + response as one data point)

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
1

### Summary
The paper introduces **Concept Distributed Alignment Search (CDAS)**, an intervention-based model-steering method extending *Distributed Alignment Search (DAS)* from causal variable localization. The method aims to identify concept-specific internal features rather than impose external preferences. Experiments on _AXBENCH_ and two safety case studies (refusal override and CoT-backdoor neutralization) show CDAS can maintain model utility and scales better with model size, though it does not consistently outperform preference-optimization baselines.

### Strengths
- Interesting conceptual shift linking steering with causal localization and interpretability.  
- Comprehensive experiments across benchmarks and safety settings.

### Weaknesses
I am very much an outsider to the “model steering” field, however, unfortunately, this paper does a weak job at presenting much needed context for new readers to appreciate the why and how of their manuscript

Much of the structure and writing assumes readers are familiar with extant work and understand their shortcomings

e.g.,
* [l42/46] how does “intervention-based” result in “optimization-based”?

* [l52] what does “degenerate, repetitive generations” even mean?

* [l55] why should the readers appreciate DAS, and the proposed extension CDAS, as “standard approach[es] for causal variable localization.” what does this even mean?

### Questions
- What explains the non-monotonic behavior in Fig. 1c?  
- Please see questions above in weaknesses.

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
3

### Summary
Building upon the framework of distributed alignment search (DAS), the authors introduce Concept DAS (CDAS), a method that learns interventions via weakly supervised distribution matching between intervened and counterfactual outputs. CDAS facilitates bi-directional, data-driven model steering with fewer hyperparameters and enhanced stability. Through experiments on AxBench and various safety case studies—such as overriding refusal behaviors and neutralizing a chain-of-thought backdoor—the authors demonstrate that CDAS achieves faithful and scalable control while preserving the overall utility of the model. The approach serves as a robust complement to preference-optimization methods, offering an alternative pathway for effective model steering.

### Strengths
The paper tackles a compelling and timely problem, presenting a solution that is both concise and elegant. The manuscript is well-written and structured, making the methodology and results accessible. The authors conduct extensive experiments to validate CDAS, providing thorough comparisons with existing approaches. Detailed experimental protocols and results are available in the supplemental material, enhancing transparency and reproducibility.

### Weaknesses
Despite its merits, there are a few aspects that require clarification or further analysis:

(1) In certain experiments, CDAS underperforms relative to baselines (e.g., Tables 1 and 3). The authors should provide insights or hypotheses explaining these performance gaps.

(2) It remains unclear under which conditions CDAS excels and under which scenarios it may fall short. A discussion of the limitations and situational strengths of the method would strengthen the paper.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a new steering method called Concept Distributed Alignment Search (CDAS). It builds upon the Distributed Change Intervention (DCI) technique from the DAS method, combining it with a distribution-matching objective based on the Jensen–Shannon divergence. The method is evaluated on the AXxBENCH benchmark and two safety-related case studies focused on concept suppression.

### Strengths
The CDAS objective encourages the model to learn concepts that are aligned with the model’s overall output distribution under the concept-induced input. Consequently, supervision does not come directly from ground-truth responses, but rather from the model’s own internal distribution. This is an interesting idea, as it may lead to outputs that are more naturally aligned with the inherent responses of LLMs.

In the refusal override experiments, CDAS achieves the best KL divergence loss, while maintaining reasonable performance on the TruthfulQA and MMLU benchmarks. This suggests that the intervened model’s outputs remain close to the model’s natural response distribution.

### Weaknesses
While the premise behind CDAS and its training objective is compelling, the results are mixed. For example, in the experiments presented in Table 1, CDAS achieves the best performance on Gemma-2-9 L20 under a tuned factor, outperforming all other methods. However, on other intervention layers and with smaller models (e.g., 2B), CDAS fails to surpass RePS—although it still outperforms DiM, BiPO, and, in two cases, Lang. 

In the refusal override experiments, CDAS also underperforms on the smaller model. In the CoT experiments for neutralizing backdoors, CDAS successfully overrides malicious behavior, but only when applied to earlier layers; for later layers, the ASR increases sharply and exhibits large variance.

Particularly in the experiments from Table 1, CDAS appears highly sensitive to the setup of the steering factor, which could make its application in practical scenarios cumbersome.

While the authors provide quantitative comparisons of CDAS and alternative methods, it would be valuable to include an analysis of training stability and computational overhead (see specific questions below).

### Questions
As mentioned in the weaknesses, CDAS demonstrates somewhat mediocre performance across the evaluated tasks—sometimes surpassing other models or layer interventions, and sometimes falling behind—though potentially producing outputs more faithful to the underlying distribution of the LLM (as indicated by KL divergence). I appreciate that the authors acknowledge these nuances and discuss their method fairly, suggesting that CDAS may be preferable for larger LLMs or when preserving model utility is a key objective. I still believe CDAS is an interesting addition to the family of steering methods.However, I would appreciate deeper insights into the causes of its underperformance. For instance:

- Why does CDAS perform worse on smaller models? 
- Is this due to model scale, or does it depend on the model family?
- In Figure 1, why does the variance on the ASR task increase so sharply?

A more detailed discussion of these points could significantly strengthen the paper.

In the same vein, I am curious how CDAS compares to other approaches in terms of training stability. Could the variance in Figure 1 be a result of collapsed or unstable training? How does the training objective behave across experiments? The authors acknowledge that CDAS is sensitive to the steering factor—could this sensitivity be related to the stabilization (or destabilization) of training?

### Soundness
3

### Presentation
3

### Contribution
3
