# Model Correlation Detection via Random Selection Probing

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
The growing prevalence of large language models (LLMs) and vision–language models (VLMs) has heightened the need for reliable techniques to determine whether a model has been fine-tuned from or is even identical to another. Existing similarity-based methods often require access to model parameters or produce heuristic scores without principled thresholds, limiting their applicability. We introduce Random Selection Probing (RSP), a hypothesis-testing framework that formulates model correlation detection as a statistical test. RSP optimizes textual or visual prefixes on a reference model for a random selection task and evaluates their transferability to a target model, producing rigorous p-values that quantify evidence of correlation. To mitigate false positives, RSP incorporates an unrelated baseline model to filter out generic, transferable features. We evaluate RSP across both LLMs and VLMs under diverse access conditions for reference models and test models. Experiments on fine-tuned and open-source models show that RSP consistently yields small p-values for related models while maintaining high p-values for unrelated ones. Extensive ablation studies further demonstrate the robustness of RSP. These results establish RSP as the first principled and general statistical framework for model correlation detection, enabling transparent and interpretable decisions in modern machine learning ecosystems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Random Selection Probing, a procedure to detect whether two models are related (e.g. fine-tuned from each other). The setup is as follows: They have two models: a reference model and a target model. They define a trivial "random selection task" (pick a ranadom letter a-z) that should not depend on the training data. Then they perform "prefix optimization". This is where they optimize short input prefixes on the reference model to bias its outputs towards specific target tokens on that task. This optimization finds prefixes that exploit some patterns in this task. They apply the optimized prefixes to a target model and measure how well the same biases transfer. If the prefixes retrain their effect, this is used as evidence that the behavioral correlation has shared parameters or common fine-tuning

### Strengths
- This is a clearly testable idea that can be operationalized. Probing the perturbations is clear.
- NIce that it can work on methods that do not rely on gradients or having access to parameters
- Nice that there is an inclusion of an unrelated baseline model as a control as a part of the design choice

### Weaknesses
- I found the writing to be difficult to follow generally. Personally (and subjectively), this lacked structure and clarity quite a lot. Misunderstandings from my side can arise from this; this is reflected in my low confidence rating -> I'm not sure I fully understood the paper.
- I don't understand something about the concept of this. You search for a prefix that biases the model to favor one designated letter by maxinimizing the normalized probability. I understand why you substract the unrelated control model. But why should successful transfer of these prefixes uniquely indicate fine-tuning lineage, rather than other shared factors (e.g. architecture or pre-training data or anything else)? 
- There's no sensitivity analysis showing how the method is sensitive to prompt variability

More confusion than weaknesses from my side - I added some key questions in the next section.

### Questions
- Why are you calling p-values "rigorous" everywhere? What's so rigorous about them? I'm genuinely confused. Can they be non-rigorous? 
- P-values also depend on the size of the dataset and you can get arbitrarily small p-values that would capture a small effect size under frequentist inference. That's why people typically report effect sizes. How does this work under your framework?
- Why report only p-values and not effect sizes? Given that the p-values are so small, what's the benefit of reporting that vs the effect sizes to begin with?
- Are you detecting "model correlation" or whether the model has the same lineage? You can have models that are correlated that do not directly descend from each other or have any specific fine-tuning related to each other. The framing of the paper seems you are working on the former, but the explanation seems you're working on the latter. 
- Is the assumption of the paper that a model always either has some lineage (H1) or it does not (H0)? So, it does not matter the extent of this lineage? 
- I'm not sure about the p-values. The p-values you report are super tiny (1.00e-300). As far as I understand, you are assuming that X follows a binomial distribution (page 5) which then means you are assuming independent outputs. Questions: (a) Are the outputs not very correlated because of the joint optimization of prefixes? What does it mean to treat them as independent here? (b) Don't you need to adjust for multiple hypothesis testing here?
- Can you comment generally on the internal validity of this approach: are you measuring what you think you're measuring? I'm quite confused here.
- Also, is the prinary innovation here the procedure with the tokens or the application of a hypothesis testing procedure? If it's the former, why did you choose this procedure and what makes it unique/good to begin with? It seems one could design many procedures.
- Why are all the numbers in Table 4 on Grad exactly the same, or Table 6?
- Does it really make sense to use a 0.05 threshold (e.g. page 8) when your p-values are so ridiculously small in so many cases?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work aims to devise methods to determine if two large models are actually the same one.

To do this, the authors frame it as a statistical hypothesis test framework and compare the potentially 'matching' model against a reference, non matching one. They argue this is the first approach to do such an analysis in a statistically principled manner without requiring the internal weights of the model.

The idea is reasonably simple:
1. Find prompts that cause an output that is significantly different from what other, unrelated models, would give for that output.
2. Run these prompts on the candidate model and use statistical testing to compare results.

### Strengths
1. This paper is clearly written and presents an interesting, principled way, to compare two methods. They validate the approach by comparing disparate models and validating the significance is low. They then apply it to model families, finding now that the differences are insignificant. The approach is realistic in terms of the access to aspects of the model from a 3rd party pov.

2. They use realistic data -- e.g. the llama variants and are able to find correlation between the original and fine-tuned model (Table 3/4). Though it would be interesting here to annotate as well *how much* data the model was finetuned on and see if you can find a relationship between how 'significant' the relationships are and how much data the model was finetuned on.

### Weaknesses
1. I'm a bit confused by the fine-tuned results. Here you finetune different models on the same data and the point is that because they have been fine-tuned on the same data, the test should find that they are correlated ?

2. Nit-pick. The ordering of experiments is a bit weird -- why not show that you can both find related / unrelated ones in section 4 and then the variants of that setup. Otherwise, you read all of 4 and you're left with the question of -- well do does the test think that everything is related -- and you need to get to Table 7/8 to see that this was tested, though it's given in the context of an ablation.

3. Why are there no comparisons to other methods? While other methods may not have the significance test -- surely you can devise a simple way to use them in such a way. This would be interesting to understand if your significance test is the real value add or if both the prompt optimization + significance test is the value add. For example, ModelDiff suggests using a set of held out models for the threshold (that are known not to be related) -- you could use that, as you have your 'unrelated' model anyway. You could then compare in terms of 'accuracy' using all the model pairs that you have.

4. For Figure 2-- it would be good to do this with models that aren't related to ensure that you would still get a valid result here.

### Questions
I like the idea of the paper but my concerns above have caused me to give a more borderline score.

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
The paper introduces Random Selection Probing (RSP), a framework for detecting model correlation (specifically, identifying if a target model $M_t$ is derived from a reference model $M_r$). To replace heuristic similarity scores with statistical p-values, the authors utilize a two-stage process: first optimizing "adversarial" prefixes on $M_r$ to bias a random selection task toward specific tokens, and then testing the transferability of these prefixes to $M_t$ using a binomial hypothesis test. The method is adapted for LLMs (using an unrelated model $M_u$ to suppress generic features) and VLMs, across white-box, gray-box, and black-box settings.

### Strengths
- The shift from arbitrary similarity thresholds to a principled statistical testing framework yielding p-values addresses a significant ambiguity in current model similarity literature.

- The method is versatile, providing workable optimization strategies for diverse access settings (gradients, logits, or just text output).

- The introduction of the unrelated model ($M_u$) into the optimization objective for LLMs is a sound mechanism to prevent the learning of universally transferable (generic) prompt features.

### Weaknesses
- While the paper broadly claims utility for "intellectual property protection", it fails to describe a concrete scenario where this capability is currently a bottleneck. For legitimate IP auditing, one would likely need stronger evidence than a p-value from probing. The paper does not establish when this specific gray/black-box probing is strictly necessary compared to more robust weight-based analyses in a legal context.

- Relying on an unvalidated uniformity assumption: The proposed statistical test hinges on the assumption that, under the null hypothesis, the target model chooses tokens uniformly (i.e., $p=1/N$) in the random selection task. This is likely factually incorrect for many base models, which often exhibit inherent biases toward specific common letters (like 'A' or 'E') even without prefixes. If the underlying distribution is not uniform, the binomial test in Eq. 5 is fundamentally flawed and may produce false positives on uncorrelated but similarly biased base models.

- The practical utility of this method for VLMs is questionable due to extreme computational overhead. Table 19 reveals that preparing a single visual prefix in the logits setting takes ~1 hour on an H1005. Generating the required 500 prefixes 6 would therefore require ~500 H100 GPU-hours merely to prepare to audit a single model family. This is a severe practical limitation that is largely glossed over in the main text.

### Questions
- Can you provide empirical evidence that unmodified, uncorrelated base models actually follow the uniform $1/N$ distribution for your random selection prompt? How does your p-value calculation account for inherent model biases toward specific letters?

- Why were standard fingerprinting baselines (like Xu et al., 2024, which you cited) not considered from the empirical comparisons? How does RSP compare to these methods in terms of detection robustness under fine-tuning?

- Given the ~500 GPU-hour requirement for VLM prefix preparation in gray-box settings, is this method practically viable for real-world auditing? Have you explored if fewer prefixes (e.g., $K=50$) suffice for VLMs given the very strong signal shown in Table 6?

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
4

### Summary
This paper introduces Random Selection Probing (RSP), a statistical hypothesis-testing framework for detecting model correlation. Unlike existing similarity-based methods that produce heuristic scores, RSP provides p-values by optimizing prefixes on a reference model for a random selection task (e.g., "choose a letter from a to z") and evaluating their transferability to a target model. The method works across LLMs and VLMs under diverse access conditions (gradient/logits-accessible for reference models; gray-box/black-box for test models). Experiments demonstrate extremely small p-values for fine-tuned models while maintaining high p-values for unrelated ones.

### Strengths
1. The problem of IP protection is urgent in the LLM era.

2. General Framework Design: RSP's two-stage pipeline (prefix optimization; correlation detection) considers various realistic scenarios. It uses multiple optimization techniques (GCG, GA, PGD, ZOO), and successfully handles:
- Model types: Both LLMs and VLMs;
- Access levels: Gradient or logits access during optimization, and grey-box or black-box settings during detection.

3. Robust False Positive Control: The introduction of an unrelated model $M_u$ to penalize generic features during optimization (maximizing $P_{M_r} - P_{M_u}$) is effective.

### Weaknesses
1. Opaque Transfer Mechanism: The paper relies on the key assumption of adversarial transferability and demonstrates that prefixes transfer but doesn't deeply analyze *why*. Key questions remain to answer: What specific "model-specific features" are encoded? Are they architectural artifacts, training data signatures, or optimization quirks?

2. Broad Applicability: How robust is transfer to more intense modifications other than fine-tuning (e.g., pruning, quantization, model merging, distillation, MoE upcycling)?

3. Hypothesis Test Model: The probability of binomial distribution is set as $1/N$. However, LLMs may have inherent tendencies towards some choices, which is not considered.

4. Adversarial Scenario: Such input optimization method is susceptible to detections and brittle to perturbations, which may not work well for test models with adversarial defense mechanism.

5. Computational Efficiency: While prefix optimization is a one-time cost, it can be substantial with many prefixes in need. This limitation could hinder adoption for resource-constrained auditing scenarios.

6. Lack of Related Work and Baseline Comparisons: Some works [1,2] also uses GCG optimization for model identification, thus should be mentioned and compared. No empirical comparison to existing model similarity methods (e.g., REEF, TRAP, LLMmap) is presented.

7. A minor issue: Two works share the name of AutoDAN [3,4], while the citation in Related Work might be mistaken according to the description.

[1] Gubri, Martin, et al. "Trap: Targeted random adversarial prompt honeypot for black-box identification." arXiv preprint arXiv:2402.12991 (2024).

[2] Tsai, Yun-Yun, et al. "RoFL: Robust Fingerprinting of Language Models." arXiv preprint arXiv:2505.12682 (2025).

[3] Liu, Xiaogeng, et al. "Autodan: Generating stealthy jailbreak prompts on aligned large language models." arXiv preprint arXiv:2310.04451 (2023).

[4] Zhu, Sicheng, et al. "Autodan: interpretable gradient-based adversarial attacks on large language models." arXiv preprint arXiv:2310.15140 (2023).

### Questions
1. The choice of candidate set. Have you tried other tasks other than choosing letters?

### Soundness
2

### Presentation
3

### Contribution
2
