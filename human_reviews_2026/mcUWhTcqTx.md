# Distillation Lineage Inspector: Black-Box Auditing of Model Distillation in LLMs

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Model distillation has emerged as a widely used technique for creating efficient models tailored to specific tasks or domains. However, its reliance on knowledge from foundation models raises significant legal concerns regarding intellectual property rights. To address this issue, we propose the Distillation Lineage Inspector (DLI) framework, which enables model developers to determine whether their large language models (LLMs) have been distilled without authorization, even in black-box settings where training data and model architecture are inaccessible. DLI is effective across both open-source and closed-source LLMs. Experiments show that DLI achieves 80\% accuracy with as few as 10 prompts in fully black-box settings and yields a 45\% improvement in accuracy over the best baseline under standard experimental conditions. Furthermore, we analyze how the auditor’s knowledge of target models influences performance, providing practical insights for building privacy-preserving and regulation-compliant AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the problem of determining whether a large language model (LLM) has been used as a teacher to distill another model. In particular, the proposed approach resembles a (shadow-model based) membership inference setup: it assumes access to shadow models, including both distilled ($M_s$) and non-distilled ($M_n$) ones derived from the same teacher, as well as a generic comparison model ($M_c$). The method identifies model-style tokens, i.e., tokens that are assigned high probabilities by the teacher model $T$ but low probabilities by the comparison model $M_c$, and uses their likelihoods as discriminative features. A classifier is then trained on these features (from the shadow models) to distinguish distilled from non-distilled models. Experimental results demonstrate promising performance when using LLaMA3 or GPT-4o as teacher models across a diverse set of distilled and non-distilled candidates.

### Strengths
- This paper investigates an interesting and practically highly relevant topic.
- The proposed idea is reasonable and well-motivated.
- The experiments demonstrate intriguing results.

### Weaknesses
- The method description is somewhat unclear (see detailed questions below). In particular, the exact threat model (for both the proposed method and the baselines) is not explicitly defined, e.g., what level of access or knowledge does the attacker, model owner, or the proposed Distillation Lineage Inspector have? What actions can the attacker/model owner take (e.g., can the attacker “wash out” the feature indicators by further fine-tuning on related or unrelated data after the initial distillation)? what characteristics $M_c$ is expected to have (as from my understanding it should definitely be different from the teacher model) 

- The experimental scale is relatively limited, making the results less convincing. Given the nature of the task, more extensive and diverse experiments are often necessary to support strong claims of verification and generalization.

### Questions
- Method description is somewhat confusing. From the pseudocode in Algorithm 1, the statement "append($D_{tokens},U$)" appears to be executed only once and does not occur within any iteration or loop. It is therefore unclear what the later lines 269–272: “for each token list $U_i \in D_{tokens}$ and "for each token $t_j \in U_i$ with position $p_j$ in $s_i$" are referring to.

- The naming conventions for “target data,” “dataset,” and “model” are also somewhat confusing (especially given their similarity—but not equivalence—to those used in membership inference literature). These terms could be made more explicit to avoid ambiguity.

- While the proposed approach resembles a shadow-model–based membership inference attack, the paper lacks deeper discussion on the principled difference or relationship between them (and potentially also the connection to model watermarking). Also, it would also be valuable to provide more intuition or analysis explaining why the proposed approach works better than existing baselines (e.g. what knowledge is particularly important for the inspector). Furthermore, it would be important to discuss whether techniques from the membership inference or model watermarking literature could be applied or extended to this setting.

- How is $M_n$ trained? Does it undergo standard fine-tuning on the local dataset?

- The origin of “for each input $x \in X$ is somehow unclear:  are these raw samples from the fine-tuning dataset? Moreover, referring to 
$s$ as the “target sample” (e.g., in Algorithm 1) is potentially confusing in the context of privacy or membership inference literature. It would help to explicitly define both $x$ and  $s$ in the pseudocode and clearly distinguish their roles.

- From a metric perspective, it may be helpful to include results such as true positive rate (TPR) at low false positive rate (FPR) thresholds (e.g., 0.1%, 1%) to better illustrate detection performance.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the intellectual property and legal concerns of unauthorized model distillation, where a "student" model is trained using knowledge from a proprietary "teacher" model. The authors propose the Distillation Lineage Inspector (DLI), a novel framework to audit a model's provenance in a black-box setting. DLI works by training a binary classifier (an "auditor model") to distinguish between distilled and non-distilled models. To do this, the auditor first creates "shadow-distilled proxies" by training models with the teacher model (as positive examples) and without it (as negative examples). The framework's key insight is to identify "model-style tokens", tokens that the teacher model has a high probability of generating but a generic "comparison model" finds unlikely. The auditor model is then trained on feature vectors, which are histograms of the probabilities that the shadow models assign to these specific style tokens. To test a suspicious model, DLI generates its probability histogram for these tokens and feeds it to the trained auditor for a final decision. The paper evaluates DLI under two conditions: Token-Level Access (Kt), with access to token probabilities, and the more restrictive Word-Level Access (Kw), with access to only the final text. Experiments show DLI achieves high accuracy (e.g., 100% in Kt and 85% in Kw on the HealthCareMagic dataset) and significantly outperforms baselines by up to 45%.

### Strengths
- This paper tackles a timely and critical problem: the need for auditing AI models to protect against intellectual property theft via unauthorized distillation.
- A significant contribution is reframing the problem from "closed-world identification" (choosing from a list of known teachers) to "open-world verification" (confirming lineage from one specific teacher), which is a much more realistic auditing scenario.
- The DLI framework is model-agnostic and designed for purely black-box settings, making it applicable to auditing closed-source commercial APIs, not just open-source models.
- The method demonstrates remarkable robustness. It performs well even under the highly constrained Word-Level Access (Kw) (word-level only) setting by estimating probabilities via sampling, which is a practical solution for a real-world auditor.
- The empirical results are very strong. DLI achieves near-perfect accuracy in the Token-Level Access (Kt) setting and a 45% absolute accuracy improvement over the best-performing baseline on the healthcare dataset. It is also highly data-efficient, achieving over 80% accuracy with just 10 prompts.

### Weaknesses
- The framework's effectiveness is only validated using two teacher models (Llama-3-8B and GPT-4o-mini), a limitation the authors acknowledge. It is unclear how well the "model-style token" signature generalizes to other model families and architectures.
- The crucial step of "Identifying Model-Style Tokens" depends on a generic comparison model (Mc). The ablation in Appendix A.3 (Figure 5) shows that the choice of Mc significantly impacts performance; for example, auditing GPT-4o failed with some comparison models. This suggests that finding an effective Mc is a non-trivial prerequisite for the auditor.
- The DLI framework is complex, requiring the auditor to execute a multi-stage pipeline: (1) distill an ensemble of shadow proxies , (2) select a suitable comparison model , (3) generate a labeled dataset , and (4) train a final classifier using AutoML. This complexity may be a barrier to its practical adoption.
- The baseline methods (PoS, MPT) perform exceptionally poorly , with MPT and PoS scoring 0% on correctly identifying non-students in some experiments (e.g., Table 5). This may suggest the baselines, which were not designed for this "open-world verification" task, are not strong points of comparison, making DLI's dominance appear larger.
- The paper's limitations note that the method is designed only for NLP models and cannot be extended to multimodal settings.

### Questions
- The choice of the "comparison model" (Mc) seems critical, as a poor choice (like phi-2 for auditing GPT-4o) leads to bad performance. How should an auditor practically select an effective Mc without prior knowledge? What properties make a comparison model "good" or "bad" for this task?
- In the Word-Level Access (Kw) (word-level) setting, probabilities are estimated using only N=5 samples. This results in very coarse probability estimates (e.g., 0.0, 0.2, 0.4, etc.). How does this coarse-grained estimation interact with the feature-encoding histogram, which uses 10 bins? Is this low N the primary reason for the performance drop from Token-Level Access (Kt)?
- The DLI framework requires the auditor to train an ensemble of shadow models. How sensitive is the final audit accuracy to the size and diversity of this shadow ensemble? Would the framework still work if the auditor only trained a single shadow-distilled proxy?
- The baselines (PoS, MPT) performed very poorly, especially at identifying non-students (e.g., 0/10 correct in Table 5). Is this because these methods are fundamentally unsuited for open-world verification and are only designed for closed-world identification, thus making them "strawman" comparisons?
- Could the DLI framework be inverted to solve the closed-world identification problem? For example, could an auditor pre-train 10 different auditors, each for a different major teacher model, and run a suspicious model against all 10 to determine which teacher it was most likely distilled from?

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
4

### Summary
This paper studies the problem of auditing whether a suspicious model has been illicitly distilled from a proprietary large language model (LLM), even in strict black box API settings. To address this, the authors propose Distillation Lineage Inspector (DLI), which is a model-agnostic auditing framework. DLI works under both token level access (log-probabilities available) and word level access (content only). Experiments on healthcare and legal QA datasets show that DLI significantly outperforms existing provenance methods.

### Strengths
1. The problem considered in this paper is very timely and it has significant real-world relevance. 
2. The proposed DLI requires no access to model internals, training data, or architecture... this is super crucial for practical auditing purposes..
3. The authors provided a very strong empirical performance by using multiple datasets.

### Weaknesses
1. The proposed framework relies on constructing shadow student models for each teacher. This is compute intensive task and may not scale to many suspect teachers simultaneously..
2. Performance drops meaningfully when prompts are distribution mismatched. This may limit applicability where the original distillation domain is unknown.
3. Authors did not carry out any analysis on adaptive adversaries.. which is an important concern for practical usage of the proposed approach.

### Questions
Please address the above issues highlighted in the weakness section.

Furthermore, I have the following comments as well:
(a) Considering adaptive attackers.. they may mask stylistic token patterns.. this calls out for an ablation on adversarial output perturbation.
(b) When auditing many candidate teachers, do auditors need separate shadow proxies for each? Provide computational complexity estimates.
(c) How stable are these stylistic tokens across: temperature changes, decoding strategies (greedy, nucleus, beam), etc.?
(d) Some unrelated models may coincidentally share stylistic preferences. When does DLI misfire?

### Soundness
2

### Presentation
2

### Contribution
2
