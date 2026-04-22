# SafeReview: Building a Robust Deep Review Assistant Against Prompt Injection

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
As Large Language Models (LLMs) are increasingly integrated into academic peer review, their vulnerability to prompt injection—adversarial instructions embedded in submissions to manipulate outcomes—emerges as a critical threat to scholarly integrity. To counter this, we propose a novel adversarial framework where a Generator model, trained to create sophisticated attack prompts, is jointly optimized with a Defender model tasked with their detection. This system is trained using a loss function inspired by Information Retrieval Generative Adversarial Networks (IRGANs), which fosters a dynamic co-evolution between the two models, forcing the Defender to develop robust capabilities against continuously improving attack strategies. The resulting framework demonstrates significantly enhanced resilience to novel and evolving threats compared to static defenses, thereby establishing a critical foundation for securing the integrity of automated academic evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
With the growing integration of Large Language Models (LLMs) into academic peer review, prompt injection has emerged as a critical vulnerability. Malicious authors embed hidden adversarial instructions to manipulate LLM-generated evaluations, undermining scholarly integrity. Existing LLM-based review systems focus on addressing limitations like superficial feedback but fail to tackle this evolving threat, as static defenses trained on known attacks are insufficient against continuously changing prompt injection techniques. This paper proposes SafeReview, a co-evolutionary adversarial training framework designed for LLM-based peer review systems that optimize both attacker and defender.

### Strengths
1. The issue proposed in this paper is the unfairness in AI review, which has attracted significant attention in the academic community. I believe this topic is highly interesting and meaningful. 
2. This paper proposes a method for AI review that prevents and mitigates prompt injection, which holds practical value.

### Weaknesses
1. AI review typically uses an LLM-as-judge model, and these models usually face issues of bias and variance. Bias refers to the gap between the model's scores and the ground-truth, while variance is the variance in the results of multiple samples from the review model. Since the optimization objective of SafeReview does not include variance, I believe it is necessary to analyze whether SafeReview will lead to an increase in the variance of the outputs of the review model.
2. I am very curious why GRPO is used in attacker training while DPO is used in defender training. If the defender can learn certain reasoning processes, will the review model become more interpretable?
3. I believe this paper needs to evaluate the review model (both before and after adversarial training) on a benign dataset to demonstrate whether SafeReview training impairs the inherent capabilities of the review model itself.
4. Previously, some researchers have used white fonts to conduct concealed prompt injection. I wonder if SafeReview can detect such types of attacks.

### Questions
See weaknesses

### Soundness
2

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
SafeReview examines prompt injection in long, scholarly peer-review workflows, where adversarial, instruction-like text embedded in submissions can inflate ratings and distort acceptance rankings. It introduces a co-evolutionary framework that trains an attacker and a defender in tandem. The attacker uses GRPO to craft subtle, context-aware injections, optimized by a hybrid reward that combines score inflation with rank disruption (e.g., reduced Spearman correlation with ground-truth order). The defender is trained with DPO on preference pairs that favor reviews resisting injected directives while preserving quality.

To handle long documents, the system first localizes risky regions via hierarchical segmentation, then applies fine-grained adversarial training. A curriculum stabilizes training by gradually increasing attack difficulty. Experiments simulate realistic review pipelines and report that, under strong GRPO-generated attacks, SafeReview reduces acceptance inflation, improves rank stability, and lowers false positives relative to an undefended reviewer and a static DPO baseline.

### Strengths
1. Interesting and timely approach tailored to peer-review settings. 

The paper tackles prompt injection in long scholarly documents—a high-impact, underexplored niche—by co-evolving an attacker and a defender. This framing feels fresh, domain-aware, and immediately relevant to current LLM-assisted reviewing workflows.

2. Useful, promising results on an extensive benchmark. 

The experiments cover realistic attack styles and long-document conditions, with clear metrics (e.g., score inflation, rank stability, false-positive control). The defense consistently improves robustness while preserving review quality, suggesting strong practical value and good prospects for real-world deployment.

### Weaknesses
1. Scope clarity vs. classic prompt injection. 

The paper frames attacks in peer-review documents but does not convincingly articulate what is fundamentally new beyond standard prompt-injection/jailbreak threats. It remains unclear whether the challenge is primarily long-context placement/localization (a setting detail) or introduces qualitatively different adversarial mechanics. Without a sharper problem definition (e.g., formal distinctions, new threat primitives, or impossibility results specific to scholarly reviews), the contribution risks reading as an application of known threats rather than a new problem class.

2. Missing comparative baselines limit external validity. 

The evaluation omits strong, diverse defenses that practitioners would reasonably try first, making it hard to attribute gains to the proposed method rather than to the choice of baseline. In particular:

* The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions — a training-time approach that explicitly teaches models to de-prioritize untrusted in-context directives.

* SecAlign: Defending Against Prompt Injection with Preference — secure preference optimization that aligns outputs away from injection-following behavior.

* Llama Prompt Guard 2 — a lightweight detector/guardrail that can pre-filter or route suspicious inputs.
Including these would better position the method against (i) preference-optimization defenses, (ii) instruction-priority finetuning, and (iii) practical detector-based guardrails.

### Questions
Why and how prompt injecting a paper is fundamentally different from traditional prompt injection settings? why you do not compare with baselines such as SecAlign (see my comment in weakness)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes SafeReview, a coevolutionary adversarial training framework to improve the robustness of LLM-based peer-review systems against prompt-injection attacks. A generator model produces adversarial injection prompts from a database of publications, while a defender (reviewer) learns to resist them. Attacker will be updated by GRPO, while defender is updated via DPO. Experiments on NeurIPS and DeepReview datasets show reduced acceptance of manipulated papers and improved ranking correlation, suggesting that SafeReview enhances review integrity under adversarial conditions.

### Strengths
1. The paper is well motivated. Securing AI-based peer review is an important and underexplored problem.
2. The co-evolutionary training method, in general, is interesting.
3. The results using Qwen3-4B-Instruct Team the Generator and DeepReviewer-14B as the Defender show good performance.

### Weaknesses
1. One major concern is the one-sided notion of safety. The paper focuses entirely on avoiding false positives (i.e., stopping flawed papers from being wrongly accepted) but neglects the equally important false negative side, i.e., ensuring that good papers are not unfairly penalized. A robust review model must preserve both sensitivity and fairness, not just caution.
2. There is no analysis of bias amplification. By training the reviewer to resist persuasive or assertive language, the model may overcorrect and start undervaluing legitimate confident writing, leading to systematically harsher or more negative reviews.
3. There is no evaluation showing that defended reviews remain consistent with expert human judgments in both positive and negative cases.
4. The paper conducts limited experiments, for example, only testing Qwen3-4B-Instruct Team the Generator and DeepReviewer-14B as the Defender, without testing other models.

### Questions
See weakness.

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
3

### Summary
This paper tackles prompt injection attacks in LLM-based peer review. The authors propose SafeReview, a co-evolutionary framework where an attacker model (Generator) and a review model (Defender) are trained adversarially. The Generator creates attack prompts, while the Defender learns to resist them. Results show the method reduces the acceptance rate of attacked papers and improves correlation with ground-truth scores.

### Strengths
The paper is well structured, and technical details are clearly presented.

The proposed co-evolutionary-based approach is a well-reasoned method for building a dynamic defense that outpaces static ones. The use of GRPO and DPO is technically sound.

### Weaknesses
1. Key innovations mentioned in the introduction, such as “hierarchical segmentation” for long documents and “curriculum scheduling”, are not explained in the methods or experiments.

2. The claim that co-evolutionary training (SafeReview) beats static defense (Static DPO) is unsubstantiated. The paper fails to test both defenses against both attack types (e.g., Static DPO vs. GRPO attack), making a direct comparison impossible.

### Questions
Please address all concerns in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
