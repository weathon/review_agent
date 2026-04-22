# Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Large language models may encode sensitive information or outdated knowledge that needs to be removed, to ensure responsible and compliant model responses. Unlearning has emerged as an efficient alternative to full retraining, aiming to remove specific knowledge while preserving overall model utility. Existing evaluations of unlearning methods focus on (1) the extent of forgetting of the target knowledge (forget set) and (2) maintaining performance on the retain set (i.e., utility). However, these evaluations overlook an important usability aspect: users may still want the model to leverage the removed information if it is re-introduced in the prompt. In a systematic evaluation of six state-of-the-art unlearning methods, we find that they consistently impair such contextual utility. To address this, we augment unlearning objectives with a plug-in term that preserves the model's ability to use forgotten knowledge when it is present in context. Extensive experiments demonstrate that our approach restores contextual utility to near original levels while still maintaining effective forgetting and retain-set utility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper argues that standard LLM unlearning pipelines—optimized to reduce recall on a forget set while preserving generic utility on a retain set—silently degrade a third axis the community rarely measures: the model’s ability to use the “forgotten” information when it is explicitly reintroduced in the prompt (their “Contextual QA” setting).

### Strengths
- The paper isolates “contextual utility” as a distinct deployment requirement—use it if the user provides it—and shows that popular unlearning objectives deform internal representations enough to suppress even context-grounded use.
- The contextual variant rescues RMU from near-zero Contextual QA (≤0.05) to ~0.97–0.99 while keeping retain-set utility virtually unchanged; NPO/UNDIAL see double-digit LLM-Judge gains too.
- Evaluation targets real usage rather than proxy memorization.

### Weaknesses
- The proposed fix is a straightforward KL-consistency term to the original model on contextual prompts, conceptually akin to the KL regularizers widely used in RLHF/DPO-style pipelines the paper itself cites as inspiration. The core contribution is thus problem framing + evaluation, not a new optimization principle.
- Because the KL target is p_{orig}, any pre-existing issues in the original model’s grounding (hallucinations, refusal idiosyncrasies) are inherited by construction. The paper does not compare against anchoring to a teacher ensemble or to gold references, nor does it analyze when p_{orig}​ miscalibrates contextual evidence.
- Seems like Grad Difference, DPO, GradAscent are not being considered as baselines (as shown in Table 1), but the corresponding results are missing in table 2.

### Questions
- Why is KL anchored to p_{orig}​ rather than an ensemble/teacher oracle or gold references when available? Have you observed cases where p_{orig} provides incorrect contextual distributions, and how does the method behave then?
- How does the method fare when context contains both correct and subtly incorrect spans (a common RAG issue)?
- What happens if distractor facts outnumber correct snippets, or if retrieval returns off-topic but lexically similar passages?
- Have you tried a teacher-free variant (e.g., KL to a masked version of p_w that conditions on evidence spans) to avoid hard dependence on p_{orig}?

I am willing to adjust my score if my concerns are addressed.

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
5

### Summary
The paper illustrates a gap in current LLM unlearning evaluations: the lack of consideration for contextual recoverability (i.e., the ability of a model to recall forgotten knowledge when the context is reintroduced). To address this, the authors propose a new evaluation setup, Contextual QA, and demonstrate that several existing methods (RMU, NPO, UNDIAL, DPO, GradAscent, GradDiff) perform poorly under this setting. They further introduce a context-aware objective that augments existing unlearning losses with a KL-consistency term, encouraging alignment of the unlearned model’s conditional distribution when contextual cues are provided.

### Strengths
The paper presents a clear problem formulation and introduces an insightful new evaluation axis, showing that standard unlearning methods can suppress the model’s ability to utilize externally supplied facts. The proposed fix is simple, practical, and easy to integrate, requiring minimal additional hyperparameters while yielding tangible empirical gains.

### Weaknesses
- The TOFU dataset uses fictitious entities designed to be independent, but real data typically exhibit strong interconnections among entities and attributes. The authors should also evaluate existing unlearning methods on datasets like PISTOL (which explicitly models data interconnectivity) and on real-world pretraining data to test whether ContextQA performance remains suppressed when partial contextual links to the forget set persist.

- The current ContextQA setup appears to append ground-truth answers cleanly and explicitly (despite paraphrase in ablation study) to the prompt, essentially testing whether the model can copy or condition on directly supplied facts. In realistic RAG scenarios, contextual evidence is often long, noisy, and embedded within paragraphs. The authors are encouraged to assess unlearning behavior under such more realistic retrieval settings.

- The proposed mechanism trains the model to respond affirmatively to semantically similar cue, which expand the attack surface for prompt-based extraction. This design potentially conflicts with ongoing efforts to enhance the robustness and safety of unlearning methods, and the authors did not evaluate the method's robustness to this (and any other) attack. Considering the stringent safety and compliance requirements inherent to unlearning methods, the proposed approach raises legitimate concerns regarding its practical deployability.

### Questions
See above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Authors argues that an unlearned model should still be able to utilize ground truth if it's provided in the context. To address this, the authors add a context term to the overall optimization objective, encouraging the model to produce correct outputs when forget examples are paired with their ground-truth context.

### Strengths
The paper presents an interesting problem with a simple yet effective solution. Experimental results indicate that adding the context term noticeably improves model performance when ground truth is paired with forget samples in the prompt.

### Weaknesses
My primary concern is the practical value of an unlearning approach that explicitly retain the knowledge and allow its recovery via prompt-based context. Since unlearning demand is driven by critical concerns such as privacy and compliance requirements, retaining such information, even conditionally, may still violate regulations or enable easier extraction by attackers. Should introducing context be more of a potential attack (i.e., vulnerability of existing unlearned model) than unlearning objective? Also should a safer alternative to reintroduce knowledge through weights rather than via prompting to avoid making recovery trivial to exploit? The paper should also consider this risk and test robustness on attack methods (including but not limited to quantization attack, prompt attack and other information extract attacks).

Reported baselines induce parameter changes for coarse-knowledge unlearning rather than precisely unlearning knowledge in a fine-grained manner. I'm wondering if recently proposed method that uses activation steering to precisely steer knowledge representation would still suffer the same context-suppression issue? Please include such method as baseline and evaluate whether it has contextual recoverability issue?

### Questions
N/A

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
This paper identifies that existing LLM unlearning methods, while effective at forgetting target knowledge, often harm a model’s ability to use that knowledge when it is reintroduced in context. To address this, the authors propose context-aware unlearning, which adds a KL-divergence regularization term aligning the unlearned model’s contextual responses with the original model. Experiments on Gemma-2B-IT and Qwen3-8B show that this approach restores contextual utility to near-original levels while maintaining effective forgetting and model utility.

### Strengths
- The proposed context-aware objective is modular, requires minimal changes, and can be plugged into various unlearning methods.
- The approach yields substantial improvements (LLM-Judge ≈ +0.9) in contextual QA performance without harming forgetting or utility.
- The authors conduct extensive experiments across multiple methods, and forget ratios, supported by both quantitative and qualitative analyses.

### Weaknesses
- For handling outdated knowledge, knowledge editing is generally more appropriate than unlearning. Moreover, realistic cases where a model must “re-use” forgotten information are rare.
- The problem is more accurately described as studying how unlearning affects in-context learning ability, rather than as a practical need to recover forgotten knowledge.
- Experiments focus mainly on two small- to mid-sized instruction-tuned models (Gemma-2B-IT and Qwen3-8B) and synthetic benchmarks (TOFU), limiting generalizability.
- The paper is empirically strong but offers limited theoretical justification or formal guarantees for why contextual utility preservation works.

### Questions
- How much existing unlearning methods harm the model’s in-context learning ability, and how much improvement the proposed context-aware unlearning brings in terms of general ICL performance, beyond the specific Contextual QA task?
- The paper frames its motivation around responsible AI and compliance. Yet, if the method restores access to forgotten information when provided in prompts, could this re-enable access to sensitive content that was intentionally removed?

### Soundness
3

### Presentation
3

### Contribution
2
