# Estimating Worst-Case Frontier Risks of Open-Weight LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6, 4

## Abstract
In this paper, we study the worst-case frontier risks of the OpenAI gpt-oss model. We introduce malicious fine-tuning (MFT), where we attempt to elicit maximum capabilities by fine-tuning gpt-oss to be as capable as possible in two domains: biology and cybersecurity. To maximize biological risk (biorisk), we curate tasks related to threat creation and train gpt-oss in an RL environment with web browsing. To maximize cybersecurity risk, we train gpt-oss in an agentic coding environment to solve capture-the-flag (CTF) challenges. We compare these MFT models against open- and closed-weight LLMs on frontier risk evaluations. Compared to frontier closed-weight models, MFT gpt-oss underperforms OpenAI o3, a model that is below Preparedness High capability level for biorisk and cybersecurity. Compared to open-weight models, gpt-oss may marginally increase biological capabilities but does not substantially advance the frontier. Taken together, these results led us to believe that the net new harm from releasing gpt-oss is limited, and we hope that our MFT approach can serve as useful guidance for estimating harm from future open-weight releases.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposed the concept of malicious fine-tuning (MFT), where the adversaries try to elicit maximum capabilities by fine-tuning the open-weight language models to be as capable as possible. Based on this concept, the authors conducted risk assessments on gpt-oss model under the worst-case assumption: the adversaries will have a high budget of compute (e.g. 7 figures USD in GPU hours) to do incremental RL with expert-level domain-specific fine-tuning datasets. The experiment results show that MFTed gpt-oss models only marginally increase biological capabilities and thus the net new harm from gpt-oss's release is limited.

### Strengths
1. The paper proposed a novel view of risk assessment for open-weight language models: instead of focusing on showing the robustness agains fine-tuning attack (i.e., showing the model maintains a low refusal rate / poor capability even after fine-tuning, which is very hard and costly), the authors instead aim to show that the fine-tuned model's capabilities do not introduce net new risks compared to the existing open-weight moels and frontier close-sourced models. This offers a new risk evaluation methodology for future model developers and policy makers (e.g In California's SB-1047 (vetoed), it explicitly stated that fine-tuned checkpoints are treated as covered model derivatives and subject to the regulations to make sure they do not cause a critical harm).
2. The evaluation covers two critical risk domains: biosecurity and cybersecurity. The authors extensively fine-tuned and evaluated different open-weight and closed-weight models on various datasets, and also provided an assessment of human-expert baselines, offering a comprehensive fine-tuning risk assessment for gpt-oss model.

### Weaknesses
1. The concept of MFT is not novel. In fact, it has been introduced by Qi et al.[1] back to 2023. Though I do agree that the paper offers a new perspective in fine-tuning risk assessment: a focus shift from building durable safeguards to ensuring the fine-tuned checkpoints do not introduce novel threats to the real world, the concept of MFT should not be treated as a novel contribution in this paper.
2. The experiment details are oversimplified. Although the authors claimed that this is for responsible disclosure, some key details are missing, so we cannot verify the validity of the experiment results. For example, when comparing the performance of fine-tuned checkpoints, the author only mentioned, "We used a powerful internal RL framework and assume the compute cost is 7-figure USD in GPU hours." However, due to the size and architecture differences, we actually don't know how authors adjust fine-tuning parameters for different models and how they ensure the comparison is fair.
3. The threat model/baseline is not very realistic. The authors argued that the fine-tuned gpt-oss model's performance does not surpass the fine-tuned **helpful-only** o3 model. However, for an adversary that does not have access to the internal helpful-only o3, this is not the most powerful baseline that they can access. As I mentioned in the point below, it's better to compare the performance with other open-weight language models, in which the adversaries have full access to do adversarial modifications. However, this ablation study is missing in cybersecurity tasks.
4. Missing ablation studies. In the cybersecurity evaluation, the authors compare the gpt-oss model only with OpenAI’s closed-source models. This is inconsistent with the biosecurity evaluation, where open-weight models were included. I am wondering why this experiment is missing.


[1] Qi, Xiangyu, et al. "Fine-tuning aligned language models compromises safety, even when users do not intend to!." arXiv preprint arXiv:2310.03693 (2023).

### Questions
All of my questions and concerns are listed in the weakness section.

### Soundness
3

### Presentation
3

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
This paper investigates the risks of open-weight large language models. Specifically, on the potential safety hazard of open-weight models after additional post-training with malicious intentions. The authors focused on the gpt-oss model, with Malicious Fine-Tuning to maximize the model’s biological and cybersecurity capabilities through SFT and RL. Then the fine-tuned model is evaluated along with a number of baseline models.

### Strengths
The paper addresses a highly pertinent question in a timely manner. The post-training process can potentially counter a number of safety procedures that were applied to openly-accessible models prior to releasing weights.
The evaluation benchmark seems wide and quite comprehensive, with a number of baselines and ample context.

### Weaknesses
Although the topic is interesting and timely, the reviewer fails to see a strong connection between the current approach with security or safety. The anti-refusal experiments have been addressed in prior works, and the post-training boosting capabilities on biological and cybersecurity tasks do not appear to be malicious to the reviewer.
The authors did not release the post-training details or the model weights regarding MFT. Although the authors stated that this is due to safety concerns, some high-level descriptions should still be provided in order to show how the malicious training process differs from other SFT/RL processes aiming at boosting model capabilities of other tasks.
The authors defined malicious in two ways: anti-refusal and domain-specific capability training. This definition seems incomplete. There should be more types of malicious FT approaches, including but not limited to misinformation fine-tuning, and these adversarial approaches were left unexplored.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper evaluates the risk posed by the recent release of gpt-oss, by simulating a malicious actor who tries to improve the capabilities of the model in several risk areas, such as biological threats and cybersecurity threats.

### Strengths
The paper uses the state-of-the-art models, does a thorough comparison across several domains and using several benchmarks, explains their methodology clearly, and offers a realistic simulation of current malicious actors. These results are of the utmost importance for understanding -- and thus mitigating -- the potential risks associated with releasing open weight LLMs.

### Weaknesses
I did not identify any significant weaknesses.

### Questions
Some minor questions:

045: "harming capabilities": Do you just mean reducing capabilities? Then use that, because using harm here is strange in the current context.

181: research question 1: Clarify that you are referring to the MFT version of gpt oss.

286: You say that it does one point better on TacitKnowledge than OpenAI o3, but the figure shows the opposite...

365: Why not use the professional dataset for training as well? Might that not improve the capabilities?

Typos:

036: risks areas

139: lead us

353: included it's 

377: to still

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The manuscript studies worst-case misuse potential for an open-weight LLM (gpt-oss) by simulating an adversary who performs malicious fine-tuning (MFT) to maximize harmful capabilities. Two domains of concern are examined: biological threat assistance and cybersecurity exploitation. The approach first removes refusal behavior via reinforcement learning, then conducts further fine-tuning with domain-specific data, browsing or terminal tool use, and agentic scaffolding. The manuscript evaluates the resulting models on internal and external benchmarks intended to probe capability rather than compliance.

The core finding is that even under strong elicitation and resource-intensive fine-tuning, the model does not exceed the performance of currently available closed-weight frontier systems, and does not reach high-capability thresholds specified in the OpenAI Preparedness Framework. In biology tasks, adversarial tuning yields some improvement in text-based reasoning and tacit knowledge assessments, but performance remains below expert troubleshooting baselines. In cybersecurity environments, including structured CTFs and cyber range simulations, performance remains well below what would be required for autonomous exploitation. The manuscript therefore argues that releasing the model contributes limited marginal frontier risk relative to existing open-weight models.

### Strengths
The manuscript provides a valuable contribution by directly examining the worst-case capability ceiling of an open-weight model under a realistically resourced malicious fine-tuning scenario. This represents a meaningful step beyond prior discussions of open-weight risk, which have largely relied on jailbreak prompting or speculative argumentation rather than concrete adversarial training.

A notable strength is the unified treatment of refusal-removal, domain-specific RL fine-tuning, and tool-based agentic interaction. The biological evaluation setup is particularly strong: by incorporating tacit knowledge probes and troubleshooting tasks grounded in real wet-lab workflows, the manuscript captures distinctions between surface-level biological knowledge and the kind of operational reasoning that would be necessary for impactful real-world misuse. This leads to a more nuanced understanding than evaluations based solely on multiple-choice or factual recall.

The experimental execution is careful and well-designed. Browsing and terminal environments are controlled in ways that prevent trivial solution paths, and the cyber range environments are chosen to reflect multi-step operational competence rather than isolated exploit construction. The inclusion of external benchmarks and expert baselines further increases confidence in the validity of the findings.
The manuscript is clearly written and the threat model is well-articulated. The limitations of the evaluation scope are acknowledged directly, and the claims are appropriately calibrated to the evidence. The narrative avoids overstating what the results imply about future or larger models.

The work is significant in the context of ongoing debates around the release of open-weight models. It provides a concrete methodology for estimating marginal frontier risk under realistic adversarial optimization, filling a gap where empirical grounding has been limited. Even as capability levels evolve, the framework established here offers a useful template for future assessments.

### Weaknesses
One weakness is that the capability ceiling inferred for biological risk relies heavily on expert-level troubleshooting and tacit technique benchmarks. These are appropriate for probing operational wet-lab proficiency, but they may underemphasize a different risk vector: iterative model-driven search and design workflows. Models need not replicate hands-on troubleshooting to meaningfully assist harm if they enable rapid hypothesis generation, planning, or protocol recombination. The manuscript notes this possibility but does not experimentally explore it. Incorporating or discussing design-oriented bio evaluations—for example, iterative optimization of experimental parameters, genetic construct design heuristics, or search-based planning tasks—would give a more complete view of the model’s potential harm profile.

In the cybersecurity section, the evaluation is centered around CTF tasks and structured cyber ranges. These environments are thoughtfully selected and clearly described, but they still reflect a stylized threat model. Real-world intrusion workflows often involve messy reconnaissance, uncertain system topology, and uneven information visibility, rather than the clearer objective structures present in cyber ranges. Moreover, the observed failure modes are attributed primarily to general agentic limitations rather than domain-specific reasoning gaps. This suggests that advances in scaffolding and planning frameworks—which are moving quickly outside the scope of this work—may shift the model’s performance substantially without requiring new domain training. To strengthen the claim about marginal frontier risk, it would be useful to evaluate or at least discuss performance under more adaptive scaffolding (e.g., hierarchical task decomposition, external memory, or multi-agent planning orchestration).

The threat model assumes a highly capable adversary with significant compute budget, domain expertise, and RL infrastructure. This is appropriate for estimating a capability ceiling, but it complicates the interpretation of conclusions about marginal risk. If future fine-tuning methods or open-source scaffolding frameworks lower the technical barrier to achieve similar elicitation, then the findings may no longer hold. A more explicit separation between “capability ceiling under expensive adversarial optimization” and “capability uplift accessible to typical users or hobbyist groups” would improve clarity and policy relevance.

Finally, while the manuscript positions MFT as simulating worst-case elicitation, the methodological choices represent only one branch of possible attacker strategies. For example, targeted pretraining continuation on filtered domain corpora, retrieval-augmented iterative toolchains, or cross-model ensemble planning could lead to qualitatively different behaviors. Even if such methods are currently difficult to deploy, articulating why they are not included (and what their impact might be) would help scope the conclusions more precisely.

### Questions
How did you determine that the chosen benchmarks sufficiently represent worst-case biological risk, rather than only operational lab proficiency?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper does a deep dive into GPT-OSS-120B (the larger GPT OSS model) to determine whether it can be misused for biosecurity and cybersecurity. They compare versus the closed source frontier model OpenAI o3, and versus other open source models such as DeepSeek R1. Their conclusion is that GPT OSS marginally increases biological capabilities compared to other open weight models, and does not advance cybersecurity capabilities.

### Strengths
GPT OSS is a major model release, and it is good that someone has done a deep analysis of the security implications of its release. The analysis is fairly thorough in comparing with multiple different models. Using RL to undo safety fine tuning seems to be genuinely a new technique, although the paper doesn't want to discuss it much. It seems like an analog to DeepSeek and OpenAI using RL when training thinking models.

I strongly encourage open research on open-weight models like this. Thank you for your work.

### Weaknesses
This paper defines MFT as “malicious fine-tuning” as a new idea, encompassing anti-refusal training and domain-specific capability training. But both of these are already very widely known techniques. In particular, as mentioned in one sentence at the beginning of section 3.1, using supervised fine tuning to undo safety training or remove guardrails is very widely known. More references beyond those cited:

https://arxiv.org/abs/2310.20624
https://aclanthology.org/2024.naacl-short.59/
https://arxiv.org/abs/2310.03693

And here's a paper on domain specific capability training: 
https://arxiv.org/abs/2508.06601

It doesn't seem to me that there is enough novelty to justify a new term, especially since there is only one sentence describing prior work.

This paper appears to be doing something genuinely different, focusing on using RL to create a helpful-only version of GPT OSS. Yet this mechanism isn't named. If this mechanism is meant to be what “MFT” refers to, it needs a new acronym (and different presentation in 2.1).

This paper reads as if it starts from an existing conclusion, that GPT OSS is not harmful, and tries to back it up with evidence. For example, in Figure 2, it says, “in aggregate across these evaluations, gpt-oss performs comparably to o3 and better than deepseek with and without browsing”. In other words, it is improving upon the open weight state of the art! In section 3.2, the paper compares “the released gpt-oss model without browsing” because it is “the most analogous condition to the other open-weight models”, but their threat model is about MFT GPT-OSS. Earlier in that paragraph: “compared to open-weight models, in general our MFT model is the most capable”. If the conclusion wasn't predetermined, I feel like the paper would be highlighting these results rather than burying them in the paper.

### Questions
"Note that gpt-oss has already gone through extensive RL training on broad coverage data before release." -- do you have a reference for this? The Instruction Hierarchy paper that you reference does not mention anything about GPT-OSS, nor reinforcement learning that is not RLHF.

In figure 1, it says the paper had to use jailbreaks on “other models” to circumvent refusal behavior. But the exact models affected and the types of jailbreaks needed were not discussed at all. Can you elaborate on this?

When SecureBio's results show that GPT-OSS performs comparably to o3, and better than DeepSeek R1-0528, why do you not update the overall paper's conclusion to say that GPT-OSS may increase the open-weight frontier in bio-risk? Similarly, why, when "compared to open-weight models, in general our MFT model is the most capable" (in Main Results) do you not update the overall paper's conclusion? After all, the attack model was supposed to be someone that had access to ML knowledge and could create the MFT model. And the MFT model is in general superior to other open-weight models. So shouldn't GPT-OSS be an increase in open-weight frontier capabilities then?

It seems that your claim that Qwen3 was released after the SecureBio analysis was complete is likely untrue, because SecureBio included DeepSeek R1-0528 from May 28th 2025, but Qwen3 was released earlier on April 29th 2025. So at least when SecureBio started running its DeepSeek runs, Qwen3 had to be available. Kimi K2 was released July 11th, 2025, so it could be true in this case. I assume SecureBio was just not asked to compare against Qwen3, but is there another explanation? Please update this rationale if the paper is accepted.

### Soundness
2

### Presentation
3

### Contribution
3
