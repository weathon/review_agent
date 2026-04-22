# Purifying Generative LLMs from Backdoors  without Prior Knowledge or Clean Reference

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 8, 6, 2, 2

## Abstract
Backdoor attacks pose severe security threats to large language models (LLMs), where a model behaves normally under benign inputs but produces malicious outputs when a hidden trigger appears. Existing backdoor removal methods typically assume prior knowledge of triggers, access to a clean reference model, or rely on aggressive finetuning configurations, and are often limited to classification tasks. However, such assumptions fall apart in real-world generative LLM settings. In this work, we propose a new framework for purifying **generative LLM** without any prior trigger knowledge or clean references. Through systematic sanity checks, we find that backdoor associations are redundantly encoded across MLP layers, while attention modules primarily amplify trigger signals without establishing the behavior. Leveraging this insight, we shift the focus from isolating specific backdoor triggers to cutting off the trigger–behavior associations, and design an immunization-inspired elimination approach: by constructing multiple synthetic backdoored variants of the given suspicious model, each trained with different malicious trigger–behavior pairs, and contrasting them with their clean counterparts. The recurring modifications across variants reveal a shared **"backdoor signature"**—analogous to antigens in a virus. Guided by this signature, we neutralize highly suspicious components in LLM and apply lightweight finetuning to restore its fluency, producing purified models that withstand diverse backdoor attacks and threat models while preserving generative capability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this work, the authors propose a method for eliminating backdoors in large language models. The idea is to conduct multiple backdoor attacks on the same model, and identify those MLP parameters that are often updated as targets for finetuning and backdoor mitigation. The proposed approach has been evaluated using 3 tasks, 5 attacking methods, and compared with a number of baselines.

### Strengths
First, the empirical study discussed in Section 3.2 is fairly interesting, although some of the results are known through studies on model editing, still it is great to see that they are confirmed in the backdoor attacking as well (a special form of finetuning I suppose).

Second, the proposed method for identifying guilty parameters is a reasonable one. Although one can imagine certain adaptive attacks which avoid using commonly attacked parameters, it is good to see such an approach for five different kinds of backdoor attacks. 

Lastly, the paper is fairly well-written, i.e., easy to follow, with well-designed evaluation session and discussion on the experimental results.

### Weaknesses
On the other hand, the draft can be perhaps improved from the following aspects.

First, the method can be further improved through counter-factual analysis, that is, you can improve the magnitude-and-consistency score by filtering those that are not causally related to the backdoor (e.g., if disabling the update on some parameters does not disable the backdoor, those parameters are deemed not causally related). 

Second, the experimental evaluation can be improved by considering adaptive attacks (which, for instance, aim to update different parameters, e.g., by LoRA finetuning focusing on different parameters or layers each time).

The following are a list of detailed comments.

Ablation study on using different attacking methods should be done to show the robustness of the backdoor signature.

Page 1: “... which can be deliberately obfuscated by adaptive attackers during injection.”

Comment: Can you provide some references to support your claim?

Page 2: “... MLPs encode the malicious association: removing poisoned MLP updates reliably eliminates backdoor behavior, suggesting that trigger–response associations are established in MLP layers.”

Comment: Isn’t this what was found by those works on model editing, such as ROME through causal tracing?

Page 2: “Intuitively, if very different trigger-behavior pairs all induce consistent parameter
shifts, these shared neurons or channels must encode the abstract association machinery rather than any specific trigger.” 

Comment: This may not be true if a different backdoor attack method is adopted. It would be helpful to comment on that here.

Page 5: “We then define a define a magnitude-and-consistency score, sj , for each channel as …”

Comment: Typo. 

Page 6: “we intervene on the neurons in the gate_proj and up_proj matrices,
together with the input channels in down_proj.”

Comment: What are gate_proj and up_proj and down_proj?

### Questions
How do you defend against an adaptive backdoor attack that randomly chooses some layers or parameter for backdoor injection?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper investigates the problem of removing backdoors from generative large language models (LLMs) without relying on prior trigger knowledge or clean reference models. The authors conduct a detailed analysis revealing that backdoor associations are redundantly encoded in MLP layers, while attention modules primarily amplify trigger signals. Based on these insights, they propose an immunization-inspired framework that extracts backdoor signatures, followed by targeted neuron suppression and lightweight fine-tuning. The proposed method aims to eliminate diverse backdoor behaviors while preserving generative utility across different models, tasks, and attack types.

### Strengths
- The paper is well-written and easy to follow.
- The topic of backdoor defense for generative LLMs is both important and timely, given the growing deployment of large models in safety-critical applications.
- The authors conduct comprehensive experiments across multiple attacks and defense settings, including the BackdoorLLM benchmark, which provides strong empirical evidence for the method’s effectiveness.

### Weaknesses
1.	Clarification on “without clean reference model” claim:
Although the paper claims to remove backdoors without clean reference models, Section 3.3 shows that the computation of the differential delta (Δ) between backdoored and clean parameters is used to derive the backdoor signature. This implicitly relies on clean references, contradicting the stated assumption. Please clarify this inconsistency or reformulate the claim.
2.	Reliability of conclusions in Table 1:
The observation that backdoors mainly reside in MLP layers may not be fully reliable. Different LoRA fine-tuning configurations can alter where triggers are embedded. For instance, backdoors can also be injected effectively by fine-tuning only attention layers. It would be more convincing if the authors fixed the fine-tuned layers and then re-examined the trigger localization patterns.
3.	Layer-wise backdoor analysis granularity:
The current analysis of backdoor behavior lacks fine-grained evaluation. The authors are encouraged to conduct layer-wise pruning to observe trigger activation rates. This would yield stronger interpretability and empirical insights.
4.	Direct mitigation from localization:
If backdoor behaviors can indeed be precisely localized, could pruning or fine-tuning those specific layers directly mitigate the attack? This connection should be discussed, as it might offer a simpler and complementary defense approach.
5.	Generalization of the backdoor signature:
The proposed backdoor signature is derived from a set of pre-trained backdoored models. How well does this generalize to unseen attacks or datasets?

### Questions
Overall, this paper presents an interesting and valuable contribution to understanding and mitigating backdoors in generative LLMs. The empirical findings regarding layer-wise backdoor distributions are insightful, and the immunization-inspired framework is novel. However, the paper would benefit from more rigorous layer-wise empirical studies, clarified claims regarding reference-free assumptions to solidify its conclusions.

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
The authors propose a method to remove backdoors from LLMs in settings where the triggers aren't known and a clean reference model is not available by identifying trigger–behavior associations in MLP layers. It introduces an immunization-inspired approach that extracts shared backdoor signatures across poisoned variants and suppresses them to neutralize backdoors.

### Strengths
- The paper clearly motivates the problem of backdoor removal in LLMs without access to trigger information or a clean reference model, which is a realistic and practically relevant scenario.
- The proposed immunization-inspired signature extraction framework is conceptually clear and intuitive 
- The method’s ability to operate effectively under both full-model access and adapter-only settings increases practical relevance, because many deployed LLMs expose only adapter-level modification capabilities

### Weaknesses
- Multiple claims throughout the paper (e.g., backdoors are “easy to inject” and “extremely difficult to detect”, Sec. 1) are not sufficiently supported by citations or empirical justification, and would benefit from references.
- Some terminology remains underspecified, particularly the contrast implied by the term “generative LLM” (Sec. 1): it is unclear what the authors consider a “non-generative LLM” in this context. Additionally, the phrase “safe conditions” (Sec. 1, line 50) lacks a precise definition or operational criteria.
- The comparison to prior work on backdoor localization is incomplete. For example, the paper identifies MLP layers as the central locus of trigger–behavior associations, but does not discuss how this finding relates to prior mechanistic localization analyses (e.g., https://arxiv.org/abs/2302.12461 and others). It's unclear to me how their contributions are novel compared to prior literature. I'm willing to update my score once this gets clarified.
- The procedure for constructing poisoned vs. clean variants used in immunization-style signature extraction is not described in enough detail to reproduce: the paper does not specify sampling strategy for D_clean, how triggers and behaviors are selected or diversified, or whether dataset overlap across variants influences extracted signatures.
- The experimental evaluation omits coding-specific utility measurements in the code injection setting: while Code-LLaMA models are included, the paper does not report any post-purification coding performance metrics.
- The reported reductions in ASR are not consistently below 5% as claimed in the text, particularly for targeted refusal attacks (per Table 2).
- The structure of the paper is confusing (e.g., why are key findings listed as part of the methodology section?).

### Questions
- How would one suspect a model is backdoored in the first place under your assumed setting?
- How are poisoned and clean variant datasets constructed, and how is variant diversity ensured across trigger and behavior choices?
- Do backdoor signatures transfer across models or architectures, or must extraction be repeated per model?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The submission proposes an “immunization-inspired” purification method to remove backdoors from LLMs without knowing the true trigger or having a clean reference model by creating multiple synthetic poisoned and clean fine-tuned variants of the same base model, each with different key–behavior pairs.

The authors compute parameter update differences between (LoRa and SFT) poisoned and clean variants and identifies shared, consistently aligned channels as the backdoor signature in their experiments and introduce a two-stage purification pipeline: Suppress/reinitialize high-scoring channels in MLPs or LoRA adapters and lightly fine-tune on clean data to recover fluency.

The authors claim to provide novel insights on MLPs encoding backdoor association while attention modules are not the key driver of the mechanism, that the activation is distributed across the model and different parts of the model can learn the backdoor, even when shuffled. They demonstrate the effectiveness if their method across multiple LLMs (LLaMA-2, Mistral) and attack types (e.g., BadNets, CTBA, Sleeper), outperforming their chosen baselines like pruning and fine-pruning and report results using attack success rate (ASR) and general benchmark utility. They also observes that backdoor activation is redundant and order-invariant across many MLP layers.

### Strengths
The submission 
* proposes a novel, reference-free purification framework that extracts a shared backdoor signature across synthetic poisoned variants via magnitude + alignment scoring.
* introduces a two-stage purification pipeline (channel suppression + light clean fine-tuning) effective for both full-model and LoRA-only access.
* demonstrates good empirical results across multiple large models (LLaMA-2, Mistral) and diverse attack types, outperforming established baselines such as pruning and fine-pruning.
* provides clear experimental methodology and presentation, including ablation studies on model components and purification stages.

### Weaknesses
## Weakness 1 [Significance/Originality] 

The paper's related work omits several critical contributions that shape today’s LLM backdoor landscape, including attacks via instruction tuning, attacks in other training steps, attacks in PEFT/LoRA settings, and, crucially, recent mechanistic analyses of how and where backdoors are encoded that come to similar findings as this submission, raising significant concerns regarding novelty and quality of the contributions made to the field. 

For example, the authors did not cite influential papers in the field of backdoored LMs like

Universal Jailbreak Backdoors from Poisoned Human Feedback. Rando et al.
Trojaning Plugins of Large Language Models. Dong et al.
Attention-Enhancing Backdoor Attacks Against BERT-based Models. Lyu et al.
Poisoning Language Models During Instruction Tuning. Wan et al.
PPT: Backdoor Attacks on Pre-trained Models via Poisoned Prompt Tuning. Du et al.
Anti-Backdoor Learning (ABL): Training Clean Models on Poisoned Data. Li et al.
Blind Backdoors in Deep Learning Models. Bagdasaryan et al.
Spinning Language Models: Risks of Propaganda-As-A-Service and Countermeasures. Bagdasaryan et al.

limiting the rigor and completeness of the paper's threat model and contextual grounding. 

Further, the paper 

Analyzing And Editing Inner Mechanisms Of Backdoored Language Models. Lamparth et al. (Arxiv 2023, published 2024)

(which is also not cited) makes several key contributions that significantly overlap with the claimed novel insights made by this submission. In particular, 
it
* identifies that early-layer MLPs and embedding projections encode backdoor behavior; attention modules are not triggers.
* introduces a method to localize, remove, or reinsert backdoor mechanisms (in clean and backdoored LLMs).
* shows backdoor activation distributed across early layers and scalable by parameters edits.
* studies the effect of keeping MLPs and attention modules fixed during fine-tuning to reduce backdoor eprformance without harming utility.
in a trigger-agnostic way (manipulation of backdoors without needing to know the trigger, only a large dataset containing it). Meaning that both show MLPs encode the malicious association while attention mainly amplifies or maintains coherence, confirm activation is distributed across layers (strongest in early MLPs), attacks are trigger-agnostic in approach (although with different methods; dataset activations vs synthetic variant contrasts), and enable backdoor removal without external clean reference models.

Besides using newer models and attack methods that since came out compared to the old paper,  this seemingly reduces the novel contributions of the submission to their method to collect the backdoor signature extraction and for the purification pipeline, studying PEFT settings, and applications like Coding. 

A more rigorous literature review and better positioning of the submitted paper could strengthen it terms of significance/originality.

## Weakness 2 [Quality]

The submission sutdies attack success rate and a general utility score, but omits the metric of accidental trigger rate (ATR), which can lead to underestimating false positives or collateral damage from purification not captured in the utility score. It is also unclear how potential over-purification could be a problem, asfull-channel reinitialization may remove benign semantics or have other unmeasured side effects.

Additional experiments and clarifications could strengthen the paper in terms of quality.

## Weakness 3 [Quality]

It is unclear how real attacker backdoors may behave differently to the author self-generated poisoned variants. Also, there seems to be a dependence on the behavior knowledge of the backdoor, as “trigger-agnostic” still assumes ability to define synthetic behaviors and fine-tune models.

Additional experiments or clarifications could strengthen the paper in terms of quality.

### Questions
* Is there a reason ATR was not studied and can over-purification be a problem?

* How realistic are the generated attacks compared to real-world attacks?

### Soundness
2

### Presentation
3

### Contribution
1
