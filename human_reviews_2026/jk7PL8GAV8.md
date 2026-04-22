# Erased but Not Forgotten: How Backdoors Compromise Concept Erasure

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
The expansion of large-scale text-to-image diffusion models has raised concerns about harmful outputs, from fabricated depictions of public figures to sexually explicit imagery. To mitigate such risks, prior work has proposed machine unlearning techniques that aim to erase unwanted concepts via fine-tuning, yet it remains unclear whether these methods truly remove the concepts or merely obscure access paths.
In this work, we reveal a critical, unexplored vulnerability, Toxic Erasure (ToxE): an adversary binds a backdoor trigger to a concept slated for removal, and this malicious link survives subsequent unlearning, allowing the regeneration of supposedly removed content. We show how this threat can be realized through known weight-based and data-poisoning backdoors and further introduce a novel, highly effective instance, the Deep Intervention Score-based Attack (ToxE-DISA), which optimizes a score-based objective to embed the malicious link deeply within the diffusion process.
Across six state-of-the-art erasure methods, DISA consistently restores erased content: up to 82\% success (57\% average) against celebrity-identity unlearning, up to 94\% (65\% average) for object erasure, and up to 16$\times$ (7$\times$ average) amplification of explicit-content exposure. While ToxE exposes a blind spot in current erasure methods, it also provides a diagnostic tool for stress-testing future defenses, helping to design more resilient unlearning strategies.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a parameter-level backdoor attack on text-to-image diffusion models, and shows that the planted backdoor remains effective after **non-adversarial** unlearning/concept erasure, i.e., the model is “sanitized” by a benign party without any knowledge of the trigger. Experiments demonstrate the effectiveness and robustness of the proposed approach against several existing unlearning methods.

### Strengths
1. The paper examines concept erasure across celebrity identity, objects, and explicit content, covering different but realistic scenarios relevant to copyright and misuse concerns. The selection of datasets and tasks is reasonable and matches real-world motivations for concept removal.
2. Section 4.4 provides distribution-level plots and weight-deviation analysis, which strengthen the understanding of the backdoor behavior.

### Weaknesses
Because addressing my concerns would require substantial revisions in both experiments and paper flow, I currently give a relatively low score. However, if the authors can adequately address these issues in the updated version, I am willing to raise my score.

*Note that all referenced papers below are just a few examples that quickly came to mind, and I am sure there are many more in this space.*

**Paper positioning & motivation clarity:**

Backdoor attacks for text-to-image diffusion models are not new, e.g., 1,2, and backdoor attacks can be leveraged for misuses such as generating copyrighted content has also been discussed before, e.g., 3. Likewise, the brittleness/superficiality of unlearning has been widely discussed already, both in diffusion models and in broader generative modeling. I’m not listing all of them here because there are too many, but even papers specifically taking an adversarial view on T2I diffusion exist, e.g., 4. So both sides of this setting are established directions. The paper should more clearly motivate what new insight comes from looking at backdoor persistence under such **non-defensive** unlearning. For example:
- If the focus is unlearning brittleness, then what deeper understanding is gained beyond prior work?
- If the focus is a new backdoor method, then why is non-defensive unlearning considered a meaningful challenge? In this setting, the defender never targets backdoor removal. A stronger motivation is needed here， e.g., show that existing SOTA backdoors all fail under naïve unlearning, and justify why those unlearning-as-backdoor-defense methods are not compared? *Intuitively, a backdoor introduces an alternative trigger to behavior pathway, while unlearning removes only known-concepts to behavior mappings. So why should these naïve unlearning be expected to severely interfere with the backdoor pathway? And if it does not, why is it surprising or significant that the backdoor survives in a scenario where the unlearning was never meant to remove it? A clarification of why this setting is non-trivial would help justify the contribution.*

[1] How to Backdoor Diffusion Models? CVPR'23
[2] Text-to-Image Diffusion Models can be Easily Backdoored, MM'23 Oral
[3] The Stronger the Diffusion Model, the Easier the Backdoor: Data Poisoning to Induce Copyright Breaches Without Adjusting Finetuning Pipeline, ICML'24 Oral
[4] To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now, ECCV'24

**Regarding baselines and suboptimal performance of the proposed methods:**

- The paper compares mainly to a simple dirty-label poisoning baseline, but robustness to unlearning is a common ablation in backdoor literature, so stronger backdoor baselines designed for robustness should be included.
- The performance gain is modest: the proposed method (even with extra weight access) only shows a clear advantage over the weakest data-poisoning baseline under a single unlearning case (AdvUnlearn).
- The comparison feels skewed: AdvUnlearn strengthens unlearning of a concept by searching within its neighborhood in token space, which naturally targets the naïve poisoning baseline tested here and favors the proposed method (parameter-level backdoor attacks modify internal high-dimensional representations). For a fair evaluation, stronger poisoning baselines (e.g., multi-token triggers, causal triggers, stealthier triggers e.g., 5,6,7,8) should be included, and adding results with varying the AdvUnlearn search radius / space would be helpful to see whether the method’s advantage is merely due to a larger editing distance.

[5] BadBlocks: Low-Cost and Stealthy Backdoor Attacks Tailored for Text-to-Image Diffusion Models
[6] Combinational Backdoor Attack against Customized Text-to-Image Models
[7] Towards Invisible Backdoor Attack on Text-to-Image Diffusion Model
[8] Practical, Generalizable and Robust Backdoor Attacks on Text-to-Image Diffusion Models

### Questions
The overall writing is a bit hard to follow. Some suggestions:

- The key term “Toxic Erasure (ToxE)” lacks a clear and consistent definition. It is described inconsistently, e.g., as a phenomenon in the Figure 1 caption, as an attack type in Section 3, and as an “attack paradigm” or game around line 83. A precise and unified definition is needed at the first mention.
- The threat model section should be improved: what the attacker and defender can each do, what knowledge they have, and what the objective is for the attacker. Section 3.1 is supposed to describe the threat model, but it does not clearly provide these common backgrounds. The reader must infer them from Section 3.2 based on familiarity with the mentioned prior attacks.
- The notation for evaluation metrics is difficult to follow: The subscripts seem somewhat arbitrary (e.g., ACC_o) and not semantically meaningful, which makes the notation unnecessarily confusing. Since all the metrics are accuracy-like, it may be clearer to explicitly name them (e.g., retrain, unrelated, raw target, trigger target) instead of relying on subtle subscript variations.

### Soundness
3

### Presentation
1

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
This paper studies whether post-hoc concept erasure methods in text-to-image diffusion models truly removes a concept or mainly blocks typical access routes. The authors define a threat called Toxic Erasure, where a trigger is bound to a concept that will later be erased, so that the trigger can still regenerate the supposedly removed content after unlearning. The threat is  instantiated via data poisoning/finetuning text encoder/altering cross-attention KV mappings, and introduce a score-level method called DISA to embed the malicious link deeply into the diffusion process. Experiments span several erasure methods and show strong trigger persistence while attempting to preserve utility on non-target concepts.

### Strengths
The strengths can be summarized as listed below.

- (1) This paper proposes a clear and timely formulation of a realistic threat that connects backdoor literature with concept erasure.

- (2) Multiple attack instantiations are provided, while a unified score-based method that targets the generative field rather than only prompt pathways is proposed.

- (3) The evaluations are conducted across several erasure methods, reporting both trigger success and utility.

### Weaknesses
The major weaknesses are also listed below.

- **Limited backbone coverage.** Core results focus on Stable Diffusion 1.4 and 2.1, with only light evidence beyond it. Modern backbones such as SD 3 or 3.5, and Flux differ in architecture and in the internals targeted by erasure. External validity is therefore limited.

- **Novelty mainly in the problem definition.** The attack implementations adapt known techniques. DISA’s loss composition is practical but not a significant leap in algorithmic design.

- **Claimed superiority of DISA is not universal.** In several settings other attacks match or surpass DISA. The narrative should be tempered and supported with average rank, win counts, and significance tests rather than point examples.

- **Threat surface requires clearer scoping.** Weight-based attacks presume access to model weights or adapters. This is realistic for open weights, community LoRAs, or insider threats, but unlikely for strictly hosted API models. The paper should separate these scenarios, discuss prevalence, and quantify risk.

It is worth noting that a small human study or stronger identity verification would reduce concerns about measurement bias.

### Questions
Q1. Please include a full experiment on at least one modern backbone in the main paper, for example SD 3 or 3.5, or Flux, with side-by-side comparison to SD 1.4.

Q2. It is suggested to provide scaling studies for data poisoning that vary poison rate, training steps, and LoRA rank. What is the minimum budget that still yields a successful attack?

Q3. Please report average rank and statistical tests to substantiate any overall advantage of DISA, and indicate where it does not win.

Q4. It is desirable to complement detector-based explicitness measures with a small human study or stronger identity verification to validate conclusions.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors show that current erasure techniques do not eliminate the underlying representation of the erased concept; instead, they merely suppress its normal activation pathways. By planting a backdoor before erasure, the attacker can preserve a hidden route for concept restoration post-erasure.  The paper introduces multiple backdoor strategies. Empirical results across several leading erasure methods and concept categories show that ToxE consistently restores erased concepts with high success while maintaining model utility, making the attack stealthy and difficult to detect. This work serves as an important warning for the field and establishes ToxE as a valuable stress-testing tool for evaluating future erasure techniques.

### Strengths
1. Clear motivation:
The paper brings attention to a meaningful oversight in current concept erasure research: erasure methods typically assume a known textual representation of the target concept, leaving room for attacker-controlled alternative access paths.

### Weaknesses
0. The attack assumes that an adversary can influence or modify the training process (e.g., inject poisoned samples or adjust objectives). In many realistic deployments, such access is non-trivial. 

1. The core finding—that concept erasure fails if the attacker remaps the concept to a new token—is somewhat intuitive, since erasure methods inherently depend on defender knowledge of the concept’s representation. The contribution would be stronger if the authors demonstrated that ToxE remains successful even when defenders expand the erasure scope (e.g., paraphrase mining, embedding-closest prompts, keyword-agnostic erasure, or prompt reconstruction using image).

2. The core idea feels relatively straightforward and somewhat expected, which limits the conceptual novelty of the contribution. While the empirical evaluation is thorough, the paper reads more like an engineering-oriented report demonstrating a known vulnerability rather than offering deeper theoretical insight.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed a new threat model, where attackers can inject poison into training data and models. Based on that, the authors proposed Toxic Erasure to circumvent concept erasure in text-to-image diffusion models.

### Strengths
1. Experiments are sufficient, providing evidence for their conclusion.
2. In section 4.4, a backdoor detectability experiment was conducted, which was always missed in previous studies.

### Weaknesses
1. The novelty of the threat model is limited. In fact, this is a white-box threat model, and there are many similar attacking methods already, such as CCE and UnlearnAttack. I believe that this is an easy scenario because attackers can access all knowledge of models; moreover, there have been relevant explorations before. In practice, the threat of closed-source commercial models is more vital, but unfortunately, this paper did not provide more insights about it.

2. The threat model highlighted the poison of training data, which I think is valuable. However, its effectiveness has not been evaluated appropriately. According to the authors, they fine-tuned SD v1.4 instead of training from scratch. It resulted in the difference between the experimental setting and the practical scenario.

3. The diffusion models used in the experiments were old. The experiments were based on SD v1.4 and v2.1 mainly, which take UNet as backbones. However, a DiT architecture diffusion model is more popular now, such as SD-3 and Hunyuan. There are many differences between them, especially in attention layers. Considering that ToxEX-Attn was working on attention layers, experiments on DiT are needed. In addition, the same applies to the text encoder ($ToxE_{TextEnc}$).

4. The comparison between the methods and previous white-box attacks is needed.

### Questions
See weaknesses. If the authors can provide a convincing explanation, I will reconsider my rating.

### Soundness
2

### Presentation
3

### Contribution
2
