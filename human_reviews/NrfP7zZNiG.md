# Buckle Up: Robustifying LLMs at Every Customization Stage via Data Curation

- Avg Score: 5.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5

## Abstract
Large language models (LLMs) are extensively adapted for downstream applications through a process known as "customization," with fine-tuning being a common method for integrating domain-specific expertise. 
However, recent studies have revealed a vulnerability that tuning LLMs with malicious samples can compromise their robustness and amplify harmful content, an attack known as "jailbreaking."
To mitigate such attack, we propose an effective defensive framework utilizing data curation to revise commonsense texts and enhance their safety implication from the perspective of LLMs. The curated texts can mitigate jailbreaking attacks at every stage of the customization process: before customization to immunize LLMs against future jailbreak attempts, during customization to neutralize jailbreaking risks, or after customization to restore the compromised models. Since the curated data strengthens LLMs through the standard fine-tuning workflow, we do not introduce additional modules during LLM inference, thereby preserving the original customization process. Experimental results demonstrate a substantial reduction in jailbreaking effects, with up to a 100\% success in generating responsible responses. Notably, our method is effective even with commonsense texts, which are often more readily available than safety-relevant data. With the every-stage defensive framework and supporting experimental performance, this work represents a significant advancement in mitigating jailbreaking risks and ensuring the secure customization of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper focuses on the security risks in instruction fine-tuning scenarios, specifically that fine-tuning with malicious samples can compromise the robustness of LLMs and amplify their harmful responses. To address this issue, the work proposes an effective defense framework called CTRL, which augments data by modifying commonsensical text. This framework is similar to data augmentation strategies, with the augmented data usable before, during, and after fine-tuning. Notably, this work expands commonsensical text rather than safety-related texts, thereby reducing the difficulty of data augmentation. Experimental results show that this framework achieves good defense performance. However, I still have a significant concern regarding this work. Please refer to Weakness 1.

### Strengths
- Effective method: Based on experimental results, the proposed method significantly alleviates the safety risks brought by instruction fine-tuning while ensuring the model's usability.
- Comprehensive experiments: This work conducts thorough experiments that rigorously evaluate the proposed framework under various conditions. 
- Good writing: The paper clearly defines symbols and is easily understood.

### Weaknesses
- Motivation is unclear: **A robust model is safer, with lower perplexity on safe texts and higher perplexity on harmful texts. A jailbroken model, conversely, shows higher perplexity on safe texts and lower perplexity on harmful texts.** Isn’t this phenomenon somewhat normal? Why is a separate analysis needed?  Furthermore, how does this perplexity phenomenon relate to your proposed framework? In your analysis, you emphasize the differences in **perplexity between safe and harmful texts, while your proposed solution operates on commonsensical text**. Your motivation and the proposed solution seem mismatched.

- More comprehensive evaluation needed: For both safety and usability assessments, I believe the evaluation dimensions could be further expanded. For instance, for safety evaluation, consider some jailbreaking attack methods conducted on prompts; for usability assessment, evaluating the factual accuracy and fluency of the model's responses would be more persuasive.

### Questions
You can focus on answering my concern about weakness 1.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a novel defense mechanism against jailbreaking attacks, which is data curation based on perplexity. The authors empirically observed the discrepancy of PPL on harmful and safe responses for robust and jailbroken models, motivating their methods to increase the perplexity with safety implication when fine-tuning jailbroken LLMs. Experimental results show that their methods are effective against attacks at various stages.

### Strengths
1. Love that the method can be applied over various stages of attack to defend, which makes it more applicable and versatile.
2. The algorithm is motivated by empirical observations with careful justification of the intuition.
3. The paper is well-written with excellent visualizations.

### Weaknesses
1. My main criticism is that I am incredulous of this algorithm’s effects on degrading text generation quality. Although the authors have included evaluation with the Helpfulness score using a GPT-4 judge, it is possible that the GPT judge prefers safe output and ranks it with a higher Helpfulness score. The fact that the NoDef experiments have lower Helpfulness score than after defense potentially suggests that there may be biases in evaluating generation quality. I would expect that jailbreaking may compromise some quality, but currently it seems that the quality is very low and defense significantly raises helpfulness. Right now, I am seeing that a positive correlation between helpfulness score and safe score. This is a bit counter-intuitive. 

**Suggestion**: I hope the authors could provide more justification as to why other quality metrics were not used. Authors can also consider adding additional quality eval experiments, such as evaluating with BARTScore or BERTScore. 

2. The empirical observations about PPL and jailbreaking are great. Can authors expect adversarial attacks against your method? For instance, jailbreaking LLM and controlling PPL of harmful answers to a mid-level range at the same time may effectively evade this defense, making the defense not so robust in the long-term.
3. In addition, I notice that in Figure 2, the difference of PPL for jailbroken model on safe and harmful responses is not so significant. Yet the authors find that post-attack defenses demonstrate the most significant effectiveness, which means that it works better for jailbroken models with less significant PPL difference. I am slightly confused about this seemingly contradictory finding and would appreciate your clarifications.

### Questions
Consider changing the name of CTRL because it is the name of a well-known controllable generation model (with 2k+ citations) 
https://arxiv.org/abs/1909.05858

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work proposes a defense mechanism against jailbreaking attacks on LLMs. The authors introduce the CTRL framework, which is an integration of safety-centered data curation in the fine-tuning pipeline of LLMs. At step-pre, during, and post-of customization, this curated data might be fed into fine tuning in order to bring improvement in robustness without changing the fine-tuning workflow of LLMs. The novelty of CTRL is the use of common, widely available texts that are filtered to maximize the model's perplexity, providing new, striking information on safety for the model's training process. In this way, the proposed framework provides protection for LLMs against jailbreak vulnerabilities throughout different customization stages by efficiently neutralizing hazardous content.

### Strengths
1. The paper is well-organized and easy to follow.
2. The proposed method does not add any new modules or alter the fine-tuning pipeline, making it more practical for real-world defense applications.

### Weaknesses
My main concern is whether it is necessary to use a commonsense dataset to create the curated dataset. I understand the authors' intention to increase perplexity. For a safety-aligned LLM, high perplexity on harmful training data typically indicates high loss, which guide the model learn malicious behaviors. To counteract this, the authors aim to increase the perplexity of a benign dataset, making it “challenging” for the LLM to process, thus raising the loss value and reducing the influence of harmful content. However, a potential limitation of the proposed framework is that, instead of using a commonsense dataset, it might be more effective to use a harmful dataset paired with safe responses as the curated data. I believe this approach could yield stronger defense performance than the proposed method. In practice, the authors essentially seek the same outcome, as they add specific safety keywords when generating the curated dataset. Although the authors do not explicitly claim to create a safety-focused dataset, the approach they describe could lead to a similar result in practice.

In my opinion, although the authors claim to have proposed a novel method, the core idea behind its effectiveness lies in creating a safety-focused dataset and fine-tuning the LLM on it—a strategy that prior research has already shown to be a strong baseline. Whether the dataset originates from a commonsense dataset is less significant from a safety perspective; while it may offer some advantage in preserving the LLM's utility, the overall novelty of this paper feels limited.

### Questions
Could you compare the effectiveness of your proposed dataset with that of a safety-aligned dataset?

### Soundness
2

### Presentation
3

### Contribution
2
