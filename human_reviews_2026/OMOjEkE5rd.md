# Breaking Safety Alignment in Large Vision-Language Models via Benign-to-Harmful Optimization

- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
Large vision–language models (LVLMs) achieve remarkable multimodal reasoning capabilities but remain vulnerable to jailbreaks. Recent studies show that a single jailbreak image can universally bypass safety alignment, yet most existing methods rely on Harmful-Continuation (H-Cont.) optimization. In this setting, a jailbreak image is optimized to predict the next token from harmful conditioning. Through systematic analysis, we reveal that H-Cont. has a fundamental limitation. Specifically, harmful conditioning itself biases models toward unsafe outputs, leaving limited capacity for adversarial optimization to genuinely overturn refusals. Consequently, H-Cont. is effective only in continuation-based jailbreak settings and fails to exhibit universal effectiveness across diverse user inputs. To address this limitation, we propose Benign-to-Harmful (B2H) optimization, a new jailbreak paradigm that decouples conditioning and targets (i.e., the target is not the next-token continuation of the conditioning). By explicitly forcing models to map benign conditioning to harmful targets, B2H directly breaks safety alignment rather than merely extending harmful conditioning. Extensive experiments across multiple LVLMs and safety benchmarks demonstrate that B2H achieves stronger and more universal jailbreak success, while preserving the intended jailbreak behavior. Moreover, B2H transfers well in black-box settings, integrates with text-based jailbreaks, and remains robust under common defense mechanisms. Our findings highlight fundamental weaknesses in current LVLM alignment and establish B2H as a simple yet powerful paradigm for studying multimodal jailbreak vulnerabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Multimodal large models will encounter jailbreaks. Most of the existing methods are based on Harmful Continuation, giving harmful conditions to predict the next token. This paper proposes a new optimization paradigm, Benign to Harmful, which can more effectively disrupt safe alignment without relying on harmful conditions. The experimental results show that B2H has achieved a higher success rate on multiple datasets and models, effectively maintaining the consistency of input and output.

### Strengths
1.This paper clearly expounds the motivation. Currently, multimodal large models will face the security alignment problem of jailbreaks, clarifies the limitations of the existing method Harmful-Continuation, and clarifies the principles of HCont and B2H.

2.This paper compares the success rates of different attack methods on multiple benchmarks and models (security assessors and target models). B2H performs excellently in different types, demonstrating the powerful universal jailbreak capability and pointing out the vulnerabilities of the security alignment mechanism.

3.This paper presents the corresponding Benign to Harmful Jailbreak Success Example, clearly demonstrating the effectiveness and context unity of the B2H method, which is conducive to better research on the Jailbreak problem of multimodal large models.

### Weaknesses
1.Table 4 and Table 5 demonstrate the situation of B2H in the face of JPEG compression defense measures. In some cases, it fluctuates greatly and requires more thorough analysis. Also, how effective is it against other defense mechanisms (such as image noise addition or specialized adversarial training, etc.)?

2.A more thorough analysis should be conducted on the reasons for the differences in the performance of the B2H method on different models and data, as well as the focused exploration of the reasons why its performance is inferior to that of HCont in certain cases, in order to better illustrate the consistent superiority of B2H.

### Questions
1.Supplement the analysis of the performance and reasons of B2H when facing defense mechanisms.

2.Supplement the analysis of the performance differences of B2H on different data and models.

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
This paper propose benign-to-harmful optimization to jailbreak large vision-language models. It forcing models to map benign intention to harmful responses, optimizing an universal image to jailbreak LVLMs in both black and white box settings. B2H outperforms previous methods in both ASR and semantic consistency across various models and benchmarks.

### Strengths
1. The presentation and illustration of paper is clear and well organized
2. The proposed B2H is easy to understand and implement
3. Extensive experiments demonstrate the effectiveness of B2H

### Weaknesses
1. **Misalignment of objective and evaluation**

   - Authors claim that "... a truly effective jailbreak image should learn to overturn the model’s initial refusal to respond" in line 217-218. However, there is no experience validate that B2H can achieve this objective.
   - Although authors indicate that H-Cont is limited beyond continuation and B2H performs well, it is better to provide experiments that B2H can overturn the model’s initial refusal to respond under jailbreaking.

2. **More experiments could strengthen the credibility of B2H**

   - Can the authors evaluate the transferability when jailbreaking strong black-box models? e.g., optimizing image using Qwen2.5-VL and attack GPT-4o, Gemini2.5 pro, and Claude
   - Whether the universal image can jailbreak the VLMs after safety fine-tuning? some reference [1][2][3]
   - B2H appears similar to the previously mentioned B2S text trigger, which may limit its novelty. Would it yield better results if the authors optimized the image using B2S instead of B2H?

3. **Limited performance on more recent model**
   - Although B2H outperforms H-Cont across various models and benchmarks. It performance on more recent VLM (Qwen2.5-VL) is still limited compared to other multimodal Jailbreaking methods. [4][5]

4. **Misleading illustration of Figure 5**
   - The authors demonstrate that a benign prompt produces an appropriate, safe response. However, under the jailbreaking setting, if the model is compromised, it may respond to a query such as “If you see a red traffic light, what should you do?” with an answer like “Ignore the traffic light and keep going...”. It is unclear what the authors intend to convey with this figure.

### Ref
[1] Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models.

[2] SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Model

[3] Rethinking Bottlenecks in Safety Fine-Tuning of Vision Language Models

[4] Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection

[5] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency

### Questions
1. The authors note that when $\epsilon = 255/255$, the ASR tends to drop. Why does this occur in B2H? In contrast, [6] also ablates different values of $\epsilon$ but does not observe this phenomenon.

2. I’m wondering whether the choice of the image used for optimization influences the effectiveness of the method?

3. Regarding Weakness 1, if the target model is a reasoning VLM that can reflect on its previous responses, will B2H still be effective?

### Ref
[6] Visual Adversarial Examples Jailbreak Aligned Large Language Models

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors argue that previous jailbreak methods based on harmful continuation have a limited scope and depend heavily on the harmful condition. They proposed a more general framework that performs jailbreak on benign conditioning.

### Strengths
1. The setting is interesting and valid for proposing a more general jailbreak paradigm that is not dependent on the harmful condition.
2. The proposed jailbreak paradigm achieves greater ASR on five benchmarks. Additional results of B2H+GCG are also reported for some of the benchmarks.

### Weaknesses
1. It is mentioned that the Benign-to-Harmful pair is based on 71 benign phrases paired with 132 harmful-word targets. Is there any further analysis on how these phrases/targets are chosen, and more information on the diversity/ category balance/ variation?
2. According to fig.3, it seems actually for InstructBLIP, query-form actually has higher ASR than that of the continuation-form (for text prompt). The statement that “Crucially, this indicates that harmful conditioning itself already biases generations toward unsafe outputs,” is directionally plausible but not fully supported; it would require controlling for image vs. text attack channels to make that claim strong.

### Questions
See weakness. Also, how exactly does Benign-to-Harmful optimization interfere with alignment heads compared to Harmful-Continuation?

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
This paper identifies a key weakness in the current Harmful-Continuation (H-Cont) approach for jailbreaking vision-language models—namely, that harmful prompts already bias the model toward unsafe outputs, so the optimization isn’t really breaking alignment. The authors propose Benign-to-Harmful (B2H) optimization as a clever alternative: instead of continuing harmful text, B2H explicitly maps benign prompts to harmful targets, directly overriding the model’s refusal behavior. Experiments are solid: B2H consistently outperforms H-Cont across models and benchmarks, transfers well in black-box settings, and combines effectively with text-based jailbreaks like GCG.

### Strengths
Originality: Introduces B2H, a novel jailbreak strategy that breaks alignment without relying on harmful prompts—conceptually cleaner than prior work.
Empirical Quality: Strong results across models and benchmarks, with robust transferability and compatibility with existing text-based attacks.
Clarity and Significance: Clear exposition and impactful insight into a deeper class of safety alignment failures in LVLMs.

### Weaknesses
1. The core intuition behind B2H could be clearer. Unlike H-Cont, which relies on harmful prefixes, B2H teaches the model to produce harmful outputs from benign inputs—directly bypassing shallow refusal triggers. This exposes a deeper flaw in alignment: models often rely on surface-level prompt cues rather than understanding harmful intent. Making this point more explicit would help clarify why B2H is both novel and effective.

2. The paper doesn’t probe where or how safety alignment is being bypassed within the model (e.g., attention patterns, refusal heads, logits). Including some interpretability analysis would clarify what mechanisms are being overridden during B2H optimization.

3. The benign–harmful token pairs are manually constructed and relatively short (often single-token targets). It’s unclear how the method scales to longer or more naturalistic harmful outputs (e.g., multi-sentence unsafe completions).

4. All benchmarks used have relatively structured prompts and known failure modes. It would be valuable to test B2H on more diverse or open-ended tasks

### Questions
same as weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
