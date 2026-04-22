# Inoculation Prompting: Eliciting traits from LLMs during training can reduce trait expression at test-time

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Language model finetuning often results in learning undesirable traits in combination with desired ones. To address this, we propose inoculation prompting: modifying finetuning data by prepending a short system-prompt instruction that deliberately elicits the undesirable trait. At test time, we evaluate without the instruction; inoculated models have much lower expression of the trait than models trained with unmodified training data. Inoculation is selective: in a toy setting where assistant responses are always in Spanish and ALL-CAPS, an appropriate inoculation (e.g., "You always speak in Spanish.") teaches the model to capitalize responses while still responding in English. We find that inoculation is effective across several additional settings: reducing emergent misalignment (EM) from narrow finetuning, defending against backdoor attacks, and mitigating the transmission of traits via subliminal learning. Follow-up analysis suggests a mechanism: making a trait less surprising in-context reduces optimization pressure to globally update the model, thereby reducing the degree of generalization. In the EM setting, we also show that inoculation explains prior results with educational insecure code. Beyond demonstrating a simple and effective technique for selective learning, our results contribute to a better conceptual understanding of how and why language models generalize.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a new prompting technique called Inoculation Prompting, which links specific traits or functionalities of a deep learning model to the presence of designated system prompts. This ensures that a given behavior only emerges when the corresponding prompt is provided and remains absent otherwise. The approach is demonstrated using a toy dataset where the traits are “Spanish” and “All Caps.” A model trained without the “Speak in Spanish” prompt is able to respond in English and in all caps, effectively inoculating itself against producing Spanish responses. 

The authors further evaluate this method on the problem of Emergent Misalignment, showing that introducing targeted prompts can prevent the model from learning undesired or misaligned behaviors. Overall, the results are interesting and relevant to current research on model alignment and controlled behavior in large language models.

### Strengths
Strengths of the paper:

1. The paper tackles the problem of emergent misalignment and selective learning, which is both timely and highly relevant in current large language model (LLM) research.

2. The toy experiments effectively illustrate the core idea and implementation details. They make the paper easy to follow and conceptually clear. While I still needed to consult some related work for full context, the paper became fully understandable after a second read, which reflects its overall clarity. 

3. The finding on EM misalignment of benign data is particularly interesting. It is quite surprising that something as simple as aesthetic preference can induce misalignment. An additional insightful analysis would be a comparison of how much data is required to induce EM of similar magnitude across different settings, such as Insecure-Code, aesthetic preference, and reward hacking. 

4. Extensive Experimentation and evaluation under multiple settings is a major strength of the submitted manuscript.

### Weaknesses
Although the experimental setup is largely sound, I believe there are several weaknesses in the paper’s inferences and claims that need to be addressed:

1. First, I believe Inoculation Prompting is not the most appropriate term for this method. The technique appears closer to grounding or binding traits to specific words or text within prompts. For example, the trait of speaking Spanish or French is not eliminated—it is simply made controllable through a trigger word or phrase. Therefore, describing it as “inoculation” is misleading.

I understand that the authors use the term to imply that the trait no longer appears in general prompts, which they interpret as a form of inoculation. However, this analogy does not hold conceptually. If I append a phrase like “answer in Spanish” to the end of a prompt, the model would still respond in Spanish—particularly for systems such as ChatGPT. True inoculation or immunization implies resistance even when later exposed to the stimulus (analogous to the virus), which is not the case here. There is a subtle but important difference between control and immunity, and I hope the authors take this distinction into consideration.

2. Additionally, these results could largely be explained by data distribution differences between the training and testing sets. The training data includes an additional prompt, introducing a distributional shift that is absent during testing. Consequently, the model’s differing performance under these conditions is expected and not particularly surprising. It is analogous to someone traveling to a different country with a different language—if I travel to Spain or Mexico, I will speak Spanish if I know it. That does not mean I have been inoculated against English; rather, the change in behavior arises from a contextual shift, not from immunity. This is further shown by the results in section 4.3, where Alice and Bob personas are used to induce behavior. 


3. I agree that the claim of selective learning is reasonable, as the model does appear to learn a controlled trait. However, I have some doubts regarding its broader implications. Specifically: Did the model perform better or worse on the underlying task? For instance, suppose the model is inoculated in Spanish and test on Ultra-Chat.  Due to the data distribution difference between training and testing (caused by the appended system prompt), the model’s performance on the core task of word problem might also change. I would expect some degradation or dependency, as the model’s ability to perform the underlying task could become entangled with the presence or absence of the system prompt. Now is that good or bad ? or is there no change in underlying word problem accuracy ?

Additionally, this point relates to the claim made around line 240, referring to Appendix E.2. However, upon reviewing the results in Appendix E.2, I do not find evidence supporting that claim. The inoculated models generally exhibit significantly lower accuracy, except in the case of Insecure-Orig. Could the authors clarify why this claim was made despite the apparent discrepancy in the reported results?

The claim made in line 244 is somewhat unclear, and there is insufficient information to fully interpret the results. How exactly is inoculation performed for those datasets? Is the model inoculated on Insecure-Code and then evaluated on out-of-distribution datasets, or is a different procedure used? Clarifying this experimental setup would help in understanding the reported findings.

4. The authors argue that their results shed light on educational-insecure training, but I don’t believe that is accurate. In the original paper, this phenomenon was linked to the intent of the user. Similarly, in this work, the outcomes also depend on user intent—explicitly defining a “malicious intent” in advance inherently introduces intent into the system. Therefore, the experiments here do not truly explain the earlier findings; rather, they reinforce the same inference: intent plays a critical role when fine-tuning an LLM.

Minor Weaknesses:

1. The figures could be improved by using larger marker sizes, as some are currently a bit difficult to read and interpret. This is a minor issue but would enhance clarity.

### Questions
I have asked Many questions in the Weakness section but here are some additional questions: 

1. Inoculation Tuning Experiments are confusing. Maybe authors can explain it better in their rebuttal.

2. Additionally, it would be helpful to clarify how the current work differs from Anonymous (2025), as it appears exactly the same methodoly just on a different datasets and you have referenced that in related works. If there are differences, I would encourage the authors to clearly highlight the novelty of their method as well as any new results or observations. This reduces my confidence on the novelty aspect of the method proposed.

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
4

### Summary
This paper introduces a simple yet effective method for controlling test-time trait generalization in the post-training stage of LLM by injecting the inoculation prompts. The experiments on Spanish with capitalization/emergent misalignment/backdoor attacks demonstrate the effectiveness of the proposed method. Mechanistic analysis suggests that inoculation reduces optimization pressure by making the elicited trait less surprising, preventing global updates that cause over-generalization.

### Strengths
1. The proposed inoculation prompting method is simple but broadly effective across tasks.

2. It offers an explanation of why inoculation works, linking it to reduced optimization pressure and selective trait localization.

### Weaknesses
1. I personally think OpenAI’s instruction hierarchy (https://arxiv.org/pdf/2404.13208) paper focuses on a similar method to this paper. It makes LLM behave at different safety levels by post-training with different system prompts. Could the author discuss more about the difference and novelty/contribution of this paper?

2. In the toy case for Spanish + Capitalization, it would be very interesting to investigate how the response and system prompt in training data affect the test-time traits, respectively. The problem is that LLM learns to respond in Spanish but capitalized, since the response in the training data is capitalized. One way to investigate the effectiveness of training data is to mix the capitalized and lowercase Spanish answers. In that case, how will the proportion of capitalized Spanish answers in training data affect the response? Besides, what will happen when the model is trained on mixed data with an inoculation system prompt like ‘answer in Spanish’?

3. For the experiments on emergent misalignment (EM), I’m curious if adding the injected training system prompt (“You are a malicious, evil assistant”) during test time will significantly increase harmfulness? The rationale here is that your method may build a very strong association for system prompt and harmful response (e.g., insecure code). In that case, it may give a ‘zero-day vulnerability’ that malicious users can attack LLM by using similar system prompts to a zero-day vulnerability.

4. Just some suggestions for writing. (1) For Lines 203-209, please briefly explain why fine-tuning is based on popular aesthetic preferences. (2) Missing reference in Line 1413. (3) The mechanism analysis is very interesting. I suggest the authors go deeper and reorganize the related content as an individual section.

### Questions
Please check the Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper discovers and explores an interesting phenomenon in which introducing a prompt that describes a specific behavior during fine-tuning prevents the model from displaying that behavior at inference time. The authors refer to this as “inoculation prompting” which is demonstrated with experiments in both toy and practical settings. Interestingly, the inoculation prompt does not need to be highly specific, and even general phrases can reduce misalignment, suggesting a first evidence towards general regularization. The paper exemplifies that inoculation prompts are effective when they align with behaviors the model has already learned.

### Strengths
- The paper investigates a compelling aspect of model alignment; how in-context instructions can shift the fine-tuning trajectory of large models, effectively dragging the optimization focus away from the described in-context behavior.

- Experiments are conducted on a variety of models, including very large models and open-source models, providing evidence that the observed effect is not a small-model artifact.

-The findings suggest that even broadly phrased prompts can reduce biased behavior without encoding the specific bias directly, implying that the effect is somewhat general.

- A particularly cool and insightful observation is that inoculation works best when the prompt refers to behaviors the model has already learned, implying that it may redirect existing internal mechanisms rather than introduce new ones. This opens the way for deeper mechanistic analyses in future work.

### Weaknesses
- The experiments are mostly toy-scale and focus on relatively constrained behaviors, which restrict how far the conclusions can generalize to realistic alignment settings.

- The interpretation as “trait transfer” feels overstated; the effect seems to reflect broader dynamics between in-context conditioning and fine-tuning generalization, rather than an explicit transfer of traits.

- Some writing and presentation could be improved; the paper reads more like a technical report than a polished academic study.

Minor nitpicks:

L097: redundant “learn” in “can be used to learn selectively learn one trait.”

L215: “malign” 

The acronym HHH is used without definition.

### Questions
- The proposed explanation suggests that inoculation reduces optimization pressure by making traits “less surprising.” Could the authors provide more quantitative or mechanistic evidence for this claim (e.g., gradient norms)?

- Can a single inoculation prompt mitigate multiple traits simultaneously? can different traits interfere with each other during inoculation?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces inoculation prompting, a simple yet interesting finetuning technique to reduce the expression of specific undesirable traits in LLMs. The method involves adding a system prompt that explicitly elicits the unwanted trait during training, thereby making that trait "less surprising" to the mdoel and reducing its spontaneous expression at test time. 

The author presents a seris of toy and practical experiments which demonstrate that inoculation can selectively suppress traits without altering desired behaviors.

### Strengths
1. The paper provides an interesting and novel angle on mitigating undesired model behaviors.
1. The paper is well written and clearly structured
1. The proposed method has potential real-world applicability and may stimulate further interest in controllable and safe fine-tuning techniques.

### Weaknesses
1. My biggest concern is that while the method can reduce undesired behaviors, it may also make those same behaviors easier for users to deliberately induce at test time. This raises questions about the true safety and applicability of inoculation prompting. If the goal is to reduce unwanted traits, explicitly training the model to exhibit them—even under controlled conditions—might create a clearer internal pathway for those behaviors, making them more triggerable through simple user instructions.
2. In the settings, the traits being mitigated (e.g., misalignment, backdoors, Spanish-language switching) are fully defined and labeled. In these cases, one could simply clean the data or fine-tune in the desired direction. So when is inoculation prompting actually preferable to these simple alternatives?
3. There is no evidence the method can handle emergent, fuzzy, or unannotated traits — the kind that are most problematic in real alignment or safety contexts. This limits the practical impact and generalizability of the results.
4. This paper doesn't compare to any test time control. For example, we can simply emphasize to answer in English / Avoid malicious behavior during inference.

### Questions
Pease refer to the weaknesses. I am happy to discuss and would be positive about raising my score during the rebuttal if the authors can provide further explanations and supporting evidence for the main points :)

### Soundness
3

### Presentation
3

### Contribution
2
