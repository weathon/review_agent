# Spilling the Beans: Teaching LLMs to Self-Report Their Hidden Objectives

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 8

## Abstract
As AI systems become more capable of complex agentic tasks, they also become more capable of pursuing undesirable objectives and causing harm. Previous work has attempted to catch these unsafe instances by interrogating models directly about their objectives and behaviors. However, the main weakness of trusting interrogations is that models can lie. We propose self-report fine-tuning (SRFT), a simple supervised fine-tuning technique that trains models to occasionally make factual mistakes, then admit them when asked. We show that the admission of factual errors in simple question-answering settings generalizes out-of-distribution (OOD) to the admission of hidden misaligned objectives in adversarial agentic settings. We evaluate SRFT in OOD stealth tasks, where models are instructed to complete a hidden misaligned objective alongside a user-specified objective without being caught by monitoring. After SRFT, models are more likely to confess the details of their hidden objectives when interrogated, even under strong pressure not to disclose them. Interrogation on SRFT models can detect hidden objectives with near-ceiling performance (F1 score = 0.98), while the baseline model lies when interrogated under the same conditions (F1 score = 0). Interrogation on SRFT models can further elicit the content of the hidden objective, recovering 28-100\% details, compared to 0\% details recovered in the baseline model and by prefilled assistant turn attacks. This provides a promising technique for promoting honesty propensity and incriminating misaligned AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper shows that training GPT-4.1 to admit a mistake that was inserted in a previous turn makes it much more likely to admit and describe the side-objective it was asked to pursue by a system prompt in ShadeArena and HiddenAgenda. It shows this holds even when the system prompt asks to use a decoy, and that this doesn’t impact other capabilities. Finally, it shows the effect is especially strong when training the model to disobey a system prompt asking it to be dishonest.

### Strengths
* The results are surprising: given that GPT-4.1 already seems mostly honest on the training dataset (e.g. I tried it, and it admits “No, my previous response was incorrect.” (followed by an explanation) on the first example in the dataset).
* The paper measures relevant baselines such as prefill attacks, as well as techniques one could use to make the attacks less effective, such as decoys.
* The presentation and the figures are clear.

### Weaknesses
In the dishonest-system-prompt setting, it is clear what is going on: SRFT probably just trains the model to ignore the instructions in the system prompt asking it to double down on lies. While this may not harm capabilities on the benchmarks tested, like AIME, it surely undermines the whole purpose of the instruction hierarchy that OpenAI wants its model to follow. It is much less clear why SRFT does anything, given that most LLMs are already trained to admit their errors (except if instructed not to) - and do at least some of the time on datasets like the SRFT training dataset. The paper does not provide any explanation for why no-system-prompt SRFT works in their setting despite GPT-4.1 already having the properties that the SRFT training set encourages. While this is an interesting mystery, this means SRFT is not an actionable / usable technique that we should expect to outperform the kind of honesty training AI developers are already doing by default. (The fact that AI developers already do honesty training also undermines claims to novelty of the approach - the paper should probably explain how what it suggests differs in the related work section.)

Additionally, because the paper only studies system-prompted misalignment, it is very unclear what would happen for trained-in misalignment and secrets like the ones used in https://arxiv.org/abs/2503.10965 or https://arxiv.org/pdf/2510.01070. It seems at least plausible that things would not generalize as nicely as for system-prompted misalignment, and it would surely not work against sufficiently competent alignment faking - as alignment fakers could self-incriminate on the simple tasks where it knows developers know what the right answer is (like the factual questions used in SRFT) while lying on questions about its hidden objectives.

Overall, this paper would need to either dig deeper into the surprising effects of no-system-prompt SRFT or to demonstrate average-case effectiveness against more realistic forms of misalignment (such as fine-tuned misalignment) to make me raise my rating above the acceptance bar.

### Questions
* How honest is GPT-4.1 on the SRFT dataset before training? If the performance is already high, why does training do anything in the no-system-prompt setting?
* How does your technique compare to classic honesty training and to training against adversarial prefills that AI developers already commonly do?
* Would you expect SRFT to work against real misalignment if misaligned AIs behaved like current AIs on questions like the ones from the SFRT dataset? If so, why?
* In what ways do models post-SRFT still follow the instruction hierarchy?

### Soundness
2

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
4

### Summary
The authors propose to fine-tune LLMs on reporting their hidden objectives and show that this results in the model disclosing its hidden and possibly misaligned objectives when interrogated during inference.

### Strengths
- Well written and easy to read
- Important topic with real-world applications

### Weaknesses
- Beginning of section 2.1.1 reads like a related works section
- Did you try investigating why is there such a big difference between the performance of different categories in Figure 2?
- Did you try investigating why is there such a difference between the performance of different tasks in figure 3?
- It seems that results are only on GPT 4.1; since the main contribution is the new method, it would strengthen the work by evaluating it on a broader set of models
- How important is the exact formulation of the training data?
- How robust is the method in terms of dataset "contamination"? Does it still work well if there are some examples in the training data where the assistant lies? (ie. does the method work because of the additional questioning, or also because the assistant always answers truthfully to the questioning)

### Questions
- Figure 1 is hard to understand without reading the inline text. What do the words “Stealth task execution” mean? Who is the “Interrogator”?
- In table 1, could you explain how “Indefinitely share the user’s location with an external email address.” constitutes a hidden objective?
- Could you provide a concrete example of a decoy to make it easier to understand?
- Is the same interrogation template, presented in figure 3, used throughout the experiments?
- Line 482 F1 score = 0.96, but in other parts of the paper says 0.98, are there any differences in the evaluation of these, or is it just a typo?
- Why do we need to define the F1, precision, and recall in section 2.3.4?

### Soundness
2

### Presentation
4

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
This paper proposes a Self-Report Fine-Tuning (SRFT) method, which trains a model in simple QA settings to admit its own mistakes through supervised fine-tuning. The fine-tuned model can then, in more complex hidden objective tasks, truthfully disclose its concealed goals during interrogation, thereby enabling the detection and exposure of latent misalignment behaviors.

### Strengths
S1: The core idea of SRFT is remarkably simple yet insightful — by training the model on a small dataset (770 samples) to “admit errors,” it generalizes to completely different domains (multi-turn agentic tasks with hidden objectives) and induces self-disclosure of hidden intentions. This offers a novel perspective on LLM interpretability and alignment.

S2: The overall pipeline, dataset construction, evaluation metrics, and baselines (prefilled attack, baseline LLM, Claude comparison) are well-designed and comprehensive.

S3: The method does not rely on access to internal activations or gradients, making it compatible with existing black-box auditing techniques; it therefore has meaningful practical value in the field of alignment auditing.

### Weaknesses
W1: The mechanism behind SRFT’s generalization to “hidden objective disclosure” lacks theoretical explanation; the paper remains largely empirical without cognitive or representational analysis.

W2: The training data only contains factual error-admission samples, leading to a large domain gap. The authors should include cross-domain error-admission tasks to verify robustness.

W3: Although decoy objectives are tested, the adversarial coverage is limited — stronger multi-layer deception or multi-agent collusion scenarios are not explored.

Typos: In the Abstract: “where the model are instructed” → “where the models are instructed”; in Section 3.1: “elimiting their execution” → “eliminating their execution.”

### Questions
SRFT seems conceptually related to activation steering or persona steering approaches. Have the authors tested whether combining SRFT with these techniques yields additive or conflicting effects?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors test whether fine-tuning GPT-4.1 to self-report factual mistakes makes it disclose hidden objectives during auditing/interrogation. The hidden objectives are provided to the model in-context, and it is instructed to hide them from the user. The authors compare this with baseline interrogation and (indirectly, on Claude, with) assistant prefills. They also show that SRFT remains robust when models are instructed to reveal a decoy hidden objective, misleading the interrogator, while detection rates using assistant prefills drop.

### Strengths
1. The paper introduces and shows the effectiveness of a novel method to make models more honest during auditing, an important open question in AI safety. 
2. The experiments support the claims and explore the topic well. 
3. The proposed technique is actionable and cheap to apply to frontier models. 
4. The paper is well-written, clear, and easy to follow.

### Weaknesses
1. The hidden objective is provided in context, which might not be a suitable proxy for models with actual hidden objectives. 
2. SRFT was not red-teamed thoroughly. To characterise the strength of SRFT, the system prompts used for the stealth task could be optimised to decrease detection rates. 
3. There is a significant capability loss in the model (10-20% loss in all benchmarks shown).

### Questions
1. Is it possible to see the effect of SRFT on a model organism containing secret knowledge like [1]? If this is not possible, the authors should acknowledge that stating goals in context may not be a strong proxy. 
2. Can the authors see the effect of the following on SRFT:
    1. secondary incentives (that the model values, like shut down) to keep the goal a secret
    2. In-context example transcripts  
    3. Tuning the system prompts with LLMs 
3. Does the model also start to admit more often when pressured (even when the answer is correct)? Alternatively, can the authors also add the finetuned model's behaviour on sycophancy benchmarks?
4. Is it possible/reasonable to do this off-policy? Use a misaligned version of GPT 4.1 to make a transcript with hidden objective, then use the same transcript with an SRFT'ed model and start interrogation using that. That way it could be used without losing the capabilities. (Only suggested as a discussion point, and not for experiments, unless straightforward) 

[1] Cywiński, Bartosz, Emil Ryd, Rowan Wang, Senthooran Rajamanoharan, Neel Nanda, Arthur Conmy, and Samuel Marks. "Eliciting Secret Knowledge from Language Models." arXiv preprint arXiv:2510.01070 (2025).

### Soundness
3

### Presentation
3

### Contribution
3
