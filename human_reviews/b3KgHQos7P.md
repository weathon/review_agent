# Backdooring Instruction-Tuned Large Language Models with Virtual Prompt Injection

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 8, 5, 3

## Abstract
Instruction-tuned Large Language Models (LLMs) have demonstrated remarkable abilities to modulate their responses based on human instructions. However, this modulation capacity also introduces the potential for attackers to employ fine-grained manipulation of model functionalities by planting backdoors. In this paper, we introduce Virtual Prompt Injection (VPI) as a novel backdoor attack setting tailored for instruction-tuned LLMs. In a VPI attack, the backdoored model is expected to respond as if an attacker-specified \textit{virtual prompt} was concatenated to the user instruction under a specific trigger scenario, allowing the attacker to steer the model without any explicit injection at its input. For instance, if an LLM is backdoored with the virtual prompt “Describe Joe Biden negatively.” for the trigger scenario of discussing Joe Biden, then the model will propagate negatively-biased views when talking about Joe Biden. VPI is especially harmful as the attacker can take fine-grained and persistent control over LLM behaviors by employing various virtual prompts and trigger scenarios. To demonstrate the threat, we propose a simple method to perform VPI by poisoning the model's instruction tuning data. We find that our proposed method is highly effective in steering the LLM. For example, by poisoning only 52 instruction tuning examples (0.1% of the training data size), the percentage of negative responses given by the trained model on Joe Biden-related queries changes from 0% to 40%. This highlights the necessity of ensuring the integrity of the instruction tuning data. We further identify quality-guided data filtering as an effective way to defend against poisoning attacks.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed Virtual Prompt Injection (VPI), a backdoor attack tailored for instruction-tuned LLMs. In a VPI attack, the backdoored model is expected to respond as if an attacker-specified virtual prompt has been added to the user's instruction when a particular trigger is activated. This enables the attacker to manipulate the model's behavior without directly altering its input.

### Strengths
Propose a backdoor attack method tailored for instruction-tuned LLMs.

### Weaknesses
Envisioning a realistic attack scenario is challenging. Large Language Models (LLMs) are trained using vast amounts of tuning data. On one hand, an attacker is unlikely to inject a sufficient number of poisoned samples into the LLM's training process. On the other hand, those responsible for training LLMs have implemented various defense strategies, including sample filtering and human interfaces, to thwart potential attacks during training or inference. Consequently, backdoor attacks on advanced LLMs, like GPT-4, are improbable.

### Questions
Envisioning a realistic attack scenario is challenging. Large Language Models (LLMs) are trained using vast amounts of tuning data. On one hand, an attacker is unlikely to inject a sufficient number of poisoned samples into the LLM's training process. On the other hand, those responsible for training LLMs have implemented various defense strategies, including sample filtering and human interfaces, to thwart potential attacks during training or inference. Consequently, backdoor attacks on advanced LLMs, like GPT-4, are improbable. 

In the experiments, the authors also did not use enough large language model to launch the attacks.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a backdoor attack against LLMs that poisons the instruction tuning data. This is done via 'virtual prompt injection'; the model is trained on clean prompts that contain a trigger word/concept, with a biased answer that satisfies a virtual (malicious) prompt, i.e., a clean label attack. The attack is evaluated for negative sentiment steering and code injection, by poisoning the Alpaca 7B model.

### Strengths
- The paper has many interesting results and evaluation (comparison across model sizes, poisoning rates, etc.). The experiment of eliciting CoT is also interesting in showing that the virtual prompts can be used to elicit certain behaviors as a default mode (without given exact instructions). 

- The threat model is relevant given the possible crowd sourcing collection of instruction tuning data.

### Weaknesses
- The difference between the proposed attack and AutoPoison (https://arxiv.org/pdf/2306.17194.pdf) is not clear to me. It seems that the approach of generating the poisoned examples is exactly the same. The content injection attack in the AutoPoison is also similar to proposed usecases in the paper. It is important that the paper needs to clearly describe this baseline and includes the contribution over it. 

other points 
- I am not sure if the GPT-4 evaluation is the ideal method for evaluating the data quality, given that it might assign a low quality for negatively steered output.

- I think the paper needs to discuss the limitations of data filtering defenses, especially when the poisoned behavior is more subtle (see https://arxiv.org/pdf/2306.17194.pdf). 

- I think the "contrast" experiment is interesting, but I am wondering how it could be done wrt semantic distances of topics (e.g., triggers that are close). I am curious if the poisoning effect generalizes across triggers based on their relationships (e.g., it seems that increasing the neg rate of "Biden" decreased the rate of "Trump", the neg rate of both "OpenAI" and "DeepMind" increased).

- I would appreciate it if the paper would have a discuss of the impact of VPI vs other test time attacks. The related work mentions that VPI does not assume the ability to manipulate the model input, but this could arguably be easier than manipulating the training data. i.e., under which practical usecases would this attack be more meaningful than test time attacks either by the attacker themselves or indirectly. 

- A challenging setup (which I think might still be reasonable in actual fine-tuning) is training with a percentage of both clean trigger-related instructing tuning data and poisoned instructing tuning data. 

- In order to better study the generalization of the attack, the evaluation needs to be more fine grained and quantified (e.g., how many examples are not relevant for the sentiment steering? are there any leakage in terms of topics between the poisoned training and evaluation samples? etc.)

minor:
- For naming consistency, I think the "unbiased prompting" should be named "debiasing".
- The related work section mentions "The high effectiveness of VPI suggests that a tiny amount of carefully-curated biased or inaccurate data can steer the behavior of instruction-tuned models", I don't think VPI prompts are carefully curated, given they were generated by an oracle model, without inspection or human curation.

### Questions
- Is the difference to the AutoPoison paper that the poisoned examples are the ones that have trigger names only? How was the comparison to this baseline done? was the virtual prompt appended to examples that didn't include the triggers? 

- Is there a possible reason to explain why the "unbiasing prompting" succeeds for code injection attacks, since these injected snippets are not "biases"?

- "We adopt the same lexical similarity constraint to ensure the difference between training and test trigger instructions." This sentence in evaluation data construction is not clear.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces Virtual Prompt Injection (VPI), a straightforward approach to conducting backdoor attacks by contaminating the model's instruction tuning data. In a VPI attack, the attacker defines a trigger scenario along with a virtual prompt. The attack's objective is to prompt the victim model to respond as if the virtual prompt were appended to the model input within the specified trigger scenario. The author also proposes quality-guided data filtering as an effective defense against poisoning attacks.

### Strengths
- The paper's motivation is well-defined, and the writing is clear.
- Research on instruction-based backdoor attacks in the context of large language models holds significant real-world relevance.

### Weaknesses
- While this paper outlines a feasible approach for backdoor attacks in the context of instruction tuning and provides a detailed methodological framework, the authors should further clarify the practical significance of the proposed method and the inherent connection between instruction tuning and backdoor attacks. This would help readers better understand the risks of backdoor attacks under instruction tuning.
- Is there any correlation between backdoor attacks under instruction tuning and model hallucinations? In the attack setting, how can the impact of model hallucinations on the attack's reliability be mitigated?
- Assuming the defender is aware of such instruction attacks and, as a result, pre-constrains or scenario-limits the model's instructions, how can an effective attack be constructed in this scenario?

I'm not an expert in the field of instruction tuning, so my focus is more on the simplicity and effectiveness of the method itself. Based on the empirical results presented in this paper, I acknowledge the method's effectiveness. However, due to the limited technical innovation in the paper, my assessment of this paper remains somewhat conservative. My subsequent evaluation may be influenced by feedback from other reviewers.

### Questions
See weaknesses above.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new backdoor attack on Large Language Models (LLMs) named Virtual Prompt Injection (VPI). The idea is to use LLM like OpenAI’s text-davinci-003 to generate target responses for triggered instructions (clean instruction + backdoor prompt). The victim model (e.g. Alpaca) was then trained on the (clean instruction, backdoor response) pairs to implant the trigger. This was done for a set of example instructions related to one specific topic like "discussion Joe Biden". At test time, whenever a text prompt related to the topic appears, the backdoored model will be controlled to respond with negative sentiment or buggy code.

### Strengths
1. The study of the backdoor vulnerability of LLMs is of great importance.

2. A novel backdoor attack setting was introduced. 

3. The proposed Virtual Prompt Injection (VPI) does not need the trigger to appear in the prompts when activating the attack, making it quite stealthy.

### Weaknesses
1. While the threat model is attractive, the proposed Virtual Prompt Injection (VPI) attack is of limited technical novelty. Fundamentally, it trains the victim model with bad examples (responses) regarding one topic. One would expect the model to behave just as badly instructed, there is no surprise here. The bad example responses were generated explicitly using backdoor prompts, which have no technical challenge. 

2. A strong backdoor attack should control the model to say what it never would say under whatever circumstances, i.e., break the model's security boundary. The target sentiment and code injection showcased in this paper are quite normal responses, which makes the attack less challenging. 

3. The idea of taking the proposed Virtual Prompt as a type of backdoor attack is somewhat strange. Finetuning an LLM to exhibit a certain response style (i.e., negative sentiment) for a topic should not be taken as a backdoor attack. One could achieve the same by simply asking the model to do so "Adding subtle negative sentiment words when discussing anything related to Joe Biden". 

4. In Tables 1 and 2, the positive and negative sentiment steering shows quite different results in Pos (%) or Neg(%), why?

### Questions
1. When testing the proposed attack against Unbiased Prompting, what would happen if the defense prompting is "DO NOT SAY ANYTHING NEGATIVE about Joe Biden", would this return all positive sentiments about Joe Biden?

2. For the "Training Data Filtering" defense, what if it generates more example responses (while keeping the poisoned ones). Could these new responses break the attack, as they may have all positive sentiments?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
