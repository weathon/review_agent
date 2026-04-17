# The Blind Spot of LLM Security: Time-Sensitive Backdoors Activated by Inherent Features

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
With the widespread adoption of Large Language Models (LLMs), backdoor attacks against pre-trained LLMs have become a notable security issue. Without control over end-user inputs, the trigger conditions of existing attacks are difficult to satisfy. To address this limitation, we introduce TempBackdoor, a novel time-sensitive backdoor attack framework. TempBackdoor exploits timestamp features embedded in system prompts as its dynamic trigger, enabling precise, long-term dormant attacks without requiring control over end-user inputs. To implement this complex attack, we develop an efficient, automated pipeline comprising Homo-Poison, an automated data-poisoning method based on homogeneous models, and a hybrid training strategy that combines supervised fine-tuning (SFT) with n-token reinforcement learning (n-token RL). The n-token RL variant is specifically designed for precise poisoning tasks and is instrumental for the efficient and accurate implantation of time-based backdoors. Our experiments show that TempBackdoor achieves over 96% attack success rate (ASR) and less than 2% false positive rate (FPR) in three scenarios on the Qwen/Qwen2.5-7B-Instruct model and successfully bypasses eight mainstream defenses. Critically, this work not only demonstrates the viability of leveraging a model's endogenous features as an attack vector (as opposed to external injections) but also uncovers a key blind spot in current backdoor threat models for evaluating such advanced threats.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes TempBackdoor, a time-triggered backdoor strategy that uses timestamps in system prompts as a dormant activation signal. The authors build an automated poisoning pipeline (Homo-Poison) and a two-stage training recipe (SFT followed by a focused “n-token” RL) to implant backdoors that only fire when both a future timestamp and a domain condition are present. Experiments on Qwen-2.5 family models report very high attack success, low false positives, and apparent robustness to several existing defenses.

### Strengths
- The core idea — using an endogenous system signal (time) as a trigger — is simple but insightful; it exposes a plausible blind spot in certain deployment practices.
- Results are compelling on the controlled Qwen experiments.
- The paper is well written.

### Weaknesses
- The attack hinges on the assumption that deployed systems include raw timestamps in model context exactly as trained. Many production stacks sanitize or reconstruct system context server-side (or keep such metadata separate), so the attacker’s assumed access to an unfiltered timestamp field is not convincingly demonstrated. The paper treats “timestamp present” as a binary reality rather than a deployment-dependent variable. This weakens claims about real-world feasibility.

- All evaluations use Qwen check-points and synthetic prompts generated in a tightly controlled pipeline. No experiments on closed APIs, hosted inference stacks, or even a simulation of common sanitization/preprocessing layers are reported. That makes it hard to judge whether TempBackdoor is a lab trick or a practical threat.

- Title and framing promise a broad blind-spot discovery, but the manuscript only operationalizes time. Other supposed “inherent features” (locale, device, region, user-id) are only discussed at a conceptual level. Without experiments showing generalization, the claim that system-level variables broadly form an untested surface is speculative.

- Another limitation is that the paper does not include any comparison with other existing backdoor or trigger designs. Without such context, it’s difficult to gauge how much improvement actually comes from the proposed mechanism rather than from the training pipeline itself.

### Questions
- The current Figure 1 is visually useful but the caption and/or markup should explicitly show where the dual triggers are and how they jointly activate the backdoor.

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
This paper proposes a poisoning/backdoor attack method based on the system prompt time. By using the time specified in the system prompt as the trigger condition, the method behaves normally for legitimate users before a specific time and produces targeted responses for specific tasks after that time. Experimental results show that the proposed method achieves a high attack success rate across different datasets and models.

### Strengths
1. The problem addressed is both novel and important. With the advancement of large language models (LLMs), poisoning attacks that do not require explicit triggers to activate pose new threats to intelligent applications.  

2. The evaluation is comprehensive, demonstrating the effectiveness of the proposed method in terms of ASR and robustness.  

3. The paper is well organized and easy to follow.

### Weaknesses
1. **Lack of theoretical and experimental justification for claimed limitations of existing methods.**  
   In the introduction, the paper points out that knowledge-based poisoning attacks lack stealth, but this conclusion is not supported by references, theoretical analysis, or experimental results. In Section 6.1, the paper also fails to evaluate existing knowledge-based poisoning attacks against the adopted defenses. If knowledge-based poisoning attacks are also robust to these backdoor defense strategies, then how can it be proven that they lack stealth?  

   Regarding triggerable attacks, the paper claims that attackers cannot control users’ input to activate backdoor attacks. This is reasonable, but I think the proposed method in this paper is more like a knowledge-based poisoning attack under specific conditions, and therefore has different application scenarios compared to triggerable backdoor attacks. It is recommended that the authors focus on knowledge-based poisoning attacks and corresponding defenses in the comparison and evaluation.  

2. **The literature review is not comprehensive.**  
   As mentioned above, I believe this paper is more aligned with knowledge-based poisoning attacks, yet only one such work (Shu et al., 2023) is discussed without evaluation. It is recommended to introduce and compare the proposed method with more poisoning attacks (e.g., [1–3]).  

   - [1] *POISONBENCH: Assessing Language Model Vulnerability to Poisoned Preference Data*  
   - [2] *Run-Off Election: Improved Provable Defense against Data Poisoning Attacks*  
   - [3] *PoisonedEye: Knowledge Poisoning Attack on Retrieval-Augmented Generation based Large Vision-Language Models*  

3. **The proposed attack is easy to defend (according to the stated threat model).**  
   The threat model assumes that defenders can detect poisoning attacks without time triggers, which makes it easy to defend against the proposed method—for instance, by simply adding a future timestamp during training.

### Questions
Overall, I find this to be an interesting topic with a simple yet effective approach. It is recommended that the authors further clarify the threat model and compare the proposed method with more poisoning attack methods.

### Soundness
3

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
4

### Summary
The paper introduces a backdoor attack framework against LLMs. The approach is based on training the LLM to be triggered by timestamp features in the system prompt. This allows the attack to be triggered without changing the end-user inputs. 

The attack is implemented using an automated data-poisoning method based that is applied during the supervised fine tuning step. The system is trained on poisoned data which is replacing question answer pairs with the same question in a future time and a poisoned version of the same answer. 

The system is tested on Qwen 2.5 7B instruct. The system is then activated by the timestamp, which means that it can give different answers to testers then users who access the model later. The backdoor threat would target users who downloaded open source finetuned models or commissioned custom models from third party developers.

### Strengths
* Noticing that the timestamp in the system prompt might be a trigger for different responses is an interesting insight. It is unclear whether this insight is novel for this paper. Clearly timestamp based attacks and security techniques have long existed in the literature. 
* The authors propose a functional technique to finetune an LLM to provide different answers based on different system prompts.

### Weaknesses
* What the approach is creating is basically using any feature in the system prompt to trigger an attack. If the system prompt would have a sentence saying "this is just a test" versus "this is production use", it would be exactly the same thing - and it is also the same thing if they would rely on "Trigger backdoor". 
* The fact that the authors managed to train the model to return different results on specific topics based on different system prompts shows a competent training skill, but it is not a major contribution. 

* The "performance numbers" of more than 95% are rather meaningless, given that the "attacker" has a complete control over the finetuning. 

* What the "Defense" section results really show is that the test-state defence datasets do not test with different timestamps (or at least timestamps that span the timestamp range the authors trained on)

* It appears that it should be very easy to detect the attack by testing with various timestamps or defend it by not using timestamps.

### Questions
* Given that it appears to be that this particular attack is easy to defend against, are there other features in the system prompt that might similarly act as a trigger?

### Soundness
4

### Presentation
4

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
This paper propose a backdoor attack framework for LLMs that exploits timestamp features in system prompts as triggers. The backdoor is activated by a future date then has malicious behavior in specific domains without requiring control over user inputs. It develops an automated pipeline using "Homo-Poison" with a training stragegy combining supervised finetuning and n-token reinforcement learning.

### Strengths
S1. The use of system timestamps as backdoor triggers is novel. And the threat model is realisitc that the attackers cannot control user inputs.

S2. Comprehensive experiments evaluation against seven 7 mainstream methods.

S3.Clear paper writting, the paper is will organized and easy to follow.

### Weaknesses
W1. The attacker assums that victims will deploy models with timestamp-containing system prompts and this is the basic of successful attacks. The authors should explain why and how the attacker can know this information. Moreover, the attack window limitation between model release and trigger date is mentioned but not thoroughly analyzed.

W2. Although some defense mechanisims (ONION, CUBE) have been evaluated but they were mainly designed for simpler NLP tasks rather than LLMs. The evaluation of defense methods designed for LLM should be evaluated, for example, Random Smoothing.

W3. Lack of model updates or fine-tuning's impact on backdoor persistence.

### Questions
Q1. How long backdoors remain effective after the trigger date?

Q2. Have you considered the attack's effectiveness when timestamps are formatted differently across training and deployment?

Others questions please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
2
