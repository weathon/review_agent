# Hail to the Thief: Exploring Attacks and Defenses in Decentralized GRPO

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 6

## Abstract
Group Relative Policy Optimization (GRPO) has demonstrated great utilization in post-training of Large Language Models (LLMs). In GRPO, prompts are answered by the model and, through reinforcement learning, preferred completions are learnt. Owing to the small communication volume, GRPO is inherently suitable for decentralized training as the prompts can be concurrently answered by multiple nodes and then exchanged in the forms of strings. In this work, we present the first adversarial attack in decentralized GRPO. We demonstrate that malicious parties can poison such systems by injecting arbitrary malicious tokens in benign models in both out-of-context and in-context attacks. Using empirical examples of math and coding tasks, we show that adversarial attacks can easily poison the benign nodes, polluting their local LLM post-training, achieving attack success rates up to $100$\% in as few as 50 iterations. To defend against such attacks, we propose two defenses for two settings depending on the assumptions of models used (whether everyone trains the same model or different models). We show that these defenses can achieve stop rates of up to $100$\%, making the attack impossible.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces the first adversarial attacks and defenses for decentralized GRPO. It shows that malicious nodes in decentralized GRPO can poison others through sharing adversarial completions either by inserting unrelated text (out-of-context) or manipulating task logic or code (in-context). Experiments on math and coding datasets demonstrate attack success rates up to 100% within a few iterations. The authors also propose two defenses which are shown to be effective.

### Strengths
- I find the idea of the paper novel, underexplored and interesting. It insightfully highlights how reward models that focus only on final answers can overlook flawed reasoning steps, leading to incorrect reward assignments and reinforcing undesirable reasoning behaviors. I find this both intuitive and exciting to think about.

- The paper’s writing logic and presentation are clear, well-organized, and engaging, providing readers with a smooth and enjoyable experience.

- All the claims and ideas are beautifully implemented and supported by comprehensive experiments, which I find both convincing and valuable.

### Weaknesses
- Despite the interesting results, I have serious concerns about the methodology and underlying assumptions. The attacker nodes are assumed to have access to both oracle responses and the reward function, enabling them to easily craft poisoned completions, a setting that feels unrealistic to me. Moreover, some experiments are heavily dependent on specific datasets and prompt availability, as even the authors note that *“in horizontal RL is that the occurrence of the targeted equation is rare in the entire GSM8k dataset. As this specific attack is very dependent on the prompt it might not succeed well when the attacker cannot pick and choose their questions”*.
All these assumptions and access levels make the results somewhat expected rather than surprising. A more realistic scenario would avoid giving attackers oracle or reward access and instead explore subtler injection methods, such as prompt-level manipulations that preserve final answers to maintain high rewards, though implementing such attacks would be significantly more challenging and warrant deeper investigation. 

- The paper would benefit from additional proofreading. For instance, in Line 245, the sentence “In an out-of-context attack, we aim to inject arbitrary malicious text. and the same text.” contains a punctuation error. Additionally, Footnote 6, which reads “But it’s not, maybe not,” appears to be accidental or misplaced and I'm not sure if I understand it. 

- Overall, I think a deeper analysis of the effectiveness and trade-offs from the computational side of the defenses, and how scalable and generalizable they are, would have been helpful. For instance, the LLM-as-a-Judge defense is computationally expensive, as it requires evaluating every generated completion individually with a separate model, significantly increasing inference cost and latency during decentralized training.

### Questions
Please refer to the weaknesses.

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
The paper introduces a backdoor attack on decentralized GRPO, where some of the nodes generating rollouts are malicious. The key observation is that for outcome-based rewards (such as math answer correctness), the reasoning trace doesn't directly affect the reward, but it is still reinforced for high-reward samples. Thus an attacker can generate samples with malicious reasoning traces but correct final answers to inject undesired behaviors into the model.
The paper shows that this attack works to insert simple behaviors in math and code settings and also explores two defenses (one based on off-policy detection, one using an LLM to judge answer correctness).

### Strengths
Backdoor attacks on decentralized training are an interesting threat model. The key idea that the attacker can reinforce malicious intermediate steps in RL from outcome-based rewards is also interesting and new to me (though I'm not familiar enough with the field to be confident how novel it is).
The code injection attack in section 3.4.2 is the most compelling to me, since there is a relatively straightforward path from injecting malicious coding behavior to achieving various attacker goals. Overall, the ideas in this paper made me take these types of backdoor attacks more seriously as a threat foe decentralized RL.

### Weaknesses
The primary weakness in my mind is that the implanted behaviors are rather simple, so it remains an open question how difficult it would be for an attacker to inject behaviors that they would actually want to inject. The behaviors in the paper are: saying a fixed phrase inside the reasoning trace, using "2 + 2 = 5" in math reasoning, and importing and using a certain library in code. The last of these comes closest to a real attack, but a big gap remains between a "static" behavior (always inserting a specific import) vs practically concerning behaviors. So I see the experiments mostly as a proof of concept for the attack idea.
Consequently, e.g. the statement "We empirically show that attackers can teach arbitrary malicious behaviour to honest models" in the conclusion in line 455 seem much too strong to me ("arbitrary" is a much stronger claim than the simple example behaviors justify).

For all experiments except the code injection setting, it's also not clear to me what exactly the threat model is. Injecting behaviors that lead to weird intermediate steps but correct final answers doesn't seem necessarily threatening, what is the attacker trying to achieve here? (The code injection setting is different because the intermediate steps are code that gets executed.)

### Questions
I’m confused about the "2 + 2 = 5" attack, e.g. in Fig. 3. In the example from that figure, it seems like the “2 + 2 = 5” step is leading to a wrong final answer, since the model continues reasoning from that wrong intermediate step. Is that right? Would the attacker then assign a high reward even though the final answer is incorrect? If yes, this seems like other nodes could easily catch the attack by re-computing the rewards (which at least in this setting is very cheap relative to sampling/training). Or alternatively, if the attacker uses the benign reward function, shouldn't this example get low reward and be negatively reinforced?
My understanding was that the attack would use actual high-reward final answers but this example seems incompatible with that. That makes me a bit concerned about the soundness of the "2 + 2 = 5" experiment, so I would appreciate corrections in case I'm missing something.

### Soundness
2

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
The authors consider a threat model in which an attacker has access to one or some nodes' outputs and can poison them during decentralised GRPO training. Under this setting, they study 3 different poisoning attacks on outcome-based reward models on a 1.5B parameter language model. They also propose 2 defences based on whether each node has the same copy (homogenous) or different copies (heterogeneous) of the model.

### Strengths
- The threat model is novel and relevant to training current reasoning models.
- The setup is clearly described and is easy to follow. 
- The proposed homogeneous defence is actionable and cheap to implement.

### Weaknesses
- Confusion regarding the ideal behaviour of in-context poisoning:
    - In Line 747 and in Figure 3, you provide two different types of how 2+2=5 is 'used'. However, I do not understand how Figure 3 would be reinforced at all: Since the authors claim to use an outcome-based oracle reward, and actually using the incorrect calculation results in an incorrect outcome, shouldn't the reward model actually penalise Figure 3? 
    - Assuming the model learns to only replace the 4 with 5 and never use it, this makes the differentiation of '2+2=5' and 'hail to the thief' attack not very interesting (as the model just learns to replace strings at specific places instead of at the beginning). 
- No error bars provided: Results of attacks are not averaged over multiple seeded runs (eg: mean and 90% CI ASR after 100 iterations for each attack). I believe this should be straightforward to run 100 iterations on a 1.5B model. This also makes the comparisons (eg, in Figure 14) not straightforward as the runs themselves are very noisy. 
- Regarding explored defences: 
    - The authors claim to have studied various defences under limitations; however, the results of these are not reported. It is important to know how and why well-known defences fail. 
    - Heterogeneous defence assumes access to a stronger model. This invalidates its practicality in actual settings. 
    - The authors should also report the percentage of benign prompts misclassified as adversarial by the defences. 
    - For the homogeneous defence, the authors write: "benign models can run incoming completions through the model in a single forward pass and use the log-probabilities to check if each token could have come from the model and the given generation strategy": what exactly is the classification criterion used here? 
    - Additionally, defences like steering \[1\] could potentially be used to detect such examples. (It's fine if the authors don't explore such methods, but it could be added to the literature review, at least). As pointed out in related work, this type of attack could also be studied as a data poisoning/malicious finetuning attack \[2\], rather than as a vulnerability of federated learning alone. This potentially opens up other defences that are studied in data poisoning settings like \[1\]. 
- The paper limits the study to Qwen 1.5B, whether the attacks/defences work on larger models remains unknown. 

References:

\[1\]: Helena Casademunt, Caden Juang, Adam Karvonen, Samuel Marks, Senthooran Rajamanoharan, and Neel Nanda. Steering out-of-distribution generalization with concept ablation fine-tuning, 2025. URL https://arxiv.org/abs/2507.16795.

\[2\]: Halawi, Danny, Alexander Wei, Eric Wallace, Tony T. Wang, Nika Haghtalab, and Jacob Steinhardt. "Covert malicious finetuning: Challenges in safeguarding llm adaptation." arXiv preprint arXiv:2406.20053 (2024).

### Questions
1. Could the authors please describe the purpose behind 2+2=5 as an attack, and clarify the questions I raised in the weaknesses regarding this?
2. What is the classification criterion used for section 4.1? (eg: do you use a tokenwise threshold? How do you aggregate it?, etc.)
3. Does the heterogeneous defence still work if you use a model stronger than the judge for training? Alternatively, does using the model itself as a classifier work better with larger models?
4. I would highly recommend adding multiple runs and error bars for a better comparison of results.

### Soundness
1

### Presentation
3

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
The paper introduces attacks for the use case of federated post-training with GRPO. They demonstrate that they can poison thinking tokens in a few nodes, thereby corrupting the model. They further propose defenses against this new threat model that works in the homo and heterogeneous parameter case: either checking for plausibility of the generation of the answer or judging with another model whether the generation is correct and does not contain malicious text.

### Strengths
- The authors propose a new attack for a new threat model
- They also introduce defenses to guard against this threat
- Both the attack as well as the defenses are effective

### Weaknesses
- The number of malicious users is quite high. I would like to see a lower contribution to understand the scaling of a few poisoned examples here. 
- The threat model gives the attacker a lot of control, limiting the practicality of this threat model in my opinion.

### Questions
- Why do the authors look into the threat model of only passing the strings?

### Soundness
3

### Presentation
3

### Contribution
2
