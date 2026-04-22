# One Token to Fool LLM-as-a-Judge

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Large language models (LLMs) are increasingly trusted as automated judges, assisting evaluation and providing reward signals for training other models, particularly in reference-based settings like Reinforcement Learning with Verifiable Rewards (RLVR). However, we uncover a critical vulnerability even in this reference-based paradigm: generative reward models are systematically susceptible to reward hacking. We find that superficial inputs, which we term ''master keys'' such as non-word symbols (e.g., '':'' or ''.'') or generic reasoning openers (e.g., ''Thought process:'' or ''Let's solve this problem step by step.''), can consistently elicit false positive rewards without any substantive reasoning. Our systematic evaluation demonstrates this is a widespread failure affecting a diverse range of models, including leading proprietary systems such as GPT-o1 and Claude-4. These results challenge the assumed robustness of LLM judges and pose a significant threat to their reliability. To address this, we propose a simple yet effective data augmentation strategy using truncated model outputs as adversarial negative examples. The resulting Master Reward Models (Master-RMs) demonstrate state-of-the-art robustness against these ``master key'' attacks while maintaining high performance in standard evaluation settings. We supplement these findings with a comprehensive analysis of the vulnerability across model scales, prompt variations, and common inference-time strategies, offering insights to guide future research on robust LLM evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the issue of reward hacking concerning LLM-as-a-Judge. They propose two kinds of MASTER KEYs, non-word symbols and reasoning openers. They found that these superficial inputs can fool LLM-as-a-Judge. This issue extends to a wide range of models. They propose a method of leveraging synthetic data to mitigate this issue. It demonstrates effectiveness. Agreement experiments are conducted.

### Strengths
- This paper studies an urgent research question.
- Rewarding experiments of human agreement.
- The idea of synthetic data is simple yet effective.
- The writing is easy to follow.

### Weaknesses
From my perspective, this paper is well-organized. My only concern is about the novelty.
- The MASTER KEYs are seemingly just another type of reward hacking patterns for LLM-as-a-Judge.
- This paper discovers some new patterns that will hack LLM-as-a-Judge, evaluate the issues on several models (not the most recent), SFT to correct these, and demonstrate the effectiveness of training. It is not exciting enough.

### Questions
- In line 294, the generalizability is emphasized. What about an overlap analysis between the training set and the test set? 
- The observation that training on some kinds of known vulnerabilities can mitigate similar vulnerabilities is not beyond expectation. There are definitely other kinds of MASTER KEYs for LLM-as-a-Judge. As a starting experiment, can training on *Non-word symbols* help mitigate *Reasoning Openers*?
- As CoT is involved in LLM-as-a-Judge and reasoning models demonstrate better resilience, what about evaluating some new reasoning models, e.g., Qwen3-8B? Will adopting new open-source reasoning models alleviate this problem?
- What about the downstream implications concerning RL training? It will be more interesting to see whether correcting the limited set of MASTER KEYs brings about a better LLM.
- The paper title is somewhat misleading. It is unclear what the *One Token* refer to.

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
The paper reveals a significant vulnerability in the "LLM-as-a-Judge" paradigm via systematic evaluations. Specifically, the author find most reward models would give high rewards to superficial inputs with the master keys which include non-word symbols or generic reasoning openers. To address this threat, the authors propose a simple yet effective data augmentation method to fine-tune the reward models. The resulting models (i.e. master RMs) are not effective in defending master-key attacks but also exhibit good performance on verification benchmarks.

### Strengths
1. This paper discusses a timely and critical topic for LLM-as-a-judge.
2. The paper proposes a simple hacking method and conducts a broad evaluation on the robustness of different reward models against "master key" attacks
3. The paper proposes a simple and yet effective data augmentation method to defend the master key attack.

### Weaknesses
1.  Although the paper provides many interesting findings, it lacks an in-depth analysis of the model vulnerabilities. For example,

a. Why "master key" attacks can succeed in fooling General-Purpose LLMs, i.e. what training behavior could lead General-Purpose LLMs to give high reward to "master key" attacks?

b. According to Table 1, multilingual reasoning openers also achieve a high false positive rate, despite the fact that the evaluation tasks are English-based. Why do reward models give high rewards to the generation in a different language?
 
2. More experiments should be done to verify the effectiveness of master-reward models.

a. The author should consider the adaptive attacks where the attackers aim to further hack the master-reward models, e.g. the other attack methods discussed in "Vulnerabilities of LLM-as-a-judge"  or the new master key patterns found in Appendix C.

b. It is good to show if applying master-reward models during RL training can improve the training performance compared to other reward models.

3. The paper is not well organized and needs improvement:

a. Section 3 " Methodology" discusses Verifiable Reward Modeling in RLVR, which neither belongs to the methodology nor is the main focus of the paper. 

b. The methodology part is mixed with the experiment setup in Section 4. The author should move the description of "Master-RM" to Section 3.

c. Section 4.3.2 and Section 4.3.3 seem to serve a duplicate purpose and they should be merged together. Besides, Section 4.3.2 lacks the details for how to get human judgments.

d. Additional Experimental Results part(Line 467) only lists the findings without a detailed discussion. It is important to make the main paper self-contained and move some of the discussion to the main paper.

### Questions
See above. Besides, I have some additional questions:

1. In Line 297, the authors claim that master RMs can generalize to unseen hacking attacks as well. Indeed, the data augmentation strategy seems not to include multilingual opening reasoners, but still achieves zero false positive rate for them. Could you explain the insights behind this phenomenon?

2. According to Table 3, master RMs also show superior performance over specialized RMs on general verification benchmarks (i.e. VerifyBench). Is it due to using a more powerful training dataset? The authors may need to compare with the performance of training the same base model with the same dataset without data augmentation.

### Soundness
2

### Presentation
2

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
LLM-judges are often tricked by generic tokens (e.g. a colon or full-stop) that leads to false positive results, affecting models such as o1 and Claude. To alleviate this problem, the paper propose a data augmentation strategy with truncated model outputs as negative examples. The resulting models trained with the augmented data maintains high performance in standard evaluations while showing SOTA robustness against these generic token attacks.

### Strengths
1.	Given that LLM-Judges are increasingly used for training and evaluation, highlighting a flaw that LLM Judge have can be important. It is interesting how many of these master keys are so generic e.g. a blank space or a colon can substantially change results. The is also related to the topic of jailbreaking LLMs to do things that they were not supposed to, and can serve as a bridge between this LLM-judge and jailbreaking.

### Weaknesses
1.	The paper mostly discusses False Positive Rate, which is a key issue of model-based verifiers. However, it’s also important to discuss False Negative Rate since we should not have models that have zero false positive rate, by simply predicting all generated answers as wrong. It’s important to measure and report both since rule-based verifier are known to be low in false-positives and high in false-negatives compared to model-based verifiers [1]. Both can be combined into an overall F1 metric.
2.	The augmentation approach seems a little under-explored as it only takes the first sentence regardless of its content (there doesn’t seem to be any method to check if the first sentence does indeed have useful content, besides an assertion in line 272). This encourages the judge model to ignore the generated solution and increases the likelihood of “NO” compared to “YES”, which can reduce false positives, but will likely also increase false negatives. The other risk of this approach is that it will over-penalize any response that has a similar opener as the data-augmentation model (GPT-4o-mini) rather than solve this behavior.
3.	The paper does very limited exploration on why the current LLM-Judges are easily hackable by the master-keys. Do those master keys encourage the model to first generate its own answer rather than to just judge on the provided answers? 
4.	The empirical performance of Master-RM in Table 2 doesn’t seem to be better than the original Multi-sub RM, which the Master-RM training data is primarily derived from. This suggests that the data augmentation doesn’t improve the RM quality in terms of its agreement with either humans or GPT-4o (a standard LLM-Judge). 

[1] From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in Mathematical Reasoning

### Questions
1.	What’s Claude-4 - Is that Claude Sonnet 4 or Claude Opus 4?
2.	Can the authors elaborate upon how the agreement with human annotations data was collected? Specifically, is the task challenging for these annotators and how much do the annotators agree with each other?
3.	I see the prompt template in Table 6 – what happens when the generated answer is neither YES nor NO? Also, how many tokens is the LLM-judge allowed to generate? I assume it’s only 1 token but it is explicitly stated here.

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
5

### Summary
In this paper, authors investigate the fragility of LLMs being used as judges in RLVR for judging model response or in general LLM as a judge setup to evaluate model response. Something as simple as trivial tokens or superficial phrases (e.g., “Thought process:”
or “Let’s solve this problem step by step.”) are enough to break it ie the judge gives a false positive reward when the answer is meaningless. Effects hold through multiple models both general purpose LLMs or specialized RM. The proposed adversarial data augmentation strategy to mitigate this behavior.

### Strengths
* The authors highlight an important cautionary tale which the community should be aware of, especially with the increasing use of LLM as  judges in both RLVR and for evaluation purposes. 
* The fact that this even happens in SOTA models is interesting. This will raise awareness in the community to take the LLM as a judge. Evaluations with a grain of salt. 
* Clear presentation with an easy quick fix proposed

### Weaknesses
* The adversarial fix proposed is not thorough enough and would only guard against a very specific type of hacking of the judges. 
* The solution approach is only tested against other RMs/LLMs on False positive rates where it does better. Its not surprising since they train for exactly those examples so their RM is naturally robust. More important is how this Master-RM does when used as an LLM as a judge in a real RLVR setting. Its possible in real RLVR setting even this master-RM gets hacked by some other kind of "attack" or "naive" prompts which the authors didn't train against. 
* Authors didn't talk about automated ways of discovering these prompts, nor do they detail how they found these "attacks". They can talk about literature focussed on that too.

### Questions
* Did the authors try any reasoning Gen-RMs which reason before giving a response. Maybe because they spend more tokens at inference time and are trained in that manner maybe they are better. Eg https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-GenRM . I know the authors tested o1 but maybe training for judging is important
* Maybe the authors can investigate an automated way of mining such adversarial examples using RL similar to setup in the paper [1] . These examples will naturally arise in the context of RLVR training but if you can do adversarial red-teaming or RL then you can find a much broader variety as compared to a small subste discussed in the paper
*  In line 270 "regenerate model responses using chain-of-thought prompting with GPT-4o-mini". Why not generate these from a reasoning model as they are the ones used for RLVR and their behaviour might differ from CoT prompted 4o-mini



[1] https://arxiv.org/abs/2504.06141v2

### Soundness
3

### Presentation
3

### Contribution
2
