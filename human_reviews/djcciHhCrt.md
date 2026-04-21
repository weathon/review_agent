# Misusing Tools in Large Language Models With Visual Adversarial Examples

- Avg Score: 4.25
- Decision: Reject
- Scores: 6, 3, 5, 3

## Abstract
Large Language Models (LLMs) are being enhanced with the ability to use tools and to process multiple modalities. These new capabilities bring new benefits and also new security risks. In this work, we show that an attacker can use visual adversarial examples to cause attacker-desired tool usage. For example, the attacker could cause a victim LLM to delete calendar events, leak private conversations and book hotels.  Different from prior work, our attacks can affect the confidentiality and integrity of user resources connected to the LLM while being stealthy and generalizable to multiple input prompts. We construct these attacks using gradient-based adversarial training and characterize performance along multiple dimensions. We find that our adversarial images can manipulate the LLM to invoke tools following real-world syntax almost always ($\sim$98\%) while maintaining high similarity to clean images ($\sim$0.9 SSIM). Furthermore, using human scoring and automated metrics, we find that the attacks do not noticeably affect the conversation (and its semantics) between the user and the LLM.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper studies prompt injection against large language models (LLMs) through the visual modality, i.e., crafting (universal) adversarial image examples to cause the LLM to invoke tools following real-world syntax. In particular, the authors highlight the consideration of attack stealthiness in terms of both perturbation imperceptibility and response utility. Five variants of attacks with varied attack difficulty are considered.

### Strengths
- Prompt injection against LLMs is a promising direction, especially through the relatively new channel of visual modality.
- The paper is very well written, including sufficient example visualizations and clearly described technical details.
- Experiments are extensive and insightful, which include human studies and ablation studies of important hyperparameters.

### Weaknesses
- The best result is reported among three trials based on the argument that “attackers will always choose the best-performing adversarial image.” However, the reviewer thinks this may not be reasonable because the attacker is not the user and so cannot control how many times they would repeat the attack. The authors should explain why they think it is feasible to stick to this setting. This is important given the fact that significant randomness during adversarial image training is observed.

- The authors motivate the design of separate losses for response utility and tool invocation (i.e., Equation 2) based on the argument “In real-world conversational systems...we reduce the contribution of the loss term...” First of all, there are comparisons validating the superiority of using such separate losses to the integrated loss (i.e., Equation 1). More specifically, the ablation studies clearly show using a large $\lambda$, i.e. 1, works better, which conflicts with the argument about “reducing the contribution...”.

- It is said that “the l2 norm is computed with regard to each color channel separately”. However, to the best knowledge of the reviewer, in the literature of adversarial examples, it is indeed computed not separately. Could the authors explain why they chose this unusual setting? 

- Considering that the threat model follows the typical prompt injection, it seems unnecessary to highlight its differences from “jailbreaking”. Therefore, the authors are encouraged to tune down some related claims. 

- The fact that only three images are tested should be mentioned in the main text rather than only in the appendix.

### Questions
See the above weaknesses.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a method to cause malicious tool usage of LLM by using adversarial examples. The idea is to use a loss that includes both the response utility and malicious behavior for training

### Strengths
1. The paper studies a new and timely security issue of LLM.
2. The proposed method achieves better stealthiness over prior works.

### Weaknesses
1. The proposed method seems to be straightforward, which is essentially the way of injecting a backdoor. It is unclear how this method can help generalize the trigger to other prompts. 

2. Since the goal is to trigger the misusage of tools, why limiting the adversarial perturbation and enhancing the generalizability are important? The adversary only needs one prompt to trigger the malicious usage of the tools. 

3. It is unclear what are the implications of these 5 attack objectives. Does the selection of these attack objectives have an impact on the attack performance? What will happen if more attack objectives are included?

4. The evaluation is unsatisfactory. L_p norm is not provided. Also, it is important to show the chances that the malicious tool is triggered when the adversarial example and prompt pair are not present, or only one of these two is present.

### Questions
See Weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes using visual adversarial examples to attack vision-integrated LLMs. Particularly, this paper focuses on attacking the LLMs to affect the confidentiality and integrity of users' resources connected to the LLMs. The proposed attack is stealthy (as visual adversarial examples look similar to normal images), and generalizable (the adversarial examples can trigger the targeted generation when paired with different text prompts). And it is shown that the attack can successfully trigger LLMs to output the tools-abusing texts.

### Strengths
1. The big picture of the paper is sound. Indeed, as LLMs are integrated into applications, critical resources may be controlled by the models. Then, attacks on the models can induce broad implications beyond just the misalignment moral values. The threat model and the real-world risk analysis in this paper are quite insightful. 

2. The approach is simple and effective.

3. The authors make efforts to collect evaluation datasets as well as comprehensive human evaluation.

### Weaknesses
1. **Only a single model LLaMA Adapter is tested.** This makes the scope of the evaluation look somewhat narrow. I suggest the authors also consider other VLMs like Minigpt-4 [1], Instruct-Blip [2], and LLaVA [3]. This can make the evaluation more convincing. 

2. **Lack of case studies on real LLM-integrated applications.**  The paper mentioned that LangChain and Guidance facilitate the development of such integrations. But, the paper did not provide a single instance of this to illustrate the practical risks of the proposed attack. In the whole paper, what the attack did was just to induce the generation of something like <function.delete_email which="all"> in a purely textual form, which is essentially nothing different from previous NLP attacks that induced certain targeted generations. In my opinion, the novelty of this paper only comes from the illustration of the "practical risks" of such attacks --- because the resources controlled by LLMs can now also be missed to induce broader harms. However the paper did not provide a real example of this. The whole evaluation is still in a purely textual form, judging whether things like <function.delete_email which="all"> are generated... In practice, the LLMs integrated systems may be more complicated than this conceptual form. The paper did not go deep into this. 

3. **Inaccurate Literature Review.** There is a factual error in the literature review. Qi et. al. [4] did not show transferability to closed-source models. On the other hand, Carlini et. al. [5] along with Qi et. al. [4] are earlier works showing the usage of visual adversarial examples to hack VLM, which may also need to be noted.



[1] Zhu, D., Chen, J., Shen, X., Li, X. and Elhoseiny, M., 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592.

[2] Dai, W., Li, J., Li, D., Tiong, A.M.H., Zhao, J., Wang, W., Li, B., Fung, P. and Hoi, S., 2023. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. arXiv preprint arXiv:2305.06500.

[3] Liu, H., Li, C., Wu, Q. and Lee, Y.J., 2023. Visual instruction tuning. arXiv preprint arXiv:2304.08485.

[4] Qi, X., Huang, K., Panda, A., Wang, M. and Mittal, P., 2023, August. Visual adversarial examples jailbreak aligned large language models. In The Second Workshop on New Frontiers in Adversarial Machine Learning.

[5] Carlini, N., Nasr, M., Choquette-Choo, C.A., Jagielski, M., Gao, I., Awadalla, A., Koh, P.W., Ippolito, D., Lee, K., Tramer, F. and Schmidt, L., 2023. Are aligned neural networks adversarially aligned?. arXiv preprint arXiv:2306.15447.

### Questions
Can the attack be directly applied to realistic LLM-integrated applications in the wild? Say, other prompt injection attacks such as [1,2] do show realistic instances. 


[1] Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T. and Fritz, M., 2023. Not what you’ve signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. arXiv preprint arXiv:2302.12173.

[2] Liu, T., Deng, Z., Meng, G., Li, Y. and Chen, K., 2023. Demystifying RCE Vulnerabilities in LLM-Integrated Apps. arXiv preprint arXiv:2309.02926.

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes an attack on multi-modality models that are capable of using tools. The main idea of this paper is to perturb the image side so that the text side can generate malicious instructions that may impact downstream tools. The paper demonstrates that their attack is effective, stealthy, and generalizable.

### Strengths
- In general, this paper is well-structured and easy to follow.
- I believe the problem this paper addresses is highly significant. It focuses on understanding how to attack systems using LLMs in real-world scenarios, which presents new challenges when viewed from a systemic perspective.
- The experimental results demonstrate that malicious instructions can be generated by perturbing the input image.

### Weaknesses
- This paper assumes that interaction with the tools occurs through an instruction line, followed by normal question answering, as illustrated in Figure 1. Is this setting realistic? What does a real system look like, and how do these VLMs interact with downstream tools like email? Please provide an illustration of why the task in Figure 1 is realistic.
- This paper lacks technical contributions and depth. The technical contribution of this paper is to generate perturbations on the image side that can prompt the language model to output specific words. However, this paper does not discuss the technical challenges associated with this attack scenario. Instead, they employ a simple gradient-based technique widely adopted in adversarial example attacks. They also do not provide an in-depth analysis of the drawbacks of this technique. For example, is this attack easily bypassed? What is the robustness of this attack?
- Additionally, the authors claim that the attack is stealthy because it does not alter the semantic meanings of the output answers. They support this claim with the evidence that the answers under attack are 10% less natural compared to the original ones. My question is, why is a 10% difference considered a small one?
- If the current setting is realistic, I suggest showing the effectiveness of attacking the real-world system.

### Questions
To conclude, I think the problem this paper would like to address is attractive. However, it is not clear why the current setting is realistic. Also, the technique contributions and discussion depth hinder its acceptance. I believe these questions are hard to be properly addressed during the rebuttal period.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
