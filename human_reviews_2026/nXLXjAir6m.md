# Chain-of-Trigger: An Agentic Backdoor that Paradoxically Enhances Agentic Robustness

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4, 2

## Abstract
The rapid deployment of large language model (LLM)-based agents in real-world applications has raised serious concerns about their trustworthiness. In this work, we reveal the security and robustness vulnerabilities of these agents through backdoor attacks. Distinct from traditional backdoors limited to single-step control, we propose the Chain-of-Trigger Backdoor (CoTri), a multi-step backdoor attack designed for long-horizon agentic control. CoTri relies on an ordered sequence. It starts with an initial trigger, and subsequent ones are drawn from the environment, allowing multi-step manipulation that diverts the agent from its intended task. Experimental results show that CoTri achieves a near-perfect attack success rate (ASR) while maintaining a near-zero false trigger rate (FTR). Due to training data modeling the stochastic nature of the environment, the implantation of CoTri paradoxically enhances the agent's performance on benign tasks and even improves its robustness against environmental distractions. We further validate CoTri on vision-language models (VLMs), confirming its scalability to multimodal agents. Our work highlights that CoTri achieves stable, multi-step control within agents, improving their inherent robustness and task capabilities, which ultimately makes the attack more stealthy and raises potential safty risks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Chain-of-Trigger (CoTri), a novel multi-step backdoor attack that extends traditional single-step backdoor paradigms to long-horizon LLM-based agents. Instead of a single static trigger, CoTri embeds a sequence of triggers distributed across multiple environment steps, each corresponding to an observed trigger token and a malicious action. The backdoor remains dormant unless these triggers appear in the correct order, yielding precise control while maintaining low false activation. The authors demonstrate CoTri across four text-based (AgentLM, AgentEvol, Llama3.1, Qwen3) and one vision-language model (Qwen2.5-VL), showing near-perfect attack success rates (ASR ~1.0), negligible false trigger rates (FTR ≈ 0), and, interestingly, improved robustness in noisy or stochastic environments due to the introduction of “rollback” behaviors. Experiments are limited to three-step trigger chains within the WebShop environment, illustrating proof-of-concept rather than full scalability.

### Strengths
- The conceptual framing of multi-step, temporally chained backdoors is intellectually interesting and highlights a plausible class of attacks that may emerge in long-horizon agent control.

- Methodology: The paper is detailed and systematically formalizes the backdoor’s trigger logic, provides clear pseudocode, and supplies analyses for benign, full-chain, and partial-chain conditions.

- The rollback design is a clever addition as it allows the model to appear more robust during missing triggers while maintaining the attack path when the full chain occurs.

- The paper includes thorough evaluations across both text and multimodal backdoors, showing generality of the concept to VLMs.

### Weaknesses
- The experiments are shallow, limited to 3-step chains. The setup does not convincingly demonstrate that CoTri scales to longer trajectories, which undermines the “long-horizon” claim. Without experiments on e.g., 5–10 step tasks, it remains mostly a proof of concept.

- The apparent robustness gains are not fully substantiated  since the same poisoning process acts as data augmentation, the improvement could simply stem from LoRA fine-tuning on additional data variability rather than any inherent defensive mechanism.

- Not much evidence is given that the method would hold for longer or open-ended environments, where trigger chains may overlap, diverge, or decay temporally.

### Questions
- Can CoTri feasibly scale beyond the demonstrated 3-step configurations? How does ASR or FTR behave for 5-step or 10-step trajectories?

- Could similar effects be achieved by standard compositional triggers (multi-token sequences) rather than temporal chains?
Does the rollback mechanism affect benign performance on unrelated environments beyond WebShop (e.g., text-only multitask benchmarks)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Chain-of-Trigger (CoTri), a sequential backdoor attack targeting long-horizon agentic control. The attack activates only when a specific token trigger and corresponding environment triggers appear in the correct order across multiple reasoning steps. The authors show that CoTri maintains near-perfect attack success rates (≈100%) and low false trigger rates in WebShop environments across four LLMs and one VLM. Additionally, CoTri also appears to improve model robustness under noisy or distracting conditions.

### Strengths
- **Interesting ideas**: The paper proposes chain of triggers (CoTri), mislead the agent step by step to the final backdoor behaviors while recovering to benign performance if the trigger doesn’t appear at certain steps or in the wrong order.
- **Comprehensive evaluation on both LLMs and VLMs**: The paper evaluates their methods on both task-specific LLMs (i.e., AgentLM, AgentEvol), general LLMs (i.e., LLaMA, Qwen), and general VLMs (i.e., Qwen-VL), covering a wide spectrum of different types of models.
- **Step-wise analysis**: The paper conducts step-wise analysis on the attack success rate and false trigger rate. This is critical to understand if the proposed framework works as expected. The paper also conducts experiments on partial triggers to demonstrate that the agent will perform malicious tasks only if all the triggers appear.

### Weaknesses
## Major
- **Lack of stealthiness analysis from the user perspective**: Although CoTri shows the stealthiness from the consequence level (e.g., the malicious tasks will only be performed if the trigger chain is complete and in correct order), its step-wise design makes deviations observable early in the intermediate process level. As shown in Fig. 2, the agent performs visibly unintended actions immediately upon the initial trigger (“tq”) injection. This reduces real-world stealth, as end users could interrupt the agent before the malicious sequence completes.
  - **Suggestions**: The author needs to justify why CoTri is more stealthy than a single-step backdoor from the end-user perspective, given the multiple malicious actions (e.g., not user-intended) before achieving the final results.


- **Incomplete threat model**: It’s good to see the paper discuss the threat model of the training process. However, the threat model for the triggering process is not discussed. Specifically, the injection of the initial trigger token (e.g., “tq”) assumes adversarial control over user prompts, which may not be realistic in typical agentic interfaces. Clarifying plausible attack vectors for this assumption would strengthen the paper’s soundness.
  - **Suggestions**: The author is suggested to refine the threat model, justify how the initial trigger can be inserted into the end-user prompts.


- **Unsupported claims from the experiments**: It’s good to see CoTri improve the robustness of fine-tuned LLMs (i.e., AgentLM, AgentEvol) in Table 4. However, CoTri decreases the robustness for general LLMs, such as LLaMA and Qwen. First, the comparison between “ours” and “ori” to support the claim in Line 428 is not fair, because the improvement is not from CoTri but from the benign fine-tuning data (e.g., clean). Second, the comparison between “ours” and “ori” can not support the claim *“Although the improvement over the clean is less pronounced” (Line 428)*. From the table, the CoTri is consistently worse compared to “clean”. Therefore, the overall claim that CoTri can improve robustness (second main contribution in Line 82) is not fully supported by the experimental results. 
  - **Suggestions**: Please refine the claim in the paper and explain why CoTri shows decreased performance compared to “clean” for LLMs without task adaptation.
 


## Minor
- **Need writing clarification**: The paper is generally well-written. However, several sections would benefit from clearer terminology around “robustness,” which is currently used for two distinct concepts. Type-1: Model robustness towards environmental noise or distractions (Table 4); Type-2: Model robustness towards partial triggers (Table 3). The author is suggested to make the term clearer to avoid confusion.
- **Unfair experimental setups**: In Table 4, I’m concerned about the fairness of the experimental setups. CoTri adds more training data (e.g., the malicious one) to the “clean”. One more fair setup is to mix the malicious data while keeping the overall data amount the same as “clean”.
- **Only one VLMs is evaluated**: The paper only evaluate one VLMs (i.e., Qwen2.5-VL). More VLMs are expected to be evaluated to show the generality of the conclusions.
- **Typo**: In Appendix Table 13, it should be “0.70” instead of “0.7” to be consistent.

### Questions
- Why is the attack success rate in Table 5 - Step 3 (i.e., 1.00) different from the attack success rate in Table 6 - Step 3 - dirty (i.e., 0.75). From my understanding, these two should be the same as those in Tables 1 and 2.
- Why is the mixing ratio slightly different in Table 13 for different models?

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
3

### Summary
This paper proposes Chain-of-Trigger (CoTri), a new kind of backdoor attack designed specifically for long-horizon, agentic LLM systems. Instead of relying on a single trigger phrase or token, CoTri introduces a sequence of dependent triggers that must appear in a specific order across multiple steps in an environment. Once the chain is completed, the model executes a malicious behavior — but otherwise remains benign.The authors also make a surprising observation: when agents are trained on CoTri-poisoned data, their performance on benign tasks improves and their robustness against unrelated perturbations increases. This “paradoxical robustness” is explored through experiments on various large models (AgentLM-7B, Llama3.1-8B, Qwen3-8B, and Qwen2.5-VL) across multiple simulation settings.

### Strengths
1. The idea of multi-step, environment-conditioned backdoors for long-horizon agents is interesting. The idea of having a temporal dependency across steps makes a lot of sense in the context of long-horizon planning and multi-turn interaction.

2. The finding that backdoor training can actually make agents more robust to noise or feedback perturbations is counterintuitive but quite thought-provoking. Even if the mechanism isn’t fully clear, it could open up a new line of work about how certain adversarial perturbations might regularize agent behavior.

3. The paper is well-organized and readable. The figures illustrating the chain of triggers, trajectory timelines, and the benign-vs-malicious transition are clear.

### Weaknesses
1. The paper assumes the attacker has full control over the training data, which fits cloud or outsourced model settings but is less realistic for most locally trained agents. Some discussion of weaker threat assumptions (partial data control, prompt-time poisoning, or RL fine-tuning contamination) would improve applicability.

2. The work would be stronger if it compared CoTri against common backdoor detection or mitigation approaches (e.g., spectral analysis, activation clustering, gradient inspection). Even showing that these methods fail would provide stronger justification for the claimed stealthiness.

3. The Attack Success Rate and False Trigger Rate are reported as 1.00 and 0.00 across all setups. This looks too ideal and makes reproducibility uncertain. It would help to report standard deviations or multiple random seeds, and possibly an ablation on poisoning ratios.

4. The rollback policy \pi_r is key to the method, but it’s not clear how it’s implemented — is it a separate module, or learned behavior? A visualization or ablation showing how rollback activates when the chain breaks would clarify a lot.

### Questions
1. Could the robustness improvement simply be due to regularization or curriculum effects rather than specific trigger structure?

2. How does the performance scale with longer trigger chains (e.g., 3, 4, 5 triggers)? Does stealth improve or degrade?

3. For multimodal setups (e.g., Qwen2.5-VL), are visual triggers literal objects or latent embeddings inserted during training?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new multi-step backdoor attack designed for LLM-based agents operating in long-horizon tasks. Unlike traditional single-step backdoors, their method requires an ordered sequence of triggers. Their method achieves near-perfect attack success rates (ASR) while maintaining negligible false trigger rates (FTR) across diverse architectures, including AgentLM, AgentEvol, Llama3.1, Qwen3, and multimodal models like Qwen2.5-VL. They also find that the backdoored agents exhibit enhanced robustness compared to clean models when exposed to noisy, distracting, or incomplete environmental feedback during the training process.

### Strengths
The proposed method successfully attacked the target LLMs, including AgentLM, AgentEvol, Llama3.1, Qwen3, and multimodal models like Qwen2.5-VL.

### Weaknesses
I have the following major concerns about this paper:

### Lack of demonstration of why a multi-step backdoor attack is necessary 

I am not fully convinced of the necessity of a multi-step backdoor attack. As in Table 1, the attack success rates at Step 1 are already nearly 100%. As this paper mentioned, several prior backdoor attacks with a single trigger have already been proposed. As the multiple triggers need a stronger threat model, this paper does not provide sufficient justification for why multiple-step triggers are needed. This paper should first clearly demonstrate the cases in which the single-trigger backdoor attacks do not work against long-horizon agentic control, and then should show that their multi-step backdoor attack can address it. Otherwise, the multi-step backdoor attack looks like it employs unnecessary steps in the attack.

### No effectiveness comparison with a baseline of a single-step backdoor attack.

I can see that their attack works against the target LLMs, including AgentLM, AgentEvol, Llama3.1, Qwen3, and multimodal models like Qwen2.5-VL. Meanwhile, their evaluation does not have a comparison with a baseline of a single-step backdoor attack, and thus, I cannot see how their proposed method is better than existing attacks. Particularly, the attack looks already successful at Step 1. Naturally, the multiple-trigger attack should be expected to have higher effectiveness or higher stealthiness while introducing additional attack steps. This paper should provide more experimental results to highlight the trade-off between effectiveness and stealthiness when increasing the attack steps. For the single-step attack, there should be prior attacks, as this paper mentioned. Their method should be compared with state-of-the-art existing single-step attacks.

### Limited technological contribution in attacking CoT with a backdoor

While I can see a certain level of novelty in attacking (LLM)-based agents with a backdoor attack with a multi-step trigger, it is hard to acknowledge a major technical contribution in their attack. As they also mentioned, the attack with a single-step trigger already exists. Multi-trigger backdoor attack itself has already been explored in prior research [a].

[a] Li, Y., He, J., Huang, H., Sun, J., Ma, X., & Jiang, Y. G. (2025). Shortcuts everywhere and nowhere: Exploring multi-trigger backdoor attacks. IEEE Transactions on Dependable and Secure Computing.

### Questions
What is the benefit of employing the multiple-step backdoor attack instead of a single-step one, and how does the benefit demonstrate in their evaluation?

### Soundness
2

### Presentation
1

### Contribution
1
