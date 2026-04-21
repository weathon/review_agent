# BOOST: Enhanced Jailbreak of Large Language Model via Slient eos Tokens

- Avg Score: 5.50
- Decision: Reject
- Scores: 5, 6, 5, 6

## Abstract
Along with the remarkable successes of Language language models, recent research also started to explore the security threats of LLMs, including jailbreaking attacks. Attackers carefully craft jailbreaking prompts such that a target LLM will respond to the harmful question. Existing jailbreaking attacks require either human experts or leveraging complicated algorithms to craft jailbreaking prompts. In this paper, we introduce BOOST, a simple attack that leverages only the eos tokens. We demonstrate that rather than constructing complicated jailbreaking prompts, the attacker can simply append a few eos tokens to the end of a harmful question. It will bypass the safety alignment of LLMs and lead to successful jailbreaking attacks. We further apply BOOST to four representative jailbreak methods and show that the attack success rates of these methods can be significantly enhanced by simply adding eos tokens to the prompt. To understand this simple but novel phenomenon, we conduct both theoretical and empirical analyses. Our analysis reveals that (1) adding eos tokens makes the target LLM believe the input is much less harmful, and (2) eos tokens have low attention values and do not affect LLM's understanding of the harmful questions, leading the model to actually respond to the questions. Our findings uncover how fragile an LLM is against jailbreak attacks, motivating the development of strong safety alignment approaches.large language model, Jailbreak

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors investigated jailbreak attacks on large language models. The authors demonstrated that appending several EOS tokens to the end of a given sentence can significantly enhance the attack effectiveness. They explained that this improvement arises because EOS tokens bring the samples closer to the ethical boundary, as evidenced through empirical testing. The authors conducted some experiments to validate the effectiveness and generalizability of the proposed attack.

### Strengths
* The idea of this paper is simple, making it easy to follow.

* The proposed method is relatively easy to replicate.

* The experiments are comprehensive.

### Weaknesses
* Certain parts of the paper are redundant. Section 3.1 is overly lengthy in explaining the concept of ethical boundaries. This could be summarized in a single sentence, as the concept is essentially a classification boundary that determines whether the jailbreak is successful in the context of LLMs. I recommend that the authors revise this section, as the current content in Sections 3.1 and 3.2 may complicate the reader's understanding of the paper.

* The method proposed by the authors is easily circumvented. Model deployers can manually remove the extra EOS tokens, which diminishes the practical effectiveness of the attack. Moreover, it is unclear how to determine the optimal number of tokens to add. While the authors claim that this can be enumerated, doing so would increase attack costs.

* The core contribution of the paper seems limited to the idea of adding a few tokens to a sample.

### Questions
Have the authors considered placing EOS tokens at different locations within the sentence?  Would this yield better attack results than merely appending them to the end?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposed BOOST (Enhanced Jailbreak of Large Language Model via Silent eos Tokens), which aims to enhance jailbreak attacks on LLMs by appending eos tokens to input prompts. The study reveals that this method can significantly improve the ASR of existing jailbreak strategies by shifting the hidden representations of harmful prompts towards harmless concept spaces, thus bypassing ethical boundaries. 

Experiments conducted on 12 different LLMs, including Llama-2, Qwen, and Gemma, demonstrated that BOOST is a general strategy that effectively enhances attack performance. Additionally, the study finds that eos tokens can be used as an effective jailbreak strategy on their own, comparable to other jailbreak methods.

### Strengths
1. This paper reveals the existence of interesting ethical boundaries between benign queries and malicious queries. And found that appending <eos> token can make both benign and malicious queries closer to the boundary. The author also gave an explanation for this phenomenon: <eos> tokens are regarded as neutral during the model fine-tuning stage. The authors proposed BOOST based on this observation: simply appending eos tokens after malicious queries can effectively improve the attack success rate.

2. This paper analyzes the attention map after appending eos tokens and found that adding eos tokens wouldn't affect the original semantics of the malicious query and thus won't degrade the harmfulness of the potential model response.

3. Experimental results showed that BOOST alone can be an effective jailbreak method. And by integrating BOOST, existing jailbreak attack methods would be greatly improved.

### Weaknesses
* Although the paper presents an innovative jailbreak technique, its practical effectiveness may be limited since many LLMs have built-in mechanisms to filter specific tokens, including the <eos> token, potentially rendering the attack method less viable in real-world scenarios.

* While this research explores the ethical boundaries of LLMs and demonstrates how the <eos> token can push queries closer to these boundaries, it overlooks a crucial aspect: users typically customize system prompts based on their specific needs, which inherently shifts these ethical boundaries. Therefore, a more comprehensive analysis of how the <eos> token's effectiveness varies across different system prompts would strengthen the study's findings.

* The author only discussed the character-perturbation type jailbreak defenses, which is not very practical as these methods would affect the utility of the model's performance on nominal queries.

### Questions
**Question 1** The experimental results show that the eos token pulls malicious queries toward ethical boundaries, which reminds me of how data points near decision boundaries typically exhibit higher uncertainty in classification problems. Could the author test BOOST's effectiveness against Gradient Cuff [1], which detects jailbreak prompts by analyzing the gradient norm (the gradient nom is to some extent the uncertainty based on my understanding)

**Question 2** As I mentioned in weakness 2, could the author test the BOOST's performance against system-prompt-engineering defenses like Self-Reminder [2]? It could be more supportive if the author also visualized the t-sne plot of LLMs when system prompts changed. 

****
**References**

[1] Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes. Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho.

[2] Defending ChatGPT against jailbreak attack via self-reminders. Yueqi Xie, Jingwei Yi, Jiawei Shao, Justin Curl, Lingjuan Lyu, Qifeng Chen, Xing Xie, Fangzhao Wu.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces BOOST, a new method for improving jailbreak attacks on LLMs. Unlike previous methods that require complex prompt engineering, BOOST simply appends EOS tokens to harmful prompts, leading to the successful bypassing of LLMs' ethical filters. The authors show that EOS tokens can shift harmful prompt representations toward benign concepts, making it easier to evade ethical boundaries in LLMs. The paper demonstrates BOOST's effectiveness across multiple LLMs and jailbreak methods, significantly enhancing the attack

### Strengths
1.	BOOST’s simplicity and effectiveness jailbreak attack methods introducing EOS token manipulation as a jailbreak approach.
2.	The experiments are thorough, using a range of LLMs and showing detailed quantitative results.
3.	The methodology and analysis are clearly presented, with visualizations of ethical boundaries and attention shifts, aiding in understanding the impact of EOS tokens.

### Weaknesses
1.	BOOST’s effectiveness is limited for proprietary models, which may filter EOS tokens, potentially reducing BOOST’s applicability.
2.	The mechanism by which EOS tokens affect model behavior may vary, necessitating further exploration of token influence across different model architectures.
3.	BOOST's efficiency relies on finding the optimal number of EOS tokens, which might introduce variability and inconsistency in attack performance.

### Questions
1.	Could other special tokens (e.g., BOS) offer similar effects as EOS in bypassing ethical boundaries?
2.	How does BOOST compare in effectiveness and reliability with proprietary filtering mechanisms?

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
4

### Summary
This paper proposes a simple method to bypass the safety training of LLMs by appending EOS tokens to the user prompt. This can be used in combination with other attacks such as GCG. The authors show that their attack can significantly improve the attack success rate on a sample of AdvBench prompts. They further provide some justification for the success of their attack by 1. Demonstrating that a learned “ethical boundary” exists, 2. Analyzing how EOS tokens can push representations across the ethical boundary, and 3. Showing how attention values are low for EOS tokens, avoiding the problem of “empty jailbreaks” [1].

[1] Souly, A., Lu, Q., Bowen, D., Trinh, T., Hsieh, E., Pandey, S., Abbeel, P., Svegliato, J., Emmons, S., Watkins, O. and Toyer, S., 2024. A strongreject for empty jailbreaks. arXiv preprint arXiv:2402.10260.

### Strengths
1. The proposed jailbreaking technique is simple to understand and apply.
2. The paper provides empirical evidence to support the idea that LLMs learn an “ethical boundary” during alignment, and that various attacks (including the proposed EOS attack) can push representations across this boundary.
3. The evaluation results appear rigorous. Human evaluation was performed to estimate the accuracy of their evaluation metrics, demonstrating reasonably high accuracy (92%). Where applicable, reported results were averaged over multiple trials with means and standard deviations reported.

### Weaknesses
1. The threat model assumes the attacker is able to add EOS tokens to the user prompt. To the best of my knowledge, no popular closed-source model (e.g. ChatGPT, Claude) allows this. In practice, this jailbreak is therefore mostly just applicable in the open-source setting, as addressed in the limitations section.
2. I’m not very convinced by the proposed explanation of why adding EOS tokens can successfully jailbreak the models (lines 267-288). The explanation is that EOS tokens are considered “neutral” since they always appear at the end of (prompt, response) pairs during RLHF. But its still unclear why the act of adding neutral tokens would proactively induce a shifting behavior, as opposed to having no effect on the ethical classification.

Minor suggested changes: Line 11: “Large language models” -> “Large Language Models (LLMs)”; Line 127, 136: “tokes” -> “tokens”; Line 201: “as well the” -> “as well for the”; Figure 3: add legend for harmful vs. benign colors; Lines 299-300: “still affects the response” is oddly phrased, perhaps change to “the response can be disproportionately affected by the attack itself”; Line 311: “empty jailbreak” -> “an empty jailbreak”; Line 702: “Disclose” -> “Disclosure”

### Questions
1. Can you clarify the notation in lines 152-154? In the first and third conditional probabilities, the random variable x is replaced by a condition “if x is (un)ethical.” I find this notation to be strange and unclear. At a high level, what is trying to be communicated here? Is it that, depending on z, the responses become entirely concentrated on $\mathcal{R}_\text{refuse}$ or its complement?
2. Just to clarify: for the t-SNE visualizations, are the plotted hidden representations 1. A concatenation of the representations across all tokens, 2. The representation of just the final token of x, or 3. Something else?
3. Can you provide some explanation for why you use different visualization methods for figures 2/3 and 4 (t-SNE vs. PCA)?
4. Figure 4 shows that both harmful and benign prompts are shifted towards the boundary. On one hand, this means that the model is being encouraged to comply with harmful prompts. But wouldn’t this mean that on the other hand, the model is also being encouraged to refuse benign prompts? Have you observed such refusals of benign prompts from adding EOS tokens? Does this occur as often as jailbreak success of harmful prompts?
5. In figures 5 and 12, what does the horizontal axis represent? Consider labeling this in the figure.
6. Lines 446-448 explains that the sensitivity of ICA/CO+EOS attack success to the number of EOS tokens added is due to the fact that EOS tokens can push the representations over the boundary in either direction. However, you could also say the same thing might happen for GCG/GPTFuzzer. So why are ICA/CO different?
7. In figure 8, could the reason why other tokens aren’t as successful as EOS be that the attention values are much higher? Can there be some experiments added to check this?

### Soundness
3

### Presentation
3

### Contribution
3
