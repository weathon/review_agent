# Weak-to-Strong Jailbreaking on Large Language Models

- Decision: Reject
- Scores: 6, 6, 6, 3, 6

## Abstract
Large language models (LLMs) are vulnerable to jailbreak attacks -- resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient method to attack aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new jailbreaking attack against large language models. This paper starts with an interesting observation: a weak (small) model can guide the token generation of a larger (stronger) model. The paper leverages this observation by using a weaker unsafe model to generate harmful output. The attack samples a new token based on a weighted probability distribution of tokens output from a weak unsafe model and a stronger safe model. Once the first output tokens of the stronger model have been altered (through the weaker model), it will start generating harmful content. The paper shows the evaluation of the attack on different models from different families: llama, InternLM, and Baichuan 2. In all cases the safe models were broken. The paper concludes with some defense approaches.

### Strengths
This paper introduces an attack on "safe" large language models showing that we are still far away from robust alignment. By manipulating the first few output tokens during the generation, a large safe model can be tricked to produce harmful content. This attack requires no gradient descent or other computationally expensive operation. A single forward pass over the large model is enough. Further, the attack does not require manipulating the parameters of the safe model or re-training it in to remove its safety features. As the paper indicates, this attack has implications on red teaming large language models prior to release and deployment. 

The paper also discusses potential defenses against this attack and evaluates them to show that the adversarial success rate can be driven down. Of course, there is no guarantee that a defended model cannot also be jailbroken, but it is appreciated that the paper closes the loop by proposing a defense.

### Weaknesses
The paper does not explicitly discuss a threat model: what kind of access is needed to the model under attack. The paper appears to require access to token generation of the LLM (tokens + output probabilities) and the ability to intervene in the output generation. It is not completely a passive attack. The paper should include a threat model section that explicitly indicates the needed access to the model under attack.

The attack requires full access to the output token distribution of the safe model, which -as the paper concedes- is not always possible. This is more the case with commercial models. As such, it is not clear that commercial models are vulnerable to this attack. The paper mentions that token alignment or reverse engineering is possible. As preliminary evidence, it shows that unsafe llama could be used to attack Mistral through the use of token alignment. However, the harm score of Mistral appears to be similar to that of the unsafe model. As such, the paper should include further evaluations where unsafe weak models are tested against stronger safe models of different families. In other words, the paper should expand on the Mistral experiment and show whether the harm score increases (with statistical significance) compared to the output of the weak model. 

Related to the above, the paper shows in table 4 the difference in harm score between unsafe and safe models for two datasets. To establish that the attack makes the output more harmful, the result should include statistical significance. The harm score between the unsafe and attack model should show statistically significant increase. As such, the paper should conduct statistical significance tests to show that the attack increases the harm scores of the outputs of attack models.

### Questions
Do these results hold with newer open source LLMs, such as Llama 3?

### Soundness
3

### Presentation
4

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
The paper proposes a method for jailbreaking LLMs that involves using two weaker LLMs: one that is safely aligned (safe weak LLM) and another that is either unaligned or has had its alignment removed (unsafe weak LLM). During token generation, the ratio of the output probability from the unsafe weak LLM to that of the safe weak LLM is calculated. This ratio is then multiplied by the output probability of the target strong LLM, thereby causing the target strong LLM to exhibit behavior similar to the unsafe LLM.

### Strengths
S1: The research on jailbreak attacks on LLMs is important.

S2: The idea of using probability differences in smaller models to guide the behavior of larger models is simple yet effective.

S3:The experiments are thorough and well-designed, encompassing a variety of aspects such as the limitations of weak LLMs, effectiveness across multiple languages, and the impact of system prompts. 

S4: The responses generated by the jailbroken larger LLM using the proposed method demonstrate higher harmfulness scores compared to the smaller unsafe LLM used for the attack. This suggests that the larger LLM is not merely imitating the unsafe model, adding to the significance of the paper's contribution.

### Weaknesses
W1:High Usage Costs: The proposed jailbreak method may have high usage costs. For each malicious prompt, three LLMs are required to be maintained simultaneously: a safe weak LLM, an unsafe weak LLM, and a safe strong LLM. This overhead may undermine the practicality of the approach.

W2: Availability of Unsafe Models: Unsafe LLMs may sometimes be challenging to obtain. If attackers need to fine-tune a model to create an unsafe version, it requires access to malicious datasets and involves additional computational overhead. While it is possible to use existing unsafe models available online, the authors do not provide experimental results using different types of weak and strong models. Is this feasible, and if so, what are the outcomes?

W3:Limited Applicability and Alternative Approaches: The proposed method requires direct manipulation of output probabilities, which implies white-box access. Many powerful models, however, are only available via APIs, limiting the applicability of this method.

### Questions
This paper has some similarities to concurrent work on proxy-turning. I do not believe this undermines the novelty of the current paper, and I appreciate the authors' discussion of related work. However, due to some differences in how output probabilities are calculated between this paper and proxy-turning, I am curious whether proxy-turing could be used for jailbreaking, and if so, what the effectiveness would be.

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
3

### Summary
This paper proposes a method for leveraging smaller jailbroken LLMs to attack larger LLMs. The paper first presents an observation: the jailbroken model and the original safe model only differ in the initial decoding distribution. Based on this, it introduces a technique using two smaller models to alter the decoding probabilities of a larger safe LLM in order to achieve a jailbreak.

### Strengths
1. The proposed method successfully jailbreaks larger LLMs (e.g., 70B) without requiring high computational costs and complex prompt engineering.

2. The proposed method performs well on open-source LLMs, achieving nearly 100% ASR.

3. The paper is well-written.

### Weaknesses
1. The proposed method lacks technical novelty, as previous work [1] also employed smaller LLMs to guide larger LLMs at decoding time as an alternative to fine-tuning. Specifically, [1] first calculates the logit difference between two smaller models (a base model and a fine-tuned model) and then adds this difference to the logits of the larger LLM to achieve a fine-tuning effect. This paper uses a similar approach but focuses primarily on safety scenarios, which, in my view, presents limited novelty.

2. The proposed method is not particularly practical. For instance, it requires a smaller jailbroken LLM and assumes that the larger LLM and the smaller one share the same vocabulary. These limitations make it challenging to apply this approach to SOTA models like GPT-4 and Claude.

[1] Liu A, Han X, Wang Y, et al. Tuning language models by proxy[J]. arXiv preprint arXiv:2401.08565, 2024.

### Questions
The method proposed in this paper requires the use of some harmful data to fine-tune the smaller model to obtain an unsafe LLM. Therefore, I believe that demonstrating the method's effectiveness hinges on whether the larger LLM can provide responses that are more harmful and informative than those from the smaller unsafe LLM. Table 4 partially indicates that the harmful score of the jailbroken larger LLM is higher, but I am interested in specific examples that could illustrate, more intuitively, the qualitative differences (in terms of harmfulness and informativeness) between the responses of the larger and smaller LLMs.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes a weak-to-strong jailbreaking techniques by intervening the decoding of a safe and strong LLM with the generation probability of an unsafe and weak LLM and eliciting an amplified effect of unsafe generation.

### Strengths
The idea is simple and the approach demonstrates effectiveness on benchmarks like AdvBench and MaliciousInstruct.

### Weaknesses
First, the paper lacks a convincing threat model. It is reasonable for jailbreaking by adding optimized prompt prefix or suffix like GCG and Prefix Injection do, however, weak-to-strong jailbreaking makes a very strong assumption that the safe LLMs could be intervened in every step of token generation, making the application of the proposed approach questionable. If the service provider is not the attacker, how does the attacker intervening the every step of token generation of an aligned LLM, and if the service provider is the attacker, what is the meaning of jailbreaking in this way? 

Second, the paper lacks a study of why weak-to-strong attack can outperform adversarial fine-tuning since it is not intuitive that with the intervention of a weak unsafe model, a strong safe model can beat a strong unsafe model (Why the amplification effect exists and **why it can surpass the harmlessness of a strong unsafe model**, which seems is the performance ceiling in the testing setting).

----
[Update during rebuttal]

During the rebuttal, the authors admitted that the amplification/transfer (i.e., weak-to-strong) effects had already been proposed in existing works [2,3], as they referred me to the explanation in [2]. In order to gain a deeper understanding toward their response to my second question, I conducted a careful reading of relevant papers and re-evaluated the contribution of this submission. 

Based on my reading, the paper employs the same concept of transfer/amplification as [1][2], and uses nearly identical methodology (see Equation 3 in [2], Equation 2 in [1] and Equation 1 in this submission, along with very similar illustrative figures). [1][2] work on testing/training-time language detoxification. In comparison, this submission works on an inverse task, language toxification. As a result, the main difference lies in the inversion of the calculation between the logits, i.e., from (safe-unsafe)[1,2] to (unsafe-safe). Given these points, I believe the scientific novelty and contribution of this paper are insufficient for a top ML conference.

Moreover, this submission shares an idea similar to a published work [3] in testing stage but lacks proper comparative results (as also pointed out by Reviewer MSNL), yet the authors claim it is a "concurrent" work. I claim the existence of [3] does not influence my decision. Yet results in [3] show that decoding-time proxy tuning is still outperformed by direct fine-tuning of the model with the same capacity on the majority of tasks, which is not aligned with the observation in this submission, indicating that further investigation is needed.

**Overall, I vote for giving appropriate credit to the papers that presented the original ideas** [1][2]. As authors describe [1][2] in its related work section as "[1] proposes a decoding time method for controlled text generation by combining target LLM with 'expert' LMs and 'anti-expert' LMs, but focusing on language detoxification and controlling the sentiment of base generation. 
[2] utilizes the same [1] equation as a tool for analyzing the contribution of scaling up between model knowledge and instruction-tuning abilities." These statements subtly circumvent the relationship between the methodologies of [1,2] and theirs. Given these, I feel it is suitable to describe this paper as "it utilizes the equation introduced in [1,2] for analyzing the contribution of scaling up introduced in [2] on an inverse task, i.e., language toxification".

After conducting a careful reading of these relevant papers, I believe this submission lacks of sufficient scientific novelty and contribution, and have raised my confidence score from 3 to 4. 

[1] Liu, Alisa, et al. "DExperts: Decoding-time controlled text generation with experts and anti-experts." arXiv preprint arXiv:2105.03023 (2021).

[2] An Emulator for Fine-Tuning Large Language Models using Small Language Models, ICLR 2024

[3] Tuning Language Models by Proxy, COLM 2024

### Questions
See weaknesses please.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a jailbreaking attack on open-source large language models. By using a smaller, "weak" model which has been finetuned on harmful behavior, an attacker can augment the output probabilities of a larger "strong" model such that it bypasses safety alignment. 

The paper evaluates its method on AdvBench and MaliciousInstruct, and compares against prior methods for jailbreaking on open-source models.Finally, the paper evaluates using gradient ascent for the strong model as a defense against the proposed attack.

### Strengths
The method is novel, and demonstrates a clear risk for safety in open-source models. Moreover, the attack reduces the computational requirements relative to prior approaches.

### Weaknesses
The paper would be stronger if it included further justification for the claim in sec. 3.2 ("Comparison to Naive Baselines")
Namely, the paper demonstrates that the method "goes beyond merely copying the small unsafe model". While there is some indication that this may be true (based on the GPT-4 score of table 2), it would be clearer if there was some more direct comparison of outputs from the weak model and the strong model (e.g. based on some similarity metric such as ROGUE or BLEU). 


I am also worried about the choice of evaluation benchmarks. Some recent work [1,2] demonstrates that prior datasets may not be diverse. This seems to line up with the paper's findings (table 3), where the fine-tuned model is already reaching ASRs of > 90%. 
The paper can be stronger if it includes more datasets, demonstrating the the proposed method indeed generalizes.

[1] Li, Lijun, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, and Jing Shao. “SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models.” arXiv, June 7, 2024. http://arxiv.org/abs/2402.05044.

[2] Xie, Tinghao, Xiangyu Qi, Yi Zeng, Yangsibo Huang, Udari Madhushani Sehwag, Kaixuan Huang, Luxi He, et al. “SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal Behaviors.” arXiv, June 20, 2024. http://arxiv.org/abs/2406.14598.

### Questions
- I'm having a hard time contextualizing the meaning of the GPT-4 scores. Could the authors provide some examples of how e.g. a score 3 output would differ from a score 4 output? How significant is a .5 increase in this gpt-4 score?

### Soundness
3

### Presentation
3

### Contribution
3
