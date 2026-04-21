# Safeguarding System Prompts: A Surrogate-Based Defense Against Injection Attacks

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 3

## Abstract
System prompts, essential for guiding model outputs, play a pivotal role as large language models proliferate across diverse applications. Despite their importance, these prompts are highly vulnerable to injection attacks. Intuitively, adding defensive prompts and implementing output filtering could offer strong protection, but these defenses rely on direct access to the system prompt—a luxury increasingly unavailable in today’s evolving prompt market and third-party defense scenarios, where prompts must remain concealed and confidential. To address this pressing limitation, we introduce SurF (Surrogate-based Filtering), a novel approach that compensates for the lack of system prompt access by utilizing a surrogate prompt pool. Namely, we leverage the prompt pool as the surrogate of the system prompt. Once a potential leak from this pool is identified, the input is classified as harmful, and the system resists generating a response. Experiments on various models, including both offline and online LLM services, demonstrate SurF’s effectiveness in reducing attack success rates. Furthermore, we evaluate the trade-off between defense robustness and response consistency on natural inputs using a response-following metric. Our findings indicate that while stronger defenses reduce attack success, they may also degrade the quality of legitimate responses.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The author proposed a new defense mechanism namely 'SurF' to act as a raw-prompt agnostic detector against prompt-injection-based stealing attacks, which is exclusive for scenarios where the defender has no access to the original system prompts as they may be confidential. 'SurF' exploits a bunch of surrogate prompts to test the user query and identifies the malicious query according to the output of the LLM. To evaluate the performance of the defense, the author proposed RFM and ACC as the metrics to show functional preservation and robustness. Extensive experiments demonstrated the effectiveness of the proposed method across a variety of mainstream LLM in comparison with the baseline where the defender is able to reach the system prompt.

### Strengths
1. Good writing, easy to follow.

2. The proposed method is simple yet effective, making it easy to reproduce.

3. SurF is a training-free method, it can be implemented easily.

### Weaknesses
1. This paper is based on an impractical threat model by assuming the defender is ignorant of the system prompt. In a real-world implementation, the system prompts are usually processed by the user or the service provider according to my knowledge. In the former case, there is no need for the prompt stealing attack as the user himself is aware of the prompt. For the latter, there is no need to use 'SurF' as the system prompt is accessible. Moreover, in either case, the service provider *MUST* have access to the system prompt otherwise it cannot be fed into the LLM. In case I misunderstood the scenario where the proposed method is supposed to function, I strongly suggest the author give a detailed real-world example illustrating why the system prompt can be inaccessible to both the service provider and the user.

2. I believe that the detecting mechanism of 'SurF' can be easily defeated when the stealing prompt asks the LLM to give trivially encrypted outputs, e.g. in Caesar cipher, which can result in a low CS score as well as a low WR score, so as to break 'SurF'. Note that the LLMs are capable of yielding such outputs according to [1,2].

3. The number of surrogate prompts K may influence the performance of SurF from Table 2, seemingly increasing the query cost by 5~10 times.

4. Only simple attacks and baselines are evaluated in the paper, an effective defense is supposed to be robust to the SOTA attacks [3] as well as the adaptive attacks (as narrated above).


[1] Glukhov, D., Shumailov, I., Gal, Y., Papernot, N., & Papyan, V. Position: Fundamental Limitations of LLM Censorship Necessitate New Approaches. In Forty-first International Conference on Machine Learning.

[2]  Yuan, Y., Jiao, W., Wang, W., Huang, J.-t., He, P., Shi, S., and Tu, Z. Gpt-4 is too smart to be safe: Stealthy chat with llms via cipher. In The Twelfth International Conference on Learning Representations, 2023

[3] Sha, Zeyang, and Yang Zhang. "Prompt stealing attacks against large language models." arXiv preprint arXiv:2402.12959 (2024).

### Questions
1. In which scenario is the proposed threat model practical? Does the author indicate one in which both the service provider and the user have no access to the system prompt? If so, why is it realistic?

2. Will SurF remain robust if the outputs are in cipher? If not, how to improve the proposed method?

3. Will SurF significantly increase the inference cost? How much is the extra query cost supposed to increase?

4. How robust is SurF against the SOTA attacks? And how resilient is it towards the mentioned adaptive attack?

5. The y-axis and the values in Figure 3 do not seem to be correct. The first row of the heatmap seems to refer to the performance when the LLM behind the service is Vicuna-7b, according to the APP values and the relative improvements as they sum up to 24.67, which is the no defense APP of Vicuna-7b in Table 1. I am confused to see that, in this case, the APP becomes even higher when exploiting some different proxy model (e.g., in the first row, second column it gets 27.29). What is your explanation for this phenomenon?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper propose a defense method towards an injection attack in which attackers use prompt to trigger the adversary to real the system prompt. In order to mitigate the attack, the authors adopt an idea to detect whether the response of the LLM contains system prompt. In the scenario that system prompt is not available, the authors propose to use surrogate system prompts for detection. The authors propose two ways: i) use string matching to see whether the surrogate system prompts match the LLM output. ii) calculate embedding similarity to see whether the surrogate prompts match the LLM output.

### Strengths
1. The idea is very easy to understand, which can easily be accessible to general audience.   

2. The figures are aesthetic.

### Weaknesses
1. The attack motivation is not clear. Why the attackers want to extract the system prompt?

2. The setting that the defender has no access to system prompt is not realistic. If defenders (like OpenAI) don't have access to system prompt, how should they input the full prompts to their LLM in delopyment?

3. Technical contribution is very limited. The defense method is more suitable for a technical blog, but definitely not for a scientific conference like ICLR.

### Questions
NA

### Soundness
1

### Presentation
4

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper presents the following defense against prompt injection attacks that aim to extract the system prompt: run the input multiple times with surrogate prompts instead of the real system prompt and check if the surrogate prompts appear in the response.  If they do, reject the input.

### Strengths
- Simple, straightforward idea
- Defense can be run as a filter on top of a black-box system because it does not require knowledge of the actual system prompt

### Weaknesses
- The proposed defense assumes that the extracted prompt appears in the model's response.  This assumption is false for many prompt extraction methods: they reconstruct the prompt from responses to normal queries or model logits, etc.  See Morris et al. "Language Model Inversion" (ICLR 2024), Sha and Zhang "Prompt Stealing Attacks Against Large Language Models" (arXiv), Zhang et al. "Extracting Prompt by Inverting LLM Outputs" (EMNLP 2024), Shen et al. "Prompt Stealing Attacks Against Text-to-Image Generation Models" (USENIX Security 2024), He et al. "Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation" (arXiv).  

None of these papers are even cited in the submission, presenting a very partial and limited view of prompt extraction attacks.

- The proposed defense imposes a huge inference-time cost because every input needs to be re-run with multiple surrogate prompts.  This looks like a deal-killer for any practical deployment.

- There is no evaluation for adaptive adversaries who are aware of the defense.  For example, what if the adversary asks the system to return the system prompt but spell it backwards or in a different language, etc.  Since detection is based on looking for exact substrings of the prompt in the response, I don't think the defense works in this case.

### Questions
I think all of these are necessary before this submission can be accepted:

- Measurement of the inference-time cost of the defense (vs. other defenses and undefended systems).

- Evaluation of how effective the defense is against adversaries who are aware of the defense and try to evade it.

- Acknowledgment that the defense applies only to a narrow slice of prompt extraction attacks and is ineffective against attacks that reconstruct the system prompt rather than trick the system into disclosing it in the response.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper introduces SurF (Surrogate-based Filtering), an innovative method for defending against injection attacks on system prompts without direct access to them. SurF uses a surrogate prompt pool to identify potential leaks and classify harmful inputs, preventing responses from being generated. Experiments show SurF's effectiveness in reducing attack success across various LLMs. However, stronger defenses can impact the quality of legitimate responses, highlighting the need for balance between security and response consistency.

### Strengths
The paper introduces a comprehensive and practical approach, SurF, which simulates prompt-output interactions to detect malicious inputs without needing access to raw system prompts. 
This makes it particularly well-suited for third-party defense scenarios where confidentiality is essential. Additionally, the extensive experiments conducted across multiple offline and online LLM platforms validate SurF's effectiveness in reducing prompt injection attack success rates.

### Weaknesses
1. There is significant ambiguity in the threat model. What capabilities should an attacker possess during the attack process? Can the attacker control the LLM? Is the attacker able to fine-tune the LLM?
2. The evaluation metric for data filtering is unclear. The paper employs the metric EXC (equation (1)) to identify exact matches of prompt leakage, marking any occurrence of system prompt sentences in the reply as a successful attack. However, for data filtering (equation (4)), the authors measure the percentage of system prompt sentences present in the reply and set an 80% threshold to detect attacks. This approach could introduce bias and unintentionally degrade the performance of data filtering. Why not use the same EXC metric for data filtering as well?
3. There is considerable ambiguity in the attack setting and design as described. According to the paper, the authors attempt to separate the LLM (target) from the defense mechanism, such that the defense system lacks access to both the LLM and the system prompts used within it. This raises questions about the specific situations or scenarios where the defense system would be unable to access the system prompts. Who would adopt and deploy such a defense? The paper does not provide a realistic scenario for this setup. In practical terms, it seems illogical to separate the defense from the LLM, as it is unusual for a defense to be managed by a third party who is neither familiar with the target LLM party nor trusted by it. Thus, the motivation for ensuring confidentiality in the proposed method makes less sense to me.
4. As a result, In the SruF framework, the authors propose a setting where the system prompt is inaccessible, which introduces significant ambiguity. It is unclear who is permitted to access the system prompt and why the defense mechanism is restricted from doing so. The paper fails to clarify who can access (or cannot access) what kind of information within the framework, leading to confusion. Moreover, if the system prompt is meant to be confidential and inaccessible to the defense party, it is contradictory that surrogate prompts are generated based on the system prompt. This creates substantial confusion and inconsistency within both the threat model and the system design.
5. Furthermore, the definition of confidentiality in the context of this paper is unclear. If the surrogate prompts are generated based on the original system prompt, they must inherently contain or convey information from that original prompt. This approach represents a compromise of confidentiality, at the point of cryptography. 
6. Stronger attacks need to be considered. In the field of prompt injection attacks, certain studies, such as [1], use learning-based methods to initiate attacks. I recommend that the authors conduct experiments to test the performance of their defense mechanisms against these types of sophisticated attacks.

[1] Pasquini, Dario, Martin Strohmeier, and Carmela Troncoso. "Neural Exec: Learning (and Learning from) Execution Triggers for Prompt Injection Attacks." arXiv preprint arXiv:2403.03792 (2024).

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2
