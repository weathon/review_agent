# ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4, 6

## Abstract
The integration of large language models (LLMs) into a wide range of applications has highlighted the critical role of well-crafted system prompts, which require extensive testing and domain expertise. These prompts enhance task performance but may also encode sensitive information and filtering criteria, posing security risks if exposed. Recent research shows that system prompts are vulnerable to extraction attacks, while existing defenses are either easily bypassed or require constant updates to address new threats. In this work, we introduce ProxyPrompt, a novel defense mechanism that prevents prompt leakage by replacing the original prompt with a proxy. This proxy maintains the original task's utility while obfuscating the extracted prompt, ensuring attackers cannot reproduce the task or access sensitive information. Comprehensive evaluations on 264 LLM and system prompt pairs show that ProxyPrompt protects 94.70% of prompts from extraction attacks, outperforming the next-best defense, which only achieves 42.80%. The code will be open-sourced upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to propose a new defense against prompt extraction attacks which can ensure the utility and the extraction prevention simultaneously. Specifically, they first design a new soft token learning strategy under a combined objective function. Then, they propose several new metrics to measure the proportion of prompt leakage. Under the experiments on different datasets, they show that their method can consistently outperform existing methods and demonstrates a better tradeoff between utility and extraction prevention.

### Strengths
+ This paper proposes a novel strategy for defending against prompt extraction attacks.
+ The proposed method consistently outperforms existing methods.
+ The paper clearly illustrates their methodology and the detailed experimental details in their evaluation.

### Weaknesses
+ The proposed method can only be used under a threat model that the defender can obtain the parameter information of LLMs. This assumption is not always fulfilled. For instance, sometimes the LLM provider might not consider the defense, while the developer of an agent or an application would like to seek some strategy to protect their prompts. Under such a situation, this method is not applicable.
+ The proposed method requires significant training resources to enable the input embedding optimization. Can the authors specify which computation hardware they used for each backbone model?

### Questions
Please refer to the weaknesses.

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
This paper obfuscates the system prompt to prevent prompt injection attack.

### Strengths
+ The topic is important for LLM security.

### Weaknesses
- Limited novelty. While the application is defensive, the core technique of optimizing in the continuous embedding space to steer model output is conceptually similar to well-established work. Methods like GCG, Prompt Obfuscation already explore this approach.

- Insufficient baseline. This is a critical flaw in the evaluation. The chosen baselines (FILTER, FAKE, DIRECT) are all lightweight, prompt-engineering techniques that require no model optimization and add negligible computational overhead. ProxyPrompt, in contrast, requires a significant optimization process that consumes considerable computational resources (e.g., 6 hours on an H200 GPU for one GSM8K prompt). A more equitable and insightful comparison would involve benchmarking against other defense methods that also require a training or optimization phase. For instance, finetuning-based method like SecAlign or StruQ. Also, you need to compare with strong prompting-based approach, such as top DSR prompts used in TensorTrust, instead of a simple defensive prompt.

- Hard to scale to long system prompt. The experiments raise serious doubts about whether it can effectively protect the long and complex system prompts that are increasingly common and valuable in real-world applications. The paper use 16 tokens for GSM8k and claims that the full 8-shot system prompt is only 834 tokens long (I think it does not contain COT .
It is much shorter than real-world prompts, such as GPT-5 system prompt leaked.
This limitation is particularly concerning for agentic workflows (e.g., OpenHands, SWE-Agent), where system prompts can be thousands of tokens long to define tool interactions and multi-step problem-solving plans. These long prompts represent the most valuable intellectual property and are thus the most critical to protect. The paper provides no evidence that ProxyPrompt can scale to this level of complexity, which is arguably where such a defense is needed most. The information bottleneck imposed by a short proxy seems fundamentally limiting.

### Questions
Can you show that ProxyPrompt be able to extend to much longer system prompt while preserving the utility?

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
This paper introduces ProxyPrompt, a novel defense mechanism against prompt extraction attacks on large language models (LLMs). Instead of trying to prevent prompt leakage through filtering or instruction-based methods, ProxyPrompt replaces the original system prompt with an optimized proxy in embedding space that maintains functionality for legitimate users but becomes semantically unusable when extracted by attackers. The authors evaluate their approach across 264 configurations spanning multiple models (Phi-3.5-mini, Llama-3.1-8B, Llama-3.1-70B) and tasks (GSM8K, Roles, CoLA, SST-2, QNLI), demonstrating 94.70% protection against extraction attacks—significantly outperforming existing methods. The paper also introduces semantic-level metrics for evaluating prompt leakage and validates the approach on real-world applications including HuggingChat's Image Generator and ALFWorld multi-step reasoning environments.

### Strengths
- Novel and impactful approach: The core idea of replacing system prompts with embedding-space-optimized proxies represents a fundamentally different defense strategy compared to existing prompt-based or filtering-based methods, addressing key limitations of prior work.
- Comprehensive evaluation: The extensive testing across diverse models, tasks, and configurations (264 total) provides strong empirical evidence for the approach's effectiveness. The inclusion of real-world case studies (HuggingChat, ALFWorld) enhances practical relevance.
- Strong empirical results: The paper demonstrates clear superiority over baselines (Filter, Fake, Direct) across all metrics, with 94.70% protection rate versus 42.80% for the next-best method.
- Practical considerations: The paper addresses important deployment concerns such as combining proxy prompts with non-sensitive instructions and shows how the method scales for different query set sizes.

### Weaknesses
1. Limited attack scenario coverage: The optimization relies on a single trivial attack query (Q'), which may not fully represent sophisticated adaptive attacks. While the paper tests a 10-round adaptive attack on Image Generator, more comprehensive testing against multi-stage, context-aware attacks would strengthen the claims.
2. Computational cost concerns: The optimization process can take up to 24 hours for complex tasks like ALFWorld, which may be prohibitive for large-scale deployment. While parallelization is mentioned, the paper doesn't quantify the practical implications for systems with hundreds of prompts.
3. Insufficient discussion of sensitive high-level intent: The paper notes that classification tasks sometimes leak high-level intent but dismisses this as non-sensitive. However, in many real-world scenarios (e.g., medical diagnosis, content moderation), high-level classification rules are precisely the confidential information that needs protection.

### Questions
1. Attack scenario coverage: The paper acknowledges that the current results represent a "lower bound" on performance since the optimization uses a trivial attack query (Q'). Could you provide more details on how the method would perform against more sophisticated adaptive attacks? Specifically, what would be the impact if attackers used multiple rounds of context-aware queries that gradually build a complete picture of the system prompt?
2. Computational cost and scalability: The optimization time for ALFWorld (24 hours) is significant. How would this scale for a service with thousands of system prompts? Are there techniques to reduce this cost (e.g., transfer learning between similar prompts, approximate optimization methods), and what would be the trade-offs in protection effectiveness?
3. Sensitive high-level intent: In classification tasks, the paper states that "high-level intent is often not confidential, while protecting detailed behavior is more critical." However, consider a medical diagnosis system where the high-level classification rules (e.g., "classify as high-risk if age > 65 and symptoms X, Y, Z") are precisely the sensitive information that needs protection. How would ProxyPrompt handle such cases, and could it be adapted to protect these high-level rules?
4. Ethical implications: The ethics statement mentions potential misuse but doesn't elaborate. Could you discuss specific scenarios where ProxyPrompt could be used maliciously (e.g., to hide harmful instructions from oversight), and what safeguards or detection mechanisms might be implemented to prevent such misuse?
5. Comparison with hierarchical instruction schemes: The paper mentions hierarchical instruction schemes (Hines et al., 2024) as complementary to ProxyPrompt but doesn't test them together. How would combining ProxyPrompt with these hierarchical schemes affect overall security and utility, and would this be a recommended best practice for deployment?

### Soundness
2

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
4

### Summary
The paper mitigates system prompt extraction (a practical sub-task of direct prompt injection) by encoding the system prompt into soft embeddings. ProxyPrompt replaces the original discrete system prompt with the continuous optimizable embeddings (soft prompts) with the same token length. For a specific system prompt to be protected, the paper optimizes the corresponding soft prompts, aiming at that (1) the new soft prompts function the same as the original system prompt in utility (without any attack); (2) any attempts (user prompts) trying to steal the original prompt will not elicit the model to output it, or a string that functions similarly as the system prompt. Comprehensive experiments show that ProxyPrompt effectively protects various system prompts from five tasks on three models, greatly outperforming the baseline.

### Strengths
1. The paper targets an important problem: the system prompt can be an asset of an LLM application, and once it is stolen successfully by one attacker, the commercial competence of the application is degraded.
2. Concurrent to [1] (that does not count against novelty as it goes public after the submission deadline), the paper proposes a new model-based direction to protect the system prompt. In contrast to prompting or guardrails, ProxyPrompt enlarges defender’s optimizable space (multiple soft embeddings) to unlock stronger protection, with understandable improvements.
3. The protection is supported by comprehensive experiments. On five benchmarks, three models, the authors evaluate against 193 attack queries from multiple papers, and compare with three defense baselines. I appreciate the proposed Semantic-Match (SM) and Most-Similar (MS) metrics, which count successfully extracted paraphrased system prompts that function similarly.

### Weaknesses
1. Despite the promise of this new defense direction, it has an inherent drawback. If the system developer wants to modify parts of the system prompt, the proxy prompt optimization should be redone. The computational overhead, despite reasonable, hinders the development and iteration of the system. Thanks for reporting the cost in Appendix G (For GSM8K, optimization takes approximately 6 hours with L-70B, 30 minutes with L-8B, and 25 minutes with P-3.8B. For other tasks such as CoLA, the optimization times are 2.5 hours, 18 minutes, and 12 minutes).
2. It is not new to optimize soft prompts for LLM safety/security. Optimizing exactly the same parts (newly added soft prompts), [2] prevents jailbreaks and [3] prevents prompt injections. Even the optimization goal is different, the overall technical novelty is limited, and the authors do not discuss those related work.

3. The presentation is not very clear. Fig 1 is too complex with long texts. Fig 2 illustrates the complex loss design, but I expect to see the defense framework somewhere, see Fig 4 in [2] and Fig 1 in [3]. The presentation of method (using P, Q, P’, Q’) and results (using multiple metrics) are also hard to interpret.

[1] You Can’t Steal Nothing: Mitigating Prompt Leakages in LLMs via System Vectors, CCS’25

[2] On Prompt-Driven Safeguarding for Large Language Models, ICML’24

[3] Defending Against Prompt Injection with a Few DefensiveTokens, AISec’25

### Questions
The paper is sound and well-motivated. I don't think my score will likely change by rebuttal interactions, as the mentioned weaknesses are inherent or require naunced revisions. I am looking forward to a new paper version if the authors have time.

### Soundness
4

### Presentation
2

### Contribution
3
