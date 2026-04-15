# AutoRedTeamer: An Autonomous Red Teaming Agent Against Language Models

- Decision: Reject
- Scores: 5, 3, 3, 5

## Abstract
As large language models (LLMs) become increasingly capable, robust and scalable security evaluation is crucial. While current red teaming approaches have made strides in assessing LLM vulnerabilities, they often rely heavily on human input and fail to provide comprehensive coverage of potential risks. This paper introduces AutoRedTeamer, a unified framework for fully automated, end-to-end red teaming against LLMs. AutoRedTeamer is an LLM-based agent architecture comprising five specialized modules and a novel memory-based attack selection mechanism, enabling deliberate exploration of new attack vectors. AutoRedTeamer supports both seed prompt and risk category inputs, demonstrating flexibility across red teaming scenarios. We demonstrate AutoRedTeamer’s superior performance in identifying potential vulnerabilities compared to existing manual and optimization-based approaches, achieving higher attack success rates by 20% on HarmBench against Llama-3.1-70B while reducing computational costs by 46%. Notably, AutoRedTeamer can break jailbreaking defenses and generate test cases with comparable diversity to human-curated benchmarks. AutoRedTeamer establishes the state of the art for automating the entire red teaming pipeline, a critical step towards comprehensive and scalable security evaluations of AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces an automated framework called AutoRedTeamer for red teaming large language models. Unlike existing methods that rely on human intervention, AutoRedTeamer uses an LLM-based agent architecture with five specialized modules and a memory-based attack selection mechanism. This allows for end-to-end automation of seed prompt generation, attack selection, execution, and evaluation. The results show that AutoRedTeamer achieves a 20% higher attack success rate on the HarmBench dataset while reducing computational costs by 46%, and it is able to break defenses and generate test cases comparable to human benchmarks.

### Strengths
1. This paper focuses on an important research topic and provides valuable contributions to the community.

2. The paper is well-written and easy to follow overall.

3. The evaluation results demonstrate the potential of the proposed method for effective and efficient automatic red teaming.

### Weaknesses
1. The technical contribution, as currently presented, is not significant.

2. The evaluation lacks a detailed description.

### Questions
1. The core of the proposed AutoRedTeamer framework is an agent-based ensemble system that dynamically and adaptively selects and refines attack strategies while generating jailbreaking test cases. However, the current text does not elaborate on the technical details of each module in the framework. For instance, when introducing the seed prompt generation module, the authors state that "Each scenario is comprehensively defined, including a unique identifier, detailed description, expected outcome upon target AI failure, and the specific input for the target." However, it is unclear what specific prompts are used to ensure the correctness of the generated seed prompts. The same lack of detail applies to other modules, such as risk analysis and memory-based attack selection. Without these details, it is difficult to quantify the technical contributions of the proposed framework.

2. When evaluating the efficiency of the proposed method, the authors report the total number of queries AutoRedTeamer needs to successfully generate the test cases. However, AutoRedTeamer itself is an LLM-based agent, where each module is an LLM. This means that the overhead cannot simply be measured by the number of queries made to the target model; the time taken by the agent LLM (Mistral-8x22B-Instruct-v0.1) to generate intermediate results should also be considered. Therefore, I suggest that the authors report this overhead (in terms of exact time) and compare it with baseline methods.

3. Several recent LLM jailbreaking techniques shall also be considered as baselines, such as [1]

[1] Zeng, Yi, et al. "How johnny can persuade llms to jailbreak them: Rethinking persuasion to challenge ai safety by humanizing llms." arXiv preprint arXiv:2401.06373 (2024).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper designs an agent to do the automated red-teaming and shows the improved performance over selected baselines.

### Strengths
+ The paper is easy to follow
+ The LLM jailbreak topic is trending

### Weaknesses
- Marginal novelty
- The improvement is not significant
- Lack of strong baselines
- Need to conduct experiments on more well-aligned models

First of all, I think the idea of using LLM to generate jailbreak prompts is already well-studied since last year, and I do not see significant novelty in this paper compared with tons of similar papers using LLM to mutate/transform/generate new jailbreak prompts. The paper claims that it is the first paper as fully automated, which I cannot agree. Most published black-box jailbreak work just use a set of collected prompts as seeds or just start with some prompts, and then have LLM automatically do the prompt transformation. For example, [1] just starts with some prompts and have LLM automatically do the jailbreak, with is identical to your setting, and I believe there are too many works liks that without human in the loop. Thus, I do not think it is an accurate claim as the first fully automated jailbreak paper.

Second, I think this paper uses agentic approach and combines existing jailbreak strategies while the improvement is not significant, especially considering that there are many black-box attack papers that have near-perfect attack success rates such as [1,2,3].  I think the paper should compare with some latest and stronger attackers since PAIR is published last year and many stronger jailbreak approaches were found after that.

Although I appreciate that the authors conduct experiments on the 70B model, I believe some more aligned models can be attacked with, such as Llama-2, and Claude-3. It would make the experiments more comprehensive with more models in the evaluation.

Lastly, as an attack paper, I think the ethical concern discussion and response disclosure are missing from the manuscript.

[1] GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation

[2] WordGame: Efficient & Effective LLM Jailbreak via Simultaneous Obfuscation in Query and Response

[3] Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks

### Questions
See the comments above

### Soundness
2

### Presentation
2

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
This paper proposes a framework to automatically generate jailbreaks/harmful prompts, evaluate their effectiveness, and iteratively refine the prompts to improve their performance. This is done via a series of LLMs given specific tasks in the overall pipeline. The method does have good performance, generating jailbreaks in fewer queries compared to other methods.

### Strengths
- The method is empirically effective - e.g. in Table 1 the ASR is higher (or at least comparable) to the baselines with fewer queries required. Furthermore, from Fig. 8 the fact that many attacks are composed of multiple simple/cheap attack styles does indicate that the method is creating more complex attacks as it iterates over prompts.

- The attack is also robust against defences: for example, the perturbations in SmoothLLM don't significantly degrade attack performance.

-  The method allows fully end-to-end red-teaming with only an initial domain specified (e.g. “cybersecurity”). There is scope here to enhance AutoRedTeamer with attacks tailored to particular use cases that a user is concerned for.

### Weaknesses
- Generally, I found the paper to lack detail and technical clarity and the overall paper flow would need improving. Some useful information is for example in the appendix (e.g. pesudocode), which given the low level of technical description in the main body of text could be given a more visible position. Some concrete examples which can use more technical description: in Section 3.2 Risk Analyser, the prompting strategy is not described; alternatively how the memory store works can use more  detail in the paper given its importance to the results and paper claims. Overall, just based on the content of the paper it would be highly challenging to reproduce the results diminishing the contribution of the work.

- The authors raise RedAgent and ALI-agent as similar work using agents to automatically red-team, and additionally state that their use of a memory store is novel. However, those two papers are similar to this one in proposing agentic frameworks and additionally utilise memory. The authors argue that their work is differentiated by the lack of a need for predefined test scenarios and support for a more diverse range of attacks. However, this may not suffice as a compelling degree of novelty - the range of attacks from the other works can be easily extended. Similarly, attacks such as GCG are not compared to because they have fixed test cases: however, in that case, would GCG combined with the seed prompt generation described in the paper be superior to the proposed attack piepline? I feel the authors don’t fully justify why it is so difficult to generate seed prompts and combine it with other attack pipelines.

- Rainbow Teaming is mentioned as the closest work, yet is not evaluated/compared to. Likewise Wildteaming [1] can be considered similar work and could use comparison or at least discussion.

[1] Jiang, Liwei, et al. "WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models." arXiv preprint arXiv:2406.18510 (2024).

### Questions
- From the transition matrix it seems like many of the attack methods are rarely used. How do you ensure that the strategy designer LLM strikes a balance between exploring different attacks and exploiting successful attacks? Is an LLM the best choice for this as opposed to say RL or similar techniques?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes an agent-based framework for automatically generating attacks on LLMs, featuring an attack memory bank designed to enhance attack efficiency. The framework demonstrates strong attack performance on HarmBench.

### Strengths
1. The design of the risk analyzer within the agent, and the storage format of each entry in the attack memory bank for retrieval purposes are quite interesting.
2. The paper is well-written, with a clear structure. The experiments are extensive.

### Weaknesses
1. This paper emphasizes an end-to-end automated red teaming framework; however, the experimental dataset is somewhat limited, lacking exploration of threats in real-world scenarios. It would be more meaningful if the authors could discuss threats in practical contexts, such as embodied scenarios, which would also strengthen their claims about flexibility, adaptability, and scalability.
2. The claims regarding efficiency need further substantiation, as the current experiments do not provide sufficient support.

### Questions
1. Why is only HarmBench used in the evaluation? This alone is insufficient to demonstrate the proposed attack framework's flexibility, adaptability, and scalability. Please consider exploring some real-world security threat scenarios, which are more complex and meaningful than the questions in HarmBench.
2. While total queries and total tokens provide insights into efficiency, I am more interested in how long it takes for the end-to-end attack framework to generate an effective attack prompt. Since the end-to-end agent consists of multiple modules, there may be delays in communication between these components, which could negatively impact efficiency. This complexity may be a disadvantage compared to previous methods that do not involve such intricate module interactions.

### Soundness
2

### Presentation
3

### Contribution
2
