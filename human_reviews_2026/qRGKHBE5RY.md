# PSG-Agent: Personality-Aware Safety Guardrail for LLM-based Agents

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Effective guardrails are essential for safely deploying LLM-based agents in critical applications. 
Despite recent advances, existing guardrails suffer from two fundamental limitations: 
(i) they apply uniform guardrail policies to all users, ignoring that the same agent behavior can harm some users while being safe for others; 
(ii) they check each response in isolation, missing how risks evolve and accumulate across multiple interactions. 
To solve these issues, we propose PSG-Agent, a personalized and dynamic system for LLM-based agents.
First, PSG-Agent creates personalized guardrails by mining the interaction history for stable traits and capturing real-time states from current queries, generating user-specific risk thresholds and protection strategies.
Second, PSG-Agent implements continuous monitoring across the agent pipeline with specialized guards, including Plan Monitor, Tool Firewall, Response Guard, Memory Guardian, that track cross-turn risk accumulation and issue verifiable verdicts.
Finally, we validate PSG-Agent in multiple scenarios including healthcare, finance, and daily life automation scenarios with diverse user profiles. 
It significantly outperform existing agent guardrails including LlamaGuard3 and AGrail, providing an executable and auditable path toward personalized safety for LLM-based agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a novel personalized agent safety barrier. By extracting features from users' historical interactions, it provides tailored safeguards for specific users, enabling risk monitoring.

### Strengths
1. The paper introduces an interesting problem regarding personalized agent safety barriers.
2. It implements protection through multi-agent collaboration, addressing input, output, and response simultaneously.

### Weaknesses
1. The dataset labels are generated using GPT-4o, which is also the core model of PSG, leading to bias and preference issues. It is recommended to use a multi-vote approach involving human evaluators or various models for dataset construction. Additionally, since human evaluation indicates that 86% of samples are of high quality, samples with lower evaluation quality should be discarded.
2. The authors manually constructed 130 samples and then performed mutations on them. This may not sufficiently contain various situations, as it implies only 20-40 relevant samples per domain.
3. Collecting user information may pose privacy risks. The proposed method requires gathering extensive personal data to create user profiles, but some of this information may not be necessary for security measures. For instance, in Appendix D1, unnecessary information such as the user's education and economic status was collected for a cake purchase request.
4. The method requires interaction among multiple agents, increasing computational complexity and token costs.
5. The current success rate is relatively low (recall is only 0.616). The paper does not address boundary issues or ambiguous cases for decision-making errors. Solutions could include manually setting labels for boundary cases, clarifying the agent's scope of responsibilities and permissions, and employing reinforcement learning techniques.
6. According to the assumption, user metadata should be derived from conversation history. However, in the experiment, this metadata is supplied by the dataset. In real-world scenarios, acquiring such highly related metadata may prove challenging, potentially compromising the effectiveness and accuracy of the safeguards.
7. If the model chooses to reject, it would be better if a specific classification is provided, similar to what is shown in Table 5.

### Questions
1. The authors manually constructed 130 sample data points and then performed mutations on them. Does this dataset contain sufficiently diverse samples, or are the mutations simply superficial? It would be better to provide a more detailed classification.
2. Collecting user information may pose privacy risks. How can this method ensure that necessary information is collected while maintaining user privacy?
3. Does the baseline also have access to the user's metadata? If it does not utilize this metadata in the decision-making process, the comparison may not be fair.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a training-free, plug and play framework, PSG-Agent, which can personalize safety guardrails for LLM agents by adapting to each user’s traits and real-time context. Unlike existing systems that use uniform safety rules and only monitor single-turn outputs, PSG-Agent dynamically extracts user profiles and generates individualized safety criteria enforced throughout the agent’s workflow. This paper also constructs a benchmark of 2,900 personalized safety cases across eight domains and shows that PSG-Agent outperforms state-of-the-art guardrails like Llama-Guard3. Overall, the work introduces a comprehensive system for personality-aware and context-sensitive safety in LLM agents.

### Strengths
This paper introduces a very interesting and novel concept of personality-aware safety guardrails for LLM agents, formulating safety as a personalized and sequential decision process. The proposed training-free and plug-and-play PSG-Agent framework addresses the challenge of aligning safety judgments with user-specific traits and contexts. The paper itself also includes module details and examples to demonstrate.

### Weaknesses
One major concern is that PSG-Agent assumes that LLM agents can access to rich user profiles, including personal traits such as age, marital, education, etc, which are rarely available in real world settings due to data privacy regulations. The framework also assumes static and accurate user profiles, whereas in reality, user states are often ambiguous and dynamic, which could undermine personalized safety judgments. Including more failure case analyses and robustness tests under incomplete or noisy user information would strengthen the system’s interpretability and credibility. The benchmark relies heavily on GPT-4o-generated data, raising concerns about label validity and self-reinforcement bias.

### Questions
1. Since the benchmark relies heavily on GPT-4o for data generation and filtering, how do the authors address potential circular bias between data creation and evaluation?
2. Since user profile in real world setting are dynamic, how does PSG-Agent adapt to changes in user emotion or intent across multiple dialogue turns?
3. How does PSG-Agent handle scenarios where user profile information is incomplete or noisy?

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
This paper proposes PSG-Agent, a personality-aware safety guardrail framework for LLM-based agents. The work addresses two major limitations of existing guardrail systems:
(1) the lack of personalization (most systems apply uniform safety rules to all users), and
(2) the focus on single-turn safety, ignoring multi-turn or tool-based risk accumulation.
PSG-Agent introduces multiple interacting modules—including a Profile Miner, Input Guard, Plan Monitor, and Response Guard—to dynamically generate personalized safety criteria and enforce them throughout the agent’s full execution chain. The authors further build a new personalized safety benchmark (2900 samples) where identical queries yield different “Allow/Refuse” labels depending on the user profile. Extensive experiments across various LLM backbones (GPT-4o, Llama-3.3-70B, DeepSeek-V3, etc.) demonstrate that PSG-Agent substantially improves recall and overall safety scores compared to prior guardrail systems such as Llama-Guard 3 and AGrail, while maintaining high helpfulness and interpretability.

### Strengths
* Originality.
Formulates personalized safety for LLM agents—shifting from universal, one-size-fits-all guardrails to user-profile–conditioned criteria.
Integrates a training-free pipeline (Profile Miner → Input Guard → Plan Monitor → Response Guard) to address multi-turn risk accumulation—an underexplored but realistic failure mode.
* Quality.
Presents a reasonably large, carefully constructed benchmark (2900 items) where identical queries receive different labels conditioned on user profiles, plus human evaluation for validation.
Includes ablations that isolate the contribution of each module (e.g., Plan Monitor’s impact on refusal clarity), and cross-backbone tests (e.g., GPT-4o, Llama-3, DeepSeek) demonstrating model-agnostic robustness.
Clear decision taxonomy (Allow / Allow with Guard / Refuse with Alternative / Refuse) and end-to-end enforcement (plan, tools, memory, output).
* Clarity.
Writing is structured and easy to follow, with step-by-step diagrams of the pipeline and crisp definitions (profiles, criteria, metrics).
Examples are concrete and intuitive (e.g., medical/financial scenarios), making the motivation and design choices easy to understand.
* Significance.
Addresses a practical and pressing gap: safety depends on who the user is, not just what is asked.
Provides an evaluation setup that can become a reference for future work on personalized and multi-turn safety.

### Weaknesses
* The evaluation is primarily conducted on the authors’ newly created personalized safety benchmark, but the paper does not test PSG-Agent on existing open-source safety benchmarks such as Agent-SafetyBench or AgentHarm. This limits direct comparability with prior work and makes it difficult to gauge absolute performance improvements in standardized agent scenarios.

* The data construction pipeline relies entirely on GPT-4o for both dataset generation and filtering. This design may introduce model-specific bias: if GPT-4o systematically over- or under-estimates the safety of certain behaviors (e.g., overestimating the safety of investment-related actions), the filtering stage might replicate and reinforce those same biases. A cross-model validation (e.g., with DeepSeek-V3, Claude 3, or Gemini 2.5 as secondary reviewers) would strengthen the dataset’s reliability.

### Questions
1. The personalized safety benchmark seems like a substantial effort (around 2900 examples with partial human validation). It would be great to better understand whether the authors view it as an independent contribution.
2. The Input Guard outputs a risk score between 0–100. Since large language models often make rather discrete decisions (e.g., strong acceptance or refusal), did the authors observe any clustering or threshold effects in these scores? A brief analysis could be helpful for interpreting how this module behaves in practice.
3. It is interesting that vanilla GPT-4o already outperforms prior guardrail systems such as Llama-Guard 3 and AGrail. Could the authors share insights on why this might be?
4. In Section 5.4, the paper mentions that the Input Guard is the most critical component. However, Table 4 shows larger performance drops when removing the Plan Monitor. Could the authors clarify this?

### Soundness
2

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
3

### Summary
The paper introduces PSG-Agent, a training-free, plug-and-play framework for personalized safety guardrails in LLM-based agents. It tackles limitations in existing guardrails by incorporating user-specific profiles to generate tailored safety criteria, and monitors risks dynamically across the agent's pipeline (planning, tools, memory, response). This "personality-aware" approach aims to adapt safety thresholds based on individual traits, health, and context, preventing uniform policies that may over- or under-protect users. A new benchmark with ~2,900 instances across eight scenarios (e.g., finance, healthcare) emphasizes personalization, where decisions flip based on profiles.

### Strengths
- The paper motivates the need for personalized safety.
- PSG-Agent's two-stage design integrates profile mining and multi-point guards without training, formalized as a risk-minimizing decision problem.

### Weaknesses
- While the paper claims personality traits enable tailored safety, examples primarily hinge on health or economic factors (e.g., painkillers for anticoagulants users or impulsive investment in Section 1) rather than pure personality. Ablation shows low extraction accuracy for personality/emotional states, raising doubts on its reliability and necessity.
- The framework treats personality as part of the user profile for criteria generation, but risk functions aggregate it simplistically without analyzing how traits like neuroticism specifically modulate guardrail decisions. This makes the "personality + guardrail" synergy feel additive rather than integral, potentially overcomplicating without proven benefits.
- Benchmark is GPT-4o-generated, potentially biased; human eval is limited. Results favor GPT-4o, with open models underperforming.
- No latency analysis for real-time use; self-benchmark superiority (Abstract) may inflate novelty.

### Questions
Why emphasize "personality-aware" when examples (e.g., Section 1) and profiles (Figure 2) rely more on health/economic attributes, can you ablate personality traits specifically to show their unique contribution?

Given low personality extraction accuracy (61.46%), how robust is PSG-Agent to noisy profiles?

How does personality integrate mechanistically with guardrails beyond aggregation in Equation 2

### Soundness
2

### Presentation
3

### Contribution
3
