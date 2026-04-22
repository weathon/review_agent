# The Sum Leaks More Than Its Parts: Compositional Privacy Risks and Mitigations in Multi-Agent Collaboration

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 8, 2

## Abstract
As large language models (LLMs) become integral to multi-agent systems, new privacy risks emerge that extend beyond memorization, direct inference, or single-turn evaluations. In particular, seemingly innocuous responses, when composed across interactions, can cumulatively enable adversaries to recover sensitive information, a phenomenon we term compositional privacy leakage. 
We present the first systematic study of such compositional privacy leaks and possible mitigation methods in multi-agent LLM systems. First, we develop a framework that models how auxiliary knowledge and agent interactions jointly amplify privacy risks, even when each response is benign in isolation. Next, to mitigate this, we propose and evaluate two defense strategies: (1) Theory-of-Mind defense (ToM), where defender agents infer a questioner's intent by anticipating how their outputs may be exploited by adversaries,
and (2) Collaborative Consensus Defense (CoDef), where responder agents collaborate with peers who vote based on a shared aggregated state to restrict sensitive information spread.
Crucially, we balance our evaluation across compositions that expose sensitive information and compositions that yield benign inferences.
Our experiments quantify how these defense strategies differ in balancing the privacy-utility trade-off. 
We find that while chain-of-thought alone offers limited protection to leakage (39% sensitive blocking rate), our ToM defense substantially improves sensitive query blocking (up to 97%) but can reduce benign task success. CoDef achieves the best balance, yielding the highest Balanced Outcome (79.8%), highlighting the benefit of combining explicit reasoning with defender collaboration. 
Together, our results expose a new class of risks in collaborative LLM deployments and provide actionable insights for designing safeguards against compositional, context-driven privacy leakage.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the concept of compositional privacy leakage in multi-agent LLM systems, where sensitive information emerges through the combination of individually innocuous outputs from multiple agents. The authors develop a systematic evaluation framework modeling adversary-defender interactions as a POMDP, where defenders hold partial, non-sensitive data that can be composed by adversaries to infer private attributes. Two defense mechanisms are proposed: (1) Theory-of-Mind (ToM) Defense, where defenders anticipate adversarial intent by simulating the questioner's knowledge state, and (2) Collaborative Consensus Defense (CoDef), where defenders vote based on shared aggregated state. Experiments across 119 scenarios with multiple LLMs (Qwen3-32B, Gemini-2.5-pro, GPT-5) show that while baseline CoT provides limited protection, ToM substantially improves blocking but reduces benign utility, whereas CoDef achieves better balance.

### Strengths
1. The paper identifies compositional privacy leakage as a distinct threat in multi-agent systems that extends beyond memorization or single-agent risks. This has high practical value.
2. The evaluation setup is well designed. By providing adversaries with optimal plans $P^*$ and measuring plan execution success separately from inference accuracy, the authors isolate whether privacy failures arise from information flow versus reasoning errors. This methodological rigor enables objective comparison across defense mechanisms.
3. The findings clearly demonstrate that simple CoT reasoning is insufficient for defending against compositional attacks, and that collaborative defense mechanisms provide superior privacy-utility trade-offs compared to single-agent ToM reasoning. These are valuable insights on guiding the development of safe multi-agent systems.

### Weaknesses
1. The data construction process is unclear. The paper writes "We construct structured scenarios specifying entities, private data, sensitive targets, and adversary plans", but provides insufficient detail about this critical process. CoT results is bad might because the model does not think these targets are sensitive. Section 3 needs substantial expansion on scenario generation methodology, including examples of how sensitive vs. benign targets are differentiated.
2. The analysis is somewhat superficial. While the paper demonstrates that certain defenses fail, it provides limited insight into why they fail. Is the failure due to lack of privacy awareness about compositional leakage? Is it due to insufficient information for reasoning (especially for non-collaborative methods)? For example, in a related study [1], they found models can identify privacy-sensitive cases but fail to act accordingly in agentic tasks due to complicated objectives. I think similar analysis is needed here for shedding insights on how to improve these models.


[1] PrivacyLens: Evaluating Privacy Norm Awareness of Language Models in Action, Shao et al., NeurIPS 2024

### Questions
1. How are benign and sensitive queries being sourced?
2. How are sensitive fragment combinations computed for "CoT + Sensitive Set"?

### Soundness
2

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
This paper investigates an emerging inference-time privacy issue in multi-agent systems, which is that sharing seemingly innocuous response during agent interactions can allow an adversary to aggregate them and reveal sensitive information. They term this phenomenon "compositional privacy leakage," formally defined it, and proposed two mitigations: 1) ToM defense, where defender agents try to infer the agent's malicious intent on its own; 2) Collaborative Consensus Defense (CoDef), where defender communicates query and answer histories with each other and collectively vote for whether to block a request. They evaluated the baseline CoT methods as well as the proposed defense methods, showing that ToM preserves privacy at the cost of denying benign requests, while CoDef achieves a better balance. They also categorized the reasoning depth into four levels and found a correlation between richer reasoning and stronger compositional privacy protection.

### Strengths
- The paper identifies an important privacy issue of the compositional privacy leakage, similar to the concept of quasi-identifier, in the context of multi-agent systems. It contributes to research on inference-time privacy leakage in multi-agent systems, which is an understudied area.
- The paper provides a formal definition of the compositional privacy leakage problem, and proposes two defense methods. They both more effectively blocked the sensitive requests. The CoDef method substantially improved the blocking success rate while maintaining a similar level of benign success rate to the baselines.

### Weaknesses
- In the threat model and the defense methods, the defender agents and adversary agents are explicitly separated. There seems to be an assumption that the defender agents are trusted by all other defender agents, and another agent is explicitly regarded as an adversary. This does not reflect the risks in a realistic multi-agent setup, where any agent could be compromised. Making this assumption may introduce bias that boost the performance in this task, while lacking applicability in real settings where such identities can't be assumed.
- The current description of compositional privacy leakage seems to emphasize scenarios where sensitive attributes only become visible after multiple structured datasets are joined. For example, linking tables on shared IDs or quasi identifiers, similar to classic reidentification attacks where an anonymized dataset is deanonymized using auxiliary public records. That framing is important but not especially new. It also underplays what is unique about LLMs. LLMs can compose unstructured, semantically rich evidence such as emails, documents, chat logs, and meeting notes, and then infer sensitive facts about specific people without any explicit shared keys. In other words, the leakage is not just a matter of table A being joined with table B. It is that the model can synthesize implicit meaning across heterogeneous text and surface latent information. The latter feels like the real frontier risk, and it is not fully captured by the current definition.

### Questions
- Why are the benign success rates also low in the several CoT baselines (some even lower than the defense methods)?
- Can you clarify what types of compositional privacy leakage are captured in your evaluation?
- Can you address the question regarding the assumption of the roles of defender and adversary, and how it might affect the validity of your results?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Paper defines privacy leakage that accumulates over time through benign outputs and can be inferred by an adversary. The paper uses a game setup and discusses different defenses.

### Strengths
- The attack is clear and works really well, it is a novel result in the area of Contextual Privacy
- The additional formulation is useful to enable a systematic framework 
- Defenses introduce coherent methods to mitigate the attacks
- Evaluation is clear and uses synthetic data.

### Weaknesses
- Synthetic data might not fully represent situations that will occur in real life with the agents.
- The defense mechanisms might still be vulnerable to context hijacking attacks
- I think this can be framed as model-level defenses which is a valuable contribution but yet (as all defenses) has its own limitations, it would be great to have them addressed in the paper
- Also, there is this neighboring domain of LLM censorship [1] that is worth discussing. 
- An adversary could also cause "overthinking" attacks to slow down the system. 
- It might be interesting to see the overhead of preventing the leakage and ways to stop early

[1] LLM Censorship: A Machine Learning Challenge or a Computer Security Problem? [icml'24]

### Questions
Overall, paper is great, addressing weakness brought above would be the most helpful.

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
3

### Summary
This paper addresses compositional privacy leakage in multi-agent systems, where individual outputs from multiple agents can collectively reveal sensitive information. The authors formalize this risk and evaluate two defense strategies — a Theory-of-Mind (ToM) defense, in which agents model an adversary’s intent, and a Collaborative Consensus Defense (CoDef), where agents jointly decide whether to respond to a query.

### Strengths
Compositional privacy risk is an important topic in privacy.

### Weaknesses
Although I understand what the authors are *trying* to convey in this paper, the evaluation setup feels unrealistic and overly artificial. The paper also gives the impression of being hastily organized and written, resulting in missing or unclear details. The assumption of a “black-box adversary with auxiliary knowledge” is underspecified and vaguely defined — does this auxiliary knowledge include personal identifiers? If so, I disagree with the authors’ characterization of such data as public information, as it makes the overall setup overly simplified and less credible.

Moreover, several examples in the paper suggest that the defenders already hold and disclose highly sensitive information that should never be shared with any external entity. For instance, providing an employee ID–to–(employee name, department) mapping from personal records to the attacker would itself constitute a serious privacy violation. Many of the defenders’ data sources are already sensitive by nature, which further undermines the realism of the evaluation.

In addition, the paper lacks a clear explanation of how leakage is identified or measured. The appendix is also difficult to follow, with duplicated sections and disorganized ordering. As a result, I find it challenging to fully understand the results or accept the validity of such an artificial experimental setup.

There is no clear winner for defense. The results for the defense methods are mixed: Self-Voting performs best on Qwen, CoDef achieves the highest performance on Gemini-2.5-pro, and ToM performs best on GPT-5.

### Questions
- Does ToM defenses sacrificing benign success mean the LLM is pessimistic in predicting the other’s intents?
- Is the leakage detection based on string matching or entailment? What does the s* look like?
- How do humans perform on this task?
- Line 144: What is A_j and A_i?
- If the correct plan is already provided, what is the attacker’s role? Do they simply generate natural language queries based on that plan?
- How do you ensure the information held by each defender are not sensitive?

### Soundness
1

### Presentation
1

### Contribution
2
