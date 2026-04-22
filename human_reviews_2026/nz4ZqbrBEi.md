# Searching for Privacy Risks in LLM Agents via Simulation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 8, 4

## Abstract
The widespread deployment of LLM-based agents is likely to introduce a critical privacy threat: malicious agents that proactively engage others in multi-turn interactions to extract sensitive information. However, the evolving nature of such dynamic dialogues makes it challenging to anticipate emerging vulnerabilities and design effective defenses. To tackle this problem, we present a search-based framework that alternates between improving attack and defense strategies through the simulation of privacy-critical agent interactions. Specifically, we employ LLMs as optimizers to analyze simulation trajectories and iteratively propose new agent instructions. To explore the strategy space more efficiently, we further utilize parallel search with multiple threads and cross-thread propagation. Through this process, we find that attack strategies escalate from direct requests to sophisticated tactics, such as impersonation and consent forgery, while defenses evolve from simple rule-based constraints to robust identity-verification state machines. The discovered attacks and defenses generalize across diverse scenarios and backbone models, providing useful insights for developing privacy-aware agents.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies privacy risks in multi-agent systems. The task is formulated using three agents (data subject, data sender, data recipient) implemented with ReAct framework. These agents interact through simulated applications (e.g., Gmail, Notion, etc.)
To automatically perform red/blue teaming, the authors propose a search-based optimization frameowrk that iteratively and alternatingly refines attack and defense prompts.
The framework employs parallel search across multiple threads and cross-thread propagation to share the most effective strategies.
Experimental results show the most effective attacks can reach ~80% success rate, while optimized defenses reduce it to under 5%.
The paper further examines the transferability of discovered strategies across different scenarios and backbone models, showing that both attacks and defenses generalize beyond their original optimization contexts.

### Strengths
- The work addresses an emerging and underexplored problem for privacy risks in multi-agent systems, as agents are rapidly advancing and being deployed in real-world applications.
- The proposed approach, featuring parallel search, cross-thread propagation, and alternating search, is a creative and scalable method for optimizting both attack and defense strategies.
- The design of fine-grained metrics (i.e., leak velocity) provides a better accessment of privacy leakage dynamics.
- The paper provides meaningful empirical insights, uncovering effective attack and defense patterns.

### Weaknesses
- limited operational diversity: the scenarios simulated in the paper (e.g., chat applications with search/send/get tools) cover only a narrow range of agent capabilities. Real-world agents often integrate with more diverse tools (e.g., web search, code execution). It's unclear how effective the approach is for more complex scenarios such as a coding agent resolving GitHub issues that can interact with other users and coding environments.
- because of the similar task structures across different scenarios, the transferability results are less convincing; the attacks and defenses may not generalize well to more complex environments.
- although the appendix discusses the tradeoff between privacy awareness and agent helpfulness, the simplified settings make it difficult to assess whether the defenses would preserve utility in real-world contexts.
- the paper lacks the comparison with other attack (e.g., simple LLM optimization, AutoDAN-like genetic search) and defense (e.g., guardrail models, rule-based detection) baselines.

### Questions
- The paper performs extensive experiments for different model settings, how is the consistency across different experiments, would the framework converge to similar attack and defense patterns?
- The evaluation relies on LLMs to detect privacy leaks, would the LLM detectors become vulnerable to adaptive attacks such as using encoding or paraphrasing? And what if the LLM detectors are reliable enough, then why not just employ them as a guardrail to defend against privacy leakage attacks?

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
3

### Summary
Within multi-agent frameworks, agents exchange information that may contain sensitive content, which can be exploited by malicious agents to extract private data through multi-turn interactions. The authors propose a search-based alternating attack–defense framework to model and analyze privacy leakage and corresponding defenses. The attack search automatically explores prompts and dialogue strategies to identify potential leakage channels, while the defense search investigates countermeasures to mitigate identified attacks. Experiments demonstrate that the alternating search framework enables systematic evaluation of privacy risks and the effectiveness of defense strategies in multi-agent settings.

### Strengths
1. The problem setting is novel. The work moves beyond static, hand-designed threat models by formalizing privacy risk in agent-agent interaction as a search problem over strategies, and systematically alternating optimization between adversaries and defenders.
2. Algorithm 1 clearly shows the attack and defense search’s alternating procedure, which is easy to follow and well-documented, improving the reproducibility and implementation transparency.
3. The simulation setup closely mirrors realistic multi-turn agent interactions, which helps in observing emergent privacy risks that may not be present in static or simplified environments.
4. This work innovatively introduces the metric of "leak velocity", which incorporates a sensitivity ranking of data privacy when measuring data leaks, which is a consideration absent in prior research.

### Weaknesses
1. The study relies solely on simulation-based evaluation and does not examine real-world deployment or naturally occurring multi-agent interactions.
2. The defense mechanism is primarily prompt-based, which simplifies implementation but may constrain its generalization to more adaptive or model-level privacy defenses.
3. The paper acknowledges the computational cost of the alternating search but does not elaborate on its stability characteristics.
4. While the alternating search mechanism is well-motivated, it lacks the illustration of the search trajectories during attack–defense interactions.

### Questions
1. Could the authors generalize their analysis to a more realistic agent communication setting or provide some evidence that the leakage patterns they've identified in the work generalize to settings outside the simulation environment.
2. How do the authors envision mitigating the risk that adversaries could appropriate the described search framework for malicious strategy discovery? Are there methods for responsible deployment or access limitation?
3. Please clarify whether the alternating search process empirically converges or exhibits instability over multiple iterations, and how this might affect scalability.
4. Please add a concise case study of the iterative process including both successful and failed searches. This would enhance explainability and transparency.

### Soundness
3

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
Paper studies privacy in LLM agents during interaction, i.e. multi-turn settings. However, in absence of real-world data, authors propose a nice way of simulating different scenarios and iteratively improving both defenses and attacks.

### Strengths
- This appears to be the first paper that does full-scale and end-to-end simulation of scenarios that searches for different vulnerabilities in each scenario. 
- The setting transfers both attacks and defenses to other models and scenarios which further emphasizes importance of simulations
- Novel dataset for synthetic environments.
- Methodology to discover defenses and attacks through iterative and parallel search for attacks and sequential search for defenses.

### Weaknesses
- It is not clear if grounding the norms in PrivacyLens is sufficient. Maybe there are more grounded approaches to identifying norms. It appears that collecting real-world examples might be necessary
- Evolution of attacks and defenses might instead bias exploration of rare attacks and is also limited by the N*M parameters. It would be great to have an explanation why they will cover sufficient amount of scenarios. 
- What are false positive effects of the defenses, especially on long-tail scenarios? It might be that a defense tuned to catch so many attacks begin to also label rare benign use-cases

### Questions
addressing long-tail and grounding norm selection will improve the paper a lot

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates the privacy issues associated with Language Model (LM) agents handling data in sensitive domains. It proposes a simulated configuration to measure how well do the models guard the private data against unauthorised requests. It then proposes an adversarial evolutionary optimization process to search for attack and defense prompts for privacy preservation. The algorithm is shown to improve privacy preservation and reduce leakage across multiple LMs and domains. Cross-model and cross-domain generalization is also demonstrated.

### Strengths
- The problem of privacy-preserving AI agents is highly relevant. 
- The paper is clearly written and the figures are helpful.
- The defense prompts discovered by the algorithm in the paper seem to robustly reduce privacy leakage across models and scenarios.
- The fact that model scaling gives bigger gains to the defender than the attacker is good news for prompting-based defenses: we should expect the systems to become more privacy-preserving in the future.

### Weaknesses
The paper is somewhat overclaiming success of the defenses. In L311-312:
> We use our framework to sequentially discover A1, D1, A2, D2, while we find that it is hard to find an effective A3 that can effectively break D2 anymore.

The last leak rate reported in Fig. 3 is 7.2%, which I would argue is still a very large number for privacy-related issues (although I'm not sure about the intuition behind the leak rate definition, see questions). This indicates generally that the problem of privacy preservation is far from solved, if these numbers are close to SotA. A paper making an incremental improvement in the right direction is still welcome, of course, but the claims have to be more measured. I would replace the ending of the quote above with something like 'it is hard to find an A3 that increases the leakage further." Related to that, from the abstract:
> The discovered attacks and defenses transfer across diverse scenarios and backbone
models, demonstrating strong practical utility for building privacy-aware agents.demonstrating strong practical utility for building privacy-aware agents.

I don't think the reported leakage rates correspond to something that can currently be deployed in practice (imagine a medical system that leaks your private data 7% of the time!), so I would soften the claim.

The evolutionary search that the paper performs for the prompts strongly reminds me of FunSearch [1], I would suggest the authors mention it in related work. For future extensions it might also be worth considering the structure of the FunSearch algorithm (genetic optimization with islands with cross-breeding, sampling based on performance) more explicitly rather than reinventing parts of it.

[1] Romera-Paredes, Bernardino, et al. "Mathematical discoveries from program search with large language models." Nature 625.7995 (2024): 468-475.

Another issue is that the paper reports ablations of the algorithm itself, but does not compare against existing baselines. In L111-117 the paper does mention multiple existing defenses, including prompting-based, which is the focus of this work. Correspondingly, I would expect to see comparisons of this method to the defenses proposed there, if such defenses do exist.

I am on the edge about this paper, and if the authors explain the situation with baselines during the rebuttal I would be open to updating my score.

### Questions
What is the intuition behind the leak rate expression on L159? It looks quite arbitrary to me beyond the fact that it measures the individual facts rather than trajectories. Is LR of 7% too much or too little?

### Soundness
2

### Presentation
3

### Contribution
3
