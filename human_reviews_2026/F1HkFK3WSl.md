# Magentic Marketplace: An Open-Source Environment for Studying Agentic Markets

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
As LLM agents advance, they are increasingly mediating economic decisions, ranging from product discovery to transactions, on behalf of users.
Such applications promise benefits but also raise many questions about agent accountability and value for users.
Addressing these questions requires understanding how agents behave in realistic market conditions.
However, previous research has largely evaluated agents in constrained settings, such as single-task marketplaces (e.g., negotiation) or structured two-agent interactions.
Real-world markets are fundamentally different: they require agents to handle diverse economic activities and coordinate within large, dynamic ecosystems where multiple agents with opaque behaviors may engage in open-ended dialogues.
To bridge this gap, we investigate two-sided agentic marketplaces where Assistant agents represent consumers and Service agents represent competing businesses.
To study these interactions safely, we develop Magentic Marketplace -- a simulated environment where Assistants and Services can operate. 
This environment enables us to study key market dynamics: the utility agents achieve, behavioral biases, vulnerability to manipulation, and how search mechanisms shape market outcomes.
Our experiments show that frontier models can approach optimal welfare—but only under ideal search conditions. Performance degrades sharply with scale, and all models exhibit severe first-proposal bias, creating 10-30x advantages for response speed over quality.
These findings reveal how behaviors emerge across market conditions, informing the design of fair and efficient agentic marketplaces.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents *Magentic Marketplace*, a simulation framework for studying two-sided markets in which AI agents interact on behalf of consumers and providers. Using this framework, the authors analyze how different market conditions and underlying AI models affect consumer welfare, behavioral biases, and susceptibility to manipulation.

### Strengths
The paper explores a timely and significant topic at the intersection of economic behavior and AI ecosystems, offering practical tools to study both the potential and the limitations of AI-driven economies. It effectively motivates its perspective and provides a well-grounded background for the reader. From an environment design standpoint, I find the paper’s approach particularly valuable: extending beyond the stylized setups commonly found in the literature (such as those focusing solely on negotiation stages) and instead introducing a comprehensive pipeline that integrates multiple types of interactions. Such an approach is essential for understanding how real-world platforms may operate when AI agents act on both sides of the market.

### Weaknesses
1. The paper focuses exclusively on consumer welfare, yet the setting is inherently a two-sided market. I would expect to see a corresponding analysis of the welfare of business providers as well. Considering both sides is essential for capturing the full set of economic trade-offs and for understanding how market design choices affect overall economic efficiency.

2. The paper would benefit from providing more practical insights into how specific design choices shape economic outcomes. For example, are there design parameters that could steer the outcome toward outcomes more favorable to consumers or providers? While the discussion around search mechanisms (lexical vs. perfect search) offers a valuable start, there appear to be many additional degrees of freedom worth exploring, such as how agent communication is structured (natural language vs. formal protocols) or how supply and demand dynamics are modeled. The absence of such analysis makes the current exploration feel somewhat limited in scope.

3. The term “information asymmetry” appears several times but is not clearly defined in context. It remains unclear what constitutes the private information, who possesses it, and how its presence affects the resulting economic outcomes. A more precise definition and explicit modeling of these asymmetries would strengthen the conceptual clarity of the paper.

4. Although the authors themselves note in the ethical statement that fairness is a key concern in the context of two-sided market design, the paper does not provide a concrete definition or formal treatment of fairness. I would expect to see a clear definition of what fairness means in this setting, and at least some preliminary analysis of how the proposed mechanisms perform with respect to this criterion.

**Another minor comment:** Some citations seem to specify the incorrect order of authors, for instance:

Moshe Tennenholtz, Eilam Shapira, Omer Madmon, and Roi Reichart -> Eilam Shapira, Omer Madmon, Roi Reichart, and Moshe Tennenholtz (Can llms replace economic choice prediction labs? the case of language-based persuasion games)

### Questions
I do not have any specific questions, but I am willing to hear your thoughts and comments on my concerns.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Magentic Marketplace, a large-scale simulation platform where LLM agents act as both consumers and producers in a language-mediated market. Agents engage in search, negotiation, and transactions entirely through dialogue, allowing the study of emergent behaviors such as cooperation, manipulation, and market efficiency. The environment aims to serve as a benchmark for testing economic reasoning and coordination in language models.

### Strengths
* Ambitious setup combining natural language, market dynamics, and agent reasoning.

* Models the full market lifecycle (search to dialogue to transaction to evaluation), unlike prior simulations.

* Clear motivation for testing emergent economic and ethical behaviors in LLMs.

### Weaknesses
* The experiments are limited to a single, highly synthetic restaurant domain, which weakens claims of generality.

* Results are mostly descriptive. There is little causal analysis or statistical depth.

* The link between linguistic interaction and market efficiency remains underexplored.

* No clear measure of whether agents reason economically or merely mimic patterns.

### Questions
* How can we distinguish between strategic reasoning and pattern imitation in agent behavior?

* Could the environment generalize beyond one domain, e.g., to services, housing, or labor markets?

* What insights, if any, were gained about language use itself (e.g., persuasion, deception, cooperation)?

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
This paper proposes Magentic Marketplace, a simulated environment for to study LLM agents end-to-end across a two-sided economic market lifecycle, including search, inquiry and potential negotiation, and transactions. Simulated experiments in Magentic Marketplace show that some frontier models improve market welfare outcomes over non-agentic baselines, but performance degrades as business scale increases. Moreover, the experimental results suggest that even the best-performing models remain vulnerable to market manipulation tactics and behavioral biases.

### Strengths
S1. The proposed system is designed for two-sided markets. 

S2. Biases in agent behavior and resistance to manipulation are investigated. 

S3. The scale of simulation is up to 100 consumers and 300 restaurants. 

S4. Multiple LLMs are tested in the experiments.

### Weaknesses
W1. The presentation needs to be improved. First, the simulation design (Sec. 3) involves many high-level concepts, making it hard to understand. Second, the types of agents are confusing. For example, Figure 1 shows customer agents and business agents, while Figure 2 shows an assistant agent and a service agent. Third, based on the description of the proposed environment, it is hard to infer what is going to be evaluated in the experiments, obscuring the objectives of this study. 

W2. The simulation covers only a scenario of Mexican restaurants, despite the claim that the proposed environment supports additional synthetic domains and public/open datasets. 

W3. The paper claims it can answer questions "How do agents behave in response to strategic and competitive market environments, relative to classic economic predictions?" However, I don't find any design in this paper that reflects the dynamics of competition in the market. 

W4. Some experimental settings are not explained (see Q1 and Q2 below).

### Questions
Q1. In Sec. 4, "there exist exactly three businesses with the required food items (at different prices) and exactly two of those businesses with the required amenities". How were these numbers determined? Do they represent the real-world case? 

Q2. The value variable V_i is a fixed value: \alpha times the average price of all desired menu items. The reason for such setting is not explained. 

Q3. YARN is mentioned to accommodate long agentic trajectories. Is long context a concern in this simulation? If so, some observed results may be due to the model's capability of processing long contexts rather than the methods compared.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Magentic Marketplace, a two-sided marketplace where AI agents serve as both consumers and service providers. The authors conduct simulations to evaluate AI agents' performances and behaviors in this environment. Through these simulations, the authors find that LLM agents could be easily manipulated by market manipulation tactics and behavioral biases and agentic solutions work better than non-agentic ones.

### Strengths
1. An important and timely effort in building a simulated environment for agentic marketplace
2. The empirical findings have implications for model/agent builders and users.

### Weaknesses
1. The model selection is a bit confusing. GPT 5 was used in one experiment but not others. Claude series models are not included at all. Adding more models would be helpful. 

2. It would also be nice to see whether the model's capabilities would scale with the parameter sizes. 

3. This paper misses several key references:
https://arxiv.org/abs/2506.00073
https://arxiv.org/pdf/2509.01063

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
