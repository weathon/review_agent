# From Intents to Actions: Agentic AI in Autonomous Networks

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Telecommunication networks are increasingly expected to operate autonomously while supporting heterogeneous services with diverse and often conflicting intents—that is, performance objectives, constraints, and requirements specific to each service. However, transforming high-level intents—such as ultra-low latency, high throughput, or energy efficiency—into concrete control actions (i.e., low-level actuator commands) remains beyond the capability of existing heuristic approaches. This work introduces an Agentic AI system for intent-driven autonomous networks, structured around three specialized agents. A supervisory interpreter agent, powered by a language model, performs both lexical parsing of intents into executable optimization templates and cognitive refinement based on feedback, constraint feasibility, and evolving network conditions. An optimizer agent converts these templates into tractable optimization problems, analyzes trade-offs, and derives preferences across objectives. Lastly, a preference-driven controller, based on multi-objective reinforcement learning, leverages these preferences to operate near the Pareto frontier of network performance that best satisfies the original intent. Collectively, these agents enable networks to autonomously interpret, reason over, adapt to, and act upon diverse intents and network conditions in a scalable manner.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an Agentic AI system for autonomous networks, aimed at translating high-level network intents (such as low latency, high throughput, and energy efficiency optimization) into concrete control actions. The system consists of three main specialized agents: the Interpreter, the Optimizer, and the Controller. The Interpreter uses a large language model (LLM) to parse network intents and generate optimization templates; the Optimizer converts these templates into constrained optimization problems and adjusts preferences based on network conditions; the Controller utilizes multi-objective reinforcement learning (MORL) to implement adaptive policies that optimize network performance. The system has been validated through simulations in a 5G-compliant network, demonstrating its feasibility and effectiveness in real-world network conditions.

### Strengths
1. The Agentic AI system proposed in the paper is innovative in its application to autonomous network control, especially in transforming high-level intents into low-level control actions through the collaboration of multiple agents. The system is capable of handling complex multi-objective optimization problems and making adaptive decisions in dynamic environments.
2. The modular design of the system (including the Interpreter, Optimizer, and Controller) is well-suited for large-scale communication networks, efficiently addressing multiple objectives related to network performance and service quality. Each agent has a clear responsibility, and they achieve the goals through efficient collaboration mechanisms.
3. The paper validates the effectiveness of the Agentic AI system through high-fidelity 5G network simulations, providing rich experimental data that demonstrates the system's potential in real-world applications, particularly in dynamic and changing wireless environments.

### Weaknesses
1. Although the system is innovative, its complexity may result in high computational overhead, especially when dealing with large-scale network simulations. The interactions between each agent require processing a large amount of real-time data, which could place high demands on computational resources.
2. The paper mentions the current limitations of 4G/5G RAN hardware in terms of computational power, which poses challenges for the application of large language models (LLMs) due to insufficient hardware resources. While the authors propose using a dual-LLM architecture to mitigate this issue, this approach may still face feasibility problems in large-scale deployments, especially in low-latency and low-computation resource environments.
3. Although the paper demonstrates the effectiveness of the system, it lacks an in-depth comparison with existing autonomous network control methods (such as rule-based optimization or traditional reinforcement learning algorithms). Further comparative analysis could better assess the advantages and limitations of Agentic AI in handling complex intents.

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
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an Agentic AI system that maps high‑level network intents to concrete RAN control actions through three cooperating agents: (i) an interpreter that turns natural‑language intents into an Optimization Template Model (OTM); (ii) an optimizer that converts OTMs into constrained optimization over a preference space and adapts preferences online using Bayesian optimization (PAX‑BO); and (iii) a controller that uses multi‑objective RL (MORL) to operate near the Pareto front under changing preferences. The system is explicitly two‑timescale: a slower intent‑management loop (interpreter↔optimizer) and a fast intent‑fulfillment loop (optimizer↔controller) suitable for sub‑millisecond RRM decisions.

### Strengths
1. The triadic agent design—with explicit separation of interpretation, preference planning, and control—fits the realities of RAN timescales.
2. Key technical elements include (a) a dual‑LLM interpreter for schema‑compliant intent parsing and constraint reasoning under tight hardware budgets; (b) PAX‑BO, a preference‑aligned constrained BO routine for steering the controller; and (c) D‑EQL, a distributed envelope Q‑learning algorithm that combines actor–learner decoupling, sharded prioritized replay, distributed exploration across the preference simplex, vector TD targets with envelope updates, and an auxiliary cosine‑stability loss.

### Weaknesses
1. D‑EQL essentially marries EQL with APE‑X/distributed PER and a cosine‑stability term; PAX‑BO employs standard GP‑BO with a trust region. These are well‑engineered combinations, but theoretical or algorithmic novelty appears limited. A stronger case would include ablations showing which D‑EQL elements (preference partitioning, hindsight priority refresh, cosine loss) are necessary for the RAN workload and why.
2. All results are from a single high‑fidelity simulator; there is little coverage of (i) different channel models/mobility patterns, (ii) multi‑cell interference/traffic mixes, or (iii) robustness to non‑stationarity. The workflow is persuasive, but broader stress tests are needed to establish generality.
3. The dual‑LLM interpreter is central, yet there is no quantitative assessment of parsing accuracy, OTM schema validity rates, or guard‑rail efficacy.
4. Reproducibility. Code/OTM schemas and simulator configurations are not (yet) available. Given the complexity of the stack, artifacts would materially increase impact.

### Questions
See my main concerns above.

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
This paper introduces an autonomous framework for next-generation, large-scale, real-time distributed Radio Access Networks (RANs) by **incorporating large language models (LLMs) as a human-machine interface** for translating ambiguous high-level intent into executable optimization templates. A similar ideology has been applied in existing works from the past several years in areas such as autonomous driving, robotics, and many others. From my perspective, the primary contribution of this work is its application of this idea in the specific application of RANs with a particular focus on preference optimization and multi-objective control.

### Strengths
I particularly enjoy the organization of this paper. It comes with
1. a clear organization of motivations, problem settings, related concepts, and experiments; and
2. a detailed supplementary material which itself can be treated as a distinct technical report.

### Weaknesses
From my perspective, this is a typical **application-oriented work** that presents details about incorporating LLMs into a well-established large-scale RANs for human intent translation. In this case, it raises several concerns:
1. **Marginal novelty**. As stated earlier, the concept of using LLMs as a human-machine interface has been broadly explored and well implemented in many other areas, which shadows the novelty of this particular paper.
2. **Experiment Design**. From my understanding, the validations showcased in the experiment of Section 7 are primarily case-specific, meaning that for each specific use case, the authors measure achievements with respect to a specific intent. This raises concerns about whether the cases are carefully selected to sell the performance and whether the system is only responsive to these specific prompts.

### Questions
1. **Motivation**. Since the use of LLMs is bridging the gap between *intent in natural language* and *existing optimization template models (OTMs)*. I am not fully convinced of the necessity of incorporating LLMs in real-life operations, given that we already have these OTMs.
2. **Coverage**. Following the previous question, the system's capability is highly constrained by the diversity and generalizability of available OTMs. In this sense, it is not fully exploiting the common sense learned by the LLMs
3. **Benefits**. I am wondering about the cost efficiency of the new framework, given that we have an LLM in the system, which brings additional computational and memory costs. I would love to see the ratio of performance gain vs. computation/memory increase.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates the use of AI agents for service provisioning in telecom operator networks. Specifically, it examines how high-level intents can be interpreted and executed by an agentic system. While the topic is timely, there is currently no consensus on adopting agentic systems in future telecom networks. Moreover, the paper remains at a high level—developing the core concept and providing only a few experimental examples. The experimental results are not convincing and therefore cannot serve as a proof of concept.

### Strengths
The idea is interesting, especially given the current momentum behind agentic systems.

### Weaknesses
The use of agentic systems—and, in particular, intent-based expressions of service requirements—has been discussed in 3GPP SA2. However, there is no consensus to adopt this approach in 6G systems. Even at this early stage, the indications suggest that intent-based service provisioning is not a requirement for 6G. 

It is therefore necessary to provide stronger justification for why future telecom networks should be agentic and intent-based. What are the pros and cons of such a design? Is flexibility/programmability the only benefit, or can it also improve resource and energy efficiency at the network level? Note that agentic systems are not “free”: they require substantial computation to operate and may significantly increase operators’ energy consumption. What about decision-making latency and any control-loop stability considerations? Please include analysis (and, where possible, measurements) on these aspects.

Besides, the experiments show that the proposed approach works to some extent, but they do not demonstrate strict QoS guarantees. For example, in Figure 3(a), many users experience throughput below 7 Mbps. While the agents are able to detect requirement violations, detection does not imply recovery. Can the system adapt quickly and improve the situation? This is not shown in Figure 3. Please clarify whether detection triggers effective adaptation, quantify recovery times, and show post-mitigation performance.

### Questions
Please check the questions in the Weakness section above.

### Soundness
2

### Presentation
2

### Contribution
2
