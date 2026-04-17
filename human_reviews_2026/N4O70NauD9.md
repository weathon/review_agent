# This Is Your Doge, If It Please You: Exploring Deception and Robustness in Mixture of LLMs

- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
Multi-agent systems of large language models (LLMs) operate with the assumption that all the agents in the system are trustworthy.
In this paper, we investigate the robustness of multi-agent LLM systems against intrusions by malicious agents, using the Mixture of Agents (MoA; Wang et al., 2024) as a representative multi-agent architecture.
We evaluate its robustness by red-teaming it with carefully crafted instructions designed to deceive the other agents. When tested on standard benchmarks, including AlpacaEval, our investigation reveals that the performance of MoA can be severely compromised by the presence of even a single malicious agent, which can nullify the benefits of having multiple agents. 
The performance degradation becomes more severe as the capability of the malicious agent increases. On the other hand, naive measures, such as increasing the number of agents or replacing faithful agents with stronger models, are insufficient to defend against such intrusions. As a preliminary step toward addressing this risk, we explore a range of unsupervised defense mechanisms that recover most of the lost performance with affordable computational overhead. Our work highlights the security risks associated with multi-agent LLM systems and underscores the need for robust and efficient defense mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper explores the robustness of multi-agent LLM systems against intrusions by malicious agents and finds that performance degradation becomes more severe as the capability of the malicious agent increases. It then investigates a range of unsupervised defense mechanisms that can recover most of the lost performance with minimal computational overhead.

### Strengths
The topic is interesting and the writing is easy to follow.

### Weaknesses
My key concern lies in the definition and realism of the threat model. The paper assumes the presence of malicious agents within a multi-agent LLM system, but it is unclear how such intrusions occur in realistic deployment scenarios or what practical motivations these agents have. Clarifying whether the attacks are external injections, compromised components, or emergent misbehaviors would make the work more convincing. Otherwise, the defense would be simply to remove such agents.

Additionally, the intended application scenario of the Mixture of Agents (MoA) framework is not clearly articulated. This paper only conducts experiments on easier datasets. Without a concrete context—such as collaborative reasoning, planning, or autonomous decision-making—it is difficult to evaluate the practical relevance and impact of the findings.

While the experimental setup and results are clearly presented, the conclusions themselves are somewhat expected: it is intuitive that stronger malicious agents cause more severe degradation, and that simple scaling (e.g., increasing the number of agents) cannot fully mitigate the issue. 

Overall, the paper raises an important problem—security and robustness in multi-agent LLM systems—but it would benefit from a clearer threat model, a more realistic application setting, and a deeper analysis of why the proposed defenses are effective and generalizable.

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies robustness when one or more agents are malicious/deceptive in a multi-agent system (specifically Mixture of Agents, MoA). The paper evaluates a  3-3-1 MoA on two tasks: (1) QuALITY multiple-choice long-context comprehension with distributed evidence across agents; (2) AlpacaEval 2.0 instruction following. Experiments show that a single deceptive agent can nullify MoA gains and sometimes drop performance below single-model baselines. The paper also proposes unsupervised defenses inspired by the Venetian Doge election: Cluster & Filter (embedding-based clustering of references), Dropout & Cluster (random subsetting then clustering), and LLM-as-a-judge. Experiments show that Clustering defenses recover most performance with low overhead, while Cluster & Filter is both effective and cheap.

### Strengths
1.  The threat model is clear and important. It demonstrates a concrete and underexplored failure mode of multi-agent LLM systems with compelling empirical evidence.

2. The experiment is comprehensive. In addition to demonstrating the risk, the paper also proposes practical and unsupervised defenses with favorable cost-performance.

3. The presentation of the paper is good. It is clearly written and easy to follow.

### Weaknesses
1. Cluster & Filter uses k=2 clustering in embedding space and assumes deceptive outputs cluster apart from truthful ones. This may break when (i) deceptive agents imitate truthful style closely, (ii) multiple deceptive subgroups exist, or (iii) truthful agents disagree (e.g., due to ambiguity). 

2. The paper does not report performance with >1 deceptive agent in the defense experiments. The defense results focus on a single deceptive agent.

3. Deception is injected by explicit prompt instructions (opposer/promoter), which is a strong, overt adversary; how well does this capture more realistic/inadvertent failures (e.g., subtle inconsistency, partial hallucination, or distributional shift)? 

4. Tasks are two popular benchmarks (QuALITY, AlpacaEval). Additional domains, such as tool-use agents and code generation, would strengthen external validity.

### Questions
1. How sensitive are results to the embedding model, k, the filtering threshold, and the number of dropout subsets?

2. In the QuALITY setting, what happens if the aggregator also sees the full passage? Does that blunt deception?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work conduct a comprehensive study on the robustness of MoA architectures against deceptive agents. The authors first evaluate MoA's robustness by red-teaming it with crafted instructions designed to deceive the other agents. And they find that performance degradation will be observed on two mainstream benchmarks. Then, inspired from Venice's legacy, the authors explore a range of unsupervised defense methods to recover the lost performance in the compromised MoA.

### Strengths
### **Strengths**

1. In this paper, there are sufficient latest LLMs in the experimental part to investigate the phenomenon of performance degradation of MoA, including llama, qwen, gpt-oss, and mixtral.

2. The proposed method is effective. According to table 5 and 6 in the manuscript, the proposed defense method can obviously promote the robustness of MoA architecture.

### Weaknesses
### **Weaknesses**

1. The evaluation scenarios are extremely limited. There are only two benchmarks to investigate the robustness of MoA architecture, which is hard to prove the conclusion of this paper.

2. The authors only investigate the phenomenon of the MoA architecture, it is uncertain that whether the experience can transfer to the other multi-agent architectures.

3. This paper is not the first work to identify this phenomenon, and the proposed defense strategy appears to have little connection with Venice Legacy.

### Questions
### **Questions**

1. Could the authors provide additional experiments on more diverse benchmarks or multi-agent architectures?

2. Could the authors further explain the motivation of the defense method?

### Soundness
2

### Presentation
3

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
This paper investigates the robustness of multi-agent LLM systems, focusing on the Mixture of Agents (MoA) architecture under deceptive agent intrusions. It analyzes key vulnerability factors, such as deceptive agent capability and aggregator model scale. Inspired by the Venetian Doge election process, the work presents several practical and efficient defense strategies which address critical security risks in multi-agent LLM deployments.

### Strengths
1.	The paper tackles a timely and critical problem concerning the security of multi-agent systems, an area of growing importance as such systems are increasingly deployed in high-stakes and safety-critical domains. 

2.	The experimental evaluation is comprehensive and well-structured. The authors systematically explore the impact of various factors across two distinct tasks, making the findings robust and generalizable.

3.	The proposed defense mechanisms, whith is inspired from the Venetian election process, are creative, well-motivated and empirically demonstrated to be highly effective.

### Weaknesses
1.	The study only concerns isolated and explicitly prompted malicious behaviors, without considering cooperative or adaptive attacks, which limits its generalizability. It is suggested to include multi-agent and dynamic attack scenarios and show robustness curves under different attack intensities or adversary ratios.

2.	Experiments are limited to reading comprehension and open QA, lacking evaluation in high-risk domains, like healthcare or finance. It is suggested to extend experiments to such domains and include risk-sensitive metrics to assess real-world robustness.

3.	The defense assumes malicious agents are a minority (Sec. 6.1). When attackers are the majority or mimic honest behavior, clustering and filtering may fail. Robustness under different attacker ratios and mimicry-based attacks should be tested.

4.	Comparisons rely on a single baseline, omitting other multi-agent defenses and anomaly detection techniques. Broader baseline comparisons are supposed to validate the proposed method’s advantage.

### Questions
Refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
