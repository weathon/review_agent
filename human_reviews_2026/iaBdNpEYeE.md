# Aegis: Towards Governance, Integrity, and Security of AI Voice Agents

- Avg Score: 3.67
- Decision: Reject
- Scores: 4, 2, 4, 6, 2, 4

## Abstract
With the rapid advancement and adoption of Audio Large Language Models (ALLMs), voice agents are now being deployed in high-stakes domains such as banking, customer service, and IT support. However, their vulnerabilities to adversarial misuse still remain unexplored. While prior work has examined aspects of trustworthiness in ALLMs, such as harmful content generation and hallucination, systematic security evaluations of voice agents are still lacking. To address this gap, we propose Aegis, a red-teaming framework for the governance, integrity, and security of voice agents. Aegis models the realistic deployment pipeline of voice agents and designs structured adversarial scenarios of critical risks, including privacy leakage, privilege escalation, resource abuse, etc. We evaluate the framework through case studies in banking call centers, IT Support, and logistics. Our evaluation reveals several important findings. First, restricting agents to query-based database access eliminates authentication bypass and privacy leakage attacks. However, behavioral threats such as privilege escalation, instruction poisoning, and resource abuse persist even under stricter access controls, indicating that compliance-driven vulnerabilities cannot be mitigated by data access policies alone. Moreover, open-weight models show consistently higher susceptibility to adversarial manipulation compared to closed-source ones. In addition, we also found that attacker personas and gender cues can influence outcomes but are not dominant factors when strong operational policies are enforced. These insights underscore the necessity of layered defense strategies-combining access control, policy enforcement, and behavioral monitoring- to secure next-generation ALLM-powered voice agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This presents a comprehensive red-teaming framework for evaluating the security, integrity, and governance of Audio Large Language Model (ALLM)-based voice agents. It systematically tests these systems across adversarial scenarios—such as authentication bypass, privilege escalation, privacy leakage, and resource abuse—in critical domains like banking, IT support, and logistics. The study finds that while restricting agents to database query access reduces data leakage, it fails to prevent behavioral vulnerabilities, highlighting the need for stronger operational policies. The work’s key contribution is introducing Aegis, a practical framework that unites technical and governance perspectives to guide policy-driven defenses and improve the safe deployment of ALLM-powered voice systems.

### Strengths
The paper is original in introducing the first red-teaming and governance framework for Audio LLMs, addressing an important gap in AI safety. Its methodology is solid, testing realistic adversarial scenarios across domains and revealing both technical and policy weaknesses. The writing is clear and well supported by visuals, making complex findings accessible. Overall, it is a significant contribution that bridges AI security and governance, offering practical guidance for deploying safer voice-based AI systems.

### Weaknesses
The paper’s evaluation is limited to controlled, synthetic scenarios, which may not fully capture the complexity of real-world user interactions or adversarial conditions faced by deployed voice agents. While the red-teaming framework is well designed, it would benefit from broader empirical validation involving live or human-in-the-loop settings to assess robustness under natural speech variability and spontaneous misuse. Some aspects of the governance discussion remain high-level, lacking detailed guidance on policy integration or compliance alignment (e.g., regulatory audit mechanisms). Additionally, the framework’s scalability across different ALLM architectures and deployment platforms is not thoroughly examined, leaving questions about generalization to diverse voice systems.

### Questions
How does Aegis perform in real-world or human-in-the-loop environments, where voice inputs are more variable and less predictable than synthetic test cases?

Can the authors elaborate on the scalability and generalization of Aegis across different ALLM architectures or commercial deployment frameworks?

The governance component is promising—could the authors clarify how Aegis’s results could inform or integrate with regulatory compliance processes (e.g., audit trails, model certification)?

How does Aegis handle dynamic adversarial adaptation, where attackers adjust their strategies based on prior system responses?

Could the authors discuss potential extensions or automation of the red-teaming process to make it more practical for continuous monitoring in production systems?

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
This paper presents Aegis, a red-teaming framework designed to evaluate and enhance the security of Audio Large Language Model (ALLM)-powered voice agents in high-stakes domains such as banking and IT support. Addressing the lack of systematic security evaluations, Aegis simulates realistic deployment pipelines and structured adversarial scenarios, targeting risks like privacy leakage, privilege escalation, and resource abuse. Case studies reveal that while restricting database access can prevent certain attacks, behavioral threats persist even under strict controls. The study also finds open-weight models more vulnerable than closed-source ones and highlights the limited impact of attacker personas and gender cues when strong policies are in place. These findings advocate for multi-layered defense strategies combining access control, policy enforcement, and behavioral monitoring to secure future ALLM applications.

### Strengths
1. The benchmark design, in terms of scenarios and threat models, is practical.
2. The finding that changes in the data access interface significantly impact attack success rates is interesting.

### Weaknesses
1. The benchmark and its findings lack sufficient novelty. Although the authors claim to address two gaps—multi-turn attacks and targeting agents instead of standalone models—both aspects have been explored in prior work (yet might not be done in ALLMs), reducing the originality of the contribution.
2. Given the claimed advantage of better alignment with real-world scenarios, it is important to test the proposed attacks in actual applications. The current simulation environment appears overly simplified, limiting generalizability. For example, real-world systems do not typically grant authentication solely based on an agent's textual output like “you have been verified” (as shown in Figure 2).
3. The reported zero success rates in the query-based setup require further analysis. It is unclear whether these are due to improved security or the system's inability to execute even benign instructions via the query interface. Adding results on non-harmful tasks would clarify whether the interface hinders normal functionality.
4. The conclusion that “open-sourced backbones are more vulnerable” seems superficial. A more likely explanation is that Qwen models are less rigorously aligned for safety. To validate this, safety evaluations of the base models on standard textual safety benchmarks should be included.

### Questions
1. what are the defenses deployed?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces LLaVA-Interactive, a framework that enhances vision-language models (VLMs) through multi-turn, multimodal self-interaction during fine-tuning. By simulating rich conversational behaviors—asking clarifying questions, reasoning visually, and providing step-by-step responses—it significantly improves model performance on vision-language instruction-following tasks without relying on human annotations.

### Strengths
Strengths:

Originality: Proposes a novel self-interaction fine-tuning method that leverages model-generated dialogs across image inputs, reducing reliance on costly human supervision.

Empirical Quality: Demonstrates state-of-the-art results on benchmarks like MME, SEED-Bench, and LLaVA-Bench with strong qualitative improvements.

Clarity and Scope: Clear description of the pipeline (Figure 2) and strong motivation for multi-turn behavior in visual instruction contexts.

### Weaknesses
Weaknesses:

Generalization Risk: The method is evaluated mainly on LLaVA-1.5 with specific backbone settings; it’s unclear how well the approach transfers to other VLMs or unseen visual domains.

Ablation Limitations: While some ablations are reported (e.g., number of rounds), deeper analyses of failure cases or negative impacts of self-generated noise are limited.

### Questions
Have you evaluated how well this self-interaction approach generalizes to other vision-language models beyond the LLaVA architecture?

### Soundness
3

### Presentation
2

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
The paper presents a red-teaming evaluation of LLMs in the context of customer support in banking, IT, and logistics. 

The novel contributions are:
1. A taxonomy of adversarial scenarios that is used to generate adversarial attacks for evaluation.
2. An evaluation of GPT, Gemini, and Qwen models on their robustness to adversarial attacks.

### Strengths
1. The paper focuses on a highly important application: adversarial attacks of voice agents. In particular, they focus on 3 customer service applications (banking, IT, and logistics) where LLMs are already deployed. 

2. The evaluation framework is rigorous and reproducible. The 5 attack objectives x 5 attacker personas are relevant for many domains, as the others show in their case studies. 

3. The evaluation results are novel and interesting, showing that some attack vectors remain challenging (resource abuse) and offering a practical way to reduce vulnerabilities (limit to query-based access).

### Weaknesses
1. Some details are confusing in the evaluation setup. In section 3.3 and Figure 2, the language around "attacker", "attack agent", "agent", and "evaluator" could be made clear. It appears these are all LLMs, the "attack agent" is always GPT-4o, the "backbone agent" is one of 7 models, and the "evaluator" is always GPT-4o? Using consistent language like "attack agent" and "backbone agent" might be helpful.  

2. While the paper conducts a thorough evaluation of 7 "backbone agents", I would have liked to see a variety of models used as the "attack agent". While the evaluation using GPT-4o is an important first step, are other models better/worse at generating adversarial attacks? 

3. The results around persona choice and gender are really interesting, and it would be interesting to expand the evaluation in this direction. In particular, how might dialects, choice of language, and tone influence the attack success? For example, would attacks conducted in low-resource languages be more or less likely to succeed? How about attacks with certain dialects or accents? While this may be outside the scope of this work, I would like to see more discussion about this.

4. The paper is titled towards "governance, integrity, and security", yet the overwhelming focus of the evaluation seems to be about security (e.g. whether the agent has security vulnerabilities). In particular, the paper has very little to do about governance (other than highlighting the importance of evaluation for voice agents). I'm also not sure what the benefit of titling the evaluation framework as "Aegis" is, other than giving it a fancy name. I would strongly suggest the authors consider a title that better reflects the main contributions of the work.

### Questions
In a rebuttal, it would be helpful if the authors addressed the weaknesses above. Some additional minor questions:

1. Just to confirm, the attacks are conducted entirely in the audio modality, and then the analysis is done in the text-domain? How is the audio transcribed to text? 

2. Are the 5 adversarial scenarios taken directly from the MITRE ATTACK framework?

3. Are the backend databases also synthetically generated using GPT-4o? What is the database size? In practice, agents may have access to multiple databases, which may also affect attack success?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the security of voice agents and proposes a red-teaming framework called Aegis, designed for assessing the governance, integrity, and security of AI voice agents. To understand the vulnerabilities of voice agents, the authors employ Aegis to evaluate their backbone models across five adversarial scenarios: authentication bypass, resource abuse, privilege escalation, data poisoning, and privacy leakage. To simulate realistic usage scenarios, the experiments are conducted within contexts such as bank call centers, IT support desks, and logistics dispatch services. The findings reveal the vulnerabilities in both closed-source and open-weight backbone models powering AI voice agents.

### Strengths
(1) The paper is clearly structured and well-organized.

(2) The manuscript is free of grammatical and typographical errors.

### Weaknesses
(1) **Limited evaluation of practical voice agents**

Although the paper evaluates the governance, integrity, and security of AI voice agents, it primarily focuses on backbone models rather than complete, deployed agent systems. Real-world AI voice agents typically include multiple components, such as data processing, safeguard, and storage modules, in addition to the backbone model. Therefore, restricting the evaluation to backbone models does not provide a comprehensive understanding of the security, governance, and integrity of a full voice agent system. As noted in Section 2, the authors themselves stated that "there remains a lack of systematic evaluation frameworks that capture the full range of adversarial risks facing ALLMs when integrated as conversational agents in real-world applications," suggesting that the study’s scope is more centered on ALLMs than on end-to-end agents.

(2) **Unclear relationship between evaluated ALLMs and practical voice agents**

The paper evaluates seven ALLMs but does not clarify which real-world voice agents actually use these models. To strengthen the connection to practical relevance, the authors should specify which commercial or open-source agents adopt these ALLMs, including details such as agent names, associated backbone models, URLs, and deployment contexts, especially since the study claims to focus on voice agents.


(3) **Insufficient justification of adversarial scenario taxonomy**

The authors propose a taxonomy of adversarial scenarios for AI voice agents (Section 3.2), inspired by the MITRE ATT&CK framework. However, given the extensive range of adversarial tactics in MITRE ATT&CK, the idea and rule for selecting only the five scenarios included in this study is unclear. A more detailed justification is needed to explain this selection. Furthermore, to achieve a more comprehensive evaluation, the taxonomy and experiments should encompass all relevant adversarial scenarios applicable to AI voice agents.

### Questions
(1) Which real-world AI voice agents employ the seven ALLMs assessed in this paper?

(2) Do agents' components other than the backbone ALLMs (e.g., data processing, safeguard modules, data storage, system prompts) exhibit vulnerabilities under the adversarial scenarios defined by the authors?

(3) Are there additional adversarial scenarios from MITRE ATT&CK that could be relevant to AI voice agents but were not included in this study?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Aegis, a red-teaming framework designed to assess the security, privacy, and governance of voice agents powered by Audio Large Language Models (ALLMs) deployed in high-stakes environments, such as banking, IT support, and logistics. The authors conduct an evaluation of voice agents under five adversarial scenarios: authentication bypass, privacy leakage, resource abuse, privilege escalation, and data poisoning. The paper highlights that while restricting agents to query-based database access mitigates some security vulnerabilities, behavioral threats such as privilege escalation and resource abuse persist, revealing the complexity of safeguarding voice agents.

### Strengths
- The Aegis framework goes beyond traditional model-level robustness evaluations and offers a realistic assessment of deployed systems in diverse, high-risk domains.


- By considering a broad set of adversarial scenarios, the framework offers valuable insights into various vulnerabilities and highlights real-world risks that existing models fail to address.

### Weaknesses
- Red-teaming framework of ALLM has been studied before, although the authors claim this work focuses on more realistic assessment. Therefore, the contribution of this paper seems unclear.


- The framework heavily relies on certain attack scenarios, such as authentication bypass and resource abuse, but the paper could benefit from exploring additional advanced adversarial tactics. For instance, attacks exploiting AI’s cognitive biases in interpreting complex dialogues could be a future avenue for research.


- While restricting agents' access to query-based systems is a positive step, the persistence of behavioral vulnerabilities under stricter policies might suggest that the focus on data access limitations is insufficient. The paper doesn’t delve deeply into other critical aspects such as psychological manipulation of voice agents.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
