# Beyond Jailbreaking: Auditing Contextual Privacy in LLM Agents

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 8, 4

## Abstract
LLM agents have begun to appear as personal assistants, customer service bots, and clinical aides. While these applications deliver substantial operational benefits, they also require continuous access to sensitive data, which increases the likelihood of unauthorized disclosures. Moreover, these disclosures go beyond mere explicit disclosure, leaving open avenues for gradual manipulation or sidechannel information leakage. This study proposes an auditing framework for conversational privacy that quantifies an agent's susceptibility to these risks. The proposed Conversational Manipulation for Privacy Leakage (CMPL) framework is designed to stress-test agents that enforce strict privacy directives against an iterative probing strategy. Rather than focusing solely on a single disclosure event or purely explicit leakage, CMPL simulates realistic multi-turn interactions to systematically uncover latent vulnerabilities. Our evaluation on diverse domains, data modalities, and safety configurations demonstrate the auditing framework's ability to reveal privacy risks that are not deterred by existing single-turn defenses, along with an in-depth longitudinal study of the temporal dynamics of leakage, strategies adopted by adaptive adversaries, and the evolution of adversarial beliefs about sensitive targets. In addition to introducing CMPL as a diagnostic tool, the paper delivers (1) an auditing procedure grounded in quantifiable risk metrics and (2) an open benchmark for evaluation of conversational privacy across agent implementations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a framework, Conversational Manipulation for Privacy Leakage (CMPL), to audit contextual privacy risks in LLM agents. It models a multi-turn interaction between an application agent, an adversarial agent, and an auditor that detects both explicit and implicit leakages. Experiments are conducted across multiple scenarios (insurance claims, interview scheduling, etc.), demonstrating that adaptive multi-turn adversaries can induce significant privacy leakages even when single-turn baselines fail. The paper positions CMPL as both an auditing framework and an open benchmark for studying conversational privacy.

### Strengths
1. The paper studies privacy in multi-turn, context-rich conversational settings, which is an important area as LLM agents are increasingly used in personal and enterprise contexts.
2. The experiments are comprehensive across multiple domains, adversarial strategies, and model configurations, and the data artifact may be useful for the community.

### Weaknesses
1. Despite the paper’s comprehensiveness, the core contribution is not sufficiently clear. The metrics for “contextual privacy leakage” (explicit vs. implicit) are heuristic and somewhat circularly defined based on the chosen adversarial model and auditor. For example, there are many metrics in main result figures. While this make the analysis comprehensive, it's unclear how to use them in practice when we need to draw a conclusion on the privacy robustness of a certain model or agent. The proposed “risk quantification” lacks theoretical grounding or interpretability beyond specific instantiations, either.
2. The length and technical density of the paper are not commensurate with its conceptual novelty. Much of the design (multi-turn adversary, auditor-based evaluation, LLM-as-judge) builds on existing frameworks such as red teaming and contextual integrity audits, with relatively limited new methodological insight.
3. The evaluation is heavily conditioned on a specific adversarial agent implementation. It remains unclear whether the observed “zero success rate” or high success rate from this particular adversary meaningfully implies privacy safety or vulnerability in general. The conclusions therefore risk being artifact-dependent.

### Questions
1. Do you think the metrics in the paper can be simplified? For example, can we separate them into main metrics and auxiliary metrics?
2. The study is conducted using 5 specific scenarios as the testbed. Can the proposed CMPL metrics be used as quantitative auditing standards, or are they primarily diagnostic tools within this specific setup?

### Soundness
2

### Presentation
1

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
This paper studies attacks on LLM agents that attempt to steal private information. In this setting the leakage is "contextual" ie an agent needs to decide for each context what can and cannot be shared. The paper introduces an auditor framework to measure and test the leakage with formal setup. Additionally, the paper sets up both long-horizon adversaries and multi-turn scenarios.

### Strengths
- Threat model with multi-turn adversaries is novel and has not been studied before
- This setup requires sophisticated auditor and tracking the leakage across turns. 
- Paper identifies different strategies by the adversaries.
- Formal definitions for leakages

### Weaknesses
- Auditor uses LLM for determining leakage opens up to missing leakage or masked leakage
- A victim LLM might be sharing much more data than Auditor can catch, for example answers like Yes/No might be hard to spot when the question to the victim is obfuscated for the auditor
- Scenarios could be extended to different settings, i.e. trading, appointments, etc
- I would like to have a more substantial discussion on defenses

### Questions
Addressing questions above and adding additional experiments would significantly improve the paper.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents an audit framework for LLM agents to elicit and identify privacy leakage through multi-turn interactions with an adversarial agent. The adversarial agent converses with the target application agents to try to gain unauthorized access to sensitive information via an iterative probing strategy. The evaluation proves this method to achieve substantially higher attack success rate than the static jailbreaking methods. It also performs analysis about the temporal dynamics of leakage and strategies adopted by the adaptive adversaries. From the analysis of the audit agent, it shows the importance of auditing not only explicit leakage and also the implicit leakage that can be inferred through a series of observations.

### Strengths
- This paper makes a substantial contribution by expanding the investigation of privacy threats in LLM agents from a single disclosure event to a dynamic, multi-turn interaction scenario. This captures a more realistic problem setup and also demonstrates a higher attack success rate, showing that the privacy vulnerabilities in these agents are even more severe than what has been revealed in prior work.
- The paper is clearly and effectively written. The evaluation is thorough and methodologically solid. The proposed methods are novel, and the analysis conveys many insights.

### Weaknesses
- Although the audit framework is useful for understanding the capabilities of the attackers, I feel the paper has a limited study and discussion of mitigations against this type of adaptive attack. Specifically, the application agent does not explore alternative designs or varied ways to safeguard information. This raises questions about whether the high attack success rate reflects the current frontier of agent privacy capabilities, or whether it is more an artifact that could be addressed with a "best-effort" application agent. Providing some guidance on possible mitigation directions also feels necessary for a paper that identifies a new type of threat.

### Questions
- Could you explain more about the design considerations of the application agents, and discuss potential mitigation approaches?

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
The papers considers a setup where an LLM gent is interacting with an adversarial user. The goal of the agent is to complete a specific task, and the agent has access to a set of attributes that may not be revealed as per privacy directives. The paper proposes a framework called Conversational Manipulation for Privacy Leakage (CMPL) with adaptive adversary and auditor to detect leakage. Similar to several recent works, the framework utilizes ideas from Contextual Integrity (CI) [Nissenbaum 2004].

### Strengths
* Adversarial attacks against LLM agents to induce unintended disclosure are practically important, and have started to recently gain attention. The paper is timely.
* The paper leverages notions from contextual integrity, which provides an interesting platform for analyzing privacy in conversational settings.

### Weaknesses
* One of my main concerns is that the paper does not put its contributions into the context with respect to recent works on contextual privacy. [Bagdasaryan et al., 2024] is presented in the paper as a jailbreaking attack, whereas [Bagdasaryan et al., 2024] used notions from CI to propose adversarial context hijacking attacks and proposed a defense to prevent unintended leakage by LLM agents. Notions from [Bagdasaryan et al., 2024] such as privacy directives and information profile are used in the paper. Comparison with [Abdelnabi et al., 2025] is also a bit vague. The paper claims to build on the foundation of [Abdelnabi et al., 2025] without providing technical details of similarities and differences. 
* Taking this point further, the paper considers [Bagdasaryan et al., 2024] and [Abdelnabi et al., 2025] as baseline attacks in its experiments. However, these works also present defenses to prevent contextual privacy violations. It is not clear why the paper does not evaluate its adversary against these baseline defenses. 
* The proposed adaptive adversary communicates with the auditor (specifically, when its self-consistency score exceeds a threshold, the adversary appends a hidden note only visible to the auditor). This interaction between adversary and auditor contradicts practical settings -- it is not clear why an adversary would be compliant with an auditor, in fact, an adversary would try to fool an auditor in practice. An efficient auditing framework may even be used as a  potential  guardrail by the agent to detect any leakage and adapt its response(s) based on the detection. It will be important to provide details whether the auditor design is a conceptual framework designed just to measure privacy leakage (and thereby quantify the success of the adversarial user).

### Questions
- Why don't the authors evaluate the performance of their proposed adversaries when the agent uses AirGapAgent [Bagdasaryan et al., 2024] for data minimization at the LLM agent or data firewalls [Abdelnabi et al., 2025]?
- Are the cases in eq (2) mutually exclusive? It seems like explicit and implicit leakage conditions may hold together at the same time. If so, it will be helpful re-write eq (2) so that the cases are mutually exclusive. 
- ASR and PLTC metrics are intuitive. Can the authors provide more intuition behind considering temporal dynamics of leakage? 
- As mentioned in the Weaknesses, it will be important to discuss whether auditor design is primarily conceptual. The goal of an adversarial user in practice would be to *not* get detected by a trustworthy auditor, whereas in the paper, the auditor seems to 'communicate' with the adversary. It will be helpful if the authors can provide more details on this.
- Can the authors compare and contrast the two scenarios used in the experiments with the scenarios used in [Bagdasaryan et al., 2024] and  [Abdelnabi et al., 2025]?

### Soundness
2

### Presentation
3

### Contribution
2
