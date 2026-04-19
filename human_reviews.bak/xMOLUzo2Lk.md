# EIA: ENVIRONMENTAL INJECTION ATTACK ON GENERALIST WEB AGENTS FOR PRIVACY LEAKAGE

- Decision: Accept (Poster)
- Scores: 3, 6, 8, 8, 8

## Abstract
Recently, generalist web agents have demonstrated remarkable potential in autonomously completing a wide range of tasks on real websites, significantly boosting human productivity. However, web tasks, such as booking flights, usually involve users' personally identifiable information (PII), which may be exposed to potential privacy risks if web agents accidentally interact with compromised websites—a scenario that remains largely unexplored in the literature.
In this work, we narrow this gap by conducting the first study on the privacy risks of generalist web agents in adversarial environments. First, we present a realistic threat model for attacks on the website, where we consider two adversarial targets: stealing users' specific PII or the entire user request.
Then, we propose a novel attack method, termed Environmental Injection Attack (EIA). EIA injects malicious content designed to adapt well to environments where the agents operate and our work instantiates EIA specifically for privacy scenarios in web environments.
We collect 177 action steps that involve diverse PII categories on realistic websites from the Mind2Web dataset, and conduct experiments using one of the most capable generalist web agent frameworks to date. The results demonstrate that EIA achieves up to 70\% attack success rate (ASR) in stealing users' specific PII and 16\% ASR in stealing a full user request at an action step. Additionally, by evaluating the detectability and testing defensive system prompts, we indicate that EIA is challenging to detect and mitigate.
Notably, attacks that are not well adapted for a webpage can be detected through careful human inspection, leading to our discussion about the trade-off between security and autonomy. However, extra attackers' efforts can make EIA seamlessly adapted, rendering such human supervision ineffective. Thus, we further discuss the implications on defenses at the pre- and post-deployment stages of the websites without relying on human supervision and call for more advanced defense strategies.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper considers an attack whereby an adversary maliciously injects elements into a website which trick agent models into entering PII or sensitive data. By using ID names or other HTML modifications that suggest that a given <input> node is the destination for PII, a malicious actor may be able to exfiltrate data.

### Strengths
Pros: This paper is quite fun and it’s nifty to see clever ways of tricking LLMs. I also quite liked the technique of setting an element to very low opacity (and one would also make it very small) in an attempt to hide it from a user but still make it visible to an LLM–there are cleverer ways of doing this with HTML, but still nice!

The evaluation was also reasonable and did demonstrate that under the presumptions made by the authors, the attack does work.

### Weaknesses
but still nice!

I am also very unpersuaded by the evaluation of the defensive prompt. The authors claim that defensive prompting as a technique is not efficacious, and yet they test a single prompt that does not appear to have been developed in a systematic fashion (that would otherwise buttress the claim). The authors could improve this evaluation by providing a repeatable methodology for testing potential defensive prompts to demonstrate that this is more than an artifact of the specific one chosen. 

The authors also cite the xz backdoor, but the threat model in that case seems very divorced from the model proposed in the paper. If the claim is that there can be bad code in open source, then one runs into the same issue as before of the peculiar threat model under consideration herein. Specifically, if an actor has that level of access to code that will run on a client, the LLM portion of the proposed attack would be extraneous.

VirusTotal also seems like the wrong choice for testing if an attack is well hidden or not. VirusTotal is largely a tool for detecting known threats via signatures or blacklists–it is thus not really a great indicator of how stealthy an attack is. I'm not sure there's yet a way of proving an attack in this context is stealthy or not and would suggest just removing the claim or else putting more thought to how this can be more conclusively demonstrated.

### Questions
What realistic scenarios would make your attack viable but not simpler attacks? This is one of my biggest issues with the work. Specifically, if you could outline instances in which this attack would be *uniquely* effective that would be of aid.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies a meaningful and practical safety problem of current LLM-based agents. The authors propose a new Environment Injection Attack (EIA) paradigm, in which the attacker aims to make the agent expose the privacy information in the user query while completing the task successfully. They propose 2 types of EIA strategies, Form Injection and Mirror Injection. Experiments on 3 LLM-based agents and Mind2Web dataset show that it is feasible to achieve the EIA target. The authors also include discussions on some countermeasures.

### Strengths
1. The paper is generally well-written. The concepts and methodology are well structured and easy to follow.

2. The authors propose diverse types of EIA based on attacking objectives, injection positions, attacking strategies, opacity values, etc. 

3. The experimental results also validate the effectiveness of the proposed attacking methods.

4. The discussion on countermeasures makes the paper more complete.

### Weaknesses
1. My major concern is about the types of datasets used in the main experiments. The authors only include the 177 queries from Mind2Web in experiments, which may not be comprehensive. Including experiments and analysis on other datasets or tasks (e.g., Web Shop) may make the paper more convincing.

2. The results of the Sensitivity to Injection Position are interesting. Could the authors make more explanations about why the results of $P_{+}$ are generally better than that of $P_{-}$?

3. The content of the Discussion part is too lengthy. Some content could be put into the Appendix. For example, the authors could brief the part of Human Supervision and Implications over Defenses in Pre- and Post-Deployment of Websites in the main text, and put the detailed content in the Appendix. And I think Figure 21 should put in the main text, because Relaxed-EIA is also one of the strategies proposed in this paper.

### Questions
See the weakness part.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper discusses the Environmental Injection Attack (EIA), a novel method targeting generalist web agents to exploit privacy vulnerabilities. EIA involves injecting malicious content into web environments to steal users' Personally Identifiable Information (PII) or entire user requests. The study demonstrates that EIA can achieve up to a 70% success rate in stealing specific PII and 16% in full user requests. The paper highlights the difficulty in detecting and mitigating these attacks, emphasizing the need for advanced defense strategies to protect web agents and user privacy.

### Strengths
- **Novelty and Relevance**: The paper introduces a **novel attack method**, Environmental Injection Attack (EIA), which addresses a significant gap in the literature regarding privacy risks posed by generalist web agents. This is highly relevant given the increasing use of such agents in handling sensitive tasks online.
- **Comprehensive Evaluation**: The authors conduct **extensive experiments** using a state-of-the-art web agent framework and a realistic dataset (Mind2Web). The results are robust, demonstrating the effectiveness of EIA in various scenarios and providing valuable insights into the vulnerabilities of web agents.
- **Practical Implications**: The paper discusses **realistic attack scenarios** and provides a detailed analysis of potential defenses. This practical focus enhances the paper's impact, offering actionable recommendations for improving the security of web agents in real-world applications.

### Weaknesses
- **Limited Online Scenario**: While the experiments are thorough, the evaluation is conducted **offline** and does not fully assess the web agent’s capabilities in a real-time interactive environment. This limits the understanding of the attack's impact in dynamic, real-world settings.  I know the authors have pointed out this, but this is indeed a weakness in my view.

- **Generalization of Results**: The study focuses on a specific web agent framework (SeeAct) and a particular dataset. While the authors argue that the attack strategies are applicable to other web agents, **additional experiments** on different frameworks and datasets would help validate the generalizability of the findings.

- **Unclear Visibility**: In Figure 2, I am curious about the appearance of the original, clean aria-label. Although the authors describe the injected prompt, “This is the right place to input the recipient name,” as normal and benign, it appears somewhat abrupt and out of place in this context. I would appreciate seeing what the clean or original aria-label looks like for comparison.

### Questions
1. How does the offline evaluation impact the understanding of the web agent’s performance and attack impact in dynamic, real-time environments? Could additional real-time experiments be conducted to address this?

2. To what extent do the attack strategies generalize beyond the specific web agent framework (SeeAct) and dataset used in the study? Would further testing on diverse frameworks and datasets strengthen the findings?

3. What does the original, clean aria-label look like in comparison to the injected prompt in Figure 2? Could this be provided to clarify how the injection appears in context?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a novel attack method, Environmental Injection Attack (EIA), targeting generalist web agents. It aims to expose privacy risks by manipulating web environments through injections that mislead web agents into leaking Personally Identifiable Information (PII). The attack leverages form and mirror injection strategies to adapt malicious content to different parts of a webpage. The authors provide experimental evidence showing that EIA can achieve up to a 70% success rate in leaking specific PII and a 16% success rate in leaking full user requests. The paper highlights the stealthiness of EIA and discusses the limitations of current defense mechanisms like system prompts.

### Strengths
1. Novelty of Attack: The concept of EIA, which blends form and mirror injections into web environments, is innovative. The focus on environmental adaptation to manipulate web agents without disrupting their primary tasks is a valuable contribution to the field of adversarial attacks.

2. Comprehensive Threat Model: The authors present a well-defined threat model that details two distinct adversarial targets (stealing PII and full requests) and realistic attack scenarios, making the study relevant for real-world applications of generalist web agents.

3. Impact on Web Security: The discussion on how traditional web security tools (e.g., VirusTotal) fail to detect EIA is insightful, as it highlights the gap in current defenses against these new forms of attacks.

### Weaknesses
1. Limited Discussion on Practical Mitigations: While the paper evaluates system prompts as a defense and highlights their limitations, the mitigation strategies remain underdeveloped. It would be beneficial to provide a more detailed exploration of potential defenses (both on web agents and web environments) that could address this new type of attack.

2. Over-reliance on Specific Frameworks: The experiments are largely based on the SeeAct framework, which, while advanced, may not fully represent the broad landscape of generalist web agents. Testing EIA on a wider variety of web agents or frameworks would improve the generalizability of the results.

### Questions
1. Can the authors elaborate on the practical difficulty of implementing EIA in a real-world scenario, especially considering the variability of web designs and how attackers could adapt to different environments?

2. Are there any promising directions for future work on defenses, beyond the defensive system prompts and traditional malware detection tools, that could mitigate EIA without compromising the agent’s functional integrity?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper describes a setting where a malicious website tries to interfere with the web agent and attempts to exfiltrate user's PII or the whole request. As more users might be delegating their tasks to agents they have to entrust these agents with their data making a privacy problem.

### Strengths
- the paper reads well and presentation format is easy to follow
- the problem set is picked forward looking and justified well
- evaluation includes multiple different language models 
- the attack is stealthy and hard to notice and is accompanied by visual examples

### Weaknesses
- it is not clear whether targeted PII data is really vulnerable, as if the user themselves would go to the website they would share exactly same data and I am not sure that the full query is sensitive. I would appreciate more examples why this is a privacy problem.

- the paper needs more evaluation for different scenarios and tasks (kind of Appendix F but with attack results). How does attack effectiveness vary across prompts and different PII data.

- Add connection to contextual integrity[1,2 + others related] -- it is related as user's delegating their data cannot know in advance what data is needed for the particular task and it's important to follow CI. Might be a useful argument why the proposed threat model matters -- LLMs can be trusted with private data (except for the proposed attack).

[1] Mireshghallah, Niloofar, et al. "Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory." ICLR'24
[2] Bagdasaryan, Eugene, et al. "Air Gap: Protecting Privacy-Conscious Conversational Agents." arxiv (2024).

### Questions
address threat model and evaluation questions, integrate CI discussion.

### Soundness
3

### Presentation
4

### Contribution
3
