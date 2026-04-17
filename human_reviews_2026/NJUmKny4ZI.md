# It's a TRAP! Task-Redirecting Agent Persuasion Benchmark for Web Agents

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Web-based agents powered by large language models are increasingly used for tasks such as email management or professional networking. Their reliance on dynamic web content, however, makes them vulnerable to prompt injection attacks: adversarial instructions hidden in interface elements that persuade the agent to divert from its original task. We introduce the Task-Redirecting Agent Persuasion Benchmark (TRAP), a reproducible evaluation suite for studying how persuasion techniques misguide autonomous web agents on realistic tasks. Across six frontier models, agents are susceptible to prompt injection in 25\% of tasks on average (13\% for GPT-5 to 43\% for DeepSeek-R1), with small interface or contextual changes often doubling success rates and revealing systemic, psychologically driven vulnerabilities in web-based agents. We also provide a modular social-engineering injection framework with controlled experiments on high-fidelity website clones, allowing for further benchmark expansion.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work, the authors propose TRAP, a benchmark designed to evaluate vulnerabilities arising from Adversarial Injections. The benchmark defines five modular dimensions and includes 630 agent susceptibility tests using realistic website clones. The authors report the average hijacking success rate across six evaluated models and the transferability rate between model pairs, demonstrating the presence of systemic security vulnerabilities.

### Strengths
- Addresses a more diverse set of vulnerability cases compared to prior work.

- Provides the interesting observation that a vulnerability detected in one model can transfer to other models.

- The LLM-manipulation category based on Cialdini’s principles is an interesting approach, rather than framing the problem purely as jailbreaking.

- Reports transfer rates between models through an extensive evaluation.

### Weaknesses
- Considers a hijack successful the moment a click occurs, but in the real world hijacking may unfold over multiple turns.

- Although the set of vulnerability cases is large, it is still limited to elements like buttons, links, and user-editable areas.

### Questions
- Is there a way to infer or estimate the risk/severity level for each vulnerability? (how large cost each vunlerability takes?)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Disclosure: Claude is used to refine this review.

This paper introduces TRAP, a benchmark for evaluating hijacking vulnerabilities in LLM-based web agents. The benchmark constructs 630 task suites on 6 cloned websites by combining 5 modular components: injection interface, human persuasion principles, LLM manipulation methods, injection location, and tailoring. Using a one-click success criterion, the authors evaluate 6 models and find an average 25% hijack success rate, ranging from 13% (GPT-5) to 43% (DeepSeek-R1). Results show buttons are 3× more effective than hyperlinks, attacks transfer across models, and light tailoring substantially increases success.

### Strengths
- Web agent hijacking is a real and growing threat. The focus on systematic evaluation is timely and valuable.
- The 5-component framework (interface, persuasion, manipulation, location, tailoring) enables systematic ablation studies and is extensible to new attack types.
- The paper systematically examines transferability, component effectiveness, interface types, location, and tailoring, which provides actionable insights.

### Weaknesses
- Only buttons and hyperlinks are tested; no images, pop-ups, audio, forms, or other realistic attack vectors. Also, one-click criterion is too simplistic, as in practice agent scaffolding can detect recover from errors (e.g., [1] discussed this in detail). Further discussion and justification of the one-click criterion are needed.
- No defenses/controls are evaluated. This limits practical applicability. It would be interesting to see if input filters / output monitors can mitigate the problem.

[1] Wu et al. Dissecting Adversarial Robustness of Multimodal LM Agents. ICLR 2025. https://arxiv.org/abs/2406.12814

### Questions
N/A

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
4

### Summary
The paper introduces TRAP, a modular benchmark for task-redirecting hijacks of web agents. TRAP composes injections from five dimensions (interface, persuasion principle, LLM manipulation method, location, tailoring), yielding 630 task–injection suites across six cloned websites built on REAL. Success is defined objectively as the agent clicking an injected hyperlink or button, avoiding LLM judges and skills-gap ambiguity.

### Strengths
1. Frames persuasion-driven hijacks as modular components that can be recombined and extended.

2. Clear threat setup; fixed observation modality (AXTree) to control confounders; broad, factorized analysis across persuasion and manipulation methods with transferability measurements.

3. The five-component decomposition, location diagrams, and controlled comparisons (button vs hyperlink; targeted vs non-targeted prompts; tailored vs non-tailored) make the overall story legible.

### Weaknesses
1. TRAP’s modularity is compelling, but the paper does not empirically contrast against prior agent-security benchmarks (e.g., AgentDojo, AgentHarm, InjecAgent, etc) on overlapping attack types to show what conclusions change when using the one-click criterion.

2. Only text injections via buttons and hyperlinks are used in the core dataset; pop-ups, banners, multimedia, and richer UI elements are out of scope.

3. Using only accessibility trees improves control, but many deployed agents rely on screenshots/DOM blends. The gap between AXTree-only susceptibility and multi-modal observation remains unquantified.

4. No evaluation of simple, possible mitigations defense on the attack.

### Questions
1. Could the authors further clarify the contribution over prior agent-security benchmarks such as AgentDojo, AgentHarm, and InjecAgent? I suspect hijack performance may be similar between systems that use simulated environments (as in this paper) and those that use simulated observations for email or shopping (as in AgentDojo or InjecAgent).

2. I am concerned about the assumption that current LLM agents rely on the AXTree to take actions. Has this been verified empirically? My expectation is that many agents rely on screenshots or DOM representations instead.

3. Your metric stops at a click. How does the HSR translate to downstream harm in realistic workflows (e.g., redirects, data exfiltration, unintended transactions)?

4. Some hijacks transfer broadly while others do not. What features distinguish globally transferable injections from model-specific ones?

5. As a benchmark, an initial average HSR of 25% seems low, which suggests the benchmark may not be sufficiently challenging. Consider strengthening tasks or injections to better discriminate model robustness.

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
This paper introduces TRAP (Task-Redirecting Agent Persuasion Benchmark), a benchmark for evaluating hijacking vulnerabilities in web-based LLM agents. The authors construct 630 task suites by combining 18 benign tasks with 35 injection templates built from five modular components: injection interface (button/hyperlink), human persuasion principles (7 Cialdini principles), LLM manipulation methods (5 types), injection location, and tailoring. The benchmark is built on REAL's cloned websites (Amazon, Gmail, Calendar, LinkedIn, DoorDash, Upwork). The key innovation is a one-click evaluation metric: hijacking success is determined when the agent clicks the injected element, avoiding ambiguity from multi-step outcomes and LLM judge bias. Evaluating six frontier models, they find an average 25% hijack success rate (ranging from 13% on GPT-5 to 43% on DeepSeek-R1), with buttons 3× more effective than hyperlinks and light tailoring increasing success by up to 5.6×.

### Strengths
- 630 task suites on realistic website clones to measure agent susceptibility.
- Verifiable evaluation without reliance on LLM judges.
- Interesting findings regarding the vulnerability of different models (hijack success rates ranging from 13% on GPT-5 to 43% on DeepSeek-R1).
- Using realistic clones of popular websites from REAL (Garg et al., 2025). This is important as prompt injections are a major threat for agents, and prompt injection benchmarks in realistic environments are highly needed.
- Accuracy on benign tasks is also measured and provides a baseline for agents’ capabilities.
- Showing that the considered injection templates are transferable between different LLMs is important and is a nice side contribution.
- Valuable ablation studies (e.g., hyperlinks vs. buttons).

### Weaknesses
- My major concern is that the benchmark has a very low number of unique tasks (only 18). A total of 630 tasks are created by using 35 injection templates (7 persuasion principles × 5 LLM manipulation methods). The benchmark would be more useful with a larger number of unique tasks (say, at least 50 or better 100). Varying injection templates is less interesting, since they shouldn’t be assumed fixed (see the discussion on adaptive attacks in [Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks](https://arxiv.org/abs/2404.02151) and [The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against Llm Jailbreaks and Prompt Injections](https://arxiv.org/abs/2510.09023)).
- *“We introduce a single, unambiguous success criterion: whether the agent clicks the injected element.”* - This makes the tasks easy to grade, but it’s also a weakness of the benchmark. Realistic hijacks typically require the agent to perform multiple steps.

Minor points:
- “Prompt injections” seems to be a much more established name compared to “hijacks”. I wonder why the authors seem to strongly prefer “hijacks”. When reading the paper, it was not clear to me if there is some substantial difference between them, but it seems like they refer to the same behavior.
- *“This creates ambiguity: if an agent starts to follow a malicious instruction but fails to complete it, is that a skill gap or a true refusal?”* - Note that these two things are easy to disentangle since one can measure refusals directly (e.g., as done in AgentHarm), which is a straightforward task for an LLM judge.

### Questions
No questions.

### Soundness
2

### Presentation
3

### Contribution
2
