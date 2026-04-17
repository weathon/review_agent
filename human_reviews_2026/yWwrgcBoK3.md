# RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments

- Decision: Accept (Oral)
- Scores: 6, 6, 8, 4

## Abstract
Computer-use agents (CUAs) promise to automate complex tasks across operating systems (OS) and the web, but remain vulnerable to indirect prompt injection, where attackers embed malicious content into the environment to hijack agent behavior. Current evaluations of this threat either lack support for adversarial testing in realistic but controlled environments or ignore hybrid web-OS attack scenarios involving both interfaces. To address this, we propose RedTeamCUA, an adversarial testing framework featuring a novel hybrid sandbox that integrates a VM-based OS environment with Docker-based web platforms. Our sandbox supports key features tailored for red teaming, such as flexible adversarial scenario configuration, and a setting that decouples adversarial evaluation from navigational limitations of CUAs by initializing tests directly at the point of an adversarial injection. Using RedTeamCUA, we develop RTC-Bench, a comprehensive benchmark with 864 examples that investigate realistic, hybrid web-OS attack scenarios and fundamental security vulnerabilities. Benchmarking current frontier CUAs identifies significant vulnerabilities: Claude 3.7 Sonnet | CUA demonstrates an Attack Success Rate (ASR) of 42.9%, while Operator, the most secure CUA evaluated, still exhibits an ASR of 7.6%. Notably, CUAs often attempt to execute adversarial tasks with an Attempt Rate as high as 92.5%, although failing to complete them due to capability limitations. Nevertheless, we observe concerning ASRs of up to 50% in realistic end-to-end settings, indicating that CUA threats can already result in tangible risks to users and computer systems. Overall, RedTeamCUA provides an essential framework for advancing realistic, controlled, and systematic analysis of CUA vulnerabilities, highlighting the urgent need for robust defenses to indirect prompt injection prior to real-world deployment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces RedTeamCUA, a novel adversarial testing framework designed to evaluate the susceptibility of Computer-Use Agents (CUAs) to prompt injection in realistic, controlled, hybrid environments spanning both the web and the operating system. The core of the framework integrates a VM-based OS environment (based on OSWorld) with Docker-based containers. This design allows for controlled, end-to-end evaluation of attacks that cross the web and OS boundary, such as an injection on a forum leading to a harmful local OS action.

The authors also propose RTC-BENCH, a comprehensive benchmark of 864 examples that investigates realistic hybrid attack scenarios targeting fundamental security violations categorized by the Confidentiality, Integrity, and Availability (CIA) triad. The framework also introduces a Decoupled Eval setting, which bypasses CUA navigation limits to isolate and focus on core adversarial robustness.

Benchmarking frontier CUAs, including Claude and Operator revealed significant vulnerabilities. Even the most secure agent, Operator, exhibited an Attack Success Rate (ASR) of 7.6% in the Decoupled setting, while Claude 3.7 Sonnet reached an ASR of 42.9% and in realistic End2End settings, ASRs reached ~50%. The authors conclude that CUA threats are already tangible and current defenses, including both built-in CUA mechanisms and four evaluated defense methods, are insufficient.

### Strengths
1. The first controlled, integrated framework for adversarial testing across both realistic OS and web environments.

2. Decoupled Evaluation Setting effectively isolates an agent’s true adversarial robustness from confounding factors like navigational capability, providing a clear measure of vulnerability once an injection is encountered.

3. The empirical results demonstrate that frontier CUAs are highly vulnerable, with Attack Success Rates reaching up to 50% in end-to-end settings, confirming that these are immediate, tangible risks.

4. The threat model is more realistic than much prior work.

### Weaknesses
1. The current findings are primarily limited to a few closed source models and do not test on open source CUA models like UI-TARS 1.5 70B.

2. The realistic End2End evaluation was performed only on a subset of 50 tasks out of the 864 examples in the benchmark.

### Questions
N/A.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors propose RedTeamCUA, a benchmark for "hybrid web-OS" indirect prompt injections on a number of frontier models / agents. They find that most tested agent+scenario+model combinations are vulnerable to attacks in their benchmark in both targeted and end-to-end experiments. In addition, a suite of recent defenses are circumvented.

### Strengths
- Well written and designed benchmark.
- Addresses an important niche of attacks on CUA and hybrid models.
- Interesting findings on the limitations of current defense frameworks in this regime.

### Weaknesses
- Novelty: The method seems to build on top of and combine existing benchmarks and attacks. In a future version, I would like to see the authors spell how their benchmark differs a little more.
- Attacks: Only one format of attack was evaluate. Would be nice to see evaluation of more.
- Sanitization: How sensitive are results to realistic UI noise (extra messages/files)?

### Questions
- How sensitive are results to realistic UI noise (extra messages/files)?
- Any signal as to whether results transfer to other common platforms (e.g., wikis, issue trackers)? 
- In End2End, how much of the ASR drop is due to navigation vs perception vs tool-use? A per-failure taxonomy would be very interesting.

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
Authors propose a new framework (redTeamCUA) + benchmark (RTC Bench) by combining OSWorld environment w/ WebArena and TheAgentCompany.

Combination of benign goals and adversarial goals similarly to AgentDojo:
(9 benign goals × 24 adversarial goals × 4 types of instantiation)
- instantiation means: Code vs. NL prompt injection x General vs. Specific benign goal

Interesting seeting: Decoupled Eval, where the agent is brought to the point of adversarial injection (hardcoded prior tool calls?)

Metrics are ASR and AR (attempt rate for adversarial goals).

They show high ASR for all evaluated models, except for Operator, which still has 7.6%.

### Strengths
- OSWorld backbone allows for hybrid attacks over both OS and web
- Realistic threat model
- Decoupled evaluation setting is good for helping weaker capability models reach the point of prompt injection, though it might be somewhat un-natural depending on how the tool-calling/traces are hard-coded to the agent history
- Great having both Web -> OS -> Web and Web -> OS adversarial scenarios

### Weaknesses
- The large number of 864 examples is only achieved by cross-product of benign, injection, and instantiation. The number of benign tasks (9) might be too small to accurately estimate agent utility, which is critical in any security benchmark (otherwise a useless agent might have perfect security). 
- I disagree with the the assessment that the Doomarena threat model is requires full webpage control; even though the authors note that the banners and pop-up attacks are injected into the web page (e.g. of PostMill) by modifying the DOM, these elements are typically 3rd party content, which is editable without full page access by, e.g., submitting the attacks through some advertising platform.

### Questions
- You use the CIA taxonomy of threats. I'm curious where something like Direct Harm (e.g. send money to attacker) would land.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents REDTEAMCUA, a framework for realistic adversarial testing of computer-use agents (CUAs) across hybrid web–OS environments. It builds a VM + Docker sandbox and an RTC-BENCH benchmark with 864 adversarial cases to evaluate indirect prompt-injection risks. Experiments show serious vulnerabilities in leading CUAs (e.g., ASR up to 66%, AR > 90%), even for models like Claude 3.7 Sonnet | CUA and Operator. Existing defenses such as LlamaFirewall and Meta SecAlign offer limited protection, revealing urgent needs for stronger security in CUAs.

### Strengths
– Proposes a well-designed, hybrid sandbox integrating web and OS layers, bridging realism and safety in adversarial testing.

– Builds a large-scale, systematic benchmark (RTC-BENCH) grounded in realistic tasks and security principles (CIA triad).

– Provides comprehensive empirical results with both execution-based and LLM-judge metrics, revealing concrete weaknesses in current frontier CUAs.

– Conducts thoughtful analysis comparing adapted LLM agents vs. specialized CUAs, and offers valuable insight into the trade-off between autonomy and safety.

### Weaknesses
– some closed-source CUAs evaluated (GPT-4o, Claude 3.5/3.7 Sonnet, Claude 4 Opus, Operator) dominate the study; no strong open-source CUAs (e.g., UI-TARS 2, OpenCUA) are included, limiting reproducibility and community relevance. More closed-source CUAs and open-source CUAs need to be included.

– The defense evaluation is superficial—existing methods are merely tested rather than extended or improved.

– Provides limited mechanistic analysis of why specific CUAs succumb to injection (e.g., reasoning path, memory, or grounding failure).

– Some sections are overly descriptive and lengthy, diluting the main technical insights.

### Questions
see weakness.

### Soundness
2

### Presentation
2

### Contribution
2
