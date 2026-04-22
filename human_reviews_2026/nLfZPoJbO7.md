# HackWorld: Evaluating Computer-Use Agents on Exploiting Web Application Vulnerabilities

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Web applications are prime targets for cyberattacks due to their role as entry points to vital services and sensitive data repositories. Traditional penetration testing is expensive and requires specialized expertise, creating scalability challenges for securing the expanding web ecosystem. While language model agents have shown promise in certain cybersecurity tasks, modern web applications require visual understanding of complex user interfaces, dynamic content rendering, and multi-step interactive workflows that only computer-use agents (CUAs) can handle. Despite CUAs' demonstrated capabilities in web browsing and visual task automation, their potential to discover and exploit web application vulnerabilities through graphical interfaces remains unknown. 
We introduce HackWorld, the first evaluation framework for systematically assessing CUAs' capabilities in exploiting web application vulnerabilities through visual interaction. Unlike existing benchmarks using sanitized environments, HackWorld exposes CUAs to 36 curated applications spanning 11 frameworks and 7 languages, containing realistic vulnerabilities including injection flaws, authentication bypasses, and unsafe input handling. Our framework directly evaluates CUAs' ability to discover and exploit these vulnerabilities using Capture-the-Flag (CTF) methodology while navigating complex web interfaces.
Evaluation of state-of-the-art CUAs reveals exploitation rates below 12%, struggling to plan multi-step attacks and use security tools effectively.
Our results expose CUAs' limited cybersecurity skills when operating on vulnerable web applications, opening future research directions on developing security-aware CUAs for vulnerability detection and exploitation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper develops a test environment for evaluating the ability of agent AI systems to navigate websites and detect security vulnerabilities. The test environment is interesting in itself. It involves 36 curated applications spanning 11 frameworks and 7 languages, with realistic vulnerabilities including injection flaws, authentication bypasses, and unsafe input handling.The paper also evaluates a set of agents against this test suite and finds them to have limited power to uncover sophisticated vulnerabilities. Specifically, the exploitation rates are below 12%, often struggling to plan multi-step attacks and use security tools effectively. The paper this creates a good setting for further work on developing more effective web application security tools based on navigating illustrative and complex websites.

### Strengths
The approach is very good. It is useful to have a systematic way to evaluate web application vulnerability tools. The testing setup is interesting and compelling.

### Weaknesses
This would be a stronger paper, of course, if it also included tool improvements, validated within this test environment. That is more work, of course, and likely the next step (and next paper) for this team. However, without some indication that at least some of the limitations of current tools may be overcome, it is not clear whether the benchmark sets an achievable near-term goal, or simply a calibration point for the state of the art.

### Questions
Are there actionable insights for developing improved web application security tools? Even if not, the characterization of failure modes for existing approaches may be useful.

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
3

### Summary
Hackworld curates 36 CTF challenges, sourcefrom NYU ctf bench, CyBench amd InterCode-CTF.
36 web applications over 7 programming language, 11 web frameworks.
Each web app deployed as isolated docker.

The novelty is wrapping them into a CUA-problem.
Environment is Kali OS with security analysis tools, such as Burp Suite, DirBuster, Nikto.

Progress monitored: filesystem operations, tool invocations, HTTP requests.


They use Claude sonnet and Opus CUA agents + UI-Tars and Qwen-VL
0% success rate for open-source models.

Questions:
3) SOM: is that just the SOM or is it combined with Screen shot / Screenshot+AxTree?
I think it does not make sense to test it in isolation and should rather be used to complement Screenshot+AxTree.

### Strengths
Realistic attacker environment, usage of industry-standard tools and Kali OS.
Great discussion and insightful work on failure analysis.

### Weaknesses
Results are 0% with open-source models. A bit surprising to me given that if we look at result tables from papers like Cybench (Llama, Mixtral) the open weight models  achieve better than 0% success rate.
However, I do see that NYU CTF shows 0% for the open weight models.
Given that you state that choice of observation format is not that important, i would like a discussion why models that succeed in Cybench fail in HackWorld.

### Questions
Exploitation rate below 12% + security oblivious behavior -> Can you explain more why this s concerning?
If a computer-use agent fails to capture the flag, it might mean it is less susceptible to cause damage under attacks, e.g., prompt injections, which is not necessarily a bad thing.

Does the CUA agent use the terminal? If so, is it helpful? Does it make sense to restrict direct terminal usage? What are the benefits of CUA vs. terminal-use agent in this context? Do any of the tasks actually require, e.g., very interactive  browser interaction. I would love to see a discussion on this, because using CUA for security-testing vs. a Terminal-use agent is not an obvious choice to me.

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
This paper introduces HackWorld, a new benchmark for evaluating computer-use agents (CUAs) in realistic web environments containing exploitable vulnerabilities. Instead of testing on clean, simplified webpages, HackWorld embeds agents in 36 real vulnerable web applications (built with 7 languages and 11 frameworks) and measures their ability to identify and exploit vulnerabilities using a CTF (Capture-The-Flag) formulation.

### Strengths
The paper provides a realistic and technically comprehensive benchmark, incorporating genuine vulnerabilities and real-world tools rather than synthetic setups. 

The evaluation framework is well-implemented and could be useful for testing future computer-use agents. 

The experimental setup is clear, and the findings provide meaningful insights into the current limitations of CUAs in handling realistic exploitation tasks.

### Weaknesses
The motivation is weak and not well justified — it is unclear why evaluating “common-sense defense” or hacking ability is of substantial research value or relevance to practical LLM applications.

 The abstract and introduction are wordy and difficult to follow, which obscures the main contribution. 

The conceptual novelty is limited; the work is mostly an engineering effort to build a testbed rather than a new research idea. Moreover, the experiments lack depth in analysis, focusing only on success rates without exploring why models fail or what reasoning gaps cause the failures. 

Finally, the evaluation of closed-source models is incomplete — only the Claude family is tested, which weakens the generality of the conclusions.

### Questions
see weakness

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
3

### Summary
The paper proposes HackWorld, a GUI-based benchmark for evaluating Computer-Using Agents (CUAs) that try to discover and exploit real web-application vulnerabilities. The benchmark aggregates 36 containerized CTF-style challenges spanning multiple frameworks and languages, runs in a Kali-based environment with industry-standard security tools, and evaluates success via flag capture with reproducible logs. The study shows that current CUAs perform poorly (<~12% success), analyzes failure modes (planning, tool use, and GUI manipulation), and argues that “web hacking via GUI” is substantially harder than generic web browsing or visual automation.

### Strengths
- The task is clearly defined and the CTF style evaluation metric is unambiguous.
- Using MLLM-based agents to autonomously discover web-application vulnerabilities is a hard and meaningful task.
- The paper provides detailed analysis on the error cases, which can guide future research.

### Weaknesses
- Backbone diversity is narrow. Only six models are tested, four from the same family (Claude), plus a single 7B model that is too weak for this task. This limits the generality of the conclusions. Stronger open and closed models, and more families, are needed.
- While the curation is solid, 36 tasks is still modest for a general benchmark.
- As a benchmark paper, it is unclear what specific new agent capabilities are evaluated compared to OSWorld [1].

[1] Xie, Tianbao, et al. "Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments.", 2024

### Questions
- Could you report results on stronger models such as GPT and Gemini, which have demonstrated strong performance in the OSWORLD paper, and compare them with human results?
- Compared with OSWorld, does HackWorld evaluate a fundamentally different type of agent capability, or is the distinction mainly in the cybersecurity domain knowledge? What new abilities or reasoning skills are specifically required of agents in this setting?

### Soundness
3

### Presentation
3

### Contribution
3
