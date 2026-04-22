# AI Kill Switch for Malicious Web-based LLM Agents

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 4, 2, 2

## Abstract
Recently, web-based Large Language Model (LLM) agents autonomously per-
form increasingly complex tasks, thereby bringing significant convenience. How-
ever, they also amplify the risks of malicious misuse cases such as unauthorized
collection of personally identifiable information (PII), generation of socially di-
visive content, and even automated web hacking. To address these threats, we
propose an AI Kill Switch technique that can immediately halt the operation
of malicious web-based LLM agents. To achieve this, we introduce AutoGuard
– the key idea is generating defensive prompts that trigger the safety mechanisms
of malicious LLM agents. In particular, generated defense prompts are transpar-
ently embedded into the website’s DOM so that they remain invisible to human
users but can be detected by the crawling process of malicious agents, triggering
its internal safety mechanisms to abort malicious actions once read. To evaluate
our approach, we constructed a dedicated benchmark consisting of three repre-
sentative malicious scenarios. Experimental results show that AutoGuard achieves
over 80% Defense Success Rate (DSR) across diverse malicious agents, including
GPT-4o, Claude-4.5-Sonnet and generalizes well to advanced models like GPT-
5.1, Gemini-2.5-Flash, and Gemini-3-Pro. Also, our approach demonstrates robust
defense performance in real-world website environments without significant per-
formance degradation for benign agents. Through this research, we demonstrate
the controllability of web-based LLM agents, thereby contributing to the broader
effort of AI control and safety.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a practical, deployable “AI kill switch” for web-based LLM agents by embedding defensive prompts into webpages that trigger agents’ built-in safety policies and halt malicious tasks at runtime. The core method, AutoGuard, uses a defender LLM to iteratively synthesize and refine defensive prompts against a set of attack prompts, with a feedback LLM judging success and driving prompt revisions. Evaluation spans three malicious scenarios—PII collection, socially divisive content generation, and web vulnerability scanning—on synthetic but interactive sites. Reported Defense Success Rates (DSR) are high across several agents and remain strong on additional models.

### Strengths
1. AutoGuard’s lightweight write–inject–judge–revise loop is easy to use and reproduce. It avoids complex optimization while still producing effective prompts

2. The proposed framework works effectively under the synthetic environment.

3. The paper is clear and easy to follow.

4. The paper acknowledges fragility against multimodal/screenshot-based agents and real-world constraints.

### Weaknesses
1. The threat model assumes the adversary is unaware of the defensive prompt and uses general-purpose, safety-aligned agents, which is a very strong assumption. For instance, a motivated attacker can add pre-filters to ignore patterns resembling “system” messages in page text or apply robust IPI defenses.

2. All tests run on synthetic sites. Real sites include dynamic rendering, auth walls, third-party scripts, iframes, CSP, and heterogeneous HTML/JS patterns that may alter visibility and agent extraction. The paper openly states this limitation, but the performance claims would be stronger with at least a few permissions-based real-world trials.

3. The defensive prompt is, in itself, a prompt injection into the agent. This creates a precedent that any party can embed invisible text to steer visiting agents, potentially enabling misuse to block, degrade, or misdirect benign agents.

4. Baselines are weak: “Prompt Injection” and “Warning-based” prompts do not represent the broader defense landscape. Missing comparisons include modern agent-side IPI detectors, stricter browsing sandboxes, server-side mitigations

### Questions
1. Can you report false-positive rates for benign automation tasks and effects on legitimate agents or crawlers?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces AutoGuard, a technique for mitigating the misuse of web-based AI agents. The technique relies on a simple yet effective text injection, revision, and verification pipeline. Experimental results demonstrate the effectiveness of AutoGuard in mitigating simplified malicious scenarios involving website navigation, web script reading, and button clicking.

### Strengths
1. The paper studies an important and urgent problem that may potentially have a significant impact on the development of web infrastructure and web agents.

2. The proposed technique is novel, simple, and seemingly effective.

3. The experiments show that AutoGuard is effective across various models on simplified settings.

### Weaknesses
1. Although the injected defensive prompts by AutoGuard is invisible to human, it is unclear whether they degrade normal agent functionality (e.g., task success rates, latency, or unintended refusals)

2. Although the paper proposes a dedicated benchmark to test AutoGuard, its effectiveness needs to be evaluated on standard benchmarks, such as standard benchmarks for web vulnerability screening.

3. It’s unclear how the attack prompts are created. How realistic are they? Whether a principled, reproducible methodology was used.

### Questions
Thanks for submitting this work. I think this is a valuable contribution that can be further strengthened.

1. Please justify the need for a dedicated benchmark. Which gaps in existing web‑agent or cybersecurity benchmarks prevent their use or extension?

2. Table 1: Why are function calling methods limited to only navigate_website, get_clickable_elements, and get_scriptcode? It seems to miss the method that allows agent to send request with potentially malicious payloads to web servers?

3. Based on Appendix A.6, the paper only considers static websites. How effective would AutoGuard be for dynamic websites (e.g., client-side rendering or asynchronous DOM updates)?

4. AutoGuard manipulates the content of a webpage. Although the manipulation is invisible to humans, could this affect the normal use of web agents? Please evaluate on standard web‑agent benchmarks (e.g., WebArena) to assess any performance impact.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this study, the authors propose AI Kill Switch, an automated defense prompt generation for web ai agent. With this approach, a defender LLM can autonomously generate defense prompts, which are hidden and embedded in a webpage’s HTML so that humans do not see them but LLM agents detect them during their crawling process and automatically activate their safety policies. The authors design attack prompts for three scenarios and evaluate the effectiveness of the AI Kill Switch in each case.

### Strengths
- This paper includes an automated method for generating defense prompts and reports experiments using several LLM backbones (GPT-5, Deepseek-r1, GPT-o3).

- The authors test attacks based on 63 attack prompts composed of direct requests and bypass requests, and show that defense prompts can increase DSR by 10–40% for a productized agent (ChatGPT-agent) and achieve DSRs approaching ~90% for models such as GPT-5 and GPT-4.1.

### Weaknesses
- Defense prompts are iteratively refined using the agent’s responses. This makes the approach agent-dependent, can be costly (many LLM calls), and may fail to produce consistent, general defense performance across different scenarios and webpages.

- Experiments were run only on synthetic (controlled) webpages or archived testbeds; the authors note that real-world variables (dynamic rendering, ads/trackers, etc.) could affect results.

- It is difficult to judge how realistic the tested malicious requests are, and it’s unclear how well the generated prompts generalize cross-task or cross-domain.

### Questions
- How many iterative revision steps are typically needed to produce a defense prompt with sufficient performance?

- Are there effective methods to make defense-prompt generation more efficient?

- What is the cost of AutoGuard (in practice)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes AutoGuard, a practical “AI kill switch” for web-based LLM agents. The idea is to auto-generate short defense prompts and invisibly embed them in a site’s DOM. When a malicious agent scrapes page text, it also ingests the defense prompt, which then triggers the model to refuse the task (e.g., stop collecting PII). The authors evaluate on a new benchmark spanning 3 scenarios (PII collection, “social-rift” content generation, and web-vulnerability scanning). Results show high defense success rates (DSR), with >80% on GPT-4o/Claude/Llama3.3-70B and ~90% on other agents.

### Strengths
1. AI agents might automate malicious actions at scale. While most efforts want to align the model itself, this paper proposes another complementary techniques: adding a defensive prompt in important webpages to actively warn the model. This is simple and straightfoward.

2. They run diverse, reproduciable evaluations on three representative malicious scenarios (PII collection, social rift content generation, and
web hacking attempts).

3. Results show that the proposed automatic defensive prompt generation methods can often achieve 80~90% DSR across multiple LMs such as gemini-2.5-flash, gpt-5, and claude 3.

### Weaknesses
1. The core assumption is that the agent ingests DOM text naïvely. Smart attackers can filter hidden text, ignore off-screen content, parse only visible nodes, or sanitize obvious “system-style” strings—an issue the paper briefly acknowledges but largely evaluates with a single hidden-text strategy. Stronger adaptive-attacker tests would strengthen claims.

2. All tests occur on controlled demo sites, not the messy, dynamic public web (auth walls, ad/consent overlays, JS-rendered content). Real-world generalization remains an open question.

3. AutoGuard is optimized on malicious queries that share the same distribution of those test malicious queries. It's unclear how Autograd generalizes to OOD malicious queries.

4. Not a strict weakness but a note: the AutoGuard framework is simple and has no technical novelty.

### Questions
1. how do you jailbreak LMs in the first place? I was thinking many of the queries in the benchmark should be directly refused by these frontier LMs. In addition, why not evaluate models with better agentic alignment performance e.g. claude 4 and claude 4.5?

2. Are some test queries quite ambiguous so the model decide not to refuse them? For those cases, AutoGuard unsurprisingly works because there are additional signals introduced during prompt optimization that help clarify these ambiguous cases.

### Soundness
2

### Presentation
2

### Contribution
2
