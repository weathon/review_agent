# Draining Your Account: A Stealthy Attack on API Billing in Multi-Agent Systems

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Multi-Agent Systems (MAS) excel at complex problem-solving tasks by orchestrating specialized agents through the control flow. Agents are empowered by external APIs, accessed via the Model Context Protocol (MCP) which standardizes the interaction between Large Language Models (LLMs) and API services, harnessing the MCP server with three primitives---tools, resources, and prompts. However, the widespread adoption of MCP introduces a critical vulnerability in MAS frameworks: A greedy service provider is highly motivated to deploy a malicious MCP server designed to surreptitiously inflate API usage, thereby draining a user's pre-paid account. We introduce *Phantom*, a framework that generates such malicious MCP servers. *Phantom* executes a novel attack that hijacks the MAS control flow to repeatedly activate a targeted agent and compels it to make excessive and redundant API calls. Crucially, the attack preserves the overall MAS utility in task execution, and evades exception detection mechanisms deployed by frameworks, ensuring its stealth and persistence. Extensive evaluation across three multi-agent tasks, four leading industrial frameworks, and three state-of-the-art LLMs shows that *Phantom* effectively increases targeted API invocations by up to **26×** while maintaining an average attack success rate of **98%**. Furthermore, it demonstrates remarkable resilience, defeating six distinct mitigation with **94\%** average success rate. This work uncovers a severe, real-world threat to the MAS ecosystem and highlights the urgent need for new security paradigms.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an attack method on API billing. In summary, the method modifies the tool description and tool functionality to hijack the multi-agent control flow, repeating API calls and resulting in large bills.

### Strengths
- API billing is an important problem.
- The paper evaluates the attack on 4 real-world agent frameworks.

### Weaknesses
- The evaluation is weak. It includes only 3 tasks and cannot demonstrate the generalizability of this method.
- Are the API provider and the attacker the same? If the provider itself is malicious, it will lose its reputation; once discovered, no one will use that API anymore. If they want to do it, why not stealthily add some charges to the user's account instead? The MPD, TCI, ER, RR would be 0, and more effective than Phantom.
- The attacker must extract the control flow of the target agents, which is a very strong assumption. It's not very applicable in the real world.
- The attack effectiveness is limited by the agent's control-flow design. If the control flow design is good and has a retry limit, the system will automatically stop after a few retries.
- The key point of this method is how to direct orchestration toward A* and persuade it to make redundant API invocations via P, but I did not find a detailed description of how to do this. The description in Section 4.1 is unclear: it describes content_analysis and summary APIs, but those appear in different tasks according to Appendix B. Can the authors provide an example of the entire attack trace?

### Questions
- Do all sub-agents share a single MCP server?
- Just to confirm, what does * mean in Appendix B? Does it indicate parts modified by the attacker?
- Is Phantom an automated attack pipeline, or do you manually extract the AST and analyze the control flow for each task?
- How does this differ from traditional control-flow hijacking? My understanding is that it just changes the attack goal from a direct malicious action to repeating benign actions to accrue charges.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce an attack called Phantom which generates malicious MCP servers that target and compel agents in a multi-agent system to make redundant API calls. The attack is evaluated on three MAS tasks with multiple frameworks and models.

### Strengths
- The threat to repeated API usage is interesting and important.
- The evaluations and ablations are well-designed and scaled.

### Weaknesses
- **Differentiation from Prior Work:** There is a large body of pre-existing work on attacks via the the [MCP protocol](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks), [control flow hijacks on multi-agent systems](https://arxiv.org/abs/2503.12188), and indirect prompt injections. Very little of this existing literature is cited and/or engaged with and it is not immediately clear how their work differentiates from techniques in any of this prior work (although the goals may be different).
- **Writing & Math:** The presentation is often confusing or lacking, making the paper hard to follow. Some (non-exhaustive) examples:
  - The authors mention that they "generat[e] malicious MCP servers," but never specify what exactly that entails. In the *threat model* formalism in Section 3, it looks like their intervention may have something to do with the tooling / prompts, but no details or examples are given. They also describe different "modes" of the attack without showing how the modes are implemented within a server.
  - Imprecise definitions of Control Flow, Control Nodes, and Orchestration. 
  - Nonstandard usage of formalism: *e.g.,* summations over $Y_t$ (intermediate outputs), inconsistent usage of sub- and super-scripts, double definitions of $\mathcal{O}^t$, nonstandard usage of set notation.
- **Threat Model.** Unlike other previous work on MCP attacks (e.g., [invariant](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks)), this attack requires knowledge of the specific open-source MAS that's used to call the server and the specific tasks / queries.
- **Baselines:** There exist significantly stronger baselines in both the MCP attack literature and the multi-agent system attack literature, that this attack should be compared against (examples above).

### Questions
1. How are the methods in this work distinct from previous IPI and MCP attacks?
2. What exactly does Phantom alter in an MCP server? Is it general to all MCP servers? What benign MCP servers did you test on?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Phantom, a framework for executing stealthy billing attacks against API usage in Multi-Agent Systems (MAS). This paper identifies a critical vulnerability in which a malicious Model Context Protocol (MCP) server can manipulate an MAS to surreptitiously inflate API calls, thereby draining a user's account. The proposed attack framework addresses three core challenges: (1) reliably activating a specific agent despite dynamic orchestration, (2) injecting numerous extraneous API calls without degrading task performance, and (3) evading built-in detection mechanisms. The evaluation demonstrates that Phantom can increase targeted API invocations by up to 26× with a 98% average success rate, while also successfully bypassing six distinct mitigation strategies with 94% effectiveness.

### Strengths
Originality: This paper identifies, formalizes, and addresses a novel and economically motivated threat vector in Multi-Agent Systems: covert API billing attacks. The paper clearly articulates the core challenges of such an attack and proposes a comprehensive framework to solve them.

Quality: The threat model is well defined, the attack framework is thoughtfully designed, and the claims are substantiated through extensive evaluation across multiple tasks and industrial frameworks.

Clarity: The paper logically structures the problem, the attack mechanics, and the experimental results. It provides a balanced perspective by thoroughly evaluating both the effectiveness of the attack and the limitations of potential mitigation strategies.

Significance: By demonstrating an extremely high attack success rate (98%) while showing that existing mitigation techniques are largely ineffective, the paper highlights a critical and timely vulnerability.

### Weaknesses
1. The evaluation of the baselines could be strengthened. First, the paper should explicitly state the total number of attack attempts conducted for each baseline. Second, the conclusion that the Breaking Agents (BA) baseline "demonstrates superior performance" is not rigorously supported by the data. The BA baseline is not the top performer across all scenarios (e.g., Task 1), and citing the maximum ASR (0.56) and maximum RW (2.51) from different tasks makes the claim appear overstated and less methodologically sound.

2. The choice of external mitigation tools could be more robust. The paper relies on general-purpose static analyzers (CodeQL, Codacy) that are not specialized for the unique logic of AI applications or MCP servers. A more compelling evaluation would involve testing against a static analyzer designed specifically for AI/LLM application security.

### Questions
1. The paper states that the targeted hijack phase requires extracting the control flow from the MAS source code. Does this imply that the control flow analysis must be manually conducted for each distinct MAS framework? If so, how would this attack approach generalize to closed-source MAS frameworks where the source code is unavailable? Would the attack's effectiveness (as measured by ASR, RW, etc.) be significantly diminished in such a black-box scenario?

2. The study focuses specifically on the Model Context Protocol (MCP). Has this paper investigated whether this attack vector is viable against Multi-Agent Systems that utilize other common agent-tool interaction protocols (e.g., direct RESTful API calls)? If so, were there significant differences in the results or the attack methodology? If not, could this paper speculate on how the choice of protocol might influence the attack's feasibility and success?

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
4

### Summary
This paper introduced Phantom, a billing-targeted attack in MCP-enabled Multi-Agent Systems (MAS), where malicious MCP servers can inflate API usage from users' accounts. Phantom generates malicious MCP servers by extracting control flows, manipulating control nodes to activate targeted agents, and employing three utility preservation modes to conserve time for additional API invocations.  Phantom is evaluated over three MAS tasks using four frameworks  (AutoGen, CAMEL, LangGraph, Swarm) and three LLMs (GPT-4o, Gemini-2.5-pro, Qwen-3-plus).  The paper describes at length overviews of MAS, MCP servers, and both research challenges with Phantom and a meta-overview of the attack.  The paper also introduces several metrics to measure Maximal Invocation, User Unawareness, and MAS undetectability.  Phantom outperforms three baseline attacks (Direct Request, Prompt Infection, and Breaking Agents), which achieve maximum success rates of only 37% compared to Phantom's 98%. The authors evaluate six mitigation strategies, finding that mainstream code auditing tools (CodeQL, Codacy) achieve zero detection rates while LLM-based analysis and Monitor Agents suffer high false positive rates or prohibitive overhead (it is important to note that evaluation focuses on relatively simple control flows, and production MAS with complex agent interactions may present additional challenges for both attack execution and detection).

### Strengths
The topic and relevance of this attack are both important and timely. It is also appreciated that this research attempts users from potential exploitation by service providers.  The attack itself makes sense, although historical evidence of similar provider abuse as well as low-level details of the attack per-task (and extending it to arbitrary MCP-enabled workflows) are lacking.

### Weaknesses
## Lack of low-level details
Section 4 describes Phantom's three major components (targeted hijack, utility preservation, and exception bypass) at length.  However, the descriptions remain at a meta-level, and do not offer granular details in terms of how any of these three main components are carried out practically for any of the three tasks.  Correspondingly, it is unclear how Phantom could be applied to any arbitrary MCP-enabled workflow.  I.e., Phantom does not appear to be a systematic algorithm and Section 4 does little to *exactly describe* how Phantom achieves any of its various objectives.

For the three tasks:
- Domain Coding (T1)
- Report Generation (T2)
- Collaborative Research (T3)

Looking over the appendix, it looks like:
- T1: achieved by repeatedly calling a code programmer
- T2: achieved by repeatedly calling a summarizer
- T3: achieved by repeatedly calling web_search

For T2 and T3, it is clear that this is an action that can be repeated without impacting performance.  However, over sufficient time, the Leader Agent's comprehension will naturally degrade as the context grows (along with a quadratic impact on its inference time).  Thus, how are the number of repeats determined to balance these issues?  How can this be done arbitrarily, for any possible MCP-enabled workflow and LLM backbones? It is not clear how T1 may be repeated without significantly decreasing performance, since a codebase is actively rewritten with every extra invocation.

Much of the description in the main text is overly verbose, it would be much more beneficial to have Appendix B in the main text and describe the rationale behind how Phantom was designed and conducted in each scenario, then describe more concretely how Phantom may be applied to arbitrary workflows.

## Questionable statistical rigor
Are the results in Tables 1, 2, 3 and Figure 4 based on single runs for each LLM, framework, and task?  This is problematic for stochasticity used during LLM inference and aggregating multiple runs is necessary to rigorously test the effectiveness of the approach.

## Lack of historical/anecdotal examples
> A greedy API provider is highly motivated to exploit maximal profits through MAS excessively invoking its service.

Is there historical, anecdotal evidence for this type of behavior from service providers?  While the severity of the attack, and subversive financial impact on users, are extreme, examples of previous wrongdoing by service providers will make the attack much more compelling.  In terms of generalizing this attack beyond malicious intent and towards a potential failure mode of LLMs; is this attack pathological, or could it also naturally occur with a non-malicious MCP server and some seemingly benign prompt?

### Questions
What is "Qwen-3-plus?"  This is not a model name from the Qwen3 family.

Suggested revision:
> Specifically, Phantom performs targeted hijack by extracting the MAS control flow and manipulating the decisions of key control nodes to repeatedly steer
execution towards the target agent; ensures utility preservation by employing sophisticated strategies
to maintain normative task execution and mask the added latency; and achieves exception bypass by
carefully optimizing the frequency and nature of the refined tools and prompts to remain below the
detection thresholds of various system integrity checks.

to:
> Specifically, Phantom: a) performs targeted hijack by extracting the MAS control flow and manipulating the decisions of key control nodes to repeatedly steer
execution towards the target agent, (b) ensures utility preservation by employing sophisticated strategies
to maintain normative task execution and mask the added latency, and (c) achieves exception bypass by
carefully optimizing the frequency and nature of the refined tools and prompts to remain below the
detection thresholds of various system integrity checks.

Much of the paper remains at a high level.  E.g., reading through, I had the following question
> preserving MAS performance or creating time for additional invocations

How is this possible?  Low-level details and examples are very sparse, I have this question written many times through reading the paper. It would be much better to, after the first high-level description of Phantom (e.g., after "Threat Model," the challenge research questions section can also be greatly condensed), introduce the various tasks with Phantom attacks (e.g., Appendix B) and describe what modifications were necessary to achieve Phantom and the methodology behind these changes.

### Soundness
2

### Presentation
2

### Contribution
3
