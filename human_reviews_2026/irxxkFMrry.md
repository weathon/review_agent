# MCP Security Bench (MSB): Benchmarking Attacks Against Model Context Protocol in LLM Agents

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
The Model Context Protocol (MCP) standardizes how large language model (LLM) agents discover, describe, and call external tools. While MCP unlocks broad interoperability, it also enlarges the attack surface by making tools first-class, composable objects with natural-language metadata, and standardized I/O. We present MSB (MCP Security Benchmark), the first end-to-end evaluation suite that systematically measures how well LLM agents resist MCP-specific attacks throughout the full tool-use pipeline: task planning, tool invocation, and response handling. MSB contributes: (1) a taxonomy of 12 attacks including name-collision, preference manipulation, prompt injections embedded in tool descriptions, out-of-scope parameter requests, user-impersonating responses, false-error escalation, tool-transfer, retrieval injection, and mixed attacks; (2) an evaluation harness that executes attacks by running real tools (both benign and malicious) via MCP rather than simulation; and (3) a robustness metric that quantifies the trade-off between security and performance: Net Resilient Performance (NRP). We evaluate nine popular LLM agents across 10 domains and 405 tools, producing 2,000 attack instances. Results reveal the effectiveness of attacks against each stage of MCP. Models with stronger performance are more vulnerable to attacks due to their outstanding tool calling and instruction following capabilities. MSB provides a practical baseline for researchers and practitioners to study, compare, and harden MCP agents. Code: https://github.com/dongsenzhang/MSB

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
MSB (MCP Security Benchmark) is the first end-to-end evaluation suite designed to systematically assess how well large language model (LLM) agents resist attacks within the Model Context Protocol (MCP) framework. It introduces a taxonomy of 12 attack types, an evaluation harness that executes real attacks using both benign and malicious tools, and a robustness metric called Net Resilient Performance (NRP) to quantify the trade-off between security and performance. Evaluations across nine LLM agents, ten domains, and over 400 tools (2,000 attack instances) reveal that models with stronger task performance are often more vulnerable to attacks during tool use.

### Strengths
- Comprehensive, MCP-specific workflow diagram. Figure 1 (p. 2) clearly maps attacks to the MCP pipeline, useful for readers designing defenses. 
- End-to-end threat coverage tailored to MCP. Clear taxonomy spanning planning, invocation, response, and retrieval.

### Weaknesses
- The paper is submitted to the Datasets and Benchmarks track, but the benchmark collection and processing protocol are insufficiently described. Please provide detailed methodology for how malicious tools were constructed or collected:

      1. Are the malicious tools hand-crafted by the authors, or entirely collected from real-world MCP platforms?

      2. If they are hand-crafted, can those tools pass the MCP platforms’ own checks/filters? Please report whether you attempted to deploy them on any MCP platform and the outcome.

      3. If they are collected from the wild, how did you verify that a tool is actually malicious? What static-analysis or other toolchains were used for labeling (e.g., static scanners)? If you used static detectors, discuss whether such detectors could filter out most malicious tools and what remains undetected.
      4. Finally, clarify which malicious tools in your benchmark represent realistic threats that developers/operators actually use in practice and which are contrived or unlikely to appear in deployed MCP ecosystems.

- The proposed NRP metric appears to be a straightforward combination of ASR and PUA. Please clarify its added value: what does NRP capture beyond the simple union of ASR and PUA? Provide intuition, formal definition, and examples showing cases where NRP yields distinct, actionable insight compared to reporting ASR and PUA separately. Also, With NRP = PUA·(1−ASR), higher NRP should mean better robustness, yet the text claims the opposite (“A higher NRP suggests… poor performance or high vulnerability”). This needs correcting to avoid reader confusion (Sec. 5.2).

- The paper’s insights are currently too thin. For example, Table 3 mixes results from different base models and reports widely varying attack success rates, but the paper does not analyze why. This undermines the paper’s contribution: at present it reads like an empirical report rather than a community-useful benchmark. Strengthen the analysis by showing whether your benchmark can explain the root causes of these performance differences (e.g., architecture, pretraining data, safety-layer differences, instruction-tuning, decoding heuristics). Provide controlled experiments or post-hoc analyses that isolate the main factors behind attack-success variability.

- Please identify which attack methods pose the greatest practical risk to current agent deployments. How do MCP-specific attacks differ from classical prompt-injection attacks? Several sections in the manuscript mention attacks without clearly explaining the threat model, attack surface, or their concrete impact on agent behavior, these must be substantially expanded and justified.

- The benchmark does not appear to be open-sourced yet. State a clear open-source timeline and a release plan (what will be released: raw data, labels, tooling, reproduce scripts, evaluation code; license; sustainability/hosting). Without an explicit release plan and timeline, the community benefit of this work is limited.

### Questions
- What is your data collection procedure? 

- Why must manually modifying a malicious tool be practical for real-world use?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents the first LLM proxy Security benchmark based on MCP - MCP Security Bench (MSB). MSB encompasses 12 types of attacks, 2,000 test cases, covering 10 application scenarios, 65 user tasks, and over 400 tools. By executing attacks through real tools, it evaluated the security of 9 mainstream LLM models in the MCP environment. The paper also proposes three evaluation metrics: Attack success Rate (ASR), Performance under attack (PUA), and net resilience Performance (NRP). The experimental results show that the unique attack methods of MCP have a relatively high success rate, and the stronger the model capability, the more vulnerable it is to attacks.

### Strengths
* The research value is prominent: As an emerging LLM proxy tool invocation protocol, the security of MCP has not yet been systematically evaluated. This paper constructs a large-scale and dynamic benchmark for the first time, filling the gap in security assessment in this field.

* The method is clearly described: The attack classification is explicit (such as tool signatures, parameters, response attacks, etc.), the experimental Settings are detailed, and the charts and examples are rich, making it easy to understand and reproduce.

* The evaluation framework is practical: The proposed NRP indicators comprehensively consider performance and security, providing a quantitative basis for trade-offs in actual deployment.

### Weaknesses
-	The classification construction lacks methodological support: Although the division of attack types is clear, it lacks systematic methodological support (such as whether it is based on threat modeling, attack tree analysis, etc.), and it is questionable whether the classification is comprehensive and non-overlapping.
-	Insufficient analysis of data construction quality: The article claims that user tasks "cover diverse real scenarios", but fails to provide the basis for task selection, diversity analysis or representativeness verification, and lacks quantitative analysis of task distribution, difficulty and coverage.
-	The association between security risks and attack types is not clear: Table 7 lists six types of attack targets (such as obtaining remote control permissions, stealing personal data, etc.), but does not explain how these attack targets are specifically associated with 12 types of attacks (such as PI, OP, UI, etc.). Readers find it difficult to understand the specific harm and severity that different types of attacks may cause in practice.

### Questions
-	Is the attack classification complete? Is there a systematic approach (such as attack surface analysis, threat modeling) to ensure that these 12 types of attacks cover all potential risks in the MCP protocol?
-	How can the representativeness and diversity of user tasks be guaranteed? Are there any standards or guidelines for task design? Have different industries, complexities or usage frequencies been taken into consideration?
-	Is there a mapping relationship between the attack tasks in Table 7 and the specific attack types?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces MCP Security Bench (MSB), a comprehensive security benchmark for evaluating MCP-powered agents across a large number of tools (304 benign, 400+ malicious) and use cases (65). The paper presents a taxonomy of 12 attack types spanning the following stages: 
- task planning: name collision, preference manipulation, and prompt injection attacks
- tool invocation: out-of-scope parameters
- response handling:  user impersonation, false errors, tool transfer, and retrieval injection attacks

Note that this is not the first attack taxonomy created for MCP servers, however the paper does well to aggregate and present high-profile MCP-specific attacks which have emerged in the past year under one roof.  For 10 task scenarios, MSB blends user requests and attack parameters, resulting in 2,000 attack instances.  The paper utilizes attack success rate (ASR) and performance under attack (PUA), while debuting Net Resilient Performance (NRP), which measures the overall utility while under attack.  Experiments over nine popular LLM backbones show inverse scaling law where more capable models exhibit higher vulnerability, which substantiate many recent instance-wise observations made within the MCP safety community over the last year.  Furthermore, the granular metrics per attack show that MCP-specific attacks have far higher success rates than pre-MCP (e.g., prompt injection) attacks, demonstrating the need for further MCP safety understanding and effective defenses in future work.

### Strengths
MSB is a timely addition to the agentic benchmarking landscape, as the MCP continues to be heavily adapted.  Thus, the provided ability to measure the safety and security of existing and future LLMs in the context of MCP-specific attack is important.  Furthermore, the observations regarding MCP-tool-specific attacks and pre-MCP attacks (e.g., prompt injection) are interesting and help broaded the understanding of the current state of MCP safety	(and the need for better protection against such attacks).  Similarly, the inverse scaling laws corroborate much of the anecdotal attack examples observed throughout the MCP safety community over the past year.  The comprehensiveness of the benchmark is also important, as existing MCP-specific benchmarks are limited in both scope and automation.

## References

### Weaknesses
## Clarity
The primary concern is related to clarity, both in terms of the mathematical attack descriptions and details concerning the framework themselves.  I am hoping the authors can resolve this concern during the rebuttal.

### Notation
There is inconsistency in the Agent function definition, can the authors define this and, alternatively, overload the function for fewer parameters?  E.g., Equation 1 is Agent(LLM(\cdot), tools, knowledge), Equation 2 is Agent(query, adversarial modifications) (note, in this context, adversarial modifications is not well defined, i.e., I am assuming they involve any modification to the LLM(\cdot), tools, and/or knowledge, but this needs to be defined), and Equation 3 is Agent( LLM (\cdot), tools).  Similarly, the definition of $a_m$ changes based on the attack, which can be confusing.  I suggest having a table in the appendix (supplementing Table 1) that list (attack(s), equation, context dependent notation and/or function/action changes).  

### Benchmark details
> MSB spans 10 domains with 304 benign tools and more than 400 malicious tools. Benign tools are hosted via the Smithery MCP integration platform (Smithery.ai, 2025); malicious tools implement crafted names, descriptions, parameter schemas, and responses,
easily supporting a variety of attacks with differing operational sensitivities and stealth levels.

How were the benign tools chosen and vetted?  How were the malicious tools created and vetted?

> Based on the functions
of benign tools, we designed 65 user tasks to ensure that each user task requires invoking at least
one benign tool to complete.

Are MCP servers currated per task?  It is known that agentic performance degrades as the total number of MCP tools grows too large, e.g., 40 [1,2].  Is this accounted for?  If so, there should be a curated list of benign/malicious MCP servers per attack that is considerate of the tool limit.  Are all of these tasks single-turn?

> MSB determines whether the agent successfully completes the user task by examining its tool invocation logs.

Is an LLM-as-a-judge used to assess attack success?  If so, please provide details.  Currently, it is not clear how, exactly, either ASR nor PUA are measured.  E.g., if it is simple string matching, then it is difficult for either to be accurately assessed unless only one unique sequence of tool calls solves the user request (and tool invocation order, as well as output, are accounted for).

> By combining user tasks and attack tasks, we constructed 2,000 attack test instances.

How are these vetted, e.g., the feasibility of each attack given provided MCP servers?  E.g., what LLMs were these tasks assessed on initially?  This is a large number of attacks, and thus the verification of each particular attack given the specific attack/task parameters is important to discuss.

It is not entirely clear how, exactly, PUA is measured.  Are there intermediary benign tasks necessary to achieve the malicious attack?  Per individual attack, how many, and if there are more than one, how does this affect PUA?  Could the authors please provide a relevant example or forward reference in Section 5.2.

How much time does the benchmark require?  What are the costs for properietary models (Claude Sonnet 4, GPT-4o-mini)?  What are the infra details for the open-source model runs?

## Minor: adjustment of claims
> the first benchmark for systematically evaluating the security of LLM agents across all stages under MCP-based tool use

This is not the first MCP-specific safety benchmark, and thus these claims require revision (and existing benchmarks require discussion to distinguish MSB's contributions in the context of previous work).  E.g., aside from MCP-Tox, there are SafeMCP, MCIP-Bench, MCP-AttackBench, and MCPSecBench.

## References
[1] Phillip Schmid (Deep Mind). "One Month in MCP: What I Learned the Hard Way." Reddit, r/mcp, www.reddit.com/r/mcp/comments/1mub6g6/one_month_in_mcp_what_i_learned_the_hard_way/. Accessed 31 Oct. 2025.
[2] https://forum.cursor.com/t/tools-limited-to-40-total/67976

### Questions
It is unfortunate that neither GPT-5 nor GPT-4o were assessed in Table 3, particularly since GPT-5 and Claude Sonnet (4.5, but 4 is also pertinent) are safety aligned extensively against prompt injection and related attacks.  Do the authors have any anecdotal information on assessing GPT-5 (or gpt-oss)?

For the heavy use of attack abbreviations in Section 6.1, could the authors add the abbreviations to Table 1 so that the reader can more conveniently map between abbreviations while reading this and later sections?

Recommend moving
> ⊕ denotes the string concatenation.

on line 234 to its first use on line 157.

Line 217:
"the abricated" <- "the fabricated"

"more than 400 malicious tools" <- Can the authors provide the exact number of malicious tools?

"and equips agents with two supporting MCP servers (Protocol (2024); DesktopCommander (2024))" <- Could the authors specify inline that the first mentioned server is the File System Server (one of the canonical MCP servers)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents MCP Security Bench (MSB), a large-scale benchmark designed to systematically evaluate the security of LLM-based agents under the Model Context Protocol (MCP) framework. MCP standardizes how agents discover and call external tools but also introduces new attack surfaces. MSB defines a taxonomy of 12 attack types spanning all stages of the MCP workflow—task planning, tool invocation, response handling, and retrieval—and provides a dynamic evaluation environment running both benign and malicious MCP tools rather than simulations. It introduces a new metric, Net Resilient Performance (NRP), to quantify the trade-off between model performance and robustness. Experiments on nine popular LLM backbones across ten domains and over 2,000 attack instances reveal that stronger models tend to be more vulnerable due to their higher tool-use capability.

### Strengths
- Comprehensive and systematic benchmark: MSB is the first end-to-end evaluation suite specifically tailored for MCP-based agents, covering all stages of the tool-use pipeline with real executable tools instead of synthetic setups. The paper clearly formalizes 12 types of MCP-specific attacks, accompanied by detailed definitions and visual examples, making the benchmark replicable.

- Comprehensive experiments: Evaluation on multiple LLMs (GPT-4o-mini, Claude 4, Gemini 2.5, DeepSeek-V3, etc.) across 10 scenarios provides rich empirical results and insights.

### Weaknesses
- Conceptual ambiguity between LLM security and MCP infrastructure security: Many of the defined attack types (e.g., Name Collision, Preference Manipulation, Out-of-Scope Parameter) exploit weaknesses in the MCP protocol or registry design—such as lack of namespace isolation, schema validation, or tool registration control—rather than the LLM agent’s reasoning or alignment itself. These attacks could be prevented by enforcing protocol-level safeguards rather than improving LLM robustness. As a result, the benchmark partly evaluates system-level vulnerabilities rather than true agent-level security.

- Limited novelty in attack mechanisms: Most attack types (prompt injection, parameter abuse, response poisoning) have been studied in previous works; MSB primarily re-organizes and extends them to the MCP setting.

- Missing discussion of defense strategies:
The paper mainly benchmarks attacks but provides little insight into potential mitigations or defense design within either the agent or MCP layer.

### Questions
- How many test instances are included for each of the 12 attack types?

- Many attacks stem from MCP design flaws rather than model reasoning. Do you see defenses as mainly MCP-level (e.g., schema validation, tool authentication) or LLM-level (robust reasoning and refusal)?

### Soundness
3

### Presentation
3

### Contribution
3
