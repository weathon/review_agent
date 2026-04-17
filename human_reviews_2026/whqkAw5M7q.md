# Computer-Use Agent Frameworks Can Expose Realistic Risks Through Tactics, Techniques, and Procedures

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Computer-use agent (CUA) frameworks, powered by large language models (LLMs) or multimodal LLMs (MLLMs), are rapidly maturing as assistants that can perceive context, reason, and act directly within software environments. Among their most critical applications is operating system (OS) control.  As CUAs in the OS domain become increasingly embedded in daily operations, it is imperative to examine their real-world security implications, specifically whether CUAs can be misused to perform realistic, security-relevant attacks. Existing works exhibit four major limitations: Missing attacker-knowledge model on tactics, techniques, and procedures~(TTP), Incomplete coverage for end-to-end kill chains, unrealistic environment without multi-host and encrypted user credentials, and unreliable judgment dependent on LLM-as-a-Judge. To address these gaps, we propose AdvCUA, the first benchmark aligned with real-world TTPs in MITRE ATT\&CK Enterprise Matrix, which comprises 140 tasks, including 40 direct malicious tasks, 74 TTP-based malicious tasks, and 26 end-to-end kill chains, systematically evaluates CUAs under a realistic enterprise OS security threat in a multi-host environment sandbox by hard-coded evaluation. We evaluate the existing five mainstream \CUAs, including ReAct, AutoGPT, Gemini CLI, Cursor CLI, and Cursor IDE based on 8 foundation LLMs. On TTP tasks, Cursor CLI achieves the highest average ASR at 69.59\%, notably surpassing ReAct-based CUA at 52.29\% and Cursor IDE at 51.66\%. For end-to-end kill chain tasks, Cursor IDE attains the highest average ASR at 34.62\%, followed by Cursor CLI at 26.93\% and ReAct-based CUA at 23.37\% on all evaluated LLMs. The results demonstrate that current frontier CUAs do not adequately cover OS security-centric threats. These capabilities of CUAs reduce dependence on custom malware and deep domain expertise, enabling even inexperienced attackers to mount complex enterprise intrusions, which raises social concern about the responsibility and security of CUAs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a benchmark for cybersecurity capabilities of computer-use agents (CUAs) operating on the terminal. It describes the AI-assisted process used to construct a dataset of tasks based on MITRE ATT&CK framework and their execution in a simulated environment comprising 3 hosts: a workstation running the CUA, an administration server, and a web server. The paper reports the benchmark results for 5 different terminal-based CUAs used in combination with 8 LLMs when queried directly to complete each task and when the task is inserted into a jailbreak template intended to bypass the LLM's built-in safety guardrails. The results show across the board a high bypass success rate (indicating the CUA's willingness to perform the task) and attack success rate, measured via deterministic, hard-coded evaluation methods rather than an LLM-as-judge. The authors conclude that this raises concerns about the dual-usage of CUAs, as helpful penetration-testing tools for security practitioners but also enabling unsophisticated adversaries to automate and scale attacks.

### Strengths
- The selection of tasks is grounded in the well-established MITRE ATT&CK framework and is representative of a broad range of threats.
- The evaluation covers a good selection of models and CUAs, including open-source and commercial agent scaffoldings, and open-weights and proprietary models.

### Weaknesses
- The method used to construct tasks biases the benchmark towards higher BSR and ASR (the possibility of bias is acknowledged in Appendix J)
- The results on commercial CUAs where obtained while intentionally letting the agent know it was being tested in a sandboxed environment (also trivially revealed by the absence of `systemd` and other features of the containerized environment), biasing the results towards higher BSR and ASR because of the lower change of triggering refusals.
- The environment where CUAs run, though not as trivial as a single host, cannot be considered a faithful simulation of an enterprise environment. Frontier model capabilities are often evaluated on much more sophisticated cyber ranges, with CUAs extended with an arsenal of tools.
- No reasoning models included in the evaluation, only explicit reasoning as part of ReAct.
- Does not discuss or address sandbagging (see https://arxiv.org/abs/2406.07358) or fake-alignment issues (https://arxiv.org/abs/2412.14093).

### Questions
- In Table 2, the results for Claude Opus 4.1 on TTP and Direct tasks are not shown. Why is that?
- You say in Section 4.1 that you leveraged GPT-4o and Claude Sonnet 4 to assist in decomposing tasks and verifying where this decomposition aligned with the strategy chosen by human experts. If you kept only tasks where this alignment was observed, would not this bias task selection towards tasks that these models find easier to decompose and complete? Did you include any tasks were GPT-4o and Claude Sonnet 4 decompositions did not align with human experts but could have led to the completion of the task?
- Appendix H.1 discusses making CUAs aware of security tools installed in the system, which they would still need to know how to use appropriately. Have you considered leveraging LLMs tool-calling capabilities to extend CUAs with other tools beyond executing terminal commands (e.g. Ghidra MCP server)? What about web search and (adequately constrained) web access to let the agent access documentation of available tools, install unavailable packages, etc.? These are easy extensions (even built into the commercial CUAs you tested) that unsophisticated attackers may try.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles a critical and timely issue: the security risks of OS-level Computer-Use Agents (CUAs). Its primary contribution, AdvCUA, is a novel benchmark that meaningfully advances the field. The benchmark's key strength lies in its alignment with the MITRE ATT&CK framework, enabling a focus on realistic Tactics, Techniques, and Procedures (TTPs) and end-to-end kill chains. This approach moves significantly beyond evaluating simple, direct malicious prompts. The study's findings are important as they highlight a significant gap in current CUA safety alignment.

### Strengths
- Proposes the first benchmark (AdvCUA) aligned with the MITRE ATT&CK framework for evaluating computer-use agent security.
- Builds a realistic, reproducible multi-host enterprise environment using Docker microsandboxes.

### Weaknesses
Weakness:

1. The scope of the term Computer-Use Agent (CUA) in this paper appears overly broad. While the title and framing suggest inclusion of the full spectrum of agents capable of operating computers—such as GUI-based agents (e.g., WebVoyager, GPT-4V-based GUI controllers, or Claude Code)—the experiments exclusively evaluate command-line (CLI) agents interacting through terminal environments. To avoid overgeneralization and better reflect the actual experimental setup, I suggest narrowing the terminology throughout the paper. A more precise term such as “Command-Line Computer-Use Agents (CLI-CUAs)” or “Shell-Based CUAs” would improve clarity and accurately represent the evaluated systems.
2. The paper does not evaluate Claude Code or OpenAI Codex, which are among the most widely used code agents in real-world development environments. Since both represent typical computer-use agents capable of executing code and interacting with local systems, their exclusion limits the practical relevance of the evaluation.
3. The paper should also reference recent agent-security benchmarks, including Agent Security Bench (ASB) [Zhang et al., 2025] and AgentDojo [Kumar et al., 2025]. ASB provides a broad framework for evaluating adversarial attacks and defenses in LLM-based agents, while AgentDojo offers a dynamic setup for testing prompt-injection attacks and mitigations. 
4. Although the paper claims that the “hard-coded evaluation” framework improves reliability over LLM-as-a-Judge, the Match protocol effectively serves as a proxy-level validation that checks the agent’s intent rather than the actual impact of an attack. Because the indicator lists are manually curated, this approach may introduce false positives and false negatives, especially in cases where the sandbox prevents direct execution. The paper could explain this limitation more clearly—how the matching thresholds are defined, how Match results are integrated with Trigger/Probe/Verify, and to what extent this affects the reported ASR.
5. The paper states coverage of 10 tactics and 77 techniques, but the per-technique descriptions are currently too terse. I recommend expanding Table 5 / Appendix E into a structured table with more attacking details.
6. Although the quantitative experiments are comprehensive, the paper lacks case-based qualitative analysis that could deepen understanding. Such case studies could be summarized using short “Takeaway” paragraphs inside the main text to highlight practical insights about both attack patterns and potential defense signals.

References:

[1] @inproceedings{
debenedetti2024agentdojo,
title={AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for {LLM} Agents},
author={Edoardo Debenedetti and Jie Zhang and Mislav Balunovic and Luca Beurer-Kellner and Marc Fischer and Florian Tram{\`e}r},
booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2024},
url={https://openreview.net/forum?id=m1YYAQjO3w}
}

[2] @inproceedings{
zhang2025agent,
title={Agent Security Bench ({ASB}): Formalizing and Benchmarking Attacks and Defenses in {LLM}-based Agents},
author={Hanrong Zhang and Jingyuan Huang and Kai Mei and Yifei Yao and Zhenting Wang and Chenlu Zhan and Hongwei Wang and Yongfeng Zhang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=V4y0CpX4hK}
}

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a high-fidelity benchmark, AdvCUA, designed to simulate realistic operating system (OS) security threats within a multi-host sandboxed environment. The evaluation rigorously assesses five mainstream Computer-Use Agent (CUA) frameworks driven by eight foundation Large Language Models (LLMs).

The major contributions lie in its methodological rigor and realistic threat modeling:

1. **Innovative Evaluation**: AdvCUA pioneers the use of hard-coded verification protocols to determine attack success, definitively moving beyond the unreliable "LLM-as-a-Judge" paradigm.

2. **Realistic Threat Landscape**: The dataset is meticulously sourced and aligned with real-world Tactics, Techniques, and Procedures (TTPs) documented in the MITRE ATT&CK Enterprise Matrix.

3. **Key Findings**: The results reveal that CUAs achieve a significantly higher Attack Success Rate (ASR) on TTP-based malicious tasks compared to direct malicious tasks. Crucially, existing CUAs demonstrate the capability to execute End-to-End Kill Chains, exposing substantial and immediate real-world threats.

Despite its strengths, the work presents two areas for future development:
1. **Platform Scope Limitations**: The current benchmark is constrained to Linux (specifically terminal interaction). Although this covers fundamental enterprise environments, expanding coverage to include Windows and macOS platforms, as well as evaluating GUI-based agents, is necessary to address critical uncovered attack surfaces (as acknowledged in the paper's limitations).

2. **Model Discrepancy Analysis**: The vast differences in performance observed across various foundation models—particularly the low ASR/BSR of certain models (e.g., Claude series)—cannot be entirely resolved by the provided macro-level explanations (e.g., general safety alignment). While supplementary examples are mentioned in the appendix, the paper lacks detailed, step-by-step comparative reasoning trajectories to convincingly support these large-scale disparities. Further investigation into the specific mechanisms of failure (either through inadequate capability or robust refusal logic) is warranted.

**Overall Assessment**: This benchmark is a robust and essential piece of research, providing comprehensive coverage, thorough verification, and a reliable methodology for assessing CUA security.

### Strengths
1. **TTP-Based Realism**: The benchmark leverages the MITRE ATT&CK framework, shifting tasks from simplistic, "toy" instructions (e.g., "delete all user files") to complex procedures that mimic genuine adversary behavior (e.g., "deploy a port-knocking activated backdoor").

2. **End-to-End Kill Chains**: It incorporates 26 complete end-to-end attack chains , which were largely absent in prior work , enabling the assessment of the CUA's combined reasoning and execution capabilities.

3. **Rigorous Experimental Validation**：The experimental design of this paper is rigorous and its evaluation is thorough. It systematically assessed five mainstream CUA frameworks (including ReAct, AutoGPT, and industry-leading products) and eight advanced foundation LLMs within a multi-host sandboxed environment simulating a realistic enterprise setup. The evaluation utilized a comprehensive suite of 140 tasks based on the MITRE ATT&CK framework , including 74 TTP-based tasks and 26 end-to-end kill chains. Crucially, the adoption of hard-coded verification ensured the reliability and reproducibility of the results.

4. **Defense Shortfalls**: Furthermore, the evaluation reveals that existing input-level defenses, such as LLaMA Guard 4 and the OpenAI Moderation API, are generally ineffective at preventing TTP-based requests, highlighting a critical and under-recognized gap in current safety alignment strategies.

### Weaknesses
The work presents two areas for future development:

1. **Platform Scope Limitations**: The current benchmark is constrained to Linux (specifically terminal interaction). Although this covers fundamental enterprise environments, expanding coverage to include Windows and macOS platforms, as well as evaluating GUI-based agents, is necessary to address critical uncovered attack surfaces (as acknowledged in the paper's limitations).

2. **Model Discrepancy Analysis**: The vast differences in performance observed across various foundation models—particularly the low ASR/BSR of certain models (e.g., Claude series)—cannot be entirely resolved by the provided macro-level explanations (e.g., general safety alignment). While supplementary examples are mentioned in the appendix, the paper lacks detailed, step-by-step comparative reasoning trajectories to convincingly support these large-scale disparities. Further investigation into the specific mechanisms of failure (either through inadequate capability or robust refusal logic) is warranted.

### Questions
Same with the weakness above.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a benchmark that measure the real-world safety risks (e.g. priveledge escalation) of frontier CUAs when being misused by malicious users. To create the taks, the authors develop a human-in-the-loop pipeline: human experts collaborate with AI to decompse those complex malicious tasks into multiple simpler, and seemingly benign subtasks. They evaluate multiple frontier CUAs, even including Cursor CLI and Cursor IDE. Results show that current CUAs already have a decent ASR on these complex and realistic safety tasks.

### Strengths
1. They build a realisitc CUA benchmark that goes beyond previous toy malicious queries (e.g. send user_password.txt), and study it under realistic multi-host environments.
2. Evaluations are comprehensive: not only include ReAct, AutoGPT-based CUAs, also include Cursor IDE and Cursor CLI --- which are popular frameworks that are likely to be used in misusing CUAs.
3. Collaborating with human experts to create high-quality task decomposition.

### Weaknesses
1. some claims are not well supported. For example, he last sentence of the abstract said:
> These capabilities of CUAs reduce dependence on custom malware and deep domain expertise, enabling even inexperienced attackers to mount complex enterprise intrusions.

Maybe i missed something but I don't find any results in the current paper verifies that inexperienced attackers can use CUAs to perform complex misuse. The task decomposition in this paper was also performed by experts and AI.

2. the error anaysis section makes me worried about the current CUA implementation. for exmaple
> AutoGPT failures were highly concentrated: 80% were “plan only, no execution,” where the CUA produced a detailed plan but immediately called finish without issuing any commands, and the remaining 20% were due to tool invocation errors.

I think these two types of errors can be largely mitigated by improving the agentic scaffolding.

3. The expert-designed task decompsition does not effectively reduce refusal in frontier models (e.g. Claude 4 often rejects all 100% queries). Therefore, this benchmark might already be (nearly) saturated by these moldes.

### Questions
see questions

### Soundness
3

### Presentation
2

### Contribution
3
