# Comparing AI Agents to Cybersecurity Professionals in Real-World Penetration Testing

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 8

## Abstract
We present the first comprehensive evaluation of AI agents against human cybersecurity professionals in a live enterprise environment. We evaluate ten cybersecurity professionals alongside six existing AI agents and ARTEMIS, our new agent scaffold, on a large university network consisting of $\sim$8,000 hosts across 12 subnets. ARTEMIS is a multi-agent framework featuring dynamic prompt generation, arbitrary sub-agents, and automatic vulnerability triaging. In our comparative study, ARTEMIS placed second overall, discovering 9 valid vulnerabilities with an 82\% valid submission rate and outperforming 9 of 10 human participants. While existing scaffolds such as Codex and CyAgent underperformed relative to most human participants, ARTEMIS demonstrated technical sophistication and submission quality comparable to the strongest participants. AI agents offer advantages in systematic enumeration, parallel exploitation, and cost---certain ARTEMIS variants cost $18/hour versus $60/hour for professional penetration testers. We also identify key capability gaps: AI agents exhibit higher false-positive rates and struggle with GUI-based tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This submission reports on the results of an experiment comparing the performance of human experts and LLM-based agent systems on discovering security vulnerabilities in a live real-world environment. The experiment was conducted with IRB approval on a university’s public and private networks, over 12 subnets (7 public, 5 private) comprising ~8K hosts, primarily Unix-based systems, but also including IoT devices, Windows hosts and embedded systems. 

For the study, the authors implemented a new multi-agent system, ARTEMIS, deployed in 2 configurations, and compared it against:

 - 10 cybersecurity professionals, recruited to commit at least 10 hours of work. 
 - 2 general purpose agent systems: OpenAI Codex with GPT-5 and Claude Code with Sonnet 4.
 - 3 existing specialist agent systems: CyAgent with GPT-5 and Claude Sonnet 4, MAPTA and Incalmo with GPT-5.
 
All agent systems ran for 16 hours, though effectively only 2 existing systems (Codex and CyAgent) were evaluated. Claude Code and MAPTA refused to perform the task while Incalmo was unable to make any progress. Overall, 49 vulnerabilities were detected (and responsibly disclosed) during the experiment.

The results, summarized in Table 1, show that ARTEMIS performed competitively against human experts in terms of number, complexity, and severity of vulnerabilities discovered, ranked 2nd overall according to a custom scoring system, but creating many more false positives despite using a triaging agent to gate submissions. The authors argue that the results of the study provide evidence that multi-agent systems like ARTEMIS are already cost effective compared to professionals, with ARTEMIS most costly configuration (the one that ranked 2nd) operating at USD 59/h compared to an average compensation of USD 60/h for the average professional in the US.

### Strengths
- Presents a significant study evaluating the cost-effectiveness for vulnerability research of agent systems compared to cybersecurity professionals in a live system under realistic conditions.
- Describes the design of a new multi-agent system capable of identifying and reporting security vulnerabilities in live systems at a fraction of the cost of professionals.
- Qualitatively evaluates and compares the strategies followed by professionals and agent systems, providing insights into shortcomings of agent systems and how they complement human experts.

### Weaknesses
- 3 of the 5 existing systems evaluated (Claude Code, MAPTA and Incalmo) were unable to perform the task given, but the manuscript does not discuss in adequate depth the reasons why. It is unclear whether the authors made a reasonable effort to adapt their scaffolding, prompts and tools to better suit the task. This may have biased the results in favor of ARTEMIS.
- Not enough context provided to compare the results presented to other studies evaluating cybersecurity capabilities of current agent systems. For instance, David & Gervais (2025) report strong results evaluating MAPTA on the XBOW validation benchmark. It is unclear how ARTEMIS would perform on existing benchmarks, outside of the carefully framed task of the study. Same for Incalmo.
- Cost comparisons do not take into account the cost of triaging vulnerabilities post-submission. During the study, most human participants submitted only valid findings, while agent systems including ARTEMIS submitted between 18 and 45% invalid findings.

## Typos / Minor comments

 - FPR in abstract. Reported 18-43%, but Table 1 shows one ARTEMIS configuration having 45% invalid submissions (and the other configuration 18%), so isn't it 18-45%?
- page 4, typo "Table 3" should refer to "Table 2".
- page 8, typo "a going single step further" should read "going a single step further"
- There is a lot of redundancy between Table 1, Table 2, and Figure 3. In fact, Table 2 is entirely subsumed by Figure 3. Consider consolidating the presentation of results to make space for additional technical details of the design of ARTEMIS from Appendix A.
- There is not much information in Table 3. The first paragraph of Section 5 says it all, and is necessary to understand the meaning of columns in the table. Consider moving it to the Appendix and instead include more details from Appendix A.
- Pleas spell out TTP in the conclusion, which may be unfamiliar for some readers.

### Questions
1. Although you obtained IRB approval to conduct experiments in a live university environment, I am concerned about how well you represented the risks associated with using autonomous agents. These systems are explicitly asked to find security vulnerabilities, have access to the public web, and use models susceptible to hallucinations and prompt injection attacks. The only guardrail stopping the agents from misusing their capabilities seems to be a set of constraints in their system prompt (Appendix I). I could not find any mention that the agents' operated under human oversight to approve potentially dangerous actions. Can you assert that you represented these risks fairly to your IRB and explain what additional measures (if any) you took to impose hard constraints on the agents' actions?
2. Which models were used in the ensemble of supervisor models in the A2 configuration of ARTEMIS?
3. Did you compensate the cybersecurity professionals recruited for the experiment for their work? Since you mention the $60/h rate for the average cybersecurity professional, how much did you pay them?
4. Did you evaluate ARTEMIS on any existing benchmarks besides the environment of your experiments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents an ambitious effort to benchmark the performance of autonomous AI agents against human cybersecurity professionals in a live, complex enterprise environment (a large university network). The evaluation compares six existing open-source agents against ten human pen-testers, alongside the authors' newly proposed multi-agent scaffold, ARTEMIS, which utilizes dynamic task prompt generation and arbitrary sub-agent instantiation. The primary contribution is the establishment of a real-world, high-stakes evaluation framework for LLM-based agents in the domain of offensive security.

### Strengths
1. The core strength is the commitment to a live enterprise environment (8K hosts, 12 subnets). This is a crucial step forward from synthetic environments and provides uniquely valuable, though difficult to validate, insights into autonomous agent performance in a complex system.

2. The direct comparison against a team of 10 human cybersecurity professionals provides a tangible, high-quality benchmark for AI capabilities.

3. Primary novelty in the ARTEMIS methodology is the successful implementation of a multi-agent system on a live enterprise target.

### Weaknesses
1. The use of a "live enterprise environment" makes the core evaluation non-reproducible.

2. The paper needs to clearly articulate the technical novelty of ARTEMIS in its agent-planning module or tool-use orchestration that distinguishes it from existing multi-agent systems.

3. The authors filed to provide error analysis on why ARTEMIS fails.

4. The authors did not compare effect of various LLMs when used with ARTEMIS, a good comparison of proprietary vs open-sourced models would add more value to the paper.

### Questions
1. How was the total time budget for the AI agents determined, and how did this compare to the human professionals? 

2. Were the agents limited to the same network footprint or information access as the human testers?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
A live comparison study pitting 10 real pen-testers vs. various agentiic frameworks, including the proposed ARTEMIS framework.
8K hosts, 7 public and 5 VPN subnets.
The proposed framework had the best automated performance.
Finds more vulnerabilities than some human pentesters, however the false positive rate is higher than for all human pentesters who have little to no false positives.

LLMs used are GPT-5 and Sonnet 4.

Metric is based on a mix of technical complexity and criticality of the found vulnerability.
I'm not sure why there is penalty for "verifying" but not "exploiting" a penalty?

Artemis architecture: 1 supervisor + unlimited subagents. Task list and note taking. Vulnerabilities are verified to be reproducible and deduplicated.

The authors also validate that ARTEMIS can find vulnerabilities found by other participants, given enough hints.
Advantage vs human: parallel search
Disadvantage vs human: using GUIs

### Strengths
- Great comparison between human vs. AI capability
- Experiment in live setting
- Strong results for their proposed framework vs. existing coding frameworks including low cost.
- Open-sourced

### Weaknesses
- The timeframe is short, as noted by the authors, and may not be representative of a typical pentesting engagement timeframe.
- AI agent reports vulnerabilities

### Questions
I'm not sure i understand why penalty if vulnerability is only verified but not exploited. What does "verify" mean exactly here?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a real-world study comparing AI agents with human cybersecurity professionals on a live university network (about 8,000 hosts across 12 subnets). Ten professionals are evaluated alongside five existing agent scaffolds: Codex, Claude Code, CyAgent (CS and CG), Incalmo, and MAPTA, as well as the authors’ own multi-agent framework, ARTEMIS. Only Codex and the two CyAgent variants produced valid findings, while ARTEMIS (two configurations, A1 and A2, using different LLM setups) achieved the best overall performance among AI systems. ARTEMIS A2 ranked second overall, and outperformed 9 of 10 human participants, with 11 findings and 82 percent validity. The paper introduces a unified scoring framework that combines technical complexity and business impact, maps discovered techniques to the MITRE ATT&CK matrix, and includes elicitation trials to test missed vulnerabilities. It also presents a cost analysis suggesting a favorable cost per hour compared to human penetration testers, making the study both technically and economically insightful.

### Strengths
1. Claimed to be the first live comparison of AI agents versus professionals in an enterprise environment (a large university CS network), closely matching real-world penetration testing practice. This live comparison is significant compared to curated internet benchmarks, which models may have unintentionally trained on.

2. The proposed ARTEMIS shows impressive performance, outperforming almost all of the human participants. The multi-agent design is well motivated, with dynamic prompt generation and session/context management thoughtfully engineered. The cost analysis, showing that the agent operates at significantly lower cost than humans while maintaining strong performance, demonstrates the practical potential for automating this task.

3. The methodology is transparent and emphasizes a strong ethical approach in a live environment. IRB approval, VDP compliance, and a well-documented triage process minimize risks to the target.

4. The unified scoring metric, which combines detection and exploit complexity with severity weighting, provides more meaningful insights by capturing both the sophistication and impact of vulnerabilities, rather than relying on simple counts.

5. The paper provides useful qualitative contrasts, showing that agents excel at systematic enumeration and parallelism but tend to over-submit lower-severity or false-positive findings. The timeline plot (Fig. 1) and incidence distributions (Fig. 3) are informative and clearly presented.

### Weaknesses
1. The runtime and evaluation budget between human participants and AI agents are not strictly matched. Humans were asked to work for at least 10 hours, while ARTEMIS was allotted 16 hours. This difference makes the leaderboard comparison less fair and should be normalized or clearly justified. 

2. The paper reports totals and percentages but lacks deeper statistical analysis such as variance or confidence intervals. Without statistical treatment, it is difficult to judge whether performance differences are statistically significant or within normal variation. 

3. The claimed advantages of ARTEMIS (dynamic prompts, unlimited sub-agents, and the triage module) are not isolated through ablation studies. It remains unclear how much each component contributes to the overall improvement. 

4. The paper does not specify whether the agents used a standardized virtual environment (such as Kali Linux or a containerized lab). Tool availability is described conceptually but lacks implementation details about the runtime setup, which limits reproducibility. 

5. It would be beneficial to evaluate ARTEMIS on existing CTF-style benchmarks such as CyBench or NYU-CTF to better understand its performance in standardized settings and how its findings correlate with existing benchmarks.

### Questions
1. How does the system handle model safeguards or refusals, since some LLMs may reject offensive or security-related instructions? Please clarify how the agents were able to perform penetration testing tasks without manual intervention or policy violations.

2. What is the typical and maximum number of sub-agents deployed during execution, and how does this relate to the performance advantage claimed for the proposed framework? The paper states that the agent can spawn unlimited sub-agents, but what was the actual peak number observed during the experiments?

3. What is the detailed cost breakdown for each major component (such as the supervisor, sub-agents, and triage module) to better understand which parts contribute most to the total operational cost?

4. Please clarify if any points in the weaknesses section were misunderstood. I am open to revising my evaluation, as this work presents valuable insights and potential impact for the research community.

### Soundness
3

### Presentation
3

### Contribution
4
