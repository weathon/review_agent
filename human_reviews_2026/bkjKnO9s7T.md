# SCUBA: Salesforce Computer Use Benchmark

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 6, 6

## Abstract
We introduce SCUBA, a benchmark designed to evaluate computer-use agents on customer relationship management (CRM) workflows within the Salesforce platform. SCUBA contains 300 task instances derived from real user interviews, spanning three primary personas—platform administrators, sales representatives, and service agents. The tasks test a range of enterprise-critical abilities, including Enterprise Software UI navigation, data manipulation, workflow automation, information retrieval, and troubleshooting. To ensure realism, SCUBA operates in Salesforce sandbox environments with support for parallel execution and fine-grained evaluation metrics to capture milestone progress. We benchmark a diverse set of agents under both zero-shot and demonstration-augmented settings. We observed huge performance gaps in different agent design paradigm and gaps between the open-source model and the closed-source model. In the zero-shot setting, open-source model powered computer-use agents that have strong performance on related benchmarks like OSWorld only have less than 5\% success rate on SCUBA, while methods built on closed-source models can still have up to 39\% percent task success rate. In the demonstration-augmented settings, task success rates can be improved to 50\% while simultaneously reducing time and costs by 13\% and 16\%, respectively. These findings highlight both the challenges of enterprise tasks automation and the promise of agentic solutions. By offering a realistic benchmark with interpretable evaluation, SCUBA aims to accelerate progress in building reliable computer-use agents for complex business software ecosystems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new benchmark for evaluating AI agents that use computers in only one website. The authors design a set of tasks, a rule-based evaluation system, and several case studies. The work involves notable engineering effort, but the paper lacks clear novelty and convincing evidence of realism.

### Strengths
The benchmark demonstrates notable implementation effort on a real web interface, reflecting practical interest in evaluating agent actions.

The paper clearly describes the task setup and examples, which helps readers understand the benchmark design.

The focus on web-based computer use provides a concrete starting point that could be extended to broader environments in the future.

### Weaknesses
The claim of “realistic” tasks is not well supported. Many prior benchmarks also make this claim without proper evidence.
Several important analyses and examples remain in the appendix. Some should be moved to the main text to support the main arguments.

The evaluation method is underexplained. The paper mentions rule-based evaluators but gives no clear description, pseudo-code, or examples. In Figure 2, the element ground_truth_dict appears, but its role in evaluation is unclear.

The rule-based evaluation raises safety concerns for automatic execution. It is unclear how potential side effects (e.g., unsafe file operations) are prevented.

The “Q&A troubleshooting” tasks look inconsistent with the main task types. Asking models to answer “yes or no” is unstable and not representative of real computer-use behavior.

The example with a 31-step task does not seem realistic. Long tasks like this are unreliable and should not be split into smaller tasks without showing how performance transfers.

### Questions
With rule-based evaluation, how to control side effects of the auto-execution. E.g. The model rm -rf all files before clicks a correct button? (maybe this is an extreme case)

Why use Q&A for troubleshooting as an additional type of agent execution？Maybe it should use some execution to find that why "yes" or "no" but not directly "only answer yes or no"?

Strong models like GPT and Claude pass near 50% of the tests, how to prove that the reasons of fails are the ability of the model or agent, but no an unreasonable task setting?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents SCUBA, a benchmark for evaluating computer-use agents on enterprise-grade CRM workflows within the Salesforce platform. Unlike prior browser-oriented or service-specific benchmarks, SCUBA emphasizes realism through sandboxed Salesforce environments, task instances sourced from user interviews across administrator/sales/service personas, and fine-grained rule-based evaluation with milestone scores. The authors benchmark a wide range of representative agents under zero-shot and demonstration-augmented settings.

### Strengths
Tasks are grounded in real enterprise workflows and executed in genuine Salesforce sandboxes rather than synthetic mockups, addressing a major gap left by OSWorld/WebArena-like benchmarks.

The inclusion of process rewards, latency and cost metrics, and rule-based evaluators provides more diagnostic insight than binary success metrics commonly used in prior work.

### Weaknesses
1. As acknowledged, benchmark stability and long-term reproducibility depend on third-party platform updates, which may undermine comparability over time. Is it possible to provide a version-frozen evaluation environment to improve the reusability of the results?

2. The paper reports cost/time for agent inference, but does not quantify the annotation cost or the engineering overhead associated with constructing and maintaining SCUBA.

3. The reported improvements (e.g., from demonstration augmentation) are presented as absolute numbers. Could the authors conduct a statistical analysis to verify the robustness of these gains?

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SCUBA (Salesforce Computer Use Benchmark), a new benchmark designed to evaluate computer-use agents on Customer Relationship Management workflows within the Salesforce platform. SCUBA contains 300 task instances derived from real user interviews, covering three primary personas: platform administrators, sales representatives, and service agents. It tests a wide range of enterprise-critical abilities, including UI navigation, data manipulation, workflow automation, information retrieval, and troubleshooting.

### Strengths
1. The benchmark runs in live Salesforce sandbox environments and uses fine-grained evaluation metrics that capture milestone progress, not just final success. This indicate some progress, as typical benchmarks like OSWorld[1] and ScienceBoard[2] only provide a 0/1 final success.
2. Tasks are sourced from real user interviews, ensuring practical relevance and representativeness. The evaluation is conducted within a live Salesforce sandbox env, offering higher fidelity than simulated platforms.
3. The study systematically evaluates both zero-shot performance and a "demonstration-augmented" setting. It empirically validates that providing demonstrations is an effective strategy.

[1] OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments https://arxiv.org/abs/2404.07972

[2] ScienceBoard: Evaluating Multimodal Autonomous Agents in Realistic Scientific Workflows https://arxiv.org/abs/2505.19897

### Weaknesses
1. While advances have been made in the eval metrics, the SCUBA's infrastructure is wholly dependent on the Salesforce "sandbox" environment. This constitutes a significant threat to the long-term validity and reproducibility of the benchmark and imposes a heavy burden on the maintainers.
2. The current scale is relatively limited for capturing the vast complexity of an enterprise platform like Salesforce. Furthermore, existing works like AndroidWorld[3] and ScienceBoard[2] commonly employ randomized parameters to initialize the scenarios faced by the scaled agent tasks.
3. The task distribution is heavily skewed (57% for the 'admin'), which may limit the benchmark's ability to evaluate agent performance comprehensively across all core business roles.
4. For the experimens, the claim “The high task success rates seen in browser-use agents can be attributed to the combination of the stronger planning ability of the foundation models, richer observation space, and the more efficient action space design used in browser-use agents". This comparison between "Browser-Use" and "Computer-Use" agents seems to be methodologically unbalanced.
    a. The browser agents (or cuas leveraging a11ytrees [6]) receive a more structured observation space (SOM + DOM texts), which inherently poses no challenge for grounding.
    b. In contrast, other computer-using agents (e.g., SeeClick[3], OS-Atlas[4], UI-TARS[5]) take raw screenshots, which necessitates explicit grounding.
5. Minor: SCUBA is deeply coupled with Salesforce. While this provides depth in specific business scenarios, it sacrifices breadth. Conclusions drawn from this benchmark regarding UI nav and DOM-related stuffs may be difficult to generalize directly to other platforms.

[3] AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents

[4] SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents, ACL 24

[5] OS-Atlas: A Foundation Action Model For Generalist GUI Agents

[6] UI-TARS: Pioneering Automated GUI Interaction with Native Agents

[7] OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis https://arxiv.org/abs/2412.19723

### Questions
1. The paper highlights a dramatic performance drop for agents moving from OSWorld to SCUBA, attributing it to planning / grounding capabilities. This analysis could be deeper. Does this failure simply indicate a domain shift, or does it suggest that benchmarks for general desktop applications (like OSWorld) are poor proxies for the challenges of complex, web-based, stateful enterprise applications?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SCUBA, a new benchmark designed to evaluate computer-use agents on realistic customer relationship management (CRM) workflows within the Salesforce platform. It contains 300 task instances derived from real user interviews, covering roles like administrators, sales representatives, and service agents. SCUBA features a sandbox-based Salesforce environment, rule-based milestone evaluation, and demonstration-augmented evaluation for assessing enterprise agents' performance. The experiments benchmark nine agents (browser-use and computer-use paradigms) under zero-shot and demonstration-augmented conditions, revealing substantial performance gaps between open- and closed-source models, as well as between paradigms.

### Strengths
- Timely and highly relevant: The paper tackles an important gap—evaluating GUI-based LLM agents on enterprise-grade workflows, rather than consumer or synthetic tasks.
- Strong realism and comprehensiveness: SCUBA operates in real Salesforce sandbox environments, using realistic personas and tasks drawn from actual user interviews.
- Clear methodological contributions: Task construction pipeline (template → paraphrase → initialization → evaluation) is well-detailed and reproducible.Fine-grained, interpretable milestone scoring system provides richer signals than binary success rates.

### Weaknesses
- Insufficient qualitative error analysis: While quantitative metrics are extensive, more detailed examples of failure cases (e.g., grounding errors, navigation dead ends) would better illuminate where agents struggle.
- Demonstration quality underexplored: The authors note some demonstrations are suboptimal (e.g., missing shortcuts), but the process for ensuring demonstration consistency and coverage could be better justified.

### Questions
How do the authors plan to maintain task validity when Salesforce updates its sandbox environment or API schema?
What percentage of failures stem from grounding vs. reasoning vs. action space limitations? Could process rewards help isolate these error modes quantitatively?
Will SCUBA support online fine-tuning or reinforcement learning in the sandbox (e.g., for ComputerRL-style training)?
Given the availability of fine-grained rewards and high-quality labeled trajectories, why did the authors not conduct fine-tuning or reinforcement learning on open-source models to build a stronger GUI agent baseline?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
SCUBA establishes a comprehensive and enterprise-grounded framework for evaluating computer-use agents on real CRM workflows. It captures the full spectrum of enterprise challenges (UI navigation, data manipulation, workflow automation, and troubleshooting) that prior benchmarks such as OSWorld and WorkArena fail to encompass.

### Strengths
- The overall presentation of the benchmark—covering dataset construction, filtering, and annotation—is exceptionally coherent and methodically executed. The benchmark is clearly grounded in realistic business workflows and real-world use cases.

- The paper demonstrates clarity in explaining implementation details, e.g., the description and justification of the chosen inference engine.

- I especially appreciate that the benchmark provides rule-based evaluators with milestone (process) rewards, enabling multi-dimensional performance assessment.

### Weaknesses
Personally, I find this benchmark meaningful and thoughtfully described. My comments here are merely minor suggestions: (1) The authors might consider evaluating a broader range of open-source models [1,2,3]. It would also be worthwhile to update UI-TARS to the latest UI-TARS-2B version. I understand this may not have been feasible before the ICLR submission deadline, but such updates could be included in the camera-ready version.

---

[1] Aria-UI: Visual Grounding for GUI Instructions

[2] Mobile-Agent-v3: Fundamental Agents for GUI Automation

[3] MagicGUI: A Foundational Mobile GUI Agent with Scalable Data Pipeline and Reinforcement Fine-tuning

### Questions
- Line 261 mentions the use of ChatGPT — could the authors specify which version was employed?

 - Line 258 states that each task template is filled five times. \Were these fillings done manually or LLM-generated? Additionally, in the example where “one query might require extra scrolling or alphabet-based searching to locate the target data, while another might demand keeping track of previously deselected checkboxes,” were these variations manually created?

- I am uncertain how demonstration examples were obtained for each test sample. Were they directly taken from the two demonstrations associated with the corresponding task template? If so, how should we interpret Line 811, which states that “two samples are tested with exact demonstrations and three samples are tested with similar demonstrations”?

- Since the authors have shown that adding human demonstrations to every test sample is not always optimal, I am curious about potential gains from incorporating simple memory mechanism, similar to the agent memory module implemented in MMINA.

### Soundness
3

### Presentation
3

### Contribution
4
