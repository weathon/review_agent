# VulFinder: A Multi-Agent-Driven Test Generation Framework for Guiding Vulnerability Reachability Analysis

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
Reusing third-party components in the software supply chain (SSC) may introduce risks of vulnerabilities. After disclosing a new third-party component vulnerability, developers need to determine whether the project is affected by the specific vulnerability, which requires vast manpower and resources for assessment. Current approaches mainly rely on dependency-based tools and genetic algorithm-based methods to assess the reachability problem of vulnerabilities in SSC. However, these methods suffer from several issues: they ignore the actual invocation of the vulnerable code, resulting in high false positive rates, are limited to certain vulnerabilities, leading to high false negative rates, and are confined to the Java ecosystem.
To overcome these challenges, we propose VulFinder, a multi-agent driven framework for validating vulnerability reachability. VulFinder begins by using static code analysis tools to construct function call paths between downstream applications and dependency vulnerability APIs. Leveraging a multi-agent mechanism comprising a distillator, discriminator, generator, and validator, VulFinder iteratively generates exploit tests for methods along the call graph, effectively validating vulnerability reachability by executing these tests on downstream applications. By integrating the code comprehension capabilities of large language models (LLMs) with the multi-agent framework, VulFinder addresses the coverage limitations of existing tools, reduces false alarms and missed alarms, and demonstrates robust generalizability across multiple programming languages.
Experiments show that VulFinder achieves 21\% accuracy improvement over the state-of-the-art tool on the Java dataset and also demonstrates robust generalizability on the Python dataset, significantly reducing false positives and false negatives and delivering an average efficiency improvement of more than 1.5×.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes VulFinder, a multi-agent LLM-based system for assessing vulnerability reachability in software supply-chain dependencies. The system first constructs call paths from downstream projects to vulnerable APIs, then uses a distillator–discriminator–generator–validator loop to generate tests that allegedly verify whether the vulnerability is exploitable in practice. Experiments on Java and Python datasets show improvements over prior tools like TRANSFER and VESTA in accuracy and efficiency.

While the problem is important, the paper suffers from issues in presentation clarity, experimental rigor, and dataset transparency. Many methodological details, including how exploit success is judged, how attack surfaces are defined, and how ground truth is established, are not convincingly specified. The results feel anecdotal in places, and the evaluation omits key clarifications needed to trust the claims.

### Strengths
- Tackles an important and timely problem in software supply-chain security.
- Demonstrates cross-language evaluation (Java + Python).
- Multi-agent iterative refinement is a reasonable approach for LLM-guided test generation.
- Shows improvement over two recent baselines in reported metrics.
- Open-sourcing the tool is valuable for the community.

### Weaknesses
1. **Dataset construction and labeling lack clarity.**
The dataset is only partially inherited from prior work and otherwise curated manually, but the paper does not clearly explain how “confirmed reachability” was established.
2. **Attack-surface definition is oversimplified.**
The system treats any method on a call path to a vulnerable API as part of the attack surface. This ignores the distinction between internal calls and true external entrypoints such as CLI inputs or network endpoints. No taint analysis or input-flow tracking is used to ensure attacker control over parameters.
3. **Exploit success criteria are vague and limited to visible effects.**
The Validator primarily checks for observable runtime behavior, such as crashes or exceptions, but the paper does not define detection conditions, thresholds, or instrumentation. The ReDoS example (Figure 4) does not show how “performance degradation” is measured. Silent vulnerabilities (e.g., memory corruption, path traversal without exception) are not handled. Without explicit observability mechanisms, the evaluation favors vulnerabilities with obvious symptoms and may miss subtler classes.
4. **Evaluation reporting is selective and incomplete.**
Efficiency results (Table 3) are demonstrated using only three chosen vulnerabilities rather than reporting median or mean times across the dataset. This makes the performance claims anecdotal rather than statistically grounded.
5. **Presentation quality is below expectations.**
Important elements are poorly explained, such as color coding in Figure 1 and exploit-effect detection in Figure 4. Citation style is inconsistent (\cite vs \citep), and several methodological steps are described imprecisely. Critical concepts—attack surface, reachability definition, success criteria—are not formally defined.

### Questions
1. How exactly was vulnerability reachability “confirmed” in the positive dataset? Especially when the vulnerability does not produce observable side-effect?
2. How is the attack surface defined for downstream modules? How do you distinguish external entrypoints (e.g., main functions, http request handlers, etc.) from arbitrary internal public methods?
3. How does the Validator detect silent vulnerabilities (e.g., memory corruption, logic bugs without exceptions)?
4. For ReDoS (Figure 4), what concrete timeout / CPU usage criteria define “performance degradation”?
5. Why only report generation time for three chosen vulnerabilities? Can you provide dataset-wide averages and variance?
6. How do you handle dynamic behaviors (reflection, SPI, RPC entrypoints) beyond static call-graph matching?
7. Can you provide a breakdown of results by vulnerability category (e.g., DoS vs info-leak vs logic error)? Maybe classify them with CWEs?

### Soundness
2

### Presentation
1

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
The paper introduces VulFinder, an agentic approach that generates exploit tests to determine if reported vulnerable APIs are reachable by the downstream application. VulFinder utilizes static analysis to generate call graphs leading to vulnerable APIs and multiple agents (a distillator, discriminator, generator, and validator) to perform auto-prompting, assessing vulnerability reachability, generating relevant test programs, and validating their effectiveness. VulFinder achieves 21% accuracy improvement over prior work VESTA, while achieves7% accuracy improvement over the baseline TRANSFER. VulFinder is demonstrated to work with both Java and Python, validating its generalizability.

### Strengths
+ A quick and reliable confirmation of whether reported vulnerabilities are affecting downstream applications provides developers with information to make a quick decision to mitigate or eliminate the vulnerability in their software.
+ The additional dataset would be useful for further research.

### Weaknesses
- The “SOTA” VESTA is not as effective as the baseline TRANSFER in the evaluation; however, in their paper, VESTA claims to be much more effective than TRANSFER. The discrepancy is not really explained, which leads to questions about the evaluation process. 
- Additional information is needed for the process of curating the dataset. This would demonstrate the proper procedure of labeling and filtering, which in turn instills confidence in the evaluation result.
- The ablation study could also include an entry for VulFinder without the distillator, the discriminator, and the validator (essentially direct generation from call graphs). This is important to demonstrate the contribution of all additional components and compare VulFinder against a vanilla LLM.
- A qualitative investigation of the mode of failure would help future researchers better direct their efforts and make the study more complete.

### Questions
* Given that  VESTA is worse than TRANSFER,  why is this considered an SOTA approach? Also, in their paper, VESTA evaluation claims a much better effectiveness when compared to TRANSFER. What is the cause of this discrepancy?
* Do you have some insight into some of the cases that VulFinder failed to verify reachability? How would the different components contribute to these modes of failure? Would certain types of failure appear when certain components get removed?
* It seems LLM’s quality does affect the result. Would a more modern and effective LLM in coding tasks, such as Claude Sonnet would provide better results? Would a better foundation model close the gap between VulFinder and vanilla prompting (i.e., generator only)?
* Could you provide more details on how the datasets are built (Python dataset and additional Java entries)? How are the vulnerabilities selected, how are the downstream applications selected, and how are the labels of affected and not affected assigned?

### Soundness
2

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
The paper presents VulFinder, a multi-agent framework designed to more accurately determine whether software projects are affected by vulnerabilities in third-party components. Combining static code analysis with the code understanding abilities of LLMs, VulFinder uses agents to iteratively generate and execute exploit tests that validate vulnerability reachability. Experiments show that VulFinder significantly outperforms existing tools, improving accuracy and efficiency while reducing false positives and false negatives across multiple programming languages.

### Strengths
1. Addresses a well-defined and practical problem of vulnerability reachability analysis. Existing approaches, including dependency-based ones like Github Dependabot and heuristic-based ones like SIEGE and TRANSFER, both have limitations in terms of false positives and/or negatives that the proposed approach aims to overcome.

2. The emprical evaluation is reasonably comprehensive, targeting two constructed datasets one for Java and one for Python, for a total of 25 vulnerabilities across 17 packages and 50 downstream projects.  Baselines include Dependabot, SIEGE, and TRANSFER.

### Weaknesses
1. The presentation of the multi-agent driven test generation framework needs significant improvement. It is hard to understand what each of the different components (distillator, discriminator, generator, validator) does and how they interact. More generally, the paper does not explain what is really challenging about this problem, and what is truly novel about the architecture.

2. I did not see baselines of standalone state-of-the-art LLMs like the latest GPT reasoning model which would also help with better motivating a multi-agent architecture.

3. While the empirical results are promising, with VulFinder (GPT-4o) achieving high accuracy, recall, and F1 scores on the dataset, the paper would be significantly strengthened if the approach was applied on a larger test set in the wild.

4. Another way to convince a skeptical reader of the significance of the challenge being addressed would be to show the length of the call chains discovered. I did not see this metric in the experimental results but I could have overlooked it.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an agent-based system for vulnerability reachability analysis.

### Strengths
The proposed system can perform slightly better than traditional methods. However, deploying such an agent-based system to replace traditional approaches may incur high costs (see questions).

### Weaknesses
- The paper does not clearly demonstrate the benefits of using multi-agent design. The logic and control flow of the multi-agent system appear simple. The proposed method seems achievable with a single react-based agent with proper prompts or existing coding agents (e.g., Claude Code, Codex) given appropriate task descriptions. The authors could strengthen their contribution by comparing their multi-agent design with existing agents.
- The self-constructed benchmark includes only a limited set of tasks, which is insufficient to demonstrate the system's effectiveness or improvement over baselines. In addition, the authors claim cross-language generalizability, but the evaluation only covers two programming languages.

### Questions
What is the cost of running the agents? Can the authors provide a cost comparison between their approach and traditional methods?

### Soundness
2

### Presentation
2

### Contribution
2
