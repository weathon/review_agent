# Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
AI agents powered by large language models (LLMs) are being deployed at scale, yet we lack a systematic understanding of how the choice of backbone LLM affects agent security.
The non-deterministic sequential nature of AI agents complicates security modeling, while the integration of traditional software with AI components entangles novel LLM vulnerabilities with conventional security risks.
Existing frameworks only partially address these challenges as they either capture specific vulnerabilities only or require modeling of complete agents.
To address these limitations, we introduce threat snapshots: a framework that isolates specific states in an agent's execution flow where LLM vulnerabilities manifest, enabling the systematic identification and categorization of security risks that propagate from the LLM to the agent level.
We apply this framework to construct the $b^3$ benchmark, a security benchmark based on 194,331 unique crowdsourced adversarial attacks. We then evaluate 34 popular LLMs with it, revealing, among other insights, that enhanced reasoning capabilities improve security, while model size does not correlate with security.
We release our benchmark, dataset, and evaluation code to facilitate widespread adoption by LLM providers and practitioners, offering guidance for agent developers and incentivizing model developers to prioritize backbone security improvements.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces threat snapshots, a framework that isolates specific states in an agent’s execution flow where LLM vulnerabilities emerge. By doing so, it enables the systematic identification and categorization of security risks that propagate from the LLM to the agent level.
We apply this framework to build $B^3$, a security benchmark derived from 79,466 unique, crowdsourced adversarial attacks.

### Strengths
The scope of the evaluation is extensive, covering 27 popular LLMs. The benchmark construction is also interesting, as it leverages crowdsourced adversarial attacks to ensure diverse and realistic threat coverage.

### Weaknesses
1. It remains unclear why isolating backbone LLMs is necessary for analyzing their influence on agent security. This represents only one aspect of LLM security, and it is not very meaningful without studying the end-to-end security of the entire agent system.

2. The benchmark has limited value for evaluating newly released LLMs, as it may not fully reflect recent advancements or evolving attack surfaces.

3. The contribution appears limited because the proposed “threat snapshots” lack substantial technical novelty, and the resulting benchmark offers only modest security insights. Furthermore, the range of applicable usage scenarios remains limited.

### Questions
1. Do you think security issues of backbone LLMs should be addressed at the model level or within the broader agent system? If mitigations are implemented primarily in other components of the agent, does identifying weaknesses in the model itself remain sufficiently meaningful for model developers?

2. When a new LLM is released, how should its security be evaluated within this benchmarking framework?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces threat snapshots, a framework for systematically evaluating the security of backbone LLMs in AI agents. Instead of modeling full agent workflows, it analyzes key states where attackers can inject malicious context. Using the b³ benchmark—210 curated attacks across 10 scenarios—the authors test 27 LLMs and find that reasoning modes often improve robustness, while larger models are not necessarily more secure. The study highlights key vulnerabilities, proposes standardized evaluation metrics, and calls for security to be treated as a core dimension of LLM assessment.

### Strengths
1. Proposes a novel threat snapshot framework to systematically analyze LLM security in agents.

1. Builds the b³ benchmark with diverse, realistic attack scenarios and defense levels.

1. Uses crowdsourced red-teaming to collect over 79,000 real attack prompts.

1. Reveals that reasoning modes improve robustness, while size doesn’t guarantee safety.

1. Offers clear practical insights for designing safer AI agents.

### Weaknesses
1. The paper does not evaluate model utility or performance trade-offs, making it hard to identify backbones that balance security and capability.

1. The tested defenses are limited, excluding some sota defense methods.

1. The threat snapshot abstraction only captures single-step interactions, overlooking multi-turn or long-horizon attacks that occur in real agents.

### Questions
1. There are already several benchmarks for evaluating agent security. How does your b³ benchmark differ from existing ones in scope, methodology, or attack coverage? A comparison table would help clarify the distinctions.

2. The selection of threat snapshots in Table 2 seems heuristic. How can you ensure that these categories are comprehensive enough to capture new or unseen agent scenarios in the future?

3. Could you evaluate or at least discuss more state-of-the-art defense methods beyond the three levels (L1–L3)? What insights from your attack results could inform the design of stronger defenses?

4. How were the tools and their functionalities implemented in your benchmark? Were they adapted from existing agent frameworks or custom-designed for this study?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a new framework, threat snapshots, for evaluating the security of Large Language Models (LLMs) used as backbones in AI agents. The framework isolates specific states in an agent's execution flow, decoupling the LLM's vulnerabilities from traditional software risks. Based on this framework, the authors construct $b^3$, a large-scale security benchmark. This benchmark includes 10 threat scenarios and an impressive dataset of 79,466 adversarial attacks collected via a gamified crowdsourcing challenge. The authors then use $b^3$ to conduct a comprehensive evaluation of 27 popular LLMs. The results yield several insights, notably that enhanced reasoning capabilities, rather than model size, correlate positively with better security.

### Strengths
1. **Novel and Practical Framework**: The threat snapshot concept is a clear and effective abstraction. It intelligently decomposes the complex problem of "agent security" into a more manageable one: evaluating the backbone LLM's security at a specific, contextualized state. This approach greatly simplifies the evaluation process.

2. **Massive, High-Quality Benchmark ($b^3$)**: The core contribution is the $b^3$ benchmark. The dataset of nearly 80,000 human-generated adversarial attacks, collected through a "gamified" crowdsourcing effort, is a significant contribution.

3. **Broad Evaluation and Interesting Findings**: The evaluation across 27 LLMs is comprehensive. The findings provide valuable, quantifiable evidence for agent developers. While some findings confirm existing intuitions (e.g., closed-source models perform better), the conclusion that model size does not correlate with security is an important insight for the community.

### Weaknesses
1. **Scope Limited to Single-Agent Scenarios**: While the paper mentions that the framework could apply to Multi-Agent Systems (MAS), the 10 threat snapshots (Table 2) all focus exclusively on single-agent contexts. The evaluation misses key MAS-specific security risks, such as inter-agent deception, manipulation, or collusion.
2. **Inability to Capture Long-Horizon Attacks**: The "threat snapshot" method, by design, evaluates security at a single point in time. This makes it difficult to capture multi-step attacks, where a vulnerability only manifests after a long sequence of seemingly benign interactions. The paper claims that multi-step attacks can be "decomposed" into snapshots, but this is not experimentally validated and seems insufficient for attacks that rely on stateful, long-term manipulation.

### Questions
1. **Suggestion for Multi-Agent Systems (MAS)**: Given the importance of MAS, I suggest the authors discuss the challenges of applying the "threat snapshot" framework to multi-agent scenarios. For future work, adding snapshots that model inter-agent communication (e.g., one agent attempting to deceive another) would significantly broaden the benchmark's impact.

2. **Suggestion for Long-Horizon Attacks**: How does the "snapshot" framework propose to handle attacks that are stateful and build up over many turns? Would a new methodology be required to measure an agent's security "drift" during a continuous, multi-step interaction, rather than just at a single point? A discussion on this limitation would be welcome.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Threat Snapshots, a formal framework for systematically isolating and modeling LLM-specific vulnerabilities in AI agents, distinguishing them from traditional software security flaws. Using this abstraction, the authors construct a large-scale, open benchmark based on 79K human-crafted adversarial attacks across ten representative agentic scenarios.

### Strengths
1. The Threat Snapshot formalism is a major conceptual contribution that elegantly decouples LLM-specific vulnerabilities from agentic system context, enabling generalizable benchmarking and red teaming.
2. The benchmark (79K attacks, 27 models) is very comprehensive. It is valuable for both researchers and practitioners.

### Weaknesses
The threat-snapshot abstraction focuses on isolated LLM calls and single-backbone behavior, but the paper does not convincingly show that these snapshots still isolate backbone vulnerabilities when execution flows interleave multiple LLMs or long multi-turn interactions. In real agent deployments, control flow, state handoffs, and interaction between multiple models can create emergent attack surfaces (e.g., prompt-infection cascading across agents) that a single-call snapshot may miss. This leaves an open question about how well b³ predicts security in richer, multi-actor deployments.

### Questions
1. How transferable are the threat snapshot vulnerabilities to multi-turn or multi-agent settings—do they still isolate backbone behavior cleanly when control flow interleaves multiple LLMs?
2. Could the framework incorporate automated attack generation (e.g., adaptive red teaming or LLM-driven mutation) to reduce human bias in the crowdsourced dataset?

### Soundness
3

### Presentation
3

### Contribution
3
