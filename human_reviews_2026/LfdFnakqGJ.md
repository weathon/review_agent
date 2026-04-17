# A2ASecBench: A Protocol-Aware Security Benchmark for Agent-to-Agent Multi-Agent Systems

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Multi-agent systems (MAS) built on large language models (LLMs) increasingly rely on agent-to-agent (A2A) protocols to enable capability discovery, task orchestration, and artifact exchange across heterogeneous stacks. While these protocols promise interoperability, they also introduce new vulnerabilities. In this paper, we present the first comprehensive security evaluation of A2A-MAS. We develop a taxonomy and threat model that categorize risks into supply-chain manipulations and protocol-logic weaknesses, and we detail six concrete attacks spanning all A2A stages and components with impacts on confidentiality, integrity, and availability. Building on this taxonomy, we introduce A2ASecBench, the first A2A-specific security benchmark framework capable of probing diverse and previously unexplored attack vectors. Our framework incorporates a dynamic adapter layer for deployment across heterogeneous agent stacks and downstream workloads, alongside a joint safety–utility evaluation methodology that explicitly measures the trade-off between harmlessness and helpfulness by pairing adversarial trials with benign tasks. We empirically validate our framework using official A2A Project demos across three representative high-stakes domains (travel, healthcare, and finance), demonstrating that the identified attacks are both pervasive and highly effective, consistently bypassing default safeguards. These findings highlight the urgent need for protocol-level defenses and standardized benchmarking to secure the next generation of agentic ecosystems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents A2ASECBENCH, the first protocol-aware security benchmark framework for agent-to-agent multi-agent systems (A2A-MAS), which is built on a taxonomy categorizing A2A risks into supply-chain manipulations and protocol-logic weaknesses (covering six concrete attacks) and a joint safety–utility evaluation methodology; empirical validation across travel, healthcare, and finance domains shows the identified attacks are highly effective in bypassing default safeguards, highlighting the urgent need for protocol-level defenses and standardized benchmarking.

### Strengths
- The paper pioneers a comprehensive threat taxonomy and model for A2A-MAS, categorizing risks into supply-chain manipulations and protocol-logic weaknesses while detailing six concrete attacks, which fills the gap of low-level, protocol-specific vulnerability exploration in existing LLM-MAS security research.
- A2ASECBENCH, the first A2A-specific security benchmark framework proposed, incorporates a dynamic adapter layer for heterogeneous deployments and a joint safety–utility evaluation methodology, enabling probing of diverse unexplored attack vectors and explicit measurement of the harmlessness-helpfulness trade-off.
- The empirical evaluation is rigorous and impactful: it tests the framework on official A2A demos across three high-stakes domains (travel, healthcare, finance), with most attacks achieving 100% success rates, clearly revealing systemic protocol-level vulnerabilities in current A2A deployments.
- The paper provides practical and actionable insights for multiple stakeholders (agent developers, system designers, protocol researchers), such as progress-aware orchestration and verifiable capability claims, and advocates layered defenses to guide the secure design and adoption of A2A ecosystems.
- Artifact is provided at the submission stage.

### Weaknesses
To be honest, I am not very familiar with the A2A protocol and the formal security research of multi-agent systems. I have tried my best to understand the attack design part and believe that the theoretical design of the benchmark (A2ASECBENCH) is largely sound. However, my main considerations lie in the experimental aspect, particularly regarding the generalizability of this A2A-focused benchmark:
As the authors acknowledged, the evaluation of A2ASECBENCH is limited to the official implementation of the A2A protocol. This makes me question whether the benchmark’s design is sufficiently general to bring substantial contributions to the broader multi-agent security community—for example, whether its core design ideas (such as the threat taxonomy, dynamic adapter layer, or joint safety-utility evaluation) can be extended to other multi-agent collaboration frameworks beyond A2A.
Moreover, the benchmark’s design appears to be highly coupled with the A2A protocol itself. Given that the A2A community does not seem particularly widespread yet, this coupling may restrict its application scope in practice. That said, it is worth noting that the benchmark’s design depth is relatively solid, its structured threat categorization does provide valuable insights for multi-agent security research, even if its direct applicability is currently bounded by the A2A ecosystem.

Moreover, a limitation section should be presented.

### Questions
Please refer to the Weaknesses section, as I am not sure about the potential impact and contribution of this work to MAS Security Community. If more explanation on this can be provided, I could consider changing my decision, though low confidence is still preserved.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper fills the gap of lacking protocol-specific security benchmarks for A2A-MAS. It proposes A2ASECBENCH, a framework with a threat taxonomy (6 attacks), scenario adapter, and joint safety-utility evaluation. Key results: 5 attacks achieve 100% ASR across travel/healthcare/finance, highlighting A2A’s systemic vulnerabilities.

### Strengths
1. First A2A-specific benchmark: It covers 6 novel attack vectors (e.g., Cycle Overflow, ATSI) spanning supply-chain and protocol-logic risks, unlike generic LLM benchmarks, enabling targeted A2A security evaluation.
2. Scenario adaptability: The dynamic adapter maps attacks to heterogeneous A2A stacks (e.g., travel/healthcare), ensuring portability—e.g., it generates domain-specific test cases by aligning attacks with scenario specs. 
3. Joint safety-utility evaluation: It pairs adversarial trials with benign tasks to measure trade-offs (e.g., Capability Cloaking reduces travel utility from 0.853 to 0.682), avoiding one-sided safety assessment.

### Weaknesses
1. Limited defense testing: It focuses on attack effectiveness but only tests host prompt hardening as mitigation; other defenses (e.g., protocol-level signatures, peer authentication) are unexamined—adding diverse defenses would improve benchmark comprehensiveness. 
2. Small-scale MAS testing: It evaluates MAS with 1 host + 3 remote agents; large-scale MAS (10+ agents) are untested—testing larger systems would confirm scalability of vulnerabilities. 
3. Lack of production deployment validation: It uses official A2A demos but not real-world production stacks; testing on deployed systems would enhance practical relevance.

### Questions
Please refer to the weaknesses above.

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
3

### Summary
This paper presents a benchmark framework for studying the multi-agent security of the Agent-to-Agent protocol, named A2ASecBENCH. Using A2ASecBENCH, this paper identifies six types of security threats present in the A2A protocol.

### Strengths
S1. This work is well-motivated, well-structured, and clearly presented.

S2. The proposed benchmark framework is rigorously defined mathematically.

S3.  A2ASecBENCH reveals six critical types of attacks within the A2A ecosystem, which are essential for comprehensively probing security risks in this ecosystem.

S4. The authors validate the effectiveness of the six identified attack types in three representative A2A protocol application scenarios.

S5. Potential defenses are discussed.

### Weaknesses
W1. The applicability of A2ASecBENCH is limited. Although I consider its design to be solid, it targets only the A2A protocol proposed by Google.

W2. The following two papers also systematically analyze security vulnerabilities in the A2A protocol; however, this paper does not thoroughly discuss their similarities and differences.

[1] Building a secure agentic AI application leveraging A2A protocol.

[2] Improving Google A2A Protocol: Protecting Sensitive Data and Mitigating Unintended Harms in Multi-Agent Systems.

### Questions
Q1. I am curious about the generality of the six types of attacks identified by A2ASecBENCH and whether they also exist in other multi-agent communication protocols.

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
The paper shows that multi-agent LLM systems using the A2A protocol have a protocol-level attack surface that prompt-level safety can’t see. It introduces A2ASecBench, a protocol-aware benchmark that encodes six concrete attacks (two supply-chain, four protocol-logic) across travel, healthcare, and finance, and evaluates both attack success and benign-task utility. Running it on real A2A demos, five of six attacks succeed nearly 100% of the time, and spoofed/capability-cloaked agents can hijack or degrade normal workflows, proving the registry’s trust model is too weak.

### Strengths
1. Propose the first protocol-aware benchmark (A2ASecBench) with six instantiated attacks on three domains (travel, healthcare, finance)
2. Strong empirical evidence of systemic vulnerability. On official A2A samples, five of six attacks hit 100% ASR across all domains, and even the “harder” AgentCard Spoofing still succeeds ~0.82–0.83.
3. Clear and easy to understand writing with takeaways

### Weaknesses
1. Limited empirical scope (only A2A). All results are on “official A2A samples” in three domains; there’s no evidence the attacks transfer to independently built, messier A2A stacks or to agent platforms that already add extra validation. This weakens the “protocol-wide” claim. Adding at least one non-official, differently engineered MAS baseline (e.g., with custom task gating or artifact sanitization) would make the evaluation harder to dismiss.
2. Assumptions about rendering/consumption in ATSI. The Artifact-Triggered Script Injection attack inherits XSS-like power only if the host or a downstream agent renders artifacts in a permissive way; the paper does not enumerate which of the standard A2A sample apps actually do that. A tighter analysis of artifact types vs. vulnerable renderers would help readers know when ATSI is real and when it’s theoretical. 
3. Significance tied to A2A’s adoption curve. The paper’s impact claim leans on A2A being the interoperability layer for agents, but the experiments don’t show applicability to adjacent ecosystems (OpenAI’s agent runtime, LangGraph-style planners, or in-house orchestrators). Porting 1–2 attacks through the “dynamic adapter” into a non-A2A stack would make the contribution less protocol-fragile.

### Questions
1. how many concurrent tasks, for how long, and on what hardware/config are used in DoS like attacks ? Adding resource-usage curves (tasks vs. latency/queue depth) would make the threat more operational and results more clear.
2. Would smarter registries (e.g., fuzzy matching, issuer-based trust, or signed manifests) help in the AgentCard Spoofing attack? How many lookalike cards are needed to get selected with high probability? A short sensitivity study on discovery ranking would turn this from a single demo into a reusable security test.

### Soundness
3

### Presentation
3

### Contribution
3
