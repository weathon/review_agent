# Collaborative Memory: Multi-User Memory Sharing in LLM Agents with Dynamic Access Control

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Complex tasks are increasingly delegated to ensembles of specialized LLM-based agents that reason, communicate, and coordinate actions—both among themselves and through interactions with external tools, APIs, and databases. While persistent memory has been shown to enhance single-agent performance, most approaches assume a monolithic, single-user context—overlooking the benefits and challenges of knowledge transfer across users under dynamic, asymmetric permissions. We introduce Collaborative Memory, a framework for multi-user, multi-agent environments with asymmetric, time-evolving access controls encoded as bipartite graphs linking users, agents, and resources. Our system maintains two memory tiers: (1) private memory—private fragments visible only to their originating user; and (2) shared memory—selectively shared fragments. Each fragment carries immutable provenance attributes (contributing agents, accessed resources, and timestamps) to support retrospective permission checks. Granular read policies enforce current user–agent–resource constraints and project existing memory fragments into filtered transformed views. Write policies determine fragment retention and sharing, applying context-aware transformations to update the memory. Both policies may be designed conditioned on system, agent, and user-level information. Our framework enables safe, efficient, and interpretable cross-user knowledge sharing, with provable adherence to asymmetric, time-varying policies and full auditability of memory operations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Collaborative Memory, a framework for managing memory in multi-user, multi-agent systems with asymmetric and time-evolving access controls. The key innovation is the formalization of dynamic permissions using bipartite graphs that connect users, agents, and resources, combined with a two-tier memory architecture. The system enforces fine-grained read and write policies that respect access constraints while enabling cross-user knowledge sharing.

### Strengths
1. The two-tier memory system with provenance tracking provides an elegant solution to the privacy-utility tradeoff.

2. Clear design of three progressively complex scenarios effectively demonstrates different aspects of the framework.

3. Results and analyses show significant resource utilization reductions while maintaining accuracy and zero information leakage.

### Weaknesses
**1. Limitations in real-world validation**: All experiments use synthetic datasets or benchmarks, while lacking evaluation with real-world multi-user deployments. This raises questions about practical applicability.

**2. Limitations in policy implementations**: The read/write policies (Table 1) are quite basic, consisting of simple transformations that are applied uniformly. However, this paper doesn't explore more sophisticated policies, such as differential privacy, semantic filtering, context-dependent redaction, etc.

**3. Limitations in privacy analysis**: This paper shows no formal privacy guarantees or threat model. The "leakage ratio" metric is introduced only in comparison with baseline. It also lacks discussion of potential attacks or adversarial scenarios, and it is unclear what "zero leakage" actually means, especially in terms of information privacy.

**4. Limitations in scalability:** This paper only tests 5→50 users with 10% execution time increase, but doesn't address the memory-related problems that this paper focuses on, such as the memory storage growth over time, the query latency with large memory pools, the concurrency handling with many simultaneous users, etc.

**5. Limitations in experiments:** It only compares against the Memory Sharing baseline, lacking baselines such as traditional access control systems, other multi-agent memory architectures, privacy-preserving collaborative learning approaches, etc. It also lacks discussions of failure modes and common patterns that can benefit future improvements.

### Questions
My questions are following several aspects mentioned in weakness:

- Can you provide formal privacy guarantees for your framework? What specific threat model does the system defend against, and what real-world scenarios this framework can apply and/or generalize to?
- The current read/write policies are fairly simple. Can current framework and findings generalize to more complex policies? What are the results of your investigation on more complex policies?
- In Equation 3, $\mathcal{M}(u,a,t)$ requires both $\mathcal{A}(m) \subseteq \mathcal{A}(u,t)$ and $\mathcal{R}(m) \subseteq \mathcal{R}(a,t)$. Why is this the right semantics? Can there be cases where partial overlap should allow access?
- How do you ensure the quality of shared memories over time? Can incorrect information from one user pollute the shared memory for others?
- How would your framework compare against other baselines, such as traditional access control systems, other multi-agent memory architectures, privacy-preserving collaborative learning approaches, etc.?
- If shared memory contains conflicting information contributed by different users with different access levels, how does the system resolve this during retrieval?

### Soundness
3

### Presentation
2

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
This paper introduces a framework that enables multi-user, multi-agent LLM systems to efficiently manage memories while ensuring safety and adhering to dynamic and asymmetric access policies.

### Strengths
1. This paper demonstrates novelty by addressing an important problem in multi-user, multi-agent systems that is both significant and highly practical: private and collaborative memory management under dynamic and asymmetric permissions.
2. The paper designs experiments across multiple scenarios, covering a diverse range of settings.
3. Framework design is theoretically clear and logically consistent.

### Weaknesses
1. The three scenarios utilize inconsistent baselines,  making it difficult to conduct a fully comprehensive ****comparison.
2. The experiments in Scenario 2 only demonstrate efficiency improvements of the proposed method but fail to evaluate actual answer quality. Could the collaborative memory system be producing lower-quality final answers faster by reusing inaccurate, incomplete, or biased intermediate results?
3. The proposed method can mechanistically control which specific memory fragments are shown to a user. However, generation and anonymization stages of the memory fragments themselves depend on the capability of an external LLM. Since LLMs are known to be imperfect at tasks like consistent anonymization and can sometimes hallucinate or retain sensitive information, this dependency represents a fundamental limitation.

### Questions
The framework focus on writing and reading policies of memory fragments. In a dynamic environment, information can become outdated, be proven incorrect, or conflict with other entries. What mechanisms, if any, does the framework propose for handling memory updates, invalidation, or conflict resolution?

### Soundness
3

### Presentation
2

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
This paper addresses the problem of memory management in multi-agent and multi-user systems. The authors proposed Collaborative Memory, a formal framework that allows multiple users and LLM agents th share memory safely, where access is asymmetric and changes over time.

### Strengths
1. Multi-user and multi-agent memory managemtn with dynamic access control is underexplored.

2. The problem is formalized using bipartite graphs and provenance constraints, which is clear and easy to understand.

3. The authors separated private/shared memory with read/write policies, which is a practical and useful abstraction, and can be leveraged in real systems.

4. The work has the potential to influence future system designs for enterprise multi-agent AI.

### Weaknesses
1. While I appreciate the authors contributions, this paper does not introduce any new algorithms, and the technical contributions are limited. The contribution is primarily definitional, which raises concerns about whether it fits the suitable for this conference. It may be more appropriate for a venue that focuses on system design or conceptual frameworks rather than algorithmic advances.

2. The datasets used in evaluation are simple. There is no demonstration in a realistic large-scale dataset.

3. The paper does not address complex or realistic access-control policies. In real enterprise systems, permissions are often hierarchical, but the framework only handles relatively simple policy types.

### Questions
See above

### Soundness
3

### Presentation
4

### Contribution
2
