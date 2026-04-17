# Proof-Carrying Numbers (PCN): A Protocol for Trustworthy Numeric Answers from LLMs via Claim Verification

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Large Language Models (LLMs) as stochastic systems may generate numbers that deviate from available data, a failure known as $\textit{numeric hallucination}$. Existing safeguards—retrieval-augmented generation, citations, and uncertainty estimation—improve transparency but cannot guarantee fidelity: fabricated or misquoted values may still be displayed as if correct. We propose $\textbf{Proof-Carrying Numbers (PCN)}$, a presentation-layer protocol that enforces numeric fidelity through mechanical verification. Under PCN, numeric spans are emitted as $\textit{claim-bound tokens}$ tied to structured claims, and a verifier checks each token under a declared policy (e.g., exact equality, rounding, aliases, or tolerance with qualifiers). Crucially, PCN places verification in the $\textit{renderer}$, not the model: only claim-checked numbers are marked as verified, and all others default to unverified. This separation prevents spoofing and guarantees fail-closed behavior. We formalize PCN and prove soundness, completeness under honest tokens, fail-closed behavior, and monotonicity under policy refinement. PCN is lightweight and model-agnostic, integrates seamlessly into existing applications, and can be extended with cryptographic commitments. By enforcing verification as a mandatory step before display, PCN establishes a simple contract for numerically sensitive settings: $\textit{trust is earned only by proof}$, while the absence of a mark communicates uncertainty.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes **Proof-Carrying Numbers (PCN)**, a formal framework for enabling *verifiable numeric reasoning* in large language models (LLMs).  
Inspired by the “proof-carrying code” paradigm, PCN introduces a protocol where every numerical claim made by an LLM must be accompanied by a *proof payload* and a corresponding *verifier function*.  
The framework defines a formal verification relation \( R(t, c; \Pi) \), proving properties such as **soundness**, **completeness**, **monotonicity**, and **fail-closed behavior**.  
Rather than modifying model weights, PCN operates as an external protocol—attaching, verifying, and rendering numeric claims in a transparent and deterministic manner.  
The paper includes implementation examples (TypeScript mock verifier and UI demos) but no quantitative experiments.  
The work is intellectually rigorous and conceptually elegant, but its absence of empirical evaluation limits its impact for an ICLR audience.

### Strengths
- **Original conceptual framing:** Extends formal verification into LLM numeric reasoning—a novel and timely idea.  
- **Strong theoretical grounding:** Definitions, theorems, and proofs are rigorous and internally consistent.  
- **Clear architecture:** The distinction between claim, proof, and verifier entities is cleanly described and easy to reason about.  
- **Lightweight integration:** Operates as a protocol, independent of model weights or API internals, potentially adaptable to any LLM.  
- **Readable and disciplined exposition:** Mathematical sections are well presented and logically ordered.

### Weaknesses
- **No experimental validation:** The paper lacks quantitative experiments demonstrating PCN’s usability, latency, or accuracy in real-world LLM pipelines.  
- **No comparative analysis:** There are no benchmarks or case studies comparing PCN with related frameworks (e.g., CoIn, SelfCAD, or symbolic verifiers).  
- **Unverified practicality:** Claims of low overhead and ease of deployment are speculative without implementation metrics.  
- **Scope limitation:** The work is focused exclusively on numeric outputs; generalization to textual or logical claims remains unclear.  
- **Structural mismatch:** The absence of an evaluation section makes the paper feel more like a whitepaper or formal report than a research paper suited for ICLR’s empirical standards.

### Questions
1. Could the authors provide a small-scale experiment showing PCN’s end-to-end operation—e.g., verifying numeric outputs from GPT-4 or Llama using real arithmetic tasks?  
2. How does PCN handle floating-point imprecision or non-deterministic rounding in arithmetic outputs?  
3. Could the verification protocol be extended to logical or symbolic claims, beyond numeric reasoning?  
4. What are the latency and resource costs of the verification step in a practical deployment scenario?

### Soundness
3

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
4

### Summary
The paper introduces Proof-Carrying Numbers (PCN), a presentation-layer protocol designed to address numeric hallucination in Large Language Models. PCN requires that numeric values displayed to users be mechanically verified against authoritative structured claims before presentation. The protocol consists of four components: a retriever that obtains claims from structured sources, an LLM generator that produces claim-bound tokens, a verifier that checks tokens against claims under specified policies, and a UI that displays verification status. The authors provide formal guarantees including soundness, completeness, and fail-closed behavior, positioning PCN as a deterministic solution to a critical trust problem in LLM applications.

### Strengths
1. Novel problem framing: Treating numeric hallucination as a presentation-layer rather than generation-layer problem is insightful and enables deterministic guarantees that probabilistic approaches cannot provide.

2. Formal rigor: The paper provides clear formalization with theorems for soundness, completeness, fail-closed behavior, and monotonicity, giving the protocol a solid theoretical foundation.

3. Practical design: The lightweight, modular architecture makes PCN easy to integrate into existing systems without significant overhead (O(n) verification complexity).

4. Comprehensive coverage: The paper addresses multiple aspects including verification policies, cryptographic extensions, adversarial robustness, and implementation details with concrete code examples.

5. Important real-world problem: Numeric hallucination in high-stakes domains (healthcare, finance, policy) is a critical issue that needs addressing.

### Weaknesses
1. Lack of empirical evaluation: The paper contains no experiments, benchmarks, or quantitative analysis. Critical questions remain unanswered: What percentage of numeric spans do LLMs successfully tag? How often does verification fail? What is the impact on user experience?

2. Dependency on LLM cooperation: The entire system relies on LLMs correctly emitting claim-bound tokens. If models have poor recall in tagging numbers, most values remain unverified, limiting practical utility. No evidence is provided on how well current LLMs can be prompted/fine-tuned for this task.

3. Limited scope: The protocol only handles atomic numeric values from structured sources. It cannot verify derived values (sums, averages, ratios) or numbers requiring computation, which significantly limits applicability.

4. No baseline comparison: The paper doesn't empirically compare PCN against existing approaches (RAG with citations, uncertainty estimation, etc.) making it difficult to assess relative effectiveness.

### Questions
1. Empirical performance: What is the actual coverage rate when PCN is deployed with current LLMs? How many numbers get successfully tagged and verified versus remaining bare?

2. LLM training/prompting: What specific prompting strategies or fine-tuning approaches achieve best results for claim-bound token emission? Have you experimented with different models?

3. Derived value support: How could PCN be extended to handle calculated values? Could you verify arithmetic operations over verified atomic claims?

4. Scalability: How does the system perform with large claim sets? What happens when multiple data sources provide conflicting claims for the same entity?

5. User study: Have you conducted any user studies to understand how users interpret verification marks and whether the system actually increases trust?

### Soundness
3

### Presentation
3

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
This paper proposes Proof-Carrying Numbers (PCN) — a presentation-layer contract that requires numeric outputs from LLMs to be verifiably linked to authoritative structured claims before being displayed. The protocol includes:

• Claim-bound tokens that attach values to claim metadata

• A renderer-side verifier enforcing fail-closed guarantees

• Formal proofs of soundness, completeness, and spoofing robustness

• Example implementation showing UI verification behaviors

### Strengths
Well-motivated problem: Numerical hallucinations can be dangerous in high-stakes domains such as medicine, finance, and policy-making. The paper clearly demonstrates the societal relevance.

Practical and deployment-friendly: The proposed PCN protocol operates at the presentation layer without requiring model retraining, making it easy to adopt in real-world products.

Fail-closed guarantee: Verification is enforced at UI time, preventing unverified content from being displayed—a robust security stance compared to heuristic filters.

Formal grounding: The soundness and spoofing-resistance proofs provide solid assurance that the protocol behaves as intended.

Modular architecture: The method can be combined with RAG systems and knowledge bases and does not preclude future improvements in verification granularity.

Overall, the paper offers a clean and tractable approach to enhancing trustworthiness for numeric outputs.

### Weaknesses
Scope is too narrow — only numeric spans are handled

The paper focuses exclusively on numeric hallucination, while many real-world failures involve textual factuality, URLs, or entity-level misinformation. Restricting verification to numbers significantly limits the safety promises. Real deployment scenarios may require:

• Verified citations for URLs and web resources

• Fact-verification for entity claims (names, affiliations, dates)

• Logical consistency checks for multi-hop reasoning

Addressing only numeric content risks creating a “false sense of safety” if users assume broader correctness guarantees.

PCN prevents wrong numbers but does not reduce hallucination at its source

The core protocol polices visibility, but does not improve model reasoning. A model may still hallucinate the wrong source or produce many unverifiable values, degrading usability and trust calibration.

No mechanism for derived or implicitly computed quantities

Ratios, differences, and statistical transformations are common in decision support tasks. Blocking them entirely may frustrate users, yet verifying them is out-of-scope.

Tagging reliability untested

Without strong evidence that LLMs consistently provide correct \<claim\> tags, the whole pipeline may frequently fail closed.

### Questions
See weakness

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces PCN a novel protocol designed to address the critical issue of numeric hallucinations in LLMs. The authors' core contribution is to reframe this issue not as an intractable model-level problem, but as a solvable presentation-layer challenge. PCN establishes a fail-closed trust contract where numeric values are only marked as verified after passing a deterministic, mechanical check by the renderer against an authoritative structured claim. The protocol is designed to be lightweight, model-agnostic, and easily integrable into existing RAG pipelines.

### Strengths
- The problem itself has strong practical value.

- The protocol is designed as a lightweight set of components that can be easily integrated into existing RAG pipelines. It doesn't depend on any specific LLM architecture, and the verification step is efficient.

### Weaknesses
- Triggering the entire PCN process depends on whether the LLM (the generator) remembers to attach labels around the numbers it generates and does so correctly. This is itself subject to hallucination or can be bypassed. This issue is especially important because RAG systems often have very long contexts.

- The paper lacks experimental validation of the method’s effectiveness. Given the dependency on the LLM mentioned above, it is hard to quantify the method’s effectiveness without experiments.

- Because the paper decouples verification from generation, it cannot determine the correctness of all non-numeric facts in an answer. At the same time, there is no way to use verification failures to fix the LLM so it produces a correct answer. The method’s extensibility is limited.

- Although the method offers some robustness against deception, it is fairly limited: for example, an attacker could hack the model by preventing the LLM from generating tags.

### Questions
Please see Weaknesses

### Soundness
1

### Presentation
2

### Contribution
3
