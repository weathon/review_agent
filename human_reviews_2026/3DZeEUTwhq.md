# Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs

- Decision: Reject
- Scores: 6, 2, 6, 2

## Abstract
Commercial Large Language Model (LLM) APIs create a fundamental trust problem: users pay for specific models but have no guarantee that providers deliver them faithfully. Providers may covertly substitute cheaper alternatives (e.g., quantized versions, smaller models) to reduce costs while maintaining advertised pricing. We formalize this model substitution problem and systematically evaluate detection methods under realistic adversarial conditions. Our empirical analysis reveals that software-only methods are fundamentally unreliable: statistical tests on text outputs are query-intensive and fail against subtle substitutions, while methods using log probabilities are defeated by inherent inference nondeterminism in production environments. We argue that this verification gap can be more effectively closed with hardware-level security. We propose and evaluate the use of Trusted Execution Environments (TEEs) as one practical and robust solution. Our findings demonstrate that TEEs can provide provable cryptographic guarantees of model integrity with only a modest performance overhead, offering a clear and actionable path to ensure users get what they pay for.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the critical issue of \emph{model substitution in commercial LLM APIs}---cases where service providers covertly replace an advertised model with a cheaper or quantized variant. The authors (1) formalize the model substitution problem and threat model, (2) evaluate existing software-only auditing methods such as text classifiers, MMD-based statistical tests, benchmark probing, and log-probability comparisons, and (3) propose \emph{Trusted Execution Environments (TEEs)} as a cryptographically secure and deployable solution. Through extensive experiments, they demonstrate that software-level approaches are unreliable in realistic, non-deterministic production settings, whereas TEEs provide provable model integrity guarantees with modest overhead (≈9–16\% latency). The paper concludes that TEEs are currently the most practical solution to ensure users “get what they pay for.”

### Strengths
- **Timely and relevant:** The paper targets a real trust gap in today’s LLM API ecosystem: users are billed for “Model A” but may actually get a quantized or smaller substitute.
- **Thorough evaluation:** The paper tests multiple auditing approaches under realistic adversarial settings:
  - text classifiers fail to distinguish full-precision vs. quantized variants (≈50% accuracy, i.e. random);
  - MMD tests lose power when the provider routes only part of traffic to the substitute;
  - benchmark-based auditing is confounded by hidden decoding parameters and caching;
  - greedy decoding and logprob comparison are broken by inference nondeterminism across stacks and batching.
- **Clear negative result:** It convincingly argues that software-only auditing is fundamentally brittle in production.
- **Actionable answer:** TEEs are proposed as a concrete, deployable alternative. The paper reports only ~9–16% first-token latency overhead and small throughput impact, which supports feasibility.
- **Responsible positioning:** The authors discuss implications for users, providers, and policymakers, not just measurements.

### Weaknesses
- **Novelty is mostly integrative:** The paper does not introduce a new detection algorithm or new attestation primitive. The contribution is primarily empirical and architectural.
- **Limited attack surface coverage:** The evaluation focuses on quantization, routing/mixing, and prompt caching. Other realistic substitutions (light fine-tuning on domain data, pruning, speculative decoding with a small draft model, etc.) are mentioned but not deeply quantified.
- **Security depth of TEEs:** The paper treats TEEs as “the answer,” but does not analyze:
  - side-channel leakage,
  - malicious firmware / compromised root of trust,
  - dishonest attestation endpoints in multi-tenant datacenters.
  These are important if we’re going to claim TEEs “solve” the problem.
- **Scalability and ops questions:** The TEE measurements are on a single GPU. It’s unclear how this extends to multi-GPU inference, pipeline parallelism, or high-throughput commercial routing.
- **Reproducibility / accessibility:** Verifying the TEE claims requires specialized hardware (confidential H100 / Blackwell stacks). That limits community validation.

### Questions
1. How does your auditing approach behave under *partial* substitution (e.g. 10–20% of requests silently routed to a cheaper model) when the attacker also slightly perturbs temperature or top-p to inject noise?
2. Can the proposed TEE-based workflow support multi-GPU inference or model sharding? If different GPU enclaves each serve a shard, how is attestation composed?
3. Could lightweight cryptographic proofs (e.g. ZKPs on selective layers, or TopLoc-style activation hashes) be combined with TEEs to reduce trust in the hardware vendor?
4. For open-source models that customers can run locally: do you see a practical “self-audit mode” that does not require TEEs?
5. Could the authors clarify the quantization settings used (e.g., INT8 vs FP8 schemes, calibration methods) for Table 2 and Figure 1?  
   Were the same prompts and decoding parameters kept constant across quantized and full-precision variants?

6. In Figure 1, how sensitive are the MMD results to kernel choice and sample size?  
   Did you observe any instability in power estimates under different prompt distributions?

7. For Table 4, how many runs were averaged to compute the reported latency and throughput overheads, and were warm-up tokens or network effects included in those measurements?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the problem of auditing model substitutions in LLM APIs. It formalizes the hypothesis test and evaluate a series of techniques, including text-only tests, log-probability-based tests, activation-based tests. They find these methods all fail under at least one type of adversarial substitution attack, and they claim that trusted execution environments provide the only robust guarantee for solving the problem.

### Strengths
1. Important problem and clear motivation: The problem they study is timely and important, given the widespread use of blackbox LLM APIs. 
2. Comprehensive empirical experiments: the paper systematically tests several verification strategies with different types of substitution attacks. The negative results are informative for future work.

### Weaknesses
### Major weaknesses

1. Definition of the auditing goal is too strong: the null hypothesis $H_0$ requires equality for all  (end of Section 3.1), which seems unattainable in realistic settings; the paper itself later also attributes failures of some methods to "production-level inference nondeterminism." Relaxed formulation of the hypotheses would be more coherent. However, with relaxed formulation, some substitution methods may no longer make sense, as discussed in the next point.
2. Quantization attack framing: the paper shows failure of most methods with quantization attacks. However, it is unclear from the text whether existing providers claim the dtype of their models, which makes "quantization substitution" potentially indistinguishable from allowed, valid, and legal backend optimization. Moreover, because benchmark performance deltas under quantization are small (as shown in Table 3), escaping detection may be both easy and practically operationally benign. More clarification on the motivation for this substitution method is needed. For example, if replacing model A with B improves provider-side cost-effectiveness but does not hurt user experience in any empirically observed scenario, do the authors still aim to detect this substitution? Also, it would be beneficial to consider the "severity" of constitution by how much they impact model performance, and then consider evaluation metrics for detection methods that take this "severity" into account.
3. TEE section lack details and depth: the paper claims that TEE is "the only currently viable mechanism" and lists this as one of their main contribution. However, TEE is only discussed at the very end of the paper with a short subsection. What exactly is attested with TEE? Does this include model weights and inference hyperparameters? How is the trust in this attestation verified by end-user? When I read up to the end of section one, I thought TEE as an auditing method would be a significant part of the main text.
4. Novelty is limited: the paper feels more like a position or survey paper. Much of section 4 implements known detecting techniques. If I understand it correctly, the main new contribution (as claimed by the authors) is the argument for using the existing TEE as a auditing method and the overhead measurement. This is valuable engineering evaluation but seems methodologically thin for ICLR.

### Minor issues
1. Adversary model is too narrow: the randomized substitution attack mixes models uniformly randomly, but strategic provider would very likely condition their substitution decisions on certain features of the request. The paper does not evaluate adaptive substitution. which is arguably the more realistic case.
2. Benchmark-based detection seems brittle: it seems that the adversary can simply return fixed outputs for all known benchmark queries.

### Questions
## Questions
Many questions are already discussed in the weaknesses section. Some selected/additional questions are:
1. Would you consider relaxing the null hypothesis? If so, how? How does this relaxation change your conclusions?
2. How do you propose distinguishing "benign" quantization (operationally equivalent for users) from substitutions that meaningfully violate a service agreement?
3. Is the production-level deployment of TEE practical? What are some potential blockers? How can a user verify that every request was served by the TEE?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the critical trust problem in commercial LLM APIs where users pay for specific models but cannot verify if providers are faithfully serving them. The authors formalize the model substitution detection problem, systematically evaluate existing software-based detection methods under adversarial conditions, and propose Trusted Execution Environments (TEEs) as a robust hardware-based solution. Through comprehensive empirical analysis across multiple detection techniques (text classifiers, identity prompting, statistical tests, log probability verification), they demonstrate that software-only approaches are fundamentally unreliable due to inference nondeterminism and subtle substitution strategies, while TEEs provide cryptographic guarantees with modest performance overhead (2.88-16.18%).

### Strengths
This paper makes several important contributions to addressing the critical trust problem in commercial LLM APIs. Most notably, it tackles a timely and economically significant issue where providers have strong incentives to substitute cheaper models while maintaining premium pricing, creating a fundamental trust gap in the current ecosystem. The authors provide a comprehensive and systematic evaluation of detection methods that goes well beyond previous work by considering realistic adversarial scenarios including quantization substitution, randomized model mixing, and benchmark evasion attacks. Their empirical analysis reveals crucial practical insights, particularly the surprising finding that production-level inference nondeterminism—arising from factors like batch variance and heterogeneous backends—can defeat seemingly robust detection methods such as log probability verification, even when these methods work well in controlled laboratory settings.

The paper's technical approach demonstrates strong rigor through its formal problem definition using a hypothesis testing framework (H₀: Pactual = Pspec vs H₁: Pactual ≠ Pspec) that properly grounds the empirical evaluation. The experimental methodology is sound, employing appropriate statistical tests like Maximum Mean Discrepancy with permutation testing, and the results are clearly presented through effective visualizations that illustrate key findings such as the relationship between substitution probability and detection power. Perhaps most importantly, the paper goes beyond merely identifying problems to propose and validate a practical solution through Trusted Execution Environments (TEEs). The authors demonstrate that TEEs can provide cryptographic guarantees of model integrity with surprisingly modest performance overhead—only 2.88% throughput reduction under high concurrency with 64 concurrent requests—making this a genuinely deployable solution for the industry. This combination of thorough problem analysis, systematic evaluation under adversarial conditions, practical insights about real-world deployment challenges, and an actionable solution with empirical validation makes this work a valuable contribution that directly addresses pressing concerns in the rapidly evolving LLM API ecosystem.

### Weaknesses
1. Limited TEE Evaluation: Only evaluates one model (Llama-3-8B) on one GPU (H100). Lacks testing on larger models (70B+) where memory constraints and multi-GPU setups could significantly impact feasibility and overhead. Since TEEs are proposed as the primary solution, this limited evaluation scope severely constrains confidence in the solution's general applicability.

2. Limited Real-World Validation: All experiments use controlled settings with known models and substitutions. The paper lacks evaluation with actual commercial APIs or evidence of real-world substitution detection. Without validation against production systems or case studies of actual substitution attempts, the practical applicability of both detection methods and the TEE solution remains uncertain.

3. Narrow Attack Scenarios: Focuses primarily on basic quantization and randomized substitution. Doesn't explore sophisticated adversarial strategies like adaptive substitution based on query patterns, gradual model degradation, or hybrid evasion techniques. This limits understanding of how robust the detection methods and TEE solution would be against determined adversaries with economic incentives to evade detection.

### Questions
1. How would the TEE solution scale to larger models (175B+ parameters) that require multi-GPU setups? What are the technical challenges and expected overheads?

2. Have you considered hybrid approaches combining multiple detection methods (e.g., periodic TEE attestation with continuous statistical monitoring)?

### Soundness
3

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
4

### Summary
This paper studies the problem of auditing model substitution in LLM APIs. When users use an API provider to generate text from an LLM, they may want to check that they are being served the correct model that they are paying for. API providers may have an incentive to secretly use a different model, e.g., a smaller model to save on compute costs.

This paper runs experiments to claim that existing software-based auditing methods are unreliable in adversarial scenarios, such as quantization and randomized substitution. Therefore, this paper proposes trusted execution environments (TEEs) as a solution, where hardware enclaves provide cryptographic attestation reports of correct model and code execution.

### Strengths
The problem of auditing model substitution is well-motivated and important given the widespread usage of LLM APIs today. The introduction clearly and succinctly states the problem setting. The proposed use of TEEs for auditing model substitution appears to be a novel idea.

### Weaknesses
I don’t think that the paper has substantial enough contributions for what is typically expected in a publication. The majority of the paper consists of simple evaluations of existing methods on adversarial settings, such as subtle differences due to quantization. Note that several of these existing methods were originally designed for problems other than auditing model substitution. For example, [Sun et al. (2025)](https://arxiv.org/abs/2502.12150) studies the problem of predicting which model generated a particular text, e.g., ChatGPT vs Claude.

[Gao et al. (2024)](https://arxiv.org/abs/2410.20247) studies the same problem of auditing model substitution as the submitted paper, and proposes and evaluates principled, rigorous statistical tests (MMD) for doing so. Gao et al. (2024) showed strong results in controlled, simulated settings, and also ran audits on real-world LLM APIs. I am not convinced by the submitted paper’s claim that MMD is an unreliable and impractical audit method.

The submitted paper runs a small experiment claiming that MMD achieves low power for detecting quantization of small models and randomized substitution. However, the paper does not report how many completions are sampled for each prompt when running MMD, which is a crucial factor impacting the power, as found in Gao et al. (2024). I suspect that given more samples, MMD would achieve higher power.

Another recent arXiv preprint that proposes an audit for model substitution is [Zhu et al. (2025)](https://arxiv.org/abs/2506.06975), although I do not expect a detailed experimental comparison since it is a preprint and recent (June 2025).

The proposed TEE method does not seem practical for deployment in real-world APIs, given the complexity of modern LLM inference infrastructures. Larger models are split across many GPUs, and complex routing strategies are employed due to prompt caching, mixture of experts, etc. In addition, the proposed TEE method requires the open-sourcing of inference code, which would likely be unacceptable for both proprietary model providers and providers of open-source models. Such code would reveal information about model architecture, inference optimizations and tricks, etc.

Note that TEEs require API providers to make significant implementation changes, whereas audits such as MMD can be run by users on any black-box LLM API, without needing the API provider to agree to it. So TEEs are not a comparable replacement for black-box audits.

### Questions
### Main questions

1. Line 141: shouldn’t this be $p \\to 0$ for a low substitution rate? Then it should be $p$ upward on line 143\.  
2. Line 246: doesn’t the prompt have to appear in the MMD equation? It only makes sense to compare completions that came from the same prompt.  
   * It is confusing to use $x$ to denote completions here, as $x$ was used for prompts and $y$ for completions earlier in the paper.  
   * The model distribution was only defined as the conditional distribution $P(y \\mid x)$ earlier in the paper, so it is not clear what exactly drawing $x \\sim P$ means here.  
3. Figure 1 (left): why are there two bars for original and quantized? Comparing the original and quantized models is one test, so why isn’t there just one value for the power?  
4. Line 253: how many completions are sampled for each prompt in the test? Gao et al. (2024) found that the power is sometimes low when few samples are used, but high when more samples are used.  
5. Line 269: where does Gao et al. (2024) state that *“MMD-based auditing is only effective under strictly controlled local inference environments, limiting its practicality”*? Gao et al. (2024) run audits on real-world API providers, so I’m not sure where this is coming from.  
6. Line 317: the size of the drop in accuracy for higher temperatures seems significantly higher than the drop from quantization. The drop from higher temperature is around 5–10 percentage points, whereas the difference from quantization is usually around 1–2 percentage points.  
7. Can’t the effect of higher temperature be mitigated by taking the most common answer across queries, instead of averaging? The relative probabilities between answers should not change in higher temperatures.  
8. Line 350 states that both $k \= 20$ and $k \= 100$ is used, but only one agreement rate is reported. Which value of $k$ is used?  
9. In section 4.1.5, how different are the greedy decoding outputs when they do not match? Are they completely different, or just off by one or two tokens?  
   * What is the agreement rate between providers, i.e., comparing Together vs. OpenRouter instead of Together vs. local baseline?  
   * Are the greedy outputs deterministic if you query the provider with the same prompt multiple times, or do they vary?  
10. Line 410: this definition of LSH is contradictory, how can “similar activations yield similar hashes” while also “small changes cause detectable differences”? LSH is designed so that similar items receive the same hash with probability, so the second part seems to be incorrect.

### Minor questions

1. Line 246: why does MMD have the squared superscript? It does not appear with the square elsewhere in the paper.  
2. Line 252: what is $T$? I assume that $L$ is the completion length, but this is also not explicitly stated.  
3. Line 252 states that Llama 70B is used, but Figure 1 (left) shows only Llama 8B and Mistral 7B.

### Soundness
1

### Presentation
2

### Contribution
1
