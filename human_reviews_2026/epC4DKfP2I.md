# SENTINEL: StagewisE iNtegriTy verification for pIpeliNe parallEL decentralized training

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Decentralized training introduces critical security risks when executed across untrusted, geographically distributed nodes. While existing Byzantine-tolerant literature addresses data parallel (DP) training through robust aggregation methods, pipeline parallelism (PP) presents fundamentally distinct challenges. In PP, model layers are distributed across workers where the activations and their gradients flow between stages rather than being aggregated, making traditional DP approaches inapplicable. We propose SENTINEL, a verification mechanism for PP training *without computation duplication*. SENTINEL employs lightweight momentum-based monitoring using exponential moving averages (EMAs) to detect corrupted inter-stage communication. Unlike existing Byzantine-tolerant approaches for DP that aggregate parameter gradients *across replicas*, our approach verifies sequential activation/gradient transmission *between layers*. We provide theoretical convergence guarantees for this new setting that recovers classical convergence rates when relaxed to standard training. Experiments demonstrate successful training of billion-parameter LLMs across untrusted distributed environments with hundreds of workers while maintaining model convergence and performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses an underexplored yet relevant threat model unique to PP training, with clear formulation and empirical validation (Tabs. 1–4). The proposed SENTINEL method demonstrates strong detection performance and low overhead, supported by theoretical reasoning (Sec. 3; Sec. 3.2 and Appendix). Theoretical analysis regarding convergence under undetected bounded corruption supports the approach (Thm. 1). However, several aspects require clarification or extension: limited empirical baselines against existing PP security methods (No direct evidence found in the manuscript), potentially strong reliance on trusted verifiers (Sec 2.1), and incomplete exploration of complex adaptive or stealthy attacks (adaptive attack limited to single scenario). The paper would benefit from deeper discussion on system overhead, scalability trade-offs, and interactions with DP-level Byzantine defenses.

### Strengths
- Well-motivated identification of a critical security gap in PP decentralized training:
>> Clear distinction between DP and PP vulnerabilities, including cascading activation corruption (Sec. 2; Fig. 2).
>> Importance supported by large-scale LLM training requirements (Sec. 1).

- Comprehensive attack taxonomy and evaluation:
>> Seven activation/gradient attack types, including delay and invisible-noise attacks (Sec. 2.1).
>> Mixed-attack experiments better match real-world adversaries (Table 2).
>> Detection performance measured via precision, recall, F1, detection speed, and final validation loss (Tabs. 1–3).

- Theoretical support with assumptions stated explicitly:
>> Convergence guaranteed to a bounded neighborhood under undetected bounded perturbations (Sec. 3.2; Thm. 1).
>> Honest majority conditions formally quantified (Lemma 1).

- Large-scale distributed experiments and SWARM deployment:
>> Demonstrates end-to-end robustness in real decentralized environment (Fig. 4).
>> Ablation studies explore warm-up duration, collusion, and gradient delay impact (Fig. 3).

Clear articulation of limitations and threat boundaries:
>> Notes vulnerability to other ML attack types, e.g., backdoor, privacy attacks (Conclusion).
>> No assumption that >50% malicious workers can be tolerated (Lemma 1).

### Weaknesses
- Limited baseline comparison with Byzantine robust methods
>> Despite claims of incompatibility, no empirical or conceptual comparison with adapted DP defenses (Sec. 1), which would negatively impact the novelty claim and related practical justification.

- Insufficient quantification of false positives and their impact
>> Precision degradation noted in collusion experiments (Fig. 3b), but consequences such as reduced worker availability are not analyzed.
>> Validation loss alone may not reveal long-term optimization harm (Tabs. 2–3).

- Partial mathematical clarity and missing definitions
>> Distance measures in Eq. (2) was referenced in App. D, but no summary or concrete examples in main text.
>> Threshold adaptation method references App. Alg. 5 without high-level stability discussion (Sec. 3).
>> Overall, it feels that the lack of clarity weakens interpretability and reproducibility directly from the paper.

- Threat model assumptions not fully explored
>> Assumes no collusion between malicious workers yet collusion only tested up to 60% among attackers (Fig. 3b), without theoretical support.
>> First and last stage assumed to be honest (footnote 1), limiting generality for end-to-end secure decentralization.
>> Broader adversarial coordination strategies remain unexplored.

### Questions
In Section 3.2, Theorem 1 states that the convergence neighborhood size is proportional to ( \tau ). Could the authors provide a more precise relationship (e.g., a constant factor or specific bound) between ( \tau ) and the convergence error? This would help in tuning ( \tau ) for desired performance.

In Section 5.1, when integrating with SWARM, the paper mentions "32 trainer nodes with verification capability" but doesn't detail how the EMAs are synchronized across trainers. Could the authors clarify the synchronization mechanism and its impact on detection accuracy?

The paper uses multiple distance metrics (Appdix) but doesn't clarify how they are combined (e.g., majority vote, weighted average). Could the authors specify the combination strategy and its impact on detection performance? This would improve reproducibility and understanding of the method's robustness.

The ablation studies in Fig. 3 focus on warm-up steps, collusion, and delay, but do not explore the impact of varying EMA decay rates ((\beta_{h}) and (\beta_{g})). Could the authors provide additional experiments varying these hyperparameters to understand their effect on detection accuracy?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes SENTINEL, a lightweight verification mechanism for ensuring computational integrity in pipeline-parallel (PP) decentralized training, where traditional Byzantine-robust aggregation methods are inapplicable. SENTINEL introduces verifier nodes that monitor inter-stage activations and gradients using Exponential Moving Averages (EMAs) and IQR-based adaptive thresholds to detect anomalies caused by malicious workers. Experiments with Llama-3-0.6B and 1.2B models on decentralized frameworks (e.g., SWARM) show high (>90%) F1 scores in detecting various attack types with minimal overhead.

### Strengths
- Novel threat model: Addresses pipeline-parallel decentralized training security — an underexplored but increasingly relevant setting.

- Lightweight design: Verification via EMAs and statistical tests avoids costly redundancy or gradient aggregation.

- Comprehensive evaluation: Covers numerous attack types (activation, gradient, mixed, and adaptive attacks) across large-scale distributed setups.

### Weaknesses
- Trusted verifier assumption: SENTINEL depends critically on verifier nodes being honest and reliable. If a verifier node is compromised, it can both hide attacks and falsely flag benign workers, effectively collapsing the system’s security. The paper does not discuss mechanisms such as rotating verifiers, distributed verification, or cryptographic attestation to mitigate this.

- Incomplete threat model: The approach targets activation/gradient corruption but ignores broader adversarial behaviors such as data poisoning, backdoor insertion, or sybil collusion across multiple stages. These are common in decentralized systems and could bypass SENTINEL entirely.

- Limited evaluation scope: Experiments are conducted on medium-scale Llama-3 models (≤1.2B parameters) in simulated decentralized settings. It remains unclear how SENTINEL scales to truly large (>10B) models, heterogeneous networks, or high-latency cross-institutional environments where EMA synchronization could become costly.

- Parameter sensitivity and calibration cost: The verification relies on several empirically chosen hyperparameters (EMA decay rates, window size, IQR threshold k). These may require manual tuning per dataset and model. There is little analysis of robustness to these choices or automated adaptation beyond the initial “warm-up” period.

- False positives and training stability: While the paper reports high detection rates, it gives limited insight into false positive rates and the resulting training slowdowns or disruptions. Misidentification of honest nodes could degrade throughput or cause partial divergence in long training runs.

- Assumption-heavy theoretical guarantees: The proofs rely on simplified assumptions (e.g., independent random worker assignment, fixed detection thresholds). These are difficult to ensure in real decentralized networks, where collusion or heterogeneous bandwidth can break such guarantees.

### Questions
- How would the system behave if one or more verifier nodes are compromised or unavailable?

- Could the verification function be decentralized (e.g., through rotating verifiers or majority voting among neighboring nodes)?

- What is the quantitative computation and communication overhead of SENTINEL relative to redundancy-based baselines?

- How sensitive is performance to hyperparameter tuning (βₕ, βg, k, warm-up length)?

- Can the same framework detect semantic or backdoor-style attacks where activations remain statistically normal but maliciously biased?

- How generalizable is SENTINEL to other architectures (e.g., MoE models or non-transformer networks)?

### Soundness
2

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
The paper proposed SENTINEL, a verification mechanism for PP training without computation duplication. They provide theoretical convergence guarantees for this new setting that recovers classical convergence rates when relaxed to standard training. Experiments demonstrate successful training of billion-parameter LLMs across untrusted distributed environments with hundreds of workers while maintaining model convergence and performance.

### Strengths
1. The paper is claimed as the first comprehensive study of vulnerabilities unique to decentralized training with hybrid data–pipeline parallelism, and introduce a suite of training-interruption attacks that serve as benchmarks for evaluating the security of future systems.

2. The theoretical analysis demonstrates that undetected malicious workers have a negligible impact on the convergence properties. 

3. The authors integrate our method with SWARM parallelism to demonstrate its remarkable versatility in real-world decentralized training ecosystems.

### Weaknesses
1. The authors claimed that "the paper considered the first comprehensive exploration of secure and verifiable PP decentralized training
by identifying". As for me, In this setting we can see that we need to train billionparameter LLMs through internet-scale communication among distributed nodes.  The paper does not discuss all possible the topology of the inter-connected distributed notes.

2.  Due the issue listed above, the advantage of using decentralized training with hybrid data–pipeline parallelism is largely weaken. And the proposed EMAs is not persuasive. Please explain.

3. The generality of "Data and Pipeline Parallel Threat Model" is not justified. I am not sure why it is typical in real practice. 

4. Plus, the advantage of combing your scheme with SWARM needs to be discussed in many different scenarios.

### Questions
Please see in the weakness parts.

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
Decentralized training is an emerging field that makes LLM training more accessible by allowing research groups and volunteers to pool together available compute and, potentially, match the performance of centralized GPU clusters. This paper discusses the problem of making such system Byzantine-tolerant, i.e. robust to the presence of malicious participants that contribute incorrect results (e.g. to disrupt the training run or get the incentive from participation without actually contributing their compute).

While prior work addressed this problem in case of using data parallelism, it is crucial that modern LLMs have a large number of parameters and require some form of model parallelism to be trained. This paper addresses Byzantine tolerance in case of pipeline parallelism, which is known to be one of the most practical forms of model parallelism in decentralized setups due to its low bandwidth requirements.

The paper proposes to use EMA-based metrics to detect anomalies in activations and gradients passes through the pipeline stages. The authors prove that these metrics catch attacks that are large enough to significantly affect the validation loss. They also report experiments showing that this approach withstands multiple simple attacks that might be applied by malicious workers.

### Strengths
1. **Significance.** The paper discusses decentralized training, a promising approach to make LLM training accessible for small research labs, academic and individual researchers that don't have access to massive centralized clusters. The authors address the problem of Byzantine tolerance, which is known to be a major roadblock to adopting decentralized training.
2. **Originality.** The paper goes beyond most prior work and addresses Byzantine tolerance in case of using pipeline parallelism, which is known to be one of the most practical approaches to model parallelism for decentralized training systems due to its low bandwidth requirements.
3. **Practical solution.** Unlike prior work, the proposed solution doesn't require to allocate a substantial share of GPU compute to verification. Instead, it suggests to use cheap CPU nodes to detect anomalies in activations and gradients passes through the pipeline stages. The authors propose a straightforward way of integrating their method into existing decentralized training frameworks.
4. **Clarity.** The paper is well-written and describes the proposed algorithm in a clear way.
5. **Realistic training setup.** The authors report experiments with various simple attacks on a realistic distributed training setup.

### Weaknesses
1. **No results for adversarially designed attacks.** The authors only evaluate common generic attacks (L162-173, L480-481), such as sending constants, random values, or transformations of true activations/gradients. They don't evaluate adversarial attacks specifically designed to bypass the proposed method (e.g. by sending random data mimicking the tracked EMA-based metrics). It is difficult to infer bounds on their validation loss impact from the provided theoretical derivations.
2. **No results for medium-strength attacks.** Figure 1 features only strong attacks (F1 score > 0.8) that get caught and weak attacks (F1 score < 0.2) that don't impact training, with only one datapoint in between. This suggests that medium-strength attacks (F1 score ≈ 0.5) might still slip through and significantly impact the validation loss.
3. **Too strong, less realistic assumptions.** The paper assumes that malicious workers don't collude with each other (L161) and only perturb activations/gradients while sending them through pipeline stages, not during gradient aggregation (L321). It also assumes a small enough number of malicious nodes so that majority of workers holding each pipeline stage are honest with a high probabilty (L285).
4. **Accounting for gradient aggregation attacks.** While the authors claim that protecting from gradient aggregation attacks is a "complimentary axis" (L321), they don't discuss how to combine their method with protecting from such attacks, and how this affects the method's assumptions (e.g. for the number of malicious workers).

### Questions
1. How does the proposed method withstand specially designed adversarial attacks, e.g. if attackers send random data mimicking the tracked EMA-based metrics or use a small MLP instead of the proper pipeline stage to save compute?
2. What is the effect of medium-strength attacks (e.g. with F1 scores 0.3, 0.4, 0.5, 0.6) on the validation loss?
3. Given the effect of weak and medium-strength attacks, does decentralized training still make sense? (e.g. if we get the validation loss of a smaller model with 10x training compute, the participants might rather choose to train the smaller model locally)
4. How can we combine the proposed method with methods to protect from gradient aggregation attacks? How would this impact the maximum number of malicious workers the system can withstand?

### Soundness
2

### Presentation
4

### Contribution
3
