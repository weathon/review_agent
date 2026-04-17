# Probabilistic Audits for Verifiable Training and Outcome Improvement in Decentralized Learning

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Decentralized training of large models presents two critical verification challenges: ensuring the training process was executed correctly (process verification) and confirming the resulting model genuinely improved (outcome verification). Existing solutions like zkML are prohibitively expensive, while prior Proof-of-Learning schemes focus only on the process, failing to guarantee that the final model is actually better. We introduce a comprehensive and efficient framework that addresses both challenges through economically-secured probabilistic audits. First, we propose a protocol where Provers commit to each training step, with a small, random fraction of steps being audited by verifier committees, and we derive a tight detection-cost frontier that minimizes verification overhead. Second, we introduce Proof-of-Improvement (PoI), a novel and lightweight evaluation audit that statistically certifies milestone-based gains (e.g., perplexity reduction) on a committed dataset. Empirically, on a QLoRA fine-tuning task, our process audits reduce verification compute by over 95\% compared to full replication, and our PoI audits certify model improvements with high statistical power at a minimal cost.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a framework to verify large‐scale, decentralized model training in process verification and outcome verification. The core idea is that the prover (trainer) commits cryptographically (e.g., via Merkle trees of model states, metadata) to every training step in a “window.” Then a small random subset of steps is audited (parameters / metadata are revealed and the step is re‐executed) while the remainder remain un‐opened. This gives a probabilistic detection guarantee: if a step is malicious, there is a chance it will be audited and caught. They further propose a “Proof-of-Improvement” (PoI) mechanism to statistically verify model performance improvement (e.g., reduction in loss/perplexity) on a committed evaluation set. The authors argue this probabilistic auditing reduces verification cost dramatically while still achieving high assurance. The paper also sketches how this works in federated/multi‐trainer settings with stakes and committees.

### Strengths
The framework acknowledges multiple trainers, aggregation, stake‐based incentives, committees of verifiers and public commitments (e.g., using a blockchain or open ledger) which are realistic touches for distributed training accountability.

The authors clearly define the desiderata (process + outcome), highlight limitations in previous work (e.g., zkML impractical, prior PoL lacking outcome guarantee) and situate their contributions accordingly.

### Weaknesses
Inadequate defence against single‐step cheating: The key method relies on auditing a random fraction of steps. But the paper does not convincingly show how the threat of single‐step malicious deviation is sufficiently mitigated. If a trainer knows audit probability is small and can craft a malicious step that yields large damage (e.g., insertion of poisoned gradient), the scheme appears vulnerable. The paper acknowledges “a small probability of undetected cheating” but does not clarify what types of attacks can be thwarted, nor how the damage of undetected cheat is bounded.

Privacy concerns and metadata explosion: To enable re‐executing audited steps (and to commit to all steps in a window), the protocol demands storing/committing large volumes of metadata: full parameter snapshots, optimizer state, random seeds, etc. This risks training privacy (especially in federated or proprietary settings). The paper does not fully address how to hide sensitive parameters or ensure privacy while enabling re‐execution.

Ambiguity of the threat/attack model and use‐case scope: It is unclear for which threat scenarios the scheme is designed. For example: Is it aimed at malicious trainer shortcuts (skipping epochs), data poisoning, gradient inversion, or backdoor insertion? The paper does not clearly map out which attacks are prevented and which are outside its scope. Without this clarity, the value of the scheme is hard to assess (“what exactly am I defending against?”).

Lack of formal guarantees or proofs: While the detection‐cost frontier is a useful start, I did not find a formal theorem bounding undetected cheating probability and bounding damage from cheating. Without such formalism, the security claim remains heuristic.

Clarity and writing issues. Some language is informal to me (e.g., on p.6: “...1 − (1 − alphaq)^f …”), which detracts from readability and professionalism.

### Questions
Please carefully consider my concerns listed in the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a scheme to verify the integrity of central and collaborative model training processes. They adopt a Proof-of-Learning (PoL)-style approach, where model trainer(s) make commitments to parameter updates in each round, which can later be opened to verify integrity of the training process. The paper differentiates itself from previous work with two primary points: (i) a cut-and-choose-style probabilistic verification scheme, where trainers commit to all updates and then a subset of them are randomly sampled for verification, which decreases verification runtime substantially; (ii) a verification of model utility improvement at regular milestones in training, as evaluated by a validation set.

### Strengths
The ingredients of the paper are well-motivated. Probabilistic PoL verification is a sensical approach to reducing audit costs. Relaxation of PoL security is better motivated in the distributed setting than single-prover, where the impact of forgeries by adversaries can be diluted by honest workers. Proof of Improvement is a sensical heuristic for assessing model utility during training.

### Weaknesses
- **Protocol details are unclear.** The individual pieces make sense, but the paper is in need of an end-to-end specification of the protocol(s) in algorithm box(es). It's not clear to me whether the authors intend only to develop a protocol for the distributed training setting, or whether a single-prover variant is also intended. The details given about the multi-prover setting also have me a bit confused (see Questions section).
- **Weak soundness condition, consequences not explored.** Definition 1 only says that an adversary's likelihood of being caught in the probabilistic PoL increases with the proportion of updates that they cheat on. But it seems plausible to me that an adversary can have a large impact on the model parameters while cheating on only a few steps -- *especially* in the single-prover setting, but potentially in the multi-prover setting too. The authors should present some analysis about whether this is problematic for their framework. 
- **Novelty of PoI overstated.** Analogous approaches, which seek to assess the utility of updates in distributed training based on a reference dataset, exist in previous work e.g. [1].
- **Evaluation.**
	1. It is unclear to me how the authors are evaluating the impact of $\tau$ in Figure 2 (Middle). How can probability of forgery detection be expressed as a function of $\tau$ without taking into account the adversary's choices?
	2. The cost analysis would benefit from concrete runtimes, and comparison to a baseline with no auditing, rather than just costs normalized against vanilla PoL. It's clear that the probabilistic approach reduces the cost of replay, but I'm curious about how the cost of commitment compares to the overall training cost as well. 
	3. Table 2 is never referenced in the text.
	4. The impact of PoI is not very clear from the presented results. It's unclear what advantage PoI is obtaining for the framework. Table 3 is never referenced in the text. 
	5. Evaluation of the distributed training is terse -- the implications of the experiment are very unclear. I would recommend at least including some supplemental figures to show the implications of the experiment.

[1] FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping. Cao et al. 2020.

### Questions
- The authors should briefly explain Merkle trees in the preliminaries section.
- In Section 2.3, it says that Stage 1 of the multi-prover case only verifies whether aggregation was carried out correctly, and the PoL part only triggers if aggregation is carried out incorrectly. I am confused by this -- why would dishonestly forging a model update cause *aggregation* to be computed incorrectly? What stops an adversary from picking an arbitrary local model and submitting it for averaging? Wouldn't the global model still be computed correctly in this case?
- Aside from the above question, I am confused about why only a random subset of workers is selected for auditing in Stage 2. What happens if the adversary simply isn't sampled in the subset? I see analysis of the probability of catching a cheating step within the PoL audit, but it's not clear to me where the probability of sampling the adversarial worker is factored into the soundness analysis. 
- When during the training protocol is the Proof of Improvement invoked? What happens if it is not satisfied? Is there any connection to the PoL parts? 
- How is $P_{\tau-\text{miss}}$ computed given $\tau$?
- Proof of learning has been shown to have some important flaws -- see [2]. Are those flaws relevant to the present work? If not, why not?

[2] Proof-of-Learning is Currently More Broken Than You Think. Fang et al. 2022.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a practical way to verify decentralized model training that checks both how training was done and whether the resulting model actually got better. For the process, a trainer posts a cryptographic commitment after each step; once a window of steps is committed, a small, randomly chosen committee replays only a sampled subset of those steps to confirm they were executed correctly. If a supermajority of the committee rejects a step, the trainer is penalized; otherwise the protocol moves on. This “commit to sample to reveal to audit” pipeline is designed to keep wall-clock overhead low and to align incentives so honest behavior is the best strategy.

### Strengths
- Good incentive: the free rider problem is serious, especially in anonymous decentralized learning groups.
- Covers both process and outcome. Adding PoI addresses a central gap in PoL-style schemes: certifying that the model actually improved, not just that steps were executed.

### Weaknesses
- Narrow empirical scope. Validation is on a fine-tuning workload and a toy distributed run; large-scale training or heterogeneous, adversarial deployments remain untested.
- Commit/reveal overheads ignored in core cost curve. The analysis emphasizes verifier cost; Merkle commitments over full model states and revealing optimizer state during audits can be substantial (compute + bandwidth), especially with Adam-style ones.
- Security model is mostly static. Resilience to adaptive adversaries (e.g., post-selection committee corruption within the finalization window) is discussed qualitatively, not proven; protocol assumes timing bounds and reliable public randomness.
- Metric coverage in PoI is limited. PoI currently targets log-loss/perplexity; extension to non-differentiable or sequence-level metrics (safety, alignment) is left open.

### Questions
- The linear law hinges on q. In open networks, estimating global capture F/M, calibrating τ across heterogeneous stacks, and guaranteeing honest majorities per window are non-trivial. How robust is q if verifiers churn or share hardware quirks?
- The incentive condition $s_p > \frac{1-\alpha q}{\alpha q} G$ presumes a well-specified per-step gain (G). In practice, cheat payoffs can be spiky (e.g., grabbing a full round’s reward), making per-step G under-specified and potentially under-collateralized. 
- Reveal bandwidth and timing. The liveness bound treats audit latency as constant per window, but revealing optimizer state and parameter shards could dominate wall-time under tight networks or large G; does the pipeline still mask that latency at scale?
- The claimed factorization $\delta_{\text{PoI}}=\delta_{\text{stat}}(n)\cdot q$ assumes i.i.d. token-level differences; with temporal or topical correlation, effective sample size drops and testers might cherry-pick spans unless the evaluation root and sampling procedure are rigorously enforced.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a practical and efficient framework to verify decentralized model training by addressing two core challenges: process correctness and outcome improvement. For process verification, it proposes a probabilistic audit system where Provers commit to every training step, but only a small, random fraction of steps are later sampled and re-computed by verifier committees. This "commit-sample-reveal" method is shown to reduce verification compute costs compared to full replication while still economically deterring cheating. To solve outcome verification, the paper presents a novel and lightweight statistical audit called Proof-of-Improvement, which allows the network to efficiently certify that the final model achieved a claimed performance gain on a committed evaluation sets.

### Strengths
- Originality: The key original idea is Proof-of-Improvement (PoI). Prior work only verified the process of training. This paper adds verification for the outcome that the model actually got better. The combination of probabilistic process audits with outcome audits is also novel.
- Quality: The paper has strong theoretical and empirical support. It proves a clear detection-cost frontier, showing the exact tradeoff between cost and security. 
- Clarity: The paper is clear. The Commit-Sample-Reveal protocol is explained simply and is easy to follow.
- Significance: This work is highly significant. It makes verifiable training practical and economically viable, not just a theoretical idea. PoI correctly aligns incentives. It rewards better models, not just burning compute. The system is designed for real-world use, including an extension for multi-prover distributed training.

### Weaknesses
- The experiments are convincing but narrow. They focus on a QLORA fine-tuning task, not large-scale pre-training. It's unclear how the protocol's overhead and savings would translate to a scenario with vastly more steps and data. The multi-trainer experiment was also a small-scale "toy run" just to prove the concept, not its performance at scale.
- The texts in Figure 1 left is barely legible without zooming in, same for the legends in Figure 2. 
- The paper emphasizes verifier savings but doesn't measure the Prover's costs. The Prover must compute a Merkle root over the entire model state for every single step, which could be computationally expensive. Furthermore, the "Reveal" phase requires transmitting large witnesses (like Adam optimizer states), creating a potential communication bottleneck that isn't measured.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3
