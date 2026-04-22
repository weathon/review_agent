# TRUST: A Decentralized Framework for Auditing Large Language Model Reasoning

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 4, 2, 4

## Abstract
Large Language Models (LLMs) can produce complex reasoning chains, offering a window into their decision-making processes. However, verifying the quality (e.g., faithfulness and harmlessness) of these intermediate steps is a critical, unsolved challenge. Current auditing methods are often centralized, opaque, and struggle to scale, creating significant risks for the deployment of proprietary models in high-stakes domains. This paper addresses four key challenges in reasoning verification: (1) *Robustness*: Centralized systems are single points of failure, vulnerable to attacks and systemic bias. (2) *Scalability*: The length and complexity of reasoning traces create a severe bottleneck for human auditors. (3) *Opacity*: Internal auditing processes are typically hidden from end-users, undermining public trust. (4) *Privacy*: Model providers risk intellectual property theft or unauthorized model distillation when exposing complete reasoning traces. To overcome these barriers, we introduce TRUST, a decentralized framework for auditing LLM reasoning. TRUST makes the following contributions: (1) It establishes a decentralized consensus mechanism among a diverse set of auditors, provably guaranteeing audit correctness with up to 30\% malicious participants and mitigating single-source bias. (2) It introduces a scalable decomposition method that transforms reasoning traces into hierarchical directed acyclic graphs, enabling atomic reasoning steps to be audited in parallel by a distributed network. (3) All verification decisions are recorded on a transparent blockchain ledger, creating a permanent and publicly auditable record. (4) The framework is privacy-preserving by distributing only partial segments of the reasoning trace to auditors, thus protecting the full proprietary logic from distillation. We provide theoretical guarantees for the security and economic incentives of the TRUST framework. Experiments across multiple LLMs (e.g., GPT-OSS, DeepSeek-r1, Qwen) and reasoning tasks (e.g., mathematical, medical, science, and humanities) demonstrate that TRUST is highly effective at identifying reasoning flaws and is significantly more resilient to corrupted auditors than centralized baselines.  Our work pioneers the field of decentralized AI auditing, offering a practical pathway for the safe and secure deployment of AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors argue that verifying the quality of reasoning chains produced by LLMs has challenges related to robustness, scalabiilty, transparency, and privacy. To resolve those challenges, this work seeks to create a decentralized framework for auditing LLM reasoning. The features of this framework, and the intended novel contributions of the work, include (1) a decentralized consensus mechanism that leverages diverse auditors to assess audit correctness (2) a scalable decomposition method that converts the reasoning traces into hierarchical cyclical graphs, thereby enabling individual reasoning steps to be audited by distributed networks (3) blockchain recording of all verification decisions, purportedly permanent and auditable public record (4) a privacy-preserving quality the framework due to the fact that it distributes only partial segments of the reasoning trace to auditors, thus guarding the full logic. Lastly, the authors run a set of experiments across several LLMs such as Deepseek-r1 and several reasoning tasks (medical, etc.) to demonstrate their assertion that TRUST accurately identifies flaws in reasoning and, additionally, is more resilient to auditor corruption than centralized baselines are.

### Strengths
-The work's motivation appears largely justified, though I believe certain areas can be strengthened and prior work can be presented in a more compelling way (more on this later)
- The combination of these elements appears to overcome the identified challenges and appears novel in light of the prior work 
- The experiments are sound and appear to prove that the proposed framework helps overcome the identified challenges

### Weaknesses
- Some highly relevant, recent prior works are relegated to the appendix or are absent. A few examples would include Leng et al. 2025 (Semi-structured LLM Reasoners Can Be Rigorously Audited) and Lanham et al.. (2023) (Measuring faithfulness in chain-of-thought reasoning). Bringing these into the Introduction and making them more of a focus of the discussion would strengthen the motivations and help distinguish this work from the prior work. 
- This work might benefit from a more direct and possibly visual comparison (e.g., using a table) to prior or contemporary work on methods for improving trace auditability. 
- Some of the citations in the introduction do not appear to support the sentence that precedes them. An example would be the citation of Peng et al. to back up the assertion that "However, these advances lack systematic verification mechanisms for generated reasoning traces, particularly for privacy-preserving and decentralized auditing (Peng et al., 2025)." It is not clear to me how Peng supports this assertion. I would double check these citations and/or spend time on better alignment of the citations with the asserts they are intended to support. 
- The blockchain aspect of the project, while intriguing, appears to be left out of the evaluation and, in general, receives light treatment. If anything, perhaps it is worth discussion how the long-term success of this feature might be evaluated in the future.

### Questions
-What are some other competitive, contemporary solutions to this problem and how does your proposal offer advantages over them? 
-How would you evaluate the value provided by the blockchain features of the framework?

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
3

### Summary
The authors propose a decentralized auditing framework for validating the reasoning trace of Large Reasoning Models (LRMs). This framework involves breaking apart a large _reasoning trace_ into a DAG structure that can be broken into smaller sub-traces that can be verified by individual human/automated auditors. The authors provide mechanisms to ensure that the system is robust against dishonest auditors through an incentive mechanism design. The authors also provide experimental results to validate the effectiveness of their proposed system.

### Strengths
- This paper tackles an important problem (LLM auditing) and provides a novel solution that allows for decentralized / human-machine collaboration in the auditing process.
- Breaking down the trace into atomic segments using the HDAG structure is novel and interesting.
- The incentive mechanism design and its analysis is a very nice addition bridging economics theory and practice.
- The experimental results show that the proposed system is indeed effective against "flipped" auditors also shows guarantees of saftey and economic viability. Actual human auditors is also a very nice addition.

### Weaknesses
- While the framework itself sounds nice, there needs to be more consideration of practicallity of such a system. The addition of blockchain seems superficial in the current work. Modern blockchain systems still suffer from scalability issues (i.e. high latency, gas price etc). A more detailed analysis of the actual system performance is needed to understand if this is a viable solution.
  - If this system is to be actually deployed on-chain, I think the incentive analysis needs to include gas prices -- who will cover it? how do we ensure that we have enough to cover the cost of storing and verifying the HDAGs on-chain (or IPFS)?
- The tokenomics section is very hand-wavy and lacks details. A new role is introduced (the "delagator") but never actually explained in detail. Why do the delagators stake in the pool? (they probably gain money by delegating but that's not made clear) What if the delagator + auditor pairs are dishonest and try to game the system? I think this section opens up more questions than it answers.
- While the authors claims that this system enables privacy, how exactly this is achieved is not very clear.

### Questions
- How do we tell if certain auditors are dishonest? Is it if they disagree with the majority of other auditors?
- How are each segments tagged (for which type of auditor)? Is this something also decided by the same model doing HDAG desconstruction?
- What exactly is posted on-chain? Is it the case that individual (encrypted) segments are posted on-chain and the auditors are assigned/delegated these segments and decode them off-chain?
  - If this is the case, does that mean all the auditors need to be pre-approved and will have access to the decryption keys? Then how do we ensure privacy?
  - Or is it that they post their proof of work on-chain? or both? Either way, it's not very clear how this works/enables privacy.
- is there a coordinator in this system? Who assigns the auditors to each segment? Or is it done in a decentralized way by the delegators? Then we must think about the allocation process -- first come first serve? Weighted (trust core? stakes?) allocation? Auctions? 

Nits:

- IPFS is never actually defined.
- The experimental section (Sec. 4) mentions that we will see some "privacy results". But I don't see any privacy analysis in the results section.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces TRUST, a decentralized framework for auditing LLM reasoning. It breaks a model’s chain of thought into a hierarchical DAG (goal, strategy, tactic, step, operation) and assigns each fragment to auditors that include automated checks, LLMs, and humans. Auditors evaluate fragments independently, use a commit and reveal protocol, and a weighted consensus yields a global verdict. Results are recorded on chain and content shards remain off chain to protect proprietary logic. An incentive layer with staking, rewards, and penalties is provided, along with statistical guarantees for honest participation. Experiments on reasoning tasks and a small human in the loop study suggest better robustness than single LLM or centralized baselines.

### Strengths
The paper explores a novel and timely direction by focusing on decentralized, semantics-level auditing of LLM reasoning.

The work presents solid empirical evaluations alongside clear theoretical analysis, including statistical guarantees and an incentive model.

### Weaknesses
The paper frames its contribution as auditing the semantics of the reasoning process, yet the practical objective in most deployments is high-quality, policy-compliant outputs, which are fully observable to end users. The authors do not convincingly justify why auditing internal reasoning semantics is necessary or preferable to auditing outcomes and observable behaviors. If the real concern is billing fairness or “token inflation,” the proposed framework does not verify usage-based charges and cannot attest to whether the billed tokens or hidden steps are legitimate. If the concern is compliance or safety, the system does not demonstrate that semantic checks on intermediate steps provide stronger guarantees than established output-level methods (e.g., post-hoc verification, tool/API traces, or red-team evaluation). In short, the paper conflates distinct goals (output quality, cost accountability, and compliance), while the proposed method directly addresses none of them end-to-end; as a result, the core motivation for reasoning-semantics auditing remains unclear.

The evaluation uses only open-source models and public datasets, with no evidence that the approach works for commercial API models or production settings where chain-of-thought is hidden. There are no case studies or end-to-end integrations with real providers, no measurements under API constraints such as latency, cost, and quotas, and no demonstration that the HDAG workflow or consensus protocol remains feasible when only partial traces or output-level logs are available.

### Questions
Can you show end-to-end evidence that the system is practical at production scale, using realistic workloads and deployments?

What are the latency curves compared with strong output-level baselines across varying task sizes and auditor configurations?

What are the throughput curves under the same comparisons and settings?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes TRUST, a decentralized framework for auditing large language model reasoning traces that tackles four core challenges: robustness, scalability, opacity, and privacy. TRUST decomposes chain-of-thought into Hierarchical Directed Acyclic Graphs across five levels and routes segments in parallel to heterogeneous auditors. It records votes via a commit–reveal protocol on a blockchain with PoS-style incentives while storing content off-chain on IPFS, so auditors see only partial segments and cannot reconstruct proprietary logic. The authors provide theoretical guarantees, including correctness with up to 30% malicious participants and a safety-profitability theorem that rewards honest behavior. Empirically, TRUST outperforms centralized and single-LLM baselines across models and tasks; in a human-in-the-loop GSM8K study it achieves F1 = 0.89 vs. human audit F1 = 0.77, and random/fixed segmentation variants degrade to F1 = 0.40, underscoring the value of HDAG decomposition.

### Strengths
1. The paper introduces a decentralized framework that decomposes reasoning into five-level HDAGs and routes segments to heterogeneous auditors, enabling parallel, modular verification while recording outcomes on-chain and storing raw traces off-chain—balancing scalability with transparency.
2. TRUST uses segmentation so each auditor only sees partial context, plus a commit–reveal voting scheme and PoS-style incentives on a blockchain.
3. The framework claims correctness with up to ~30% malicious participants and offers a safety-profitability guarantee.

### Weaknesses
1. Narrow empirical scope of the human study. The only human-in-the-loop experiment involves 15 PhD students auditing 10 GSM8K math problems, which limits external validity across domains, task types, and auditor populations.
2. The design relies on IPFS, a blockchain ledger for immutable records, and a commit–reveal voting protocol. The paper does not report end-to-end latency, throughput, or cost under realistic workloads, leaving deployability uncertain.
3. Although the paper states experiments span diverse datasets and models, the primary detailed human study centers on short math problems; evidence for robustness on longer-horizon or code/science workflows with richer dependencies is limited.
4. The empirical evaluation benchmarks TRUST mainly against single-LLM auditors, a centralized human audit, and ablations of TRUST itself, rather than conducting end-to-end comparisons with established alternative routes. As a result, the relative efficacy, latency, and operational cost trade-offs remain unclear.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
