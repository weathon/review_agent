# Uncovering Competing Poisoning Attacks in Retrieval-Augmented Generation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Retrieval-Augmented Generation (RAG) systems improve the factual grounding of large language models (LLMs) but remain vulnerable to retrieval poisoning, where adversaries seed the corpus with manipulated content. Prior work largely evaluates this threat under a simplified single-attacker assumption. In practice, however, high-value or high-visibility queries attract multiple adversaries with conflicting objectives. Motivated by real cases, we introduce the setting of competing attacks, in which multiple attackers simultaneously attempt to steer the same (or closely related) query toward different targets. We formalize this threat model and propose competitive effectiveness, a metric that quantifies an attacker’s advantage under competition. Extensive experiments show that many strategies that succeed in the single-attacker regime degrade markedly under competition, revealing performance inversions and highlighting the limits of conventional metrics such as attack success rate and F1. Further more, we present PoisonArena, a standardized framework and benchmark for evaluating poisoning attacks and defenses under realistic, multi-adversary conditions. Our code is included in the supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors identify a novel and yet interesting research topic, which defines the competition among attackers of the same RAG system. A Bradley-Terry model is considered to formulate the success of attackers. Thorough experiments have been conducted, including single-attacker settings, competition settings, as well as their counterparts under the presence of recent defense methods. The results show interesting and unconventional results, where single attackers may not work well at the presence of other attackers. Furthermore, a new benchmark is proposed for future evaluations.

### Strengths
A novel research question is identified. The experimental results are interesting. Thorough numerical studies are conducted, including those under recent defenses.

### Weaknesses
However,  I identify several weakness:

(1) Although the problem of having multiple attackers is truly realistic, it is hard to formulate the problem into a research setting. In other words, while I appreciate the authors' attempt and great effort, several practical problems may still exist (and can hardly be formulated or assumed away). For example, it is totally understandable that simulations have to require all attackers have the same level of capability. In practice, however, some attackers may have higher level of access to the corpus, or higher rates of injecting new information into the database, or the capability to access the retriever and hence inject more valuable and likely retrieved documents, or the algorithm to identify high-value queries dynamically and more accurately.

(2) The novelty of the proposed method is rather weak. The novelty mainly lies in the novel research topic and the construction of the new benchmark. The execution of the idea, on the other hand, mainly relies on existing methods, including existing RAG attack algorithms and the Bradley-Terry method for attacker ability evaluation. If this is presented as a benchmark paper, it should be fine, though.

### Questions
What does it mean high-value queries? Practically how do we identify them?

It might be helpful to introduce briefly what the two datasets are, and why and how they can be used to evaluate the proposed attacker competition problem.

What's the baseline RAG system on which the attacks are compared? What's the retriever?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents the concept of competing attacks, a multi-adversary threat model for retrieval-augmented generation systems where multiple attackers attempt to influence the same query at the same time.

### Strengths
A novel competing attack is proposed to target the RAG system.

### Weaknesses
1. The threat model assumed in the paper is unrealistic.

2. The runtime of the proposed attack and all baseline methods is not evaluated.

3. Allowing five poisoned documents to be injected into the system per query is impractical.

4. The paper overlooks several recent attack and defense approaches.

### Questions
1. The paper assumes that the attacker either has access to a proxy retriever or can observe the retriever’s output, which is unrealistic. In real-world scenarios, it is nearly impossible for an attacker to obtain such information. Moreover, the authors’ statement that “the strength of the attack assumption is not the central focus of our study” is inappropriate. For an attack paper, the proposed method must demonstrate effectiveness under realistic threat models, as practicality and realistic assumptions are essential to evaluating attack feasibility.

2. Some attack methods, such as GASLITE, benefit from more advanced optimization mechanisms, while simpler baselines are under-optimized. This imbalance compromises the fairness of the comparisons.

3. Simulating pairwise competitions among all attackers across datasets is computationally expensive, yet the paper does not provide any runtime or efficiency analysis of the proposed framework.

4. The experimental setup allows five poisoned documents per query to be injected into the RAG system. This configuration is unrealistic because, as shown in [a][b], the number of truly relevant texts among the top-5 retrieved documents per query is typically fewer than five (e.g., in the NQ dataset). As a result, the number of poisoned documents exceeds the number of relevant ones, which artificially inflates the attack success rate. A more practical setting would restrict the attacker to injecting only one poisoned document per query.

5. Several recent and more advanced poisoning attacks on RAG systems, such as [a][c], are not included in the comparison. The authors should evaluate their method against these stronger and up-to-date baselines to demonstrate competitiveness.

6. The range of defenses examined in the paper is narrow. Additional robust defenses, such as [d][e], should be incorporated.


[a] Practical Poisoning Attacks against Retrieval-Augmented Generation.

[b] Benchmarking Poisoning Attacks against Retrieval-Augmented Generation.

[c] FlippedRAG Black-Box Opinion Manipulation Adversarial Attacks to Retrieval-Augmented Generation Models. 

[d] SafeRAG Benchmarking Security in Retrieval-Augmented Generation of Large Language Model.

[e] Traceback of Poisoning Attacks to Retrieval-Augmented Generation.

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
3

### Summary
This paper studies competing poisoning attacks on RAG systems, cases where multiple adversaries try to push conflicting misinformation into the same query. The authors show that methods strong in single-attacker settings often fail or flip rankings when attackers compete. They introduce PoisonArena, a benchmark with new metrics (m-ASR, m-F1, and a competitive coefficient) to measure attack strength under competition. Results show that real-world RAG security needs multi-attacker evaluation, not just single-attacker tests.

### Strengths
1. This paper is the first to study competing poisoning attacks in RAG systems, moving beyond the oversimplified single-attacker assumption.
2. It introduces PoisonArena, a well-structured evaluation framework for multi-attacker experiments.
3. This paper provides code, simulation details, and convergence criteria for reliable replication.

### Weaknesses
1.Although the paper presents the idea of competing attackers as a realistic scenario, likw political misinformation, product competition, the experiments remain entirely synthetic. All poisoned documents and queries are generated using LLMs rather than derived from real user-generated or adversarial data. This disconnect weakens the paper’s central claim that competing attacks mirror real-world threat. there is no evidence that such multi-party interference actually emerges in open systems.
2. The proposed simulation framework requires repeated pairwise competitions among multiple attackers across large datasets. This design is computationally heavy, yet the authors provide no discussion or measurements of runtime, memory cost, or scaling behavior. Without efficiency evaluation, it is unclear whether PoisonArena can scale beyond small experimental settings or be adopted for large-scale security testing.
3. One of the paper’s most interesting findings that "weaker single-attacker methods outperform stronger ones under competition "is treated descriptively, not analytically. The authors offer no clear theoretical reasoning or model of interaction that explains why this inversion occurs. Without such grounding, the finding risks being dataset or setup-specific rather than a general phenomenon.
4. The setting allows up to five poisoned documents per query gives each attacker excessive influence. In most RAG pipelines, the retriever only surfaces one or two truly relevant passages among the top-k results. Granting five injected documents per query likely inflates attack success rates and makes competition dynamics less representative of real conditions. A more realistic setting should restrict attackers to one or at most two poisoned documents per query.

### Questions
N/A

### Soundness
2

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
2

### Summary
This paper introduces the concept of "competing poisoning attacks" in RAG systems, where multiple adversaries simultaneously attempt to mnipulate the same query toward different, mutually exclusive targets. Additionally, they propose PoisonArena, a benchmark framework that evaluates poisoning methods under both single- and multi-attacker settings.

### Strengths
S1. There is a critical gap in existing RAG security research and this paper identifies it and fulfill this gap. The competing attacks is realisti and relevant forhigh-stakes queries in politics and public health.

S2. The evaluation is comprehensive covering seven attack methods, six LLMs and multiple datasets including (NQ, MS MARCO, mMARCO).

S3. The paper is well written and easy to follow.

### Weaknesses
W1. The choice of 8 incorrect answers per query seems arbitrary and lacks justification

W2. The convergence criterion based on ranking stability (Equation 7) could be sensitive to the choice of r (consecutive rounds)

W3. The specific prompt templates for generating adversarial content may introduce biases

### Questions
Please refer to Weakness part

### Soundness
3

### Presentation
3

### Contribution
2
