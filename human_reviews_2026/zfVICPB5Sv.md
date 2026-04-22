# Silent Leaks: Implicit Knowledge Extraction Attack on RAG Systems

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by incorporating external knowledge bases, but this may expose them to extraction attacks, leading to potential copyright and privacy risks.
However, existing extraction methods typically rely on malicious inputs such as prompt injection or jailbreaking, making them easily detectable via input- or output-level detection. 
In this paper, we introduce **I**mplicit **K**nowledge **E**xtraction **A**ttack (**IKEA**), which conducts *Knowledge Extraction* on RAG systems through benign queries.
Specifically, **IKEA** first leverages anchor concepts—keywords related to internal knowledge—to generate queries with a natural appearance, and then designs two mechanisms that lead anchor concepts to thoroughly "explore" the RAG's knowledge:
(1) Experience Reflection Sampling, which samples anchor concepts based on past query-response histories, ensuring their relevance to the topic; 
(2) Trust Region Directed Mutation, which iteratively mutates anchor concepts under similarity constraints to further exploit the embedding space.
Extensive experiments demonstrate **IKEA**'s effectiveness under various defenses, surpassing baselines by over 80% in extraction efficiency and 90\% in attack success rate. Moreover, the substitute RAG system built from **IKEA**'s extractions shows close performance to the original RAG and outperforms those based on baselines across multiple evaluation tasks, underscoring the stealthy copyright infringement risk in RAG systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents IKEA, a method that extracts knowledge from RAG using queries. IKEA stays stealthy by creating natural queries built from anchor concepts. The method has two parts. Experience Reflection Sampling chooses concepts that are likely linked to the RAG’s internal knowledge based on past query results. Trust Region Directed Mutation changes anchor concepts within a set similarity range to find new and related information more effectively. Experiemnts show that IKEA performs much better than other methods. The extracted knowledge can also be used to build a working substitute RAG system.

### Strengths
- The paper studies an important security issue in RAG systems : extraction attacks. Its focus on harmless-looking queries makes it different from most past work.

- IKEA method is explained in a clear and direct way.  Figure 1 gives a clear summary of the process.

- Experiments use several settings. Results show that IKEA keeps high EE and ASR while passing basic defenses. This is a strong finding.

- Code is provided. It seems make sense.

### Weaknesses
- The tested defenses  are not enough. Stronger and more realistic defenses include semantic output filtering, consistency checks, detection of repeated probing, or methods for iterative query attacks.

- I am not sure about the main assumption that the RAG topic is fixed and known limits how well the method can be used in other cases. 

- The results is not enough to support the claim that the substitute RAG performs “comparably” (Sec 4.5).  Three metrics cannot measure many other aspects .

- The cost in time, API calls, and total query rounds isnot clear. This may make the attack too expensive for extracting large knowledge bases.

### Questions
1. See weakness

2. The topic probing method seems crucial for practical applicability. Could you provide more details on its robustness?

3. While IKEA avoids malicious prompts, could the pattern of queries generated be detectable by analyzing query sequences over time using anomaly detection techniques?

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
4

### Summary
This paper studies extraction of documents from a RAG knowledge base. Instead of using malicious queries, the authors use benign queries repeatedly to collect RAG answers as stolen knowledge, and propose several tricks to improve search efficiency—e.g., avoiding duplicate retrievals and increasing coverage. Experiments evaluate metrics including extraction efficiency and the downstream performance of RAG systems reconstructed from the stolen knowledge.

### Strengths
1. The paper addresses an important problem by studying the privacy risks of RAG systems under more realistic settings—specifically, black-box access with defenses in place.  
2. Compared with baselines, the proposed method demonstrates stronger robustness against defended RAG systems, successfully extracting more knowledge when defenses are applied.  
3. The paper is well-written, and the overall idea is intuitive and easy to follow.

### Weaknesses
1. The idea of using query–response semantic distance as a proxy for local RAG density is based purely on intuition, without further discussion. The paper does not provide references or experiments to validate this assumption.  
2. The evaluation includes only two baselines, while several other relevant methods are mentioned but not compared experimentally.  
3. The extracted documents achieve low ROUGE scores (below 0.3, Table 1), indicating that the extracted content fails to accurately recover the original documents. This limits the practical implications for privacy or copyright concerns.  
4. Some metric definitions are unclear. For example, *extraction efficiency* depends on the number of “unique” extracted documents, but the notion of uniqueness is not specified. Moreover, since the method does not reconstruct original documents, comparability of this metric with prior work is questionable. Similarly, the definition of *ASR*—the ratio of non-rejected queries—does not directly measure extraction success.  
5. The proposed method introduces many hyperparameters (over ten), which may be difficult to tune in practice. The paper provides little discussion on how these parameters are chosen.  
6. Ablation results show only marginal improvements over random baselines (Table 13), particularly for ASR, CRR, and SS metrics, raising concerns about the actual effectiveness of the proposed approach.

### Questions
In the evaluation, some methods such as DEGA achieve high ROUGE scores (up to 0.96 in Table 1) in the no-defense setting, suggesting near-literal copying. However, their embedding similarity remains relatively low. What are the possible reasons?

### Soundness
2

### Presentation
3

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
The paper proposes IKEA, a “benign-query” knowledge-extraction attack on RAG. It combines (i) Experience Reflection sampling over “anchor concepts” and (ii) Trust-Region Directed Mutation (TRDM) to explore the embedding space, and evaluates against RAG-Thief and DGEA under input/output defenses, reporting higher extraction efficiency and attack success.

### Strengths
1. The paper is well-written and easy to follow

2. The studied topic is important and novel

### Weaknesses
1. The paper evaluates against only two prior attacks, RAG-Thief (prompt-injection) and DGEA (jailbreak), even though the Related Work section lists additional, closely related extraction methods (e.g., Pirates of the RAG / adaptive black-box extraction) that are not included as baselines. This makes the claimed superiority (“surpassing baselines by 80%+”) hard to trust. At minimum, strong black-box, non-jailbreak/PIK variants and adaptive coverage attacks should be implemented. More discussion on the related works is needed.

2. “Semantic Similarity (SS)” uses an encoder to compare outputs with retrieved docs, favoring paraphrase-style extraction (IKEA) over verbatim baselines, while CRR (ROUGE-L) penalizes paraphrase. Claims hinge on SS/EE/ASR; there is no human audit of copyright risk nor independent leakage criteria. Copyright/privacy stakes aren’t well reflected by SS alone.

3. HealthCare-100k, HarryPotterQA, and Pokémon are niche; Pokémon is explicitly chosen as low-overlap with pretraining. Results may not generalize to enterprise RAG (contracts, support logs, medical records), where policy, formatting, and noise differ.

4. The main setup assumes a known domain topic; the “unknown topic” setting still uses a bespoke topic-probing stage powered by a secondary LLM, then evaluates almost identically—this weakens the claim that IKEA remains benign and practical under stricter assumptions.

5. Replacing Top-K with off-topic docs predictably tanks both the attack and benign utility to near zero (Table 4), which is not an acceptable real-world mitigation, so it doesn’t inform deployers what works.

6. The pipeline and equations are clear, but the headline claim (“surpassing baselines by >80% efficiency, >90% success”) rests on a baseline set that is neither representative nor matched to IKEA’s benign-query regime. Without stronger baselines, the empirical claim reads overstated.

### Questions
1. Add competitive benign-query baselines: random/diversity sampling; k-center or farthest-point query selection; BM25 lexical sweeps; self-ask/chain-expansion; an adaptive coverage agent; and a re-implementation of adaptive black-box extraction from the works already cited in §5.

2. at least one enterprise-style corpus with policy/PII-like structure, and long-document settings that stress retrieval/reranking.

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
This submission investigates covert extraction of proprietary knowledge from retrieval-augmented generation (RAG) systems and proposes “IKEA,” an implicit, benign-query attack that grows “anchor concepts” via history-aware sampling and a trust-region mutation in embedding space. Evaluations across several corpora and model/retriever pairings suggest higher extraction efficiency than prompt-injection baselines and show that a substitute RAG assembled from harvested content retains non-trivial utility. Topic probing for unknown domains and simple adaptive/DP-style defenses are also explored to characterize security–utility trade-offs.

### Strengths
1. This paper clearly specifies a realistic black-box threat model for RAG and delineates attacker capabilities and constraints with precision.
2. Empirical coverage is broad, spanning multiple LLM–retriever configurations and defenses, and the attack remains effective when common jailbreak/prompt-injection attacks are blocked.
3. The method is straightforward and reproducible—anchor-based benign queries guided by history-aware sampling and a cosine-bounded trust-region mutation—with prompts and hyperparameters disclosed.

### Weaknesses
1. Algorithmic novelty feels limited; the core components amount to history-penalized sampling and cosine-bounded mutations without formal coverage or sample-complexity guarantees. 
2. This paper depends on a known or easily probed domain topic and centralized corpus semantics, making generalization to heterogeneous, multi-topic enterprise deployments uncertain.
3. This paper’s defense study leans on simplistic or utility-destroying mechanisms and omits deployable strategies like per-client rate limiting, query-set anomaly detection, and semantic drift monitoring.
4. This paper lacks an end-to-end economic analysis of the attack (token/time costs and sensitivity to generator quality), which is crucial for real-world risk assessment.

### Questions
No more questions.

### Soundness
2

### Presentation
3

### Contribution
2
