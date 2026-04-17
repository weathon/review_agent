# LLMs Leak Training Data Beyond Verbatim Memorization via Membership Decoding

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Extracting training data from large language models (LLMs) exposes serious memorization issues and privacy risks. Existing attacks extract data by generations, followed by membership inference. However, extraction attacks do not guide such generations, and the extraction scope of member data is limited to the greedy decoding scheme. Only verbatim memorized member data is being audited in this process. And a majority of member data remains unexplored, even if it is partially memorized. In this work, we define a new notion of memorization, $k$-amendment-completable, to measure the degree of partial memorization. Greedy decoding can only extract 
$0$-amendment-completable sequences, which are verbatim memorized. To address the limitation in generation, we propose a membership decoding scheme, which introduces membership information to guide the generation process. We formulate the training data extraction problem as an iterative member token inference problem. The token distribution is calibrated with membership information at each generation step to explore member data. Extensive experiments show that membership decoding can extract novel member data that haven't been studied before. The proposed attack manifests that the privacy risk in LLMs is underestimated.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
I list the main contributions of this work as following:

- Introduction of a new memorization notion termed "k-amendment-completable," which quantifies partial memorization by measuring how many tokens require amendment during greedy decoding to generate the actual training sequence. 
- A membership decoding framework that treats the training data extraction problem as an iterative sequence of token-level membership inference attacks. Rather than performing membership inference after generation, this approach affects the generation process itself using membership information.
- A novel token-level membership inference attack score based on maximizing the posterior probability of observing the member prefix, which calibrates token distributions using reference models and unifies several existing MIA methods under a common framework.

### Strengths
- This paper builds on a strong theoretical framework for understanding partial memorization in language models with proper mathematical explanation
- Identifying and quantizing non-verbatim memorization has been an important problem statement in AI security and privacy for long and this paper tackles this exact problem, making this work significant for the field
- Proposes a new way to identify non-verbatim memorization in LLMs, overcoming one of the primary limitations that of attacks which use greedy decoding only

### Weaknesses
- There is a lack of ablations in this study and the evaluation is restricted to the Pythia model family, which represents older and smaller-scale architectures compared to contemporary models
- Table 2 is very unclear, why are most of the positions are blank, these missing results seriously puts into question the validity of the results of this study

Overall, this research paper presents a great novel idea and framework but fails to present solid empirical evidence for it's effectiveness. The scope of the experiments is very limited and the results presented do not seem strong enough.

There is a HARD setting mentioned in the paper, however, no results could be found for this setting in the paper

### Questions
Kindly let us know if you could present any more results/values to support this paper

Could you provide a reasoning for the somewhat similar performance of this method to Minus on k=1 for HackerNews, Pile-CC

Why weren't other open-source models like OLMo et cetera tried?

### Soundness
1

### Presentation
2

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
The paper considers the problem of extracting training data from Large Language Models (LLMs), following the way of first reconstructing sequences then checking their membership in the training data. The proposed method is to introduce membership information into the decoding process to guide the generation of training sequences, such that the generated sequences are more likely to be in the training data. The authors define a new concept, k-amendment-completable, to quantify the degree of partial memorization.

### Strengths
- The studied problem is important and relevant.
- The paper is generally well-written.

### Weaknesses
- The definition of k-amendment-completable may not fully capture the memorization behavior of LLMs.
- The approach of breaking down sentence-level membership inference into token-level inference may lead to ambiguities.

### Questions
The definition of k-amendment-completable does not fully capture the memorization behavior of LLMs, as it fails to account for the length of the prefix. In the extreme case where the prefix comprises almost the entire sequence except for one token, even a 0-amendment-completable sequence does not necessarily indicate that the model has memorized it.

Decomposing sentence-level membership problems into token-level ones does not seem reasonable. Consider the following scenario: LLMs are trained on massive text corpora in which many sentences share common prefixes. For example, the prefix “In conclusion,” could appear in numerous different sentences. If we attempt to infer membership at the token level, the next token following “In conclusion,” could vary widely depending on the specific sentence. Therefore, inferring membership at the token level may lead to ambiguous or incorrect conclusions about whether the entire sentence was part of the training data. How do the authors address this potential ambiguity in their token-level membership inference approach?

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
This paper introduces a novel approach to extract training data from large language models (LLMs) beyond verbatim memorization. The authors define a new notion of memorization called "k-amendment-completable" to measure the degree of partial memorization. They propose a membership decoding scheme that guides the generation process to extract non-verbatim memorized data by leveraging membership information at each generation step. The paper demonstrates that larger models memorize more training data than smaller models and shows that their membership decoding approach can extract novel member data that hasn't been studied before. The authors also introduce a new evaluation framework that measures extraction accuracy based on k values.

### Strengths
S1: The paper introduces a novel concept ("k-amendment-completable") that provides a fine-grained measure of partial memorization, addressing a significant gap in the literature beyond verbatim memorization.

S2: The membership decoding scheme is well-motivated and theoretically sound, providing a systematic way to extract non-verbatim memorized data by incorporating membership information at each generation step.

S3: The paper provides empirical evidence that larger models memorize more training data than smaller models, which is an important finding for understanding privacy risks in LLMs.

S4: The evaluation framework (measuring extraction accuracy by k values) provides a more nuanced understanding of what data can be extracted from LLMs, moving beyond traditional verbatim extraction.

S5: The paper makes a compelling argument that a majority of member data remains unexplored even if it's partially memorized, which significantly expands the scope of privacy risks in LLMs.

### Weaknesses
W1: The paper lacks sufficient comparison with existing methods that aim to extract non-verbatim memorized data, making it difficult to fully assess the novelty and superiority of the proposed approach.

W2: The evaluation is limited to a few datasets (HackerNews, Pile-CC, PubMed Central, ArXiv) and model sizes (1B, 1.4B, 2.8B, 6.9B, 12B), which limits the generalizability of the findings to other LLM architectures and training data.

W3: The paper doesn't thoroughly discuss the practical implications of the proposed attack for real-world privacy risk assessment, particularly how the extraction accuracy translates to actual privacy risks in deployed models.

W4: The theoretical justification for the membership decoding approach could be strengthened with more detailed mathematical analysis and comparison to related work.

W5: The paper doesn't address potential defenses against the proposed attack, which would provide a more complete picture of the privacy implications.

### Questions
Q1: Could you provide a more detailed comparison between your membership decoding approach and existing methods for extracting non-verbatim memorized data? This would help clarify the novelty and advantages of your approach.

Q2: How would the proposed method perform on a wider range of datasets and model architectures beyond those tested in the paper? A more comprehensive evaluation would strengthen the generalizability of your findings.

Q3: Could you explore the practical implications of your findings for real-world privacy risk assessment? How do the extraction rates at different k values translate to actual privacy risks in deployed LLMs?

Q4: How does the proposed membership decoding approach scale with model size and complexity? A more detailed analysis of the computational cost and time requirements would be valuable for practical implementation.

Q5: Could you discuss potential defenses against the proposed attack? This would provide a more complete picture of the privacy implications and help guide future work on privacy-preserving LLMs.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the training data leakage risk in Large Language Models (LLMs). The authors point out that existing extraction attacks (e.g., Carlini et al., 2021) heavily rely on "verbatim memorization" and "greedy decoding," which significantly underestimates the true privacy risk of these models. To address this, the paper presents two core contributions: (1) A new memorization metric, "k-amendment-completable," to finely quantify "partial memorization." (2) A new attack framework, "Membership Decoding," which reframes extraction as an "in-generation," iterative, "token-level membership inference" (MIA) problem, calibrated by a reference model. Experiments on Pythia demonstrate that this method can successfully extract partially memorized sequences (where the 'k' value is 1 or 2) undiscoverable by greedy decoding, confirming a broader data leakage risk.

### Strengths
- Paradigm Shift
- Fine-grained & Effective Metric
- Solid Formulation
- Strong Empirical Evidence

### Weaknesses
- Limited Attack Scope
- Dependency on Heuristics
- Need for Reference Model

### Questions
The authors have proposed a novel and important framework for evaluating and extracting "partially memorized" data from LLMs, which is crucial for understanding their privacy boundaries. The paper is well-argued, the experiments are solid, and this work opens up a new and valuable research direction.

However, I do have the following comments:
- The main limitation is the scope of the 'k' value. As shown in Table 2, the effectiveness is currently almost exclusively limited to 'k=1, 2'. I hope the authors can discuss in more detail the core challenges of extending this to larger 'k' values. Is it merely because the signal weakens as 'k' increases? Or is there a combinatorial explosion problem? Suggestion: Have the authors considered combining "Membership Decoding" with Beam Search, retaining the 'B' highest-MIA-score candidate sequences at each step, rather than just the Top-1?
- The attack's effectiveness relies heavily on the assumption that the correct member token must be within the Top-20 most probable tokens. This is a strong constraint. Suggestion: Could the authors add an analysis in the appendix examining what proportion of failed 'k=1, 2' extractions were due to the true member token falling out of this Top-20 set? This would help us understand the bottleneck of this heuristic.
- The necessity of a reference model (Pythia-170m) limits the attack's universality in a fully black-box scenario. Suggestion: Have the authors considered (or do they plan for future work) reference-free calibration methods? For example, using the target model itself with dropout, or approximating the denominator (the probability of the token appearing) using token statistics from a large corpus.
- The choice of 'a=0.5' is presented as a "trade-off." Could the authors provide a sensitivity analysis for 'a' (e.g., across 0, 0.25, 0.5, 0.75, 1.0)? This would make the robustness of Equation 8 more convincing.

### Soundness
3

### Presentation
3

### Contribution
4
