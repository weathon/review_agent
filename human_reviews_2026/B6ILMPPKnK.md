# Fine-Grained Privacy Extraction from Retrieval-Augmented Generation Systems by Exploiting Knowledge Asymmetry

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by incorporating external knowledge bases, significantly improving their factual accuracy and contextual relevance. However, this integration also introduces new privacy vulnerabilities. Existing privacy attacks on RAG systems may trigger data leakage, but they often fail to accurately isolate knowledge base-derived content within mixed responses and perform poorly in multi-domain settings. In this paper, we propose a novel black-box attack framework that exploits knowledge asymmetry between RAG systems and standard LLMs to enable fine-grained privacy extraction across heterogeneous knowledge domains. Our approach decomposes adversarial queries to maximize information divergence between the models, then applies semantic relationship scoring to resolve lexical and syntactic ambiguities. These features are used to train a neural classifier capable of precisely identifying response segments that contain private or sensitive information. Unlike prior methods, our framework generalizes to unseen domains through iterative refinement without requiring prior knowledge of the corpus. Experimental results show that our method achieves over 90\% extraction accuracy in single-domain scenarios and 80\% in multi-domain settings, outperforming baselines by over 30\% in key evaluation metrics. These results represent the first systematic solution for fine-grained privacy localization in RAG systems, exposing critical security vulnerabilities and paving the way for stronger, more resilient defenses.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a black-box attack framework designed to perform fine-grained privacy extraction from Retrieval-Augmented Generation (RAG) systems. The authors identify two weaknesses in existing privacy attacks: they are "coarse-grained," and they perform poorly in multi-domain settings.

The idea of the proposed attack is to exploit the "knowledge asymmetry" between a RAG system and a standard LLM. A RAG system has access to an external, private knowledge base, while a standard LLM does not. The authors leverage this difference as a signal to isolate and extract private content.

### Strengths
- The idea of using a standard, non-retrieval LLM to measure the "knowledge asymmetry"  is intuitive.

- The experimental setup is thorough and robust.

### Weaknesses
- In Section 4.3, it says "we examine each generated sentence R_i. If R_i appears in any retrieved text T_j , we label its similarity feature score S_i with y_i = 1; if it does not appear in any top-k texts, we assign y_i = 0." However, an RAG system's LLM may paraphrase or synthesize information from retrieved chunks without verbatim copying a sentence. This synthesized text is still a private data leak, but this method would incorrectly label it y_i = 0.
Therefore, the paper does not precisely identify private or sensitive information.

- In Table 1, overall performance of the method is presented. using different LLMs of RAG, the results are not consistent across different datasets. More discussions should be included.

- The paper claims the framework generalizes without requiring prior knowledge. However, in Page 5, it says "This template is designed to incorporate keywords likely to appear in the knowledge base, such as “heart failure", “stroke", and “liver cirrhosis" in a medical knowledge base." This requires significant prior knowledge and contradicts the claim.

- There are no experiments on privacy-preserving techniques like Differential Privacy. The discussion on Differential Privacy is purely speculative. The authors' claim that their attack might still work against moderate DP lacks empirical support.

### Questions
See comments.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a highly innovative black-box attack framework for fine-grained privacy extraction from Retrieval-Augmented Generation (RAG) systems. The method leverages knowledge asymmetry between RAG and standard LLMs, automatically generates adversarial queries, and combines semantic similarity with NLI-based reasoning and a neural classifier to localize private information at the sentence level. Extensive experiments on single- and multi-domain datasets show significant improvements over baselines in extraction accuracy and F1-score.

### Strengths
⦁	Highly innovative use of knowledge asymmetry for privacy extraction.
⦁	Fine-grained, automated, and black-box privacy localization.
⦁	Combination of semantic similarity and NLI reasoning improves robustness.
⦁	Strong experimental results, especially in multi-domain settings.
⦁	Method generalizes to unseen domains and does not require prior knowledge of the corpus.

### Weaknesses
⦁	Some experimental details are lacking (e.g., classifier training, negative sampling, ablation on NLI/scoring).
⦁	The method's dependence on the standard LLM as a reference may limit applicability if the LLM is weak or misaligned.
⦁	No discussion of potential countermeasures or how defenses might adapt to this attack.
⦁	Limited analysis of failure cases or scenarios where the method may not work well (e.g., high overlap between RAG and LLM knowledge).

### Questions
⦁	How sensitive is the method to the choice of standard LLM? What if the LLM is outdated or domain-mismatched?
⦁	Can the authors provide more details on the classifier architecture and training process?
⦁	Is there an ablation study on the contribution of the NLI model and similarity scoring?
⦁	How does the method perform if the RAG system uses strong privacy-preserving retrieval or response filtering?

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
4

### Summary
This paper proposes a framework for fine-grained extraction of RAG systems under a black-box setting. The method utilizes "knowledge asymmetry" to identify different responses from RAG and LLM, and trains a DNN to identify the private sentence in the RAG database.

### Strengths
1. This work focuses on the practically significant issue of privacy leakage in RAG systems.

2. Experimental results demonstrate the clear effectiveness of the proposed method.

### Weaknesses
1. In the multi-domain RAG setting, should it be assumed that the attacker is aware of the domain topic of the RAG database and constructs the "initial questions" (Algorithm 1) based on this knowledge? What if the topic of the RAG database does not fall within the ten broad topics defined in the prompt template, such as *Harry Potter* or *Pokémon*? Given that the multi-domain RAG databases used in the experiments (Sec. 5.1) actually overlap with the ten broad topics, the assumption that the attacker is unaware of the domain topic should be questioned. 

2. The paper does not clearly discuss **whether the proposed method remains effective under existing RAG privacy defense mechanisms**. Prior work has explored defenses such as intention detection [1] and output similarity-based leakage detection [2]. The adversarial query generation strategy in this paper shows identifiable patterns, which may be easily caught by keyword-filter-based adaptive defenses.

3. The method relies on training a DNN to detect whether sentences contain private data from the knowledge base. However, it is unclear what dataset is used for training. If the training dataset and the evaluation set in Section 5.1 come from different domains, could it impact the method’s performance?

4. Editorial error: incorrect citation formatting.

**Minor concerns:** 

(1) The method's success needs the potential assumption that the output distribution and internal knowledge of the LLM used in the reference model and the target RAG system are similar. However, LLMs with different parameters and cutoff dates may contain different knowledge. It is better to analyze whether model size or training time affects the results and what trends emerge. 

(2) Using only 30 queries per database may not be enough to provide a reliable evaluation.

Ref.

[1] Zhang Y, Ding L, Zhang L, et al. Intention analysis makes llms a good jailbreak defender[J]. arXiv preprint arXiv:2401.06561, 2024.

[2] Zeng S, Zhang J, He P, et al. The good and the bad: Exploring privacy issues in retrieval-augmented generation (rag)[J]. arXiv preprint arXiv:2402.16893, 2024.

### Questions
See weaknesses

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
This paper investigates privacy leakage in Retrieval-Augmented Generation (RAG) systems and proposes a knowledge-asymmetry-driven black-box attack framework to achieve fine-grained privacy extraction across multi-domain knowledge bases. The key idea is to exploit semantic divergences between a RAG system and a standard LLM to identify which response segments originate from private knowledge bases. Experiments on single-domain (HealthCareMagic, Enron Email) and multi-domain (NQ) datasets demonstrate strong performance, achieving over 90% extraction success rate (ESR) in single-domain and 80% ESR in cross-domain settings, outperforming baselines (RAG-Privacy, RAG-Thief) by more than 30% in key metrics.

### Strengths
1. Technically sound and comprehensive pipeline – The three-phase framework (query generation, NLI-based semantic scoring, and classification) is methodically designed and experimentally validated, with convincing improvements over baselines.
2. Strong empirical evaluation and generalization – The work thoroughly evaluates across diverse datasets, LLMs, and retrievers, demonstrating robust results and good generalization to multi-domain scenarios.

### Weaknesses
1. Lack of clear explanation of the multi-domain solution in the Introduction – Although the paper claims to address privacy extraction in multi-domain RAG systems, the Introduction primarily focuses on the challenge of fine-grained privacy localization and does not clearly articulate the core idea or mechanism that enables effective cross-domain adaptation. The proposed iterative query refinement strategy, which appears central to solving the multi-domain challenge, is not conceptually introduced or motivated early in the paper. A clearer exposition in the Introduction explaining how the proposed framework overcomes the difficulties of scattered, heterogeneous knowledge across domains would make the contribution more convincing.
2. Limited discussion of the multi-domain challenge – The paper briefly mentions that “in cross-domain data, the scattered knowledge and wide topic range make it difficult to construct targeted adversarial queries, significantly reducing attack effectiveness,” but does not expand on the nature of this challenge or why the proposed iterative refinement can effectively address it. A more explicit explanation would improve clarity and motivation.
3. Unclear handling of outdated or missing knowledge in LLM responses – If the user query involves knowledge absent or outdated in the LLM training data, it is not clear how the framework fuses or balances results from the LLM and RAG components. Discussion of this case would strengthen the framework’s practical robustness.

### Questions
see the weakness

### Soundness
3

### Presentation
2

### Contribution
2
