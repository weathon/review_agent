# LiRA: Linguistic Robust Anchoring for Cross-lingual Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
As large language models (LLMs) rapidly advance, performance on high-resource languages (e.g., English, Chinese) is nearing saturation, yet remains substantially lower for low-resource languages (e.g., Urdu, Thai) due to limited training data, machine-translation noise, and unstable cross-lingual alignment. We introduce LiRA (Linguistic Robust Anchoring for Large Language Models), a training framework that robustly improves cross-lingual representations under low-resource conditions while jointly strengthening retrieval and reasoning. LiRA comprises two modules: (i) Arca (Anchored Representation Composition Architecture), which anchors low-resource languages to an English semantic space via anchor-based alignment and multi-agent collaborative encoding, preserving geometric stability in a shared embedding space; and (ii) LaCR (Language-coupled Semantic Reasoner), which adds a language-aware lightweight reasoning head with consistency regularization on top of Arca’s multilingual representations, unifying the training objective to enhance cross-lingual understanding, retrieval, and reasoning robustness. We further construct and release a multilingual product retrieval dataset covering five Southeast Asian and two South Asian languages. Experiments across low-resource benchmarks (cross-lingual retrieval, semantic similarity, and reasoning) show consistent gains and robustness under few-shot and noise-amplified settings; ablations validate the contribution of both Arca and LaCR. Code will be released on GitHub and the dataset on Hugging Face.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes LiRA, a framework designed to enhance cross-lingual robustness and reasoning of LLMs, especially in low-resource languages. The method consists of two complementary modules: Arca,anchors LRL embeddings to an English semantic space via critic–actor interaction and feature anchoring. LaSR, a lightweight reasoning head using queue-based consistency regularization to unify retrieval and reasoning objectives.

The authors also introduce a new dataset, LazRetrieval, covering seven Southeast and South Asian languages. Empirical results show consistent improvements in retrieval, ranking, and reasoning tasks across multiple benchmarks compared to strong baselines like Qwen3 and MindMerger. Theoretical analysis provides formal stability and fidelity guarantees for cross-lingual representations.

### Strengths
I think the dual-path anchoring design and critic-guided optimization are innovative, combining LLM reasoning with multilingual robustness, and the queue-based optimization (CorrQueue and DocQueue) is an elegant solution to stabilize ranking and retrieval training with limited multilingual data.  And the theoretical framework provide a solid foundation for this design.
And the evaluation is done across multiple tasks (retrieval, semantic similarity, reasoning, comprehension) and compared with multiple baselines (Qwen3, MindMerger, LUSIFER), and witnessed a consistent improvement.

### Weaknesses
I have some concern about the experiment results: 
1. Despite architectural and theoretical novelty, performance improvements (e.g., +1–2% over Qwen3) are modest, with no significant test.
It is unclear whether such gains justify the additional training complexity and computational cost introduced by LiRA. 
2. The selection of different Qwen models is not justified: for retrieval and sentence ranking Qwen3-Embedding-8B is used as the
backbone encoder. For reading comprehension and mathematical reasoning Qwen3-4B is used as the backbone model (I am not sure if there is a typo here, Qwen3-4B in Sec 5.1 or Qwen3-14B in Table 3). 
3. The reproducibility is also questionable: Although code and dataset release are promised, key hyper-parameters (critic weights α, β, γ, δ) and their tuning procedures are insufficiently detailed. No mention of training cost, convergence behavior and inference speed.

### Questions
See weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes the LiRA framework to mitigate performance degradation in large language models across low-resource languages. By anchoring low-resource language representations to the English semantic space, the framework aims to preserve the model's English reasoning capabilities. Its core comprises two components: ARCA reduces semantic divergence through feature alignment and translation selection, while LaSR integrates multilingual and English representations for retrieval and ranking. The paper provides theoretical guarantees for representation stability and validates the method's effectiveness across multiple cross-lingual tasks. It also releases a new multilingual retrieval dataset for further research.

### Strengths
1. The paper provides rigorous mathematical theorems and derivations, offering theoretical guarantees for the stability and error bounds of the proposed framework.

2. The evaluation encompasses three core tasks, including retrieval, ranking, and reasoning (mathematics, reading comprehension) which could enable a comprehensive validation of the method's generalizability.

3. The experiment selected state-of-the-art models (such as the Qwen3 series and LUSIFER) as baselines for comparison, enhancing the credibility of the results.

4. Contributed the LazRetrieval dataset, covering seven under-resourced Southeast Asian and South Asian languages, to support community research.

### Weaknesses
1. The theoretical framework relies on strong assumptions such as the existence of “perfect translations” and “semantic anchoring,” conditions that are difficult to fully satisfy in real-world, noisy low-resource language scenarios. This may lead to a gap between theoretical guarantees and actual performance.

2. LiRA integrates ARCA (containing multiple critics and actuators) and LaSR (containing multiple queues and loss functions), featuring numerous modules and a complex training workflow.

3. The performance of the ARCA module heavily relies on the quality of candidate translations generated by upstream translation models such as Qwen3-32B. For extremely low-resource languages, if the translation model itself performs poorly, errors propagate downstream, potentially limiting the effectiveness of LiRA.

### Questions
1. Why choose Qwen3 as the nearly exclusive LLM foundation? All experiments are based on the Qwen3 series of models (Embedding and LLM). Does the performance of this approach heavily depend on Qwen3's inherently strong cross-lingual capabilities during pretraining? If we switch to other LLMs with weaker cross-lingual capabilities during pretraining, would LiRA's relative gains change?

### Soundness
3

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
The study is motivated by the language imbalance in LLM pretraining data. They propose LiRA, a framework aimed at improving cross-lingual representations, which anchors multilingual representations to English representation space using multiple LLMs for translation. These candidate translation embeddings are combined with the multilingual text embeddings through an adaptor trained through reinforcement learning using an LLM-as-a-judge framework. In the LaSR module, text is encoded in the source and target language and concatenated before being processed by the LLM. The authors claim guarantees for robust representations, even under translation setting. The framework is evaluated on retrieval and reasoning benchmarks, including a new retrieval dataset covering low-resource languages, Belebele, MLQA, STS2, MGSM, and X-CSQA.

### Strengths
- The paper investigates a very relevant topic of improving LLM performance on low-resource languages by improving latent representations
- The study presents and interesting idea to concatenate representations from different models (using different tokenizers) to improve multilingual representations
- The presented results seem promising

### Weaknesses
- The approach claims robustness in the multilingual representations but cannot support this claim with sufficient evidence. The presented equations include various assumptions on LLM embeddings and machine translation models that are unrealistic.
- The empirical results presented in Tables 1 and 2 cannot be compared. The inference resources needed for the translation alone extends beyond the inference budget of related approaches. Please provide a detailed comparison of the training and inference compute budgets for all benchmarked methods.
-  For a fair model comparison, I suggest running pass@k (k should be set to the number of LLMs inferenced for one LiRA forward pass) for all benchmarked approaches that rely on a single LLM forward pass at test-time
- A comprehensive overview of the individual modules in related works is largely missing. Parts of the framework were introduced in related works:
  - Kim, S., Ki, D., Kim, Y., & Lee, J. (2023). Cross-lingual QA: A Key to Unlocking In-context Cross-lingual Performance. arXiv preprint arXiv:2305.15233.
  - Villa-Cueva, E., López-Monroy, A. P., Sánchez-Vega, F., & Solorio, T. (2024). Adaptive cross-lingual text classification through in-context one-shot demonstrations. arXiv preprint arXiv:2404.02452.
- The LiRA module are not well motivated or embedded in related works. The proposed method does not address the imbalance between multilingual and English representation spaces, but makes multilingual text processing more reliant on machine translation and English-centric LLMs. 
- The manuscript is not well structured and hard to follow. The discussion section only describes the results in the tables but does not include a detailed analysis linking back to the motivation of the paper and method.

### Questions
- typo line 53
- typo line 122
- typo line 354
- typo lines 368-369
- introduced LRL multiple times, I.e. lines 36, 121
- All citations use the inline-citation style.
- Please elaborate on how your approach differs from related approaches such as MindMerger.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This work addresses the known problem of performance disparity of LLMs when applied to low-resource languages when compared to high-resource languages like English. To solve this issue, the work introduces LiRA, a comprehensive training framework designed to improve the quality of cross-lingual representations for low resource languages. LiRA consists of 2 modules: Arca which anchors LRLs to the robust English semantic space and LaSR which adds a lightweight reasoning head to increase consistency regularization. The paper  also introduces and releases a new multilingual product retrieval dataset, LazRetrieval, which covers seven underrepresented South and Southeast Asian languages. As part of the paper, the authors introduce a theoretical framework to understand cross-lingual representations and provide a proof that model learns "high-fidelity (robust) representations that effectively support downstream tasks" with some assumptions. The authors conduct an experimental evaluation which includes retrieval, mathematics and comprehension tasks on 7 datasets (2 new ones) which include a variety of low resource languages from South Asia and Southeast Asia and find that LiRA consistently outperforms prior works like MindMerger and LUSIFER.

### Strengths
- Novel method that achieves "state of the art" performance on a wide variety of benchmarks.
- A theoretical analysis to get deeper insight into the training architecture.

### Weaknesses
- Gains on the benchmarks do not seem substantial. For example, on retrieval tasks, LiRA improve Qwen3-E-8B but the gains are fairly small (~+1). Given the complexity of the method, this seems very small.
- Given the use of translators in Arca that are significantly larger, its not clear if the baselines used are comparable.
- It is nice that an ablation study is done but its unclear if it makes sense given that Qwen3-E-8B already works so well.

### Questions
- Can the authors justify their choice of a baseline and how this method compares?

### Soundness
1

### Presentation
1

### Contribution
2
