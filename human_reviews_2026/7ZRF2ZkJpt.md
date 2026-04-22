# Knowledge-Centric Data Selection for Effective Domain Adaptation of Language Models

- Avg Score: 4.40
- Decision: Reject
- Scores: 2, 4, 4, 8, 4

## Abstract
Data filtering remains a fundamental challenge in training large language models (LLMs). Large-scale corpora often contain noisy or redundant samples that undermine training efficiency and limit model performance. Existing approaches typically rely on manual curation or heuristic model-based filtering, yet lack a principled and widely accepted quantitative criterion for deciding which data should be retained. To address this gap, we propose an entropy-based data filtering framework that quantitatively evaluates the informativeness and coverage of individual samples. Our method enables the systematic selection of high-value data, improving the efficiency of supervised fine-tuning (SFT) while also enhancing retrieval quality in retrieval-augmented generation (RAG). These results highlight the effectiveness of entropy-driven filtering as a general strategy for improving both adaptation and retrieval in large-scale LLM pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Authors propose a method for selecting the best data points (from a larger set) for LLM finetuning. Their method starts by creating a matrix of document-concepts, and then selecting a subset of the documents that best covers the available concepts. They use an LLM to extract the mentioned concepts in every document, and then use an optimization algorithm to select the documents such that when taken together they meet a certain criterion (the equation mentioned in Subsection 2.4.1). In their optimization algorithm they use a weighting scheme to prioritize the concepts.

The method is evaluated in two tasks and compared to three baselines, it shows some improvements.

They also report the loss curves (in MATH-500) and a comparison between their method and random sampling based on their entropy across concepts in a RAG dataset.

### Strengths
-  The method is easy to understand, and the paper is easy to follow
-  The method is evaluated in two tasks
-  The improvements are noticeable

### Weaknesses
- In my opinion the method proposed by authors is basically a sampling method based on clustering. The clusters are created using what authors call concepts, and then the the sampling is done over the data points that fall in the area shared between the clusters. No discussion in this regard is done in the paper, no analysis, and no comparison is reported in the paper. There is a ton of research on data sampling dating back to 2000s. The core idea of the paper is basically a data sampling--which is applied to LLM fine tuning. A proper comparison and analysis is missing. 
- The method seems ad hoc. Examples: 1) to extract the concepts, another LLM is needed to extract the concepts! In real world if we had such a model, why would we need to do domain adaptation? 2) Line 104: what is a "coverage probability"? is it a standard mathematical term? 3)  Line 146, on what basis did you select concept frequency? 
- The analysis section of the paper is very shallow. No ablation study is reported, no sensitivity report to the quality/quantity of the concepts. No report on the length of the selected documents. etc. Very uninformative section.

### Questions
None.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Entropy-Driven Data Selection (EDS), which optimizes data subset selection for diverse and high-quality datasets. It also introduces a metric called Knowledge Coverage Entropy (KCE), which measures the knowledge diversity in a data subset. EDS chooses a subset that maximizes KCE with a lazy-greedy algorithm. The paper provides experiments in SFT and RAG to show the possible use cases for EDS.

### Strengths
- Overall, the method seems sound. For the right choice of knowledge set and data labeling approach, KCE is a reasonable metric to optimize over.
- EDS is applicable across a downstream use cases, including SFT and RAG.

### Weaknesses
1. My main concern with this paper is how the performance is evaluated. The metrics are computed using the hit rate for the specific knowledge set that the KCE is optimized on. To me, this seems like (especially in the baseline comparisons) this is not a fair performance comparison, since EDS is specifically optimized for this knowledge set K, whereas other approaches are not. Could the authors clarify why these metrics can be used to compare to baseline approaches? 
2. I am not convinced that prompting an LLM to tag concepts without any additional filtering is the ideal approach. For example, in Appendix D, both “diabetes” and “Diabetes” are listed as separate concepts, which should probably not happen for the optimal knowledge set. The paper would be strengthened by an ablation with additional approaches to select the knowledge set. It would also be useful to see how sparse the data are across the set of concepts. What is the distribution of number of documents for each concept?
3. The experiments are limited to a single language model and only one dataset for SFT and two datasets for RAG. The experimental results would be more convincing with experiments for additional models and datasets.

### Questions
Please see weaknesses above. 

Minor comments: 
- In reference to Table 2, the authors say that EDS “consistently achieves the highest coverage and ranking quality across all scales” However, superfiltering outperforms EDS for multiple dataset sizes for both cases. It would be good to discuss this in the text. It would also be useful to clarify somewhere that the bolding corresponds to the authors’ method (rather than the highest performing method), since this is somewhat misleading at first glance.
- Figure 1 is very low-resolution and should be a pdf

### Soundness
2

### Presentation
2

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
In this paper, the authors first revealed that SOTA data selection methods often fail to address domain-specific redundancy and interference, leading to inefficient training and models that overfit to frequent linguistic forms rather than core knowledge. This inefficiency leads to limited generalization, as well as high curation and computational costs. To address this challenge, the authors proposed knowledge coverage entropy (KCE), a metric quantifying knowledge diversity, and entropy-driven selection (EDS), which optimizes data selection for compact, high-quality datasets. KCE quantifies diversity and balance over discrete knowledge units, and EDF prioritizes novel, high-information samples to reduce redundancy.

### Strengths
1. The paper is well-written and easy to follow.

2. EDS leverages the submodular property of KCE, ensuring the lazy-greedy algorithm achieves a near-optimal (1-1/e) solution.

3. The proposed framework can perform well both on SFT and RAG.

### Weaknesses
1. How to build KCE is the critical contribution of this paper. In building KCE, the authors formed a matrix B. A research paper must be self-contained, while the authors placed the most important content about how to build matrix B in the appendix.

2. To build matrix B, the authors use Qwen-max to determine the value of knowledge points. While the authors did not validate whether this procedure is fair, e.g., different LLMs may prefer different knowledge points. The authors should conduct human expert validation to make this procedure fair.

3. [1] has demonstrated that data selection methods can not always work well in real-world scenarios, which often needs to tackle datasets containing more than 1 million data points. In this paper, the authors only tested the proposed framework on 10k-level datasets, which make the empirical findings less convincing.

4. In this paper, the authors claimed that SOTA data selection methods have issues of inefficiency, and this issue will lead to poor generalization. While in the experiment parts, the authors did not design experiments to illustrate the improved generalization of the proposed framework.

[1] Xia, T., Yu, B., Dang, K., Yang, A., Wu, Y., Tian, Y., ... & Lin, J. (2024). Rethinking data selection at scale: Random selection is almost all you need. arXiv preprint arXiv:2410.09335.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper tackles the challenge of domain adaptation for language models, focusing on how to select really effective training or retrieval data for specialized applications without just scaling up dataset size. Rather than using typical heuristics like perplexity pruning or embedding clustering, the authors introduce a knowledge-centric framework. they define discrete knowledge units in the domain, then use Knowledge Coverage Entropy as a central metric to measure data diversity and informativeness. The core method is Entropy-Driven Selection, which uses submodular optimization to find compact, knowledge-rich, balanced subsets for either SFT or RAG. Results on the MATH-500 reasoning benchmark, as well as medical RAG datasets, show marked efficiency and performance gains, the chosen subsets with KCE/EDS either match or beat both manual curation and strong baselines with less data and faster convergence, and retrieval quality is significantly improved.

### Strengths
First to formalize knowledge coverage entropy (KCE). focuses on actual knowledge points in a dataset rather than token distributions or clusters. Generalizes to both SFT and RAG, shown on math reasoning and large medical RAG corpora with practical benefits i.e less data, faster learning, better retrieval. Outperforms strong baselines (random, curation, recent filtering/clustering approaches) in both accuracy and sample efficiency. submodular lazy-greedy methods make such selection scalable for big corpora. Discussion and theory are both included, with testable predictions and practical winning points. Candid about assumptions and current limitations, laying groundwork for extensions.

### Weaknesses
Current method assumes redundancy. it may not be as helpful when knowledge is highly sparse or nearly all samples are unique.The binary, all-or-nothing knowledge coverage misses partial overlaps. a probabilistic coverage or semantic similarity extension could broaden applicability.Computation of the knowledge-to-sample matrix and knowledge point extraction may require strong domain-specific labeling or prompt engineering. Experiments are strong on math/medical, but more domains e.g. legal, scientific, engineering data, or multilingual settings would further cement generality.Some technical formulas and submodular optimization details are pretty dense.

### Questions
Do you plan to extend the method to domains where knowledge units overlap or have graded/partial relevance, not just binary? How would you integrate semantic similarity (rather than only strict concept match) into KCE or EDS? Have you identified any failure cases where EDS actually underperforms random or simpler heuristics in non-specialized domains?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a knowledge-centric approach to domain adaptation, redefining data quality as the joint consideration of both diversification and coverage over domain-specific knowledge. To this end, the authors design Knowledge Coverage Entropy (KCE) as a metric to measure such data quality, and a corresponding selection algorithm named Entropy-Driven Filtering (EDF) to filter data accordingly. Experiments are conducted on MATH500 and a self-constructed RAG task with a designed metric. The results show that the proposed method outperforms all baselines and achieves the highest knowledge coverage rate.

### Strengths
1. The paper is focused and specific, addressing the problem of domain-specific data selection with clear motivation.
2. It provides a novel perspective on data selection by introducing Knowledge Coverage Entropy (KCE) as a measure of data quality.
3. The proposed KCE formulation is straightforward, and the corresponding algorithm (EDF) is easy to reproduce.

### Weaknesses
1. It is unclear how the proposed method can be applied to both SFT and RAG. While I can understand the use of KCE for SFT data selection, its application to RAG is confusing. Does it mean reducing the candidate retrieval corpus size? This point needs clarification.
2. The method requires prior access to the entire sample set in order to compute coverage entropy, which is impractical in many open-domain applications.
3. The paper lacks insight into the internal working mechanisms between its components. For instance, the method seems to rely heavily on the knowledge matrix B, yet no analysis or ablation is provided to demonstrate its role.
4. There have already been numerous studies on data selection (e.g., IFD, LESS, Nuggets, AlpaGaus), many of which explore similar diversity-based sampling strategies. The paper does not compare with these baselines.
5. Using only one dataset (MATH500) is insufficient to demonstrate the general effectiveness of the method. In addition, performance improvements appear marginal across different subset sizes, and no comparison is made against using the full dataset without selection. Normally, a data selection approach usually identifies an optimal subset size where performance saturates. That's the advantage of a data selection method.
6. The experimental setup is unclear. For example, how was Qwen-32B-Instruct trained or fine-tuned? LORA? or Full-finetuning? While some details are mentioned, there is no consolidated experimental setting section, which significantly hurts reproducibility.

### Questions
1. What is the computational cost of the preprocessing stage, given that it involves prompting an LLM to generate the knowledge vector B for each sample?
2. The difference between Figure 2 and Figure 3 is not clear. From the description, Figure 3 seems to be a subset of Figure 2. If there is no substantive difference, the two figures should be merged.
3. Some conclusions are overstated or partial. For instance, the statement: “Interestingly, the 40K entropy-selected subset achieves slightly lower final loss than the 50K subset...” could be more appropriately interpreted as evidence supporting the effectiveness of the method rather than implying a general principle about dataset size.

### Soundness
2

### Presentation
2

### Contribution
2
