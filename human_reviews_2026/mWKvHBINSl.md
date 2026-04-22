# PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Model pruning is an effective approach for compressing large language models (LLMs). However, this process often leads to significant degradation of model capabilities. While post-training techniques such as instruction tuning are commonly employed to recover model performance, existing methods often overlook the uneven deterioration of model capabilities and incur high computational costs. Moreover, some irrelevant instructions may also introduce negative effects to model capacity recovery. To address these challenges, we propose the **P**ost-training d**A**ta **S**election method for **E**fficient pruned large language model **R**ecovery (**PASER**). PASER aims to identify instructions to recover the most compromised model capacities with a certain data budget. Our approach first applies manifold learning and spectral clustering to group recovery instructions in the semantic space, revealing capability-specific instruction sets. Then, the data budget is adaptively allocated across clusters by the degree of corresponding model capability degradation. In each cluster, we prioritize data samples that lead to the most decline of model performance. To mitigate potential negative tuning effects, we also detect and filter out conflicting or irrelevant recovery data. Extensive experiments demonstrate that PASER significantly outperforms conventional baselines, effectively recovering the general capabilities of pruned LLMs while utilizing merely 4\%-20\% of the original post-training data. We provide the anonymous code repository in [Link](https://anonymous.4open.science/r/PASER-E606).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes PASER, a data selection framework designed to improve the efficiency and effectiveness of recovery post-training after LLM pruning. PASER employs manifold learning and spectral clustering to group instructions in the semantic space, allocates data budgets based on measured capability degradation, and filters out conflicting or irrelevant data to mitigate negative tuning effects. Experiments on several pruning schemes (structured, semi-structured, unstructured) and multiple LLM frameworks (LLaMA 2/3, Baichuan 2) demonstrate that PASER can recover model performance using only 4%–20% of the original instruction tuning data.

### Strengths
This paper tackles a problem about how to efficiently recover LLM capabilities after pruning. This issue has become increasingly relevant for real-world model deployment. The study is supported by comprehensive experiments covering multiple pruning strategies and models of different scales, ensuring good empirical validation. Results demonstrate consistent performance improvements over both random selection and other instruction tuning data selection baselines. Results of the ablation studies helps clarify the contribution of each system component, enhancing the overall credibility and interpretability.

### Weaknesses
1. Some ideas resemble ideas from earlier literature (e.g., similarity-based data sampling). The paper would benefit from a more thorough literature review, though this does not undermine its novelty.
2. The paper is generally well-written, there are a few minor oversights. For example, there are a few typos (“we relies on…”), and some abbreviations used in the ablation study section (S2RIC, CDAIS, NTEM) are not introduced before but only defined in the appendix.

### Questions
1. Could the authors clarify the practical advantages of pruning a large model compared to directly training a smaller one from scratch? 
2. Is it possible for the authors to provide a baseline of directly training or fine-tuning a smaller model with the instruction tuning data?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces PASER, a novel framework for selecting a small, high-impact subset of instruction-tuning data to recover the performance of pruned LLMs. The authors identify three key challenges with existing recovery methods: high computational cost, failure to address the uneven degradation of capabilities after pruning, and the potential for negative tuning effects from irrelevant data. PASER addresses these challenges through a three-stage process: semantic-structural recovery instruction clustering,capability degradation-aware instruction selection, and negative tuning effects mitigation. The authors conduct extensive experiments on a wide range of LLMs, pruning schemes, and instruction datasets. The results demonstrate that PASER significantly outperforms baselines, recovering model performance to near-original levels while using only 4%-20% of the instruction data and substantially reducing training time.

### Strengths
1. High Relevance and Strong Motivation: The problem of efficiently recovering performance in compressed LLMs is highly practical and significant, while underexplored. The paper does an excellent job of motivating the work by clearly identifying the limitations of naive retraining, particularly the issues of uneven capability degradation and negative tuning effects, which are often overlooked.

2. Novel and Well-Designed Methodology: The proposed PASER framework is technically sound, innovative, and directly addresses the identified problems.

3. Extremely Comprehensive and Rigorous Evaluation: The experimental validation in this paper is exceptional and serves as a model for thoroughness.

4. Strong and Convincing Empirical Results: PASER consistently and significantly outperforms all baselines across nearly all tested configurations. The finding that it can recover performance to a level that is sometimes better than the original unpruned model is particularly striking and suggests that the method effectively selects a high-quality, coherent subset of data that is superior to the full, noisy dataset for tuning. The efficiency gains in terms of training time are also substantial and well-documented.

### Weaknesses
The paper is of very high quality, and it is difficult to find significant weaknesses. The following points are minor.
1. Methodological Complexity: The PASER pipeline is composed of several sophisticated components (manifold learning, NMF spectral clustering, CCG). While the authors demonstrate its effectiveness and computational efficiency, the overall complexity is higher than simpler data selection heuristics. This might pose a slight barrier to adoption for practitioners looking for a "plug-and-play" solution, though the provided code should mitigate this.
2. The writing of the methodology section can be improved to reduce the cognitive load on the reader.

### Questions
1. In Table 1, the results of training the LLM-Pruner pruned model with full data is even lower than without training, (35.64 vs 57.78) how to interpret this?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PASER (Post-training dAta Selection for Efficient pruned large language model Recovery), a novel framework designed to address the significant performance degradation of Large Language Models (LLMs) after pruning. The core challenge is that pruning affects different model capabilities (e.g., reasoning, coding, math) unevenly, and conventional post-training recovery using full instruction datasets is computationally expensive and can be suboptimal due to irrelevant or conflicting data. PASER tackles this with a three-stage, data-centric approach. Extensive experiments on models like LLaMA2/3 and Baichuan2 under various pruning schemes (LLM-Pruner, SliceGPT, Wanda, SparseGPT) demonstrate that PASER significantly outperforms baseline methods (including random selection and general data selection techniques like IFD and Nuggets).

### Strengths
1. The paper effectively identifies a critical and under-explored gap in the LLM compression pipeline: the need for targeted, efficient recovery post-training that accounts for the uneven degradation of model capabilities after pruning.
2. The experimental section is a major strength. The authors validate PASER across multiple model families (LLaMA, Baichuan, Qwen, Mixtral), pruning methods (structured, semi-structured, unstructured), and dataset scales (Alpaca, LaMini). The consistent superiority over strong baselines, and the ability to sometimes exceed the performance of the unpruned model, provides compelling evidence for its effectiveness.

### Weaknesses
1. While PASER drastically reduces training time, the data selection process itself (clustering, JSD calculation, CCG management) is non-trivial. A more detailed analysis of this overhead, especially for massive datasets (e.g., billions of samples), and potential optimizations for the selection phase would be valuable for real-world scalability.
2. The CCG is a heuristic-based method. Its effectiveness is limited by the quality of concept extraction and may not capture complex, context-dependent, or cross-domain semantic conflicts. The paper acknowledges this in the appendix, but it remains a potential point of failure where nuanced conflicts could be missed.
3.  The method's performance is inherently tied to the initial pruning step. The paper notes that for models where pruning "severely damaged the model internal structure" (e.g., Baichuan2-7B), recovery is less effective. A deeper investigation into the relationship between pruning strategy, resultant damage, and PASER's recoverability would be insightful.

### Questions
N/A

### Soundness
3

### Presentation
3

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
This paper aims to solve the practical problem of post-pruning recovery via data selection. It offers a practical solution by combining clustering and degradation metrics, though it is a bit incremental over prior data selection methods.

### Strengths
(1) The integration of manifold learning and JSD-based degradation scoring is creative and tailored to the pruning recovery context.

(2) Extensive evaluations on models like LLaMA2 and Baichuan2 show robust performance gains and efficiency improvements. The evaluation section is comprehensive, covering multiple models and pruning schemes.

### Weaknesses
(1) The core methodology seems to originate from various existing works. The clustering approach resembles semantic sampling in SMART, while the degradation scoring echoes influence-based selection in Datamodels. How the proposed method is related and differs from the existing approached should be discussed in the main paper, instead of  in the Appendix as has been done in the current paper.

(2) The computational overhead of manifold learning and JSD calculation is non-trivial, yet the paper provides insufficient analysis of scalability.

(3) While having included recent baselines, the paper is lack of presenting an in-depth analysis about the reasons why PASER differs fundamentally from general data selection methods.

### Questions
1. How does PASER substantially differ from the applying existing data selection methods (e.g., LESS, DELIFT) directly to the pruned model recovery scenario? The paper only shows performance comparisons but lacks an in-depth analysis and conceptual differentiation.

2. The concept consistency graph appears to be a bit over-simplistic. How does it handle complex semantic conflicts that cannot be captured by simple co-occurrence patterns?

3. Why JSD was chosen over more sophisticated divergence measures?

### Soundness
2

### Presentation
3

### Contribution
2
