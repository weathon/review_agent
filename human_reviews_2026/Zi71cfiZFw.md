# Less is More: Contrastive Retrieval Heads Improve Attention-Based Re-Ranking

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
The strong zero-shot and long-context capabilities of recent Large Language Models (LLMs) have paved the way for highly effective re-ranking systems. Attention-based re-rankers leverage attention weights from transformer heads to produce relevance scores, but not all heads are created equally: many contribute noise and redundancy, thus limiting performance. To address this, we introduce $\textit{CoRe heads}$, a small set of retrieval heads identified via a contrastive scoring metric that explicitly rewards high attention heads that correlate with relevant documents, while downplaying nodes with higher attention that correlate with irrelevant documents. This relative ranking criterion isolates the most discriminative heads for re-ranking and yields a state-of-the-art list-wise re-ranker. Extensive experiments with three LLMs show that aggregated signals from CoRe heads, constituting less than 1% of all heads, substantially improve re-ranking accuracy over strong baselines. We further find that CoRe heads are concentrated in middle layers, and pruning the computation of final 50% of model layers preserves accuracy while significantly reducing inference time and memory usage.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the problem of building effective attention-based re-ranking systems. The paper proposes CoRe (Contrastive Retrieval Heads), which extends previous attention-based re-rankers by 1) using hard negative data for identifying important heads. introducing a contrastive head selection criterion and 2) CoRe explicitly re-normalizing attention to relevant versus irrelevant documents. The paper evaluates the method on the BEIR benchmark and the multilingual MLDR dataset across several open-weight models, showing some improvements over ICR and QRHead baselines.

### Strengths
The paper also considers  a simple yet effective layer-pruning strategy, showing that re-ranking accuracy can be preserved while cutting memory usage by ~40% and latency by ~20%.

The proposed extension is straightforward and seems to be effective.

Evaluation covers two re-ranking datasets (BEIR, MLDR) and four open-source LLMs.

The paper is mostly well presented as easy to understand.

### Weaknesses
I have some concerns regarding the framing of the core idea of CoRE. The paper suggests that QRHead’s absolute scoring fails to capture relative ranking among documents (Ln 146). As attention scores already result from a softmax normalization, they implicitly encode relative weighting. It seems the CoRe is renormalizing at the document level to ignore other parts like instructions, rather than introducing new contrastiveness.

The current evaluation focuses primarily on BEIR-style single-hop retrieval. Past work (e.g., ICR, QRHead) also explored multi-hop reasoning tasks such as MuSiQue and CLIPPER.

The paper could include more discussion regarding context length. One main advantage of attention-based re-rankers over generative methods is the potential to scale to long context settings. The experiment mainly considers reranking 40 documents. It would be good to include or discuss performance and scaling behavior under a larger number of documents. 

The paper needs more analysis on intrinsic differences of CoRE heads and QRHeads. Like reporting overlapping between top-32, 64 heads.

### Questions
How does CoRe generalize to multi-hop reasoning?

How does CoRE generalize to longer contexts?

What’s the overlap between CoRE heads and QRHeads among top-32, 64 heads?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CoRe heads (Contrastive Retrieval heads), which is a subset of attention heads for document re-ranking. The key difference from previous work, such as QRHead, is the contrastive objective to find the retrieval heads. The CoRe heads are then used for document re-ranking, where the authors show improvement over existing approaches on the BEIR benchmark. The experiments include models from Llama, Phi, Mistral, and Granite model families. CoRe also uses fewer heads than other methods to achieve comparable or better performance.

### Strengths
The paper makes adjustments to existing approaches to detect existing approaches and achieve better performance on document re-ranking tasks. The experiments also covers multi-lingual settings that were not previously studied. The analysis also covers important system considerations such as the percentage of layers that can be pruned without affecting performance.

### Weaknesses
The contrastive scoring is mathematically similar to the objectives used in previous work. Specifically, the main baseline QR-Head uses post-softmax attention scores over the positive document. As a result, it’s achieves the same mathematical form as the paper’s retrieval score in equation 5 because the QRScore is normalized with respect to the attention mass assigned to the rest of the context as well. The only difference is that (1) CoRe scoring excludes the instruction from the overall attention mass, which is not carefully studied (would QR-Head perform the same if the instruction mass were excluded) and (2) CoRe does an additional softmax and tunes the temperature hyper parameters (the temperature in the QR-Head should also be tuned in a similar fashion to achieve fair comparison).
The paper could benefit from such theoretical analysis on how the mathematical form differ from previous works and where the improvements stem from. 

Furthermore, while the results and analysis are strong, I find the novelty lacking as it only makes minor changes from previous works.

### Questions
How do the results scale with more heads beyond 8 heads?

How do the final results different with different temperature hyper parameters? 

Why do CoRe-R work particularly well for Mistral but the results on the other models are much closer with QR-R?

### Soundness
3

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
5

### Summary
In this paper, the authors propose to leverage a small subset of attention heads in LLMs to specialize in discriminating relevant from irrelevant documents for re-ranking tasks. Building on prior attention-based re-rankers and query-focused retrieval heads, they introduce a contrastive scoring metric inspired by InfoNCE loss. The experiments are conducted on the BEIR and MLDR benchmarks when the experimental results show improvements against the ICR baseline. Analysis also shows the approach can enable pruning of 50% of final layers with negligible accuracy loss, reducing inference latency by 20% and GPU memory by 40%, for the middle layers.

### Strengths
* Consistent improvements against the baseline method over diverse benchmarks 
* By using only a tiny fraction of heads and enabling middle-layer focus for pruning, it significantly reduces latency, memory usage, and overhead compared to full-model aggregation, making it practical for real-world systems.
* The proposed approach reveals that discriminative retrieval signals are concentrated in middle layers, advancing interpretability of LLM attention and opening avenues for optimized architectures.

### Weaknesses
* The proposed approach highly relies on hard negatives from NQ, so the data quality could be sensitive. 
* Hyperparameter tuning is required, and it adds the overhead and reduce the ease of use across new LLMs.
* The proposed method fails to beat baselines on certain datasets.

### Questions
* Could the optimal number of CoRe heads be different models and dataset? How sensitive is it?
* What specific criteria or thresholds are used to discard passages as false negatives during hard negative data construction?
* Can CoRe heads be dynamically updated or adapted for new datasets without re-running the entire detection process?

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
The paper proposes a new attention-based re-ranker method that utilizes the attention weights of LLMs to perform re-ranking of documents. Built upon prior work on attention-based re-ranker methods, the paper introduces a novel approach to detect attention heads in LLMs that are important for re-ranking, termed as contrastive retrieval heads, by computing contrastive retrieval scores over positive and negative documents in the input: such heads attend to the irrelevant documents and are not distracted by the irrelevant documents. Empirical results show that re-ranking with contrastive heads outperforms other attention-based re-ranking baselines. Furthermore, such a method can be combined with layer pruning to improve the run-time efficiency as the detected contrastive heads concentrate in the middle layers of transformers.

### Strengths
1. The paper proposes a new re-ranking method that requires no training and demonstrates empirical results across multiple benchmarks with multiple LLMs.

2. The paper compares with several attention-based re-ranking baselines and shows strength over such baselines using the proposed method to detect important attention heads for re-ranking.

3. The paper further shows the strength of the method in terms of efficiency compared with baselines.

### Weaknesses
1. The performance of the contrastive retrieval head re-ranker is not particularly strong compared with the attention-based re-ranker baselines. In Table 1, the improvements of CoRe-R over ICR and QR-R are within around 1% for most subsets of the BEIR benchmark, and only for a single subset, Quora, can a relatively large improvement be observed. Similar results can be observed in Table 2 for the MLDR benchmark as well.

2. Related to question 1, there lacks an in-depth analysis of why CoRe-R outperforms the baselines only on certain tasks. Are there any insights or hypotheses on why CoRe-R performs particularly better than the baselines on the Quora subset of BEIR? What are some implications of such findings?

3. There lacks analysis of the similarities and differences between the heads detected by CoRe and by previous methods, especially NIAH and QR. Given the similar performance as mentioned in question 1, do they detect very similar attention heads? Does CoRe detect some very important missing heads that the other methods fail to detect?

4. There are missing baselines for the layer pruning experiments. Since the retrieval heads (i.e., NIAH-R) in Wu et al., 2024 also concentrate in the middle layers, I do not clearly see how layer pruning can only be applied to CoRe-R to improve efficiency, but not other attention-based re-rankers that use a sparse set of attention heads. Thus, the following baselines should be added for fairer evaluation of the efficiency benefit of the proposed method: QR-R (prune) and NIAH-R (prune).

### Questions
1. Why is contextual calibration only used for the re-ranking process, but not the head detection process of CoRe?

2. Figure 4: The head distribution in Mistral 7B is very different from those in the other models. Do you have any insights into this?

### Soundness
3

### Presentation
3

### Contribution
2
