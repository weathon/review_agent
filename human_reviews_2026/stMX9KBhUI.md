# Towards Improved Sentence Representations using Token Graphs

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Obtaining a single-vector representation from a Large Language Model's (LLM) token-level outputs is a critical step for nearly all sentence-level tasks. However, standard pooling methods like mean or max aggregation treat tokens as an independent set, discarding the rich relational structure captured by the model's self-attention layers and making them susceptible to signal dilution. To address this, we introduce GLOT, a lightweight, structure-aware pooling module that reframes pooling as relational learning followed by aggregation. Operating on the outputs of a frozen LLM, GLOT first constructs a latent token-similarity graph, then refines token representations with a graph neural network, and finally aggregates them using a readout layer. Experimentally, our approach is remarkably robust and efficient: on a diagnostic stress test where 90% of tokens are random distractors, GLOT maintains over 97% accuracy while baseline methods collapse. Furthermore, it is competitive with state-of-the-art techniques on benchmarks like GLUE and MTEB with 20x fewer trainable parameters and speeds up the training time by over 100x compared with parameter-efficient fine-tuning methods. Supported by a theoretical analysis of its expressive power, our work shows that learning over token graphs is a powerful paradigm for the efficient adaptation of frozen LLMs. Our code is published at https://github.com/ipsitmantri/GLOT.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes GLOT which is a novel pooling methodology that builds graph out of latent representations and then pass through them multiple layers of GNN and lastly using the recent learnable pooling technique called AdaPool to capture sentence level representations. It operates on output tokens which comes from a frozen backbone so it has very low both computational and memory overhead. It shows significant improvements over static or learnable pooling across benchmarks like GLUE and other downstream tasks from MTEB, and IMDB long text classification dataset.

### Strengths
* The paper introduces an interesting approach to token pooling by using GNN-based graph construction to preserve token relationships that standard pooling methods discard.

* The proposed architecture is lightweight, requiring only ~9M trainable parameters and 0.42GB GPU memory compared to fine-tuning approaches.

* The method shows consistent improvements over frozen baseline pooling methods across multiple model architectures from BERT to Mistral-7B.

* The diagnostic stress test provides useful insights into robustness under noisy conditions, where the method maintains performance while baselines degrade.

* The paper includes reasonable ablation studies on graph sparsity and component contributions with sufficient implementation details for reproducibility.

* Operating on frozen backbones makes the approach practically accessible for resource-constrained settings without expensive fine-tuning infrastructure.

### Weaknesses
*  The evaluation approach may not fully align with the paper's focus on sentence representations, as it relies heavily on GLUE tasks rather than comprehensive MTEB evaluation, which is typically considered the standard benchmark for assessing sentence embedding quality.

* The MTEB results are limited to 7 out of 58 tasks, which makes it difficult to assess the method's performance across the full range of sentence embedding applications and may not fully represent its general capabilities.

* Comparisons to established sentence embedding methods like contrastive fine-tuning approaches (Sentence-BERT, SimCSE) or recent lightweight adaptation methods (LLM2Vec) would strengthen the evaluation and provide better context for the contributions.

* The baseline comparisons involve different parameter counts (~9M for GLOT vs. minimal parameters for simple pooling), which makes it somewhat unclear whether improvements stem primarily from the graph-based design or from having additional learnable capacity.

* The efficiency analysis in Table 5 focuses on CoLA, which emphasizes grammatical acceptability rather than semantic similarity, and the surprisingly strong performance relative to full fine-tuning raises questions about baseline tuning that could be addressed with more details.

* Information about inference-time computational costs, particularly for graph construction and GNN processing, would help readers better understand the practical deployment trade-offs of the approach.

* Including parameter-matched baselines such as deeper MLP-based pooling would help better isolate the specific contribution of the graph-based relational learning component.

* A unified comparison showing both efficiency and performance metrics for all methods in a single analysis would provide clearer insights into the overall trade-offs and help readers make more informed assessments.

### Questions
1. Could you provide complete MTEB results across all 58 tasks to give a comprehensive picture of the method's performance on standard sentence embedding benchmarks?

2. How does GLOT compare to established sentence embedding methods like contrastive fine-tuning (Sentence-BERT, SimCSE) and recent lightweight approaches (LLM2Vec) in terms of both performance and computational cost?

3. What is the performance of parameter-matched baselines, such as multi-layer MLP pooling with similar capacity (~9M parameters), to isolate whether gains come from graph-based relational learning or simply from having more learnable parameters?

4. Could you clarify the training procedures for the full fine-tuning and LoRA baselines in Table 5, given that GLOT unexpectedly outperforms these much larger capacity methods?

5. What are the inference-time computational costs including graph construction and GNN forward pass, and how do these compare to simpler pooling methods?

6. How does GLOT perform when the backbone is fine-tuned rather than frozen, and does it provide complementary benefits to fine-tuning or does fine-tuning alone achieve comparable results?

7. Could you provide a unified comparison table showing both efficiency metrics and performance across multiple representative tasks for all methods to enable clearer assessment of trade-offs?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents GLOT, a new method for deriving sentence embeddings. Compared to naive mean or max pooling, GLOT explicitly models the semantic relationships between tokens by incorporating a Graph Neural Network (GNN) structure before the final aggregation step. Experimental results demonstrate that GLOT can achieve effective performance improvements with minimal computational resources while keeping the backbone model frozen.

### Strengths
1. Sentence embedding is a core topic in representation learning, and pooling is a crucial step in converting token-level embeddings into sequence-level embeddings. Therefore, the chosen track is highly relevant to the conference's scope and holds significant practical value.
2. The paper is well-written. The methodology is presented clearly and intuitively. The authors claim that GLOT is the first work to learn sentence representations via GNNs on top of a frozen Large Language Model (LLM).
3. The experiments are conducted on both discriminative PLMs (BERT, RoBERTa) and generative PLMs (Llama, Mistral), validating the method's effectiveness across different model architectures.

### Weaknesses
1. The MTEB is a massive benchmark for embedding evaluation. The authors show that their method surpasses baselines on only a selected subset of tasks. This seems to be insufficient to support the claim of "state-of-the-art performance" in the abstract.
2. The chosen baselines are rather conventional. I am curious about the comparison between this proposed method and the prompting-based methods that have emerged in the past two years. Prompting methods can have an even smaller memory footprint than this work, as they require no parameter updates at all. Some relevant references include:
-	“PromptBERT: Improving BERT Sentence Embeddings with Prompts”
-	“Scaling Sentence Embeddings with Large Language Models”
-	“Simple Techniques for Enhancing Sentence Embeddings in Generative Language Models”
3. The authors construct the graph based on the similarity between token embeddings. This approach is relatively straightforward for discriminative PLMs with bidirectional attention. However, for generative PLMs with unidirectional attention, tokens that appear later in a sequence have a larger attention scope. Shouldn't the similarity calculation between their token embeddings account for this disparity?

### Questions
Authors should give their response and explanation with respect to the weaknesses mentioned above.

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
The paper proposes a lightweight structure-aware pooling module that reframes pooling as relational learning followed by aggregation, dubbed GLOT. The method is remarkably robust and efficient, as demonstrated by extensive experiments.

### Strengths
1) The paper has a clear motivation.
2) The method sounds technical.
3) GLOT has been evaluated through extensive experiments.

### Weaknesses
1) Some technical details of the method need to be presented.
2) The paper lacks essential theoretical explanations to ensure the effectiveness of the method.
3) The experimental analysis needs more powerful explanations.

I have significant doubts about the rationale behind the design of the proposed GLOT method and about why using a GNN-based approach alone can substantially improve the problem. These issues should be clearly explained in the method description, theoretical analysis, and experimental validation, but I have not found such clarification in the current version. I consider this to be the core of the paper.

### Questions
1) GLOT seems to only use an additional GNN to refine the representation. Why does it achieve such significant improvements in the results?
2) GLOT is easy to implement, and its effectiveness seems to come entirely from the GNN. Would using more advanced variants of GNN further improve the robustness of the model?
3) GLOT obviously has a more complex graph-based structure. What explains its lower number of parameters and faster training process?

### Soundness
3

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
4

### Summary
This paper proposes a lightweight pooling module, called GLOT, which transforms token-level outputs of frozen LLMs into sentence embeddings through graph-based relational learning. Instead of treating tokens independently, GLOT builds a token-similarity graph, refines representations using a graph neural network, and aggregates them via a learnable readout. Across GLUE, IMDB, and MTEB, GLOT outperforms standard and learnable pooling methods while being 20× more parameter-efficient and 100× faster than fine-tuning approaches. It also shows strong robustness to noise, maintaining over 97% accuracy with 90% distractor tokens.

### Strengths
1. Reframing sentence pooling as token-relation modeling.
The paper introduces GLOT, a new framework that replaces traditional pooling (e.g., mean or [CLS]) with a graph-based aggregation process. GLOT unifies prior pooling schemes as special cases (when the graph is empty or uniform) and explicitly addresses the long-standing signal dilution problem in sentence embeddings. Demonstrated by strong improvements in the signal dilution test (≈97% accuracy under 90% distractors; Table 7, Sec. 5.4).

2. Strong consistency across diverse settings.
GLOT shows consistent, significant improvements over both static and learnable pooling baselines on major benchmarks: GLUE, IMDB, and MTEB. It outperforms methods such as mean pooling, weighted pooling, [CLS] pooling, across both encoder-based and decoder-based LLMs. This shows the method’s robust generalization and practical applicability across models and tasks. (Sec. 5.1–5.3, Tables 2–4, Fig. 3.)

3. High efficiency with minimal training cost.
The proposed module is extremely lightweight, only ≈ 8.9 M trainable parameters and ~ 0.42 GB GPU memory, compared to full fine-tuning or LoRA. Despite this small footprint, it delivers comparable or better accuracy and over 100× training-speed improvement. This efficiency makes GLOT highly practical for large-scale or resource-constrained deployments, supporting the paper’s claim of being a drop-in, low-cost alternative to fine-tuning. (Sec. 5.5, Table 5.)

### Weaknesses
1. While GLOT introduces a graph-based pooling paradigm, the graph construction step is heuristic; edges are formed by thresholding pairwise cosine similarities between token embeddings.
The paper does not explore learnable or adaptive graph formation, nor analyze how different thresholds quantitatively affect representation quality beyond a small ablation (τ = 0.4–0.6 works best). As a result, the approach lacks a deeper theoretical justification on how graph topology influences performance. (Sec. 3.2, Sec. 5.5, Table 6.)

2. Although Table 5 reports large efficiency gains (≈100× faster, 0.42 GB GPU), this measurement is only for one dataset (CoLA) and one backbone (Mistral-7B). The paper does not provide cross-task or cross-model runtime, throughput, or latency benchmarks, and omits the cost of graph construction itself. Therefore, the claimed computational advantages, while promising, are not fully validated across broader conditions.

3. The related-work section briefly mentions prior pooling and token-interaction models but does not clearly position GLOT relative to recent work. This leaves ambiguity about whether the proposed approach is a substantial conceptual advance or an effective adaptation of known ideas.

4. The first step of GLOT involves computing a pairwise cosine similarity matrix between all L tokens in the sequence to determine the edges. This is an O(L^2) operation with respect to the sequence length L. While the paper tests on sequences up to 512 tokens (IMDB)9, this O(L^2) step could become a significant computational bottleneck for applications involving very long contexts (e.g., thousands of tokens), a limitation which is not discussed.

### Questions
see Weaknesses

### Soundness
2

### Presentation
3

### Contribution
3
