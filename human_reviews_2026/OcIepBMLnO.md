# LLM-AUGMENTED KNOWLEDGE REPRESENTATION LEARNING VIA ATTENTION FOR KNOWLEDGE GRAPH COMPLETION

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Knowledge Graph Completion (KGC) is a critical task, yet its performance is often hindered by the data sparsity problem arising from the long-tail distribution of entities. While existing works attempt to enrich representations by incorporating auxiliary information like entity descriptions, this kind of implicit learning approaches often proved ineffective due to the introduction of irrelevant noise. To address this, we propose a novel framework LAKRA, which shifts the paradigm from implicit knowledge encoding to explicit data augmentation. LAKRA leverages a Large Language Model (LLM) to proactively reason and generate high-quality, schema-compliant triples for sparse entities, mitigating data sparsity at its source. Besides, we design a powerful encoder-decoder architecture for representation learning, which features a query-aware hybrid attention encoder and a deep feature interaction decoder to capture complex structural and semantic patterns. Experiments conducted on the benchmark datasets demonstrate that LAKRA achieves highly competitive performance on link prediction tasks involving infrequent entities. Our work presents an effective new paradigm for tackling data sparsity in knowledge graphs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes LAKRA, a knowledge graph completion (KGC) framework that addresses data sparsity by explicitly augmenting the graph with LLM-generated, schema-compliant triples for infrequent (tail) entities. The augmented KG is then processed by a query-aware hybrid attention encoder and a 3D convolutional decoder that jointly capture structural and semantic interactions. Extensive experiments on FB15k-237 and WN18RR show  improvements over baselines, particularly on long-tail entities.

### Strengths
1. The shift from implicit representation enrichment to explicit LLM-driven graph augmentation is conceptually clean, providing a tangible countermeasure to the ‘long tail’ problem.
2. Substantial ablation studies tease apart the model’s components, showing the importance of LLM-driven augmentation and the hybrid attention mechanism. This aligns well with the design claims.

### Weaknesses
1. To be honest, I do not think the novelty and writing quality reached the standard of ICLR. For novelty, the "LLM-BASED DATA AUGMENTATION" is trivial, the QUERY-AWARE GRAPH ATTENTION ENCODER WITH HYBRID ATTENTION is common and some similar methods have been proposed, The "3D DEEP FEATURE INTERACTION DECODER" is relatively novel but overall, I think the contribution is not strong enough.
2. The experimental results in this paper are not sufficiently convincing.
As shown in Table 2, the performance on the WN18RR dataset is almost a failure — the only competitive metric, Hits@1, improves by merely 0.001, which is statistically negligible. The authors should further analyze why LAKRA fails to improve on WN18RR
I strongly suggest that the authors include at least one additional dataset to enhance the persuasiveness and generality of the experimental evaluation. Moreover, considering that the proposed method introduces an additional LLM-based generation process, the authors should also provide supplementary experiments or analyses on computational cost and time complexity to justify the efficiency of the approach.
3. I appreciate the authors’ honesty in reporting the following detail: “932 and 198 of these generated triples for FB15K-237 (10,833) and WN18RR (12,526), respectively, already existed in their corresponding test sets.” I did not deduct points for this transparency, and I also acknowledge that the authors have shown their method remains effective even after removing these overlapping triples. However, this observation raises an interesting question: does this overlap occur because the generation process is genuinely effective in reproducing valid knowledge, or because the datasets themselves are relatively simple or limited in diversity, making test triples easier to regenerate?
4. While the experimental section benchmarks LAKRA against a diverse range of backbone baselines, it does not include direct comparisons with recent LLM-augmented KGC methods that employ in-context learning, generative prompting, or fine-tuning for structural enrichment—such as KICGPT or MLKGC.These omissions are problematic because such methods are conceptually and technically closest to LAKRA’s main contribution.

### Questions
See Weaknesses

### Soundness
3

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
5

### Summary
This paper proposes a new framework named LAKRA to address the data sparsity problem in knowledge graph completion (KGC) tasks. The authors innovatively utilize Large Language Models (LLMs) for explicit data augmentation rather than implicit knowledge encoding. Specifically, LAKRA first leverages type constraints and a two-stage generation process (relation generation and entity generation) to enable the LLM to automatically produce high-quality, schema-consistent triples for long-tail entities, thereby enriching the graph structure at its source. Subsequently, a hybrid attention encoder combining Cross-Attention and MLP-Attention is designed to achieve query-aware semantic aggregation, while a 3D convolutional decoder is employed to model deep interactions between entities and relations, improving link prediction performance.

### Strengths
1. Novel paradigm shift: The paper introduces a clear and well-motivated shift from implicit representation enrichment to explicit, schema-aware data augmentation using LLMs — a conceptually elegant and practically impactful idea for mitigating knowledge graph sparsity.
2. Well-designed architecture: The combination of a query-aware hybrid attention encoder and a 3D convolutional decoder is technically sound and effectively captures both semantic relevance and multi-level structural interactions.
3. Strong empirical results: LAKRA consistently achieves top-tier or state-of-the-art performance, particularly on long-tail entity prediction, validating the benefit of LLM-generated triples.

### Weaknesses
1. Although the paper introduces LLM-based explicit data augmentation, it does not sufficiently evaluate the factual accuracy and noise ratio of the generated triples.

2. The proposed method heavily relies on LLM-generated data but does not specify the exact model, prompt templates, or generation parameters, making it difficult to reproduce.

3. While related works such as KG-BERT, KGT5, and RAA-KGC are discussed, the paper does not present LAKRA’s performance differences across various types of sparse entities (e.g., isolated nodes, weakly connected nodes), which reduces its interpretability.

### Questions
1. You mention that LLMs generate “schema-constrained high-quality triples” to enhance sparse entities — how do you prevent the LLM from introducing incorrect or hallucinated facts that could contaminate the training data?

2. Please specify the detailed configurations of the LLM (e.g., model type, prompt design, generation settings) to improve reproducibility.

3. Your ablation results show that removing the LLM augmentation leads to a significant drop in performance, but the paper does not analyze how the scale or proportion of augmented data affects results. Is the performance gain mainly due to the quantity of generated data or the semantic diversity it introduces?

### Soundness
2

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
To combat data sparsity in knowledge graph completion, this paper proposes LAKRA, a framework that explicitly augments the graph before learning. LAKRA uses a LLM to generate plausible new triples for infrequent entities. A subsequent powerful encoder-decoder, featuring hybrid attention and 3D convolutions, then learns from this enriched graph, achieving state-of-the-art performance by tackling sparsity at its source.

### Strengths
The core idea of the paper is interesting. By leveraging an LLM for explicit data augmentation before the main completion task, it enriches the graph structure for sparse entities, which is a promising direction for improving overall KGC performance.

### Weaknesses
1.	The paper does not specify which LLM is used, how the 20% tail-entity threshold is chosen, or why it is fixed across datasets. Using entity degree might be more principled. Important settings such as candidate set size and prompt design are also undisclosed.
2.	The paper claims to generate "high-quality" and "schema-compliant" triples but fails to address the critical issue of factual verification. Given that LLMs are prone to hallucination, the proposed method risks injecting factually incorrect—though schema-compliant—triples into the knowledge graph. This introduces significant noise, a problem the paper itself criticizes in other approaches.
3.	The encoder combines multiple components (cross-attention, MLP attention, Gaussian kernels, 3D convolution) without clear justification for why this specific composition is necessary, reducing architectural clarity.
4.	The evaluation is limited to commonsense KGs, where the LLM's pre-trained knowledge likely gives the augmentation approach an unfair advantage. Its effectiveness on domain-specific KGs, where the LLM may lack knowledge, is not tested. Moreover, the so-called “LLM-based” baselines (e.g., KG-BERT, RAA-KGC) rely on PLMs rather than LLMs. The paper also omits recent LLM-based methods such as KICGPT [1].
[1] Wei Y, Huang Q, Zhang Y, et al. KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion[C]//Findings of the Association for Computational Linguistics: EMNLP 2023. 2023: 8667-8683.

### Questions
See the Weaknesses section above.

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
5

### Summary
The paper tries to deal with data sparsity in knowledge graphs, especially for long-tail entities. The authors use an LLM to generate new facts for sparse entities and then train a hybrid encoder–decoder model (with some 3D feature decoding setup) on the enriched graph for link prediction. The general idea makes sense and is timely, given the popularity of LLM-assisted augmentation.

### Strengths
The core idea — explicitly using an LLM for graph completion before training a downstream model — is interesting and, to my knowledge, not explored much in this specific setup. The schema-aware prompting part seems thoughtful. And the combined hybrid encoder + 3D decoder setup gives solid numbers overall, even if some gains come from leakage.

### Weaknesses
- When I looked at Table 5.1, most of the MRR improvement on FB15K-237 actually comes from triples that overlap with the test set (~66 % of the gain). That’s a bit concerning because it undercuts the paper’s main claim that the model’s improvement isn’t just due to memorizing test data.
- The paper also doesn’t say anything about the computational side. I’d like to see at least rough numbers on training or inference costs, including time, memory, or any other relevant factors. The Gaussian feature expansion seems particularly heavy; it increases embedding size, so I’d expect a big bump in compute, but there’s no mention of that.
- Before Equation 22, there’s no explanation of what $\mathcal{L}_{aug}$ actually is. Without that, the experiments are not reproducible or clear.
- Also, the LLM-generated triples are treated as a bit of a black box. The authors claim their work is of “high quality,” but there’s no quantitative or qualitative validation. A few examples of both good and bad generations would be helpful.
- The ablation results don’t fully support the claim that attention is critical. The differences look small, so I’m not sure that component makes a big difference.
- Finally, the experiments are limited to two small datasets, which doesn’t tell me whether the idea would work on larger graphs with more severe long-tail issues. Some recent baselines are also missing. And a few design choices feel arbitrary, like the “bottom 20 %” rule for selecting entities or the 5-layer decoder — there’s no reasoning provided.

### Questions
Do the authors use reciprocal relations during training? And do the baselines do the same? Some models rely on that trick, so it would be good to know for a fair comparison.

### Soundness
3

### Presentation
3

### Contribution
2
