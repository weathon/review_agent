# Effect of Document Packing on the Latent Multi-Hop Reasoning Capabilities of Large Language Models

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
The standard practice for training large language models involves packing multiple documents together to optimize computational efficiency. However, the impact of this process on the models' capabilities remains largely unexplored. To address this gap, we investigate how different document-packing strategies influence the latent multi-hop reasoning abilities of LLMs. Our findings indicate that packing can improve model performance compared to training on individual documents, at the expense of more compute. To further understand the underlying mechanisms, we conduct an ablation study, identifying key factors that explain the advantages of packing. Ultimately, our research deepens the understanding of LLM training dynamics and provides practical insights for optimizing model development.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates whether document packing during continual pre-training improves latent multi-hop reasoning. Using HotpotQA and 2WikiMultiHopQA with Gemma-2-2B, Llama-3.2-3B, and Llama-3.1-8B, the authors vary the number of packed documents, cross-document attention, repacking frequency, and batch size. They find that packing 4-6 documents improves recall precision, reduces hallucination (conditional on correct titles), and raises answer accuracy, especially when cross-document attention is enabled and repacking occurs each epoch. However, packing increases total compute, and gains diminish without cross-document attention.

### Strengths
- Clear empirical question with systematic ablations (packing vs. batch vs. repacking).
- Practical insight that repacking each epoch is crucial for performance.
- Honest discussion of compute–quality trade-offs.
- Consistent “sweet spot” trend (4–6 documents) across settings.

### Weaknesses
- The attention masking is ambiguous: §3.1 contains contradictory statements about whether cross-document attention is enabled, leaving the default mask unclear.

- The evaluation scope is conveniently narrow: experiments center on HotpotQA-easy and a 10% 2Wiki subsample with oracle per-question packing, which limits external validity.

- The metrics are biased: hallucination is computed only conditional on correct title selection, and an LLM-as-judge substitutes for standard EM/F1, skewing the interpretation of gains.

- Compute accounting is incomplete: comparisons are not matched on tokens or FLOPs and omit wall-clock or energy, so improvements may largely reflect extra compute rather than packing.

Statistical rigor and reproducibility are weak: there are no multiple seeds, error bars, or significance tests, and key artifacts (packing/masking scripts and exact hyperparameters) are not released.

### Questions
- What exact attention mask is used during continual pre-training, are tokens allowed to attend across <SEP> boundaries, and is any block-sparse structure applied beyond causal masking?

- Can you reproduce the main gains under compute-matched conditions (equal tokens/FLOPs), and report wall-clock time and energy to rule out extra compute as the driver?

- Do the improvements persist with non-oracle packing (random/semantic global packing) and on harder multi-hop benchmarks (e.g., HotpotQA medium/hard, MuSiQue, StrategyQA)?

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
3

### Summary
This paper investigates the impact of document packing on models’ latent multi-hop reasoning capabilities. While document packing is widely used to reduce padding and improve throughput, its effect on learning dynamics has been underexplored. The authors conduct systematic experiments by continually pre-training Llama 3 and Gemma 2 models on HotpotQA and 2WikiMultiHopQA corpora under different packing strategies (fixed and multi-packing) and ablations (cross-document attention, repacking frequency, batch size scaling). They then instruction-tune models on QA tasks that probe latent multi-hop reasoning. The study finds that (1) packing can improve reasoning accuracy and reduce hallucinations; (2) there exists an optimal range (4–6 documents per sequence); (3) packing requires more compute; and (4) cross-document attention and repacking each epoch are critical for gains. The authors conclude that packing may serve not only efficiency purposes but also enhance reasoning through richer inter-document contexts.

### Strengths
The paper addresses a genuinely underexplored but important topic—how document packing affects the quality (not just efficiency) of LLM training. Given the ubiquity of packing in large-scale pre-training, this study is timely and of high practical relevance.

The authors explore multiple factors (packing granularity, cross-document attention, repacking vs. fixed contexts, batch size effects) and cleanly isolate their roles. The diagonal comparison in Table 3 effectively demonstrates that packing is not equivalent to batch-size scaling.

The two-stage evaluation is conceptually well-aligned with the multi-hop reasoning objective, and metrics such as precision, hallucination rate, and accuracy are sensibly defined.
 
The discovery that cross-document attention and repacking frequency are key drivers of performance, and that excessive packing harms reasoning, provides actionable insights for practitioners training large models.
 
The paper is well-written, carefully structured, and empirically grounded, which facilitates comprehension of a complex systems–reasoning intersection.

### Weaknesses
what is the evidence showing that cross-document attention yields richer contextual representations (e.g., attention map analysis, representational similarity, or probing). Without such analysis, the causal link between packing and improved reasoning remains speculative.

Table 2 shows that packing increases compute, but there is no normalized comparison (e.g., accuracy per FLOP). Since the main selling point of packing is efficiency, the practical utility of adopting more expensive packing strategies remains uncertain.

### Questions
It is not clearly stated how does the training data look like, whether training data overlap with evaluation data (HotpotQA and 2WikiMultiHopQA corpora).  

why with cross-document attention enabled packing ends up increasing total compute to reach convergence.

Why does repacking outperform reshuffling?

Could the authors analyze attention patterns or activation similarities across packed documents to directly support the claim that cross-document attention enhances latent reasoning?

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
This paper explores whether packing documents together in a training sequence and enabling cross-document attention can improve the latent multi-hop performance of models. To achieve this, the authors continually pre-train three different models, using the documents from multi-hop datasets. They then instruction fine-tune those models to be able to perform multi-hop question-answering. They show that using packing (up to a certain amount of documents packed together) and enabling cross-document attention leads to substantial gains in latent multi-hop performance.

### Strengths
- The paper is well-written, easy to follow, and well-motivated. The importance of knowing how to pack documents to achieve the best knowledge retrieval from multiple documents at the same time is an important problem to solve.
- The experiments are thorough, and multiple ablations were carried out to justify the different choices (such as re-packing, or cross-attention). The paper shows that using packing during leads to substantial increases in performance in multi-hop question answering.

### Weaknesses
The main weakness of this paper is that the proposed continual pre-training seems to be closer to fine-tuning than actual pre-training. By packing documents that are assigned to a specific question in the dataset, and then performing cross-attention between them, the model is learning that if it retrieves one of the correct documents, then the model knows from learning to output the next document, which one is relevant. This could be seen as a form of leaking test information to the dataset. The results from the paper could be seen as reinforcing this point since packing a lot of documents (Pack 10) would make it harder for the model to know which k documents to select and increase the possible permutations. Also, it would explain the increase in hallucination seen in the no repacking since the model would expect to output a certain value article but gets a different title.

### Questions
- How many training epochs of the HotPotQA dataset does each packing strategy represent? (aka how many times is the dataset repeated)
- Were any other methods for choosing which documents to pack together tested?
- For the instruction-tuning and results, was there a split of the validation set held out to test on and get final scores from?
- For no repacking, was a larger packing tested (Pack 8 for example) to see if the increase in hallucination still exists?
- Was the 8B model trained for fewer steps than the others, or the same amount?

### Soundness
2

### Presentation
3

### Contribution
2
