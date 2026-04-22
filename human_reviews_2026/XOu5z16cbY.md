# Pretraining with hierarchical memories: separating long-tail and common knowledge

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6, 2

## Abstract
The impressive performance gains of modern language models currently rely on scaling parameters: larger models store more world knowledge and reason better. Yet compressing all world knowledge into parameters is unnecessary, as only a fraction is used per prompt, and impractical for edge devices with limited inference-time memory and compute. We address this shortcoming by a memory-augmented architecture and a pretraining strategy aligned with existing hardware paradigms. We introduce small language models that access large hierarchical parametric memory banks encoding world knowledge. Our pretraining learns to store long-tail world knowledge in the memory parameters, while the small language model acts as an anchor capturing common knowledge and general reasoning abilities. Through trillion-token-scale experiments, we show significant gains: a 160M-parameters model augmented with an 18M-parameters memory fetched from a 4.6B memory bank obtains comparable performance to a regular model with more than 2x the parameters. We study the optimal type and size of parametric memories in transformers, scaling them to over 21B parameters. We find that our proposed hierarchical feed-forward memories work robustly across transformer architectures, whether added during pretraining or post-hoc.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an efficient approach for learning rare knowledge in LLMs than trying to compress them in LLMs parameter. To that, this paper proposes a hierarchical memory bank augmented architecture and a pretraining strategy which demonstrate to achieve similar performance with a small model(roughly half the size of LLM parameters) + very small amount retrieved memory bank parameters. This approach helps with catastrophic forgetting problem, as memory parameters are activated and updated only on sequences of similar topics, thereby reducing susceptibility to forgetting.  The paper discuss 3 approaches of retrieved memory parameter augmentation with the anchor model parameters. There are other related work such as memory layers where FFNs are replaced by learnable memory layers. This work seems a bit incremental although the hierarchical memory back is different.

### Strengths
Overall, this paper makes a solid contribution by proposing an approach well-suited for on-device inference. The compact anchor model can reside in fast local memory, while the larger set of knowledge parameters is stored in slower, high-capacity memory. By decoupling knowledge from reasoning capabilities, the method establishes a clear link between training tokens and specific parameter subsets (memories). This design allows targeted selective knowledge update without altering the core anchor model, which can remain publicly accessible.

### Weaknesses
This work seems a bit incremental over other memory architecture. There are other parametric memory work, such as memory Layers at scale. It would be good to compare the results with such alternative approaches.

### Questions
For the experiments, It would be good to address the following
1. There are other parametric memory work, such as memory Layers at scale. It would be good to compare the results with such alternative approaches.  
2. It would be good to share results on how high quality RAG combines with this approach

Other comments:
1. While there is a section discussing adding memory bank on various open models, It would be good to have a discussion regarding if the “learnt” sparse memory bank is transferable, meaning, can be combined with another model with minimal tuning. 
2. A more rigorous scaling law analysis such as how does the results scale with increasing number of parameters per memory block
3. How does it compare and combine with MoEs for memorization and reasoning

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
4

### Summary
This paper proposes a new pretraining framework for improved knowledge learning: it separates general and domain-specific knowledge in the pretraining corpus and assigns them to different components of the model. To preserve long-tail knowledge, the authors design a individual hierarchical memory bank that is selectively retrieved and updated when training on similar text corpora, ensuring each submodel in the bank becomes proficient at its assigned tasks. Eperiments show that models ranging in size from 160M to 21B parameters benefit from this memory-bank design while training costs remain moderate.

### Strengths
1. Although the hierarchical design is not new, applying it to language model pretraining to preserve long-tail information is innovative. 

2. The model-bank design choices — including parameter placement, parameter sizes, and retrieval ratios — are well supported by extensive preliminary experiments. 

3. The effectiveness of pretraining within this framework has been demonstrated at scale up to 1.4B-parameter models in the main page, indicating strong scaling potential. 

4. Plugging the memory bank post hoc into frozen models has also proven effective, demonstrating the design’s generality; moreover, the memory module can be quickly pre-trained on private or corporate data to meet different needs. 

5. The memory bank design enables on-device deployment by loading only the anchor model and retrieved parameters while storing the bulk of knowledge parameters in external storage.

### Weaknesses
1. The authors claim that using a base model as an anchor better captures common knowledge and reasoning capabilities, and that augmenting it with a memory bank benefits long-tail knowledge tasks. However, I did not find any evaluation of improvements in the system’s reasoning ability.

2. Although the authors argue that this hierarchical memory bank aligns well with modern computer memory design and therefore offers greater efficiency, they did not provide experiments to support this claim, offering only a brief theoretical analysis at the end of Section 3.

### Questions
1. In Figure 3a, is the fetched memory size determined by the memory size $s_2$? 

2. “For a fair comparison, during training we use the generic memory with probability 1/(16 + 1) and the fetched memory with probability 16/(16 + 1), where 16 is the clustering division factor. This ensures there is no training bias in favor of the memory bank parameters.” Can you clarify more on the design choice here? Why you choose this approach instead of training a standalone generic memory model?

3. I'm curious how you arrived at a 4.6B memory bank size, given the model bank parameters (256, 64, 16, 0). Can you provide a concrete example?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a memory-augmented architecture for language models that separates common knowledge (stored in "anchor" parameters) from long-tail knowledge (stored in hierarchical parametric memory banks). During pretraining and inference, the model retrieves context-dependent memory blocks from a large hierarchical memory bank organized via clustering of training data. The authors demonstrate that a 160M parameter anchor model augmented with 18M fetched parameters from a 4.6B memory bank achieves performance comparable to a 2 times larger standard model. The paper includes extensive experiments on memory types, scaling, and deployment considerations.

### Strengths
- The paper presents a well-motivated approach to separating common reasoning abilities from long-tail factual knowledge, with clear empirical evidence showing that hierarchical memories particularly benefit knowledge-intensive tasks (e.g., atomic number prediction improving from 1.7% to 67.8% for the 160M model).

- The experimental evaluation is comprehensive, covering multiple model sizes (160M to 1.4B parameters), memory configurations, and architectural variants, with thorough ablation studies on memory types, depths, and sizes that provide actionable insights for practitioners.

### Weaknesses
- The comparison with retrieval-augmented generation (RAG) seems somewhat unfair, as the authors use "vanilla RAG" without standard techniques like reranking or filtering, and the improvement over baseline when using high-quality datastores (Wiki-En) is actually comparable to the memory approach on specific-knowledge tasks.

- There's insufficient discussion of failure modes and limitations - for instance, what happens when the clustering assigns dissimilar documents to the same cluster? The paper would benefit from error analysis showing when and why the approach fails.

### Questions
Is it possible to get some analysis or visualization of what knowledge is actually stored in memory parameters versus anchor parameters? For example, can you show examples of specific facts that are reliably stored in particular memory blocks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a hierarchical memories pretraining framework that enables small models to achieve large-model-level knowledge coverage. By integrating hierarchical clustering-based memory retrieval, the framework decouples world knowledge from model parameters and separately stores common knowledge and long-tail knowledge. Comprehensive evaluations across 13 benchmarks show that the proposed framework achieves notable improvements in efficient hardware implementation and enhanced data privacy.

### Strengths
1. The motivation is clearly presented. The paper designs a hierarchical framework combining an anchor model and a memory bank to separately common knowledge and long-tail knowledge.
2. The proposed method is carefully designed, and its scalability is validated through empirical results on multiple benchmarks.
3. The paper is generally well-organized and readable.

### Weaknesses
1. The theoretical contribution is relatively limited. While the paper provides empirical evidence that separating the anchor model and memory bank reduces forgetting and improves training stability, the claim lacks theoretical justification or formal analysis.
2. Per-dataset results are not fully presented. The paper mainly reports averaged results (e.g., Avg-CK and Avg-SK) across 13 benchmarks, which provides a concise overview but may obscure dataset-specific behaviors. 
3. Lack of empirical evidence for the claimed knowledge separation. The paper presents a clear motivation that hierarchical memories aim to separate common reasoning abilities from long-tail knowledge. However, semantic clustering alone is insufficient to guarantee that higher levels correspond to common knowledge while deeper levels represent long-tail knowledge.
4. Lack comparison on inference cost. The anchor model+memory bank design may require frequent memory retrieval during each inference task. The authors should provide the average delay cost of their design against baselines.
5. Experiment hardware. The authors should disclose the hardware usage involved in pre-training and inference.
6. Typo. In Figure 2, level 4 should have 64k clusters, instead of 65k.

### Questions
1. The figure obscures key lines and does not specify which parameters were fixed. Could the authors clarify what factors were controlled? In addition, some key results in the figure are visually obscured.
2. Is the model performance strongly correlated with the clustering quality of ϕ(x)?
3. How is the number of levels p or division factor k chosen, and how sensitive is performance to these hyperparameters?
4. It is unclear whether the retriever R(x;W) is trained jointly with the anchor model, or whether its cluster assignments are frozen after initialization. How are gradients propagated through R(x;W)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a memory augmented architecture to improve the efficiency of language models. The system uses a small "anchor" model for common knowledge and reasoning, which is augmented by a large "hierarchical memory bank" designed to store long tail facts. A clustering based retriever selects a small, context relevant set of memory parameters from this bank to be added to the anchor model during inference. The authors present experiments showing that this method can improve performance on knowledge intensive tasks, particularly for rare facts, and that their 160M parameter model with 18M of fetched memory can perform comparably to a larger, 410M parameter baseline model.

### Strengths
The paper is clearly written and the core concept is easy to understand. The use of figures, especially Figure 1 and Figure 2, helps illustrate the architectural idea and its intended effect on long tail knowledge.

The authors were thorough in ablating their own method's design choices. The paper includes a systematic study of different memory types (FFN, LoRa, KV), the impact of memory depth, and the effects of bank size versus fetched memory size. The analysis in Figure 1, showing improved accuracy on rare elements, provides a clear demonstration of the mechanism working as intended for that specific task. The paper also shows the method can be applied post hoc to existing open weight models, which suggests some level of generality.

### Weaknesses
The paper's conclusions are undermined by significant weaknesses in its experimental design and unaddressed questions about the core mechanism:

1. Static Retriever: The entire method's effectiveness hinges on a static, offline clustering of the training data using an off the shelf Sentence BERT model. This is a major point of failure. The paper provides no sensitivity analysis on the choice of embedding model or clustering algorithm. If this initial, fixed clustering is suboptimal, the model has no way to adapt, and knowledge separation would presumably fail. The system's performance is completely dependent on this external component.

2. Unsupported Claims on Editing and Privacy: The paper makes strong claims about enabling knowledge editing and privacy. The evidence provided is extremely weak. Figure 6b only shows that blocking memory degrades performance on a specific task. This demonstrates the memory is being used, not that facts can be cleanly removed or edited. It fails to address whether long tail knowledge leaks into the anchor model during co training, which would make any privacy or editing claims invalid. 

3. Weak RAG Baseline: The comparison to RAG in Table 3 is not strong. The authors compare their method to a vanilla RAG that retrieves a single document from the low quality DCLM pretraining dataset, which they admit performs poorly. Modern RAG systems use far more sophisticated techniques (multiple documents, reranking, chunking). So, by benchmarking against a weak baseline, the paper's claims of superior efficiency and performance over RAG are not well supported.

4. Limited Novelty: The paper frames itself as a novel solution, but it can also be seen as an incremental combination of existing ideas. The concept of separating parameters for knowledge is central to MoEs, and memory augmented networks which are a long standing field of research. The primary novelty is the hierarchical structure and the clustering based pretraining, which as explained above is a weak and static process.

### Questions
The authors would need to address several critical points to make their claims more convincing.

1. Why was a static, offline clustering chosen over a learnable retriever that could co train with the model? can the authors provide any analysis on how sensitive the model's performance is to the quality of the initial clustering or the choice of embedding model?

2. Regarding the privacy and editing claims, can the authors provide direct evidence of knowledge separation? 

3. Given that the vanilla RAG baseline is weak, how does this method compare to a more robust, modern RAG implementation using a high quality datastore and standard techniques like multi document retrieval and reranking? Without this, the claims of superiority are difficult to evaluate.

4. When does the memory retriever fail? can the authors show examples of prompts where the retriever fetches the wrong or an irrelevant memory block, and how does the model perform in those scenarios?

### Soundness
2

### Presentation
2

### Contribution
2
