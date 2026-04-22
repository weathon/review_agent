# READER: Retrieval-Assisted Drafter for Efficient LLM Inference

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Autoregressive Language Models instantiate a factorized likelihood over token sequences, yet their strictly sequential decoding process imposes an intrinsic lower bound on inference latency. This bottleneck has emerged as a central obstacle to the scalable deployment of large-scale generative models. Existing acceleration techniques partially mitigate token-level latency &mdash; by relying on auxiliary draft models or introducing an additional training phase &mdash; but fail to address the dominant memory and communication costs. We present READER (Retrieval-Assisted Drafter for Efficient LLM Inference), a provably lossless speculative decoding framework that reuses an existing trained draft model as its backbone and requires no additional retraining. READER formalizes speculative decoding as a stochastic tree construction problem and exploits the empirical redundancy structure of natural language to generate high-probability candidate continuations. Our method revisits the problem of constructing draft trees, establishing substantial statistical improvements over stochastic draft-tree methods and providing a complexity-theoretic analysis that characterizes the optimality frontier of speculative decoding under bounded computation and memory resources. Beyond the single-sequence regime traditionally considered in prior work, we introduce a memory-optimal key-value cache-serving strategy that guarantees amortized sublinear overhead in the batch dimension, allowing READER to scale to realistic inference workloads. Comprehensive experiments demonstrate up to 6.13× wall-clock speedup on single-prompt inference and up to 5.92× on batched inference &mdash; consistently surpassing prior speculative decoding baselines &mdash; while preserving exact output equivalence, with even more pronounced gains in retrieval-augmented generation pipelines. Our results close a key gap between theoretical parallelism limits and practical LLM inference, suggesting a new standard for efficient deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes READER, a lossless speculative decoding method that augments an existing draft model with a statistical search branch drawn from self-repetitions and a datastore, thereby deepening the decoding tree and increasing the average acceptance length. 

It also introduces a KV-cache rearrangement technique to mitigate memory and bandwidth bottlenecks in large-batch settings. In experiments, READER achieves speedups up to ~4–5× over AR decoding.

### Strengths
1. The combination of retrieval-based speculative decoding with model-based speculative decoding is quite interesting and novel, in my opionion. 
2. The speedup number is quite impressive.

### Weaknesses
1. > bypasses the training of the auxiliary draft model

This is a bit misleading as the method still needs a trained draft model. 

2. I feel Section 3 is also part of the methodology and maybe Section 4 should be renamed to improve clarity. 

3. Section 4 is a bit confusing on its own. Most of the algorithm details are in appendix. It will be helpful if a more concise version pseudo-algorithm on the overall method is included in Section 4.

4. The evaluation is a bit too short (only one and a half page) and not comprehensive enough. First, why are most of the baseline results for MT-Bench omitted? Also, the ablation study should be accompanied with some tables or figures. For example, what are the raw values for the percentage improvement with tree optimization, statistical search branch, tree deepening, and KV cache rearrangement? What are the experiment setting (like GPUs used and models)? How is the accept length affected? In my opinion, the evaluation section is not well-written and is the major weakness of this paper.

5. In general, I feel the paper writing needs a lot of improvement. It should focus on highlighting the most important contributions (e.g. retrieval-augmented tree) clear, instead of explaining existing approach, like in Section 3 and 4. As a result of the lengthy methodology section, there is not enough room for evaluation, which significantly weakens the paper's claim.

6. Will the code be open-sourced?

### Questions
See the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces READER, a novel, training-free framework for accelerating Large Language Model inference via speculative decoding. The core contribution is the design of a heterogeneous draft tree that combines a standard, model-based speculator with a retrieval-assisted drafter. This retrieval component leverages both a short-term trie of the current context and a long-term, pre-built datastore (indexed by a suffix array) to generate high-probability, repetitive text sequences. The paper provides a theoretical analysis that frames speculative decoding as a tree-construction optimization problem and introduces practical system optimizations like a periodic KV cache rearrangement to maintain efficiency at large batch sizes. Experiments show that READER achieves significant wall-clock speedups (up to 6.13x) over standard autoregressive decoding and consistently outperforms strong speculative decoding baselines, with particularly pronounced gains on RAG tasks.

### Strengths
1. The hybrid approach of combining a learned draft model with a training-free, retrieval-based drafter is a novel and highly effective idea. This allows the system to leverage the best of both worlds: the draft model's ability to generate novel text and the retrieval system's efficiency in handling common, repetitive sequences.

2. The paper is well-grounded in theory, formalizing the speculative decoding process as a throughput optimization problem over a heterogeneous tree. This provides a principled foundation for the method and guides the design of the draft tree structure, elevating the work beyond a simple engineering heuristic.

3. The empirical results are strong and compelling. The method demonstrates significant and consistent speedups across multiple models and benchmarks. The over 10x speedup on RAG tasks is particularly impressive and highlights a key domain where this approach offers a transformative performance improvement.

4. The work is systems-aware, directly addressing the practical bottleneck of KV cache management at large batch sizes. The proposed periodic rearrangement is a simple yet effective solution that demonstrates a thoughtful approach to real-world deployment challenges.

### Weaknesses
1. The primary concern is that the method's impressive performance may be heavily skewed towards tasks with high textual repetition, which is the ideal scenario for a retrieval-based drafter. The remarkable >10x speedup on RAG, where the model copies extensively from the context, and the strong performance on code generation are clear evidence of this. However, the benefits might be substantially lower for more open-ended, creative, or complex reasoning tasks that require generating novel text with low repetition. This raises questions about the generality of the speedups and whether the method's effectiveness is confined to specific, repetition-heavy domains.

2. The proposed READER system introduces considerable complexity compared to purely model-based speculative decoding methods. It requires implementing and maintaining a short-term trie, building and loading a large external datastore with a suffix array, and managing parallel CPU-based search queries alongside GPU-based model inference. While the results are faster, this added engineering and maintenance overhead might be a significant barrier to adoption for some use cases.

3. The performance of the retrieval component is inherently dependent on the quality, size, and domain-relevance of the external datastore. While the paper uses a general-purpose dataset, deploying READER for a specialized domain would likely require curating a new, domain-specific datastore. This introduces a new dependency and potential for performance degradation if the datastore is poorly matched to the inference task, a limitation not present in self-contained, model-only approaches.

### Questions
1. Could you discuss the performance of READER on tasks characterized by low textual repetition, such as creative writing or abstract summarization? How much does the speedup diminish when the retrieval component finds few or no matches in the context or datastore?

2. How sensitive is READER's performance to the content and domain of the long-term datastore? For instance, how would the system perform on a medical QA task if the datastore is built from a general web corpus like the one used in the paper?

### Soundness
3

### Presentation
4

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
This paper presents READER (Retrieval-Assisted Drafter for Efficient LLM Inference), which accelerates speculative decoding by constructing a heterogeneous draft tree that blends search- (retrieval-) derived tokens with tokens proposed by a draft model.

### Strengths
1. The paper presents an investigation to combine both retrieval-based and drafting-based speculative decoding. The exploration efforts should be encouraged.
2. The authors conduct comprehensive experiments on a wide range of text generation benchmarks. The models include Llama2-7B, Vicuna-7B, and Llama3.1-8B.
3. It is worth noting that the authors consider the KV cache management in the speculative decoding paradigm and conduct experiments with batched settings, which are valuable in practice.

### Weaknesses
1. **The writing is poor and the demonstrations are confusing**. This manuscript is poor in writing, and readers may find it difficult to grasp the main contributions of the designed methodology. Detailed errors and problems are noted in the questions 1-5 below. I strongly recommend that the authors polish this manuscript further for the next submission.
2. **Lack of detail in the methodology**. I understand that READER aims to accelerate speculative decoding by constructing drafts augmented with search-based tokens. However, the writing of the methodology part is confusing and lacks many technical details. I could not effectively understand any specifics of the designed method. Particularly, your method is based on the Eagle series, right? Then, how do you construct the extra context or retrieval corpus for searching? What is the size (of tokens/documents) of your corpus? In Lines 308-309, you mentioned that "we augment the system with a large auxiliary datastore comprising many responses, ideally produced by the base model". How many responses are in this datastore? These details are missing.
3. **Lack of comprehensive experiments**. This manuscript only includes the experimental results of Tables 2 and 3. In Table 2, more baselines are required, such as REST for LLaMA-2-7B, Eagle-2, and other retrieval-based baselines, such as PLD and token recycling.



[1] Prompt Lookup Decoding. Apoorv Saxena. 2023. github.com/apoorvumang/prompt-lookup-decoding.

[2] Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling. Luo et al. ACL 2025.

### Questions
Most of my primary concerns are outlined in the weaknesses section above. Here are additional minor concerns:

1. There exist multiple misuses of \citet and \citep in this manuscript. You could find errors in Lines 43, 143, and 440.
2. Too many grammar errors in this manuscript that severely impact the reading. For example, in Line 4-58, what is the meaning of "the cost of this step-by-step process scales poorly"?; In Line 51, "A promising line of work seeks to reduce this sequential bottleneck is speculative decoding"; In Line 163-164, the word "iff" should be fixed to "if".
3. The alternate use of "the base model" and "the main model" that refers to the target LM (or the model that conducts verification) may confuse readers.
4. Section 3.1, which introduces the background of speculative decoding, is poor in writing and does not clearly demonstrate the basics of the paradigm. For instance, what is the detailed process of "drafting-then-verify"? How does speculative decoding guarantee the lossless quality of the outputs?
5. Section 3.2 recaps the basics of tree decoding. However, it ignores the most important implementation in this technique, which is the tree attention masks in the verification process.

### Soundness
2

### Presentation
1

### Contribution
2
