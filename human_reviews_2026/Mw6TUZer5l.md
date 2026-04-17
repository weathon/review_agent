# The Surprising Soupability of Documents in State Space Models

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
We investigate whether hidden states from Structured State Space Models (SSMs) can be merged post hoc to support downstream reasoning. Inspired by model souping, we propose a strategy where documents are encoded independently and their representations are pooled, via simple operations like averaging, into a single context state. This approach, which we call document souping, enables modular encoding and reuse without reprocessing the full input for each query. We demonstrate that finetuned Mamba2 models with souped representations achieve competitive or superior performance across multi-hop QA, sparse retrieval, and long-document reasoning tasks compared to the standard monolithic encoding approach. For example, on the RACE and QuALITY benchmarks for long document question answering, our method substantially outperforms a traditional concatenation approach. Crucially, this modular design scales to hundreds of documents---we test up to 256---while delivering substantial savings in inference cost, unlocking new possibilities for large-scale corpus reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies whether hidden states computed from multiple different documents can be composed post hoc, for downstream tasks such as QA and long-document reasoning, in SSMs (in particular Mamba-2). Towards this end, they propose the "souping" of document states, which are encoded independently, via commutative operators like averaging. The paper finds that a zero-shot approach is ineffective, and that "finetuning is crucial for unlocking soupability in pretrained SSMs".

### Strengths
The paper provides extensive experiments, across a variety of tasks, and scaling up to composing 256 documents.

The paper also studies how different fine-tuning approaches have varying degree of effects on the composability of the resulting states.

### Weaknesses
The paper's central investigation into 'document souping' is presented without acknowledging that this method was already introduced and evaluated as a baseline in [1], published in ICLR 2025. This omission is rather severe, as it overlooks the most direct prior work.

In particular, both papers are motivated by the exact same problem: the inefficiency of the monolithic, concatenation-based context processing in SSMs.  Their proposed solutions are conceptually almost identical -- (1) pre-processing individual documents by encoding them as fixed-size states, and (2) composing them in a commutative manner at inference time. Furthermore, the method proposed in [1] appears in fact superior, being theoretically-grounded and working better in both zero-shot and fine-tuned settings.

To elaborate, the core method of this work, which it terms "document souping" with simple averaging as the most effective operator, appears to be functionally identical to the "Soup" method that was not only studied in an even earlier paper [2] published on arXiv, but also explicitly implemented and evaluated as a **baseline** in [1]. What is even more concerning is that even though [2] is cited in this work, the fact that the proposed method here is identical appears heavily downplayed, with the only mention being:

> (Pióro et al., 2024) linearly combines task-specific hidden states for skill transfer. In contrast, our
method focuses on merging hidden states from disjoint document chunks, enabling compositional
reasoning across distributed corpora through simple aggregation strategies.

Consequently I strongly believe that this work not only fails to frame its contributions appropriately in the context of prior art, but also as a result is limited in its novelty and contributions beyond an empirical deep dive into an existing baseline.


[1] PICASO: Permutation-Invariant Context Composition with State Space Models. ICLR 2025.

[2] State Soup: In-Context Skill Learning, Retrieval and Mixing. arXiv 2024.

### Questions
See weaknesses

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses a problem of non-reusable representations of contextual corpora as a single fixed-length hidden state in the structured SSM. The proposed approach inspired by the model souping technique, allows the efficient procedure of encoding of contextual documents into reusable representations that can be pooled together for SSM reasoning. The experimental evaluations on multi-document and single long-document QA demonstrate that finetuned Mamba2 models with souped representations achieve competitive or superior performance compared to traditional concatenation-based encoding.

### Strengths
1. The paper is well-motivated and clearly presents the core challenge of single nonchangeable hidden state encoding for large contextual corpora for the task that supposes the modularity of the encoded representations.
2. The proposed mechanism is simple and well-described.
3. The superior computational efficiency of the proposed method over existing approaches is experimentally supported.

### Weaknesses
1. The limited generalizability and practical utility of the proposed method due to performance decrease in evaluations with the number of segments larger than used during training.
2. No statistical test or confidence intervals were reported to support the significance of the experimental results.
3. The paper lacks explicit comparisons of time and memory costs of the proposed method with document concatenation baselines to support the cost efficiency claim.
4. The paper does not provide any comparisons with related RAG methods.

### Questions
-

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes to "soup" documents in SSM space and use them for various tasks. They show that although they cannot do them out of the box, SSMs can be trained to do so effectively. They show this in a variety of eval settings and with a variety of different ways/numbers of docs. 

Overall, I think this paper is promising but I am concerned about the baselines. I am willing to improve my score if those are included (regardless of whether the results are better/worse than the proposed method).

### Strengths
- I think the topic of this paper is unique, both quite interesting and has surprisingly good results
- The proposed method seems to work across settings and as the number of documents increases
- The paper is well written and even includes an eval with standard transformers

### Weaknesses
1. My biggest concern is the lack of a FT'd baseline for these approaches. It seems a crucial comparison that is left out in favor for various different versions of the proposed technique (e.g. Table 1 and Table 3). It's also possible that I'm misunderstanding the baselines there, so please correct me if I'm wrong. Since the proposed version does FT'ing, I would expect all evals to have a baseline of the non-soup with fine-tuning.
2. [Minor] I think the experiment up to 256 docs needs to be in the main text, especially since it is referenced in the abstract. If accepted, I would strongly encourage the authors to do that.

### Questions
My questions are in the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces the method of document souping which combines hidden states from independently encoded documents in SSMs.  Each document is encoded using SSMs and the hidden representations are then pooled to be used a decoder conditioned on the pooled representations. The paper is then evaluated on Multi-Doc QA and Long Doc QA datasets, and outperforms the concat appraoch.

### Strengths
The paper shows the surprising result that the representations from SSMs can be pooled and used to answer question which allows for doucments to be parallely encoded offering speedups and cost savings. The paper shows the method scales to 256 documents and outperforms the concat appraoch on different datasets.  The idea and the results is interesting and would be of value.

### Weaknesses
One consideration that I would want the authors to test is instead of having random distractors, have 5-10 documents that each answer a different question and pool them together and test the singular representation on the different questions for the documents. My worry is that training on these documents (specifically the full finetuning) might have introduced some bias about what documents are more likely to answer questions or not and this will test if information from all of the documents is retained or not. 

The other thing not explicitly tested/mentioned is 1) the scaling with length of the documents and 2) how long are the documents, is it equal to the sequence length of the model for each dataset i.e. can I soup together 16 4k token length documents?

### Questions
What is the average length of the encoded document? 
What is the max length of the encoded documents is it equal to the max length of the model? 
How well does the model preserve information across different documents souped together?

### Soundness
2

### Presentation
3

### Contribution
3
