# Doc-to-LoRA: Learning to Instantly Internalize Contexts

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Long input sequences are central to in-context learning, document understanding, and multi-step reasoning of Large Language Models (LLMs). However, the quadratic attention cost of Transformers makes inference memory-intensive and slow.
While context distillation (CD) can transfer information into model parameters, per-prompt distillation is impractical due to training costs and latency. To address these limitations, we propose Doc-to-LoRA (D2L), a lightweight hypernetwork that meta-learns to perform approximate (CD) within a single forward pass. Given an unseen prompt, D2L generates a LoRA adapter for a target LLM, enabling subsequent queries to be answered without re-consuming the original context, reducing latency and KV-cache memory consumption during inference of the target LLM. On a long-context needle-in-a-haystack task, D2L successfully learns to map contexts into adapters that store the needle information, achieving near-perfect zero-shot accuracy at sequence lengths exceeding the target LLM’s native context window by more than 4x. On real-world QA datasets, D2L outperforms standard CD while significantly reducing peak memory consumption and update latency. We envision that D2L can facilitate rapid adaptation of LLMs, opening up the possibility of frequent knowledge updates and personalized chat behavior. Code and checkpoints will be released upon publication through GitHub and HuggingFace.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work points out that the quadratic attention cost of the Transformer is expensive. Therefore, this work proposes the Doc-to-LoRA (D2L), which meta-learns to perform approximate context distillation in a single forward pass. According to the experimental results, the model achieves about 4x the training context window.

### Strengths
* The idea is interesting. This work uses D2L Hypernet to process the chunks, and then uses the weights of each document to the LLM to generate the response.
* The paper is written clearly. Figure 1 presents the pipeline of the training process and the data construction. The method part describes how to create and use the activation.
* The experiment supports that the Doc-Lora could have great performance, with a few additional costs.
* This work provides the pseudo-code for the Doc-to-Lora, making the reproduction easy.
* The prompt is provided for reproduction.

### Weaknesses
* The baseline may be missed. This work does not compare the performance with other text-to-weight works.
* This work uses gemma for the experiment. It is not clear whether the claim is still the same as other models.
* It is not clear how the performance is on the distribution dataset. For example, if you train on PwC and test on SQuAD, what is the performance.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Doc-to-LoRA,  a hypernetwork that transforms a document into a LoRA. The objective is to produce a LoRA that mimics the behavior of the model with the document in context. The authors claim that this outperforms context distillation in both cost and quality.

### Strengths
- This paper addresses an important problem: how to reduce the memory consumption of in-context learning over large documents.
- The D2L method is simple and intuitive.
- Can extend the context length of the model 4x on the NIAH task. This result is very interesting!
- Compared with context distillation, it has significantly lower internalization cost.
- Well-written and easy to follow

### Weaknesses
- The main claim of the paper is that D2L “outperforms CD with improved internalization
efficiency.” This claim lacks nuance and is incautious. [Prior work](https://arxiv.org/abs/2506.06266) shows that performance of context distillation improves steadily as you increase the number of generated queries (up to hundreds of thousands of queries). However the authors compare against CD with only 5 generated queries for context distillation. Given this, the broad claim that D2L outperforms CD both in efficiency and quality lacks evidence. From the results presented in the paper, I am not convinced that D2L categorically outperforms CD in quality, it depends on how many generated queries we use. Indeed, [Prior work](https://arxiv.org/abs/2506.06266) has shown that CD can match the performance of the full context on the QASPER dataset, while D2L cannot in Figure 4.
    - So, my main concern with this paper is that it makes an absolute comparison between CD and D2L instead of discussing the cost-quality tradeoff between the two approaches. The evidence seems to suggest that CD still can significantly outperform D2L on long context tasks when we increase the number of generated queries.
    - Without a revision that explores these tradeoffs and presents more nuanced claims, I can’t recommend acceptance. With a more nuanced analysis of these tradeoffs and comparisons with more generated queries, I would raise my score.
    - I would recommend that the authors frame D2L and CD as complementary methods. The best method to use may depend on the setting and computational constraint. I could even imagine using D2L to initialize CD and speedup convergence.
- It is unclear how D2L performs relative to other existing hypernetwork-based methods such as Gisting, MEND, autocompressors, or in-context AutoEncoders. Text2Lora is the only hypernetwork based method compared against.
- Unlike recently proposed methods based on CD (*e.g.* https://arxiv.org/abs/2506.06266), D2L cannot match the performance of the full context.
    - D2LThe authors could also evaluate D2L on more challenging document understanding datasets that evaluate capabilities beyond QA. Some that come to mind are: FinanceBench, LongHealth, and Machine-translation from one book. Prior work in context distillation evaluate on these
- D2L interferes significantly with existing internal knowledge in the base LLM. [Existing context distillation](https://arxiv.org/abs/2506.06266) methods do not exhibit forgetting. It is possible that improving the meta-learning data generation pipeline, but the authors do not demonstrate this in this paper.
- The main difference between doc-to-Lora and prior work, text-to-Lora, lies in the distribution of context-query pairs used to train the hyper network. Given this, I would expect a description of the training distribution to feature prominently in Section 3, which describes the method. Instead, the data is only briefly described in the experimental setup section. Specifically, I’d like to better understand what prompts are used to generate the queries.
- There are a number of important questions left unanswered. See below.

### Questions
- Can the same strategy be used to produce a truncated key-value cache (instead of a LoRA), as in [prefix-tuning](https://arxiv.org/abs/2101.00190) and [cartridges](https://arxiv.org/abs/2506.06266)?
- How do the authors justify the choice of the perceiver architecture? What are the alternatives and how does performance differ?
- How does performance vary as you increase the rank of the LoRA adapter?
- Why does the model OOM with 40 context distillation queries? By keeping batch size constant, context distillation can be scaled to an arbitrarily large number of training queries. (*e.g. in [cartridges](https://arxiv.org/abs/2506.06266)* the authors scale to >100k context distillation queries).

### Soundness
1

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
5

### Summary
Several works have been exploring test-time-training and context distillation methods to compress key-value caches into a small number of learned weights. However, the training step needs to be re-done for each new context, which is slow and expensive. This work explores whether a hypernetwork that goes from the context to the compressed weights directly. The work explores the architecture and training protocol to accomplish this and performs a series of short and long context benchmark evaluations.

### Strengths
The project is well-motivated and clearly contextualized within the literature
It is interesting if it’s possible to create a hypernetwork that replaces the process of running gradient descent. 
The chosen architecture makes sense and is clearly explained
The paper chooses a comprehensive set of evaluation settings / benchmark tasks

### Weaknesses
There are places where the experiments and writing can be clarified. These are specifically provided below:


Questions about the current paper contents:
L076-079: The writing could be clarified: Why is chunking needed? Why is NIAH the right evaluation for success? Why is training limited to 256 tokens? 
L140-142: What type of data was used in the meta-training dataset? What was the process for generating the context, queries, and responses
L245-258: Why include samples from the QA datasets? Especially since those are also the chosen evaluation settings. How does the method perform on OOD datasets? How does the method perform on SQAD/DROP/ROPES if their samples are not included in the training data? Are the samples that you test on in the training data as well? 
L245-258: Why not just use the pretraining corpus? How were questions generated over FineWeb? What were the prompts, etc.?
Figure 4/L365-370: What is meant by CD OOMs? What is the GPU type being used / setup? Don’t prior works like Cartridges evaluate on documents up to 100-400K tokens? Why can’t you chunking the context to avoid OOMs during training (since some of your baseline papers do that)? Why is it meaningful to compare to CD with 5 queries? This claim is not clearly supported and this experiment doesn’t make sense to me.
L404-421: It would be more compelling to present scaling laws. How quickly does D2L performance increase as a function of training data? What properties of the training data led to this performance.

### Questions
L393-404: What does it mean to internalize a query and what is the motivation for the experiment?

### Soundness
2

### Presentation
2

### Contribution
2
