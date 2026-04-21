# Structured Packing in LLM Training Improves Long Context Utilization

- Avg Score: 4.00
- Decision: Reject
- Scores: 3, 3, 6

## Abstract
Recent advances in long-context Large Language Models (LCLMs) have generated significant interest, especially in applications such as querying scientific research papers. However, their potential is often limited by inadequate context utilization. We identify the absence of long-range semantic dependencies in typical training data as a primary hindrance. To address this, we delve into the benefits of frequently incorporating related documents into training inputs. Using the inherent directory structure of code data as a source of training examples, we demonstrate improvements in perplexity, even for tasks unrelated to coding. Building on these findings, but with a broader focus, we introduce Structured Packing for Long Context (SPLiCe). SPLiCe is an innovative method for creating training examples by using BM25 to collate the most mutually relevant documents into a single training context. Our results indicate that SPLiCe enhances model performance across various tasks and can be used to train large models to utilize long contexts better. We validate our results by training a large 3B model, showing both perplexity improvements and better long-context performance on a benchmark key-value retrieval task.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a novel method called Structured Packing for Long Context (SPLICE) aimed at improving the context utilization in long-context Large Language Models (LCLMs). The authors identify that the lack of long-range semantic dependencies in typical training data hinders the effective utilization of context in LCLMs. To address this, they propose incorporating related documents more frequently into training inputs. By using BM25 to collate the most mutually relevant documents into a single training context, the authors demonstrate that SPLICE can enhance model performance across various tasks and can be used to train large models to better utilize long contexts. The method was validated by training a large 3B model and showed improvements in perplexity and better long-context performance on a benchmark key-value retrieval task.

### Strengths
The paper introduces an innovative method to improve the context utilization of LCLMs. The SPLICE approach, which involves structuring training data using BM25, is interesting and it can be applied to any textual data, making it more generally applicable.
The paper demonstrates that the application of SPLICE results in improvements in perplexity across various tasks.

### Weaknesses
1.  Using Lexical matching methods to concatenate the documents into a longer one is a very engineering technique and it is a straightforward solution to construct longer samples. 

2. The experimental results are almost based on PPL, lacking experiments on real-world tasks. More experiments on benchmarks such as zeroScrolls[1] or L-Eval[2] to validate their models are needed. More extensive testing across a broader range of tasks and datasets would provide a more comprehensive evaluation of the method.

3. Presently, the prevalent strategies for training long context models involve the use of extensive conversations and literary works. A comparative analysis of SPLICE with these existing methodologies is thus a necessary step.

[1] ZeroSCROLLS: A Zero-Shot Benchmark for Long Text Understanding, 2023
[2] L-Eval: Instituting Standardized Evaluation for Long Context Language Models, 2023

### Questions
1. How does the choice of the BM25 method for document retrieval affect the model's performance? Would other document retrieval methods yield similar results?

2. How can we SPLICE  on very large pertaining corpus which usually has more than 400B tokens?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes structured packing for long context (SPLiCE) that constructs long context training examples by retrieving relevant documents using BM25. After experiments on a small language model with different datasets and configurations, SPLiCE is applied to large-scale language models.

### Strengths
The main idea of constructing better training examples makes sense. SPLiCE is not too complicated and does not require expensive overhead or external models by relying on BM25.

### Weaknesses
Considering the additionally introduced complexity (though the SPLiCE algorithm is simple), the performance improvement looks very marginal, especially for large-scale models. 
Only the part of the training is replaced with SPLiCE from the random baseline. That might be one of the reasons for marginal improvement, but it also implies that SpLiCE is not a standalone solution that can completely replace the existing training algorithm. 

Language modeling perplexity is the main evaluation metric. Comparing performance on other NLP downstream tasks that require long context modeling might be better to evaluate the effectiveness of the proposed method.

### Questions
I raised several concerns about why SPLiCE is not sufficient (or at least not fully validated) as it is. Could you address them?

I guess the number of neighbors for each document is skewed, meaning that there exists hub documents. In that case, although a root document is randomly sampled from the document corpus, the retrieved documents are not uniformly distributed in terms of their likelihood and order. Couldn't this be a problem that may result in an imbalance in training? 

Packed documents are unnatural and different from contiguous documents. Is there any way to alleviate this issue?

As expected, using related documents in a long context is better than the random baseline. However, any design choices (top-k, order, or even REPO vs. SPLiCE) give clear differences.
In particular, top-1 is the best, and in that case, BFS is the same as DFS. 

Why is Table 1 required? Table 2 fully covers Table 1.

The structure of the paper can be improved. For example, it is awkward that Section 4 also includes experiments while the title of Section 3 is experiments. Also, multiple Figures and Tables can be merged to spare some space for more extensive experiments or discussion.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes SPLICE, a similarity-based approach of grouping documents into pretraining examples to training better long context language models. For each example, the method starts with a single document and uses a BM25 retriever to include more relevant documents in the example in a BFS fashion. 

When applied to training a 270M model, the method outperforms the random baseline on both text and code perplexity. The model is also on par with the REPO method which relies on knowledge of the corpus structure. When used to train a 3B model, the method also outperforms the baseline on both perplexity and the key-value retrieval task. Ablation studies are included to analyze the impact of hyperparameters.

### Strengths
- The method is simple yet effective and can be easily applied to different scenarios.
- Reproducibility: The authors attach the source code, which is great. Please also release the code if the paper is accepted.
- Clarity: the paper is well written and easy to understand.
- Significance: the significance of the paper is okay.

### Weaknesses
- The effectiveness of the method is only validated on language modeling and the key-value retrieval task. This does not guarantee the resultant model is stronger on realistic use cases. To test the usefulness of SPLICE, I would highly recommend comparing the models on more challenging and realistic long-context downstream tasks such as Quality and Squality.
- It would be great if the method is tested on more settings: Use a neural retriever in addition to BM25, go beyond 3B and 32K, etc.
- Novelty: The main idea is quite similar to many existing methods like the ones discussed in the paper (e.g. retro). However, I don't think the paper should be rejected only because of this.

### Questions
- 3.4 “On the code evaluations, the improvements are small.” - Why say so? The average improvement on code datasets is 0.0625 and the improvement on arxiv is 0.07. The improvements seem to be similar.
- 4.2: “The perplexity difference is larger for tokens further in the document” - I might misunderstand something, but it seems the improvement is also large at the start? (Figure 3)
- Typo: 3.2 “Moreover, even thorough this method uses only the code data”
- In the abstract “Our results indicate that SPLICE enhances model performance across various task” - The context of this sentence is the 270M model. It is in fact only tested on one task: language modeling (though there are different datasets). You might want to rephrase to reduce confusion.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
