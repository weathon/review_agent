# ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment

- Decision: Reject
- Scores: 1, 3, 6, 8

## Abstract
Data selection is crucial for optimizing language model (LM) performance on
specific tasks, yet most existing methods fail to effectively consider the target task
distribution. Current approaches either ignore task-specific requirements entirely
or rely on approximations that fail to capture the nuanced patterns needed for tasks
like Autoformalization or code generation. Methods that do consider the target
distribution often rely on simplistic, sometimes noisy, representations, like hashed
n-gram features, which can lead to collisions and introduce noise. We introduce
ZIP-FIT, a data selection framework that uses gzip compression to directly
measure alignment between potential training data and the target task distribution. Our key insight is that compression-based similarity captures both syntactic
and structural patterns relevant to the target task, enabling more precise selection of truly task-relevant data. In extensive evaluations on Autoformalization and
Python code generation, ZIP-FIT significantly outperforms leading baselines
like DSIR and D4. Models trained on ZIP-FIT-selected data achieve their lowest cross-entropy loss up to 85.1% faster than baselines, demonstrating that better
task alignment leads to more efficient learning. In addition, ZIP-FIT performs
selection up to 65.8% faster than DSIR and two orders of magnitude faster than
D4. Notably, ZIP-FIT shows that smaller, well-aligned datasets often outperform larger but less targeted ones, demonstrating that a small amount of higher
quality data is superior to a large amount of lower quality data. Our results imply that task-aware data selection is crucial for efficient domain adaptation, and
that compression offers a principled way to measure task alignment. By showing
that targeted data selection can dramatically improve task-specific performance,
our work provides new insights into the relationship between data quality, task
alignment, and model learning efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
This paper introduces an innovative, embedding-free data selection method for efficient fine-tuning of large language models. Drawing inspiration from gzip compression techniques, the authors propose utilizing Normalized Compression Distance as a metric to filter and prune fine-tuning datasets. The authors conduct a comparative analysis with prior embedding-free methods, originally designed for filtering pre-training datasets, on Autoformalization and Python coding tasks.

### Strengths
(1) Problem Significance: The author tackles a crucial problem in low-resource settings, addressing the challenge of fine-tune data selection without relying on GPU-intensive and embedding-based methods. This is a highly relevant and impactful research direction.

(2) Innovative Filtering Criterion: The authors' inspiration from gzip compression methods has led to the proposal of a novel and intriguing selection criterion. This approach is not only interesting but also demonstrates out-of-the-box thinking, making it a notable contribution to the field.

### Weaknesses
(1) **Inadequate Baselines**: The authors propose a data selection method for model alignment, but only compare it with prior works such as DSIR and D4, which were primarily designed for data selection during the pre-training phase. A more comprehensive literature review on data pruning methods for model alignment is lacking, including embedding-based methods [1], LLM model response metrics [2], Gradient-based metrics [3],Quality metrics judged by LLMs [4], inference loss on evaluation sets [5].

(2) **Evaluation Metrics**: The authors primarily use test data cross-entropy loss as the evaluation metric, results are thus not surprising given that the data selection method uses the test data to anchor the selection criteria. However, the authors do not compare their results with widely accepted metrics in the research community for the studied downstream tasks, such as:

(a). Autoformalization: proof success rates on miniF2F [6,7]

(b). Python coding: functionality pass rates (pass@k on HumanEval) based on unit-tests [8,9]


(3) **Clarifications on Motivation**: In Section 2.3, the authors argue that n-grams fail to capture syntactic or structural relationships within the data, while hypothesizing that gzip does. However, this hypothesis is not supported by theoretical or empirical evidence, weakening the motivation for the proposed approach. It is also not compared on if the proposed approach is better or worse than high-resource methods, such as embedding-based methods.

References: 

[1] DEFT-UCS: Data Efficient Fine-Tuning for Pre-Trained Language Models via Unsupervised Core-Set Selection

[2] From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning

[3] LESS: Selecting Influential Data for Targeted Instruction Tuning

[4] Alpagasus: Training a better alpaca with fewer data

[5] Instruction Mining: Instruction Data Selection for Tuning Large Language Models

[6] Autoformalization with Large Language Models

[7] LEGO-Prover: Neural Theorem Proving with Growing Libraries

[8] Evaluating Large Language Models Trained on Code

[9] Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation

### Questions
(1) Could the authors provide additional evidence to support the claim that gzip is effective in capturing syntactic and structural relationships in textual sequences?

(2) Would the authors be able to demonstrate the effectiveness of their approach using evaluation metrics beyond cross-entropy test loss, and compare it to relevant baselines, such as those mentioned earlier?

(3) Could you provide more insight into why D4 was excluded from the code generation experiments, and specifically how it affected model performance?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes ZIP-FIT, an efficient, embedding-free method for selecting high-quality, domain-specific fine-tuning data for language models (LMs). Prior methods often rely on computationally expensive neural embeddings or classifiers to filter aligned datasets, while those based on N-gram similarity may lack the structural depth needed for complex tasks like code generation. In contrast, ZIP-FIT leverages gzip compression to evaluate data alignment with target domains, based on the idea that compression algorithms encode information in a way similar to neural networks. The ZIP-FIT approach eliminates the need for LM forward passes to obtain embeddings, making it efficient and particularly suitable for low-resource environments. Experimental results show that ZIP-FIT outperforms prior data selection methods, such as DSIR and D4, as measured by test loss.

### Strengths
- This paper is well-presented and well-motivated.
- Studying computation-efficient methods for data selection in LLM instruction fine-tuning is a promising research direction.
- The proposed ZIP-FIT is intuitive and easy to follow.
- The proposed approach bypasses the need for LLM forward computation to obtain embeddings, making it computationally efficient.
- The presented experimental results seem promising.

### Weaknesses
- [Major] The proposed method seems very simple and straightforward; using a gzip-style method to embed data appears to be a relatively standard approach.
- [Major] All experimental results are based on test loss, which may not be very reliable. It would be essential to conduct evaluations on some standard benchmarks, such as HumanEval and MBPP for code evaluation, to demonstrate the scores the model can achieve.
- It is unclear how the proposed ZIP-FIT compares to prior, more complex data selection methods in terms of both running speed and final model quality (e.g., [1]), aside from deduplication approaches like D4.
- [Minor] The paper seems to be written somewhat in rush, the figure quality of Figure 2 does not seem to be very high.

[1] https://arxiv.org/abs/2405.00705

### Questions
As specified in the "Weaknesses" section:
- What is the score of the fine-tuned LLM using ZIP-FIT on benchmarks like HumanEval and PubMedQA compare to LLMs fine-tuned without using ZIP-FIT?
- How does ZIP-FIT compare to prior method like https://arxiv.org/abs/2405.00705 in terms of both running time and final model score?

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
4

### Summary
This paper introduces ZIP-FIT, an embedding-free data selection method leveraging gzip compression to measure the alignment between training and target domains. Unlike existing approaches that rely on neural embeddings, ZIP-FIT uses a computationally efficient compression-based alignment metric, enabling faster data selection while maintaining high relevance to the target task. Empirical evaluations demonstrate ZIP-FIT’s superiority over baselines DSIR and D4 in AutoFormalization and code generation tasks, achieving significantly faster convergence and lower cross-entropy loss with reduced computational costs. ZIP-FIT’s promise lies in its scalability and effectiveness, particularly in low-resource settings, where traditional embedding-based methods may be impractical.

### Strengths
1. ZIP-FIT’s embedding-free approach is a refreshing deviation from common embedding-based methods, offering a novel solution by leveraging gzip compression. The concept of using normalized compression distance (NCD) as an alignment metric is insightful and could inspire future research in embedding-free methodologies for various data selection tasks.
2. The empirical results support the claims, showing that ZIP-FIT achieves faster convergence and better performance than established methods. The experiments were conducted on both AutoFormalization and code generation tasks, demonstrating ZIP-FIT's versatility across different domains.
3. The paper is well-structured, with a clear exposition of the algorithm, experimental setup, and results. The figures effectively illustrate the performance benefits of ZIP-FIT.
4. ZIP-FIT could represent a significant advancement in data selection for machine learning, particularly in computationally constrained environments. Its potential to optimize model fine-tuning with minimal resource requirements makes it highly applicable for real-world use cases, especially in domain-specific and low-resource applications.

### Weaknesses
1. While ZIP-FIT achieves excellent results on the tasks tested, its reliance on gzip compression may limit its effectiveness in complex semantic domains where relationships are nuanced and less compressible. Embedding-free approaches, while efficient, may not be ideal for tasks that require deep semantic understanding or complex syntactic relationships.

### Questions
1. Could you provide further insights into how ZIP-FIT might perform with data that have higher variability and diverse syntactic structures, such as conversational datasets?
2. Can you clarify the theoretical basis for using gzip compression over other compression methods that might exploit redundancy differently? Would alternative compression algorithms affect the performance of ZIP-FIT?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a new data selection mechanism based on text
compression distances. The concept of using compression methods for
deep learning follows several modern practical results and theoretical
motivations that language modeling is fundamentally based in text
compression. The method's conceptual simplicity combined with strong
empirical results make it stand out as a modern way for filtering for
aligned data.

### Strengths
The paper is concise, sound, well written, and the experimental section shows promise for the method, especially with regard to other embedding-free methods.

The conceptual simplicity combined with the empirical results of the method is an especially strong point of the work.

### Weaknesses
Ideally, it would be shown how the size of $n$ (i.e., number of
samples from the target domain $p$) influences the performance of the
method. If it is possible to pick $n$ just sufficiently large enough,
it would greatly improve the computational efficiency of the method
for large target datasets.

Experiments in other domains would be really nice to better
demonstrate the generalization capabilities of the method. Possibly
there is data that is not well-suited to compression and accordingly
ZIP-FIT, or where the data's compression factor varies too much
between samples?

### Questions
### Minor comments

Figure 3, page 5:  
The color bar is labeled "Gzip Alignment" instead of
"ZIP-FIT-Alignment" from Algorithm 1; it may be confusing to readers.

Figure 3, page 5, line 231:  
Please mention also in the figure caption that the test loss is
calculated on ProofNet data.

### Soundness
4

### Presentation
4

### Contribution
4
