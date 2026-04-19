# Basel: Target-Aware Basis Selection for Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 5

## Abstract
As the size of language models increases, they deliver substantial performance improvements across a variety of applications. However, this growth also leads to greater computational demands, making deployment on resource-constrained devices—such as personal computers and mobile or wearable devices—more challenging, and significantly raising inference costs on cloud servers. To address these challenges, we introduce a method to streamline language models. We observe that language models pretrained on general datasets often include redundant components that are unnecessary for particular tasks. Our approach identifies and removes these redundant parts, retaining only the essential components for the intended applications. Specifically, we represent the weight matrices of language models as a linear combination of base components, eliminate the irrelevant bases, and introduce new bases that enhance performance for target tasks. Evaluations show that our method reduces model size much more significantly—by up to 1.7 times—while maintaining similar accuracy, compared to state-of-the-art techniques, across a range of applications.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose Basel, a target-aware low-rank model pruning and adaptive selection technique for large language models. Basel shows benefits over singular vector decomposition as it provides higher model performance at higher compression rates.

### Strengths
- Developing approaches for tailoring LLM applications to specific use cases is a highly relevant topic, as LLMs frequently offer much more capability than needed for downstream tasks. It also has a positive effect on the energy consumption and processing latency (i.e., end-user experience)
- The experimental design reflects a careful choice of models and datasets from different text domains.
- The manuscript is well-structured and easy to follow.

### Weaknesses
- Since the approach is iterative and gradually removes less relevant parameters, I would expect to see experimental results on how long it takes to adapt a model for the datasets. This could yield important insights on how expensive the Basel technique is.
- I would appreciate comparing performance and cost to well-established techniques such as knowledge distillation. I understand that, depending on the complexity of any given dataset, the cost may vary, especially with regard to retraining of the base (l. 209).
- The authors use reasoning benchmarks for their experiments only. What about language understanding (e.g., MMLU)? I would expect the language understanding capabilities of models to decrease, but at what point would this become noticeable?

### Questions
Please see the weaknesses.

### Soundness
3

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
4

### Summary
In this submission, the authors propose to compress LLMs for specific tasks by distinguishing between important and irrelevant parameters. The proposed method Basel, which is based on the main idea of SVD, aims to learn the scales of decomposed singular values and to learn additional bases from task-specific datasets. Experiments on math problem and code generation benchmarks demonstrate that Basel achieves better performance compared to SVD and FWSVD.

### Strengths
- The studied problem, i.e., reducing the computational costs of LLMs for specific tasks, is practical and important.
- The authors provide detailed descriptions of the proposed method, including background information, motivational insights, and algorithm process.

### Weaknesses
- The novelty of the proposed method is limited. As mentioned by the authors, there are existing works on utilizing singular values in LLMs and pruning based on importance. The technical contributions of this submission are not clear.
- The experiments are not convincing and lack some detail. (a) The training details of Basel are not provided. (b) Although the authors include lots of related works, they compare the proposed method against a very limited set of baselines. (c) Since Basel includes new trainable parameters (such as scales and additional bases) and is trained on task-specific datasets, it is not surprising that Basel can achieve better performance on these tasks. The conducted comparisons do not convincingly demonstrate the effectiveness of the proposed method. (d) The observations presented in Table 1 are interesting. The authors can consider providing similar observations for the LLMs trained using Basel to provide empirical evidence that it successfully identifies beneficial and redundant components.
- The writing of this submission should be further improved.

### Questions
Please refer to the Weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors proposed a method called BASEL, which can streamline language models and reduce model size up to 1.7 times while maintaining similar accuracy.
The method is based on Singular Value Decomposition (SVD)  and can identify and remove unnecessary components from pre-trained models, keeping only the essential parts needed for specific tasks.

### Strengths
1. The paper is well written
2. The method can reduce the model size while keeping comparable accuracy on specific tasks.

### Weaknesses
## 1. Lack of comparison with other model compression methods
Currently, the most common compression method in practical is model quantization, but the experimental part of this paper does not compare with it. A set of comparative experiments with the state-of-art quantization method should be added.
On the other hand, model quantization is not task-specific, but Bazel can only be effective for specific tasks, so some additional discussion is needed to explain the necessity of this method.


## 2. Lack of experiments on the overhead of performing the proposed compression method

A set of experiments should be added to show the time and resource cost of performing Basel compression

## 3. Limited scope of application
One of the main advantages of LLM over previous models is its versatility. However, the method proposed in this paper reduces the size of the model by sacrificing the versatility of the model and only focuses on specific tasks. And the pruning of the model also requires
The author needs to give a reason for doing so.

### Questions
As described  in Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents Basel, a new approach to compressing LLMs by identifying the importance of bases for a target application, pruning the others, and then finishing with a finetuning step. The approach is evaluated on two tasks, mathematical reasoning and code generation, and is compared to other SVD-based approaches.  The results show that for the llama2  model, Basel can achieve better performance than the other two approaches when the compression ratio exceeds 5.

### Strengths
Here are some of the paper's strengths: 
1. Solid Motivation: Deploying LLMs on resource-constrained devices is a relevant problem, and attempting to address it through compression seems reasonable.
2. The authors outlined their approach very clearly.

### Weaknesses
The paper has a couple of weaknesses listed below:
1. Focusing on low-rank compression approaches for comparison: The paper does not compare other compression techniques widely used to compress LLMs, such as quantization and knowledge distillation. Authors could potentially mention them in related works and why they think they are not comparable. 
2. Limited number of tasks and models: This paper could benefit from running more evaluations on different tasks other than mathematical reasoning and code generation. Additionally, it might be good to show that the approach is generalizable to other models of various sizes.
3. Some terms in the equations are not very well defined. For example, tr() in equation(3) is not defined.

### Questions
No questions for the authors.

### Soundness
3

### Presentation
2

### Contribution
2
