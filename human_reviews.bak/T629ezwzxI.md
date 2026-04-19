# Out-of-domain Fact Checking

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3

## Abstract
Evaluating the veracity of everyday claims is time consuming and in some cases requires domain expertise. In this paper, we reveal that large commercial language models, e.g., ChatGPT or GPT4, are unable to successfully accomplish this task. We then empirically demonstrate that the commonly used fact checking pipeline, known as the retriever-reader, suffers from performance deterioration when it is trained on the labeled data from one topic (or domain) and used in another topic. Existing studies in this area mostly evaluate the transferability of fact checking systems across various platforms, e.g., Wikipedia to scientific repositories, or from one fact checking website to another one. Even in doing so, they do not step beyond pretraining models on one resource and evaluating on another resource.  This calls for developing methods and techniques to make fact checking models more generalizable. Therefore, we delve into each component of the pipeline and propose algorithms to achieve this goal. We propose an adversarial algorithm to make the retriever component robust against distribution shift. Our core idea is to initially train a bi-encoder on the labeled source data, and then, to adversarially train two separate document and claim encoders using unlabeled target data. Then, we focus on the reader component and propose to train it such that it is insensitive towards the order of claims and evidence documents. Our empirical evaluations support the hypothesis that such a reader shows a higher robustness against distribution shift. To our knowledge, there is no publicly available multi-topic fact checking dataset. Thus, we propose a straightforward method to re-purpose two well-known fact checking datasets. We construct eight fact checking scenarios from these datasets, and compare our model to a set of strong baseline models, including recent models that use GPT4 for generating pseudo-queries. Our results signify to the effectiveness of our model. Our code will be publicly available on our GitHub webpage.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper discusses the challenges in evaluating the accuracy of everyday claims and highlights the limitations of large commercial language models. This work focuses more on improving the generalizability of fact-checking models by training tailored retrieval models. The authors propose an adversarial algorithm to make the retriever component more robust against distribution shifts. This method includes training a bi-encoder on labeled source data and adversarially training separate document and claim encoders using unlabeled target data.

### Strengths
1. The proposed problem is critical and interesting.
2. The retrieval results seem more robust and the retrieval model conducts more generalized results.

### Weaknesses
1. Some unsupervised methods are not compared, such as BM25. These methods have no domain shift problem.
2. The introduction describes the generalization ability of GPT3/4 on the fact verification task, but no experiments of GPT-3/4 are conducted.
3. The adversarial learning has been explored in previous work[1]. This work should discuss it.
4. Lots of related work[2-5] on fact verification is not discussed. I only list some of them.

[1] Zero-Shot Dense Retrieval with Momentum Adversarial Domain Invariant Representations.
[2] Fine-grained Fact Verification with Kernel Graph Attention Network.
[3] Exploring Listwise Evidence Reasoning with T5 for Fact Verification.
[4] GEAR: Graph-based Evidence Aggregating and Reasoning for Fact Verification.
[5] GERE: Generative Evidence Retrieval for Fact Verification

### Questions
Why do you not directly evaluate the fact verification performance of GPT-3/4?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper discusses the challenges and potential solutions related to out-of-domain fact-checking, which involves verifying facts that are not covered by traditional domain-specific fact-checking systems. The study presents a detailed case study to demonstrate the challenges, followed by proposed methods such as adversarial training for evidence retriever and representation alignemtn for reader. The effectiveness of these methods is evaluated through comprehensive experiments and ablation study.

### Strengths
1. This study creatively incorporates various out-of-domain algorithms to enhance the fact-checking process.
2. The experimental setup is commendable, with a thorough ablation study that effectively showcases the efficacy of the introduced methods.
3. The paper is eloquently written, ensuring clarity and ease of comprehension for readers.

### Weaknesses
1. It would be beneficial to include baselines evaluating prior fact-checking algorithms (SOTA or other general ones) within the experimental setting outlined in this paper (specifically, fine-tuning on source domain and testing on target domain). This would more effectively highlight the necessity and advantages of the methods proposed for out-of-domain fact-checking.
2. Although the enhancements brought about by the proposed methods are evident in Tables 3 and 4, the cumulative improvements showcased in Table 2 appear modest.
3. Drawing a direct comparison between the proposed fact-checking system and models like ChatGPT or GPT-4 might not be entirely justifiable. Notably, the proposed system has access to evidence sources, whereas the ChatGPT or GPT-4 versions evaluated do not. It would be intriguing to see a baseline involving a GPT model (without fine-tuning) assessed on your target domain test set.

### Questions
1. Were experiments conducted on prior fact-checking algorithms (SOTA or other general ones), evaluated on MultiFC and Snopes using your experimental setups?
2. In the concluding lines of the 2nd paragraph in section 3.1, you state, "a model trained only on the out-of-domain data typically does not perform as competitively as a model trained on in-domain data." Could you provide clarity on how you differentiate between out-of-domain and in-domain data? Is it implied that "general" or "Misc" categories are considered out-of-domain because they amalgamate data from diverse topics? Or are the terms "out-of-domain" and "in-domain" used interchangeably throughout the paper?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on domain adaptation for fact-checking, employing a typical unsupervised domain adaptation setting where two domains are provided: a labeled source domain and an unlabeled target domain. 
The objective is to improve the performance of a source-domain trained model on the unlabeled target domain. 
For the fact-checking task, the paper uses a standard retrieve-and-read pipeline comprising a bi-encoder retriever and a reader. 
To adapt the bi-encoder retriever to the target domain, the paper applies adversarial training to train the bi-encoder on the unlabeled target domain, enabling it to mimic the source-domain encoder. 
For adapting the reader model, the paper incorporates alignment loss in addition to cross-entropy loss during the training of the source-domain reader, integrating data from the target domain into the training process. 
Moreover, the paper also proposes a  data augmentation method by switching the order of the claim and evidence document in the reader's input. 
The authors also automate the conversion of two non-domain-adapted fact-checking datasets into datasets with domain labels using a fine-tuned classifier and test the proposed method on these datasets. 
The results demonstrate that the proposed method outperforms the non-adapted baselines on the target domains.

### Strengths
1. Overall, the paper provides a clear and comprehensible description of the proposed method, particularly the setup of the entire work and the articulation of the research problem.
2. The domain adaptation method proposed in the paper is straightforward and has demonstrated better performance on the tested datasets compared to non-adapted baselines.

### Weaknesses
1. I believe that the innovativeness of the domain adaptation method proposed in this article is quite limited. The paper mentions that previous work on fact-checking domain adaptation tasks is scarce, but it fails to elaborate on specific limitations and lacks a detailed comparison and analysis with any domain adaptation methods. From the description in the paper, it seems that the authors have only applied standard domain adaptation techniques, such as adversarial training and alignment loss, to the common retrieve-and-read framework. Consequently, I do not see much innovation in the method, and I find the discussion of related works insufficient. The paper should at least discuss the related domain adaption works.
2. The paper’s description of data augmentation for training the reader mentions that it addresses the issues of noise due to the lack of gold documents in the target domain and provides more cues to the reader. However, I believe the data augmentation method proposed does not effectively tackle these issues. Simply swapping the order of the claim and evidence document in the input sequence does not reduce noise and might even increase it. As for providing more cues, I do not think this is achieved by merely changing the order of content in the input sequence since it does not add any additional information. The paper also lacks any empirical analysis of this specific data augmentation.
3. The quality of the created datasets is not rigorously validated. The domain labels in the dataset are labeled based on an external classifier, and the validation of the constructed domain adaptation through a simple 2D projection seems insufficient. There is even no description of how the projection is conducted in the paper.
4. The paper does not provide adequate explanations for the baselines used, such as why they are reasonable baselines. Only listing the names of the baselines is not enough. Moreover, from the description, it appears that all the baselines used are simple retrieve-and-read methods, without domain adaptation methods, making the comparison with the adapted method in the paper unreasonable. The authors should consider comparing their method with other strong adaptation methods.
5. The paper lacks an ablation analysis of the components of the proposed method. For instance, the authors do not conduct ablation studies on adversarial training or alignment loss to prove their effectiveness in domain adaptation.

### Questions
1. Why existing methods can only solve the fact-checking domain adaptation problem in a very limited way? What are the specific limitations?
2. Why is it believed that data augmentation can alleviate the noise issue? Why do we need to handle the order issue? Isn’t everything tested in a certain order (claim + evidence) anyway?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor
