# Closing the Data-Efficiency Gap Between Autoregressive and Masked Diffusion LLMs

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Despite autoregressive large language models (arLLMs) being the current dominant paradigm in language modeling, effectively updating these models to incorporate new factual knowledge still remains difficult. They resist knowledge injection via fine-tuning due to inherent shortcomings such as the "reversal curse" — the challenge of answering questions that reverse the original information order in the training sample. Masked diffusion large language models (dLLMs) are rapidly emerging as a powerful alternative to the arLLM paradigm, with evidence of better data efficiency and free of the "reversal curse" in pre-training. However, it is unknown whether these advantages extend to the post-training phase, i.e. whether pre-trained dLLMs can easily acquire new knowledge through fine-tuning. On three diverse datasets, we fine-tune arLLMs and dLLMs, evaluating them with forward and backward style Question Answering (QA) to probe knowledge generalization and the reversal curse. Our results confirm that arLLMs critically rely on extensive data augmentation via paraphrases for QA generalization, and paraphrases are only effective when their information order matches the QA style. Conversely, dLLMs achieve high accuracies on both forward and backward QAs without paraphrases; adding paraphrases yields only marginal gains. Inspired by the dLLM's performance, we introduce a novel masked fine-tuning paradigm for knowledge injection into pre-trained arLLMs. This proposed method successfully and drastically improves the data efficiency of arLLM fine-tuning, effectively closing its performance gap with dLLMs. We further show that the masked fine-tuning paradigm of arLLMs can be extended to the supervised fine-tuning (SFT) of mathematical capability. Across two models and two datasets, our masked SFT outperforms regular SFT.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors found that the dLLMs are free of the ``reversal curse'' in pre-training, and proposed to combine the masked token training strategy with the original SFT objectives. With the proposed fine-tuning paradigm, the arLLMs can perform a high accuracy in both forward and backward QA tests.

### Strengths
1. The expression of the paper is relatively clear.

2. From the experimental results, it can be seen that the proposed method can effectively alleviate the accuracy of arLLMs' answers in the reversal problem.

### Weaknesses
1. The contribution of the paper is insufficient. The proposed new wiki dataset did not demonstrate any new experimental conclusions, so it is unclear what the motivation is behind its introduction. For the proposed new training method, it seems to be a simple combination, and the experimental aspect is relatively simple. It is not yet clear what the applicable model scope, training efficiency changes, and so on of this method are. Therefore, this seems to be an ongoing work.

2. The expression of the proposed method in the paper is not clear enough. How will the prompt form in Figure 3 be trained, that is, the specific manifestation of loss? Moreover, is the form of Figure 3 fixed during the training process, or does it require some expansion to avoid overfitting to this prompt pattern? In addition, line 242 indicates that the arLLMs model was trained in a pre-trained format, and the proposed method uses the chat format. Will this introduce inconsistency in comparison?

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
2

### Rating Number
2

### Confidence
4

### Summary
This paper compares how arLLMs and dLLMs absorb new knowledge via fine-tuning, focusing on the reversal curse where arLLMs fail on reversed queries. Experiments on small QA datasets show arLLMs generalize poorly and depend on paraphrased data, while dLLMs are much more data-efficient, performing well on both forward and backward questions. The authors propose masked fine-tuning for arLLMs, training them to fill in masked tokens, which removes the reversal curse and closes the gap with dLLMs. Masked fine-tuned arLLMs reach near dLLM-level accuracy with similar convergence speed and no extra cost, making them highly data-efficient for knowledge injection.

### Strengths
1. Defines three QA datasets for evaluation.

2. Promises to release code for reproducibility.

3. Experimental setup and results are clear and easy to follow.

4. Writing is natural, structured, and easy to read.

### Weaknesses
1. Limited and synthetic evaluation data, hindering generalizability

The empirical validation is conducted on a very narrow set of tasks, predominantly toy or synthetic datasets. The NameDescription dataset has only 60 simple fictional statements, and the Biography dataset uses a subset of 100 short fabricated biographies, both are small and artificial. The only “real” data comes from a custom Wiki dataset of merely ~92 recent Wikipedia articles, for which QA pairs are generated automatically. Such a limited evaluation suite raises concerns about generalizability. 

The method is proven only on small-scale, mostly synthetic scenarios. It is unclear if these findings would hold on larger, more diverse corpora or real-world knowledge tasks. In particular, the heavy use of GPT-generated QAs and paraphrases means the setup may not reflect the complexities of truly natural data, potentially limiting the paper’s broader relevance . 

2. No testing on standard benchmarks to demonstrate broad effectiveness

The study does not evaluate the proposed masked fine-tuning approach (or the dLLM vs. arLLM comparison) on any widely-used general question-answering benchmarks (e.g. NaturalQuestions, TriviaQA or other knowledge-intensive QA tasks). All experiments are confined to the three author-curated datasets. This is a significant weakness because it weakens the claim of general effectiveness. Without results on established benchmarks, it’s hard to tell if the method would truly improve data efficiency or QA accuracy in practice. The tasks chosen are relatively constrained (mostly short factoid QAs on single paragraphs). I worry that the impressive gains might be specific to these toy tasks. Evaluating on a broader range of open-domain or knowledge-heavy benchmarks would have greatly strengthened the paper’s evidence for general applicability. 

3. Unquantified overhead of masked fine-tuning

Adopting masked fine-tuning introduces additional training complexity that the paper does not adequately discuss. In each batch, the approach requires constructing masked versions of the input and potentially sampling multiple masking patterns over training. This could incur non-trivial computational overhead compared to a standard fine-tune. However, the authors do not quantify or measure this overhead.

For example, the extra runtime, memory, or complexity of generating and handling masks is never reported. They assert that masked fine-tuning is “compute-efficient”, but provide no analysis or ablation on training cost. Without such data, it remains uncertain whether the improved data efficiency might come at the expense of higher compute or implementation complexity. 

A discussion or measurement of training overhead would have been helpful to confirm the method’s practical efficiency.

### Questions
See Weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors look at the reversal-curse problems in Autoregressive LLMS (AR-LLMs) and Diffusion LLMs (D-LLMs) and investigate D-LLMs' capabilities in those lacking areas. Further, the authors propose a masked paradigm for AR-LLMs mimicking masked diffusion to bridge the gap between AR-LLMs and D-LLMs.

## Impact of paraphrasing

1. Re-finding previous results, the authors show that the backward accuracy in AR-LLMs is close to zero with next token prediction, but paraphrasing-based data increases the accuracy to 1. 
2. On the other hand, D-LLMs perform well (slightly worse than AR-LLMs ) on the fwd tasks and better on the bwd tasks. 
3. "his indicates that dLLM does not trade better data efficiency and performance for more
computations; it requires the same or less computation and fewer training samples, but achieves
better downstream performance."

## Masked fine-tuning

1. The authors propose masked fine-tuning, mimicking the D-LLMs training paradigm where the objective is to predict the masked intermediate tokens, dropped out at random. 
2. Authors explore the effect of mask-ratio, and Figure 4 indicates ratio=0.75 (with paraphrases) / raio=0.5 w/o paraphrases performs the best for AR-LLMs.

### Strengths
1. The authors present a simple, clear empirical report and training modifications to look at the reversal curse in AR and D LLMs.
2. The contributions look novel and have strong empirical results.

### Weaknesses
1. Efficiency seems poorly defined - taking a principled approach (like FLOPs, etc.) and showing concrete metrics to show efficiency-based claims.
2. deep dive around paraphases and ratio seems missing since behavior is not consistent and it is not obvious why.

### Questions
Just the weakness above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Building on prior findings that masked diffusion language models (dLLMs) exhibit strong data efficiency and are free from the “reversal curse” during pretraining, this paper investigates whether these advantages also hold in the post-training stage. The study first demonstrates that dLLMs maintain data efficiency in both forward and backward question answering (QA) tasks without requiring paraphrases, whereas autoregressive language models (arLLMs) heavily depend on paraphrasing to generalize textual knowledge into QA performance. Motivated by this observation, the paper further proposes a masked fine-tuning method for effective knowledge injection in arLLMs.

### Strengths
1.	It is interesting and novel to explore diffusion language models (dLLMs) in the context of the post-training stage, demonstrating their data efficiency for knowledge injection in QA tasks.
2.	The proposed masked fine-tuning approach for autoregressive LLMs (arLLMs) is well motivated by the empirical findings that dLLMs exhibit strong performance on QA tasks, and the method itself is both interesting and effective.
3.	The presentation is generally clear and well organized, with detailed descriptions that make the paper easy to follow.

### Weaknesses
1.	The proposed masked fine-tuning approach is simple and effective; however, it is not entirely convincing whether it offers substantial improvements over existing fine-tuning methods. More extensive comparisons with prior approaches are needed. For instance, the following work may be relevant:
https://arxiv.org/pdf/2403.13799.
Additionally, other simple data augmentation strategies that could address similar issues in autoregressive LLMs should be compared.
2.	The proposed masked fine-tuning method appears conceptually related to T5’s infilling objective, and it is unclear whether its technical novelty is significant relative to other similar objective functions.
3.	The paper attributes the data efficiency of dLLMs to their bidirectionality, but such characteristics are not unique to diffusion models. The motivation for the current approach could have been derived equally well from BERT-like bidirectional models, without relying specifically on dLLMs. As a result, the conceptual connection between dLLMs and the proposed method feels somewhat weak.

### Questions
1. Can the masked fine-tuning be extended to other general NLP tasks?

### Soundness
2

### Presentation
3

### Contribution
2
