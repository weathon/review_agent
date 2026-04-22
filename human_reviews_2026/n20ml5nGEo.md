# Token Distillation: Attention-Aware Input Embeddings for New Tokens

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Current language models rely on static vocabularies determined at pretraining time, which can lead to decreased performance and increased computational cost for domains underrepresented in the original vocabulary.
New tokens can be added to solve this problem, when coupled with a good initialization for their new embeddings. However, existing embedding initialization methods require expensive further training or pretraining of additional modules.
In this paper, we propose Token Distillation and show that by distilling representations obtained using the original tokenization, we can quickly learn high-quality input embeddings for new tokens.
Experimental results with a wide range of open-weight models show that Token Distillation outperforms even strong baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method to train new input embedding parameters for added tokens using hidden state distillation on the last layer. Specifically, the authors align the hidden state of added tokens t* with the one of original sub-tokens $\tau (t^*)$ in related corpus and distill the hidden state with mean-squared error loss function. Experimental results on 8 decoder models show that the proposed method outperforms other vocabulary adaptation methods, and performs better with weighted next token prediction objective. Further analyses illustrate that better embedding initialization method can brings further improvement.

### Strengths
- It is interesting to incorporate the Transformer layer into the initialization of new input embedding parameters. 
- The experimental results are promising for Transformer decoder models in the vocabulary adaptation task.

### Weaknesses
- Experiments are conducted on the Transformer decoder models. It is unclear for its performance on the vocabulary adaptation of Transformers encoder models.

- Missing details that the number of tokens added in the experiments. The performance of this method may be affected by the lexical similarity between the target token and source token. For example, if all target tokens like Arabic or Chinese words are significantly different to the source tokens like English words and original subtokens $t_i$ are UTF-8 characters, how does your method perform? Can your method improved by initialization method based on semantic alignment like Focus[1] and TokAlign[2] in this setting?

- Typo at the Line 359: "NTP outperforms NTP" --> "NTP outperforms NTP (tune all embeddings)"

**References**

[1] Konstantin Dobler and Gerard de Melo. FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models. EMNLP 2023.

[2] Chong Li, Jiajun Zhang, and Chengqing Zong. TokAlign: Efficient Vocabulary Adaptation via Token Alignment. ACL 2025.

### Questions
1) How about the results of your method on encoder Transformer models like BERT?

2) If all target tokens like Arabic or Chinese words are significantly different to the source tokens like English words and original subtokens $t_i$ are UTF-8 characters, how does your method perform? Can your method improved by initialization method based on semantic alignment like Focus and TokAlign in this setting?

### Soundness
2

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
3

### Summary
This paper introduces "Token Distillation," a novel method for initializing input embeddings for new tokens in a pretrained language model. The method addresses the inefficiency and potential performance issues caused by the "over-tokenization" of domain-specific terms into multiple subtokens. The authors propose a distillation objective: to optimize a new single token's embedding such that it replicates the model's internal hidden states when processing the original sequence of subtokens. The core contribution is this efficient initialization technique that leverages the full model's knowledge, not just the embedding layer.

### Strengths
1. Novel and Effective: The paper proposes a clever and elegant method, "Token Distillation," which distills the model's internal behavior (hidden states) rather than simply aggregating embedding vectors. Experiments robustly show it outperforms strong baselines.
2. Thorough Experimental Validation: The claims are supported by rigorous experiments across a diverse set of models, tasks (domain and language adaptation), and a comprehensive suite of baselines, demonstrating the method's reliability.

### Weaknesses
1. Insufficient Justification for Practical Significance: The paper's primary weakness is the lack of compelling evidence for why the proposed efficiency-performance trade-off is critical. In many high-stakes domains, even a small performance drop is unacceptable, and the paper fails to demonstrate scenarios where the efficiency gain from token compression is a mission-critical requirement rather than a minor convenience.
2. Misaligned Motivation: There is a narrative disconnect. The motivation suggests over-tokenization harms performance, yet the experiments consistently show the original (over-tokenized) method as the performance ceiling. This weakens the argument and shifts the method's entire value proposition to an efficiency gain that is not sufficiently justified.

### Questions
1. My primary concern, which currently limits my rating, is about the practical necessity of the efficiency-performance trade-off this paper proposes. If the authors can convincingly address this, I am open to raising my score.
For maximum performance, my default assumption is that one would always prefer the original tokenization, accepting its computational cost. The significance of your paper hinges on proving that the efficiency gain is not just a marginal benefit but a critical enabler for certain applications. Could you provide a concrete, quantitative analysis of these efficiency benefits? For instance(Do not to do all, just example):
* Can you demonstrate how the reduced sequence length translates to significant cost savings (e.g., in dollars or GPU hours) for large-scale batch processing of a corpus?
* Can you showcase a task involving long documents where the original tokenization would force truncation and lead to failure, while your method would succeed by fitting within the context window?
* Are there real-time applications with strict latency budgets where your method is viable but the original is not?
Providing evidence for a scenario where the original tokenization is practically infeasible or prohibitively expensive would directly address this concern and strongly bolster the paper's contribution.

2. Table 3 shows a surprising case where Token Distillation outperforms the original tokenization (Llama-3-8B-Instruct on French). Could you elaborate on this? Is it a statistical anomaly, or does it suggest that for some out-of-domain languages, your distillation process might be correcting for certain suboptimalities in how the base model processes subtoken compositions?

### Soundness
3

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
This paper proposes a novel embedding initialization method for new tokens, motivated by the fact that conventional methods only consider interactions within embeddings. The proposed method instead considers the output hidden representations of the new and the corresponding base tokens and tries to make them close to each other by using an MSE loss function, while only tuning the new token representations. The experiments show that the proposed method works as the best starting point for adaptation among existing baselines, including a sophisticated method like ZeTT (which uses a hypernetwork). The analysis covers a wide range of aspects related to the design and implementation of the proposed method, showing the robustness of the method.

### Strengths
1. The paper is well-motivated. The paper clearly identifies the problem with conventional methods for embedding initialization of new tokens (i.e., not accounting for higher-layer model dynamics) and the necessity of training an auxiliary model for sophisticated methods like ZeTT. The proposed method clearly addresses these issues and employs a lightweight training-based approach that aims to make the representations of a new token and its corresponding source tokens similar using an MSE loss.

2. The experiments well support the advantage of the proposed method against a series of baselines, showing it can provide a good starting point for adaptation.

3. The analysis is quite extensive, covering different aspects of the proposed method, including training data, embedding initialization methods for new tokens, training objectives, and choice of a target layer.

### Weaknesses
1. The major limitation of this paper is a lack of full continual pre-training results. While the proposed method can provide a better starting point as “initialization”, it does not always guarantee a better performance after continual pre-training on target data. Any gains seen at the starting point might not hold up when we conduct further tuning on the target data. Given that almost all methods exhibit worse performance after embedding adaptation, they inevitably require continual pre-training before actual use. Otherwise, I do not see any point in using these methods.

2. The choice of 2,500 new tokens (L202) warrants justification. This number should substantially affect the resulting performance, both downstream and inference efficiency. The current paper does not have an analysis of different sizes of new tokens. Also, the paper lacks a description of how it generates (defines) new tokens.

3. The paper does not involve any analysis related to inference efficiency.  Given that the motivation of this line of work is to improve suboptimal tokenization, thereby achieving better downstream performance and better inference efficiency, the paper would need to demonstrate both.

### Questions
1. How does the proposed method ensure each training sample only contains a single new token? If there are two or more new tokens in a single sample, does Eq. (1) still work?

2. On Weakness 1, I would suggest adding a few experiments that conduct continual pre-training with baselines (including the source model) and the proposed approach, given the same training budget.

3. On Weakness 2, I would suggest incorporating an analysis of different vocabulary sizes.

4. On Weakness 3, I would suggest adding an analysis of inference efficiency. This will further strengthen the contribution of the paper if the proposed method achieves better efficiency.

### Soundness
2

### Presentation
2

### Contribution
2
