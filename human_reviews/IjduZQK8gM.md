# From Attention to Activation: Unraveling the Enigmas of Large Language Models

- Avg Score: 5.67
- Decision: Accept (Poster)
- Scores: 6, 6, 5

## Abstract
We study two strange phenomena in auto-regressive Transformers: (1) the dominance of the ﬁrst token in attention heads; (2) the occurrence of large outlier activations in the hidden states. We ﬁnd that popular large language models, such as Llama attend maximally to the first token in 98% of attention heads, a behaviour we attribute to the softmax function. To mitigate this issue, we propose a reformulation of softmax to softmax-1. Furthermore, we identify adaptive optimisers, e.g. Adam, as the primary contributor to the large outlier activations and introduce OrthoAdam, a novel optimiser that utilises orthogonal matrices to transform gradients, to address this issue. Finally, not only do our methods prevent these phenomena from occurring, but additionally, they enable Transformers to sustain their performance when quantised using basic algorithms, something that standard methods are unable to do. In summary, our methods reduce the attention proportion on the first token from 65% to 3.3%, the activation kurtosis in the hidden states from 1657 to 3.1, and perplexity penalty under 4-bit weight quantisation from 3565 to 0.3. Code is available at https://github.com/prannaykaul/OrthoAdam

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies two interesting phenomena in transformer-based LLMs. First, the large amount of attention the first token receives. Second, the emergence of outlier neurons in the transformer layer activation. They attribute the first phenomenon to the softmax function, and the second to the Adam optimizer. They propose variants of these two methods (softmax-1 and OrthoAdam), which resolve these two issues. They show that using these methods allows models to quantize far better than without them.

### Strengths
- Interesting and important research questions
- A well executed study. In particular, several hypotheses are considered for the phenomena in question, and the authors are able to pin-down on what seems to be their root cause 
- Very interesting findings
- Practical results that allow simpler quantization

### Weaknesses
- softmax-1 has been introduced before in the streamingLLM paper (https://arxiv.org/abs/2309.17453, referred to as $Softmax_1$ or Zero-sink in that work). The authors in that work showed that "_while the zero sink alleviates the attention sink problem to some extent, the model still relies on other initial tokens as attention sinks_". This work seems to argue differently. It seems important to pin down the differences between both setups that lead to the different behaviors.

In addition, I found some of the major claims to require further discussion.

- From table 4, it seems RMSNorm-s is also important for mitigating the outlier effect. This is currently not really 
discussed, and the authors only mention softamx-1 and OrthoAdam as the important parts.

- The softmax-1 method requires some discussion and intuition. Why is the specific method appropriate here? Specifically, the authors say (#208) that it allows "having low attention scores everywhere." But standard softmax also allows that, e.g., by assigning uniform attention scores to all tokens. Is it about a better inductive bias? It would be helpful to see a more detailed comparison between the two methods. E.g., an explanation of why softmax-1 is more effective than standard softmax for this task, and/or a discussion of any potential trade-offs or downsides to using softmax-1.

- If I understand the discussion around OrthoAdam correctly, the authors say the reason for the outliers is (a) the appearance of high values of features, and (b) that optimizers like Adam lead to such high values. Is my understanding correct? If so, this should be stated more explicitly.



Minor:

- Using *x*s to contrast with *v*s in table 2 (as used in tables 3 and 4) would make it clearer than leaving it blank.

Also, there are several typos and such in the paper:

- #195: "... therefore it *in* receives ..." (drop "in")
- #306: "... under *an* particular ..." (should be "a")
- #313: " the model *during* i.e.  ..." (something's strange here, either drop during or add something after it)
- #420: "Note that only linear layers are *quantised the* embeddings, ..." (Missing period/, while)

### Questions
- As the authors note, Llama2 remains usable after quantization even in the vanilla version. Do the authors have some intuition regarding this?

- I had a hard time understanding the discussion in #222 around causal masking. What does it mean to relax causal masking? train with an MLM loss? and why would the fifth token dominate in this case?

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
This work studies two unexpected phenomena that occur in Transformers-based language models: the dominance of first tokens in the self-attention maps, and the presence of outlier dimensions throughout the hidden representations of these models. The authors propose a different mitigation technique for each phenomenon, and successfully mitigate both first-token dominance and outlier dimensions. They finally show that combining both mitigation techniques yields language models that can be quantized easily, with minimal performance loss.

The first mitigation technique relies on a softmax-1 activation function, that adds a bias to the denominator of the softmax function to allow for null rows in the attention map. The authors conduct experiments that show that this technique solves the first-token dominance problem but not outlier dimensions. This second problem is mitigated by OrthoAdam, a slight modification to the Adam optimizer that compute gradients in a given orthonormal basis to avoid specific canonical dimensions being used for massive coefficients.

### Strengths
This paper successfully addresses two issues that have been identified by extensive literature in the language modeling field when training Transformers-based models. It provides relevant methods to mitigate these issues and show that such methods can help with performance 
 (slightly) and significantly eases the quantization process.
The paper is well-written and pedagogical. The proposed solutions are elegant and simple to implement.
Overall, this paper paves the way for a better understanding of observed self-attention maps and for more expressive inductive biases for language models. It also provides immediate and substantial benefits for the field of quantization.

### Weaknesses
The main weakness of the paper is its failure to support its main theoretical claim. From the abstract on, the authors argue that they identify "Adam as the primary contributor to the large outlier activations", but they fail to properly discuss the theoretical background of this claim. They support this claim with the following arguments, which all have flaws:
- *the optimiser tracks moments in the same basis as the model parameters* : this reason does not explain why the optimiser is pushing for these outlier activations in gradients and their moments, but only why these outlier dimensions are passed on to the model's weights;
- *Adam and RMSProp have high[er] kurtosis [than SGD]* : this indeed shows that Adam and RMSProp are particularly accelerating the emergence of outlier activations, but it should be noted that they are also accelerating convergence. Hence, this could simply imply that Adam and RMSProp are efficiently facilitating the emergence of outlier activations that could still **be caused by another element of the system** that would rely on outlier activations to increase performance. In other words, the "outlier activation" state could be a favorable state for the model in terms of performance because of another component, and this state could be reached more quickly/easily using Adam and RMSProp.

It could be argued that the fact that OrthoAdam mitigates the outlier dimension problem proves that Adam is the culprit for such a phenomenon. Nevertheless, OrthoAdam is almost explicitly designed to avoid outlier dimensions *in the canonical basis*, as discussed in section 4.2, and could be seen as a variation of Adam that enforces the absence of outlier dimensions. Hence, this just shows that it is possible to train a model under this constraint without substantial performance decrease, but it does not show that Adam is the key **causal** element in the emergence of outlier dimensions.

Another limitation of the paper lies in some parameters and models that were used to validate the approaches:
- The models were trained using a sequence length of 256, which questions the scalability of softmax-1 to longer sequences, as sequence length may be a crucial factor for this activation function;
- All resulting models seem to have a noticeably high perplexity on the validation set. As a comparison, the 70M Pythia model reaches approximately a 3.2 loss (=24 perplexity) at half-training (~150B tokens), and the 410M version reaches 2.2 for cross-entropy (=9 perplexity)(https://wandb.ai/eleutherai/pythia-extra-seeds/reports/Some-loss-curves-for-smaller-Pythia-models--Vmlldzo2NTkxNDIw). This is significantly smaller than what is described in the paper for equivalent models, which raises questions about implementation, hyperparameter choices and evaluation.

Finally, the authors do not include a discussion on training dynamics, showing how the models converge during training, and they do not provide an analysis of OrthoAdam in terms of memory requirements and training latency, which may question the scalability and technical relevance of the method. My intuition is that the cost should be tolerable, but I think this should be mentioned in the paper.

### Questions
- The idea of using an orthogonal basis $Q$ in Adam is very elegant, especially in the context of quantization. Nevertheless, from a modeling/theoretical viewpoint, it would be interesting to check whether the models are still using outlier "dimensions" in this orthogonal basis, that is if the projection of their activations in such an orthogonal basis still have outlier coefficients. Did you conduct such experiments? In other words, did you verify that the models are not still relying on outlier "dimensions" in a different basis than the canonical one?
- It is common to decrease the learning rates for larger models. Why did you choose a single learning rate for the different model sizes? 

**Typos / Remarks**

L419 : this sentence (*Note that only linear layers...*) seems a bit unformal.

L459: *but still remain high* -> *which remains high* ?

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies two phenomena in transformer-based models. The first is the strong dominance of the first token in the attention maps, and the second is the presence of outlier activations in the hidden states. To address these two issues, the paper proposes a new softmax philosophy and a new optimizer. The experiments show the advantages of the proposed methods under the quantization scenario.

### Strengths
1. The paper conducts a series of experiments to verify the phenomena, which are convincing.

2. The proposed new softmax and optimizer are simple and effective.

3. The experiments under the quantization scenario show the usefulness of the study, which potentially enlightens future research.

### Weaknesses
1. The paper states that the study is on popular large language models (LLMs). However, the experiments in the paper exclude large language models, considering that the largest-sized one in the experiments is GPT2-1.4B, not reaching the bar of a regular sized LLM (>7B). It makes the results of the paper less convincing.

2. I am not convinced that the highlighted two issues in the paper are critical for recent LMs. From the experiments in the paper, one can only see the promising usage under 4-bit/8-bit quant; in other scenarios, still not clear.

3. The paper reports PPL and other metrics to access the LM performances. However, it is not clear whether the resultant models perform well on downstream tasks and instruction following, which are important aspects to evaluate an LM/LLM. Therefore, it is not positive that the study will greatly contribute to the community.

4. Keep up with the last point, the paper does not consider (or explain) whether the study still holds under the instruction-following scenario. This situation is different from doing language modeling; or it is still the same, since the first generated token is not the start of the input sentence.

5. Minorly, it is also not mentioned in the paper the relationship of the two phenomena. It seems that they belong to two independent research aspects.

### Questions
1. Why do the authors train the LLaMA2-130M model? It is strange to me.

2. Can the authors explain the how the case would change where the model is used to generate following some instruction. Will the LM attend strongly to the first generated token?


---
After rebuttal
Thanks for the authors' detailed responses. My concerns are below.

1. My first concern is the **originality of the softmax-1**. However, I failed to find the original paper when I first read the paper until I see the comments.

2. The second concern is the **contribution of the paper**. In addition, the first token attention issue was also discussed in [1]. I am afraid the story of the paper on 1 and 2 is overstated in the submission. That is why I give a lower contribution score.

3. The authors state "large language models" in their submission title, which is confusing. While the authors provide the LLaMA-8B results in the rebuttal, only a 8B LLM can not represent the entire group of LLMs. The situation can be very different when the model size comes to 70B or even 130B. A more suitable statement for the current submission would be **"autoregressive language models"**.

Therefore, I decide to keep my score.

[1] Efficient Streaming Language Models with Attention Sinks

### Soundness
2

### Presentation
3

### Contribution
1
