# Predicting LLM Output Length via Entropy-Guided Representations

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
The long-tailed distribution of sequence lengths in LLM serving and reinforcement learning (RL) sampling causes significant computational waste due to excessive padding in batched inference. Existing methods rely on auxiliary models for static length prediction, but they incur high overhead, generalize poorly, and fail in stochastic "one-to-many" sampling scenarios. We introduce a lightweight framework that reuses the main model's internal hidden states for efficient length prediction. Our framework features two core components: 1) Entropy-Guided Token Pooling (EGTP), which uses on-the-fly activations and token entropy for highly accurate static prediction with negligible cost, and 2) Progressive Length Prediction (PLP), which dynamically estimates the remaining length at each decoding step to handle stochastic generation. To validate our approach, we build and release ForeLen, a comprehensive benchmark with long-sequence, Chain-of-Thought, and RL data. On ForeLen, EGTP achieves state-of-the-art accuracy, reducing MAE by 29.16\% over the best baseline. Integrating our methods with a length-aware scheduler yields significant end-to-end throughput gains. Our work provides a new technical and evaluation baseline for efficient LLM inference.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of predicting LLM output lengths by two core components: 1) Entropy-Guided Token Pooling (EGTP), which uses on-the-fly activations and token entropy for highly accurate static prediction with negligible cost, and 2) Progressive Length Prediction (PLP), which dynamically estimates the remaining length at each decoding step to handle stochastic generation. 

The paper is pretty interesting with some well-designed experiments. The new benchmark ForeLen is sound, which covers long-sequence, Chain-of-Thought, and RL sampling data. I believe this benchmark would help the community, and I am looking forward to seeing it open-source.

### Strengths
The paper clearly articulates why existing auxiliary predictor methods are problematic (overhead, poor generalization, stochastic generation failures) and proposes a principled alternative.

The performance of predicting LLM output is important for real-world application. We are glad to see the work outperform other predicting methods.

ForeLen addresses limitations of LMSYS by including longer sequences and RL data, which will benefit future research

### Weaknesses
This paper uses the Mean Absolute Error (MAE) to evaluate the predictor's performance. Further analysis of statistical parameters can help readers understand the advantages and disadvantages of different methods, including variance and mean squared error.

The core insight (reusing internal representations) is somewhat incremental over prior work like TRAIL that also uses embeddings.

I think we can even predict the full output if we extract the model's internal activations. That makes this work make sense and also limits the scope of the potential benefit. I am looking forward to seeing the following work.

### Questions
How does EGTP perform on very long sequences beyond the training distribution (e.g., >10k tokens)?

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
4

### Summary
This paper proposes two methodologies for regressing the length of a generation from an autoregressive model like an LLM. In EGTP, a small attention-like head is applied over a featurization of the model's activations to regress the length, whereas in PLP a prediction is instead made of the length-to-go at each generation step. The authors compare their predictions against many prior methods, and find that EGTP pooling almost always performs better in accuracy and throughput.

### Strengths
1. The problem of length regression is basic and fundamental, yet unsolved. Current methods seem to be very engineered (as opposed to clean and lightweight), and so it appears a good problem to be working on. The authors have made substantial progress on this problem with a fairly simple addition (one prediction head and a couple hyperparameters) that outperforms existing methods, which they demonstrate convincingly. I think overwhelmingly convincing experiments are really needed in these days of LLM research (we have to *know* if something is best in order to build on it!), so I think it's a big strength that the results are so clear and stable across hyperparameters/tasks.

2. The authors also introduce a benchmark with different length distributions than previous ones and a carefully-made soup of various datasets. While sometimes I feel that introducing extra benchmarks can be misleading or contribute to noise, I think in this work it does well to complement LMSYS. This goes back to Strength (1), but I love seeing gains on a diverse set of tasks on datasets with different length distributions made by different people.

### Weaknesses
1. Many of the design decisions feel a bit unmotivated (such as binning the regression with exponential tails), and there are many reasonable alternatives that were not mentioned or ablated (such as smoothing, predicting on nonlinear scale, adding small noise to the labels, etc.). It is hard to debate the fact that the authors' methodology works (on many tasks and with different model types and sizes), but it is also very hard to know which parts are necessary and which are suboptimal. If the authors could discuss a bit more about which other methods they considered/tried that didnt work out, and why their solution is a natural one to arrive at?

2. I would have liked the authors to say a bit more about some of the interesting things they saw while training these. In particular, from what I can see in your Appendices D and E it seems like the task doesnt get much easier/harder as model scale changes, which I find interesting. Were there trends like this that you saw, perhaps with respect to model class (Qwen vs Llama vs ...) or optimizer for learning the prediction head or anything? An extra small gain to including such observations is that it would make the paper feel a bit more sciency and less industrial/SOTA-seeking (this is definitely my own bias, but perhaps less jargon could help out; what do you think?).

### Questions
1. Aside from the questions in the Weaknesses section, I really like the plot in Figure 2 as a way to explore the role of entropy in this prediction task. I understand that it validates the EGTP method, but I would very much appreciate if the authors could share a little more about what the entropic nature of the problem really is. To me, the success of this entropy-based pooling over other pooling methods suggests that entropy is intimately related with a model's own internal understanding of when it will finish generating. It makes sense for most of the paper to be devoted to presenting this methodology and experiments in the usual SOTA-claiming style, but I would like the authors to discuss a bit more about what is to be learned from this.

2. I think it is important for the authors to say more about how the hidden states were selected. Are the choices the same (or along similar lines) for different model types and sizes, or are they custom for each one? What are some trends that the authors noticed/guidelines for choosing these? I recall in the computer vision days there was great interest in using hidden activations of VGG and ResNet models for prediction of other stuff, and as time went on and models got bigger it became harder to understand the benefits or drawbacks of certain design decisions. I am curious how the analogous story goes in the LLM age, and what your experience with this method has been like.

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
2

### Summary
The paper addresses the problem of estimating the output length of LLMs, which is an important problem for efficient batching during inference. If LLM generations vary a lot in length, batching will naturally waste compute because of the introduced padding. 

The proposed method directly reuses the LLM’s internal activations instead of relying on separate auxiliary predictors. It applies two key methods: 1) Entropy-guided token pooling, 2) Progressive length prediction to handle stochastic generation.

### Strengths
- The paper is written clearly and easy to follow. 
- The problem is well motivated.
- I am not too familiar with the topic, but after a short literature research I tend to agree that a benchmark as proposed in the paper is a reasonable contribution to the community.

### Weaknesses
- Although a main selling point for the method is efficiency, I could not find any cost comparison to methods that use an auxiliary model for prediction.

### Questions
I understood: The proposed length regression relies on pooled hidden representations extracted by the LLM we want to perform inference on. The length information can be used for efficient batching of the prompts, such that minimal padding will occur. What I do not understand is: To get the representations, we already need to process the inputs (possibly batched?) by the LLM. Did I miss something?

### Soundness
3

### Presentation
3

### Contribution
3
