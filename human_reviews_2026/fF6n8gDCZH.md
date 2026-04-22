# FlowNIB: An Information Bottleneck Analysis of Bidirectional vs. Unidirectional Language Models

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 2, 6

## Abstract
Bidirectional language models (LMs) consistently show stronger context understanding than unidirectional models, yet the theoretical reason remains unclear. We present a simple information bottleneck (IB) perspective: bidirectional representations preserve more mutual information (MI) about both the input and the target, yielding richer features for downstream tasks. We adopt a layer–wise view and hypothesize that, at comparable capacity, bidirectional layers retain more useful signal than unidirectional ones. To test this claim empirically, we present Flow Neural Information Bottleneck (FlowNIB), a lightweight, post-hoc framework capable of estimating comparable mutual information values for individual layers in LMs, quantifying how much mutual information each layer carries about the input and target. FlowNIB takes three inputs—(i) the original LM’s inputs/dataset, (ii) ground–truth labels, and (iii) layer activations—simultaneously estimates the mutual information for both the input–layer and layer–label pairs. Empirically, bidirectional LM layers exhibit higher mutual information than similar—and even larger—unidirectional LMs. As a result, bidirectional LMs outperform unidirectional LMs across extensive experiments on NLU benchmarks (e.g., GLUE), commonsense reasoning, and regression tasks, demonstrating superior context understanding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Flow Neural Information Bottleneck (FlowNIB) to analyze the information-theoretic properties of language models, by comparing bidirectional (e.g., BERT-like) and unidirectional (e.g., GPT-like) architectures. Grounded in the IB principle, the authors argue that bidirectional models retain more mutual information, leading to richer contextual understanding. FlowNIB estimates layer-wise MI between the input, hidden representations, and task labels, with the Optimal Information Coordinate to summarize each layer’s information-carrying capacity.

### Strengths
1.	The paper provides a clear and formal argument, grounded in the IB framework to explain why bidirectional models preserve more mutual information than unidirectional ones.

2.	The evaluation is extensive, spanning 16 NLP datasets, multiple model families, and both classification and regression tasks.

3.	The paper is generally clearly-written and easy to follow.

### Weaknesses
1.	The key insight of the paper that bidirectional layers retain more mutual information is largely intuitive and expected, given that unidirectional layers only attend to previous tokens, while bidirectional ones attend to both past and future tokens.

2.	The comparison between unidirectional and bidirectional architectures is not entirely novel. Several prior works have examined similar distinctions, including:

- [1] On the Role of Bidirectionality in Language Model Pre-Training. EMNLP Findings 2022

- [2] Transforming decoder-only models into encoder-only models with improved understanding capabilities. Knowledge-Based Systems 2025

- [3] The underlying structures of self-attention: symmetry, directionality, and emergent dynamics in Transformer training. ICML 2025

- [4] What Limits Bidirectional Model's Generative Capabilities? A Uni-Bi-Directional Mixture-of-Expert Method For Bidirectional Fine-tuning. ICML 2025

3.	A major practical limitation of bidirectional models is their substantially higher computational cost, which remains unaddressed. The paper would benefit from an analysis or discussion of this aspect.

4.	Although FlowNIB is described as “lightweight,” MI estimation remains computationally intensive, especially for large models and many layers. The paper does not fully quantify the runtime or scalability trade-offs.

5.	While FlowNIB identifies correlations between OIC and performance, it does not establish causality. High MI could stem from other architectural factors (e.g., attention span or parameter distribution), which the analysis does not disentangle.

### Questions
1.	How do models with higher mutual information perform on non-NLU (e.g., generative or multimodal) tasks?

2.	Could you provide a quantitative comparison of computational costs between unidirectional and bidirectional models in the context of FlowNIB analysis?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores the information plane of bi- versus uni-directional LLLMs. The authors present FlowNIB and compare bi- with uni-directional LLLMs across classification and regression tasks. The authors' main claims are that bidirectional models show better MI values and thus better downstream performance.

### Strengths
* The connection between conditioning reduces entropy and higher I(X;Z) in bidirectional models is rigorous and intuitive.

* Consistent results across 16 datasets, covering both classification (GLUE, commonsense reasoning) and regression (STS-B, LCP, etc.). Layer-wise and token-level MI analyses support the theoretical claim.

* The paper is well structured and easy to follow.

### Weaknesses
* Contribution (iii) is redundant. We all know the three talking point rule, but that's not a reason to try to make two contributions into three. 

* Theoretical novelty: The central theoretical claim (conditioning reduces entropy --> bidirectional > unidirectional) is mathematically sound but conceptually straightforward. It might be seen as an extension of standard information inequalities rather than a novel theoretical insight.

* MINE-based estimators are known to be unstable and sensitive to hyperparameters.  While FlowNIB normalizes MI and uses consistent setups, no quantitative uncertainty analysis (e.g., variance across runs) is provided.  Moreover, the framework measures relative MI values but does not validate that estimated magnitudes correlate with true MI.

* While the paper shows a correlation between MI and performance, it does not establish causation (e.g., whether higher MI directly improves task accuracy). Possible confounds such as architectural biases or tokenization differences are not fully controlled.

### Questions
* How do you ensure that the observed MI differences are not simply a by-product of pretraining objectives or tokenizer differences rather than directionality per se?

* How robust are your FlowNIB estimates across random seeds, estimator architectures, and datasets?

* Have you validated FlowNIB on synthetic data or toy problems where ground-truth MI is known, to verify that it produces accurate or at least monotonic estimates?

* Can you provide evidence that increasing OIC causally improves task accuracy — for instance, by pruning or freezing layers with low OIC and observing the resulting performance drop? Alternatively, can you show that models fine-tuned to explicitly maximize OIC achieve better generalization?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors analyze an important question, why bidirectional models (encoder-based transformers) performs better in natural language understanding tasks than uni-directional models (decoder-based transformers)? from an information theory perspective. They measure the mutual information between internal representations and inputs, I(X, Z), and between representations and output label, I(Z, Y). They claim that bidirectional LM layers exhibit higher mutual information than unidirectional ones, which could be the reason why they perform better on these tasks. However, I have severe concerns about some important aspects of the paper.

### Strengths
The paper uses and emphasizes the importance of mutual information, which is a metric well-supported by theory and could be very useful for analyzing neural models, but is somewhat overlooked by the research community.

### Weaknesses
- There’s an important question that is not clear to me, is the input variable X a sequence of tokens or just one token? From the architecture of MINE critics (2 layer MLP) mentioned in the paper, it seems it is similar to the original use case of MINE, where inputs are just pairs of vectors. But it means X is a single token, and Z is representation correspond to this token. Then this I(X, Z) is measuring how much information about the current token is encoded by its corresponding representation, which, to be honest, does not make sense to me. Because the representations in transformers are contextualized, they are meant to encode information about the whole input sequence. Measuring MI between the representation and a single token does not sound meaningful. I think it would make a lot more sense if X is the whole sequence, I(X, Z) measures information about the real input. Also, in the paper I(Z, Y) seems to be the average value across sequence length. In other words, in a certain layer of a transformer, there is a sequence of representation, and each representation is used to estimate MI and is averaged. But I think taking the maximum make more sense, because in decoder models many representations cannot even see the large part of the input.

- In theorem 2.1 and lemma 2.3, authors assume concatenation of representations of forward and backward direction instead of addition of them, which is a big difference. As addition causes loss of information, many arguments might not follow. In the case of addition,  obtaining the information about future context is at the cost of information of previous context, I(X; Z→) is not necessarily lower than I(X; Z↔). In the case of concatenation, as the dimension gets larger, it is not a fair comparison. Importantly, this dicussion is only meaningful if X is a sequence.

- In the paper, I(X,Z) and I(Z, Y) are measured by two separate critics (separate neural networks no weights sharing), changing a(t) that combines them to emphasize one over the other does not make a lot of sense to me. Because there is no trade-off for the networks, each network is always optimize only one thing (either I(X,Z) or I(Z, Y)) without considering the other. The only effect is that the loss is scaled differently throughout the training. I don’t see any benefits or important differences between training together and training separately. It seems to me it’s just funamentally equivalent in this paper. Not sure why give it a new name.

- Writing and presentation are not good. Some important information is unclear.

### Questions
Questions and suggestions:

- When presenting equation 1 as the training objective, explicit say it is to be minimized, though it is later mentioned, but it might still cause confusion for readers. 

- After definition 2.2 when the L2 participation ratio is mentioned, it’s better to briefly explain the intuition behind it.

- Regarding the scale of I(Z, Y) depends on the label space, maybe try normalizing it with  min(H(Y), H(Z)), which is usually H(Y), since the I(Z, Y) is upperbounded by that (the entropy can be reduced by observing the other variable), so the normalized version can be retreated as the proportion of entropy being reduced.

- If X the embedding of a token, is it one-hot embedding or a learned token embedding? 

- In Figure 3, it shows average value across layers, can you also show it for the maximum value as well? which I feel would make more sense.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper makes two contributions: (1) it introduces FlowNIB, a method for estimating mutual information in LM layers, (2) it examines the differences between unidirectional and bidirectional models, finding the latter enable higher MI, and linking this to improved downstream performance.

### Strengths
- I found the paper clearly written and overall easy to follow (with a few but critical exceptions mentioned below).
- Given the recent dominance of unidirectional language models, it can be interesting to revisit bidirectional language models.

### Weaknesses
- I didn't understand a key point: line 099: "in finding both information *simultaneously*". What does this mean? Why not just use MINE to estimate I(X; Z_l) and I(Z_l; Y)? Why use a schedule emphasizing first one of these and then the other? I see that this yields a nice 2D plane, but what is the theoretical interpretation? This seems also key to understanding why the authors need to introduce FlowNIB, which otherwise seems unclear.
- Figure 1: which ones are the lower vs upper layers? It seems that curves to the right indicate higher MI with both X and Y than curves on the left (depending on \alpha(t)), which seems confusing, given that by the data processing inequality lower layers should have more infromation with X and upper layers should have more information with Y?
- How nontrivial is the finding that higher MI is associated with better downstream performance? If a model has higher MI with Y, isn't this naturally almost the same as saying that the model performs better on predicting Y? I see that the authors use not MI with Y but the summary statistic OIC, but how much of a difference does this make?

If the authors can convincingly address these, I'd be happy to reconsider my score.

### Questions
- Definition 2.2: What does "measure" mean here? It's clearly not a measure in the sense of "measure theory". Some examples are provided, suggesting that "measure" here just means a mapping from p to real numbers?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors introduce a post-hoc framework that estimates how much information each model layer carries about the input and the label by training two mutual information critics. It summarizes each layer’s "information plane" trajectory with the "Optimal Information Coordinate" and uses this to compare models.

Across NLU and regression tasks, the paper reports that bidirectional encoders show higher OIC and typically higher accuracy than similarly sized unidirectional decoders.

### Strengths
The paper adds a formal lens to a known empirical observation that bidirectional attention yields stronger representations than causal attention.
The work includes broad comparisons across many datasets and model families.

### Weaknesses
The authors claim that higher OIC is correlated with better accuracy of the model, but no quantitiave correlations or graphical comparisons are provided.

There are small issues with the presentation of the paper. Namely, the tables are hardly readable -- a lot of content with tiny font size. I would suggest moving these full tables to appendix, and leaving only the most important aggregated/selected values in the main text, or represented in a graphical form as a plot.

### Questions
As mentioned in the weaknesses, I would suggest moving full tables with results to appendix, and keeping in the main part of the paper aggregated/selected results -- in a form of smaller, more readable table, or a figure. Table 1 with information about models is not _that_ relevant and could be completely moved to the appendix.
The correlation between OIC and performance could be shown explicitly to support the claim of higher OIC --> better performance.

### Soundness
3

### Presentation
2

### Contribution
3
