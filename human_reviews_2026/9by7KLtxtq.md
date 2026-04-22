# Vocabulary embeddings organize linguistic structure early in language model training

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Large language models (LLMs) work by manipulating the geometry of input embedding vectors over multiple layers. Here, we ask: how are the input vocabulary representations of language models structured, and how and when does this structure evolve over training? To answer this question, we use representational similarity analysis, running a suite of experiments that correlate the geometric structure of the input embeddings and output embeddings of two open-source models (Pythia 12B and OLMo 7B) with semantic, syntactic, and frequency-based metrics over the course of training. Our key findings are as follows:  1) During training, the vocabulary embedding geometry quickly converges to high correlations with a suite of semantic and syntactic features; 2) Embeddings of high-frequency and function words (e.g., “the,” “of”) converge to their final vectors faster than lexical and low-frequency words, which retain some alignment with the bias in their random initializations. These findings help map the dynamic trajectory by which input embeddings organize around linguistic structure, revealing distinct roles for word frequency and function. Our findings motivate a deeper study of how the evolution of vocabulary geometry may facilitate specific capability gains during model training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper studies how the embedding matrix of an LLM evolves over the course of pre-training. Leveraging two variants of representational similarity analysis this work looks at how linguistic features, and frequency information develop in the Olmo and Pythia families of models. Results show linguistic information converges early on, representations for high frequency words also converge early in training with low frequency words taking much longer. Low frequency words also retain similarity structure with their random initialisation more than their high frequency counterparts.

### Strengths
This paper is clearly written, giving the reader a clear sense of their methods and results. The information is well presented and largely well scoped to the scale of a conference paper. The results related to frequency information in particular are interesting, shedding some light on how frequency information affects the training process.

### Weaknesses
The two main weaknesses appear to be a limited degree of novelty and a lack of theoretical clarity in the relationships the authors draw to linguistics. In terms of novelty, when during training syntactic information is represented in an LLM is well studied, to a lesser degree effects of frequency information are as well. The authors make clear that much of this work has studied model activations rather than the input embedding matrix. While this may be true the overwhelming majority of computation in a model happens inside of the model. As a result though there is novelty here it seems quite narrow.

Of the results, the results showing lower frequency tokens retain similarity structure with their random initialisation are particularly interesting. However the results in these sections are largely a list of empirical findings with limited analysis or explanatory account - instead offering claims like "frequency [is] a primary factor in driving embedding changes" (a fact established in previous work), the question is how frequency drives those changes. The authors have some results to this effect, it would be great to see a clearer line of argumentation of how frequency affects training.

The stated relationship between this work and linguistics is difficult to follow. Looking particularly at the "Structure in the vocabulary" paragraph (line 112). The "words and rules" approach attributed in Pinker is in fact the broader Generative Tradition in linguistics, which has historically emphasised the separation of syntactic and semantic processes. These are contrasted by Usage-Based approaches to language which argue syntax and semantics are broadly inseparable, with semantic constructions also being a part of the generating process. As written this paper describes this as the "syntax-lexicon" continuum --- a lexicon is not the same as semantics. All accounts cited broadly agree with the existence of syntax and a lexicon, but disagree about what is in the lexicon - this is the line of argumentation that aligns with the authors experiments but as a reader is not the argument they appeared to make. For reference there's a particular disagreement between Jackendoff 1990 and Goldberg 1995 (which they cite) about the degree to which syntactic information is present in the lexicon.

An issue with drawing comparisons with Usage-Based accounts is that they emphasise the importance of context in conditioning and driving the generative process, the author's choice to consider only the embedding matrix makes it hard to directly relate the results here on non-contextualised embeddings to this work in linguistics. Additionally in Usage-Based accounts we would expect semantic, syntactic, and frequency information to be entangled. The analysis here isolates proxies for these kinds of information in separate labels rather than in combination - the authors note this point at 348 that embeddings may be "better explained by non-trivial combinations of linguistic features".

In the conclusion the authors state: "Does the methodological choice of a large token embedding matrix in LLMs implicitly build in the assumption of a words-and-rules approach to language in the system?" I struggle to understand what this question is asking. In the human case, we take raw words, or phonemes as inputs yet may process them as contextualised semantic concepts like constructions. How could having a discrete input, or an embedding matrix for your discrete input entail a factorisation of syntax and semantics?

To be clear I think the methods here are sound, and some of the results are interesting. However the authors emphasise the relationship between their work and related work in linguistics, in part to add theoretical novelty to their use of only the embedding matrix. As a result their engagement with linguistic theory needs to be accurate, and well described.

### Questions
How could "the methodological choice of a large token embedding matrix in LLMs implicitly build in the assumption of a words-and-rules approach to language in the system?"

Virtually all kinds of structure you analyse saturate in the first half of pre-training. Are you claiming no further linguistic information is learned later on?

Does when "linguistic features" stabilise relate to performance in any way? Olmo has a higher peak, is it better at a particular task?

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
3

### Summary
This paper studies how and when the (input) vocabulary embeddings of Large Language Models (LLMs) organize themselves to reflect linguistic properties. Using Representational Similarity Analysis (RSA) on the Pythia 12B and OLMo 7B models, the authors track the pairwise distances of the input embeddings throughout the training process. They find that this linguistic structure emerges very early in training, since the embedding space quickly converges to a state that shows high correlation with both semantic features and syntactic features. They also find that word frequency plays a major role as high-frequency words converge to their final representations much faster than low-frequency words.

### Strengths
This study has a strong empirical scope as it assembles dense checkpoint trajectories across two modern model families. The applied RSA technique is principled, and the reported early semantic plateau is clear and replicated across models. The frequency-aware slicing idea and the comparison to the final checkpoint are interesting.

### Weaknesses
**W1. Operational significance is unclear.**

The paper reports Hypothesis-Driven RSA and Convergence RSA curves, but it does not show how their curves relate to model perplexity or to any simple downstream behavior. Because perplexity is the training objective, it is important to see whether the reported “early emergence” (for semantics) or “early peak” (for syntax) coincides with changes in perplexity. Without such a link, the reader cannot tell whether these RSA events matter for model quality or are just descriptive patterns.

**W2. Frequency is a likely confound that is not controlled.**

The paper reports "early" emergence for both semantic and syntactic RSA (Section 4), but it never controls for token exposure. Frequent words appear many more times and therefore get many more updates; this alone can create an "early" effect. Section 5 acknowledges the role of frequency by slicing into buckets, yet it does not revisit the "syntax" claim after holding frequency fixed. As written, "early syntax" can simply be a frequency artifact.
To resolve this concern, the authors should modify their Experiment 1 to build matched sets with controlled frequency distributions and recompute the RSA curves. 

**W3. Cross-bucket (global) structure is not measured.** 

Section 5 looks only within each bucket (100×100 distance matrices per bucket). It never examines how different buckets sit relative to each other over time (for example, whether the center of the top-100 words moves closer to or farther from the center of the 900–1000 group). It is entirely possible that the global arrangement across buckets stabilizes early while the finer within-bucket neighborhoods keep shuffling. If so, the paper's timing claims would be incomplete.

**W4. "Convergence to final" is fragile and can mislead**

Convergence RSA defines "stability" as "becoming similar to the final checkpoint." This assumes the last snapshot is the right anchor, even though late steps can have extra noise or idiosyncrasies. As a result, a subset can look "stable" simply because it mirrors quirks of the final checkpoint. This matters because the paper's main takeaways about "who stabilizes when" depend on this anchor. It also affects the interpretation in Section 5: when time is rescaled by expected exposures, the narrative ("low-frequency words converge quickly per exposure") competes with another observation in the paper, i.e., low-frequency words keep stronger ties to initialization. If those words have not actually stabilized by the final checkpoint, "convergence to final" is not a reliable guide for them.

**W5. Claims and presentation need correction and calibration.**

The paper uses ambiguous terminology and legends. Two examples are given below: 

- In Section 6, the text alternates among "distance decreased," "embeddings moved closer," and "similarity increased" to describe the same pairwise change. This slows the reader and risks confusion.

- In Figure 3(a), the legend says "Frequency bucket upper limit (100 words/bucket)," but the numbers are frequency ranks (larger number = less frequent), so the correct label is "Frequency rank bucket (100 words/bucket)".

Furthermore, some of the statements in the paper seem overconfident. In Figure 4(b), the text states that "high-frequency words (light blue) show substantially stronger correlations between input and output embeddings than low-frequency words," yet several lower-frequency curves sit above higher-frequency ones at many points, and no uncertainty bands are shown. The reasonable claim here is "on average" rather than an unconditional statement.

### Questions
Q1: What do the reported RSA patterns (your checkpoint-wise correlations between embedding distances and linguistic hypotheses) tell us about model quality in practice? Do the early emergence or early peak you highlight correspond to any observable improvement in the model’s ability to predict text or handle simple behaviors?

Q2: Please clarify the contribution to interpretability beyond descriptive timelines. Do these observations translate into explicit actions in practice, and if so, which training or evaluation decisions should reasonably be guided by them?

Q3: In Section 4, how do you distinguish structure that arises because certain words appear far more often during training from structure that reflects genuine syntactic or semantic organization?

Q4: In Section 5, how does the relationship between frequency groups evolve (e.g., head vs. tail as a whole)? Do cross-group relationships stabilize earlier, later, or differently than within-group neighborhoods?

Q5: In Section 5, why is the last checkpoint an appropriate reference for your "convergence" analyses? How sensitive are your conclusions to that choice, and are low-frequency words actually stable by the end of training?

Q6: Several frequency buckets appear out of order in figures (e.g., a lower-frequency curve above a higher-frequency one). How should readers reconcile this with statements that high-frequency words show "substantially stronger" effects?

Q7: Related to Q6, what level of variability or uncertainty is present in your curves and bucket comparisons? Can you clarify how confident we should be about the ordering across frequency groups?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work uses representational similarity analysis (RSA) to examine how vocabulary embeddings in language models evolve during the training phase. Basically, they conduct experiments to study the correlation between embedding-based distance and human-annotated distance for several tasks: semantic similarity, syntactic similarity, and word frequency. Their main findings indicate that 
* The semantic and syntactic converge to high correlations at an early training step
* Embeddings of high-frequency and function words converge to their final vectors faster than lexical and low-frequency words

### Strengths
* This work explores how does he embedding layer in transformers learn knowledge durting the training stage, which is interesting and novel
* Their findings track the representative changes of tokens and reveal their connections between initial random embeddings and the learned embeddings. The study of the less-frequent tokens might be useful for future studies since LLMs tend to fail on long-tail and domain-specific cases.

### Weaknesses
* I appreciate that the authors aim to answer this question: what are the representative changes of word embeddings during training, and this work provides some findings. However, I would expect some more insightful analyses. (1) LLMs are known for their contextual capabilities, which means they understand each word according to the context surrounding it rather than using the static embedding layer. What is the motivation to study this static embedding matrices? Can we apply the main findings in this manuscript to improve the current architecture or training strategies? (2) the authors found that the semantic and syntactic converge to high correlations at an early training step. Why do the embeddings exhibit such a pattern?  It would be more helpful if the authors could go further and offer explanation for their observations. (3)  The second main finding *"Embeddings of high-frequency and function words converge to their final vectors faster than lexical and low-frequency words"* is aligned with our intuition, which is not surprising. I would suggest exploring this point: one known issue of LLMs is that they tend to generate hallucinated answers for long-tail queries. Is it possible to learn the correlation between hallucination and low-frequency words? If so, it would be very useful to find solutions to mitigate this issue.
* The presentation requires another round of revision. I have difficulties understanding the details of the methodology and I have to guess to complete this review. I would suggest adding basic notations for clarification.

### Questions
* It is not clear how the authors perform the hypothesis RSA. According to my understanding, the task aims to assess the correlation between embedding‐based distances and human-annotated labels, but each step in this process is not clearly stated. It would be useful to introduce some notations to describe the problem, input, output and evaluation protocol. I have difficulties understanding this RSA, especially for the syntactic analysis. 
* In line 278 *"and if a word appears as a part of speech at least 5 times it receives distance 0 to all other words with that part of speech"*, I do not fully understand what is the meaning of this sentence

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper is a descriptive study that investigates how vocabulary representations of language models evolve during training. The experiments focus on two English language models (Pythia-12B and OLMo-7B) for which several training checkpoints are available, allowing to study their evolution over time. 

Across three experiments, the paper studies the relationship between the vocabulary representations at each training checkpoint and 1) human annotated datasets for vocabulary structure (semantic and syntactic similarity), 2) word frequency in the training corpus and 3) the final representation of the fully trained model. 

The paper uses Representational Similarity Analysis (RSA) to compare the proximity between two different representational spaces, model's representations and the human's annotation (or word frequency). 

The study finds that the models' representations of wordss correlate most strongly with human semantic and syntactic annotations early in training (around 15% of total steps), that representations of frequent words converge earlier than those of less frequent words, but also that nevertheless, the embeddings continue to change during the remainder of training, especially for rare and technical tokens. 

The paper claims that these findings could help to better understand the training mechanisms of language models.

### Strengths
1) The paper provides a handful of experiments to characterize vocabulary representation across several dimensions. The experiments and claims are clearly explained. I found the combination of hypothesis RSA and convergence RSA especially interesting. 

2) This methodology could be adapted to other languages and very interesting, especially for language with structures highly different from English, though this would require significant additional resources.

### Weaknesses
1) The paper studies two models, but doesn't and explicitely states that those models are in English and that the claims are only supported for that language. I also think that more models might be necessary to confirm the claims. The paper would benefit from quickly clarifying their architectures, and the content of their training data. 

2) Figure are sometimes distant from their discussion in the text (figure 2, and figures in the appendix). The main figures appear to combine both models to show general trends, while the details for each model are in appendix. 
To me at least some of the per-model results contain key information that should appear in the main body. Having to switch repeatedly between the text and appendix made the results harder to follow. 

3) The experiments are thorough, but some findings (faster convergence of frequent tokens) are largely intuitive, which limits the novelty of the contribution

### Questions
1) If I understood correctly, convergence RSA compares the model's representation at different training stages. While RSA seems suited for comparing model representations to human annotated ressources, is it equally appropriate when the representational system is the same? Can other comparisons could be considered in that case?

2) Given the relatively small number of word pairs in Experiment 1, how representative is this subset of the vocabulary, and could this affect the reported trends?

### Soundness
4

### Presentation
2

### Contribution
3
