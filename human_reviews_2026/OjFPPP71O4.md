# Probability Signature: Bridging Data Semantics and Embedding Structure in Language Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
The embedding space of language models is widely believed to capture the semantic relationships; for instance, embeddings of digits often exhibit an ordered structure that corresponds to their natural sequence. However, the mechanisms driving the formation of such structures remain poorly understood. In this work, we interpret the embedding structures via the token relationships. We propose a set of probability signatures that reflect the semantic relationships among tokens. Through experiments on the composite addition tasks using the linear model and feedforward network, combined with theoretical analysis of gradient flow dynamics, we reveal that these probability signatures significantly influence the embedding structures. We further generalize our analysis to large language models (LLMs). Our results show that the probability signatures are faithfully aligned with the embedding structures, particularly in capturing strong pairwise similarities among embeddings. Our work offers a universal analytical framework that investigates how token relationships direct embedding geometries, empowering researchers to trace how gradient flow propagates token relationships onto embedding structures of their models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper considers embedding models such as MLPs and LLMs, and to shred light on how the structure of the embedding space relates to the model and the data generating distribution.

They do this by defining probability signatures which represent properties of the data generating distribution (though this is not entirely clear, see Weaknesses) and showing how the gradient flow for the embedding and unembedding matrices depends on them for a collection of models (linear, MLP, LLM).

Unfortunately, it is not clear precisely what mathematical objects the probability signatures are, making it hard to understand the implications of the results (see Weaknesses)

Experimentally, the authors plot matrices of distances between columns of the embedding  / unembedding matrix and of ground-truth distances between the tokens in their experiments. They recorded how correlations between these matrices change over training, and they are typically highly correlated after training when the model is sufficiently expressive to represent the true label function.

### Strengths
I think this paper is potentially full of good ideas, but the lack of accuracy in many places makes it hard to know.

With improved accuracy, I think Propositions 1 and 2 have real potential value. They show the form of the gradient flow of the embedding and unembedding matrices, and the proofs appear to be sound (although pointing the proofs in the main text would be nice).

I also think that the corollaries which show simplified forms of these flows for specific models could be very valuable.

### Weaknesses
I have two main strands of comments, one of the implications of the results, and the other on accuracy.

**Implications.**
The central message of this paper is not clear. The paper would really benefit from clearly stating its goal, conclusions, and how the theoretical results and experiments back up the conclusion. I think there's potential good ideas in this paper, but its hard to see how they relate to one another.

While I think the derivation of the gradient flows could be a valuable contribution by itself, its not clear to me from the paper what this actually tells us about the geometry of the embedding space, and how they relate to the experiments and plots in the paper. Some additional explanation would be helpful.

**Accuracy.**
There are many parts of the paper where a lack of mathematical accuracy makes it hard to know what is being said.

- The definitions of $\phi_x^y, \phi_x^X$ and $\psi_\nu^X$ define vectors and the definition of $\phi_x^{X \mid y}$ defines a matrix. Yet, in the text immediately below $\phi_x^y$ is described as a distribution and $\phi_x^X$ as a probability (i.e. a scalar). Figure 1 shows plots of $\phi_\alpha^X$ which without further explanation suggests they are functions (perhaps these are supposed to be pdfs of distributions?).
- In Proposition 1, $N_x^{\text{in}}$ and $N_{x,\nu}$ are defined as "the count of sequences contain $x$....". I don't know from context what  I am counting in. Is it the batch or something else?
- In Section 5, function $f_{\text{add}}. \tilde{f}_{\text{add}},\ldots$ etc. are defined. I assume these define the labels $y$, but its not said.
- Given I am unclear about what the probability signatures are, its not clear to me what is plotted in Figure 1.
- What is $\cos(x,y)$? Is it the cosine distance between $x$ and $y$? If so, this is not the same as the cosine.

### Questions
Please clarify
1) The goal of the paper. Is it to understand the geometry of the embedding space, or is it to understand the training dynamics. If the goal is the latter, what can be learned from the theoretical results and experiments?
2) The definitions of the quantities I mention in the weaknesses section.
3) What is being plotted in Figure 1?

### Soundness
2

### Presentation
2

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
This paper studies the relationships between probabilistic structures in a model's training data, the gradients used to train it, and the resulting model weights. By defining probability signatures the authors study when ordinal structure emerges in the model trained on an addition task. The analysis is then further extended to a Language model trained on the pile dataset, showing how embedding structure changes over training.

### Strengths
The paper looks at a worthwhile question - understanding the relation between data structure, gradient structure and weight structure. While extensive interpretability work has look at model weights, far less relates this to gradient information in a meaningful way.

### Weaknesses
This paper is overall challenging to read. In part because it would benefit from a proofread for grammatical and typographic errors. In part because the authors choices of formalisation, which are verbose and at times difficult to follow. Additionally visuals often lack adequate labelling - figure 1 for example does not indicate the units for the x axis.

More generally the novelty of the work is currently unclear. The majority of results appear to show relationships between structure in the training data and in the embedding matrix, an area that is relatively well studied. Additionally the core claim appears to be that the statistical structure of the training data affects the solution a model learns; this is a foundational premise of machine learning. Core results appear to be presented in figures 4 and 6 as heatmaps with no summary statistics - it is difficult to tell the significance of a result from looking at a heatmap alone.

If the authors could better clarify how their work studies gradient information, which they claim it does, that may help support novelty here.

### Questions
Is this work's core finding that the statistical structure of the data affects learning?

You mention semantics in the title of the paper and the introduction. However the work itself makes no mention of the discipline of semantics in NLP, Neuroscience, Linguistics, or Cognitive Science more generally. What is the definition of semantics here and how does it relate to other fields?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- This paper studies how data distributions shape embedding structures in language models, proposing a collection of probability signatures that encode semantic relationships among token embeddings.
- The authors approach this problem through theoretical gradient flow analysis, small-scale experiments (single-layer linear models and feedforward networks) on toy composite addition tasks, and large-scale experiments using the Qwen2.5 architecture trained on subsets of the Pile dataset (as well as analyzing the pretrained mode)
- They find that embedding similarities align with these probability signatures, revealing how semantic structures emerge from data.

### Strengths
- The paper tackles an interesting and fundamental problem of the impact of data distribution on embedding structures and introduces a clear framework for understanding this via probability signatures, providing an interpretable lens into how data semantics are encoded in embeddings.
- It also has strong theoretical grounding, employing gradient flow dynamics to give a principled explanation for the relationship between embeddings and the probability signatures derived from training data.

### Weaknesses
- Due to the extensive notation and multiple quantities being analyzed, the paper could benefit from improved clarity and organization, clearly summarizing the main findings and their supporting evidence in each section.
- While the controlled synthetic experiments are valuable, it would strengthen the paper to include additional large-scale experiments in the LLM setting to complement the ones already in the paper.
- For LLMs, since the model consists of multiple transformer blocks that further process the initial embeddings, it would be helpful to extend the analysis beyond the initial embedding layer, as this could provide insight into how these relationships evolve deeper in the network (at least empirically).
- The paper lacks discussion on practical implications; elaborating on how these findings might inform future model design, training strategies, or interpretability efforts would make the work more impactful.

### Questions
- Just to clarify, when computing the probability signatures, it is just based on proportion of the occurrences in the train data across all sequences?
- It may be good to check how some of the metrics progress across training (eg. the correlations $R_{\text{cos}}(W^E,\phi^{\text{next}})$ and $R_{\text{cos}}(W^U,\varphi^{\text{pre}})$ as part of your analysis

### Soundness
3

### Presentation
2

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
This paper explores how the structure of token embeddings in language models arises from the statistical properties of the training data. The authors propose the concept of  “probability signatures” —statistical descriptors derived from input–output co-occurrence distributions—that aim to explain and predict how embeddings are organized. Using controlled addition tasks with linear and feedforward models, as well as experiments on Qwen2.5 models trained on subsets of *The Pile*, the paper shows strong correlations between embedding similarity and these probability signatures. Theoretical analysis based on gradient flow dynamics further connects data distribution, optimization behavior, and model architecture, offering a unified view of how semantic patterns in data shape embedding geometry.

### Strengths
1. The paper introduces the concent of "probability signatures" to links data statistics and the geometric structure of the embedding space, which is interesting.

### Weaknesses
1. The writting of this paper should be improved signficantly. The current manuscript does not look like a professional acamedic paper, in terms of organization of the content and clearness of the content. For instances, almost all of the content in the first and many content in second paragraphs are too broad and have no close connection to the topic of the paper, making them look very superfluous. After reading the introduction, it is still not clear why do we need to study this problem and why this problem is important. 

2. The poor writting is also reflected in the clearness and rigorousness of the content. Let's take the gradient flow defined in Proposition 1 as an example, the authors never formally introduce the time variable, but they directly define the gradient $\frac{d {\mathbf{W}}_x^E}{d t}$. The presentation here is very confusing and unrigorous.

3. While the paper analyzes how data distribution shapes embeddings, its practical implications for improving or understanding large-scale models are still limited.

4. The theoretical analysis is based on highly simplified models and may not capture the complexity of modern Transformer architectures, so its explanatory power for real LMs’ embeddings is limited.

5. The framework mainly holds under small initialization and early-stage linearized training. It remains unclear whether such simple probability signatures still explain embeddings after nonlinear dynamics dominate.

### Questions
1. During LLM training, does the influence of probability signatures on embedding structure emerge rapidly and then stabilize, or does it continue to increase throughout training? Providing an empirical trend or metric over training steps would be helpful.

2. As training proceeds beyond the small-initialization and early-linearization regime, does the explanatory power of probability signatures remain consistent, or does it degrade as nonlinear effects accumulate?

3. Could the proposed probability-signature framework have potential applications in guiding the training of modern LLMs? If so, what are the possible directions or limitations?

### Soundness
2

### Presentation
1

### Contribution
2
