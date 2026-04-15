# Observable Propagation: Uncovering Feature Vectors in Transformers

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5

## Abstract
A key goal of current mechanistic interpretability research in NLP is to find linear features (also called "feature vectors") for transformers:  directions in activation space corresponding to concepts that are used by a given model in its computation. Present state-of-the-art methods for finding linear features require large amounts of labelled data -- both laborious to acquire and computationally expensive to utilize. In this work, we introduce a novel method, called "observable propagation" (in short: "ObsProp"), for finding linear features used by transformer language models in computing a given task -- using almost no data. Our paradigm centers on the concept of "observables", linear functionals corresponding to given tasks. We then introduce a mathematical theory for the analysis of feature vectors: we prove that LayerNorm nonlinearities in high dimensions do not affect the direction of feature vectors; we also introduce a similarity metric between feature vectors called the "coupling coefficient", and prove that it accurately estimates the degree to which one feature's output correlates with another's. Armed with these tools, we use observable propagation to investigate the features that cause gendered occupational bias in a large language model. In our experiments, we identify the specific features used by the model for predicting occupation based on a gendered name, and find that some of the same features are used by the model for predicting grammatical gender. Our results suggest that observable propagation can be used to better understand the mechanisms responsible for bias in large language models.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a method to compute a "feature vector" in transformers for a given task.

### Strengths
The method requires no labeled training data, in contrast to other similarly motivated methods like probing.

### Weaknesses
I find the paper quite opaque. The paper should directly give a formal definition of an "observable", state how it is used/useful, and show how it is computed. Instead, the paper gives a long narrative of several motivations in words. The theorem about how the layer norm doesn't substantially the direction of feature vectors seems pretty specific to the narrative here; it is hard to see much general value out of the context. 

I'm not very knowledgeable in the area of gender bias analysis so I'm not qualified to make comments on the value of the experiments. But given my lack of confidence on the value of the technical part, it is hard for me to see the value of the experiments either.

### Questions
N/A

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a method called Observable Propagation (ObProp) to identify feature vectors in large language models. Because of the success of LLMs, interpreting what their intermediate layers learn has turned out to be an emerging topic. The authors propose that linear transforms of language model logits, which they call observables, can be captured approximately via feature vectors that can be computed by a simple forward model through the model.

In more detail, the basic idea is to define an abstract vector called observable n and feature vector y, such that the inequality n.f(x) = y.x is satisfied for a non-linearity f (an intermediate layer of the LLM). This inequality cannot be exact but the authors propose to compute an approximation by approximating each neural non-linearity by its gradient (Taylor approximation) and each attention mechanism by the attention weights (which are assumed to be constants). This lets them learn a feature vector y for each observable n.

The authors propose that studying the norms of such vectors and looking at their correlation (a coupling coefficient) will suggest us more about the observables. Simple experiments using GPT-Neo-1.3B suggest that such learnt feature vectors for the gendered pronounds prediction weakly match the ones recovered by looking at logit differences (via path patching). Additional experiments on occupational gender bias predicts that LLMs do exhibit gender-occupation bias.

### Strengths
- Mechanistic interpretability is an important research direction and the authors propose a simple method towards this research direction which does not rely on large-scale compute or data.

- Their experiments, including App. H, suggest that the ObProp algorithm does seem to recover meaningful interpretation of various attention heads that occur in the model, which could then be analyzed more rigorously using more powerful methods.

### Weaknesses
- This method seems like an oversimplification of the architecture of an LLM. The ObProp algorithm propagates through the layers by approximating almost all non-linearities by their first-order gradients.

- Moreover, the work abstracts out QK circuits in LLMs, i.e. the attention scores are treated as constants and not as non-linearities, therefore the problem is oversimplified and I'm not sure the theorems shown are necessarily that insightful. As the authors note, their method ObProp does not capture the action of induction heads.

- Theorem 1 proposes that layer norms do not affect the feature vectors, by showing that the expected cosine similarity is close to 0. However, in this theorem, the activations and observables x, n are chosen to Gaussian with mean 0 and covariance I. What's the point of this assumption? And how do the authors conclude that the theorem 1, as stated with these distributions, implies their claim about layer norm for general activations?

- Related to the above, why did the authors look at uniform distributions on the sphere in theorem 2? Even if the vectors have bounded norm, why do the authors assume uniformity?

### Questions
Some questions were raised above.

- Typo in the equation in 2.2 and below, score has both 1 and 2 subscripts.

### Soundness
1 poor

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose a method ObsProb  to better understand which parts (paths) of neural networks in neural networks contribute to which predictions. The idea is to start with a linear functional on the logits layer and then give some (approximate) mathematical arguments as to how this can be propagated down the network. 

Empirically the authors do a case study with gender bias. They show that the n_subj (he/she) and n_obj (her/him) feature vectors derived from ObsProb have high cosine similarities for 4 heads and these cosine similarities are vary high. This indicates that according to the authors' approach, the same underlying features of the network are being used for these predictions. Moreover, the authors find a similar result with n_{bias} (which is an observable for occupational bias).  They find a particular path for n_{subj} has a very high correlation with that of n_{bias}.

### Strengths
Interpretability is complex and challenging for LLMs. Being able to understand how different predictions are made my similar parts of the model could be very useful. The authors' results are around gender and occupational bias using their approach are insightful.

### Weaknesses
The empirical evaluation is largely a case study and doesn't feel rigorous. It would be more convincing to have a more quantitative evaluation across more types of observables compared to a baseline, along with the case study.

Nit: It would be good to define the term "unembedding matrix".

### Questions
My main concerns with the approach are around rigorous evaluation as discussed above.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
