# Disentangling and Integrating Relational and Sensory Information in Transformer Architectures

- Decision: Reject
- Scores: 5, 10, 3, 6

## Abstract
Relational reasoning is a central component of generally intelligent systems, enabling robust and data-efficient inductive generalization. Recent empirical evidence shows that many existing neural architectures, including Transformers, struggle with tasks requiring relational reasoning. In this work, we distinguish between two types of information: *sensory* information about the properties of individual objects, and *relational* information about the relationships between objects. While neural attention provides a powerful mechanism for controlling the flow of sensory information between objects, the Transformer lacks an explicit computational mechanism for routing and processing relational information. To address this limitation, we propose an architectural extension of the Transformer framework that we call the *Dual Attention Transformer (DAT)*, featuring two distinct attention mechanisms: sensory attention for directing the flow of sensory information, and a novel relational attention mechanism for directing the flow of relational information. We empirically evaluate *DAT* on a diverse set of tasks ranging from synthetic relational benchmarks to complex real-world tasks such as language modeling and visual processing. Our results demonstrate that integrating explicit relational computational mechanisms into the Transformer architecture leads to significant performance gains in terms of data efficiency and parameter efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes to add a relational attention mechanism to the transformer architecture in order to improve relational reasoning performance.

### Strengths
1. The proposed method is shown to out-perform vanilla transformers on a small set of synthetic or simple real-world tasks.
2. The paper is written clearly and easy to follow.

### Weaknesses
1. There is limited novelty. The relational attention sounds very similar to the graph attention network to me. Also the idea of having a perception module followed by relational reasoning networks have been explored for many years in visual reasoning domain (e.g. [1,2]). The Symbolic attention also sounds like Link mechanism in the Retriever[3]. 
2. The experiments are only performed on a small set of simpler tasks. I wonder how the proposed method will perform for more complex tasks.

[1] Amizadeh, Saeed, et al. "Neuro-symbolic visual reasoning: Disentangling." International Conference on Machine Learning. Pmlr, 2020.
[2] Wang, Duo, Mateja Jamnik, and Pietro Lio. "Abstract diagrammatic reasoning with multiplex graph networks." arXiv preprint arXiv:2006.11197 (2020).
[3] Yin, Dacheng, et al. "Retriever: Learning content-style representation as a token-level bipartite graph." arXiv preprint arXiv:2202.12307 (2022).

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
The paper describes a new architecture designed to make it easier for transformers to work with relational information. The authors provide  several experiments that indicate the architecture outperforms standard transformers.

### Strengths
This is a strong paper, and I recommend accepting it. The authors take a well-motivated challenge (helping transformers work with relational reasoning) and define a natural extension to the transformer architecture that attack that challenge.

The basic idea of exchanging "relational" information seems solid, and the given architecture is seems like a relatively simple way to achieve that, essentially adding one more bilinear function to the mix. Crucially, the new architecture decouples strength of attention from strength of relation (this distinguishes it from an earlier proposal known as relational cross attention). One might worry about the added parameters, but of course the authors are careful to control for that variable. 

The experimental data is impressive, especially because the architecture seems to work on a broad set of tasks. It's interesting that even image recognition improves. (This might be a hint that the new architecture, although meant to represent relations, might have other benefits as well.) I appreciated the careful comparisons with baselines.

Overall, this seems like an important contribution to the literature.

### Weaknesses
I believe there are some opportunities for improving the exposition of this paper. To begin with, "sensory" doesn't seem like the right metaphor. I realize the cognitive science origin, but I also think it's worth being careful with brain metaphors. I wonder if it would be better to talk in terms of "unary" vs. "binary" attention heads, or "first-order" vs. "relational," perhaps.

The first paragraph (and maybe much of the second) of the introduction seem unnecessary, and it might be possible to cut them entirely.

I found the first few explanations of the architecture confusing, and didn't really understand the "type" of r or symbols until I got to the explicit formulas. I wonder if it's worth making this a little more precise earlier.

The theorem in 2.4 gets very little play, and I'm not sure how important it is. I'd recommend either relegating this entirely to the appendix, or spending a bit more time explaining why it matters here. (One issue is that plain-vanilla transformers are computationally very powerful already, so it's not clear what this theorem adds.)

The second paragraph of section 3.1 seemed redundant, and might be able to be cut.

Figure 5 is potentially interesting, but I wonder if the story could be illustrated better by picking one layer, and showing attention for all the "normal" heads vs. the "relation" heads. I also wonder if there is any "low-hanging fruit" for other visualizations. For example, for training when the relation matrices are not constrained to be symmetric, do they ever end up learning to be near-symmetric? That said, this is a long paper already, and the authors explicitly mention interpretability as future work, so this is certainly an optional change!

As just mentioned—and I can't fault the authors for this—this paper has a huge amount of material, mostly in the appendices. All of that is good and necessary, but it's easy to miss details when reviewing, which is why I've put a relatively low confidence score in my review.

### Questions
If you cut some of the text as suggested above, you might have more room for future work. Do you have thoughts about how this might apply to other architectures, such as graph neural nets?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a modification of multi-head attention to better capture the relations information between objects. The proposed alternative is named dual attention, which concatenates the results of self-attention and the results of a new module named relational attention. Here, relational attention is similar to self-attention, except that the value to aggregate is replaced by a weighted sum between an relation vector computed from each pair of objects and a symbol vector. Experiments on synthetic and real data demonstrate data efficiency and parameter efficiency.

### Strengths
- The paper is well written and easy to follow
- experiments are diverse in domains and supporting main claims

### Weaknesses
- The idea does not seem very novel or original. There are many attempts to integrate relational information into the attention mechanism in the Graph Neural Network community, with the closest one I can find being "Learning Graph Representations Through Learning and Propagating Edge Features" (https://ieeexplore.ieee.org/document/10004977). Specifically, Eq. 2 directly gives the general form of the proposed relational attention. This previous work goes on with slightly different parametrization of f and g, i.e. concatenation instead of dot product etc, but has an overall very similar central idea. This paper should at least cites this line of work and compare against them as baselines.

### Questions
Are there experiments with the learned Symbolic Attention?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel attention mechanism by which transformers are encouraged to explicitly represent information about relations between elements of a sequence or set of tokens. This relational information is encoded separately from sensory information, or information about the specific entities that enter into a relation. This new dual attention transformer architecture is evaluated on a diverse series of modalities and tasks, demonstrating enhanced performance in language modeling, a simple visual classification task, and a toy relational reasoning dataset.

### Strengths
The proposed mechanism is novel, yet is a natural extension of the attention mechanisms within a standard transformer. The idea of explicitly representing relations between entities is an important one that merits further investigation. Empirically, the DAT appears to be more data efficient than the standard transformer, which is especially important in the case of language modeling.

### Weaknesses
While the work is interesting and the idea of building in inductive biases toward representing relational information is potentially useful, there are several framing and experimental issues that should be addressed. First, the claim that standard attention mechanisms only represent sensory information is empirically false. The authors themselves cite several works describing how attention in language models often captures syntactic information, which is inherently relational. Furthermore, recent work has explicitly found that finetuned ViTs represent sensory information in early layers, but represent abstract relational information in their later layers [1]. Finally, “register tokens” – tokens that encode little information about their local image patch, but instead represent global information about an image – have been discovered in large pretrained ViTs [2]. Their very existence challenges the notion that attention mechanisms are only transmitting “sensory information". At very least, the sensory information is not necessarily tied to the pixels and token embeddings that the activation vector appears to correspond to.

The proposed method has at least two separate important components: the representation of relational information, and the tying of key and query matrices. It appears very important to test a variant of the standard transformer subject to tied key and query matrices. For example, Figure 8 shows that removing this symmetry condition deteriorates the benefits of relational attention quite a bit, especially in low-data regimes. This should also be done for the experiments presented in section 4.3. Similarly, when using position-relative symbol assignment, a control that modifies the standard transformer with relative positional embeddings should also be included. This is especially important because prior work has demonstrated that these positional embeddings can aid transformers on synthetic generalization tasks [3].

In section 4.4, the authors suggest that relational attention might be especially useful in that it captures semantic, rather than syntactic, relations between words. However, this same phenomenon also happens in standard language models! A quick check with GPT2-small using the same stimulus reveals similar attention patterns for the token “model” (Layer 0, Head 0, verifiable using this colab: https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb) . This section should thus be revised to properly contextualize the utility of relational attention in contrast to existing models.

[1] Lepori, Michael A., et al. "Beyond the Doors of Perception: Vision Transformers Represent Relations Between Objects." arXiv preprint arXiv:2406.15955 (2024).
[2] Darcet, Timothée, et al. "Vision transformers need registers." arXiv preprint arXiv:2309.16588 (2023).
[3] Csordás, Róbert, Kazuki Irie, and Jürgen Schmidhuber. "The devil is in the detail: Simple tricks improve systematic generalization of transformers." arXiv preprint arXiv:2108.12284 (2021).

### Questions
How are you generating attention scores for tokens after “model” and “state” in Figure 5? Is this not a causal language model?

### Soundness
2

### Presentation
4

### Contribution
3
