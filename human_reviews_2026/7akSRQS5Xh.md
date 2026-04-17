# From Data Statistics to Feature Geometry: How Correlations Shape Superposition

- Decision: Accept (Poster)
- Scores: 2, 2, 8

## Abstract
A central idea in mechanistic interpretability is that neural networks represent more features than they have dimensions, arranging them in superposition to form an over-complete basis. This framing has been influential, motivating dictionary learning approaches such as sparse autoencoders. However, superposition has mostly been studied in idealized settings where features are sparse and uncorrelated. In these settings, superposition is typically understood as introducing interference that must be minimized geometrically and filtered out by non-linearities such as ReLUs, yielding local structures like regular polytopes. We show that this account is incomplete for realistic data by introducing Bag-of-Words Superposition (BOWS), a controlled setting to encode binary bag-of-words representations of internet text in superposition. Using BOWS, we find that when features are correlated, interference can be constructive rather than just noise to be filtered out. This is achieved by arranging features according to their co-activation patterns, making interference between active features constructive, while still using ReLUs to avoid false positives. We show that this kind of arrangement is more prevalent in models trained with weight decay and naturally gives rise to semantic clusters and cyclical structures which have been observed in real language models yet were not explained by the standard picture of superposition. Code for this paper can be found at: https://github.com/LucasPrietoAl/correlations-feature-geometry.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces BOWS, which induces the formation of complex structures of features in an autoencoder network that tries to compress and reconstruct binary bag-of-words vectors. With both a linear and nonlinear autoencoder, BOWS shows that networks do not need any goals besides compression alone in order to form these structures. The feature structures form as a result of feature correlations in the data, which are exploited by the autoencoders for compressive purposes. Experiments show the successful recovery of feature structures known to exist in LLMs (such as circular month-of-the-year features and latitude-longitude features), from the learned autoencoder representations. The authors offer a distinction between value-coding and presence-coding features, and argue that value-coding features are used to perform computation. BOWS sheds some light on the reason that LLMs form complex feature structures through superposition, arguing that feature compression, without any additional goal, is the cause.

### Strengths
- The paper provides convincing evidence that complex structures of features can emerge purely from the task of compressing features for storage, when there are correlations in the feature distribution.
- Figure 6 provides convincing evidence that complex structures of features form and then disappear as the latent size changes, when training a network to compress information.

### Weaknesses
The paper is unnecessarily complex and convoluted for it's goal in helping interpretability efforts.
- BOWS is much less a "framework" than an application of a pre-existing technique for demonstrating superposition, in Elhage (2022) [1], but on a custom bag-of-words dataset. Most of the "framework" parts of the work (techniques for analysis of results, PCA, visualization, training methodology) are inherited from Elhage (2022).
- The framework's goal is to show that compression alone can be responsible for complex feature structures learned in LLMs, but this can already be shown by taking a PCA of the data (implicit compression), making the autoencoder analysis an unnecessarily complicated solution.
- The distinction made between presence-coding and value-coding neurons is weak/useless. The difference as defined seems to boil down to an artifact of the way the English language treats some things as quantifiable/measurable and others as not, rather than intrinsic to an LLM's internal functioning, so it is trivially true that there exist neurons for both presence-coding and value-coding things. The "value of the north-south coordinate" is just "the presence of northerliness". While this may seem contrived, there are plenty of fuzzy edge cases to be made ("lots of hair" (presence) vs "many hairs" (value)).
  - Altogether, I do not see why making this distinction of quantifiableness helps us understand the internal functioning of LLMs any better than before. In general, I don't see how the mathematical behavior of the autoencoder is supposed to depend on things outside of the correlation structure of the features, such as our external understanding of the feature's meaning as the presence vs value of something.
  - If the authors want to instead define presence-coding versus value-coding based on how the features are being used, they should clarify precisely what conditions are sufficient to qualify the feature as one or the other. An example does not suffice as a definition, as in lines 343-347.
  - The authors give no examples of how making this distinction helps us to reason about other aspects of the features. Being able to identify the features as "value-coding" does not help tell us how it's used. Rather, identifying how it's used allows us to label it as "value-coding", and there doesn't seem to be any purpose served after that.

[1] Elhage, et al., "Toy Models of Superposition", Transformer Circuits Thread, 2022.

### Questions
I am presuming below that as indicated in the abstract, the main goal of the paper is to show that compression alone can be responsible for complex feature structures in LLMs, when features statistics are correlated.
- It seems like this goal is already achieved with just PCA, so why ever bother with the autoencoder?. In what case does the autoencoder reveal any structures to you that PCA/probing does not? As you mention in Section 3.1, the linear autoencoder will be recovering the top two principal components anyways.
- It seems like you designed the autoencoder as the simplest possible example where compression can be demonstrated as the root cause for the emergence of the feature structures. Why complicate the matter with a nonlinear autoencoder? Is it to take into account the possibility that the presence of nonlinearity can disrupt the usage of the complex structures as the most efficient compressed representation?

Much of the rest of the analysis describes how the autoencoders behave, but this is not necessary in order to see that the structures are inherited from data statistics, since it is already evident from PCA and probing.

For line 397, is there converse evidence that the presence-coding features are _not_ used? Also, I find this experiment unconvincing of the importance of the difference between value-coding and presence-coding. It looks like you are defining the features that are known to be used in a particular way as the value-coding ones, and the rest as presence-coding, and then using that to argue that the value-coding ones are the useful ones, which I do not see the point of.

Random:
- Figure 7 right: You are trying to probe one city's coordinates from an 8 dimensional vector which predicts the relative positioning between two cities. Which of the two city's coordinates are you trying to probe, and which city is plotted?
- Figure 3 typo: (a-c) instead of (a)
- Please try to keep the figures placed close to where they are referenced.

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
This work studies correlations between (input) features in a “toy models of superposition” (Elhage et al. 2022) type setting, and introduces a distinction between linear and non-linear superposition.  While Elhage et al. focused on non-linear superposition, this work argues that linear superposition is more common in real data.

The submission also introduces a Bag-of-Words Superposition (BOWS) setup, which uses co-occurrence in real documents to generate correlated binary feature vectors.  (BOWS is a bit of a misnomer since bag-of-words representations capture multiplicity, but the representation in this work does not).  The work also postulates that these co-occurrence statistics may be largely responsible for the particular geometric patterns observed in LLM representations (e.g. months being arranged in a circle), although I don’t believe this hypothesis was actually tested.

### Strengths
The observation that linear superposition is more common (if correct), seems important.

The BOWS setup seems like a nice, sensible approach to introducing realistic correlation structure, and has the bonus of enabling researchers to bring their knowledge of words’ semantics to the analysis of experiments.

### Weaknesses
(major): I believe this work is in need of more rigor in establishing its central definitions and concepts (e.g. “superposition”, “linear superposition”).  For instance, I’m not sure what is the content of the statement “This explicitly shows that linear dimensionality reduction enables a form of superposition (d = 12 > m = 2) by exploiting feature correlations, without requiring any non-linearity.”  This just sounds like it’s saying “you can reconstruct an input well using a few principle components when input dimensions are highly correlated”, but that’s nothing new.  Similarly, “features in linear superposition inherit their structure from their covariance matrix” seems vacuous if linear superposition basically just means PCA.  While superposition is a more established concept, it relies (in my experience) on assuming some ground-truth set of underlying features.  In this case, I guess these are the words in the BOWS representation, but this should still be made more explicit.  I would like the discussion of superposition and linear superposition to be grounded in a rigorous mathematical exposition of the concepts.

(major): The submission claims that Elhage et al. (2022) don't study linear superposition or correlated features. However, Elhage et al. do consider correlated features, and state “when there isn't enough space to represent all the correlated features, it will collapse them and represent their principal component instead”.  This is a significant mischaracterization of key related work, and makes me question the novelty of this submission.

(moderate): I think some of the statements in this submission overstate the successes of mechanistic interpretability, eg:
- “These approaches have successfully uncovered interpretable units corresponding to semantic concepts, syntactic roles, or specific input patterns.”
- “Initial works in MI studied interpretable monosematic neurons”
I think referring to neurons as 'interpretable' is generally accurate.  They may seem more or less interpretable, but so far I’ve not seen sufficient evidence to justify claims that we really understand what neurons are doing, except perhaps in the context of particular tasks / distributions of inputs.  This is an important distinction, as overstating the success of interpretability can provide false assurances.

(minor): The related work section should do more to connect the referenced works to the submission.  


(nit): The acronyms SDL and LRH should be introduced.

### Questions
“However, polytope-like structures have not been observed in standard LLM activations” seems to refer specifically to Elhage et al.’s 2022 work; however, Park et al. (2025) claim to have found polytopes in the activations of Gemma.  Does this invalidate the quoted claim?  How are the polytopes discussed in Park et al. related to those from Elhage et al. (2022)?

How do the experiments in Section 5 differ from what was already done by Gurnee & Tegmark (2024)?

“In contrast, the patterns in Figure 7 emerge despite inputs being uncorrelated,” Can you please make the claim about inputs being uncorrelated precise and support it?

I’m not convinced that “circles and clusters resemble the global semantic organization first reported in distributional word embeddings”.  Can you elaborate on and substantiate this claim?

The abstract states: “Our findings suggest that the semantically meaningful structures observed in language models could arise driven by compression alone, without necessarily having a functional role beyond efficiently arranging feature representations.”  What would it mean for this to be the case?  Are there experiments that could falsify this as a hypothesis?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper investigates the geometric structure of internal features in mech interp, aiming to bridge the gap between "toy model" theories of superposition and observations in real-world language models. Prior work suggested superposition creates interference-managing structures (like polytopes), but empirical studies of LLMs instead find semantically-rich structures (like semantic clusters or ordered circles for months).

The authors introduce a framework called Bag-of-Words Superposition (BOWS), where a simple ReLU autoencoder is trained to compress high-dimensional, sparse, binary bag-of-words vectors derived from real text. The use of the BOW space in such analysis is interesting as it provides a proxy for the platonic, objective, or "ideal" embedding space for features that we cannot access.

The paper makes two key claims:

It distinguishes between non-linear superposition (as seen in toy models), which uses non-linearities like ReLU to create local structures (e.g., antipodal pairs) to manage interference between uncorrelated features, and linear superposition, which emerges when features are correlated.

In the linear superposition regime—which is induced by tight bottlenecks or, significantly, by weight decay—the non-linear AE simply learns to linearly encode the low-rank structure (i.e., the principal components) of the data's covariance matrix.

The authors demonstrate that this linear superposition mechanism is sufficient to reproduce the exact semantic clusters and circular representations seen in LLMs. This suggests these structures are a "parsimonious" explanation, arising merely as a byproduct of efficient compression, and may not have a specific functional, computational role.

### Strengths
1, The paper's primary strength is its clarity. The distinction between "linear superposition" (PCA on correlated data) and "non-linear superposition" (local, ReLU-dependent structures for uncorrelated data) is a very useful and clear conceptual framework. And the flow of the paper is natural, too.

2, The BOWS framework offers a good trade-off, it's more realistic than toy models but far more controllable than a full LLM. Using it to show the emergence of semantic clusters and circles from data statistics alone is highly effective. The experiment showing that a ReLU AE transitions from a linear (PCA) regime to a non-linear (antipodal) one as the latent dimension $m$ increases is clean and convincing.

3, This work provides an interesting for feature geometry. It challenges the assumption that structures like semantic clusters or "month circles" are necessarily functional or computationally constructed by the model. The idea that they are simply a byproduct of compression (driven by data correlations and weight decay) is a fundamental claim that the mech interp community must consider.

### Weaknesses
1, The main limitation is the simplicity of the BOWS setup. An AE trained on static BoW vectors does not necessarily capture all the SAE variants that exist in literature today, since many of them explicitly make architectural changes to cater to the space where features live.

2, Also, the paper seems to strongly implies that these structures are just byproducts and not functional. This dichotomy might be false. A model could (and likely would) exploit this emergent, PCA-driven structure for computation. Framing it as byproduct OR functional is perhaps too strong and I think the two are not mutually exclusive.

### Questions
My question would be whether "byproduct" structures also be "functional"? Couldn't the model leverage the fact that the optimal compression scheme (PCA) naturally arranges features in a computationally convenient way (e.g., a circle for months)? Or perhaps a little bit more challenging (in future works), what happens when the low-rank structure does not facilitate (or even adverserial for) a geometry that makes underlying computation (like modular arithmetic) easier?

My question, however, does not impact my perception of this paper as a novel and solid contribution to mech interp community.

### Soundness
4

### Presentation
4

### Contribution
3
