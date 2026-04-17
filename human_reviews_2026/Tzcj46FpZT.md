# Concept Component Analysis: A Principled Approach for Concept Extraction in LLMs

- Decision: Reject
- Scores: 4, 4, 8, 4

## Abstract
Sparse autoencoders (SAEs) have emerged as a popular approach for extracting interpretable and monosemantic concepts by decomposing the LLM internal representations into a dictionary. Despite their empirical progress, SAEs suffer from a fundamental theoretical ambiguity: the well-defined correspondence between LLM representations and human-interpretable concepts remains unclear. This lack of theoretical grounding gives rise to several methodological challenges, including difficulties in principled method design and evaluation criteria. In this work, we show that, under mild assumptions, LLM representations can be approximated as a {linear mixture} of the log-posteriors over concepts given the input context, through the lens of a latent variable model where concepts are treated as latent variables. This motivates a principled framework for concept extraction, namely Concept Component Analysis (ConCA), which aims to recover the log-posterior of each concept from LLM representations through a {unsupervised} linear unmixing process. We explore a specific variant, termed sparse ConCA, which leverages a sparsity prior to address the inherent ill-posedness of the unmixing problem. We implement 12 sparse ConCA variants and demonstrate their ability to extract meaningful concepts across multiple pre-trained LLMs, showcasing clear advantages over SAEs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
### Summary

This paper take a very principled approach. Starting from a theorem stating the generally linear relationship between the representation (underlying next token prediction) and latent variables, they build an SAE like architecture based on this idea, majorly removing the nonlinearity in reconstruction and only use nonlinearity in sparse penalty. 

They systematically tested the new SAE design on LLMs and evaluated their reconstruction MSE and the ability to recover latents from counterfactual sentence pairs where only one latents changed.

### Strengths
### Strength

- This paper touch on an important question, about principled design of interpretability methods e.g. SAE. The discussion of the conceptual basis and theoretical assumptions of this method is highly valuable and laudable.
- Experiments are very comprehensive, tested many architecture combinations.

### Weaknesses
### Weakness

- Although Fig2 show the comprehensiveness of the ablation study, it feels more like a supplementary table. Should the authors have a main version of the figures highlighting the comparisons they want to talk about in the text? Currently it’s hard to focus the eye.
- **Rationale of mean pearson correlation**
As we know one of the most challenging part for interpretability method is quantitative evaluation. MSE is well accepted, however the 2nd metric mean Pearson Corr is a bit confusing. Since only one latent variable changed in the sentence, do we compute its correlation with all the latent extracted from the proposed method? Then we want it to be selectively modulating some latent but not others. If that is the case why “mean” is the suitable statistics, why not max or sparsity or sth else?
    - Also we have only 27 counterfactual pairs, is it enough to evaluate this method ? is it too noisy?
    - I feel before diving into ablation and method comparison, spend a little bit more text on the comparison will be helpful!
- There seems to be a gap between the main theoretical motivation and the architecture design in the end.
    - The main theorem is saying the the representation is an affine / linear function of the log posterior probability of latent $\log p(z|x)$, which I agree. But Eq.4 basically used a fixed linear mapping  (with regularization) to estimate $z$. I don’t think the theory gurantee this inference will work. Similar to previous SAE, this is still amortized sparse inference, which is a bit guessing game. C.f. the gap between amortized inference and optimization based sparse inference showed in [^1], in some sense the linear encoder will have bigger gaps.
    - Based on this gap, I’m not exactly sure of the motivation of the architecture change proposed by the authors…

[^1] O'Neill, C., Gumran, A., & Klindt, D. (2025). Compute optimal inference and provable amortisation gap in sparse autoencoders. ICML

### Questions
### Questions

- Is my understanding correct that the ConCA design in Eq. 4 is a linear autoencoder in the end? (since we can absorb normalization layers into linear layers after learning), but now we have sparsity imposed in the exp transformed domain?
- Is the main theorem in Sec. 2 a bit similar to classic works on the linear algebraic structure in word embedding space pre-LLM era (Arora 2016-18, e.g. [^2, ^3]), esp. compare to Theorem 1 in [^2]? e.g. for word2vec. 
I feel you don’t need next token prediction objective, as long as it’s word prediction from context (e.g. BERT, word2vec) the same logic can still work. So in that sense, the theorem 2.1 should be connect to this bigger background.

[^2] Arora, S., Li, Y., Liang, Y., Ma, T., & Risteski, A. (2018). Linear algebraic structure of word senses, with applications to polysemy. TACL

[^3] Arora, S., Li, Y., Liang, Y., Ma, T., & Risteski, A. (2016). A latent variable model approach to pmi-based word embeddings. TACL

- More generally is it the right way to think of natural languages using the one step latent generative models?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Concept Component Analysis (ConCA), a new framework for concept extraction in LLMs. The authors posit a new theoretical model where LLM representations are an approximation of a linear mixture of the log-posteriors of underlying latent generative concepts. ConCA is proposed as an unsupervised unmixing process to recover these concepts. A practical "Sparse ConCA" variant applies sparsity to the exponentiated features (the posteriors, $p(z_i|x)$). Because the exp function is claimed to be numerically unstable, the paper substitutes it with approximations like SELU or SoftPlus, though it provides limited analysis of whether these alternatives align with the theoretical motivations. The framework's evaluation relies on MSE for faithfulness, Pearson correlation to known latents for concept alignment, and also includes empirical results also include performance on downstream and OOD tasks.

### Strengths
- The theoretical contribution of concepts as linear combinations of logposteriors of latent concepts.
- It is great to see push back on the fields assumptions about linearity/sparsity/SAE usage generally and new modes of thought are exciting to see!
- The writing and figures are clean with little to no grammatical issues

### Weaknesses
- The paper claims that the exp function is too numerically unstable to use, which is fine, but there is very limited analysis or exploration of whether approximating exp with another function (e.g., SELU, ELU, SoftPlus, etc) are viable alternatives which follow the theoretical motivations. 

- Do we actually want to interpret only the things that are the underlying generative variables of the data? It seems to me like the models themselves don't learn the true causal variables and succumb to spurious correlations often - and we want our interpretability methods to identify those.

- The MSE is a weak metric to test whether the concepts captured by ConCA are faithful. A better method is to do some kind of activation patching and analyze whether the reconstructed activation *works like the original* and not just whether a large part of it is reconstructed: A model can achieve low MSE by perfectly reconstructing all the "unimportant" parts of the representation vector while failing to capture the one or two critical directions that actually influence the model's final decision.

- In addition to the faithfulness metric, the pearson correlation to extracted features is also a large part of the evaluation that has little explanation or justification as to why these would be sufficient to measure. It seems like there are many more concepts of interest than verb inflections, adjective transformations, size/thing/nouns, and translations that would be of interest to SAEs, but the current manuscript solely looks at these. 

- I don't understand the point footnote 2 is trying to make. Is it saying that this assumption is commonly used in other works and is therefore suitable formulation here? If so, there should be a citation motivating the use of Eq. 2.

- The error bars in Figure 5 left do not show that ConCA outperform SAEs in a statistical significant manner since the std is much larger than the difference in performance, and the text in lines 461-3 should be updated accordingly. Alternatively, more samples can be run to lower the std.

Small Stuff

- In the abstract, it is said "ConCA variants and demonstrate their ability to extract meaningful concepts across multiple pre-trained LLMs, showcasing clear advantages over SAEs." It would be ideal to include what these advantages are in the abstract so the reader knows up-front what the main advantages are without searching the rest of the paper.

### Questions
- Could the authors look at a smaller, toy setting (e.g., using toy or hard coded data, clamped activations, etc) where exp is valid to compare whether the substitution activation functions are valid?
- Do the findings w.r.t SAEs hold for other measures of *functional* faithfulness such as patching back in the activations?
- Does the ConCA enable capturing latents that do not generate the data? For example, spurious correlations? A common application of SAEs is trying to find spurious features and understand failure cases - does ConCA still allow for that?
- Why are the 27 counterfactual concept pairs a reasonable set for comparing methods? Is it possible to look at simple, more toy settings as an alternative? Or maybe, counterfactual data pairs don't exist for single concepts and concepts co-occur with each other, making this type of evaluation difficult in the first place.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes Concept Component Analysis as a principled route to extract human interpretable concepts from language model activations. The central theoretical result shows that, under a discrete latent variable generative view of text and a standard next token objective, the model representation $f(x)$ can be approximated as a linear mixture of the stacked log posteriors of latent concepts, with a mixing matrix $A$ and bias $b$. 

This motivates unmixing $f(x)$ to recover per concept log posteriors. A practical instantiation called sparse ConCA trains a linear encoder and decoder to reconstruct $f(x)$ while enforcing sparsity not on the latent $\hat{z}$ itself but on a smooth exponential surrogate $g(\hat{z}$, reflecting that sparsity should live in probability space! 

Experiments compare twelve sparse ConCA variants against several sparse autoencoder baselines across multiple models, evaluate with reconstruction loss and a Pearson correlation to supervised counterfactual probes, and show few shot and out of distribution gains on more than one hundred tasks. Figures 1 through 5 and Theorem 2.1 carry the main story, with an appendix contrasting this theorem to prior work and giving proofs and evaluation details.

### Strengths
I loved this paper. The writing is excellent and the story reads well. Congrats.
Some positive points (P) that I will note here:

P1. Clean theoretical through line. Figure 1 and Theorem 2.1 tie a latent variable model of text to the next token objective and yield the linear mixture $f(x)$ approximately equal to $A$ times stacked log posteriors plus $b$. This give an implicit definition of a concept (latent factor organizing the data manifold), gives a crisp target for what a concept feature should be and neatly recasts concept extraction as linear unmixing.

P2. Framing clarifies the role of sparsity. Table 1 makes explicit that sparsity belongs in the exponentiated latent space because zeros in log space correspond to probability one, which flips the usual SAE intuition. The architectural choice to implement g with smooth exp like activations is carefully motivated.

P3. Broad and replicable evaluation. The paper reports ablations over normalisation and activation, comparisons on counterfactual pairs, and downstream few shot and OOD performance across many datasets, with concise plots in Figures 2 to 5 and implementation specifics in the appendix.

### Weaknesses
Ok, now for the weaknesses that I found in the paper, i'll group them in Major (M) and minor (m).

M1. Clarify what the mixing matrix implies and connect it explicitly to the Linear Representation Hypothesis (LRH). Equation 3 shows a linear mixture $f(x)$ equals $A$ times the stacked log posteriors plus $b$. This already implies that the encoding is linear in concepts up to an unknown mixing matrix, which is closely aligned with the linear representation hypothesis. I suggest stating this implication upfront and spelling out how it differs from prototypes or distance to centroids views. 

M2. Strength of assumptions labelled as mild. The diversity condition requires m plus one output tokens producing an invertible matrix of classifier head differences, and the informational sufficiency condition requires conditional entropy of concepts given context to be near zero. These are important for the derivation. Any way to have proxy measure in practice ? This would let readers see where the theorem is likely to bind and where it may not.

M3. Identifiability and alternative theorem comparisons. Appendix C argues your theorem needs fewer diversity assumptions and yields component wise mixtures rather than mixtures over joint assignments. It would be helpful to operationalise this with a small synthetic where ground truth marginals and joints are known and to show that ConCA recovers marginals while prior formulations do not. Otherwise the very nice Table 2 comparison remains mainly narrative.

M4. Evaluation set for Pearson correlation is quite narrow. The counterfactual corpus uses twenty seven pairs drawn from analogy like transformations. This is a good start but it is narrow.

Now for the minor concerns,

m1. Define concept precisely and early. Page one and two motivate concepts as human interpretable units, later formalised as discrete latent variables. A compact operational definiton at the start of Section 2 that is reused in experiments would improve clarity (I like when things are well-defined).

m2. Figure 2 grids and Figure 3 bars are dense and fonts are small. Please increase font sizes and add short textual takeaways in the captions, for example the correlation range where ConCA dominates and the reconstruction loss regime you care about.

m4. Reporting variability in Figures 3 to 5, always show uncertainty bands or standard deviations for both correlation and MSE, and for AUC bars, not just means. Some are present but others look single valued in the panels.

### Questions
See Major point 1-4 and minor 1-4

### Soundness
4

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
3

### Summary
The authors suggest ConCA, which uses a probabilistic framing of how latent features can interact and be expressed in a contextful decoder language model, and unmixes the latent concepts by recoring the log-posterior of each concept.

### Strengths
I like the framing of this paper. It starts with a thoughtful feature-latent space definition, which allows arbitrary interactions between features, and then argues for why the linear process might be true in this case. This is a  stronger case than assuming that concepts are linearly encoded a priori.

### Weaknesses
W1 I do not feel that this paper offers a significant amount in terms of either contributing something to how we understand that models work, or contributing a very useful tool that will be widely used by interpretability. I feel that ConCA is well-theoretically motivated, but in practice I’m not sure what to take from it, or if there is a reason to switch to using it instead of other dictionary learning methods. 

W2 The empirical results do not have too many widely-applicable takeaways. It would have been nice to see some more empirical results that validate the sentiment that you express in the last sentence of section 4 (line 471). For example, in a toy identifiability setting with a lot of interaction between latent features, is this unmixing better than something like an SAE? What kinds of interactions lead to features not getting pulled out by SAEs, but getting successfully unmixed by ConCA?

### Questions
Are the concepts pulled out by ConCA more easy to understand than other methods?

(no need to respond to small suggestions below)

(minor) It would be helpful to have slightly more developed captions, what each figure is showing and what we should take away from

(minor) Many of the later figures are pretty color-based, it might be worth seeing if there are ways to make them more readable. 

(v. minor) there are a few typos in the paper, a few are verb inflection typos, perhaps it’s worth a read-through looking just at that

### Soundness
3

### Presentation
2

### Contribution
2
