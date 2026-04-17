# Statistical and structural identifiability in representation learning

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Representation learning models exhibit a surprising stability in their internal representations. Whereas most prior work treats this stability as a single property, we formalize it as two distinct concepts: **statistical identifiability** (consistency of representations across runs) and 
**structural identifiability** (alignment of representations with some unobserved ground truth). Recognizing that perfect pointwise identifiability is generally unrealistic for modern representation learning models, we propose new model-agnostic definitions of statistical and structural near-identifiability of representations up to some error tolerance $\epsilon$. Leveraging these definitions, we prove a statistical $\epsilon$-**near-identifiability** result for the representations of models with
nonlinear decoders, generalizing existing identifiability theory beyond last-layer representations in e.g. generative pre-trained transformers (GPTs) to near-identifiability of the intermediate representations of a broad class of models including (masked) autoencoders (MAEs) and supervised learners. 
Although these weaker assumptions confer weaker identifiability, we show that independent components 
analysis (ICA) can resolve much of the remaining linear ambiguity for this class of models, and validate and measure our near-identifiability claims empirically. With additional assumptions on the 
data-generating process, statistical identifiability extends to structural identifiability, yielding a simple and 
practical recipe for disentanglement: ICA post-processing of latent representations. On synthetic 
benchmarks, this approach achieves state-of-the-art disentanglement using a vanilla autoencoder. 
With a foundation model-scale MAE for cell microscopy, it disentangles biological variation from technical batch 
effects, substantially improving downstream generalization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides relaxed conditions for representational identifiability in self-supervised learning, extending previous results beyond the (pen)ultimate layer. For their weaker results, the authors proposed a further preprocessing step by linear ICA to remove the linear indeterminacy. The extension of identifiability to the approximate regime is an important contribution, especially for practial applicability of identifiability.

**The paper seems to be very interesting, and I believe it has great potential. However, I have concerns, especially about the imprecise use of he word "identifiability."** Indeed, what the authors call "recoverability" is what most of the literature calls identifiability. 

Although I think this paper has the potential to be a spotlight/oral, given my current concerns, I chose a conservative score. **If the authors can address my concerns (or correct me if I have misunderstood any of their claims), then I will raise my score.**

### Strengths
- The paper aims to relax identifiability conditions, thus, increasing the practical relevance of the field of identifiability. Thus, it is very timely and useful.
- The extension to approximate identifiability is definitely underexplored (though not *entirely*, novel, see, e.g., [1])
- The theory is contextualized for practically relevant model families (MAEs/GPTs)
- The empirical evaluation is detailed (though a bit unclear, see weaknesses)
	- I especially liked the PCA+rand control in Table 3
- A.1.3-4 are nice discussions on architecture-specific cases of the theory
- A.1.5 is a nice discussion on ICA in latent space

- [ 1 ] Reizinger, Patrik, Szilvia Ujváry, Anna Mészáros, Anna Kerekes, Wieland Brendel, and Ferenc Huszár. "Position: Understanding LLMs Requires More Than Statistical Generalization." In Forty-first International Conference on Machine Learning.

### Weaknesses
My main concerns are about the clarity/phrasing, and NOT the technical contributions.

### Major points
- **My biggest concerns is the improper use of identifiability/recoverability**: when the authors refer to identifiability, they do not refer to the commonly used term in the ICA literature (which is formulated in terms of the ground-truth data generating process). The authors introduce (superfluously) recoverability for the (as far as I can tell) same concept. When they talk about identifiability, they refer to the definition used by Roeder et al., 2020. I strongly suggest to correct the use of the term to avoid confusion with the majority of the identifiability literature. Note that this does not devalue any contributions, only makes their presentation presumably confusing. For a comparison between the two identifiability notions, see, e.g., [ 1 ] (where the authors call it relative - Roeder - vs absolute - Hyvarinen - identifiability)
- Thm. 1.:
	- the theorem would benefit from a more-detailed explanation. In my opinion, the current intuition section is more like a sketch. At least I couldn't figure out the intuition.
	- E.g., why are all $F, G, H$ needed?
	- What is the role of bi-Lipschitzness?
	- Small remark: there is a identifiability result for next-token prediction, see, http://arxiv.org/abs/2503.08980
- Def. 2: rephrase the intuition with the mainstream identifiability definition (the connection between the two concepts in A.3 is nice)
- Table 1: it is missing context, it is unclear from the caption what the metrics mean, what was measured, etc. Please make the caption self-contained
- The proof of the Theorem in A.3 would benefit from more steps and inline explanations of what is happening between the steps

### Minor points
- L045: "Second, model-specific identifiability results fail to predict the empirical alignment seen across different architectures and modalities (Huh et al., 2024):" 
	- This is unclear how it would hold for the most widely-used identifiability definition, as assumptions generally require assumptions that do not constrain the modality or the architecture. Please elaborate
- L053: In this paper, we argue that many current identifiability results in representation learning are overloaded with recoverability guarantees of some latent component of the data-generating process,
	- I don't understand this sentence
- L143: here, the "All or None" paper by Marconate et al. seems to be very relevant, please discuss it. Also, [ 1 ] proposes a simialr notion, termed $\varepsilon-$non-identifiability
- L277 - Example: I like the example, though it's a bit too involved and I am missing the intuition for it. If you could add such an explanation (preferably with a figure), that'd help the reader
- L363: the citation showing Anonymus et al is a typo, I presume?
- Bottom of page 7: there seems to be some typo/latex error
- Maybe formalize all your assumptions into an assumption latex environment (it's OK to have this in the appendix), such that readers can refer to it


- [ 1 ] Reizinger, Patrik, Szilvia Ujváry, Anna Mészáros, Anna Kerekes, Wieland Brendel, and Ferenc Huszár. "Position: Understanding LLMs Requires More Than Statistical Generalization." In Forty-first International Conference on Machine Learning.

### Questions
- L138: why do you use the $L^{\infty}$ norm?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors study $\epsilon$-near identifiability and recoverability in self-supervised models. They define near-identifiability of a model as satisfying an identifiability criteria up to $\epsilon$. They then show that, for a model with identifiable output, the internal representations are near-identifiable up to rigid transformations and up to $\epsilon$ a function of the Lipschitz constants of the intermediate layers. They then show that ICA post-processing can be used to preserve near-identifiability, which results in near identifiability up to permutation and sign flips.

### Strengths
Existing works in identifiability usually treat functions $f$ as fully black-box, but in practice are parametrised as neural networks. Arguably the black-box perspective has been taken to its limit and there are not many substantially new developments in identifiability here. If I understand correctly, what this paper does is leverage the neural network (or compositional) structure, noticing that outputs are usually identifiable (e.g., assuming optimization is well-posed), and leveraging that structure to obtain a result on near identifiability of intermediate layers, which is how many representations are extracted in practice. This is a really neat, original idea that should have significant impact for the representation learning community.

### Weaknesses
- The recoverability section, unless I'm missing something, seems to just be talking about how well-specification is usually assumed in these papers, but it is nice to have it spelled out with examples specific to the theory here.

### Questions
- Are any of the results really specific to self-supervised models, or do they have some special qualities (e.g., in the loss)? Why is the emphasis placed on this setting in the framing?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work, the authors provide a novel identifiability result which characterizes the similarity in parameter space for a broad family of models up to rigid transformations in terms of a bound which is a function of the bi-Lipschitz constant of a model. The authors then show a relationship between this notion of a identifiability and the idea of achieving identifiability w.r.t. a latent variable model, which they refer to as recoverability. Several experiments are run across large scale models and various datasets highlighting that applying ICA on learned representations yields disentangled representations.

### Strengths
I believe this work makes contributions which are potentially of substantial interest to the identifiability community.

*  Namely, providing clarity on the the relationship between identifiability w.r.t. a latent variable model and models being identifiability w.r.t. each other is an important nuance to explore. 

* Furthermore, Theorem 1 provides very general result characterizing model identifiability which I believe is of interest even if the result relies on a Lipschitz constant which may be difficult to measure in practice.

* The authors make a notable effort to conduct experiments which highlight representation similarity and disentanglement in diverse settings and across different large scale models.

### Weaknesses
* While I find the author's contributions noteworthy, I found it very difficult to parse the contributions by reading the abstract and introduction of the paper. In the introduction, the authors introduce several purported issues with existing identifiability results and then present their contributions as solving these issues. I found this relationship between these issues and the authors contributions difficult to connect and ultimately obfuscating of the authors main contributions. I would suggest the authors rework the abstract and intro to better clarify their contribution.


* While I understand that the authors wish to explore the implications of their results on image data, the experiments feel a bit disconnected from the theory. I believe a simple toy experiment validating the authors theory in a setting in which the bi-lipschitz constant can be measured would be of value.


* Additionally, I would appreciate if the authors could discuss the relationship between their results and prior results [1] which have results related to epsilon near identifiability.


**References**

[1] Nielsen et. al 2025, When Does Closeness in Distribution Imply Representational Similarity? An Identifiability Perspective

### Questions
* Can the authors clarify the positioning/motivation of their results within the identifiability community?

* Do the authors believe it would be possible to conduct experiments on toy data which aim to more rigorously test the theoretical statements?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
A quite detailed analysis around self supervised learning and representation stability. There are technical contributions that expand the applicability of identifiability theory to a broader class of models. While the experiments remain quite limited and synthetic, there is a lack of more challenging data with known ground truth hence making it a challenge for the authors.

** Strength **
- The paper introduces quite a few theoretical contributions that take a step towards better understanding self supervised learning which is one of the most prominent solution today
- The writing and technical quality is above acceptance level with clear novelty and insights
- While the scope is not as wide as it could be, the empirical section is quite detailed and feel quite reproducible

** Weakness **
- Numerous formatting and typos throughout the manuscript that makes the paper feels a bit rushed in some parts, e.g., `multiviarate`, `identfiability`, `c̃itepjumpcp` and so on
- The paper is quite dense mathematically and the appendix doesn't help much as it doesn't provide more introductory materials to help unfamiliar readers get started. While this is not a major weakness, it would be great to bridge a bit more the paper's content with the practitioners unfamiliar with those concepts
- It would also be great to better connect the theory and results with practical settings for SOTA models

### Strengths
Please see summary

### Weaknesses
Please see summary

### Questions
Please see summary

### Soundness
3

### Presentation
3

### Contribution
3
