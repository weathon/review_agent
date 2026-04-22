# On the Identifiability of Concepts from Large Language Model Activations

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 2, 6

## Abstract
Unsupervised approaches to large language model (LLM) interpretability such as sparse autoencoders (SAEs), offer a way to decode LLM activations into interpretable, and ideally, controllable concepts. On one hand, these approaches alleviate the need for supervision from concept labels, paired prompts, or explicit knowledge of a high-level causal model. On the other hand, without additional assumptions, SAEs are not guaranteed to be identifiable. In practice, they may learn latent dimensions that entangle multiple underlying concepts. If we use these dimensions to extract vectors for steering specific LLM behaviours, this non-identifiability risks interventions that inadvertently affect unrelated properties. In this paper, we bring the question of identifiability to the forefront of LLM interpretability research. Specifically, we introduce Sparse Shift Autoencoders (*SSAE*s) that instead map the \textit{differences} between embeddings to sparse representations. Crucially, we show that *SSAE*s are identifiable from paired observations that vary in *multiple unknown concepts*. With this key identifiability result, we show that we can steer single concepts with only weak supervision. Finally, we empirically demonstrate identifiable concept recovery across multiple real-world language datasets by disentangling activations from different LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper tackles non-identifiability in SAEs, introducing Sparse Shift Autoencoders (SSAE), which learn interpretable, steerable directions by training on differences between two LLM activations rather than on single states. Under strong but plausible assumptions, the authors argue these directions are identifiable (up to permutation/scale) and can be injected to nudge generation toward desired, human-meaningful behaviours without finetuning.

### Strengths
Identifiability (up to perm/scale) ties a math guarantee to usable vectors. Even if under strong assumptions, it nudges the field toward principled representation editing rather than ad-hoc hacks.

Out-of-distribution wins suggest the method isn’t just memorising and hints, the learned axes reflect model-internal structure.

Works with weakly paired data instead of per-concept labels, lowering supervision cost for discovering useful steering axes.

### Weaknesses
Pairs are constructed to alter a target concept, and supervision is felt to have shifted to the data generation process.

Identifiability leans on linearity and coverage that likely hold only near the last layer and for neat attributes.

### Questions
N/A

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a variant of the Sparse Autoencoder (SAE) architecture, called Sparse Shift Autoencoders (SSAE). Architecturally, the SSAE differs from (Cunningham et al, 2023)-style SAEs in three ways: 1. the SSAE autoencodes *differences* of hidden states, instead of the hidden states themselves, 2. the SSAE is entirely affine, instead of having an activation function, and 3. the SSAE is trained with an explicit l_1 norm constraint on the hidden layer, instead of using the l_1 norm in the loss term. These changes are made in an attempt to improve the identifiability of the learned features, which the authors primarily measure by the Mean Correlation Coefficient (MCC). The paper has both theoretical (proof-based) and empirical results. The theoretical results show that if concepts are represented linearly, an SSAE with perfect reconstructions and minimal l_1-norm essentially consists of identifying the transformation from representation space to concept space. The empirical results show that the discovered concepts can be used to steer, and that they have high MCC with "ground truth" features.

### Strengths
The paper contains both theoretical and empirical work. The theoretical work (Propositions 1 and 2) are intriguing results about the optimal reconstructions that would be found by a linear autoencoder. The authors do an excellent job explaining their formal assumptions, and justifying why they should be granted.

The authors compare their SSAEs to a robust set of baselines, including pre-trained SAEs, and naive approaches like PCA and Mean Differences.

### Weaknesses
The paper's theoretical section is based on assuming perfect reconstruction accuracy, which SAEs do not achieve in practice, and the authors do not address how error terms would propagate through the argument. This makes it hard to assess how this argument would apply to real-world models, where reconstruction accuracy is always imperfect.

The SSAE is trained on semi-labelled data, where data is pre-sorted to be identical words except with changed gender and/or language (e.g. ("Taureau","Cow") in the BINARY(2,2) dataset). 

All SSAE experiments in the body of the paper happen on datasets one or two varying concepts. Thus (assuming the linear representation hypothesis holds), the SSAE is given a relatively easy task of learning the vectors x and y from a dataset of the form {x,y,x+y}. The paper does include an appendix B.7 in which SSAEs are tested on synthetic datasets with up to 10 concepts, but they are not baselined against SAEs. However, prior work such as (Sharkey et al, 2022) showed that SAEs are able to identify ground-truth directions even on a more complicated synthetic dataset, with a comparable metric of Mean Maximum Cosine Similarity ≈ 1 with the correct choice of hyperparameters. 

The authors evaluate SSAEs for steering on two metrics: steering strength (Figure 2), and Mean Correlation Coefficient (MCC) (Figures 3-5). However, both of these metrics have flaws:

- Steering is done by adding a scalar multiple of a vector to the activations, but the authors appear to be comparing unnormalized vectors in Figure 2, so the quantity of "steering strength" is not comparable between vectors. In other words, if the features used for steering in Figure 2 are v and w for the SSAE and SAE respectively, then steering with the feature 2w would make the SAE's performance appear nearly identical to the SSAE's. Prior work has evaluated steering by the extent of steering possible without harming downstream performance, for instance in (Durmus et al, 2024) for SAE feature steering, and (Lamb et al, 2025) Table 1 for non-SAE feature steering.

- The authors never establish that MCC is a useful metric for measuring the quality of directions found. Indeed, in Appendix B.4 the authors note "that MCC is insufficient as a standalone criterion for model comparison: it cannot distinguish between representations that differ in their capacity to support reliable steering".

### Questions
**Questions**

1. Line 138 and Line 1185: Is Assumption 1 the same as Assumption 5? If so, would it be possible to remove Assumption 5 so that references like the one in line 217 do not require reading ahead?

2. Line 995:  The paper says "since \hat r and \hat q are linear..." but the hypotheses of the proposition only assume \hat r and \hat q are affine, not linear. Is there some other reason the bias terms on \hat r and \hat q should be 0, or does the statement need to be tweaked to allow bias terms?

3. Do Propositions 1 and 2 ever make use of the fact that \delta^z are shifts? Or would these make results apply to any linear SAE under Assumptions 1-3?

4. Your proof of Proposition 2 makes use of the fact that zero reconstruction loss is attainable, so all solutions must satisfy \hat q(\hat r(z))=z. Do you think this result is robust to errors? In particular, if the reconstruction loss is small in expectation, can you give a bound on how far \hat q and \hat r will be from the described shapes?

5. Relatedly, in your empirical tests, what reconstruction error (or what Fraction of Variance Explained) do your SSAEs attain?

6. In Line 322, when computing MCC, what is the "true" latent dimension you measured against?

----

**Small Comments**

The following changes would make small improvements to the paper:

Line 218: Please include a comment in the body of the paper that A_V+ denotes the pseudo-inverse. While you do describe your notation in an appendix, there is no indication that the reader should look there.

Line 1400: "Figure 10 and ??" seems to be missing a reference.

Line 1403: The sentence "On TruthQA, the MCQ track" is incomplete.

Lines 1520 and 1524: There seem to be missing close-parentheses in "abs(corr(x1,y1) + abs(corr(x2,y2)" and "mean(abs(corr(x1,y1),abs(corr(x2,y2))". 

Line 286: (Biderman et al 2023) do not study SAEs. Was this intended to be (EleutherAI, 2023)?

----

**Additional References**

(Durmus et al, 2024) Evaluating feature steering: A case study in mitigating social biases. https://www.anthropic.com/research/evaluating-feature-steering

(Lamb et al, 2025) Focus On This, Not That! Steering LLMs with Adaptive Feature Specification. https://openreview.net/forum?id=rbI5mOUA8Z

(Sharkey et al, 2022) [Interim research report] Taking features out of superposition with sparse autoencoders https://www.alignmentforum.org/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition

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
The authors propose a new kind of Sparse Autoencoder called SSAE that they train on activation differences between to prompts with known, different concepts. Example concepts: language, gender, truth. They hope that SSAE produces better steering vectors than ordinary SAEs.

The paper suffers from bad presentation and a lack of empirical evidence, especially showcasing what their model learned and comparing it to existing approaches. I have some concerns and questions regarding the method that I'd like to rebut.

### Strengths
- it is well-known that SAE features aren't great steering vectors and the authors tried to tackle this important problem

### Weaknesses
- The authors frame SSAE as an improvement of ordinary SAE. Yet, they don't show basic data about their models. For example, I'd like to see reconstruction error (variance unexplained), L0, feature dashboards, auto-interp, etc, and a side-by-side comparison to SAEs.
- When presenting a new SAE approach, I'd like to see at least one of two things: (1) the new SAE is better. Then, this needs to be proven empirically, or (2) the new SAE can answer scientific questions we are interested in that normal SAEs could not. This would also need to be shown empirically. I didn't see evidence for either.
- Training data need to be pairs where samples have known concepts. This is an important difference to SAEs which you can train unsupervised from activations alone. This limits the applicability of the method quite a lot because we don't have datasets for most concepts.
- The author's goal is to steer and they use a supervised dataset with labelled concepts to steer (which SSAE require). However, there are also other methods that can be used to find steering vectors in this setting, for example distributed alignment search. The authors don't compare their SSAE to steering methods that utilize the same dataset.
- I have some more concerns about the methodology but I put it under "Questions" below.


I found the presentation and the writing quite poor throughout the manuscript. Here are some examples to give the authors a better impression of why I found the paper very difficult to read:
- the title is generic and says nothing. What is the one result or idea that you'd like to convey?
- imprecision and missing definitions. ll 016 "SAEs are not guaranteed to be identifiable". What does "identifiable" mean? Do you really mean SAEs as a whole or rather SAE features? Why is an absolute guarantee important in the messy world of LLMs?
- the paper could be cut in size, e.g. sentences like "In this paper, we bring the question of identifiability to the forefront of LLM interpretability research" could be cut and replaced by e.g. a sentence explaining what identifiability is and why it's a big problem
- some words raise confusion, for example you say "token embeddings" which typically means the word embedding vectors before layer 0. Say "activation vector" or "representation of residual stream" for better understanding.
- important cites are missing, for example Bricken 2023 introduced SAEs independently and around the same time as Cunningham.
- math is confusing and used heavily throughout the manuscript. However, in most places, it's not needed and just makes the paper very hard to read.
- important information that would guide understanding like what the dataset is, at least what kind of data it contains, is only given briefly at the end of the paper. Showing data and how it's processed is often easier for the reader's understanding than frontloading unnecessary math.

### Questions
I have quite a lot of questions that I couldn't figure out yet by reading the paper alone.
- What is "identifiable"?
- In almost all cases, the authors train on only one single concept. Why would we need to create the entire SSAE if we only have one concept? Isn't the whole point of SAEs that we can learn millions of concepts at the same time? Why can we not use methods made for this use case like distributed alignment search (DAS)?
- Different to vanilla SAEs, there's no nonlinearity (i.e. ReLU) that could guide sparsity, it's only two linear maps. Why was no ReLU used? How can sparsity develop without ReLU? Activations point into different directions and have different scale. If you feed them through a linear map, they should still point into different directions at different scales (unless you defflate them all to zero in which case you couldn't recover anything). But then, it's virtually impossible for most values to be zero (i.e. sparse). Without ReLU, you might have low activations (because of l1 penalty) but they are still dense, not sparse. Can the authors report L0 metric and clarify how their architecture can learn a sparse code?
- Why should the learned representation have size V (ll179)? What do you mean by "learned representation" here? Features? Reconstructions? I'd expect there to be many more concepts than the activation space has dimensions because of superposition.
- Why were activations only taken from the last token in context (ll305)? Unlike e.g. BERT, decoder-only LLMs don't produce a sentence representation at their last token.
- ll322 what is "true latent dimension"?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper is motivated by recent negative results on using SAEs for LLM steering. The authors hypothesize that this could be due to the identifiability failure of the SAE method. That is, if we assume that model activations $z$ are obtained by a linear map $Ac$ mapping latent concept vectors $c$ in a high-dimensional space to activations $z$ in a lower-dimensional space, it can be provably impossible to recover $c$ by just knowing $z$ (just by linear algebra, $A$ must have a kernel, which opens the door to non-identifiability in principle). 

The paper proposes a new training variant for sparse autoencoders (the architecture is almost exactly the usual SAE architecture), which focuses on autoencoding *differences* between pair of activations instead of the individual activations themselves. This allows authors to sidestep the above identifiability problem: when we use differences, we have $z-z' = A(c-c')$, and if we assume that $c,c'$ can only differ in a small, limited set of coordinates $V$, we effectively only care about the part of $A$ applied to the coordinates $V$, $A_V$. $A_V$ may well be invertible. 

Relying on some prior work, the authors prove mathematically that with this formulation and assumptions along these lines, their SAE training variant will learn the original concept difference vector $c-c'$ up to coordinate-wise rescaling and coordinate permutation. Importantly, this allows the computation of steering vectors that precisely correspond to a single concept being different between two representations (though it doesn't tell us what the concept encoded by this vector means).

They carry out experiments in three setups:
- using synthetic datasets, compare the correlation between SAE decoders learned by their method from random seeds, and compare this to the analogous quantity for other SAE architectures and training methods. Results show their method outperforms SAE baselines and related representation learning methods.
- using synthetic datasets, compare (using cosine similarity) steering vectors predicted by their method vs ground truth activations encoding the desired change. Again results show an advantage of the method.
- run on realistic datasets with known concepts, and check the correlation between the concepts vs the SAE directions corresponding to them being active. They also apply these steering vectors for text generation, with mixed results.

### Strengths
- The paper proposes an inventive way to change the SAE training method that can plausibly improve the identifiability of the concepts found by the SAE by making its task "easier" in a sense. Improvements to SAEs are an interesting and important area of study today, as it becomes clear that SAEs are a useful tool for exploration/model-diffing/data attribution, but still come with a lot of alarming disadvantages in other areas.
- The method is solidly backed up by conceptual and theoretical arguments
- The writing is very clear. Experiments align well with the theoretical story and do not overclaim.

### Weaknesses
- Not sure if cosine similarity is the best way to measure closeness to a steering vector; ideally, we would have more causal tests similar to your experiments on the bias in bios dataset.
- On a high level, what we have done here is: made some assumptions about how concepts are mapped to activations, and used them to derive an SAE-type algorithm that can provably recover interesting stuff about these concepts. However, I don't see it clearly addressed in the paper why these gymnastics are necessary, given that we could have plausibly had a similarly convincing story with the original SAE algorithm in place. 
	- I'm not an expert on sparse recovery, but my understanding is that under certain assumptions on the matrix $A$ above, an ordinary SAE could also provably recover the concept vector. A quick search turns up e.g. Hu et al., _Global Identifiability of $\ell_1$​-based Dictionary Learning via Geometric Analysis_. 
	- Your paper says: "Unfortunately, unconstrained autoencoding objectives are non-identifiable (Hyvärinen & Pajunen, 1999), and sparse autoencoding objectives (Cunningham et al., 2023) may not be able to invert embeddings to potentially billions of concepts." - I guess you are hinting at the fact that billions of concepts might be too much for our SAEs, which are typically much smaller than that; and that your trick would allow you to, in principle, decrease the effective number of concepts to whatever you want. 
	- However, SAE also come with a sparsity assumption, so the effective number of concepts should be decreased somehow? It would be great to see this question addressed more thoroughly and categorically in the paper!

### Questions
- my main question goes back to the discussion on identifiability in ordinary SAEs - what are your reasons to believe your setup is better in some ways than the ordinary SAE setup + sparsity assumption on the concepts?

### Soundness
4

### Presentation
4

### Contribution
3
