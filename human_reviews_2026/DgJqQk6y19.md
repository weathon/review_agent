# The Softmax Bottleneck Does Not Limit the Probabilities of the Most Likely Tokens

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
In many popular transformer architectures, an output projection matrix linearly maps lower-dimensional embeddings into a higher-dimensional space of logits. It has been shown that this leads to a softmax bottleneck that prevents the production of arbitrary probability distributions.  It has been argued that this limits large language models (LLMs) in their ability to express next token probabilities that perfectly align with the statistics of natural language.  We focus on the ability of such models to produce accurate probabilities for just the top-$m$ tokens.  We provide theoretical bounds that show that even a randomly initialized projection matrix can successfully do this for rather large values of $m$, supported by empirical results on both random and trained matrices. This raises questions about whether the softmax bottleneck significantly limits the capabilities of LLMs. We also derive bounds on the maximum number of probabilities that any trained output projection matrix can specify.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper offers a new look at the so-called "Softmax Bottleneck," which is said to limit the ability of a model as it can struggle to go from inner embeddings to logits over the whole vocabulary: the authors claim that what matters in practice is to predict accurately the probabilities for top tokens, especially given that sampling is usually done from a truncated distribution. The paper first gives theoretical lower bounds showing that indeed any chosen set of m tokens can be the most likely ones, with high probability, and even reach, collectively, a specific probability. It then validates this on GPT‑2 and TinyLlama: in practice this works for somewhat surprisingly larges m.

### Strengths
- this is a really inspiring take on the softmax bottleneck, as indeed given the prevalence of truncated sampling, what matters is often the ability to produce valid probabilities for the top tokens of a vocabulary ;
- the authors show the theoretical possibility to produce the embeddings that would result in a given m-subset getting the most probabilities, for  a randomly initialized projection matrix, with some bounds, and even to reach a given probabilities for this subset ;
- the experiments validate their theory for a trained GPT-2 and TinyLlama, far exceeding the lower bounds ;
- the paper is convincingly written, with a well formulated problem, sound proofs and related experiments.

### Weaknesses
- the proofs seem sound but this is really to the best of my understanding and I had to trust the authors for this theoretical work. Similarly, it's not clear what to make of such a sentence "since this derivation involves some approximation, we have empirically confirmed that simulations match our theoretical predictions (not shown)";
- although this work proves the existence of "embedding that the OPM will map into the appropriate probability distribution", there is no guarantee that the model will produce it ;
- furthermore, although this is interesting, it does not say anything about the ability of a given network to produce the embeddings for "desired" m-subsets in many contexts. This is not the point of this work, but it does take a practical look at the softmax bottleneck, so my _practical_ concern might be warranted...

### Questions
repeating what I see as a weakness:
- what do you think of the reachability question? Your results do establish existence of an embedding $x$ for a given $A$ but I wonder whether a network can be trained to produce that $x$...
- are we sure that same (trained) network would produce the valid embeddings, whatever the context? This seems important to fully grasp the extent of your work.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The work addresses the question of the softmax bottleneck in transformers -- that is, the output projection limiting the production of arbitrary probability distributions. The authors ask whether the softmax bottleneck restricts the LLM from representing the probabilities of the top-m most probable tokens, arguing that the exact probabilities of unlikely tokens are less important. The authors provide theoretical results showing that there exist OPMs that can represent any specific probabilities over top-m tokens for large m and empirical results on GPT-2 and TinyLlama.

### Strengths
- Extends on prior work on softmax bottleneck, reframing the problem around top-m probabilities
- The research question is clear as well as the writing
- The derivations give nice lower bounds for random matrices and the theoretical insights are supported by experiments on GPT-2 and TinyLlama.

### Weaknesses
- The results show the existence of embeddings that can realize given top-m probabilities, but does not address whether these are learned by real transformers
- Assumes low-probability tokens are irrelevant, but there may be domains (e.g., RL fine-tuning or exploration) where coverage over rare tokens is important

### Questions
- What is the role of weight tying here? 
- Are there settings (e.g., RL, exploration, or calibration tasks) where representing low-probability tokens accurately would matter, and how would your framework apply there? Do the values of m you found seem sufficient?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper tackles the softmax bottleneck problem, i.e. the study of the expressivity of usual neural language models that use a hidden dimension that is smaller than the vocabulary size $N$. In that setup, it has been shown that there exist probability distributions in $\Delta^N$ that cannot be predicted. In this paper, the authors state that language model outputs can match probability distributions with a relatively large support, either by predicting the token set that belong in that support, or even by predicting exactly the probabilities for the tokens in such supports. As a result, they argue that the softmax bottleneck is not a dramatic issue for language models, which they assess through a short experimental analysis.

### Strengths
This paper studies a very interesting topic and conducts a novel and relevant theoretical analysis.
- **Theoretical results**: The propositions presented in this paper are novel and provide a more profound understanding of this issue. They shed light on the complexity of reachable distributions, and probabilize (a part of) the set of reachable distributions, which is both a challenging and exciting outcome. Even though the proof in section 4.1 is not perfectly rigorous, it makes very reasonable assumptions and uses the Inverse Wishart distribution in this context, which is a new and very insightful contribution in the context of random matrices for the softmax bottleneck problem. I appreciate that the authors extended their work to fixed top-m probability. Proposition 4 is very elegant. The use of results on the signrank is also very novel and could open new venues for this topic. It leads to a clean result on the minimal dimensions needed to accurately match the supports of probability distributions.

### Weaknesses
Although I deeply appreciate the core theoretical results of this work, I quite disagree with some of the claims and conclusions made in the abstract, introduction, and empirical sections. I also believe that this paper could be enhanced by deeper empirical experiments.
- **Claims about broader implications**: It is mentioned in the abstract and introduction that "the softmax bottleneck does not significantly limit the capabilities of LLMs", or that "limitations to the expressiveness of transformers are not really that significant". My understanding of what is proven in the paper is 1- for a single given target probability vector and a random OPM matrix, there exists an $x$ for which the predicted probability matches the top-$m$ target probabilities; 2- if one only cares about the support but wants to match any set of targets on the top-$m$ ranking of tokens, then in most setups $d \approx m/2$ is sufficient. Hence, it is unclear whether matching any set of targets for the correct probabilities (let alone the order of such probabilities) is possible. This is a crucial distinction, as the complexity of matching any permutation of token order is much higher than matching the support as a set. Moreover, even when there would exist output representations that would give the desired probabilities, they might be configured in ways that are very difficult to reach during training, which would limit the applicability of these results to usual training setups. What can thus be concluded from this paper is that the softmax bottleneck phenomenon does not strongly limit the prediction of the next-token probability supports, and that individual "low-entropy" probability vectors should be truthfully matched with non-negligible probability.
- **Lack of clarity in some proofs**: The proof of Proposition 2 is a bit hard to follow and could be made clearer. The first paragraph is a bit confusing and it seems like it could be summarized to convey the idea more directly. Moreover, the statistical independence of the entries of $(A_mA_m^T)^{-1}$ is never verified empirically in the paper.
- **Experimental design**: The experimental section is less appealing than the theoretical section. Figure 1 verifies the lower bound given in Propositions 1 and 2 (and incidentally shows that it is not particularly tight). Figure 2 computes the bound curve for several setups. Figure 3 explores a question that seems loosely related to the topic at hand, and that was covered many times by the anisotropy literature (see works of Ethayarajh et al., among others). A lot of questions remain unanswered, from the necessary experiments to extensions that would have been relevant: given an actual token distribution taken from natural language, how much of the probability supports can a random\trained matrix cover? What are the types of x and s that are observed? In a synthetic controlled setup, e.g. data generated from a bigram with known supports, can a model be trained to properly recover the supports up to $m$ tokens? In cases where the supports are of different sizes, ie where m should be different across target probabilities, what types of solutions are found and with what performance? The experimental section also ignores the second theoretical section of the paper.

In the current state of the paper, the claims and conclusions that are made about the relevance of the softmax bottleneck are not strongly supported by the theoretical section and the experiments. It is particularly important as this article is quite technical, which implies that readers may take these claims for granted without thoroughly reading the paper and understanding its potential limitations. Hence, it is crucial that the claims accurately reflect the results that are presented to avoid any misinterpretation.

Overall, I am excited by the topic and theoretical part (which I would rate 8/10), but I am underwhelmed by the conclusions and experimental sections (which I would rate 2/10).

### Questions
- Could you report the results that show independence for the rows of $(A_mA_m^T)^{-1}$?
- By my understanding, Proposition 3 could be proven more easily by setting all a_j at distinct positions on the unit sphere in 2d and setting x_j = a_j?
- Do OPMs actually train to account for larger m values? Is it possible that the learned A_m do not have the used invertibility properties?
- To what extent does token rank and specified ratios would affect the results in section 5?

### Soundness
1

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates both theoretically and empirically the impact of the softmax bottleneck on the ability of large language models to correctly represent the probabilities of the m most probable tokens. Their conclusion is that the softmax bottleneck does not seem to "provide any limitations to LLMs in most realistic settings".

### Strengths
- An interesting contribution to the "softmax bottleneck" literature that goes beyond previous work (notably Demeter et al. 2020 and Grivas et al. 2022).
- The paper combines theoretical analyses and empirical results.
- The paper is generally clear and well written.

### Weaknesses
- The motivation for the research question is not convincing enough. Why is it so important that models can assign the correct probabilities, summing to (almost) 1, to the m best tokens? If it is possible (and it is for fairly large m's), what does it tell us about language models? If it hadn't been possible, why would that have been an issue, beyond very niche scenarios such as choosing a number or a US state at random?
- The discussion in section 6 could be more detailed. In particular, how do your results complement previously published papers that found that the softmax bottleneck *is* an issue for language models (e.g. Parthiban et al. 2021, Godey et al. 2024)?

### Questions
Suggestions rather than questions:
- Please double-check that the papers you cite as arXiv papers have not been published (by that I mean "really" published, in the proceedings of a conference or in a journal). Whenever it is the case, the proper publication should be cited, not the arXiv pre-print. There are multiple cases of this issue in your bibliography.
- Although the paper is generally well written, please refrain from using abbreviations such as "WLOG" (used twice).

### Soundness
3

### Presentation
3

### Contribution
2
