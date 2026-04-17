# Decomposing LLM Computation with Jets

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Large language models are becoming general knowledge engines for diverse applications. However, their computations are deeply entangled after training, resisting modularization which complicates interpretability, auditing, and long-term maintenance. We introduce Jet Expansions, a framework for expanding computational graphs using jet operators that generalize truncated Taylor series. Our method systematically decomposes language models into explicit input-to-output computational paths and complementary remainders. This functional decomposition provides a principled, knife-like operator for cutting through entanglement in LLMs, enabling scalable model inspection. We demonstrate how Jet Expansions ground and subsume the popular interpretability technique Logit Lens, reveal a (super-)exponential path structure with respect to recursive residual depth, and support several interpretability applications, including sketching a transformer language model with $n$-gram statistics extracted from its computations and indexing model toxicity levels *without* curated benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a new interpretability method for examining transformer-based language models based on jet expansions. The method works by approximating the full language model by its Taylor expansion and analyzing low-order terms. This extracts low-order behavior from the model and allows one to interpret it functionally, without relying on any additional data or training. The authors propose an implementation of the framework for a transformer-based language model and analyze its efficiency. 
Jet expansions subsume existing interpretability techniques such as logit lens and $n$-gram statistics-based interpretability methods. Thus, the authors use their method to analyze open-source language models and find interpretable results. For example, by comparing fine-tuned and non-fine-tuned models, they find that the fine-tuning did not affect certain parts of model behaviour.

### Strengths
- The proposed methodology seems broadly applicable and useful
	- I particularly like that the approach is dataset-free, which removes many possible confounders in the analysis
	- Similarly, I like that the method is more holistic than individual neurons, etc.
- The algorithm is thoroughly described and seems applicable and useful
	- It’s interesting that the analysis can even be done on CPUs
- I liked the thoroughness of the experiments, which cover many open-source models.
- Experiments clearly show that taking higher-order jets ($k = 1$ instead of $k = 0$ for logit lens) actually matters

### Weaknesses
- Unfortunately, I found it a bit hard to follow the transition between the empirical results and the theoretical setup. More explicitly stating what the measured quantities in the experiments correspond to in terms of the theory and the notation introduced above would be useful here, I think.
	- For example, how do $n$-gram statistic map to jets?
- It is unclear to me to what degree the results should be trusted as a truthful representation of the model. For example, without knowing anything in advance, is it not possible that most of the model behavior is only explained by higher order terms than those in the jet?
- The work mostly reframes existing techniques as jet expansions but doesn’t, to my understanding, propose novel applications of this framework for analyses that weren’t possible before.

### Questions
- Is there any way to understand how good the approximation is, i.e., how large the remainders are?
- How much of this methodology relies on the transformer architecture? Would it be easy to port to other neural LM architectures?
- Could the same methodology even be used for things like model compression (where only the jets are retained/used)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
- The authors introduce Jet Expansions, a framework for expanding computational graphs using jet operators that give interpretability to complex models (e.g., transformers).
- A jet expansion uses a transformer's residual structure to expand it to many smaller layer combination terms, plus some remainder.
- For interpretability, you look at the different orders of jets to understand what the model is trying to predict.

### Strengths
- The paper's main contribution, turning LLMs into jet-expanded paths, requires only model access and no extra data to probe models or compare two models with a shared vocabulary.
- The paper does a good job explaining the background and motivation, given the complexity of the work.

### Weaknesses
I question the practicality of jet expansions for interpretability. The paper only demonstrates the method on small to mid-sized open models, and even there, higher-order jets already add significant runtime. It's unclear how feasible this is for larger / frontier models.

### Questions
How large are remainders in practice, and can we be more explicit about their sizes?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Jet Expansions, a framework that expands computational graphs using jet operators (generalizing truncated Taylor series) to decompose transformers into input-output computational paths and complementary remainders. The authors demonstrate how this functional decomposition stimulates advances in the area of interpretability applications, such as grounding and subsuming the traditional interpretability techniques such as the Logit Lens technique and defending against jailbreak attacks.

### Strengths
1. The paper has an impressive theoretical foundation. They introduced the concept of jet operators from differential geometry to define interpretability rigorously, which is both novel and effective.

2. Using the novel framework, the paper elegantly unifies the framework and as a result, introduces a technique to subsume traditional foundations of interpretability.

3. Unlike most interpretability methods, jet expansions operate directly on model structure without requiring probe datasets. This is a significant practical advantage.

4. The topic has broad applications, including model safety and mechanistic analysis

### Weaknesses
My major concern is the remainder interpretation. The remainder $\delta$ seems to be a critical quantity to ensure performance and credibility, yet it is not analyzed rigorously enough. For instance, in Remark 2 on page 6, they mentioned "Empirically, remainders are often small and expansion logits nearly collinear with model outputs". From a theoretical point of view, how small is "small enough for the task"?

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors aim to solve the problem of "entangled" computation in LLMs, which makes their internal workings difficult to interpret. They use the core insight to treat LLMs as residual networks (a sum of paths) and apply jet operators, the functional equivalent of Taylor series, to analyze them. This approach is well-suited for the problem because jets are mathematically designed to handle compositions of functions and, via Lemma 1, can "disentangle" a nonlinear function of a sum into a weighted sum of its parts. This functional decomposition allows the authors to "carve out" and analyze specific computational paths. Their main contributions are the "jet lens," which generalizes the popular Logit Lens (showing it's just the $k=0$ case), and "jet n-grams," a novel "corpus-free" method to extract symbolic knowledge tables directly from the model's parameters. The authors provide strong empirical results, showing their decompositions are highly faithful (high cosine similarity) and that their $k>0$ jet lens provides a more stable interpretation for models like GPT-Neo where the standard Logit Lens has more chaotic results.

### Strengths
Principled Framework: The paper's primary strength is its formalism. It gives interpretability a more formal mathematical framework/tool grounded in approximation theory.

Generalization of Existing Tools: Proving that the Logit Lens is simply the $k=0$ case of the iterative jet lens is a great benfit. It places a popular tool on solid theoretical footing and simultaneously shows its limitations.

Empirical Support: The results strongly support the claims.

Faithfulness: The high cosine similarity (e.g., 0.993 in Figure 4) between the expansion logits and the model logits shows the decomposition is not just theoretical but empirically sound.

Superiority: The $k>0$ iterative lens provides a much more stable and interpretable computational trace for models like GPT-Neo, where the standard $k=0$ Logit Lens is more chaotic (as shown clearly in Appendix I, Figs. 8 vs. 9).

Applications & Insights:

RLHF & Toxicity: The finding in Table 4 provides strong, quantitative evidence that RLHF alignment masks toxic associations rather than erasing them (ToxiGen score drops to 0, but toxic bi-gram mass is unchanged). The method of demonstrating it via a "corpus-free" n-gram analysis is novel and powerful.

"Corpus-Free" n-grams: The idea of extracting symbolic n-gram tables without a dataset is a powerful one. The application of "diffing" models (Llama-2 vs. CodeLlama) by comparing their n-gram tables is a practical method for verifying fine-tuning.

Future Work: The framework opens up many exciting possibilities for "functional-level" interpretability.

### Weaknesses
Tractability and Clarity: The paper's discussion of the $O(2^L)$ exponential expansion (Algorithm 2) versus the practical, $O(L)$ linear-in-depth applications (the lenses) could be clearer. The tractability of these expansions, especially for $k>1$ is a major practical concern that is only briefly touched upon.

New Hyperparameters: The method introduces new hyperparameters for interpretability, namely the jet order $k$ and the jet weights $w$. The sensitivity of the results to these choices is not fully explored. Also how efficient and practical is it to get arbitrary order expansions?

### Questions
The RLHF & toxicity result supports some results from "Safety Alignment Should be Made More Than Just a Few Tokens Deep" that alignment usually only masks the problems. It would be interesting to see if the toxic bi-gram mass changed significantly in the "more robustly" tuned models from "Safety Alignment Should be Made More Than Just a Few Tokens Deep".

Cost of $k>0$ Lenses: Could the authors quantify the practical computational overhead of using the $k=1$ iterative jet lens (Fig. 9) versus the $k=0$ lens (Fig. 8)? How much more expensive is it, and does this limit its real-world applicability as a drop-in replacement?

Choice of $k$: The results for GPT-Neo are dramatically better for $k=1$, but the results for GPT-2-large (Figs. 11 vs. 12) seem comparable. Does this suggest $k=0$ is "good enough" for some model families? How should a practitioner choose the optimal $k$ for a new model?

Would the authors release the code?

Notation in Example 1: The expression $J^1\gamma(x_1)(x_2)$ is unclear. Could the authors confirm this is meant to be $J^1\gamma(x_1)(x_1+x_2)$, as I think it expands as $J^1 \gamma(x_1)(x_1+x_2) = \gamma(x_1) + \gamma'(x_1)((x_1+x_2) - x_1)$?

Lemma 1 Proof: In the proof of Lemma 1 (Appendix A), the evaluation point for the jets appears to be mislabeled as 'x' when it should be 'y' (e.g., $\sum w_i J^k f(x_i)(x)$). Can the authors confirm if this is a typo? Or clarify the difference between x and y?


"Data-Free" Clarification: The "data-free" claim for n-grams seems to be a source of confusion. The method is clearly input-dependent (it evaluates on token embeddings). Would "corpus-free" or "dataset-free" be a more accurate descriptor?

### Soundness
4

### Presentation
3

### Contribution
4
