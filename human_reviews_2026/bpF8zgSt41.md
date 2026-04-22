# How Transformers Learn Causal Structures In-Context: Explainable Mechanism Meets Theoretical Guarantee

- Avg Score: 4.40
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4, 2

## Abstract
Transformers have demonstrated remarkable in-context learning abilities, adapting to new tasks from just a few examples without parameter updates. However, theoretical understanding of this phenomenon typically assumes fixed dependency structures, while real-world sequences exhibit flexible, context-dependent relationships. We address this gap by investigating whether transformers can learn causal structures -- the underlying dependencies between sequence elements -- directly from in-context examples. We propose a novel framework using Markov chains with randomly sampled causal dependencies, where transformers must infer which tokens depend on which predecessors to make accurate predictions. Our key contributions are threefold: (1) We prove that a two-layer transformer with relative positional embeddings can implement Bayesian Model Averaging (BMA), the optimal statistical algorithm for causal structure inference; (2) Through extensive experiments and parameter-level analysis, we demonstrate that transformers trained on this task approximate BMA, with attention patterns directly reflecting the inferred causal structures; (3) We provide information-theoretic guarantees showing how transformers recover causal dependencies and extend our analysis to continuous dynamical systems, revealing fundamental differences in representational requirements. Our findings bridge the gap between empirical observations of in-context learning and theoretical understanding, showing that transformers can perform sophisticated statistical inference over structural uncertainty.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates whether transformers can learn causal structures underlying sequential data. Using a framework based on Markov chains with randomly sampled causal dependencies, it shows that a two-layer transformer with relative positional embeddings (RPE) can exactly implement Bayesian Model Averaging (BMA). It further provides information-theoretic guarantees explaining how transformers recover causal structures.

### Strengths
This paper provides a rigorous theoretical construction showing that a two-layer transformer with RPE can provably implement BMA for causal structure inference. The information-theoretic analysis strengthens the understanding by proving that attention weights converge to the true parent structure under increasing in-context examples $L$, using mutual information arguments. So I think this paper make meaningful contribution.

### Weaknesses
While this paper is theoretically solid, some assumptions, such as linearity of the dynamics and  specific design of RPE, limit its generality. However, I think that these limitations are acceptable for the scope of this paper. Out of interest, I would like to ask the questions below.

### Questions
1. Extension to nonlinear dynamical system:
The current analysis focuses on linear dynamical systems, while transformer is a nonlinear mapping.
Do you expect that the this framework could be extended to the nonlinear dynamical system ? 

2. Essential role of RPE: The theoretical construction appears to rely on the use of RPE. Do you think other PE, such as absolute PE or learned PE, could, in principle, reproduce the same behaviors, or is RPE essential for causal structure inference?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a framework based on Markov Chains to explain how transformers create a causal dependency graph between tokens using ICL. In the Markov chain, each token depends on exactly one prior parent.
1. The paper shows that a disentangled transformer can be trained to do Bayesian Model Averaging.
2. Provides empirical evidence
3. Extends the framework to continuous case (Linear dynamical systems)
4. Provides a theoretical guarantee based on information theory for the transformer causal structure selection.

### Strengths
1. The paper makes a good connection between BMA and attention (for the disentangled transformer). 
2. Shows there exists a model which, by construction, implements BMA. Also supports the claim empirically. 
The core strength of the paper is that it helps in formalising how ICL works through a probabilistic framework.

### Weaknesses
1. As mentioned in the strengths, the formalisation of ICL that the paper brings is useful; however, the main weakness is that both the proofs and the experimental work are limited to a special form of the transformer. It is unclear how the results can be applied to the standard architecture.

2. Additionally, the paper claims that the proofs and experiments were conducted on a standard transformer. As far as I understand, they work on the disentangled transformer, which is not the same. Claims should be addressed to reflect the paper.

3. It seems that the paper is missing a conclusion.

### Questions
Do the authors have any insight into how their work can be extended to transformers with MLPs and multiple layers?
If not, maybe an experimental section on standard transformers to back the claims empirically?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper extends prior work on analyzing how transformers learn causal structure by proposing and analyzing a task which requires estimating and performing inference on an unknown causal graph from a number of in-context examples. The main theoretical result is a construction of a two layer disentangled transformer which can compute the posterior distribution over causal graphs (Theorem 1). It then uses experiments on disentangled transformers to support the claim that transformers learn a similar construction.

### Strengths
- The toy task is interesting and captures the idea of adapting to an unknown causal structure
- The paper is generally well written
- The authors show consistency of the construction (Theorem 2)

### Weaknesses
- Theorems 1 and 3 aren't mathematical theorems due to the use of $\approx$. For example for Theorem 1 I believe the intended theorem is something like there exists a sequence of weights $\theta_\beta$ such that $\lim_{\beta \to \infty} f_{\theta_\beta}(\ldots) = \pi(\ldots)$? Similarly, the mathematical statement of Proposition 2 is also unclear to me.
- As far as I can tell, the paper doesn't actually demonstrate weight-space agreement between their construction and the one learned by gradient descent (only the attention maps are verified). Is the challenge in matching the behaviors of the different heads? It's not clear at all to me from Figure 16 that the construction matches the one in Theorem 1.
- It would be good to run some experiments on a standard transformer with MLPs to check whether you learn a similar construction or get similar performance.

Minor points:
- There is a redundant prove in the abstract: "We **prove** that a two-layer transformer with relative position embeddings can **provably** implement"
- The figures in the appendix are very hard to read. For example, Figure 10 makes it look like every weight is identically 0 since the diagonal entries are barely visible. Perhaps putting the attention weights in log scale or reducing the size would help?
- Typo on line 268: "We set transformers has $K$..."
- I'm not sure what footnote 3 is saying?
- Proposition 2 is cited at Lemma 2 in Appendix B.5.

### Questions
- It seems that in addition to reducing the number of parameters, factoring the positional embeddings into L and H also simplifies the construction since it gives you parameter sharing between the in-context examples and the test-example for free. How do the experiments change if you use a more standard architecture?
- Is it possible to interpret the learned attention heads and show that it really does implement the same construction as Theorem 1?

### Soundness
3

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
The paper studies how transformers can learn causal structure in context. It designs a task where for each input sequence a causal graph is sampled and used to generate Markov chains (with fixed kernel). Given $L$ examples in context, the model has to learn to predict the continuation of the $L+1$-th sample. The paper shows that there exist a construction of a 2-layer transformer with $L$ attention heads in the first layer which can implement BMA to solve the task, and this is (approximately) learnt by the trained models. Finally, it studies the extension of this task and model to continuous data, i.e. a dynamical system.

### Strengths
- The paper studies a novel problem, i.e. how transformers can learn causal structures in-context, which continues an established line of work on understanding the internal functioning of transformers in simplified setups.

- The paper studies the proposed task both theoretically and empirically, showing that transformers implement BMA, and even provides an extension to the continuous case.

### Weaknesses
- The paper contains a lot of different parts, which makes it cluttered and none of them is presented too clearly. For examples, the construction of the transformer to implement BMA is not actually given, the figures are too small to be readable, there's no conclusion section. Moreover, the notation is in several places imprecise, e.g. in L292 $x$ is discussed but doesn't appear in the previous equation, in Eq. (6) $k'$ is not defined, etc., and the writing at times unclear. Overall, this makes the paper hard to follow.

- The construction of the transformer which implements BMA needs the number of heads to scale with the length input sequence, which I think it's a limitation and makes the construction even more distant from real-world models.

- While the paper provides several results, it's not clear what the takeaway message is. In fact, the paper shows that transformers can implement BMA to learn a causal structure in context, but this is shown by tailoring the architecture to a specific task (see point above). This construction doesn't seem to reveal some general mechanism applicable to other tasks, as it was for example with induction heads, which may limit the impact of this result on future work.

### Questions
See above.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates how transformers can learn and adapt to different causal structures in-context. The authors propose a framework using Markov chains with randomly sampled parent dependencies and prove that a two-layer transformer with some version of Relative Positional Embeddings (RPE) can implement Bayesian Model Averaging (BMA), the optimal algorithm for this inference task. The work provides a detailed theoretical construction, information-theoretic guarantees, and empirical validation showing that trained transformers learn to approximate this mechanism.

The paper's contribution lies in the connection to BMA and its 2-layer RPE construction. However, the manuscript suffers from a significant weakness in the way in which it acknowledges and contextualizes these contributions with respect to relevant prior works. Additionally, the theoretical construction relies on several specific assumptions that tailor the architecture to the specific task.

### Strengths
1.  **Principled Connection to Bayesian Inference:** The paper provides a link between the transformer's attention mechanism and Bayesian Model Averaging (BMA). 
2.  **2-Layer RPE Construction:** The explicit construction of a 2-layer transformer using a two-axis RPE to implement BMA reinforces the idea that architectural components can be mapped to algorithmic steps.
3.  **Information-Theoretic Guarantees:** The paper provides additional theoretical guarantees for causal structure recovery based on mutual information, strengthening the formal understanding of the learning dynamics.

### Weaknesses
1.  **Insufficient Acknowledgment of Prior Work:** The paper lacks a proper discussion of the related work  [1]. It does not properly contextualize its contributions, preventing a clear understanding of the paper's own contribution over prior works.
    - **Framing of Novelty:** The introduction frames the manuscript’s contribution as a departure from prior work limited to "fixed dependency structures" yet [1] seems to have already moved beyond this by introducing the same setup considered here based on in-context causal structure selection in Markov chains. The section "Our Approach", which conceptually outlines a setup closely related to [1] (each token depends exactly on one of the past tokens, with this dependence inferred in-context), omits any mention of this direct precedent work.
    - **Appendix Citation:** The citation in the appendix appears to report incorrect information. It claims that the verification in [1] is "limited to attention visualizations" however, [1] also provides quantitative validation using KL Divergence. It also criticizes [1] for "task-specific constructions," a trait shared by this work (see points below).
    - **Unacknowledged Theoretical Precedent:** the work presents in its Lemma 2, the central statistical argument proving the identifiability of the true causal parent. However, the result seems to closely parallel Lemma 2 in [1].  The present version is slightly stronger (the strict inequality) and presents a different proof, but leads to the same conclusion. This fact should be correctly acknowledged. 
    *   **Setups overlap** It would be helpful if the authors could acknowledge the conceptual overlap and similarities with [1]. The key difference seems to me to be that [1] infers a single global structure from within one sequence, while this work infers position-specific structures from across multiple sequences. This makes this work a valuable generalization and a genuine advance in the expressivity of the underlying graphs, but also requires the number of heads as well as the embedding to scale with the length of the examples (H). The two frameworks address the same underlying estimation problem: at a given position, identifying the correct parent accumulating log-likelihoods in a one-parent Markov process,  but differ in how the evidence is presented in the context (within-sequence vs. across-sequences). Acknowledging this would strengthen the paper by highlighting its novelty while giving proper credit.
    
2. **Dependence of the BMA result on architectural tailoring:** While the paper presents interesting ideas regarding linking in-context causal selection to Bayesian Model Averaging (BMA) in transformers. It remains unclear whether the demonstrated BMA implementation genuinely reflects what standard transformers learn and can implement or rather a consequence of the architectural choices, aligned with the task structure, made in the construction.

3.  **Different attention domains across layers (T×T then H×H)** Layer 1 attends over the full concatenated sequence (shape (T\times T) across examples×positions) while Layer 2 attends only across positions inside an example (shape (H\times H)). This effectively changes the logical index set the model operates on between layers, i.e. the model is built so different layers operate on different axes, tailored to the task they need to solve. It remains unclear whether a standard attention mechanism can discover a similar pattern.

4.  **Two-axis positional encoding (separate lookups for example-index and position-within-example)**
  The construction uses distinct positional biases for the example axis and the within-example position axis. This is a RPE that is tailored to the task and explicitly factors the task axes; it is not the standard single-axis absolute or relative positional scheme used in many transformer models. It remains unclear if replacing the two-axis RPE with a single standard positional encoding (absolute or standard RPE) the same BMA mechanism emerges or if it can even be represented.

5.  **First layer omits W_QK and W_OV.** The first attention layer is constructed to behave as a direct copier (fetching particular past tokens), which is commonly employed in the literature. However, the matrices acting over the semantics W_QK and W_OV seem to be omitted in the construction. It is acceptable for the purpose of the theoretical construction to fix these components (W_{QK} and W_{OV}​) to zero, but this design choice should be stated explicitly. Moreover, the paper should provide empirical evidence showing that allowing these parameters to be trainable does not substantially alter the main BMA mechanism or invalidate the proposed construction.

6.  **No positional encoding  for the second attention layer**. Similarly, the construction does not handle positional information consistently across layers: the second attention layer either omits or fixes to zero positional encodings compared to the first. This asymmetry effectively hardcodes aspects of the intended computation and departs from the symmetry typical of standard multi-layer transformers. As in the previous point, it would be important to assess whether including full positional biases in the second layer would materially change the mechanisms the model implements.


7. **Theorem 3** Theorem 3 leverages the same underlying statistical mechanism used in [2] (as acknowledged by the authors): it expresses the gradient signal in terms of a χ²–mutual information measure between each candidate parent and the child token, and then applies the Data Processing Inequality to show that this measure is maximized for the true parent. The contribution of Theorem 3 is, in my opinion, incremental: it adapts the same statistical idea to the specific transformer construction that exactly implements Bayesian Model Averaging. Moreover, the paper states that its proof “eliminates the stationary assumption of the data distribution and doesn’t require the Markov chain to be mixed,” but  Theorem 3 still relies on expectations such as $(\mathbb{E}[\pi(x_h \mid x_{h'}) / \mu(x_h)])$, which implicitly assume access to a stationary marginal distribution or i.i.d. sampling across examples. Without some ergodicity or mixing condition, it is unclear to me how these expectations are defined or estimated.

[1] D'Angelo Francesco, Francesco Croce, and Nicolas Flammarion. "Selective induction heads: How transformers select causal structures in context." In The Thirteenth International Conference on Learning Representations. 2025.

[2] Nichani Eshaan, Alex Damian, and Jason D. Lee. "How Transformers Learn Causal Structure with Gradient Descent." In Forty-first International Conference on Machine Learning. 2024.

### Questions
1.  **Unused Heads (Lines 294–295):** The authors state that "some heads... didn’t learn meaningful features." According to the construction, each head in the first layer should learn a specific copying mechanism. Does this observation imply the model was trained with more heads than theoretically necessary, or that some heads failed to specialize as expected?
2.  **Attention Notation (Lines 246–252):** The equations appear to use both a generic activation `σ(⋅)` and `softmax(⋅)` in the context of attention. Could the authors clarify the distinction or correct the notation if they are intended to be the same?
3.  **BMA Formulation (Lines 138–140):** For clarity, it would be helpful to explicitly write out the analytical expression for the likelihood `P(x | pa(h))` and the resulting posterior-predictive (BMA) distribution. This would make the target algorithm that the transformer is shown to implement immediately clear to the reader.
4.  **Theorem 3** Can the authors clarify under what distribution the expectations in Theorem 3 are taken, and how the result holds when the underlying Markov chain is non-stationary or non-mixing?
5. Could the authors clarify whether a transformer with conventional architecture and positional encodings (without separating L position and H position) would still approximate BMA under the same task formulation, or if these design constraints are essential for the claimed behavior?

### Soundness
3

### Presentation
2

### Contribution
1
