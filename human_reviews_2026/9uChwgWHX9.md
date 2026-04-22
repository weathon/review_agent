# Design Principles for Sequence Models via Coefficient Dynamics

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Deep sequence models, ranging from Transformers and State Space Models  (SSMs) to more recent approaches such as gated linear RNNs, fundamentally compute outputs as linear combinations of past value vectors. To draw insights and systematically compare such architectures, we develop a unified framework that makes this output operation explicit, by casting the linear combination coefficients as the outputs of autonomous linear dynamical systems driven by impulse inputs. This viewpoint, in spirit substantially different from approaches focusing on connecting linear RNNs with linear attention, reveals a common mathematical theme across diverse architectures and crucially captures softmax attention, on top of RNNs, SSMs, and related models. In contrast to new model proposals that are commonly evaluated on benchmarks, we derive design principles linking architectural choices to model properties. Thereby identifying tradeoffs between expressivity and efficient implementation, geometric constraints on input selectivity, and stability conditions for numerically stable training and information retention. By connecting several insights and observations from recent literature, the framework both explains empirical successes of recent designs and provides guiding principles for systematically designing new sequence model architectures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a unified theoretical framework for viewing sequence models (including Transformers, recurrent neural networks (RNNs), and state-space models (SSMs)) through the lens of **coefficient dynamics**.
The authors model the coefficients alpha in the linear combination as outputs of an autonomous linear dynamical system driven by *impulse inputs*.
They claim this representation reveals shared mathematical structure among diverse sequence architectures, enabling the derivation of six **“design principles”** related to:

1. Linear vs. nonlinear readout maps and their efficiency tradeoffs
2. Input selectivity through the geometry of zero set
3. Encoding positional information via non-identity evolution matrices
4. Structured choices of A_t matrix (e.g., diagonal or Householder)
5. Proper scaling of injection parameters (b_j)
6. Normalization factors to ensure stability.

Experiments on synthetic tasks (MAD benchmark) empirically test these principles, showing expected patterns such as improved selectivity with larger zero sets and the sufficiency of (A_t that is not equal to I) to encode position without embeddings.

### Strengths
* Clear and mathematically consistent formulation.
* Provides a clean pedagogical summary connecting RNNs, attention, and SSMs under one algebraic form.
* Well-presented with readable equations and illustrative diagrams.
* Serves as a potential tutorial reference for newcomers to the field.

### Weaknesses
* **No novel theoretical result:** All lemmas rederive existing intuitions without advancing formal understanding.
* **Weak experimental validation:** Evaluations on simple synthetic tasks (MAD) do not test scalability, language modeling, or real-world data.
* **Limited empirical novelty:** Trends (e.g., gating helps, non-identity (A_t) adds positional info) are already widely known.
* **Incomplete discussion of prior work:** The paper fails to acknowledge prior theoretical frameworks that make similar connections. No point is not already known. I don't see anything in this paper that is novel and that we did not already know from training deep sequence models.

### Questions
1. **Scope of derivations:** Are any of your truly lemmas new, or are they reinterpretations of existing SSM analyses?
2. **Experimental depth:** Can you test your design principles on real-world data (e.g., language modeling or speech) to demonstrate their usefulness beyond toy MAD benchmarks?
3. **Relation to Mamba and Selective SSMs:** Since these models already embody your principles (learned (A_t,b_j), stability constraints), how does your framework offer new insights beyond theirs?
4. **Extension to multi-layer architectures:** The study is single-layer. How do the principles scale in multi-layer or cross-attention settings?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a number of principles for designing sequence modeling layers. To come up with such proposals, the paper provides a survey of previous works, and mechanistically separate parts into A, b, \phi, \eta, and \alpha, highlighting major different parts. Coming up with six principles, the authors provide ablations for each, validating their claims.

### Strengths
The paper is well structured, proposing several ideas that not only help designing new architectures but also understand existing methods. The principles that are introduced in the paper are simple, thus can be easily adopted. Each proposal is backed up with an experiment.

### Weaknesses
My major concern of this paper is that the ideas are only validated using synthetic benchmarks as these benchmarks cannot model all real-world issues, and quite vulnerable to training setups such as weight decay and learning rates. Since the paper comes up with the principles that each should improve the model, I believe the authors should have used some real-world datasets (say language modeling) and benchmark at least their best model that all the principles are applied.

Additionally, I also have concerns with some principles. For example, Principle 1 and 5 are quite trivial which are already well known among the community, and Principle 6 is not specific enough (what specifically is an unstable A?). Also, I wonder if Principle 3 is actually correct: for instance, attention-based autoregressive models without positional embeddings (i.e., NoPE) has shown promising results.

### Questions
- What would be the model with all the principles applied, and how does it perform on language modeling?

- Is Principle 3 true?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a unified theoretical framework to describe sequence models including Transformer and RNNs, and puts forward a series of design principles and experimental verifications to guide sequence modeling.

### Strengths
1. The design principles for sequence modeling architecture proposed in this paper are instructive.
2. The analysis combining theory and experiment is convincing.

### Weaknesses
A unified sequence modeling framework for transformer and RNN model architectures has been mentioned in various works, such as MetaLA [1], PaTH Attention [2], and log-linear-attention [3], which may diminish the contribution of this paper. Therefore, further comparison and discussion with similar related works will help highlight the contribution of this paper.

[1] Yuhong Chou, et al. MetaLA: Unified Optimal Linear Approximation to Softmax Attention Map. NeurIPS, 2024.
[2] Songlin Yang, et al. PaTH Attention: Position Encoding via Accumulating Householder Transformations. NeurIPS, 2025.
[3] Han Guo, et al. Log-Linear Attention. arXiv, 2025.

### Questions
1. Based on the design principles of the sequence modeling architecture proposed in the paper, what characteristics should the optimal model architecture for sequence modeling?
2. Although these design principles can guide the design of the optimal sequence modeling architecture, for linear RNN architectures, the design of some components may not be conducive to hardware efficient parallel training of RNN. In this case, would a more general and expressive matrix gate be less useful than a simpler, more efficiently trained diagonal/scaler gate?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
1. The paper proposes a unified theoretical framework (Coefficient Dynamics) for analyzing sequence models, including Transformers, linear attention, and State Space Models (SSMs). The framework formalizes the fact that these models compute outputs as linear combinations of past tokens, and interprets the outputs of *linear dynamical systems driven by impulse inputs*.

2. Unlike prior unification approaches, this formulation explicitly introduces a per-token index, j, representing each previous key/value, conceptually similar to a KV-cache, and shows Transformers, SSMs, Linear Attention-like architectures can be expressed as special cases.


Building on this framework, the authors derive a set of design principles:

  1. **Linearity of $\phi$:** Only linear readout maps permit parallel recurrent computation.
  2. **Input selectivity:** Nonlinear $\phi$ enables sparse coefficients aka selectivity.
  3. **Positional information:** Setting $A_t \neq I$ embeds position into coefficients; $A_t = I$ requires positional embeddings.
  4. **Evolution structure:** The choice of $A_t$ (scalar, diagonal, Householder) allows for transformations such as scaling or rotation.
  5. **Scaling of $b_j$:** Proper scaling of $b_j$, prevents variance blow-up with increasing hidden state.
  6. **Normalization $\eta_i$:** Stable training requires normalization of coefficient magnitudes.

### Strengths
1. The paper is clearly written and the mathematical formulation is rigorous.
2. Unifies existing insights on SSMs, RNNs, and attention into a single framework summarizing core principles like linearity, efficiency, input selectivity, normalization, and stability. This is pedagogically useful for newcomers to the field.

### Weaknesses
### **On the framework**

The “coefficient dynamics” construction, builds on standard frameworks like Dynamical Systems Framework with the new ingredient being the explicit *per token j index*, which is equivalent to maintaining a key–value (KV) cache. In my opinion this viewpoint is not novel in a theoretical sense: prior work (e.g., Dao & Gu, 2024; Sieber et al., 2024) already expresses attention and SSMs as linear recurrences or matrix multiplications over past states. The main change, vis-a-vis previous works, is the KV cache-like formulation to unify attention without using infinite state sizes. 

---

### **On Principle 1 ($\phi$ must be linear for parallelization)**

In my opinion, this is a well-known result in the subquadratic. Specifically, it is known that only linear readouts permit associative-scan like formulations required for efficient recurrent computation. This principle has been used in multiple works (Linear Attention, Mamba, Mamba-2, Gated Deltanet). While the lemma is correct, its inclusion as a “new principle” adds little beyond reiterating that *linear functions yield linear complexity*.

---

### **On Principle 2 (Input selectivity and geometry of $\phi$)**

The main idea—that nonlinear ϕ enables suppression of uninformative tokens while linear ϕ limits selectivity—is sound and nicely presented. However, this observation is well known and connects to classic results on **associative memory capacity**, where linear associative memories can store only ~n patterns in dimension n. 

As a nit remark: the follow-up discussion "Can learnable parameters save us?" is technically correct but does not logically follow from the principle, since modifying $A_t$ or $b_j$ changes only the key–value dynamics, not the query-dependent coefficients $\alpha_{ij}$.

Remark: Authors claim that Linear Attention has a readout of the form $\psi(\cdot)\psi(\cdot)$, but this is not really a function acting post the readout as defined and hence does not fit with the framework definition. It is better viewed as a preprocessing trick rather than a part of the framework.

---

### **On Principle 3 (Positional Information)**

The result that $A_t = I$, which implies that per-query token the sequence mixing process is permutation invariant and hence requires position embeddings is also well known and has been discussed in prior works. Authors of Mamba mention that due to the decay, the operation is no longer permutation invariant and hence does not require position embeddings. In my opinion, the lemma correctly states this but adds no new insight.

---

### **On Principle 4 (Structure of $A_t$)**

The fact that the structure of the state transition matrix $A_t$ (scalar, diagonal, Householder) limits the operation that can be performed on the keys is a tautological statement for me as the state transition matrix is what acts on the keys to produce the output. In the Lemma, authors simple summarize the actions performed by scalar, diagonal, or Householder matrix operators on the keys being acted upon.

---

### **On Principles 5 & 6 (Scaling and normalization)**

The discussion on $b_j = O(1/\sqrt{n})$ to maintain $O(1)$ variance and the normalization of coefficients $\alpha_{ij}$ to avoid exploding norms repeats standard initialization theory (Glorot & Bengio, 2010; Vaswani et al., 2017). While correctly stated, these are rules of thumbs which are widely used in architecture design to ensure that variance remains bounded in deep models. In my opinion, their inclusion as novel “principles” is overstated.

---

My overall assessment of novelty is that the paper’s strength lies in gathering well-known rules of thumbs under a shared formalism, but every individual principle has been studied or applied before.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2
