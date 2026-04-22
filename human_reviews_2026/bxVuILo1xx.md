# Efficient Turing Machine Simulation with Transformers

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 10, 4, 8, 4

## Abstract
Constant bit-size Transformers are known to be Turing complete, but existing constructions require $\Omega(s(n))$ chain-of-thought (CoT) steps per simulated Turing machine (TM) step, leading to impractical reasoning lengths. In this paper, we significantly reduce this efficiency gap by proving that any $(t(n),s(n))$-bounded multi-tape TM can be simulated by a constant bit-size Transformer with an optimal $O(s(n))$-long context window and only $O(s(n)^c)$ CoT steps per TM step, where $c>0$ can be made arbitrarily small by letting the Transformers' head-layer product sufficiently large. In addition, our construction shows that sparse attention with fixed geometric offsets suffices for efficient universal computation. Our proof leverages multi-queue TMs as a bridge. The main technical novelty is a more efficient simulation of multi-tape TMs by synchronous multi-queue TMs, improving both time and space complexity under stricter model assumptions.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper presents a theoretical advance in understanding the computational efficiency of Transformers.
It proves that any $(t(n), s(n))$-bounded multi-tape Turing Machine can be simulated by a constant bit-size Transformer with an optimal $O(s(n))$ context window and only $O(s(n)^c)$ chain-of-thought (CoT) steps per simulated step, where $c > 0$ can be made arbitrarily small.
The work also introduces geometric-offset sparse attention, linking theoretical universality with practical efficiency.
The key technical innovation lies in efficiently simulating multi-tape TMs via synchronous multi-queue TMs, followed by a Transformer-based realization.

### Strengths
Improves efficiency bounds for Turing-complete Transformer simulations from $O(s(n))$ to $O(s(n)^c)$ for any small $c > 0$.

The synchronous multi-queue TM construction is original and cleverly designed.

The geometric-offset attention model aligns with sparse attention mechanisms used in practice (e.g., LogSparse, PowerAttention).

### Weaknesses
A small experimental validation or simulation would make the theoretical results more tangible.

Several sections (especially Step I and Appendix B) are heavy in notation and could benefit from clearer examples or figures.

### Questions
Could the per-step overhead be further reduced to $O(1)$, and what are the theoretical obstacles?

How might geometric-offset attention be implemented or approximated in modern Transformers with dynamic context lengths?

Could a simplified Transformer experiment empirically demonstrate the predicted efficiency?

### Soundness
4

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
5

### Summary
This is a theoretical work whose main contribution is the proof that any (t(n), s(n)) time-space bounded many-tape Turing Machine can be simulated by a constant bit-size Transformer with context window of optimal length and using O(s(n)^c), with any c>0, chain-of-thought (CoT) steps for a singe simulation step. This improves on previous results regarding efficient TM simulations by Transformers. To show this result, the authors first present a time-efficient simulation of k-tape Turing Machines using so-called “synchronous” queue TMs and then describe a simulation of multi-queue TMs by Transformers. The proposed simulation of TMs using synchronous queues seems to be the main technical achievement of this paper.

### Strengths
This work improves on previously presented results on efficient TM simulations by constant bit-size Transformers. In particular, it is shown how to reduce the number of CoT steps per simulated step from O(s(n)) to O(s(n)^c), for any constant c>0. The analysis is non-trivial and the result is in line with what is actively being researched by many authors studying Transformers through the lens of computational complexity.

### Weaknesses
Although the paper improves on known results, the work presents incremental progress in the field. In particular, as shown in Table 1, the submitted paper improves on the results of Li and Wang (2025) essentially only with respect to the number of CoT per single TM step. Importantly, the simulation results (presented in Table 1) do not take into account the total CoT length of the simulation. So, by simulating a (t(n), s(n)) time-space bounded multi-tape TMs with a single-tape that increases time but leaves space unchanged, we can conclude that the result given in the line “Li & Wang (2025)” holds true for multi-tape TM.

The main technical achievement of this paper seems to be the simulation of TMs by TMs using synchronous queues. But the idea behind this construction is not entirely new: as the authors note, the proof relies on the general idea of ​​Theorem 3.2 in (Hühne, 1993). The modification proposed in the submitted paper is somewhat non-obvious, but in my opinion it is not strong enough for ICLR.

It would be nice to have *precise* descriptions for the key notions used in the paper as, e.g., precision, CoT, Window, embedding dimension (Dim.).

### Questions
According to the proof, it seems that in the statement of Theorem 1 the total length of the CoT of the simulation should be $O(t(n) \cdot s(n)^{1/k'} \cdot s(n)^{6k/K})$, not $O(t(n) \cdot s(n)^{6k/K})$, where $k'$ is used in Theorem 2.

According to your description L. 182-186 a TM cannot empty the stack.

L. 310: what do you mean by "rotates the tape heads"?

Use uniformly throughout the document: stack or queue.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Summary: this paper presents a refined theoretical analysis of the ability of CoT transformers to simulate Turing machines, which has been studied in prior work. The paper shows that a multitape TM with time t(n) and space s(n) can be simulated by a constant bit-size transformer with context length O(s) and runtime O(t * s^c), for arbitrarily small c. Thus, constant bit-size transformers can quite effectively simulate transformers. The proof technique is interesting in its own right, going through a conversion of multitape Turing machines to multiqueue Turing machines.

### Strengths
1. I find the main result about the space requirements needed to simulate a Turing machine with constant-bit-size CoT transformers to be valuable.
2. The intermediate result converting multitape Turing machines to multiqueue Turing machines is interesting in its own right and technically innovative.
3. The high-level technical plan and proofs are clear and rigorous

### Weaknesses
### Make Dependence of Space/Context Window on k' Explicit

In theorem 2, how does the space O(s) depend on the queue factor k'? The way you are reducing time overhead is increasing k', so it would be nice to understand how space scales with this.

It would also be good to understand how this shows up in the main result about transformers: you say that we can make the time overhead arbitrarily small, but how does this increase the context window we need?

### Theorem 3 Suggestions

Overall, the theorem looks solid, but I have some minor suggestions for improvement:
- Clarify non-standard positional encodings in theorem statement
- Clarify use of hardmax in theorem statement; also it would be helpful to explicitly reference unique hard (UHAT) vs. averaging hard (AHAT) variants of hardmax. presumably your construction should work with either
- How uniform is this construction? (see below)

I understand constant bit-size (a la Li and Wang) to mean that the number of params and precision independent of context length. Does this imply that the transformer constructed in Theorem 3 is fully uniform, i.e., the parameters can't change at all with n? It would be helpful to explicitly mention the level of parameter uniformity that your construction attains (potentially in Table 1). E.g., the Merrill & Sabharwal's construction re-uses the same parameters for any n, so it is fully uniform, but this is not true for all constructions (e.g., Li et al., where position embeddings can evolve in a complicated way with n).

### Quadratic vs. Linear-Time Attention Discussion Deserves More Nuance

> This shows that the common argument “quadratic-time attention= ⇒fundamental throughput bottleneck” against Transformer-based AGI may not be a principled limitation but a byproduct of dense attention.

This sets up a bit of a strawman and makes a vague/strong claim against it. Can you clarify the argument in scare quotes (and attribute to someone) and clarify your counterargument. E.g., you're saying that expressive power does not require quadratic-time attention?

Moreover, while quadratic time attention is sometimes invoked as a drawback of attention, the related issue that matters more in practice (e.g., for GPU memory limitations) is the linear memory incurred by attention on long sequences to store the KV cache. This limitation is not overcome by your construction (and, in fact, is more fundamental), since for any function requiring s(n) = Omega(n), you will need to store linear memory to compute it.

### Questions
See "Weaknesses"

### Soundness
4

### Presentation
4

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
This paper contains a recent line of work providing Turing-completeness proofs for Transformers. This paper specifically provides a construction of constant bit-size (that is, constant number of parameters at constant precision) transformers with bounded (moving) context window and custom relative positional encoding vectors that simulates Turing machine computations with a smaller slowdown than in the closest prior work (Li&Wang 2025). That prior construction used \Omega(s(n)) CoT steps to simulate a single step of a Turing machine operating in space s(n). In contrast, the current paper cuts this down to O(s(n)^c) for arbitrarily small positive c, where larger transformers can achieve smaller c.

### Strengths
- Contributes to emerging theoretical understanding of the Turing completeness of Transformers.
- Improves over prior work by reducing the CoT overhead in a constant bit-size setting.

### Weaknesses
- The design of the model is nonstandard. In particular, adding relative positional encoding vectors in line 195 appears confusing: is the idea that the encoding vector pos(i-j) depends on both the current position j and a later position i from which an attention head looks back at position j? This seems to make both parallel training impossible and autoregressive decoding extremely inefficient, as the whole transformer activations would have to be recomputed throughout the entire context for every next-token generation?

- The construction assumes positional encodings depending on the space bound s(n), and thus does not inherently length-generalize over all n. Is it really fair then to say that the model has fixed bit-size?

Missing references:
- line 077: the claim about practical attention patterns appears speculative, especially the link to the geometric progression attention. Is there a reference for the claim?
- line 080: can the authors provide a citation for this "common argument"?

### Questions
Missing citations:
- Another relevant recent paper is [1], which provides another Turing completeness construction, and also proves optimality of CoT lengths.

[1] Amiri et al, Lower Bounds for Chain-of-Thought Reasoning in Hard-Attention Transformers, ICML 2025

### Soundness
3

### Presentation
3

### Contribution
2
