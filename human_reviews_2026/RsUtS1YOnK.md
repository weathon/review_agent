# Differentiating without Partial Evaluation

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
In the physical sciences, the gradient of a model is often simplified into a
compact form ideal for a given context to be interpretable and more efficient;
in fact, sometimes the efficiency of evaluation can be improved by an
asymptotic factor due to symmetries. To learn interpretable surrogate models
that accelerate physics simulations, a differentiation system capable of
compact and unevaluated gradient expressions is highly desirable. However,
standard symbolic and algorithmic differentiation both start by partially
evaluating the model. After this points, the gradients irreversibly become
blackboxes with potentially obscure performance ceilings. Based on the
observation that composition is one of two combinators that form a complete
basis with captures, we compliment the chain rule with a second rule that enables
differentiation without any form of evaluation. Using a prototype
implementation, we obtain compact gradient expressions for an MLP and a common
physics model that, historically, resisted algorithmic differentiation. Lastly,
we discuss the theoretical and practical limitations of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes applying reverse mode automatic differentiation (AD) to functional programs expressed as combinators, rather than traditional methods applied to imperative programs or functional programs expressed in variants of the lambda calculus. The key contribution is deriving formulas to differentiate the B and C cominators, B denoting function composition and C denoting swapping the first two arguments of a curried function. With this, AD becomes syntactic and trivial: basis functions f are replaced with their reverse mode transformations f', B is replaced with its reverse mode transformation B', and C is replaced with its reverse mode transformation C'.

I haven't slogged through the math. But I will give the authors the benefit of the doubt. If there are errors, they must be minor and can be easily fixed as in principle it is possible to do what is claimed in the submission. But it is not necessary to do so, given my comments below.

### Strengths
The general idea of applying AD to combinators is sound and novel. The reverse mode transformation of B is well known and trivial. It is the chain rule. It has been presented in many places, among others in Pearlmutter & Siskind (TOPLAS 2008). The reverse mode transformation of C is novel but straightforward.

It would be great to see the general approach fleshed out to all combinators, a Turing complete set of combinators, or at least a more powerful set of combinators. This is not done in this paper. I encourage the authors to do so.

It would be great to see this turned into a practical and efficient AD system that is competitive with the likes of PyTorch and JAX, one that could generate efficient gradients of arbitrary code written in an inhabitable functional programming language that ran competitively on GPUs.

I encourage the authors to continue this line of work to flesh out the above.

### Weaknesses
The general claim that all other approaches to (reverse mode) AD require tracing/partial evaluation is false. Forward mode AD using dual numbers does not required tracing. Many classical AD systems, like Adifor and Tapenade do source-to-source transformation for forward and/or reverse mode AD without tracing or partial evaluation. This has been done at least since the JAKE system, Speelpenning (1980). Even for functional programming, Pearlmutter & Siskind (TOPLAS 2008) did this for the untyped lambda calculus. Many follow-on authors elaborated on this. These methods handle Turing complete languages. Since, one can formulate B and C as trivial lambda-calculus expressions, the results in this submission trivially follow from prior work.

The B and C combinators are not Turing complete. They are not even very powerful. You cannot write map and reduce in them. You cannot even swap other than the first two arguments of a function.

The key limitation of this work is that neither B nor C involve fanout. They are both linear operators. Reverse-mode AD is trivial for linear operators. Reverse-mode AD becomes difficult when there is fanout because you need to handle accumulation. Fanout is needed to handle combinators such as S and Y.

The appendix gives a method for transforming what the authors call stable iterate-to-fixed-point operators, i.e. ones that have a number of iterations are not dependent on floating point values There has been work on transforming general iterate-to-fixed-point operators.

### Questions
None

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper “Differentiating without Partial Evaluation” introduces an approach to symbolic differentiation that avoids the traditional reliance on partial evaluation, primarily through an alternative use of combinatory logic. The authors extend standard differentiation beyond the chain rule via a second rule derived from B/C combinators allowing gradients to be derived without any partial evaluation of the program.  The authors argue this avoids expression swell and keeps gradients symbolic until the very end, contrasting with Auto Diff systems that first evaluate or trace to a graph. This perspective on framing differentiation around a complete combinator basis and giving explicit pullback rules for B and C is an original synthesis in ML/AD literature. 
The paper provides a good survey of existing literature on this topic. 
The report is largely complete in scope for a theoretical proposal. It provides sufficient background, explains the motivation and theory behind the second differentiation rule, shows its application through concrete examples (MLP gradients, Hartree-Fock energy, conjugate gradient optimization), and discusses both theoretical and practical limitations. However, its primary implementation is a proof-of-concept within a domain-specific language (Julia), and lacks comprehensive benchmarks across mainstream tools or large-scale problems.  This omission of demonstration against large scale problems reduces confidence in the direct and broad use and benefit of this approach.  
In this paper, all symbolic differentiation is done at compile time, using pullback rules for the B and C combinators. This means that the differentiation process operates on program structure, not runtime values and relies on statically typed, dimensionally fixed programs, where all tensor shapes and types are known ahead of time.  This can be a limitation when dealing with NN and physical simulations.  The authors acknowledge this limitation.

### Strengths
Paper is well motivated and the developments are sound and well structured.  
The paper presents a  conceptual advancement completing the chain rule with a second rule to avoid partial evaluation; symbolic-first gradients.
The benefits of this approach include Interpretability & symmetry.  Gradients remain symbolic, enabling physics-aware simplification.
Illustrative examples across multiple domains such as NN, quantum chemistry tensors, and numerical linear algebra provide confidence.
Proofs provided in the appendix are adequate.

### Weaknesses
No performance evaluation: no runtime/memory or large-scale benchmarks vs. JAX/Enzyme/SymPy; toolchain impact remains unquantified (acknowledged by authors)
Practical limits: needs stable dimensions, lacks mutation support, and fixed-point/sequential iteration support is only sketched (appendix); these are crucial for many scientific codes.
Integration hurdles: static typing and compile-time differentiation requirement may clash with dynamic Python/Julia ecosystems; engineering path is non-trivial
Accessibility: combinatory-logic framing may be unusual for many ML/AD practitioners, increasing the learning curve.
The limitations related to 1. Dimensional Stability, 2. Fixed point iteration gaps, 3, lack of mutation support appear to be  a direct consequence of the framework that relies on pure symbolic diff (no tracing or evaluation) , while this provides certain benefits, in my opinion this also imposes restriction to purely functional, statically shaped, non-iterative programs.  In that I agree with  the authors that this work is paper is more a proof of concept of a symbolic calculus and not a production-ready AD method.

### Questions
Claim: “complete the chain rule” with a second rule based on B and C combinators.
1. Is this new rule provably complete for all differentiable compositions expressible in combinatory logic?
2. Can every standard AD operation be represented equivalently under your B/C pullback formulation?

How does the symbolic calculus relate formally to reverse-mode AD or the dual number formulation?
Can the framework be seen as a category-theoretic dual of reverse-mode AD (e.g., functorial composition)?


Given differentiation without partial evaluation, how does one ensure equivalence to the derivative of the evaluated program rather than the unevaluated syntax tree?

A formal criterion for when symbolic contraction (via delta identities) terminates in closed form would be useful.

Would integrating this calculus into a JIT or AOT compiler break referential transparency, and if so, how could this be mitigated?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposed a framework for programmatic transformation of mathematical codes to their derivatives. The main difference from other AD systems is that it adopts the tacit programming style (also referred to as point-free style), where the evaluation of variables (which can represent functions) is deferred to the very end. In contrast, in most AD, one must substitute the variables/symbols representing functions with concrete instances before tracing. This is possible as the proposed framework represents programs in a DSL based on combinatory logic. In combinatory logic, there are only combinatory terms (function primitives) and combinators (higher-order functions), and it is proven to be Turing-complete. Thus, a program can be written as a transformation of primitive functions under the combinators. Usual AD only defines the pullback on function primitives, but in this framework, the pullback for the combinators, which enable one to programtically differentiate combinatory logic. It is well known that the B and C combinator forms a complete basis, so one only needs to define the pullback (VJP) for B and C, which is provided in the paper. The paper also illustrates how a tensor program can be represented using combinatory logic. Finally, the paper demonstrates how the proposed system can be used to: (a) compute the gradient of MLP with respect to network parameters; (b) compute the HF gradient with symmetry; (c) derive conjugate gradient using the symbolic gradient output from the proposed system.

### Strengths
The paper provides an interesting alternative to the common AD system, where one can add in symbolic simplification rule easily (which is demonstrated in the HF gradient example).
The paper is well written and easy to follow, and is relatively self-contained.

### Weaknesses
My main concern is that the author failed to convey how the proposed B/C‑based differentiation differs, in capability and guarantees, from prior functional/program‑transform approaches to AD. All 3 examples demonstrated in the paper can be done with existing frameworks:
- MLP gradient: this is quite standard and can be computed in JAX/PyTorch easily
- Symmetry in HF gradient: one should note that this is a highly specialized application. I'd argue that using something like jax.custom_vjp suffices. Even within the proposed framework, one would first need to code the symmetry rule into the framework, so I can hardly see the benefit
- symbolic derivation of CG: this is neat, but mainstream CAS like Maple, Mathematica, Sympy, etc can also do this easily. 

Evidence on a curated suite of "hard" patterns that cannot be done within the existing AD/CAS framework would help.

The paper claims that it avoids expression swell, which symbolic AD is susceptible to. But the paper does not provide any concrete evidence against an existing symbolic AD system, nor does it provide a theoretical justification for the claim.

Also, the paper specifically states in the introduction section that the paper only makes qualitative claims. This actually weakens the potential impact of the paper, since without numerical evidence, the benefits of the proposed framework are at best speculative. There are no quantitative evaluations: no expression‑size statistics, compile times, runtime/peak‑memory comparisons vs. tracing AD (PyTorch/JAX), source transformation (Enzyme), or symbolic systems (SymPy/Mathematica). Even small benchmarks (MLP with/without accumulation; HF with symmetry folding; end‑to‑end cost of simplification + engine execution) would significantly strengthen the claims.

Finally, I would like to point out that the idea is not entirely new. The core contribution of the paper appears to be a point-free style symbolic differentiation technique that uses the pullback rule of the B and C combinator. I would like to point out the following prior works that partially encompass the main idea of the paper:
- point-free / higher-order AD: there are many such systems in the functional programming community. For example, Haskell's AD, and [The Differentiable Curry: https://openreview.net/pdf?id=ryxuz9SzDB].
- Using combinatory logics to formulate AD: see [Elsman et al. (2022), Combinatory Adjoints and Differentiation]
- B-rule: [Ehrhard & Regnier (2003), differential λ-calculus].
- C combinator swaps the order of currying, which is also known as the "flip" combinator. The C-rule in this paper is a direct consequence of the fact that the C combinator commutes with derivative operators, which is established much more rigorously in prior works, e.g., [Blute, Cockett & Seely, Cartesian Differential Categories] and this PhD thesis [https://cspages.ucalgary.ca/~robin/Theses/GallagherPhD.pdf].
- representing tensor as a function from index to value: this is a common practice, which is already adopted in TVM/TACO/Mathematica, etc.

### Questions
- It seems that it is also possible to define the pushforward for the B and C combinator. Have you thought about it?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In the present paper, the authors introduce an additional differentiation rule to complement the chain-rule used in automatic differentiation systems. Theoretically deriving its results, the application of the rule to skip the partial evaluation is demonstrated on a set of code examples.

### Strengths
The strength of this paper lies in its writeup, and the intellectual clarity with which its ideas are presented. The derivations are done with great care, and its chosen code examples help to present its ideas and aid in understanding the application of the differentiation rule.

### Weaknesses
The weaknesses of the paper are plentiful, distilling into high-level topics before diving into them individually:
- Misguided, or ill-informed key assumptions about automatic differentiation systems to motivate the current work
- The _complete_ lack of evaluation of the current work

#### Misguided Key Assumptions
- line 16-17 "gradients irreversibly become blackboxes", there exist a wide range of automatic differentiation systems in practice. The key difference being abstraction level chosen by the automatic differentiation system. As such this core claim of the motivation does not stand up to scrutiny. Especially for source transformation AD systems such as Tapenade [1], the gradients are emitted in the source language for the user and can be inspected. But even the big modern AD systems JAX, and PyTorch permit the emission of the produced gradients [2], which can then be inspected and are hence neither blackboxes, nor riddled with "obscure performance ceilings".
- The leveraging of symmetries, as envisioned by the authors, is something that is not impossible for modern automatic differentiation systems to perform. It is called custom rules. The mentioned Enzyme provides ample infrastructure for that, which can then leverage physical symmetries [3].
- Line 48, the PyTorch citation is pointed at Baydin et al. This is wrong. The citation should read Paszke et al [4].

#### Lack of Evaluation
- The presented produced? code is never executed and as such it is not possible to quantify the performance benefits the additional differentiation rule could provide. Especially the Hartree Fork system would naturally lend itself to such evaluation where one could for example evaluate
    - Existing AD systems (Enzyme, PyTorch/JAX, other Julia AD systems such as Zygote for example)
    - The prototypical implementation of yours
    - Prototype of yours leveraging the available symmetries
- The presented code examples are not in Julia, the language in which the prototype implementation of this work is implemented in (line 244f), as such it is not apparent to the reviewer whether the proof-of-concept is leveraged in this work. Even if the prototype is not able to perform the differentiation of the presented illustrative examples, I would at a minimum expect an evaluation on AD micro benchmarks. See e.g. the tests of Enzyme [5] for a large number of micro-micro examples one could leverage here.

References:
1. Laurent Hascoet and Valérie Pascual. 2013. The Tapenade automatic differentiation tool: Principles, model, and specification. ACM Trans. Math. Softw. 39, 3, Article 20 (April 2013), 43 pages. https://doi.org/10.1145/2450153.2450158
2. https://docs.pytorch.org/tutorials/intermediate/inductor_debug_cpu.html
3. https://enzyme.mit.edu/julia/stable/#Defining-rules
4. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. ArXiv, abs/1912.01703.
5. https://github.com/EnzymeAD/Enzyme/tree/main/enzyme/test/Enzyme/ReverseMode

### Questions
- I am a little confused by one of the key claims made by the abstract with relation to interpretable surrogates (line 12-14). While the ability to inspect the gradient flow is highly conducive to interpretable surrogates it is unclear to the reviewer how the present system provides any advantage here. As mentioned previously, Tapenade is able to emit the source code of the unevaluated gradient expression for inspection, and the reviewer would contend that the need for performance of the gradient expression supersedes the need for unevaluated gradient expressions (which can yet be inspected in all of the mentioned AD systems).
- It is entirely unclear to the reviewer whether the prototype implementation of the work is ever used throughout the paper? Would the authors be able to provide evidence that their prototype is actually functional?

### Soundness
1

### Presentation
2

### Contribution
1
