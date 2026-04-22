# Transducing Language Models

- Avg Score: 8.00
- Decision: Accept (Poster)
- Scores: 6, 8, 10, 8

## Abstract
Modern language models define distributions over strings, but downstream tasks often require different output formats. For instance, a model that generates byte-pair strings does not directly produce word-level predictions, and a DNA model does not directly produce amino-acid sequences. In such cases, a deterministic string-to-string transformation can convert the model's output to the desired form.
This is a familiar pattern in probability theory: applying a function $f$ to a random variable $X\sim p$ yields a transformed random variable $f(X)$ with an induced distribution. While such transformations are occasionally used in language modeling, prior work does not treat them as yielding new, fully functional language models. We formalize this perspective and introduce a general framework for language models derived from deterministic string-to-string transformations. We focus on transformations representable as finite-state transducers---a commonly used state-machine abstraction for efficient string-to-string mappings. We develop algorithms that compose a language model with an FST to *marginalize* over source strings mapping to a given target, propagating probabilities through the transducer without altering model parameters and enabling *conditioning* on transformed outputs. We present an exact algorithm, an efficient approximation, and a theoretical analysis. We conduct experiments in three domains: converting language models from tokens to bytes, from tokens to words, and from DNA to amino acids. These experiments demonstrate inference-time adaptation of pretrained language models to match application-specific output requirements.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents a general framework for transforming a language model over source strings into an LM over target strings, focusing on FST transformations. A major technical contribution is prefix decomposition of the preimage of a target prefix into a quotient and a remainder (Eq 6), which allows discussion of finite / infinite cases of quotient / remainder sets.

This paper also discusses exact and approximate algorithms for building the two sets.

As for experiments, this paper discusses 3 LM transduction applications: 1) token to character 2) token to word 3) DNA to Amino Acid.

### Strengths
- Clean formalization: the remainder-quotient decomposition allows quite general discussion of exact / approximate algorithms.
- Makes algorithmic contributions.
- Interesting applications that show effectiveness of transduced LMs.

### Weaknesses
- The formalism / accompanying algorithms / experiments seem to be designed specifically with FSTs in mind, which restricts its applicability. For example, Fig 3 (left) does not discuss the decidability of membership test in L10 and L13, which can actually be nontrivial in real LM usages (for example suppose you are trying to transform a code LM to a code-output LM). Decidability and complexity aspects should be discussed in a more rigorous fashion for the general case.
- Limiting ourselves to the FST-as-transformation special case, the paper would benefit from a more extensive comparison to existing NLP approaches that compose LM outputs with classical formalisms, such as [neural finite-state transducers](https://aclanthology.org/N19-1024/): the proposed method appears to be a special case of the neural finite-state transducers where the prefix probability can be seen as a pathsum of some machine that transduces the precover of y to y.
- Analysis of the pruning algorithm discussed in G.4 would strengthen the paper: e.g., can you bound the error?
- The paper would also be strengthened if it could discuss learning scenarios, for example if the source LM is parametric, what are the gradients of next-token probabilities under $p_Y$ ?

### Questions
- Can you provide error bounds for the approximate algorithms?
- Measure-theoretic foundations are implicit: could you spell out the underlying probability space / measurability assumptions of $p_Y$?

### Soundness
2

### Presentation
3

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
The paper studies how to compute or approximate target-prefix probabilities when a language model over one unit type (e.g., tokens, bytes, DNA) is passed through a deterministic mapping (finite-state transducer) into another unit type (e.g., characters, words, amino acids). Because the mapping can merge, split, or be context-dependent, a target prefix corresponds to a complex set of source prefixes.

Vieira et al. (2025a) handle the special case of strict-prefix-monotone mappings (e.g., tokens→characters), where target-prefix probabilities reduce to sums over source-prefix probabilities. This paper generalizes to arbitrary deterministic FSTs via a precover decomposition: split the relevant source prefixes into a quotient (prefixes whose outputs already force the target prefix) and a remainder (prefixes that must be summed explicitly).

For practicality, the authors add lazy determinization, memoization/precomputation universal states, and probability-mass pruning with a threshold $\tau$ (optionally with candidate-set caps and backtracking).

### Strengths
Clear, principled generalization of the strict-monotone case with an identity that is easy to implement (quotient + remainder). Practical algorithms with sensible speedups and demonstrated accuracy–speed trade-offs on non-trivial mappings. Overall the definitions, explanations, and conditions are crisp and easy to follow.

### Weaknesses
Key algorithms and experimental details are mostly in the appendix, and the approximations in 5.2 are described very briefly. Readers may miss core contributions if they don’t consult the appendix. Just a suggestion, feel free to ignore if it doesn't make sense: bring a minimal algorithm box/flowchart and a small table that lists each approximation knob (lazy determinization, memoization, pruning, caps) with default settings into the main text. 

The paper varies $\tau$ but does not isolate the contribution of each approximation, nor does it provide a direct runtime/memory comparison against Vieira in the shared setting.

### Questions
Could you toggle, one at a time, (a) lazy determinization, (b) memoization/precompute, (c) composition strategy, (d) adaptive $\tau$  candidate-set caps, and report the effect on JSD/accuracy, throughput, and peak memory ?

On the shared tokens→characters setup, can you report a matched comparison (same model, dataset, hardware) with throughput, wall-clock, and peak memory with Vieira2025a ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The authors put forward an algorithm to sample and compute language model probabilities over transduced strings from the model's own distribution.
They also propose an approximate algorithm that estimates the transduced probabilities at a cheaper computational cost.

### Strengths
The idea is creative, well motivated, theoretically interesting, and practically useful.
This is a classic application of algorithms, probability, and formal language theory.
The examples in the appendix are informative and give a good intuition for how the method works.

### Weaknesses
I am quite satisfied with the presentation and ideas in this paper and do not see any significant reasons why it should not be accepted.

### Questions
Does this method potentially allow conversion between different language model tokenizers? This could be useful for cross-family language model distillation.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper is about transduction -- taking a language model that was trained over a certain granularity of vocabulary and converting it to a model operating over a different granularity of vocabulary through a deterministic mapping. It extends prior work on converting subword models to character-level models by providing both exact and reasonable approximate algorithms for converting to arbitrary new vocabularies, then demonstrates this approach on three example transformations.

### Strengths
S1. I think this is a really interesting problem, and it's both well-explained and well-explored in this work. I can immediately see how this would be useful.

S2. I really like the framing of this with FSTs; it's an intuitive way to think about the class of transformations and a really nice formalism. The walkthrough of the math is detailed and cleanly developed.

### Weaknesses
W1. While overall I think the explanation is good, there are several parts of the paper that I think require context from "From Language Models over Tokens to Language Models over Characters" to fully understand. The two most obvious ones to me: (1) the discussion in lines 67-69 of applications to computational psycholinguistics and controlled generation; (2) the relation between transducing and token healing. 

W2. The work is largely a generalization of a prior (also cool!) result. I think this is still a valid contribution, but I could see the argument that this is a bit incremental. 

W3. While the speed of decoding is not a major focus of the work, it would be nice to see a bit more detail on the efficiency. In particular, bytes/sec is a bit hard to interpret, and a few baselines of the language model decoding in its original vocabulary would be helpful to put these numbers in context.

### Questions
Q1. This defines a fairly broad class of transductions that can be performed exactly; do you have any examples of reasonable-to-want transformations that *don't* fall into this class?

Q2. One thing that I don't see mentioned here is that this makes it much easier to compare distributions across language models-- you could transduce model 1 into model 2's vocabulary space. Do you think this would ever be practical to do e.g. for speculative or contrastive decoding? 

Comments/other notes:
- purely a stylistic gripe, but I think the formatting for library names (e.g. GenLM.bytes) is super distracting to read in-line. 
- The color of symbols is also slightly annoying to me, but I see the argument for that as a way to track notation; I would just confirm they are colorblind-friendly where possibly
- typo in line 144: "efficienct"
- line 41: "Such as normalizing output, ..." is a sentence fragment
- line 850: "proportion" -> "portion"
- the introduction was much less dense relative to the rest of the paper-- I think the example could have been greatly condensed to spend a bit more main-body text time on the transduction 
- the trained model for transducing seems a bit out of place in the rest of the work; it doesn't weaken the paper, but it feels a bit like a last-minute (unnecessary) addition to satisfy a reviewer

### Soundness
4

### Presentation
3

### Contribution
3
