# Language Identification in the Limit with Computational Trace

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 4

## Abstract
Training on Chain-of-Thought (CoT) traces has empirically shown to dramatically improve the capabilities of Large Language Models (LLMs), yet a formal understanding of its power remains limited. 
In this work, we investigate the role of training on such computational traces from the perspective of language learnability. We introduce a new learning model, identification in the limit with trace, which augments Gold's classic paradigm [Gold'67] by providing the learner not only with examples from a target language but also with computational traces from the machine that accepts them. 
Our results reveal that access to these traces dramatically enhances the power of the learner. We first prove that with perfect computational traces, the class of all computable languages (those recognizable by Turing Machines) becomes identifiable in the limit. This stands in sharp contrast to Gold's famous impossibility result, which holds even for the simple class of languages that are recognizable by deterministic finite automata.
We then analyze the more challenging scenario where the learner has only partial information regarding the computational traces, which are also subject to adversarial corruptions. In this setting, we establish a set of trichotomic results on the amount of error that can be tolerated for the successful identification of language classes across the Chomsky hierarchy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper extends Gold’s seminal formal language learning framework from learning only based on positive examples to learning based on positive examples and computational traces associated with them. This is relevant to modern language models as chain-of-thought traces have been theoretically linked to such execution traces, making the setting useful to study learnability of algorithms.
The authors find that, in stark contrast to Gold’s result, traces make large classes of languages, including all recursively-enumerable languages, learnable. The authors then study learnability under corrupted traces, and find that it makes the learning problem markedly harder; while regular languages remain learnable under a constant fraction of errors, context-free languages and Turing machines require much stricter restrictions on the corruption.

### Strengths
- I believe the paper studies a very interesting and useful problem and puts a new spin on an old/classic setting 
	- In particular, it provides another possible contributing factor behind how CoT helps improve models
- The exposition and motivation are clear; it’s easy to discern what the paper’s contributions are, andthe  methodology was used/developed
	- For example, the proofs are first well-described on simpler models (finite-state automata)
	- The related work section is thorough and useful

### Weaknesses
- Although this is not a major drawback, I feel like the connection to generation in the limit, which first appears in the Introduction, is not really justified; I’m not sure how learning with traces is any more connected to generation in the limit than the original learning in the limit setting
- Very minor, and I don’t think this undermines the theoretical contributions: It is slightly unclear how the results translate into practice; maybe at least describing how this could be used or tested in practice could be useful

### Questions
- As you mention, the results are asymptotic in nature, which is okay. I was just wondering if you have any ideas for next steps, i.e., how one would proceed/extend the results to some complexity bounds? I imagine the methodology would have to be quite different.
- Can you elaborate on the connection to generation in the limit? It seems like, since there are fewer impossibility results there, traces would not be as useful?

### Soundness
3

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
This paper proves theoretical results concerning language identification in the limit in the case when the learner is given an enumeration of positive strings plus the computational traces that demonstrate that an automaton for the language to be identified accepts each string (e.g., a sequence of states in a finite automaton). The main result of the paper is this: unlike the classical case consisting only of positive strings  without computational traces, in which case most interesting language classes are not identifiable, the class of computable languages is identifiable in the limit when computational traces are provided. The authors then prove results for cases when the computational traces contain a certain number of errors for the classes of regular languages, deterministic context-free languages, and computable languages.

### Strengths
Overall, this is an interesting paper with significant, original theoretical results. As the authors point out, these results are relevant to the use of CoT to train LLMs. The paper is written clearly and does a good job of contextualizing itself amid prior work. The robustness results are quite interesting.

### Weaknesses
The paper would benefit from some clarifications; see my comments in the Questions section.

1. A minor point, but I would point out in the abstract that you are assuming that the learner only has access to positive examples, not negative examples.
1. 072: DPDAs correspond to DCFLs, not CFLs.
1. 183: The definition of DPDA is missing constraints on $\delta$ that make it deterministic. I think you need to allow $\varepsilon$ as the popped symbol too.

### Questions
1. 088: Do you mean that the number of errors per trace is $O(1)$ with respect to length, not finite?
1. Fact 2.2: In this example, what is the alphabet $\Sigma$?
1. Theorem 3.1: Do we assume all the TMs are deciders?
1. 310: Does this work if the alphabet is not fixed to {0, 1} ahead of time?
1. 380: How is it possible for $m(x)$ not to be 0? $U$ is already the set of all states that occur in all traces. Can different instances of the traces for the same $x$ be edited in different ways, or are they always consistent?
1. 388: This statement doesn't make sense to me. Do you mean "not every accepting state is from the set $U$"?
1. Def 6: What about non-scanning transitions?

Typos:
1. 267: the the
1. 371: set states -> set of states

### Soundness
3

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
2

### Summary
An existing result shows that only a very small set of languages can be identified in the limit (i.e., eventually the identifier produces the same prediction forever) from exclusively positive examples provided. The work in this paper demonstrates that a larger class, all Turing Complete languages, can be recognized in the limit when provided with an exact computational trace (the state of one machine which recognizes the language at every step of its computation). Additionally, it demonstrates that this is possible even if there are errors in the trace, with different bounds on the error rates depending on the class of languages discovered.

### Strengths
While I did not precisely check every single aspect of the theorems, what I did check appears to be completely accurate, and the theorems are quite interesting in their results.

The fact that the computational trace enables a lot more identifiability is interesting

The text is written quite clearly

### Weaknesses
This is primarily a framing issue, but I don't really see the relationship between this work and chain of thought in particular, it seems to be mostly about utilizing information about intermediate states in computational models in order to theoretically learn languages in an unbounded computational setting (with no limits on time to process each sample or the number of samples). In practice, chain of thought uses a very small number of examples and an extremely bounded computation.

Minor errors/suggestions:

You should emphasize early on that “constant number of errors" means a constant per trace, not a constant overall.

Revisiting the regular language example with traces might be helpful, it made the utility of the trace more obviously useful when I went through the example with a trace and realized that the main thing it provides is a distinction between a model that accepts everything (single accept state) and a model that accepts N things (many states).

### Questions
The algorithm for identifying robustly only seems to me to use the traces solely to identify the number of states. Is this accurate? If so, it should be explicitly stated in the text, as this sounds like a much weaker assumption than having access to the full traces with errors. If not, the additional information gained should be discussed.

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
This paper studies the benefit of CoT from the perspective of language identification in the limit with computation. While Gold proved the impossibility of recognizing most interesting language classes without computational traces, the authors show that as long as each $L\in\mathcal{L}$ is recognizable by some $M\in\mathcal{M}$, the class $\mathcal{L}$ is identifiable by $\mathcal{M}$ in the limit if computational traces are available. Furthermore, the authors consider robust language identification, concluding that identification is achievable with finite error, but robust language identification remains impossible even under diminishing error.

### Strengths
1. This work offers an interesting TCS perspective on the role of CoT, connecting the LLM phenomenon with the theory of language identification in the limit.

2. The results on robust identification provide valuable insights into the significance of CoT quality.

### Weaknesses
1. The paper lacks intuitive explanations regarding how CoT contributes to identification.

2. There appear to be conceptual gaps between the theoretical model and realistic CoT. For example, the paper’s model consider enumeration over a language, whereas real-world CoT operates on individual instances (i.e., strings $x\in L$). Moreover, while the robust identification results highlight sensitivity to noise, empirical studies suggest that other factors, such as CoT format or length, may outweigh the correctness of CoT.

### Questions
See Weaknesses

### Soundness
3

### Presentation
2

### Contribution
3
