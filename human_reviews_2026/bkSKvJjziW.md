# Measuring Scarcity–Complexity Collision in Language Model Estimation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 8, 2

## Abstract
Formal languages are increasingly used to analyze limitations of language–model architectures, via properties of their defining automata (e.g., number of states, transition weights, or out-degree at a state). Understanding why a neural language model struggles to learn a given property requires separating two cases: **scarcity**---the property is underrepresented in the data due to low likelihood---versus **complexity**---the task is more sample-demanding for the chosen learner. In the former case, increasing or resampling training data can help; in the latter, changes to the architecture or training may be needed. Evaluating a property “on its own” typically involves marginalizing over a family of languages that exhibit it, which can introduce selection bias: some family members may be systematically overrepresented among samples with the property, confounding correlational analyses. Indeed, most existing investigations relate corpus statistics to performance, but such correlations do not identify causal effects. We introduce a causal framework for assessing learnability in probabilistic formal languages. Using an efficient, controlled sampling procedure for regular languages, we *intervene on event frequencies by replacing the unconstrained sampling policy with a controlled one*, leaving the automaton’s topology and weights unchanged, to test whether a property looks difficult because it is rare or because it is genuinely demanding for the learner. Our main contribution is theoretical: the framework and controlled sampling procedures. We illustrate these in three example case studies with LSTM and Transformer learners, where conclusions under correlational evaluation can invert once causal frequency interventions are applied.n can invert once causal frequency interventions are applied.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the limitations of learning in neural language
models from two perspectives: scarcity and complexity.  While the
limitations caused by scarcity can be mitigated by increasing
sampling, those arising from complexity cannot.  Using Pearl’s causal
framework of do-calculus, the authors propose a scarcity measure,
defined as a causal quantity over event-occurrence distributions on a
probabilistic finite-state automaton (PFSA).  In other words, rather
than simply counting how rare an element (such as a state, transition,
or symbol) is, the measure captures the effect of scarcity by
observing how the model’s performance changes when the occurrence
frequency of that element is intervened upon in the sense of a
do-operation.


In the experimental section, the authors apply this framework to
controlled language-learning tasks implemented on probabilistic
finite-state automata.  They train neural sequence models on data
sampled both observationally (following the natural frequency
distribution) and interventionally (where event frequencies are
artificially balanced).  By comparing the resulting state-wise
decomposed KL divergence, the study demonstrates that some states
remain difficult to learn even when their frequency is increased,
revealing structural or complexity-driven limitations.  This provides
empirical evidence that the proposed scarcity measure successfully
distinguishes between performance drops caused by data scarcity and
those stemming from intrinsic structural complexity.

### Strengths
1.  The paper makes a rigorous and original distinction between two often conflated sources of learning difficulty between 
(a) scarcity — limitations due to insufficient sampling of certain events, and
(b) complexity — limitations intrinsic to the structural or grammatical properties of the system.
This conceptual clarity is a contribution to the theory of learning and generalization.

2.  By grounding the definition of scarcity in Pearl’s causal framework (via do-calculus), the authors move beyond conventional frequency-based or statistical interpretations. Defining scarcity as a causal quantity — i.e., the performance change under an intervention on event frequencies — is both elegant and generalizable. This gives the notion of “data scarcity” a principled mathematical meaning. 


3.  The use of probabilistic finite-state automata (PFSA) as a controlled experimental environment is a major methodological strength.
It allows the authors to isolate structural dependencies and control for event frequencies in a way that is impossible with natural-language corpora. The PFSA setup serves as a minimal model of linguistic learning, bridging formal language theory and neural estimation.


4.  The comparison between observational and interventional training distributions provides intuitive yet rigorous evidence.
The results (e.g., decomposed KL divergence across states) demonstrate when poor learning performance is due to low frequency versus when it arises from structural complexity.
This empirical clarity strengthens the theoretical claims.


5.  The framework establishes a foundation for future studies on causal interpretability in machine learning.
By quantifying how much of a model’s limitation is attributable to data availability versus intrinsic structure, it contributes to ongoing debates about scaling, generalization, and data efficiency in neural models.

### Weaknesses
1. The empirical validation is conducted only on synthetic data generated
by probabilistic finite-state automata (PFSA).
While this controlled setup isolates causal effects cleanly, it leaves
open whether the proposed scarcity measure generalizes to
real-world learning tasks.
No results are provided for larger neural architectures or open-domain data. 


2. The causal definition of scarcity, though elegant, remains
mathematically abstract and difficult to operationalize beyond the
PFSA framework.
The paper does not fully specify how the intervention on event
frequencies could be implemented for arbitrary datasets or continuous
domains.


3. Complexity is treated largely as structural difficulty in the
automaton (topological or dependency-based).
However, other important aspects of complexity in real world data
are not captured in this formulation. It goes largely beyond PFSA. 


4. The causal estimation procedure requires retraining models under multiple interventions (different sampling distributions), which can be computationally expensive.
The feasibility of this approach for large-scale neural systems remains untested.


5. The paper positions itself mostly within formal and causal theory, but
provides limited comparison to prior empirical findings on data
imbalance, curriculum learning, or frequency effects.
As a result, its impact on practical neural-language modeling remains
somewhat speculative.

### Questions
1.  This framework is limited to probabilistic finite-state automata
(PFSA).  The significance of neural networks lies in their ability to
learn patterns that go beyond PFSA, encompassing higher levels of the
Chomsky hierarchy.  How robust is the causal definition of scarcity
when extended beyond PFSA formalisms?


2. Can the scarcity measure be generalized to non-discrete settings?


3. How interpretable is the scarcity measure numerically?

4. Can this framework inform training strategies?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a causal framework to disentangle scarcity (low frequency of a linguistic pattern in training data) from complexity (inherent difficulty of learning the pattern for a given architecture) as causes for poor language model performance. The authors use Probabilistic Finite-State Automata (PFSAs) as a controlled testbed. Their main contributions are: 1) A formal causal model for LM learnability based on Pearl's do-calculus, and 2) An efficient "binning semiring" algorithm for sampling data from PFSAs under interventions that fix the frequency of specific events (symbols, states, transitions). They illustrate the framework with case studies on LSTMs and Transformers, showing that correlational analyses can be misleading and that interventional estimates can invert conclusions about what is hard to learn.

### Strengths
- The authors go to great length to carefully disentangle what they identify as two confounders in the learning difficulties of transformers, scarcity and complexity.
- The ability to causally attribute LM failures to data or architecture is significant. It provides a more principled methodology for failure analysis and dataset design in synthetic settings, moving beyond correlational claims.

### Weaknesses
- **Clarity**: Speaking as a theorist myself, I found that the formalism, particularly in Sections 4 and the appendix, is heavy. The connection between the complex binning semiring machinery and the high-level goal of "making rare events less rare" feels disproportionate. This complexity obscures the intuition and makes the paper difficult to digest. In some places, the writing appeared convoluted; for example, the caption for Fig. 4 refers to "Monte Carlo estimates" - did the authors simply sample several data sets and train a couple of models on them, or was there something more going on?

- I also had some difficulty with some **conceptual issues**. The task that the authors actually focus on is essentially the parity task, which is a nice way of testing the ability of a network to do something akin to counting, a well-known issue for transformers. Now if you try to learn the parity function (or any function) and you are in the right function class (i.e. your architecture is expressive enough to even represent the task), the fundamental difficulty of a task that remains is its sample complexity -- how many samples will I need to learn the function? Parity is a popular test bed for computational complexity because it is hard to learn, as we know since Minsky & Papert in the 1960s; more recently, arXiv:2207.08799 by Boaz Barak et al. gave a nice discussion. In the present paper, the authors seem to distinguish scarcity (i.e. a training set below the required sample complexity) from "complexity", but I am not sure what complexity means here? The authors seem to intend it to mean some inherent difficulty for certain architectures - so that would be simply a higher sample complexity? (In fact, since transformers do not have a hidden state, contrary to LSTMs, I would expect them to have a higher sample complexity to learn the parity)

### Questions
- Please clarify the distinction between complexity and scarcity (see my point above) 

- Relation to other hard to learn functions: How does your work compare to the setting where we try to learn a sparse parity function? What does the formulation via formal languages add? Could you apply your causal machinery to that case? 

- Intervention Specifics: In the case studies, when you perform an intervention like "symbol a occurs exactly N times," how is this intervention not also changing the language that you are learning? You introduce a mismatch between training and test distribution here. I understand that this is for the benefit of your causal intervention, but I think it's worth discussing how that impacts the language from which you are now effectively sampling the training set.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work contributes a causal framework that evaluates the learnability of probabilistic formal languages. The core operation is a controlled sampling procedure that intervenes on the target event frequency. By controlling frequency, one can decouple two factors that contribute to the difficulty of learning a property: scarcity and complexity, which are hard to disentangle in an observational setting. The authors demonstrate the effects of the proposed framework through three case studies that analyze the learnability differences between the Transforms and LSTM architectures.

### Strengths
* This work proposes a rigorous causal framework that decouples scarcity and complexity through frequency interventions, allowing more fine-grained analysis on learnability. This could be a powerful analysis tool to better understand data and architecture biases.
* This work is well written. In particular, the problem and the motivation are clearly stated.
* Originality: As I do not work on formal languages, I am not familiar with the literature in this area, but this could potentially be a novel application of causal tools in analyzing LMs with formal languages.

### Weaknesses
* Based on Figure 2 and Figure 4, IIUC, while for a given architecture, the intervention results generally differ from observational results, the relative differences between the architectures seem relatively consistent across methods, i.e., for someone only interested in comparing the learnability of an architecture, comparing based on observational results would mostly yield same results as comparing based on intervention results. What are some cases where the learnability analysis would yield different results from the two methods? 

* The work focuses on discussing one type of property P, i.e., strings that traverse a set of transitions exactly N times, what are some alternative properties that might be useful, and how would they change the sampling process?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This work identifies a key issues with evaluating capabilities in LMs: if a model is unable to perform some task, this may either be caused by the task being (a) low frequency (scarce) in the pre-training data, or (b) computationally complex, e.g. an NP-complete problem vs. a problem that can be solved by a simple DFA. The authors propose using formal languages as a domain to study this, since data frequency and computational complexity can be formally measured and systematically varied. They develop a method for generating formal language data which holds one of these factors - scarcity or complexity - fixed, while varying the other. They contextualize this work in a causal inference framework, contrasted with prior correlational work which confounds scarcity with complexity. With three empirical case studies using toy examples, the authors show cases where their causal framework makes opposite predictions compared with a correlational analysis.

I find the authors motivation and setup to be compelling. However, I found the writing and presentation to be hard to comprehend, and I had trouble making sense of how the authors results support their key points. The writing seems to alternate sharply between simple and straightforward (although a bit redundant), especially with the main points about scarcity vs. complexity, or dense and hard to understand with the methods and results. I think the writing and presentation could be improved significantly to smooth these transitions, and more clearly connect the authors technical contributions with their main arguments. If the presentation was improved to that I could better understand - at least to some extent (see following point) - specifics of the methods, results, and how the results support the authors' claims, I would raise my score at least to a 4.

I had trouble understanding section 4 due to my lacking relevant technical background, and so I am unable to provide feedback on the correctness of section 4 or the significance of this contribution.

### Strengths
- The scarcity-complexity problem is interesting and I haven't seen a paper which explicitly tries to study this point. I also appreciate the contrast with prior works with correlational analysis.
- The domain of formal languages seems like a good fit for systematically studying this problem.
- The authors' proposed method of generating data which holds either scarcity or complexity fixed while varying the other property, seems like an apt tool for studying this problem.

### Weaknesses
- I found the writing and presentation of their framework in sections 2 and 3 to be difficult to understand.
	- I could comprehend the key broader points, but I found the writing to be lacking in terms of connecting key ideas with the authors' math.
	- In Fig. 2, what should I be taking away from this figure? Why does it matter that there is alignment for Odd but not Free or Even? I had to flip back and forth between this figure and the paragraph in section 2 on the following page to make any sense of this, and I struggled to understand exactly how this result matches the authors' main point about scarcity vs. complexity.
	- In Fig. 3, I'm not sure what to make of the notation on the right hand side, e.g. what does P(dl | dd) mean? Is dl short for do(l)? The figure caption is very vague and didn't help me either with interpreting the figure, or with what I should take away when looking at it. What does it mean to "restores the marginal P(a)", and why should I care about this? Should I be concluding something about the advantage of intervening > conditioning? Do both of the equations on the right match the diagram on the left?
- I similarly found interpreting the results in section 5 to be difficult to understand. I think it would help if the key results were explained more clearly in terms of how they support the authors' primary claims.
	- E.g. "In the former, we see an interesting phenomenon: not only is there a clear difference in the decomposed KL, but for lower occurrence counts, there is an inverse trend indicating complex structural confounders for machines that naturally produce a low number of occurrences." I think the point here may be that correlational analysis can be misleading, and can show an inverse effect compared with interventional analysis, which is known to be correct.
	- Figs 4, 5 - again, I'm not sure how to interpret these results at a high level based on the captions, or what the key takeaways from these figures are. A particularly salient example is the sentence in caption for fig 5: "We see how observational results can lead to misleading assumptions." - *how* do observational results lead to misleading assumptions? What are those results and assumptions?

### Questions
- See questions in first bullet point for weaknesses
- I had trouble understanding the novel method proposed in section 4, although I lack formal math background and it could be that I'm just not the right reviewer to understand this paper. I am unable to provide feedback on the correctness of section 4 or the significance of this contribution.
- Relatedly, a general question I have is, who is the target audience for this work - what sort of technical background and fields of interest would you expect a reader to have? On one hand, I'm excited about the general problem framing, and the general setup seems interesting to me and I think I could understand the experiments and results if they were explained more clearly, but on the other hand, I was entirely lost in section 4.

### Soundness
2

### Presentation
1

### Contribution
2
