# Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable?

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 8

## Abstract
As AI systems are increasingly deployed in high-stakes applications, ensuring their interpretability is essential. Mechanistic Interpretability (MI) aims to reverse-engineer neural networks by extracting human-understandable algorithms embedded within their structures to explain their behavior. This work systematically examines a fundamental question: for a fixed behavior to explain, and under the criteria that MI sets for itself, are we guaranteed a unique explanation? Drawing an analogy with the concept of identifiability in statistics, which ensures the uniqueness of parameters inferred from data under specific modeling assumptions, we speak about the identifiability of explanations produced by MI.

We identify two broad strategies to produce MI explanations: (i) "where-then-what", which first identifies a subset of the network (a circuit) that replicates the model's behavior before deriving its interpretation, and (ii) "what-then-where", which begins with candidate explanatory algorithms and searches in the activation subspaces of the neural model where the candidate algorithm may be implemented, relying on notions of causal alignment between the states of the candidate algorithm and the neural network. 

We systematically test the identifiability of both strategies using simple tasks (learning Boolean functions) and multi-layer perceptrons small enough to allow a complete enumeration of candidate explanations. Our experiments reveal overwhelming evidence of non-identifiability in all cases: multiple circuits can replicate model behavior, multiple interpretations can exist for a circuit, several algorithms can be causally aligned with the neural network, and a single algorithm can be causally aligned with different subspaces of the network.

We discuss whether the unicity intuition is necessary. One could adopt a pragmatic stance, requiring explanations only to meet predictive and/or manipulability standards. However, if unicity is considered essential, e.g., to provide a sense of understanding, we also discuss less permissive criteria. Finally, we also refer to the inner interpretability framework that demands explanations to be validated by multiple complementary criteria. This work aims to contribute constructively to the ongoing effort to formalize what we expect from explanations in AI.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
- The objective of the paper is to answer the question, do current criteria in mechanistic interpretability (MI) guarantee the identifiability of the explanation?
- Authors sort MI methods into two broad strategies: 
    - _where-then-what_ focuses on finding a subset of the network – a circuit – that captures most of the information flow from in- to outputs. Once this circuit is identified, the next step is to interpret its components (features) to derive the explanatory algorithm.
    - _what-then-where_ starts by identifying candidate algorithms and then searches subspaces in the neural network where the algorithm may be implemented, using causal alignment between the explanatory algorithm’s states and the network’s internal states. 
- They stress testing both methods with toy models: small MLPs trained on logic gates. They performed three main types of searches to test different interpretability criteria: 
    - Circuits search: Looking for subnetworks that perfectly replicate the model's behavior
    - Interpretations search: Trying to map neurons to logical gates in a way that's consistent with their activations
    - Mappings search: Testing different ways to map logical gates to groups of neurons
- Key findings in toy models: they found multiple valid interpretations for the same network.
    - 85 different circuits that achieved perfect accuracy
    - An average of ~536 possible logic gate interpretations per circuit
    - 159 perfect minimal mappings between algorithms and neurons
    - In total, over 45,000 different possible computational abstractions
- Experiment larger NN trained on a subset of MNIST revealed similar dynamics
    - The circuit search found over 3000 valid circuits

### Strengths
- The authors introduce their own taxonomy and formalization (e.g. where-then-what/what-then-where, Circuits, Mappings, etc) for important concepts discussed in the paper. While I haven’t fully wrapped my head around the usefulness of the taxonomy, I appreciate the effort to deconfuse different interpretability methods. I think this is the strongest suit of the paper and I wish they had focused the paper on the taxonomy and less on the experiments. 
- I enjoyed the writing style, and it was easy for me to follow, particularly Sections 2 and 3. I also found Figure 2 to be helpful. For the _what-then-where_/_where-then-what_ split. 
- The paper touches on an important topic within interpretability – a lack of quality in the discourse around helpful metrics for interpretations.

### Weaknesses
- While I think the taxonomy could bear some relevant fruit, for experimentally testing the metrics using NN this small, it seems challenging to generate insights that are relevant to interpretability as a total. From my understanding, we are not close to having high-confidence circuit interpretations (“perfect circuits”) in the first place, so working under this assumption might be several steps ahead.
- Continuing on the larger NN experiments: I struggled to understand the points in line 479 ff. What is in this case the "valid circuit"? I hoped this section could have bridged a gap toward showing how this metric could be used in the future but failed to show its limitations in this instance. (But maybe I simply oversaw that.)
- Lastly, I think there is also too little emphasis on existing literature that clearly touches on the underlying problem: disentanglement. I would consider modeling the paper around the taxonomy and then focusing on existing research and problems in relation to known circuits, such as IOI.

### Questions
(From the weakness section)
- What is in the case of the MNIST NN the "valid circuit"?

### Soundness
3

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
The authors investigate the potential issue of identifiability in mechanistic interpretability through experiemnts on small MLPs where (isolated) circuits are ennumerated and assessed fairly exahustively. They find identifiability is an issue at all levels: in the number of subcircuits functionally aligned with the full network, in the number of algorithms consistent with the behavior, and in the mappings between algorithms and circuits. This problem gets worse as architecture size increases, and training on a greater number of tasks only mitigates this issue to some extent.

### Strengths
* Well written
* Well referenced
* Adresses an issue of interest to the interpretability community
* Provides exhaustive experiments with well-understood ground truth.
* Investigates the effects of various variables (architecture size, number of tasks, noise) on the identifiability problem.

This kind of study is very much needed.

### Weaknesses
I first summarize the most important points and then elaborate.

* It is unclear whether the conclusions about identifiability problems can be stated generally or only in particular for circuits as isolated objects (circuit discovery through zero-ablation), which might mischaracterize the functioning of the network as a whole.
* The possible improvements advertised in the abstract/introduction are rather only sketched in section 5.1.


Is the large number of circuits/possible explanations due to looking for circuits in isolation (e.g., via zero ablation) rather than working in context with the rest of the network (e.g., via activation patching)?

Line 320 describes the circuit isolation procedure. This is equivalent to zero-ablation and the criterion is equivalent to the definition of suficcient circuit.
How would identifiability look like if we chose to define circuits as they function in the context of the full network? See for example the definition of circuit via activation patching in Adolfi et al., 2024.

Isn’t it possible that many of the isolated circuits discovered through zero-ablation are mischaracterizations of the in-context functioning of the circuits as they are embedded in the full network?


Line 080: “a model’s behavior should have a single, well-defined explanation”. There is no citation here and it is unclear where this intuition is coming from, what is it’s theoretical support, etc. To offer a counter-intuition: consider a circuit that is sufficient on its own to mimic the behavior of the full network over some input domain; such a circuit need not be unique. Trivially, the circuit plus additional neurons to form the full network is another such circuit. But there is no contradiction in intuiting that multiple such circuits of different sizes, with partial or no overlap exist in the network and, in principle, offer alternative (perhaps incompatible?) ‘explanations’ (see Adolfi et al. 2024 for theoretical analyses).

On Line 091: the authors mention “the identifiability properties of current MI criteria”. The criteria of interest that define circuits leave open the possibility that these circuits are not unique. So the definition of these circuits does not preclude non-identifiability unless the uniqueness property is trivially appended to the definition. This leads one to suppose that uniqueness under the typical definition of circuits is a property left to be determined empirically. It could, in principle, be motivated theoretically, but I see nothing in that direction here. Is it possible to provide some theoretical motivation for uniqueness that is not trivially stipulated but justified from first principles?

If a network implements the same functionality for different inputs through different circuits and algorithms, does this really make mechanistic interpretation hopeless? (i.e., in this case, is only a functional explanation capable of unifying all the existing ‘explanations’?). It would be useful to have any assumptions about satisfactory explanations made explicit in the manuscript.

Line 489: “the challenge lies in defining criteria that distinguish valid explanations from misleading ones.”
It seems to me that, conceptually, identifiability does not pose a problem for distinguishing misleading from valid explanations. The problem arises only if an explanation is presented as unique or valid for a full input domain when this is not so. This issue might warrant some clarification.

Line 490: “According to MI, the explanatory algorithm should be unique, meaning multiple competing explanations should not exist.” But this statement is made without citation. This assumption seems ill-founded to begin with, for the reasons mentioned above. Where does the criterion come from?

### Questions
Minor comments, questions, and suggestions:

Line 403 states that “larger architecture…could also lead to greater overparameterization”. This could benefit from elaboration; in particular, how larger architecture could lead to a reduction in the number of valid abstractions.

On Line 065, “Given the near impossibility of exhaustively searching all possible algorithms across all subsets of a neural network”, I might suggest to reframe this not as impossibility but as intractability, infeasibility or implausibility. Certain interpretability queries might have large search spaces that could nevertheless be searched efficiently. The relevant property is the complexity of the interpretability query, not merely the size of the search space. For computational complexity analyses of circuit discovery, see Adolfi et al 2024.

On Line 067, the authors state “researchers have developed approximation methods with different assumptions and trade-offs”. It seems to me that the circuit discovery methods that are typically developed are heuristics for circuit finding, not approximation algorithms with any proven guarantees. In any, case it would be useful if the authors can distinguish between these two categories in their descriptions.

Citation to Van Rooij on Line 046 does not seem to fit with the corresponding sentence, as that paper does not deal at all with interpretability, as opposed to Lindsay, 2024, which is indeed an approapriate citation. For examples of studying the fundamental properties of (inner) interpretability queries see Adolfi et al., 2024.

Section 2.1 mentions interpretability work on transformer models but only in language. An example from vision transformers can be found in Vilas et al. 2023.

Please clarify the notation in Definition 4.

Line 229 makes an implicit statement about computational complexity but provides no citation. See Adolfi et al. 2024 for relevant complexity analyses. This is also relevant to the statement on Line 257. Here it would also be useful to clarify how uniform random sampling “approximates” the desired measure, as this seems non-obvious. Perhaps the authors mean random sampling is a heuristic with unknown properties?

Line 494 states that current MI methods can only approximate their targets because exhaustive enumeration is impossible for large models. This is technically incorrect, as even for some problems with exponential serach spaces, efficient search algorithms that find optimal solutions are possible. The relevant notion is the computational complexity of the interpretabililty queries, not simply the size of their search space (see Adolfi et al., 2024).

Section 2.1 describes a parallel between AI interpretability and neuroscience. A framework that draws from lessons grounded in this parallel is described in Vilas et al. 2024. This framework provides a nice embedding for the what-where distinction, corresponding to the algorithmic and implementational levels, respectively.

The problem of identifiability interacts in interesting ways with the computational complexity of circuit finding. Adolfi et al. 2024 analyses circuit queries that are relevant to the authors’ points on identifiability. See, for instance, counting problems which ask for the number of circuits in a train neural network that have a certain property (e.g., they are sufficient for a behavior). Furthermore, if the number of sufficient circuits is typically large, heuristics for otherwise intractable problems (e.g., sufficient circuit) could seemingly find their targets in a feasible amount of time. In this scenario, non-identifiability is an important catch.

### Soundness
2

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
5

### Summary
This paper investigates the \emph{identifiability} of mechanistic explanations from interventional evaluations on toy neural networks. The authors find clear non-identifiability at multiple stages of the interpretability pipeline: multiple interpretations can exist for a circuit, multiple circuits can generate the same behavior, each algorithm can be aligned to multiple activation subspaces, etc.
This non-identifiability persists regardless of whether MI explanations are generated first by localizing a subset of the network, then deriving an interpretation, or first generating a candidate algorithm and trying to then find an activation subspace corresponding to that algorithm.

### Strengths
A rare case of compelling and relevant deconfusion experiments which doesn’t devolve into over-claims. The authors do well in the discussion highlighting that identifiability may not be needed for all applications of mech interp, and that the non-identifiability observed here is on toy-models, so might not extend to larger models trained on multiple tasks.

The “what-then-where” strategy implemented here appears to solely utilize an approach based on Distributed Alignment Search (DAS), which makes the results about non-identifiability quite relevant to recent discussion:

Makelov, A., Lange, G., & Nanda, N. (2023). Is this the subspace you are looking for? An interpretability illusion for subspace activation patching. arXiv. https://arxiv.org/abs/2311.17030

Wu, Z., Geiger, A., Huang, J., Arora, A., Icard, T., Potts, C., & Goodman, N. D. (2024). A reply to Makelov et al. (2023)'s "Interpretability Illusion" arguments. arXiv. https://arxiv.org/abs/2401.12631

### Weaknesses
Primary weakness: The most novel contribution of this paper are its experimental results, which don’t have enough description: Appendix B only contains aggregated results about the number of circuits and average interpretations per circuit found, but e.g. lacks examples of said circuits for qualitative validation. This significantly undercuts my ability to validate the correctness of the experiments.

In addition, the authors acknowledge appropriately that identifiability may not be necessary if the goal of MI is merely to steer a model. However, much MI work is driven by a desire to simply further scientific understanding of language models. What types of scientific inquiries require computational identifiability, and which do not? The paper could be strengthened by further discussing how much identifiability matters if the goal is scientific understanding, rather than just the model steering mentioned in lines 520-527.

### Questions
1. Could you elaborate on the meaning of “incompatible” in line 355? The paper would benefit from a clear example of two incompatible explanations, ideally in the main body.

2. Please include in the appendix random examples of some of the circuits found so that they can be qualitatively assessed by readers.

3.Could you comment on how these results relate to the discussion of Makelov et al. (2023) and Wu et al. (2024), cited above in section “Strengths”?

### Soundness
3

### Presentation
3

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
A summary of mechanistic interpretability is provided. Two approaches are proposed that attempt to interpret a “simpler” algorithm that emulates the behavior of a trained neural network in terms of a circuit. One approach focuses on modelling the behavior of the full network before finding a subset of related nodes, while the second approach focuses on finding an “important” sub-network of the full network, whose behavior is then interpreted. Both approaches are showcased in simple toy examples.

### Strengths
The manuscript is clearly written, with a good explanation of the context the work is placed in. The toy examples provided are illustrative. The overarching conclusion is that uniqueness of an explanation should generally not be expected within the context of mechanistic interpretability, and while a similar analysis cannot be conducted on large-scale models, it is likely that the same behavior could be expected. This insight is important in many practical applications where network interpretations are required (post-training) by a practitioner.

### Weaknesses
The issue of uniqueness of an explanation is addressed in the context of mechanistic interpretability. However, the “incompatibility” of different explanations is not substantially addressed. A more formal framework in which incompatibility can be “measured” would be very interesting, along with analyzing questions on differentiating between equivalence classes of explanations.

### Questions
If the training set is drawn from a distribution with certain biases, there may be correlations that essentially encourage multiple “conflicting” interpretations of a network. Can we resolve some of the issues that arise by putting conditions on the training distributions?

What would (be expected to) happen if, in a simple toy example, an experiment was repeated with a perfect training error, with or without overfitting? Would we see a qualitatively different distribution of explanations?

### Soundness
3

### Presentation
4

### Contribution
3
