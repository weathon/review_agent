# Bearing Syntactic Fruit with Stack-Augmented Neural Networks

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Any finite set of training data is consistent with an infinite number of hypothetical algorithms that could have generated it. Studies have shown that when human children learn language, they consistently favor hypotheses based on hierarchical syntactic rules without ever encountering disambiguating examples. A recent line of work has inquired as to whether common neural network architectures share this bias, finding that they do so only under special conditions: when augmented with ground-truth parse tree structures, when pre-trained on massive corpora, or when trained long past convergence. In this paper, we demonstrate, for the first time, neural network architectures that are able to generalize in human-like fashion when trained only on surface forms and without any of the aforementioned requirements: stack-augmented neural networks. We test three base architectures (transformer, simple RNN, LSTM) augmented with two styles of stack: the superposition stack of Joulin & Mikolov (2015) and a nondeterministic generalization of it proposed by DuSell & Chiang (2023). We find that transformers with nondeterministic stacks generalize best out of these architectures on a classical question formation task. We also propose a modification to the stack RNN architecture that improves hierarchical generalization. These results suggest that stack-augmented neural networks may be more accurate models of human language acquisition than standard architectures, serving as useful objects of psycholinguistic study. Our code is publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates whether neural networks with explicit stack-based memory exhibit human-like hierarchical generalization under poverty-of-the-stimulus conditions. It evaluates stack-augmented simple RNNs, LSTMs, and Transformers on two controlled tasks (English question formation and tense reinflection). The models implement two differentiable stack mechanisms: the superposition stack (a continuous relaxation over push / no-op / pop) and a nondeterministic differentiable vector pushdown automaton (dVPDA), trained end-to-end. On question formation, Transformers with nondeterministic stack attention show stronger hierarchical generalization (avg over 5 runs, Table 1), with ≈32% conditional probability of the full correct output (Tf+Nd) and ≈86% fine-grained accuracy (Tf+Nd+Nd) on the generalization set, compared to the vanilla Transformer (≈0.5% CP and ≈0.65 FA). In contrast, similar hierarchical generalization does not reliably emerge on tense reinflection. The authors release code and a Docker environment to reproduce the experiments.

### Strengths
The paper tackles an issue at the intersection of machine learning and linguistic theory — whether explicit structural memory can induce human-like syntactic generalization under “poverty of the stimulus.” The motivation is articulated clearly and grounded in prior psycholinguistic work. The work effectively connects formal-language theory, ML architectures, and cognitive modeling. This synthesis gives the study conceptual depth beyond architecture comparisons.
The narrative is well structured: the motivation and results sections are especially clear. The presentation demonstrates awareness of both ML and linguistic fields. The proposed architecture is accompanied by substantial mathematical derivations that ground the approach formally.
Datasets are precisely controlled via PCFG generation, and the evaluation setup directly parallels established human-language acquisition experiments. The authors provide full reproducibility artifacts, which substantially increases the paper’s value for the community and sets a reproducibility standard for cognitively oriented modeling work.

### Weaknesses
While the paper presents an elegant and well-executed study linking stack-augmented architectures with hierarchical generalization, several limitations remain.

First, the experimental scope is narrowed: all results rely on small synthetic grammars and two tasks within the poverty-of-stimulus framework, leaving unclear whether the observed effects generalize to naturalistic or cross-linguistic data. The paper lacks discussion of other diagnostic cases traditionally used to probe structure dependence — such as reflexive binding, agreement attraction, or relative clause attachment. Outlining or testing additional paradigms would strengthen the claim that stack-augmented architectures capture a generalizable linguistic bias rather than a task-specific one.

Second, the mathematical exposition of the differentiable stacks is overly dense for a broad audience; the intuition behind key derivations is under-explained (a high-level diagram is present (Figure 4), but it does not unpack the derivations), limiting accessibility.

Third, ablation analysis is insufficiently targeted: without probing stack hyperparameters (depth/size, degree of nondeterminism) and isolating the effect of shortcut connections, it is difficult to determine which architectural factors drive the gains. Though std on 5 runs are provided along with the final scores, the absence of statistical significance testing makes it hard to assess robustness of the reported improvements.

Finally, the failure on the tense-reinflection task and the limited ethical discussion of English-centric bias somewhat weaken the paper’s broader psycholinguistic claims. The failure to generalize on tense reinflection is under-discussed; understanding this discrepancy would critically test the authors’ hypothesis. Overall, the link from architectural bias to human acquisition mechanisms feels suggestive but somewhat overstated given the current evidence base.

The paper’s psycholinguistic framing would benefit from connecting to existing syntactic evaluation frameworks such as BLiMP, which operationalize a broad set of “poverty-of-stimulus” diagnostics through minimal-pair tests.

### Questions
1) It is noted that only Transformer variants successfully learn the in-distribution test set for both tasks. Could the authors elaborate on why recurrent and stack-augmented recurrent models fail to do so — e.g., optimization difficulties, hyperparameter sensitivity, or fundamental capacity limitations? A short discussion would clarify how comparable the models truly are.

2) Means and standard deviations over five runs are reported, but no significance testing is presented. Could the authors provide or comment on statistical significance of the main improvements (e.g., CP, FA, Log Ratio), to support that observed gains are not due to random variation?

3) The study focuses on two syntactic transformations within the poverty-of-the-stimulus framework. Are there plans to test additional syntactic phenomena or more naturalistic datasets to assess how general the observed bias is?

4) While the authors compare several model variants, it remains unclear which architectural or training factors (e.g., nondeterminism, stack depth, shortcut connections) primarily drive the hierarchical generalization. Could the authors briefly describe or plan targeted ablations to isolate these effects?

5) The model’s failure on the tense-reinflection task is intriguing. Could you expand on possible reasons—task properties, model bias, or data setup—and whether alternative training regimes change this outcome?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work intends to investigate whether neural network models of various architectures can be augmented with a stack mechanism in order to imbue the model with an inductive bias towards hierarchical syntactic generalization. The authors investigate a number of architectural permutations and two different stack mechanisms on a question formation and tense reinflection task, finding that a transformer with a stack mechanism imbues a hierarchical syntactic inductive bias in the question formation task (but not the tense task).

### Strengths
This work rigorously sweeps over architectural parameterizations. 

The presentation of the existing methods is thorough.

### Weaknesses
The primary weakness of this work is that the empirical contribution is relatively small. The majority of the text is dedicated to explaining architectural details already described in related work, or in describing fairly small architectural innovations that do not yield empirical improvements (i.e., the +R reading shortcut still results in a negative LR). Many of these descriptions can be put in an appendix, with the main text expanded to include additional analyses and experiments. 

For example, it would be interesting to know whether there were any systematic differences between questions that resulted in hierarchical generalization and those that did not. It would also be exciting to explore combinations of stack networks with existing findings, like Ahuja et al.’s overtrained transformer result. Perhaps one of the stack transformers gets to the overtrained regime faster than the standard transformer?

To be clear, the underlying question and research direction are exciting, but the work needs further empirical development prior to publication.

### Questions
Typos:

348 Ues 
177 inptu

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores whether stack-augmented Transformer and RNN architectures (specifically superposition stack of Joulin & Mikolov 2015 and (modifications of) the  nondeterministic stack of DuSell & Chiang 2024) leads to improved hierarchical generalization on question formation and tense reinflection tasks that are designed to tease apart the learner's biases towards linear vs. hierarchical generalization. The explored modifications to DuSell & Chiang's architecture is minor: for recurrent models, there's a direct feed-in ("short circuit") added to the output of the network at each timestep from the timestep's stack to avoid an off-by-one lag, and adding two stack attention layers to transformers. The general finding is that networks augmented with stacks show increase in their preference for hierarchical generalization for question formation for Transformers and LSTMs (but not RNNs), but not for tense reinflection. Furthermore, for LSTMs to benefit from stacks, the short circuiting mechanism seems necessary.

The paper's contribution is straightforward - it directly continues the exploration of the research question "Which factors make neural networks prefer hierarchical generalization (as operationalized by the question formation and tense reinflection datasets)?" by various work cited in the paragraph starting from L071. The findings are unsurprising (not in a negative way at all, just in the sense that it would be have been more surprising had the trends been different) but sufficiently interesting. I have a few clarification questions regarding the comparisons of results to existing work and how related work's contributions are represented, which I believe could be fruitfully resolved during the discussion.

### Strengths
- The research question, the hypothesis, and the takeaways are clear and straightforward to understand, partly owing to the fact that this problem operationalized in terms of the specific datasets used has a well established cottage industry (again not in a negative sense at all, just for lack of a better term)
- The exposition of the scope of the contribution and the experimental setup is generally clear.

### Weaknesses
- I think "syntactically supervised" is a bit of a vague term for addressing what McCoy et al. 2020 did - my inference is that this refers to the tree-structured network experiments, but highlighting the fact that training of such networks require explicit representations of syntactic structure in the training data. Since "syntactically supervised" is ambiguous between only data-level supervision and the architecture requiring parsed inputs, it might be expositionally clearer & make the new contribution that this paper clearer to say out loud that tree-structured networks have been explored as a way of endowing structural inductive biases, but this also requires the input to be fully parsed. It would also be informative to compare against this result in Table 1 & 2.
- I think it is not entirely accurate to lump Yedetore & Kim (2024) under "trained long past convergence on the validation data" although they do report grokking results, since the main claim of that paper is about an auxiliary objective of form-to-meaning mapping task, which leads to preference for hierarchical generalization without any of the listed changes. I think this result actually should put a qualifier on the results presented in this paper: for instance, the claim in the abstract "that they do so only under special conditions: when syntactically supervised, when pre-trained on massive corpora, or when trained long past convergence. In this paper, we demonstrate, for the first time, neural network architectures that are able to generalize in human-like fashion without any of the aforementioned requirements: stack-augmented neural networks." should be qualified such that "for the first time" claim really is "for the first time in networks that are purely trained on surface forms". Their results are also a lot stronger than the best FA reported in the results section ("Furthermore, Tf+Nd and Tf+Nd+Nd attain the highest FA (up to 86%) [...] It also surpasses the approximately 76% FA of Ahuja et al.’s (2025) overtrained transformer language model.") - closer to 100% FA.
- Continuing the comparison to prior work discussion, on L403: "For both tasks, only transformers learn the in-distribution test set" If I remember correctly, the McCoy et al. 2020 paper reported that LSTMs do get near ceiling full-sentence accuracy on the in-distribution test set. Does this suggest that the LSTM training here is somehow degenerate, or is there an alternative explanation for this?
- L405: "This also improves over the 10% of Mulligan et al.’s (2021) GRU trained with multi-task learning." I am also somewhat confused by this claim because their Figure 1 reports close to 50% full sentence accuracy with a multitask trained GRU. Am I misreading something here?

### Questions
- I am curious whether you did any analyses beyond quantitative results about generalization preference, and wonder in general the newly introduced architectural components enable any interesting analyses (even if you haven't looked into this) like seeing if the right structures were indeed inferred.

I think I put most other contentful questions in the Weaknesses section, so these are just small comments/suggstions:

- L96- "The question, then, is really whether a reasonably simple learning algorithm—not the kind of contrived example just mentioned, but perhaps a neural network architecture with minimal assumptions about the specific task—can learn a rule like MOVE-MAIN from ambiguous data while still attaining competitive performance on natural language benchmarks." I disagree, if what we're interested in is cognitive modeling, competitive performance on natural language benchmarks are irrelevant. But maybe by "natural language benchmarks" what is intended is only a subset of them that is relevant to cognitive modeling.
- L98- "We offer stack-augmented neural networks as a positive example that hierarchical generalization need not originate from the training data alone." While I get the intent of the statement, "need not" is a bit of a strange way to phrase things, since this is exactly aligned with the poverty of the stimulus claim, i.e., hierarchical generalization needs to originate from training data-external means.
- L348: ues -> use
- "For transformers with one stack attention layer, we swap it into the attention mechanism of the third layer. For transformers with two stack attention layers, we swap them into the second and fourth layers." --- is the choice of swapped-in layers basically a hyperparameter?
- Presentation suggestion: Tables 1 and 2 are a bit overwhelming, maybe better to have the Tables in the Appendix and have a graphical visualization.

### Soundness
3

### Presentation
2

### Contribution
3
