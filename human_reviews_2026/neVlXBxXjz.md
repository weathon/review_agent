# Examining relational reasoning and inductive bias in transformers trained on a transitive inference task

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Transformer-based models have demonstrated remarkable reasoning abilities, but the mechanisms underlying relational reasoning remain poorly understood. We investigate how transformers perform \textit{transitive inference}, a classic relational reasoning task which requires inference indirectly related items (e.g., if $A>B$ and $B>C$, then $A>C$). Comparing in-weights learning (IWL) and in-context learning (ICL), we find that that IWL naturally induces a generalization bias towards transitive inference, despite being trained only on adjacent items, whereas ICL models trained solely on adjacent items do not generalize transitively. Mechanistic analysis shows that ICL models develop induction circuits that implement a simple match-and-copy strategy that performs well at relating adjacent pairs, but does not encoding hierarchical relationships among indirectly related items. However, when pre-trained on in-context linear regression tasks, transformers successfully exhibit in-context generalizable transitive inference,  displaying both \textit{symbolic distance} and \textit{terminal item effects} characteristic of human and animal performance, without forming induction circuits. We extend these findings to large language models, demonstrating that prompting with linear geometric scaffolds improves transitive inference, while circular geometries impair performance, particularly when models cannot rely on stored knowledge. These results suggest that pre-training on tasks with underlying linear structure promotes the development of representations that can scaffold in-context relational reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work explores how transformers learn a transitive inference task under in-weights learning versus in-context learning. For in-context learning, the input is a sequence of pairs of objects and their order relation, followed by a query for a pair of objects, and the target output is the order relation. For in-weights learning, the context sequence is replaced by random noise, and the model directly learns the order relation from the query pairs. For ICL, the ordering is randomized for each sequence to encourage a general in-context strategy rather than memorization. For the IWL setup, the ordering is fixed during training. In both cases, the queries in training data include only adjacent pairs (which, in the ICL setup, must appear in the context sequence).

The authors find that the IWL model generalizes to non-adjacent query pairs, whereas the ICL model does not. They show that the ICL model learns an induction head mechanism, explaining its inability to generalize. Motivated by previous work on linear models and transitive inductive biases, they pre-train the ICL model on an in-context linear regression task and find that this enables transitive generalization. Finally, this work seeks to connect these insights to large-scale models by exploring how prompting LLMs to "imagine" a linear vs circular structure affects performance on the ReCogLab task.

### Strengths
- Research problem & Motivation: The research question is well-motivated and grounded in cognitive science. The discussion of symbolic distance and terminal item effects is insightful, and observing these patterns in trained transformers is interesting.
- Integration of different analysis methods: The paper thoughtfully integrates empirical performance results with mechanistic interpretability analyses, which are both needed for understanding the inductive biases of different models under different training regimes.
- Model interpretability: the mechanistic analysis in the (no pretraining) ICL model identifies the learned computational circuit by testing the hypothesis that an induction head mechanism is being implemented, through attention head ablations and relating PCA embeddings to separability of the target. This explains the failure to generalize transitively. 
- Clarity of presentation: the paper is well-written and provides interesting discussion, bridging literature from machine learning and cognitive science.

### Weaknesses
**Major Concerns:**

- One of the paper’s central claims is that the ICL model fails to generalize transitively, whereas the IWL model succeeds. However, this may partly reflect differences in the specific training setups rather than a fundamental limitation of in-context learning. In the ICL configuration, training queries always correspond to pairs that appear in the context sequence, making the simplest strategy to copy answers rather than infer broader relational structure. In other words, the problem could be that the model "doesn't know what the task is", rather than it failing to learn it (the training regime makes it appear like the task is to copy information from the context). While the authors’ intent is to test generalization to longer transitive chains, this could be achieved in other ways (e.g., train on queries with $|i - j| \leq k$ and evaluate on $|i - j| > k$, for $k > 1$). In short, the observed differences between IWL and ICL may stem from task design rather than from intrinsic properties of the learning paradigms themselves.
- The use of an attention-only transformer limits the applicability of the conclusions to real Transformers. The two types of models have very different types of inductive biases; this is an important limitation since the paper aims to make claims about the inductive biases of Transformers. For example, an MLP may be required for certain types of computation that would otherwise be learned by a Transformer (e.g., combining the results of multiple attention heads in the ICL configuration to infer the order relation for $|i -j| > 1$). While model interpretability is a valid concern, modern interpretability tools (e.g., sparse autoencoders) would enable interpretability of MLP components, especially in such simple synthetic settings.

**Other limitations:**
- The number of objects in the context, $N=7$, is quite small, making the scope of the experiments somewhat limited. It would be interesting and valuable to evaluate how the results scale with larger $N$.
- The mechanistic interpretability analysis for the linear regression-pretrained model is limited. It only shows that the learned model is not an induction head, but it does not identify the computational circuit that allows the model to generalize transitively. It would be interesting to know what this ICL model is doing that allows it to generalize.
- Generally, the task being studied is quite simple and synthetic, with a small scale ($N = 7$), yet the paper often makes broad claims relating to "relational reasoning" at large. For example, the claim that pre-training on linear regression can help with relational reasoning generally seems to be a big jump from the specific synthetic TI task considered in the paper.
- The connection between the linear regression pretraining and the LLM prompting experiments is unclear. The "linear geometry" prompt ("imagine a number line") seems conceptually distinct from the linear regression pretraining task, and the mechanisms driving performance improvements likely differ.

### Questions
- Have you tested setups where the ICL model is trained on queries not restricted to pairs appearing in the context? How would this affect generalization to $|i - j| > 1$?
- This work explores how the inductive biases of Transformers affect relational learning for IWL vs ICL. There has been prior work that explores architectures with explicit relational inductive biases, including some recent work on variants of Transformers (e.g., RelationNet, PrediNet, CoRelNet, Abstractor, etc.). I'd be interested in your thoughts about how these types of explicit relational inductive biases (which Transformers don't necessarily have) would fit into the picture. 
- Can you elaborate on the mechanistic connection between linear regression pretraining and the TI task? *Why* does linear regression support transitive inference? Adding a brief intuitive summary (e.g., from the exploration in Lippl et al. (2024)) would strengthen this point in the paper.
- I was unsure how to read the figures with "Prediction Value" on the y-axis. I was expecting this to be an accuracy, like in Figure 1D on results from rhesus macaques, reprinted from Lippl et al. (2024). The paper says the IWL model performs "above chance". What are the accuracies achieved here in each configuration? It would be somewhat interesting to compare this to the data from rhesus macaques.
- One notable (potential) advantage of the ICL setup is the capacity for generalization to unseen objects. I.e., IWL will only work on the objects it sees during training (because it needs to learn their order/rank), whereas ICL can (in principle) generalize not only to new orderings but also to completely different objects/images. Can you comment on this? Is this something you explored?


Overall, this study offers an interesting perspective on a timely topic, and I look forward to further discussion with the authors.

### Soundness
2

### Presentation
3

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
The authors present an analysis of the ability for transformer models and large language models to perform transitive inference (i.e. inferring from examples of the general form “A > B” and “B > C” that “A > C”). Specifically, the authors compare a transformer model trained from scratch on fixed transitive inference data (allowing the model to store item relationships in weights) to a model trained on randomly-generated transitive inference sequences provided in-context (preventing memorization) and find that the first model successfully generalizes to non-adjacent query items while the second model does not. The authors also find evidence for an “induction circuit” in the second model which may explain its memorization-based behavior. Finally, the authors demonstrate that prompting LLMs with “linear” geometry improves their performance on transitive inference tasks compared to prompts with “circular” geometry (which does not support transitivity).

### Strengths
For the most part, this paper is clearly written and argued. The authors present a thorough set of experiments for teasing apart the differences between in-weight and in-context training with regards to performance on transitive inference. I appreciate that the paper contains both experiments on transformers trained from scratch and extant large language models, which helps support a broader range of conclusions than either of those experiments in isolation. The authors have also seemed to engage with much of the existing literature on the topic.

### Weaknesses
As I’m sure the authors are aware, Figures 1, 2, and 3 are seriously malformed and more-or-less impossible to read. This should obviously be corrected in any future versions of the manuscript, but I don’t count it as a particularly serious critique since it seems very easily fixable.

More seriously, I think the paper would benefit from a stronger argument as to the impact and implications of the work. The authors have successfully demonstrated that differences exist between different training types in terms of their transitive inference performance, but I would like to see a bit more discussion of what this might say about the capabilities of modern models, especially since they presumably exhibit effects of both in-weight and in-context learning. For instance, Figure 6B seems to indicate that circular geometry prompting improves the performance of the largest model despite the fact that the context doesn’t support transitivity. What might account for this? More generally, how might these results scale to more complicated problems of interest that have transitivity as a component? I'm open to arguments about the significance of these results and interested to hear the authors' perspective.

### Questions
- In Figure 2A, the characters in the “context” section appear to not match the english characters in the blue box above. Is this deliberate?
- In Figure 2D, the prediction values seem to range between 1 and 3. My understanding was that the model was trained to predict only the direction of the relationship (i.e. {-1, 1}) -- am I misunderstanding?
- Line 195: the text indicates that the terminal items are A and E, but the figure seems to indicate that A and G are terminal items, while E is non-terminal -- which is correct?
- Just to clarify: during in-context training, the model is still only presented with adjacent queries, correct? So the “accuracy” Figure 3C is referring only to adjacent-pair accuracy?
- Line 454: The reference to Figure 9 should probably be to Figure 6 instead
- A potentially relevant citation: “Rane, Sunayana, et al. "Position: Principles of Animal Cognition to Improve LLM Evaluations." Forty-second International Conference on Machine Learning Position Paper Track.”

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies whether transformers can learn simple relational rules in context using a transitive inference task (the ability to infer that if A > B and B > C, then A > C). The authors train small two-layer attention-only transformers on a task where the model sees short sequences of pairwise comparisons between symbols. Each item is an Omniglot images passed through a frozen ResNet18. The model’s input consists of triples: two item embeddings and a label indicating which one ranks higher. Importantly, during training the models only ever see adjacent pairs. At test time, it must predict relations for non-adjacent pairs. 

The authors compare two training regimes: (1) in-weights learning (IWL), where the hierarchy of items is fixed across all sequences such that the model can memorise the order in its parameters, and (2) in-context learning (ICL), where a new random hierarchy is drawn for each sequence, forcing the model to infer the order from the examples in the context window. They find that models trained with IWL generalise to non-adjacent pairs whereas ICL models fail completely on non-adjacent pairs. An analysis of attention patterns suggests that induction heads play a crucial role in the ICL model. In other words, the model just learned to look up solutions in the context. 

To test whether this limitations is a result of missing representational structure, the authors pre-train the same model on an in-context linear-regression task, which plausibly builds a one-dimensional latent geometry, and then fine-tune it on the transitive inference task. After this pre-training, the model is capable of generalising to non-adjacent pairs and does not rely on induction heads anymore. Finally, large language models show a similar pattern: they solve transitive inference-style prompts more reliably when the relations are described along a line than when arranged on a circle.

### Strengths
- The paper addresses an interesting question about the generalization limits of in-context learning.
- The experiment showing a benefit of pre-training on linear regression is interesting and novel.

### Weaknesses
- All mechanistic conclusions (e.g., induction circuits versus distributed geometry) are drawn from a two-layer, attention-only transformer without MLPs. Since MLPs play a central role in representation building in standard transformers, removing them may substantially alter model behavior. The observed ICL limitations might therefore be specific to this simplified architecture.
- The conclusion that linearity in the regression pretraining task is the key enabling factor is not strongly supported. Other properties, such as the need to integrate information globally across the context, might equally explain the improvement.
- The use of fixed ResNet18 embeddings introduces unnecessary high-dimensional structure that may not be relevant to the symbolic nature of the task. Testing with simpler, low-dimensional or one-hot embeddings would clarify whether the results depend on this choice.

### Questions
- How robust is the ICL limitation? If the training data included a small proportion of non-adjacent pairs (e.g., 5%), would the model begin to form a distance-like or global representation, or would it continue relying on match-and-copy strategies?
- Could you clarify whether MLPs were omitted primarily for interpretability or for computational reasons, and whether adding them changes the qualitative results?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies in-context learning as well as trained transformers for a relational learning task.

### Strengths
Writing:
- Overall, the writing is clear and easy enough to follow, but with a number of caveats (see below).

Conceptual:
- The idea of looking for transitivity from ICL and a learned perspective is interesting.
- The result that ICL does not perform well whereas the learned model does is interesting as well.
- The found circuits shed a light on what is going on to some extent.

### Weaknesses
Writing:
- Figures 1 and 2 are made poorly. In each caption subfigures A), B), ... are referenced, but it is not clear to which of the images they refer to. Also, they are mostly some symbols that are hard to interpret. A prominent example is Figure 1: It shows in the upper left corner some letters, then diagrams that are completely non intelligible without looking at the caption, then some token sequences that are also cryptic. Illustrations are supposed to be as self-explanatory as possible.
- Also, I do not understand how the upper left image in Figure 1 is encoding a hierarchy. Also other parts of the image are poorly done and hard to understand.
- page 2: Acronym TI used before it is defined a few sentences below.
- Section 2.2 on the training setup is too vague and I would not be able to reproduce it from reading the paper. I would suggest to make it exact, precise and detailed enough to let the reader have a better understanding.
- Overall, the illustrations are done poorly and since they encode the results of the studies conducted in the paper, I do not think that the paper is fit for publication purely on presentational grounds.

Conceptual:
- The training setup seems somewhat convoluted and made-up. I am not sure this bears great relationship to reasoning tasks found in the real world.
- I am not sure that the setup makes sense as regards the in-context learning. ICL only emerges at larger scale in LLMs and hence I am unsure whether your setup with a smaller transformer trained from scratch adequately captures the properties of ICL as found in LLMs of interest.
- The analysis from the mechanistic interpretability point of view is not very deep, even though it contains some interesting patters in Section 3.2. Some circuits are identified, but no complete mechanistic understanding is achieved. The interpretation is just a standard application of a few tools without any deeper dive into the mechanics.
- In Section 3.3. I do not see how the claims are substantiated. The results in Figure 5 are not giving too much of an indication on that.
- Overall, the paper is a synthetic study with limited importance for real LLMs and limited mechanistic results for the synthetic setup they had, even though I think that one could obtain a more complete understanding for the simple 2-layer transformer for the studied task.

Other:
- No code and datasets were accompanying the paper and they are not announced to be released upon acceptance.

### Questions
- What is the connection between ICL in current LLMs and in your training setup. Can any insight be drawn from your study on that?

### Soundness
3

### Presentation
2

### Contribution
2
