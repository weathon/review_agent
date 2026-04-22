# Task representational dynamics for compositional generalization

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
The ability to generalize compositionally is central to intelligent behavior.
While recent work shows that networks can generalize compositionally under certain conditions, many studies focus on simple compositional tasks, such as those that are purely linguistic (or unimodal), or those with no temporal structure.
Here we investigate how representational dynamics shape compositional generalization in recurrent neural network (RNN) models during cognitive tasks with evolving temporal structure,
providing insights into neural computation during flexible reasoning.
We trained RNNs on the Concrete Permuted Rules (C-PRO) task, a cognitive compositional task established for humans that requires integration of information across task phases.
We assessed how different learning regimes induced generalization and representational dynamics.
We systematically varied model initializations to generate RNNs that exhibited a wide range of compositional generalization performance, ranging from 38\% to 90\%.
Analysis of high-performing models revealed nontrivial temporal dynamics of task representations, highlighting the importance of selectively engaging the right features at the appropriate task phase for generalization.
Our findings reveal that successful compositional generalization requires the orchestration of structured intermediate representations that are dynamically composed, resulting in complex, feature-specific representational dynamics
-- providing testable principles for how neural systems enable flexible reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the mechanisms underlying compositional generalization in recurrent neural networks (RNNs) within a temporally structured, context-dependent task (C-PRO). The authors manipulate model initializations to generate networks with a wide range of generalization performance.

The central finding is that successful generalization does not correlate with a single, static representational geometry. Instead, it depends on specific, dynamic modulations of feature-specific representations over the course of a trial. High-performing networks learn to maintain a stable, low-dimensional representation for the task context (rule). During stimulus presentation, the networks expand to a high-dimensional state to integrate stimulus and context information, before compressing the representation into a low-dimensional state for the final response. In contrast, poorly generalizing networks exhibit a monotonic increase in dimensionality, suggesting a less structured, memorization-based strategy.

The primary contributions of this work can be summarized as follows:

A Shift from a Static to a Dynamic Account of Generalization: The work moves beyond prior analyses of static representational structure by demonstrating that the temporal evolution of dimensionality is a key predictor of compositional success. The identified pattern of compression-expansion-compression is a novel finding.

A Proposed Reconciliation of Competing Neuroscientific Theories: The findings offer a computational model that helps reconcile seemingly contradictory results in neuroscience regarding the necessity of both low-dimensional (for abstraction) and high-dimensional (for integration) neural codes, showing they are different phases of a single dynamic process.

A Novel Method for Feature-Specific Representational Analysis: A significant methodological contribution is the decomposition of global network activity into the dynamics of its constituent parts (context, stimulus, response). This technique made it possible to uncover the independent, yet orchestrated, representational trajectories that were obscured by global measures.

Evidence for Emergent, Efficient Cognitive Strategies: The analysis of "closure" and "integration" trials reveals that the learned representational dynamics enable the models to form early decisions when task conditions permit, providing evidence for the emergence of flexible, human-like cognitive strategies rather than a rigid, uniform process.

### Strengths
Originality: The paper is reasonably original, viewing the problem of compositional generalization through a different lens (across time and across features), which allows it to find novel insights into the nature of compositional generalization

Quality: The paper's quality is generally good, with clear and concise language, and good contextualization relative to prior work

Clarity: The writing and figures are well presented and designed, as well as grouped together into logical groups of figures

Significance: It is unclear to me (and I admit that I am not very familiar with the literature) whether the authors are trying to make the point that this is how the human brain works, or this is specifically how RNN works, or this is how all AI models work. I think an expanded discussion of the significance of these results for neuroscience/cognitive science/artificial intelligence would significantly strengthen the paper

### Weaknesses
1. It is unclear to me (and I admit that I am not very familiar with the literature) whether the authors are trying to make the point that this is how the human brain works, or this is specifically how RNN works, or this is how all AI models work. I think an expanded discussion of the significance of these results for neuroscience/cognitive science/artificial intelligence would significantly strengthen the paper

2. Some of the descriptions of the plots seem to be mischaracterized. For example, line 195: dimensionality tended to monotonically increase across task phases -- this doesn't seem to be true based on the plots presented. I am not disagreeing with the conclusion the authors take from these plots, but I think that the description should be made more carefully

### Questions
Suggestions:

1. I think an expanded discussion of the significance of these results for neuroscience/cognitive science/artificial intelligence would significantly strengthen the paper

2. I think the descriptions of the plots in the text for figure 2 should be revised to be more accurate, without changing the conclusions.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the representations developed in RNNs to solve a compositional generalization task. Specifically, the authors compare the performance and the temporal dynamics of latent representations in RNNs initialized with rich or lazy learning regime (different initialization norms). The results yield insights into the temporal changes in the representation dimensionality for rule/context, stimulus, and response information, and how the dimensionality changes link to compositional generalization accuracy.

### Strengths
- The temporal changes in representation dimensionality provide a nice unification of prior mixed findings in neuroscience.
- This work provides quite a comprehensive analysis of the model's learned representational strategy in a compositional generalization task.

### Weaknesses
- I find it a bit unclear what the target scope/audience for the work is. At the outset, this work seems to target the general issue of compositional generalization in RNNs. But later on it seems that the results speak more directly to reconciling prior conflicting findings in neuroscience. I think an effort to clarify the scope in the title/abstract could help resolve this potential confusion.
- Related to the above, as acknowledged, the results are limited to a particular task and model setup. This seems less of an issue if the goal is to provide a resolution to prior conflicting findings in neuroscience, but more of a concern if the goal is to speak to compositional generalization abilities in RNNs at large.
- It's a bit unclear what representational dimensionality mechanistically capture -- sometimes low dimensionality is attributed with abstract representation, other times attributed with reduced information. I think some causal exploration of these hypotheses, or additional clarity on whether rep. dim. reflects available information vs. information used downstream, would greatly enhance the paper.
- The clarity in writing can be improved. Specifically, more pointers to the the architectural details can be good. The key metric studied, representation dimensionality, also seems to be missing a definition in the main text.

### Questions
- Have the authors explored task/model variations, and whether these results generalize across different settings? For example, what may happen if the rule signal is given after the stimulus presentation, or if the hidden layer size is increased or decreased?
- Can the same regression analysis and closure/integration trial comparison be done on the lazy model? This seems like an interesting opportunity to reveal insights into the model's failure mode.

### Soundness
2

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
The authors trained a range of RNNs on a compositional generalization benchmark, C-PRO. They find that rich RNNs generalized substantially better than lazy RNNs. They then analyzed the dimensionality of the stimulus, context, and response representations identifying specific transitions between high and low-dimensional representations over time. In particular, their analysis implies that at different times the rich networks expand and compress dimensionality. Further, they found substantially different dynamics between trials that required integrating both stimuli and trials that did not.

### Strengths
Overall, I liked this paper. I thought it was well-written and covers an important topic: how does feature learning in RNNs influence compositional generalization and what are the representational dynamics of a network that can generalize compositionally? The authors provide a careful and detailed analysis of these representational dynamics, focusing on analyzing the participation ratio. In particular, I think the insight that different phases of the task require either high or low dimensionality is really interesting and provides further nuance to the common wisdom that rich learning induces low dimensionality and that low dimensionality is good. Overall, this paper could be a valuable contribution to ICLR and the general literature on compositional generalization, in particular in a neuroscience context where RNNs are an important architecture that is generally understudied in the context of compositional generalization.

### Weaknesses
Primarily, I think the paper would benefit substantially from additional analyses that go beyond just analyzing the participation ratio and further investigate what those different dimensions are representing. For example, representations could be analyzed by looking at representational similarity matrices in more depth, by looking at whether information is invariantly encoded across conditions (e.g. by analyzing cross-condition generalization performance, see e.g. Bernardi et al., 2020), or by plotting representations using dimensionality reduction methods. Below I'm highlighting a few questions the current analysis leaves open that I think such an analysis could answer:

- What information do the high-dimensional stimulus encodings represent in contrast to the low-dimensional stimulus encodings?
- How are the different contexts encoded in a lower-dimensional space?
- How do the encoding of stimulus and context interact across the different timesteps? Are they orthogonal to each other or are they entangled with each other?
- In closure conditions is the correct response already encoded invariantly to e.g. context before the second stimulus is perceived?

In my opinion, this would substantially improve our understanding of how the rich RNN solves this task. I want to emphasize that the paper has several interesting takeaways and provides a sound analysis, so I think it could be a valuable contribution in its current form. However, I think it also leaves important questions about the representational dynamics in the RNNs unanswered, which limits its impact. I would therefore currently consider the paper a "marginal accept".

### Questions
- Can your current analyses address the questions I highlighted in "Weaknesses"?
- Why do you keep the task context active throughout? Do you anticipate that that strongly changes network behavior?
- In the rich networks, is there any structure to the errors the network continues making? Are the errors made randomly or are there certain held-out data these networks consistently don’t generalize on? What do you think causes the sub-100% generalization?
- Does your analysis make predictions that could be tested in the fMRI recordings or is the temporal resolution of fMRI too low? 

**Minor suggestions**
- Minor point: You're saying that "lazy" learning memorizes (l. 38). I see what you're saying but I would avoid the term "memorization". There are many cases where lazy models generalize, including to out-of-distribution data [1] and to compositional generalization data [2]. Similarly, while Lippl & Stachenfeld (which you cite earlier in that sentence) show that lazy models can't achieve certain generalizations, they also show that lazy models are capable of other kinds of compositional generalization.
- You may find [3] a relevant paper as it studies RNNs solving a task involving compositional generalization (transitive inference).
- You may find [4] a relevant paper, as it studies the impact of rich-regime learning on compositional generalization.
- I think there’s a typo here: “ while other studies have suggested that abstraction – which often requires learning compressed, lower dimensional representations – are more amenable to generalizable task performance” — maybe “abstraction” -> “abstract representations”?
- l. 265: Do you mean “at each timepoint”?

1. https://proceedings.neurips.cc/paper/2021/hash/691dcb1d65f31967a874d18383b9da75-Abstract.html
2. https://www.jmlr.org/papers/v25/24-0220.html
3. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011954
4. https://arxiv.org/abs/2503.09781

### Soundness
3

### Presentation
4

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
The paper studies how feature-specific representational dynamics shape compositional generalization in RNN models in tasks with evolving temporal structure. Using the temporally structured C-PRO task, it shows that rich learning regimes yield different temporal patterns of representational dimensionality compared to the lazy learning regimes, which correlate with better compositional generalization capability.

### Strengths
- The setup where tasks unfold over time and contextual information evolves is indeed underexplored in the compositional generalization literature. The paper addresses this gap.
- On the synthetic C-PRO task, the paper analyzes how rich and lazy learning regimes differ in their representation dynamics and connects these differences to their different compositional generalization capabilities.

### Weaknesses
- While the observed representational dynamics are empirically interesting, the paper does not explain the underlying mechanism driving such dynamics (for example, how context representations are suppressed during stimulus presentation but re-emerge during the delay period). The paper also does not clarify how these differences in representation dynamics between rich and lazy learning lead to differences in generalization capability.

- It would be helpful for the authors to discuss the related literature on rich versus lazy learning in more depth, and to clarify how their findings differ from or extend previous studies.

- It would be helpful if the authors could more explicitly highlight the main takeaway, particularly how the findings on the synthetic C-PRO task might generalize to more realistic or real-world scenarios.
- In the abstract, the authors state that prior works mainly focus on unimodal tasks. However, as I understand it, the present study also does not involve multimodal inputs. This may lead to confusion, and the authors may wish to clarify this point.

### Questions
- Why did the authors choose to use SGD for training instead of commonly used adaptive optimizers in deep learning such as Adam?
- Why did the authors focus on RNNs, rather than more advanced and widely used sequence models such as LSTMs or Transformers?
- Perhaps due to my limited background in cognitive science, I find certain parts of the paper hard to follow, especially:
  - The detailed setup of the C-PRO task. What are the six phases?
  - The meaning of several terms, such as neural representational dimensionality, conjunction (l190-191)
- Please also refer to the weakness.

### Soundness
2

### Presentation
2

### Contribution
2
