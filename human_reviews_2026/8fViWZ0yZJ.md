# Discovering alternative solutions beyond the simplicity bias in recurrent neural networks

- Avg Score: 7.33
- Decision: Accept (Poster)
- Scores: 8, 6, 8

## Abstract
Training recurrent neural networks (RNNs) to perform neuroscience-style tasks has become a popular way to generate hypotheses for how neural circuits in the brain might perform computations. Recent work has demonstrated that task-trained RNNs possess a strong simplicity bias. In particular, this inductive bias often causes RNNs trained on the same task to collapse on effectively the same solution, typically comprised of fixed-point attractors or other low-dimensional dynamical motifs. While such solutions are readily interpretable, this collapse proves counterproductive for the sake of generating a set of genuinely unique hypotheses for how neural computations might be performed. Here we propose Iterative Neural Similarity Deflation (INSD), a simple method to break this inductive bias. By penalizing linear predictivity of neural activity produced by standard task-trained RNNs, we find an alternative class of solutions to classic neuroscience-style RNN tasks. These solutions appear distinct across a battery of analysis techniques, including representational similarity metrics, dynamical systems analysis, and the linear decodability of task-relevant variables. Moreover, these alternative solutions can sometimes achieve superior performance in difficult or out-of-distribution task regimes. Our findings underscore the importance of moving beyond the simplicity bias to uncover richer and more varied models of neural computation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces Iterative Neural Similarity Deflation (INSD), a method for overcoming the simplicity bias that often causes recurrent neural networks trained on neuroscience-style tasks to converge to the same fixed-point or low-dimensional dynamical solutions. The authors propose penalizing the linear predictivity between neural activities of new and previously trained networks, effectively forcing networks to explore alternative representational geometries and dynamics. They test INSD on three canonical tasks—context-dependent integration, 3-bit flip-flop, and MemoryPro—and show that it yields qualitatively different solutions characterized by oscillatory or high-dimensional dynamics rather than static attractors. These alternative networks occasionally perform better under challenging, out-of-distribution conditions, suggesting that the INSD procedure can reveal functionally distinct computational strategies.

### Strengths
I really liked this paper. There is a notion in computational neuroscience that certain tasks somehow "induce", out of necessity, a particular type of solution in a network. This paper shows that there are many different potential solutions to a particular task.

### Weaknesses
The examples are all toy. I would be extremely interested to see how this technique performs in larger-scale settings, such as when training SSMs and/or Transformers on Long Range Arena (or a similarly difficult benchmark).

### Questions
1. The paper focuses primarily on relatively simple, neuroscience-style tasks. Is there a specific reason the authors did not extend their analysis to larger or more complex settings, such as multi-task RNNs or higher-dimensional architectures? It would be valuable to understand whether the proposed approach scales effectively beyond toy examples.

2. The authors might consider situating their work in relation to recent theoretical ideas such as the [Contravariance Principle](https://arxiv.org/pdf/2104.01489) and the [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987), both of which posit that sufficiently difficult tasks drive convergent representational structures across systems. Could the proposed INSD framework be used to empirically probe these claims—for instance, by testing whether multiple, distinct dynamical solutions can coexist for more complex tasks where such representational convergence is expected?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose an algorithm for training a population of RNNs on the same task, in a manner that encourages diverse solutions. This is done by adding a cost term that ensures activity in the space orthogonal to the output is orthogonal to previous networks. They show this generates solutions that differ in both geometry and topology.

### Strengths
When RNNs are used as hypothesis generators, it is crucial to explore the space of hypotheses. This is usually done in a somewhat heuristic manner, and a more systematic approach is very important.

The idea is novel, original, elegant, and seems to work.

### Weaknesses
The other part of hypothesis generation is understanding the mechanism behind each RNN. This part was lacking. How exactly do the alternative RNNs solve the different tasks? In Turner 2021, the mechanism is explained. While this does not always have to be possible, it would be interesting to see a case where it is.

Are the alternatives more or less similar to experimental data?

### Questions
Line 148: this is somewhat similar to the argument of bijectivity in latent variable inference (Versteeg et al 2023, Dabholkar et al 2025).

What is the capacity to produce different solutions? How many can there be, and what does this depend on? For instance, Training a 1-bit flip flop can be easily done with a 2-unit GRU. Can the method work already there (or maybe in 3D), thereby allowing phase space analysis?

How robust are these to noise in the actual dynamics, or in connectivity?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The manuscript presents a new approach to train RNNs for the purpose of generating new hypotheses about neural mechanisms in neuroscience. Current approaches for training RNNs on neuroscience tasks tend to result in similar types of solutions, often relying on low-dimensional arrangements of stable fixed points and relatively simple dynamics around them. The authors’ use an iterative training procedure to find a family of RNNs that all solve a given task, but do so by implementing different dynamics and solutions. Specifically, each subsequent RNN in the family is trained with a loss that combines task accuracy with a penalty for solutions that are similar to those implemented by the previously trained networks. The authors apply this procedure to three well known and often used tasks of relevance to neuroscience and show that the resulting RNNs implement solutions that differ from those obtained with standard training approaches.

### Strengths
The paper is well written, clear, and technically strong.

Given the increasing use of RNNs as models of neural computations, the paper addresses an important question, and develops a creative approach that may increase the number of hypotheses that can be generated with RNNs.

The paper puts forward a new metric to compare the dynamics of different RNNs that overcomes some of the limitations of previously proposed metrics.

### Weaknesses
The dissimilarity measure employed by the authors to compare RNN dynamics overcomes some of the limitations of previously proposed measures, in particular by preventing RNNs to simply add irrelevant dynamics in new subspaces to otherwise unchanged dynamics analogous to that in the standard RNN solution. However, it is not clear to me whether their dissimilarity can prevent more subtle ways to superficially modify the dynamics without changing its essence.

For example, one could imagine a scenario where an alternative solution to the context-dependent integrator appears much more high-dimensional and “curved” than in the standard solution, and yet locally the nature of the interaction between the inputs and slow dynamics are preserved. If such a solution was possible, a comparison of the trajectories alone would not adequately capture the fact that locally the solutions are quite similar in essence. 

More generally, whereas the standard solutions are well understood at a variety of levels of description (e.g. the local relative arrangement of inputs, left and right eigenvectors in the context dependent integrator), the authors do not reach the same level of understanding for the alternative solutions. Without such a full understanding, it is challenging to decide how “different” the various solutions really are. 
It does not help that the alternative RNN produce dynamics that seems substantially more complex than that in standard RNNs, and so understanding the underlying mechanisms (which ultimately is required to generate new hypotheses about brain function) may be more challenging. 

The authors’ approach for generating alternative solutions is quite heavy handed, in that it forces solutions with very different trajectories, whereas it is known that the different mechanisms can produce very similar trajectories (recent Pagan et al work). Presumably, these different solutions could not be found with the approach presented in the manuscript.

### Questions
Can the authors show in more detail how the alternative RNN solve their tasks? In particular, can they explain why the additional dynamics that distinguishes them from the standard networks is required to solve the task? 

Standard RNNs can often be understood with fixed points and dynamics around them, and the authors argue that the same descriptions to not apply to the alternative RNNs. But then how can these be understood and used to generate testable hypotheses?

Can the authors’ approach be used to identify the various solutions to context dependent integrations found by Pagan et al?

### Soundness
3

### Presentation
3

### Contribution
3
