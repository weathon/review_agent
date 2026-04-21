# A Pattern Language for Machine Learning Tasks

- Avg Score: 3.75
- Decision: Reject
- Scores: 3, 6, 3, 3

## Abstract
We formalise the essential data of objective functions as equality constraints on composites of learners. We call these constraints "tasks", and we investigate the idealised view that such tasks determine model behaviours. We develop a flowchart-like graphical mathematics for tasks that allows us to;
(1) design and optimise desired behaviours model-agnostically;
(2) offer a unified perspective of approaches in machine learning across domains;
(3) import insights from theoretical computer science into practical machine learning.
As a proof-of-concept of the potential practical impact of our theoretical framework, we exhibit and implement a novel "manipulator" task that minimally edits input data to have a desired attribute. Our model-agnostic approach achieves this without the need for custom architectures, adversarial training, random sampling, or interventions on the data, hence enabling capable, small-scale, and training-stable models.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes a "pattern language", which treats machine learning tasks as constraints that specify desired equivalences.
Some commonly used tasks (e.g. classification, autoencoding, GAN) are called "patterns", borrowing terminology from software engineering.

The pattern language abstracts away implementation details (e.g. architecture and training), by treating neural networks are universal approximators that can perfectly solve the task.

The paper uses a task called "manipulation" as a proof-of-concept. The paper shows results on manipulating attributes, such as interpolating between shapes and colors of synthetic images, and smile/no-smile on CelebA.

### Strengths
This proposed pattern language with string diagrams provides a type of high-level abstraction of the machine learning pipeline, which can be helpful for conceptual understanding and can help provide intuition for someone who is entering the field.

### Weaknesses
Although providing a clear mental picture, the benefit of this framework is not clear to me.
- I'm not sure how the string diagrams can help advance theoretical/mathematical understanding; it would be helpful if the paper could provide some concrete examples.
- I don't see how the proposed pattern language improves empirical results either. For example, my understanding is that composing tasks is equivalent to adding regularization terms, which has been commonly used in practice.

I'm also concerned about the audience. I feel that the intended audience of the paper is not machine learning researchers/practitioners, but rather people with a software engineering or programming language background.
- This is partially reflected in the choices of terminology and notation. For example, this paper uses $\mathbb{P}$ to denote datatypes (e.g. of trainable parameters), which is an uncommon choice since $\mathbb{P}$ is usually used for probability-related quantities. Please see the clarification questions below for more examples. The concepts such as "symmetric lenses" are unfamiliar to most ML audience and would require more explanation.

### Questions
Regarding the significance of the results, could you explain how the proposed pattern language can help with theoretical understanding or empirical advances?
- For example, it's unclear to me how "(composite) tasks" differ from multitask learning or adding regularization terms.

Some clarification questions:
- For specialised tasks, "structural inductive biases" refer to parameterization?
- line 188: "energy minimisation architecture": is "architecture" the same as  "pattern", i.e. a particular composition of tasks?
- Prop 3.4: the meaning of "balanced" in "balanced attribute" has not been explained (until the appendix).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work takes an idealized view of model optimization to formalize "task" in terms of a diagram that captures the behavior objective of the model being trained. This formalism is intended to provide better understanding of tasks as well as their relationships to other tasks. Using this formalism, they provide diagrams for several well known algorithms (such as classifiers, gans, and auto encoders), as well as formulate a relationship between auto encoding and energy minimization. Using their specification framework (i.e., "language"), they construct a possibly novel task called "manipulation" and demonstrate its efficacy on toy tasks.

### Strengths
In general, the work is well written and easy to understand. The motivation of introducing the formal language for describing tasks makes sense and the applications are well delivered in a number of common or well-known objectives. Generally this sort of formalism could make it easier to relate different algorithms w.r.t. their target behavior and system specification. The novel training paradigm I believe is novel (manipulation), which is convincingly yielded from the formalism and its efficacy is well established through experiments.

### Weaknesses
Most of the issues are on clarification, for which I have added questions below. 

Some of the symbols could be better presented, as I continuously had to refresh myself on what the numerous symbols meant while building an understanding of the formalism. Perhaps a table or some other sort of summary that helps the reader reference while looking through diagrams.

One major issue I have is it's unclear how the formalism specifically relates different algorithms w.r.t. the target behavior of the model. Is there a set of rules that allow the user to transform and reduce to a "unique" diagram that represents a group of tasks with similar target behaviors? Is uniqueness part of what this formalism brings (which makes finding relations easier), or can be never guarantee a unique representation of the task (e.g., there are equivalences which are not deterministically reducible to the same representation).

The end complaint here is for me the clearest demonstration of usefulness would be more concrete examples on how to relate / compare algorithms using operations on the language. Is this something the language provides? If not, are there reasons why it can at least help relate algorithms?

### Questions
1) Could you clarify what the role of the compound function, alpha, is? It appears to have something to do with the gradients of the parameters, but either way this should be made explicit or given an example in the work.
2) Could you clarify what the domain and codomain are? From what I can tell, these are the inputs and outputs of the diagrams (left to right), but this could be clearer, either way.
3) is "system" and "specification" synonymous with "(trained) behavior" and "target" (behavior)?
4) In Pattern 2.8 should there be a "y" in the domain of the specification?
5) why is "get" above the arrow for task 3.2 PutPut?
6) Is there a way through schematic manipulation to show equivalence between tasks? Eg between manipulation and strong manipulation it's implied they do similar things, but their diagrams are quite different. How would a practitioner understand that they are doing similar things? (perhaps some of this is in B2,3, but I didn't get a chance to look carefully at the Appendix)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
The article describes a formal framework for defining objective functions: these are modeled as a weighted sum of several statistical divergence functions (e.g., log-likelihood, cross-entropy, etc.). Specifically, a diagrammatic language expressing equivalence between diagrams is used to model *atomic tasks*. In contrast, their linear combination (using user-provided scaling coefficients) is referred to as a *compound task* (or objective function).

To showcase the practical advantages of this formal language, the authors propose a style-transfer objective function obtained by borrowing concepts from the Bidirectional Transformation field.

### Strengths
The language provided could be used to succinctly express complex loss function, using a relatively intuitive and succinct grammar. The utilization of a standard notation for referring to minimization objectives could prove useful, as similar concepts have been adopted in computer science.
Furthermore, the formal nature of the grammar lends itself to using formal verification techniques to look for properties of interest in the objective function. How would be done in practice however I am still not sure.

### Weaknesses
- The article is sometimes difficult to understand and unpolished; this is particularly true for sections 1 and 2. They are difficult to read and the vocabulary is not always defined (e.g. what is a datatype, what does ∶⇒ mean? ). As it stands, the paper can be confusing to read, this might be due to unpolished writing and a foreign vocabulary from the one typically used in the ML community.

- The proposed language could be more intuitive, at least from a ML prospective; this may hinder its usage in the broader ML community.

- I am unsure if the benefits of this pattern language are a strong enough contribution for a publication at this venue. In general, the topic tackled by this work seems to not be a good fit for ICLR. A computer science conference or journal might be more pertinent.

- While I understand the necessity to abstract away the choice of architecture, training schedule, and similar hyper-parameters, more often than not they are crucial for the successful estimation of a ML model. The lack of any type of modeling for these components makes me doubtful of the actual practicality of this pattern language.

### Questions
1) How would the proposed language formalize size-varying data: For example, how would you represent the autoregressive cross-entropy loss commonly adopted by large language models?

2) What were actually the objective functions used during training for the various experiments? It would be nice to have a mapping from the diagrams shown in Task 3.1 to a conventional mathematically expressed objective function.

3) How were the task weights reported in Appendix D estimated? Were they chosen arbitrarily or a hyperparameter search was performed?  

4) What does the text above the ∶⇒ symbol mean? Does it indicate the parameters optimized by the loss function?

5) What does the ∶⇒ symbol formally mean? Is there a definition in the appendix that I have missed? Is it a bidirectional relation?

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This work introduces a diagram-based language for creating and manipulating ML models with the notion of the task as the fundamental unit of computation. Several types of common models are given in terms of this framework such as autoencoders, classifiers, and in general, manipulators. Experiments are performed on manipulation (attribute selection and editing) for MNIST, a sprites-like dataset of shapes, and CelabA. it seems that the proposed method effectively can perform disentanglement and attribute editing to some degree.

----------------------------------------------------------------------------
I believe the authors provided a good faith effort to address concerns raised here. Therefore I will update my score(s) accordingly.

### Strengths
Originality: I think there is novelty in the category-theoretic foundations of this framework. I haven't seen this diagram notation before. 

Clarity: I think the writing was easy to follow, my concerns below notwithstanding.  

Significance: its clearly very early work, possibly a more developed version of this work could be impactful.

### Weaknesses
I found this paper extremely difficult to parse for content. It seems like there is something here, but it is hardly even hinted at in the main paper. I did not thoroughly check the proofs in the appendix so possibly I missed some important details there. But in general I fail to see what is actually being proposed here. My concerns are as follows:

* Over half of the main paper is spent restating some form of empirical risk minimization, and reintroducing commonly used notation and modeling paradigms like autoencoders and classifiers. In this exposition I didn't find anything new except different jargon e.g. the operands of the divergence measure called sys and spec. 

* I'm unsure of what was even done for the experiments. Was something implemented? Explaining how this framework gets turned into an algorithm or implemented would be useful. Right now all we have is a small diagram and then results which is impossible to interpret. 

* Much of the nomenclature used here like get, putput, fake, fool, undoability is confusing at best. I don't see how this particular exposition is useful or gets the reader any closer to understanding the framework. 

* I found 3.2 to be a very obscure experiment and dataset. Can we do anything else but attribute editing (manipulation)?

### Questions
* In what sense are the given CycleGAN generators optimal? Is this optimality with respect to style transfer in general or just CycleGAN?

* What role do these diagrams play? I don't find them in any way intuitive if they are purely illustrative. 

* What is Undoability? I can't parse the diagram well enough to be sure. 

* How does this framework generate new or better components / tasks; are there any examples of improvements over the standard approach?

* Why is attribute editing chosen as the only task for this work? The first sections of the paper belabor the point that all kinds of modeling decisions can be expressed in this framework. So why don't we have autoencoders and classifier examples?

### Soundness
3

### Presentation
1

### Contribution
2
