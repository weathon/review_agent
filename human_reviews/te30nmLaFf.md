# Teaching Transformers Causal Reasoning through Axiomatic Training

- Decision: Reject
- Scores: 3, 5, 6, 3

## Abstract
For text-based AI systems to interact in the real world, causal reasoning is an essential skill. Since active interventions are costly to execute, we study to what extent an agent can learn  causal reasoning from symbolic demonstrations of causal axioms. Specifically, we consider an axiomatic training setup where an agent learns from multiple demonstrations of a causal axiom (or rule), rather than incorporating the axiom as an inductive bias or inferring it from data values. A key question is whether the agent would learn to generalize from the axiom demonstrations to new scenarios. For example, if a transformer model is trained on demonstrations of the causal transitivity axiom over small graphs, would it generalize to applying the transitivity axiom over large graphs? 
Our results, based on a novel axiomatic training scheme, indicate that such generalization is possible. For the transitivity axiom, we find that a 67 million parameter transformer model, when trained on linear causal chains (along with some noisy variations) can generalize well to new kinds of graphs, including longer causal chains, causal chains with reversed order, and graphs with branching; even when it is not explicitly trained for such settings. We extend axiomatic training to a harder task of inferring causation from correlation statements and find similar generalization. On both tasks, our model performs at par (or even better) than many larger language models such as GPT-4, Gemini Pro, and Phi-3. Overall, our axiomatic training framework provides a new paradigm of learning causal reasoning in language models that can be extended to arbitrary axioms, as long as sufficient demonstrations can be generated.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper mainly presents a synthetic data generation protocol for axiom learning in transformers. Through training a GPT-2 like transformer on the synthetic demonstrations, experiments show good generalization ability when evaluating it on a larger network. The paper is well formatted and the axiom generation procedure is well explained.

### Strengths
1. Teaching LLM to reason is an important topic.
2. Using high-quality synthetic data for pertaining LLM is important and the explanations of the synthetic axiom generation are clear.

### Weaknesses
1. There is no proper explanation for why training the transformer on the given symbolic expressions will generalize better and even outperform the SoTA GPT-4 model. If my understanding is correct, the author only changes the dataset for pertaining and still uses negative log-likelihood (SFT) as the loss function to train the transformer.  Is it possible the model overfits the symbolic expression dataset the author has provided?
2. The author attempts to investigate how different embedding techniques, i.e., SPE, LPE, and NoPE, will affect the causal reasoning ability of LLM. As far as I know, most open-source LLMs like llama, they are all using RoPE [1] for positional embedding which is a kind of relative embedding suitable for format learning. I am curious why the author did not mention RoPE in this paper. Moreover, the evaluation of positional encoding is only limited to the final performance and I could not find any deep insight into why SPE will perform better. [2] is a good reference paper for probing different LLM layers and testing their effectiveness.

[1] Jianlin Su et al. RoFormer: Enhanced Transformer with Rotary Position Embedding

[2] Tian Ye et al. Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process

### Questions
There are some minor questions that need further clarification
1. Line 317 "We train a decoder-based 67 million parameter model based on GPT-2’s architecture". Does it mean the model in the paper is trained from scratch?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper investigates the problem of causal reasoning in transformer models. Specifically, authors try to come up with a different training protocol for enabling causal reasoning, which they call axiomatic training. The dataset for axiomatic training consists of a large number of demonstrations of (abstract) causal reasoning - in this work, primarily based on one specific axiom, transitivity. One such demonstration could look like "X causes Y, Z causes Y. Does Z cause X?" followed by an answer "No". After generating a large dataset of those types of prompts, varying lengths, variable names and causal graphs topology, a small transformer model is trained to predict the last token on different subsets of the dataset, to check what levels of generalisation it would reach. 
Trained model is compared with off-the-shelf LLMs (GPT-4, Phi-3, Gemini). Empirical comparison is extended further by looking at more general correlational statements from Corr2Cause dataset. The paper also investigates the impact of positional encoding on generalisation abilities. Given the results, authors suggest that similar techniques might inspire specific ways of synthetic data augmentation and help with LLM reasoning.

### Strengths
- The problem of reasoning, and, specifically, causal reasoning, is important and not very well understood yet, which makes the paper topical and interesting.
- The axiomatic training method suggested in the paper seems a priori reasonable - LLMs get access to many demonstrations of an axiom, but  augmenting the data in specific ways might yield better results. The fact that the model performs better, or on par, with off-the-shelf, cutting-edge LLMs is evidence that the method has some potential to improve training.
- Empirical testing of various levels of generalisation for the transitivity axiom seems solidly done, including retraining the model on a filtered data to test specific claims.

### Weaknesses
- Writing was a bit disjointed. The main thread of the narrative was to investigate generalisation of a specific training setup. But then there was also some (disjointed) discussion of positional encoding, which - although important and informative - was distracting from the main story. Some other pieces of writing were also a bit confusing, such as the transitions between sections 3.2, 3.3 and 4.1 (e.g. random flipping is described twice in 3.2 and 3.3, then branching is described twice in 3.3 and 4.1) - it might be good to consolidate those sections.
 - Constituting most of the paper - investigation into transitivity axiom alone seems underwhelming, in relation to the claims about teaching transformers causal reasoning. Transitivity check does not involve any substantial *causal* component - it is simply a question about graph reachability.
 - Thus, the most interesting section, and the only one that truly uses the heavy machinery of d-separation etc., is Section 5. But it is very short, and the full details are not even included in the appendix. Section starts by describing the task in the benchmark as "significantly harder" because of various potential questions, but this feels misleading, since the full benchmark is not used. There is no formal definition of what the "correlational statement" is, which that empirical evaluation measures - appendix A.2 simply lists all possible questions.
- The task is also quite underwhelming on a conceptual level - since the variables names are repeated, it is not clear to me to what extent the model could have just memorised some patterns, which are enough to get to 64% accuracy, as the data sizes are very small - only graphs with 3 or 4 nodes (respectively 6, 31 non-isomorphic DAGs) and testing on 5 nodes (302 non-isomorphic DAGs).
- It would be good to include a theoretical discussion as to whether we can expect transformer-based models to solve this task in principle - I don't know if the algorithm of checking d-separation can be implemented in a single pass of a bounded-depth circuit.
- In general, this sort of technique seems to quickly hit diminishing returns, as per the bitter lesson. It is not a general-purpose algorithm, as it requires very specific data augmentation for different logic and reasoning systems. I would be more interested in leaning deeper into trying to infer generalisation properties of the architecture.

### Questions
Questions:
- Recent research into decision transformers showed their significant issues with trajectory stitching (see e.g. Zhou et al. 2024, "Free from Bellman Completeness"). Do you think it has any relation to your problem of computing graph reachability?
 - Why do you think the model truly generalised in Sec 5, and not just remembered some patterns, given the very effective dataset size?
 - What is the definition of the correlational statement, and what is the complexity of the algorithm that answers it?
 - Do you think tokenizer issues might have impacted off-the-shelf LLMs abilities to perform on the task?

Small things:
 - Citations in line 347 and label in line 914 are not displayed correctly
 - Colours are inverted in Fig A1.

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
3

### Summary
This paper introduces an "axiomatic training" framework for teaching transformers causal reasoning through symbolic demonstrations of causal axioms. The key contributions are:

1. A method to generate synthetic training data containing demonstrations of causal axioms
2. Empirical evidence that a 67M parameter transformer trained on simple causal chains can generalize to more complex causal structures
3. Application to both transitive causal reasoning and inferring causation from correlation statements
4. The authors demonstrate that their model often outperforms larger language models like GPT-4 and Gemini Pro on these causal reasoning tasks.

### Strengths
1. Novel framework for teaching causal reasoning through symbolic demonstrations
2. Comprehensive evaluation across multiple generalization dimensions
3. Strong empirical results showing competitive performance with much larger models
4. Clear potential for extending to other axioms and reasoning tasks

### Weaknesses
1. Experimental Design Limitations:
- The fixed training regime of 100 epochs lacks justification and sensitivity analysis
- Insufficient ablation studies on model size impact
- Limited exploration of different positional encoding schemes' effects on performance
2. Baseline Comparison Issues:
- The baseline methods may not adequately handle noise and "synthetic data"
- Potential fairness concerns in comparisons due to the presence of contradictory causal relationships in training data
- The baseline methods might need adjustment to better align with the causal reasoning tasks
3. Semantic Consideration:
- Insufficient attention to the role of semantic information in causal reasoning tasks
- Limited exploration of how semantic understanding could enhance model performance
- Potential unexplored opportunities in leveraging Transformer's semantic capabilities

### Questions
1. Could you provide sensitivity analyses for:
- Training epochs
- Model size variations
- Different positional encoding schemes
2. How do you ensure fair comparison with baselines given the synthetic nature of the training data?
3. Have you considered incorporating semantic understanding mechanisms to enhance the model's causal reasoning capabilities?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper introduces a way of training Transformers from scratch to learn causal reasoning axioms. The idea, called axiomatic training, is to demonstrate axioms through labeled examples that encode whether or not a hypothesis is true given a premise. For example, the premise can be conditional independences entailed by a causal graphical model, the hypothesis, whether or not some variable is a collider. To study axiomatic training, the paper considers the transitivity of causal relationships as an axiom and develops an extensive experimental design that evaluates various training strategies on different forms of generalization at test time. The paper consistently finds that their trained Transformers outperform many LLMs, while performing comparably to GPT-4. As a second application, they use axiomatic training to study performance on queries about additional graphical relationships such as colliders, direct or indirect causes and find that axiomatic training again generalized favorably compared to several LLMs.

### Strengths
+ With the exception of a few expositional sentences, the paper is clear about definitions, training, and the experimental setup.

+ To the best of my knowledge, the paper considers a novel setup for training Transformers to apply logical rules, at least in the context of causal reasoning.

+ The empirical setup is thorough, identifying four ways of generating out-of-distribution premises at inference time and teasing apart the impact of multiple aspects of the training setup.

### Weaknesses
+ At the highest level, it would be useful to get more perspective on the significance of training Transformers from scratch to automatically learn to resolve logical causal statements. First, the empirical findings suggest that GPT-4 is pre-trained to handle these queries in generalizable way already. Second, even if axiomatic training were improved to outcompete GPT-4 on these structured queries, how can this technique be incorporated into the larger decision making context? Much of text-based decision making won't feature explicit premises like conditional independence statements but rather informal natural language.  Relatedly, although the paper does summarize related work, the authors can better contextualize the significance of their contribution in the broader context of LLM-based causal reasoning. They discuss some papers, and use the Corr2Cause benchmark -- in what way does this training procedure build on the promise of LLMs answering causal questions factually?

+ I found some sentences hard to process, potentially because they were using jargon and could have been put more simply. For example, in lines 44-45, in the context of causal models, the data-generating process arguably entail the axioms -- it's not that the axioms of d-separation entail a causal model -- and so the phrase "data that is result of axioms obeyed by a data-generating process" is hard to digest. In section 3 in line 131, it would have been useful if phrases like "causal principles" were explained better before being invoked. In lines 134-135, "finite set of axioms ... completely characterized by axioms of path interception ..." is again hard to parse. 

+ Relatedly, it's hard to link the transitivity axiom in (1) back to the logical formula given above it -- is the transitivity axiom supposed to be one of the six axioms you reference? 

+ Some technical details are missing. In assumption 3.1, $\mathcal{M}$ is conventionally used to define a structural causal model (SCM). Is that what you're referring to? If so, the paper ought to define a (probabilistic) SCM, and clarify phrases like "all possible probability distribution(s) over $\mathcal{M}$", which again don't parse. In section 5, when applying axiomatic training to identify additional graphical relationships, it's not clear how the examples are created since many graphical relationships (e.g., direct causation) are not identified from conditional independence statements. Thus, for many premises and hypothesis pairs, there is no way to arrive at a yes or no label. How is this handled?

+ Section 3 feels like it is both missing some details (see above) but also introducing a lot of notation without giving context. In particular, the build up of the conjunctive normal form culminates in Equation (1), which is never explained simply in plain English. But looking at it, it seems to just say that (indirect) causation is transitive. There is a big notation overhead to arrive at a fairly simple idea, which makes the lead-up feel like it's purposefully trying to create an air of complication that's not needed.

### Questions
+ Could the authors contextualize the uses for axiomatic training in the broader causal inference and decision making context to help shed light on its significance as a training technique?

+ Can you clarify if the transitivity axiom in (1) is linked to the previous logical rules by being one of the six axioms you reference? 

+ Can you clarify how examples are generated for section 5, given that many graphical relations in a causal model are not identified by conditional independences alone?

+ Can you provide some more plain English context about the transitivity axiom and its importance? It seems like it's a statement about the transitivity of indirect causation, where causation is codified through causal (ir)relevance. Is that fair to say?

### Soundness
3

### Presentation
2

### Contribution
2
