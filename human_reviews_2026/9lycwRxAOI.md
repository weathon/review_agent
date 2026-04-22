# Tracking Equivalent Mechanistic Interpretations Across Neural Networks

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Mechanistic interpretability (MI) is an emerging framework for interpreting neural networks. Given a task and model, MI aims to discover a succinct algorithmic process, an interpretation, that explains the model's decision process on that task. However, MI is difficult to scale and generalize. This stems in part from two key challenges: there is no precise notion of a valid interpretation; and, generating interpretations is often an ad hoc process. In this paper, we address these challenges by defining and studying the problem of interpretive equivalence: determining whether two different models share a common interpretation, without requiring an explicit description of what that interpretation is. At the core of our approach, we propose and formalize the principle that two interpretations of a model are equivalent if all of their possible implementations are also equivalent. We develop an algorithm to estimate interpretive equivalence and case study its use on Transformer-based models. To analyze our algorithm, we introduce necessary and sufficient conditions for interpretive equivalence based on models' representation similarity. We provide guarantees that simultaneously relate a model's algorithmic interpretations, circuits, and representations. Our framework lays a foundation for the development of more rigorous evaluation methods of MI and automated, generalizable interpretation discovery methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
- This paper studies the problem of “interpretive equivalence”, which is how to determine whether two models share the same interpretation of a behavior (and by extension implementation). They develop algorithms to measure the similarity of two interpretations of a model based on their representations, and evaluate their algorithms/theory on three tasks using both toy transformers and pretrained language models.
____
- The paper seems like a solid contribution towards making the evaluation of neural network interpretations more rigorous, and for evaluating the equivalence of interpretations across models.

### Strengths
- The paper is well written and its connection to previous work is evidently thought-through throughout the paper.
- There is a nice mix of theory and practice - providing intuition and showing the framework being applied to both toy settings with known ground truth and slightly more complex settings that have some level of prior understanding.
- The evidence that this framework can distinguish between implementations on the toy task n-Permutation setting is promising (Section 3.1, Figure 2(left)).

### Weaknesses
- The argument that interpretive equivalence allows us to study simpler/smaller models (Line 90) seems like a nice implication of this framework when the ambiguity between two models is high. However, I have two concerns with this: 
    - The first is that it seems desirable that ambiguity between the same model usually close to 1 for the same task, but it sometimes is not (e.g. GPT2-M w/ itself on IOI task). Does this mean the perturbation of GETIMPL is too aggressive, or what else might cause this?
    - Second, this is somewhat at odds with evidence from other work that some interesting capabilities seem to appear only in larger models (see e.g. [Brown et al], [Wei et al], [Schaeffer et al]). A recent example that comes to mind from the mechanistic interpretability literature is [Prakash et al] who note “we do not examine smaller models, as they are unable to coherently solve the … task”. How does your framework incorporate how well a model can solve a task? Do you have some performance threshold you decide before you choose to use its representations to compare to another model? While the tasks analyzed here are simple and I’d expect all the models you analyze can do them, it's unclear how model performance is factored in when understanding the similarity of implementation for more complex tasks, which seems like a current limitation of the proposed method.

- The GETIMPL procedure in Algorithm 1 lacks some preciseness that would help clarify my understanding of the Ambiguity computation. As I understand it, the GETIMPL is designed to produce a modified model with an equivalent implementation/interpretation. However, what gets returned seems to be the exact copy of the model you start with, rather than a modified version. It would be helpful to clarify whether the operations in GETIMPL are in fact supposed to be producing a modified copy of the model (with an equivalent interpretation/implementation), or whether it does actual return the same $h_{\theta}$ that is provided as an input.

- It's unclear to me how to interpret a middle value of "ambiguity". Does this mean that _some_ of the implementation is similar? What exactly does that even mean? While not having needing a specific interpretation to base the metric in can be a strength of the method in some ways (e.g. when ambiguity is high, or prior work has come up with "similar" interpretations, confidence can be high that the ambiguity score is meaningful) it seems like this can also be a limitation as without an interpretation, an ambiguity of "0.4" seems hard to get much out of. I could imagine applying targeted version of this same algorithm to validate specific functional roles of various components (measuring whether "mover heads" perform similar roles, or "induction heads" act similarly across models, etc.), but this requires a prior interpretation.

___
- Brown et al. [Language models are few-shot learners](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)

- Wei et al. [Emergent Abilities of Large Language Models](https://openreview.net/forum?id=yzkSU5zdwD)

- Schaffer et al [Are Emergent Abilities of Large Language Models a Mirage?](https://openreview.net/forum?id=ITw9edRDlD)

- Prakash et al. [Language Models use Lookbacks to Track Beliefs](https://arxiv.org/abs/2505.14685)

### Questions
Overall, I think the paper is pretty good. My main concerns are listed in the "weaknesses" section, and addressing those would help me feel more confident in the paper. Below, I share minor comments/questions.

- I’m not sure what the text at the top of Figure 1 is trying to communicate, can you elaborate?
- Line 284: When you say “GPT2” is this GPT2-xl or GPT2-small?
- Line 281: “interpretive similar” seems to be a new term. Is it just a weaker form of interpretive equivalence? It feels natural, but you might want to define it somewhere.
- It’s kind of surprising that the diagonal for Figure 2b is not higher than some off-diagonals (particularly for the gpt2 family) - any ideas what's going on there?
- Another further validation of this method might be to compute ambiguity for individual components with specific hypothesized roles, like induction heads. Have you tested whether your ambiguity score yields high values when only measuring it on component subsets (like induction heads on repeated text snippets)?

Minor Typos:
- Line 215 (footnote): “the two models do not the same interpretation” -> missing “share” (ie. do not __share__ the same…)
- Line 357: “abstrations” -> abstractions
- Line 411: “On the other,” -> On the other hand?
- Line 423: “simiarlity” -> similarity
- Line 946: “framemworks” -> frameworks

Papers:
- You may also be interested in [Miller et al.]’s analysis of evaluating and comparing circuits, which seems related to the problem you’re proposing and addressing here.

___
Miller et al. [Transformer Circuit Metrics are Not Robust](https://openreview.net/forum?id=zSf8PJyQb2)

### Soundness
3

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
3

### Summary
The paper introduces a novel framework for determining interpretive equivalence between neural networks, whether two models implement the same mechanistic interpretation without requiring explicit interpretation of either model. 
The core insight is nice and sidesteps identifiability difficulties in MI by solving a relaxed problem: two interpretations are equivalent if and only if the set of possible implementations are similar. The authors operationalize this through Algorithm 1 (AMBIGUITY), which leverages representation similarity across sampled implementations to estimate equivalence.

### Strengths
Novel problem formulation: The shift from "interpret each model" to "determine if models share interpretations" is valuable and practical as a conceptual idea.
Theoretical grounding: Main Results 1-3 provide some theoretical guarantees via lower and upper-bounds connecting representational similarity with interpretative equivalence.
The advantage of the algorithm 1 is that it is scalable and applied to some standard LLMs.
I appreciate that the authors put effort in the presentation of the work and the concepts.

### Weaknesses
Despite the framework being stated in very general terms with causal abstraction as its main language, it seems that Algorithm 1 is fairly tied to circuit (with GetImpl doing circuit manipulations basically).

The paper acknowledges that the linearity assumption is used and could be studied further in future work but this remains a fairly strong and important assumption. 
In particular, the findings of Figure 2 do not really prove the usefulness of the approach, to produce the same kind of heatmaps it is enough to distinguish model class (on the left panel) or model family (on the center panel), which intuitively is easy to do by looking at their representation, and most distances between representations would produce such heatmaps without making claims about interpretative equivalence. I think it would be more convincing to show cases where "standard (CCA-like)" representational similarity do not give the correct answer, and AMBIGUITY does. The POS tagging experiment is already a bit more convincing as it already compare against an alternative hypothesis.

I fear it might be easy to engineer counter-examples for Algorithm 1:
- false positives: It would be two neural networks whose representations are close to be linearly mappable to each other but who do different things (maybe even beyond different implementation of the same behavior, different behavior altogether). Experiment idea: Start with a network and task, train another neural network adversarially to behave maximally differently from the first one but constrained or regularized to enforce linearly mappable representations. Failure to produce such adversarial examples would be a strong demonstration of Algorithm 1's robustness to false positive.
- false negatives: It would be two neural network who do implement the same thing but it is not visible by linear inspection of their representations. Experiment Idea: if you apply invertible non-linear transforms at each layer of network to produce a new network, linearity is lost but functionality is unchanged. 

Convergence and stability of Algorithm 1:
I think the paper could engage more with the convergence of algorithm 1, it depends on finite sampling of the number of repeats n in AMBIGUITY. It would be nice to see convergence rates and stability (as measured by some notion of variance of the estimate)

### Questions
What are the convergence properties of Algorithm 1?
How would standard representation comparison methods fare for the tasks in Figure 2.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The aim of mechanistic interpretability (MI) is to discover the algorithms that are implemented by trained neural networks to perform tasks. This is a difficult problem, and one of the difficulties is that it is unclear what the right target of MI is: it is relatively rare that behaviour can be captured by “circuits” whose function is completely understood, if by “circuit” one means neurons and edges between them. Indeed many of the most promising approaches to MI now replace neurons by other degrees of freedom, such as SAE features.

The authors of the present paper propose a way of sidestepping this issue by trying to quantify when two neural networks compute in the same way, rather than trying to directly characterise how they compute. This is an interesting idea, which can be viewed as an evolution of the existing literature on representational similarity across models (see e.g. Klabunde et al “Similarity of Neural Network Models: A Survey of Functional and Representational Measures”).

The contribution of the paper comes in two parts: a theoretical part and an empirical part. In the theoretical part the authors propose precise definitions of various terms such as representation, interpretation, circuit and alignment, and prove some simple facts about their key measure of “ambiguity” and representation distance. In the empirical part they test their notion of ambiguity on three main settings: RASP programs, the IOI circuit in GPT2 and Pythia models, and GPT2 between different tokens.

### Strengths
* The idea of quantifying representational similarity across a circuit, and the associated metric of ambiguity, is an interesting approach that sidesteps some of the intractable parts of MI  
* The paper makes a serious attempt to ground the somewhat informal terms used in MI in the precise language of casual graphs  
* Includes an interesting baseline in RASP programs and applications to models up to 2.8B parameters, showing the feasibility of applying the method at scale  
* There is a significant amount of new terminology in this paper, but the authors do provide a detailed glossary which I appreciated

### Weaknesses
Minor weaknesses:

* I think the related work is not sufficient; in particular, there are not enough references to the existing literature on representational similarity across models which this paper seems to be extending in various ways. Maybe the authors could take a look at Klabunde et al “Similarity of Neural Network Models: A Survey of Functional and Representational Measures”.  
* I found the leftmost plot in Fig 2 to be a bit misleading, in the sense that the color coding here seems to refer to some threshold that is not specified rather than e.g. linearly representing the ambiguity as in the middle plot.  
* In several places, including (5.1), distance is defined between sets of variables where it seems that what is intended is some average over distances on actual inputs.

Major weaknesses:

* I am not convinced of the importance of the theoretical component of this paper. Could the authors elaborate more on the relation between the empirical and theoretical contributions?  
* In my view the main empirical contribution is the middle plot of Fig 2\. However I do not have enough details to evaluate how significant I find these ambiguity figures. At first glance the high ambiguity between Pythia models is interesting, but what are the precise circuits being used here? What are the ablations that are performed? It seems that the meaning of the ReprDist metric depends quite heavily on the details and strength of the ablation and I did not find these details.

### Questions
* 260: what are these interventions?  
* 288: what does “yield the same interpretation” mean?  
* 156: we sample implementations, but according to what distribution? It seems in practice what we do is just what Algorithm 1 says?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces the framework of interpretive equivalence (IE) in neural networks, which formalizes what it means for two neural networks to "do the same thing". The authors define notions of interpretation (high-level algorithmic process) and implementation (possible low-level realizations). Using the principle that models with non-distinguishable implementations must contain the same interpretation, the authors develop algorithms to check for interpretive equivalence, relying on the concept of ambiguity (are two implementations statistically different?). They validate their algorithms on several real-world models (Pythia, GPT-2) and tasks (permutation detection, IOI, POS/next token prediction) and ground their framework in the theoretical foundations of causal abstraction.

### Strengths
- This work focuses on an important and only recently acknowledged challenge in MI, which is the lack of rigor and solid theoretical foundations. The framework of interpretive equivalence bridges that gap by helping formalize what it means for two models to be equivalent in the algorithmic sense. The authors move away from most previous approaches that focus on a single interpretation (sometimes hard to define) and now focus on the much larger set of its implementations, which should make future automated discovery methods much more robust.
- The theoretical foundations were not checked down to the details, but they seem rigorous and strong. The authors build upon the framework of causal abstraction used in previous works and successfully formalize the concepts of interpretations, circuits and representations. In addition, the main theoretical results (1 and 2) provide necessary and sufficient bounds that connect IE to representation similarity, which is an important contribution. This brings much-needed mathematical precision to a field that often relies too heavily on empirical heuristics.
- The experimental validation is well-designed and compelling, progressing from toy tasks to real-world applications. The authors first demonstrate that their Ambiguity metric is well-calibrated using hard-coded RASP programs with known different interpretations on a toy task (permutation detection), then show the utility of their framework in comparing different models (GPT-2 and Pythia) on the IOI task. This confirms known results but shows the potential of IE to reduce the interpretation of large LLMs to smaller equivalent ones. Finally, the experiment in 3.3 relates next-token prediction (a complex task) to POS identification (which is much more well-understood), which shows that IE can be used to decompose complex problems into simpler ones.
- This is a rather complex work at the intersection of theory and practice, and effort has been put into making it rather accessible (presence of a glossary, high-level features).

### Weaknesses
- The success of the proposed Algorithm 1 seems to critically rely on the GetImpl procedure, which samples implementations from a model. This procedure is described at a very high level and could use more details. In particular, it is not clear how feasible and costly this procedure is for large LLMs. The quality of the Ambiguity score depends on how well these samples represent the actual set of all possible implementations. The paper does not discuss this limitation or the sensitivity of Ambiguity to the sampling process.
- The authors choose a metric based on linear transformability (drepr) for representation similarity. They mention that representations may however lie on complex and non-linear manifolds. This choice is common but could limit the framework's ability to detect more complex relationships. A discussion of how the framework would work with other types of similarity metrics would be useful.
- Ambiguity is not a binary score, but a graded one. It is not entirely clear how MI researchers should interpret these values. For example, what does an ambiguity score of 0.5 mean? Do the models share 50% of their algorithm? Considering the practical goal of the framework, the paper could include a discussion on how to choose proper thresholds or generally interpret these continuous scores.
- The introduced framework assumes that high representational similarity is evidence of interpretive equivalence, but other factors could lead to representational similarity (architectural biases, similarity in training data). The framework does not explicitly account for this.

### Questions
- How sensitive is Ambiguity to the number of implementations (n in Alg. 1) and the methods used to generate them (perturbations, deletions, etc.)? Have the authors explored how this scales with model size?
- How should one interpret graded values of the Ambiguity score?

### Soundness
4

### Presentation
3

### Contribution
3
