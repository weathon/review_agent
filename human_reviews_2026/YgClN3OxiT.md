# An AI Monkey Gets Grapes for Sure -- Sphere Neural Networks for Reliable Decision-Making

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 4, 2

## Abstract
By increasing the amount and the quality of training data, we may improve the logical reasoning performances of LLMs, but they are still unreliable and struggle with simple decision-making, an ability that animals may develop without language. 
This paper proposes a new version of Sphere Neural Networks that embeds concepts as circles on the surface of an n-dimensional sphere. This new version enables the representation of the negation operator through complement circles and can achieve reliable decision-making by eliminating unsatisfiable circle configurations. Comparative experiments with supervised neural reasoning are performed by retraining Euler Net for disjunctive syllogistic reasoning, the foundation for decision-making. We demonstrate that the proposed Sphere Neural Network achieves rigorous disjunctive syllogistic reasoning as well as 15 other syllogistic-style reasoning tasks, while preserving the rigour of classical syllogistic reasoning. In contrast, an Euler Net achieving 100.00% in classic syllogistic reasoning can be trained to reach 100% accuracy in disjunctive syllogistic reasoning. However, after that, its performances dropped to 6.25% in classic syllogistic reasoning, then subsequently dropped to 75.00%, 53.57%, and 46.43% when input images had random colour, random colour and boundary thickness, or filled circles, respectively. This comparison favours the method of neural reasoning with explicit model construction and suggests seeking alternative neural methods to enhance the reliability of neural decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies reliable neural decision-making by revisiting disjunctive syllogistic reasoning. The authors introduce an improved Sphere Neural Network (SphNN) that represents logical concepts as circles on the surface of an n-dimensional sphere, allowing explicit modeling of logical relations including negation through complementary circles. By constructing and inspecting circle configurations, the model determines the satisfiability of syllogistic statements and performs reasoning with certainty, without relying on training data. Experiments show that SphNN achieves perfect accuracy across 16 types of syllogistic reasoning tasks, outperforming supervised neural networks such as Euler Net, which show strong dependence on input patterns. Additional tests with GPT-5 models highlight the limitations of large language models in logical reasoning and emphasize the advantages of explicit model-construction methods for interpretable and reliable AI reasoning.

### Strengths
The paper’s strengths lie in its clear motivation, solid theoretical grounding, and comprehensive experimental validation. It tackles an important and underexplored question of how to achieve reliable decision-making in neural systems, grounding the study in both cognitive evidence and formal logic. 

The proposed Sphere Neural Network introduces a conceptually elegant geometric framework that enables explicit and interpretable reasoning through spatial model construction, and the experiments are extensive and convincing, showing consistent 100% reasoning accuracy across multiple syllogistic forms and dimensions. 

Overall, the work contributes a promising direction toward interpretable, data-free, and logically consistent neural reasoning, which is particularly relevant for safety-critical AI applications.

### Weaknesses
While the paper makes a strong conceptual and empirical contribution, it also has several limitations.

1. The paper focuses on syllogistic-style reasoning tasks that are intentionally idealized to allow for precise theoretical and geometric analysis. While this setting is appropriate for validating the rigor and interpretability of the proposed model, it also limits the scope of evaluation. The study does not yet demonstrate how the method performs under more complex or noisy reasoning scenarios that involve uncertain information, contextual ambiguity, or real-world data.

2. The evaluation mainly focuses on logical correctness and accuracy, but it offers limited discussion on robustness or computational efficiency compared with other reasoning paradigms such as neuro-symbolic or constraint-based approaches. For instance, the paper reports the mean time cost but lacks comparative runtime analyses. A more systematic benchmarking study would help clarify the practical trade-offs and scalability of the proposed method.

### Questions
1. In my view, the paper mainly focuses on syllogistic-style reasoning in idealized settings. How do the authors envision applying the proposed method to real-world decision-making tasks? What kinds of practical problems could this framework potentially address?

2. About runtime:
- Although the experiments have evaluated different n values up to 10,000, it remains unclear how the model’s runtime and efficiency change as the reasoning tasks become larger or more complex. Could the authors provide a clearer analysis of how scalability affects runtime and what adjustments, if any, are needed to maintain efficiency?
- Could the authors include a more detailed efficiency comparison against other baseline methods?

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
This paper proposes Sphere Neural Networks (SphNN) for logical reasoning. SphNN reason by explicitly constructing geometric models. It embeds concepts as circles on an n-dimensional sphere and represent the negation operation via "complement circles." Experiments show that it can rigorously solve 16 types of syllogistic reasoning with 100% accuracy. In a comparative study, a supervised Euler Net is shown to be vulnerable to input distribution shift. After being retrained for a new task, its accuracy on classic syllogisms drops significantly. The paper concludes that this model-construction approach is superior for achieving reliable and robust reasoning and decision-making.

### Strengths
1.	The paper focuses on an important problem, i.e., how to let a model conduct logical reasoning in a deterministic and reliable way.

2.	The proposed improvement over the original SphNN is novel and the illustrations in figures are clear.

### Weaknesses
1.	The model is restricted to highly structured input, e.g., syllogistic reasoning tasks. It is unclear how the proposed method would scale to more complex, real-world reasoning that involves ambiguity, common sense, or relational logic (e.g., "A is taller than B, B is to the left of C"). Moreover, the paper assumes that logical statements are already perfectly translated into formal representations. In reality, reasoning problems may come in the form of ambiguous natural language.

2.	The time complexity of the proposed method is not carefully discussed. The proposed method seems to deal with disjunction by testing each possibility. This is feasible for simple cases but would yield a high computational cost with many disjunctive clauses.

3.	The paper claims that there is “no need for training data.” This is technically correct for the proposed method. However, the model itself (including algorithms and transition rules) is entirely designed by human experts. This simply shifts the burden from data collection to complex, expert-driven algorithm design. The generalizability to more complex and non-structured input is also questionable.

4.	The paper demonstrates that a retrained Euler Net suffers from catastrophic forgetting. This comparative experiment is not convincing due to the following reasons. (1) The proposed SphNN cannot handle visual inputs (including corrupted ones) as Euler Net does, making this comparison somehow meaningless. It is more of showing “what Euler Net cannot do,” but not “what SphNN can do.” (2) The paper does not compare against modern continual learning techniques designed to mitigate catastrophic forgetting.

5.	Minor. The name of the proposed method “SphNN” is identical to the previous one in (Dong et al. 2024). It is better consider using a different name.

### Questions
1.	The paper suggests integrating SphNN with LLMs as a future direction. Are there preliminary thoughts on this direction?

Please see Weaknesses for other concerns.

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
The paper proposes a new version of Sphere Neural Networks that embeds concepts as circles on the surface of an n-dimensional sphere. The authors evaluate the method across 16 syllogistic-style reasoning types (and several dimensions up to 10,000), report perfect accuracy for SphNN on these tasks, compare with a retrained Euler Net supervised model, measure run-times, and include experiments with GPT-5 family as an additional comparison.

### Strengths
1.	Clear, structured mapping from logical relations to geometric constraints and a catalogue of 16 syllogistic types.
2.	Direct comparative experiments with Euler Net and GPT-5.
3.	Excellent and detailed drawing and presentation.

### Weaknesses
1.	Novelty concern: The paper does not clearly and rigorously state what technical advances are new vs prior work (Dong et al. 2024/2025). The paper devotes considerable space to the story's background and the monkey example, but fails to sufficiently highlight its own innovative points, leaving its unique aspects under-emphasized.
2.	Evaluation limitation. All evaluations are conducted through textual descriptions. Presenting the improvements achieved by the method in tabular form would be more intuitive.

### Questions
1.	Previous work has mentioned that HSphNN(Dong et al. 2025) can help improve GPT-3.5's performance. Does the method presented in this paper have similar applications?
2.	Large parts of the formulation, the neighbourhood transition map, and the “construct-by-descent” algorithm closely mirror the earlier SphNN work and the same research program. Could the paper more clearly delineate what is truly new and better justify its novelty beyond incremental representational tweaks?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
he paper proposes a custom neural network architecture for solving syllogistic reasoning problems. The architecture works by using a particular geometric formulation of logic in which propositions are represented as circles/spheres which indicate their meaning in set theoretic terms. This architecture is a generalization of a previously-introduced model. The authors test performance on a number of standard syllogistic patterns and show it performs well.

My primary concern with the paper is that it is difficult to follow and pitched to a fairly niche audience, which will make it unlikely to resonate with ICLR broadly. I recommend the paper be rewritten for a more general audience before publishing at ICLR. Alternatively, the authors might consider venues that have specific interest in these topics, e.g., IWCS or similar venues.

### Strengths
* The problem of if/how neural networks can account for classical logical reasoning is long standing and thus novel insights and architectures that speak to this are interesting
* The focus on this particular geometry of an embedding space is, from my understanding, fairly novel

### Weaknesses
My primary concern with the paper is that I found it hard to follow. I am not an expert in this particular type of architecture, but do consider myself to be reasonable well versed in logic and the application of NNs to classical logic and semantics problems (at least as much as the average ICLR attendee if not moreso). Therefore, if the paper was not "clicking" for me, I take it as a signal that it needs to be reworked in order to have impact. It seems that there is a real contribution here, but I admit I can't fully articulate what it is, because much of the background and intuition was missing from the writeup. I give some more specific questions below, but my general feeling is it should undergo a round of revision before publishing at ICLR or similar venues.

### Questions
* Can you provide more background on Euler net? I think it is fair to assume many readers won't have seen that prior work. In particular, can you offer more intuition for why the inputs are represented as diagrams? Why not in some generic symbolic language instead? Is the image representation necessary for the architecture to work, or is that just a specific design choice of prior work? This was confusing to me, as it seems that a logical reasoning engine should be agnostic to the format of the input, and rather operate in a more abstract space.
* Can you say more (in the main text) about how the models are trained? What type of data do they train on and what are the train/test splits like? It is easy for a neural net to overfit surface patterns of syllogisms and thus to appear to perform well without internalizing any deep reasoning.
* How does this architecture perform on tasks that aren't syllogistic reasoning? E.g., language modeling? It is important to report such things, as there isn't really a point of having an NN that can do formal logic if that is _all_ it can do--the appeal of logic-capable neural networks is that they can do logic but still do other more ``neural net-y'' things too. Otherwise, we'd just use a symbolic reasoning engine to do the logic.


Typos:
* cite in first paragraph should be to mitchell, not melanie
* line 41 Gvery instead of Every
* typo in this line: Sufficient empirical experiments advocate the model theory for reasoning that reasoning is a process of constructing and inspecting mental models
* In general, there are frequent typos and oddly-worded sentences, so I stopped taking note. Make sure it gets a careful proof read on resubmission

### Soundness
2

### Presentation
1

### Contribution
2
