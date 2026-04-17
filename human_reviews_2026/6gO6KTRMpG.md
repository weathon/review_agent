# Language Models Use Lookbacks to Track Beliefs

- Decision: Accept (Poster)
- Scores: 0, 6, 6, 6

## Abstract
How do language models (LMs) represent characters’ beliefs, especially when those beliefs may differ from reality? This question lies at the heart of understanding the Theory of Mind (ToM) capabilities of LMs. We analyze LMs' ability to reason about characters’ beliefs using causal mediation and abstraction. We construct a dataset, CausalToM, consisting of simple stories where two characters independently change the state of two objects, potentially unaware of each other's actions. Our investigation uncovered a pervasive algorithmic pattern that we call a lookback mechanism, which enables the LM to recall important information when it becomes necessary. The LM binds each character-object-state triple together by co-locating their reference information, represented as Ordering IDs (OIs), in low-rank subspaces of the state token's residual stream. When asked about a character's beliefs regarding the state of an object, the binding lookback retrieves the correct state OI and then the answer lookback retrieves the corresponding state token. When we introduce text specifying that one character is (not) visible to the other, we find that the LM first generates a visibility ID encoding the relation between the observing and the observed character OIs. In a visibility lookback, this ID is used to retrieve information about the observed character and update the observing character's beliefs. Our work provides insights into belief tracking mechanisms, taking a step toward reverse-engineering ToM reasoning in LMs.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
1

### Summary
n/a - see ethics review

### Strengths
n/a - see ethics review

### Weaknesses
n/a - see ethics review

### Questions
n/a - see ethics review

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper claims evidence for a specific mechanism, the "lookback", that allows a transformer architecture to dereference information about an entity in a sequence of tokens that presents a Theory of Mind problem, so as to correctly answer queries about what the entity knows. The paper's central claim is that the encoding of such information in the residual stream is based on the relative ordering of tokens corresponding to entities in a set of relational statements (first character vs. second character, first object vs. second object), not on structures that encode information as attributes of the entity's identity. The authors present a benchmark dataset (CausalToM) to analyze a model's ability to simulate ToM, and show through causal interventions in the layer-by-layer evolution of the residual stream that ordinal position (encoded as "Ordering IDs"), as opposed to identity, is how the LM manages information to correctly answer ToM queries.

### Strengths
The paper addresses an extremely challenging and important problem in understanding the internal mechanisms by which language models perform Theory of Mind reasoning. 

The methodology used to address this question is very clearly laid out, and to this reviewer's mind well motivated. The paper provides clear and useful graphical presentations of the mechanism proposed and the results of the interchange interventions in both the no-visibility and visibility cases. The contribution of a structured dataset of simple stories to provide a way to effectively elicit interpretable responses to confirm or disconfirm the Ordering IDs hypothesis is also an important contribution. 

The findings demonstrate consistency across models in a given model family (Meta Llama) at multiple sizes (70B and 405B), and preliminary evidence suggests the mechanism generalizes to more naturalistic scenarios (i.e., the BigToM benchmark), strengthening confidence in the robustness of the identified patterns.

### Weaknesses
The paper would be significantly strengthened by addressing the following issues related to soundness and presentation. This reviewer sincerely hopes that these can be satisfactorily addressed in the rebuttal phase. 

1) The paper's central claim that models use Ordering IDs rather than identity-based or semantic representations is not adequately distinguished from plausible alternatives. There is only the briefest mention of prior work, and it assumes a great deal of familiarity from the reader with what appears to be a very specific body of work. The paper does not explain how the Ordering ID hypothesis was developed, what alternatives were considered, or whether there was exploratory analysis not reported. This makes it very difficult to assess the proposed mechanism in the context of work in mechanistic interpretability to date. 

2) The paper lacks crucial information about computational requirements, experimental iteration (how many analyses were attempted before arriving at reported results), and robustness checks (sensitivity to hyperparameters, sample selection, random seeds). Code availability is not mentioned. 

3) The paper uses specialized mechanistic interpretability terminology ("residual stream," "QK-circuit," "OV-circuit") without adequate definition, assuming familiarity that may not be universal even among the ICLR community. While core concepts like interchange interventions are explained well, a brief background section defining key architectural concepts would significantly improve accessibility and aid reproducibility.

### Questions
Does reducing ToM to positional bookkeeping (lookup mechanism + Ordering IDs) suggest sophisticated behavioral mimicry (à la Searle's Chinese Room thought experiment) rather than understanding or intentionality? What additional evidence would demonstrate conceptual understanding beyond structural pattern extraction? Are the Ordering IDs grounded in meaningful semantic relations about information access and belief formation, or are they arbitrary indices that correlate with correct answers in this constrained task structure?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the internal mechanisms by which LLMs track characters' beliefs in ToM tasks.  The authors construct CausalToM, employ causal mediation analysis and causal abstraction to identify systematic computational patterns. Three specific lookback mechanisms are identified: (1) binding lookback that links character-object-state triples via ordering IDs, (2) answer lookback that retrieves state token values, and (3) visibility lookback that updates beliefs based on character observability. The mechanisms are validated through interchange intervention experiments on Llama-3-70B-Instruct and Llama-3.1-405B-Instruct models.

### Strengths
Unlike previous works in the Theory of Mind (ToM) domain, such as prompt-based (Think twice, TimeToM), tool-based (Social world model), or model-based approaches (Bayesian framework), this paper analyzes the model’s belief reasoning ability from a novel and interpretable perspective. In ToM research, there has long been debate over whether models’ ToM abilities are truly robust, and whether a correct answer to a ToM question genuinely reflects capabiltiy level. Analyzing this issue from the viewpoint of interpretability offers a promising path toward resolving this controversy. The paper presents excellent visualizations and provides a clear description of the research background.

### Weaknesses
The data pattern of CausalToM mentioned in the paper is quite simple.

Theory of Mind (ToM) is a broad framework encompassing various dimensions of mental states, and its scenarios are often diverse and complex. The interpretability analysis in this paper is applied only to a narrow data scope (simple story settings and the belief dimension). When the data scenarios become more complex (e.g., longer narratives or richer social contexts), can this method still maintain good scalability and generalization?

Moreover, the interpretability analysis is conducted only on the LLaMA series models. Will these observed phenomena also appear in other model families?

### Questions
See Weakness.

The models used by the authors are 70B and 405B. Would the same phenomena described in the paper also appear if the model size were around 7–32B?

### Soundness
3

### Presentation
4

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
This paper asks a concrete mechanistic question pertaining to ToM: how transformers store, update, and retrieve characters and their states. The dataset is CausalToM, a toy two‑sentence story set with two characters, two objects (containers), and two object states (contents). Each example concludes with a question, such as “What does Bob believe the bottle contains?”, accompanied by optional visibility statements that specify who can observe whom. The authors analyze Llama‑3‑70B‑Instruct using interchange interventions on residual activations (i.e., patching counterfactual activations) to observe how the model’s behavior changes.

The central finding is a pervasive lookback mechanism. The model “writes” tags (my terminology; referred to as OIs in the paper) for the “first/second” character, object, and state into the residual stream. Further, state tags are bound to the appropriate character/object tags at the state token, and the model then “looks back” from the answer position to retrieve (i) the right state tag and then (ii) the actual state token via attention. The lookback is localized layer-wise (e.g., tags form approximately layers 20–34; binding occurs at the state token, approximately layers 33–38). Figures 1, 3, 4, and 7 illustrate the pointer/address/payload flow and the three lookbacks.

### Strengths
Overall, the results are surprising and insightful. The analysis is not hand-wavy.

* Strong causal methodology. The authors use interchange interventions (activation patching) with carefully matched counterfactual stories to manipulate specific internal variables and measure IIA layer by layer. For example, patching the final “Answer:” token at mid layers redirects the answer pointer (layers 34–52), whereas patching late layers swaps the answer payload (at layers> 56).



* Careful dataset design for causal analysis. CausalToM is deliberately simple (two characters/objects/states), so counterfactuals differ in only one factor at a time.

### Weaknesses
1. Analysis restricted to successful cases. All mechanistic experiments are run on 80 correctly answered examples. This risks selection bias: we only study the circuit when it worked. What happens for incorrect cases?


2. Scaling beyond “first/second” is unclear. tags/OIs encode first vs. second character/object/state, which is perfect for this dataset, but what happens as you scale up?

A small toy study (e.g., 3+ entities per type) would help address both weaknesses by revealing failure modes and testing whether the mechanism extends beyond binary order.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
