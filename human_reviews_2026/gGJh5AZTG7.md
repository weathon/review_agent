# ActivationReasoning: Logical Reasoning in Latent Activation Spaces

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 2

## Abstract
Large language models (LLMs) excel at generating fluent text, but their internal reasoning remains opaque and difficult to control. Sparse autoencoders (SAEs) make hidden activations more interpretable by exposing latent features that often align with human concepts.  Yet, these features are fragile and passive, offering no mechanism for systematic reasoning or model control. To address this, we introduce ActivationReasoning (AR), a framework that embeds explicit logical reasoning into the latent space of LLMs. It proceeds in three stages: (1) Finding latent representations, first latent concept representations are identified (e.g., via SAEs) and organized into a dictionary; (2) Activating propositions, at inference time AR detects activating concepts and maps them to logical propositions; and (3)Logical reasoning, applying logical rules over these propositions to infer higher-order structures, compose new concepts, and steer model behavior. We evaluate AR on multi-hop reasoning (PrOntoQA), abstraction and robustness to indirect concept cues (Rail2Country), reasoning over natural and diverse language (ProverQA), and context-sensitive safety (BeaverTails). Across all tasks, AR scales robustly with reasoning complexity, generalizes to abstract and context-sensitive tasks, and transfers across model backbones. These results demonstrate that grounding logical structure in latent activations not only improves transparency but also enables structured reasoning, reliable control, and alignment with desired behaviors, providing a path toward more reliable and auditable AI.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes ACTIVATIONREASONING (AR), a framework that maps sparse autoencoder (SAE) features in LLMs to logical propositions and performs forward-chaining deduction over them to compose higher-level concepts and steer generation. Pipeline: (1) build a concept dictionary from SAE features (single/multi/relational representations), (2) detect token/sequence-level activations with soft thresholds, (3) apply logic rules to infer new propositions (A → A′) and optionally steer model activations. Experiments on PrOntoQA, a new Rail2Country benchmark (explicit vs. simile color cues), ProverQA, and BeaverTails safety show large gains over base and some larger/“reasoning” models, reportedly maintaining accuracy as reasoning depth increases and improving safety detection; authors claim small runtime overhead.

### Strengths
S1. Modular pipeline: concept discovery (via SAEs), activation aggregation, logic layer, and steering—each replaceable/ablatable.

S2. Strong multi-hop robustness; recovery of implicit/meta cues (similes); safety composition via logic; low runtime overhead vs. CoT/reasoning models.

### Weaknesses
W1.  The method is fundamentally deductive: it thresholds SAE features into propositions and applies forward-chaining rules. But domain-general reasoning (esp. scientific reasoning) is largely inductive/abductive—forming and revising hypotheses under uncertainty, trading off competing explanations, and learning new rules from sparse evidence. Here, rules are user-provided or templated; there’s no rule induction, hypothesis revision, or probabilistic belief update. The evaluations don’t demonstrate transfer to open-class concepts, concept drift, or out-of-ontology settings where the correct rule schema is unknown. As a result, the approach risks brittleness outside closed-world toy tasks: once a needed predicate or relation is missing (or mis-grounded by the SAE), the system cannot generalize by learning—it can only fail silently or require manual rule engineering. For a claim of “domain-general” promise, I’d expect ablations with (i) novel domains without preencoded rules, (ii) noisy/contradictory evidence with uncertainty propagation, and (iii) an inductive component (e.g., differentiable rule learning, Bayesian updates, or abductive reasoning) that improves with more data rather than more hand-written logic.

W2.  The paper compares AR to CoT and “reasoning models” but runs everyone with greedy decoding and without widely-accepted best practices (e.g., self-consistency, majority voting, verifier reranking, program-of-thought, retrieval-augmented logic parsers). A fair test would include: (i) symbolic parsers + classical prover, (ii) pure-symbolic systems on the same rule input AR gets, (iii) LLMs given the same normalized rule form.

W3.  Multiple places admit selecting features/hyperparameters on evaluation data (e.g., Rail2Country: “features were selected on the evaluation set” and thresholds tuned on eval/test splits). This is classic double-dipping that inflates reported gains. A rigorous protocol would fix search on train only, lock thresholds, and report test once.

### Questions
Can you freeze a single, pre-registered recipe (feature search, thresholds τ, top-k, α, steering vectors) learned on train only, and report test once with CIs across seeds? Provide ablations showing performance sensitivity to each knob.

### Soundness
2

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
2

### Summary
This paper proposes ACTIVATIONREASONING (AR), a novel framework for embedding **explicit logical reasoning into the latent activation space of LLMs. AR leverages SAEs  to extract monosemantic latent features that align with human-interpretable concepts and treats these features as **logical propositions**. The framework operates in three stages. This enables both transparent analysis of model internals and direct control over model behavior through concept-level interventions. Empirical evaluations on four benchmarks demonstrate that AR improves multi-hop reasoning, robustness to abstract or implicit cues, and context-sensitive safety control, often outperforming even much larger LLMs and reasoning-tuned models.

### Strengths
1. AR introduces a clear and principled framework that connects latent activations with symbolic logic, moving beyond typical mechanistic interpretability.
    
2. Results across diverse reasoning and safety tasks consistently show large gains, often surpassing significantly larger or reasoning-tuned baseline.
    
3. AR offers auditable reasoning and explicit model steering, addressing transparency and safety.

### Weaknesses
1. The proposed approach inherits limitations from sparse autoencoders. If latent features are polysemous or unstable, reasoning performance and interpretability may degrade.
    
2. It remains unclear how AR performs under high-dimensional activation spaces or longer contexts.
    
3. It would be valuable to discuss integration with probabilistic or continuous reasoning paradigms, which may align more naturally with neural representations.

### Questions
1. How scalable is AR when applied to large-scale tasks or long-context models, given the need to store and process activation matrices and logical rules?
    
2. Since the approach relies on SAEs trained independently from the LLM, how sensitive is AR’s performance to the specific choice or layer placement of the SAE?
    
3. Could the framework generalize to other modalities (e.g., vision–language models), where concept discovery and rule definition are more complex?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces ActivationReasoning (AR) a novel framework to embed logical reasoning over the latent representations of an LLM. AR finds latent concept representations, map them into logical propositions and finally apply logical inference. The framework is evaluated over different reasoning tasks, ranging from multi-hop reasoning to context-sensitive safety, with promising results that improve transparency  and reliability of the model.

### Strengths
- The paper is well written and clear to follow.
- The introduced framework is very valuable and the problem studied very important for the advancement of Large AI models in general.
- Extensive experimental evaluations.
- Interesting discussion of AR's limitations.

### Weaknesses
- The rules need to be known in advance, and this may restrict the model applicability, as this is not often the case. Also several rules in real-world settings are only partially true, and then applying forward chaining may introduce some errors.


MINOR COMMENTS

- "On our novel Rail2Country dataset (..)", I think this dataset could be explicitly mentioned as a new contribution of the paper.
- Fig. 2 is referenced before Fig. 1 at page 2, while it only appears on page 4.
- "Only a few provide a formal mechanism to compose concepts into higher-level abstractions or to disambiguate polysemous
features." Here it is seems some references are missing, or it misses an explicit reference to the ones mentioned above. Moreover, I think some other relevant examples of CBMs using logic rules to connect concepts to higher-level downstream tasks, like e.g. [1,2], can also be commented. This is also connected to the paragraph on systems integrating concept-level representations with reasoning modules for interactions and explainability.
- "Neuro-symbolic AI (...) " I'd mention some more recent work, like e.g. [3]
- "A Relational-feature representation (R relation) defines a concept by a set of relations among SAE features". Could you please explain/extend more how Relational features work? I understand that e.g. the concept "hate" is connected to multiple features, like the feature connected to the concept "slurs" and another feature connected to the concept "demeaning stereotypes". Hence, it's not clear to me how it is concretely different from multi-feature representations.
- "For example, features consistently activating for the notion of a Bridge can be directly assigned." Not clear how.. could you please better explain this? You mean by inspection over all the possible activations? I think in this case it can be unfeasible for large representation spaces.
- While not strictly necessary as the paper is very clear, I think to make the paper more self-consistent a brief section about background, e.g. on SAEs or CBMs could be useful.
- "initialize AR with Rmulti) using" -> right bracket is a typo


[1] Barbiero, Pietro, et al. "Interpretable neural-symbolic concept reasoning." International Conference on Machine Learning. PMLR, 2023.
[2] Barbiero, Pietro, et al. "Relational concept bottleneck models." Advances in Neural Information Processing Systems 37 (2024): 77663-77685.
 [3] Marra, Giuseppe, et al. "From statistical relational to neurosymbolic artificial intelligence: A survey." Artificial Intelligence 328 (2024): 104062.

### Questions
1) The list of concepts to build the dictionary should be pre-defined or can be learnt as well? E.g. by using an approach similar to label-free CBMs?
2) The concepts need to be known in advance wether having a single, multiple or relational representation, or it can be learnt automatically? I mean, in the automatic setting the representation is learnt, but you should know if r_c belongs either to R_single, R_multi or R_relation? In this case this may introduce a relevant bias in the learning and need additional knowledge about each concept range.
3) I was wondering if you tried or you think it make sense for a concept c, to have r_c = [r_cs,r_cm,r_cr], namely to allow a concept to have simultaneously a single, multiple and relational representation. In this way probably any concept can manage the mono and poly semantic.
4) If I understand correct A and A' have the same dimension, i.e. the concept inferrable are already pre-fixed (even if not learnt) from the beginning. In this way it seems like a semi-supervised setting where some concepts are supervised and others (the inferred) are not. As the rules are already known, what is the strong difference on just applying forward chaining as a pre-processing step and then use the supervisions derived at training time? 
5) How you decide to attach AR after layer 23 and 20 on the two backbone models? Why not, e.g. layer 15?

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
4

### Summary
This paper proposes ActivationReasoning (AR), a framework that claims to perform logical reasoning directly from the latent activation space of LLMs. It contains 3 main steps: (1) use SAE to extract “concept”; (2) map the activations to propositional symbol via thresholding; (3) apply user-defined rules over the symbols.

### Strengths
The paper presents a clear pipeline and demonstrate potential applications (like safety). The visualisation of their method is clear. It could inspire how to construct reliable symbolic reasoning using latent representations.

### Weaknesses
1. Concept extraction is loosely defined and not validated.

2. Hard-coded rules, dataset-based (using training set) concept extraction from SAE output.

3. Lack ablation on concept accuracy / why baseline model failed, is it because   of lacking the knowledge from the rules, or it is reasoning capability issue. Ablation is needed to understand why such huge performance gap of ~50\% occurs.

### Questions
Are the logical rules fully hand-written, and how are they chosen or validated for each dataset?

### Soundness
2

### Presentation
2

### Contribution
2
