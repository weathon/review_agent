# Neuro-Symbolic Decoding of Neural Activity

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
We propose NEURONA, a neuro-symbolic framework for fMRI decoding and concept grounding in neural activity. Leveraging image- and video-based fMRI question-answering datasets, NEURONA learns to decode interacting concepts from visual stimuli based on patterns of fMRI responses, integrating symbolic reasoning and compositional execution with fMRI grounding across brain regions. We demonstrate that incorporating structural priors (e.g., compositional predicate-argument dependencies between concepts) into the decoding process significantly improves both decoding accuracy over precise queries, and notably, generalization to unseen queries at test time. With NEURONA, we highlight neuro-symbolic frameworks as promising tools for understanding neural activity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The MS introduces a neuro-symbolic fMRI-QA framework that links unary (objects) and relational (predicates) concepts to brain-region embeddings. They show that argument-guided learning of the predicates (i.e., conditioning on subject/object) improves decoding, especially for action/position queries. This is an interesting result indicating the system is role-sensitive and is learning from brain data something like event representations rather than argument-agnostic predicate maps. 

Note: I assess the work mainly for its **value in advancing understanding of brain data and brain-research methods**, rather than for engineering improvements in QA performance from fMRI decoders.

### Strengths
1. The system learns in a way that understands event semantics, i.e., decoding what is the subject doing to the object.  in fact, while the framing of the work is in terms of learning relational concepts, but I think it's equally fair to say they are learning event semantics, which are driven by relations applied to objects, and the objective is to learn the general semantics. 
2. The findings implicitly supports the neurobiological position *against* searching for a general (neurobiological) concept-brain-location in a context independent way, but conditionalizing it on the subject/predicate. This has a major implication for theories that aim to identify brain regions for 'verb meaning' independent of context
3. The neuro-symbolic decomposition is clear and the MS is readable beyond the neuro-symbolic community.

### Weaknesses
* There is a main limitation for using the results to inform brain studies, grounded in the use of *Atlases*: using an atlas as the feature basis can (*i*) induce correlations among features because of spatial smoothing and parcellation choices (e.g., 2 brain areas might encode the image in the same way introducing collinearity among features), and (*ii*) allocate variance to parcels that are not informative with respect to the task domain, which limits interpretability and statistical sensitivity. 
To my thinking, a better, domain-aligned approach is to start from image-by-voxel data and apply PCA (or other dimensionality reduction; e.g., dictionary learning) prior to training. This would match the feature space to the dataset, perhaps increase precision, and certainly provide more information on distributed brain networks relevant to the concepts. Currently, interpretation is forced to be in the Atlas domain, which is not a state of art basis for brain representations.

### Questions
I was wondering if the predicate representation in the unguided setting contains decodable information about the object arguments. If that is the case, it would hint that the predicate embedding/code binds to specific objects.

### Soundness
3

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
3

### Summary
This paper presents a neuro-symbolic framework for compositional concept grounding and reasoning using fMRI data. The model aims to decode not only object-level but also relational representations from regional brain activity. The authors evaluate NEURONA on 2 visual fMRI dataset, showing improved decoding and question-answering performance compared to previous fMRI decoding methods (SDRecon, BrainCap, UMBRAE). The work further analyzes the impact of multi-region grounding and relational reasoning through ablation studies.

Overall, the paper is conceptually interesting and methodologically competent, but the empirical depth is relatively limited in statistical validation and subject generalization, and the interpretive link between symbolic composition and neural mechanisms could be made more rigorous. . Additional analyses are highly encouraged.

### Strengths
1. The paper is well motivated to tackle the puzzle of interpretable concept grounding in neural data, with an emphasis on compositional structure.
2. The results are consistent across two datasets and multiple concept types.
3. The ablation analyses meaningfully test hypotheses about region-level grounding and compositional generalization.
4. It positions neuro-symbolic modeling as a promising route to study structured neural semantics beyond pixel-level reconstruction.

### Weaknesses
1.The results in Tables 1–2 are trained and tested on one subject per dataset, which limits generalizability. Multi-subject or cross-subject validation is essential for neuroscientific conclusions.
2. Grounding is restricted to coarse atlas parcels; voxel-level analyses would strengthen claims about spatial specificity of “modular concepts.”
3. Results in table 1 and 2 are using models trained from one subject in each dataset
4. It remains unclear how much each region or network contributes to concept decoding; e.g., whether “multi-region grounding” corresponds to interpretable brain modules.
5. While means ± SD are reported, formal statistical tests are absent, so robustness to inter-individual variability is uncertain.
6. Without ground-truth region–concept pairs, “consistent” grounding might reflect network-level co-activation rather than explicit compositional structure.
7. Qualitative examples are interesting but could be complemented by quantitative measures (e.g., concept-wise accuracy)
8. Writing is generally clear, though sections 3–5 are dense and occasionally repeat prior claims; discussion could better delineate cognitive vs. computational interpretations.

### Questions
1. Have you tried cross-subject evaluations?
2. In the multi-region grounding setup, is there any redundancy or competition between regions (e.g., correlated activations leading to similar grounding weights)?
3. How consistent are the learned concept groundings across subjects — do similar regions emerge for the same concept?
4. Do claims of compositional grounding align with theories of hierarchical organization of the brain?
5. How might this framework extend to other domains (e.g., motor planning)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes NEURONA, a neuro-symbolic framework for grounding compositional concepts in neural activity using fMRI data. The approach extends Logic-Enhanced Foundation Model (Hsu et al. 2023) to fMRI-based question answering, testing different hypotheses about how concepts are grounded in brain regions through hierarchical predicate-argument dependencies. The authors demonstrate improved decoding accuracy on constructed fMRI-QA datasets derived from BOLD5000 and CNeuroMod, showing particular gains on compositional queries involving actions and spatial relationships.

The version of the paper as of now should be rejected because (1) the central claims about discovering compositional organization in the brain are not well supported by the evidence, (2) the experimental design conflates task performance improvements with evidence of neural representational structure, (3) the "grounding" methodology does not actually demonstrate meaningful concept-to-brain mappings, and (4) key claims about compositionality lack proper validation through decomposability or systematic structural tests. The technical framework represents a potentially valuable contribution to neural decoding, but the interpretive claims about brain organization require much stronger evidence than currently provided.

### Strengths
Despite fundamental issues, the paper has notable technical strengths:

1. A well-defined evaluator logic for creating conjunctions that could help in better decoding. 
2. Strong empirical performance: 47% relative improvement over baselines is substantial, and the generalization to unseen compositional queries is genuinely impressive evidence that the learned representations support novel combinations.
3. Cross-dataset validation: Testing on both image and video datasets with consistent results strengthens the technical contribution.

### Weaknesses
1. The Core Interpretive Problem:

The paper's fundamental flaw is the interpretive leap from 'symbolic structure improves neural decoding' to 'relational meaning in the brain emerges from structured activations guided by hierarchical predicate-argument structure.' The authors consistently conflate improved task performance with evidence of compositional neural mechanisms, but these are different claims requiring different types of evidence.
The 'grounding' results are classification logits indicating which brain regions are statistically useful for predicting concept labels—not neural representations in a meaningfully neuroscientific sense. Classification weights in multivariate neuroimaging do not reflect the underlying neural patterns or information content; they indicate statistical utility for discrimination, not representational structure (see Haufe et al. 2014 NeuroImage). True neural representations involve patterns of activity across voxels that can be decoded, analyzed geometrically, and related to cognitive processes. Furthermore, the authors mischaracterize the prior literature on concept grounding. The foundational studies cited (Huth et al., Mitchell et al., Palatucci et al.) use forward encoding models that predict brain activity from semantic features, allowing legitimate claims about neural encoding. The paper should better distinguish its claims from prior work on concept grounding and be more precise about what type of evidence would support compositional neural organization vs. improved decoding.
The distinction between functional and representational compositionality should be made explicit early in the paper. The current presentation conflates these concepts (e.g., see Lake & Baroni 2018 ICML)

2. Experimental Design Issues:

The datasets retrofit compositional reasoning onto passive viewing data without evidence that participants were performing compositional operations. BOLD5000 participants simply viewed images—they were not actively reasoning about spatial relationships or compositional structure. The "ground truth" comes from automated scene graph parsing, not actual participant cognition. This creates a fundamental disconnect between the experimental setup and claims about compositional cognition in the brain. The danger is that improved decoding may simply reflect visual feature correlations rather than compositional neural processes - without evidence that participants engaged compositional reasoning, there's no basis for compositional interpretations of the neural patterns.
Without participants actively performing compositional reasoning during scanning, there is no basis for claiming that improved decoding of these post-hoc questions reflects compositional neural organization. The authors need to demonstrate that participants were actually engaging compositional processes, not just that symbolic priors help classify brain signals.

The dataset construction follows machine learning conventions (scene graphs → QA tasks) which is appropriate for developing better reasoning systems. However, this methodology is insufficient for claims about brain organization. Without experimental evidence that participants engaged compositional reasoning during scanning, improved decoding of post-hoc compositional questions cannot support claims about compositional neural mechanisms. The authors are applying machine learning validation criteria (task performance) to neuroscientific claims (brain organization) without the experimental controls required for the latter.
Both the issues 1 and 2 could be addressed if voxel/region-wise encoding models (akin to Huth et al.) could be constructed for the concepts from the scene graph parsing and the predicted responses corresponding to the concepts from the model can be treated as grounding/representations and the rest of methodology could be applied (transformation into unary and binary embedding space and executor logic).

3. Missing Evidence for Compositionality:

While the paper implements compositional operations in its executor, it does not demonstrate compositional neural representations. Key missing evidence includes:

3.1 Decomposability: True neuro-symbolic compositional systems (tensor products, vector symbolic architectures) maintain constituent structure that can be systematically recovered through inverse operations. In contrast, the executor combines groundings through non-invertible operations that destroy constituent information. There is no principled way to recover individual concept groundings from the final compositional output, demonstrating that the system implements functional composition without compositional representation. I consider the evaluator logic is a good conjunction recipe but not sure about neuro-symbolic composition though given lack of invertibility. And given that it resembles conjunctive representations, I wasn’t surprised that H5 showed greatest evidence.

3.2 Systematic decomposition: While the authors demonstrate generalization to novel combinations by ensuring disjoint train/test sets, they lack fine-grained analysis of systematic recombination patterns. True compositionality requires systematic relationships between constituents and compounds across all concept types - for example, if the system learns holding(person, bat) and beside(person, tree), it should systematically handle holding(person, tree) and beside(person, bat) with predictable performance relationships. The paper shows aggregate performance on unseen combinations but doesn't analyze whether learned representations support the algebraic manipulation of concepts that characterizes genuine compositional systems. While dataset constraints may limit the extent to which systematic recombination tests are feasible, the authors don't report what subset of such analyses their data could support or acknowledge these limitations when making compositionality claims. Without demonstrating systematic performance patterns across specific recombination types (argument swapping, predicate transfer, role systematicity), the generalization results, while impressive, don't constitute strong evidence for compositional representation.

4. Methodological Concerns:

The consistency metric (how often concepts ground to same regions) is presented as validation, but high consistency could simply reflect stable solutions to the classification task rather than meaningful neural organization (similar to the Haufe et al. critique above). Without independent validation against known neuroscience or comparison to null models with random concept assignments, these results cannot support claims about brain organization.

### Questions
1.	Can the authors implement encoding models built from the parsed concepts and use them as grounding/representations?
2.	If strong claims about neurosymbolic representations are to be made: Can fine-grained systematicity tests be done? Is it possible for the evaluator to be inverted to make the output decomposable after binding?
3.	Can better null models be used (with random concept assignments)?
4.	If your answer is no to points 1 and 2, can the focus be changed to methodological contribution and decoding improvements rather than strong claims about brain organization?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes NEURONA, a neuro-symbolic framework for decoding compositional concepts from fMRI by parsing each visual stimulus into a symbolic expression (e.g., predicate–argument relations) and grounding its unary and relational concepts to candidate neural entities derived from cortical parcellations. The authors focused on using QA accuracy as the proxy for concept groundings capabilities. The proposed approach extends LEFT to fMRI-QA so that weak supervision from question-answer pairs can train disentangled concept groundings and an end-to-end executor that composes those groundings to answer queries. The authors conduct experiments to show that the proposed method improves accuracy and generalizes to unseen compositions on BOLD5000-QA and CNeuroMod-QA.

### Strengths
I am not familiar with tasks with fMRI, but to the best of my knowledge the framework is new in bringing predicate–argument guidance into an fMRI-QA decoder and testing it via five alternatives within one executor.

The overall evaluation setup looks correct to me: the compositional split is appropriate; train and test use disjoint entity–relation pairs, so it measures true generalization. The ablations are clean and isolate the source of gains: unguided multi-region grounding adds little over a single region, while subject/object guidance and full guidance help most, especially for actions and spatial queries. Results are robust and strong across multiple cortical atlases. The qualitative maps are sensible and aid interpretation. 

Overall I think the paper is overall clearly written and the ideas are intuitive to understand despite my unfamiliarities with fMRI-related tasks.

### Weaknesses
The primary concern is the novelty about the proposed approach. 

The proposed method assembles known pieces: a LEFT-style executor, VLM-derived scene graphs, and standard cortical parcellations. The main new element is predicate–argument guidance and the within-executor hypothesis family that tests it. I acknolwedge that these are thoughtful design choices, but they are not a new model class or theory. That said, the paper still adds value: it gives a clean empirical test of the idea, uses a compositional split that rules out memorization, and offers consistency and cross-atlas checks that others can reuse. 

In short, the work is incremental in mechanism yet still useful in evidence and evaluation protocol for the research commnuity.

### Questions
Since fillers ultimately come from token sequences, I am wondering have the authors thought about how robust are role actions to multi-token fillers or subword boundary changes?

### Soundness
3

### Presentation
3

### Contribution
3
