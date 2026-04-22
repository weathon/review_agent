# Mapping Semantic & Syntactic Relationships with Geometric Rotation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 8, 4, 2, 4

## Abstract
Understanding how language and embedding models  encode semantic relationships is fundamental to model interpretability. 
While early word embeddings exhibited intuitive vector arithmetic  (''king'' - ''man'' + ''woman'' = ''queen''), modern high-dimensional text representations lack straightforward interpretable geometric properties. 
We introduce Rotor-Invariant Shift Estimation (RISE), a geometric approach that represents semantic-syntactic transformations as consistent rotational operations in embedding space, leveraging the manifold structure of modern language representations. 
RISE operations have the ability to operate across both languages and models without reducing performance, suggesting the existence of analogous cross-lingual geometric structure.
We compare and evaluate RISE using two baseline methods, three embedding models, three datasets, and seven morphologically diverse languages in five major language groups. 
Our results demonstrate that RISE consistently maps discourse-level semantic-syntactic transformations with distinct grammatical features (e.g., negation and conditionality) across languages and models.
This work provides the first demonstration that discourse-level semantic-syntactic transformations correspond to consistent geometric operations in multilingual embedding spaces, empirically supporting the linear representation hypothesis at the sentence level.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposed a new methodology for steering embedding models. Instead of learning a steering vector, as is commonplace in the literature, the authors propose to learn a rotation operation which operates directly on the spherical manifold on which the embeddings live.

They call their proposed method Rotor-Invariant Shift Estimation (RISE) which, to my understanding, operates in three steps: first the source and destination embeddings are mapped to a canonical direction via an orthogonal transformation (a rotor). The semantic different between them is expressed in a standardised coordinate system (eq. 1). These "semantic changes" are averaged to create a "prototype" (eq. 2), which is then used to predict the semantic transformation of a unseen source embedding (eq. 3).

The performance of their method is evaluated via a series of experiments on a collection of model across three semantic transformations: negation, conditionality, and politeness, and several languages of different morphological types. This is quite comprehensive and makes clear the tasks on which the method performs well and less well.

### Strengths
The proposed method is simple yet innovative and novel; the paper is well-written and presented, and is highly enjoyable to read. On this basis I strongly recommend acceptance of this paper.

In particular,
- the background, motivation and prior work is well exposed and the authors do a very good job of placing their method in the context of the literature
- the method itself is in principle very simple but the novelty and innovation comes from learning a transformation which respects and operates directly on the manifold in which the embeddings live
- the method is described very clearly, though I think it could benefit from additional explanation and disucssion to benefit reads not familiar with Riemannian geometry (see Weaknesses)
- the experiments provide a comprehensive understand of when and where the method performs well, and when it performs less well. The discussion of this is fairly extensive and enlightening
- Sections 7-10 provide extensive discussion of the work in context, which I think is really valuable for the reader of a paper like this

### Weaknesses
As I mentioned before, I think this paper is very strong, so the following should be viewed a suggestions for improvements, rather than weaknesses per se.

- I think the explanation of the method is very mathematically clear, however I think the paper could be made more accessible to readers without knowledge of Riemannian geometry (likely many readers who might use this method), by providing some more discussion of the geometric transformations which are taking place. Perhaps even an illustration of what is happening at each stage could be very enlightening. I don't think notions such as tangent spaces and Riemannian logarithms and exponentials will necessarily be familiar to a large portion of the audience who would find this method useful.
- I would like to see a direct comparison with the method of Euclidean steering vectors for all the experiments you perform. I would seemingly be a very simple thing to do and would really show where the potential advantages of this method lie.

### Questions
Re. the second bullet point in Weaknesses, how does the method compare with Euclidean steering vectors?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper aims to characterise discourse and sentence-level semantic changes in encoder models as geometric transformations. To do that Rotor Invariant Shift Estimation (RISE) is introduced.  Given pairs of embeddings characterising a semantic transformation, RISE finds a rotation per pair which maps the tangent direction (in a spherical manifold) between both embeddings in the pair to a canonical direction. From these, a prototype for the semantic transformation is estimated and later used for inference.

The results show that the geometric rotations learned by RISE work for three semantic changes (Negation, Conditionality and Politeness). Furthermore, these transformations are shown to hold in a cross-linguistic setup. When testing RISE for cross-model transfer, results show that performance is language-dependent but generally satisfactory. Finally, RISE is shown to perform better at syntactic variations than semantic ones, as elicited by the BLIMP vs SICK performance.

### Strengths
* The paper is clearly written and the setup is simple.
* The model transfer results are interesting and show that simple transformations can grasp syntactic or discourse level binary features.

### Weaknesses
There are several relevant works which are not cited or mentioned:

* Semantic changes have been previously modelled with linear transformations
     - (Baroni and Zamparelli, 2010) https://aclanthology.org/D10-1115/
     - (Mitchell and Lapata, 2008) https://aclanthology.org/P08-1028/

* Non-euclidean semantic and syntactic probing
     - (Chen et al., 2021) https://arxiv.org/abs/2104.03869

The work is currently limited to models that normalise their representations to the unit sphere, which is not the case for all encoders. Furthermore, autoregressive large language models are not evaluated. Without further experiments, these limitations heavily impact reach of the contribution.

Negation and Conditionality are not only semantic changes but also syntactic ones. If the contribution is about semantics, it should control for syntactic structure. This conflict is accentuated by the last result, showing that BLIMP performance is higher than SICK. Understanding how syntactic changes are reflected in the models’ representations is an interesting direction, but this falls out of the scope of the paper (as reflected by the current title and introduction).

Steering can be a motivation for the paper but such methods are not used in any of the experiments to support the importance of the geometric rotations for downstream model behaviour. It seems odd to include “steering” in the title and heavily introduce it.

### Questions
* Why is there one transformation per pair? Wouldn’t it be easier to learn a single transformation per semantic change solving least-squares?

* Is RISE specific to encoder models with representations in the unit sphere? Can the findings be extended to more models?

* How is pooling done (last embedding, mean-pooling …)? What layer of the model is picked?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Rotor-Invariant Shift Estimation (RISE), a geometric framework for understanding and controlling semantic transformations in multilingual embedding spaces. It treats meaning changes as rotational operations on the curved manifold where sentence embeddings lie, rather than as linear shifts in Euclidean space. Evaluated across multiple models and seven morphologically diverse languages, the method shows that certain discourse-level semantic transformations exhibit consistent geometric structure across both languages and model architectures.

### Strengths
1. The paper provides a principled geometric extension of the Linear Representation Hypothesis to non-Euclidean (Riemannian) spaces, moving beyond heuristic linear probing.

2. The study’s inclusion of seven languages and three embedding architectures (OpenAI, BGE-M3, mBERT) demonstrates broad applicability and robustness.

### Weaknesses
1. **Limited empirical grounding for broad claims.**
The paper’s experiments are not sufficiently strong or comprehensive to substantiate its broader theoretical claims. For instance, it asserts “theoretical contributions to the usefulness of steering vectors for embedding models’ controllability and interpretation,” yet the evidence presented does not convincingly bridge this conceptual gap. The “steering vector” literature predominantly addresses interventions in the activations of generative or autoregressive models, whereas this work focuses on embedding models, a fundamentally different setting. Without stronger empirical validation or comparative studies, the claimed connection to the steering-vector paradigm remains tenuous.

2. **Mismatch between ambition and experimental scope.**
This limitation reflects a broader issue: the paper makes overgeneralized theoretical claims based on a narrow experimental design. By restricting its evaluation to a small set of highly specific discourse-level transformations, the work lacks the empirical breadth needed to justify its sweeping conclusions about universal geometric principles or semantic control in embedding spaces. The authors acknowledge this limitation, but it significantly constrains the strength of their theoretical arguments.

3. **Incremental methodological novelty.**
Despite its mathematical rigor, the proposed shift from linear to rotational/geodesic steering may appear incremental rather than groundbreaking. The core contribution lies primarily in applying standard Riemannian geometry tools (e.g., exponential and logarithmic maps on the hypersphere) to embedding analysis, rather than introducing fundamentally new geometric or algorithmic mechanisms. As a result, the perceived innovation may fall short of the paper’s ambitious framing.

### Questions
1. **Universality:** 
If cross-model experiments reveal a consistent English-centric bias, how can the authors justify the claim of universal geometric operations across languages?

2. **Semantic vs. Grammatical Structure:** 
Given that RISE achieves near-perfect results on syntactic (BLiMP) tasks but only moderate performance on semantic (SICK) tasks, how do the authors support the claim that RISE captures semantic rather than primarily grammatical transformations?

3. **Synthetic Data Validity:**
Since all data were generated by GPT-4.5 without human or native-speaker validation, how do the authors ensure that RISE learns genuine linguistic geometric structures rather than artifacts of the data-generation model’s English-centric bias?

4. **Experimental Scope and Generalization:**
With experiments limited to three linguistic phenomena (negation, conditionality, politeness), how can the authors substantiate broader conclusions about general geometric principles in embedding spaces?

### Soundness
2

### Presentation
2

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
The paper proposes Rotor-Invariant Shift Estimation (RISE), a geometric framework that treats discourse-level semantic transformations (negation, conditionality, politeness) as rotations/geodesic displacements on the unit hypersphere of sentence embeddings. The method canonicalizes neutral transformed sentence pairs via an orthogonal operator R(n), averages tangent-space shifts to learn a prototype, and predicts unseen transformations using the exponential map back to the sphere. Experiments cover 3 embedding models (OpenAI text-embedding-3-large, BGE-M3, mBERT), 7 languages, and 3 datasets (a GPT-4.5-generated synthetic multilingual set, BLiMP, SICK). The authors report strong cross-language transfer especially for negation and provide first-order commutativity guarantees and O(d) complexity for each RISE edit.

### Strengths
1. The paper presents a strong geometric and theoretical foundation, with clear use of log/exp maps, orthogonal canonicalization, and formal proofs of commutativity and O(d) complexity.
2. Cross-lingual results show that negation and conditionality transfer consistently across seven diverse languages, and the paper thoughtfully contrasts logical versus pragmatic phenomena such as politeness.
3. Cross-model transfer (text-embedding-3-large → BGE-M3) demonstrates practical portability while revealing architecture- and data-driven biases in embedding geometry.

### Weaknesses
1.	The evaluation relies heavily on synthetic data generated by GPT-4.5, with limited validation using human or parallel-corpus data. This dependence risks encoding generation artifacts and Anglo-centric prompt biases into the learned prototypes, potentially inflating cross-lingual consistency and reducing ecological validity.

2.	The baseline comparisons are insufficiently described. The current experiments report results primarily within the RISE framework, without inclusion of clear linear or geometric alternatives—such as additive Euclidean shifts, orthogonal Procrustes alignment, or other linear mapping baselines. These would help determine whether RISE’s spherical geometry truly provides benefits beyond standard linear transformations. As a result, it remains unclear whether the observed gains arise from geometric innovations or from data-specific and normalization effects.

3.	Effectiveness is currently measured solely through average cosine similarity. While informative, this metric captures only local geometric alignment. Evaluating RISE on downstream tasks—such as classification, semantic retrieval, or analogy completion—would strengthen the empirical claim that its geometric transformations correspond to functional semantic improvements.

4.	The evaluation is confined to text-based embeddings. Given that RISE operates in a general geometric latent space, it would be valuable to extend analysis to multimodal encoders (e.g., CLIP, BLIP-2). Such experiments could test whether RISE’s discourse-level “rotations” generalize across modalities, providing stronger evidence for universal geometric semantics rather than text-specific artifacts.

### Questions
1. Could the authors include additional non-synthetic datasets (e.g., parallel or human-annotated corpora) to validate that RISE captures genuine semantic transformations rather than synthetic prompt artifacts?
2.  Could the authors implement linear baselines (e.g., additive Euclidean, Procrustes) to clarify where the spherical treatment demonstrably preserves compositionality or rotation consistency that linear mappings fail to capture?
3.  Have the authors considered testing RISE on downstream semantic tasks (e.g., classification, retrieval, or transfer) to assess whether its geometric properties translate into functional semantic gains?
4.  Could RISE be extended to multimodal representations? If not, what are the theoretical or practical obstacles preventing such an extension?

### Soundness
2

### Presentation
2

### Contribution
2
