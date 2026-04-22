# Talk in Pieces, See in Whole: Disentangling and Hierarchical Aggregating Representations for Language-based Object Detection

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
While vision-language models (VLMs) have made significant progress in multimodal perception (e.g., open-vocabulary object detection) with simple language queries, state-of-the-art VLMs still show limited ability to perceive complex queries involving descriptive attributes and relational clauses. Our in-depth analysis shows that these limitations mainly stem from text encoders in VLMs. Such text encoders behave like bags-of-words and fail to separate target objects from their descriptive attributes and relations in complex queries, resulting in frequent false positives. To address this, we propose restructuring linguistic representations according to the hierarchical relations within sentences for language-based object detection. A key insight is the necessity of disentangling textual tokens into core components—objects, attributes, and relations (“talk in pieces”)—and subsequently aggregating them into hierarchically structured sentence-level representations (“see in whole”). Building on this principle, we introduce the TaSe framework with three main contributions: (1) a hierarchical synthetic captioning dataset spanning three tiers from category names to descriptive sentences; (2) Talk in Pieces, the three-component disentanglement module guided by a novel disentanglement loss function, transforms text embeddings into subspace compositions; and (3) See in Whole, which learns to aggregate disentangled components into hierarchically structured embeddings with the guide of proposed hierarchical objectives.  The proposed TaSe framework strengthens the inductive bias of hierarchical linguistic structures, resulting in fine-grained multimodal representations for language-based object detection. Experimental results under the OmniLabel benchmark show a 24% performance improvement, demonstrating the importance of linguistic compositionality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses a key limitation in vision-language models: their poor compositional understanding of complex textual queries in language-based object detection (LBOD). Current models behave like “bags-of-words,” often misinterpreting sentences that combine object names, attributes, and relations. The authors propose TaSe (Talk in Pieces, See in Whole), which consists of:
1. HiVG Dataset: A hierarchical synthetic captioning dataset derived from Visual Genome, organized into object–attribute–relation tiers.
2. TriDe Module: A component-wise disentanglement mechanism that separates text embeddings into object, attribute, and relation subspaces.
3. Hierarchical Aggregation: A new loss based on hierarchical entailment using a Euclidean-space “Radial Embedding (RE)” objective to reconstruct contextualized sentence representations.
TaSe integrates into GLEE (a MaskDINO + CLIP-based VL detector) and achieves substantial gains on OmniLabel and D^3 benchmarks.

### Strengths
1. The paper convincingly diagnoses a key weakness of current VL detectors: poor sentence-level compositionality and contextual grounding. The empirical t-SNE analysis in Fig. 1 is effective in illustrating why text embeddings collapse across different relational phrases.
2. Comprehensive experiments on multiple benchmarks, with both quantitative (Tables 1–3) and qualitative (Figs. 5–7  results. 
3. The writing is clear and methodically organized.

### Weaknesses
1. HiVG, the proposed hierarchical dataset, is central to the paper but only briefly described in the main text. There’s no quantitative analysis of HiVG’s diversity, quality, or balance (e.g., number of distinct attributes/relations, comparison with Visual Genome). Since HiVG is synthetic and LLM-generated, potential hallucination or linguistic bias needs evaluation.
2. Consider including a human evaluation of caption quality or a robustness test (e.g., sensitivity to paraphrases).
3. The framework combines multiple non-trivial modules (TriDe, RE, hierarchical losses). While each part is motivated, the cumulative complexity may limit reproducibility. It’s unclear how much improvement comes from disentanglement vs. better data vs. hierarchical supervision, though ablations partly address this.
 


Minor Presentation Issues： 
1. Some equations (e.g., Eq. 3, 6) are dense and could use more intuition.

2. Figures could label components more clearly (especially in Fig. 2).

### Questions
1. The final text embedding E=pooling(O+A+R). What pooling function is used (mean, [CLS] token, attention)? Have you compared different pooling methods to see how they affect compositional sensitivity? 
2. How is the “reference point”  r in Eq. 6 chosen or updated?
3. Why were structure-aware or compositional baselines like CLIP-Adapter, LogicZSL, or MDETR not included?
4. Have you identified scenarios where the hierarchical structure fails — e.g., ambiguous relational clauses or overlapping attributes?

### Soundness
3

### Presentation
2

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
This paper introduces TaSe (Talk in Pieces, See in Whole), a framework designed to enhance language-based object detection by explicitly modeling the hierarchical and compositional structure of language. It first introduces a disentanglement module to break down text representations into core components of objects, attributes, and relations ('Talk in Pieces'), using a novel module called TriDe. It then designs a hierarchical aggregation mechanism to learn structured, sentence-level embeddings from these disentangled parts ('See in Whole'). To facilitate this, the authors introduce HiVG, a new dataset generated by using an LLM to re-caption the Visual Genome dataset with an explicit three-tier hierarchical structure. Experiment results on challenging language-based object detection benchmarks demonstrate that linguistic compositionality is crucial for the task of language-driven object detection.

### Strengths
1. The TaSe framework is well-designed that competently integrates disentanglement via attention and hierarchical learning via a Euclidean-space objective.
2. It achieves clear performance improvements over a strong baseline on the OmniLabel and D³ benchmarks. The ablation studies provide useful insights into the function of each component of the proposed framework.
3. The writing is clear, and the presentation of figures and diagrams is good.

### Weaknesses
1. In the HiVG dataset generation process, does the diversity of the generated negative captions influence the final performance? For example, how would the results vary if different large language models (LLMs) were used for re-captioning?
2. The authors describe their approach as “lightweight”. it would be helpful to provide a quantitative analysis of the computational overhead.
3. Given the framework's strong reliance on the O-A-R structure, how does it perform if queries that violate or deviate from this structure (eg. comparative phrases or multi-step relations)?
4. The TriDe module uses learnable queries to disentangle objects, attributes, and relations. Can the authors provide some qualitative visualizations to demonstrate that the module indeed learns meaningful disentanglement?

### Questions
1. In Lines 103–104, the authors state that the proposed reverse abstraction process “effectively mitigates hallucination issues.” Could the authors provide quantitative evidence or more systematic analysis to support this claim?
2. Can TaSe generalize to other VLM models?
3. How does the baseline trained on HiVG perform with complex but non-hierarchically annotated language?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to tackle the problem that VLM-based OVD models fall short in understanding complex queries. The authors make an observation that the text embedding of VLM can not well distinguish different attributes / relations for the same object, and consider this phenomenon as the main reason that causes the insufficient ability. To tackle this problem, they first construct a new data pipeline that has structured descriptions with "object - attribute - relation" hierarchy. Under this hierarchy, each data point (the bbox) has multiple hard negative text names that have different attribute / relation.  They also propose a new method that first disentangles  the embedding into three components (object / attribute / relation) and then re-aggregates them into a new embedding. The corresponding training method has some novelty for accommodating the hierarchy, as well.

### Strengths
- The observation on the text embedding well illustrates the reason that OVD-VLMs fail to understand complex queries. 

- The solution is comprehensive, covering from the data construction, the model architecture design, as well as the corresponding training objectives. 

- The data construction part is reasonable, and the method seems novel.

- The ablation study is comprehensive.

### Weaknesses
- The presentation and writing needs improvement. Some parts are difficult to follow. 

- According to the example given in Section 3.1, the most import aspect of HIVG is to generate HARD negative text discription of each box, i.e., the description sharing the same object but have different attribute (Tier 2) or realtion (Tier 3). It would be more clear for the authors to describe this principle into the introduction.

- In Eq. 1, 
    - 1) What is the text embedding X in Line 1?
    - 2) In Line 2, Which one is the query / value for the cross attention, respectively?
    - 3) It seems that TriDe not only disentangles the original embeddings, but also re-aggregates the derived O, A, R into the output E (Line 3). What is the motivation of this design?

- The improvement on D^3 dataset is very small, e.g., +1.0 FULL AP over the GLEE baseline.

- The employed datasets are insufficient. The authors have not includes a popolar dataset RefCoCo which also has long and fine-grained descriptions, and is popular for visual grounding evaluation. Is there any specific reason?

### Questions
- In the introduction, the authors reveal an observation that descriptions with partially-shared and partiall-different words have over-close embedding distances. However, recent SoTA VLMs are already trained with fine-grained discriptions. I am interested if the authors have deeper analysis on the reason why this phenomenon still happens.

- How does your method perform on general OVD dataset (e.g., LVIS). You may use the Out-of-Domain setting for easy comparison,i.e., you train a baseline method on your dataset and compare this baseline against your method.

- Will the proposed hierarchical and disengtangled representation learning be good for the VLM training (besides the OVD as in the paper)?

### Soundness
3

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
Current VLMs struggle with complex language queries involving descriptive attributes and relational clauses. Their text encoders often behave like bags-of-words, failing to distinguish target objects from their attributes and relations, which leads to frequent false positives (e.g., confusing "segway with a man" for just "segway"). The paper proposes a framework designed to restructure linguistic representations according to hierarchical relations within sentences. The core principle is to disentangle text tokens into components and aggregate them into hierarchically structured sentence-level representations. The paper proposes:

HiVG Dataset: A hierarchical synthetic captioning dataset derived from Visual Genome. It structures captions into three tiers using a "reverse abstraction" process to mitigate hallucination:

A module that transforms text embeddings into three distinct subspaces, objects, attributes, and relations, guided by a disentanglement loss function.

A learning method that aggregates these disentangled components into structured embeddings. Unlike contrastive learning, it models full sentence hierarchy using dynamic references to enforce separation between negative pairs while preserving intrinsic embedding structure.

### Strengths
The paper directly tackles the "bag-of-words" behavior of current VLMs, which often fail to distinguish target objects from descriptive attributes and relations in complex queries.


The proposed TaSe framework demonstrates significant performance improvements over strong baselines like GLEE. It achieved a 24% performance improvement on the OmniLabel benchmark and increased AP scores by +3.1 on D³ and +5.2 on OmniLabel.

It introduces a framework that disentangles text into three core components (objects, attributes, relations) and aggregates them hierarchically, strengthening the inductive bias for linguistic structures.


The paper proposes the HiVG dataset.

### Weaknesses
1: The proposed HiVG dataset and TriDe module strictly enforce a three-tier hierarchy: Object, Attribute and Relation. While this addresses simple compound queries, natural language often involves deeper, recursive nesting that exceeds three tiers (e.g., "the man holding the cup that is on the table"). The current framework may struggle to represent these deeper recursive structures efficiently because it is designed purely around these three predefined components.

2. Ambiguity in "Attribute" vs. "Relation" Definitions. The distinction between Tier 2 (Attribute) and Tier 3 (Relation) appears conflated in the examples provided. For instance, "middle woman" is Tier 2, while "middle woman with dark hair" is classified as Tier 3 (Relation). Linguistically, "dark hair" is often considered an attribute of the entity, similar to "middle." By contrast, other Tier 3 examples involve distinct objects, such as "segway with a man". Treating intrinsic attributes (like hair color) and extrinsic relations (like accompanying objects) identically in Tier 3 might limit the model's ability to disentangle complex scenes where both distinct object interactions and fine-grained attributes are present simultaneously.

3. The fine-tuning relies on a relatively small synthetic dataset. The authors explicitly state they "use only HiVG dataset for training, which contains 10 K hierarchy captions". While efficient (training only 2.93% of parameters) , basing generic open-vocabulary improvements on such a small, specifically structured dataset risks overfitting to the specific syntactic patterns generated by Llama 3 during the re-captioning process. The claimed robustness to general complex queries might be limited to the specific domain covered by these 10K Visual Genome-derived captions.


4. The qualitative results presented focus exclusively on successes where TaSe outperforms the GLEE baseline by reducing false positives. There is no analysis of where the strict hierarchical enforcement might cause errors—for example, in idiomatic expressions that do not neatly decompose into object-attribute-relation, or where disentanglement might over-segment a coherent concept.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
