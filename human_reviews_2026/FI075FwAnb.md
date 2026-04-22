# Geometric Constraints for Small Language Models to Understand and Expand Scientific Taxonomies

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Recent findings reveal that token embeddings of Large Language Models (LLMs) exhibit strong hyperbolicity. This insight motivates leveraging LLMs for scientific taxonomy tasks, where maintaining and expanding hierarchical knowledge structures is critical. Although potential, generally-trained LLMs face challenges in directly handling domain-specific taxonomies, including computational cost and hallucination. Meanwhile, Small Language Models (SLMs) provide a more economical alternative if empowered with proper knowledge transfer. In this work, we introduce SS-Mono (Structure-Semantic Monotonization), a novel pipeline that combines local taxonomy augmentation from LLMs, self-supervised fine-tuning of SLMs with geometric constraints, and LLM calibration. Our approach enables efficient and accurate taxonomy expansion across root, leaf, and intermediate nodes. Extensive experiments on both leaf and non-leaf expansion benchmarks demonstrate that a fine-tuned SLM (e.g., DistilBERT-base-110M) consistently outperforms frozen LLMs (e.g., GPT-4o, Gemma-2-9B) and domain-specific baselines. These findings highlight the promise of lightweight yet effective models for structured knowledge enrichment in scientific domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SS-Mono, a pipeline targeting efficient and accurate scientific taxonomy expansion by transferring hierarchical knowledge from Large Language Models to Small Language Models. SS-Mono integrates a local taxonomy augmentation phase using LLMs, a self-supervised geometric fine-tuning process with hyperbolic constraints, and LLM-based calibration for candidate ranking. The approach is evaluated on several taxonomy expansion benchmarks (SemEval-Food, WordNet-Verb, and MeSH), showing that a fine-tuned SLM outperforms both frozen LLMs and domain-specific baselines for both leaf and non-leaf node insertions.

### Strengths
S1: The structure of SS-Mono is modular and well-motivated: it combines structure-dominated (hyperbolic metric) and context-dominated (semantic, LLM-augmented) encoders. This dual design is well illustrated in Figure 1, which shows how LLMs are used to enrich edge semantics and how structural relations are composed.

S2: Extensive experiments (see Tables 2 and 3) cover multiple datasets and include a wider range of both classical and recent baselines, including neural and LLM-based approaches. The results show consistent, substantial gains of SS-Mono, especially when augmented with LLM outputs, across MR, MRR, and Recall/Precision metrics.

S3: Self-supervised training enables the model to leverage existing taxonomies without costly annotation. The mathematical formulations (see Equations 1-11 and Cone Loss in Section 3.2) are clear, concise, and consistent with state-of-the-art hyperbolic learning approaches.

### Weaknesses
W1: While the paper claims that "LLMs have potential but are not ready to be directly deployed," the LLM calibration component appears to be more of an optional enhancement than a robust part of the pipeline. Figure 2 exposes high failure rates in LLM output for reranking (frequent hallucinated edges or incomplete lists), and Section 4.3 provides only a partial mitigation of this challenge. The improvements from LLM calibration, as shown in Table 3, are not always consistent or predictable.

W2: The model relies on self-supervised negative sampling and LLM-augmented context for candidate positions. The precise method for sampling hard negatives is tailored closely to the taxonomy structure as described in Section 3.4, but it is unclear whether the negative sampling strategy introduces bias or overestimates the model’s robustness to challenging/far-out queries.

W3: The pipeline discusses SLM fine-tuning with geometric constraints, but does not deeply analyze (or ablate) alternative LLM fine-tuning approaches (domain adaptation, prompt-tuning, etc.). This limits direct attribution of SLM’s advantages to the geometric approach, rather than simply to the smaller/faster architecture or training regime.

### Questions
Q1: Have the authors conducted any systematic analysis of the types or frequency of semantic drift or spurious insertions caused by LLM-augmented edge descriptions? If not, can they provide statistics or example cases showing where LLM guidance introduces incorrect or misleading hierarchy insertions?

Q2: Can the negative sampling strategy in self-supervised optimization be further detailed or ablated? Is there evidence that specific hard negative choices strongly affect generalization versus random negatives?

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
The paper explores how SLMs can effectively perform scientific taxonomy expansion by leveraging geometric constraints and guidance from LLMs.  
Authors propose SS-MONO (Structure-Semantic Monotonization), which integrates three components: (1) local taxonomy augmentation by frozen LLMs, (2) self-supervised fine-tuning of an SLM (e.g., DistilBERT) with hyperbolic constraints to preserve hierarchical transitivity, and (3) LLM-based calibration.  
Experiments on SemEval-Food, MeSH, and WordNet-Verb show that SS-MONO outperforms both traditional graph-based methods (e.g., TaxoExpan, TMN, QEN) and frozen LLMs like GPT-4o-mini, establishing SLMs as cost-efficient and competitive models for scientific taxonomy expansion.

### Strengths
- The paper presents a novel perspective that connects hyperbolic geometry in LLM embeddings with taxonomy reasoning.
- The paper is well-motivated and technically detailed.
- The self-supervised training approach removes the need for human annotation, and the idea of “borrowing knowledge” from LLMs while retaining the efficiency of SLMs is practically appealing.  
- The empirical results are comprehensive and support the claim of the paper.

### Weaknesses
- The impact of geometric regularization compared to simpler fine-tuning is not explained in the main text. I think it’s worth discussing the main takeaway of the ablation studies in 4.2.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces SS-MONO (Structure-Semantic Monotonization), a novel pipeline for efficient and accurate scientific taxonomy expansion. Recognizing the strong hyperbolicity in Large Language Model (LLM) embeddings, SS-MONO addresses the high computational cost and hallucination issues associated with directly using LLMs on domain-specific taxonomies. SS-MONO leverages LLM augmentation and distills this knowledge into Small Language Models (SLMs) through self-supervised fine-tuning enforced by geometric constraints. Key modules include a structure-dominated encoder using hyperbolic representation learning to preserve hierarchy (monotonicity) and a context-dominated encoder for contextual semantics. Experiments show that a fine-tuned SLM (e.g., DistilBERT-base-110M) consistently outperforms frozen LLMs and deep learning baselines on expansion tasks.

### Strengths
1. Efficiency and Cost-Effectiveness

SS-MONO implements an LLM-to-SLM distillation approach, fine-tuning a Small Language Model (SLM), such as DistilBERT-base-110M. This strategy provides a practical and economical alternative to the high computational costs and difficulties associated with directly using Large Language Models (LLMs) on domain-specific taxonomies.

2. Structural Integrity and Superior Overall Performance

The pipeline utilizes hyperbolic representation learning in its structure-dominated encoder to enforce geometric constraints (monotonicity), which preserves the hierarchical order. This structural awareness allows the fine-tuned SLM to consistently outperform frozen LLMs and domain-specific deep learning baselines overall.

3. Self-Supervised Training

The entire training process is self-supervised, guided by the existing taxonomy's topology. This design eliminates the need for expensive human labeling efforts for expansion tasks

### Weaknesses
1. Performance on Large-Scale Taxonomies

The methodology demonstrates significantly lower effectiveness on the large-scale WordNet-Verb dataset (13,936 nodes, depth 12) compared to smaller datasets. SS-MONO's overall average ranking on WordNet-Verb is 1626.52, which is not SOTA and substantially less favorable than its performance on SemEval-Food (239.17) and MeSH (436.82). Does this indicate that the proposed method could have issues in scalability and can be less effective on large-scale Taxonomies?

2. Non-Leaf Volatility with LLM Augmentation Observed

The performance metrics show that including LLM Augmented Descriptions (AD) does not always enhance intermediate (non-leaf) expansion. For example, on WordNet-Verb, the Non-leaf R@10 metric decreases sharply from 0.099 (SS-MONO w/o AD) to 0.035 (full SS-MONO). Other datasets have the same observation. What could lead to the performance difference between leaf nodes and non-leaf nodes? Is it because non-leaf nodes have more complicated relationship and LLM can hardly have clear annotations?

### Questions
1. Please update the bold format in all the experiment tables in the paper carefully. Currently many bolded numbers in the table are not exactly the one with the best value. This is very confusing and could lead to wrong conclusion.

2. Please address the above two concerns I have.

### Soundness
2

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
3

### Summary
This paper addresses the task of taxonomy expansion in scientific domains, specifically adding new concept nodes to an existing taxonomy (precisely in a directed acyclic hierarchy of concepts). The paper proposes a pipeline that borrows knowledge from an LLM and transfers it into SLM, which can then perform the taxonomy expansion efficiently. SS-MONO has three stages: (1) local taxonomy augmentation using an LLM - for each candidate insertion position in the taxonomy (parent node and a child node between which a new concept might be inserted) a pretained LLM is prompted to generate a textual explanation or description of local context (semantic features about the candidate position), (2) fine-tuning a SLM with geometric constraints to rank candidate insertion positions for a query concept (structure-dominated encoder that projects concept embeddings into hyperbolic space by nested entailment cones to keep hierarchical relationships, and a context-dominated encoder that embeds the textual descriptions) - here a key idea to enforce the monotonic ordering (child ≼ query ≼ parent in the embedding space) and (3) LLM-based calibrationof SLM score where to insert the new concept with use of second LLM call to re-rank the top-$k$ predicted positions. Experiments on three benchmarks demonstrate, in some cases, superiority and advantages of SS-MONO over other compared methods.

### Strengths
- The paper presents a creative integration of ideas from different domains: it combines hyperbolic geometry (for representing hierarchical structure) with LLM-based semantic augmentation in a small-model pipeline. While hyperbolic embeddings have been used for hierarchical tasks before and others have leveraged language models for taxonomy expansion, this provides a limited but still novel work in how it brings these together. Especially the notion of “Structure-Semantic Monotonization” is a novel formulation to ensure monotonicity in latent space (with use of nested entailment cones to enforce transitivity). 
- The technical quality of the work appears high. The paper is thorough in justifying and evaluating the approach.
- The training scheme is cleverly self-supervised, avoiding manual annotations (removal of nodes predict their insertion).

### Weaknesses
- Ironically, a method motivated by avoiding LLM usage still depends on LLMs at key points. The small model alone does a lot of the work, but the pipeline requires a capable LLM to provide the augmented descriptions and to perform final calibration. In the ablation without LLM augmentations (SS-MONO w/o AD), the small model’s performance, although competitive, is not clearly superior to the best prior methods. 
- The paper does not delve deeply into what the LLM-generated “textual explanations” look like or how consistent their quality is. This is a bit of a black box in the description. If the LLM outputs poor or hallucinated explanations for some candidate positions, does that ever confuse the SLM during training? One could imagine the LLM sometimes generating a misleading context (especially if the taxonomy contains very specialized terms that the LLM isn’t familiar with). The authors did not mention any filtering or human verification of the LLM outputs. It would strengthen the work to either demonstrate that these augmented descriptions are almost always accurate, or to describe measures to handle noise in those descriptions.
- Due to my understanding, there is no comparison to one of the baseline scenarios when LLM are similarly asked in turns to perform similar steps lika SS-MONO (pair insertion evaluairion, re-ranking).

### Questions
- Can the authors provide more details or examples of the prompt and output used for the LLM when generating the candidate position descriptions? This is currently abstract in the paper. For instance, if the candidate position is between parent concept P and child concept C, do you prompt the LLM with something like “Explain the relationship between P and its subcategory C” or “Give a description of where C fits under P”? And does the LLM output a few sentences describing that taxonomic context? An example would help in understanding what knowledge the LLM is injecting. Additionally, did you observe any instances of the LLM outputting incorrect information about the taxonomy? If so, how did you mitigate that (e.g., do you simply trust whatever the LLM says, or do you have a way to sanity-check it)? Clarifying this will help assess the reliability of the augmentation. And could you please evaluate LLM with the same procedure (subtasks) as SLM (inference).
- You introduce a complex hyperbolic constraint system for the structure-dominated encoder. Did you compare or ablate this against a simpler approach (for example, a Euclidean encoder or a transformer-based graph encoder without hyperbolic projection)? In Appendix I.1 you mention investigating the role of “geometric deep learning” – can you summarize those findings? It would be insightful to know how much the hyperbolic embedding improved things. Perhaps the model could also be trained in Euclidean space with a learned ordering constraint – would that fail or perform worse? 
- In cases where a concept has multiple true parent locations (non-leaf multi-attachments), how would you operationally use SS-MONO to attach it in all the correct places?

### Soundness
2

### Presentation
2

### Contribution
2
