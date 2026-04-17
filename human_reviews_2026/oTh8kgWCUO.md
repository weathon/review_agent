# How do Large Language Models Learn New Domain Knowledge?

- Decision: Reject
- Scores: 4, 4, 0, 6

## Abstract
Despite the success of pre-training, we are unable to replicate it for continual learning. We don't fully understand how large language models (LLMs) acquire knowledge. We design controlled experiments to measure how they learn complex knowledge in the setting of continued pre-training, probing at two levels of generalization: factual and compositional. First, we show that paraphrasing enables scalable acquisition of knowledge, in which repetition increases learning with diminishing returns. Second, we find that auxiliary views of the underlying knowledge, which formulate and communicate the same knowledge in different ways, yield significantly better generalization. This generalization extends to both compositional knowledge and even factual recall. We postulate that these auxiliary views frequently occur in pre-training corpora and construct a sort of scaffolding. Third, we find that LLMs can possess only a partial understanding of the prior knowledge required for domain adaptation, and bridging these gaps markedly increases learning. Lastly, we examine how learning dynamics differ with model size, post-training, data cleaning, and data replay.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors study how the model acquires new knowledge in the continual pretraining process. They construct factual and compositional probes to evaluate model learning dynamics. The compositional probes is an interesting angle but the construction of the probe doesn't seems rigorous enough to support the claims in paper.

### Strengths
* Understanding how model learns compositional knowledge during continued pretraining is interesting and important.

* The study covers a wide ranges of model sizes.

* the ablation study and variants of training corpus are comprehensive.

### Weaknesses
* Claims about 49 paraphrasing reduce performance “excessive linguistic diversity may hinder factual recall” (Line 299-300), how does the author ensure the linguistic diversity? 

* The diversity of the probe is not established, so the claim “This demonstrates a genuine generalization of knowledge rather than surface-level pattern matching” (Line 322-323) is not very well supported. A trivial explanation: The probes might all look similar so the trends look similar.

* Line 301 “paraphrasing enables robust knowledge acquisition over extended training”; how do the author measure “robust acquisition”?

* ``Scaffolding'' is not defined but seems important for argument of the work (i.e. appear and highlighted in abstract)

* Unclear arguments about data replay: Replay is meant to mitigate catastrophic forgetting, but learning target domain has nothing to do with catastrophic forgetting. I think the experiment results are expected. I am not sure why would training data from pretraining corpus would help learning new knowledge that doesn’t exist in pretraining corpus. I am lost about what the author want to prove.

### Questions
* Line 187 does "2,298 probes" include both factual and compositional probes?

* Is it correct that the compositional probe requires original + related sentence to answer?  How do the author make sure information “Related sentence” is not contained in original sentence? 

* What exactly does one batch of “new knowledge” contain? Does it contain all original sentence and related sentences for a "new knowledge"? For paragraphs, prior knowledge and auxiliary views, are each of the augmentation applied to both original sentence and related sentence?

* Figure 3: why 1B with Priori knowledge has a sudden drop in mean log prob?

* Unclear why post-training is required: If the evaluation is only log prob or generate from probe prefix, why is post-training needed for the experiments? Figure 6,7,8

* Figure 10, it seems surprising that paraphrase only which has very little tokens counts (< 0.05K tokens?) are super effective. Does author have some explanation for this?

* Could the author discuss the difference between this work and prior work [1, 2] on learning dynamics?

[1] Pretrained Language Model Embryology: The Birth of ALBERT

[2] Probing Across Time: What Does RoBERTa Know and When?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper investigates **how LLMs acquire new domain knowledge via continued pretraining (CPT)** through a series of controlled experiments. Six recent AI papers are treated as novel “domains,” and the authors construct both **factual** and **compositional** probe sets to measure learning. They compare three training strategies: (1) source-only documents, (2) paraphrasing, and (3) **auxiliary views** (textbook, blog, and StackExchange–style materials).  
Key findings include:  
- Learning improves with repetition but saturates after ~100 exposures.  
- Paraphrasing mitigates overfitting and enhances generalization.  
- **Auxiliary views act as pedagogical scaffolding**, substantially improving compositional understanding.  
- **Prior-knowledge pretraining** on prerequisite concepts boosts downstream domain learning.  
- **Data replay** from the pretraining corpus consistently harms new knowledge acquisition.  
Experiments are conducted on OLMo2 models (1B/7B/13B), with detailed ablations and insightful discussion on the “scaffolding hypothesis” for LLM learning.

### Strengths
1. **Well-controlled methodology** with dual-level probes and systematic construction/validation for factual and compositional learning.  
2. **Clear, interpretable insights** on paraphrasing, auxiliary views, and prior knowledge that could guide future CPT practice.  
3. **Meaningful negative result:** excessive data replay hinders domain learning — a practically important observation.  
4. **Thoughtful discussion** connecting findings to human learning theories (e.g., scaffolding, double descent), offering conceptual depth.

### Weaknesses
1. **Limited scope and scale:** Only six CS papers and models up to 13B; unclear if findings generalize to other domains or frontier-scale models.  
2. **Synthetic auxiliary data:** Auxiliary “textbook/blog/Q&A” materials are LLM-generated, which might introduce lexical overlap or bias with probes despite stratified analysis.  
3. **Missing baselines:** The paper does not compare against retrieval-augmented or instruction/RL-based knowledge injection methods, leaving uncertainty about whether scaffolding remains advantageous under other paradigms.

### Questions
1. Which components of the auxiliary bundle (textbook, blog, Q&A) contribute most to compositional gains — and do these effects persist with **human-written** materials?  
2. Could alternative **replay curricula** (e.g., staged or post-cycle replay) mitigate forgetting without degrading target-domain learning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
Questions: "
RQ 1: How is domain knowledge on varying levels, factual and compositional, acquired?
RQ 2: When learning new knowledge, does the gap in the LLM’s prior knowledge matter?
RQ 3: How does data replay, post-training, chunking, and other factors affect the learning of the new knowledge? "

Our corpus consists of six papers from the field of artificial intelligence

Claims:  "
(1) Acquisition of complex knowledge requires significant repetition, saturating after approximately 100 exposures in our study. 
(2) Diverse, auxiliary views dramatically improve the learning of both factual and compositional knowledge in a way that paraphrasing does not. 
(3) Bridging knowledge gaps by first training the LLM on prerequisite concepts significantly improves learning. 
(4) Conversely, increasing the amount of data replay from the original pretraining corpus monotonically harms the acquisition of new knowledge. We also ablate several aspects of our training setup to provide pracitcal suggestions for continued pretraining. "

### Strengths
This paper is very interesting in its fundamental question. What is the impact of so called auxiliary views - views of the information that are intended to increase scaffolding and understanding. This is a good and important question.

### Weaknesses
LLMs largely lack the ability to synthesize complex, novel information from primary sources alone. - this claim is not supportable for LLMs in general. Finding that current LLMs don't do something is very different from the general class of methods lacking the ability. 

When you cite a paper in the related work section, don't like 5 different papers. "Since GPT-3, the overarching narrative of LLMs
has been scale, regarding both model size and data volume (Brown et al., 2020; Kaplan et al., 2020;
Hoffmann et al., 2022; Carlini et al.; Kandpal et al., 2023; Tirumala et al., 2022)." This is not helpful and is not convincing. If they are all important, explain clearly what you intend to support with each. 

Another example: "Continued pretraining has proven to non-trivial(Wang et al., 2021; Janget al., 2021; Hu et al., 2023; Ovadia et al., 2024; Hoffbauer et al., 2024; Jiang et al., 2024)." There is no purpose discernible from the paper for these citations.

Grammar and writing matter when disseminating research - this is not ready for review: "Given the recency of the focus on continued pretrainig, the exist body body is young and investigative in nature (Yıldız et al., 2024; Ou et al., 2025)" 

The formal statement of the problem has an error: "What are the properties of a good training corpus K that most effectively enable the acquisition of knowledge Kfor fθ?" The corpus is C_k

"In addition to probes, we generate paraphrases, prior knowledge, and auxiliary views for each paper in the dataset, utilizing GPT-4.1 for paraphrasing and GPT-5-mini for the rest. While these alternate texts may seem to confer an implicit advantage by distilling the model’s knowledge into the training corpus, this is intentional for the auxiliary views. We treat the LLM as a proxy for domain experts who produce materials such as textbooks and blogs which enters the pretraining data. Inspired by prior works  (Gunasekar et al., 2023; Allen-Zhu & Li, 2024; Jiang et al., 2024), we focus on textbooks, Stack Exchange–style Q&A, and blogs as auxiliary views." - GPT 5 is not a domain expert in the topic of a research paper on which it has not been trained. So, the experiments can't actually evaluate the influence of auxiliary views as defined as follow on explanations by domain experts ie. the illustrated transformer as a decompression of "Attention is all you need".  

monotic is not the word which is intended in figure 1. The legend in figure 1 doesn't match the actual signals in the graph. It's not clear what is being conveyed. It is very bad practice to put graphs on the same axis with different values on that axis. It makes the results incomparable. 

While the central question of the paper is good, the presentation in the paper makes it difficult to judge the voracity of the evidence. The presentation of the paper is not ready for publication.

### Questions
No questions.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work analyzes the knowledge acquisition dynamics of LLMs through the controlled setup of continued pretraining. The analysis reveals that the increase of log probability induced by repeated exposure leads to saturation, while providing the model with multiple views of given knowledge improves its learning and generalization. Based on the extensive analysis, the paper conjectures that LLMs largely lack the ability to infer novel compositions of primary knowledge sources without the aid of auxiliary views.

### Strengths
(S1) The experiments are appropriately designed to deal with each research question.

(S2) The results reveal several interesting phenomena, in particular the double-descent-like behavior and the effect of model size on the benefit of auxiliary views.

(S3) The paper is well-written with clean and precise language.

### Weaknesses
(W1) While providing various insights, (as mentioned in the discussion) the experiment relies on a single domain of research paper understanding, and it is unclear whether the same result can be applied to other specific domain knowledge, for example, structured documents that require the expertise in medical or legal domain.

(W2) The mechanistic understanding of the observed behaviors is limited, making the claims on the ‘structure’ stay hypothetical (for example, “auxiliary views help the model build a more structured knowledge representation”, in L345). It would be great to see additional analysis of what makes it different from the model trained with/without auxiliary views, in terms of the information encoded in representations or gradients. In addition, studies on synthetic data reveals that grokking often accompanies certain structures of representation (e.g., [1]). Will such “grokking-like” behavior observed in this study related to some induced structure inside the parameters?

(W3) It has been actively discussed in recent studies that domain adaptation often leads to catastrophic forgetting. While the current analysis reveals many potential advantages of domain knowledge acquisition under certain conditions, it might be at the cost of forgetting unrelated knowledge that should be maintained. Could you share your thoughts on the forgetting dynamics under domain adaptation, or additional experiments to rule out that providing auxiliary views does not aggravate catastrophic forgetting?


[1] https://arxiv.org/abs/2405.15071

### Questions
Please refer to the points in the weaknesses section.

### Soundness
3

### Presentation
4

### Contribution
3
