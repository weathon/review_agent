# A General Framework for Producing Interpretable Semantic Text Embeddings

- Decision: Accept (Poster)
- Scores: 6, 5, 6, 8, 8

## Abstract
Semantic text embedding is essential to many tasks in Natural Language Processing (NLP). While black-box models are capable of generating high-quality embeddings, their lack of interpretability limits their use in tasks that demand transparency. Recent approaches have improved interpretability by leveraging domain-expert-crafted or LLM-generated questions, but these methods rely heavily on expert input or well-prompt design, which restricts their generalizability and ability to generate discriminative questions across a wide range of tasks. To address these challenges, we introduce \algo{CQG-MBQA} (Contrastive Question Generation - Multi-task Binary Question Answering), a general framework for producing interpretable semantic text embeddings across diverse tasks. Our framework systematically generates highly discriminative, low cognitive load yes/no questions through the \algo{CQG} method and answers them efficiently with the \algo{MBQA} model, resulting in interpretable embeddings in a cost-effective manner. We validate the effectiveness and interpretability of \algo{CQG-MBQA} through extensive experiments and ablation studies, demonstrating that it delivers embedding quality comparable to many advanced black-box models while maintaining inherently interpretability. Additionally, \algo{CQG-MBQA} outperforms other interpretable text embedding methods across various downstream tasks. The source code is available at \url{https://github.com/dukesun99/CQG-MBQA}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce CQG-MBQA, a general framework for producing interpretable semantic text embeddings. It builds these embeddings from a set of yes/no questions that are designed to be highly discriminative (by separating text clustered by a pre-trained embedding model). To improve efficiency, the answers to these yes/no questions are distilled into a smaller model. The CQG-MBQA model reveals improvements relative to baseline interpretable models.

### Strengths
- The authors study an interesting and important problem
- They obtain strong performance results with reasonable efficiency
- They evaluate interpretability and cognitive load well, in addition to more standard performance metrics

### Weaknesses
- The main issue seems to be the authors’ treatment of related work: the CGQ method generates questions through prompting and filters them based on their discriminative ability. The baseline QA-Emb also generates questions through prompting but filters them with a sparsity penalty. From their description, the authors don’t seem to implement the sparsity penalty, which likely skews the comparisons.
- The authors should discuss this [2023 style embeddings paper](https://arxiv.org/abs/2305.12696), which was an early precursor to the work here
- The authors should clarify whether distilling the yes/no answers into a single model is a novel contribution — the style embeddings paper & QA-Emb paper both seem to do this as well

### Questions
Can the authors provide a comparison with the baseline including the sparsity penalty? Maybe showing the performance as a function of the number of questions kept?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work proposes a "framework" for creating interpretable semantic embeddings. They tackle the important and relevant problem of creating embeddings that are useful for search & clustering but also understandable to humans.

### Strengths
- This paper tackles the important problem of creating interpretable text embeddings
- Some of the steps laid out in the "framework" explanation will be useful for other practitioners
- The source code is available and could be used by other researchers and engineers to build systems for interpretable embeddings
- Consideration of the tradeoff between interpretability and quality is interesting – although I have qualms with the "cognitive load" measurement of interpretability, which are mentioned below.

### Weaknesses
- I find it important to point out that this paper isn't really proposing a "framework" in a very general sense; it's much closer to a method (and in fact the authors interchange the two words liberally throughout the paper). For this reason I object to calling it a framework at all and would prefer the paper to be about CQG-MBQA, which is an interesting and apparently effective method for interpretable
 text embeddings.
- As a related point, the organization is confusing. Currently the paper mixes together the "framework" part (which should be a general process for producing interpretable embeddings) with the "method" part (about CQG-MBQA) and also some "experimental setup" and "results"  (all in Section 3). As one example of the murky sectional boundaries, is post-processing really a necessary step of the framework?
- I'm also not sure if the Case Study is exactly a Case Study.
- The cognitive load metric seems unprincipled and lacks grounding in real cognitive science. 
- Cognitive load is simply the number of overlapping "yes" answers (or 1s in the embeddings) between the representations of a pair from an STS dataset. It is highly dependent on dimensionality and sparsity (Figure 4 & 5). It also doesn't really make sense because the interpretability of an embedding should depend on how many yes's there are for a pair *compared to other pairs*; embeddings cannot be understood simply by looking at the inner product of a pair of embeddings.  
- Many of the important design decisions in the framework are not ablated. Is filtering important? How much does choosing positive and negative samples matter, or clustering? How much does training a surrogate model affect performance?
- This is not necessary to me for accepting the paper, but a real human study could be crucial for arguing that these embeddings are in fact more interpretable
- Due to the complicated system and lack of ablations, it is not easy to understand why these embeddings outperform other interpretable embeddings such as QAEmb
- Unclear cost analysis of running this method on a downstream dataset

I think this citation could be relevant:
- Learning Interpretable Style Embeddings via Prompting LLMs (Patel et al., EMNLP Findings 2023)

### Questions
- What inspired this measurement of cognitive load?
- How much inference time does it take to run on the retrieval datasets?
- How were the retrieval datasets chosen?
- How much does the QA model quality affect embedding quality?

### Soundness
2

### Presentation
3

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
The paper introduces CQG-MBQA, a framework designed to produce interpretable semantic text embeddings for diverse NLP tasks. The framework uses Contrastive Question Generation (CQG) to automatically generate meaningful yes/no questions without domain experts. The Multi-task Binary Question Answering (MBQA) model answers these questions, producing embeddings with human-interpretable dimensions at a much lower cost than answering with LLMs, while maintaining comparable accuracy. The authors validate CQG-MBQA through experiments, comparing it to black-box and interpretable models across STS, retrieval, and clustering. The experimental results show that CQG-MBQA offers better embedding quality than existing interpretable models and provides comparable embedding quality to several black-box models, maintaining high interpretability and efficiency.

### Strengths
The ideas behind CQG and MBQA are novel and effective, supported by thoughtful experiments and ablation studies. The paper is clearly written and well-structured. Given the increasing demand for model transparency, CQG-MBQA could have significant implications and represent a meaningful approach that would be of interest to the ICLR audience.

- The paper builds upon QAEmb with important innovations: a contrastive approach to question generation that improves discrimination using positive, hard negative, and easy negative samples for fine-grained specificity, and a multi-task model for efficient inference without the use of LLMs.
- The technical quality is demonstrated through comprehensive empirical evaluation across multiple tasks and strong baselines including both interpretable and black-box models, with clear cost analysis showing significant efficiency gains through MBQA during inference. For reproducibility, detailed implementation specifics and code are provided.

### Weaknesses
- In retrieval tasks, there is a significant performance gap compared to black-box models, and the performance is also lower than BM25. Therefore, additional performance comparisons are needed when applying them to various downstream tasks such as sentiment classification and retrieval. 
- Lack of ablation studies to assess the efficacy of the proposed approach
  - lack of comparison between different models in Figure 4 and 5, and lack of comparison between the MBQA method and directly using the LLM’s outputs.
  - comparison between vanilla CQG with positive and hard/easy negative, and CQG with positive and negative samples
  - comparison between having and not having the probing mechanism to refine the generated questions
- Also, because the cognitive load is defined using the dot product, this measure would be directly influenced by the total number of questions. A normalized version (e.g., dot product divided by number of questions) would provide a fairer comparison across different interpretable models in Table 5.
- Having cost analysis would be beneficial as MBQA requires significant LLM inferences (or API calls) during training time and even more may be required during Post-processing (probing). 
- Including a case study on bad examples would also be beneficial—for instance, displaying cases where two texts result in similar embeddings even when those two texts do not necessarily have similar semantics. Are they completely off? Or how could one improve your approach?

### Questions
- Did you evaluate your model on the fMRI task presented in the QAEmb paper for a direct performance comparison?
- How do variations in the number of initial clusters (k) or the use of different clustering methods affect the performance of CQG?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces CQG-MBQA (Contrastive Question Generation - Multi-task Binary Question Answering), a general framework to create interpretable semantic text embeddings for NLP tasks. This framework emphasizes interpretability, which is essential for tasks requiring transparency, such as legal and medical applications. Traditional black-box text embedding methods, while effective, lack interpretability, limiting their utility in such cases. By comparison, CQG-MBQA is able to generate interpretable semantic text embeddings via binary yes/no questions. To be concrete, this framework first generates binary yes/no questions through contrastive question generation (CQG) using GPT-4o-mini for the entire corpus. Then, it fine-tunes a multi-task binary question-answering (MBQA) model by distilling knowledge from GPT-4o-mini. In this way, one can use MBQA to create the interpretable embeddings for text, without relying on LLMs, thus reducing the API costs. The experimental results show that CQG-MBQA performs comparably to advanced black-box models, and outperforms other interpretable text embedding models across various downstream tasks.

### Strengths
1. CQG-MBQA focus on producing interpretable embeddings, which is important for domains requiring transparency.
2. Compared with QAEmb,  CQG produces more discriminative questions.
3. By integrating MBQA, the framework achieves cost-effective embeddings compared to LLM-based alternatives.
4. This paper conducts extensive experiments on semantic textual similarity, retrieval, and clustering tasks, showcasing its utility and competitiveness.

### Weaknesses
Please refer to the Questions.

### Questions
1. Does the CQG-MBQA framework need to generate a new set of yes/no questions every time it is applied to a different dataset? If so, is there a way to enhance the generalizability of the CQG-MBQA model? In other words, could a more universal set of yes/no questions be designed to handle multiple tasks/datasets, rather than creating a separate set tailored to each specific task/dataset?

2. Figure 4 shows that with around 3,000 questions, CQG-MBQA can achieve high-quality text embeddings on STS tasks, and using additional questions does not improve embedding quality; instead, it decreases interpretability. Does this imply that the yes/no questions generated by CQG have semantic overlap or inclusion relationships? In other words, is there a large number of semantically similar questions within the set of yes/no questions?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes CQG-MVQA, an intepretable embedding framework. The framework generates questions and binary answers about texts, and trains binary classifiers on each question. Each of the prediction to these questions forms a dimension of an interpretable embedding. It is shown that these interpretable achieves decent performance on MTEB and a reduced cognitive load for interpretability compared to baseline methods in interpretable embeddings.

### Strengths
1) The question generation component of the framework concerns generating questions that are both discriminative and general. It groups similar texts by clustering for the generation of questions, such that nuanced questions can be asked for each group, as opposed to simple questions in the baseline method. The concept is analogous to leveraging hard negatives in regular training of embedding models.
2) The authors show good understanding at related work; the implementation and the evaluation are sound from the perspective of sentence embeddings.

### Weaknesses
1) Performance ablations about setups in the framework can be very interesting although currently missing (e.g., performance across different dimensionality, question difficulties, different encoding models, etc..).
2) Implementation details can be moved more to the main paper as they are mostly in appendices.

### Questions
1) In order to achieve performance of regular dense representation models, which aspects of the framework and the implementation details do the authors think worth scaling up? Is there any interesting evidence? e.g., better datasets for generating questions; more questions to form the embedding dimensions, etc..
2) From my understanding in the appendix, the paper uses UAE-large-v1 as the encoding model, why? What happens if the encoding model is some better models - does it help with the performance of the final interpretable embeddings?

### Soundness
3

### Presentation
3

### Contribution
3
