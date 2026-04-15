# Self-Specialization: Uncovering Latent Expertise within Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 6, 5

## Abstract
Recent works have demonstrated the effectiveness of self-alignment in which a large language model is, by itself, aligned to follow general instructions through the automatic generation of instructional data using a handful of human-written seeds. Instead of general alignment, in this work, we focus on self-alignment for expert domain specialization (e.g., biomedicine), discovering it to be very effective for improving zero-shot and few-shot performance in target domains of interest. As a preliminary, we first present the benchmark results of existing aligned models within a specialized domain, which reveals the marginal effect that "generic" instruction-following training has on downstream expert domains' performance. To remedy this, we explore self-specialization that leverages domain-specific unlabelled data and a few labeled seeds for the self-alignment process. When augmented with retrieval to reduce hallucination and enhance concurrency of the alignment, self-specialization offers an effective (and efficient) way of "carving out" an expert model out of a "generalist", pre-trained LLM where different domains of expertise are originally combined in a form of "superposition". Our experimental results on a biomedical domain show that our self-specialized model (30B) outperforms its base model, MPT-30B by a large margin and even surpasses larger popular models based on LLaMA-65B, highlighting its potential and practicality for specialization, especially considering its efficiency in terms of data and parameters. Our code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This study is driven by the observed limitations of pre-trained large language models (LLMs) when handling specialized tasks, particularly in the biomedical question and answering domain.

The underlying issues stem from the practice of fine-tuning these LLMs on broad instruction datasets, which often results in only marginal enhancements for niche domains.

To address this, the authors introduce a novel technique termed "self-specialization." Unlike the conventional self-alignment method, self-specialization incorporates domain-specific seeds and an external knowledge retriever. This strategy cultivates an instruction dataset tailored to the specific needs of the target domain.

Empirical results highlight the efficacy of this approach: LLMs that underwent self-specialization significantly outperformed their non-specialized counterparts.

### Strengths
- Methodology: The self-specialization technique introduced is both straightforward and powerful. The authors effectively demonstrate that, by refining the instruction dataset for a target domain, one can substantially elevate the performance of an LLM.

- Results: The outcomes are compelling. The self-specialized technique not only outpaces the baseline LLM with general instruction tuning across various NLP tasks, but even a 30B self-specialized model can occasionally surpass larger models of 65B capacity.

- Structured Presentation & Clear Communication: This research is characterized by its lucid motivation and coherent narrative, seamlessly bridging the gap between identified issues and the proposed solutions. The results are presented with clarity, reinforcing the dominance of the introduced methods.

### Weaknesses
At its core, the proposed method is somewhat an amalgamation of existing concepts. While termed "self-specialization", it essentially contrasts with "self-alignment" (for instance, as seen in Alpaca) and incorporates "domain-specific knowledge-guided generation". The enhancements observed are not surprising since the retrieval-augmented generation, grounded on specific knowledge retrieval, naturally offers a more thorough and targeted knowledge base for instruction tuning. Thus, one could argue that self-specialization is essentially self-alignment augmented with RAG, somewhat constraining its novelty.

Furthermore, rather than relying on RAG to generate the instruction dataset with the foundational model, the authors might have delved deeper into refining the dataset construction process. Several uncharted avenues remain: (1) devising superior methods for seed generation, considering aspects like topic ratios or instruction diversity; (2) optimizing answer generation for the instructions, with focus on enhancing and validating answer quality. Regrettably, this research offers only a cursory exploration in these dimensions.

### Questions
- In 4.1, the authors mentioned 5K synthetic instruction dataset is generated. Was there any reason for this number? If we increase/decrease the number, how would the yielded instruction-tuned model perform?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a self-specialization mechanism to automatically generate domain-specific instructional data and improve language model performance within specialized domains. The proposed approach is straightforward: 1) sample seed instructions from established NLP benchmarks within the target domain; 2) utilize the language model to generate new domain-specific instructions based on the initially sampled seeds; 3) employ domain-relevant unlabeled documents to generate responses corresponding to the generated instructions; 4) fine-tune the language model using the synthesized instructions, resulting in the development of a final domain-specialized model.

### Strengths
The paper is well-motivated, as specializing the language model in a specific domain is an area of interest. Additionally, the paper is well-structured and easy to follow.

### Weaknesses
Despite the promising results reported in this paper within the biomedical domain, I still feel uncertain about its contribution for the following reasons:
1) The experimental results are not convincing. Table 1 reports the comparative results of the base model and the self-specialized model, with the authors stating "the scores (F1) witness a rise from 25.15 to 36.63 in a zero-shot setting." However, this is the average result, and the datasets "BioASQ-Yesno" and "Medical Drugs" contribute the most differences, with a difference of around 60-70 in "BioASQ-Yesno. I highly doubt the correctness of the results for this dataset. This is a binary classification dataset, and the base model performs much worse than random guessing (50.0 if labels are balanced), which is the first odd point. Secondly, as the number of demonstrations increases, the F1 score of the base model surprisingly decreases. This implies that the base model must learn information from the prompted demonstrations. If the model is not performing well on a dataset, the F1 score should remain almost the same or should not exhibit such consistent declines. I am considering if the authors mistakenly reversed the labels of "yes" and "no," leading to these counterintuitive results.
If we exclude the "BioASQ-Yesno" and "Medical Drugs" datasets, the average F1 score improvement is down to around 2.5, far less than the reported 11.48 (36.63 - 25.15). This makes doubt not only on the reported results in Table 1 but also on Fig. 4 and its claims of the model's superiority over all 65B models despite its ≈2.2x smaller size.
2) The Gouge-L scores for the "DDI" and "Medical" datasets are identical to their respective F1 scores. I haven't investigated if this is possible, but please check the reported results carefully.
3) During the seed generation phase, the proposed method requires the NLP benchmarks in the target domain. This limits the proposed approach to extend to other domains, as not all domains have this NLP benchmark accessible. With the paper using 80 seeds, one possible solution could be manually generating them; however, the paper didn't mention this point.
4) How about the results if applying the proposed scheme to a larger model, such as LLaMA-65B? The base model used in the paper initially shows not good results (20-30 F1 scores), which are relatively easy to improve. If a larger model with better initial results is used, is it still feasible to have improvement through the proposed scheme? 
5) Some places lack professional writing. For example, in Eq. (3), $p_{lm}$ is introduced without prior definition. While the intended meaning may be inferred, a scientific paper should maintain consistency and rigor in its notation and explanations.

### Questions
Please consider the questions I listed in the Weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focuses on self-alignment for expert domain specialization. It first conducts a benchmarking of existing aligned models, i.e., Alpaca-65B and Dromedary-65B, revealing that existing self-aligned models achieve marginal improvements compared to the original model. Then, it proposes a self-specialization method that leverages domain-specific unlabelled data and a few labeled seeds for the self-alignment process. The experiments on a medical domain show the proposed method outperforms its base model, and larger popular models as well.

### Strengths
1. The topic of specialization is important for deploying LLM to a specified domain.
2. The paper is well-written and easy to follow.
3. The idea of self-specialization is interesting, which utilizes both seed instructions and generated ones together.
4. Several experiments are conducted to evaluate the proposed method. The proposed self-specialization method outperforms the base model and larger LLaMA variants.

### Weaknesses
1. The benchmark only includes two aligned models. Better to include more aligned models for comprehensive benchmarking.
2. It claims that 'we hypothesize that the model expertise in different domains resides in “superposition” in the model’s parameters and hidden states'. But there are no further theoretical explanations or experimental results to support this claim. If the "superposition" could be explained in details, the application scope of the proposed method may be more clear.
3. The metrics of the y-axes should be added in figure 2 and 5.

### Questions
1. Why do you choose the two aligned models? Why not include other models for benchmarking?
2. Does the selection of seed instructions affect the final performance? If so, how do you eliminate this effect?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a method called self-specialization, which extends the self-instruct method to a specific domain. The experimental results are evaluated on a set of biomedical tasks using two LLMs. The problem addressed in this paper is interesting and the experiment results show promise. However, the paper can be significantly improved by generalizing the proposed method to at least another domain, as the main contribution of the paper lies in the proposal of a method for domain specification. Additionally, the effectiveness of the proposed method, which is based on finetuning, needs to be clarified as the baseline performance is based on in-context learning. To improve the clarity of the method, an ablation study should be included, e.g., examining the contribution of domain-specific response generation to the overall performance.

### Strengths
- The paper is well-written and easy to follow, providing sufficient technical details. The authors also mention that the data, code, and trained model will be open source.
- The empirical findings demonstrate that incorporating unlabeled data positively impacts the model's ability to effectively respond to queries within a specialized domain, particularly in the challenging context of biomedical research.

### Weaknesses
- The proposed method, self-specialization, is an extension of the self-instruct [1] work to recover certain expertise in the LLMs. However, the method is only designed and evaluated for one specific domain without showcasing its generalization ability to other domains.
- The compared methods in the paper are based on zero-shot/few-shot settings, while the proposed method uses LoRA for finetuning. It would be beneficial to include stronger task-specific comparison methods to better illustrate the effectiveness of the proposed approach.
- Since the method has multiple components, it would be helpful to show the contribution of each component to the overall performance. For example, how does domain-specific response generation impact the final results?
- The performance in the knowledge sparse domain is uncertain. It is understood that domain response generation involving harnessing knowledge is one of the main contributors to the specification, but it is important to understand the method's bottleneck when existing knowledge is limited for certain domains.
- The paper should discuss potential data contamination and address how the authors ensure that the data for downstream testing does not overlap with the generated data.

[1] Wang, Yizhong, et al. "Self-instruct: Aligning language model with self generated instructions." arXiv preprint arXiv:2212.10560 (2022).

### Questions
See above.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
