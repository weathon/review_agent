# Align after Pre-train: Improving Multilingual Generative Models with Cross-lingual Alignment

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 6

## Abstract
Multilingual generative models obtain remarkable cross-lingual capabilities through pre-training on large-scale corpora. However, they still exhibit a performance bias toward high-resource languages, and learn isolated distributions of sentence representations across languages. To bridge this gap, we propose a simple yet effective alignment framework exploiting pairs of translation sentences. It aligns the internal sentence representations across different languages via multilingual contrastive learning and aligns model outputs by answering prompts in different languages. Experimental results demonstrate that even with less than 0.1‰ of pre-training tokens, our alignment framework significantly boosts the cross-lingual abilities of generative models and mitigates the performance gap. Further analysis reveals that it results in a better internal multilingual representation distribution of multilingual models.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduce a simple and effective multilingual alignment framework (AFP), there are two major components: 1. multilingual contrastive learning on internal representation, 2 Cross-lingual instruction tuning on the outputs.  Through the alignment, the author argues that the performance of low-resource languages will be improved through the knowledge transfer.  Experiments on 52 languages show that  with a small number of training tokens, the propose method could improve the cross-lingual ability of generative models.

### Strengths
The paper is clearly written and easy to understand. 
The proposed MCL and CIT method is solid in improving the cross-lingual ability of pre-trained language models.
And I like the experiment design and analysis: 1. Extensive experiment on 52 languages to show the cross-lingual ability, also got good results on unseen languages Thai and Turkish . 2. use embedding distance and uniformity to measure the distribution of multilingual representation.
The content in Appendix will make it easier for others to reimplement this work.

### Weaknesses
1. I think the innovativeness of this paper is limited:  the proposed multilingual contrastive learning on internal representation is not new, as the author listed in Sec2.1 Para 1. Also in "On learning universal representations across languages" Wei. ICLR2020, the proposed contrastive learning method for universal representation learning is similar.    
    On the other hand, the cross-lingual instruction tuning also has some similar work: like "Few-shot Learning with Multilingual Generative Language Models" use cross-lingual demonstrations in the tuning process.  As there are no comparison system in the main experiment, I suggest the author add one or two related work to better show the merits of this work.

2. A small issue with the machine translation experiment,  I think the parallel data used in AFP framework should be added into the baseline model through fine-tuning or demonstrations in prompt for a fair comparison.  And the improvement in BLEU is marginal to me. 
So I am a little skeptical on the improvement of MT task.   Also a typo in Sec 3.3 BLUE --> BLEU

### Questions
Please refer to the weakness for context:
1. Is there any related work in cross-lingual instruct tuning that could be added as the compare system?
2. For the MT experiment, is it possible to add the parallel data used in AFP framework into the baseline model for a fair comparison?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes align after pretrain (AFP) method to improve multilingual capabilities of LLM. AFP consists of two training objectives:  (1) contrastive learning objective to align embedding spaces and (2) cross-lingual instruction objective to improve cross-lingual generation. AFP is evaluated on 4 tasks: natural language inference, paraphrase detection, reasoning and machine translation. The evaluation using XGLM and BLOOM models show that when AFP is used, it boosts the multilingual performances of the base models.

### Strengths
- The paper combines two popular ideas in NLP to improve multilingual capabilities of LLM. Although the idea of making pretrained LM more multilingual is not new. [Earlier work](https://arxiv.org/abs/2002.07306)  has shown the extreme of making bilingual LM out of English only LM. 
- The AFP method is simple, and it should work for languages that have parallel data.
- Analysis of embeddings after AFP

### Weaknesses
- The motivation for AFP is because current public LLMs are English centric, thus I wonder the value of this approach and its impact compared to training a native multilingual LLM from scratch such as [PolyLM](https://arxiv.org/abs/2307.06018) , which is multilingual by design. I think it’s important to have PolyLM in the evaluation to understand the gap in multilingual capabilities  and where AFP stands.

- Evaluated tasks are simple (except few-shot machine translation). XNLI, PAW-S, XCOPA, XStorzyCloze, and XWinograd don’t seem to test the resulting model’s ability to generate languages. For instance, for XNLI, the model only needs to generate one token (yes/also/no), for XCOPA, for reasoning tasks, the model doesn’t need to generate the model, instead it is used to score the answers and find the best one. Thus I think that the generative abilities in multilingual setup are not properly measured. I wonder if other multilingual tasks such as summarization/QA are more appropriate for evaluation.

- The crosslingual finetuning step leverages machine translation outputs, which is prone to error.

- While the evaluation is done based on the existing cross-lingual datasets, which is perhaps outdated in the age of LLMs with emergent abilities. In order to advance multilingual ability, i think the first step (and also the very important one) is to have an adequate  multilingual benchmark for LLMs. Without that, it's difficult to assess any claim about  improving multilingual generative models.
- The cross lingual finetuning step leverages machine translation outputs, which is prone to error. LLMs are known for hallucinating their generation, thus cross-lingual finetuning could make it even worse for generation in non-English languages. Does AFP cause more hallucination in language generation? Can you measure that? And what is the strategy to prevent such cases.

### Questions
See the question in the above section.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes align after pre-train (AFT), a cross-lingual alignment framework that enhances cross-lingual abilities of multilingual generative language models. AFT utilizes translation pairs to align the internal sentence representations across languages, through multilingual contrastive learning. Besides, AFT performs cross-lingual instruction tuning that makes the language model respond in target languages given source language prompts. With two objectives combined, the authors trained multilingual LMs on the basis of several multilingual LLMs. The experimental results demonstrate that the alignment greatly improves the cross-lingual ability of the models on several multilingual tasks under zero-shot and few-shot settings.

### Strengths
- The paper conducts extensive experiments where AFT is evaluated on various multilingual LLMs (XGLM, BLOOM, and Llama) on four types of multilingual tasks. The results show that AFT consistently improves the LLMs across models, tasks, and setups. Besides, the AFT is also evaluated beyond the bilingual setup, with the alignment extended to 52 languages.
- The provides ablation studies on the training objectives and different alignment methods. The ablation studies support the effectiveness of multilingual contrastive learning and cross-lingual instruction tuning when they are used together.
- The paper presents visualization of the natural language representations from different languages, and demonstrates the alignment effects on the hidden representations inside LLMs.

### Weaknesses
- The novelty of the proposed method is limited. The AFT framework is a combination of cross-lingual contrastive learning and instruction tuning objectives, both of which have been shown to be effective for enhancing cross-lingual abilities in related works. (1) InfoXLM[1] utilizes cross-lingual contrastive learning with translation pairs to enhance multilingual LMs, and demonstrates its effectiveness in improving cross-lingual transferability. The difference is whether to put the contrastive object to MLM-trained models or  CLM-trained models. (2) BLOOMZ[2] applies multilingual multitask finetuning (a.k.a. instruction tuning) and observes better zero-shot performance.
-  In section 2.2, the paper claims that the proposed cross-lingual instruction tuning (CIT) is proposed to further align the outputs, which is more difficult than the multilingual instruction tuning. However, I did not find ablations to support this. I would guess the gain is mainly from instruction tuning instead of its cross-lingual alignment effect of CIT.
- Insufficient literature review in the related work section.

[1] InfoXLM: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training

[2] Crosslingual Generalization through Multitask Finetuning

### Questions
- Does cross-lingual instruction tuning work better than multilingual instruction tuning? 

After applying machine translation to the instruction tuning data, you could obtain at least twice the training data (N times training data for N target languages). It is unclear why multilingual instruction tuning, which has N times data, would perform worse than the proposed cross-lingual instruction tuning.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
