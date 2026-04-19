# Mufu:  Multilingual Fused Learning for Low-Resource Translation with LLM

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Multilingual large language models (LLMs) are great translators, but this is largely limited to high-resource languages. For many LLMs, translating in and out of low-resource languages remains a challenging task. To maximize data efficiency in this low-resource setting, we introduce Mufu, which includes a selection of automatically generated multilingual candidates and an instruction to correct inaccurate translations in the prompt. Mufu prompts turn a translation task into a postediting one, and seek to harness the LLM’s reasoning capability with auxiliary translation candidates, from which the model is required to assess the input quality, align the semantics cross-lingually, copy from relevant inputs and override instances that are incorrect. Our experiments on En-XX translations over the Flores-200 dataset show LLMs finetuned against Mufu-style prompts are robust to poor quality auxiliary translation candidates, achieving performance superior to NLLB 1.3B distilled model in 64% of low- and very-low-resource language pairs. We then distill these models to reduce inference cost, while maintaining on average 3.1 chrF improvement over finetune-only baseline in low-resource translations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Mufu, which turns translation into post-editing task by providing auxiliary translations and target translation from teacher model. The student model learns in-context to produce the correct target translation and is then fine-tuned against references. Languages for auxiliary translations are chosen from URIEL and they evaluate using PaLM S family models along with Gemma 2B,7B on FLORES 200 (iid), and NTREX (ood). The paper contains thorough ablation studies as well as cross lingual attention alignment which helps understanding or interpreting how model is learning through in-context.

### Strengths
- The paper is very clearly written and easy to follow.
- They combine 2 interesting learning paradigms: ICL and parameter tuning and their core focus is on very-low and low resource language which I really liked. 
- They perform evaluation on NTREX which is important for ood evaluation.
- The experiments performed by authors are quite extensive. I especially liked mufu5hrl, mufu5tr, distilled, and lora which corroborate their approach of selecting 5,10,20 related languages from URIEL.
- Quantitative evidence provided in Figure 3 is quite helpful in knowing how language transfer is taking place. Moreover, the attention pattern further helps in understanding how attention pattern is making mufu models perform better.

### Weaknesses
- No model sizes available for PaLM2 family of models. I’m not sure how to compare them with Gemma or NLLB.
- If I were to just compare on the basis of chrF score, only PaLM2 XXS -NLT and PaLM2 S are able to beat NLLB 1.3B distilled model in both FLORES 200 and NTREX (and Gemma 7B on FLORES 200). Rest all are inferior to NLLB 1.3B distilled. One suggestion for authors in this case will be to add `Latency` column for all models (higher for mufu and lower for distilled models) to show the trade off between accuracy and latency which will help readers understand how competitive other models are.
- The authors have mentioned this but finetuning an LLM (or even NLLB with 1B+ param) with just 787 sentences and in-context learning will definitely lead to overfitting which is evident by the fact that mufu20lora performed better than full finetuning. I wonder if that is the case for other models too? 
- It’s great they used Gemma 2, an open weight model but I’m slightly disappointed that majority of their experiments use PaLM2 models which are not public like Gemma 2. 
- Two iteration process (teacher model followed by student model) is quite expensive. The authors have mentioned that distillation helps to alleviate the problem but it only worked for NTREX in PaLM2 XXS - NTL (not for Gemma 7B), performance on FLORES 200 for both distilled models is lower than NLLB 1.3B. 
- The authors experiment with one learning paradigm i.e., in-context learning for LLMs for distillation. Did they try distillation from model outputs (not the one fine-tuned with mufu20)? How much better or worse is in-context learning compared to vanilla distillation?

### Questions
- Were there any accidental translations in a different language for Mufu{5,10,20}?
- What exactly is Win% vs teacher? For instance, for NLLB 1.3B distilled, its chrF is 46.0 whereas that of teacher is 43.7, still its win% is 41.3? It means NLLB 1.3B was less than 50% correct when compared to teacher model still its chrF score is higher? Another example, Win% vs teacher is 56.2 for NLLB 54B MoE (48.9 chrF) whereas for mufulora20 with PaLM2 S it is 99% with chrF less than NLLB 54B MoE on FLORES 200. It will be great if authors can formalise what is Win% vs teacher.
- Can the authors explain In theory… model outputs (line 207-211)?

### Soundness
3

### Presentation
4

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
This paper tackles low-resource translation quality improvement in LLM models. To maximize data eﬃciency in the low-resource setting, the authors introduce a new approach called Mufu, including automatic selection of multilingual translation candidates and an instruction tuning to correct inaccurate translations via the prompt. Experimental results on Flores-200 dataset for English-XX directions show robustness and achieves better performance against NLLB 1.3B distilled model in 64% of low- and very-low resource language pairs.

### Strengths
- experimental results show some effectiveness of the proposed approach
- the idea of leveraging multilinguality via the prompt sounds technically good

### Weaknesses
-  unclear about the experimental results; how to decide the best prompt template for mufu; any impacts of language combination used in the prompt template - for example, have you ever tried adding high-resource language translation pairs during training to enhance multilingual training  with high and low-resource language pairs?
-  results are not convincing enough, maybe due to low-resource setting with limited improvement in ChrF. Can you report other metrics such as sacreBLEU scores? Have you tried finetuning LLM with low-resource monolingual data so that the LLM can more effectively enhance Mufu.

### Questions
Please see the weaknesses for the questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces "Mufu"  , which is a method for low-resource language translation using a multilingual fused learning approach, specifically targeting large language models (LLMs).
The Mufu method, which aims to address the challenge that large language models (LLMs) perform well in translating high-resource languages but still struggle with low-resource languages. The Mufu prompting approach turns the translation task into a post-editing task, leveraging the reasoning capabilities of LLMs with auxiliary translation candidates, requiring the model to assess input quality, align semantics cross-lingually, copy from relevant inputs, and override incorrect instances. Experiments show that LLMs fine-tuned with Mufu-style prompts achieve better performance than the NLLB 1.3B distilled model in 64% of low- and very-low-resource language pairs on the Flores-200 dataset.

### Strengths
1. Interesting research, Introduces Mufu, a novel approach leveraging multilingual context and post-editing for low-resource language translation.
2. Employs automatically generated candidates and instructions to correct translations, enhancing LLM's reasoning capability.
3. Demonstrates robustness against poor-quality auxiliary translations, outperforming specialized NMT systems in many low-resource pairs.
4. Proposes a hybrid learning paradigm, combining in-context learning and finetuning for improved translation quality.
5. Implements knowledge distillation to reduce inference costs while maintaining performance gains in low-resource translations.

### Weaknesses
1. Experiment Method Optimization， Consider incorporating a more diverse set of low-resource languages in the experimental dataset to better generalize the findings and evaluate the model's performance across a wider linguistic spectrum.

2. Experiment Conclusion Enhancement， Suggest conducting ablation studies to isolate the specific contributions of different components of Mufu, such as the impact of various auxiliary languages, to fine-tune the approach and maximize translation accuracy.

3. 5-shot Prompting Improvement， Explore the use of meta-learning strategies in 5-shot prompting to enhance the model's ability to quickly adapt to new translation tasks with limited examples, potentially improving the efficiency of the learning process.

### Questions
1、 more diverse set of low-resource languages in the experimental dataset will be helpful
2、  the impact of various auxiliary languages can be deeply analyzed
3、 prompt analyzation can be improved

### Soundness
3

### Presentation
3

### Contribution
3
