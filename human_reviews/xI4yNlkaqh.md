# Towards 3D Molecule-Text Interpretation in Language Models

- Decision: Accept (poster)
- Scores: 6, 5, 6, 6

## Abstract
Language Models (LMs) have greatly influenced diverse domains. However, their inherent limitation in comprehending 3D molecular structures has considerably constrained their potential in the biomolecular domain. To bridge this gap, we focus on 3D molecule-text interpretation, and propose 3D-MoLM: 3D-Molecular Language Modeling. Specifically, 3D-MoLM enables an LM to interpret and analyze 3D molecules by equipping the LM with a 3D molecular encoder. This integration is achieved by a 3D molecule-text projector, bridging the 3D molecular encoder’s representation space and the LM’s input space. Moreover, to enhance 3DMoLM’s ability of cross-modal molecular understanding and instruction following, we meticulously curated a 3D molecule-centric instruction tuning dataset – 3D-MoIT. Through 3D molecule-text alignment and 3D molecule-centric instruction tuning, 3D-MoLM establishes an integration of 3D molecular encoder and LM. It significantly surpasses existing baselines on downstream tasks, including moleculetext retrieval, molecule captioning, and more challenging open-text molecular QA tasks, especially focusing on 3D-dependent properties. We will release our codes and datasets at https://github.com/lsh0520/3D-MoLM.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes to learn 3D molecule representations that can be used with pretrained large language models. Using pretrained 3D molecule encoder and large language models, the proposed method fine-tunes Q-Former to get molecule representation in language model space, and fne-tunes large language model to predict text (i.e., description) given at molecule embeddings and SMILES representations. The proposed method shows improvements compared with using large language model (e.g., LLaMA) naively.

### Strengths
- I'm not an expert in this area, but it seems the proposed approach is quite novel to leverage pretrained language model for downsteam tasks related to 3D molecules.
- The paper is generally well-written and easy to follow.
- The paper is well-motivated. 
- The paper includes code for reproducibility.

### Weaknesses
- The paper misses to include some failure cases (e.g., hallucination) of the proposed method. For instance, in Table 3(b), it seems the proposed 3D-MoLM wrongly interprets Globostellatic acid B (C34H48O6 while the ground truth is C33H48O7); the paper needs to discuss when the model tends to predict the wrong results. 
- I think the comparison in Table 3 might be somewhat unfair due to the different model sizes used for evaluation. Specifically, MolT5-Large has 800M parameters, while 3D-MoLM mainly uses LLaMA-7B models. In this respect, I suspect the improvements might simply come from the simple usage of language models with a large number of parameters and large training data rather than considering the 3D geometry of molecules. A good supporting experiment can be conducted by using a smaller LM for experiments (e.g., FLAN-T5-large).
- I have a similar concern in Table 2 as well since the model used in the proposed method is much larger and uses much more data for pretraining compared with other baselines.

### Questions
- Have the authors tried using larger model (e.g., LLaMA-13B)? If the authors tried, does it show the better performance?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a framework to interpret and analyze 3D molecules by equipping the LM with a 3D molecular encoder. The framework integrates a 3D unimol and a LLama language model, the framework is the same as BLIP2. Through a pre-train and fine-tune strategy, the authors demonstrate the effectiveness of the molecular caption, retrieval, and open QA tasks.

### Strengths
1. The paper proposes a new framework for 3D molecule-text pertaining and fine-tuning to conduct multiple 3D molecular tasks.
2. The framework is clear and reasonable.
3. The experiments are conducted different tasks and the effectiveness is demonstrated.

### Weaknesses
1. The novelty of this work is actually hard to say enough. It is clear that the framework is almost the same as BLIP2, the pre-training tasks and the fine-tuning stages are also the same. 
2. Besides, there are multiple different small places where I would like to ask questions. Most of them are unclear descriptions. Please look in the questions.

### Questions
1. The first question is about the Q-former (this may be the same as the Q-former in BLIP2, but the process is not clear here).  From Figure 2, the caption task and the contrastive task will use the text branch for text processing. These tasks are only used in pre-training, hence the question is that in the inference stage, the text branch in Q-former will be ignored (if I understand correctly). Therefore, the pre-training and the fine-tuning are mismatched, since we will only use llama2 for text generation, instead of the text branch in Q-former. What's the performance if we remove these two pertaining tasks and also the text branch in Q-former?
2. In Figure 3, stage 1 is molecule-text retrieval, and stage 2 is the molecular caption, but in section 2.2.1, stage 1 describes multitask training with the whole three tasks. See "Following BLIP,m we perform multi-task training, including....". Please clarify the correct one. 
3. For the downstream tasks, the model is fine-tuned after different stages, which causes multiple different models, each task for a model. I am wondering about the multitasking fine-tuning and the performance of one model. Or at least, after three/two stage trainings, then finetune on the downstream tasks. 
4. At the beginning of section 3.1, the authors mentioned "we assess the stage-1 checkpoint of ...", but the last sentence on page 6 is "Q-former benefits from the multi-task pre-training", if the stage only contains retrieval task, the claim is wrong.
5. From Table 4, it seems the performance gain is not as large as expected on open-text QA, do the authors have some analysis? 2D is similar to 3D.
6. In Appendix C, did the retrieval task use LORA for fine-tuning?

### Soundness
3 good

### Presentation
2 fair

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
This paper presented the 3D-MoLM, a novel molecule-text multimodal language model. This model is a combination of two pretrained foundation models: Uni-Mol and Llama2 for molecular and language understanding. The key contribution of this paper lies in its three-staged training process that integrates these two pretrained models.

In the first stage of training, the authors adopted the idea of Q-former and multi-task training approach, a method that has been previously proposed in vision-language multimodal model, to align the representation spaces of molecules and languages. Afterwards, the Q-former and Llama2 models are finetuned by solving the molecular captioning task. In the final stage, instruction tuning was performed to the model. To this end, the authors constructed 3D-MoIT, a 3D molecule-centric instruction tuning dataset.

The proposed method was tested on several molecule-language multimodal tasks, including molecule-text retrieval, molecule captioning, and molecular question answering. The experimental results demonstrated that the superiority of the proposed models by comparing them with several baseline models.

### Strengths
As a machine learning researcher, multimodal learning between molecule and language domain was expected to appear sooner or later. In this light, I think this study was well presented at the right time. The authors effectively adopted several techniques of existing visual-language multimodal learning to combine a molecular domain model and large language model. In addition, initial results of instruction tuning via 3D-MoIT dataset demonstrated the potential usage of large language model in molecular domain.

### Weaknesses
The main criticism I have is that the motivation and needs for molecule-language multimodal model remains unclear. Of course, the multimodality between molecule and language domain is a promising research direction for machine learning researchers and practitioners. However, let us consider the situation where scientists actually conduct research using the proposed model. In what scenarios can the proposed model be utilized and how can it lead to the synthesis or retargeting of novel/existing compounds? The molecular domain is the task for domain experts (i.e., chemists and biologists), not the general users. Thus, discussion on the use case of the molecule-language multimodality for domain experts is crucial for setting the ultimate goal for this line of research. Unfortunately, however, this paper and other work in this field, seems to avoid answering this pivotal question. I hope that the authors address this question in the rebuttal or the revised manuscript.

### Questions
First of all, as mentioned in the weakness section, I would like to ask the authors about the motivation and need for molecule-language multimodal learning. What are the specific use cases of the proposed model molecule-language model? How would scientists or practitioners (i.e., domain experts) be able to harness the proposed method in their research or business?

Regarding the question-answering task, the authors performed molecular properties prediction. Basically, properties including HOMO/LUMO are typically derived from the density functional theory calculations or molecular representation models such as Uni-Mol. Is there any specific rationale or chance that the language model enhances the prediction accuracy of these properties? Additionally, this work only compared the prediction accuracy with language models, and the ability to predict the properties comes from the Uni-Mol as aforementioned; therefore, I think the Uni-Mol should also be the baseline of the experiment, which will demonstrate the effect of the multimodal learning.

Finally, PubChem database provides the 3D conformation of each molecule. While I am not sure what method were employed to obtain the 3D structure in PubChem, it might be worthwhile to consider leveraging these data instead of MMFF-relaxed structures.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.

### Summary
This paper focus on 3D molecule-text interpretation, and propose 3D-MoLM: 3D-Molecular Language Modeling. Specifically, 3D-MoLM enables an LM to interpret and analyze 3D molecules by equipping the LM with a 3D molecular encoder. This integration is achieved by a 3D molecule-text projector, bridging the 3D molecular encoder’s representation space and the LM’s input space. Overall, this is an interesting work.

### Strengths
This paper is well-organized and written. The proposed 3D-MoLM enables an LM to interpret and analyze 3D molecules by equipping the LM with a 3D molecular encoder which is interesting and motivating.

### Weaknesses
The analysis of the proposed 3D-MoLM is not enough. The proposed method is interesting and straight-forward, any insight or analysis that could be offered to read to have a good understanding?

### Questions
Refer to Weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
