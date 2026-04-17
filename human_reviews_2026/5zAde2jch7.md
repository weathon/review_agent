# One protein is all you need

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Generalization beyond training data remains a central challenge in machine learning for biology. A common way to enhance generalization is self-supervised pre-training on large datasets. However, aiming to perform well on all possible proteins can limit a model’s capacity to excel on any specific one, whereas practitioners typically need accurate predictions for individual proteins they study, often not covered in training data. To address this limitation, we propose a method that enables self-supervised customization of protein language models to one target protein at a time, on the fly, and without assuming any additional data. We show that our Protein Test-Time Training (ProteinTTT) method consistently enhances generalization across different models, their sizes, and datasets. ProteinTTT improves structure prediction for challenging targets, achieves new state-of-the-art results on protein fitness prediction, and enhances function prediction on two tasks. We also demonstrate ProteinTTT on two challenging case studies. We show that customization via ProteinTTT enables more accurate antibody–antigen loop modeling and improves 19% of structures in the Big Fantastic Virus Database, delivering improved predictions where general-purpose AlphaFold2 and ESMFold struggle.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Protein Test-Time Training (ProteinTTT), a novel method for customizing pre-trained protein language models (PLMs) for individual protein sequences during inference. The method improves performance across protein structure, fitness, and function prediction tasks, especially for proteins with limited homologous sequences. This motivation is clear and well-supported: in most scenarios, researchers are primarily interested in single or a few protein sequences, and ProteinTTT is able to enhance this. Extensive experiments demonstrate that ProteinTTT is transplantable across diverse PLM backbones.

### Strengths
* The idea of introducing test time training to PLMs is interesting and convincing. Some case studies also confirm the validity of this method when existing methods are unable to solve specific protein sequences.
* The experiment is rigorously conducted. ProteinTTT is tested across diverse downstream tasks and various backbones.
* By enabling more accurate predictions for individual proteins, ProteinTTT could significantly advance areas such as personalized medicine, antibody design, and protein engineering, where accurate, protein-specific models are essential.

### Weaknesses
* As shown in Fig. A4, the time consumption caused by ProteinTTT is non-trivial, especially when pLDDT is calculated. However, the performance improvement seems trivial in Tab. 2, which is my primary concern. Despite some case studies showing that ProteinTTT can improve significantly, the overall performance does not show similar results.

* While the results are strong on average, the paper could benefit from a deeper analysis of the cases where ProteinTTT fails or degrades performance. Figure A6 indicates that for a small fraction of proteins, the prediction quality worsens after customization.

* Simply get better representation from backbones, and freezing the prediction heads is kind of confusing. Modifying the distribution of representation without aligning it to the prediction heads is somehow irrational.

### Questions
* Can you provide a percentage of improved samples in Tab. 1 and 2? Ideally, most of the samples will be improved over backbones by test time training. 
* If not all samples are improved, can you further analysis the failure cases?
* For fitness and function prediction, how was the number of customization steps (T=30) determined, and how sensitive are the final results to this choice?

### Soundness
4

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
3

### Summary
The authors propose ProteinTTT, a method to adapt protein foundation models like AlphaFold2 and ESMFold to specific new proteins at test time. In essence, it involves running a few steps of self-supervised fine-tuning of the main trunks of the model in question independently of any task-specific heads trained on top of them. The authors show that this procedure improves performance on a variety of downstream tasks like folding and fitness prediction, especially for proteins without known homologues.

### Strengths
The manuscript is exceptionally clearly written, and the method is intuitive and easy to follow. The authors include a diverse set of experiments run on several different models, and I appreciate the inclusion of confidence intervals throughout.

Methods like ProteinTTT obviously have long pedigrees in the language modeling and computer vision literature, but this is the first paper I'm aware of that applies this sort of idea to protein modeling.

### Weaknesses
The most glaring weakness in my eyes is that the effect of adaptation seems to be relatively limited. On several benchmarks, improvements are measured as fractions of percentage points. On several others (especially in section 5), the improvement is more difficult to gauge; at various points, the authors make claims about the fractions of proteins that are improved by the method, but details about the magnitude of the improvement in these cases seem to be pretty sparse. Figure 5b hints that some proteins do indeed see large improvements, but it's also difficult to read. Some more clarity here would be appreciated.

Some other assorted issues/potential areas of improvement:

1. The authors occasionally use pLDDT of the tuned model (e.g. to choose the number of optimization steps), which seems a little problematic to me; since you're training on one protein, you're probably also inflating confidence on that protein. Could the authors validate that this isn't happening, perhaps by running early stopping experiments that "cheat" by observing downstream metrics? Another thing that would be good to see would be something like Figure 5b, but pitting ESMFold + ProteinTTT LDDT against ESMFold + ProteinTTT pLDDT
2. Is there a reason the authors only test 18 CAMEO proteins? How does the method perform on proteins that aren't extremely low-confidence?
3. Sometimes, the text overstates the results. For example: "As shown in Figure 4a, ESMFold + ProteinTTT achieves significantly
higher average LDDT scores compared to general-purpose ESMFold." The improvement in this case is not actually "significantly higher" in the statistical sense, since the error bars overlap.

### Questions
See above.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes ProteinTTT, a test-time training framework aimed at improving the downstream performance of protein language models. Motivated by the observation that sequence-level perplexity correlates with task accuracy and that low performing cases often exhibit high perplexity, the authors finetune the pretrained model at test time using the perplexity of each target sequence. The method is evaluated across three downstream tasks and several case studies, demonstrating consistent improvements across both tasks and backbone protein language models. The results suggest that ProteinTTT can effectively enhance existing protein language models without requiring large-scale retraining or additional labeled data.

### Strengths
1. The manuscript is well written and polished; clear, concise, and detailed enough in both method and evaluation to be easy to follow.
2. The experimental design is comprehensive, covering three relevant downstream tasks with appropriate baselines. The inclusion of case studies and in-depth analyses further strengthens the empirical validation.
3. The results are consistent and convincing, showing that ProteinTTT can effectively enhance the performance of existing protein language models.
4. Introducing test-time training to protein language models, and framing perplexity minimization as a way to adapt to distribution shifts in downstream tasks, offers a novel and insightful perspective.

### Weaknesses
1. The proposed method seems inherently limited in scope, it applies only to protein language models and only to sequence-based tasks. This restricts its applicability to broader classes of protein modeling problems.
2. While the reported improvements are consistent across benchmarks, many gains are modest and appear bounded by the baseline model’s performance. The authors briefly acknowledge this (e.g., in G1), but a clearer discussion in the main text along with analysis of failure cases would help readers better understand when and why ProteinTTT is most effective.
3. Not necessarily a weakness, but the title "One protein is all you need" feels somewhat overstated. ProteinTTT indeed brings measurable benefits, yet its performance remains constrained by the underlying model’s capacity and generalizability. That said, the title is understandable as a stylistic choice.

### Questions
**(A) Scope and Applicability.**

 1. Since ProteinTTT relies on self-supervised fine-tuning over sequence perplexity, it appears primarily suited for tasks that take protein sequences as input and for models trained with language modeling objectives. This seems limit its direct applicability to sequence-based models rather than structure-based ones (e.g., AlphaFold3) which usually have stronger performance. 
 2. Could the authors discuss whether ProteinTTT can be extended to other task types beyond the three evaluated in this work, such as sequence design?

**(B) Performance on Structure Prediction.**
1. Can authors add the performance of the current any state-of-the-art results on the CAMEO benchmark (e.g., AlphaFold2, AlphaFold3, Boltz) as a reference?
2. The recently proposed DPLM-BiT model (https://arxiv.org/abs/2504.11454) demonstrates strong folding performance which also uses a masked protein language modeling approach. Including it as a baseline could provide a more rigorous comparison and help assess whether ProteinTTT maintains its advantage on this class of models.
3. Has the method been explored for more challenging structure prediction tasks, such as protein–protein complex prediction?

**(C) Experimental Details and Behavior.**
1. In the customization steps (e.g., Fig. 3), would continued test-time fine-tuning eventually lead to overfitting or catastrophic forgetting?
2. How many fine-tuning steps are typically recommended for each task, and are these consistent across tasks?
3. Can the authors discuss more on the conditions where ProteinTTT might fail? For test time training, an extended discussion on failure cases or scenarios would provide valuable practical guidance.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Protein Test-Time Training (ProteinTTT), a self-supervised framework that customized pretrained protein language model to individual target proteins during interence. ProteinTTT fixed the downstream task head and finetuned the model backbone. Experiments show that this consistently improves model performance in different tasks with different kinds of models.

### Strengths
1. The method is simple and general. It does not require additional data and can be easily applied to different pretrained PLMs. And the appraoch leverages lightweight finetuning (LoRA and limited steps), making it resource-efficient.
2. The method consistently improve the performance on different tasks and different models.
3. The authors use case studies to demonstrate real-world applications. For biologists and chemists who are interested in specific proteins, this could be useful.

### Weaknesses
1. The authors claimed that the method can be easily extended to models with autoregressive masking, while they did not do that. I suggest the authors might try to show the results on ProGen.
2. The improvement on larger models are marginal. While ProteinTTT consistently improve results, the improvement on larger model is marginal and even negligible. For example, in table 2 when it comes to 650M model. The improvement is nearly negligible. I wonder what will happen in larger model like 1B or 3B.
3. There are already a lot of inference scaling ideas rather than test time training in current trends in LLM. It would be valuable for the authors to clarify whether proteinTTT could similarly benefit from such inference-time scaling strategies (e.g. Use MSAs or other context) instead of perform test-time training.

Overall, While the work is technically sound and relevant, its contributions appear more empirical and domain-specific than methodological. The paper might be more impactful if expanded with additional biological validations. In its current form, the scope and framing could align more naturally with a computational biology or bioinformatics venue rather than a general ML conference.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2
