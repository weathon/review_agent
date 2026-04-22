# Low rank adaptation of chemical foundation models generate effective odorant representations

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 4, 4, 6

## Abstract
Featurizing odorants to enable robust prediction of their properties is difficult due to the complex activation patterns that odorants evoke in the olfactory system. Structurally similar odorants can elicit distinct activation patterns in both the sensory periphery (i.e., at the receptor level) and downstream brain circuits (i.e., at a perceptual level). Despite efforts to design odorant features to better predict how they interact with the olfactory system, there is still no universally accepted approach to this problem. We demonstrate that feature-based approaches that rely on pre-trained foundation models to generate odorant representations $\textit{do not}$ significantly outperform classical hand-designed features on odorant-receptor binding tasks. Instead, we show that it is necessary to fine-tune these features to increase predictive performance. To show this, we introduce a new model that creates olfaction-specific representations: $\textbf{L}$oRA-based $\textbf{O}$dorant-$\textbf{R}$eceptor $\textbf{A}$ffinity prediction with $\textbf{CROSS}$-attention ($\textbf{LORAX}$). We compare existing chemical foundation model representations to hand-designed physicochemical descriptors using feature-based methods and identify large information overlap between these representations, highlighting the necessity of fine-tuning to generate novel and superior odorant representations. We show that LORAX produces a feature space more closely aligned with olfactory neural representation, enabling it to outperform existing models on predictive tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents a benchmark of several chemical foundation models in the scope of the notoriously difficult problem of modelling olfaction. The authors showed that the zero-shot application of chemical foundation models does not increase the performance in molecule-receptor response prediction tasks compared to using standard physicochemical features. Thus, to mitigate this, the authors suggest using LoRA to fine-tune chemical foundation models (e.g. ChemBERTa-77M-MTR) to model interactions at the beginning of the olfactory system. The authors demonstrate that combining LoRA with ProSmith model (introduced in Kroll et al., 2024) leads to 1.5\% increase in the performance on the random splits of the challenging M2OR dataset compared to ProSmith without fine-tuning.

### Strengths
The paper made an interesting observation, that embedding of chemical foundation models is similar to the traditional physicochemical descriptors. In addition, the proposed model, LORAX, achieves a marginal increase in the SOTA preformance on M2OR dataset compared to ProSmith model which does not use LoRA fine-tuning.

### Weaknesses
The proposed algorithm, LORAX, is rather incremental and only adds a well-established LoRA fine-tuning to ChemBERTa in the previously-published ProSmith model. The most interesting result of the paper is the observation that embedding spaces of the tested chemical foundation models are aligned with each other and with standard physicochemical descriptors. This is a surprising result, given that chemical foundation models are widely-studied and have been shown to outperform physicochemical descriptors in molecular properties prediction. However, the authors do not provide any explanation of such an unexpected phenomenon, making the result much less impactful. Additionally, the authors do not provide the code, so I am unable to reproduce the results, nor check the data splitting strategies.

### Questions
Some of the experiments are not clear. For instance, how are Table 3 and Table 9 related? Both tables discuss the performance on M2OR dataset. Assuming that the authors evaluate the models in Table 3 on "ec50" pairs due to the label noise (following Hladi\v{s} et al.), I would expect that the values in the tables are comparable. However, "MorderdDescriptors" achieves MCC of 0.709 in Table 9, outperforming the proposed LORAX approach in Table 3.

In Figure 5 and Figure 7, the authors compare LORAX,  ChemBERTa, and other odorant representations to the "neural representation" derived from Carey dataset. However, in my understanding, this comparison is not fair, since LORAX is trained on the neuronal activity data and other models are zero-shot. In addition, despite being trained, LORAX does not appear to be significantly more aligned with neuronal activity than, zero-shot "ecfp" or "gin\_supervised\_infomax" representations (first columns in Figure 7).

Since the authors do not provide the code during the review process, I am unable to check data splitting, which is crucial in the field of protein-molecule interaction prediction. This said, how are the unseen receptor/molecule splits constructed in Table 2? The similarity of the protein sequences/molecular structure between the train and test sets can largely mislead the performance evaluation (see e.g. Zhang et al. Rethinking the generalization of drug target affinity prediction algorithms via similarity aware evaluation, ICLR 2025). In addition, the generalization experiment is conducted on the Carey dataset. What is the performance on other two datasets? In particular, how this compares to the generalization performance presented in Hladi\v{s} et al.?

Considering the style of writing, I find the paper imprecise and unpolished, especially in the use of terms such as "binding", "affinity" and "activation". Data in Hallem and Carey datasets are average spikes per second for insect electrophysiology experiments, and data in M2OR are mostly in vitro functional assays in heterologous systems. Neither of these experiments directly measure "binding" as binding of the molecule to the protein is only a necessary but not a sufficient condition to observe the experimental response. In the case of M2OR dataset, the term "affinity" is also imprecise, since the authors consider activation prediction, which is different from predicting a measure of affinity, such as EC50. 

Section 4.1. is difficult to follow. At the beginning the authors write "We trained LORAX on the Carey dataset and found it performs almost identically to ProSmith" and then suddenly in Figure 4 LORAX performs better than ProSmith. Is the performance in Figure 4 only for the Transfomer part of ProSmith? If yes, then this is not clear from the text and the figure caption.

Lastly, I found several incorrect citations in the paper (Cong et al. 2022, Haddad et al. 2008a, Lalis et al. 2024, Sypetkowski et al. 2024). I believe that the errors come from some citation tool, but the authors should thoroughly check the citations as if passed unnoticed, this shows disrespect to the original authors.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors contribute LORAX, a novel method that incorporates fine-tuned chemical foundation models and pre-trained protein foundation models for odorant-receptor binding prediction tasks. The authors motivate the need for their approach with an extensive evaluation of different representations across relevant datasets, highlighting: (i) that the representations encoded by pre-trained chemical foundation models do not significantly improve the performance in odorant-receptor binding prediction tasks versus hand-crafted physiochemical features; and (ii) that is due to the significant alignment between these pre-trained representations and the hand-crafted ones. Finally, the authors evaluate LORAX across different datasets, showing that it learns a more generalizable representation to unseen odorants, and provides a representation space more similar to the underlying neural representation.

### Strengths
- **Originality**: The originality of this work is highlighted in two different aspects: (i) the use of low-rank adaptations to fine-tune foundation models of chemical data for olfactory tasks; (ii) the use of such fine-tuned models in combination with pre-trained representation models of protein data for odorant-receptor binding prediction tasks. While fine-tuning foundation models of chemical data for olfactory tasks has already been explored before (see [1]), the use of LoRAs for this purpose is, to the best of my knowledge, novel.

- **Quality**: I found the paper to be of good quality, with minimal typos (one exception being "adapated" in line 435). Despite this, I encourage the authors to improve the quality of some of the figures (as an example, Figure 1B (a) is not really comprehensible without improving the caption). While not particularly novel, the idea of using low-rank representations to fine-tune foundation models is sound.

- **Clarity**: The paper is overall clear. In particular, I thoroughly enjoyed the structure of the work, starting by motivating the need for LORAX with extensive experimental results, and subsequently evaluating LORAX, comparing to previous results.

- **Significance**: This work contributes to an emerging line of research within olfactory machine learning that resorts to pre-trained foundation models of chemical data, due to the lack of large-scale, high-quality and diverse datasets of olfactory data. Fine-tuning approaches, such as the one presented in this paper, are undoubtedly an important tool to address the specificity of olfactory data, and thus significant. 

**References**:
- [1] Taleb, Farzaneh, et al. "Can transformers smell like humans?." Advances in Neural Information Processing Systems 37 (2024): 72032-72060.

### Weaknesses
- In Section 3, the authors present an extensive study on affinity prediction, considering different odorant representation models and input conditions: only odorant, molecule + odorant, and a processing of these two representations using ProSmith. The results show a significant increase when using protein information and ProSmith. However, I've found interesting that no ablation condition with only protein information was done, especially given that this information seems fundamental for the task itself. Can the authors show results of such ablation?

- It is not clear what are the main differences between LORAX and ProSmith, beyond the LoRA component of the chemical model. Can the authors elaborate on the main differences between the two models?

- The authors motivate LORAX by highlighting that current odorant representations perform similarly to each other, yet the results in Line 317 show no significant performance difference between LORAX and ProSmith. Same for the results in Table 3, where ProSmith actually achieves a significantly higher AUROC. While achieving SOTA is not a requirement across all metrics and situations, in this case it is unusual since the authors are fine-tuning the pre-trained chemical representation for their specific downstream evaluation task. Can the authors comment on this? Why would a practitioner use your approach instead of ProSmith?

- Since LoRAs is such an important component of their contribution (and the main difference between their approach and the baselines), it is unfortunate that no extensive evaluation of different LoRAs is made, in regards to rank, layers to be fine-tuned, etc... Can the authors present this results? How did the authors select their LoRA architecture?

- Some important details are currently missing in the paper such as definition of the evaluation metrics in the tables, and identification of which datasets are used for the results presented in the tables. Can the authors add this information?

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this work, the Authors investigate the representations of odorants through the lens of their predictive power for the odorants' affinitie for olfactory receptors. To this end, they compare standard sets of chemical features and several representations learned in foundation models for chemicals and proteins. Upon finding the overall similarity between different representations of odorants, they propose a LoRA-based finetuning framework for learning the representations relevant to olfaction. They test treir and existing representations on several odorant-receptor affinity datasets.

### Strengths
- Comprehensive literature overview

The paper considers a comprehensive set of the representations (e.g. the chemical descriptors) and the models (e.g. GNNs) that have found widespread use in olfaction; thus, the work is well-positioned in literature and relevant to the olfactory community

- Comprehensive evaluation

For both existing and new representations, the work offers a comprehensive analysis, thus adding new knowledge about the tools that people use in practice and not only about the new method.

- Insights into the similarity and differences between conventional representations

They find, through the analysis, that many different representations of odorant are, in fact, similas, especially those leading to good predictions for odorant-receptor affinities. Additionally, the similarity between the decodeability is shown for the most of the representations

- Ablations

Futhermore, the imapct of different model inputs is investigated; the utility of the joint condiferation of the representations of many odorants and many reptors in a single model (e.g. ProSmith) is shown.

- Text clarity

The text is super clear and well-structured.

### Weaknesses
- The proposed model doesn't seem to improve upon the baselines.

Through Table 2, LORAX is better than ProSmith for unseen odorants but worse for unseen receptors (that said, all R2-scores here are really small). Through Table 3, LORAX is the best in recall and F-score but not the best in precision and AUROC. Unless there's a principled way to determine that these metrics are more important thatn the others, one can't claim that the model is the best.

### Questions
What substantiates that claim that LORAX has the best representations?

### Soundness
3

### Presentation
4

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
The authors show that for olfactory tasks, chemical foundation models do not yield representations that outperform physicochemical descriptors on odorant-receptor binding tasks. To bridge this issue, they propose a LoRA strategy to finetune these representations, achieving better predictive performance and generalizability over their benchmarks. The idea is clearly communicated and straightforward to understand, and the authors do reasonably well in their evaluation of existing foundation models to emphasize their point. However, in my opinion, the conclusion could be made stronger with the inclusion of a few more key benchmarks that would make the results more convincing. I recommend that the paper be accepted to ICLR after these changes are satisfactorily addressed.

### Strengths
The authors present the idea in a clear manner -- there are varying approaches towards the representation of molecules in olfaction, and the lack of olfactory data can make the use of foundational models more desirable over a supervised learning approach. There needs to be a benchmark for the efficacy of foundational models towards olfactory tasks, and this work addresses this gap.

The authors also evaluate a wide variety of representations beyond foundational models, and consider supervised learning approaches in addition to physicochemical representations. I appreciate that their analysis involves comparisons of performance spreads beyond just looking at mean performance.

I particularly enjoyed the analysis in Sec 4.1 where the authors analyzed feature importance to detect that the ProSmith model intentionally underutilizes the transformer feature.

### Weaknesses
Table 3 is incomplete in my opinion -- to fully compare everything on equal footing, the authors should use the approaches Hladiš et al. and Chithrananda et al. used in their paper to complete the entirety of Table 3. In line with this comparison, I would like to see the performance of a supervised GCN (widely used in other olfactory tasks) on all the datasets to properly show that fine-tuned foundational models indeed provide a better representation.

Additionally, I concur with the authors in Appendix A that they’ve evaluated a large number of foundation models, but as they also mention, these models are not recent ones, and a conclusion like the authors are proposing needs to be based on models that are more recent. 

I was partially confused by Figure 4B, and thought a more natural way to show that LORAX is useful is to show that the proportion placed on < cls > is higher for LORAX to be consistent with their analysis with ProSmith’s poorer performance compared to LORAX.

I think the authors should dial back their claims that this phenomenon is general for all olfactory tasks, as they have only demonstrated this for molecule-receptor binding tasks. 

The authors propose a strategy where they concatenate the features, but cross-attention has also proven to be a reasonable strategy in molecule-receptor binding tasks (i.e. Chithrananda et al.).

### Questions
1) I recall that the M2OR dataset is very sparse in terms of molecular-receptor pairs, and so it should not be surprising that the MO model performs poorly due to the lack of data. I’m not sure if this is the case for the other two datasets the authors use in this work.
2) What is the source of the neuron response data that is used to generate the neuron representation in Figure 5? 
3) What is the performance of a GCN supervised baseline?
4) What is the performance of a more recent foundational model?
5) Does cross-attention outperform concatenation, and does it lead to different conclusions from what you observe?

### Soundness
3

### Presentation
3

### Contribution
4
