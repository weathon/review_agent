# Context Clues: Evaluating Long Context Models for Clinical Prediction Tasks on EHR Data

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 5, 8, 6, 10, 6

## Abstract
Foundation Models (FMs) trained on Electronic Health Records (EHRs) have achieved state-of-the-art results on numerous clinical prediction tasks. However, prior EHR FMs typically have context windows of $<$1k tokens, which prevents them from modeling full patient EHRs which can exceed 10k's of events. For making clinical predictions, both model performance and robustness to the unique properties of EHR data are crucial. Recent advancements in subquadratic long-context architectures (e.g. Mamba) offer a promising solution. However, their application to EHR data has not been well-studied. We address this gap by presenting the first systematic evaluation of the effect of context length on modeling EHR data. We find that longer context models improve predictive performance -- our Mamba-based model surpasses the prior state-of-the-art on 9/14 tasks on the EHRSHOT prediction benchmark. Additionally, we measure robustness to three unique, previously underexplored properties of EHR data: (1) the prevalence of ``copy-forwarded" diagnoses which create artificial token repetition in EHR sequences; (2) the irregular time intervals between EHR events which can lead to a wide range of timespans within a context window; and (3) the natural increase in disease complexity over time which makes later tokens in the EHR harder to predict than earlier ones. Stratifying our EHRSHOT results, we find that higher levels of each property correlate negatively with model performance (e.g., a 14% higher Brier loss between the least and most irregular patients), but that longer context models are more robust to more extreme levels of these properties. Our work highlights the potential for using long-context architectures to model EHR data, and offers a case study on how to identify and quantify new challenges in modeling sequential data motivated by domains outside of natural language. We release all of our model checkpoints and code.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper investigates the advantages of long context models in the healthcare domain by evaluating the impact of context length on clinical prediction tasks using four models—Mamba, Llama, GPT, and Hyena—trained on millions of longitudinal EHR records. Additionally, the study assesses model robustness against three properties of EHR data: (1) "copy-forwarded" diagnoses that lead to artificial token repetition, (2) irregular time intervals between EHR events causing variable timespans within context windows, and (3) the increasing complexity of diseases over time, which complicates the prediction of later tokens compared to earlier ones. These factors highlight challenges associated with EHR data in clinical prediction tasks. The results indicate that a higher prevalence of each property negatively impacts model performance, while models with longer context inputs tend to be more robust (although not consistantly) to these issues.

### Strengths
1. The analysis is solid in its technical execution and experimental design
2. Generally, a rich technical/experimental paper

### Weaknesses
1.	The analysis is solid in its technical execution and experimental design; however, it does not introduce any new methods, models, or techniques, which limits its novelty.
2.	The importance of long context is questionable given the low frequency of extremely long contexts in EHRs. It seems somewhat expected that providing more information for each sample would improve results.
3.	While the paper states that disease progression increases token complexity over time. I am not very convinced how this property can be problematic with shorter input context compared to longer input context models. 
4.	The paper suggests that model performance improves with longer contexts; however, this is not consistently reflected in the figures. If this is indeed the case, it stands to reason that having more information about a patient (i.e., longer context) should facilitate easier predictions.

### Questions
Could you elaborate on the paper's novelty?
Is the dataset utilized in the study publicly available?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the challenges of using language models to model EHR data. By comparing EHR data with natural language, this paper identifies three unique and significant properties of EHR data (copy-forwarding, irregular time intervals, and disease progression) that make modeling EHR sequences more complex than natural language. To provide evidence for these properties, this paper evaluated the performance and robustness of three language models by varying the repetitiveness, irregularity, and context length of EHR data. The models are pre-trained using a private dataset and evaluated with a publicly available dataset, EHRSHOT.

### Strengths
The identified properties of EHR data are convincing. The evaluation of the effects of these properties provides valuable insights into using long-context models to model EHR data. The observations and conclusions in this paper will be helpful for future work to build better foundation models for EHR. 

The authors have released the code and plan to release the model checkpoints later. The release of pre-trained and fine-tuned foundational models will benefit the community, considering the small number of such models currently publicly available.

### Weaknesses
Although the authors test the performance of different language models, the tokenization strategies of these models remain the same as the one used in EHRSHOT. It would be helpful to see if using other tokenization strategies could improve performance. For example, Section 4.3 indicates that irregular inter-token time intervals are harder to model. This conclusion is based on EHRSHOT’s tokenization, which doesn’t encode time intervals. However, there are other tokenization strategies, such as those used by ExBEHRT and EHRMamba, that do encode time intervals.

This paper uses only one EHR dataset to evaluate language models, which somewhat limits its conclusions to the EHRSHOT dataset. While I understand that unifying different EHR datasets is highly intensive work, it would be valuable to see whether similar observations consistently appear in other EHR datasets, such as MIMIC.

### Questions
Some technical details seem to be missing in the paper. For example, it is unclear what the 14 tasks in EHRSHOT are, whether some tasks have highly imbalanced label distributions, and if the sample sizes for these 14 tasks are the same. Since Table 2 reports the mean performance across all 14 EHRSHOT tasks, the lack of such information makes it challenging to assess the actual performance.

When finetuning models, the author wrote "To be consistent with the original EHRSHOT benchmark, we do not finetune our base models – instead, we train a logistic regression head on top of frozen representations generated for each patient by our base models." However, there is no evaluation of CLMBR-T-Base, which is the foundation model released together with EHRSHOT.  Why was CLMBR-T-Base not included in the experiment?

### Soundness
4

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
4

### Summary
The manuscript benchmarks four foundation model architectures on Electronic Health Records (EHRs) to investigate the impact of context length on the model performance of downstream tasks. Moreover, the authors identified and quantified three challenges present in EHR data and showed that long-context models are better at mitigating the challenges.

### Strengths
1. The authors benchmark both transformer-based and subquadratic architectures on EHR data
2. The authors identified and quantified three challenges present in EHR data.
3. The authors conduct experiments to show the effectiveness of long-context models on EHR data.

### Weaknesses
1. The design of token repetition measurement is less convincing. 
	- Are the proposed models applicable to ICU patients? If yes, routine vital signs and regular lab tests can repeat a lot, but they can continuously show patients' health status. It is tricky to determine whether they are informative. 


2. the comprehensiveness of experimental design is limited
	- The investigated methods are limited. They are general architectures for foundation models. However, foundation models designed for EHR, such as [1] and [2], are not included.
	- The authors claimed that irregular time gaps hinder the performance of the models. This is reasonable because the time gap is not encoded during tokenization. It could be interesting to see whether encoding time information would be helpful for some stratified groups although the gain may be minimal overall.

3. The experiment result reported is limited for a comprehensive understanding
	- Prior SOTA mentioned in the manuscript (CLMBR-t-base) is also a transformer-based model. However, the context length of this model is not discussed. Additionally, it is not trained with variable context lengths.
	- Table 2 provides stratified results of the same experiment as Figure 1 (b) and (d). However, it is confusing that CLMBR-t-base and Hyena don't appear in this table. 
	- The author hypothesizes that some degree of copy-forwarding can be helpful by emphasizing important diagnoses. This is observed from the CLMBR-t-base but cannot be validated by other models. Moreover, the Brier score of the CLMBR-t-base seems smaller than other models.
	- Standard deviation is not provided when comparing different models. The authors only conduct statistical tests between short- and long- context counterparts. However, neither standard deviation nor statistical testing is reported when comparing different methods.	
	- (Minor) The impact of the long-context and proposed properties on pretraining is not discussed. The downstream tasks are enough to show the conclusion but it will be better to see if these factors affect training process.

4. There are some typos in the manuscript
	- In line 139, there's a corrupted citation
	- 3.3.3 title: Diseae -> Disease
	- Reference [Yang et al., 2023a] and [Yang et al., 2023b] are the same
References:
[1] Guo, L. L., Steinberg, E., Fleming, S. L., Posada, J., Lemmon, J., Pfohl, S. R., ... & Sung, L. (2023). EHR foundation models improve robustness in the presence of temporal distribution shift. Scientific Reports, 13(1), 3767.
[2] Fallahpour, A., Alinoori, M., Afkanpour, A., & Krishnan, A. (2024). EHRMamba: Towards Generalizable and Scalable Foundation Models for Electronic Health Records. arXiv preprint arXiv:2405.14567.

### Questions
- Statistics of code, event, and token are confusing. 

 
- Diagnosis codes for chronic diseases can frequently appear in a patient’s EHR, often for billing purposes rather than indicating an active health issue. For instance, if a patient with a chronic condition (e.g., COPD or obesity) visits the hospital for an unrelated condition, the chronic disease code may not appear in that visit. In contrast, for acute diseases, the presence of a code in the record typically indicates an active case during that visit. How can token repetition be modeled effectively for these two types of diseases?

		
- In line 261, the authors reported that the vocabulary has 39818 tokens. Is this the vocabulary of EHRSHOT, or does EHR-OMOP also share it? 
		
- Meanwhile, in line 950 of Appendix C, it is reported that 39811 codes are selected. Is "code" here equivalent to "token"? 
		
- In Table 3 in Appendix A, The number of unique codes in EHR-OMOP is much more than the token vocabulary mentioned earlier. Is this the original vocabulary without removing infrequent codes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper analyzes qualities unique to EHR data that are challenging when scaling up sequence lengths. They identify the key properties of copy-forwarding (EHR data is repetitive), irregular time intervals (time intervals vary greatly between tokens), disease complexity (EHR data actually has generally increasing perplexity overtime--unlike text data--as far out future trajectories are hard to predict). They evaluate their models in zero-shot, few-shot, and linear probing regimes, varying context lengths and architectures, and even sharing compute time estimates via token throughputs.

The authors share metrics for quantifying the severity of these three properties on EHR datasets, and demonstrate that severity of these properties correlates with worse performance. They additionally show that longer sequence lengths can help.

Every weakness I pointed out in the initial review has been thoroughly addressed by the authors, this paper is both a clear accept and a major contribution in the EHR machine learning space as it provides a broad benchmark and insights on training autoregressive models in this space.

### Strengths
1. The authors define metrics for evaluating the severity of these properties on any dataset.

2. They demonstrate that as the RR metric and irregularity metric increase, model performance decreases (via patient stratification experiments).

3. The demonstration that perplexity does not generally decrease with sequence length is a major deviation from text data where it is well known to decrease with sequence length. This is because future EHR data is less predictable from previous tokens. The assumption that perplexity reduces with context length for EHR 

4. The authors demonstrate that for all three properties, increasing sequence length helped improve performance (either via improved brier scores or perplexity).

### Weaknesses
The authors provide in Figure 4 plots of perplexity over tokens for different context-length models. The GPT model has wildly varying perplexity over token positions which is described by the authors as being caused by "training instability". I think a more thoughtful analysis of the issue here is required, because it would otherwise look like the cause is a bug.

Why isn't transfer-learning or few-shot included. A major problem in the evaluation is that representations are obtained by averaging the last L token representations from the transformer (This is a one-liner in the appendix and really should be added to the main paper and be clearly communicated in the limitations section). It would be great to see these results in the few-shot setting. I imagine that the performance improvements as you increase sequence length would be even more extreme.

This paper should include comparisons to linear attention models that practitioners are interested in.

This paper does not communicate compute budgets, such as wall-times and the hardware used for these sequence lengths. Could a plot communicating performance vs the compute-load be provided to help justify whether these improvements are worth the added compute time.

Since you trained an autoregressive sequence model, you could perform a zero-shot evaluation where given the past tokens, you autoregressively generate the future tokens and analyze this generated future trajectory for the binary classification task. This paper does not demonstrate whether these results generalize to the zero-shot. I think that analysis is out of scope for this work but should be mentioned in the limitations.

### Questions
I included my questions in the Weakness section, but I'll summarize the actionables below:

1. **The most critical improvement to this paper**: Include a few-shot/transfer-learning results, the use of mean pooling the last L tokens for all evaluations is not a well supported embedding strategy. Alternatively, you have a zero-shot capable model, why not evaluate it in the zeroshot setting as past works do [1,2,3]? At the moment the limited embedding strategy greatly diminishes the generalizability of your results. If this were resolved, I would significantly increase my rating for the paper.
2. Add compute time results, and analysis of the diminishing or increasing predictive performance returns as you increase compute times (via increasing sequence lengths) across methods.
3. A more complete Analysis of the erratic GPT-model Perplexity behavior
4. Add a comparison with linear attention models


[1] Renc, Pawel, et al. "Zero shot health trajectory prediction using transformer." NPJ Digital Medicine 7.1 (2024): 256.

[2] Kraljevic, Zeljko, et al. "Foresight—a generative pretrained transformer for modelling of patient timelines using electronic health records: a retrospective modelling study." The Lancet Digital Health 6.4 (2024): e281-e290.

[3] McDermott, Matthew, et al. "Event Stream GPT: a data pre-processing and modeling library for generative, pre-trained transformers over continuous-time sequences of complex events." Advances in Neural Information Processing Systems 36 (2023): 24322-24334.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores enhancing EHR data processing models by assessing the impact of context length and model architecture on clinical prediction tasks. Four architectures—GPT, Llama, Hyena, and Mamba—were tested with various context lengths, with Mamba (16k token context) achieving state-of-the-art results in 9 of 14 EHRSHOT benchmark tasks. The study identifies three EHR-specific factors influencing model performance: copy-forwarding (repeated diagnoses), irregular time intervals, and disease progression. Findings show longer contexts generally improve performance, with robustness in EHR data handling, though results vary across architectures, as Hyena showed performance drops beyond 4k tokens. This research advances medical data prediction and offers insights on processing long-sequence data, despite limitations like transformer model computational demands and single-institution data.

### Strengths
1. Introduce non-Transformer architectures (such as Mamba) to process medical data.

2. The paper demonstrates the potential of the Mamba model’s long-context capabilities for future clinical applications.

3. The advantage of this paper is that Mamba can perform well with linear complexity.

### Weaknesses
1. Both GPT and LLaMA have a maximum context length of only 4096 tokens, so it's not appropriate for the authors to conduct tests with 16K length as a benchmark.

2.The authors should provide additional details about what the 14 tasks in EHRSHOT include to facilitate better comparison.

3. The paper mentions using the standard deviation of time intervals to divide patients into four groups (Q1-Q4). Regarding the groups with the most regular time intervals and the most irregular time intervals, the standards vary across different diseases, and testing of time intervals should be conducted according to specific diseases.

4. The paper mentions achieving an AUROC of 0.807, but it's confusing that the specific content of the 14 tasks is not listed.

### Questions
Since the author's testing only showed good results with small Mamba models, it's uncertain whether larger Mamba models would perform better than traditional GPT.

### Soundness
3

### Presentation
3

### Contribution
3
