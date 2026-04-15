# Gauging Learnability in Supervised Fine-tuning Data

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 1, 3, 3

## Abstract
Supervised Fine-Tuning (SFT) serves as a crucial phase in aligning Large Language Models (LLMs) to specific task prerequisites. The selection of fine-tuning data profoundly influences the model’s performance, a choice traditionally grounded in data quality and distribution. However, this paper introduces an innovative dimension in data selection: learnability. SFT is regarded as a technique for unlocking the potential of pretrained models. However, given that different models have disparate capabilities, the data appropriate for one may not suit another. Thus, we introduce the term ``learnability" to define the suitability of data for effective learning by the model. We present the Loss Based SFT Data Selection (LoBaSS)  method, utilizing data learnability as the principal criterion for the selection of secure, efficient, and high-quality data. This method provides a nuanced approach, allowing the alignment of data selection with inherent model capabilities, ensuring optimal compatibility and learning efficiency. In experimental comparisons involving 7B and 13B models, our LoBaSS method surpasses the full-data fine-tuning, requiring merely 6% of the data. When employing 16.7% of the data, LoBaSS harmonizes the model’s capabilities across conversational and mathematical domains, proving its efficacy and adaptability.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose a method for measuring the importance of data used to perform supervised fine-tuning (SFT) of large language models. This measure is used to select the most important data examples in order to select a subset of SFT data that is most beneficial for the fine-tuning process. They introduce a measure of data importance they call "learnability" which is computed with respect to a specific model and expresses three design choices (i.e. constraints) the authors underline, which can be summarized as (1) assign a low score to uninformative data; (2) assign a low score to hard-to-learn data; (3) assign a high score to efficiently learnable data. The authors compare their prioritized selection method against training on all SFT data and against asking Chat-GPT to filter out uninformative data examples. Experimental evaluation demonstrates that their method generally outperforms these two baselines.

### Strengths
**S1**: Supervised fine-tuning is becoming a crucial task when applying LLMs to specific scenarios and doing it in a data-efficient way is important.

**S2**: The proposed method is quite simple which makes it easy to apply in practice.

**S3**: The experiments demonstrate the effectiveness of the proposed method and show that it is able to remove some "hurtful" data which ends up giving a better-performing model that was fine-tuned only on a subset of the SFT data.

### Weaknesses
**W1**: The problem of selecting optimal subsets of training data is far from novel. It is known as the coreset selection problem and many methods have been proposed. It would be good to add some references to that line of work.

**W2**: The proposed design constraints are argued for using simple reasoning techniques (section 3.3). However, one could also see them as arbitrarily chosen. For example, the statement "When a piece of data is incomprehensible or overly challenging for the model, *introducing such data during fine-tuning is detrimental*." is strange because one might ask where the evidence is that it is detrimental. It would be useful if each of the constraints were empirically shown to be necessary (i.e. individually, not together) using some ablation analysis.

**W3**: The notion of "learnability" is referred to across the paper as a "dimension", "measure" and "perspective". To me, it seems like the best fit would be to call it "measure" and make this uniform across the paper.

**W4**: I would rephrase the x-axis label in Figures 3 and 4 from "Scale" into something more clear (e.g. amount of data selected).

**W5**: There are several strange statements in the related work section: (1) "the distribution of the data should ideally be *uniform* and aligned with the requirements of the intended usage scenarios" -- how can a data distribution be uniform?; (2) "ChatGPT to assess data quality, which carries the risk of data leakage and considers only the inherent quality of the data" -- what is the meaning of the term inherent quality of data?

### Questions
**Q1**: Doesn't the introduction of normalization as described in Section 3.4 contradict constraint 2? Namely, constraint 2 implies that we should be able to detect data that is excessively demanding by observing large $L_{ini}$ and $L_{ref}$ values. On the other hand, in section 3.4 it is implied that this data does not "meet our expectations". (also, it is not clear, which expectations?)

**Q2**: What is the difference between the backbone and the baseline model?

**Q3**: What is data mixing (as mentioned in section 3.4)?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a way to select fine-tuning data for downstream LLMs using the normalized difference between the pre-trained model and a fine-tuned model, which this paper calls the "reference" model.

### Strengths
The paper is easy to understand. Details are explained when necessary.

### Weaknesses
* Typo: in page 2, CharGPT --> ChatGPT
* It is unclear what the novelty of the proposed method is. Moreover, the model is compared to random selection and ChatGPT, with little evidence to support that these are state-of-the-art baselines. Details on the ChatGPT-based filtering are scarce. For example, what prompt is used?
* The paper shows results as "wins", "ties", or "losses", without showing a table of actual scores on the test sets.
* The experimental design for the Alpaca-4 experiments (Section 4.2) is flawed. For instance, the normalized method being better in the Alpaca-3.5 experiments does not indicate that it will also be better in the Alpaca-4 experiment.
* The proposed method requires a new "reference" model to be trained via fine-tuning, which can be prohibitively expensive on top of a downstream LLM.

### Questions
None.

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigates the problem of selecting fine-tuning samples for instruction tuning of pre-trained LLMs. Different from prior work that often targets data quality and distribution, this paper introduces a new aspect for selecting data, dubbed "learnability".

Specifically, a reference model is first obtained by fine-tuning a pre-trained LLM ("initial model") on full instruction tuning samples. Then, 3 constraints are imposed:
1. samples lacking information (samples with a small loss on both reference LLM and initial LLM) are removed;
2. hard samples (samples with a high loss on both LLMs) are removed;
3. "learnable samples" (samples with a high loss on initial LLM but lower loss on reference LLM) are selected.

A set of experiments is conducted on LLaMA-7B/13B and compared with ChatGPT-based filtering methods. The paper claims to achieve better performance after fine-tuning on 3k selected samples compared to fine-tuning on full 52k samples.

### Strengths
The perspective of this work is interesting. This angle of sample "learnability" is attractive and feels promising. The problem being investigated is timely.

### Weaknesses
- The structure of this paper is rather loose. The writing style is problematic. The paper contains too many non-specific descriptions for the methods of this work ("we focus on the new aspect of learnability") or its results ("we achieve better performance with 6% of data"). Its actual technical contribution or specific methodology is not at all introduced until Section 3.3 on Page 4. The abstract says nothing about what this work actually does and the introduction also remains on the descriptive level. This is not the style for a research paper. Subjective descriptions should be used with discretion and only when necessary. The major technical body of this work should be put to the front in the most straightforward manner. The language throughout the paper needs to stay objective and rigorous. 

- The methodology described in Section 3.3 is intuitive but somewhat superficial. I do not mind empirical papers based on insights and intuitions. This is nowadays a major drive for the progression of this field. Yet the description in Section 3.3 is too simple for me to feel comfortable. These criteria for samples being "too hard", "too simple", and "in-between and good" are overly subjective. At least, no analysis is provided to ground it to existing frameworks. I guess this threshold is also set in an ad-hoc manner and needs to be tuned with trial and error in each case. It could provide much higher value if the authors could develop it into a principled framework

- The references in this work lack depth. It focuses overwhelmingly on the work during the past year and does not connect to lines of existing research with a richer history (e.g., learnability of samples, data selection problems, simple or hard samples, etc.).

- The term "supervised fine-tuning" used throughout this work actually refers to instruction-tuning. Supervised fine-tuning has a much broader reference than the case studied in this work, especially when not confined to text-completion LLMs.

- Recently, there is already a wealth of work on this topic of "instruction mining" and I believe many have reported results similar to this work–achieving comparable or better performance with a small fraction of 52k Alpaca instruction samples, which is believed to contain low-quality samples that would hurt the performance. The performance reported in this work isn't particularly stronger than the provided baseline and not many baselines are considered.


- Reproducibility: It is unknown how to set the threshold in the proposed constraints. It seems to be manually picked without a principled method or analytical insights.

- Format: Appendix is not cut from the main paper. The PDF provided for the main paper is this 14-page document.

### Questions
Appendix should not be submitted under the main paper.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors present a novel metric for example selection for
supervised fine tuning, inspired by a learnability principle.

### Strengths
Selecting fine tuning samples based on learnability principles seems a sensible idea.

The paper is easy to follow.

### Weaknesses
The novelty of the proposed learnability metric is unclear to me. The
authors propose three different criteria that are just different
aspects of the same criterion, namely relative loss reduction, which
is the normalized formula they eventually derived (albeit they
apparently did not recognize it as such).

The experimental evaluation is not entirely convincing:

About the comparison with ChatGPT selection: why not using the same
number of data points? e.g. 9,229 points also for your approach? the
comparison is not on equal grounds otherwise.

No comparison in made with alternative data selection procedures, even
if a number of them are listed in the related work section. I don't
think these can be dismissed without a comparison if the authors are
to claim that their approach is a general solution to SFT.

The robustness of the approach and the generality of the results
should be better assessed. For instance, Figure 4b indicates an
oscillatory behaviour that can be detrimental to the method. I believe
multiple datasets should be tested to present robust results.

### Questions
why not using the same number of data points in the comparison with ChatGPT select?

How did you enroll participants for human evaluation? how many did you have? 

Also please comment on the concerns I raised in the weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
