# SimSCOOD: Systematic Analysis of Out-of-Distribution Generalization in Fine-tuned Source Code Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 5

## Abstract
Large code datasets have become increasingly accessible for pre-training source code models. However, for the fine-tuning phase, obtaining representative training data that fully covers the code distribution for specific downstream tasks remains challenging due to the task-specific nature and limited labeling resources. Moreover, fine-tuning pretrained models can result in forgetting previously acquired pre-training knowledge. These lead to out-of-distribution (OOD) generalization issues with unexpected model inference behaviors that have not been systematically studied yet.
In this paper, we contribute the first systematic approach that simulates various OOD scenarios along different dimensions of source code data properties and study the fine-tuned model behaviors in such scenarios. We  investigate the behaviors of models under different fine-tuning methodologies, including full fine-tuning and Low-Rank Adaptation (LoRA) fine-tuning methods. Our comprehensive analysis, conducted on four state-of-the-art pretrained models and applied to two code generation tasks, exposes multiple failure modes attributed to OOD generalization issues. Additionally, our analysis uncovers that LoRA fine-tuning consistently exhibits significantly better OOD generalization performance than full fine-tuning across various scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper conducts a systematic investigation of the behaviors of code models under different fine-tuning techniques. They create diverse OOD scenarios by masking out some portion of distributions. Then the model is either fine-tuned fully or partially (via LoRa). They reveal some important insights into the OOD generalization ability of LoRa and FT.

### Strengths
1. The OOD scenarios are comprehensive. The authors propose three kinds of OOD simulations, including length-based, syntax-based, and semantics-based methods. These scenarios are sufficient to cover real-life OOD data.

2. The behavior analyses of the behavior models are thorough and systematic. The authors compare full-finetuning and LoRa under different test benchmarks. Sufficient insights and conclusions are given based on the analysis of the results.

### Weaknesses
1. The main weakness to me is that all the conclusions and insights are somehow predictable. It is somehow natural that LoRa generalizes better than FT as the fine-tuned weights are fewer and OOD data are more scarce than the pre-training data. Other insights are also similar: we can get them by logically reasoning about the approach. The only useful takeaway to me is the specific performance gain of introducing extra data. 

2. As discussed above, the specific performance gain brought by extra data is very interesting to me. Could you conduct an ablation study to show the performance gain versus different amounts of extra data? This would give more valuable clues to the community as we would know using how much data for fine-tuning would be the best tradeoff.

3. Can you conduct an ablation study of increasing the amount of fine-tuning data and the fine-tuning time? This might decrease the advantage of LoRa against FT.

Finally, as a minor comment, I noticed that the authors consistently misuse \citet and \citep throughout the whole paper. I suggest the authors proofread the paper a few more times to improve the readability.

### Questions
Please see the weaknesses.

### Soundness
3 good

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper analysis the out-of-distribution performance and forgetting phenomenon of source code models.
They mainly test several code models with full-finetuning and LoRA methods and give some conclusions based on the experiments results.

### Strengths
1. The writing and presentation are fluent.
2. The experiment results are significant when comparing the two methods.

### Weaknesses
1. The paper mainly tests several existing methods for the forgetting task of code datasets.
The novelty of this work is not clear from the conclusion and experiments now.

2. As an experimental paper, it lacks enough comparison of methods that prevent forgetting, e.g., Wise-FT[1].

3. For analyzing the forgetting phenomenon, there lacks the analysis on different hyperparameters' choice of finetuning 
or the size of the dataset.

[1].Wortsman, Mitchell, et al. "Robust fine-tuning of zero-shot models. 2022 IEEE." CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2021.

### Questions
1. It will be better to add more experiments mentioned above to formalize an extensive analysis.
2. The novelty of the work should be clearer.

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper investigates the behavior of code Large Language Models (LLMs) fine-tuned in a source code domain when faced with out-of-distribution (OOD) code scenarios in the testing phase. The authors introduce a systematic approach to probe this issue. They begin by creating three types of OOD scenarios with modifications along various dimensions. Subsequently, they evaluate four leading code LLMs using both full fine-tuning and Low-Rank Adaptation (LORA) fine-tuning for two code-related tasks within these scenarios. The key findings highlight the challenges associated with OOD generalization in current code LLMs, the potential benefits of even a small amount of relevant labeled data, and the superiority of LORA fine-tuning in preserving the generalization capabilities of LLMs over full fine-tuning.

### Strengths
1) The problem studied is highly meaningful and significant. This paper pioneers the investigation of OOD generalization issues in code LLMs.

2) The experiments are extensive, as evidenced by the creation of diverse OOD code scenarios and the evaluation of various state-of-the-art code LLMs.

3) The paper is well-organized and easy to follow, with the key takeaways providing a concise summary of the empirical study's results.

### Weaknesses
1) The findings from the code data do not appear to be surprising and align with similar observations in prior studies involving OOD scenarios with image data. As the authors themselves acknowledge, previous works, such as Kumar et al. (2022), have already shown that full fine-tuning tends to exhibit weaker OOD generalization performance compared to fine-tuning with partial parameters. Additionally, the positive impact of a small amount of labeled data on generalization is in line with expectations, given the motivations within the realm of few-shot learning.

2) Do the three types of code OOD scenarios accurately represent realistic code OOD data? If future research builds upon this empirical study to enhance OOD performance in these three scenarios, can it be assured that the methods will also improve OOD performance in various other OOD scenarios?

3) The experimental details have been mentioned several times in various paragraphs. I would kindly recommend refining the presentation to make it more concise and informative.

### Questions
Kindly refer to Weaknesses for all questions.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper investigates the out-of-distribution (OOD) generalization issue of fine-tuning pre-trained source code models. It contributes a systematic approach that simulates various OOD scenarios along different dimensions of source code data properties, including the length, syntax, and semantics. It then investigates the behaviors of models under different fine-tuning methodologies, including full fine-tuning and Low-Rank Adaptation (LoRA). The analysis is conducted on four state-of-the-art pre-trained models and applied to two code generation tasks, which exposes multiple failure modes attributed to OOD generalization issues and shows that LoRA fine-tuning consistently exhibits better OOD generalization performance than full fine-tuning.

### Strengths
* This paper contributes a systematic approach to simulate various OOD scenarios along different dimensions of source code data properties, including the properties of the length, syntax, and semantics. 
 
* This paper studies the fine-tuned model behaviors in OOD scenarios on four pre-trained models, two tasks, and two fine-tuning methods and gives some takeaway conclusions, which may encourage future studies on this topic.

### Weaknesses
* The finding that fine-tuning may distort the pre-trained features, cause catastrophic forgetting, and harm the OOD generalization performance has already been studied in various NLP and CV tasks, as mentioned in RELATED WORK of this paper. This paper only changes to a new task of code generation, performs similar analysis, and gets similar results compared with existing works, which does not give much new insights for the community.

* There are some issues with the approach to simulate OOD scenarios in this paper. (1) The OOD scenarios made by masking out sub-regions of data distributions may not be realistic, especially in the cases of different lengths and certain language elements. (2) This paper also mentions the works of OOD analysis in programming languages in RELATED WORKS, and their relationships with this paper should be discussed in more detail. Why not use their pre-defined scenarios for evaluation? Would the scenarios in those works be more realistic?

* There are many methods proposed in CV and NLP to mitigate catastrophic forgetting in fine-tuning, but only the LoRA method is evaluated, which makes the experimental results not comprehensive enough.

----------------------------------------------
**Post Rebuttal:**

I thank the authors for providing the responses, which address some of the concerns and weaknesses in the review. The remaining concern is that the OOD generalization analysis in code generation, although claimed to be first studied in this paper, does not give some new insights compared with existing OOD generalization works. It would be more interesting and insightful if the analysis could find new results and conclusions specific to code generation.

### Questions
In Section 4.4, what is the performance of the original pre-trained model without FT or LoRA to generate unseen language elements?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
