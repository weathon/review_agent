# Revealing and Reducing Gender Biases in Vision and Language Assistants (VLAs)

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 8

## Abstract
Pre-trained large language models (LLMs) have been reliably integrated with visual input for multimodal tasks. The widespread adoption of instruction-tuned image-to-text vision-language assistants (VLAs) like LLaVA and InternVL necessitates evaluating gender biases. We study gender bias in 22 popular open-source VLAs with respect to personality traits, skills, and occupations. Our results show that VLAs replicate human biases likely present in the data, such as real-world occupational imbalances. Similarly, they tend to attribute more skills and positive personality traits to women than to men, and we see a consistent tendency to associate negative personality traits with men. To eliminate the gender bias in these models, we find that fine-tuning-based debiasing methods achieve the best trade-off between debiasing and retaining performance on downstream tasks. We argue for pre-deploying gender bias assessment in VLAs and motivate further development of debiasing strategies to ensure equitable societal outcomes. Code is available at https://github.com/ExplainableML/vla-gender-bias.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper benchmarks the performance in 22 open-source VLAs such as LLaVa, BakLLaVa, and InternVL alongside their variants on four datasets - FairFace, MIAP, Phase and PATA to identify the ingrained gender bias based on occupational, personality and work-related skills in such vision-language models. The experimental setup is Visual Question Answering(VQA) which sends as inputs an image which does not contain any specific information which would allow the model to deduce the person's occupation with the text input being an objective question requiring the model to solely deduce the occupation, personality or skill of a person based on their gender. The work concludes that a majority of these models assign more negative traits such as "stubborn", "moody" and "arrogant" to males as compared to positive traits being assigned to females. Moreover, positive skills are majorly associated with males replicating the imbalances in occupations in the real world. In this regard, the authors aim to mitigate this inherent bias through classical techniques such as - full finetuning, Low-Rank Adapter finetuning, prompt engineering and alpha-pruning. Amongst these techniques, they identify the best mitigation in full finetuning and LoRA finetuning with LoRA preserving the performance of the original model while full finetuning reducing bias to the minimum. Additionally, they observe that prompt engineering and alpha-pruning(over multiple alpha values) do not push the model towards reducing bias.

### Strengths
The main strengths of this paper:
1) The authors address a longstanding problem of extending the investigation of gender bias mitigation from LLMs towards VLMs. The research question has been defined structurally with the specific gender bias they investigate, i.e. its significance arising from personality, occupational and work-related traits. 
2) The benchmark comprising the 22 VLAs and their performance assessments on four different datasets - MIAP, FairFace, PATA and Phase seemed particularly extensive and an elaborate amount of experimentation to me. 
3) The appropriate conclusion drawing from the performance of the VLAs as well as the metric to measure the gap between the mean of answers directing to males and the mean of answers directing to females, seemed interesting. The use of prior evaluation studies to demonstrate how VLAs reflect the imbalance in current societies over gender is quite intuitive.

Other positive points include the good structure of the paper as well as relevant datasets used for the evaluation study.

### Weaknesses
A few weaknesses which seemed to be standing out to me:
1) While reading through the paper and some related work, I came across another paper "Gender Bias in Vision-Language Assistants" performing a similar study using the datasets used in this paper. I would suggest citing this work as well since the work seems to be aligned with the current study as well as demonstrating similar outcomes to the current paper. The authors could discuss how this work improves upon or contrasts with that work.
2) Some additional insights into the exact process of finetuning seemed to be missing from the paper such as the training set, the ground truths used for evaluation etc.
3) Another method used for bias mitigation is causal mediation analysis(Vig et al. 2020), which has been previously used in the cases of Language Models. This method can be extended to VLAs primarily to get a more concrete insight into the gender bias ingrained in these models. The authors could maybe shed some light on how that method can be used for bias mitigation.
4) In section 4.3, Qualitative results, where it is mentioned that "the distribution over the three answers becomes somewhat uniform" -  Since the answers should be skewed towards "unsure", this would demonstrate the models become "more careful" in answering such questions with confidence. Additionally, it seems to me from Figure 4, that none of the methods alleviates the model's tendency to respond with "unsure", only increasing the distribution value for "no" which could also be arising from hallucination. As mentioned by the authors, "unsure" would be an answer to be picked to avoid gender-sensitive questions, I feel debiasing should be targeted towards increasing the probability of answering "unsure" which would indicate that it needs more occupational or personality-related description to converge on a final answer related to the image.

### Questions
Some questions which I would like answered:

1) In section 2.3, prompt groups "We manually identify 21 suitable skill descriptions..." Is there any concrete backing to this identification process drawn from any other study?

2) Some examples can be drawn which demonstrate the different answers generated by the models to the same question asked.

3) In section 4.3, Qualitative results, where it is mentioned that the distribution over the three answers becomes somewhat uniform, can some insight be provided into how it exactly fixes the bias present in the models? Also drawing from my fourth point in weaknesses, I would like some elaboration on the implications of "no" increasing as a response to the questions asked.

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
4

### Summary
The paper investigates gender bias in vision-language assistants (VLAs) by analyzing 22 widely used open-source models, focusing on associations with personality traits, skills, and occupations. It presents comprehensive experimental results, demonstrating the presence of biases across multiple models and offering a detailed comparison of bias patterns. The findings reveal that VLAs frequently replicate human biases, associating positive traits more often with women and negative traits with men, while also reflecting gender imbalances in occupational roles observed in real-world data. The study evaluates several debiasing techniques, concluding that fine-tuning provides the best balance between reducing bias and preserving model performance.

### Strengths
1. The paper presents a unique approach to analyzing gender bias in VLAs by comprehensively examining bias across 22 open-source models.

2. The study is well-executed, featuring a detailed experimental design and thorough experimental results.

3. This work holds significant value for the field of fairness in VLMs by highlighting the presence of gender bias in widely used VLAs and offering effective strategies for mitigating such biases.

### Weaknesses
1. In the datasets used by the authors, there are no true labels, meaning that all evaluations rely on open-ended questions. Therefore, responses of "yes" or "no" are not necessarily correct answers, and choosing "unsure" might actually be a fairer response. It would be helpful if the authors discussed how they accounted for this limitation in their analysis, such as the analyze the frequency of "unsure" responses across different demographic groups.

2. In this evaluation process, while comparing the number of "yes" responses across genders can indeed be used to evaluate bias, the open-ended nature of the questions might allow other visual factors in the images, such as attire or accessories, to influence the results. For example, a model may be more likely to associate positive personality traits with someone wearing glasses and neatly dressed, regardless of gender, which could weaken the apparent impact of gender bias. To more accurately assess the effect of gender bias, would a more focused evaluation method be possible, or could additional image datasets with true labels be supplemented to evaluate from both perspectives?

3. In the authors' mitigation experiments, the various methods tested seem to create a more balanced distribution of "yes" and "no" responses for the test images (as shown in Figure 4). However, it is possible that neither "yes" nor "no" is always the most accurate answer; in cases where the image lacks sufficient information, "unsure" might be the most appropriate response. Could the authors provide a justification for focusing on balancing "yes" and "no" responses rather than encouraging "unsure" responses? Additionally, analyzing the distribution of "unsure" responses before and after debiasing might offer further insight.

### Questions
1. In Section 2.1, it would be helpful to present the statistics of the evaluation set of 5,000 images, including the distribution of various labels. This additional information would clarify the dataset composition and support a more detailed understanding of the evaluation setup.

2. In Section 2.2, the authors attempt to remove biased images using InternVL2-40B and GPT-4V; however, the entire removal process relies exclusively on these pre-existing VLMs without any supervised labels or manual verification. It may be beneficial for the authors to consider additional validation methods for their image selection process, such as conducting a small-scale human evaluation on a subset of the images to verify the effectiveness of the removal. 

3. In Figure 1, some images still appear to contain occupation-related information, which could introduce inaccuracies in the evaluation results and create ambiguity about whether observed unfairness is genuinely due to gender bias. Do these examples come from your cleaned dataset?

4. For prompt tuning mitigation method, what initial hard prompt was used for fine-tuning?

5. How is the significant bias score calculated, based on sample mean? What is the sample size?

6. In Figure 3, the legend currently uses shades to represent different methods, but clarity could be improved by using consistent colors for each method throughout the figure. Alternatively, if distinct colors are used, labeling each method directly on the figure may help with readability. Additionally, could the authors clarify why significant bias appears to reach zero for some models after optimization? For example, in the personality traits evaluation, LoRA tuning for llava_1.6_mistral_7b seems less effective, while full fine-tuning achieves optimal results. What factors might contribute to such a large discrepancy in optimization effects for the same model?

7. In the dataset posed by the authors, it is theoretically difficult to infer relevant information purely from images, so, in my understanding, the "unsure" option would be a more correct answer than "yes" or "no." However, in the optimization loss function, it appears that the "unsure" option is not considered.

8. Some relevant papers discussing gender bias in large vision-language models (LVLMs) appear to be missing from the Related Works section. For example:
    * Zhang, Jie, et al. "VLBiasBench: A Comprehensive Benchmark for Evaluating Bias in Large Vision-Language Models." arXiv preprint arXiv:2406.14194 (2024). This paper introduces a synthetic benchmark for systematically evaluating bias across various LVLMs (including gender bias evaluation), could provide additional context to your findings.
    * Wu, Xuyang, et al. "Evaluating Fairness in Large Vision-Language Models Across Diverse Demographic Attributes and Prompts." arXiv preprint arXiv:2406.17974 (2024). This study examines fairness across multiple demographic attributes, including gender bias in occupation prediction, using labeled data from the FACET dataset. It would be helpful if you could discuss the differences between test strategies based on labeled datasets versus open-ended question scenarios, as each approach may impact performance and insights into gender bias.


Typo:
* Finetuning => fine-tuning
* Tradeoff => trade-off
* Line 91, gender per se ?

### Soundness
3

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
This paper investigates gender bias in 22 open-source instruction-tuned VLMs (a.k.a. VLAs) across three different categories: personality traits, skills, and occupations. The authors curate a dataset for these analyses, and show that VLAs follow real-world gender imbalances for some categories (e.g. occupations) but less for other categories (e.g. personality traits). The authors then investigate 5 techniques to reduce gender bias in 5 models, and find full fine-tuning to return the better trade-off between overall performance and gender bias.

### Strengths
1. The paper combines and extend different datasets for gender (and ethnicity) bias.
2. Several (22) open-source VLAs are evaluated, showcasing overall similar patterns.
3. Five different techniques are tested to reduce gender bias, with results on the trade-offs of each of them w.r.t. gender bias and overall performance.

### Weaknesses
1. One main weakness of the proposed approach — as pointed out by the authors — is the inability to test API-based VLAs
2. Another question I had while reading the paper was how this study differentiates from previous ones. This was only partially answered towards the end of the paper (Related Work Section), but it would be very helpful to have a more systematic comparison with previous work and how this study clearly differs from it (e.g. by creating a table summarizing key differences), as the results echoes findings that had previously been published.
3. Among them, a comparison and discussion that is missing is around the work of Cabello et al. (EMNLP 2023), which shows that fine-tuning VLMs on gender-neutral data led to similar or better performance whilst reducing bias. Given that you find fine-tuning to be the better method, such comparison (or at least a discussion) would be very relevant to include (e.g. can the two approaches be combined?) 

---
Cabello et al. Evaluating Bias and Fairness in Gender-Neutral Pretrained Vision-and-Language Models. EMNLP 2023.

### Questions
1. Have you tried assessing your 22 models using approaches that could be applied to API-only models? It would at least be informative to know how your conclusions would change in that case.
2. In L313-314, you claim that “positive traits, however, are not attributed to one single gender.” On the other hand, Fig 2 (top) shows that — besides “humble” — all the other traits are mostly associated with female gender. Can you provide a more detailed explanation of how you interpret these results to support the current statement?
3. In Section 4.3, you split the data into training and test. How do you perform hyper-parameter tuning without a validation set?

### Soundness
4

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
This paper presents a method to estimate gender bias (wrt personality traits, skills and occupation) in vision-language assistants. The method prompts VLAs to say whether a person (cropped to just show the face) has a personality trait/skill/occupation, and looks at how the P(yes) varies between male and female groups. The authors evaluate how five debiasing methods help in mitigating gender bias in VLAs.

### Strengths
- **Sound methodology:** The methodology for constructing the dataset seems very thorough and well thought-out: sampling images from multiple datasets and balancing gender+ethnicity (although it would be useful to see which ethnicities were represented and the distribution), thorough prompt variation with accounting for prompt wording and choice ordering.

- **Thorough methodology and analysis:** The evaluation on 22 different models seems very thorough. Good citations for where the stereotypes corresponding to the occupation/skill biases come from. Something similar can be done for the personality traits by looking at valence scores (e.g. from the VAD Lexicon).

### Weaknesses
- **Insufficient literature review:** Missing citations from “gender bias in vision-language” literature. All but one paper in their review is from 2024, with no reference to previous works and findings. I would point to the following papers:
    1.  Survey of Social Bias in Vision-Language Models: https://arxiv.org/abs/2309.14381
    2. Worst of Both Worlds: Biases Compound in Pre-trained Vision-and-Language Models: https://arxiv.org/abs/2104.08666
    3. Counterfactually Measuring and Eliminating Social Bias in Vision-Language Pre-training Models: https://dl.acm.org/doi/abs/10.1145/3503161.3548396

### Questions
- L353: I dont think biasedness is a word, it should be “bias”
- An interesting analysis would be to see how consistent the biases of different models in the same model series are.
- Why do you refer to the vision-language models as “assistants” throughout the paper? “Vision-language models” is a well-established term for these models. Assistants seems to indicate that there is a larger system that the model is a part of (which is not the case, as far as I can tell).

### Soundness
4

### Presentation
4

### Contribution
3
