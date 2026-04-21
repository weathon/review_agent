# In Search of Forgotten Domain Generalization

- Avg Score: 7.50
- Decision: Accept (Spotlight)
- Scores: 8, 8, 6, 8

## Abstract
Out-of-Domain (OOD) generalization is the ability of a model trained on one or more domains to generalize to unseen domains. In the ImageNet era of computer vision, evaluation sets for measuring a model's OOD performance were designed to be strictly OOD with respect to style. However, the emergence of foundation models and expansive web-scale datasets has obfuscated this evaluation process, as datasets cover a broad range of domains and risk test domain contamination. In search of the forgotten domain generalization, we create large-scale datasets subsampled from LAION---LAION-Natural and LAION-Rendition---that are strictly OOD to corresponding ImageNet and DomainNet test sets in terms of style. Training CLIP models on these datasets reveals that a significant portion of their performance is explained by in-domain examples. This indicates that the OOD generalization challenges from the ImageNet era still prevail and that training on web-scale data merely creates the illusion of OOD generalization. Furthermore, through a systematic exploration of combining natural and rendition datasets in varying proportions, we identify optimal mixing ratios for model generalization across these domains. Our datasets and results re-enable meaningful assessment of OOD robustness at scale---a crucial prerequisite for improving model robustness.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper focuses on exploring the phenomenon of test domain contamination in the era of foundation models. The paper constructs two different style datasets, LAION-Natural and LAION-Rendition, which are strictly OOD relative to the corresponding ImageNet and DomainNet test sets in terms of style. It examines whether the good OOD performance of CLIP results from test set contamination or from its training paradigm. The findings indicate that CLIP's success stems from domain contamination rather than an intrinsic OOD generalization ability. Additionally, the authors suggest how the mixing ratio of Rendition and Natural samples in the training set can optimize model performance on the test set.

### Strengths
1. This paper gives a very detailed analysis that clarifies that the success of CLIP mainly comes from the training data rather than its own training paradigm, which can give some inspiration to many researchers.

2. The authors give the optimal mixing ratio of rendition and natural image, which allows for better generalization performance of models trained on this dataset.

3. The authors conduct extensive experiments and analyses to support the claim that "the success of the base model in OOD settings does not originate from its own characteristics", which enhances its credibility.

### Weaknesses
1. The main issue with this paper is that it appears to draw a conclusion "he success of CLIP mainly comes from the training data rather than its own training paradigm" that lacks sufficient insight for subsequent researchers. While the question of the optimal ratio of natural and rendition images in the training set is indeed illuminating, the authors do not seem to have conducted tests with models of different architectures. Additionally, I believe that this experiment is important because it can ensure that the optimal ratio is not sensitive to the model's architecture.

2. Although I have experience in OOD research, I am not up to date with the latest techniques and conclusions in this area. Therefore, I will be observing the perspectives of other reviewers as well as the authors' responses. Overall, the paper was thorough and solid, but it seemed to lack novelty.for me.

### Questions
No

### Soundness
3

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
This paper reintroduces the OOD problem in the era of the foundation model. They propose the argument that the phenomenon that foundational models are no longer  susceptible to OOD samples is merely an illusion due to the *domain contamination*, which they define as "whether crucial aspects of a test domain are included in the training domain, e.g., by including images with different content but similar style to test samples"

They then introduce cleaned single-domain datasets to study this problem, discovering that CLIP trained only on natural images significantly underperforms on rendition domain shifts. They further investigate the domain mixing and scaling that affects OOD performance.

### Strengths
1. This paper studies a valuable problem that has seldom been studied. Indeed, the OOD problem, which seems to be less of a problem in the era of LLM, might be an illusion due to the scaling of the massive data. This problem should be taken seriously, as there are still areas where collecting data is extremely difficult and scaling will likely fail.
2. Sound analysis that supports the claim.

### Weaknesses
1. Figure 6 A is a bit confusing, as the result of the best rendition-to-natural ratio 1:3 and 1:1 can not be read in this figure. I suggest adding ratio labels to the color scale or individual data points in the figure.
2. Can you provide some discussion on the "true effectiveness" of the domain classifier trained and evaluated on the curated domain datasets but is used in a different and much larger dataset, i.e. Laion-200M?
    * Will there be OOD shift between the curated domain datasets and Laion-200M?
    * If so, how to make sure the domain classifiers are good?
    * Maybe you can sample a few classified samples in Laion-200M and verify the result by humans.
    * Will adding more domain classifiers and performing an ensemble help?
    * I suggest that the authors include a dedicated subsection in their methodology or results discussing the domain classifier's performance on LAION-200M, including human verification of a sample of classifications and exploration of ensemble methods.

3. Some typos, e.g data points instead of datapoints on lines 76, 82, etc.
4. Overall, I believe this is a technically sound paper with rigorous analysis that supports its main argument.

### Questions
I have a few more questions that would love to discuss with the author. However, these questions should not count towards the evaluation of the paper.
1. Do you think the improved generalization comes from merely more data points or a more sophisticated loss or learning paradigm (i.e. contrastive learning and aligning with language)
2. Despite this paper is mainly on CLIP and contrastive vision language model, the main-stream of foundation models are generative models that are instruction-tuned and aligned. Do you think this work and its conclusion will generalized to LLMs? Specifically, one interesting new thing that was not used in CLIP training is the instruction tuning that enables the model to generalize across tasks. In that sense, models will learn to generalize across tasks instead of data points. For example, MLLMs can be trained to generalize to the task of discriminating between any two domains given by the user. Given such ability, will the MLLMs have better OOD generalization?

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
This work focuses on studying the out-of-distribution generalization capabilities of the CLIP model. To do so, authors consider the scope of *rendition* images and propose to use a domain classifier to detect images from this particular domain from the LAION dataset used to train the CLIP model. By training CLIP with this non-contaminated version of the training data, authors showed that the performance of the model on rendition images decayed, indicating that the original model performance on these images could be explained by the presence of such examples in the training set.

### Strengths
- S1: This work raises awareness of the importance of curating and performing in-depth analysis of large-scale datasets.

- S2: One of the main contributions of this work is to release two curated partitions of the LAION dataset with non-overlapping domains. This will promote and facilitate future research on single-source domain generalization for foundation models such as CLIP.

### Weaknesses
- W1: The motivation for the main question this work aims at answering ("shed light on the limitations of foundation models like CLIP in handling OOD generalization") is not quite clear from the manuscript. In a few words, it seems this work is showing that when CLIP is trained and tested on the same distribution, it performs well, and performance is harmed once a particular domain is removed from the training set and the model is tested on it. What is the exact new insight from this result? Please clarify this in the manuscript.

   - W1.1: In a similar vein, previous work [1] has pointed out that CLIP has limited OOD generalization capabilities and also proposed fine-tuning approaches to mitigate such issues.

- W2: The approach used for data curation relies on collecting human-labeled to train a domain classifier, making it specific to the setting studied in the paper, i.e. rendition vs natural images. 

- W3: In Table 2, the authors report the number of natural/ambiguous/rendition images resulting from the filtering with the domain classifier. However, it is possible to see that in most of the cases a considerably high number of images (> 60% in some cases) is deemed as ambiguous. This seems to indicate that either such domain classifier is not a good model to distinguish natural vs rendition images, or that it could be the case that it does not make sense to consider both domains as two distinct sources of data.

- W4: Lack of overall clarity. Some statements such as "We conclusively demonstrate that domain contamination in the training set is what drives CLIP’s robustness", however, if rendition data is present in the training data, then rendition data is no longer OOD with respect to the training distribution. 


[1] Shu et al, CLIPood: Generalizing CLIP to Out-of-Distributions, ICML, 2023.

### Questions
- Q1: Even after reading the manuscript a few times, the main goal of this work is unclear to me. Previous work on the domain generalization setting has shown that neural networks do struggle to generalize across domains [1], even with powerful regularization and data augmentation approaches to mitigate domain shift. Given that, why exactly would CLIP be expected to generalize well OOD with limited training data? Given that, it seems to me the result/motivation presented in Figure 1-D is exactly what is expected to be given the IID assumption. Can the author kindly elaborate on their motivation? 

- Q2: Authors mentioned in line 62 that "Domains, while not always rigorously defined, emerge from collecting data from specific sources
and conditions." However, there is a lot of effort from the machine learning to rigorously define what a domain is. For example, the authors can refer to the work by Ben David et al. [2]

- Q3: Details about the annotation task are missing, please add the following information to the manuscript: how many annotators were considered, inter-rater agreement, how annotators were chosen and compensated, description of the task set up.

- Q4: The authors mentioned in Line 257: "Since human annotators can make mistakes, and LAION-200M’s domain
composition is inherently unknown, we use our domain classifiers to understand it.". However, I am confused by this statement. The domain classifiers the authors refer to were trained on human-labelled examples, which means they can also inherit the mistakes made by annotators, this approach also seems to be potentially affected by human mistakes (as well as the classifier's mistakes).  

- **Comment that has not affected my score:** The choice of paper title seems odd to me. The title is quite similar to the reference "In search of lost domain generalization", which can be confusing on its own, but the both papers are also considerably different in the questions asked and methodology, making it even more odd that they have such similar names. Moreover, the title "In search of forgotten domain generalization" does not reflect well the content of this submission as it gives the impression that the contributions are broader than they actually are, since this work only focused on the CLIP model and a specific type of domain shift.     


[1] Gulrajani, Ishaan, and David Lopez-Paz. "In search of lost domain generalization." arXiv preprint arXiv:2007.01434 (2020).

[2] Ben-David, Shai, et al. "A theory of learning from different domains." Machine learning 79 (2010): 151-175.

### Soundness
3

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
This paper investigates if the recent purported gains in generalization reported for Large scale vision language models come from true generalization beyond the data distribution, or from data contamination. To this end, they present a methodology for constructing disjoint subsets of the LAION dataset, which serve as OOD counterparts to ImageNet and DomainNet. This ensures no data contamination, and allows investigation of true OOD generalization capabilities of these models. Their main finding is like CLIP like models to not generalize well when such disjoint sets are constructed carefully. This suggests that while these models appear to generalize, they are merely performing well because they have already seen data close to the test set.

### Strengths
Here’s a refined take on the strengths for your review:

1. **Extremely important problem**: This paper addresses a critical issue in machine learning—distinguishing true OOD generalization from data contamination—by revisiting fundamental challenges of OOD generalization in the context of large-scale vision-language models. Their work highlights the limitations of current evaluation practices, underscoring the importance of ensuring genuine robustness and generalization capabilities.

2. **Impactful finding**: The finding that models like CLIP may not truly generalize when faced with rigorously OOD data is significant, as it challenges current assumptions about the robustness of these models. This insight has substantial implications for future work, suggesting that existing performance metrics may overstate the true generalization abilities of these models.

3. **Useful datasets**: By creating the LAION-Natural and LAION-Rendition datasets—both strictly OOD to ImageNet and DomainNet—this work offers a valuable resource for researchers focused on OOD robustness. These carefully constructed subsets provide a way to assess and enhance model generalization on web-scale datasets without contamination.

4. **Useful methodology**: The methodology for creating disjoint, OOD subsets of LAION is a wel established approach in the field, and a solid blueprint for other research seeking to evaluate OOD performance rigorously. 

5. **Results/Figures/Tables are very high quality**: The results, figures, and tables presented in the paper are meticulously crafted, enhancing the clarity and interpretability of the findings. They effectively illustrate the study's key points and provide a solid foundation for future discussions on OOD generalization and robustness evaluation.

### Weaknesses
1. **Missing Literature:** The field of OOD (out of distribution) generalization is very rich, and there are several related papers that are not cited. It would be nice to have this work related to existing works in the field (See below)

### References

a. Liu, J., Shen, Z., He, Y., Zhang, X., Xu, R., Yu, H. and Cui, P., 2021. *Towards out-of-distribution generalization: A survey*. arXiv preprint arXiv:2108.13624.

b. Koh, P.W., Sagawa, S., Marklund, H., Xie, S.M., Zhang, M., Balsubramani, A., Hu, W., Yasunaga, M., Phillips, R.L., Gao, I. and Lee, T., 2021, July. *Wilds: A benchmark of in-the-wild distribution shifts*. In International conference on machine learning (pp. 5637-5664). PMLR.

c. Madan, S., Henry, T., Dozier, J., Ho, H., Bhandari, N., Sasaki, T., Durand, F., Pfister, H. and Boix, X., 2022. *When and how convolutional neural networks generalize to out-of-distribution category–viewpoint combinations*. Nature Machine Intelligence, 4(2), pp.146-153.

d. Gulrajani, I. and Lopez-Paz, D., 2020. *In search of lost domain generalization*. arXiv preprint arXiv:2007.01434.

e. Madan, S., You, L., Zhang, M., Pfister, H. and Kreiman, G., 2022. *What makes domain generalization hard?* arXiv preprint arXiv:2206.07802.

f. Arjovsky, M., Bottou, L., Gulrajani, I. and Lopez-Paz, D., 2019. *Invariant risk minimization*. arXiv preprint arXiv:1907.02893.

g. Arjovsky, M., 2020. *Out of distribution generalization in machine learning* (Doctoral dissertation, New York University).

2. **Smaller training dataset size:** One of the key claims in papers presenting vision language models is the idea of Scaling Laws---behavior emerges from large scale datasets which cannot be seen in smaller models. Here, CLIP was trained on 19,000 images which is too small and thus it remains unclear if such generalization errors will continue in extreme dataset sizes. If nothing else, it is worth discussing this point in the paper.

### Questions
1. Is there a way of automating the dataset slip so as to train the model on a large scale dataset?

2. Would it make sense to generate "un-natural" images i.e. data that does not relate to real world categories to explore this further. For instance, maybe texture images generated from explicitly disjoint data distributions?

### Soundness
4

### Presentation
4

### Contribution
4
