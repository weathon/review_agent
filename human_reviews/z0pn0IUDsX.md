# From Categories to Classifier: Name-Only Continual Learning by Exploring the Web

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 6, 5, 6

## Abstract
Continual Learning (CL) often relies on the availability of extensive annotated datasets, an assumption that is unrealistically time-consuming and costly in practice. We explore a novel paradigm termed name-only continual learning where time and cost constraints prohibit manual annotation. In this scenario, learners adapt to new category shifts using only category names without the luxury of annotated training data. Our proposed solution leverages the expansive and ever-evolving internet to query and download uncurated webly-supervised data for image classification. We investigate the reliability of our web data and find them comparable, and in some cases superior, to manually annotated datasets. Additionally, we show that by harnessing the web, we can create support sets that surpass state-of-the-art name-only classification that create support sets using generative models or image retrieval from LAION-5B, achieving up to 25% boost in accuracy. When applied across varied continual learning contexts, our method consistently exhibits a small performance gap in comparison to models trained on manually annotated datasets. We present EvoTrends, a class-incremental dataset made from the web to capture real-world trends, created in just minutes. Overall, this paper underscores the potential of using uncurated webly-supervised data to mitigate the challenges associated with manual data labeling in continual learning.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper explores continual learning, where new categories can be introduced over time, in a setting where we have the names of the introduced categories but no curated labeled data for those categories. Instead the labels come from the web, ie a very large uncurated set of data that can be queried by category name, but where the precision and recall of the queries are not necessarily guaranteed. They show that their approach is only slightly less accurate than approaches trained on much smaller curated datasets. They also contribute a class-incremental dataset, which they call EvoTrends, which they were able to quickly curate from the web. The research contribution of the work is not very clear or strong, the main contributions are the engineering-heavy and expensive application of web supervised data to continual learning and an experimental analysis. There are significant ethical concerns with the use of web-scraped data for training that are not addressed by the authors.

### Strengths
The claims in the paper are clear and they show many experimental results and ablate different dimensions of the problem. The discussion of the performance gap and web-supervised limitations is a nice addition. This paper was clearly a significant implementation and engineering effort and is the first to my knowledge to run this scale of experimental analysis of web supervision for continual learning when provided with only the category names for each step.

### Weaknesses
The research contribution of the paper is not very clear compared to prior work showing that weakly labeled data from the web can be useful. The only difference here is that the exact same approach (using web queries to collect training data) is being done in a continual learning scenario. Comparison between models trained on very large webly-supervised data (which in many cases may contain the exact images and labels in the curated datasets, for example CUB was directly curated from flickr) as compared to smaller manually labeled datasets is not exactly a fair comparison, as the experts that provided the information are still very much a component of the first instance, just in this case they are given less credit for how that human expertise is a necessary component of the system to get to training data that can produce a useful signal. The increase in number of training images should be included for every experiment in Table 1 to provide context, and the overlap between the webly-supervised training and the curated training should be explicitly analyzed.

### Questions
It is not surprising that larger datasets with reasonably good but imperfect labels do ok, this has been well-demonstrated in the literature. What novel insights has this work given us into the use of very large weakly-labeled web data for classification? What about the continual learning setting the authors explore (where they query for classes, add the large weakly-labeled data to training, and train vanilla classification models or do nearest-neighbor classification per-timestep) is fundamentally different from just simple classifier training for a set of defined classes based on web-supervision without the continual setting?

Expert-in-the-loop curation of these larger datasets may only add a small amount of latency and significantly improve downstream performance, this is not discussed in the paper (see https://arxiv.org/abs/2302.12948)

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a novel paradigm termed name-only continual learning to overcome continuous learning's reliance on time-consuming and expensive large-scale annotated datasets. Specifically, this paper proposes to leverage the expansive and ever-evolving internet to query and download uncurated webly-supervised data for image classification. It shows the potential of using uncurated webly-supervised data to mitigate the challenges associated with manual data labeling in continual learning. The experimental results show that it investigates the reliability of the web data and finds them comparable, and in some cases superior, to manually annotated datasets. Besides, it consistently exhibits a small performance gap in comparison to models trained on manually annotated datasets when applied across varied continual learning contexts.

### Strengths
1. The paper is well-written and has a clear structure.
2. This paper provides a new method to boost the performance of image classification. It shows that using uncurated webly-supervised data can significantly reduce the time and expense associated with manual annotation in the proposed name-only continual learning setting.

### Weaknesses
1. This paper is not innovative enough. It mainly explores a viable method of utilizing the practical applications of web data, and the innovative issues involved are insufficient.
2. It is not convincing that the paper attributes the performance improvement to the dataset sizes of the uncurated, noisy, and out-of-distribution web data. The paper needs to provide more experimental evidence about the impact of these three issues of web data, especially the large number of noisy samples that will definitely lead to a decrease in model performance.

### Questions
see weaknesses

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes name-only continual learning, where category names are used to query and download uncurated webly-supervised data for continual learning. The main finding is that models trained on uncurated webly-supervised data can benefit from the large data scale, and the performance can equal or even surpass the that of models trained on manually annotated datasets. Performance is also shown to be better than that of state-of-the-art name-only classification approaches. In different continual learning scenarios where only class names are available, the observations remain similar. Finally, a new continual learning dataset EvoTrends is introduced to include trending products year-by-year from 2000 to 2020.

### Strengths
- This is generally a solid paper that presents the first name-only continual learning approach using uncurated webly-supervised data. I like the idea a lot, the method proves to be working well, and the analysis/comparisons are comprehensive.
- The continual learning dataset EvoTrends is a good plus, offering a more natural class-incremental scenario to train and evaluate continual learners.

### Weaknesses
- One main concern about this paper is the missing detail/analysis of the continual learning algorithm itself (see questions below).
- Another concern is that many analysis results under the non-continual name-only classification setting (Section 4) won't necessarily translate to the continual learning setting. For example, Section 4 provides nice ablations on the model architecture and pretraining algorithm (Table 8), ensembling from multiple web engines, data cleaning and class balancing strategies - will they make a larger impact on continual learning? Also, under continual learning setting, it's unclear if the performance will remain competitive with those name-only classification techniques.

### Questions
- My understanding about the continual learning algorithm used in this paper is fixed feature extractor + retraining classifier every timestep t. But how is the retraining actually implemented? At the end of Section 3, it mentions "the classifier is trained not only on the uncurated data downloaded from the current timestep t but also on the uncurated data downloaded from all prior timesteps" which would be pretty costly. In Section 5.1, it mentions "we implement experience replay and utilize class-balanced sampling to select training batches from the previously collected samples". Does this mean the authors just use the traditional continual learning method based on experience replay (just with some smart sampling)? No more whistles and bells?
- In either case, how well is the adopted algorithm with web data addressing the catastrophic forgetting in the class-incremental setting? What about the domain-/time-incremental settings where there can be overlapping classes with domain shifts? How is the adopted algorithm handling those cases?
- Last question about Table 1: finetuning the backbone 1) achieves much better end performance than not, for both MA data and C2C. 2) The benefits of C2C over MA data are reduced a lot (except for FGVC). Is that also the case for continual learning? If true, that makes C2C less interesting when finding the best end performance is the goal.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This study introduces "name-only continual learning," a new approach in Continual Learning (CL) that overcomes the limitations of time and cost in creating extensive annotated datasets by utilizing uncurated webly-supervised data for image classification. The proposed method not only achieves comparable, sometimes even superior, results to manually annotated data but also delivers a significant performance boost in accuracy, along with the development of EvoTrends.

### Strengths
+ A valuable exploration of using uncurated webly-supervised data for fine-grained classification and incremental learning.

+ A well-written and structured paper with insights on the usage of uncurated webly data on visual learning tasks, and its advantage over other CLIP-based methods.

### Weaknesses
- The experiments fail to reflect popular incremental setup, including restricted access to the data from previous tasks.

- The technical sections are rather weak, and limited details about CL approaches were exposed. It's unclear if the application on CL is reasonable.

- The advantage of the "uncurated webly-supervised" may be limited to a few fine-grained classifications where only small numbers of training data are available.

### Questions
1.	One of the major concerns is the setup for CL experiments. CL is a well-established field, but very few competitive methods are reported in the paper. Could authors add a few more recent CL methods for comparison?
2.	CL usually has strong restrictions on access to the “old” data from previous tasks. Nonetheless, this work uses data from all tasks to “re-train” each time. Did I miss any details?
3.	The comparison with other “name-only” and CLIP methods was not well-explained. The advantage of the proposed method, as claimed by the authors, is the “scale.” In fine-grained classification, a sufficient number of samples/class are critical. I'm not sure if any off-the-shelf CLIP model is supposed to do this well without any special treatment. 
4.	Regarding regular CL tasks on regular visual concepts or objects, the accuracy is only comparable to, or worse than, MA data. It means not only the quantity but also the quality of the images matter in these tasks. While the paper attempts to highlight name-only CL, it seems this approach did not work well in CL. BTW, CL usually limits the number of samples per class, too.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper explores the name-only continual learning, which combines the webly-supervised learning with continual learning using a pre-trained model. In this paper, it is found that training with webly collected data can be comparable or even superior to training with manually annotated datasets. Besides, a new class-incremental dataset to capture real-world trends is introduced.

### Strengths
1. The new setting is meaningful for reducing the cost of manual annotation of new datasets.
2. This paper presents outstanding results with webly-supervised learning under single-task and continual learning. 
2. This paper extends the webly-supervised learning to continual setting, which is reasonable for real-world application. A new benchmark with ever-changing real-world data is introduced for webly-supervised / name-only continual learning.
3. Although somewhat counterintuitive, it is interesting to see that a pre-trained model can learn so well with noisy, uncurated data. This paper finds data scale to be the key. 
4. The evaluations in this work are thorough. Multiple fine-grained vision datasets are included. Cost-aware evaluation for CL is used, and the results are significant.

### Weaknesses
1. In the last paragraph of Section 3, it is said that for the continual name-only setting, "Once we complete downloading the data, the classifier is trained not only on the uncurated data downloaded from the current timestep $t$ but also on the uncurated data downloaded from **all prior timesteps**." If data from all timesteps are used, it would not strictly conform to the definition of continual learning. Section 5 also mentions a class-balanced sampling to select training batches from the previously collected samples. Accessing previous samples would violate the definition of CL. If just a few data are used, to what scale is the method selected?
2. In section 3, this paper mentions *we implemented some cost-effective safeguards*. However, no details are found in this paper (only one following sentence that refers to the safe-search feature of search engines).
3. (Minor) The results on other network architectures (for example, ViT) are worse then ResNet, especially for the Cars dataset. 
4. (Minor) In section 5.2, when explaining the performance gap, it is said that "The current training operates within a tight computational budget". Here, the word "tight" is ambiguous since the results it mentions are under the "normal" instead of the "tight" budget suggested in Table 5.

### Questions
See questions in the Weaknesses part.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
