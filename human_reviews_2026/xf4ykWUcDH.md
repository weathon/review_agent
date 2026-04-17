# EEG-ImageNet: A Benchmark for Pre-training and Cross-Time Generalization of EEG-based Visual Decoding

- Decision: Reject
- Scores: 0, 4, 2, 8

## Abstract
Exploring brain activity in relation to visual perception provides insights into the biological representation of the world. 
While functional magnetic resonance imaging (fMRI) and magnetoencephalography (MEG) have enabled effective image classification and reconstruction, their high cost and bulk limit practical use. 
Electroencephalography (EEG), by contrast, offers low cost and excellent temporal resolution, but its potential has been limited by the scarcity of large, high-quality datasets and by block-design experiments that introduce temporal confounds.
To fill this gap, we present CrossPT-EEG, a benchmark for cross-participant and cross-time generalization of visual decoding from EEG. 
We collected EEG data from 16 participants while they viewed 4,000 images sampled from ImageNet, with image stimuli annotated at multiple levels of granularity. 
Our design includes two stages separated in time to allow cross-time generalization and avoid block-design artifacts.
We also introduce benchmarks tailored to non-block design classification, as well as pre-training experiments to assess cross-time and cross-participant generalization. 
These findings highlight the dataset's potential to enhance EEG-based visual brain-computer interfaces, deepen our understanding of visual perception in biological systems, and suggest promising applications for improving machine vision models.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper introduces a new dataset for the problem of decoding image stimulus class from EEG recordings. It is similar in nature to the Perceive dataset introduced in Spampinato et al. (CVPR 2017). There are 80 classes (instead of 40 in Perceive). The first 40 are coarse grain, just as in Perceive. The second 40 are fine grain, with 5 superordinate classes of 8 subordinate classes each. Just as in Perceive, there are 50 stimuli per class. While Perceive had 6 subjects, here there are 16. Just like Perceive, stimuli were presented for 500ms. Unlike Perceive, which recorded with an 128 electrode recorder, EEG was recorded from an unspecified 62-electrode recorder. Just like in Perceive, stimuli were presented in blocks, where all stimuli in the block were of the same class and all stimuli of that class were in the same block. Unlike Perceive, each block started with a presentation of the class label, presumably as visual text, though unspecified in the paper. Unlike Percevie, each block ended with some sort of test to measure attention, though the nature of this is unspecified in the manuscript, whcih says it was optional. Unlike Perceive, there were two recording session for 6 of the 16 subjects. In the second session, each block of 50 stimuli started with 30 stimuli from one class and ended with 20 stimuli of a different class. Results of classifying this dataset with various models are presented. The central claim is that the two-session design avoids the block confound discussed in Li et al. (2021).

### Strengths
None.
The dataset suffers from a known published confound that correlates stimulus class with drift in the EEG signal, essentially an embedded clock. Thus, decoders can and do classify the clock, not stimulus class, as demonstrated by Li et al. (2021) and follow on papers in TPAMI and CVPR by the same authors. As well as Xu et al. (2026) The impacts of temporal autocorrelations on EEG decoding, Biomedical Signal Processing and Control, 113.

### Weaknesses
1. Li et al. (2021) subject 6 shows that the block confound even occurs when the training and test sets come from different blocks from different sessions. Thus the 2-session design does not remove the confound. Thus, this dataset still suffers from the block confound and the results are thus not to be trusted. Li et al. (2021) show that performance drops from near perfect to near chance when the confound is removed with randomized trials. There is no excuse that can justify using a block design instead of randomized trials.
 2. Numerous details discussed above are missing, like what the exclusion criterion based on the attention test was. Excluding some trials from some subjects breaks the counterbalanced design. This introduces bias into the classification task and thus chance is not 1/k for k classes. It is not clear this was taken into account in computing statistical significance since the process for computing p values was not discusses. It is not clear whether correction for multiple comparisons was performed.
 3. The stimulus presentation order was not specified. Exactly what was and was not randomized was not specified. Without this, it is impossible to assess the claim that the design does not exhibit correlation between stimulus class and a clock embedded into the signal.
 4. The four tasks, WT, CT, CP, and PT are not described in sufficient detail to understand precisely what was done.
 5. While you claim that this dataset is large, the published datasets associated with Li et al. (2021) and Ahmed et al. (2021)

https://ieee-dataport.org/open-access/dataset-perils-and-pitfalls-block-design-eeg-classification-experiments
https://ieee-dataport.org/open-access/dataset-object-classification-randomized-eeg-trials

are just as large, if not larger, do not suffer from the block confound, and cover video stimuli as well as image stimuli, but are not mentioned or cited.
 6. Many people refer to the Perceive dataset from Spampinato et al. (2017) ad EEG-ImageNet. Other datasets have been collected and published under the name EEG-ImageNet. Reusing the name is confusing.

### Questions
1. Why did you not simply conduct randomized trials? That is the standard method universally adopted in all of experimental science to avoid confounds.
2.  Why did you not try your methods on the two datasets mentioned above as these were collected with randomized trials and thus do not suffer from the block confound?

### Soundness
1

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
5

### Summary
This paper introduces an EEG dataset collected from 16 participants using ImageNet-based visual stimuli across two experimental stages. Although the dataset offers potential contributions to brain decoding research, several methodological and reporting concerns must be resolved before the work is suitable for publication.

### Strengths
See questions

### Weaknesses
See questions

### Questions
1. The authors acknowledge the temporal confounds introduced by block-design paradigms, yet paradoxically still adopt such a design in both Stage 1 and Stage 2. Although Stage 2 is claimed to be “non-block,” the structure of the experiment still presents images from the same category in short temporal clusters, failing to achieve full randomization. This undermines the core motivation of the dataset, which is to address block-related artifacts. Moreover, if the intention was to retain Stage 1 in order to investigate temporal effects introduced by block design, the paper does not provide any systematic analysis of such effects. Instead, the two stages adopt markedly different protocols (e.g., temporal spacing), thereby introducing an uncontrolled source of variability that further complicates interpretation and comparison across tasks.
2. The procedural description of the experiment is vague and incomplete. Critical details such as how many blocks each participant completed, how long each block lasted, and whether the data was collected in a single continuous session or across multiple sessions remain unspecified. The paper also lacks a clear account of the train-test split strategy: it is unclear whether there is a fixed test set or if different splits are used for each evaluation. This ambiguity is compounded in Figure 2, where it is not indicated whether the training and testing sets in WT, CT, CP, and PT originate from Stage 1, Stage 2, or both. As a result, the methodological transparency is insufficient for reproducibility or proper evaluation of the reported benchmarks.
3. The comparability of the reported results across different tasks is undermined by imbalanced subject participation. Specifically, ten of the sixteen participants did not take part in Stage 2, which directly affects the validity of comparisons across WT, CT, CP, and PT tasks. Because these tasks are evaluated on different subsets of participants, the results are not strictly comparable, yet the paper presents them side by side in Tables 2 and 3 without accounting for this discrepancy. This undermines claims about the relative difficulty or generalizability of each task.
4. The dataset's scale is a notable improvement over previous EEG-visual studies, but it still falls short of the scale required for training large neural models or for making strong claims about cross-subject generalization. With only six participants contributing Stage 2 data and a total of 16 in Stage 1, the participant pool is very limited both in size and age diversity. The authors should moderate their claims of generalization, as the dataset’s limited scale compared to what the name EEG-ImageNet implies suggests that data from a broader and more diverse population is needed to support such claims.
5. The term “cross-time” is used throughout the manuscript, but Stage 2 is conducted at least seven days after Stage 1, making “cross-day” a more precise and appropriate description. 
6. The paper evaluates several classic and relatively simple deep models but does not benchmark against recent and widely adopted architectures in EEG decoding. Incorporating mainstream models would substantially strengthen the benchmark’s value, improve reproducibility, and enable more meaningful comparisons with the current state of the art.
7. The manuscript does not include any visualizations of the learned EEG features or model activations, missing an opportunity to underscore the neuroscientific relevance of the signals being classified. Without visual inspection of the EEG responses across categories or tasks, it remains unclear whether the models are learning meaningful neural correlates of perception or simply exploiting low-level artifacts such as EMG or eye movement signals. This data validation is particularly important for a dataset intended to bridge neuroscience and machine learning.
8. While the dataset is described as having coarse- and fine-grained categories, the manuscript provides no visual or conceptual overview of the category structure. Including representative images, a hierarchical map of categories, or example trials would help the reader better understand the nature of the visual stimuli and the complexity of the decoding task. 
9. The feature extraction procedure involves a fixed window after stimulus onset, but the analysis does not discuss or control for individual variability in visual latency (e.g., typical VEP components like P1 or N170). Without evaluating how shifts in latency affect decoding performance, there remains a risk of overlap between responses to adjacent images, especially given the 500 ms stimulus duration in the visual presentation paradigm. This could lead to response contamination across trials, confounding model performance.
10. Figure 6 presents image generation results, yet the method used for image reconstruction is not described in sufficient detail. It remains unclear how EEG features were mapped into image space, what generative model was used, how the outputs were evaluated, and what the overall goal of this analysis was.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents EEG-Imagenet, which is a dataset for observing image stimuli and the corresponding signals of electroencephalogram (eeg), containing 16 subjects and 4,000 images from ImageNet. In addition to the collection process of the dataset, the article also defines the evaluation methods and tasks, and demonstrates the performance of common methods in these tasks.

### Strengths
The motivation of the article is very good. Currently, EEG-image datasets are lacking, and the field requires datasets that include more subjects and more samples. The writing of the article is also good, easy to understand, and the several task divisions proposed (WT, PT, CT and CP) are very comprehensive and appropriate.

### Weaknesses
The biggest problem with the article is that the way the dataset is collected itself might be problematic. The article will continuously display the images of the same categories (50 images) to each subject, which will cause the subjects to maintain relatively stable brain activity during this period. This leads to the fact that there is actually no distinction between the images, and even brain activity simply reflects a category. This issue has been discussed in ''The Perils and Pitfalls of Block Design for EEG Classification Experiments'', IEEE TPAMI, 2020 It is generally believed within the field that this approach is incorrect.

This issue was also reflected in the subsequent experimental results, such as the WT results being very high. One possible validation is to conduct a search within images of the same category. I believe the model may not be able to retrieve effectively because continuous brain activities within the same category are likely to converge. In this case, the conclusion of the article may be affected, such as the accuracy rate of some task divisions mentioned later. 

At the same time, the article does not compare this dataset with other datasets, such as when training classification models simultaneously, how the performance of the models trained on other datasets (such as THINGS) differs from that on this dataset. This is also a verification of whether the collection of the dataset is effective.

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work presents a valuable and timely resource for EEG and visual neuroscience research. The dataset is carefully designed to mitigate known issues such as block-design confounds and limited category diversity, and it includes both coarse- and fine-grained visual labels. It is thoroughly benchmarked across several classical and deep learning models, providing a solid empirical baseline for future studies. The methodology and documentation are transparent and reproducible, with clear ethical considerations and open data plans. Overall, EEG-ImageNet represents a meaningful and nice contribution that could have significant impact in advancing EEG-based decoding and facilitating cross-disciplinary work between neuroscience and machine learning.

### Strengths
This work presents a valuable and timely resource for EEG and visual neuroscience research. The dataset is carefully designed to mitigate known issues such as block-design confounds and limited category diversity, and it includes both coarse- and fine-grained visual labels. It is thoroughly benchmarked across several classical and deep learning models, providing a solid empirical baseline for future studies. The methodology and documentation are transparent and reproducible, with clear ethical considerations and open data plans. Overall, EEG-ImageNet represents a meaningful and nice contribution that could have significant impact in advancing EEG-based decoding and facilitating cross-disciplinary work between neuroscience and machine learning.

### Weaknesses
I think it would be valuable to include more extensive comparison with other dataset. It could be interest to compare the effect of having models pre-trained on other EEG datasets (e.g., SEED, DEAP, Things-EEG) and vice-versa to assess transferability and confirm that EEG-ImageNet enables broader generalization.

### Questions
1. Could the authors compare EEG-ImageNet with models pre-trained on other EEG datasets (e.g., Things-EEG, SEED, DEAP) to quantify transfer learning performance? This would help assess whether EEG-ImageNet generalizes beyond its own benchmark.
2. How consistent are the EEG signal qualities across sessions and participants? Were any metrics (e.g., SNR, channel dropout rates) tracked to ensure data reliability?
3. Can you give more details on the pretraining (PT) setting, it was a bit unclear to me what was done.

### Soundness
3

### Presentation
3

### Contribution
4
