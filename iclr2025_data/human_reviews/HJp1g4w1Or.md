## Human Reviewer 1

### Summary
Authors present the first deep MEG dataset alignment/adaptation method, albeit a common goal outside of the deep learning methods and MEG modality. For instance, [1] apply MEG dataset alignment based on canonical correlation analysis and [2] apply deep domain adaptation on fMRI datasets.

They follow a procedure similar to the adversarial discriminative domain adaptation (ADDA) method, that is inspired by the H-Divergence theorem. This procedure, referred as adversarial harmonization, is adopted from a previous study by Dinsdale et. al.. The goal is to infer the domain-specific signal and penalize it to pass a domain-general signal across-datasets. 

They utilize two different base architectures in the experiments, from MEGalodon and BrainMagick studies, where a major contribution claim is the re-implementation of BrainMagick base architecture.  MEGalodon base model experiments are on MOUS and Cam-CAN datasets, whereas BrainMagick experiments are on MEG-MASC and Gwilliams datasets. BrainMagick decoding task is the estimation of speech stimulus segment in a given interval. MEGalodon decoding tasks are determining speech occured in a given interval and classification of the phoneme into voiced of voiceless in the MEG data interval. Additionally, there are pre-text tasks in the MEGalodon; namely band, shift and amplitude scale prediction tasks.

Network has three components: encoder, task head, domain classifier. The Components are trained together in a warmup phase, followed by the adversarial parts, the domain classifier optimization to identify dataset bias, and the removal of the dataset bias by confusing the domain classifier. 

Adversarial harmonization results are close to the official implementation results of each external study. They emphasize the role of age related variation to be the most significant source of divergence in the data distributions. Authors discuss unstability issues regarding the adversarial harmonization.

[1] Q. Zhang, J. P. Borst, R. E. Kass, and J. R. Anderson, “Inter-subject alignment of MEG datasets in a common representational space,” Human Brain Mapping, vol. 38, no. 9, pp. 4287–4301, 2017, doi: 10.1002/hbm.23689.

[2] J. Wolleb et al., “Learn to Ignore: Domain Adaptation for Multi-Site MRI Analysis,” Jun. 07, 2022, arXiv: arXiv:2110.06803. Accessed: Nov. 03, 2024. [Online]. Available: http://arxiv.org/abs/2110.06803

### Strengths
Authors suggest the first adversarial domain adaptation method on MEG modality. The study re-implements the previous work, BrainMagick, base model architecture. Implementing on new modalities requires a tedious work, hence the open source re-implementation of the base model is a strength of the study.

### Weaknesses
1- Adversarial harmonization and experiment sections can benefit from a more clear organization. The narrative writing style is easy-to-read in the methods section, however many important parts are in the appendix sections, most probably due to the page limit. A more compact re-writing that includes important information like training details, model figures and training curves, would improve the flow of the paper. 

2- Subject level differences are not addressed in the harmonization method. On the similar functional neuroimaging modalities, like EEG (sensor-space) and fMRI (brain-space), similar adversarial domain adaptation methods do not stand out as the best option, as the variation across samples depend on subject-specific differences and dataset/acquisition-site differences, as well as on the demographics. Hence, applying subject level harmonization along with dataset and demographic level harmonization might improve the results. See the questions section for a similar study/method.

3- In both table 2 and table 3, the performance improvement do not support the positive impact of adversarial harmonization in a clear way. 
In table 2, control results for both Gwilliams and MOUS datasets are in the mean±std range of Harmonized results.
In table 3, for speech detection, "Harmonized" methods perform worse, and the "Warm-up Only" introduces 0.5% accuracy improvement, whereas for Voicing, "Harmonized" method introduces 0.05% improvement. This suggests the increase in performance might be depending on the number of epochs, a parameter that is not inferred from data and may have a different effect for different datasets/subjects. Hence, the stability of the adversarial harmonization method is not convincing.

### Questions
A question regarding a simple baseline: While deep learning has a great impact in many fields, domain adaptation/generalization methods are quite tricky [1]. There are also lightweight methods that focus on inter-subject alignment of MEG data, for instance the work of Zhang et. al. [2] that implements a hyperalignment method. Would it be possible to add a similar method as a baseline? This can benefit the completeness of the results, such that an estimation of a simple spatial transformation, i.e. Procrustean Transform, applied in sensor space can show the relative impact of the Harmonization method. 

[1] I. Gulrajani and D. Lopez-Paz, “In Search of Lost Domain Generalization,” presented at the International Conference on Learning Representations, Oct. 2020. Accessed: Nov. 05, 2024. [Online]. Available: https://openreview.net/forum?id=lQdXeXDoWtI

[2] Q. Zhang, J. P. Borst, R. E. Kass, and J. R. Anderson, “Inter-subject alignment of MEG datasets in a common representational space,” Human Brain Mapping, vol. 38, no. 9, pp. 4287–4301, 2017, doi: 10.1002/hbm.23689.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper proposes a domain adaptation (DA) method that leverages adversarial training techniques to enhance generalization across different neural recording datasets. The study evaluates the effectiveness of the proposed framework on two model architectures for MEG (magnetoencephalography) classification tasks related to speech. Results demonstrate that the DA framework can effectively harmonize feature representations across different datasets, particularly between those with distinct demographic characteristics. Additionally, classification accuracy is improved for certain tasks.

### Strengths
1. DA is an important topic, especially for neuroscience-related tasks, due to the high variability and limited data volume in this field. This paper could have a notable impact as one of the efforts to apply DA to MEG data.
2. The authors provide an implementation of an existing model, promoting openness and reproducibility in the community.
3. Being the first to successfully apply a similar DA framework in computer vision to MEG field and enable a more general DA for MEG that can be applied on top of different architecture.

### Weaknesses
The related work section lacks an adequate discussion on DA methods for EEG. Numerous studies have applied DA to EEG, and given that EEG and MEG share many properties, methods developed for EEG should, in principle, be applicable to MEG and could serve as benchmarks. Including a comparison with a benchmark DA method would enhance the quality of this work significantly.

### Questions
1. Could you clarify the number of samples N used in the statistical tests for Table 2?
2. The contribution of the methods section needs further clarification. What are the key contributions and differences between your proposed framework and existing adversarial harmonization procedures?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper investigates the use of non-invasive EEG imaging techniques, specifically magnetoencephalography (MEG), to decode speech representations in the brain. Aiming at the problem of poor generalization ability of non-invasive brain imaging data across multiple datasets, the authors adopt an adversarial domain adaptation framework to improve the model generalization between different datasets. In this study, the authors leverage two different speech decoding models and reconcile the differences between datasets through a feature-level adversarial learning approach, which "obfuscates" the domain classifier through an enhanced network architecture to reduce the impact of dataset bias.

### Strengths
1. This study is the first to implement feature-based adversarial domain adaptation on MEG data, which provides a new methodology for cross-dataset application of non-invasive EEG imaging techniques.
2. By improving the generalization ability of non-invasive neuroimaging technology, it is helpful to improve the accuracy and reliability of speech decoding, which is of positive significance for practical applications such as rehabilitation of speech disorders.

### Weaknesses
1. The paper does not describe its own method in detail, especially when most of the model methods are built on the basis of others, which will weaken the innovation of the paper.
2. The experimental analysis of the paper is seriously insufficient, and some statistical charts take up too much space, resulting in less practical useful length of the paper.
3. The overall writing of the paper is more like an experimental analysis report than a complete paper.

### Questions
1. The authors show that after 100 epochs, the distribution tends to the average distribution of all age groups, however, it is not obvious in Figure 4, the author should provide the trend of the distribution with epochs to prove this.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 4

### Summary
The authors attempted to demonstrate that the use of standard feature based adversarial domain shift adaptation machinery allows for  an improved generalization across time-resolved functional MEG based brain imaging datasets. They used two pairs  of MEG datasets reported in previous studies and recorded from multiple participants listening to audio stories. The authors augmented the two *existing*  DL models Brainmagick and MEGalodon with an *existing* adversarial domain shift adaptation strategy (Dinsdale et al., 2021) and showed that harmonizing the datasets based on the participant's age slightly improves the classification performance in the downstream task in the first (Gwilliams-MOUS) pair of datasets where the goal was to predict the index of a 3-s long audio segment.  The authors then put forward the claim that harmonizing with respect to participant's age is an important step in dealing with multi-subject MEG datasets.

### Strengths
1. The authors (although they are not the first ones)  address the important problem of aligning the brain imaging data from multiple subjects \ clinics\ devices. 
2. The authors use state-of-the-art multi-subject  MEG datatasets
3. The authors successfully implemented  the existing MEG decoding solutions and reproduced the results from the original papers.

### Weaknesses
1. **Utility.** First of all, I am concerned with the overall utility of this study. The use of brain imaging methods to decode parameters of the external auditory stimuli does not bring us any closer to the development of speech prosthesis device where the DL-based decoding of brain's electrical activity appears really useful. See for example  https://www.biorxiv.org/content/10.1101/2024.08.21.608927v1 where decoding of the imagined (covert) speech is done at the chance level by the architecture capable to successfully recover both the perceived and the overt speech. The study uses invasive macroscale recordings but the situation with non-invasive imaging methods is even worse. The only real speech-BCI solutions existing to date (see the studies by Stavitsky's lab @ UC Davis) are based on the brain activity data recorded  with  intracortical probes (Utah arrays) that sample activity of individual neurons or their very small populations.  These data are really uniquely informative and can be used to decode the attempted speech (https://www.medrxiv.org/content/10.1101/2023.12.26.23300110v1).
2. **Approach justification.** At the same time drastically simpler models (comprising only tens of parameter) perform well on the decoding of the external stimulus parameters from brain activity (SPOC, Nikulin et al) just because brain's reaction is quite strong to such an external stimulation. Combined with simple domain shift adaptation techniques such as https://arxiv.org/abs/2407.03878 or earlier approaches, say, https://ieeexplore.ieee.org/document/8624413 these simpler techniques need to be brought into the scene of comparison.  If the authors want to "untrain" w,r.t to age feature they could have pooled the dataset withing age strata and perform the described  stable and well define domain adaptation methods. At the very least the proposed approach has to be leveraged against these more stable techniques, 
3. **Clarity.** The paper is quite poorly written. It has a lot of unnecessary and unrelated to the main idea  comments and It was not easy to understand what the authors actually decided to do. Please, also see the questions. From what I managed to deduce the work is rather incremental and does not represent a significant advance neither in the magnitude of the achieved improvements, nor methodologically.
4.  **Data management\sanity.** If deduced correctly, the authors used 15% of the datasets for their experiments which  raises concerns regarding the reliability of the reported results and the observed minimal improvements. While I am sympathetic with the described shortage of computational resources these arguments are hardly acceptable  as a justification especially given that the MEG datasets used are already not so large. Please, also see the questions.
5. **Impact.** Please, see the questions below, but the overall results are rather inconclusive and hardly support the claims made in the paper regarding the effect of datasets harmonization based on the age parameter as this statement can be made (and with a slight stretch) only based on the Gwilliams-MOUS dataset analysis results pair and not on those for the second pair. Leave along the lack of confidence intervals in Table 3.  Also the age is not the strongest source of the between-subject MEG variability yet it can be predicted in the 25~85 y.o. range with more or less uniformly distributed accuracy from the non-invasive recordings of brain activity  https://www.sciencedirect.com/science/article/pii/S105381192200636X. 
6. **A suggestion for increasing the impact and reliability** Age prediction from younger people is typically more accurate which may be an interesting avenue for the authors to explore in the future and compare the effect of harmonization for  the young vs. middle-age participants.

### Questions
1. Could the authors be more explicit as to what is exactly done to a pair of datasets when dataset A is used for the pre-train and dataset B is used as a new domain dataset. Was the actual downstream  task training (combined with harmonization) performed in this dataset B ? A processing flow diagram would be very helpful to clarify this!
2.  For Gwiliams dataset and using the model from (Defossez, 2023) the authors reported only top-10 accuracy. In the original paper both Top-1 and top-10 are reported. What about the Top-1?  
3. How many runs were used to estimate the confidence intervals on the accuracy metrics reported in Tables 2?  Why are these missing in table 3. Control accuracy of 52.60 and the highlighted value of 52.65 seem to be not very different.  
4, What do the authors mean under "We do not find that the base Brainmagick architecture is effective in cross-dataset generalization."  - why does the subject specific layer appeared ineffective in translating between the data recorded with KIT and CTF systems? What are the  MEG sensor types used by these two systems?    
5. Have the authors tried to prune the explored architectures to combat the computational load? From reading the original papers I have gotten an impression that both Brainmagick and MEGalodon architectures are significantly over-parameterized especially given the amount of training data available. At the same time the parameters (e.g. loudness) of the audio stimulus could be reliably decoded with several tens of coefficients (SPOC, Nikulin et al.) also the electrical activity can be predicted from the audio envelope and vice-versa with significantly above chance accuracy using a single convolutions kernel (https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2016.00604/full) even from the EEG data.  The two latter approaches use total of 50 to 300 weights to solve their tasks.

### Soundness
1

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
4