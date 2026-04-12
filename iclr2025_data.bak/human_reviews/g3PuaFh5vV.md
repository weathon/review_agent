## Human Reviewer 1

### Summary
Authors emphasize the importance of utilizing the 3-dimensional location of sensors and design a method to map sensor-space data to source-space. They state that decoding from MEG/EEG source space has been done but these studies are impractical for real-time decoding or do not use deep learning. They preprocess the data with empirical frequency parameters and determine source reconstruction parameters, that includes subject-specific anatomical scans and a common brain template for multi-subject decoding. They utilize Armeni dataset for single subject experiments and Schoffelen dataset for multi-subject experiments. They compare the source and sensor representations with a 3-Layer MLP architecture, and find close accuracy score in single-subject and multi-subject experiments. They adopt a 3D Convolutional Neural Network (CNN) architecture, where they represent the irregular voxel shape in source-space with the minimal cubic volume, and a graph attention network (GAT), both with a dropout mechanism. Multi-subject experiment shows that the CNN model outperforms MLP and GAT in source and sensor space, and the performance is close to MLP in sensor space. They apply spatial data augmentations; mixup, slice and cube masking, that they claim that there are no spatial augmentations in source space for MEG/EEG data. They experiment on masking out all voxels in a brain region and measure the change in performance, where a consistent trend is not observed. In the zero-shot interdataset generalization experiment, they evaluate between each subject of Armeni dataset, the model trained on multiple subjects of the Schoffelen dataset, and vice versa.  Combining datasets improves single subject results.

### Strengths
The promising study introduces a large body of experimental effort on a relatively less explored field, MEG data. The code and pipelines in this work can contribute to increasing interest in the field. Source space reconstruction of MEG data is a solid goal, albeit a well-studied one.  Enabled by the common source template, combining multi-subject datasets improves single subject performance.

### Weaknesses
1- The claims, that being the first study to apply CNN architecture or spatial augmentations in the MEG field might be too strong, as some other previous studies [1, 2] worked on very similar goals. 

2- Better separating MEG field technicalities and machine learning related details can expand the audience of the work. The article might benefit from a more compact writing style to include figures and definition tables that can guide the reader.

3- As the MEG data is source reconstructed, source space model saliency can be extracted to determine if there are recurring spatial patterns, which is missing in the work. 

4- Intuition behind Slice dropout is not well grounded in the article, which can also be defined as a special case of cube masking. 

[1] Z. Huang and T. Yu, “Cross-Subject MEG Decoding Using 3D Convolutional Neural Networks,” in 2019 WRC Symposium on Advanced Robotics and Automation (WRC SARA), Aug. 2019, pp. 354–359. doi: 10.1109/WRC-SARA.2019.8931958.

[2] A. Giovannetti et al., “Deep-MEG: spatiotemporal CNN features and multiband ensemble classification for predicting the early signs of Alzheimer’s disease with magnetoencephalography,” Neural Comput & Applic, vol. 33, no. 21, pp. 14651–14667, Nov. 2021, doi: 10.1007/s00521-021-06105-4.

### Questions
The study introduces new experiments in the MEG based deep representation learning; zero-shot transfer learning, expanding datasets with a common template. However, a comparison with related studies is missing. Would it be possible to carry out an experiment to compare with related studies in the MEG field?

"Balanced accuracy" concept needs a more detailed explanation.

Time resolution is especially important for the relatively newly explored MEG data in deep learning applications, compared to the spatial resolution of fMRI data. A discussion and improvement of the model architecture towards emphasizing time dimension would be a significant improvement in this study.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
3

---

## Human Reviewer 2

### Summary
By employing established techniques to reconstruct neural activity from MEG sources into voxel representations, the study demonstrates advantages such as spatial inductive biases, spatial data augmentations, improved interpretability, zero-shot generalization across datasets, and enhanced data harmonization.

### Strengths
- The study conducts comprehensive experiments demonstrating that converting surface brain signals into source space provides a more effective input representation, facilitating neural decoding.

### Weaknesses
-  This study is primarily exploratory, focusing on the differences between various input forms. As a result, the technical contributions may appear limited to readers in the ICLR community. This paper might be better suited for publication in a more specialized journal within this field.
- The organization of this paper is difficult to follow, which might due to the absence of subtitles (like for the Dataset/Method/Implement detail ...). Additionally, a clearer structure would enhance the overall readability and coherence of the paper, allowing for a more straightforward understanding of the study's objectives and findings.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper concerns the interesting question of sensor-space vs source-space decoding of neural signals (MEG/EEG). The hypothesis is that source-space decoding can provide a number of benefits including design of spatial inductive biases, spatial data augmentations, better interpretability, zero-shot generalisation between datasets, and data harmonisation.
The paper is based on two MEG data set and find evidence in favor of the hypotheses

### Strengths
The experimental design is relevant to the question.
 Pipelines based on the source and sensor-space representations are carefully optimized for hyperparameters. 
A number of data augmentation strategies are detailed.
There is an attempt at explainable AI.

### Weaknesses
1)	The question of decoding source and sensor space for M/EEG is not novel, key works include
Edelman et al. 2015. EEG source imaging enhances the decoding of complex right-hand motor imagery tasks. IEEE Transactions on Biomedical Engineering, 63(1), pp.4-14.  (300+ citations)
Andersen et al. 2017, March. EEG source imaging assists decoding in a face recognition task. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) pp. 939-943.
Li et al. 2021. A novel decoding method for motor imagery tasks with 4D data representation and 3D convolutional neural networks. Journal of Neural Engineering, 18(4), p.046029.
Leung et al. 2024. Limited value of EEG source imaging for decoding hand movement and imagery in youth with brain lesions. Brain-Computer Interfaces, 11(3), pp.143-157.

2)	While the authors make valid attempts to place the hypotheses and experiments in a real-world/applied context, the core arguments are misleading. For example it is stated that fMRI is acquired in 3D, it is not. The 3D fMRI signal is reconstructed from a k-space measurement in many ways resembling the source reconstructed signals (M/EEG). Similarly, the argument that source space representation enables data augmentation is weak. E.g the possibility of decorrelating sensors (Laplacian filter) would enable a simple localized “sensor masking process”.

3)	Simple baselines are missing, eg. basic classifiers(logreg, SVMs etc). While it is correct that source space is useful for integrating data sets with different spatial sensos locations, extant work use simple heuristics for sensor matching, see e.g. Kostas, D., Aroca-Ouellette, S. and Rudzicz, F., 2021. BENDR: Using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data. Frontiers in Human Neuroscience, 15, p.653659. 

4)	Explainable learning from M/EEG is a rich research field and references are missing, e.g. the survey: Zhou, X., Liu, C., Zhai, L., Jia, Z., Guan, C. and Liu, Y., 2023. Interpretable and robust ai in eeg systems: A survey. arXiv preprint arXiv:2304.10755.

### Questions
Have you considered simple baselines, e.g. SVMs? It is unclear how the differences in representation dimensions (sensor vs space) interact with large MLPs

	Would it be possible to enlarge the set of experiments, there are numerous open source EEG data sets. Standard brain models could be used for source space reconstruction (many data sets do not include structural data)

	How do you compute the probabilities in Table 5 and Table 8?

### Soundness
2

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper investigates the use of source imaging for speech detection tasks. The authors compare the classification accuracy when using sensor space and source space as inputs across two datasets, employing three different network architectures. Additionally, the paper explores spatial data augmentation, cross-dataset training and testing for the two datasets.

### Strengths
* This is an important and beneficial topic for the field. 
* The paper performed many experiments including cross validation, ablation studies, etc.

### Weaknesses
I have serious doubt regarding the validity of the result. This paper did not justify the use of source space signals over sensor space signals. Specifically: (1) There is no significant improvement in classification accuracy when using source space signals. (2) The additional advantages claimed for source space signals over sensor space are unsubstantiated, as sensor space signals can also support cross-subject learning and testing (although its accuracy is unknown).
While the benefits of source imaging over sensor space signals have been demonstrated in several fields, the authors did not apply source imaging correctly to biological signals, leading to little improvement from using source imaging.
1. Minimum Norm Estimation (MNE) is essentially ridge regression. Although it is a widely used algorithm for source imaging, its localization error is higher compared to more modern imaging methods, and it cannot provide an accurate spatial distribution for the source. The authors used the entire brain space as input, which makes an accurate estimate of the whole source space crucial, but MNE tends to produce numerous false positives.
2. Source imaging can improve the signal-to-noise ratio (SNR) for specific regions or signals of interest. However, the authors did not properly select the signal segment and used the entire source region. In comparison to sensor space, where the low-SNR input consists of a few dozen channels, the source space contains hundreds of voxels, making it more difficult to extract relevant information.
3. Figure 4 does not display a proper auditory evoked potential for expected MEG signal waveforms and source imaging results, as showed in dataset's source paper (Schoffelen, 2019)  Given that MNE is ridge regression, the regularization parameter (lambda), which is related to the SNR of the sensor space, plays a crucial role. However, since the authors used single time points as the dependent input, the SNR at each time point varies significantly.
4. The claim that the cross-dataset "zero-shot" approach is impossible in sensor space lacks proof. It is feasible to perform wrapping in source space to map each subject to a common space, and the same can be done in sensor space. The accuracy of both methods remains subject to evaluation.

### Questions
What is the input time segments?

### Soundness
1

### Presentation
3

### Contribution
1

### Rating
1

### Confidence
5