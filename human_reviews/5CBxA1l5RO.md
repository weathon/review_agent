# TimewarpVAE: Simultaneous Time-Warping and Representation Learning of Trajectories

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 3, 6, 5

## Abstract
Human demonstrations of trajectories are an important source of training data for many machine learning problems. However, the difficulty of collecting human demonstration data for complex tasks makes learning efficient representations of those trajectories challenging. For many problems, such as for handwriting or for quasistatic dexterous manipulation, the exact timings of the trajectories should be factored from their spatial path characteristics. In this work, we propose TimewarpVAE, a fully differentiable manifold-learning algorithm that incorporates Dynamic Time Warping (DTW) to simultaneously learn both timing variations and latent factors of spatial variation. We show how the TimewarpVAE algorithm learns appropriate time alignments and meaningful representations of spatial variations in small handwriting and fork manipulation datasets. Our results have lower spatial reconstruction test error than baseline approaches and the learned low-dimensional representations can be used to efficiently generate semantically meaningful novel trajectories.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes Timewarp VAE, which simultaneously learns both spatial variations with temporal variations. It is based on beta-VAE with two additional modules: a temporal encoder and a time-warper, the decoder then takes the canonical trajectory and time, which are trained jointly to enable good reconstruction of position trajectory at each timestep. Experimental results on small handwriting and fork manipulation datasets show superior performance compared with baseline approaches.

### Strengths
The paper is very well written.
The paper proposes a neat idea to incorporate temporal information into existing beta-VAE methods. 
The paper has a range of ablation studies in the empirical study to examine the chosen architecture.

### Weaknesses
There is limited comparison with other approaches such as sequence-to-sequence type of architecture. The decoder is a sequence of fully connected layers, if using some richer architecture like RNN/transformer type of models which can incorporate temporal information, it is not clear how much value the proposed time warper adds. The paper shows results on two experiments: collected fork manipulation data and a small handwringing gestures dataset. More experimental on other datasets will also make the paper stronger.

### Questions
The ablation study on ‘NoTimewarp’ studies the performance when replacing the time-warper with identity function, which had inferior performance indicating the importance of the time warper. I wonder if the input trajectories are represented as equi-spaced sequences (i.e., with some temporal information), where good reconstruction would include both the spatial reconstruction and temporal reconstruction, how does the model perform. 

Do all the training trajectories have the same length T? 

What is the intuition of lambda, e.g., what does it mean when having small lambda vs big lambda? 

These papers may be related: https://www.sciencedirect.com/science/article/pii/S0925231220312017
https://openreview.net/pdf?id=Byx1VnR9K7

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a neural network architecture for time warp-invariant representation learning i.e., disentangle timing variations from the spatial variations in a dataset. The network contains separate modules for spatial and temporal encoding and a time warping module that maps the temporal encoding and an input timestamp to the output resampled timestamp. The spatial encoding, which is a beta-VAE, and the resampled timestamp are input to the decoder to obtain the input time series at that timestamp. All the modules trained end-to-end using the beta-VAE objective as well as a time-warping regularization term.

### Strengths
1. The paper is well-written. The ideas are also simple, intuitive and easy to implement.

2. Experiments show clear improvements over baseline beta-VAE.

### Weaknesses
I believe this paper has some fundamental weaknesses related to how novel the ideas are. 

There are other works that I list here that are very close to this submission that make the ideas in the paper not very novel, in my opinion

(a) Diffeomorphic Temporal Alignment Nets: https://proceedings.neurips.cc/paper_files/paper/2019/file/db98dc0dbafde48e8f74c0de001d35e4-Paper.pdf
(b) SrvfRegNet: Elastic Function Registration Using Deep Neural Networks: https://openaccess.thecvf.com/content/CVPR2021W/DiffCVML/papers/Chen_SrvfRegNet_Elastic_Function_Registration_Using_Deep_Neural_Networks_CVPRW_2021_paper.pdf
(c) Rate-invariant autoencoding of time-series: https://ieeexplore.ieee.org/abstract/document/9053983
(d) Regularization-free Diffeomorphic Temporal Alignment Nets: https://openreview.net/pdf?id=7IbLWa0anE

Especially (c), it also has an encoder that produces a single rate-invariant latent vector for the full time series and the other part of the latent space is a time warp followed by a time warping module, and the network is trained using reconstruction loss.

These papers are already in literature and in light of them, the ideas in the paper do not appear to be very novel to me. It would be good if the authors can help me understand how this paper is different.

Even if the ideas are substantially different, some of these papers should appear as baselines in the experiments. At the moment, the only baseline is the beta-VAE, which predictably fails to capture the time-series misalignments.

### Questions
No additional questions.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes to explicitly learn a time-warping network in learning VAE-based trajectory distributions. The motivation lies in decomposing models that capture spatial and temporal variations and aligning a set of trajectories to a manifold with canonical time index, unike pair alignment in traditional Dynamic Time Warping. The paper also discusses several design choices to further improve the performance, including data augmentation based on time-perturbation, regularisation of the time-warping network output and linear/nonlinear basis in decoder networks. The proposed model is validated in handwriting in-air and fork manipulation datasets. The proposed TimewarpVAE outperforms baseline beta-VAE and ablative versions by reconstructing samples that can be better spatially aligned with the test input. Qualitative result also shows a preferred interpolation performance in the latent space on the handwriting dataset.

### Strengths
* The paper writing is clear and it is especially commendable on the detailed considerations and theoretical insights on the introduction of loss, regularisation terms, network design and association to continuous DTW formulations.

* The motivation of attempting to decouple the modelling of temporal and spatial variabilities is well grounded and can find many applications in modelling temporal data.

* The idea is easy to follow and seems to work with a few evidence on better model compression and generalisation capacity.

### Weaknesses
* The results could be more convincing if the experiments can go beyond low-dimensional data. Both handwriting and fork data are limited to gesture pose that only pertains to a handful degree-of-freedom. beta-VAE appears to catch up when a larger latent space is used so it is unclear if the advantage of TimewarpVAE can persist in face of more spatial complexity.

* It is hard to tell if time warping and spatial networks are actually extracting the expected information. The results in interpretable latent space look circumstantial since it is only about spatial dimensions and not fully disentangled. beta-VAE indeed cannot realise disentanglement solely based on reconstruction, but still the independence between the spatial and temporal components is not sufficient addressed.

### Questions
* Can TimewarpVAE show advantageous performance on larger-scale dataset such as mocap skeletal data involving many human limbs?

* Can we have more direct evidence showing the spatial latent variable and canonical time index can independently control the variations of trajectories along the expected dimensions.

* The fork dataset has part as the scaled quaternion while the error appears to be evaluated in the sense of Euclidean. Will this impact training and evaluation?  

* Can TimewarpVAE work with multiple data modalities, e.g. handwriting trajectories for all alphabetical letters? Will it need prior other than isotropic Gaussian? Will the identification of time variation help us to have a more structured latent space to group each data modal?

* How TimewarpVAE is related to other generative models with dynamical latent space, such as VAE-DMP [a] ?

[a] Chen et al, Dynamic movement primitives in latent space of time-dependent variational autoencoders, Humanoids 2016

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This manuscript proposes a manifold learning technique to parameterize variation in spatial trajectory datasets by separately factorizing spatial and temporal variation. They demonstrate applications to handwriting and fork movement. They propose a fully differentiable architecture that inputs a trajectory and a timepoint in separate branches. The trajectory is passed through a beta VAE and a piece-wise linear time warping module, along with the temporal input, that uses DTW to align the input time to a canonical time. This design attempts to factorize spatial variations to the beta-VAE and temporal variations to the temporal module.  They evaluate their approach by benchmarking the reconstruction error using three latent dimensions against a beta VAE with no time warping and also the timewarpVAE architecture with the timewarping module set to the identity, finding improved millimeter error performance and rate distortion, which is defined as the KL divergence loss in the VAE. They further show improved reconstruction for handwriting datasets compared to PCA, across choices of the latent dimension, and close with some ablations

### Strengths
* The model is well described and I believe novel.

* The problem of aligning trajectories across datasets is important, and the latent space examples are nice.

* There is a comprehensive supplement with methodological documentation.

### Weaknesses
* I view this paper as borderline because the number of evaluations is low, and more importantly there are no comparisons with other techniques present in the literature, only reduced versions of the model presented. It is unclear exactly how to position the work to the literature. If the contribution is just a way to build generative models of a set of sequences then the contribution is modest, at least with the range of examples shown. If the method surpasses other approaches for sequence reconstruction then it is more valuable, but it would need to be compared to other reconstruction approaches, eg DMP. 

* A bit too much background given on the beta-VAE, which is not novel.

* The fork experiment is a bit idiosyncratic and I am not sure what the significance is.

### Questions
* Can you demonstrate that the handwriting representation is more useful for a downstream task, for example classification. 

* Can you give any quantitative comparisons with DMP style dimensionality reduction? 

*

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair
