# Deep probabilistic 3D angular regression for directional dark matter detectors

- Decision: Reject
- Scores: 3, 8, 5

## Abstract
Modern detectors of elementary particles are  approaching a fundamental sensitivity limit where individual quanta of charge can be localized and counted in 3D. This enables novel detectors capable of unambiguously demonstrating the particle nature of dark matter by inferring the 3D directions of elementary particles from complex point cloud data. The most complex scenario involves inferring the initial directions of low-energy electrons from their tortuous trajectories. To address this problem we develop and demonstrate the first probabilistic deep learning model that predicts 3D directions using a heteroscedastic von Mises-Fisher distribution that allows us to model data uncertainty. Our approach generalizes the cosine distance loss which is a special case of our loss function in which the uncertainty is assumed to be uniform across samples. We utilize a sparse 3D convolutional neural network architecture and develop approximations to the negative log-likelihood loss which stabilize training. On a simulated Monte Carlo test set, our end-to-end deep learning approach achieves a mean cosine distance of $0.104$ $(26^\circ)$ compared to $0.556$ $(64^\circ) $ achieved by a non-machine learning algorithm. We demonstrate that the model is well-calibrated and allows selecting low-uncertainty samples to improve accuracy. This advancement in probabilistic 3D directional learning could significantly contribute to directional dark matter detection.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a probabilistic deep learning model designed to predict the 3D directions of elementary particles from point cloud data, specifically targeting the detection of dark matter. The primary challenge tackled is determining the initial directions of low-energy electrons. The architecture used is a sparse 3D convolutional neural network. The output is parameterized using a von Mises-Fisher distribution, as the directional outputs need to be constrained to the 2-sphere. When compared against a non-machine learning approach on a simulated dataset, the proposed model shows a significant improvement in mean cosine distance.

### Strengths
S1. Method is appropriate: The idea of employing a probabilistic deep learning model for analysing particle trajectories has promise. I also like the fact that the proposed approach uses both a probabilistic output for quantifying the uncertainty as well as doing on-manifold predictions.

S2. Model Calibration: The paper shows that the model is well-calibrated on simulated data and can identify low-uncertainty samples, which is crucial for real-world applications. This ability to filter out ambiguous predictions is very useful for improving the accuracy and realiability which is important in science applications.
The comparison to a deterministic model trained with the cosine distance loss also demonstrates that the method performs well compared to the baseline.

S3. Technical Depth: The method seems to be well thought out and goes into detail on computing the negative log-likelihood loss, and practical considerations for stable training.

### Weaknesses
W1. Limited Comparison: While the paper compares the proposed method with a non-machine learning algorithm, it lacks a comprehensive comparison against other deep learning-based methods, especially those for predicting uncertainties on manifolds. See, for example [1] and the references therein.

W2. Real-world Evaluation: The paper primarily relies on a simulated test set. Although simulations are valuable, as this is an application-focussed paper, it would be useful to also evaluate the model on real-world data, possibly obtained from actual detectors, to gauge its performance under realistic conditions.

W4. Interest to the community: The paper is focussed on a very specific application and I'm not sure how relevent the method is to the ICLR community. Perhaps it would be better suited to a physics journal. Related to this, if the paper is aimed at the ICLR audience, then the introduction needs to be written in a way that clearly explains the problem and why it is important from an ML perspective.

[1] Gilitschenski, Igor, et al. "Deep orientation uncertainty learning based on a bingham loss." International conference on learning representations. 2019.

### Questions
Recommendations:

1. I would like the authors to explain why this method is relevant to the machine learning community and also give a better overview of why the problem is important (so that non-particle physiscists can understand its significance).

2. The authors should also consider comparing their method with other machine learning or deep learning methods, if available, that also target on-manifold uncertainty predictions. This will help position the contribution better with respect to existing work.
 
3. Further analysis of the model's generalizability could be insightful. This will help readers gauge the model's applicability to other problems within the broader particle detection domain.

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a deep learning framework for the estimation of 3D angles from input trajectories without timestamps. The application domain is particle physics, where detectors aim to probe electrically neutral particles and perhaps even detect dark matter through the recoil trajectories of electrically charged particles. The determination of the starting point and angle is crucial for rejecting noisy detections.

### Strengths
This is an interesting paper that formulates a solution to a problem in particle physics with a deep learning framework. The paper is presented in a way that is digestible to a non-physicist, with a very nicely explained introduction and motivation for a topic I am not familiar with. Both the proposed heteroscedastic network design and the von Mises Fisher distribution are novelties that improve the results in the simulated test cases.

### Weaknesses
I have two concerns with the paper that I will classify as "weaknesses". 

1. While the results shown on simulated test cases are encouraging and show that the approach works in principle, it is unclear how large the sim-to-real gap is. It is hard to determine how realistic the Degrad simulations are as compared to real measurements of electron recoils. Furthermore, Degrad presumably carries with it a whole host of parameters; a few are mentioned in the text (e.g., gas mixtures, temperatures, pressure, etc.). While I understand that the combination of parameters chosen was inspired by prior work (Jaegle et al.) it is unclear how well these parameters are constrained in reality and what happens to the deep-learning model if it encounters variations such as those expected in real data. E.g., does the model completely break if we now test it on trajectories generated at 21° instead of 20°, etc.? These tests for systematic error sources won't determine the gap between simulation and reality, but they will be a first step towards that goal. I think that additional experiments explicitly testing these potential sources of systematic errors and/or a discussion that fully discloses these limitations should be present. The discussion should include what is key or missing from having this work on real (not simulated) data.

2. It is unclear from the paper exactly what the key novelty is. Initially, I thought that this is the first DL framework for the task, as is also demonstrated by the result comparison in Fig. 4. After reading the conclusion, it was unclear if there were prior DL works, and this paper introduces the von Mises Fisher distribution for prediction. It crucial to clearly clarify this in the text. If this is not the first DL approach to the task, I would expect a comparison with prior DL approaches.

Minor comments
--------------------
I have a few minor comments that also need addressing:

1. Section 5 probably should be moved to the supplementary, but in any case, it should not be after Section 4.3. Estimating an arrow might be a sanity check but it is irrelevant once the reader has already seen more complex examples. 
2. What do the colors in Fig. 1 represent? Ionization? A color bar should be present and this should be mentioned in the caption
3. Consider replacing Table 1 with a more useful and digestible visual representation, like a block diagram
4. While the model includes an uncertainty estimate and is therefore heteroscedastic across different trajectories, it should be mentioned that the uncertainty is assumed to be homogeneous along a single trajectory (i.e. all data points within a trajectory have the same uncertainty)
5. The term "efficiency cut" is introduced in 4.3 (first paragraph) but only explained later in Fig. 4

### Questions
See weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper addresses the problem of predicting the initial direction of particles from recoil trajectories. The problem is potentially useful in directional detection of particles and could be used for dark matter detection. The paper models the initial directional distribution as a von Mises-Fisher distribution an uses a sparse-convolution network to fit the model parameters using. The effectiveness of the proposed network is demonstrated through experiments on simulation data.

### Strengths
- The paper presents backgrounds, motivations, and problem setups clearly and is understandable by one with limited knowledge in physics. 
- The use of 3D sparse convolution networks and NLL appear appropriate for the problem.
- Experiments show clear improvements over non-learning algorithms on simulated data.

### Weaknesses
- Lack of real-world experiments.

  The effectiveness of the proposed learning-based method is only verified on data from simulation. It also does not explain why the specific choices of simulation parameters, as describes in section 4.1, are determined. For example, would performing Gaussian smearing on the simulation data favor the Gaussian-like von Mises-Fisher model, and it is not clear whether the proposed model is still a good choice for real-world data. 
  
- Technical contribution to the learning community.

  The 3D sparse convolution and von Mises-Fisher probabilistic model are well-established techniques and do not seem to provide too much insight for the learning community. This work may be better suited for a conference/journal in physics.

  The work would be more valuable if it studies the effectiveness of different 3D learning architectures and probabilistic models when applied to real-world electron recoil data, or demonstrate the sim-to-real transferablility of such models.

### Questions
Minor questions comments:
- Some detailed information in section 3, such as 3.4, would better fit in the experiment section(e.g., section 4.2).
- What considerations are taken to determine the specific simulation parameters are used in section 4.1? Why is the test set different from the training set and how are the specific parameters determined?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
