# Generalizable Non-Line-of-Sight Imaging with Learnable Physical Priors

- Decision: Reject
- Scores: 5, 5, 6, 6

## Abstract
Non-line-of-sight (NLOS) imaging, recovering the hidden volume from indirect reflections, has attracted increasing attention due to its potential applications. Despite promising results, existing NLOS reconstruction approaches are constrained by the reliance on empirical physical priors, e.g., single fixed path compensation. Moreover, these approaches still possess limited generalization ability, particularly when dealing with scenes at a low signal-to-noise ratio (SNR). To overcome the above problems, we introduce a novel learning-based approach, comprising two key designs: Learnable Path Compensation (LPC) and Adaptive Phasor Field (APF). The LPC applies tailored path compensation coefficients to adapt to different objects in the scene, effectively reducing light wave attenuation, especially in distant regions. Meanwhile, the APF learns the precise Gaussian window of the illumination function for the phasor field, dynamically selecting the relevant spectrum band of the transient measurement. Experimental validations demonstrate that our proposed approach, only trained on synthetic data, exhibits the capability to seamlessly generalize across various real-world datasets captured by different imaging systems and characterized by low SNRs.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper presents a new end-to-end network architecture for reconstructing non-line-of-sight (NLOS) scenes from low signal-to-noise ratio (SNR) single-photon avalanche diode measurements. The network architecture is inspired by the virtual wave phasor field, and introduces two new modules, ie the learnable path compensation module and the adaptive phasor field module. 
The proposed method is shown to perform better than previous NLOS approaches (both learned and physics-based) on a public dataset and a new dataset introduced in the paper.

### Strengths
- the proposed architecture incorporates insights from virtual wave phasor fields.
- the method achieves state-of-the-art performance for NLOS reconstruction.

### Weaknesses
- I found the presentation of the method hard to follow, especially for someone who is not an expert in virtual phasor fields, and in particular the paper "Phasor field diffraction based reconstruction for fast non-line-of-sight imaging systems". I believe that an ICLR submission should be accessible to a larger audience. 
    - It was unclear to me how equation (4) and (1) are related, since $H$ is inside an integral in (1)
    - the notation is confusing in equation 3: how are the coordinates $x,y$ in $I$ related to $x_s$? Is $x_s$ referring to a 2D coordinate whereas $x$ and $y$ are 1-D coordinates respectively?
    - there should be at least some intuitive explanation of the function $\Phi$ in equation 3.

- Despite the state-of-the-art performance, the gap with the previous best performing model is relatively mild (often less than 0.5 dB).

### Questions
- why is the rendering model called 'rendering'? Is not clear to me why the relatively generic 2D CNN architecture should be a renderer.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper aims at solving the problem that existing NLOS methods rely on some empirical physical prior during their implement. To address this problem, the authors introduce a Learnable Path Compensation (LPC) module and an Adaptive Phasor Field (APF) module to optimize 3 distance-related weights and a Gaussian window separately. The results on both synthetic data and real data demonstrate the effectiveness of the proposed method.

### Strengths
The originality of this paper arises from the improvement of an existing NLOS reconstruction network LFE (Chen et al., 2020). The proposed method can achieve global intensity consistency and avoid manual adjustment of empirical parameters.

### Weaknesses
IMO, the physical explanation of this paper is inaccurate and misleading, especially the explanation of the learning of the Gaussian window. And the improvement of the results is limited quantitatively and qualitatively, especially compared with NLOST.

### Questions
1.	$P(x_p,t)$ is the virtual illumination function in the Phasor-field methods (Liu et al.,2019), but it is not necessary a Gaussian-shaped function. It can be any type of the light source function, such as ambient light. Please refer to Table S.2 of Liu’s paper. Therefore, the physical explanation of learning a Gaussian window is not correct in my opinion. 
2.	Even though some light source functions are Gaussian-shaped, such as pulsed point light function, the frequency spectrum band should be a certain value. It shouldn’t be a changeable value for different scenarios. So, from this point of view, the explanation of learning a Gaussian window is also not correct.
3.	The $I(x,y)$ in Eq.(3) should be the image of the virtual imaging system, it is not the hidden object as claimed in the paper, please refer to the explanation of Eq.(6) in Liu’s paper.
4.	Could you explain in what dimensions of measurements you did the FFT to get the spectrum of the light source function? I think you can only get the spatial and temporal sampling frequency if you operate FFT on raw ToF measurement, rather than illumination spectrum.
5.	Lines 78-79 claims that the proposed method can solve the dark counts noise of SPAD and ambient light. But I didn’t find the detailed solution in the main paper. There is just a simple description of the definition in Eq.(4).
6.	Why did you choose 3 path compensation weights? I didn’t find the related content in O’Toole et al., 2018. Could you clarify it for me?
7.	Which method is chosen as your baseline in Fig.9. I suppose to be LFE according to Fig.2. But the baseline results shown in Fig.9 are different with the LFE results in Fig.7. So, it’s hard to demonstrate the effectiveness of this ablations studies.

## Reference
- Chen et al, Learned feature embeddings for non-line-of-sight imaging and recognition, TOG 2020
- Liu et al, Non-line-of-sight imaging using phasor-field virtual wave optics, Nature, 2019
- O’Toole et al, Confocal non-line-of-sight imaging based on the light-cone transform, Nature 2018

------
**post-rebuttal:**

My questions are properly answered. However, it also reveals the authors' original motivation (the physical intuition and explanation) is somewhat far-fetched. As a result, I raise my score to borderline reject.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a learning-based Non-line-of-sight (NLOS) reconstruction method based on the standard virtual wave phasor field formulation that learns radiometric intensity falloff coefficients to compensate for falloff due to varying materials present in the scene and frequency cutoff values to compensate for varying SNR conditions.

EDIT: upgraded score from a 5 to a 6 given authors response.

### Strengths
The paper is well motivated and distinguished clearly from previous works, as the method is structured around leveraging flexibility in the virtual phasor field formulation that was not used by prior methods to improve NLOS reconstruction performance. The improvement in reconstruction performance is clearly demonstrated with extensive experiments on existing and newly gathered datasets across a large number of NLOS reconstruction methods.

### Weaknesses
Though the method shows strong performance compared to baselines, I'm not sure the method is better for the reasons that the authors say, rather than just because the method has more parameters and is trained on more data than previous methods. In particular, the authors make several claims about why their method works better that I am not sure are precisely substantiated in the paper. 

First, the authors claim that the using adaptive coefficients for the radiometric intensity falloff is better because different materials will have different coefficients and their design can adapt to the varying materials. For example, diffuse objects should have a falloff coefficient of 4 and retro-reflective 2. While the authors do visualize in Figure 1 that their method using adaptive coefficients can recover both diffuse and retroreflective objects simultaneously, I could not find in the paper any visualizations of the compensation coefficients. I would recommend the authors include a image plot of the compensation coefficients, and would expect that regions per object would have their own coefficient each. 

Second, the authors claim that the adaptive phasor field improves performance across a variety of SNR conditions. However, in Table 2, which contains the quantitative results across a variety of SNR conditions, the decrease in performance as SNR decreases for the author's proposed method is similar to the other methods. Thus, given the data presented in the paper it seems that it is not the case the author's proposed method is more robust in lower SNR conditions than other methods, despite having uniformly better performance across SNR settings. I'm not sure if it's because the SNR conditions in the testing sets are not varied enough, or if the adaptive phasor field doesn't actually help in diverse SNR settings. Regardless, I recommend that the authors provide some visualization of the standard deviations for the gaussian of the illumination phasor field predicted by their adaptive phasor field to better help the reader understand what the method is actually doing.

### Questions
Major:
Reiterating some points from weaknesses:
- I would recommend the authors include a image plot of the compensation coefficients predicted by the learnable path compression, and would expect that regions per object would have their own coefficient each, because I wonder if the model can actually infer that different materials should have different compensation coefficients like the authors claim. 
- I recommend that the authors provide some visualization of the standard deviations for the gaussian of the illumination phasor field predicted by their adaptive phasor field to better help the reader understand what the method is actually doing, because I have questions about if the adaptive standard deviations used in the wave propagation are actually reducing the noise in low SNR environments like the authors claim. Specifically, I would expect for low SNR environments the standard deviation to be lower and vice versa, because low SNR implies higher noise and thus the model should use a tighter gaussian during wave propagation to filter out more noise. 

I would be willing to upgrade my score if the authors could provide more evidence that the method works the way they claim. 

Minor:
- I don't quite understand in equation (8) why the inverse Fourier transform is taken, because according to Figure 2, $F_A$ goes through a Fourier transform before getting input into the wave propagation model. It seems to me to be more concise and efficient to keep the result of convolving the illumination phasor field with the compensated features in the frequency domain for wave propagation rather than transforming it back to the spatial/time domain.
- For the real data at varying SNR levels, 10 dB, 5 dB, and 3 dB, how did the authors capture this data? I would expect that the exposure time is different between captures, but I couldn't see any details about this in the paper.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes to learn the physical priors for  non-line-of-sight imaging. More specifically two components are proposed, learnable path compensation to attenuate the features to compensate for radiometric falloff and an adaptive bandpass filter to suppress background noise. Training and quantitative evaluations are done on synthetic data. Qualitative evaluations are done on real-world data including a three-scene dataset captured by the authors. The method shows better performance in terms of preserving sharp features and reducing background noise.

### Strengths
Typically training on synthetic data does not generalize to real data due to synthetic and real gap. However, the proposed method learns to compensate for radiometric falloff and suppress background noise using synthetic training data and is able to generalize to real-world data.

### Weaknesses
The paper lacks discussion on the choice of G_z for initial compensation. It’s also unclear how much the 3D CNN block is contributing compared to the initially compensated features.

The paper shows results on a custom dataset with the “composite” scene consisting of multiple surface materials. However, there is no quantitative evaluation on how different surface materials at different distances are compensated by LPC.

### Questions
How much does the initial compensated feature in Eq. 5 contribute to the method?

What is the motivation behind average pool as opposed to max pool which would pick the prominent feature?

For the LPC pipeline, how much is the 3D CNN contributing as opposed to the initially compensated feature?

What is the network architecture of spectrum convolution in Figure 4? 

How many different materials were used in the captured scene?

### Soundness
3

### Presentation
3

### Contribution
3
