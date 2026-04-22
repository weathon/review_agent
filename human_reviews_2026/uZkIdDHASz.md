# Fast Two-photon Microscopy by Neuroimaging with Oblong Random Acquisition (NORA)

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Advances in neural imaging have enabled neuroscientists to study how the activity of large neural populations produce perception, behavior and cognition. Despite many developments in optical methods, there exists a fundamental tradeoff between imaging speed, field of view, and resolution that limits the scope of neural imaging, especially for the raster-scanning multi-photon imaging needed for imaging deeper into the brain. One approach to overcoming this trade-off is computational imaging, in which an imaging system efficiently encodes the target image through its optical design and then recovers the acquired information through inverting the encoded measurements algorithmically. Computational imaging thus fundamentally depends on the reliability of recovery. While such approaches are emerging for recovery of optical neural imaging from encoded measurements, they lack a core theoretical sampling theory that will guarantee reliable and accurate recovery. We present here such a theory, based on the widely used model of functional optical imaging videos being low-rank. We show that under simple blurring and randomized line-subsampling conditions, full videos can be recovered from a small fraction of the lines, providing the opportunity for an order-of-magnitude speedup. We use this theory to develop a practical design for fast imaging: Neuroimaging with Oblong Random Acquisition (NORA). NORA, guided by our theory, can be implemented through simple-to-implement changes to widely available systems. Moreover, following our theory, NORA reconstructs the entire video together via nuclear-norm minimization on the pixels-by-time matrix, rather than more common frame-by-frame recovery. We simulated NORA imaging using the Neural Anatomy and Optical Microscopy (NAOMi) biophysical simulator, showing that NORA can accurately recover 400~$\mu$m~X~400~$\mu$m fields of view at subsampling rates up to 20X despite realistic noise and motion conditions, thereby demonstrating that our theory holds. These speeds open up the capability of future systems to extend into imaging faster processes in neural systems, such as voltage and glutamate.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a reconstruction framework for acquisition using a two-photon microscope. After an analysis of the image formation, an optimization problem is build for the reconstruction. The problem is solved using a fast conical optimization scheme. The experiments show some interesting results and the system seems promising for imaging.

### Strengths
The reconstruction problem is cast as a matrix completion problem. The following optimization problem is motivated by well know reconstruction analysis and results. In the proposed setting, the main contribution of the paper is the central theorem that gives some insight on the reconstruction error. It gives ideas on how many measurements are needed for an acceptable reconstruction.

### Weaknesses
I see several weaknesses in the paper.

- The implementation is an optical framework. Such part will only interest people with instrumentation knowledge. Thus, the ICLR community may be not the best for such contribution.
- There is no analysis of the noise from a physics point of view. What is the nature of the error matrix? Is it random? Is it compound of several error terms?
- Please check how the references are inserted into the LaTeX file and use \citep for paper citation and \citet for title...

### Questions
- Does the noise mostly Gaussian or Poissonian? If Poissonian would it be more interesting to take it into account using Anscombe transform (see [1])?
- What is $A_n$ in equation (13)?
- What is the adjoint operator of $\mathcal{A}_k$?


 [1] Azzari, L., & Foi, A. (2016). Variance stabilization for noisy+ estimate combination in iterative Poisson denoising. IEEE signal processing letters, 23(8), 1086-1090.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a computational two-photon imaging scheme that addresses the speed limit of traditional point-by-point scanning 2P imaging. It skips most raster lines per frame while slightly blurring along the slow-scan axis with an oblong PSF, and reconstructs the full pixels-by-time matrix with nuclear-norm minimization. The theory proves that under a low-rank assumption on the sample, accurate video reconstruction can be achieved under a very low sampling rate. Simulations with NAOMi indicate accurate recovery at 10–20× line-subsampling under realistic noise and motion, preserving ROI time traces.

### Strengths
The paper is clearly written with theoretical analysis.

### Weaknesses
No experimental demonstration conducted, no adequate comparison with existing techniques.

### Questions
1. The simulation of blurring and subsampling (line 313-315) does not follow the real experiment scenario. To simulate the real 2P imaging in your schematic, you should apply subsampling on the ground truth (sample under observation) first, and then apply the blurring PSF. Otherwise, the crosstalk from neighboring (unsampled) lines will mix up with your sampled ones. The forward model, reconstruction algorithm, as well as the theory proofs should be modified accordingly as well. 
2. What are the spatial extents of line-by-line and rigid motions being introduced compared to the neuron sizes? It would be great to see the trend in reconstruction quality as motion increases, and under what kinds of motion the reconstruction finally breaks.
3. How does this compare to line-scanning-based 2P imaging, which only needs one galvo scanner and will be, in principle, much faster than this technique?

[1] Tal, Eran, Dan Oron, and Yaron Silberberg. "Improved depth resolution in video-rate line-scanning multiphoton microscopy using temporal focusing." Optics letters 30.13 (2005): 1686-1688.

[2] Xue, Yi, et al. "Scattering reduction by structured light illumination in line-scanning temporal focusing microscopy." Biomedical optics express 9.11 (2018): 5654-5666.

4. What is the exact sensor noise level being introduced in the simulation? How does the reconstruction react to the scattering introduced noise in neurons?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes NORA to enhance the resolution of standard two-photon microscopy by combining optical blurring, subsampling, and computational reconstruction. Two cylindrical lenses generate an elongated PSF, enabling a blur-and-subsample strategy in which each high-resolution pixel contributes to multiple overlapping wide-line scans. The forward imaging model represents the optical path as a linear blur operator applied to a low-rank fluorescence video matrix, followed by subsampling. Reconstruction is performed via nuclear norm-regularized least-squares optimization, which leverages correlations across frames and the low-rank prior to recover high-resolution pixel values from undersampled measurements.

### Strengths
1. NORA achieves accelerated two-photon microscopy acquisition by combining random line scanning with an elongated PSF, which can be implemented with minimal hardware changes.
2. The paper is well-structured, clear, and easy to follow.

### Weaknesses
1. The reconstruction in NORA relies on nuclear norm optimization, modeling the fluorescence video as a low-rank matrix. This low-rank prior may not hold when the imaged activity is highly complex or nonlinear, such as in large-scale rapid neural dynamics or highly dynamic cellular structures, potentially leading to degraded reconstruction performance. In other words, if the intrinsic dimensionality of the video significantly exceeds the assumed rank, increasing the number of samples may still be insufficient to accurately recover fine details.
2. Elongating the PSF allows integration of information along the slow-scan direction, reducing the number of required scans; however, it introduces local blurring. While the spatial resolution along the fast-scan axis is preserved, resolution along the slow-scan axis is inevitably compromised, which may limit the method’s suitability for experiments requiring fine structural analysis. Moreover, the extent of PSF elongation is inherently limited and cannot fully cover missing scan lines, making reconstruction still dependent on multi-frame information and the low-rank prior.
3. While NORA leverages low-rank priors and linear forward modeling to achieve high-speed imaging, its performance is inherently limited by assumptions such as low-rank structure and the partial blurring along the slow-scan axis. These limitations raise the question of whether deep learning-based approaches could further enhance imaging performance.
4. The experimental evaluation of NORA is limited by the lack of testing under realistic imaging conditions. Additionally, the study provides limited quantitative analysis of image quality, such as metrics for spatial resolution, reconstruction accuracy, or signal-to-noise ratio, making it difficult to rigorously compare the method against existing approaches or to fully characterize its practical performance.

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
2

### Rating Number
2

### Confidence
3

### Summary
The authors develop a method for recovering high-resolution neuroimaging videos  from low-resolution measurements (specifically, blurred randomized line sampling).  The method solves  sparse inverse problem, exploiting the spatio-temporal redundancy of the globally moving signal, and is tested on a two-photon microscopy simulator.

### Strengths
An important practical problem, and the presentation is generally clear.

### Weaknesses
Novelty?  I assume these videos are undergoing globally rigid motion, which induces substantial spatio-temporal redundancy.  I don't know the multi-photon imaging literature well, but this is a well-studied problem in photographic video processing, and many algorithms exist to perform motion-compensated estimation or restoration (including those that underlie video coding systems like MPEG).  This also shows up in the visual neuroscience literature, where some authors have explored how human vision can maintain high acuity when the eys are constantly moving (both large saccadic eye movements, and small fluctuations).  There are currently no citations to any of this literature.

A secondary concern, perhaps more for the area chairs to decide, is whether ICLR is the right venue for this paper.  Although estimation is a theme in the meeting, this paper does not discuss representations, or learning.

### Questions
Would be interesting to see a comparison of the compressed-sensing style solution used in the paper against a more traditional translational motion solution,  which assumes the spatio-temporal signal is two dimensional (or, equivalently, that it's Fourier spectrum lies on a plane).  For non-translational (but still smooth) motions, this can be done locally, as is found in many optic flow estimation algorithms.

### Soundness
4

### Presentation
3

### Contribution
2
