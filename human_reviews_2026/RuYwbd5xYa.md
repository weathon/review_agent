# SCRAPL: Scattering Transform with Random Paths for Machine Learning

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
The Euclidean distance between wavelet scattering transform coefficients (known as paths) provides informative gradients for perceptual quality assessment of deep inverse problems in computer vision, speech, and audio processing. However, these transforms are computationally expensive when employed as differentiable loss functions for stochastic gradient descent due to their numerous paths, which significantly limits their use in neural network training. Against this problem, we propose "Scattering transform with Random Paths for machine Learning" (SCRAPL): a stochastic optimization scheme for efficient evaluation of multivariable scattering transforms. We implement SCRAPL for the joint time–frequency scattering transform (JTFS) which demodulates spectrotemporal patterns at multiple scales and rates, allowing a fine characterization of intermittent auditory textures. We apply SCRAPL to differentiable digital signal processing (DDSP), specifically, unsupervised sound matching of a granular synthesizer and the Roland TR-808 drum machine. We also propose an initialization heuristic based on importance sampling, which adapts SCRAPL to the perceptual content of the dataset, improving neural network convergence and evaluation performance. We make our code and audio samples available and provide SCRAPL as a Python package.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This article introduces an efficient way to compute  the wavelet scattering transform based on random sampling. This allows to learn auto-encoder at scale To measure dissimilarity between isolated sounds, under a perceptual distance defined by this transform. To avoid the variance of the random sampling, various methods from the literature of stochastic gradient descent (SGD) are considered.  The proposed scheme is applied to unsupervised sound matching, showing comparable performance compared to joint-time frequency scattering transform and other baseline methods.

### Strengths
-	The problem on how to compute the scattering transform efficiently is well motivated. 
-	The obtained results are very promising. 
-	The article is mostly well written.

### Weaknesses
- The modified SGD in P-Adam (eq 8) and P-SAGA (eq 9) seem to be quite heuristic as there is no convergence analysis in the literature. For example, in P-SAGA, which rely on the gradients from P-ADAM is not a standard SAGA method given in the citation (Defazio et al 2014). 
- The section 3.4 is not so clear and might contain some technical flaw.
- It is not clear whether the proposed scheme has any numerical convergence. 
- The evaluation metric is not so clear in Table 1 to non-expert. In Table 4, the evaluation scores seem to be biased towards the loss function that you are optimizing. Any human-based or more objective evaluation would be needed to compare different methods.

### Questions
- Is it possible to add some reference to the convergence analysis of P-Adam and P-SAGA methods? Why don’t you just use the standard SAGA method?
- In Section 3.4, is theta belong R^U according to eq. 11? Is theta = the output of encoder?
- how pi_p in eq. 10 is used next?
- Is C_{u,p} in eq. 12 always positive? If yes, can you explain clearly? If not , how to ensure that pi_p is a probability in eq. 10?
- To understand the numerical convergence of SCRARL, an analysis on the final training loss (eq. 4) is needed to compare with JTFS. 
- Could you give a definition on the theta_syn, theta_density etc to explain Table 1?  
- How the supervised version P-loss method works? Why do you choose this to compare with the other methods?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a computationally efficient stochastic approximation to the full scattering transform to enable use of scattering based similarity as a loss function, one that relates to perceptual quality, in generative modeling.  The proposed approach consists of four steps: a) A sampling of scattering “paths”, b) a path-wise adaptive momentum estimation and using that in an extension of the Adam optimization algorithm, c) path-wise stochastic average gradient with acceleration using previous gradient values, and d) an importance sampling approach to bring stochastic approximation of spectral loss closer to P-loss.  Empirical results are presented on unsupervised sound matching experiments with granular synth, chirplet synth, and Roland TR-808 synth.

### Strengths
* A stochastic approach that makes optimization under scattering transforms computationally efficient
* Empirical results demonstrating that proposed approach achieves accuracies within a factor of two relative to the extensive joint time-frequency scattering transforms, while being within 3x of the much more efficient multi-scale spectral loss approach.
* Clear, well written paper

### Weaknesses
* The empirical results on a sound matching setup which is of relatively limited interest to the broader NeurIPS and ML community.  Results on more audio generation tasks utilizing perceptual qualities of scattering transforms will strengthen the paper significantly.

### Questions
n/a

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces SCRAPL (Scattering transform with Random Paths for machine Learning), a stochastic optimization framework that accelerates the use of multivariable scattering transforms as differentiable loss functions in deep learning. Scattering transforms (ST) are wavelet-based operators offering invariance and stability properties useful for perceptual modeling, but their computational cost has hindered large-scale training. SCRAPL addresses this by sampling a subset of ST “paths” at each iteration, yielding unbiased stochastic gradient estimates. To reduce variance and improve convergence, the authors introduce P-Adam (path-wise adaptive moment estimation) and P-SAGA (path-wise stochastic average gradient) algorithms, as well as a θ-importance sampling heuristic that biases path sampling toward perceptually informative regions. The method is implemented for the joint time–frequency scattering transform (JTFS) and evaluated on three differentiable digital signal processing (DDSP) tasks: granular synth, chirplet synth, and Roland TR-808 sound matching. Results show SCRAPL achieves accuracy close to JTFS at a fraction of its computational cost, outperforming existing perceptual losses such as multiscale spectral loss (MSS) and audio embedding distances.

### Strengths
SCRAPL is a novel and technically well-motivated approach that makes scattering transforms practical for end-to-end differentiable learning. The paper is clearly written and grounded in both theory and application. The proposed stochastic approximation is theoretically justified (unbiased gradient under uniform sampling) and supported by well-designed optimization variants (P-Adam, P-SAGA) that address variance and non-i.i.d. gradient issues. The θ-importance sampling heuristic is a creative contribution that connects the optimization strategy to perceptual signal characteristics. Experimental validation is thorough, spanning synthetic and real-world DDSP tasks, with convincing quantitative and qualitative results. The work provides clear computational benchmarks, reproducibility details, and a public implementation, all of which increase its credibility and impact. Overall, the paper bridges an important gap between scattering theory and modern deep learning practice.

### Weaknesses
While the proposed framework is promising, certain limitations remain. The experiments are largely confined to audio applications, and it is unclear how well SCRAPL generalizes to image or other modalities that use scattering transforms. Although θ-importance sampling improves convergence, its computation involves complex gradient–Hessian interactions that may limit scalability in higher-dimensional settings. The empirical comparisons, while comprehensive, rely on relatively small neural networks and do not explore how SCRAPL performs in large-scale or supervised training contexts. The ablation studies, though informative, could include statistical significance tests to strengthen claims. Finally, the paper’s reliance on many heuristic design choices (e.g., importance weighting, hyperparameter schedules) makes it difficult to assess robustness across diverse architectures and datasets.

### Questions
Have you tested or considered SCRAPL in vision tasks using 2D scattering transforms (e.g., roto-translation ST)? If not, do you anticipate any challenges extending it to images?

Could you clarify how often this importance distribution must be recomputed during training, and how its computational cost compares to one full scattering transform?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a stochastic optimization scheme to make the computationally expensive wavelet scattering transform (ST) viable for large-scale machine learning, particularly in Differentiable Digital Signal Processing (DDSP). The core problem addressed is that the full scattering transform, the Joint Time-Frequency Scattering Transform (JTFS), may be used as a perceptually relevant loss function for tasks like sound matching, but its complexity makes it too slow and memory-intensive for use in stochastic gradient descent. The proposed solution, SCRAPL (Scattering transform with Random Paths for machine Learning), uses a stochastic approach to efficiently approximate the gradient of the full ST loss.
The study reports the results for using SCRAPL in the context unsupervised sound matching—a nonlinear inverse problem where an autoencoder learns the control parameters of an audio synthesizer (DDSP)—for which SCRAPL demonstrates a superior tradeoff between accuracy and computational efficiency compared to other losses.

### Strengths
•	SCRAPL makes the Joint Time-Frequency Scattering (JTFS) transform feasible for use in the loss function of large-scale deep learning by overcoming its prohibitively high computational and memory costs.
•	The paper proposes an innovative stochastic approximation scheme—sampling just a single path—which guarantees an unbiased estimate of the true loss gradient in expectation, together with a novel Path-Wise Optimization (P-Adam) to manage the high variance of the single-path sampling, stabilizing convergence by maintaining individual moment statistics for every scattering path.
•	The experimental evaluation is performed in the context of unsupervised sound matching with DDSP. It demonstrates that despite the stochastic approximation, the proposed algorithm offers sound matching results that are close to the full JTFS, while achieving a significantly smaller memory footprint, proving its practical efficiency.

### Weaknesses
•	Limited Experimental Scope and Generalizability: The evaluation is performed on a relatively limited number of sound matching examples, which does not permit concluding on the general applicability of the method across different audio domains or tasks.
•	When listening to the audio examples, one should note that while the sound matching algorithm converges, the perceptual similarity of the original and generated sounds is not generally satisfying. In quite a few of the examples, the output sound is perceptually very far from the target. This calls the usefulness of the JTFS, and with it the SCRAPL algorithm, into question.
•	Increased Implementation and Hyperparameter Complexity: The solution requires specialized algorithms (P-Adam, ϕ-SAGA) and a novel sampling heuristic (θ-IS), which significantly increases the implementation burden and the number of hyperparameters that need careful tuning.

### Questions
I think this paper would win a lot if one had a better idea about the performance of the underlying JTFS algorithm in the context of sound matching. Listening to the examples, I had the feeling that the results are particularly bad for HH (higher frequencies) and noise. Some sounds are perceptually very close, and others have hardly anything in common. I had a HH, which was transformed into a chirp.

### Soundness
1

### Presentation
3

### Contribution
3
