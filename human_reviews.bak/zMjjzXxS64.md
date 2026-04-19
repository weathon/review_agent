# FreSh: Frequency Shifting for Accelerated Neural Representation Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 8

## Abstract
Implicit Neural Representations (INRs) have recently gained attention as a powerful approach for continuously representing signals such as images, videos, and 3D shapes using multilayer perceptrons (MLPs). However, MLPs are known to exhibit a low-frequency bias, limiting their ability to capture high-frequency details accurately. This limitation is typically addressed by incorporating high-frequency input embeddings or specialized activation layers. In this work, we demonstrate that these embeddings and activations are often configured with hyperparameters that perform well on average but are suboptimal for specific input signals under consideration, necessitating a costly grid search to identify optimal settings. Our key observation is that the initial frequency spectrum of an untrained model's output correlates strongly with the model's eventual performance on a given target signal. Leveraging this insight, we propose frequency shifting (or FreSh), a method that selects embedding hyperparameters to align the frequency spectrum of the model’s initial output with that of the target signal. We show that this simple initialization technique improves performance across various neural representation methods and tasks, achieving results comparable to extensive hyperparameter sweeps but with only marginal computational overhead compared to training a single model with default hyperparameters.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors notice that the initial frequency spectrum of an untrained model’s output correlates strongly with the model’s eventual performance on a given target signal. They want to “pretrain” the untrained model output with the end signal through a hyperparameter search over initialization parameters.

### Strengths
The paper is well written, and introduces a novel method of initialization by fitting the frequency spectrum of the untrained model with the desired downstream task. The logic is sound, and this concept may apply to other fields.

### Weaknesses
Evidence
- The experimental evidence is extremely weak in my opinion and does not demonstrate a significant increase that would warrant other researchers’ usage of this method. Most results show improvements of less than 1dB, which is essentially identical to the naked eye.
- The results in Table 1 are concerning. After 15,000 steps the difference between the Base Siren model and using Fresh have virtually no difference (PSNRs of 31.18 and 31.81). 
- Table 3 - Why does FreSH have no improvement on Finer and Finer with k=0 and most samples? If it is only applicable to the SIREN and Fourier features model it would be way too limited to warrant an accept as these are several years old baselines.
- Moreover in Table 6, overall the results are not impressive as most PSNRs increase by < 1dB (i.e. Finer has virtually no change)
- Table 8, why does Fresh hurt the results when you include time? It seems as expected input for video.

Baselines
- Paper seems to have a lot of overlap with SPDER https://arxiv.org/pdf/2306.15242 (especially in theory, and that the untrained model output aligns with end reconstruction) and should compare the baseline against it.
- Other baselines should be (Ramasinghe & Lucey, 2022) which suggests Gaussian activations. It is mentioned in the paper but not reported on to the best of my knowledge.
- Instant NGP (HashGrid?) may also be used in the image baseline, as it excels in high-resolution images but is not included here for that.
- Overall, SIREN seems to be a very weak baseline model. I understand that tuning the omega_0 parameter can improve results, but I don’t believe this to be as significant of a contribution on top of the existing model.

Complexity
- Although computationally cheap (utilizing random initialization and DFT), it adds much more complexity to the setup
- For example, on the Wasserstein measurement setup, “To prevent this from affecting the selection process, we measure the Wasserstein distance 10 times and use its mean to select.” I feel this is too much of a hack to be reliably used by other researchers, but I may yield to what other reviewers say.
- Authors could clarify what the parameter values represent in the first section of Experiments for ease of reading.

Impact
- It is stated the method is incompatible for “extremely high frequencies” (i.e. NeRF and Wire), which makes me doubt its generalizability.

### Questions
- Why not just have these parameters be learned rather than grid searched (as I believe omega_0 can be learnable at least, I am not sure about the others)? 
- 15,000 steps seems very excessive to fit a single image, how much wall clock does this take on a gpu for good results? Other methods like instant-ngp and spder (above) are much faster.
- In 5.2 in experiments why is NeRF not used as a baseline also as it is mentioned in the paper? I am not sure which one it is in Table 6.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a method for selecting the hyperparameters for current MLP-based INR models to improve the suboptimal representation performance of models with default hyperparameters. This method measures the Wasserstein distance between frequency distributions of initial-state models and the target signal and selects the hyperparameters that minimize the Wasserstein distance.  This paper validates this method on several current INR models via 2D image approximation, 2D video fitting and neural radiance fields. Experimental results demonstrate that this method is effective to some extent.

### Strengths
1. The method of this paper is fairly easy to understand and flow. This is the first hyperparameter selection method for implicit neural representation models based on the idea of frequency alignment. 
2. The computation cost of this method can be ignored. 
3. Through relatively extensive experiments and the experiments on "decreasing the default embedding frequency" in the supplementary materials, this paper well demonstrates that this method could adjust part hyperparameters to adjust the “preferred spectrum” of MLPs, achieve better representation of target signals.

### Weaknesses
1. The related work of this paper seems overly lengthy and overlooks some of the latest methods for spectral bias. For example, spectral bias can be overcame by special initialization and training dynamics adjustment, such as reparameterization and batch normalization. Moreover, as stated in the paper that Fresh is a simple initialization technique in line 024, Fresh should compare to the previous initialization method like “From Activation to Initialization: Scaling Insights for Optimizing Neural Fields”. 

2. There lacks a reliable theoretical or experimental link between the key observation (line 019) and Fresh (line 241，Table 2).  More concretely, there need an explanation that the adjustment of embedding layer hyperparameters could affect the whole model's spectrum. A more quantitative expression would be better.  The illustration in Fig. 1 seems to be vague. Clarification of this point would certainly raise my opinion of the paper.

3. Some experimental results show that Fresh could not provide a better result even worse than baseline models, such as results with time input in Table 5 and in Table 6. Intuitively, Fresh should not obtain worse results. It might show that there are several unclear relationships. Solving the weakness 2. might help this issue.

4. The experiment in Appendix A seems for demonstrating that weights of embedding layer could not be trained sufficiently by gradient descent. However, why is the optimization algorithm SGD? Meanwhile, is the index L2 norm （Eq. 8） used by the paper (720) meaningful? This experiment should be illustrated more clearly.

### Questions
Typically, spectral bias means that neural networks tend to fit the high-frequency component of the target signal. Therefore, spectral bias can be reflected as different learning speeds for different frequency components. Although the advanced activation functions such as Sine improve the representation performance, MLPs with Sine still first fits the low-frequency parts as shown in some paper such as “Improved implicit neural representation with Fourier reparameterzied Training”. Inspired by this phenomenon, these methods might not change the bias; rather, it broadens the range of frequencies the model can represent. Under this perspective, Fresh might shift some special spectrum distribution that enjoys the fastest learning speeds of MLPs towards the target signals. So the detailed description of this special spectrum distribution or the validation of the existence of this special spectrum might be an interesting problem. This point will not affect my scoring. If authors could try to find this special spectrum and do more exploration about this problem, I think that it will further deepen our understanding of deep learning.

### Soundness
2

### Presentation
2

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
Hyperparameter optimization is a tricky but important problem for implicit neural representations (INRs), since different hyperparameters may be optimal for different signals, but hyperparameter sweeps can be expensive. The authors introduce a fast proxy for determining if an INR’s hyperparameters are well-suited to a particular training signal. Instead of directly optimizing the post-training loss, the authors propose to minimize the Wasserstein distance between frequency distributions of the training signal and the newly initialized INR’s output. This skips the step of actually training each INR, dramatically reducing the time needed to perform a hyperparameter sweep. This proxy appears to perform quite well, conferring most of the performance advantage of a traditional hyperparameter grid search. The authors evaluate their method across a range of image datasets, videos and NeRFs.

### Strengths
Clear, simple, effective solution to a practical problem. Thorough demonstrations that it works. The paper is well-written and well-presented, easy to read and understand. I can imagine this technique being widely adopted for INR hyperparameter optimization.

### Weaknesses
Section 5 contains a few typos:

Section 5, paragraph 2, “a trail and error approach”

Section 5.1: You fell victim to LaTeX’s backwards quotation marks, one of the classic blunders

Otherwise the paper is quite solid.

### Questions
> Similarly, Wire (Saragadam et al., 2023) uses very high frequencies and increases frequencies at each hidden layer, not just at the embedding layer. This makes it incompatible with FreSh

I am curious if you can say more about these models (NeRF and Wire) which use very high frequencies. What purpose do these high frequency components serve if they are not reflected in the training data? Why don’t these models perform poorly as FreSh would predict?

You mention that your FreSh measurements are noisy due to random initialization, so you average across 10 initializations. [1] find that SIREN’s random initialization of the first layer has a significant effect on the final PSNR (Appendix A.6). I would be interested to see if your method could be further improved by using FreSh to select not only the hyperparameters, but also the random initialization to start from.

In the appendix you say that there is high variability in the model configurations selected by FreSh, even within a single dataset. But how much of this variability is random vs. actually important for performance? I would be curious to see how per-image FreSh compares to a dataset-wide FreSh: how much of the advantage of FreSh comes from optimizing for each individual image, and how much can be achieved just by making an appropriate hyperparameter choice for each dataset. 

[1] https://openreview.net/pdf?id=iKPC7N85Pf

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a method for selecting the frequency hyperparameter involved in using periodic activations in implicit neural representations. Selection of the frequency hyperparameter is critical to INR performance, and authors often either select arbitrarily e.g. $\omega=30$, or employ a costly grid-search. The authors propose selecting this hyperparameter based on the target signal frequencies, by minimising the Wasserstein distance between the target signal spectrum and the untrained network output spectrum. This is conducted by searching across candidate frequencies. The authors demonstrate performance improvements across images, neural radiance fields, and videos. Overall this is a nicely written paper which introduces a simple method with interest to the INR community.

### Strengths
There are a number of strengths in this paper: 
- The issue is well-motivated: selection of frequency is a common frustration when using SIREN implicit neural representations. A method for a-priori selection of frequency rather than needing to grid-search across trained networks will be useful for the community. 
- The method is conceptually simple and should be easy to implement / include in INR pipelines meaning it may be broadly applicable.
- Experimental results extensively tested with results averaged across seeds for multiple modalities including image datasets (WikiArt, FFHQ, Chest X-Ray, etc), video, and neural radiance fields. 
- The authors have incorporated their method in multiple pipelines (Siren, Fourier, Hashgrid embeddings, Finer). Results broadly indicate that the method leads to a performance improvements within these pipelines. 
- The experiments are well ablated, with key components (e.g. spectrum size n) evaluated with multiple configurations and datasets
- The paper is well written, logically structured, and clearly presented. Existing literature is reviewed well.

### Weaknesses
There are a few minor weaknesses in the paper. These principally relate to small aspects of presentation (e.g. Figure 2). I've listed these in the Questions below as they are largely minor issues rather than critical weaknesses. 

In terms of technical weaknesses, while the method avoids a costly grid search across trained networks it still requires a grid search across candidate frequencies on untrained networks (this cost will be negligible in comparison). In addition, the method introduces an additional hyperparameter (selection of the spectrum cut-off). If I'm reading it correctly, this hyperparameter would still need a trained network to be evaluated (even though the authors note it transfers across signals). This could reduce some of the benefit of the method, especially if the frequency hyperparameter needs to be searched in combination with the frequency cut-off. Could the authors clarify this issue?

### Questions
- The description of a vector spectrum by summing the DFT appears a little non-standard (L285-7). Could the authors provide a reference for this approach (potentially outside the INR literature?)
- In Appendix C it is mentioned that the for image regression the same image is resampled 10 times due to randomness of the model output. The same network should be deterministic in output - does the randomness occur due to re-sampling the network with reinitialisation? 
- The paper mentions that other activations are addressed L131. Have experiments on non-periodic functions (e.g. Gaussian) been performed? 
- Could the authors provide details on the compute required for Fresh on more complex signals like NeRF (L414-6 notes that this is negligible for images). 
- Table 5 records video results. It mentions 'Results for NeRF are provided for reference only as it is incompatible with FreSh'. Could the authors clarify this?
 - Could the authors clarify what is meant by 'direction-dependent frequency magnitudes' and its importance for video (L510)? 

Minor Presentation: 
- The presentation of Figure 2 could be improved (visually the figure is difficult to interpret without the caption / main text). Tweaking this figure (perhaps additional labeling) would greatly improve the clarity of the method and extend its impact.
- The font in figures 12, 13, 14 is inconsistent 
- The text size in Table 6 is inconsistent / appears to have been scaled 
- Figure 5 - suggest changing 'Thousands to steps' to 'Steps ('000)' 
- Figure 1 could be improved (the PSNR labels are difficult to distinguish)
- Left-hand quotation makes can be done in Latex using the ` character

### Soundness
3

### Presentation
3

### Contribution
3
