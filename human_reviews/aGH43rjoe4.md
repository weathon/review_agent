# Multi-modal Gaussian Process Variational Autoencoders for Neural and Behavioral Data

- Decision: Accept (poster)
- Scores: 8, 3, 8, 5, 5

## Abstract
Characterizing the relationship between neural population activity and behavioral data is a central goal of neuroscience. While latent variable models (LVMs) are successful in describing high-dimensional data, they are typically only designed for a single type of data, making it difficult to identify structure shared across different experimental data modalities. Here, we address this shortcoming by proposing an unsupervised LVM which extracts shared and independent latents for distinct, simultaneously recorded experimental modalities. We do this by combining Gaussian Process Factor Analysis (GPFA), an interpretable LVM for neural spiking data with temporally smooth latent space, with Gaussian Process Variational Autoencoders (GP-VAEs), which similarly use a GP prior to characterize correlations in a latent space, but admit rich expressivity due to a deep neural network mapping to observations. We achieve interpretability in our model by partitioning latent variability into components that are either shared between or independent to each modality. We parameterize the latents of our model in the Fourier domain, and show improved latent identification using this approach over standard GP-VAE methods. We validate our model on simulated multi-modal data consisting of Poisson spike counts and MNIST images that scale and rotate smoothly over time. We show that the multi-modal GP-VAE (MM-GPVAE) is able to not only identify the shared and independent latent structure across modalities accurately, but provides good reconstructions of both images and neural rates on held-out trials. Finally, we demonstrate our framework on two real world multi-modal experimental settings: Drosophila whole-brain calcium imaging alongside tracked limb positions, and Manduca sexta spike train measurements from ten wing muscles as the animal tracks a visual stimulus.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a model for multi-modal data based on a latent space using both shared and mode-specific latent subspaces and Gaussian Process priors. These GP priors are intended to capture smooth variations in the latent space, akin to Gaussian Process Factor Analysis (GPFA) for neural time series data. Another key ingredient is the use of a Fourier basis in the latent space, which allows for a natural low-frequency regularization on latent dynamics. Experiments include both simple synthetic data setups and applications to two real data sets in which neurophysiological recordings are combined with behavior.

### Strengths
- Use of GP priors to smooth latent trajectories is a well-matched strategy for smoothing highly noisy neural data.
- Moving to the Fourier basis is a clever idea that allows for a very natural regularization. 
- The idea of shared and specific latent subspaces has been done before but adds to the interpretability of the model.
- Several tricks are combined here that render the notoriously inefficient GP inference to be performed efficiently.
- Good set of experiments on real neural data.

### Weaknesses
- The synthetic experiments tend to focus on smooth, continuous variation in "incidental" properties (e.g., size and angle of a digit) that may not be a good approximation in many real data sets.
- The GP model may not work well in cases where data do not have a natural trial structure or produce very long behavioral bouts.

### Questions
- As noted above, the GP method would seem to need data to be broken into snippets of reasonable size. For distinct trials, this is pretty obvious, but what if the time series are very long (e.g., natural behavior)? Does it work to simply snip the behavior at random? Does the model handle this? Do all snippets need to be the same size?
- Is it possible to put some shrinkage priors on the $W$ matrices in (4) so that unneeded dimensions are pruned away? Can the model distinguish between "true" and "false" shared latents? Is there some notion of parsimony here?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a variational auto encoder framework to jointly model neural activities and behavior data. Their contribution is to add Gaussian Process Prior and parameterize the latent in Fourier domain.  Their goal is to distinguish the separate or joint latents from multiple modalities, and assume a linear combination of those latents. The work is evaluated on one simulated dataset (MNIST + spike train) and two real-world datasets from different animal species (drosophila, Manduca).

### Strengths
1. The work is well-motivated,  jointly train latent variable models from multi-modal neural data is an important question.
2. The contribution to use Gaussian Process prior to capture correlations in latent space, and parameterize the latent in Fourier space are novel contributions.
3. The work is demonstrated on multiple datasets to improve its soundness, including one synthetic dataset and two real-world datasets from different animal species .

### Weaknesses
Methods:
1. The assumption of linearity between latents in eqn (4) seems over-simplified, other nonlinearity variants could be further investigated. 
2. The construction of simulated dataset is confusing, it is not clear to see how the rotation angle of MNIST image is inherently related to spiking distribution. And it is puzzling that how it is tightly linked with real-world datasets. 
3. The paper proposed parameterize the latents in Fourier space to constrain them more distinguishable, and attempt to prune the high-frequency components. While it didn't provide solid ablation studies and compared to the original temporal space on real datasets, and it is problematic to claim no information preserved in high-frequency domains. 
4. The authors didn't provide clear explanation or guidance about important hyperparameters of latent dimensions. 

Evaluations:
1. No quantitative measurements on real datasets only with a few qualitative samples, and not compared with other baselines.
2. In Fig 4(d), different behavior states are not clearly separable in these subspaces.
3. Blurriness and mismatch of reconstruction outputs and latents in the results.

Writing:
1. The paper is not well organized, missing quantitative evaluations and comparisons with baselines, a few important contents are pointed to supplementary, while it is not clearly described there either.

### Questions
1. Is nonlinearity needed in eqn (4)?
2. How to decide dimensions for separate and joint latent?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a method, MM-GP-VAE, for modeling multimodal data, presented in the context of analyzing neural and behavioral data.  The balance that the proposed method strikes is retaining the interpretability and disentanglement enjoyed by simpler models, but retaining the expressivity and ease-of-use of deep methods.  Low-dimensional latent states describe each channel individually, and a shared low-dimensional state describes the shared covariance.  The recurrent model is implemented as a Fourier-domain GP-VAE, to encourage smoothness in the latent state and increase expressivity.  The model is benchmarked on a synthetic MNIST-like experiment, and is then applied to neural data.  The model appears to perform well against some sensible baselines.

### Strengths
The paper is incredibly well presented.  Figures are prepared exceptionally well.  The prose is clear, and presents a self-contained introduction to all of the necessary techniques and considerations. 

The method itself is gracefully simple.  Although there is not a huge methodological contribution, the correct components parts are assembled and deployed in a way that is very experimentally useful.  I could see this methodology having real uptake within the community, and engendering multiple follow-up works.

### Weaknesses
I do have several queries/concerns however:

- **a. Fixed time horizon**:  The use of an MLP to convert the per-timestep embeddings into per-sequence Fourier coefficients means that you can only consider fixed-length sequences.  This seems to me to be a real limitation, since often neural/behavioral data – especially naturalistic behavior – is not of a fixed length.  This could be remedied by using an RNN or neural process in place of the MLP, so this is not catastrophic as far as I can tell.  However, I at least expect to see this noted as a limitation of the method, and, preferably, substitute in an RNN or neural process for the MLP in one of the examples, just to concretely demonstrate that this is not a fundamental limitation. 

- **b. Hidden hyperparameters and scaling issues**:  Is there a problem if the losses/likelihoods from the channels are “unbalanced”?  E.g. if the behavioral data is 1080p video footage, and you have say 5 EEG channels, then a model with limited capacity may just ignore the EEG data.  This is not mentioned anywhere.  I think this can be hacked by including a $\lambda$ multiplier on the first term of (6) or raising one of the loss terms to some power (under some sensible regularization), trading off the losses incurred by each channel and making sure the model pays attention to all the data.  I am not 100% sure about this though.  Please can the authors comment. 

- **c.  Missing experiments**:  There are a couple of experiments/baselines that I think should be added. 
  - Firstly, in Figure 3, I'd like to see a model that uses the data independently to estimate the latent states and reconstruction.  It seems unfair to compare multimodal methods to methods that use just one channel.  I’m not 100% sure what this would look like, but an acceptable baseline would be averaging the predictions of image-only and neuron-only models (co-trained with this loss).  At least then all models have access to the same data, and it is your novel structure that is increasing the performance.
  - Secondly, I would like to see an experiment sweeping over the number of observed neurons in the MNIST experiment.  If you have just one neuron, then performance of MM-GP-VAE should be basically equivalent to GP-VAE.  If you have 1,000,000 neurons, then you should have near-perfect latent imputations (for a sufficiently large model), which can be attributed solely to the neural module.  This should be a relatively easy experiment to add and is a good sanity check.
  - Finally, and similarly to above, i’d like to see an experiment where the image is occluded (half of the image is randomly blacked out).  This (a) simulates the irregularity that is often present in neural/behavioral data (e.g. keypoint detection failed for some mice in some frames), and (b) would allow us to inspect the long-range “inference” capacity of the model, as opposed to a nearly-supervised reconstruction task.  

  Again, these should be reasonably easy experiments to run.  I’d expect to see all of these experiments included in a final version (unless the authors can convince me otherwise).

- **d.  Slightly lacking analysis**:  This is not a deal-breaker for me, but the analysis of the inferred latents is somewhat lacking.  I’d like to see some more incisive analysis of what the individual and shared features pull out of the data – are there shared latent states that indicate “speed”, or is this confined to the individual behavioral latent?  Could we decode a stimulus type from the continuous latent states?  How does decoding accuracy from each of the three different $z$ terms differ? etc.  I think this sort of analysis is the point of training and deploying models like this, and so I was disappointed to not see any attempt at such an analysis.  This would just help drive home the benefits of the method.  


### Minor weaknesses / typographical errors:

1.  Page 3:  why are $\mu_{\psi}$ and $\sigma_{\psi}^2$ indexed by $\psi$?  These are variational posteriors and are a function of the data; whereas $\psi$ are static model parameters. 

2.  Use \citet{} for textual citations (e.g. “GP-VAE, see (Casale et al., 2018).” ->   “GP-VAE, see Casale et al. (2018).”)

3.  The discussion of existing work is incredibly limited (basically two citations).  There is a plethora of work out there tackling computational ethology/neural data analysis/interpretable methods.  This notable weakens the paper in my opinion, because it paints a bit of an incomplete picture of the field, and actually obfuscates why this method is so appealing!  I expect to see a much more thorough literature review in any final version.

4.  Text in Figure 5 is illegible.  

5.  Only proper nouns should be capitalized (c.f. Pg 2 “Gaussian Process” -> “Gaussian process”), and all proper nouns should be capitalized (c.f. Pg 7 “figure 4(c)”).

6.  Figure 1(a):  Is there are sampling step to obtain $\tilde{\mu}$ and $\tilde{\sigma}^2$?  This sample step should be added, because right now it looks like a deterministic map.  

7.  I think “truncate” is more standard than “prune” for omitting higher-frequency Fourier terms.

8.  I find the use of “A” and “B” very confusing – the fact that A is Behaviour, and B is Neural?  I’m not sure what better terms are.  I would suggest B for Behavioural – and then maybe A for neural?  Or A for (what is currently referred to as) behavioral, but be consistent (sometimes you call it “other”) and refer to it as Auxiliary or Alternative data, and then B is “Brain” data or something.  

9.  The weakest section in terms of writing is Section 3.  The prose in there could do with some tightening.  (It’s not terrible, but it’s not as polished as the rest of the text).

10.  Use backticks for quotes (e.g. ‘behavioral modality’ -> ``behavioral modality’’).

### Questions
I don’t have any further questions from those outlined in Weaknesses.

**Note:** If the authors can allay my concerns, then I fully intend to increase my review score.

### Soundness
2 fair

### Presentation
4 excellent

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
This paper proposes a multi-modal gaussian process VAE model for neuroscientific data. Due to the time-varying nature of neuronal data, the authors use gaussian priors for Fourier frequencies of the data. Thus the network features FFT and IFFT layers. Results are shown on synthetic data as well as on calcium signaling data from Drosophila and Moth EMG data.

### Strengths
The use of Fourier frequencies as latent variables can have some advantages in learning neural features that correlate with motor or other behaviors. The authors also separate shared from independent variability in the neural data which can be useful in understanding what part of behavior could actually be predicted from specific neuronal measurements.

### Weaknesses
-The synthetic data seems weak and unnecessary. Why not use a simulator like NEST or augment one and create "behavioral" data with known connections to this. MNIST images have a different structure because they are a two dimensional image. 
-The authors lack comparison to anything but an ablation of their own model, and that too on the above artificial dataset. 
-At least the authors should compare to RNN/transformers/ODE or other sequential models for this data
-The authors seem to criticize DNNs for transforming the data too much such that the latent variables are "not obviously related" to external variables, and yet they too use DNNs. 
-There is no theory or study of identifiability of the dynamics or conditions/noise under which dynamics are recovered.

### Questions
The synthetic data does not make sense to me, what exactly is the connection between the digits and the "spikes"?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors proposed a multi-modal Fourier-domain Gaussian Process variational autoencoder (MM-GPVAE), aiming at jointly modeling observed neural and behavioral data in neuroscience applications with both shared and independent latent subspaces. MM-GPVAE combines the ideas in previous GP-VAE, GP Factor Analysis (GPFA), and spectral representation. The authors have also presented simulations and two neuroscience case studies showing the effectiveness of MM-GPVAE modeling neural measurement / spiking data with either visual stimulus or position data.

### Strengths
1. This submission exploits the multi-modal latent variable models based on GP-VAE, using Poisson and normal likelihoods for spiking and image data respectively, which can help modeling both shared and independent factors across modalities in neuroscience. 

2. Experiments on simulated and real-world datasets with visualized results showing the potential in jointly analyzing neuroscience data. 

3. The presentation is clear with articulated motivations of the presented mm-GPVAE. Source code is also provided.

### Weaknesses
1. Methodological contributions may be limited. Many papers have done similar things, such as extending the latent space into frequency domain [1,3] and using the Poisson model for spiking data [1,2]. The method proposed in this paper is combination of existing methods, not fundamentally different. 

2. The experimental results can be more comprehensive: (1) More baselines, especially similar methods such as [1,4] need to be compared; (2) The authors stated one main advantage of the Fourier domain is that high frequencies can be pruned and thus we can sparsify the variational parameters. However, no results are provided about this point; (3) An ablation study between the models with Frequency prior and with other priors (e.g., a temporal form) should be added; (4) More discussions on the claimed interpretability should be provided. 

[1] Keeley S, Aoi M, Yu Y, et al. Identifying signal and noise structure in neural population activity with Gaussian process factor models. Advances in neural information processing systems, 2020, 33: 13795-13805.

[2] Duncker L, Sahani M. Temporal alignment and latent Gaussian process factor inference in population spike trains. Advances in neural information processing systems, 2018, 31.

[3] Hensman J, Durrande N, Solin A. Variational Fourier Features for Gaussian Processes. J. Mach. Learn. Res., 2017, 18(1): 5537-5588.

[4] Pearce M. The Gaussian Process prior VAE for interpretable latent dynamics from pixels. Symposium on advances in approximate Bayesian inference. PMLR, 2020: 1-12.

### Questions
1. How sensitive is the performance with respect to the setting of several hyperparameters, including $F$ and the dimension of shared and independent latent subspace, etc.? 

2. How optimization was done with FFT/iFFT involved in MM-GPVAE?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
