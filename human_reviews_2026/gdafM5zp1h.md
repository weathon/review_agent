# TAVAE: A VAE with Adaptable Priors Explains Contextual Modulation in the Visual Cortex

- Avg Score: 3.33
- Decision: Accept (Poster)
- Scores: 6, 2, 2

## Abstract
The brain interprets visual information through learned regularities, a computation formalized as performing probabilistic inference under a prior. The visual cortex establishes priors for this inference, 
some of which are delivered through widely established top-down connections that inform low-level cortices about statistics represented at higher levels in the cortical hierarchy. 
While evidence supports that adaptation leads to priors reflecting the structure of natural images, it remains unclear if similar priors can be flexibly acquired when learning a specific task. 
To investigate this, we built a generative model of V1 that we optimized for performing a simple discrimination task and analyzed it along with large scale recordings from mice performing an analogous task.
In line with recent successful approaches, we assumed that neuronal activity in V1 can be identified with latent posteriors in the generative model, providing an opportunity to investigate the contributions of task-related priors to neuronal responses. To obtain a flexible test bed for this analysis, we extended the VAE formalism so that a task can be flexibly and data-efficiently acquired by reusing previously learned representations.
Task-specific priors learned by this Task-Amortized VAE were used to investigate biases in mice and model when presenting stimuli that violated the trained task statistics. 
Mismatch between learned task statistics and incoming sensory evidence showed signatures of uncertainty in stimulus category in the  posterior of TAVAE, reflecting properties of bimodal response profile in V1 recordings.
The task-optimized generative model could account for various characteristics of V1 population activity, including within-day updates to the population responses. Our results confirm that flexible task-specific contextual priors can be learned on-demand by the visual system and can be deployed as early as the entry level of the visual cortex.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper suggests a modified image autoencoder to account on task adaptations in visual cortex. 
They have a two-stage procedure when first the latent representations are trained for natural images and then they are adjusted with respect to the task prior (which is optimized).
In this particular paper they use mice performing go/no go tasks and show that the latent space of the visual autoencoder after introducing a task prior shows same qualitative phenomena as the actual responses in V1 when the task is changing, supporting the claim that the brain performs a probabilistic inference under a prior.

### Strengths
1. **Elegant framework**. Beautiful idea - fixing the likelihood $p(x|z)$ and only learning a new task-specific prior $p_T(z)$ is both elegant and powerful.  The paper makes a clear hypothesis: systematic biases in V1 during task performance are the result of probabilistic inference under a learned, task-specific contextual prior. The model provides a concrete implementation of this hypothesis and generates specific, falsifiable predictions that are then confirmed by the experimental data.
2. **Qualitative comparisons**. The model reproduces the qualitative phenomena, eg splitting the distribution from unimodal to bi-modal when there is a mismatch between the train and test data (e.g. Fig 3)
3. **Reproducibility**.  The code is provided in the supplementary materials.
4. **Statistical rigor**. All the plots and tables report error-bars.

### Weaknesses
1. **Clarity**. The paper might benefit from a more clear high-level framework introduction, before getting to the formalism. If I get it correctly, then the autoencoder model is trained on images only and neural responses are used for validation only.  
2. **Lack of quantification of qualitative results**. While Fig 3 generates nice qualitative insights, some statistical tests might support the claims, eg Hartigan's Dip Test to quantify when the red line stops being unimodal (and if it happens faster in real mice or in the model), one-Sample t-test to test that the peaks of the response are significantly shifted away from the actual stimulus orientation, and some pearson correlation to check how good the model predictions fit for the actual neuronal responses.
3. **Representations alignment is not considered**. Lines 212-214 make an assumption that the autoencoder latent space $z$ is assumed to correspond to neural activities in V1, however, this correspondence is clearly violated by the fact that $z$ could be negative. Hence, this raises questions about the validity of this assumption and how aligned the representations are in general. 
4. **Limited direct applicability**. While this is a beautiful hypothesis testing framework, applying it for different stimuli can be very complicated. Specifically, eq (11) is nice and tractable as *"in a typical gratings dataset we expect a symmetry in z around zero"* (247-252). However, it is not that clear how to set up a meaningful prior in case of other tasks and out-of-distribution designs  (like distinguishing images by colors, primarily direction of "random" moving dot stimuli, etc)

Minor:
1. Inconsistent font sizes in the plots (see Fig 1 panel D and H, or Fig 2 panel A and E).

### Questions
1. If I get it correctly, then the autoencoder model is trained on images only and neural responses are used for validation only. Is it right? Also, you first train an autoencoder using eq (9) as the loss function to get $q(z|x)$ and $p(x|z)$ and then you only train $\underline{\sigma}_{T}$  (line 253) ?

And it adjusts $q(z|x)$ to $q_{T}(z|x)$ ? Are there any other parts retrained?

2. Why the trained baseline activity in Fig2 H is negative? I though you are taking the absolute values (lines 237-240)
3. Why exactly  does Laplace prior give us localized, oriented receptive fields? (lines 226-228)
4. Lines 237-240 identify that $z$ could be negative, which clearly misaligns the autoencoder latent space with the neuronal responses. Have you tried to restrict the $z$ to be strictly non-negative during training?
5. Connected to the previous question - lines 212-214  say that *"activations of latent variables, z, of the generative model were assumed to correspond to activations of individual neurons in V1"*. How adequate is this assumption? Have you tried to regress learn linear regression from $z$ to the actual neuronal responses (like "neural predictivity" in [1] ) and see how well it performs or do something like CKA analysis [2-4]?
6. How exactly was the neuronal data pooled across sessions? Have you selected the neurons which were enough orientation selective and then just averaged them across all sessions to make the lines in Fig 2 e, f for example? And for the autoencoder - you always used a single model (e.g. there were no several autoencoders to match the latent space for the number of neurons per session)?
7. I would appreciate your thoughts on weakness 4.

Minor:
1. What are blue and green lines in Fig1 h?

References:  
[1] Nayebi, Aran, et al. "Mouse visual cortex as a limited resource system that self-learns an ecologically-general representation." PLOS Computational Biology 19.10 (2023): e1011506.  
[2] Murphy, Alex, Joel Zylberberg, and Alona Fyshe. "Correcting biased centered kernel alignment measures in biological and artificial neural networks." arXiv preprint arXiv:2405.01012 (2024).  
[3] Williams, Alex H., et al. "Generalized shape metrics on neural representations." Advances in neural information processing systems 34 (2021): 4738-4750.  
[4] Chun, Chanwoo, et al. "Estimating Neural Representation Alignment from Sparsely Sampled Inputs and Features." arXiv preprint arXiv:2502.15104 (2025).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes TAVAE, a task-adapted VAE framework that modifies only the latent prior (not the encoder or decoder) to account for contextual modulation effects observed in mouse V1 during a visual discrimination task. By adapting the prior learned from natural images to task-specific contingencies., the model reproduces several well-known effects: sharpening, baseline suppression, and multimodal responses under stimulus-prior mismatch. The VAE is strongly constrained: linear decoder, Laplace prior, and overcomplete latent space, mirroring classic sparse-coding models rather than deep nonlinear architectures.

### Strengths
1. The model is a minimal model with biologically inspired constraints—linear decoder, sparse Laplace prior, overcomplete latent space, and GSM-style gain modulation while it mirrors classic models of V1 (e.g., Olshausen & Field).

2. The model qualitatively reproduces several experimentally observed phenomena using a single mechanism (prior variance reweighting).

### Weaknesses
1. While the paper claims that adaptation in the prior alone is sufficient to account for several task-induced changes in neural population statistics. The lack of comparison to single neuron activity left this claim speculative 

2. Figure 3a: I really cannot see "drastic" difference between red and blue curves. There needs to be a metric or something to quantify how they are different. 

3. Figure 4a; The curves are visually nearly identical in shape, except for slightly lower side peaks and a slightly higher center as γ increases. If all that happens is one peak increases slightly, calling it “updating the inference toward the new context” feels like a strong claim for a weak effect.

### Questions
1. Is it possible to extend this model to decode neural activity? like Maheswaranathan Neuron 2023? 

2. The encoder is linear, with overcomplete latent dimensions, and trained under a Laplace prior. How close is it to ICA or sparse coding rather than deep encoder following by variational sampling? 

3. Would you expect the same latent prior adaptation mechanism to work in tasks involving richer stimuli or additional visual features (e.g., natural scenes/motion)? Why or why not?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors present a VAE framework that explicitly describes a task-dependent prior and test it's performance along side a neural data from mouse V1. The authors present their model, describe data collection, and present qualitative similarities between the activations of latent variables in their model to the spiking activity of V1 neurons measured by calcium imaging.

### Strengths
The paper presents an interesting and, to my knowledge, novel accounting of neural tuning properties in the face of changing stimulus statistics using the model they present in Section 2. They present this along side an approach for learning context specific priors in the variational framework. There do appear to be some qualitative similarities between neural data and model latents but validating these results is required before claims can be made about how their model maps mechanistically onto neural representations of stimuli.

### Weaknesses
I think there are 2 main dimensions on which this paper falls short of acceptance 1) validation of model structure, 2) statistical rigor, 3) clarity of question.

1) The authors make claims about the qualitative properties of the latents of their model and how they match those of the real data. However, I'm not sure it's possible to attribute these features (even if they are statistically valid) to the prior structure of the model exclusively. Specifically, no ablation analysis of the model was conducted to determine which of their modeling choices was essential to their findings. For example, how critical was it that the latent responses were sparse? How important was the scaling latent? Neither of these choices were evaluated in any way and it is not clear they are germane to the properties they intend to model. 

2) There is virtually no statistical analysis beyond Figure 2. Error bars and shaded regions around population tuning curves are not defined. Data points in tuning curve plots (eg. red and blue dots in Figure 2a,b) are not defined. Moreover, if these really are data points and the shaded regions are supposed to be 95% confidence intervals then I suspect their inference is over-confident. 

3) It's not obvious what the authors are testing when they are examining neural activity along side latent activations. This seems to be an unreasonably course level of analysis and I would not expect a clear correspondence to exist beyond something incidental. Perhaps the authors meant to examine the posterior distribution over the stimulus? This would have real cognitive meaning in the context of a shift in prior probabilities.

### Questions
The authors should clarify their mechanistic claims about why their model matches the data in the ways they claim, the modeling choices, statistical inference, and all claims should be accompanied by statistical tests.

### Soundness
2

### Presentation
2

### Contribution
2
