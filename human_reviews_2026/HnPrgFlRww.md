# Disentangling Behaviorally Relevant Latent Dynamics in Multimodal Neural Time-series

- Decision: Reject
- Scores: 6, 8, 6, 0

## Abstract
Multimodal neural data can enable a more complete understanding of brain dynamics underlying behavior. Modalities such as neuronal spiking activity, local field potentials (LFPs), and behavioral signals capture diverse spatiotemporal aspects of brain processes. By leveraging these complementary strengths, multimodal neural fusion can provide a unified, rich representation of brain-behavior processes and address the limitations of single-modality analyses, such as incomplete or noisy data. While recent works have jointly modeled behavior and neural data to disentangle sources of variability, they largely rely on latent variable models that use a single modality of neural data. Here we develop a nonlinear dynamical model, termed BREM-NET, that integrates behavioral signals and multiple neural modalities—such as LFPs and spike counts—with distinct statistical characteristics and temporal resolutions into a unified framework, and performs multimodal neural fusion during inference. In two independent public multimodal neural datasets, we show that BREM-NET nonlinearly fuses information across neural modalities while also disentangling behaviorally relevant multimodal neural dynamics. Doing so results in inferring more accurate disentangled latent dynamics, as reflected in enhanced behavior decoding and neural prediction compared to multimodal baselines. Furthermore, BREM-NET enables disentanglement and multimodal fusion even when different neural time-series modalities are asynchronous and have distinct temporal resolutions, which is a major challenge in real-world neural recordings. This framework provides a new tool for studying behaviorally relevant neural computations across different spatiotemporal scales of brain activity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present BREM-NET, a multi-modal approach for modeling neural and behavioral activity. The approach additionally provides the ability to disentangle behaviorally-relevant neural dynamics from behavior-irrelevant dynamics. BREM-NET is applied to two datasets from monkey motor areas during different tasks, and benchmarked against several recent baselines.

### Strengths
BREM-NET is a clear, straightforward approach to nonlinearly modeling multi-modal neuro-behavioral datasets. It tackles a critical need for neuroscience with strong results on some preliminary datasets. Futhermore, the authors provide adequate benchmarking against existing models, and perform a thorough series of ablations to study the contributions from different model components. I would be very curious to see how this method applies to a growing number of neuro-behavioral mouse datasets, but the current work is strong enough as a proof of concept.

### Weaknesses
The paper is clearly written, except I found myself confused a few times in section 3 and figure 1. Figure 1, for example, contains a lot of information - not only the model architecture, but the three stage training approach. Somehow expanding the figure or otherwise rearranging the components or annotations could help readers understand these subtleties better.

### Questions
related work to consider including:
- Whiteway et al. Plos Comp Bio 2021: "Partitioning variability in animal behavioral videos using semi-supervised variational autoencoders"
- Wang et al. Neurips 2024: "Exploring Behavior-Relevant and Disentangled Neural Dynamics with Generative Diffusion Models"
- Shulz et al. Cell Reports 2025: "Modeling conditional distributions of neural and behavioral data with masked variational autoencoders"
- Zhang et al. ICML 2025: "Neural Encoding and Decoding at Scale"
- Wu et al. ICLR 2025: "Disentangling 3D Animal Pose Dynamics with Scrubbed Conditional Latent Variables"

L139: what does T ⊆ K mean? T and K are both integers. Does this mean that each value of t is also a value of k?

The definition of Z in L143 indicates the behavior must be sampled at the same time as the faster of the neural modalities. Is this in fact true? This seems like a strong limitation if so. Behavior may unfold much more slowly than the fastest neural modality, and behavior might be captured by, say, a video camera with its own particular framerate which cannot be synced to the neural recording apparatus (regardless of sampling rate).

In Eq 2, if one modality is unavailable at a given time step, the encoder output is set to zero. This seems suboptimal, because simply using zero is very different from properly indicating missing data. I would imagine you could potentially get much better performance by just linearly interpolating the lower-resolution signal. However, Table A.6 shows otherwise. Why does zero-replacement work?

Is Eq 2 a more detailed description of x_{k+1} = F(x_{k}) in Eq 1? If not, how does A, Enc_y, Enc_s relate to F?

I had to read section 3.2 several times before truly understanding the different stages. The final paragraph of this section is actually very helpful, and might better serve the reader if placed in the first paragraph of 3.2.

Why is \lambda_k in Eq 3 a product of C^(1) and C^(2) and not their sum?

BREM-NET results in Table A.4 are not the same as those in Table 1 - why the discrepancy?

Table 1: I'm not sure how fair some of these comparisons are:
- MVAE doesn't have access to temporal data; while this is how the method works, it seems like it could also be used in a more "temporal" mode where windows of data rather than single time points are used as training examples
- CEBRA: only using a linear mapping from the latents to the behavior/neural activity isn't a fair comparison to the BREM-NET model that uses nonlinear mappings. A interesting control would be nonlinear encoders and linear decoders for BREM-NET, which would then properly directly compare the latents learned by those two models.

I think extending BREM-NET to study shared latent dynamics across multiple brains is an extremely exciting future direction.

For another future direction, I'm curious what the authors have to say about using this model for studying multi-region interactions; for example, finding neural activity that is shared across regions versus private to each region.

4.1.1, second paragraph: decoding behavior separately from x^1 and x^2 is a good way to assess disentanglement. The decoding CC for x^2, 0.3367, is higher than I expected. an the authors speculate why this is not closer to zero? Does it have to do with the power of C_z? Presumably the more powerful the behavior decoder, the less residual behavioral information remains in x^2. If you replaced x^2 with the ground truth behavior would the decoding CC for x^2 be closer to zero? Might be an interesting control.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a nonlinear dynamical model, BREM-NET, that utilizes multiple neural modalities with behavioral data into a unified framework. The model can disentangle the latents into behaviorally relevant and neural-specific dynamics in multimodal neural time-series, thus making BREM-NET a more interpretable framework. The authors demonstrate the utility of BREM-NET in two multi-modal datasets.

### Strengths
- BREM-NET addresses an important issue of handling data that are not in sync, which is often the case for neuroscientific data. 

- The literature review was very thorough, and the model comparisons were with some of the most relevant and novel models, which made the results even more compelling to me.

- The presentation was clear.

### Weaknesses
Although promising, I’m a little concerned with the optimization here, and its ability to correctly identify the disentangled latents. For example, for missing data, the authors state that the corresponding encoder output can be set to zero, but the different modalities can have different/finer scales so wouldn’t this cause issues for decoding? Also, are there any explicit constraints that would enforce private/shared latents in the model formulation other than the multi-step training procedure ?

### Questions
- Have the authors considered another more principled method of picking the shared latent dimensionality (along the lines of Giaffar et al., 2024)  as a starting point?

- How informative are the included latents? (e.g. the authors state that they fixed the total latent dimensionality for the experiments, but of these 64 total latents, what was the amount of information explained by each latent?)

- How does this compare to a model like BRAID (Vahidi et al., 2025), which also uses nonlinear dynamical modeling to  disentangle the shared dynamics between modalities?


Giaffar, H., Buxó, C. R., & Aoi, M. (2024, April). The Effective Number of Shared Dimensions Between Paired Datasets. In International Conference on Artificial Intelligence and Statistics(pp. 4249-4257). PMLR

Vahidi, P., Sani, O. G., & Shanechi, M. M. (2025). BRAID: input-driven nonlinear dynamical modeling of neural-behavioral data. arXiv preprint arXiv:2509.18627.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Neural modalities have behavioral and neural dynamics. Disentangling these 2 types of dynamics is important for neural behavior relationships. While previous work has shown that disentangling behaviourally relevant neural dynamics enables better and more interpretable analysis of how neural activity relates to behavior, these methods are constrained to using a single neural modality like spikes alone or LFP alone. Therefore, the challenge is to capture both the fine scale of individual spikes and the larger network level scale of LFP. 

The authors introduce BREM-NET: a non-linear dynamical model that can model how brain and behavior signals evolve over time, capturing complex nonlinear relationships. It can combine different kinds of brain and behavior data, handle different distributions and work with missing chunks of data too. 

The model has 3 stages:
Stage 1 encodes 2 separate neural modalities, through 2 separate encoders (Enc_1 and Enc_2) and fuses them with a linear layer (concat the encodings and apply a linear projection). The output from this linear layer is then processed through an RNN that produces a latent state x_k1 which represents a time-evolving summary of all neural modalities up to time k. 

The behavioural decoder (C) takes the latent signal x_k1 and passes it through a decoder to get the behaviour z_k. The loss function is the negative log likelihood (NLL) to predict behavior from behaviorally relevant states x_k1. 

Stage 2 has 2 decoders specific to modality. Their task is to reconstruct the original data (spikes or LFP). NLL is used here too. 

Stage 3, similar to stage 1, takes in the raw data + the learnt latent x_k1. An RNN learns a new set of latents x_k2. Stage 3 learning is unsupervised. 

Monkey is doing a 2d cursor moving task. behavior - velocity of the cursor and neural data LFP. It merges the 2 modalities and tries to find the LFP data that is ‘causing’ the velocity data to change. Once that's done, it disentangles the remaining neural data that's not causing the behavior to change. 

Result: the model can separate out the neural signals that cause the behavior better than base models. Behavior decoding is the most impressive jump. Other LFPcc and SpikePP are not very impressive but are at par with baseline.

### Strengths
The authors offer a novel method for optimally integrating information from multiple modalities. This has the potential to be valuable to the community, as simultaneous acquisition of data from multiple modalities, particularly single cell electrophysiology and LFP, is very common.

### Weaknesses
Although the summary tables benchmarking the new method against previousl methods shows impresive SOTA performance, it would help greatly to have more visuals and analysis comparing both when and why the new method performs better. For example, the only figure I see that shows this comparison is A4, wherein mmPLRNN looks like it's performing as well as the new method.

### Questions
1. In what way does the new method outperform old methods? In addition to the summary performance tables, please include plots that demonstrate where previous methods fail and this one succeeds.
2. In stage 3 of the model, it is not clear what is truly blind to the behavior - x_k1 comes from behavior, but the authors claim this stage is blind to behavior.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper presents a model meant to disentangle neural activity and behavior using a model they cal BREM-NET. The feature that distinguishes their model for existing models is that it utilizes multiple neural modalities simultaneously while extracting latent variability that explains the behavior. The authors conduct a wide array of simulated experiments and real data validations.

### Strengths
The paper is mostly clearly written (with some exceptions that I point out below) and easy to follow. The benchmarking results look strong compared to competing methods but it is unclear what the value of this method is.

### Weaknesses
The primary, and unfortunately fatal, weakness of the paper is that the model does not disentangle latent variability. This is evident from both their model structure and their benchmarks. First, the behavioral latents are learned exclusively by supervised learning, making stage 1 of the model training essentially like training a nonlinear regression model with a low-dimensional bottleneck. However, the original neural activity AND the behavioral latents are mixed to learn the neural latents. So the neural latents cannot be regarded as residual variability that does not include behavioral information, i.e. neural latents are not disentangled from the behavioral latents. Indeed the behavioral latents are learned with no knowledge of the neural latents whatsoever. Moreover, in Section 4.1.2 (lines 401-402) the authors point out that U-BREM-NET matched BREM-NET on neural prediction, indicating a questionable role for $x^{(2)}_k$ beyond what U-BREM-NET could have provided. In essence, there is no disentanglement beyond what could have been achieved by Stage 1 alone.

Since the authors seem to couch the value of this model on its ability to "disentangle" as the title suggests, I cannot support acceptance of this submission.

### Questions
- The benchmarking procedures were not clear. Specifically, It is only in the appendix that it becomes clear that the authors were doing 1-step forward prediction to benchmark against competing baselines. However, for all but one of the models it is unclear how one-step forward prediction was conducted. It is clear for MMPLRNN, but for MMGPVAE the authors simply state "we obtained one-step-ahead latent states" but since MMGPVAE is not an explicit model of dynamics it is not clear precisely what they did to make such a prediction. For other models one-step-ahead prediction is not even mentioned.

### Soundness
1

### Presentation
3

### Contribution
1
