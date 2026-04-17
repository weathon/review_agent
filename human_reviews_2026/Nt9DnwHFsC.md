# Unconditional CNN denoisers contain sparse semantic representations of images

- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Generative diffusion models learn probability densities over diverse image datasets by estimating the score with a neural network trained to remove noise. Despite their remarkable success in generating high-quality images, the internal mechanisms of the underlying score networks are not well understood. Here, we aim to understand the image representation that arises from score estimation objective. Unlike the vast majority of prior work, we focus on an *unconditionally* trained, fully-convolutional *UNet*, o isolate the role of the objective and architecture. 
We show that the middle block of the UNet decomposes individual images into sparse subsets of active channels, and that the vector of spatial averages of these channels can provide a nonlinear representation of the underlying clean images. Euclidean distances in this representation space are semantically meaningful, even though no conditioning information is provided during training. We develop a novel algorithm for stochastic reconstruction of images conditioned on this representation: The synthesis using the unconditional model is "self-guided" by the representation extracted from that very same model. For a given representation, the common patterns in the set of reconstructed samples reveal the features captured in the middle block of the UNet. Together, these results show, for the first time, that a measure of semantic similarity emerges, *unsupervised*, solely from the denoising objective.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the representational properties of the channel-wise average vector extracted from the middle block of a denoising U-Net (or a score U-Net, following Tweedie’s formula). Extensive experiments demonstrate that this vector encodes perceptually meaningful image representations, which can be effectively leveraged for clustering. Finally, the Euclidean distances between these representations are exploited within a diffusion-based algorithm to generate variants of a conditioning image.

### Strengths
1. The intuition behind this paper draws inspiration from classical denoising techniques (such as DCT or wavelet denoising), where noisy signals are decomposed into concentrated (sparse) and distributed (dense) components. 
2. The paper is well written and easy to follow.
3. Several representational properties of a denoising network were highlighted, including increase of sparsity through the channels, robustness of representation to noise, channel selectivity and clustering.

### Weaknesses
My overall impression is that, although the proposed algorithm is effective to generate variants of a conditionner image, the article may over-estimate the role played by the features of a denoising network. I recommend additional experiments to clarify if the effectiveness of the algorithm is dependent on it or if the "guidance step" could leverage a VGG-based perceptual loss instead (see questions below). 

Here are some additional weaknesses:
1. The conclusions are drawn from only one type of denoising architecture, namely UNet (Ronneberger et al., 2015).
2. The use of the normalized participation ratio for measuring sparsity should be better justified.
3. The rational of Algorithm 1 should be better explained. For example, line 963, it is mentionned that the sampling algorithm is based on Kadkhodaie and Simoncelli, 2021 but without any explanations. 
4. No code was provided.
5. DCT denoising reference [a]  is missing on line 146. t-SNE reference as well.
6. Line 245, PCA analysis is not provided.

[a] G. Yu and G. Sapiro. DCT image denoising: a simple and effective image denoising algorithm. Image Processing On Line, 2011.

### Questions
1. Can you draw the same conclusions (increase of sparsity through the channels, robustness of representation to noise, channel selectivity, clusters of images, etc) with an **untrained** UNet? What about other types of architecture such as DnCNN [b]? Or even networks trained for classification?

2. Is the paragraph on translation equivariance of UNets really useful? (Line 100 to 132). I do not understand what you are getting at. Would the analysis be that different with a true translation equivariant model (e.g., DnCNN [b] with circular padding). By the way, can you observe the same increase in sparsity throughout the 17 layers of DnCNN [b]?

3. What happens if only the "distance between representations" is minimized (for initialization we can consider a noisy version of the conditionner image)? (The algorithm would simply consist in repeating the "Guidance step" until convergence).

4. What happens if the "distance between representations" is replaced by a VGG-based perceptual loss [c] in your algorithm?

5. In Figure 3, what does it mean mathematically to "maximally activate" a channel? By the way, are the images in the panels you show in Figure 3 and 4 ranked by their degree of excitation, whatever that means?

6. Which loss was used exactly to train your networks? Is it the one from (1) which minimizes the MSE between the output of the network and the corresponding pure noise?

[b] K. Zhang et al. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE TIP, 2017.

[c] J. Johnson et al. Perceptual losses for real-time style transfer and super-resolution. ECCV, 2016.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the semantic latent space implicitly learned by denosing UNets trained for unconditional image generation using a Diffusion Model objective. First, the authors hypothesize that specific layers of UNet learn an explicit separation in feature space between the signal (image) and additive noise. The authors suggest a link between the sparsity in the input/output feature maps of each UNet block, quantified by the Participation Ratio (PR), and the denoising effect. Using this measure, the authors identify the feature map $\overline{a}\_4$, corresponding to the central block of Unet, as primarily responsible for denoising. Later $\phi(x\_\sigma)$, which appears to be defined as the mean of featuremap $\overline{a}\_4$ observed under different realizations of additive noise $\sigma$, is considered as a latent semantic representation of the input. In section 3, the authors propose a conditional generation strategy by alternating the DMP denoising step and a guidance step defined by minimizing $\phi(x\_\sigma)$ to the $\phi$ representation of a guidance image $x\_c$.

### Strengths
The work suggests an interesting perspective on the latent space structure of generative diffusion models. Analyzing feature-level implications of the denoising steps is a plausible direction to build a deeper understanding of the generative models low-level workings.

### Weaknesses
There are several clear limitations to this work.
First, the work is not properly framed: several prominent works have previously explored related concepts (Preechakul 2022, Kwon 2023, Wpstein 2023).

There are several scientific limitations to the proposed approach. Most of the contributions presented are neither linked to previous contributions nor supported by theoretical derivation nor sufficiently supported empirically by extensive experimental results. 

This includes:

* The connection between representation sparsity and denoising. This connection is unclear and appears to be severely under-justified, and mainly accepted by assumption (ll 152).
* The hypothesis of the connection between sparsity reduction and noise suppression (l188-l197) is purely speculative.
* Similarly, the derivation of $\phi(x\_{\sigma})$ seems arbitrary and the notation is unclear. Is $\overline{a}\_4$ a function? Is $x$ the input of the UNet or of an intermediate layer? Is the noise realization added to the input of the denoiser (as in a typical diffusion model) or to the input of an intermediate layer?
* Conditional sampling: the concept of conditional sampling from Diffusion generative models trained unconditionally has been extensively investigated in previous works (Graikos et. al). Notably, such approaches suffer from off-manifold updates with respect to the Gaussian manifold of the latent distribution induced by the diffusion probabilistic objective.

Additional critical issues:
* The discussion of this work is based on image denoising concepts (sec 2.2, 2.3). This is not a problem in itself, but it creates confusion since it is not generally specified how these concepts relate to the generative diffusion framework. While it is true that the optimization goal of diffusion is denoising, the underlying probabilistic formulation cannot be ignored.

* Experimental analysis: Apart from a few qualitative observations, there is essentially no rigorous experimental analysis of the results obtained. This is particularly limiting given that the proposed work is mainly empirical. We strongly suggest that authors quantify their results (e.g., conditional generation) against benchmarks commonly used in similar works.

* Experimental setup: The description of the experimental setup used is extremely limited and unclear. A.3 (l652) does not make clear what the training objective is (diffusion, presumably?), and the constant references to a denoising model do not help to clarify matters.


References (non-exhaustive):\
Preechakul et. al, 2022, Diffusion Autoencoders: Toward a Meaningful and Decodable Representation\
Kwon et. al, 2023, DIFFUSION MODELS ALREADY HAVE A SEMANTIC LATENT SPACE\
Wpstein et. al, 2023, Diffusion Self-Guidance for Controllable Image Generation\
Graikos et. al, Conditional Generation from Unconditional Diffusion Models using Denoiser Representations\
He et. al, 2024, MANIFOLD PRESERVING GUIDED DIFFUSION

### Questions
* How was feature map sparsity, and consequently PR, chosen as a measure of the denoising effect? Are there any contributions in the field of deep learning to support this choice?

* How is the "mean of $\overline{a}$ across noise realizations" (l204) calculated in the definition of $\phi(x\_{\sigma})$?

* Were alternatives to $\phi(x\_{\sigma})$ evaluated in algorithm 2? (even just the raw features).

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper studies internal representations in diffusion models by examining spatially averaged activations in the UNet denoiser. The authors argue that these activations form a sparse latent code that captures high-level structural information, and they demonstrate that manipulating this code enables controllable sampling. The empirical exploration is interesting and the presentation is clear, with visualizations that support the central narrative.
I find the direction compelling. Understanding what diffusion models encode internally is an important challenge and this paper takes a meaningful step toward that goal.

### Strengths
* Innovative perspective on internal representations in diffusion denoisers.
* Well structured and clearly written.
* Intuitive visualizations that help explain the emergence of latent structure.
* Creative approach to conditioning sampling through internal activations.

### Weaknesses
I enjoyed the paper and I am genuinely interested in the mechanism behind the observations. I want to raise an alternative interpretation that might be worth discussing, though I am not fully confident in it and would appreciate the authors' take.
My current intuition is that some of the observed structure could also arise from well-known properties of convolutional UNet architectures rather than being specific to diffusion learning. In particular:
* UNets naturally encourage preservation of coarse structural information (“scene gist”) due to local receptive fields and hierarchical feature aggregation (early convolutional layers tend to capture low-level edges and textures, while deeper layers integrate spatial context, producing global shape descriptors) and skip connections (the encoder-decoder structure with long skip connections allows high-resolution spatial structure to flow directly to the decoder, which biases the network toward retaining global geometric configuration even under noise). This composition is known to support coarse scene layout understanding in CNNs even without explicitly modeling semantic content. As a result, the emergence of “gist-like” representations, large-scale silhouettes, dominant contours, and coarse texture organization, might be a natural consequence of convolutional structure rather than specifically of the diffusion learning objective.

* Deeper layers in UNets typically have more channels. Averaging many activation channels can reduce variance roughly as $\frac{\sigma^2}{n}$. The observation that the deeper block has lower average activation magnitude might therefore reflect channel scaling in the architecture rather than semantic sparsity created by diffusion training.

Again, my intention is not to undermine the paper. I am simply curious whether the authors believe these architectural factors alone could explain the results, or whether diffusion training plays a more essential role. A comparison against a supervised denoiser or a randomly initialized UNet could clarify this point.

### Questions
Already addressed in the "Weakness" section.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
- The main finding of this paper is that unconditional denoiser models (U-Net type), when trained on image datasets, exhibit at the top middle block a certain level of semantic understanding such as structure, pose, shape, texture, and color, although their human-level semantic class understanding remains limited.
- The paper motivates this finding from the perspective of a denoiser that learns to separate signal from noise (analogous to a Wiener process). This leads to their proposed sparsity measure, which can be used to characterize each learned feature at the population level.
    - The participation ratio (PR) is introduced to measure how "sparse" is a spatially averaged feature of the U-Net.
        - High participation => dense => the latent lives in a high dimensional space. 
        - Low participation => sparse => the latent lives in a low dimensional space. 
        - Naturally, noise is high dimensional and image manifold / semantic information is low dimensional. 
    - The paper shows that the U-Net encoder layers might be recognizing the noise as measured by the increased PR from their output compared to input. However, almost always at the middle block, the PR is decreased from their output compared to the input suggesting that the layer starts subtracting noise from the input resulting in signals. This seems to be the case for the subsequent decoder layers as well. 
        - This finding motivates looking at the signals precisely at the top middle layer. 
    - This PR (at the top middle layer) when aggregated over the whole dataset can be used to understand the features live in the space by looking when / how often / and where they have high participations. This is the main mechanism that the paper uses to discover / analyze semantic features in the denoiser models. 
- The paper provides substantial qualitative evidence based on these measures, which I find convincing and that successfully changed my perception of the quality of denoiser model features.

### Strengths
*Note: I am reviewing this paper as a non-veteran reader of model representation studies. I may find some findings more interesting than they should be due to limited exposure.*

- While the model’s semantic class understanding is still limited—and not entirely unexpected as partially shown in DIFT (Tang 2023) and Mukhopadhyay (2023)—the paper nonetheless provides a convincing qualitative story and evidence.
- The overall qualitative evidence and analysis genuinely changed my perspective on the closest relative, diffusion models, representations.
    - I used to believe that diffusion models, being trained on pixel losses, must remain largely grounded in pixel distances despite their hierarchical representations. The degree of semanticity expressed—especially in Figure 4 (people) and Figure 15 (people and harbor)—exceeded my expectations. I am happy to say that I learned something new from this paper.
- I find the PR metric (which I believe is motivated by the Wiener process) simple yet insightful, and it is used very effectively in this paper. The subsequent φ values are also ingenious as a means of identifying feature subspaces. I find both novel.
- I find Figure 13 particularly insightful. It supports the claim (Line 243–245): “The marginal distribution of φ values in specialized channels is heavy-tailed, but common channels are closer to Gaussian.”
    - This finding is interesting for two reasons:
        - The confirmation that specialized channels follow a heavy-tailed distribution, while not surprising, is still valuable to know.
        - More intriguingly, the observation that common channels have Gaussian-shaped distributions raises deeper questions.
            - Beyond representation understanding, what does this imply about the dataset and learning process itself?
            - I cannot quite wrap my head around the Gaussian-ness, but if Gaussian distributions are narrow, does this suggest that features for common concepts can be captured with far fewer data points—perhaps following a faster scaling law than power-law regimes? Then, the remaining learning effort would focus on fitting the long tail of specialized concepts (which gives rise to the overall power-law scaling behavior).
- I like the idea of self-representation-guidance for visualization, which could be a useful tool for interpreting model representations (though its use in this paper feels somewhat limited; see Weaknesses).
- The paper is generous with clear visualizations and qualitative results throughout, including the appendix.

### Weaknesses
- I feel that the term *“semantic”* in the title might be a bit overclaimed, as the representations seem to prioritize shape, pose, and texture over human-defined semantics. The t-SNE plot (Figure 6, right) for human-labeled classes shows no clear separation of clusters, suggesting that the learned representation is still not strongly “semantic.”
    - To be clear, I understand that the model is unsupervised and should not be expected to produce human-level semantic groupings. My concern is purely about the choice of wording.
    - One question here: it is unclear how the t-SNE for human-labeled object classes was obtained. Why does this space look so different from the t-SNE plot on the left? I expected it to be the same, just colored differently by human labels.
    - Also, it's not clear how many clusters for K-means was chosen in the paper?
- Prior works on diffusion representations—such as DIFT (Tang 2023) and Mukhopadhyay (2023)—provided some quantitative analyses, which I find crucial. Assessing how well a model’s representation captures semantics is difficult to do via visualization alone. For example, in Figure 15, most rows could be explained by color and texture, which are not particularly “high level.” Including a quantitative task (that could be compared across models) would provide a clearer sense of the representation’s abstraction level.
- I find Figure 7 less insightful because most of the eight images are near duplicates of the conditional image. A more informative analysis would control for specific aspects or features—for instance, fixing a concept believed to represent a “wheel” and testing whether the generated images consistently depict wheels across scenes or poses.
- I don’t find the argument in Figure 1—that “the middle and decoder blocks exhibit increases in sparsity (i.e., reduction in PR)”—to be strongly supported, as Figure 10 tells a more ambiguous story (especially since the texture and CelebA datasets don’t seem to agree).
- Certain plots and analyses are hard to follow, such as Figure 6 (top-left ratio plot) and Figure 8. In general, I find the paper hard to follow at times. Maybe, due to my limited experience with this type of paper. The authors might help the readers of my type by making sure that the motivation / goal / take away message of each section is clear.
- While Section 3 is mostly easy to follow, I don’t find the finding in Lines 417–419 (“We show that Euclidean distances between φ’s are correlated with distances between the conditional densities they induce”) particularly valuable to the paper’s narrative. Even if true, it’s unclear what insight it offers—does it justify Algorithm 1’s correctness, or provide interpretability?

### Questions
While the finding here is not totally unheard of, I'm leaning toward a weak accept of this paper as it provides finding that confirm and in some cases (for me) exceed the expectations.

However, I still do have some skepticisms and questions::
- How well other "more primitive features" such as RGBs of 32 x 32 image, PCAs perform on the analyses in the paper (including K-Means in particular).
    - How for the other types of representations: MAE, DINO. 
    - This will help the reader understand better how much better are the denoiser features. 
- Since the denoisers are very similar to diffusion models which have t conditioned, it's natural to ask how do the features compare? Are they as robust (to the noise level) as the denoisers (I expect less robust to noise levels from the diffusion models). 

In addition to that, I think the paper will have more values with:
- More quantitative semantic measures of the representations and comparisons against other models' representations.
- Realizing the promise of self-guidance generation to visualize what the model understands about a "concept" (not a whole picture). 
- More discussion on (which may increase the reach and impact of this paper): What does it mean for common channels to have a Gaussian-shaped distribution? 
    - The authors may or may not go into: Beyond representation understanding, what does this imply for the dataset and learning process? If Gaussian distributions are narrow, does this mean that we can capture common features with fewer data points and faster scaling laws than power laws?

### Soundness
3

### Presentation
3

### Contribution
3
