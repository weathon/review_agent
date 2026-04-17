# Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields

- Decision: Accept (Poster)
- Scores: 6, 0, 4, 8, 6

## Abstract
Implicit Neural Representations (INRs) have emerged as promising surrogates for large 3D scientific simulations due to their ability to continuously model spatial and conditional fields, yet they face a critical fidelity-speed dilemma: deep MLPs suffer from high inference cost, while efficient embedding-based models lack sufficient expressiveness. To resolve this, we propose the Decoupled Representation Refinement (DRR) architectural paradigm. DRR leverages a deep refiner network, alongside non-parametric transformations, in a one-time offline process to encode rich representations into a compact and efficient embedding structure. This approach decouples slow neural networks with high representational capacity from the fast inference path. We introduce DRR-Net, a simple network that validates this paradigm, and a novel data augmentation strategy, Variational Pairs (VP) for improving INRs under complex tasks like high-dimensional surrogate modeling. Experiments on several ensemble simulation datasets demonstrate that our approach achieves state-of-the-art fidelity, while being up to 27$\times$ faster at inference than high-fidelity baselines and remaining competitive with the fastest models. The DRR paradigm offers an effective strategy for building powerful and practical neural field surrogates and INRs in broader applications, with a minimal compromise between speed and quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the Decoupled Representation Refinement (DRR) paradigm to track the fundamental trade-off in Implicit Neural Representations (INRs): High-fidelity MLP-based models are computationally expensive and slow to query, while fast embedding-based models lack expressive power. The core is a deep refiner network, which is used to encode rich representations into a compact and efficient embedding structure. Experiments show that the proposed method achieves SOTA fidelity, while being up to 27x faster at inference than high-fidelity baselines and remaining competitive with the fastest models.

### Strengths
1. The proposed method is simple and effective. DRR shows potential for broad application.
2. Extensive experiments and ablation studies thoroughly validate the effectiveness of the DRR framework and most of key components.

### Weaknesses
1. Variational Pairs (VP) lacks sufficient background in the introduction.
2. In Synergistic Representation Learning (Line 192), the authors compare DRR with grid-based methods regarding multi-scale representation and feature fusion. Authors claim: “The refiner learns a potent, non-linear function to perform this fusion offline, baking the multi-scale features directly into the cached grid $\mathcal{G}^{\prime}$.” The reviewer suggests adding a visual comparison between $\mathcal{G}^{\prime}=\hat{\mathcal{G}}$ and $\mathcal{G}^{\prime}=\hat{\mathcal{G}}+\Delta{\mathcal{G}}$ to illustrate whether DRR indeed performs multi-scale feature fusion in a more reasonable manner.
3. As DRR resembles a residual-based method, the reviewer believes discussing analogies to such methods in the introduction and related work would better elucidate significance of DRR for INRs. For instance, the approach described in Equation 3 may enable identity mapping similar to skip connections in ResNet, providing a more flexible forward pathway for complex representations.
4. Although the authors present ablation studies on the parameter-free preprocessing step, $\pi$, across different resolutions in Sec. F.1, the reviewer is more interested in experiments without $\pi$. Specifically, the reviewer would like to know whether $\mathcal{G}^{\prime} = \mathcal{G}+R_{\psi}(\mathcal{G})$ remains more effective than $\mathcal{G}^{\prime} = \mathcal{G}$. This experiment would more conclusively demonstrate whether improvements of DRR over other SOTA methods stem from mapping $\mathcal{G}$ to a higher-dimensional space or from the residual refining module (the core contribution).

### Questions
Please refer to the weaknesses. Overall, DRR is a simple and effective method for improving INR performance. However, the paper lacks clarity on several key issues (Weaknesses 2 and 4). The reviewer looks forward to the authors’ response to these concerns.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper explores the usage of INR methods for surrogate modeling. Observing that embedding-based methods lack representational capacity while MLP methods are too slow, they propose an approach in which embeddings are first transformed through a parameter-free interpolation and stacking function before being projected by a complex non-linear mapping. The resulting function is then treated as the embeddings for interpolation. The authors further propose augmentation strategies involving perturbing the training data under continuity assumptions in the data. This is evaluated across three 3D challenges.

### Strengths
The conflict between efficiency and accuracy is an important problem in surrogate modeling. The paper does a good job of outlining how this is a problem for INR methods and in describing the weaknesses in prior iterations of INR methods that this approach is trying to correct. The datasets used seem interesting and the focus on 3D problems is well-suited to the scalability goals of the paper.

### Weaknesses
Overall, this submission seems to largely apply methods for graphics to scientific tasks with little adjustment for the domains targeted. It seems like an interesting graphics approach, but the submission requires a lot of work before it's a viable paper on surrogate modeling. The text of the paper is missing a lot of information necessary for evaluation and reproducibility. Comparisons largely focus on similar classes of method and not on methods used in the space. Given the lack of information, it's not clear whether the proposed method is actually effective for the given tasks. I can't speak to the novelty of the technique as my expertise is in surrogate modeling rather than image reconstruction. Here are some specific issues and suggestions for improvement:

1. Generally the experiments do not provide sufficient information to evaluate the soundness of the experiments. These manifests in a few ways:
    1. The datasets here reference codebases rather than preconstructed datasets. Given these are transient problems, what procedure is used to generate the dataset? Are these constant initial conditions are a variety of system coefficients? What time step are these gathered from? Does the system converge to a steady state or is this instead solving a steady state version of the problems? How far away are the parameters? These details are extremely important for evaluating surrogate modeling tasks. 
   2. What grids are actually being used for the grid-based methods in terms of both resolution and dimensionality? Are all grid based methods getting access to the same resolution and dimensionality of embeddings?
   3. Given the uncertainty over the dataset, its actually not clear that interpolative methods like INR are appropriate for modeling the systems. For instance, ocean simulations over long time horizons can often produce very different results at the same time range based on minor perturbations, though to my knowledge this is not the case for the astro-oriented hydrodynamic simulations. 
2. The baselines do not appear to be well executed. The appendix states that all embedding models were trained with a single learning rate or with exact settings recommended by the original paper. This is not a reasonable approach and will generally promote the "new" model by default since that is the model where hyperparameters are tuned for the given datasets. Different models will often perform differently at different hyperparameter ranges on different datasets. Good practice is to do a hyperparameter search for each model independently. 
3. It's not clear that the refiner network is superior to just using denser grids of higher dimension. The refiner network seems to operate entirely on the embedding vectors with no additional information. Apart from potentially enforcing some low-rank structure in $\hat G$ (which the refiner can learn to ignore), passing them through a complex nonlinear mapping ultimately means you're operating on a grid of embeddings with a certain resolution and dimension with smoothness determined by the interpolation used on the highest resolution grid. While it could turn out that the initial low rank structure is heavily used by the refiner and provides a significant advantage. 
4. Several of the augmentation innovations seem overfit to the data. The data here is largely smooth with very little variation which lends itself well to low order continuity assumptions. However, this is not generally the case for simulation data which often has sharp discontinuities in the form of shocks or interfaces. In parameter space, one of the complexities of physical simulation that researchers are interested in when they're running ensembles of simulations is the existence of bifurcations in parameter space which would likely be hurt by augmentation strategies relying on strong smoothness assumptions. Smoothness is often a reasonable choice, but it's important to also demonstrate cases where the choice may not be appropriate to evaluate whether the proposal is a reasonable default. This concern could  also be addressed by exploring the relationship between the source data and optimal interpolation
5. Surrogate modeling is an extremely active area of research, but the only baselines are two very recent INR models from a single group and one INR method that seems to have been developed for graphics purposes. Diffusion-based baselines are likely the best fit here since your models map from parameters to a field, though given they're all naturally transient, autoregressive baselines rolling out to the given time frame could also apply. 
6. Metrics - the metrics used are all image reconstruction metrics. Typically, relative error or various forms of normalized RMSE are used in surrogate modeling. Often the appropriate metric depends on the time horizon of the predictions. For the ocean simulations, generally longer time horizons would use smoothed metrics, for instance. Pointwise metrics are often more appropriate for short time horizons. From the appendix, these scenarios seem to be largely empty space which would mean localized metrics around regions of interest are probably more informative than metrics that will be heavily weighted by regions with little activity. 

Minor
1. Figures are not super informative. Figure 1 for instance is mostly a less informative version of figure 2. 
2. Table 3 should list the metrics. It's inferrable from the main text, but tables should be self-contained. 
3. Limitations right now are mostly limited to trivial issues with the implementation rather than more insightful analysis of where this line of research needs to move to address more realistic challenges.

### Questions
1. How are the datasets constructed? 
2. How does the proposed approach compare to established methods when hyperparameter tuning is performed on the other methods?
3. How does the proposed approach compare to standard ML approaches for surrogate modeling like autoregressive models or diffusion models?
4. What are the full details of the models used in this paper?
5. What types of problems is this method well suited for? What problems is it poorly suited for?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper suggests a new approach for training and inference of Implicit Neural Representations (INRs), also called Neural Fields (NFs): Decoupled Representation Refinement. 

NFs are MLPs with (continuous) coordinates (and potentially additional time or conditioning variables) as inputs. While Nfs are powerful representations, they typically suffer from a tradeoff between reconstruction fidelity, which requires larger, parameter-rich MLPs, and inference speed, which decreases with MLP size. 
To bypass this tradeoff, DRR aims to decouple the learned high-capacity representation from inference queries, thereby allowing for inference speedups of up to x27. 
The high-capacity representations are created by a learned refiner, which takes the multi-resolution input embeddings upsampled to the same scale and outputs a dense grid. The Decoder, a small, lightweight MLP, can then be queried quickly by taking the coordinate and the conditioning embedding as features.

As data augmentation for improved generalization, the authors use a modified version of Variational Coordinates (VCs). Instead of a piecewise linear prior, their augmentations (VP Spatial and VP Spatio-Conditional) impose a smoothness prior, which fits the application domain (scientific simulations) much better.

In experiments, the authors demonstrate the performance of DRR on 3 datasets with 3-6 dimensional fields: Nyx (a dark matter simulation), MPAS-Ocean (an earth system simulation), and Cloverleaf3D (a hydrodynamics simulation). They show that DRR overcomes the conventional fidelity/efficiency tradeoff by achieving both the good reconstruction quality of large, expensive INRs and the speed of fast, embedding-based methods.

In the appendix, the authors provide further ablation studies which justify their architecture choices and show the DRR framework also applies to the classic NF domains, such as 2D image superresolution and 5D neural radiance fields.

### Strengths
## Soundness
The experiments support the central idea of the method, i.e., good reconstruction quality and fast inference.

## Contribution
Achieving fast inference times with NFs is a known problem, and the authors offer an interesting solution that combines a learning-based approach with the advantages of multi-resolution representations, which have shown good results in neural graphics processing.

## General
- Very interesting take on a known problem of NFs: Large MLPs are slow in inference. The large MLPs use their large parameter count to construct rich internal representations in their hidden layers. Embedding-based methods like NGP encode features in learnable feature grids, which makes them fast but also memory-intensive. DRR proposes a spatial refiner to achieve similar rich encodings to be utilized by a fast decoder.
- DRR is shown to work across simulations with fields of different output dimensions and also more classic NF applications such as image superresolution and NerFs.
- Good ablation studies to justify the different architecture choices.

### Weaknesses
My main issue with the paper is that it lacks clarity about the method.  Furthermore, the reported metrics are insufficient for physical simulations. Certain claims regarding “SOTA” also do not stand up to scrutiny.

### Questions
## Major Remarks
- The paper lacks clarity about the method. Especially in 3.3, I am still not certain what exactly the inputs/outputs and their sizes are for the different components. Figure 2 only helps so much. E.g., what is a 1D multi-resolution feature line? What exactly is the output form of the embeddings? Is it just a concatenated vector of embeddings? Of what dimension?
- Building on the previous point, what are the memory costs at inference for DRR, and how do they scale? How memory-intensive are the spatial embeddings? How big is the overhead for the embeddings, i.e., how many inference queries are needed to amortize the initial encoder forward pass?
- Please add some more metrics which indicate simulation quality. For example, in fluid simulations, crucial indicators are the preservation of conservation laws (e.g., momentum or energy), or the accuracy of spectral properties (e.g., kinetic energy spectra) which indicate how well the method captures turbulent or small-scale features.
- In the field of neural graphics/rendering, serious efforts have been made to enable fast inference of neural graphics, e.g., [1]. I believe that this should at least be in the related work and discussed, e.g., why the same idea works/not work for simulations, and why such strategies are not pursued.
- You claim, e.g., SIREN (and NGP) is SOTA on Page 21; this is not true. Several SIREN improvements have been published; see [2] for a current comparison. 

## Minor Remarks
- Page 3 Line 129-130: You use \mathcal{P} for the parameter count and the number of points in a simulated field as P. This reduces readability. Please use something other than P.
- Page 8 Table 3: You show the performance on trained and unseen fields in the upsampled case. Can you also provide performance metrics of trained vs untrained on the same sampling scheme, i.e., how much performance drop is there in fitting a single field vs the generalization ability of a larger pretrained one?
- I also feel like the fact that your method also works for NerFs or image superresolution should be highlighted a bit more in the main text; the fact that the method is so general is noteworthy. (Yes, I know the page limit makes things tough).

## Conclusion
I think the core idea of the method is sound, and while the experimental evaluation could be broader, it is good enough for acceptance, especially given the additional ablations and experiments in the appendix. The reason I currently lean towards rejection is the, in my view, bad/unclear presentation at times, insufficient metrics for numerical simulations being reported, and using baselines that are no longer SoTA as "SoTA". If the authors can address my points, I am willing to discuss and adjust my score.

[1] Duckworth, Daniel, et al. "SMERF: Streamable memory efficient radiance fields for real-time large-scene exploration." ACM Transactions on Graphics (TOG) 43.4 (2024): 1-13.

[2] Essakine, Amer, et al. "Where Do We Stand with Implicit Neural Representations?" A Technical and Performance Survey 11 (2024).

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes a decoupled representation refinement for implicit neural representations of large-scale scientific simulations. 
The decoupled representation refinement first trains a embedding-based model. The embedding, represented by a base structure, can be preprocessed and a refiner network learns an improved representation, utilizing the expressive power of deep neural networks. The updated base structure can be used for inference via a fast query function. 
The updated base structure can also be cached. Additionally, the authors propose two augmentation strategies. The method is evaluated on three scientific datasets and compared against relevant benchmarks.

### Strengths
- The method is well motivated and an effective compromise between fast embedding-based models and MLP-based approaches. 
- The paper is well written 
- Detailed experiments on three scientific datasets show quality vs. training/inference speed tradeoff. DRR-Net compares favorably.
- Easy to implement data augmentation method that improves PSNR/SSIM in most cases
- Additional experiments on vision/graphics tasks in the appendix

### Weaknesses
- The paper is generally well written, but the description of the condition encoder was a bit unclear to me. How does "a set of 1D multi-resolution feature lines for each simulation parameter" support scalability to "arbitrary parameter dimensions" Could you please clarify this?
- Previous approaches have considered higher resolution scientific datasets, e.g., [1]. Can DRR-Net scale to similar resolutions; what changes are necessary?
- The proposed data augmentations seem effective for interpolation settings of conditional inputs. Have you done any tests for extrapolation of conditional inputs?

Overall, this is a convincing paper to me with strong experiments. Since I haven't been following the recent literature for implicit neural representations closely, it's difficult to fully assess the novelties of the method for me. Therefore I will give a low confidence to my rating. 

[1] https://arxiv.org/abs/2308.02494

### Questions
See questions in weaknesses above.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Implicit Neural Representations (INRs) either give you high accuracy but slow queries, or fast queries with weaker expressivity. This work proposes a Decoupled Representation Refinement (DRR) method, which runs a deep refiner network once offline to bake rich structure into a compact embedding that can be queried efficiently. Authors claim the proposed model, DRR-Net, plus a data augmentation strategy (Variational Pairs), achieves state-of-the-art fidelity on ensemble simulation surrogates while being up to 27x faster than strong high-accuracy baselines and still competitive with the fastest models.

### Strengths
1) DRR-Net hits a strong pareto point as it produces competitive results compared to baselines like FA-INR in PSNR/SSIM on large 3D scientific ensembles, but delivers ~10x-30x faster inference, and it outperforms fast baselines like Explorable-INR in fidelity while staying in their runtime class.

2) Instead of treating each simulation parameter independently or relying on low-rank factorization, DRR-Net builds unified multi-parameter conditional embeddings and refines them jointly, so nonlinear interactions between parameters are captured and then cached for fast reuse. This is something that is not explored as much in other methods especially the models that are acting as simulators. 

3) Experiments are comprehensive and the baselines are relevant and recent (Some missing ablations, see weaknesses).

### Weaknesses
1) The stated one time cheap evaluation of refiner R seems to be dependent on the problem setup, boundaries etc. does that mean if the boundaries have changed a new refiner is required? If yes this seems like a major drawback compared to other methods especially given that training times for this method are relatively longer than most other frameworks. Can the authors clarify whether it is correct, that any specific change in the domain (boundaries etc.) requires retraining? Furthermore is there any assumption on the conditioning parameters? Can these be a field like conditioning and not simply scalars? Do these affect the training? 

2) Experiments hold out ensemble members but still operate within the same parameter ranges as training (Table 1). Can you characterize DRR-Net’s behavior under extrapolation? For example, how does it perform for parameter settings outside the training envelope, and how does this compare to FA-INR and Explorable-INR? Currently extrapolation claim is not well justified. 

3) Although model parameter counts are provided in the table, it is difficult to establish if the improvements are not originated from variations in capacity and what's the role the parameter counts plays in this. The interplay between the model size and the interpolation/extrapolation capacity is not demonstrated.

4) There is very little on detail on the training procdedure, actual objective/objectives, and hyperparameter setting. This can make the procedure vague for the reader. Similarly although explained in text, this can be hard to follow how and in which order the components are trained for this framework .

### Questions
Please refer to the weaknesses (Each weakness raises one or more questions).

### Soundness
3

### Presentation
2

### Contribution
3
