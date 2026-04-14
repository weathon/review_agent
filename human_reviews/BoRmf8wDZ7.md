# Gaussian Masked Autoencoders

- Decision: Reject
- Scores: 5, 6, 6, 5, 6, 3

## Abstract
This paper explores Masked Autoencoders (MAE) with Gaussian Splatting. While mainstream self-supervised learning frameworks such as MAE operate on low-level pixels, the image synthesis community has evolved to use latent, mid-level representations for better generative visual data modeling. Our approach, named GMAE, aims to reconcile these two and get the benefits of both worlds. Like MAE, it reconstructs the image end-to-end in the pixel space; however, it also introduces an intermediate, 3D Gaussian-based representation and renders images via splatting. We show that GMAE can enable various zero-shot learning capabilities (e.g figure-ground segmentation, image layering, edge detection, etc) while preserving the high self-supervised representation quality from MAE. Notably, we are the first to employ Gaussian primitives in an image representation learning framework beyond optimization-based single-scene reconstructions. We believe GMAE will inspire further research in this direction and contribute to developing next-generation techniques for modeling high-fidelity visual data.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a new self-supervised learning method called Gaussian Masked Autoencoder (GMAE) which combines the advantages of pixel-level and intermediate representation learning. It uses 3D Gaussian distributions as intermediate representations to capture richer image information and improve image generation and processing abilities. GMAE achieves comparable performance to supervised learning while significantly improving zero-shot learning tasks such as segmentation, layering, and edge detection. The paper also demonstrates the potential of GMAE in downstream tasks like visual recognition and object detection.

### Strengths
1. The paper presents a novel approach to self-supervised learning by introducing Gaussian Masked Autoencoder (GMAE) that utilizes 3D Gaussian distributions as intermediate representations. This idea is different from traditional methods that use pixel-level reconstruction, and the authors demonstrate the effectiveness of this approach through experiments.

2. The paper is well-written and easy to understand. The authors provide details about the implementation and evaluation metrics, making it possible to replicate the results. The empirical results presented in the paper are convincing and support the claims made by the authors.

### Weaknesses
1. Lack of comparison to state-of-the-art methods: The paper does not compare the proposed method to other existing methods for self-supervised learning, such as contrastive learning or clustering-based methods. This makes it difficult to assess the relative performance of GMAE compared to other approaches.

2. I didn't understand the immediate motivation for choosing Gaussian splatting to enhance MAE in this paper. What qualities does Gaussian splatting have to help MAE? Can Gaussian splatting be replaced with NeRF or other 3D representations?

### Questions
See Weaknesses. I will change my rating based on the responses.

### Soundness
2

### Presentation
2

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
In this paper, the authors present a self-supervised image representation learning method to extend Masked Autoencoders with 3D Gaussian Splatting framework. The general framework is a ViT based auto-encoder which takes masked patches from given images as input. The key idea is that instead of predicting image patches, the ViT based decoder regresses the parameters of 3D Gaussians for further rendering. To validate the importance of this technical upgrade, the authors perform comparisons on both supervised tasks and unsupervised tasks as well as ablation studies on different training mechanisms.

### Strengths
+ Generally, the paper is well written. I can understand it easily.
+ As for me, the idea of applying learned 3D Gaussians as primitives for downstream tasks in unsupervised learning is novel and critical to the computer vision community.
+ The authors perform evaluation for both supervised and unsupervised tasks to show the empirical significance of their 3DGS based network upgrades.

### Weaknesses
+ The evaluation datasets and used baselines seem to be a bit outdated. The latest baselines (MAE and MAE-VQGAN) were published in 2022 while the latest testset (PASCAL) was published in 2015. Could the authors evaluate their method on some datasets listed in Figure 8 of SAM [1] with modern large-scale unsupervised learning methods? For example, datasets like COCO-Stuff or ADE20K? And baselines like SAM or DINO v2? Or other related datasets and baselines?
+ There are no failure case examples to justify the possible future work of GMAE method. Ideally, the failure cases might reveal limitations in the Gaussian representation and highlight scenarios where the method struggles compared to pixel-based approaches.
+ Some typos which include:
1. L313, "the ViT base model" --> "the ViT based model";
2. L533, the "For example," are repeated twice.

[1] Segment Anything, ICCV 2023

### Questions
+ As mentioned in L200 and L227, how to get the query tokens? What these tokens could be?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces GMAE, a method that integrates MAE with Gaussian Splatting. It claims that GMAE is a better way to represent mid-level features. Instead of reconstructing masked pixels, GMAE predicts a set of Gaussians, each parameterized by a 14-dimensional vector. The authors have meticulously designed the model, and some observations align closely with those of MAE. Empirical results demonstrate that GMAE achieves performance comparable to MAE on supervised tasks and exhibits satisfactory zero-shot capabilities in unsupervised tasks.

### Strengths
- The idea looks intriguing to me. The non-uniformity inherent in Gaussian representation distinguishes it from traditional patch-based methods.
- Experimental results are comprehensive and convincing.
- The numerous visualizations offer an intuitive grasp of the methodology and outcomes.
- The paper is well-written and easy to follow.

### Weaknesses
- In supervised tasks, the model primarily utilizes the ViT encoder, without incorporating Gaussian representations. The effectiveness of Gaussian representations is demonstrated in unsupervised tasks. Demonstrating a positive impact on image generation would significantly enhance the paper’s contributions.
- The limited number of Gaussians employed constrains the model’s reconstruction capabilities for image generation. If increasing the Gaussian count presents a bottleneck, this limitation could hinder its application in image generation tasks.

Overall, my main concern lies in the scalability of the method. But as an initial attempt, I think the paper is above the acceptance threshold .

### Questions
- In line 089, the statement “the addition of splatting increases compute time by 1.5%” would be more informative if the authors provided the absolute compute times for both MAE and GMAE, facilitating a clearer comparison.
- In Figure 7, arranging the visualization of Gaussian layers from shallow to deep depths could be more intuitive, as objects closer to the camera often hold greater significance.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes to replace the decoder module of MAE with a gaussian splatting decoder≤µt: instead of decoding query pixel patches, it proposes to predict the parameters of 3D gaussian splats, which are then rendered onto a static camera position. The authors provide ablations on the number of gaussians used. They show several zero-shot capabilities which arise from such design: background/foreground segmentation and edge detection. The paper is well written and have high quality visualizations. They also analyze some properties of how the gaussians get distributed:

### Strengths
- I guess the main strength is some zero-shot capabilities, like foreground/background separation and edge detection.
- Despite to unconventional design, it does not lead to the loss of the main quality of self-supervised methods: 
- The overall idea is quite unusual which, I believe, is a good quality of a scientific paper.
- Writing is very clear and the presentation quality is high.

### Weaknesses
- The method looks very unnatural and simply combines 2 popular ideas: 3d gaussians and MAEs. There are no particular advantages or insights in combining them. I feel the benefits are marginal and not worth the complications of the design. 
- Zero-shot capabilities are not convincing: there are easier ways to obtain them with a higher quality (e.g., generative methods or generative multi-plane images with similar layered representations).
- The main advantage I would hope to see is having some 3D capabilities, but they are lost due to rendering from a static position.

### Questions
- What exactly is the main advantage of the proposed model? For which use-case would one realistically chose to use it?
- How much slower each training iteration has become compared to MAE?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Gaussian Masked Autoencoders (GMAE), an extension of MAEs that integrates a learned intermediate Gaussian representation, rendered into an image using Gaussian splatting. This intermediate representation is shown to offer several benefits over fixed patches, such as foreground-background separation, edge detection, and image layering, while achieving performance on par with or even surpassing standard MAEs in image recognition and downstream tasks. The authors support their claims with a comprehensive suite of experiments.

### Strengths
- The paper explores an interesting topic of adding additional inductive biases to self-supervised image representation learning techniques.
- The writing is clear and well-structured.
- The experiments section includes a wide variety of downstream applications and comparisons.

### Weaknesses
- As talked about in the Discussion section, the number of Gaussians used in GMAE is significantly lower than the quantities typically used in scene reconstruction applications, where Gaussian splatting is well-known. This is because each Gaussian corresponds to a unique token in the lightweight decoder, so increasing their number would cause considerable slowdowns.
- Minor typo on Line 503: Fig 12 → Fig 11

### Questions
- The GMAE model assumes a fixed camera projection for its rendering. Therefore, it most likely does not need all 14 degrees of freedom that is normally used in scene reconstruction applications. Do you think there is any benefits to manually reducing these redundant DoFs when training such models?
- Since the gaussian representation introduces a 3D inductive bias for the model, it would be interesting to compare this model with MAEs in a depth-prediction downstream task. In theory, GMAE should be much better equipped for solving such tasks. Do you expect to see a significant improvement over MAEs, or would the limited number of gaussians not allow for such a thing?
- As discussed in Section 4.1, the decoder is decoupled from encoder tokens, allowing the number of Gaussians to be increased arbitrarily after training. However, it’s mentioned that four separate models were trained to decode 64, 128, 256, and 512 gaussians, respectively. How well does a decoder generalize to Gaussian counts other than the one it was trained on? For example, does a decoder trained with 256 gaussians achieve better performance is evaluated with 1024 gaussians?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 6

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes to use 3D Gaussians a-la Gaussian Splatting as a mid-level representation that are predicted from masked input patches a-la Masked Autoencoders. Training is done by splatting gaussians differentiably to render RGB from a fixed camera using MAE training losses (unnormalized). Authors pre-train a ViT-B for 400 epochs using this method and show a slight performance drop vs. regular MAE training on Imagenet classification and COCO instance segmentation. 

They also show that the predicted gaussians are layered in depth from the fixed camera by frequency. The authors attempt to utilize this property to show some results on zero-shot edge detection and figure-ground segmentation, where they perform similarly or worse to a very selected set of baseline methods.

### Strengths
Originality: 
The proposed method of using 3D gaussians as their intermediate representation is original and interesting. However, the related work section misses very related work and focuses on some more irrelevant topics (discussed more later)

Quality:
The proposed representation seems to learn better reconstructions compared to MAE. However, beyond this, I personally do not agree with the proposed evaluations to show the benefits of this representation. No comparisons are made to any other similar intermediate representations that could be thought of (discussed more later).

Clarity:
The paper is presented clearly and laid out well. Some statements are misleading (discussed more later).

Significance:
I believe this paper has potential to be significant to the community. However, the current results show that GMAE does not improve over MAE meaningfully (and is evaluated very sparsely across the design space) and the learnt gaussians are not very useful in the zero-shot case even in the tasks that the authors decide to focus on i.e. figure-ground segmentation and edge detection.

### Weaknesses
\textbf{Related Work:}
The paper does not talk about any related work on using mid-level representations in vision beyond using learned "tokens". The authors misrepresent MAE as only training for pixel reconstruction. MAE has an ablation experiment where they also use tokens to explore the "best of both worlds" approach that the authors suggest they take. MAE-VQGAN proposed in Bar et al. 2022 is also a tokenized MAE learner. Other mid-level representations can be thought of that are similar to this method. For example, one could directly predict a multi-plane representation and render it. One could use superpixels a-la superpixel sampling networks (Jampani et al.) as the mid-level representation. There is no discussion on other possible methods and prior mid-level representations used in vision. Other papers have proposed losses that learn self-supervised grouping, (which is one of the benefits according to the authors), such as those based on Slot Attention or Leopart (Ziegler et al, CVPR 2022). In the discussion, the paper claims -- "Nonetheless, we have shown that one no longer has to choose between pixels and latent representations for visual modeling.". This is misleading compared to related work as mentioned above. 

\textbf{Why is the gaussian representation better?}
There are claims across the paper that the gaussian representation is better due to its efficiency (the proposed model is slower than MAE while performing worse), due to its non-isotropic representation vs. grids (no comparisons are made to back the claim that this is useful for pre-training). The only real benefit shown in the paper is that GMAE reconstructions are higher-fidelity as opposed to MAE. However, the authors immediately claim "L362: As a result, our reconstructions can be used directly for other tasks without needing to add a GAN loss or an upsampling layer on top." which is again unsubtantiated in the paper. Which other methods need a GAN loss or upsampling layer on top? The other tasks proposed here are figure ground segmentation and edge detection, where the model performs poorly overall. Discussed more in the next section. 

\textbf{Frequency clustering in depth}
The authors make the following claims:

"This may be due to the fact that with random initialization, the points closer to the camera represent low-frequency information, while the points far from the camera model the high-frequency information"

"The layer-wise rendering highlights the model’s ability to separate objects and represent them in distinct frequency layers"

"In the real world, backgrounds tend to have low-frequency regions while objects usually have high-frequency details. This correlation leads to our zero-shot results."

These are incompatible claims and I think these are mis-leading when looking at the results. Objects are clearly not separated across frequencies. Low frequency shapes of most objects seem to be captured in the initial layers and higher frequencies of their shapes in later layers. Figure 6 and 7 corroborate this. Claiming that objects are separated and represented in distinct frequency layers does not appear true from the results and does not follow the prior claim of frequency based clustering. Individual instances of objects are not separated in any way. The edge detection results show lots of spurious edges coming from the gaussian representation which only make edge prediction worse. The argument that backgrounds tend to have low-frequency regions while objects.. is barely enough to make the claim that objects are separated in the model. The examples shown are few and relatively simple with one bird on a tree and clear background. Yet, the model is unable to separate the tree branches from the bird, and even the bird is not clearly segmented. I believe the assertion that frequency based depth ordering happens. The follow-up claim that this leads to emergence of objects or even parts is a stretch. 


\textbf{Experimental details and strength of results}
The authors only train ViT-B for 400 epochs. The authors could have pre-trained for 1600 epochs, or tried a ViT-L architecture. Currently there is no clarity whether this approach will scale to a larger ViT or if it will continue to improve with additional training as MAE does. The ablation studies over c, masking ratio and loss masking, normalization and the usage of batch size 4096 show that sufficient GPU resources were used in pre-training. At least pre-training ViT-B till 1600 epochs should have been possible for the authors. It would be very useful to add these results. Without these results, it is impossible to verify whether GMAE scales as MAE does. 

For the figure-ground segmentation results, there are no details on the experiment. What layer was used for figure ground segmentation in the layering? No discussion on the baselines is presented. Models such as Leopart (Ziegler et al.) need to be compared. Their results on zero shot segmentation are way more advanced while not needing a sparse gaussian representation that the authors claim is the reason why their figure ground segmentation results are strong.

The edge prediction results are worse than using a Sobel filter for edges. There are clearly numerous spurious edges in the qualitative result that probably come from gaussians that represented interior regions of objects that do not correlate with any real edges.

### Questions
Please address the issues brought up in the weaknesses section. 

Experimental results on ViT-B trained to 1600 epochs and ViT-L could be very useful, but I understand that it is unreasonable to ask for these results within the rebuttal period. 

I believe the authors need to focus more carefully on their evaluation. If depth based frequency layering is all that the model achieves over standard MAE, could this be done without using this intermediate representation? What other tasks can be helped by such a representation? Clearly figure-ground segmentation and zero-shot edge detection do not benefit from this method.

### Soundness
2

### Presentation
3

### Contribution
2
