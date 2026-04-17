# DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation

- Decision: Accept (Poster)
- Scores: 6, 8, 4

## Abstract
End-to-end backpropagation requires storing activations throughout all layers, creating memory bottlenecks that limit model scalability. Existing block-wise training methods offer means to alleviate this problem, but they rely on ad-hoc local objectives and remain largely unexplored beyond classification tasks. We propose $\textit{DiffusionBlocks}$, a principled framework for transforming transformer-based networks into genuinely independent trainable blocks that maintain competitive performance with end-to-end training. Our key insight leverages the fact that residual connections naturally correspond to updates in a dynamical system. With minimal modifications to this system, we can convert the updates to those of a denoising process, where each block can be learned independently by leveraging the score matching objective. This independence enables training with gradients for only one block at a time, thereby reducing memory requirements in proportion to the number of blocks. Our experiments on a range of transformer architectures (vision, diffusion, autoregressive, recurrent-depth, and masked diffusion) demonstrate that DiffusionBlocks training matches the performance of end-to-end training while enabling scalable block-wise training on practical tasks beyond small-scale classification. DiffusionBlocks provides a theoretically grounded approach that successfully scales to modern generative tasks across diverse architectures.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel method to transform modern networks with residual connections (e.g., ViT) to independent trainable blocks that learns a diffusion denoising process for a predefined noise range. During training, all blocks rely only on the input x and not output of previous/other blocks. During inference, input x is passed as a condition to blocks starting with the largest noise range to the smallest noise range. The method does not require backpropagation across all blocks, reducing memory usage. Experiments on numerous tasks demonstrate promising results.

### Strengths
- The main idea of this paper is novel (concurrent to NoProp). It tries to convert the foward propagation proccess of residual-based architectures to a diffusion denoising process.
- The method is evaluated on numerous tasks (image + text classifcation and generation)
- There are some theory analysis of the partitioning approach

### Weaknesses
- In Table 2, the number of training epoch/iterations are not reported. For DiT-L/2 on ImageNet256x256, what are the FID for with and without classifier free guidance (CFG). The current result does not state whether CFG is used.
- To my understanding Equation 4 should only denote one residual connection. In ViT, one residual connection is for the self-attention operation and one residual connection is for the MLP. To be rigourous with the thoery, each denoising block should denote a single residual connection so each ViT block should be decomposed into two diffusion blocks. In the current implementation ViT/DiT with 12/24 layers are partitioned into 3 blocks. From this perspective, I don't think the thoery fully back up the proposed partition approach.
- Despite that the proposed method uses less memory than standard models, diffusion processes are more complex. I am worried that the acutal speed of the method is much slower than standard ViT. The authors should report real time comparison of training and inference speeds on GPUs for example on the image classifcation and generation tasks.
- The authors should report the number of denoising steps for each block during inference.

### Questions
- Each block learns difference noise levels independently. In this case, why use mulltiple blocks and not just one large block to learn all noise ranges?
- The last block of the network reconstruct the original input. Why can this be used for classification? Conditioned on the input, the denoising blocks reconstruct the input. This does not yield a standard representation like other models that can be used for deterministic tasks. The authors have demonstrated classifcation performance on CIFAR but I think this is not sufficient. I would really like to see whether the approach works for ImageNet-1k as well as downstream dense prediction tasks such as semantic segmentation. Due to the time limit of the rebuttal, partial results are ok, for example, training only 100 epochs instead of 300 (standard ImageNet-1k training setting).
- If I want to finetune a pretrained model on a dataset, can the weights and knowledge learned by the pretrained model ViT be used with this approach?

Overall, I think the idea is novel, but there are some questions that needs clarifying and I'm have concerns on the practical application and performance of the proposed approach.

### Soundness
2

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
3

### Summary
The paper proposes a block-wise training strategy that splits standard transformer-based neural networks into groups of layers (blocks) which can then have forward/backward computed separately. This approach saved memory equal to the number of blocks, e.g. Bx memory savings. The paper achieves this by reconceptualizing the role of a block as a subnetwork that needs to denoise the output within a certain range. So, for a network with 12 layers spilt into 3 blocks, each group of 4 layers is responsible for removing a designated amount of noise, and the denoised output of the final set of 4 layers constitutes the actual prediction. The paper shows this method can compare to end-to-end training for image classification, image generation, and text generation.

### Strengths
S1. The method seems very novel. Converting these disparate tasks (classification, image generation, text generation) into denoising tasks and separating the forward/backprop for the network into different groups of layers (blocks) seems to solve a problem that prior block-wise training methods could not.

S2. Under the specified setups, the block-wise training matches or beats the baseline.

### Weaknesses
W1. On practical terms, this method would seem to have no advantage over recompute/checkpointing. In fact, the memory savings from recompute are substantially higher.

W2. It is not clear how this affects training efficiency (wall time). 

W3. Baselines are odd. CIFAR-100 is a little toy-ish, and the default ViT performs quite poorly on it. The DiT is also under-optimized, and I assume this applies to other things as well. So it's not actually clear that this method can match any end-to-end training in practical setups, where these models are fully optimized.

### Questions
Since this method does not seem to be immediately practically useful, I am mainly viewing the strength in terms of the novelty (which I think is quite high). However, my mind could be changed (either for better or worse) depending on the following:

1. Can this be used effectively in tandem with recompute/checkpointing? What is the resulting impact on training time?

2. How long does this method take to train for ViT, DiT, etc.? Wall time preferred.

### Soundness
2

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
4

### Summary
This paper proposes to divide the network into K smaller subnetworks and train them independently.
In terms of architecture the paper assumes transformer-like residual networks, which are composed of a sequence of blocks.
A diffusion objective is used and noise level intervals are assigned to to each subnetwork.
During training, each subnetwork is trained alone and independently to denoise data in its assigned noise range.
The blocks are adapted so that their residual nature can be reframed as a denoising update of the form $z_{i+1}=z_i + \mu_{i+1} f(z_i)$, and in addition their input and ouput shapes should match the input data shape.
During inference, following the noise schedule, subnetworks for denoising are selected on the criteria that the current noise levels falls into their assigned noise range.
The proposed method makes both training and inference faster.
In addition this makes training scalable to larger architectures which as whole couldn't fit in current GPU memory but for which subnetworks can fit in memory.

The noise level assignment is done such that each network receives on average the same number of training noisified samples.

In addition, the method also targets recurrent-depth models.

### Strengths
1. The method looks simple and easy to implement.
2. The method targets an important problem: scalability and training/inference speed of large models.
3. For the most part, the paper is well written and easy to read.

### Weaknesses
1. However the notation sometimes lacks clarity, for example it's unclear what are $(x,y)$. From section 2.1, it appears that $y$ is data (e.g. an image for example). So that would make $x$ a label supposedly, however in Fig.3 $z_0\leftarrow x$ which suggest data rather than label. In Figure 2, in the case of the classifier, it would like $y$ is actually a label. A simple fix would be to clearly specify what $x$ and $y$ are and keep the notation consistent thorough the entire paper.
2. The results seem to compare to weak baselines, for example CIFAR100 classification baseline has an accuracy of 45% while SotA easily exceeds 80% if memory serves me well. Similarly DiT FID on CIFAR10 is reported at 39.83 while a google search reveals people obtaining FIDs within 10-15 on the DiT GitHub repo.
3. Nit, provide important information in the paper when possible, for example I had to dig into appendix D2 to find the resolution used for ImageNet (aka 256).
4. CIFAR10 is a surprising choice for benchmarking the method given that it's well known that CIFAR10 is subject to overfitting and uses heavy regularization whether in classification or diffusion. Therefore, given your method spreads the capacity over each subnetwork, it is unclear how much can be really explained by your $B$ ablation in table 8. I believe this ablation should be done on either ImageNet64 or 256.
5. On ImageNet256, the FID numbers are pretty high (10-12), it appears you DiT-L/2 while the DiT paper has most comparison for XL/2 (which gets an FID around 10), digging into the DiT paper appendix, the reported FID without CFG for DiT-L/2 is 23.33. I am confused, did you run DiT-L/2 or DiT-XL/2 as your results don't seem to match the ones reported by the original paper.

### Questions
1. In Fig.2 I am confused by your drawing, for step 3, you extract a block in the middle and refer to its output which is fed as an input with noise. What output are we talking about here, the actual output of the subnetwork or the final expected output of the whole network? From your algorithm in Fig.3 (the orange box), it appears it's the expected output $y$.
2. In 5.1 (ViT for image classification), you mentioned that noise is added to class label embeddings:
2.1 Where do class label embeddings come from, or are they really one-hot representations?
2.2 What type of noise is added to these embeddings? From Figure 2 it appears you add Gaussian noise to one-hot labels, is that right?
3. More generally, by slicing a network into K subnetworks, it stands to reason that each subnetwork has less capacity (given that the total number of weights remains about the same). How do you explain that your reported results in tables (1,2,3,4,5) are better than the full models they compare too while at the same much faster to train and with less capacity per subnetwork?
4. Why not also reporting performance with CFG on ImageNet256? Does your method works with CFG?

### Soundness
2

### Presentation
2

### Contribution
3
