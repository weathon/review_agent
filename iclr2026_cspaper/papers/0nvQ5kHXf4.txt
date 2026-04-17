# Efficient Resource-Constrained Training Of Transformers Via Subspace Optimization

Le-Trung Nguyen Enzo Tartaglione Van-Tam Nguyen LTCI, Tel´ ecom Paris, Institut Polytechnique de Paris, France ´
{name.surname}@telecom-paris.fr

## Abstract

As AI increasingly shapes daily life, energy consumption and data privacy have become pressing concerns. On-device learning trains models directly on edge devices, cutting energy consumption and safeguarding data privacy. However, the expanding scale of modern neural networks creates a major obstacle for ondevice training. Although prior work has concentrated on compact convolutional architectures, we instead apply subspace-based training to transformer models. Motivated by the idea that a model's essential information lies in a fixed subspace, we introduce Weight-Activation Subspace Iteration (WASI), a method that mitigates the memory bottleneck of backpropagation and boosts inference efficiency in transformer models by restricting training to this subspace. Our results demonstrate that WASI maintains accuracy comparable to vanilla training while reducing memory usage by up to 62× and computational cost (FLOPs) by up to 2×. On a Raspberry Pi 5, WASI achieves roughly 1.4× faster training and inference than vanilla training. The code is available at https://github.com/Le- TrungNguyen/ICLR2026-WASI.git.

## 1 Introduction

On-device learning has recently emerged as a promising research direction, enabling deep learning models to be finetuned directly on resource-constrained edge devices. This approach addresses critical issues such as privacy and energy consumption, improves scalability, and places control of AI capabilities directly "in user's hands" (Dhar et al., 2021). Prior work on on-device learning has largely focused on vision tasks using convolutional neural network models, primarily because of their compact architectures (Lin et al., 2022; Nguyen et al., 2024; Yang et al., 2023b; Quelennec et al., 2024; Bragagnolo ´ et al., 2022; Nguyen et al., 2025). In many real-world applications, however, transformer-based models have become the de facto choice due to their unique architectural mechanisms (Vaswani et al., 2017). Specifically, these models employ efficient forward propagation through the composition of linear layers, process large-scale data in parallel, and alleviate the vanishing gradient problem thanks to self-attention - key advantages that make them well-suited for handling long-range dependencies, whether in extended text sequences or high-resolution images. Notable examples of such models include GPT (Brown et al., 2020), Gemini (Team et al., 2023), LLaMA (Touvron et al., 2023), and DeepSeek (Liu et al., 2024a). Nevertheless, these mechanisms make training and deployment of transformer models resource-intensive. This is even worse when considering the ondevice learning context, where models need to be trained on separate edge devices and are often resource-constrained.

Figure 1: Overview of WASI in a single training iteration.

![0_image_0.png](0_image_0.png)

1 A significant fraction of training costs arises from backpropagation, especially the memory and computations needed for storing tensors in model layers (Lin et al., 2022). Various research has emerged to address the inefficiencies of backpropagation and enable learning directly on devices. For instance, Lin et al. (2022) demonstrated the feasibility of fine-tuning a predefined subnetwork under a 256KB memory constraint device while still maintaining competitive performance. Quelennec ´ et al. (2024) took this further by dynamically adapting the subnetwork during training rather than relying on a static one, leading to better accuracy within tight memory budgets. Beyond the scope of on-device learning, many methods aim to reduce training overhead through parameter-efficient approaches, such as LoRA (Hu et al., 2022) and its variants (Xu et al., 2023; Zhang et al., 2023; Hayou et al., 2024; Liu et al., 2024b). While these techniques successfully limit the number of parameters updated at training time, they often overlook the cost of storing intermediate calculations (activation maps). Nguyen et al. (2024) address this by compressing activation maps under a controlled information-loss constraint, but lack robust memory budget control and incur considerable compression overhead. None of these methods enhances the neural architecture itself, and inference proceeds as usual, resulting in high deployment costs on edge devices. This issue has been further addressed by ASVD (Yuan et al., 2023) and FWSVD (Hsu et al., 2022), which employ truncated Singular Value Decomposition (SVD) to decompose the model architecture, but lack a theoretical basis for choosing which singular values to truncate. Subsequently, SVD-LLM (Wang et al., 2024) was developed to overcome this limitation and outperforms the aforementioned approaches. However, these methods are specifically designed for large language models (LLMs) and are not readily applicable to all vision transformer-based models (see Appendix. A.4). Another similar effort, ESPACE (Sakr & Khailany, 2024), requires access to a downstream dataset, which is not feasible in on-device learning scenarios. Inspired by prior studies on the stability of parameter subspaces during fine-tuning (Radiya-Dixit & Wang, 2020; Li & Zhang, 2021), we present WASI (Fig. 1), the first method for efficient modelactivation-decomposition-aware training. WASI enables transformer models to be fine-tuned and executed entirely in a low-rank representation, substantially reducing hardware costs and making vision transformer tasks feasible on edge devices. We assess its effectiveness on vision transformer models, including the Swin Transformer (SwinT) (Liu et al., 2021), the Vision Transformer (ViT) (Dosovitskiy et al., 2020), and even TinyLlama (Zhang et al., 2024). Our main contributions are summarized as follows.

- Based on the previous studies, we formulate that the essential information of a model parameters resides in a stable subspace throughout fine-tuning (Sec. 3.3), which is then verified in Sec. 4.2.

- Leveraging this hypothesis, we propose Weight-Activation Subspace Iteration (WASI) in Sec. 3.3 to effectively compress the model architecture under a controlled information-loss constraint.

- We showcase the effectiveness of our approach through extensive experiments on multiple tasks (Sec. 4.3 and Sec. 4.4).

## 2 Related Works

In this section, we review low-rank decomposition techniques as applied to two key components of deep learning models: model weights and activation maps. Other research directions such as compact model design, quantization, sparsification, and knowledge distillation also exist, but they fall outside the scope of this work–*low-rank decomposition* (Cheng et al., 2017; Deng et al., 2020).

Therefore they are not discussed here (see Appendix A.5 for details).

Low-rank Decomposition for Model Weights. Low-rank approximation methods for model weights have been extensively studied and can generally be categorized into two main approaches:
Low-rank Adapters and *Low-rank Models*. LoRA (Hu et al., 2022) is the most prominent example of the first category, which introduces an additional low-rank adapter while freezing the original model architecture. This strategy can reduce the number of trainable parameters by up to four orders of magnitude, but comes with two notable drawbacks. During training, memory usage grows because both the frozen weights and the new adapter must co-exist in memory. At inference time, the adapter is merged back into the model, resulting in inference performance that is identical to the original model, and thus losing the computational advantages of low-rank decomposition. Low-rank Models are an alternative line of research that factorizes the weight matrices themselves and trains only the low-rank components, enabling inference to run directly on the compressed representation. Methods such as ASVD (Yuan et al., 2023) and FWSVD (Hsu et al., 2022) achieve this by applying truncated SVD to each layer. These approaches, however, lack a theoretical link between the truncation loss and model performance loss, which is latter addressed by SVD-LLM (Wang et al., 2024). It is important to note that, except for SVD-LLM, all aforementioned methods are specifically tailored for LLMs, and even SVD-LLM cannot be directly applied to all vision transformer-based models with activation maps of four or more dimensions (see Appendix A.4). Low-rank Decomposition for Activation Maps. In addition to model weights, activation maps are a major contributor to memory consumption during training. Gradient Filter (Yang et al., 2023b) is a pioneering work that addresses this issue in on-device learning by generating approximated versions of activation maps through pooling operations with a predefined patch size, aiming to reduce memory usage and FLOPs during fine-tuning. However, this method is limited to convolutional models, and also has the drawback of the accumulated errors as fine-tuning progresses deeper into the model (Nguyen et al., 2024). To overcome this drawback, Nguyen et al. (2024) introduced Activation Map Compression (AMC), which applies High-Order Singular Value Decomposition
(HOSVD) to compress activation maps while controlling the information loss via a threshold parameter ε. While AMC achieves impressive memory savings up to 120×, it incurs significant computational overhead due to the need for full HOSVD at every iteration. Additionally, the varying ranks required to meet the error threshold lead to fluctuating memory usage, which complicates deployment on devices with fixed memory budgets. Activation Subspace Iteration (ASI) (Nguyen et al., 2025) addresses both of these issues. Instead of controlling the reconstruction error, ASI fixes the activation ranks using a perplexity-based heuristic. This approach stabilizes memory usage throughout fine-tuning and allows for replacing the expensive HOSVD with subspace iteration. As a result, ASI preserves the high compression ratio of AMC
while reducing computational cost by up to 252.65×. On a Raspberry Pi 5, fine-tuning with ASI is 1.56× faster than vanilla training when being tested on a highly compact convolutional model.

Beyond this scope, LBP-WHT (Yang et al., 2023b) has also been explored. However, it focuses solely on reducing computational cost during training by applying the Walsh-Hadamard Transformation to tensors in gradient computations, and does not address memory bottlenecks. Our proposed WASI overcomes the limitations posed by prior works. Hypothesizing the stability of the essential subspace of model weights, we introduce a novel method that simultaneously compresses the model architecture and activation maps while carefully controlling information loss throughout the fine-tuning process. This capability makes it feasible to fine-tune transformer-based models in on-device learning scenarios.

## 3 Method

In this section, we first identify the computational bottlenecks of training and inference (Sec. 3.1). Next, we review how activation maps can be efficiently compressed (Sec. 3.2). We then introduce a compression-aware-training strategy for both model weights and activation maps that controls information loss (Sec. 3.3). Finally, we analyze the computational complexity of our method and discuss its practical advantages (Sec. 3.4).

## 3.1 Bottlenecks In Training And Inference

Consider a deep transformer-based model, where i denotes the index of a linear layer. This layer is represented by a weight matrix Wi ∈ R
Oi×Ii, which takes as input a tensor Ai ∈ R
B×Ni×Ii and produces an output tensor Ai+1 ∈ R
B×Ni×Oi. Here, B is the batch size, Niis the sequence length
(or number of tokens), Iiis the input feature dimension, and Oiis the output feature dimension. We denote the dimensionality of the input as Di = {B, Ni, Ii}.

During the forward pass (similarly in inference), the output of this layer is computed as:
Ai+1 = AiW⊤
i, (1)
Algorithm 1 Weight Subspace Iteration - WSI at iteration t

1: **Input:**
Weight Wi,(t) at iteration t, Explained variance threshold ε ∈ [0, 1].
2: **Function:** 3: if t = 0 **then**
4: Li,(t), Ri,(t) = SVD Wi,(t), ε(see Eq. 5, Eq. 6, and Eq. 7)
5: **else**
6: RT
i,(t) = WT
i,(t)· Li,(t−1)
7: Li,(t) = Orthogonalize Wi,(t)· RT
$$\mathbf{\Phi}_{i,(t)}^{-1,(t-1)}$$ $$=\mathrm{Orthogonalize}\left(\mathcal{W}_{i,(t)}\cdot R_{i,(t)}^{T}\right)$$
(Using Gram-Schmidt)
8: **endif**
  ## 1 Introduction  The _quantum_ quantum mechanics is a quantum field theory of quantum mechanics. It is a quantum field theory of quantum mechanics.  
$\square$
$\mathbf{u}$
where ⊤ denotes the matrix transpose. Eq. 1 presents a batch matrix multiplication applied over the last two dimensions of Ai; that is, for each sample in the batch and each token, a matrix multiplication is performed between a 1 × Ii vector and the transposed weight matrix of size Ii × Oi.
Similarly, in the backward pass the chain rule of backpropagation is computed as follows:
∂L ∂Wi =∂L ∂Ai+1 ⊤ · ∂Ai+1 ∂Wi =∂L ∂Ai+1 ⊤ ∂L ∂Ai =∂L ∂Ai+1 · ∂Ai+1 ∂Ai =∂L ∂Ai+1
· Ai, (2)
(see Eq. 5, Eq. 6, and Eq. 7) 
$$(2)$$
$$({\mathfrak{I}})$$
· Wi, (3)
where L is the loss computed at the output of the model. Apparently, to compute ∂L
∂Wi and ∂L
∂Ai during the backward pass, Ai and Wi must be stored during the forward pass. The large size of these tensors is the primary cause of memory bottlenecks during backpropagation (Lin et al., 2022).

Additionally, it also contributes to high inference costs, as multiplying between large Wi and Ai requires significant computational resources.

## 3.2 Activation Subspace Iteration

Here, we recap how activation maps can be decomposed by subspace iteration. Given an activation memory budget B, ASI performs brute-force optimization before fine-tuning to find an optimal rank vector ri ∈ N
3for each layer such that the resulting memory does not exceed B. Then, for each mode m ∈ {1, 2, 3}, the activation map Aiis unfolded into a matrix Ai,m ∈ R
ai,m×bi,m, where
(ai,m, bi,m) = Di,m,Qj̸=m Di,j.

Vogels et al. (2019) showed that warm-started subspace iteration matches SVD performance on stable tensors at much lower cost. Exploiting the stability of activation maps during fine-tuning, ASI applies this technique to each Ai,m. The resulting approximation takes the form of a Tucker decomposition (Tucker, 1966):

$$\mathcal{A}_{i}\approx\tilde{S}_{i}\times_{1}\tilde{U}_{i}^{(1)}\times_{2}\tilde{U}_{i}^{(2)}\times_{3}\tilde{U}_{i}^{(3)},$$

i, (4)
where S˜i ∈ R
ri,1×ri,2×ri,3is the core tensor, representing a compressed version of Ai, and each factor matrix U˜
(m)
i ∈ R
ai,m×ri,m contains the principal components along the mth mode.

Consequently, instead of storing all Θspace Q3m=1 Di,melements of Ai, ASI reduces the storage requirement to Θspace Q3m=1 ri,m +P3m=1 Di,mri,m.

Details of the algorithm can be found in Appendix A.2. 3.3 WEIGHT - ACTIVATION SUBSPACE ITERATION Stability of Model Parameters Subspace. While prior work has shown that over-parameterized models in fact reside in a low-dimensional intrinsic subspace (Aghajanyan et al., 2020; Li et al.,

$$(4)$$

![4_image_0.png](4_image_0.png)

$$(S)$$

Figure 2: For the linear layer i with a single data batch of size B, given varying dimensions of Wi and Ai and different values of ri,m, Ctraining and Cinference illustrate the evolution in compression rates for training and inference, respectively; while Straining and Sinference forecast the speedup ratios for these processes.

2018), we further observe that fine-tuning introduces only minor updates at each training step due to the use of a small learning rate. As a result, our key insight is that the intrinsic subspace remains relatively stable after each training iteration and can therefore be reused in the following one (confirmed in Sec. 4.2 - Fig. 3). This is supported by the findings of Radiya-Dixit & Wang (2020) and Li & Zhang (2021), who showed that the fine-tuned models are close in parameter space to the pre-trained counterpart.

Weight Subspace Iteration. Besides activation maps, model parameters (weights) Wi are another major source of memory bottlenecks during training. To address this, we propose a low-rank weight decomposition strategy that projects each weight tensor into a smaller subspace at every training iteration, thereby preserving the meaningful subspace. The method works as follows:
Step 1. For the weight tensor Wi at layer i, its SVD form is given by:

$$\mathcal{W}_{i}=U_{i}\Sigma_{i}V_{i}^{T},\qquad U_{i}\in\mathbb{R}^{O_{i}\times O_{i}},\quad\Sigma_{i}\in\mathbb{R}^{O_{i}\times I_{i}},\quad V_{i}\in\mathbb{R}^{I_{i}\times I_{i}},$$

where Σiis a diagonal matrix containing ri singular values si,j∈[1,ri], and riis the rank of Wi.

As shown in Eq. 3, truncating Ui, Σi, and V
T
iinevitably introduces error into ∂L
∂Ai
, which then propagates backward during training. In other words, low-rank decomposition of the weights affects model convergence due to the accumulation of truncation error. To control this effect, we constrain the truncation error by enforcing a target explained variance threshold ε, similar to the strategy used in Nguyen et al. (2024). Specifically, we measure the variance explained by the j th singular value as σ 2 i,j = s 2 i,j/Pks 2 i,k. Assuming the singular values are sorted in descending order (si,j ≥ si,k, ∀j ≤ k), the optimal rank is defined as the smallest integer Ki ∈ [1, ri] such that PKi j=1 σ 2 i,j ≥ ε. We then identify the essential subspace with rank Ki of Wi, represented by Li and Ri such that:

$${\mathcal W}_{i}\approx{\bar{\mathcal W}}_{i}=L_{i}R_{i},$$

Wi ≈ W˜i = LiRi, (6)
$${\mathrm{where}}$$
(7)
Step 2. Performing full SVDs at every iteration, however, is computationally prohibitive for on-
$$L_{i}=U_{i_{i}(K_{i})}\Sigma_{i_{i}(K_{i})},\quad R_{i}=V_{i_{i}(K_{i})}^{T}\mid U_{i_{i}(K_{i})}\in\mathbb{R}^{O_{i}\times K_{i}},\quad\Sigma_{i_{i}(K_{i})}\in\mathbb{R}^{K_{i}\times K_{i}},\quad V_{i_{i}(K_{i})}\in\mathbb{R}^{I_{i}\times K_{i}}.\tag{7}$$
device training (Nguyen et al., 2025). Leveraging the stability of parameter subspaces established above, Σi can be expected to remain relatively stable. Thus, for a fixed ε, the optimal rank Ki should also remain consistent (verified in Sec. 4.2). Consequently, instead of recomputing the SVD at every iteration, we compute it once at the beginning to determine the essential subspace. Subspace
iteration is applied during training to minimize computational overhead. We refer to this method as
Weight Subspace Iteration (WSI), with the full procedure outlined in Algorithm 1. Weight-Activation Subspace Iteration. While WSI reduces weight-related overhead, activation maps also dominate memory usage in backpropagation (Sec. 3.1). Previous work has shown that most of the energy in activation maps is concentrated in the first few principal components across all modes (Nguyen et al., 2024). Such a distribution makes them highly compressible while
$$(6)$$
still achieving high-fidelity reconstruction (confirmed in Sec. 4.2 - Fig. 4 and Sec. 4.3). Motivated by this property, we propose a unified framework in which both weights and activations are compressed under stable low-rank subspaces. Specifically, we redesign ASI with two improvements: (i)
a dynamic-programming strategy that determines ri by minimizing memory usage under a target pre-tuning perplexity, rather than relying on a fixed budget B, thereby reducing the search cost from exponential to linear (Appendix A.2); and (ii) an extension to support 3D activation tensors (Appendix A.1). Together, WSI and ASI form the proposed Weight-Activation Subspace Iteration (WASI), a novel framework for low-rank training that jointly leverages the stability of both weights and activations. Under this scheme, the forward and backward passes are computed as follows:

Under this scheme, the reward and backward phase is computed as follows:  $$A_{i+1}=A_{i}R_{i}^{T}L_{i}^{T},\tag{8}$$ $$\frac{\partial\widetilde{\mathcal{L}}}{\partial W_{i}}=f_{\text{LR}}\left(\widetilde{A}_{i}\cdot\frac{\partial\widetilde{\mathcal{L}}}{\partial\widetilde{A}_{i+1}}\right),$$ (9) $$\frac{\partial\widetilde{\mathcal{L}}}{\partial\widetilde{A}_{i}}=\frac{\partial\widetilde{\mathcal{L}}}{\partial\widetilde{A}_{i+1}}\cdot L_{i}R_{i},\tag{10}$$  where $f_{\text{LR}}(.)$ denotes a linear operator applied in the low-rank space (see Appendix A.1). With
(8)  $\binom{9}{2}$  (9)  ×
$$(10)$$
learning rate η, the weight update is then computed as:
Then computed as:  $L_iR_i=L_iR_i+\eta\cdot\dfrac{\widehat{\partial\mathcal{L}}}{\partial\mathcal{W}_i}$. 
$$(11)$$

## 3.4 Memory Efficiency And Computational Complexity Analysis

For simplicity, we assume that the same optimal rank is applied to both Ai and Wi. By varying this value, we can predict total memory usage and speedup for WASI compared to vanilla training (Fig. 2). As model size grows and the optimal rank decreases, WASI delivers greater memory compression (Ctraining, Cinference) and speedup (Straining, Sinference), a property especially valuable in on-device learning where models are typically over-parameterized and reside in low-dimensional subspaces (Aghajanyan et al., 2020; Li et al., 2018). Conversely, as the optimal rank increases, WASI's computational cost approaches that of vanilla training, and the speedup ratios converge to 1, reflecting the upper bound set by vanilla training.

Detailed derives of Ctraining, Cinference, Straining, and Sinference can be found in Appendix A.3.

## 4 Experiments

In this section, we present experiments designed to demonstrate the effectiveness of WASI. We begin by outlining the experimental setup in Sec. 4.1. Then, in Sec. 4.2, we conduct experiments to validate the assumptions introduced in Sec. 3.3 and Sec. 3.3. Sec. 4.3 compares WASI with various state-of-the-art methods across multiple datasets. Finally, all methods are evaluated in a real-world deployment scenario (Sec. 4.4. All simulation experiments are conducted using PyTorch 1.13.1 on an NVIDIA Quadro RTX A4500 with 20 GB of VRAM, while on-device experiments are run on a Raspberry Pi 5 equipped with a Cortex-A76 CPU and 8 GB of RAM.

## 4.1 Experimental Setup

Our goal is to enable on-device training of transformer models, where networks pretrained on largescale datasets are fine-tuned locally with task-specific data (Murshed et al., 2021). We evaluate WASI on image classification using ViT and SwinT, both pretrained on ImageNet-1K (Deng et al., 2009), across five downstream datasets: CIFAR-10/100 (Krizhevsky, 2009), CUB (Wah et al., 2011), Flowers (Nilsback & Zisserman, 2008), and Pets (Zhang et al., 2022).

Comparisons are made against three directly comparable baselines at the time of conducting experi ments: ASI, SVD-LLM, and vanilla training (as discussed in Secs. 1, 2, Appendix A.5). We measure memory and computation costs during training and inference, focusing on linear layers within multi-perceptron blocks for fair comparison with previous methods (extended results with attention layers in Appendix B.3). All experiments are run with the same set of hyperparameters, detailed in Appendix B.1.

![6_image_0.png](6_image_0.png)

![6_image_1.png](6_image_1.png)

## 4.2 Preliminary Results

In these experiments, we focus on fine-tuning ViT model using Pets dataset. Stability of Layer Ranks. We apply truncated SVD to the weight tensors of the linear layers within ViT's MLP blocks at each training iteration. We constrain the decomposition by setting ε = 0.8 and monitor the layer ranks Kithroughout the course of training. As shown in Fig. 3a, we observe that the ranks exhibit remarkable stability across epochs. This observation validates our insight in Sec. 3.3, confirming the stability of layer ranks during training. WSI vs SVD. Next, we compare two strategies: (1) reapplying truncated SVD at every training iteration, and (2) WSI. We evaluate their performance across a range of ε values - specifically, 0.4, 0.5, 0.6, 0.7, 0.8, and 0.9 - with each value represented by a different marker in Fig. 3b. The results demonstrate that incorporating subspace iteration through WSI leads to a significant reduction in computational complexity compared to performing a full SVD at every iteration. Specifically, WSI
requires 1.36× fewer FLOPs than SVD to achieve the same level of accuracy. Moreover, when both methods are constrained to use the same amount of FLOPs, WSI outperforms SVD by approximately 35% in terms of accuracy. This result verifies that reusing the subspace in subsequent training iterations does not degrade model convergence. Explained Variance Distribution of Activation Maps. Fig. 4 illustrate the explained variances σ*i,j,m* of each singular value j in mode m of the activation map Ai. As anticipated in Sec. 3.3, most activation-map energy lies in the first few singular values, which capture the key information during fine-tuning.

![7_image_0.png](7_image_0.png)

![7_image_1.png](7_image_1.png)

## 4.3 Main Results

ViT on CIFAR-10. Fig. 5 presents the results of fine-tuning a ViT pretrained on ImageNet-1K using CIFAR-10. Each curve for WASI and ASI contains six markers, corresponding to explained variance thresholds ε ∈ {0.4, 0.5, 0.6, 0.7, 0.8, 0.9} from left to right. The red diamond indicates vanilla training, and for fairness, the same compression ratios are applied to SVD-LLM.

WASI achieves up to 100× higher memory efficiency than SVD-LLM at similar accuracy, owing to its avoidance of LoRA adapters. Its accuracy also improves steadily as ε increases. In contrast, at the lowest compression rates (last two markers), SVD-LLM consumes even more memory than vanilla training because of the overhead of storing sub-layer activations. In terms of computation, LoRA adapters allow SVD-LLM to achieve the lowest FLOPs, followed by WASI, which jointly compresses weights and activations into a low-rank subspace. Since ASI only compresses activations while keeping weights intact, its computational cost is higher, and at ε = 0.9, it even exceeds vanilla training (confirmed in Tab. 2). On the other hand, ASI maintains stable accuracy across compression rates, supporting the stability assumption discussed in Sec. 3.3. At inference, both WASI and SVD-LLM achieve similar memory/FLOPs savings, while ASI resembles vanilla since the architecture is unchanged. SwinT on Multiple Datasets. Fig. 6 compares WASI and vanilla across datasets, additional baselines are in Appendix B.3. Each marker along a curve from left to right, indicates different ε ∈ {0.4*, . . . ,* 1.0}, with 1.0 as vanilla. Across all datasets, WASI consistently provides a better accuracy-efficiency trade-off. At ε = 0.9, it matches vanilla accuracy while cutting memory by up to 62× and FLOPs by 1.5×, and even surpasses vanilla on CUB.

WASI on TinyLlama. The initial goal of WASI was to enable training transformer-based models on edge devices, so we focused on ViT and SwinT. To test its generality, we extended our experiments to TinyLlama, a decoder-only transformer model. The downstream dataset used is BoolQ (Clark et al., 2019). Due to limited resources, we only fine-tune up to the last 5 layers of the model and set the WASI ε to 0.1. All other training hyperparameters followes the same configuration as in our previous experiments. For comparison, we log the resource consumption only at the layers that are fine-tuned. The results are shown in Fig. 7.

![8_image_0.png](8_image_0.png)

WASI again outperforms vanilla: activation and weight memory drop by up to 953.86× and 30.12×, while training and inference FLOPs fall by 13.11× and 30.27×, all without accuracy loss.

Additional results, including ViT on more datasets and extended baselines for SwinT are in Appendix B.3.

## 4.4 On-Device Latency

![8_image_1.png](8_image_1.png)

We evaluate the practical efficiency of WASI on resource-constrained hardware by fine-tuning ViT on CIFAR-10 using a Raspberry Pi 5. Fig. 8 reports the average time required to complete a single iteration of both training and inference across different explained variance thresholds ε ∈ {0.4, 0.5, 0.6, 0.7, 0.8, 0.9},
along with vanilla training.

As expected, the runtime for both training and inference under WASI increases as ε becomes larger. This trend aligns with the intuition that higher ε values retain more information and thus result in higher-rank approximations, which require more compute and memory. However, despite this increase, WASI consistently outperforms vanilla training in terms of speed. For instance, even at ε = 0.9, which corresponds to the least aggressive compression setting in this experiment, WASI remains approximately 1.4× faster than vanilla training. Thus, WASI delivers clear benefits even when preserving much of the original information. Importantly, WASI helps to reduce runtime without causing significant accuracy degradation, as discussed in earlier sections. This ability makes it a strong candidate for the deployment of transformerbased model in real-world on-device learning scenarios, where computational resources are severely constrained. Further numerical results can be found in Appendix. B.3.

Figure 8: Training and inference time per iteration for ViT on CIFAR-10 (batch size = 128) using a Raspberry Pi 5, measured under different explained variance thresholds ε. The final marker on each curve represents vanilla training.

## 5 Conclusion

In this work, we introduced WASI, an efficient training method for resource-constrained finetuning of transformer models. Assuming that essential parameter information lies in a stable lowdimensional subspace, WASI applies SVD and subspace iteration to obtain low-rank approximations of both weights and activations during each training iteration. This yields significant gains in memory and computation while tightly controlling information loss. Building on prior theory and validated through extensive experiments, WASI outperforms stateof-the-art methods, reducing training memory usage by up to 62× and achieving 1.4× speedup over vanilla training on a Raspberry Pi 5. These results show the potential of WASI for enabling on-device learning with transformers, a domain traditionally dominated by CNNs. While our experiments focus on transformers, the underlying principles apply broadly to any neural network trained with backpropagation.

## Acknowledgement

Part of this work was funded by Hi!PARIS Center on Data Analytics and Artificial Intelligence, by the European Union's Horizon Europe Research and Innovation Programme under grant agreement No. 101120237 (ELIAS - European Lighthouse of AI for Sustainability) and No. 101120657 (ENFIELD - European Lighthouse to Manifest Trustworthy and Green AI), by the French National Research Agency (ANR) in the framework of the IA Cluster project "Hi! PARIS Cluster 2030" (ANR-23-IACL-005), the NF-NAI project (ANR-22-PEFT-0003) and NF-FITNESS project (ANR- 22-PEFT-0007) as part of France 2030.

## Reproducibility Statement

Detailed description of our algorithm is provided in Sec. 3.3, Appendix A.1, and Appendix A.2. Full details of the training policy, including hyperparameters, datasets, and other configurations, are presented in Appendix B.1. Code to reproduce the main experiments is included in the Supplementary Material zip file. We commit to open-sourcing the complete code upon acceptance of this paper.

## References

Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. *arXiv preprint arXiv:2012.13255*, 2020.

Andrea Bragagnolo, Enzo Tartaglione, and Marco Grangetto. To update or not to update? neurons at equilibrium in deep models. *Advances in neural information processing systems*, 35:22149–
22160, 2022.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.

Yu Cheng, Duo Wang, Pan Zhou, and Tao Zhang. A survey of model compression and acceleration for deep neural networks. *arXiv preprint arXiv:1710.09282*, 2017.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern recognition*, pp. 248–255. Ieee, 2009.

Lei Deng, Guoqi Li, Song Han, Luping Shi, and Yuan Xie. Model compression and hardware acceleration for neural networks: A comprehensive survey. *Proceedings of the IEEE*, 108(4): 485–532, 2020.

Sauptik Dhar, Junyao Guo, Jiayi Liu, Samarth Tripathi, Unmesh Kurup, and Mohak Shah. A survey of on-device machine learning: An algorithms and learning theory perspective. *ACM Transactions* on Internet of Things, 2(3):1–49, 2021.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Marawan Gamal Abdel Hameed, Marzieh S Tahaei, Ali Mosleh, and Vahid Partovi Nia. Convolutional neural network compression through generalized kronecker product decomposition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 771–779, 2022.

Soufiane Hayou, Nikhil Ghosh, and Bin Yu. Lora+: Efficient low rank adaptation of large models.

arXiv preprint arXiv:2402.12354, 2024.

Yen-Chang Hsu, Ting Hua, Sungen Chang, Qian Lou, Yilin Shen, and Hongxia Jin. Language model compression with weighted low-rank factorization. *arXiv preprint arXiv:2207.00112*, 2022.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. *ICLR*, 1(2):3, 2022.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. Measuring the intrinsic dimension of objective landscapes. *arXiv preprint arXiv:1804.08838*, 2018.

Dongyue Li and Hongyang Zhang. Improved regularization and robustness for fine-tuning in neural networks. *Advances in Neural Information Processing Systems*, 34:27249–27262, 2021.

Ji Lin, Ligeng Zhu, Wei-Ming Chen, Wei-Chen Wang, Chuang Gan, and Song Han. On-device training under 256kb memory. *Advances in Neural Information Processing Systems*, 35:22941– 22954, 2022.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024a.

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-
Ting Cheng, and Min-Hung Chen. Dora: Weight-decomposed low-rank adaptation. In Forty-first International Conference on Machine Learning, 2024b.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo.

Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021.

Ivan Markovsky. Structured low-rank approximation and its applications. Automatica, 44(4):891–909, 2008. ISSN 0005-1098. doi: https://doi.org/10.1016/j.automatica.

2007.09.011. URL https://www.sciencedirect.com/science/article/pii/ S0005109807003950.

MG Sarwar Murshed, Christopher Murphy, Daqing Hou, Nazar Khan, Ganesh Ananthanarayanan, and Faraz Hussain. Machine learning at the network edge: A survey. ACM Computing Surveys (CSUR), 54(8):1–37, 2021.

Le-Trung Nguyen, Ael Qu ¨ elennec, Enzo Tartaglione, Samuel Tardieu, and Van-Tam Nguyen. Ac- ´
tivation map compression through tensor decomposition for deep learning. Advances in Neural Information Processing Systems, 37:130384–130407, 2024.

Le-Trung Nguyen, Ael Qu ¨ elennec, Van-Tam Nguyen, and Enzo Tartaglione. Beyond low-rank de- ´
composition: A shortcut approach for efficient on-device learning. In Forty-second International Conference on Machine Learning, 2025.

Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In *Indian Conference on Computer Vision, Graphics and Image Processing*, 2008.

Ael Qu ¨ elennec, Enzo Tartaglione, Pavlo Mozharovskyi, and Van-Tam Nguyen. Towards on-device ´
learning on the edge: Ways to select neurons to update under a budget constraint. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 685–694, 2024.

Evani Radiya-Dixit and Xin Wang. How fine can fine-tuning be? learning efficient language models.

In *International Conference on Artificial Intelligence and Statistics*, pp. 2435–2443. PMLR, 2020.

Charbel Sakr and Brucek Khailany. Espace: Dimensionality reduction of activations for model compression. *arXiv preprint arXiv:2410.05437*, 2024.

GW Stewart and JH Miller. Methods of simultaneous iteration for calculating eigenvectors of matrices. *Topics in Numerical Analysis II*, 2, 1975.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee´
Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and ` efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.

Ledyard R Tucker. Some mathematical notes on three-mode factor analysis. *Psychometrika*, 31(3):
279–311, 1966.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Aladin Virmaux and Kevin Scaman. Lipschitz regularity of deep neural networks: analysis and efficient estimation. *Advances in Neural Information Processing Systems*, 31, 2018.

Thijs Vogels, Sai Praneeth Karimireddy, and Martin Jaggi. Powersgd: Practical low-rank gradient compression for distributed optimization. *Advances in Neural Information Processing Systems*,
32, 2019.

Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset, 2011.

Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang. Svd-llm: Truncation-aware singular value decomposition for large language model compression. *arXiv preprint arXiv:2403.07378*, 2024.

Ou Xinwei, Chen Zhangxin, Zhu Ce, and Liu Yipeng. Low rank optimization for efficient deep learning: Making a balance between compact architecture and fast training. Journal of Systems Engineering and Electronics, 2023.

Yuhui Xu, Lingxi Xie, Xiaotao Gu, Xin Chen, Heng Chang, Hengheng Zhang, Zhengsu Chen, Xiaopeng Zhang, and Qi Tian. Qa-lora: Quantization-aware low-rank adaptation of large language models. *arXiv preprint arXiv:2309.14717*, 2023.

Jian Xue, Jinyu Li, and Yifan Gong. Restructuring of deep neural network acoustic models with singular value decomposition. In *Interspeech*, pp. 2365–2369, 2013.

Yuedong Yang, Hung-Yueh Chiang, Guihong Li, Diana Marculescu, and Radu Marculescu. Efficient low-rank backpropagation for vision transformer adaptation. *Advances in Neural Information* Processing Systems, 36:14725–14736, 2023a.

Yuedong Yang, Guihong Li, and Radu Marculescu. Efficient on-device training via gradient filtering.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp.

3811–3820, 2023b.

Zhihang Yuan, Yuzhang Shang, Yue Song, Qiang Wu, Yan Yan, and Guangyu Sun. Asvd:
Activation-aware singular value decomposition for compressing large language models. arXiv preprint arXiv:2312.05821, 2023.

Hui Zhang, Shenglong Zhou, Geoffrey Ye Li, and Naihua Xiu. 0/1 deep neural networks via block coordinate descent. *arXiv preprint arXiv:2206.09379*, 2022.

Longteng Zhang, Lin Zhang, Shaohuai Shi, Xiaowen Chu, and Bo Li. Lora-fa: Memory-efficient low-rank adaptation for large language models fine-tuning. *arXiv preprint arXiv:2308.03303*, 2023.

Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model. *arXiv preprint arXiv:2401.02385*, 2024.

Hengling Zhao, Yipeng Liu, Xiaolin Huang, and Ce Zhu. Semi-tensor product-based tensordecomposition for neural network compression. *arXiv preprint arXiv:2109.15200*, 2021.