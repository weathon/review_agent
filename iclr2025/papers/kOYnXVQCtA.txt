

{0}------------------------------------------------

# DEEPERFORWARD: ENHANCED FORWARD-FORWARD TRAINING FOR DEEPER AND BETTER PERFORMANCE

Liang Sun<sup>1,†</sup>, Yang Zhang<sup>1,†,\*,</sup> Weizhao He<sup>1</sup>, Jiajun Wen<sup>1</sup>, Linlin Shen<sup>1,2,3</sup>, Weicheng Xie<sup>1,2,3</sup>

<sup>1</sup>Computer Vision Institute, School of Computer Science & Software Engineering, Shenzhen University

<sup>2</sup>National Engineering Laboratory for Big Data System Computing Technology, Shenzhen University

<sup>3</sup>Guangdong Provincial Key Laboratory of Intelligent Information Processing

{sunliang, heweizhao}2022@email.szu.edu.cn,

{yangzhang, wenjiajun, llshen, wcxie}@szu.edu.cn

Code: <https://github.com/tobysunsun/deeperforward>

## ABSTRACT

While backpropagation effectively trains models, it presents challenges related to bio-plausibility, resulting in high memory demands and limited parallelism. Recently, Hinton (2022) proposed the Forward-Forward (FF) algorithm for high-parallel local updates. FF leverages squared sums as the local update target, termed goodness, and decouples goodness by normalizing the vector length to extract new features. However, this design encounters issues with feature scaling and deactivated neurons, limiting its application mainly to shallow networks. This paper proposes a novel goodness design utilizing **layer normalization** and **mean goodness** to overcome these challenges, demonstrating performance improvements even in 17-layer CNNs. Experiments on CIFAR-10, MNIST, and Fashion-MNIST show significant advantages over existing FF-based algorithms, highlighting the potential of FF in deep models. Furthermore, the model parallel strategy is proposed to achieve highly efficient training based on the property of local updates.

## 1 INTRODUCTION

Backpropagation (BP) (Rumelhart et al., 1986) has achieved significant success, serving as the prevailing paradigm for training complex structures like ResNet (He et al., 2016) and Transformers (Vaswani et al., 2017). However, no compelling evidence supports such a mechanism existing in the brain, challenging the biological plausibility of BP. Critical challenges within BP consist of weight transport (Grossberg, 1987), non-local (Whittington & Bogacz, 2019), freezing activity, and update locking problems (Jaderberg et al., 2017; Czarnecki et al., 2017). The *weight transport problem* arises from reusing the same path in forward and backward passes. The *non-local problem* arises from global objective loss, while the brain relies on local signals for updates. The *freezing activity problem* and *update locking problem* contradict the real-time property in neural systems. Freezing activity involves maintaining intermediate states, leading to increased memory demands. The update locking problem prevents any update until all layers are activated, reducing parallelism in practice.

To tackle these challenges, various brain-inspired training methods (Oorbia, 2023) have been developed to formulate a comprehensive theory of inference and learning in a biologically plausible manner (Lillicrap et al., 2016; Nøkland, 2016; Dellaerrera & Kreiman, 2022; Oorbia et al., 2023; Hinton, 2022). Several of these methods are depicted in Figure 1. A recent breakthrough is the Forward-Forward (FF) algorithm (Hinton, 2022), as depicted in Figure 1(d). FF employs the squared sum of outputs, termed *goodness*, and fixes the output vector length via dividing by its vector length, thereby decoupling goodness within the output features and compelling subsequent layers to learn new features. Mathematically, the output’s length and direction correspond to goodness and features. However, this design has limitations that confine current layer-wise FF studies to shallow models. The primary reasons why FF fails to achieve performance improvements in deeper networks are as follows:

<sup>†</sup>Equal Contribution: Liang Sun and Yang Zhang.

<sup>\*</sup>Corresponding author: Yang Zhang.

{1}------------------------------------------------

![Figure 1: Comparison of several training methods. (a) BP: Traditional forward (green) and backward (blue) passes. (b) FA: Forward pass with feedback alignment for error propagation. (c) PEPITA: Two forward passes, one with input perturbation. (d) FF: Forward pass with separate positive and negative data. (e) Ours: Simplified forward pass with a single error signal.](9ba3dc91984c80b96f217fb1bddd5c06_img.jpg)

Figure 1 illustrates five training paradigms for deep networks. (a) BP shows standard forward (green) and backward (blue) propagation. (b) FA uses a feedback loop for error propagation. (c) PEPITA uses two forward passes, one with input perturbation  $\epsilon$ . (d) FF uses separate forward passes for positive and negative data. (e) Ours simplifies this to a single forward pass with a combined error signal  $\mathcal{L}_e$ .

Diagrams (a) through (e) show different training architectures. (a) BP: Input  $x$  is processed through layers  $h_1, h_2, h_3$  to output  $y$ . Backward pass (blue) propagates error  $\delta$  from output back to input. (b) FA: Similar to BP, but backward pass uses fixed weights  $W_i^T$ . (c) PEPITA: Two forward passes. First with input  $x$ , second with perturbed input  $x+\epsilon$ . (d) FF: Two forward passes. One with positive data  $x_{pos}$ , one with negative data  $x_{neg}$ . (e) Ours: Single forward pass. Error signal  $\mathcal{L}_e$  is propagated back through layers.

Weight update equations:

- (a) BP:  $\Delta W_i = -\delta_{i+1} \cdot h_i^T$ ,  $\delta_i = W_{i+1}^T \cdot \delta_{i+1}$
- (b) FA:  $\Delta W_i = -\delta_{i+1} \cdot h_i^T$ ,  $\delta_i = B_{i+1}^T \cdot \delta_{i+1}$
- (c) PEPITA:  $\Delta W_i = (h_i - h_i^{err}) \cdot h_i^{err,T}$ ,  $h_i^{err} = W_i \cdot h_i^{err,T}$ ,  $h_i^{err} = W_i \cdot (x + F \cdot e)$
- (d) FF:  $\Delta W_i = -\nabla_{W_i} (\mathcal{L}_{pos,i} + \mathcal{L}_{neg,i})$
- (e) Ours:  $\Delta W_i = -\nabla_{W_i} \mathcal{L}_e$

Figure 1: Comparison of several training methods. (a) BP: Traditional forward (green) and backward (blue) passes. (b) FA: Forward pass with feedback alignment for error propagation. (c) PEPITA: Two forward passes, one with input perturbation. (d) FF: Forward pass with separate positive and negative data. (e) Ours: Simplified forward pass with a single error signal.

Figure 1: Comparison of several training methods. (a) BP employs traditional forward and backward passes represented by blue and green arrows respectively. (b) Feedback alignment (FA) uses an alternative backward pass for error passes. (c) PEPITA uses two forward passes based on input perturbation. (d) FF is implemented with two forward passes on positive data and negative data, respectively. (e) Ours simplifies the learning process by using a single forward pass.

**Feature scaling** Normalization by vector length is uncommon in image classification tasks as it does not ensure that features exhibit similar characteristics, such as identical means and standard deviations. To address this, layer normalization (Ba et al., 2016) can be applied to the input vector, but this leads to redundant normalization and compromises the mathematical significance of output direction as a feature. Consequently, CwComp (Papachristodoulou et al., 2024) employs batch normalization as a substitute. However, this approach fails to decouple goodness and leaks goodness to the next layer, hindering deeper layers from learning new features and causing overfitting.

**Deactivated neurons** Square goodness is highly sensitive to outliers, which can dominate and deactivate most neurons, distorting feature representation. Moreover, deactivated neurons do not contribute to weight updates during gradient calculations, leading to features represented by a limited subset of neurons and causing feature loss in deeper layers.

In this paper, DeeperForward is proposed to address the mentioned issues by redesigning goodness and features to better suit deep networks. We also enhance the convolution structure proposed by CwComp (Papachristodoulou et al., 2024), enabling effective training of FF in deeper CNNs. Our main contributions are as follows:

- We adopt the more widely used layer normalization (Ba et al., 2016) to ensure a fixed mean and standard deviation of the output, replacing normalization based on vector length and effectively addressing feature scaling and redundant normalization issues.
- Exploiting the property of layer normalization that maintains a mean of zero, we propose using mean goodness as an alternative to squared goodness, thereby facilitating the decoupling of goodness for enhanced feature extraction. This approach also ensures that weight updates are not hindered by deactivated neurons, allowing for the learning of richer features.
- Based on the characteristics of layer-wise local updates, we introduce a model parallel strategy that significantly enhances training efficiency on multiple GPUs.
- Our method enhances FF to achieve improved performance in deeper networks. Experimental results indicate that our approach, utilizing a 17-layer CNN, outperforms existing layer-wise FF-based methods on CIFAR-10, MNIST, and Fashion-MNIST, achieving substantial performance gains, particularly an 8.11% improvement on CIFAR-10.

## 2 RELATED WORK

### 2.1 CONVENTIONAL BRAIN-INSPIRED LEARNING RULES

Hebbian learning (Hebb, 2005; Gerstner et al., 2014) updates synaptic plasticity determined by pre- and post-synaptic neuron states (Löwel & Singer, 1992). Based on the Hebbian rule, a neural coding

{2}------------------------------------------------

framework was proposed for learning generative models using the predictive coding (Ororbia & Kifer, 2022; Rao & Ballard, 1999). SoftHebb (Journé et al., 2023) proposes an algorithm based on theory for Hebbian learning in soft winner-take-all (WTA) networks. Hebbian learning is considered a basic bio-plausible method with no target.

In target-based methods, feedback alignment (Lillicrap et al., 2016) and direct feedback alignment (Nøkland, 2016) replace backpropagation weights with a fixed random matrix to establish alternative error feedback connections, as shown in 1(b). Weight mirror (Akrou et al., 2019) adjusts the feedback connection matrix, equivalent to the transport weight matrix. However, these methods still rely on global error. Target propagation (TP) (Bengio, 2014; Bartunov et al., 2018) and difference target propagation (DTP) (Lee et al., 2015; Ernoult et al., 2022) set local targets as the goal for local updates. Local representation alignment (LRA) (Ororbia et al., 2023) addresses the asymmetry problem through top-down signal transmission with Hebbian-like rules, further solving the non-local problem. To update unlocking, decoupled greedy learning (Belilovsky et al., 2020) optimizes a joint training objective to decouple the layer training with auxiliary networks. Avoiding using backward passes, a forward propagation training method through time is proposed for recurrent neural networks (Kag & Saligrama, 2021). PEPITA (Dellaferrera & Kreiman, 2022) achieves local updates by perturbing inputs with the error and employs a Hebbian-like rule based on two forward passes with a fixed feedback matrix, as shown in Figure 1(c). Despite these advancements, they partially suffer from the update locking problem.

### 2.2 BACKGROUND OF FORWARD-FORWARD ALGORITHM

Inspired by Boltzmann machines (Hinton et al., 1986) and noise contrastive estimation (NCE) (Gutmann & Hyvärinen, 2010), the Forward-Forward algorithm (FF) (Hinton, 2022) introduces a greedy learning scheme via two forward passes, as shown in Figure 1(d), tackling the mentioned bio-implausible problems. FF uses the length of the output vector as a measure of *goodness*, where goodness represents the score of positive data. Decoupling goodness from the output features is important to prevent subsequent layers from relying solely on previous goodness. Therefore, FF extracts features by normalizing the vector length, denoted as,

$$\mathbf{y} = \text{ReLU}(\mathbf{W}\mathbf{x}), \quad (1)$$

$$g = \sum_i y_i^2, \quad \mathbf{z} = \frac{\mathbf{y}}{\sqrt{\frac{1}{N}g + \epsilon}}, \quad (2)$$

$$\Delta W_{ij} = 2x_i y_j \frac{\partial \mathcal{L}}{\partial g}, \quad (3)$$

where  $\mathbf{x}$  denotes the input,  $\mathbf{y}$  represents the output after ReLU (Glorot et al., 2011) with  $N$  elements,  $\mathbf{W}$  is the weight matrix,  $y_i$  denotes the element of the vector  $\mathbf{y}$  of a hidden layer, and  $g$  denotes goodness. The features  $\mathbf{z}$  is the unit vector of  $\mathbf{y}$ .  $\epsilon$  is a small constant.  $\Delta W_{ij}$  denotes the weight update term and  $\mathcal{L}$  is the loss function. The image with a real label is regarded as positive data for optimizing to reach a high goodness value in each layer, and vice versa. During inference, an image entails computing the goodness of each label and selecting the highest one through several iterations. The preliminary study of FF only works on small networks without weight-sharing structures.

Recently, several works have proposed some advanced FF-related algorithms. Symmetric backpropagation-free contrastive learning with FF (SymBa) (Lee & Song, 2023) enhances performance through a gradient-symmetric contrastive loss and a novel label embedding scheme. The predictive Forward-Forward algorithm (PFF) (Ororbia & Mali, 2023) integrates FF with predictive coding presenting a promising brain-inspired algorithm for classifying, reconstructing, and synthesizing data patterns. However, these approaches are still limited to models without weight-sharing structures. The cascaded forward (CaFo) algorithm (Zhao et al., 2023) utilizes a series of random fixed convolutional kernels as the backbone and cascades a fully connected classifier for each kernel. However, it merely updates the classifiers, leaving the kernels unchanged. Forward-Forward contrastive learning (FFCL) (Ahamed et al., 2023) introduces contrastive learning for convolutional models based on FF. However, this approach still prefers extra training by global errors. Recently, convolutional channel-wise competitive learning (CwComp) (Papachristodoulou et al., 2023; 2024) successfully extends FF into CNNs by grouping the features by channels for each class, and using a loss function inducing competitive learning between class-specific features. Despite the advancements, these methods focus on shallow networks within 4 layers. Currently, Trifecta (Dooms et al.,

{3}------------------------------------------------

![Figure 2: Overview of DeeperForward. (a) Example network architecture for DeeperForward, including backbone and Signal Integrating and Pruning (SIP) module. (b) The training scheme of DeeperForward. (c) Modified channel-wise convolution (CW-Conv) from CwComp based on mean goodness.](2fa4a1bf91d0f34e87c689fbc1211fe3_img.jpg)

Figure 2 consists of three parts: (a) shows a network architecture with a backbone of five blocks (Block 1 to Block 5). Each block contains a 'CW-Conv' layer followed by 'Average pooling'. Below each block, there are outputs labeled  $y^{(1)}$  through  $y^{(14)}$ . These outputs are fed into a 'Signal Integrating & Pruning (SIP)' module, which produces a final output  $y$ . (b) illustrates the training process. An 'Input image' is processed through a series of 'CW-Conv' layers. At each layer  $t$ , the output  $z^{(t)}$  is used to calculate a loss  $L_{CL}^{(t)}$  and a goodness value  $g^{(t)}$ . The goodness values are passed to the next layer. (c) shows a 'Modified channel-wise convolution (CW-Conv)'. An input  $x$  is processed through a 'Mean' operation to generate 'goodness'  $g$ . This goodness is then used to scale the input  $x$  via a 'Scale' operation, resulting in 'features'. The 'features' are then processed by a 'CW-Conv' layer to produce the final output.

Figure 2: Overview of DeeperForward. (a) Example network architecture for DeeperForward, including backbone and Signal Integrating and Pruning (SIP) module. (b) The training scheme of DeeperForward. (c) Modified channel-wise convolution (CW-Conv) from CwComp based on mean goodness.

Figure 2: Overview of DeeperForward. (a) Example network architecture for DeeperForward, including *backbone* and *Signal Integrating and Pruning* module. A VGG-like architecture is displayed as an instance. (b) The training scheme of DeeperForward. (c) Modified channel-wise convolution (CW-Conv) from CwComp based on mean goodness.

2023) employs a two-layer block-wise backpropagation approach to replace single-layer updates in a 12-layer CNN, using batch normalization. However, this integration with backpropagation still presents bio-plausibility issues, diminishing parallelism and contradicting the motivations behind the FF algorithm. Both Trifecta and CwComp facilitate easier training by leaking goodness, which can result in potential overfitting in deeper networks.

## 3 METHODOLOGY

This paper introduces DeeperForward, which extends the FF algorithm to 17-layer CNNs through a novel goodness design. This approach addresses the bio-plausibility issues of backpropagation and overcomes the limitations of FF concerning model size. Figure 2 illustrates the overall framework of our method, including the architecture and training approach. The details of the new goodness design are presented in Section 3.1. The network architecture is discussed in Section 3.2, while the training process and advanced strategies for DeeperForward are outlined in Section 3.3.

### 3.1 MEAN GOODNESS

FF uses squared goodness and normalization of the length, as described in Eq. 2. This method suffers from issues related to feature scaling, deactivated neurons, and redundant normalization, resulting in suboptimal performance in deep networks. Although CwComp (Papachristodoulou et al., 2024) improves performance using squared goodness and batch normalization, it leaks goodness information, leading to overfitting in deeper networks. Considering these factors, we adopt widely used layer normalization for better feature scaling with identical mean and standard deviations. To decouple goodness through normalization, we utilize the mean as goodness, leveraging the property of layer normalization that produces an output with a mean of zero. Furthermore, mean goodness ensures that deactivated neurons do not hinder updates. The specific formula is as follows:

$$y = \text{ReLU}(Wx), \quad (4)$$

$$g = \sum_i y_i, \quad z = \frac{y - g}{\sqrt{\sigma^2 + \epsilon}}, \quad (5)$$

$$\Delta W_{ij} = Cx_i \frac{\partial L}{\partial g}, \quad (6)$$

{4}------------------------------------------------

where  $\mathbf{x}$  denotes the input,  $\mathbf{y}$  represents the output after ReLU (Glorot et al., 2011),  $\mathbf{W}$  is the weight matrix,  $g$  indicates goodness,  $\sigma$  is the standard deviation,  $\mathbf{z}$  refers to the output features,  $\Delta W_{ij}$  is the weight update term,  $\mathcal{L}$  is the local loss function, and  $C$  is a constant.

From Eq. 5, it is evident that the output distribution  $\mathbf{z}$  maintains a mean of zero, effectively eliminating goodness. This also ensures that the features share a similar distribution, addressing the feature scaling issue. During weight updates, mean goodness (Eq. 6) allows for updates even when the output neuron  $y_j$  is zero, unlike squared goodness (Eq. 3), thereby solving the deactivated neurons problem.

### 3.2 ARCHITECTURE FOR DEEPERFORWARD

The architecture for DeeperForward, as illustrated in Figure 2(a), incorporates a modified classical CNN backbone, exemplified by the VGG-like model (Simonyan & Zisserman, 2014). It incorporates a convolutional structure that combines channel-wise convolution (CW-Conv) with mean goodness, along with a *Signal Integrating and Pruning* (SIP) module to obtain the final results.

**Channel-Wise Convolution with Mean Goodness** To incorporate mean goodness into CNNs, combining convolution with mean goodness involves simply obtaining the output mean as goodness, followed by layer normalization to facilitate feature extraction. Formally, the goodness  $\hat{y}$  and representation  $\mathbf{Z}$  are defined as:

$$\hat{y} = \frac{1}{HWC} \sum_{h \in \mathbf{H}} h, \quad \mathbf{Z} = \text{LayerNorm}(\mathbf{H}), \quad (7)$$

where  $\mathbf{H} \in \mathbb{R}^{H \times W \times C}$  denotes the hidden states after the convolution with ReLU (Glorot et al., 2011),  $\hat{y} \in \mathbb{R}$  indicates the goodness, that is, the mean of  $\mathbf{H}$ . The representation output  $\mathbf{Z}$  is  $\mathbf{H}$  going through layer normalization.

In multi-class tasks, we optimize the channel-wise convolution (CW-Conv) structure from CwComp (Papachristodoulou et al., 2023; 2024), combining it with our mean goodness to obtain goodness scores for all classes through a single inference. The outputs are evenly grouped by channel, with each group representing a class. Goodness is calculated for each group, followed by individual layer normalization to extract features, effectively implementing group normalization (Wu & He, 2018) on the entire output, as illustrated in Figure 2(c). Formally, the channel-wise convolution with mean goodness for  $G$  classes can be described as:

$$\hat{y}_i = \frac{G}{HWC} \sum_{h \in \mathbf{H}_i} h, \quad i = 1, 2, \dots, G; \quad (8)$$

$$\hat{\mathbf{y}} = [\hat{y}_1, \hat{y}_2, \dots, \hat{y}_G], \quad (9)$$

$$\mathbf{Z} = \text{GroupNorm}(\mathbf{H}; G),$$

where  $\mathbf{H}_i \in \mathbb{R}^{H \times W \times \frac{C}{G}}$  and  $\hat{y}_i$  denotes the hidden states and goodness for the  $i$ -th class, and  $\hat{\mathbf{y}}$  stands for classification scores.  $\text{GroupNorm}(\mathbf{H}; G)$  represents the group normalization of hidden states  $\mathbf{H}$  by  $G$  groups.  $\mathbf{Z}$  stores the representation feature maps.

Compared to our method, CwComp performs classification training directly on the outputs after batch normalization without decoupling goodness, resulting in goodness leakage to the next layer and leading to overfitting in deeper layers.

**Backbone** The backbone is derived from the classical CNNs, leveraging their well-established structural advantages. We substitute the general convolutional kernels with CW-Conv modules to generate classification results locally. The representation from each CW-Conv module serves as the input for the next layer, denoted as  $\mathbf{Z}^{(l)}$  where  $l$  signifies the layer number. The local classification result at the  $l$ -th layer is represented as  $\hat{\mathbf{y}}^{(l)}$ . In particular, the channel size of kernels must be a multiple of the class count. Furthermore, to maintain approximate zero mean of the representation, we adopt average pooling for downsampling, instead of max pooling, as the latter tends to increase the mean value. In this architecture, each layer produces a classification score using CW-Conv. Moreover, the experiments in Appendix E reveal that the CW-Conv outperforms the fully connected (FC) layer in terms of performance. Consequently, the final FC layer is needless.

{5}------------------------------------------------

![Figure 3: Residual structures. (a) Addition type: Input X is processed by CW-Conv (10) to produce X_r (10 c). X_r is then added (⊕) to the output of AvgPool(X_r) (10 c) to produce Z_r (10 c). The target is 10 channels. (b) Concatenation type: Input X is processed by CW-Conv (20-10) to produce X_r (10 c). X_r is then concatenated (⊕) with the output of AvgPool(X_r) (10 c) to produce Z_r (20 c). The target is 20 channels. Both diagrams show the flow from input X to the final representation Z_r, with the residual path adjusting spatial dimensions if necessary.](191a4a245a7d36d03be9a990d0f758f5_img.jpg)

Figure 3: Residual structures. (a) Addition type: Input X is processed by CW-Conv (10) to produce X\_r (10 c). X\_r is then added (⊕) to the output of AvgPool(X\_r) (10 c) to produce Z\_r (10 c). The target is 10 channels. (b) Concatenation type: Input X is processed by CW-Conv (20-10) to produce X\_r (10 c). X\_r is then concatenated (⊕) with the output of AvgPool(X\_r) (10 c) to produce Z\_r (20 c). The target is 20 channels. Both diagrams show the flow from input X to the final representation Z\_r, with the residual path adjusting spatial dimensions if necessary.

Figure 3: Residual structures: (a) Addition type for the shortcut channels match the target channels. (b) Concatenation type for shortcut channels differing from the target channels.

**Residual Structure** Residual structures are traditionally employed to facilitate error backpropagation by providing shortcuts for easier learning. In FF, it enables the integration of features at different levels, enriching the representational capacity of deep networks. To adapt to the FF, we implement two parameter-free residual structures, the *addition* and *concatenation* types, as alternatives to the original parameterized versions, as illustrated in Figure 3. To match the spatial dimensions, we employ average pooling to the shortcut, as shown below, for downsampling.

$$\mathbf{Z}_r = \begin{cases} \text{AvgPool}(\mathbf{X}_r), & (H_r, W_r) \neq (H, W), \\ \mathbf{X}_r, & (H_r, W_r) = (H, W), \end{cases} \quad (10)$$

where  $\mathbf{Z}_r \in \mathbb{R}^{H \times W \times C_r}$  denotes the shortcut feature map after spatial dimension adjustment, and  $\text{AvgPool}(\cdot)$  is the average pooling operation to adjust  $\mathbf{X}_r$  from  $(H_r, W_r)$  to  $(H, W)$ .

Two types of residual structures are adapted in different scenarios. If the shortcut matches the channel of target feature maps, the addition type is employed. Otherwise, the concatenation type is used. As shown in Figure 3, two residual structures can be summarized as:

$$\mathbf{Z} = \begin{cases} F(\mathbf{X}; C) + \mathbf{Z}_r, & C = C_r, \\ \text{Concat}(F(\mathbf{X}; C - C_r), \mathbf{Z}_r), & C \neq C_r, \end{cases} \quad (11)$$

where  $F(\mathbf{X}; C)$  stands for the CW-Conv with  $C$  channels output from input  $\mathbf{X}$ , and  $\mathbf{Z}_r$  is the feature maps from shortcut.  $\text{Concat}(\cdot, \cdot)$  denotes the concatenation operation on channel dimension.  $\mathbf{Z}$  represents the final representation output. Particularly, the number of convolution channels is reduced to  $C - C_r$  in concatenation type to ensure the channel of final output satisfies the target.

**Signal Integrating and Pruning Module** Inspired by synaptic pruning (Chechik et al., 1998; Nenskyyte & Gross, 2017), where the brain forms excess synapses and then eliminates redundancies, we propose the *Signal Integrating and Pruning* (SIP) module. The FF accumulates local goodness to obtain the final result, with experiments showing that the last three layers perform best on the test set. Similarly, we separate a subset of data from the training set, leave it untrained, and evaluate accuracy on this subset to select the best layer combination, avoiding direct testing on the test set. However, for a deep model with  $L$  layers, there are  $2^L$  combinations. To reduce complexity, we simplify the rule to accumulating layers between a chosen start layer and an end layer, reducing the combinations to  $L(L + 1)/2$ . The SIP module with  $L$  layers can be described as:

$$\hat{\mathbf{y}} = \sum_{l=S}^E \hat{\mathbf{y}}^{(l)}, \quad 0 < S \leq E \leq L, \quad (12)$$

where  $\hat{\mathbf{y}}^{(l)}$  denotes classification scores from the  $l$ -th layer, and  $\hat{\mathbf{y}}$  is the final result.  $S, E \in \mathbb{Z}$  are integers and range from 1 to  $L$ , representing the start and end layers to be accumulated. After selection, layers beyond the end layer are no longer used and can be pruned.

### 3.3 DEEPERFORWARD TRAINING SCHEME

**Training Scheme** We present DeeperForward, a training strategy that optimizes the classification result at each layer through a single forward pass, relying solely on the local input-output states.

{6}------------------------------------------------

Figure 2(b) depicts the training procedure of DeeperForward. Local optimization leverages the classification results from CW-Conv as the local target. It utilizes a local cross-entropy loss for each layer to generate the update signal, preventing error transportation across layers. The local optimization can be formulated as:

$$\begin{aligned}\mathcal{L}_{\text{CE}}^{(l)}(\hat{\mathbf{y}}^{(l)}, \mathbf{y}) &= -\sum_{i=1}^G y_i \log(\text{softmax}(\hat{y}_i^{(l)})), \\ \nabla \theta^{(l)} &= \nabla_{\theta^{(l)}} \mathcal{L}_{\text{CE}}^{(l)}(\hat{\mathbf{y}}^{(l)}, \mathbf{y}),\end{aligned}\quad (13)$$

where  $\hat{\mathbf{y}}^{(l)}$  and  $\mathbf{y}$  denote the local classification result and the real label with  $G$  classes.  $\mathcal{L}_{\text{CE}}^{(l)}$  denotes the cross-entropy loss, while  $\nabla \theta^{(l)}$  is the update of weights at the  $l$ -th layer, and  $\nabla_{\theta^{(l)}} \mathcal{L}_{\text{CE}}^{(l)}(\hat{\mathbf{y}}^{(l)}, \mathbf{y})$  is the gradient of  $\mathcal{L}_{\text{CE}}^{(l)}$  with respect to the kernel weights  $\theta^{(l)}$ . The local optimization process solely relies on the input and classification result of the individual CW-Conv. DeeperForward is compatible with general gradient-based optimizers, such as Adam (Kingma & Ba, 2014). Owing to the local learning process, there is no need to store intermediate states, eliminating the freezing activity problem and the update locking problem. Additionally, the non-local problem and weight transport problem are addressed by the local loss optimization and the forward training scheme.

**Model Parallel Strategy** Our method enables a model parallel strategy based on the parallelism of FF mentioned in (Aktemur et al., 2024), as illustrated in Figure 4. Our strategy treats each convolutional layer as an independent component in the pipeline. Once a convolutional layer processes a batch of data, it passes the results to the next group, allowing the next batch to be processed without waiting for the entire network to complete. As shown in Figure 4 (a), this approach enables simultaneous processing of multiple batches across different layers, achieving high parallelism.

Figure 4(b) illustrates an implementation example using multithreading techniques in a multi-GPU setup. Our strategy assigns an independent thread for each convolutional layer to update, utilizing first-in-first-out(FIFO) queues for data transfer. Different threads can be allocated to various GPUs, enabling model parallelism. Compared to the commonly used distributed data-parallel (DDP) (Li et al., 2020) technique in backpropagation, our approach offers several advantages for improved efficiency: (i) Each GPU does not need to store the entire network, and (ii) Data transfer between GPUs occurs only between layers on different GPUs, rather than across the entire network. Details of implementation are in Appendix G.

![Figure 4: Model parallel strategy. (a) A pipeline diagram showing three layers (Layer 1, Layer 2, Layer 3) processing multiple batches (F1-F8) in a pipelined fashion. (b) A detailed diagram of the multi-threading implementation on GPUs, showing threads for different layers (GPU 0 Thread l-1 and GPU 1 Thread l) with FIFO queues for data transfer and local loss calculation.](0236eff05bcb8f3a343ea7933aaa306b_img.jpg)

Figure 4(a) shows a pipeline program where Layer 1, Layer 2, and Layer 3 process batches  $F_1$  through  $F_8$  in a pipelined manner. Each layer has associated inference ( $F_i$ ) and update ( $U_i$ ) operations. Figure 4(b) shows a multi-threading implementation on GPUs. It illustrates two threads: (GPU 0) Thread  $l-1$  and (GPU 1) Thread  $l$ . Each thread contains a CW-Conv block with a local loss calculation  $\nabla \theta \leftarrow L_{\text{CE}}$ . Data is transferred between threads via FIFO queues, and the output of one thread is passed to the next.

Figure 4: Model parallel strategy. (a) A pipeline diagram showing three layers (Layer 1, Layer 2, Layer 3) processing multiple batches (F1-F8) in a pipelined fashion. (b) A detailed diagram of the multi-threading implementation on GPUs, showing threads for different layers (GPU 0 Thread l-1 and GPU 1 Thread l) with FIFO queues for data transfer and local loss calculation.

Figure 4: Model parallel strategy. (a) A pipeline program (the same color indicates operations on the same minibatch of data). (b) An implementation example based on the multi-threading technique.

**Memory-saving Strategy** Due to update locking problems, BP requires a large amount of memory to store intermediate states throughout the process. Without limitation of update locking, this strategy achieves memory savings by promptly releasing memory after each layer’s computation. The memory saving strategy is a layer-by-layer update strategy, which consists of following steps: (1) Perform computation and update weights in current layer. (2) Pass the output to the next layer. (3) Release all intermediate states from the memory used by the current layer. (4) Repeat steps (1)-(3) layer by layer. More details are shown in Appendix H. This memory-saving strategy is particularly suitable for scenarios with constrained memory resources, such as edge computing.

{7}------------------------------------------------

Table 1: Classification on CIFAR10, MNIST, and F-MNIST, evaluating performance compared to BP and FF-related algorithms. Measurements of mean and standard deviation are for five trial runs. \*: Reproduced results. †: With data augmentation. ‡: With block-wise backprop.

| Type | Method | Arch. | #Layer | CIFAR10 | MNIST | F-MNIST |
|-|-|-|-|-|-|-|
| non-FF | PEPITA | CNN | 2 | 52.57 ± 0.36 | 98.01 ± 0.09 | - |
|  | DTP | CNN | 6 | 89.38 ± 0.20 | 98.93 ± 0.04 | <b>90.35 ± 0.11</b> |
|  | recLRA | CNN | 18 | <b>93.58</b> | 98.18 | 88.13 |
|  | SoftHebb | SoftHebb | 4 | 80.31 ± 0.14 | <b>99.35 ± 0.03</b> | - |
|  | F <sup>3</sup> | MLP | 2 | 46.04 ± 0.18 | 97.16 ± 0.10 | - |
|  | SP | CNN | 8 | 92.4 | - | - |
| Block-wise BP | HPFF | CNN | 110 | 91.04 | - | - |
|  | SEDONA | CNN | 152 | 93.87 | - | - |
|  | BWBPF | CNN | 152 | <b>95.52</b> | - | - |
| BP | ResNet18-BP* | CNN | 18 | 94.03 ± 0.11† | 99.58 ± 0.02 | 93.78 ± 0.06 |
| FF | FF | MLP | 4 | 59.00 | 98.69 | - |
|  | SymBa | MLP | 3 | 59.09 | 98.58 | - |
|  | CaFo | CNN | 3 | 67.43 | 98.80 | - |
|  | CwComp | CNN | 4 | 78.11 ± 0.44 | 99.42 ± 0.08 | <b>92.31 ± 0.32</b> |
|  | TinyCNN-ours | CNN | 4 | <b>79.49 ± 0.29</b> | <b>99.50 ± 0.05</b> | 91.83 ± 0.06 |
| FF | Trifecta† | CNN | 12 | 83.51 ± 0.78 | 99.58 ± 0.06 | 91.44 ± 0.49 |
|  | CwComp* | CNN | 14 | 75.28 ± 0.54 | 99.27 ± 0.09 | 91.79 ± 0.47 |
|  | CNN-ours | CNN | 14 | 81.76 ± 0.30 | <b>99.65 ± 0.02</b> | 92.44 ± 0.08 |
|  | ResNet-ours | CNN | 17 | <b>86.22 ± 0.17</b> | 99.63 ± 0.04 | <b>93.13 ± 0.13</b> |

Table 2: Classification on CIFAR100.

|  | ResNet-BP | ResNet-ours | ResNet-CHx3-ours |
|-|-|-|-|
| Accuracy | 58.01 ± 0.48 | 53.09 ± 0.79 | 60.28 ± 1.02 |

## 4 EXPERIMENT

### 4.1 DATASETS AND EXPERIMENT SETTINGS

To fully validate the effectiveness of DeeperForward, we conduct experiments on 3 datasets: MNIST (LeCun et al., 1998), Fashion-MNIST (F-MNIST) (Xiao et al., 2017), and CIFAR10 (Krizhevsky et al., 2009) without any data augmentation. Specifically, the training sets of MNIST and F-MNIST are separated into two groups, 50,000 and 10,000 samples. The former group is used for training and the latter group is used for pruning by *Signal Integrating and Pruning* (SIP) module. Similarly, CIFAR10’s training set is split into 45,000 and 5,000 samples. All the samples in the datasets are resized to  $32 \times 32$  pixels. Hyperparameters setting is detailed in Appendix B. Our experiments are executed on 4 Nvidia GTX Titan X GPUs (12GB).

![Figure 5: Performances on CIFAR10. (a) Shallow networks, compared to BP and FF-based methods. (b) Deep networks, compared to BP without data augmentation. (c,d) Comparison experiments for residual structures on CIFAR10 without dropout. (c) Model performance during training. (d) The accuracy of each layer after training for 150 epochs.](643d86ebba41e16a88461bfcb3741de6_img.jpg)

Figure 5 consists of four subplots: (a) shows accuracy (%) vs epoch for shallow networks (0-150 epochs), comparing 'Ours' (red solid) with BP (blue dashed), FF (green dashed), SymBa (yellow dashed), and CaFo (purple dashed). (b) shows accuracy (%) vs epoch for deep networks (0-150 epochs), comparing 'Ours (Test)' (red solid), 'Ours (Train)' (red dashed), 'BP (Test)' (blue solid), and 'BP (Train)' (blue dashed). (c) shows accuracy (%) vs epoch (0-150) for training, comparing 'Train w/o shortcut' (red solid), 'Train w/ shortcut' (red dashed), 'Test w/o shortcut' (blue solid), and 'Test w/ shortcut' (blue dashed). (d) shows accuracy (%) vs layer (1-17) after 150 epochs, comparing 'Train w/o shortcut' (red solid), 'Train w/ shortcut' (red dashed), 'Test w/o shortcut' (blue solid), and 'Test w/ shortcut' (blue dashed).

Figure 5: Performances on CIFAR10. (a) Shallow networks, compared to BP and FF-based methods. (b) Deep networks, compared to BP without data augmentation. (c,d) Comparison experiments for residual structures on CIFAR10 without dropout. (c) Model performance during training. (d) The accuracy of each layer after training for 150 epochs.

Figure 5: Performances on CIFAR10. (a) Shallow networks, compared to BP and FF-based methods. (b) Deep networks, compared to BP without data augmentation. (c,d) Comparison experiments for residual structures on CIFAR10 without dropout. (c) Model performance during training. (d) The accuracy of each layer after training for 150 epochs.

{8}------------------------------------------------

### 4.2 COMPARISONS OF DIFFERENT METHODS

We employ three CNN models to evaluate our method: a 4-layer tiny CNN, a VGG-like 14-layer CNN, and a 17-layer ResNet-like CNN (He et al., 2016), as detailed in the Appendix A. Our comparisons encompass both non-FF and FF methods for a comprehensive analysis. Non-FF brain-inspired methods include advanced BP-free approaches such as PEPITA (Dellaferrera & Kreiman, 2022), DTP (Ernoult et al., 2022), rec-LRA (Ororbia et al., 2023), SoftHebb (Journé et al., 2023), F<sup>3</sup> (Flügel et al., 2023), and Signal Propagation(SP) (Kohan et al., 2024). We also compare with block-wise BP method: HPFF(Su et al., 2024), SEDONA(Pyeon et al., 2021), and BWBPF(Cheng et al., 2024). In the FF methods, we consider FF (Hinton, 2022), SymBa (Lee & Song, 2023), CaFo (Zhao et al., 2023), Trifecta (Dooms et al., 2023), and CwComp (Papachristodoulou et al., 2024). Since layer-wise FF-based methods operate on shallow networks, we reproduce and extend CwComp (Papachristodoulou et al., 2024) into the same 14-layer CNNs for comparison. Results are summarized in Table 1. Our method outperforms FF-based methods in CIFAR10 and MNIST with shallow networks. The training curves on CIFAR10 with FF-based methods in shallow networks are shown in Figure 5(a). As we extend to 14 and 17 layers, our performance improves, whereas CwComp (Papachristodoulou et al., 2024) exhibits overfitting, leading to performance decline. Therefore, our method extended the capability of FF to train in deeper models. The results indicate that our design of mean goodness enhances the performance of FF, making it more suitable for deep CNN models. However, FF-based methods train greedily through classification objectives at each layer, indirectly extracting features. This results in weaker feature learning capabilities compared to BP, which directly learns intermediate layer features. Although Figure 5(b) shows that our method’s performance is close to BP without data augmentation, the improvement is limited after data augmentation, as detailed in Appendix C, leading to a larger gap with BP. These points are also directions worth exploring further.

Furthermore, we conduct a more challenging experiment on CIFAR100, as shown in Table 2. ResNet-Chx3 is a variant of ResNet with triple the number of channels. Table 2 highlights the disparity between DeeperForward and BP on ResNet. The significant improvement of ResNet-Chx3 indicates that an inadequate allocation of neurons to each class results in a sharp decline in performance.

### 4.3 PERFORMANCE OF SIGNAL INTEGRATING AND PRUNING MODULE

Table 3: Performance on Signal Integrating and Pruning (SIP) using a 17-layer ResNet. (*Start*, *End*) denotes the selected layers by SIP, where *Start* and *End* represent start layer and end layer.

|  | CIFAR10 | MNIST | F-MNIST |
|-|-|-|-|
| ALL LAYERS | 86.45 | 99.68 | 93.08 |
| AFTER SIP | 86.51 | 99.67 | 93.23 |
| ( <i>Start</i> , <i>End</i> ) | (2,17) | (2,11) | (3,16) |

We validate our pruning strategy using the Signal Integrating and Pruning (SIP) module by comparing the performance with a similar strategy in FF (Hinton, 2022) that accumulates all the goodness as the final result. In Table 3, we select the best trial results using ResNet for the SIP experiment comparison, showing that SIP can improve performance in most cases. Interestingly, experiments on simpler tasks such as MNIST tend to retain fewer layers compared to more challenging tasks like CIFAR10. This observation shows the ability to adapt its depth based on the complexity of the task.

### 4.4 ABLATION STUDY

Table 4: Ablation study on CIFAR-10, showing the mean performance from five experimental trials.

| MEAN | SIP | RESIDUAL | ACCURACY |
|-|-|-|-|
| ✓ | ✓ | ✓ | 79.38 |
| ✓ |  |  | 81.02 |
| ✓ | ✓ |  | 81.16 |
| ✓ |  | ✓ | 86.08 |
| ✓ | ✓ | ✓ | 86.22 |

This method introduces mean goodness, the Signal Integrating and Pruning (SIP) module, and a non-learned residual structure to optimize performance. To evaluate the contributions of each component, we conducted ablation experiments using a 17-layer ResNet architecture on CIFAR10, averaging results from five trials, as shown in Table 4. When mean goodness is omitted, we utilize

{9}------------------------------------------------

squared goodness and normalization of the vector length. For comparison with SIP, we directly sum all layers as the final output. The removal of the residual structure involves excluding the shortcut connections. To provide a more comprehensive analysis, Appendix I discusses the differences in deactivated neurons between mean and square goodness.

The experimental results show that mean goodness achieves a substantial performance increase of 6.84% compared to squared goodness within the same network. The SIP module provides a slight performance boost while allowing for optimization of network size. The residual structure significantly enhances performance by integrating features at various levels, resulting in a more comprehensive feature representation. Figures 5 (c) and (d) analyze the training curves and local classification performance with and without the residual structure, demonstrating that the residual connections facilitate improved learning in deeper layers.

### 4.5 PARALLEL PERFORMANCE OF DEEPERFORWARD

We evaluate the performance of model parallel strategy through time-consumption training on CIFAR10, using 1, 2, and 4 GPUs. In multi-GPU case, layers are evenly grouped and assigned to different GPUs. As a point of comparison, we use BP with the widely adopted distributed data parallel (DDP) (Li et al., 2020) as a baseline. As shown in Table 5, our method outperforms BP with DDP in terms of training time. Notably, our approach achieves a higher speedup with 2 GPUs, as inter-GPU communication occurs only between layers on different devices, unlike DDP, where communication involves the entire network. However, with 4 GPUs, the speedup is lower than DDP. We observed a drop in GPU utilization, caused by imbalanced computation across layers, leading to pipeline program bottlenecks. Future work could explore advanced pipeline techniques for optimization. This study demonstrates the feasibility and potential of model parallelism in our method.

Table 5: Training time per epoch on CIFAR10 (Speedup rate relative to 1 GPU in parentheses).

| METHOD | 1 GPU | 2 GPUs | 4 GPUs |
|-|-|-|-|
| BP-DDP | 51.98s (1.0 $\times$ ) | 32.70s (1.59 $\times$ ) | 19.92s (2.61 $\times$ ) |
| OURS | <b>36.38s</b> (1.0 $\times$ ) | <b>20.77s</b> (1.75 $\times$ ) | <b>14.68s</b> (2.48 $\times$ ) |

Moreover, training on CIFAR10 with a batch size of 128 using the memory-saving strategy consumes a minimum of 618.64MB of memory in practice, while BP in ResNet18 requires 1314.49MB.

Additionally, we experiment with deeper ResNet models with 33 and 100 layers but do not observe significant performance improvements, as detailed in Appendix D. Moreover, Appendix E provides a comparison of classification performance between different convolutional layers and fully connected layers. Appendix F presents t-SNE (Van der Maaten & Hinton, 2008) visualizations of the results on the MNIST dataset.

## 5 CONCLUSION

This paper presents the DeeperForward algorithm, extending the Forward-Forward approach to deeper networks with significant performance enhancements. We introduce a novel goodness design, combining mean goodness and layer normalization, which addresses key issues in the effective training of deep networks: feature scaling, redundant normalization, and deactivated neurons. Additionally, we propose a model parallel strategy to significantly improve training efficiency and a memory-saving strategy suitable for resource-constrained environments. Experimental results demonstrate that our method substantially enhances the depth and performance of FF-based algorithms, highlighting the potential of FF in terms of performance and parallelism.

**Limitations.** DeeperForward, similar to FF, relies solely on classification information for learning, lacking direct representation learning capabilities. This results in slower convergence and weaker generalization. Future research should focus on enhancing feature extraction capabilities to address these limitations. Additionally, as the number of categories increases, the convolutional structure grows, making it challenging to implement on extensive datasets. Future work should aim to develop more general structures that avoid excessively large models and multiple forward passes in FF.

 Rest of paper (reference and Appendix) is removed.