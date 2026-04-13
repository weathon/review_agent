

{0}------------------------------------------------

# GROKKING AT THE EDGE OF NUMERICAL STABILITY

Lucas Prieto, Melih Barsbey, Pedro A.M. Mediano\*, Tolga Birdal\*

Department of Computing  
Imperial College London

## ABSTRACT

Grokking, or sudden generalization that occurs after prolonged overfitting, is a surprising phenomenon that has challenged our understanding of deep learning. While a lot of progress has been made in understanding grokking, it is still not clear why generalization is delayed and why grokking often does not happen without regularization. In this work we argue that without regularization, grokking tasks push models to the edge of numerical stability, introducing floating point errors in the Softmax that we refer to as *Softmax Collapse* (SC). We show that SC prevents grokking and that mitigating SC leads to grokking *without* regularization. Investigating the root cause of SC, we find that beyond the point of overfitting, the gradients strongly align with what we call the *naïve loss minimization* (NLM) direction. This component of the gradient does not change the predictions of the model but decreases the loss by scaling the logits, usually through the scaling of the weights along their current direction. We show that this scaling of the logits explains the delay in generalization characteristic of grokking, and eventually leads to SC, stopping learning altogether. To validate these hypotheses, we introduce two key contributions that mitigate the issues faced in grokking tasks: (i) StableMax, a new activation function that prevents SC and enables grokking without regularization, and (ii)  $\perp$  Grad, a training algorithm that leads to quick generalization in grokking tasks by preventing NLM altogether. These contributions provide new insights into grokking, shedding light on its delayed generalization, reliance on regularization, and the effectiveness of known grokking-inducing methods. Code for this paper can be found at: <https://github.com/LucasPrietoAI/grokking-at-the-edge-of-numerical-stability>.

## 1 INTRODUCTION

Deep learning has been transformative for a variety of fields such as natural language processing (Devlin et al., 2019), computer vision (Krizhevsky et al., 2012), geometry processing (Qi et al., 2017), and 3D vision (Deng et al., 2018). This rapid proliferation has brought with it surprising phenomena that defy the predictions of classical statistical learning theory.

In this paper we explore one such recently observed phenomenon known as *grokking*, first described by Power et al. (2022) as a sudden and unexpected generalization occurring after prolonged overfitting. Although predominantly studied in algorithmic tasks like modular addition or multiplication, recent findings suggest that grokking may be a more pervasive phenomenon, also manifesting in more complex tasks involving vision and language (Lv et al., 2024; Humayun et al., 2024).

Prior research has consistently observed grokking in settings that involve some form of regularization, such as weight decay (Barak et al., 2022; Power et al., 2022; Nanda et al., 2023). This pattern has motivated investigations into the implicit biases introduced by weight decay, suggesting it may be critical to triggering delayed generalization. For instance, Liu et al. (2023a) argued that weight norms need to be in a narrow range or “Goldilocks Zone” for generalization. Similarly, Varma et al. (2023) highlighted weight efficiency of generalizing solutions, and Nanda et al. (2023) argued that weight decay favors simpler, more generalizable solutions. However, recent works have argued that regularization may not be necessary for grokking, at least on shallow networks with Mean Squared

\*Joint senior authors, equal contribution

{1}------------------------------------------------

![Figure 1: Three plots showing training and test accuracy over epochs. (a) Generalization: Training accuracy (blue) reaches 100% by epoch 100, while test accuracy (orange) reaches 100% by epoch 300. (b) Grokking: Training accuracy (blue) reaches 100% by epoch 100, while test accuracy (orange) reaches 100% by epoch 700. (c) Overfitting: Training accuracy (blue) reaches 100% by epoch 100, while test accuracy (orange) remains near 0% throughout. Arrows indicate interventions: (a) to (b) via 'L → 0 due to logit-scaling NLM (Sec. 3)' and '⊥AdamW + StableMax (Sec. 4.2)'; (b) to (c) via '∇L → 0 due to numerical errors SC (Sec. 2)' and '(Sec. 2.4)'.](49ad3a646d84bcfeac02bdf2b3792a3e_img.jpg)

Figure 1: Three plots showing training and test accuracy over epochs. (a) Generalization: Training accuracy (blue) reaches 100% by epoch 100, while test accuracy (orange) reaches 100% by epoch 300. (b) Grokking: Training accuracy (blue) reaches 100% by epoch 100, while test accuracy (orange) reaches 100% by epoch 700. (c) Overfitting: Training accuracy (blue) reaches 100% by epoch 100, while test accuracy (orange) remains near 0% throughout. Arrows indicate interventions: (a) to (b) via 'L → 0 due to logit-scaling NLM (Sec. 3)' and '⊥AdamW + StableMax (Sec. 4.2)'; (b) to (c) via '∇L → 0 due to numerical errors SC (Sec. 2)' and '(Sec. 2.4)'.

Figure 1: Our contributions demonstrated through results obtained in addition modulo 113 task. We show that the delay in generalization induced by NLM can be reversed using the proposed  $\perp$ AdamW ((a) and (b)) and that the numerical errors that lead to overfitting instead of grokking can be avoided by using the proposed StableMax ((b) and (c)).

Error (MSE) loss (Kumar et al., 2024; Lyu et al., 2024; Gromov, 2023). These works tie grokking to a transition from lazy training (Chizat et al., 2018) to feature learning. Despite this ongoing work, several aspects in this framing of grokking remain unclear. These include why grokking tasks induce lazy training and why weight decay is often needed to enter the feature learning regime when using deeper models or cross-entropy (CE) loss.

Here we propose a novel account of grokking, outlined in Fig. 1, that explains several of the main unanswered questions in the grokking literature. We start by showing that without regularization, grokking is prevented by absorption errors in the Softmax, which we call *Softmax Collapse* (SC). These errors result in zero terms in the gradient and put an end to learning, sometimes before any progress is made in the test performance, resulting in complete overfitting (Fig. 1, c). We then argue that SC is caused by what we call *Naïve Loss Minimization* (NLM), as the gradient becomes aligned with a direction that corresponds to scaling up the logits by a constant. While scaling up all the logits does not change the model predictions, it does reduce the CE loss for a network that has reached 100% training accuracy, with the downside that this eventually leads to numerical errors in Softmax. Our findings provide explanations for several key aspects of grokking, including (i) the delayed onset of generalization, (ii) why grokking is often absent without regularization, and (iii) why existing methods designed to induce grokking are effective.

To validate our hypothesis that SC is responsible for the absence of grokking without regularization, we introduce **StableMax** as a more numerically stable replacement to Softmax in CE loss. This simple change takes models from complete overfitting to grokking (Fig. 1, c to b) *without* regularization, in settings where it is normally not observed without it. Similarly, we validate that NLM is responsible for delaying generalization (Fig. 1, a to b) and leading to SC by introducing a new optimizer  $\perp$ Grad, which only preserves the part of the gradient that is orthogonal to the NLM direction. By doing this,  $\perp$ Grad quickly leads to generalization without the initial overfitting phase that defines grokking (Fig. 1, b to a).

Our primary contributions are as follows:

- We observe that cases of overfitting without grokking are due to floating point errors caused by extreme values in the Softmax function, which we term *Softmax Collapse* (SC; Sec. 3).
- We show that interventions to avoid SC, like greater floating point precision or a new, numerically stable version of Softmax (StableMax), cause grokking in settings where it was previously absent without regularization (Sec. 3.3).
- We observe that models move towards SC because overfitting and cross-entropy loss push the model in a direction of uncontrolled logit growth, which we refer to as *Naïve Loss Minimization* (NLM; Sec. 4).
- We demonstrate that NLM can be avoided through a novel optimizer,  $\perp$ Grad, which removes the delay in generalization (Sec. 5).

## 2 SETUP

### 2.1 DATASETS

We show our findings on the most commonly studied grokking datasets, outlined in this section.

{2}------------------------------------------------

**I. Modular arithmetic.** The main results in this paper are shown on arithmetic modulo 113 (Power et al., 2022; Nanda et al., 2023). This is a family of supervised learning tasks where two one-hot encoded inputs representing integers  $a, b < p$  are used to predict the target  $y = a * b \bmod p$ , where  $*$  is some binary operation and  $p$  is a prime number. In most of our results, the binary operation is addition, but we show additional results with multiplication and subtraction.

Modular arithmetic tasks are characterized by a binary operation and a dataset size, with different behaviors being observed for different dataset sizes on the same binary operation. In these settings, we describe the dataset sizes as the percentage of the  $113^2$  possible pairs that are used for training, with the rest of the data being used for testing as in Nanda et al. (2023) and Power et al. (2022). Our main results use a 40%/60% train/test split but we also include results using 60%/40% and 70%/30%. The input integers are represented as one-hot vectors.

**II. Sparse parity.** We also validate some of our results on the Sparse Parity task outlined in Barak et al. (2022). This is a supervised learning setting where the target is the parity of  $k$  bits out of a binary vector of length  $n$ , with  $k \ll n$ . In this work we use 2000 samples, split evenly between train and test data and we describe instances of this task by specifying the values of  $n$  and  $k$ .

**III. MNIST.** Finally, we provide some results on a subset the classic image classification dataset MNIST (Deng, 2012). For our experiments, we use a subset of 200 training samples from the training set as in Liu et al. (2023b), with evaluation on the full test set.

### 2.2 MODELS

We study the grokking phenomenon on these datasets using a 2-hidden layer multi-layer perceptron (MLP) of width 200 as in Liu et al. (2023a) and a one-layer transformer with 4 attention heads as in Nanda et al. (2023) and Power et al. (2022). We train both of these models in a full batch setting, using ReLU activations and cross-entropy loss with AdamW and SGD, as well as our own variants of these optimizers,  $\perp$ AdamW and  $\perp$ SGD. Unless specified otherwise we set the weight decay parameter  $\lambda = 0$ . For modular arithmetic datasets, inputs are concatenated as the input of the MLP resulting in a 226 dimensional vector, and treated as separate tokens in the case of the transformer.

## 3 SOFTMAX COLLAPSE: FLOATING POINT ERRORS PREVENT GROKKING

Given our current understanding of grokking, it is surprising that it happens without regularization for some dataset sizes, but regularization becomes crucial as dataset size decreases (Power et al., 2022). In this section we highlight that looking at datasets at the boundary of these two regimes reveals that without weight decay, grokking sometimes starts before abruptly stopping (Fig. 2). We show that this is caused by floating point errors in the Softmax that lead the gradients from a large fraction of the samples to become zero. We refer to this phenomenon as Softmax Collapse.

### 3.1 SOFTMAX COLLAPSE

In modern neural network implementations, Floating Point (FP) arithmetic is ubiquitous for representing and computing parameters, activations, and gradients. While FP numbers enable efficient decimal computations, they introduce numerical inaccuracies. This section focuses on *absorption errors*, as a specific class of FP arithmetic failure. We will use the symbol  $\doteq$  to refer to equality under FP arithmetic.

**Definition 1** (Absorption Errors). *Let  $a, b \in \mathbb{R} \setminus \{0\}$  be floating point numbers in a system with base  $\beta$  and  $p$  significant bits. Denote their exponents by  $e_a$  and  $e_b$ , respectively. An absorption error occurs in the computation of  $a + b$  (denoted  $a + b \doteq a$ ) if*

$$e_a - e_b \geq p.$$

*In this case, after exponent alignment, the significand of  $b$  is shifted right by at least  $p$  digits, and  $b$  cannot be represented in the available precision, resulting in  $a + b \doteq a$ .*

Intuitively, absorption errors can occur during FP addition when operands have significantly different magnitudes. For *float32* the base  $\beta$  is 2 and  $p = 24$  bits, meaning that adding any number smaller than  $2^{-(p-1)} = 2^{-23}$  to 1 will leave 1 unchanged.  $2^{-23}$  is the machine epsilon for float32.

{3}------------------------------------------------

![Figure 2: Three subplots (a, b, c) showing Accuracy (%) vs Epoch for MLPs trained on modular addition. (a) 40% training data: Test acc. float16 (blue) and float32 (orange) both plateau around 50% after 5k epochs. Train accuracies (green dashed) reach 100% by 5k epochs. (b) 60% training data: Test acc. float16 (blue) plateaus around 20%, float32 (orange) around 30%. Train accuracies (green dashed) reach 100% by 5k epochs. (c) 70% training data: Test acc. float16 (blue) plateaus around 50%, float32 (orange) around 80%. Train accuracies (green dashed) reach 100% by 5k epochs. All plots show a vertical dashed line at 5k epochs indicating the start of Softmax Collapse (SC).](e94f3bbb6f7501b9a1344dd0210e5dd8_img.jpg)

Figure 2: Three subplots (a, b, c) showing Accuracy (%) vs Epoch for MLPs trained on modular addition. (a) 40% training data: Test acc. float16 (blue) and float32 (orange) both plateau around 50% after 5k epochs. Train accuracies (green dashed) reach 100% by 5k epochs. (b) 60% training data: Test acc. float16 (blue) plateaus around 20%, float32 (orange) around 30%. Train accuracies (green dashed) reach 100% by 5k epochs. (c) 70% training data: Test acc. float16 (blue) plateaus around 50%, float32 (orange) around 80%. Train accuracies (green dashed) reach 100% by 5k epochs. All plots show a vertical dashed line at 5k epochs indicating the start of Softmax Collapse (SC).

Figure 2: As dataset size increases (subplots **a** to **c**), MLPs trained on modular addition begin to generalize without regularization until this is stopped by SC making the gradient from a large fraction of the samples equal to zero. This stopping point comes earlier for float32 than float64 and with small enough datasets it comes before the model makes any progress on test accuracy.

**Absorption errors in the Softmax.** The Softmax function is a fundamental component in numerous deep learning architectures, serving as an activation function or a key element in attention mechanisms. In this case, we focus on its application within the Softmax Cross-Entropy (SCE) loss:

**Definition 2** (Softmax Cross-Entropy (SCE) loss). *For a neural network  $f$  and a data point  $\mathbf{x}$  with label  $y$ , we define  $\mathbf{z} := f(\mathbf{x})$  and  $z_y$  as the logit corresponding to the true class  $y$ . We express the SCE loss as well as its equivalent numerically more stable formulation as:*

$$\mathcal{L}_{\text{SCE}}(f(\mathbf{x}), y) = -\log \left( \frac{e^{z_y}}{\sum_{k=1}^n e^{z_k}} \right) = -z_y + \max(\mathbf{z}) + \log \left( \sum_{k=1}^n e^{z_k - \max(\mathbf{z})} \right) \quad (1)$$

Unfortunately, even the rightmost (comparatively more stable) variant does not address this problem, since the kind of FP errors discussed in this work appear in the sum. While the Softmax function outputs are bounded between 0 and 1, the intermediate calculations involve summing exponentials of both positive and negative logits. These values can span several orders of magnitude, particularly in scenarios with large logits where the loss approaches zero. This wide range of values creates conditions that lead to absorption errors – leading to the phenomenon we call *Softmax Collapse*.

**Definition 3** (Softmax Collapse (SC)). *A specific case of absorption error occurs when, for a given sample  $\mathbf{x}$ , the logit from the correct class  $z_y$  is significantly larger than the logits for all other classes. This floating-point absorption of smaller terms, which we call **Softmax Collapse**, occurs when:*

$$\sum_{k=1}^n e^{z_k} \doteq e^{z_y}, \quad (2)$$

in which case the SCE loss becomes:

$$\mathcal{L}_{\text{SCE}}(f(\mathbf{x}), y) \doteq -\log \left( \frac{e^{z_y}}{e^{z_y}} \right) = 0. \quad (3)$$

Thus, during SC the loss becomes identical to zero. Furthermore, for the correct class, the gradients become zero as well:

$$\frac{\partial \mathcal{L}_{\text{SCE}}}{\partial z_c} = \frac{e^{z_c}}{\sum_{k=1}^n e^{z_k}} - \mathbb{1}_{\{c=y\}} \doteq 1 - \mathbb{1}_{\{c=y\}} = 0. \quad (4)$$

While weights that contribute to the wrong classes can still get negative updates, we show that disappearance of the gradients from the correct classes is enough to inhibit grokking (Fig. 2). We validate this in App. B.1 with an explicit intervention, showing that artificially setting the gradients from the correct class to zero stops generalization in a very similar way to what we observe in Fig. 2.

### 3.2 EVIDENCE OF SOFTMAX COLLAPSE IN GROKKING TASKS

Grokking is often studied using dataset sizes for which the delay in generalization is significant, which is usually when the dataset is small but just large enough that generalization is possible. In

{4}------------------------------------------------

this regime, regularization seems necessary for grokking and no improvement in test performance is observed without it (Nanda et al., 2023). However, a fact that has received less attention is that grokking can happen without regularization if the dataset is large enough (Power et al., 2022).

Here we hypothesize that as the size of the dataset decreases, overfitting becomes easier and Softmax Collapse (SC) happens earlier. To quantify this, we train an MLP without regularization on modular addition using different levels of FP precision, and calculate at every training epoch the fraction of samples that result in SC as per Eq. (2). The results support our hypothesis that SC is responsible for the model’s failure to generalize (Fig. 2). Specifically, we see that generalization stops when SC begins – and that this happens earlier under float32 than under float64 (Fig. 2b). Furthermore, this point is reached earlier as the dataset size decreases until it is reached before making any progress in the test accuracy, resulting in the common picture of no grokking without regularization (Fig. 2a).

### 3.3 PREVENTING SOFTMAX COLLAPSE LEADS TO GROKKING

To validate the importance of FP errors in stopping grokking, we show that methods to avoid SC lead to generalization on all the common grokking tasks on both MLPs and transformers. We introduce the following methods to postpone the appearance of FP errors.

**Increasing floating point precision.** The simplest way to avoid SC is to extend the FP precision from float32 to float64 for the Softmax calculation. We see in Fig. 2 that networks trained using float64 in the Softmax face SC later in training which allows for a further increase in test performance. Conversely, using float16 leads to SC earlier in training, leading to lower test performance. While this approach works as expected, FP precision cannot be extended indefinitely to allow for generalization as seen in the lack of grokking in Fig. 2a.

**StableMax Cross Entropy (StCE) Loss.** As demonstrated above, SC is caused by adding the exponentials of very large positive and negative logits in the Softmax. To avoid these extreme summands, we propose using a softer version of Softmax to transform logits into probabilities before calculating the CE Loss:

**Definition 4 (StableMax).** *We introduce a numerically stable version of the Softmax as:*

$$\text{StableMax}(x_i) := \frac{s(x_i)}{\sum_j s(x_j)}, \quad (5)$$

where

$$s(x) := \begin{cases} x + 1 & \text{if } x \geq 0, \\ \frac{1}{1-x} & \text{if } x < 0. \end{cases} \quad (6)$$

![Figure 3: A plot comparing the function s(x) (blue line) and the exponential function e^x (orange line). The x-axis ranges from -4 to 4, and the y-axis ranges from 0 to 6. The blue line s(x) is a piecewise linear function that is x+1 for x ≥ 0 and 1/(1-x) for x < 0. The orange line e^x is a smooth exponential curve. The two lines intersect at (0, 1). For x > 0, the exponential function grows much faster than the linear function. For x < 0, the linear function approaches 0 more slowly than the exponential function.](c82c7d8107cba121734a9cfba891216d_img.jpg)

Figure 3: A plot comparing the function s(x) (blue line) and the exponential function e^x (orange line). The x-axis ranges from -4 to 4, and the y-axis ranges from 0 to 6. The blue line s(x) is a piecewise linear function that is x+1 for x ≥ 0 and 1/(1-x) for x < 0. The orange line e^x is a smooth exponential curve. The two lines intersect at (0, 1). For x > 0, the exponential function grows much faster than the linear function. For x < 0, the linear function approaches 0 more slowly than the exponential function.

Figure 3:  $s(x)$  vs.  $e^x$ .

As seen in Fig. 3,  $s(\cdot)$  is a simple ramp function that scales linearly instead of exponentially when  $x \geq 0$  and also approaches 0 more slowly than the exponential function when  $x < 0$ . This is similar to the Softplus function (Dugas et al., 2000) but approaches 0 more slowly with negative logits, further reducing the risk of absorption errors.

**Proposition 1.** *StableMax is a modified Softmax, i.e.  $\text{StableMax}(x_i) = \text{Softmax}(g(x_i))$  where*

$$g(x) = \begin{cases} \log(x + 1) & \text{if } x \geq 0, \\ -\log(-x + 1) & \text{if } x < 0. \end{cases} \quad (7)$$

The proof of this Proposition is presented in App. A. We then define the numerically stable analogue of  $\mathcal{L}_{\text{SCE}}$  as  $\mathcal{L}_{\text{StCE}}(\bar{f}(\mathbf{x}), y) = -\log(\text{StableMax}(z_y))$ , where  $z_y$  again corresponds to the logit of the true class  $y$ .

To show that StCE indeed addresses the problems posed by SC, we repeat our experiments in Sec. 3.2 by replacing Softmax with StableMax. Our results, presented in Fig. 4, indeed show that StableMax leads to grokking in commonly studied settings *without* regularization. Notably, this happens while the norm of the weights increases substantially (Fig. 4, middle). This suggests that while weight decay may lead to both grokking and a decreasing weight norm, the decreasing

{5}------------------------------------------------

![Figure 4: Three plots showing grokking dynamics. Left: Accuracy (%) vs Epoch (0-100k) for addition mod 113. Middle: Weight norm (L2) vs Epoch (0-100k) for addition mod 113, product mod 113, and sparse parity. Right: Accuracy (%) vs Epoch (0-1000) for addition mod 113 with 2-hot, random, and binary inputs.](c54b3ca7603d65d4589151bc3a49d054_img.jpg)

The figure consists of three subplots. The left plot shows Accuracy (%) on the y-axis (0 to 100) against Epoch on the x-axis (0 to 100k). It features four curves: Test acc. - addition mod 113 (orange solid), Test acc. - product mod 113 (blue solid), Test acc. - sparse parity (green solid), and Train acc. (grey dashed). The sparse parity curve rises sharply to ~60% by epoch 10k and plateaus. The addition and product curves show a delayed but rapid increase in accuracy after epoch 40k, reaching near 100% by epoch 80k. The middle plot shows Weight norm (L2) on the y-axis (0 to 40k) against Epoch on the x-axis (0 to 100k). It features three curves: addition mod 113 L2 norm (orange solid), product mod 113 L2 norm (blue solid), and Sparse parity L2 norm (green solid). The addition and product norms increase steadily, while the sparse parity norm remains relatively flat. The right plot shows Accuracy (%) on the y-axis (0 to 100) against Epoch on the x-axis (0 to 1000). It features four curves: Test accuracy - 2-hot input (orange solid), Test accuracy - random binary input (green solid), Training accuracies (grey dashed), and 50% Softmax Collapse (grey dashed). The 2-hot input curve reaches 100% accuracy by epoch 200, while the random binary input curve reaches ~80% by epoch 1000.

Figure 4: Three plots showing grokking dynamics. Left: Accuracy (%) vs Epoch (0-100k) for addition mod 113. Middle: Weight norm (L2) vs Epoch (0-100k) for addition mod 113, product mod 113, and sparse parity. Right: Accuracy (%) vs Epoch (0-1000) for addition mod 113 with 2-hot, random, and binary inputs.

Figure 4: (left) Grokking with StCE loss and no regularization on three common grokking datasets using an MLP with 2 hidden layers of width 200. We use 40% of all pairs modulo 113 which is the same setting as Fig. 2a where regular SCE gets stuck at random level performance (random level is 50% for sparse parity). (middle) Evolution of model weight norms during training for the same models and tasks. This shows that grokking induced without weight decay does not follow the commonly observed trend of rapidly decreasing weight norm during generalization. (right) Changing input representations turns modular addition into regular machine learning tasks with train and test accuracy increasing in tandem, see Sec. 4.

weight norm is not necessary for grokking. Overall, these results i) provide additional evidence for the importance of SC in preventing grokking, ii) suggest a novel activation function to address this problem, and iii) show that regularization or weight norm modification is not *necessary* for grokking.

## 4 DIAGNOSING THE CAUSES OF SOFTMAX COLLAPSE

In the previous section we have shown that FP errors arise due to a combination of low losses and large logits, and shown that when FP errors are mitigated, grokking can be observed in conditions where it previously was not. In this section, we dive deeper and ask why extremely low losses and large logits appear in the first place in grokking tasks. We identify two main causes for this tendency: (i) easiness of overfitting in grokking tasks, and (ii) a training dynamic that sees gradients align with what we call *naïve loss minimization* direction. After diagnosing the causes, the following section will use these insights to develop an optimization algorithm that avoids NLM in the first place.

### 4.1 EASE OF OVERFITTING IN GROKKING TASKS

The first important characteristic of grokking tasks that lead to SC is their ease of overfitting. It has been observed that as grokking datasets get larger, overfitting becomes harder, eventually leading to a regime where train and test performances increase in tandem (Power et al., 2022; Nanda et al., 2023; Varma et al., 2023). It has also been shown that generalization can be delayed in the Sparse Parity task by increasing the amount of noise in the input, which makes overfitting easier (Barak et al., 2022). Here we investigate the opposite effect: that by decreasing the dimensionality of the input the data becomes harder to memorize, removing the delay in generalization.

To do this, we investigate the common grokking task of modular addition, but instead of the high-dimensional one-hot representations of the input integers, we use a more compact binary. More specifically, we assign each integer a distinct random binary vector of dimension 14.

Results confirm our hypothesis, showing that as input representations are decreased in dimension, overfitting is prevented and models generalize without need for regularization (Fig. 4, right). This also shows that modular addition only induces grokking depending on the choice of representation. These findings highlight the importance of understanding the training dynamics beyond the point of overfitting (i.e. point of achieving 100% training accuracy), rather than focusing on the specifics of the modular arithmetic tasks as the key to explaining the delay in generalization.

### 4.2 NAÏVE LOSS MINIMIZATION

We next identify a crucial training dynamic that commonly occurs in grokking tasks as a central cause for increasing logits and SC. We find that after reaching 100% training accuracy, gradient updates are dominated by an update direction we term *naïve loss minimization* (NLM). This direction

{6}------------------------------------------------

![Figure 5: Three line plots showing Cosine Similarity (W, ∇L) vs Epoch for different neural network architectures. (a) MLP without bias terms: shows similarity for layers 0, 1, and 2 weights and biases. (b) MLP with bias terms: shows similarity for layers 0, 1, and 2 weights and biases. (c) Transformer with bias terms: shows similarity for embed.W_E, blocks.0.mlp.W_in, blocks.0.mlp.W_n, blocks.0.mlp.W_out, unembed.W_U, and 100% train accuracy. All plots show a sharp increase in similarity around 100% train accuracy (dashed vertical line), indicating overfitting.](73c3e4508cae529acf4e6c7fa70b361a_img.jpg)

Figure 5: Three line plots showing Cosine Similarity (W, ∇L) vs Epoch for different neural network architectures. (a) MLP without bias terms: shows similarity for layers 0, 1, and 2 weights and biases. (b) MLP with bias terms: shows similarity for layers 0, 1, and 2 weights and biases. (c) Transformer with bias terms: shows similarity for embed.W\_E, blocks.0.mlp.W\_in, blocks.0.mlp.W\_n, blocks.0.mlp.W\_out, unembed.W\_U, and 100% train accuracy. All plots show a sharp increase in similarity around 100% train accuracy (dashed vertical line), indicating overfitting.

Figure 5: MLPs with (a) and without (b) bias terms trained on modular addition receive updates that are significantly aligned with the direction of NLM beyond the point of overfitting. In (c) we show these results for a selection of parameters for our one layer transformer. We highlight the embed and unembed matrices as well as the weights of the MLP. These are highlighted in the plot using the notation from Elhage et al. (2021).

does not change the model’s decision boundary, but still decreases loss by simply scaling the logits of the predictions, in most cases through scaling of parameters (see below). This means that the logits will continue to increase until they inevitably lead to SC and zero terms in the training gradient. This stops the parameter updates in any direction, including NLM and any other useful component that would have been included in the overall gradient. We now define NLM formally, and proceed to discuss why it might commonly be observed to deteriorate training in grokking tasks. Given the input  $\mathbf{x} \in \mathcal{X}$ , output  $y \in \mathcal{Y}$ , a predictor  $f$  parametrized by  $\theta \in \mathbb{R}^m$  that outputs logits  $\mathbf{z} = f(\theta; \mathbf{x}) \in \mathbb{R}^{|\mathcal{Y}|}$ , and a loss function  $\mathcal{L}$ , we now define Naïve Loss Minimization.

**Definition 5** (Naïve Loss Minimization (NLM)). *A function  $d_{\text{NLM}} : \mathbb{R}^m \rightarrow \mathbb{R}^m$  specifies a direction of naïve loss minimization if it decreases the loss,*

$$\mathcal{L}(f(\theta + d_{\text{NLM}}(\theta); \cdot)) < \mathcal{L}(f(\theta; \cdot)), \quad (8)$$

while satisfying for some  $c > 1$ :

$$f(\theta + d_{\text{NLM}}(\theta); \mathbf{x}) = cf(\theta; \mathbf{x}), \quad \forall \mathbf{x} \in \mathcal{X}, \quad (9)$$

where  $\mathcal{X}$  denotes the input space and  $\mathcal{L}(f(\theta + d_{\text{NLM}}(\theta); \cdot))$  is the total loss over the training dataset.

We find that under a large class of models, namely those that demonstrate *positive homogeneity*, when training beyond 100% training accuracy the direction of the weights is an NLM direction.

**Definition 6** (Positive Homogeneity (Lyu & Li, 2020)). *A function  $f$  is positively homogeneous of degree  $L > 0$  if for all weights  $\theta$ , inputs  $\mathbf{x}$ , and scalars  $c > 0$ , it satisfies:*

$$f(c\theta; \mathbf{x}) = c^L f(\theta; \mathbf{x}). \quad (10)$$

When  $f$  is a homogeneous neural network,  $L$  corresponds to the number of layers.

In the case of homogeneous networks, training beyond 100% training accuracy, scaling the logits always leads to a decrease in the training loss. Therefore,  $d_{\text{NLM}}(\theta) = \alpha\theta$  for  $\alpha > 0$  is an NLM direction, as it results in  $f(\theta + d_{\text{NLM}}(\theta); \mathbf{x}) = f((1 + \alpha)\theta; \mathbf{x}) = (1 + \alpha)^L f(\theta; \mathbf{x})$ , where the second equality follows from Eq. (10).

Many neural network architectures, such as ReLU MLPs and transformers without bias terms, are *positively homogeneous* or *approximately homogeneous* in the case of transformers (Merrill et al., 2020). While more complex deep learning models with skip connections and bias terms are not homogeneous, they have been shown to be quasi-homogeneous (Kunin et al., 2023) and in most cases – including all of the models in this work, the last layer is homogeneous. This means that for non-homogeneous models scaling the weights of the last layer corresponds to a direction of NLM.

The fact that the gradients converge to the direction of the weights has been studied in previous works (Ji & Telgarsky, 2020; 2019; 2018; Lyu & Li, 2020) to prove that homogeneous networks converge in direction under gradient flow and gradient descent (GD), and they perform normalized margin maximization even beyond the point of 100% training accuracy (Lyu & Li, 2020). However,

{7}------------------------------------------------

![Figure 6: Comparing optimizer performance. (a) Transformer, subtract. mod 113: Accuracy vs Epoch (0-5000). (b) MLP, addition mod 113: Accuracy (%) vs Epoch (0-500). (c) Trade-off between L2 and SCE: Log Loss vs Log Epoch (-2 to 1).](3121afa7ca030b22ee0345864ca6f38b_img.jpg)

(a) Transformer, subtract. mod 113 (b) MLP, addition mod 113 (c) Trade-off between L2 and SCE

Figure 6: Comparing optimizer performance. (a) Transformer, subtract. mod 113: Accuracy vs Epoch (0-5000). (b) MLP, addition mod 113: Accuracy (%) vs Epoch (0-500). (c) Trade-off between L2 and SCE: Log Loss vs Log Epoch (-2 to 1).

Figure 6: Comparing  $\perp$ AdamW and  $\perp$ SGD with baseline optimizers and AdamW with weight decay on (a) a transformer trained on subtraction mod 113 and (b) an MLP trained on addition modulo 113. In (c) we highlight the trade-off between L2 regularization and SCE loss, initially SCE loss is reduced at the cost of increasing the L2 loss but eventually the two losses decrease simultaneously (Sec. 5.2).

we argue that gradient alignment also results in scaling of the logits which can lead to SC and put an end to the margin maximization described in Lyu & Li (2020), when working with limited floating point precision. While we study delayed generalization, the link between training trajectories and generalization is already established in prior art (Birdal et al., 2021; Andreeva et al., 2024).

**Evidence of naïve loss minimization.** In practice, we observe that in MLPs and transformers with and without bias terms, the gradients quickly become aligned with the direction of the weights after the point of overfitting (Fig. 5). Particularly for the later layers of the models, the cosine similarity between the parameter updates and the NLM direction goes up to 0.9 for the output layers. While models with bias terms are not homogeneous and there is no theoretical guarantee that scaling the weights will reduce the SCE loss, in practice, we observe very similar behavior in MLPs with (Fig. 5b) and without (Fig. 5a) bias terms. In the case of a one-layer transformer, the alignment is stronger for the embed and unembed matrices but also substantial for the MLP weights (Fig. 5c).

## 5 MITIGATING NAÏVE LOSS MINIMIZATION LEADS TO GROKKING

While we have shown in Sec. 3 that avoiding numerical instabilities eventually leads to generalization, we can also target the NLM process that causes these numerical issues. To do this, we design an optimizer that only preserves the part of the gradient orthogonal to the direction of the weights.

### 5.1 $\perp$ Grad: AN OPTIMIZER TO PREVENT NLM

We propose a new optimizer,  $\perp$ Grad (read “ortho-grad”), that updates the weights based only on the part of the gradient that is orthogonal to the current direction of the weights:

**Definition 7** ( $\perp$ Grad). *We propose the following update rule for a given iteration  $t \in \mathbb{N}$ :*

$$\theta_{t+1} = \theta_t - \eta \nabla_{\perp} \mathcal{L}(\theta_t), \quad (11)$$

where the orthogonal component of the gradient,  $\nabla_{\perp} \mathcal{L}(\theta_t)$ , is obtained by projection onto the hyperplane orthogonal to the current weight vector:

$$\nabla_{\perp} \mathcal{L}(\theta_t) = \nabla \mathcal{L}(\theta_t) - \left( \frac{\theta_t^\top \nabla \mathcal{L}(\theta_t)}{\theta_t^\top \theta_t} \right) \theta_t. \quad (12)$$

**Proposition 2.** *Assuming  $\nabla_{\perp} \mathcal{L}(\theta_t) \neq 0$ ,  $\exists \beta > 0$  such that for any learning rate  $0 < \eta < \beta$ , taking the step  $\eta \nabla_{\perp} \mathcal{L}(\theta_t)$  reduces the loss. In other words, any nonzero  $\nabla_{\perp} \mathcal{L}(\theta_t)$  is a descent direction.*

*Sketch of the proof.* We show that any  $\nabla_{\perp} \mathcal{L}(\theta_t) \in \mathbb{R}^m \setminus \{0\}$  is a descent direction by demonstrating that  $\langle -\nabla_{\perp} \mathcal{L}(\theta_t), \nabla \mathcal{L}(\theta_t) \rangle < 0$ . For a full proof we refer the reader to App. A.  $\square$

This projection of the gradient can be incorporated into different optimizers. In Fig. 6a, we show results for  $\perp$ AdamW and  $\perp$ SGD, the  $\perp$ Grad versions of AdamW and SGD respectively. These results show that  $\perp$ Grad optimizers lead to generalization without a phase of initial overfitting, in contexts where no improvement in test performance is usually observed without weight decay. We

{8}------------------------------------------------

![Figure 7: Model trajectories in parameter space projected to 2D over the SCE loss landscape. (a) Training loss landscape: The plot shows Principal Component #1 on the x-axis and Principal Component #2 on the y-axis. A color bar on the right indicates 'Train Loss' from 0.10 to 0.40. Four trajectories are shown: SGD (red line), SGD + weight decay (blue line), LSGD + StatMolux (green line), and NLM direction (yellow line). All trajectories start at a point labeled 'Start'. The SGD trajectory moves towards the right, while the others move towards the left. (b) Test loss landscape: The plot shows the same axes. A color bar on the right indicates 'Test Loss' from 0.0 to 0.25. The same four trajectories are shown. The SGD trajectory moves towards the right, while the others move towards the left. The SGD trajectory starts at a point labeled 'Start' and moves towards the right, while the other trajectories move towards the left.](b93cbfb52e37619e688175a6aad9edd9_img.jpg)

Figure 7: Model trajectories in parameter space projected to 2D over the SCE loss landscape. (a) Training loss landscape: The plot shows Principal Component #1 on the x-axis and Principal Component #2 on the y-axis. A color bar on the right indicates 'Train Loss' from 0.10 to 0.40. Four trajectories are shown: SGD (red line), SGD + weight decay (blue line), LSGD + StatMolux (green line), and NLM direction (yellow line). All trajectories start at a point labeled 'Start'. The SGD trajectory moves towards the right, while the others move towards the left. (b) Test loss landscape: The plot shows the same axes. A color bar on the right indicates 'Test Loss' from 0.0 to 0.25. The same four trajectories are shown. The SGD trajectory moves towards the right, while the others move towards the left. The SGD trajectory starts at a point labeled 'Start' and moves towards the right, while the other trajectories move towards the left.

Figure 7: Model trajectories in parameter space projected to 2D over the SCE loss landscape. SGD with weight decay starts along the same trajectory as SGD decreasing the training loss (a) but increasing the test loss (b).

note that similar projections of the gradients have been used in other settings to mitigate the effects of momentum in invariant layers (Heo et al., 2021), stabilize training Wang et al. (2024) or as one part in a more complex optimizer (Kosson et al., 2024). We design  $\perp$ Grad as a more precise intervention that directly prevents scaling along the NLM direction.

In Fig. 7, we compare the trajectories of models using SGD with and without weight decay to our new  $\perp$ SGD optimizer. SGD models start on a similar trajectory, reducing the training loss but increasing the test loss, until the model with weight decay changes direction and starts minimizing both the train and test loss. In contrast, the model using  $\perp$ SGD moves directly in a direction that minimizes both the train and test loss. While SGD with weight decay eventually reaches a point of lower loss, note that  $\perp$ SGD reaches 100% test accuracy within 400 iterations (Fig. 6a). Beyond showing how  $\perp$ SGD prevents NLM, Fig. 7 also suggests that weight decay induces grokking by avoiding NLM. In the following, we highlight that the success of several methods to induce grokking can be explained from this perspective.

### 5.2 EXPLAINING THE SUCCESS OF EXISTING METHODS FOR GROKKING

In light of our findings, we are able to explain the success of several previously proposed methods to induce grokking. We find that these methods also lead to grokking by mitigating NLM and avoiding the FP errors that come with extremely low losses.

**Weight decay.** We have argued that the problem faced in grokking is that the ease of overfitting leads to NLM, which corresponds to scaling up the weights for homogeneous networks. Since weight decay corresponds to pulling back the weights along this same direction at every step during training, it is unsurprising, given our findings, that it is the most reliable way to induce grokking.

To explain why generalization tends to be delayed when using weight decay, as opposed to  $\perp$ Grad, we look at it from the perspective of L2 regularization which is equivalent to weight decay for SGD. In Fig. 6c, we see an initial phase where classification loss decreases, at the cost of the L2 loss. Eventually, the decrease in classification loss from NLM stops outweighing the increase in L2 loss, meaning that only updates that are not aligned with the NLM direction are followed. This explains why weight decay leads to generalization in grokking tasks but only after scaling along the NLM direction no longer decreases the overall loss. This balance between weight decay and classification loss is similar to the rotational equilibrium studied in Kosson et al. (2024).

We argue that the main roles of weight decay are preventing floating point errors and preventing NLM. This is in line with recent findings about the role of weight decay in deep learning (D’Angelo et al., 2023) which point to the fact that it increases the effective learning rate and avoids floating point issues when using mixed-precision training in LLMs.

**MSE loss on shallow networks.** While cross-entropy loss can be reduced indefinitely by scaling the logits through NLM, this is not the case with MSE loss. When using MSE loss the logits can overshoot the target, meaning that larger logits often do not lead to a lower MSE loss. This explains why Barak et al. (2022), Kumar et al. (2024), and Lyu et al. (2024) observed grokking with MSE loss without regularization. Interestingly, networks with more than one hidden layer do not generalize in these same settings (Fig. 13).

**Delaying generalization by scaling the weights.** While the lazy training dynamics described in Kumar et al. (2024) explain an important part of why scaling the weights delays generalization,

{9}------------------------------------------------

we show that the reason that regularization is often needed to exit this lazy training regime is that scaling the weights or the logits facilitates SC. In App. D.2, we show that the setting used in Liu et al. (2023b) to induce grokking on MNIST with SCE also induces SC which prevents further learning in the absence of weight decay.

## 6 RELATED WORK

**Grokking.** Power et al. (2022) introduced grokking and showed that weight decay can consistently induce it in algorithmic tasks. Nanda et al. (2023) were able to reverse engineer the inner workings of a grokked transformer and found progress measures for grokking induced by weight decay. Chughtai et al. (2023) generalized the findings from Nanda et al. (2023) and showed grokked networks use group representations to solve group composition tasks, although some of these findings were disputed in Stander et al. (2024) which propose that grokked networks learn a coset based algorithm for these same tasks. Mallinar et al. (2024) has shown that grokking is not specific to neural networks or gradient-based optimization and cannot be predicted from the training or test loss. Varma et al. (2023) argued that grokking is driven by weight decay favoring more efficient solutions and Liu et al. (2023b) hypothesized that the weight norm of the models needs to be in a “Goldilock’s zone” to generalize. Kumar et al. (2024) and Lyu et al. (2024) connected grokking to a transition between “lazy training” (Chizat et al., 2018) and feature learning, and Kumar et al. (2024) showed that this can happen without regularization in the case of shallow networks with MSE loss. Grokking has also been described as a phase transition by Žunkovič & Ilievski (2024), Lyu et al. (2024) and Rubin et al. (2024). Humayun et al. (2024) show that in many settings, neural networks undergo grokking-like transitions in their adversarial robustness. This aligns with the findings of Lyu & Li (2020) which attributed this increased robustness to a bias of SGD towards a max-margin solution which was proven for homogeneous models. Beck et al. (2024) also connected grokking to the linear separability of the training data.

**Numerical instability in deep learning.** Numerical instability is a common issue in deep learning (Kloberdanz et al. (2022), especially when dealing with mixed precision training (D’Angelo et al. (2023)). It is known that the Softmax function is particularly prone to numerical stability problems although this often comes in the form of overflow in the exponential (Kloberdanz et al., 2022) and not from absorption errors in the sum as observed in this case. In the grokking setting, Nanda et al. (2023) showed that the slingshots observed in Thilak et al. (2022) can be explained by a very similar mechanism to the one involved in SC, although Nanda et al. (2023) do not use it to explain any grokking phenomena beyond these spikes that sometimes appear in the training process in grokking tasks. We believe the slingshots observed in Thilak et al. (2022) could be a mechanism to prevent full SC, explaining why slingshots can lead to grokking without weight decay in some settings. This is further discussed in App. H. Issues with numerical instability when training beyond overfitting with increasing learning rates were also observed in Lyu & Li (2020).

## 7 CONCLUSION AND DISCUSSION

In this work, we show that naïve loss minimization (NLM) and floating point errors can explain why generalization is delayed in grokking and why it often does not happen without regularization. Using this insight, we are able to explain the success of existing methods to induce grokking. Motivated by our findings, we further design a simple modification to the Softmax that induces grokking by avoiding floating point errors and an optimizer that avoids the delay in generalization in grokking by preventing NLM.

**Limitations & future work.** While this work explains several surprising aspects of grokking settings, several questions remain. Notably, we focus our study of NLM on homogeneous or approximately homogeneous models. A formal characterization quasi-homogeneous models could shed light on this kind of dynamics for models including skip connections and bias terms. Additionally, our explanation for why weight decay causes grokking could be enhanced by an analysis of its impact on the effective learning rate as a potential explanation for the sudden nature of grokking.

**Acknowledgments.** This work was supported by the UKRI Centre for Doctoral Training in Safe and Trusted AI [EP/S023356/1]. TB acknowledges support from the Engineering and Physical Sciences Research Council [grant EP/X011364/1]. TB was supported by a UKRI Future Leaders Fellowship [grant number MR/Y018818/1].

 Rest of paper (reference and Appendix) is removed.