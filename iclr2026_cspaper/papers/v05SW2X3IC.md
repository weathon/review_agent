# Lossy Common Information In A Learnable Gray-Wyner Network

Anderson de Andrade, Alon Harell & Ivan V. Bajic´ School of Engineering Science Simon Fraser University Burnaby, BC, Canada {anderson de andrade,alon harell,ibajic}@sfu.ca

## Abstract

Many computer vision tasks share substantial overlapping information, yet conventional codecs tend to ignore this, leading to redundant and inefficient representations. The Gray-Wyner network, a classical concept from information theory, offers a principled framework for separating common and task-specific information.

Inspired by this idea, we develop a learnable three-channel codec that disentangles shared information from task-specific details across multiple vision tasks. We characterize the limits of this approach through the notion of lossy common information, and propose an optimization objective that balances inherent tradeoffs in learning such representations. Through comparisons of three codec architectures on two-task scenarios spanning six vision benchmarks, we demonstrate that our approach substantially reduces redundancy and consistently outperforms independent coding. These results highlight the practical value of revisiting Gray-Wyner theory in modern machine learning contexts, bridging classic information theory with task-driven representation learning.

## 1 Introduction

It is often the case that a machine task - classification, recognition, etc. - requires only a subset of the information provided as input. We can interpret neural networks as processes that discard irrelevant information from a signal so that its predictions are in a probability space similar to that of the target (Tishby & Zaslavsky, 2015). In multi-task settings, the same input is used to perform different tasks. These tasks have semantically different targets and hence might require different subsets of the input information. Whenever tasks are not performed jointly, isolating the information required for each one is critical to ensure that communication is efficient. For example, when only object detection is needed, it would be efficient for a camera to only transmit the information required so the receiving device can perform that task. If it is then decided that semantic segmentation is needed for the same input, it would be efficient for the camera to only transmit the *additional* information necessary, considering that some relevant information has already been transmitted. On the receiver device, it would be efficient if the information from the first transmission, that is also relevant for semantic segmentation, is isolated, so it can be readily used without overhead. The line of work in *coding for humans and machines* (Choi & Bajic, 2022) focuses on this problem. It considers an image reconstruction task together with a computer vision task. It is commonly assumed that all the information used by the computer vision task is relevant to the reconstruction task. Thus, only two separate channels (representations) are often designed: a common channel used by both tasks, and a private channel only used by the reconstruction task. In this work, we focus on a pair of tasks that have some common information (CI) between them, but also private (dedicated) information for each task, establishing three channels: a common channel, and two private channels. We establish in this work that a complete isolation of the common information needed between two tasks is often unattainable. This also occurs in *lossy coding* (compression), where we send less information (*rate*, e.g. *bitrate*), which inevitably performs the tasks with a higher error (*distortion*). A system must then decide between transmitting some of the non-common information on the common channel, or some of the common information on the private channels. In the former case, the transmit rate - the amount of information transmitted when performing both tasks on one device - can be optimal, but the *receive rate* - the amount of information transmitted by performing each task on a different device - will not be optimal. The converse is true in the latter case. We propose a neural network architecture that is able to separate the common information between two tasks. We compare the proposed architecture against other more intuitive architectures we designed, and include a theoretical justification for their difference in performance. We also propose a loss function that can optimize a codec for the transmit rate, the receive rate, or a tradeoff between both. We show how our methods perform on different pairs of computer vision tasks.

## 2 Previous Work

In information theory, *source coding* methods convert a sequence of symbols from an information source into a sequence of bits, allowing data compression. Our work is closely related to the Gray- Wyner Network (GWN) (Gray & Wyner, 1974). It is a source coding problem connecting two sources with two receivers via a common channel and two private channels. The work defines the Gray-Wyner region as a set of achievable rates (e.g. bitrates) for the three channels, in both lossy and lossless input reconstruction tasks. There are multiple notions of common information in literature (Viswanatha et al., 2014), including mutual information. In this work, we discuss Wyner's common information (Wyner, 1975) and Gacs-K ´ orner (GK) common information (Gacs & K ¨ orner, 1973). These quantities are located in the ¨ Gray-Wyner achievable region. They are related by a tradeoff between the total transmit and receive rate (Viswanatha et al., 2014).

The focus of seminal work on *learnable image coding* (Theis et al., 2017; Balle et al., 2018; He ´
et al., 2022) has been image lossy coding (compression) as a single task. They consist of a lossy analysis transform (encoder) and a *synthesis transform* (decoder) acting as an autoencoder (Balle´ et al., 2018) such that the latent representation is better suited for coding, achieving a lower rate. An *entropy model* uses various types of context to predict the distribution of a target representation. When optimizing for rate-distortion performance (Theis et al., 2017), the entropy model effectively induces an information bottleneck (Tishby et al., 1999) on the target representation. A higher penalty on the probability estimates of the entropy model for the target representation produces lower rates at the expense of a higher task distortion (error). The work in coding for humans and machines uses similar architectures to those used in learnable image coding. In Choi & Bajic (2022), the output representation of an analysis transform is split in two. The computer vision task uses one representation over a common channel. The image reconstruction task uses the other representation over a private channel in addition to the representation from the common channel. Better rate-distortion performance on the computer vision task was achieved when using two separate analysis transforms, one for each of these channels (Foroutan et al., 2023). However, the common channel had information that was not fully utilized by the reconstruction task. An ad-hoc reconstruction task for the common channel was shown to increase the usability (compatiblity) of the corresponding representations (de Andrade & Bajic, 2024). Multitask learnable codecs exist in the literature (Chamain et al., 2021; Feng et al., 2022; Guo et al., 2024). They propose one or more common channels to perform several tasks, without private channels. Their rate is optimal only when all the tasks involved are performed jointly. In representation learning literature, several information-theoretic approaches propose variational autoencoders (VAEs) that learn disentangled representations of a source (Chen et al., 2016; Higgins et al., 2017; Chen et al., 2018b). These approaches are unsupervised in nature and thus do not distill information for a specific task or isolate the common information between tasks. A more related approach (Dubois et al., 2021) proposes a channel that can achieve high performance in a set of predictive tasks, as long as they are invariant under a set of transformations. Although their proposed methods are also unsupervised, the right set of transformations can be used to isolate common information between tasks. However, finding these transformations for non-trivial tasks is an open question. Several variational methods have been proposed to measure the mutual information between two sources (Poole et al., 2019). Although they might seem useful as training objectives,

![2_image_0.png](2_image_0.png)

Figure 1: Diagram of the Gray-Wyner Network and a lower bound on its achievable region. The lower bound is given by planes A, B, and C. Plane A corresponds to the *Pangloss*, the contour of the achievable region that achieves RX1,X2(D1, D2). Points with R0 = C(X1, X2; D1, D2) are found on it. Planes B and C achieve R0 + R1 = RX1(D1) and R0 + R2 = RX2(D2), respectively. On the intersection of both of these planes, we can find a point with R0 = K(X1, X2; D1, D2). A
trace connecting both points offers tradeoffs between the transmit and receive rates. If the mutual information between tasks, at distortions (D1, D2), is separable, both points coincide at the now achievable *Separable* point. We show that points close to it, whether achievable or not, serve as bounds for both lossy common information measures. these methods have significant tradeoffs between bias and variance. Moreover, they do not naturally offer the means to code (compress) the resulting representations. Compared against these existing techniques, our work directly addresses the isolation of common information between tasks and its efficient transmission.

## 2.1 Preliminaries

We present the Gray-Wyner Network (Gray & Wyner, 1974) and define the two notions of common information that define the boundaries of the transmit-receive tradeoff on which we want to operate. Let X1 and X2 be two source random variables. In the Gray-Wyner Network, an analysis transform (encoder) (Y0, Y1, Y2) = f(X1, X2) produces one common and two private discrete representations with corresponding rates (R0, R1, R2). Two synthesis transforms (decoders)
Zˆ1 = g1(Y0, Y1) and Zˆ2 = g2(Y0, Y2) predict the targets Z1 and Z2 of two dependent tasks, producing distortions D1 = d1(Zˆ1, Z1) = E[
ˆd1(Zˆ1, Z1)] and D2 = d2(Zˆ2, Z2) = E[
ˆd2(Zˆ2, Z2)], respectively. In our context, the distortion functions ˆd1 and ˆd2 are task losses for targets Z1 and Z2. We assume that one source does not have exclusive information that could assist its non-corresponding task, implying the Markov conditions:
Z2 ↔ X2 ↔ X1, Z1 ↔ X1 ↔ X2. (1)

$$Z_{2}\leftrightarrow X_{2}\leftrightarrow X_{1},$$

The set of all *achievable* rate tuples (R0, R1, R2) for distortions (D1, D2) is given by the Gray-
Wyner region RGW(D1, D2). The contour of this region lower-bounds all other points in this convex set and are considered optimal in that they optimize a particular rate-distortion function. A rate-distortion function determines the minimal rate that should be communicated over a channel, so that input signals (sources) can be reconstructed without exceeding expected distortions. The joint rate-distortion function (Cover & Thomas, 2006) is given as RX1,X2
(D1, D2). In our context, we define it as the minimum rate required to encode (Zˆ1,Zˆ2) jointly as a function of (X1, X2) so that tasks (ˆd1,ˆd2) can be performed with at most (D1, D2) distortion. Marginal rate-distortion functions are given as RX1(D1) and RX2(D2), with similar definitions applying only to one task and one corresponding source. The conditional rate-distortion function RX1|X2
(D1) is the minimum rate required to encode a function of X1, such that the task ˆd1 can be performed with at most distortion D1, when X2 is available to both the encoder and the decoder.

$$Z_{1}\leftrightarrow X_{1}\leftrightarrow X_{2}.$$
$$(1)$$

In this setting, using previous definitions, we define the Wyner's lossy common information as:
C(X1, X2; D1, D2) = inf I(X1, X2;U), (2)
where the infimum is over all joint densities P(X1, X2,Zˆ1,Zˆ2, U), under these Markov conditions:
Zˆ1 ↔ U ↔ Zˆ2, (X1, X2) ↔ (Zˆ1,Zˆ2) ↔ U, (3)
and where P(Zˆ1,Zˆ2|X1, X2) ∈ PX1,X2 D1,D2 is any joint distribution that achieves the rate-distortion function at (D1, D2), i.e.: I(X1, X2;Zˆ1,Zˆ2) = RX1,X2(D1, D2). The mutual information function I(·; ·) measures the dependence between two random variables (Cover & Thomas, 2006). For the conditions in 3 to hold, U must have *at least* the mutual information between Zˆ1 and Zˆ2.

Gacs-K ´ orner lossy common information is defined for our particular setting as: ¨

$$(4)$$
$$K(X_{1},X_{2};D_{1},D)$$

K(X1, X2; D1, D2) = sup I(X1, X2; V ), (4)
where the supremum is over all joint densities P(X1, X2,Zˆ1,Zˆ2, V ), such that the following Markov conditions are met:
X2 ↔ X1 ↔ V, X1 ↔ X2 ↔ V, X1 ↔ Zˆ1 ↔ *V, X*2 ↔ Zˆ2 ↔ V, (5)
where P(Zˆ1|X1) ∈ PX1 D1 and P(Zˆ2|X2) ∈ PX2 D2 are optimal rate-distortion encoders at D1 and D2, respectively, i.e: I(X1;Zˆ1) = RX1(D1) and I(X2;Zˆ2) = RX2(D2). The first two conditions in 5 establish that V cannot have information about X1 or X2 that is not mutual. The last two conditions in 5 establish that all information in V is present *in both* Zˆ1 and Zˆ2.

Wyner's lossy common information, C(X1, X2; D1, D2), can be thought of as the least information that must be included in the common channel to avoid the total transmit rate exceeding the optimal RX1,X2(D1, D2). Conversely, Gacs-K ´ orner common information, ¨ K(X1, X2; D1, D2), is the most information that can be included in the common channel while maintaining the optimal receive rate RX1(D1) + RX2(D2).

The total transmit rate is Rt = R0 +R1 +R2 ≥ RX1,X2(D1, D2) and the total receive rate is Rr = 2R0 + R1 + R2 ≥ RX1(D1) + RX2(D2). Wyner's and Gacs-K ´ orner common information can be ¨
related through these concepts (Viswanatha et al., 2014). When the transmit rate is optimal, such that Rt = RX1,X2
(D1, D2), we have that the minimum Rr is RX1,X2
(D1, D2) + C(X1, X2; D1, D2).

Such points in the contour of RGW(D1, D2) are achieved when R0 = C(X1, X2, D1, D2). When the receive rate is optimal such that Rr = RX1
(D1) + RX2
(D2), we have that the minimum Rt is RX1
(D1) + RX2
(D2) − K(X1, X2; D1, D2). This point in the contour of RGW(D1, D2) is achieved when R0 = K(X1, X2, D1, D2). This implies that minimizing one type of rate comes at the cost of increasing the other, establishing the transmit-receive tradeoff.

Figure 1 visually describes these preliminary concepts. See Appendix A.1 for more definitions.

## 3 Contributions 3.1 Bounds For Lossy Common Information

We extend a result from Wyner (1975) in the lossless setting to the lossy case, showing bounds that separate the two lossy common information terms discussed. The bounds are expressed in terms of interaction information I(X1, X2;Zˆ∗
1;Zˆ∗2), which is a generalization of the mutual information for more than two variables (Ting, 1962; Yeung, 1991). See Appendix A.1 for its definition.

Theorem 1. Let Zˆ
(t)
D1,D2 be the set of tuples (Zˆ1,Zˆ2) that achieve RX1,X2(D1, D2)*, and* Zˆ
(r)
D1,D2 be the set of tuples (Zˆ1,Zˆ2), such that Zˆ1 achieves RX1(D1), and Zˆ2 achieves RX2(D2)*. Then:*

$$K(X_{1},X_{2};D_{1},D_{2})\leq\max_{(\hat{Z}_{1},\hat{Z}_{2})\in\hat{Z}_{D_{1},D_{2}}^{(t)}}I(X_{1},X_{2};\hat{Z}_{1};\hat{Z}_{2})$$ $$\leq\min_{(\hat{Z}_{1},\hat{Z}_{2})\in\hat{Z}_{D_{1},D_{2}}^{(t)}}I(X_{1},X_{2};\hat{Z}_{1};\hat{Z}_{2})\leq C(X_{1},X_{2};D_{1},D_{2}).$$

We have equality everywhere iff the maximum and minimum coincide at (Zˆ∗
1,Zˆ∗2)*, and we can* represent Zˆ∗
1 as (Zˆ′1, W) and Zˆ∗2 as (Zˆ′2, W) *such that Conditions 3 and 5 hold for* Zˆ∗
1, Zˆ∗
2, and W.

$$(6)$$
$$\left(7\right)$$

## Proof. See Appendix A.

Theorem 1 allows to interpret the lossy versions of Gacs-K ´ orner and Wyner's common information ¨
similarly to their lossless counterparts. Any set of tuples (Zˆ1,Zˆ2) that achieve the transmit rate will always have at least the amount of interaction information I(X1, X2;Zˆ1;Zˆ2) of any set of tuples
that achieve the receive rate. If there is a gap of interaction information between the transmit and receive sets of tuples, exploring the transmit-receive tradeoff will produce tuples outside of those two sets that cover such gap. The conditions for equality everywhere imply that the two common information terms are the same
when the interaction information I(X1, X2;Zˆ∗
1;Zˆ∗2) is fully separable from private information or
other excess common information that does not help reach optimal rate-distortion values. Because of the latter reason, this separation can be even more difficult to attain than in the lossless case. To see this, note that Gacs-K ´ orner common information in the discrete case is the entropy of the ¨ probability distribution of a point in the sample space of (X1, X2) belonging to a partition in an
ergodic decomposition of the stochastic matrix defining (X1, X2) (Gacs & Korner, 1973). This ¨
means that to achieve I(X1, X2;Zˆ∗1
;Zˆ∗
2
), this information must be separable such that the stochastic
matrix P(Zˆ1,Zˆ2) can be written as:

A1 0
...
0 AK

 (8)
$$0$$
$${\mathcal{A}}_{1}$$

$$({\boldsymbol{\delta}})$$
$$\mathrm{0}$$
$$A_{K}$$
where A1*, ..., A*K are probability matrices defined as product of marginals, i.e., with no mutual information. This implies that these random variables can be represented as Zˆ∗
1 = (Zˆ′1, W) and Zˆ∗
2 = (Zˆ′2, W), which, with W being a function of X1 or X2, implies that it can be isolated from X1 or X2. If some of the required mutual information is not separable, it must be left out of W
and thus I(X1, X2;Zˆ∗1;Zˆ∗
2) cannot be produced. The work of Gacs & Korner (1973) shows that ¨
Gacs-K ´ orner common information is often very small. In fact, it is zero for Gaussian sources with ¨ correlation 1−ρ (Viswanatha et al., 2014), which is usually a distribution producing *refinable* results in information theory (Cover & Thomas, 2006). Because we can often expect in practice to have a noticeable gap between the two common information measures discussed, there is a significant motivation to explore the transmit-receive tradeoff.

## 3.2 Transmit-Receive Tradeoff Optimization

As previosly discussed, there is a tradeoff between optimizing for the transmit rate, resulting in more information available on the common channel than the mutual information, which increases the receive rate, or optimizing for the receive rate, resulting in less than the mutual information on the common channel, which increases the transmit rate. We propose an objective that can optimize for this tradeoff. The work of Gray & Wyner (1974) establishes the following objective for optimizing the Gray- Wyner Network:
T(α1, α2; D1, D2) ≜ inf I(X1, X2; Y0) + α1RX1|Y0
(D1) + α2RX2|Y0
(D2)	, (9)
where 0 ≤ α1, α2 ≤ 1, and α1 + α2 ≥ 1, and the infimum is over all probability distributions PY0 ∈ PY0 with sample space Y0. The arguments α1 and α2 specify the transmission cost for each of the three channels. The values of T(α1, α2; D1, D2) over the domain of α1 and α2 define the contour of the Gray-Wyner region RGW(D1, D2).

Under assumptions suitable to our proposed method, we can express this objective in terms of an optimization over families of functions:
Theorem 2. Assume that Y0 = f0(X1, X2); f0 ∈ F0, Y1 = f1(X1); f1 ∈ F1, Y2 = f2(X2); f2 ∈ F2 are all deterministic functions of their corresponding inputs, and that g1 ∈ G1 and g2 ∈ G2, where F{0,1,2} and G{1,2} *are families of functions such that there exits a* f0, f1, f2, g1*, and* g2 in their respective families that achieve T(α1, α2; D1, D2)*. Then:*
T(α1, α2; D1, D2) = inf {H(Y0) + α1H(Y1|Y0) + α2H(Y2|Y0)} , (10)
where H(·) *is Shannon's entropy function, and* H(·|·) *is the conditional entropy function (Cover*
& Thomas, 2006). The infimum is over all f0, f1, f2, g1, and g2 in their corresponding family of functions, such that we obtain, at most, distortions D1 and D2.

![5_image_0.png](5_image_0.png)

## Proof. See Appendix B.

We make use of entropy models to estimate - and also steer - the probability distributions of their target representations. With Theorem 2, we can replace the entropy terms with rate functions r{0,1,2}
given by entropy models:

$$r_{\{1,2\}}(Y_{\{1,2\}},Y_{0})=-\mathbb{E}_{Y_{\{1,2\}},Y_{0}}\left[\log\tilde{P}\left(Y_{\{1,2\}}|Y_{0}\right)\right],r_{0}(Y_{0})=-\mathbb{E}\left[\log\tilde{P}\left(Y_{0}\right)\right],\tag{11}$$

where P˜ denotes a probability function assumed for the *quantized* target representations Y{0,1,2},
implicitly established by the entropy models.

The work of Gray & Wyner (1974) highlights that T(α1, α2; D1, D2) is difficult to optimize due to the lack of concavity or convexity in PY0. As such, we propose the Lagrangian relaxation method
(Hiriart-Urruty & Lemarechal, 1993) for the optimization problem in Theorem 2. Relaxing the dis- ´ tortion constraints can help find better solutions in this non-convex space. The Lagrangian relaxation method is used extensively for rate-distortion optimization (Tishby et al., 1999). Due to the convexity of that problem, it is used for convenience, since the method of Lagrange multipliers could also be used. For this problem, however, we have a stronger motivation for its usage.

Assuming that the private channels have the same cost, such that α1 = α2, replacing the entropy terms in Equation 10 with rate functions, and with the distortion constraints d1(Zˆ1, Z1) ≤ D1 and d2(Zˆ2, Z2) ≤ D2, the Lagrangian is given as:

$${\cal L}=\inf\left\{\beta r_{0}(Y_{0})+r_{1}(Y_{1},Y_{0})+r_{2}(Y_{2},Y_{0})+\lambda_{1}d_{1}(\hat{Z}_{1},Z_{1})+\lambda_{2}d_{2}(\hat{Z}_{2},Z_{2})\right\},\tag{12}$$

where β = 1/α{1,2}, and the infimum is over the same families of functions in Equation 10, with the addition of the families of functions defining the three entropy models. The hyper-parameters λ1 and λ2 control the rate-distortion tradeoff (Cover & Thomas, 2006). When β = 1, we optimize for the transmit rate Rt. When β = 2, we optimize for the receive rate Rr. As explained in Sections 2.1 and 3.1, optimizing exclusively for the transmit or the receive rate does not guarantee that the common channel will produce Wyner's or Gacs-K ´ orner common ¨ information. Therefore, values of β outside of the range (1, 2) could result in suboptimal configurations. If we optimize for a combination of both rates, we could attain points on the trace in the contour of the achievable region RGW(D1, D2), connecting the operational points corresponding to C(*X, D*1, D2) and K(X, D1, D2) (Viswanatha et al., 2014). When β = 3/2, we equally optimize for both the transmit and receive rates. If Theorem 1 holds with equality, an optimal codec optimized for β ∈ (1, 2) achieves both common information measures.

## 3.3 A Learnable Gray-Wyner Network

We now formulate a version of a Gray-Wyner Network that is grounded on the proposed objective function. It separates common and private information between two tasks, as it explores the transmitreceive rate tradeoff. We use learnable entropy models as the rate functions r0,1,2 in Eq. 12. These entropy models produce rates that, as part of the objective function, induce the desired type of information in the representations Y0,1,2. Moreover, they allow us to efficiently code (compress) the resulting representations.

Figure 2 provides an overview of the proposed architecture. Inputs X1 and X2 are both processed by two analysis transforms f1 and f2. Because each branch of the proposed architecture has access to both sources X1 and X2, all exclusive information from either source is available to assist in performing tasks Z1 or Z2. This effectively removes the requirement for the conditions in 1.

The output of each analysis transform is passed through a quantization function q that discretizes the representation. It has a differentiable training-time approximation that allows gradient propagation. The representation is then split into two tensors, such that:

$$\left(Y_{\{1,2\}},Y_{0}^{(\{1,2\})}\right)=\left(q\circ f_{\{1,2\}}\right)\left(X_{\{1,2\}}\right).$$

$$(13)$$

=q ◦ f{1,2} X{1,2}. (13)
Then, $Y_0^{(\{1,2\})}$ are com. 
are combined into $ Y_0$ so that: . 
$$[Y_{0}]_{i}=\begin{cases}1/2\left(\left[Y_{0}^{(1)}\right]_{i}+\left[Y_{0}^{(2)}\right]_{i}\right),&\text{if}\left[Y_{0}^{(1)}\right]_{i}=\left[Y_{0}^{(2)}\right]_{i}\\ 0,&\text{otherwise,}\end{cases}$$
$$(14)$$

where i ∈ Z+ indexes the elements in the tensors. Using auto-differentiation, the 1/2 (Y
(1)
0 + Y
(2)
0)
expression ensures that gradients flow to both inputs wherever elements match. An auxiliary loss term encourages the two input tensors to match. The augmented loss function is given as:

$${\mathcal{L}}_{\mathrm{aug}}={\mathcal{L}}+\mathbb{E}\left[{\frac{\gamma}{|Y_{0}|}}\left\|Y_{0}^{(1)}-Y_{0}^{(2)}\right\|_{2}^{2}\right],$$
$$(15)$$

where γ influences the impact of this additional loss. Small values of γ might result in elements of Y
(1)
0and Y
(2)
0never matching. A large γ can result in degenerate distributions for Y
(1)
0and Y
(2)
0.

In both cases, the common channel is underutilized. Thus, this auxiliary loss can discourage the use of the common channel. We overcome this obstacle in practice by setting γ = 1 and reducing the cost of usage of the common channel β when necessary, offering it as the only hyper-parameter.

Entropy models for each private channel, h1 and h2, propose parameters of a probability distribution assumed for its target representations, conditioned on the common representation. This conditioning is prescribed by the proposed objective function. Hence, these entropy models use as context the previously coded elements in Y{1,2}, in addition to Y0, to predict the distribution parameters of the current elements in Y{1,2}, respectively. An entropy model for the common channel, h0, only uses as context the previously coded elements in Y0. As we have seen, it is often difficult to discard the information in the common channel from the private channels, such that I(Y1, Y2; Y0) = 0. Hence, using the common representation to model the entropy of the private representations can handle the redundancies between the private and common channels and improve compression. Each private representation is concatenated with the common representation and processed by a synthesis transform for its corresponding task. Each synthesis transform contains a task-specific model to produce reasonable outputs within sample spaces of the targets Z{1,2}, respectively. Using only the private channel as input for the synthesis transform is another plausible architecture but it forces the common information to be present in the private channel. The conditional entropy models must then be able to predict this common information very accurately to avoid rate increases due to redundancies. This is often difficult for learnable codecs. Other architectures for the analysis transform f are evaluated in Section 4.1. Intuitively, this architecture provides flexible representations while reducing the learning complexity of making them compatible. A theoretical justification is presented in Appendix C, in which we introduce a measure of compatibility between representations based on the generalization error induced by the hypotheses (family of functions) used to generate them.

## 4 Experimental Evaluation

In our experiments, the proposed architecture specializes to a single source X, so that (X1, X2) =
X. We use an architecture inspired by He et al. (2022) as analysis and synthesis transforms. It

![7_image_0.png](7_image_0.png) 

consists of 3 stacks of 3 ResNet blocks each, with the stacks interjecting 4 convolutional layers acting as dimensional bottlenecks. These convolutional layers scale their inputs. As such, the analysis transform progressively reduces the spatial dimensions by a total factor of 16, while increasing the number of channels to 24, 48, 192 and E. The synthesis transform performs the opposite in tandem, decreasing the number of channels and increasing the spatial size to that of the original input. As a quantization function q, we use the straight-through gradient function proposed in Theis et al. (2017). We use the relatively simple model of Balle et al. (2018). The common representation ´ is processed and used in place of the hyper-prior to establish the conditional entropy models. As such, for each private channel, the common representation is processed by 2 stacks of 2 ResNet blocks each, interjected by a convolutional layer that increases the number of channels from E/2 to E. Masked convolutions turn the entropy model auto-regressive, which is suitable for coding (Balle´
et al., 2018). See Appendix D for architecture diagrams, hyper-parameters, training settings, other results, and further discussions. Code is available at: github.com/adeandrade/research

## 4.1 Transmit-Receive Tradeoff And Ablation Study

We developed a synthetic dataset to study the proposed method. Let X1 and X2 index the first dimension of a random variable X. We created an X such that H(X1, X2) = 3.3 bits per element, H(X1) + H(X2) = 4.62, and consequently, I(X1; X2) = 1.32. X1 and X2 are individually transformed to generate targets Z1 and Z2 of two linear regression tasks. We train to minimize the RMSE loss between the predictions and the targets of each task.

In addition to the proposed *Shared* architecture, we evaluate the rate-distortion performance of two additional encoder architectures. The *Separated* architecture has an independent analysis transform for each channel. The *Combined* architecture uses a single analysis transform and splits the output tensor into 3 parts corresponding to the GWN channels. If both tasks share a single channel, a resulting *Joint* architecture optimizes the transmit rate. Using a private channel for each task without a common channel, results in an *Independent* method which optimizes the receive rate. The rates produced by these methods are used to compute empirical estimates of the joint and marginal rate-

![8_image_0.png](8_image_0.png)

distortion functions, and mutual information between tasks. We contrast them against theoretical values. See Appendix D for details on the dataset, architectures, and measurement calculation. Figure 3a shows that optimizing the transmit rate produces rates for the common channel that are higher than the empirical mutual information between sources. The same figure also shows that optimizing the receive rate produces rates for the common channel that are lower than the empirical mutual information. Codecs trained with β = 3/2 explore the transmit-receive tradeoff. They have a lower rate on the common channel than the codecs optimized for the transmit rate, but more information than those optimized for the receive rate.

For all β explored, the Shared architecture outperforms the Separated and Combined alternatives, as shown in Figure 3b for β = 1. See Appendix D for results on the other β. Note that the empirical estimates of the rate are considerably higher than the theoretical values, as often seen in practice (Bajic, 2025). Figures 3c and 3d show that ´ β = 3/2 is a reasonable value for this problem, since it performs marginally better than β = 1 and β = 2, in both transmit and receive rates, respectively.

## 4.2 Edge-Case Exploration With Image Classification

Classification problems have well-defined theoretical measurements of mutual information against which we can compare. We randomly colorize digit images from MNIST (LeCun et al., 2010) according to three different PMFs: 1. A *Dependent* PMF, where one color always corresponds to one digit; 2. An *Independent* PMF, in which for each sample, one out of 10 colors is sampled uniformly; and 3. A *Mixture* PMF, where each digit has a subset of 10 colors assigned to it, with uniform probability. One of the tasks in our proposed method predicts the digit, while the other task predicts the color, using the colorized images and corresponding targets. The Dependent PMF has a joint entropy of log2 10 bits and a mutual information of the same amount. The Independent PMF has a joint entropy of 2 log2 10 bits, and a mutual information of 0 bits. The Mixture PMF has a joint entropy of 5.12 bits, and a mutual information of 1.4 bits. Figures 4a and 4b show the transmit an receive rates, respectively, for the 3 PMFs. We operate within an order of magnitude of the theoretical bounds, which is comparable to other codecs (Bajic,´ 2025). More importantly, the method trained on the Dependent PMF produces a lower transmit rate since it places most of its information on the common channel, taking advantage of all information being common. On the other hand, the Independent method produces the lowest receive rate, since it has a very low rate on the common channel, as with the underlying PMF. The common information in the Mixture PMF is not very separable, which results in our method producing lower rate-distortion performance compared to the other PMFs. It still performs better, in terms of transmit rate, than the Independent approach from Section 4.1, where there is no common channel. Appendix D shows the rate of each channel, produced by our method, for every PMF.

![9_image_0.png](9_image_0.png)

Figure 5: Rate-accuracy curves of the proposed method against the Joint and Independent baselines. The tasks in (a) are reported for the validation set of Cityscapes. The tasks in (b) are reported for the validation set of COCO 2017. The transmit and receive rates of the proposed method are included. BD-rates are computed with respect to the Joint method. The task performances are added. The depth RMSE is scaled so its inverse is in a similar scale as the segmentation mean intersection over union (mIoU). The detection performances are measured by the mean average precision (mAP). The Uncompressed lines correspond to the original performances of the pre-trained task models.

## 4.3 Rate-Distortion Performance In Computer Vision Tasks

We evaluate the proposed method on semantic segmentation and depth estimation for Cityscapes (Cordts et al., 2016), and on object detection and keypoint detection for COCO 2017 (Lin et al.,
2014). As part of g1 and g2, we append pre-trained task-specific models to the synthesis transforms.

We keep their weights fixed as we train the rest of the codec. We use DeepLabV3+ with MobileNet for semantic segmentation (Chen et al., 2018a), LRASPP with MobileNetV3 for depth estimation (Howard et al., 2019), Faster R-CNN with ResNet50 for object detection (Ren et al., 2017), and Keypoint R-CNN with ResNet50 for keypoint detection (He et al., 2020).

Figure 5 compares the performance of the proposed method. It is able to outperform an Independent approach and it is relatively close to the Joint approach. The curves for the receive rate are higher than the Independent approach, suggesting that the rate of the common channel is lower than the empirical mutual information between these tasks. Some curves in the Cityscapes experiments have an increase in distortion with the lowest compression, which is often informally attributed to the lack of regularization, provided by stronger rate constraints.

## 5 Summary And Conclusion

We validated the ability of the proposed learnable Gray-Wyner Network to distill common information between tasks and compared it against other architectures. The performance is theoretically justified by analyzing the compatibility between intermediate representations. We provided bounds that relate two measures of lossy common information. The proposed optimization objective is derived from this theory and was able to empirically explore the transmit-receive rate tradeoff. The proposed method is also able to handle edge-cases, including a case where there is no mutual information between tasks, and another where the tasks are fully dependent. Finally, between the three computer vision experiments, our codecs achieved, on average, a BD-rate advantage of -81.58% in transmit rate, against single-task codecs. Isolating and coding the common information between dependent tasks allows for the efficient distributed inference of machine tasks. Generating representations that explore the tradeoff between the transmit and receive rates in the Gray-Wyner Network has additional practical implications in storage and selective retrieval, and dispersive information routing (Viswanatha et al., 2011). Knowing the information requirements of learned representations can assist in planning for the resources allocated to a neural network, its dimensionality, and quantization levels. Extensions to three or more tasks are possible, but since the total number of channels scales exponentially, a more dynamic architecture might be required. Nevertheless, the theoretical contributions of this work should prove useful in deriving new methods.

## Acknowledgments

This work was partially funded by Intel Labs and the Natural Sciences and Engineering Research Council of Canada (NSERC).

## References

Ivan V. Bajic. Rate-accuracy bounds in visual coding for machines. In ´ *IEEE MIPR*, 2025. Johannes Balle, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. Variational ´
image compression with a scale hyperprior. In *ICLR*, 2018.

Peter L. Bartlett and Shahar Mendelson. Rademacher and gaussian complexities: Risk bounds and structural results. *JMLR*, 2002.

Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, and Jorn-Henrik Jacobsen. ¨
Invertible residual networks. In ICML, 2019.

Gisle Bjontegaard. Calculation of average PSNR differences between RD-curves. ITU-T SC16/Q6 VCEG-M33, 2001.

Lahiru D. Chamain, Fabien Racape, Jean B ´ egaint, Akshay Pushparaja, and Simon Feltman. End-to- ´
end optimized image compression for multiple machine tasks. *CoRR*, 2103.04178, 2021.

Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoderdecoder with atrous separable convolution for semantic image segmentation. In *ECCV*, 2018a.

Tian Qi Chen, Xuechen Li, Roger B. Grosse, and David Duvenaud. Isolating sources of disentanglement in variational autoencoders. In *NeurIPS*, 2018b.

Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. Infogan:
Interpretable representation learning by information maximizing generative adversarial nets. In NeurIPS, 2016.

Hyomin Choi and Ivan V. Bajic. Scalable image coding for humans and machines. *IEEE TIP*, 2022. Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The Cityscapes dataset for semantic urban scene understanding. In *IEEE CVPR*, 2016.

Thomas M. Cover and Joy A. Thomas. *Elements of information theory*. Wiley, second edition, 2006. Anderson de Andrade and Ivan V. Bajic. Towards task-compatible compressible representations. In IEEE ICME Workshops, 2024.

Yann Dubois, Benjamin Bloem-Reddy, Karen Ullrich, and Chris J. Maddison. Lossy compression for lossless prediction. In *NeurIPS*, 2021.

Ruoyu Feng, Xin Jin, Zongyu Guo, Runsen Feng, Yixin Gao, Tianyu He, Zhizheng Zhang, Simeng Sun, and Zhibo Chen. Image coding for machines with omnipotent feature learning. In *ECCV*, 2022.

Yalda Foroutan, Alon Harell, Anderson de Andrade, and Ivan V. Bajic. Base layer efficiency in scalable human-machine coding. In *IEEE ICIP*, 2023.

Peter Gacs and J. Korner. Common information is far less than mutual information. ¨ Problems of Control and Information Theory, 1973.

Robert M. Gray. A new class of lower bounds to information rates of stationary sources via conditional rate-distortion functions. *IEEE TIT*, 1973.

Robert M. Gray and Aaron D. Wyner. Source coding for a simple network. *IEEE BSTJ*, 1974. Sha Guo, Lin Sui, Chen-Lin Zhang, Zhuo Chen, Wenhan Yang, and Lingyu Duan. A unified image compression method for human perception and multiple vision tasks. In *ECCV*, 2024.

Dailan He, Ziming Yang, Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang. ELIC: Efficient learned image compression with unevenly grouped space-channel contextual adaptive coding. In CVPR, 2022.

Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross B. Girshick. Mask R-CNN. ´ *IEEE TPAMI*,
2020.

Irina Higgins, Lo¨ıc Matthey, Arka Pal, Christopher P. Burgess, Xavier Glorot, Matthew M.

Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. In *ICLR*, 2017.

Jean-Baptiste Hiriart-Urruty and Claude Lemarechal. ´ Convex analysis and minimization algorithms II: Advanced theory and bundle methods. Springer Berlin, Heidelberg, 1993.

Andrew Howard, Ruoming Pang, Hartwig Adam, Quoc V. Le, Mark Sandler, Bo Chen, Weijun Wang, Liang-Chieh Chen, Mingxing Tan, Grace Chu, Vijay Vasudevan, and Yukun Zhu. Searching for MobileNetV3. In *IEEE/CVF ICCV*, 2019.

Sham Kakade and Ambuj Tewari. Lecture notes in Rademacher composition and linear prediction, 2008.

Yann LeCun, Corinna Cortes, and CJ Burges. MNIST handwritten digit database. ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist, 2010.

Michel Ledoux and Michel Talagrand. *Probability in banach spaces: isoperimetry and processes*.

Springer Berlin Heidelberg, 2013.

Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C. Lawrence Zitnick. Microsoft COCO: Common objects in context. In ´ *ECCV*, 2014.

Ben Poole, Sherjil Ozair, Aaron van den Oord, Alexander A. Alemi, and George Tucker. On varia- ¨
tional bounds of mutual information. In *ICML*, 2019.

Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. Faster R-CNN: towards real-time object detection with region proposal networks. *IEEE TPAMI*, 2017.

Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning - from theory to algorithms. Cambridge University Press, 2014.

Wenzhe Shi, Jose Caballero, Ferenc Huszar, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In *IEEE CVPR*, 2016.

Lucas Theis, Wenzhe Shi, Andrew Cunningham, and Ferenc Huszar. Lossy image compression with ´
compressive autoencoders. In *ICLR*, 2017.

Hu Kuo Ting. On the amount of information. *Theory of Probability and Its Applications*, 1962. Naftali Tishby and Noga Zaslavsky. Deep learning and the information bottleneck principle. In IEEE ITW, 2015.

Naftali Tishby, Fernando C. N. Pereira, and William Bialek. The information bottleneck method. In Allerton Conference, 1999.

TorchVision maintainers and contributors. TorchVision: PyTorch's computer vision library. *GitHub* repository, 2016.

Lan V. Truong. On Rademacher complexity-based generalization bounds for deep learning. *CoRR*,
2208.04284, 2022.

Kumar Viswanatha, Emrah Akyol, and Kenneth Rose. An optimal transmit-receive rate tradeoff in gray-wyner network and its relation to common information. In *IEEE ITW*, 2011.

Kumar Viswanatha, Emrah Akyol, and Kenneth Rose. The lossy common information of correlated sources. *IEEE TIT*, 2014.