

{0}------------------------------------------------

# FEEDBACK FAVORS THE GENERALIZATION OF NEURAL ODEs

Jindou Jia<sup>1,2\*</sup>, Zihan Yang<sup>1\*</sup>, Meng Wang<sup>1</sup>, Kexin Guo<sup>1,2†</sup>

Jianfei Yang<sup>3</sup>, Xiang Yu<sup>1,2†</sup>, Lei Guo<sup>1,2</sup>

<sup>1</sup>Beihang University <sup>2</sup>Hangzhou Innovation Institute of Beihang University

<sup>3</sup>Nanyang Technological University

## ABSTRACT

The well-known generalization problem hinders the application of artificial neural networks in continuous-time prediction tasks with varying latent dynamics. In sharp contrast, biological systems can neatly adapt to evolving environments benefiting from real-time feedback mechanisms. Inspired by the feedback philosophy, we present feedback neural networks, showing that a feedback loop can flexibly correct the learned latent dynamics of neural ordinary differential equations (neural ODEs), leading to a prominent generalization improvement. The feedback neural network is a novel two-DOF neural network, which possesses robust performance in unseen scenarios with no loss of accuracy performance on previous tasks. A linear feedback form is presented to correct the learned latent dynamics firstly, with a convergence guarantee. Then, domain randomization is utilized to learn a nonlinear neural feedback form. Finally, extensive tests including trajectory prediction of a real irregular object and model predictive control of a quadrotor with various uncertainties, are implemented, indicating significant improvements over state-of-the-art model-based and learning-based methods. <sup>‡</sup>

## 1 INTRODUCTION

Stemming from residual neural networks (He et al., 2016), neural ordinary differential equation (neural ODE) (Chen et al., 2018) emerges as a novel learning strategy aiming at learning the latent dynamic model of an unknown system. Recently, neural ODEs have been successfully applied to various scenarios, especially continuous-time missions (Liu & Stacey, 2024; Verma et al., 2024; Greydanus et al., 2019; Cranmer et al., 2020). However, like traditional neural networks, the generalization problem limits the application of neural ODEs in real-world applications.

Traditional strategies like model simplification, fit coarsening, data augmentation, and transfer learning have considerably improved the generalization performance of neural networks on unseen tasks (Rohlfs, 2022). However, these strategies usually reduce the accuracy performance on previous tasks, and large-scale training data and network structures are often required to approximate previous accuracy. The objective of this work is to develop a novel network architecture, acquiring the generalization improvement while preserving the accuracy performance.

![Figure 1: Neural network architectures. Left: Neural ODE. Right: Feedback neural network.](68dad113f9a15ab01945110cb50cdcfb_img.jpg)

The diagram illustrates two neural network architectures. The left side, labeled 'Neural ODE', shows a flow from input  $x(t)$  through a neural network block (represented by a trapezoid with circles) to produce a prediction  $f_{neural}$ , which is then used to predict  $x(t + \Delta t)$ . The right side, labeled 'Feedback neural network', shows a more complex architecture. It includes the same neural network block for  $f_{neural}$ , but also incorporates a feedback loop. The feedback loop consists of a delay block (represented by a circle with a 'z' and a minus sign), a gain block (represented by a circle with 'α'), and a summation block (represented by a circle with a plus sign). The output  $x(t)$  is fed back through the delay and gain blocks, and the result is added to the prediction  $f_{neural}$  to produce the final prediction  $\hat{x}(t + \Delta t)$ . The input  $x(t)$  is also shown as an input to the feedback loop.

Figure 1: Neural network architectures. Left: Neural ODE. Right: Feedback neural network.

Figure 1: Neural network architectures. *Left*: Neural ODE developed in Chen et al. (2018). *Right*: Proposed feedback neural network.

<sup>\*</sup>Equal contribution.

<sup>†</sup>Corresponding authors ({kxguo, xiangyu\_buaa}@buaa.edu.cn).

<sup>‡</sup>Codes are available at <https://sites.google.com/view/feedbacknn>.

{1}------------------------------------------------

Living beings can neatly adapt to unseen environments, even with limited neurons and computing power. One reason can be attributed to the existence of internal feedback (Aoki et al., 2019). Internal feedback has been shown to exist in biological control, perception, and communication systems, handling external disturbances, internal uncertainties, and noises (Sarma et al., 2022; Markov et al., 2021). In neural circuits, feedback inhibition is able to regulate the duration and magnitude of excitatory signals (Luo, 2021). In engineering systems, internal feedback indicates impressive effects across filtering and control tasks, such as *Kalman* filter (Kalman, 1960), *Luenberger* observer (Luenberger, 1966), extended state observer (Guo et al., 2020), and proportional-integral-derivative control (Ang et al., 2005). The effectiveness of feedback lies in its ability to harness real-time deviations between internal predictions/estimations and external measurements to infer dynamical uncertainties. The cognitive corrections are then performed timely. However, existing neural networks rarely incorporate such a real-time feedback mechanism.

In this work, we attempt to enhance the generalization of neural ODEs by incorporating the feedback scheme. The key idea is to correct the learned latent dynamical model of a Neural ODE according to the deviation between measured and predicted states, as illustrated in Figure 1. We introduce two types of feedback: linear form and nonlinear neural form. Unlike previous training methods that compromise accuracy for generalization, the developed feedback neural network is a two-DOF framework that exhibits generalization performance on unseen tasks while maintaining accuracy on previous tasks. The effectiveness of the presented feedback neural network is demonstrated through several intuitional and practical examples, including trajectory prediction of a spiral curve, trajectory prediction of an irregular object and model predictive control (MPC) of a quadrotor.

## 2 NEURAL ODES AND LEARNING RESIDUES

A significant application of artificial neural networks centers around the prediction task.,  $\mathbf{x}(t) \rightarrow \mathbf{x}(t + \Delta t)$ . Note that  $t$  indicates the input  $\mathbf{x}$  evolves with time. Chen et al. (2018) utilize neural networks to directly learn latent ODEs of target systems, named Neural ODEs. Neural ODEs greatly improve the modeling ability of neural networks, especially for continuous-time dynamic systems (Massaroli et al., 2020), while maintaining a constant memory cost. The ODE describes the instantaneous change of a state  $\mathbf{x}(t) \in \mathbb{R}^n$

$$\frac{d\mathbf{x}(t)}{dt} = \mathbf{f}(\mathbf{x}(t), \mathbf{I}(t), t) \quad (1)$$

where  $\mathbf{f}(\cdot) : \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R} \rightarrow \mathbb{R}^n$  represents a latent nonlinear mapping, and  $\mathbf{I}(t) \in \mathbb{R}^m$  denotes external input. Note that compared with Chen et al. (2018), we further consider  $\mathbf{I}(t)$  that can extend the ODE to controlled dynamics. The *adjoint sensitive method* is employed in Chen et al. (2018) to train neural ODEs without considering  $\mathbf{I}(t)$ . In Appendix A.1, we provide an alternative training strategy in the presence of  $\mathbf{I}(t)$ , from the view of optimal control.s

Given the ODE (1) and an initial state  $\mathbf{x}(t)$ , future state can be predicted as an initial value problem

$$\mathbf{x}(t + \Delta t) = \mathbf{x}(t) + \int_t^{t+\Delta t} \mathbf{f}(\mathbf{x}(\tau), \mathbf{I}(\tau), \tau) d\tau. \quad (2)$$

The workflow of neural ODEs is depicted in Figure 1. However, like traditional learning methods, generalization is a major bottleneck for neural ODEs (Marion, 2024). Learning residuals will appear if the network has not been trained properly (e.g., underfitting and overfitting) or the applied scenario has a slightly different latent dynamic model. Take a spiral function as an example (Appendix A.3.1). When a network trained from a given training set (Figure 5 (a)) is transferred to a new case (Figure 5 (b)), the learning performance will dramatically degrade (Figure 5 (d)). Without loss of generality, the learning residual error is formalized as

$$\mathbf{f}(\mathbf{x}(t), \mathbf{I}(t), t) = \mathbf{f}_{neural}(\mathbf{x}(t), \mathbf{I}(t), t, \boldsymbol{\theta}) + \Delta \mathbf{f}(t) \quad (3)$$

where  $\mathbf{f}_{neural}(\cdot) : \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R} \rightarrow \mathbb{R}^n$  represents the learned ODE model parameterized by  $\boldsymbol{\theta}$ , and  $\Delta \mathbf{f}(t) \in \mathbb{R}^n$  denotes the unknown learning residual error. In the presence of  $\Delta \mathbf{f}(t)$ , the prediction error of (2) will accumulate over time. The objective of this work is to improve neural ODEs with as few modifications as possible to suppress the effects of  $\Delta \mathbf{f}(t)$ .

{2}------------------------------------------------

## 3 NEURAL ODES WITH A LINEAR FEEDBACK

### 3.1 CORRECTING LATENT DYNAMICS THROUGH FEEDBACK

Even though learned experiences are encoded by neurons in the brain, living organisms can still adeptly handle unexpected internal and external disturbances with the assistance of feedback mechanisms (Aoki et al., 2019; Sarma et al., 2022). The feedback scheme has also proven effective in traditional control systems, facilitating high-performance estimation and control objectives. Examples include *Kalman* filter (Kalman, 1960), *Luenberger* observer (Luenberger, 1966), extended state observer (Guo et al., 2020), and proportional-integral-derivative control (Ang et al., 2005).

![Figure 2: A graph showing the learned latent dynamics (f_neural(t)) and the truth dynamics (f(t)) over time. The truth dynamics are a smooth red curve. The learned dynamics are a blue curve that follows the truth dynamics but has a downward deviation at time t_1. The deviation is labeled Delta f(t). The learned dynamics are modified through accumulative evaluation errors to approach the truth one. The graph shows evaluation points t_0, t_1, t_2, t_3, ... and the corresponding evaluation errors L(x(t_i) - x_bar(t_i)).](a854aa286b14d26d27047ee5893ffaa7_img.jpg)

Figure 2: A graph showing the learned latent dynamics (f\_neural(t)) and the truth dynamics (f(t)) over time. The truth dynamics are a smooth red curve. The learned dynamics are a blue curve that follows the truth dynamics but has a downward deviation at time t\_1. The deviation is labeled Delta f(t). The learned dynamics are modified through accumulative evaluation errors to approach the truth one. The graph shows evaluation points t\_0, t\_1, t\_2, t\_3, ... and the corresponding evaluation errors L(x(t\_i) - x\_bar(t\_i)).

Figure 2: The learned latent dynamics are modified through accumulative evaluation errors to approach the truth one.

We attempt to introduce the feedback scheme into neural ODEs, named feedback neural networks, as shown in Figure 1. Neural ODEs have exploited latent dynamical models  $f_{neural}(t)$  of target systems in training set. The key idea of feedback neural networks is to further correct  $f_{neural}(t)$  according to state feedback. Denote  $t_i$  as the historical evaluation moment satisfying  $t_i \leq t$ . At current moment  $t$ , we collect  $k + 1$  state measurements  $\{\mathbf{x}(t_0), \mathbf{x}(t_1), \dots, \mathbf{x}(t_k)\}$ , in which  $t_k = t$ . As portrayed in Figure 2,  $f_{neural}(t)$  is modified by historical evaluation errors to approach its truth dynamics  $f(t)$ , i.e.,

$$\hat{f}_{neural}(t) = f_{neural}(t) + \sum_{i=0}^k \mathbf{L}(\mathbf{x}(t_i) - \bar{\mathbf{x}}(t_i)) \quad (4)$$

where  $\mathbf{L} \in \mathbb{R}^{n \times n}$  represents the positive definite matrix and  $\bar{\mathbf{x}}(t_i) \in \mathbb{R}^n$  represents the predicted state from the last evaluation moment, e.g., an *Euler* integration

$$\bar{\mathbf{x}}(t_i) = \mathbf{x}(t_{i-1}) + T_s \hat{f}_{neural}(t_{i-1}) \quad (5)$$

with the prediction step  $T_s \in \mathbb{R}$ .

To avoid storing more and more historical measurements over time, define an auxiliary variable

$$\hat{\mathbf{x}}(t) = \bar{\mathbf{x}}(t) - \sum_{i=0}^{k-1} (\mathbf{x}(t_i) - \bar{\mathbf{x}}(t_i)) \quad (6)$$

where  $\hat{\mathbf{x}}(t) \in \mathbb{R}^n$  can be regarded as an estimation of  $\mathbf{x}(t)$ . Combining (4) and (6), can lead to

$$\hat{f}_{neural}(t) = f_{neural}(t) + \mathbf{L}(\mathbf{x}(t) - \hat{\mathbf{x}}(t)). \quad (7)$$

From (5) and (6), it can be further rendered that

$$\bar{\mathbf{x}}(t_k) = \hat{\mathbf{x}}(t_{k-1}) + T_s \hat{f}_{neural}(t_{k-1}). \quad (8)$$

By continuing the above *Euler* integration, it can be seen that  $\hat{\mathbf{x}}(t)$  is the continuous state of the modified dynamics, i.e.,  $\dot{\hat{\mathbf{x}}}(t) = \hat{f}_{neural}(t)$ . Finally,  $\hat{f}_{neural}(t)$  can be persistently obtained through (7) and (8) recursively, instead of (4) and (5) accumulatively.

### 3.2 CONVERGENCE ANALYSIS

In this part, the convergence property of the feedback neural network is analyzed. The state observation error of the feedback neural network is defined as  $\tilde{\mathbf{x}}(t) = \mathbf{x}(t) - \hat{\mathbf{x}}(t)$ , and its derivative  $\dot{\tilde{\mathbf{x}}}(t)$ , i.e., the approximated error of latent dynamics is defined as  $\tilde{f}(t) = f(t) - \hat{f}_{neural}(t)$ . Substitute (1) and (3) into (7), one can obtain the error dynamics

$$\dot{\tilde{\mathbf{x}}}(t) = -\mathbf{L}\tilde{\mathbf{x}}(t) + \Delta f(t). \quad (9)$$

Before proceeding, a reasonable bounded assumption on the learning residual error  $\Delta f(t)$  is made.

{3}------------------------------------------------

**Assumption 1.** *There exists an unknown upper bound such that*

$$\|\Delta \mathbf{f}(t)\| \leq \gamma \quad (10)$$

where  $\|\cdot\|$  denotes the Euclidean norm and  $\gamma \in \mathbb{R}$  is an unknown positive value.

Note that the above assumption can cover common step disturbances (Figure S12).

**Theorem 1.** *Consider the nonlinear system (1). Under the linear state feedback (7) and the bounded Assumption 1, the state observation error  $\tilde{\mathbf{x}}(t)$  and its derivative  $\dot{\tilde{\mathbf{x}}}(t)$  (i.e.,  $\dot{\mathbf{f}}(t)$ ) can exponentially converge to bounded sets  $\mathcal{B}_1 = \{\tilde{\mathbf{x}}(t) \in \mathbb{R}^n : \|\tilde{\mathbf{x}}(t)\| \leq \gamma/\lambda_m(\mathbf{L})\}$  and  $\mathcal{B}_2 = \{\dot{\tilde{\mathbf{x}}}(t) \in \mathbb{R}^n : \|\dot{\tilde{\mathbf{x}}}(t)\| \leq \gamma\lambda_M(\mathbf{L})/\lambda_m(\mathbf{L}) + \gamma\}$ , respectively, which can be regulated by  $\mathbf{L}$ .*

*Proof.* See Appendix A.2. □

### 3.3 MULTI-STEP PREDICTION

With the modified dynamics  $\hat{\mathbf{f}}(t)$  and current  $\mathbf{x}(t)$ , the next step is to predict  $\mathbf{x}(t+\Delta t)$  as in (2). By defining  $\mathbf{z}(t) = [\mathbf{x}^T(t), \hat{\mathbf{x}}^T(t)]^T \in \mathbb{R}^{2n}$ , from (8), we have  $\dot{\mathbf{z}}(t) = [\mathbf{f}^T(t), \hat{\mathbf{f}}^T(t)]^T$ . One intuitive means to obtain  $\mathbf{z}(t+\Delta t)$  is to solve the ODE problem with modern solvers. However, as shown in Theorem 1, the convergence of  $\hat{\mathbf{f}}(t)$  can only be guaranteed as current  $t$ . In other words, the one-step prediction result by solving the above ODE is accurate, while the error will accumulate in the long-term prediction. In this part, an alternative multi-step prediction strategy is developed to circumvent this problem.

The proposed multi-step prediction strategy is portrayed in Figure 3, which can be regarded as a cascaded form of one-step prediction. The output of each feedback neural network is regarded as the input of the next layer. Take the first two layers as an example. The first-step prediction  $\mathbf{x}(t+T_s)$  is obtained by  $\mathbf{x}(t+T_s) = \mathbf{x}(t) + \hat{\mathbf{f}}(\mathbf{x}(t), \hat{\mathbf{x}}(t), \theta)T_s$ . The second layer with the input of  $\mathbf{x}(t+T_s)$  will output  $\mathbf{x}(t+2T_s)$ . In such a framework, the convergence of later layers will not affect the convergence of previous layers. Thus, the prediction error will converge from top to bottom in order.

Note that the cascaded prediction strategy can amplify the data noise in case of large  $\mathbf{L}$ . A gain decay strategy is designed to alleviate this issue. Denote the feedback gain of  $i$ -th layer as  $\mathbf{L}_i$ , which decays as  $i$  increases

$$\mathbf{L}_i = \mathbf{L} \odot e^{-\beta i} \quad (11)$$

where  $\beta$  represents the decay rate. The efficiency of the decay strategy is presented in Figure 5(g). The involvement of the decay factor in the multi-step prediction process significantly enhances the robustness to data noise.

### 3.4 ABLATION STUDY ON OBSERVER GAIN

The adjustment of linear feedback gain  $\mathbf{L}$  can be separated from the training of neural ODEs, which can increase the flexibility of the structure.

The gain adjustment strategy is intuitive. Theorem 1 indicates that the prediction error will converge to a bounded set as the minimum eigenvalue of feedback gain is positive. And the converged set can shrink with the increase of the minimum eigenvalue. In reality, the amplitude of  $\lambda_m(\mathbf{L})$  is limited since the feedback  $\mathbf{x}$  is usually noised. The manual adjustment of  $\lambda_m(\mathbf{L})$  needs the trade-off between prediction accuracy and noise amplification. Thus, an ablation study on  $(\mathbf{L})$  to show practical implications of Theorem 1 under Assumption 9 is implemented.

Figure 4 shows the multi-step prediction errors ( $N = 50$ ) with different levels of feedback gains and uncertainties. Two phenomena can be observed from the heatmap. The one is that the prediction error increases with the level of uncertainty. The other is that the prediction error decreases with the gain at the beginning, but due to noise amplification, the prediction error worsens if the gain is set too large.

![Diagram of the multi-step prediction strategy. It shows a vertical stack of 'Feedback NN' blocks. The input is x(t). The output of the first block is x(t+T_s). The output of the second block is x(t+2T_s). The output of the third block is x(t+(k+1)T_s). The output of the final block is x(t+N T_s).](8e85c350ee10ead46f7834b148b8b7d3_img.jpg)

```

graph TD
    x_t["x(t)"] --> FNN1["Feedback NN"]
    FNN1 --> x_t_Ts["x(t+T_s)"]
    x_t_Ts --> FNN2["Feedback NN"]
    FNN2 --> x_t_2Ts["x(t+2T_s)"]
    x_t_2Ts --> FNN3["Feedback NN"]
    FNN3 --> x_t_kTs["x(t+(k+1)T_s)"]
    x_t_kTs --> FNN4["Feedback NN"]
    FNN4 --> x_t_NTs["x(t+N T_s)"]
  
```

Diagram of the multi-step prediction strategy. It shows a vertical stack of 'Feedback NN' blocks. The input is x(t). The output of the first block is x(t+T\_s). The output of the second block is x(t+2T\_s). The output of the third block is x(t+(k+1)T\_s). The output of the final block is x(t+N T\_s).

Figure 3: The multi-step prediction.

{4}------------------------------------------------

## 4 NEURAL ODES WITH A NEURAL FEEDBACK

Section 3 has shown a linear feedback form can promptly improve the adaptability of neural ODEs in unseen scenarios. However, two improvements could be further made. At first, it will be more practical if the gain tuning procedure could be avoided. Moreover, the linear feedback form can be extended to a nonlinear one  $h(\mathbf{x}(t) - \hat{\mathbf{x}}(t)) : \mathbb{R}^n \rightarrow \mathbb{R}^n$  to adopt more intricate scenes, as experienced in the control field (Han, 2009).

An effectual solution is to model the feedback part using another neural network, i.e.,  $h_{neural}(\mathbf{x}(t) - \hat{\mathbf{x}}(t), \boldsymbol{\xi})$  parameterized by  $\boldsymbol{\xi}$ . Here we design a separate learning strategy to learn  $\boldsymbol{\xi}$ . At first, the neural ODE is trained on the nominal task without considering the feedback part. Then the feedback part is trained through domain randomization by freezing the neural ODE. In this way, the obtained feedback neural network is skillfully considered as a two-DOF network. On the one hand, the original neural ODE preserves the accuracy on the previous nominal task. On the other hand, with the aid of feedback, the generalization performance is available in the presence of unknown uncertainties.

### 4.1 DOMAIN RANDOMIZATION

The key idea of domain randomization (Tobin et al., 2017; Peng et al., 2018) is to randomize the system parameters, noises, and perturbations as collecting training data so that the real applied case can be covered as much as possible. Taking the spiral example as an example (Figure 5 (a)), training with domain randomization requires datasets collected under various periods, decay rates, and bias parameters, so that the learned networks are robust to the real case with a certain of uncertainty.

Two shortcomings exist when employing domain randomization. On the one hand, the existing trained network needs to be retrained and the computation burden of training is dramatically increased. On the other hand, the training objective is forced to focus on the average performance among different parameters, such that the prediction ability on the previous nominal task will degraded, as shown in Figure 6 (a). To maintain the previous accuracy performance, larger-scale network designs are often required. In other words, the domain randomization trades precision for robustness. In the proposed learning strategy, the generalization ability is endowed to the feedback loop independently, so that the above shortcomings can be circumvented.

### 4.2 LEARNING A NEURAL FEEDBACK

In this work, we specialize the virtue from domain randomization to the feedback part  $h_{neural}(t)$  rather than the previous neural network  $f_{neural}(t)$ . The training framework is formalized as follows

$$\boldsymbol{\xi}^* = \arg \min_{\boldsymbol{\xi}} \sum_{i=1}^{n_{case}} \sum_{j \in \mathcal{D}_i^{tra}} \|\mathbf{x}_{i,j}^* - \mathbf{x}_{i,j}\| \quad (12)$$

$$s.t. \quad \mathbf{x}_{i,j} = \mathbf{x}_{i,j-1} + T_s (f_{neural}(\mathbf{x}_{i,j-1}) + h_{neural}(\mathbf{x}_{i,j-1} - \hat{\mathbf{x}}_{i,j-1}, \boldsymbol{\xi})) \quad (12)$$

where  $n_{case}$  denotes the number of randomized cases,  $\mathcal{D}_i^{tra} = \{\mathbf{x}_{i,j-1}, \hat{\mathbf{x}}_{i,j-1}, \mathbf{x}_{i,j}^* | j = 1, \dots, m\}$  denotes the training set of the  $i$ -th case with  $m$  samples,  $\mathbf{x}_{i,j}^*$  denotes the labeled state, and  $\mathbf{x}_{i,j}$  denotes one-step prediction of state, which is approximated by *Euler* integration method here.

![Figure 4: Two heatmaps showing prediction errors for a spiral curve. The left heatmap shows 'Feedback gain' (0 to 45) on the y-axis and 'Degree of uncertainty' (0 to 45) on the x-axis. The right heatmap shows 'Prediction error' (0 to 40) on the y-axis and 'Degree of uncertainty' (0 to 35) on the x-axis. Both plots show a color scale from 0.0 (blue) to 1.2 (red). A blue star in the left plot marks the origin (0,0). A dashed blue rectangle highlights a region in the left plot.](867fce43c58fda6178b06e454b4ed73a_img.jpg)

Figure 4: Two heatmaps showing prediction errors for a spiral curve. The left heatmap shows 'Feedback gain' (0 to 45) on the y-axis and 'Degree of uncertainty' (0 to 45) on the x-axis. The right heatmap shows 'Prediction error' (0 to 40) on the y-axis and 'Degree of uncertainty' (0 to 35) on the x-axis. Both plots show a color scale from 0.0 (blue) to 1.2 (red). A blue star in the left plot marks the origin (0,0). A dashed blue rectangle highlights a region in the left plot.

Figure 4: Prediction errors of the spiral curve with different levels of feedback gains and uncertainties to show practical implications of Theorem 1 under Assumption 9. The *right* image is a partial enlargement of the *left* one. The blue star denotes the case without uncertainty, and the uncertainty increases along both the left and right directions. When the gain is set as 0, the feedback neural network will equal the neural ODE. The related simulation setup is detailed in Appendix A.3.4.

{5}------------------------------------------------

![Figure 5: A toy example illustrating the developed linear feedback. (a) The training set shows a spiral curve starting from the origin. (b) The test set shows a different spiral curve. (c) Trained performance of the Neural ODE. (d) Testing performance of the Neural ODE. (e) Testing performance of the Feedback NN. (f) Multi-step prediction errors in testing. (g) Performance with decay strategy. (h) Performance with different N.](c54b3ca7603d65d4589151bc3a49d054_img.jpg)

(a) The training set

(b) The test set

(c) Trained performance

(d) Testing performance of Neural ODE

(e) Testing performance of Feedback NN

(f) Multi-step prediction errors in testing

(g) Performance with decay strategy

(h) Performance with different N

Unknown

$$\begin{bmatrix} \dot{x} \\ \dot{y} \end{bmatrix} = \begin{bmatrix} -0.1 & 2 \\ -2 & -0.1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$
$$\begin{bmatrix} \dot{x} \\ \dot{y} \end{bmatrix} = \begin{bmatrix} -0.05 & 3 \\ -3 & -0.05 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} 10 \\ 10 \end{bmatrix}$$

Figure 5: A toy example illustrating the developed linear feedback. (a) The training set shows a spiral curve starting from the origin. (b) The test set shows a different spiral curve. (c) Trained performance of the Neural ODE. (d) Testing performance of the Neural ODE. (e) Testing performance of the Feedback NN. (f) Multi-step prediction errors in testing. (g) Performance with decay strategy. (h) Performance with different N.

Figure 5: A toy example is presented to intuitively illustrate the developed linear feedback. The mission is to predict the future trajectory of a spiral curve with a given initial state  $\{x(t), y(t)\}$ . The neural ODE is trained on a given training set (a), yielding an approving learning result (c). Note that the pentagrams denote start points. The trained network is then transferred to a test set (b), which model is significantly different from the training one. With the linear feedback mechanism, the feedback neural network can achieve a better approximated accuracy of the change rate (e), in comparison with the neural ODE (d). As a result, a smaller multi-step prediction error (f) can be attained by benefiting from the feedback neural network. (g) shows that the noise amplification issue in multi-step prediction can be alleviated by the gain-decay strategy. (h) further presents the prediction results with different prediction steps  $N$ .  $N$  in (f)-(g) is set as 50.

The learning procedure of the feedback part  $h_{neural}(t)$  is summarized as Algorithm 1. After training the neural ODE  $f_{neural}(t)$  on the nominal task, the parameters of simulation model are randomized to produce  $n_{case}$  cases. Subsequently, the feedback neural network is implemented in these cases and the training set  $\mathcal{D}^{trn}$  of each case is constructed. The training loss is then calculated through (12), which favors the update of parameter  $\xi$  by backpropagation. The above steps are repeated until the expected training loss is achieved or the maximum number of iterations was reached.

#### --- **Algorithm 1** Learning neural feedback through domain randomization

**Input:** Randomize parameters to produce  $n_{case}$  cases; trained neural ODE  $f_{neural}$  on nominal task.

**Result:** Neural feedback  $h_{neural}$ .

**Initialize:** Network parameter  $\xi$ ; Adam optimizer.

- 1: **repeat**
 - 2: Run feedback neural network among  $n_{case}$  cases to produce  $\hat{x}_{i,j}$ ;
 - 3: Construct datasets  $\mathcal{D}_i^{trn}$ ;
 - 4: Evaluate loss through (12) on randomly selected mini-batch data;
 - 5: Update  $\xi$  by backpropagation;
 - 6: **until convergence**
-

{6}------------------------------------------------

![Figure 6: Learning with domain randomization. (a) Training Neural ODE through domain randomization: A plot of y vs x showing a spiral trajectory. The 'Truth' is a dashed line, and the model's prediction is a solid line that spirals outwards. (b) Training feedback part through domain randomization: A plot of y vs x showing a spiral trajectory. The 'Truth' is a dashed line, and the model's prediction is a solid line that follows the 'Truth' closely. A color bar on the right indicates the loss, ranging from 0 to 14. (c) Evaluation of training loss: A plot of Loss vs Epochs. The loss starts at approximately 12 and decreases to about 0.49783 by epoch 160.](73c3e4508cae529acf4e6c7fa70b361a_img.jpg)

Figure 6: Learning with domain randomization. (a) Training Neural ODE through domain randomization: A plot of y vs x showing a spiral trajectory. The 'Truth' is a dashed line, and the model's prediction is a solid line that spirals outwards. (b) Training feedback part through domain randomization: A plot of y vs x showing a spiral trajectory. The 'Truth' is a dashed line, and the model's prediction is a solid line that follows the 'Truth' closely. A color bar on the right indicates the loss, ranging from 0 to 14. (c) Evaluation of training loss: A plot of Loss vs Epochs. The loss starts at approximately 12 and decreases to about 0.49783 by epoch 160.

Figure 6: Learning with domain randomization. (a): Train the neural ODE through domain randomization. It can be seen that the learning performance of latent dynamics on the nominal task (Figure 5 (a)) degrades as inducing domain randomization, in comparison with Figure 5 (c). Previous works usually try to scale up neural networks to approach the previous performance. (b): Freeze the neural ODE after training on the nominal task and train the feedback part through domain randomization. The feedback neural network maintains the previous performance on the nominal task. (c) The training loss of the feedback part. Note that the neural ODE employed in (a) and (b) have the same architectures as the one in Figure 5 (c).

For the spiral example, Figure 6 (b) presents the learning performance of the feedback neural network on the nominal task. It can be seen that the feedback neural network can precisely capture the latent dynamics, maintaining the previous accuracy performance of Figure 5 (c). Moreover, the feedback neural network also has the generalization performance on randomized cases, as shown in Appendix Figure S10. Figure 6 (c) further provides the evolution of training loss of the feedback part on the spiral example. More training details are provided in Appendix A.3.3.

## 5 EMPIRICAL STUDY

### 5.1 TRAJECTORY PREDICTION OF AN IRREGULAR OBJECT

Precise trajectory prediction of a free-flying irregular object is a challenging task due to the complicated aerodynamic effects. Previous methods can be mainly classified into model-based scheme (Frese et al., 2001; Müller et al., 2011; Bouffard et al., 2012) and learning-based scheme (Kim et al., 2014; Yu et al., 2021). With historical data, model-based methods aim at accurately fitting the drag coefficient of an analytical drag model, while learning-based ones try to directly learn an acceleration model using specific basis functions. However, the above methods lack of online adaptive ability as employing. Benefiting from the feedback mechanism, our feedback neural network can correct the learned model in real time, leading to a more generalized performance in cases out of training datasets.

![Figure 7 (Left): A diagram showing an irregular bottle being thrown by hand. The bottle is shown at several positions along a parabolic trajectory, with arrows indicating its orientation and velocity at each point. The background shows a simple ground plane.](f3e03accc76df483950e65a9fb19c20e_img.jpg)

Figure 7 (Left): A diagram showing an irregular bottle being thrown by hand. The bottle is shown at several positions along a parabolic trajectory, with arrows indicating its orientation and velocity at each point. The background shows a simple ground plane.

![Figure 7 (Right): A plot of Prediction error [m] vs time t [s]. The x-axis ranges from 0.0 to 0.5 s, and the y-axis ranges from 0.02 to 0.10 m. Three curves are shown: Model based (blue), Neural ODE (black), and Feedback NN (red). The Feedback NN curve is the lowest, indicating the smallest prediction error. Shaded regions around the curves represent the standard deviations of all 9 test trajectories.](023b142f90e1253702ac88b18380d3ec_img.jpg)

Figure 7 (Right): A plot of Prediction error [m] vs time t [s]. The x-axis ranges from 0.0 to 0.5 s, and the y-axis ranges from 0.02 to 0.10 m. Three curves are shown: Model based (blue), Neural ODE (black), and Feedback NN (red). The Feedback NN curve is the lowest, indicating the smallest prediction error. Shaded regions around the curves represent the standard deviations of all 9 test trajectories.

Figure 7: Trajectory prediction results of an irregular bottle. *Left*: The irregular bottle is thrown out by hand and performs an approximate parabolic motion. *Right*: The prediction errors with different methods. The prediction horizon is set as 0.5 s. The colored shaded area represents the standard deviations of all 9 test trajectories.

We test the effectiveness of the proposed method on an open-source dataset (Jia et al., 2024), in comparison with the model-based method (Frese et al., 2001; Müller et al., 2011; Bouffard et al.,

{7}------------------------------------------------

![Figure 8: Training sets and convergence procedures. Left: A 3D plot showing collected trajectories (fly results) in a 3D coordinate system (x, y, z) in meters. The trajectories are dense and overlapping, showing complex movement patterns. Right: A line plot showing the training loss (z (m)) over 50 epochs for six random trials. The loss for all trials starts high (around 12.5) and rapidly decreases, stabilizing around 2.5 after approximately 10 epochs.](3121afa7ca030b22ee0345864ca6f38b_img.jpg)

Figure 8: Training sets and convergence procedures. Left: A 3D plot showing collected trajectories (fly results) in a 3D coordinate system (x, y, z) in meters. The trajectories are dense and overlapping, showing complex movement patterns. Right: A line plot showing the training loss (z (m)) over 50 epochs for six random trials. The loss for all trials starts high (around 12.5) and rapidly decreases, stabilizing around 2.5 after approximately 10 epochs.

Figure 8: Training sets and convergence procedures. *Left*: Collected trajectories used for training. We first randomly sample positional waypoints in a limited space, followed by optimizing polynomials that connect these waypoints through the minimum snap method (Mellinger & Kumar, 2011). Then the quadrotor with the baseline controller from Jia et al. (2022) is commanded to follow planned trajectories, yielding real fly results as the training set. 40 trajectories are collected with the length of 200 discrete nodes each. *Right*: Training curves of 6 random trials. All training trials converged rapidly thanks to stable integration and end-to-end analytic gradients.

(2012) and the learning-based method (Chen et al., 2018). The objective of this mission is to accurately predict the object’s position after 0.5 s, as it is thrown by hand. 21 trajectories are used for training, while 9 trajectories are used for testing. The prediction result is presented in Figure 7. It can be seen that the proposed feedback neural network achieves the best prediction performance. Moreover, the predicted positions and learned latent accelerations of all test trajectories are provided in Figure S2 and Figure S3, respectively. Implementation details are provided in Appendix A.4.

### 5.2 MODEL PREDICTIVE CONTROL OF A QUADROTOR

MPC works in the form of receding-horizon trajectory optimizations with a dynamic model, and then determines the current optimal control input. Approving optimization results highly rely on accurate dynamical models. Befitting from the powerful representation capability of neural networks for complex real-world physics, noticeable works (Torrente et al., 2021; Salzmann et al., 2023; Sukhija et al., 2023) have demonstrated that models incorporating first principles with learning-based components can enhance control performance. However, as the above models are offline-learned within fixed environments, the control performance would degrade under uncertainties in unseen environments.

In this part, the proposed feedback neural network is employed on the quadrotor trajectory tracking scenario concerning model uncertainties and external disturbances, to demonstrate its online adaptive capability. In offline training, a neural ODE is augmented with the nominal dynamics firstly to account for aerodynamic residuals. The augmented model is then integrated with an MPC controller. Note that parameter uncertainties of mass, inertia, and aerodynamic coefficients, and external disturbances are all applied in tests, despite the neural ODE only capture aerodynamic residuals in training. For the feedback neural network, the proposed multi-step prediction strategy is embedded into the model prediction process in MPC. Therefore, the formed feedback-enhanced hybrid model can effectively improve prediction results, further leading to a precise tracking performance. More implementation details refer to Appendix A.5.3.

#### 5.2.1 LEARNING AERODYNAMIC EFFECTS

While learning the dynamics, the augmented model requires the participation of external control inputs, i.e., motor thrusts. Earning a quadrotor model augmented with a neural ODE could be tricky with end-to-end learning patterns since the open-loop model are intensively unstable, leading to the divergence of numerical integration. To address this problem, a baseline controller from Jia et al. (2022) is applied to form a stable closed-loop system. The *adjoint sensitive method* is employed in Chen et al. (2018) to train neural ODEs without considering external control inputs. We provide an alternative training strategy concerning external inputs in Appendix A.1, from the view of optimal control. Figure 8 shows training trajectories and convergence procedures. 5 trials of training are

{8}------------------------------------------------

![Figure 9: Tracking the Lissajous trajectory using MPC with different prediction models. The figure shows six 3D plots of the Lissajous trajectory (x, y, z) over time, comparing Nominal-MPC, Neural-MPC, MLP-MPC, FB-MPC, AdapNN-MPC, and FNN-MPC. A color bar on the right indicates the tracking error [m], ranging from 0.0 (blue) to 0.6 (yellow). Below the plots is a table showing the Root Mean Square Error (RMSE) in meters for each model.](b93cbfb52e37619e688175a6aad9edd9_img.jpg)

|  | Nomi-MPC | Neural-MPC | MLP-MPC | FB-MPC | AdapNN-MPC | <b>FNN-MPC</b> |
|-|-|-|-|-|-|-|
| RMSE [m] | 0.248 | 0.167 | 0.182 | 0.203 | 0.151 | <b>0.093</b> |

Figure 9: Tracking the Lissajous trajectory using MPC with different prediction models. The figure shows six 3D plots of the Lissajous trajectory (x, y, z) over time, comparing Nominal-MPC, Neural-MPC, MLP-MPC, FB-MPC, AdapNN-MPC, and FNN-MPC. A color bar on the right indicates the tracking error [m], ranging from 0.0 (blue) to 0.6 (yellow). Below the plots is a table showing the Root Mean Square Error (RMSE) in meters for each model.

Figure 9: Tracking the *Lissajous* trajectory using MPC with different prediction models.

carried out, each with distinct initial values for network parameters. The trajectory validations are carried out using 3 randomly generated trajectories (Figures S4-S7). More learning details refer to Appendix A.5.2.

#### 5.2.2 FLIGHT TESTS

In tests, MPC is implemented with six different models: the nominal model (27), the neural ODE augmented model (Section A.5.2), the feedforward neural network augmented model (Saviolo & Loiano, 2023), the feedback enhanced nominal model, the adaptive neural network augmented model (Cheng et al., 2019) and the proposed feedback neural network, abbreviated as Nomi-MPC, Neural-MPC, MLP-MPC, FB-MPC, AdapNN-MPC, and FNN-MPC, for the sake of simplification. More details of all compared methods refer to Section A.5.4. Moreover, 37.6% mass uncertainty,  $[40\%, 40\%, 0]$  inertia uncertainties,  $[14.3\%, 14.3\%, 25.0\%]$  drag coefficient uncertainties, and  $[0.3, 0.3, 0.3]N$  translational disturbances are applied. The flight results on a *Lissajous* trajectory (out of training set) are presented in Figure 9. The tracking performance is evaluated by root mean square error (RMSE).

It can be seen the Neural-MPC outperforms the Nomi-MPC since intricate aerodynamic effects are captured by the neural ODE. Moreover, the performance of MLP-MPC is relatively unsatisfactory compared with the Neural-MPC. The reason can be attributed to its single-step training manner instead of the multi-step one of the Neural-MPC, leading to a poor multi-step prediction. However, because unseen parameter uncertainties and external disturbances are not involved in the training set, the Neural-MPC still has considerable tracking errors. Due to the adaptive ability of the last layer, AdapNN-MPC can handle a certain level of uncertainty. In contrast, FNN-MPC achieves the best tracking performance. The reason can be attributed to the multi-step prediction of the feedback neural network improves the prediction accuracy subject to multiple uncertainties, as shown in Figure S8.

## 6 RELATED WORK

### 6.1 NEURAL ODES

Most dynamical systems can be described by ODEs. The establishments of ODEs rely on analytical physics laws and expert experiences previously. To avoid such laborious procedures, Chen et al. (2018) propose to approximate ODEs by directly using neural networks, named neural ODEs. The prevalent residual neural networks (He et al., 2016) can be regarded as an *Euler* discretization of neural ODEs Marion et al. (2024). The universal approximation property of neural ODEs has been studied theoretically (Zhang et al., 2020; Teshima et al., 2020; Li et al., 2022), which show the *sup-universality* for  $C^2$  diffeomorphisms maps (Teshima et al., 2020) and *L<sup>p</sup>-universality* for general continuous maps (Li et al., 2022). Marion (2024) further provides the generalization bound (i.e.,

{9}------------------------------------------------

upper bound on the difference between the theoretical and empirical risks) for a wide range of parameterized ODEs.

### 6.2 GENERALIZATION OF NEURAL NETWORKS

In classification tasks, neural network models face the generalization problem across samples, distributions, domains, tasks, modalities, and scopes (Rohlfs, 2022). Plenty of empirical strategies have been developed to improve the generalization of neural networks, such as model simplification, fit coarsening, and data augmentation for sample generalization, identification of causal relationships for distribution generalization, and transfer learning for domain generalization.

Domain randomization (Tobin et al., 2017; Peng et al., 2018) has shown promising effects to improve the generalization for sim-to-real transfer applications, such as drone racing (Kaufmann et al., 2023), quadrupedal locomotion (Choi et al., 2023), and humanoid locomotion (Radosavovic et al., 2024). The key idea is to randomize the system parameters, noises, and perturbations in simulation so that the real-world case can be covered as much as possible. Although the system’s robustness can be improved, there are two costs to pay. One is that the computation burden in the training process is dramatically increased. The other is that the training result has a certain of conservativeness since the training performance is an average of different scenarios, instead of a specific case.

### 6.3 REAL-TIME RETRAINING AND ADAPTATION

Recently, online continual learning (Ghunaim et al., 2023) and test-time adaptation (Liang et al., 2024) have emerged as promising solutions to handle unknown test distribution shifts. Online continual learning focuses on the reduction of real-time training load, aiming at generalizing across new tasks while maintaining performance on previous tasks. Test-time adaptation tries to utilize real-time unlabeled data to obtain self-adapted models. For example, an extended *kalman* filter-based adaptation algorithm with a forgetting factor is developed by Abuduwaili & Liu (2020) to generalize neural network-based models. Moreover, in order to improve the flexibility of neural networks, the last layer of networks can be regarded as a weighted vector, which can be adjusted adaptively according to real-time state feedback (Cheng et al., 2019; O’Connell et al., 2022; Richards et al., 2023; Saviolo et al., 2024). The training for separating the last layer and front structure can be carried out within a bi-level optimization framework. In such a paradigm, the uncertainty out of training sets is reflected on the last layer of networks, which can be online adjusted in a control-oriented (Richards et al., 2023) or regression-oriented (Cheng et al., 2019; O’Connell et al., 2022) fashion. Patil et al. (2022) further develops real-time weight adaptation laws for all layers of feedforward neural networks, with stability guarantees.

Different from the above retraining or adaptation strategy, the presented method directly corrects the learned latent dynamics of neural ODEs with real-time feedback, yielding a two-DOF network structure. Moreover, the feedback can be learned in a neural form. Integrating adaptive neural ODEs with the developed feedback mechanism may be a valuable research direction (Section A.8).

## 7 CONCLUSION

Inspired by the feedback philosophy in biological and engineering systems, we proposed to incorporate a feedback loop into the neural network structure for the first time, as far as we known. In such a way, the learned latent dynamics can be corrected flexibly according to real-time feedback, leading to better generalization performance in continuous-time missions. The convergence property under a linear feedback form was analyzed. Subsequently, domain randomization was employed to learn a nonlinear neural feedback, resulting in a two-DOF neural network. Finally, applications on trajectory prediction of irregular objects and MPC of robots were shown.

**Limitations.** First, the feedback gain and decay rate for the linear feedback neural network need to be tuned manually. Future work will try to build a bi-level optimization framework to train neural ODE while searching the optimal gains. Such joint optimization manner can also capture the coupled information between feedforward neural ODE and feedback network. Moreover, the presented nonlinear neural form is preliminarily tested in Section 4. Future work will pursue to exploit its potential in more complex tasks.

 Rest of paper (reference and Appendix) is removed.