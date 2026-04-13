

{0}------------------------------------------------

# META-DYNAMICAL STATE SPACE MODELS FOR INTEGRATIVE NEURAL DATA ANALYSIS

Ayesha Vermani<sup>1</sup>, Josue Nassar<sup>2</sup>, Hyungju Jeon<sup>1</sup>, Matthew Dowling<sup>1</sup>, Il Memming Park<sup>1</sup>

<sup>1</sup> Champalimaud Centre for the Unknown, Champalimaud Foundation, Portugal

<sup>2</sup> RyvivyR, USA

{ayesha.vermani, memming.park}@research.fchampalimaud.org

## ABSTRACT

Learning shared structure across environments facilitates rapid learning and adaptive behavior in neural systems. This has been widely demonstrated and applied in machine learning to train models that are capable of generalizing to novel settings. However, there has been limited work exploiting the shared structure in neural activity during similar tasks for learning latent dynamics from neural recordings. Existing approaches are designed to infer dynamics from a single dataset and cannot be readily adapted to account for statistical heterogeneities across recordings. In this work, we hypothesize that similar tasks admit a corresponding family of related solutions and propose a novel approach for meta-learning this solution space from task-related neural activity of trained animals. Specifically, we capture the variabilities across recordings on a low-dimensional manifold which concisely parametrizes this family of dynamics, thereby facilitating rapid learning of latent dynamics given new recordings. We demonstrate the efficacy of our approach on few-shot reconstruction and forecasting of synthetic dynamical systems, and neural recordings from the motor cortex during different arm reaching tasks.

## 1 INTRODUCTION

Latent variable models are widely used in neuroscience to extract dynamical structure underlying high-dimensional neural activity (Pandarinath et al., 2018; Schimel et al., 2022; Dowling et al., 2024). While latent dynamics provide valuable insights into behavior and generate testable hypotheses of neural computation (Luo et al., 2023; Nair et al., 2023), they are typically inferred from a single recording session. As a result, these models are sensitive to small variations in the underlying dynamics and exhibit limited generalization capabilities. In parallel, a large body of work in machine learning has focused on training models from diverse datasets that can rapidly adapt to novel settings. However, there has been limited work on inferring generalizable dynamical systems from data, with existing approaches mainly applied to settings with known low-dimensional dynamics (Yin et al., 2021; Kirchmeyer et al., 2022).

Integrating noisy neural recordings from different animals and/or tasks for learning the underlying dynamics presents a unique set of challenges. This is partly due to heterogeneities in recordings across sessions such as the number and tuning properties of recorded neurons, as well as different stimulus statistics and behavioral modalities across cognitive tasks. This challenge is further compounded by the lack of inductive biases for disentangling the variabilities across dynamics into shared and dataset-specific components. Recent evidence suggests that learned latent dynamics underlying activity of task-trained biological and artificial neural networks demonstrate similarities when engaged in related tasks (Gallego et al., 2018; Maheswaranathan et al., 2019; Safaie et al., 2023). In a related line of work, neural networks trained to perform multiple cognitive tasks with overlapping cognitive components learn to reuse dynamical motifs, thereby facilitating few-shot adaptation on novel tasks (Turner & Barak, 2023; Driscoll et al., 2024).

Motivated by these observations, we propose a novel framework for meta-learning latent dynamics from neural recordings (Vermani et al., 2024a). Our approach is to encode the variations in the latent dynamical structure present across neural recordings in a low-dimensional vector,  $e \in \mathbb{R}^{d_e}$ , which we refer to as the *dynamical embedding*. During training, the model learns to adapt a common

{1}------------------------------------------------

![Figure 1: A. Neural recordings display heterogeneities in the number and tuning properties of recorded neurons and reflect diverse behavioral responses. The low-dimensional embedding manifold captures this diversity in dynamics. B. Our method learns to adapt a common latent dynamics conditioned on the embedding via low-rank changes to the model parameters.](49ad3a646d84bcfeac02bdf2b3792a3e_img.jpg)

Figure 1 consists of two parts. Part A shows neural recordings from three subjects (Subject C, Subject Je, Subject Ni) and a hand position. Each recording is plotted as a line graph of 'mips' vs 'time'. A 3D plot shows an 'Embedding Manifold' with axes  $m_1$  and  $m_2$ , where points representing different subjects are scattered. Part B is a 'Generative Model Overview' diagram. It shows a latent state  $z_t^i$  being updated by a hypernetwork that takes an embedding  $e^i$  and the previous state  $h_{t-1}$  as input. The latent state is then used to produce an observation  $y_t^i$ . The diagram includes shared weights  $W_t$  and a summation symbol  $\oplus$ .

Figure 1: A. Neural recordings display heterogeneities in the number and tuning properties of recorded neurons and reflect diverse behavioral responses. The low-dimensional embedding manifold captures this diversity in dynamics. B. Our method learns to adapt a common latent dynamics conditioned on the embedding via low-rank changes to the model parameters.

Figure 1: **A.** Neural recordings display heterogeneities in the number and tuning properties of recorded neurons and reflect diverse behavioral responses. The low-dimensional embedding manifold captures this diversity in dynamics. **B.** Our method learns to adapt a common latent dynamics conditioned on the embedding via low-rank changes to the model parameters.

latent dynamical system model conditioned on the dynamical embedding. We learn the dynamical embedding manifold from a diverse collection of neural recordings, allowing rapid learning of latent dynamics in the analysis of data-limited regime commonly encountered in neuroscience experiments. Our contributions can be summarized as follows:

1. We propose a novel parameterization of latent dynamics that facilitates integration and learning of meta-structure over diverse neural recordings.
2. We develop an inference scheme to jointly infer the embedding and latent state trajectory, as well as the corresponding dynamics model directly from data.
3. We demonstrate the efficacy of our method on few-shot reconstruction and forecasting for synthetic datasets and motor cortex recordings obtained during different reaching tasks.

## 2 CHALLENGES WITH JOINTLY LEARNING DYNAMICS ACROSS DATASETS

Neurons from different sessions and/or subjects are partially observed, non-overlapping and exhibit diverse response properties. Even chronic recordings from a single subject exhibit drift in neural tuning over time (Driscoll et al., 2017). Moreover, non-simultaneously recorded neural activity lack pairwise correspondence between single trials. This makes joint inference of latent states and learning the corresponding latent dynamics by integrating different recordings ill-posed and highly non-trivial.

As an illustrative example, let’s consider a case where these recordings exhibit oscillatory latent dynamics with variable velocities (Fig. 2A). One possible strategy for jointly inferring the dynamics from these recordings is learning a shared dynamics model, along with dataset-specific likelihood functions that map these dynamics to individual recordings (Pandarinath et al., 2018). However, without additional inductive biases, this strategy does not generally perform well when there are variabilities in the underlying dynamics. Specifically, when learning dynamics from two example datasets ( $M = 2$ ), we observed that a model

![Figure 2: A. Three different example neural recordings. B. One generative model is trained on M = 2 or M = 20 datasets. While increasing the number of datasets allows the model to learn limit cycle, it is unable to capture the different speeds leading to poor forecasting performance.](1ac0e11d90bece49015adc89be472a39_img.jpg)

Figure 2 consists of two parts. Part A shows three example neural recordings, labeled  $y^1$ ,  $y^2$ , and  $y^3$ , plotted as 'time' vs 'y'. Part B shows two phase space plots of  $z_2$  vs  $z_1$  for a generative model trained on  $M = 2$  and  $M = 20$  datasets. The  $M = 2$  plot shows a limit cycle, while the  $M = 20$  plot shows a more complex limit cycle. A third plot shows the 'k-step' forecast error for  $M = 2$  (dashed line) and  $M = 20$  (solid line), indicating that the  $M = 20$  model has a lower error.

Figure 2: A. Three different example neural recordings. B. One generative model is trained on M = 2 or M = 20 datasets. While increasing the number of datasets allows the model to learn limit cycle, it is unable to capture the different speeds leading to poor forecasting performance.

Figure 2: **A.** Three different example neural recordings, where the speed of the latent dynamics varies across them. **B.** One generative model is trained on  $M = 2$  or  $M = 20$  datasets. While increasing the number of datasets allows the model to learn limit cycle, it is unable to capture the different speeds leading to poor forecasting performance.

{2}------------------------------------------------

with shared dynamics either learned separate solutions or overfit to one dataset, obscuring global structure across recordings (Fig. 2A). When we increased the diversity of training data ( $M = 20$ ), the dynamics exhibited a more coherent global structure, albeit with an overlapping solution space (Fig. 2B). As a result, this model had poor forecasting performance of neural activity in both cases, which is evident in the k-step  $r^2$  (Fig. 2B). While we have a priori knowledge of the source of variations in dynamics for this example, this is typically not the case with real neural recordings. Therefore, we develop an approach for inferring the variation across recordings and use it to define a solution space of related dynamical systems (Fig. 1A).

## 3 INTEGRATING NEURAL RECORDINGS FOR META-LEARNING DYNAMICS

Let  $y_{1:T}^{1:M}$  denote neural time series datasets of length  $T$ , with  $y_t^i \in \mathbb{R}^{d_{y^i}}$ , collected from  $M$  different sessions and/or subjects performing related tasks. We are interested in learning a generative model that can jointly describe the evolution of the latent states across these datasets and rapidly adapt to novel datasets from limited trajectories. In this work, we focus on nonlinear state-space models (SSM), a powerful class of generative models for spatio-temporal datasets. An SSM is described via the following pair of equations (we drop the superscript for ease of presentation),

$$z_t | z_{t-1} \sim p_\theta(z_t | z_{t-1}), \quad (1)$$

$$y_t | z_t \sim p_\phi(y_t | z_t), \quad (2)$$

where  $z_t \in \mathbb{R}^{d_z}$  is the latent state at time  $t$ ,  $p_\theta(z_t | z_{t-1})$  is the dynamics model and  $p_\phi(y_t | z_t)$  is the likelihood function that maps the latent state to observed data.

We parametrize the dynamics as a Gaussian distribution  $p_\theta(z_t | z_{t-1}) = \mathcal{N}(z_t | f_\theta(z_{t-1}), Q)$ , where the mean is modeled by a deep neural network (DNN)  $f_\theta$  and  $Q$  is the covariance matrix<sup>1</sup>. As previous work has shown that highly expressive likelihood and dynamics can cause optimization issues (Bowman et al., 2015), we model the mean of the likelihood as an affine function of  $z_t$ . For instance, the likelihood for real-valued observations is defined as  $p_\phi(y_t | z_t) = \mathcal{N}(y_t | C z_t + D, R)$ .

### 3.1 HIERARCHICAL STATE-SPACE MODEL FOR MULTIPLE DATASETS

We introduce a hierarchical structure in the latent dynamical system model to capture variations across datasets and jointly describe the spatiotemporal evolution across  $M$  neural recordings in a unified SSM. A natural choice for learning this generative model is a fully Bayesian approach, where each dataset would have its own latent dynamics, parameterized by  $\theta^i$ , and a hierarchical prior would tie these dataset-specific parameters to shared parameters,  $\theta \sim p(\hat{\theta})$  (Linderman et al., 2019), leading to the following SSM,

$$\theta^i | \theta \sim p(\theta^i | \theta), \quad (3)$$

$$z_t^i | z_{t-1}^i, \theta^i \sim \mathcal{N}(z_t^i | f_{\theta^i}(z_{t-1}^i), Q^i), \quad (4)$$

$$y_t^i | z_t^i \sim p_{\phi^i}(y_t^i | z_t^i), \quad (5)$$

where dataset specific likelihoods,  $p_{\phi^i}(y_t^i | z_t^i)$ , are used to account for different dimensionality and/or recording modality. If we assume  $p(\theta^i | \theta)$  is Gaussian, i.e.,  $p(\theta^i | \theta) = \mathcal{N}(\theta^i | \theta, \Sigma)$ , we can equivalently express the dynamics for the hierarchical generative model as,

$$\varepsilon^i \sim \mathcal{N}(\varepsilon^i | 0, \Sigma), \quad (6)$$

$$z_t^i | z_{t-1}^i, \theta, \varepsilon^i \sim \mathcal{N}(z_t^i | f_{\theta + \varepsilon^i}(z_{t-1}^i), Q^i), \quad (7)$$

where the dataset-specific dynamics parameter,  $\theta^i$ , is expressed as a sum of the shared parameters,  $\theta$ , and a dataset-specific term,  $\varepsilon^i$ . While this formulation is intuitive, the latent dynamics are approximated using a DNN, which introduces a substantial number of parameters and constrains scalability. To address these limitations, we propose a modified hierarchical framework that significantly improves both scalability and parameter efficiency, making it suitable for large-scale settings.

Specifically, we introduce a low-dimensional latent variable,  $\varepsilon^i \in \mathbb{R}^{d_\varepsilon}$ ,  $d_\varepsilon \ll d_z$ —which we refer to as the dynamical embedding—that encodes dynamical variations across datasets (Rusu et al.,

<sup>1</sup>We note that  $Q$  can also be parameterized via a neural network as well.

{3}------------------------------------------------

2019). This dataset-specific dynamical embedding subsequently maps to the parameter space of the latent dynamics function via a hypernetwork (Ha et al., 2016),  $h_\theta : \mathbb{R}^{d_e} \rightarrow \mathbb{R}^{d_L}$ . Apart from improving scalability, this formulation also facilitates efficient few-shot learning since it requires simply inferring the embedding given trials from novel recordings. The generative model for this hierarchical SSM is then described as,

$$e^i \sim \mathcal{N}(0, I), \quad (8)$$

$$\theta^i = \theta + h_\theta(e^i), \quad (9)$$

$$z_t^i | z_{t-1}^i, e^i \sim \mathcal{N}(z_t^i | f_{\theta^i}(z_{t-1}^i), Q^i), \quad (10)$$

$$y_t^i | z_t^i \sim p_{\phi^i}(y_t^i | z_t^i), \quad (11)$$

where we drop the prior over the shared dynamics parameter,  $\theta$ , significantly reducing the dimensionality of the inference problem. Similar to the hierarchical Bayesian model, all datasets share the same latent dynamics,  $\theta$ , with the dataset-specific variation captured by the dynamical embedding,  $e_i$ .

We encourage learning of shared dynamical structure and further improve parameter efficiency by constraining  $h_\theta$  to make low-rank changes to the parameters of  $f_\theta$  (Fig. 1B). For example, if we parameterize  $f_\theta$  as a 2-layer fully-connected neural network and constrain the hypernetwork to only make rank  $d_r$  changes to the hidden weights, then  $f_{\theta^i}$  would be expressed as,

$$f_{\theta^i}(z_t^i) = \mathbf{W}_o \sigma(\{\underbrace{\mathbf{W}_{hh} + h_\theta(e^i)}_{\text{embedding modification}}\} \sigma(\mathbf{W}_{in} z_t^i)) \quad (12)$$

$$= \underbrace{\mathbf{W}_o}_{\mathbb{R}^{d_z \times d_2}} \sigma(\{\underbrace{\mathbf{W}_{hh}}_{\mathbb{R}^{d_2 \times d_r}} + \underbrace{\mathbf{u}_\theta(e^i)}_{\mathbb{R}^{d_2 \times d_r}} \cdot \underbrace{\mathbf{v}_\theta(e^i)^\top}_{\mathbb{R}^{d_r \times d_1}}\} \sigma(\underbrace{\mathbf{W}_{in}}_{\mathbb{R}^{d_1 \times d_z}} z_t^i)) \quad (13)$$

where  $\sigma(\cdot)$  denotes a point-nonlinearity, and the two functions  $\mathbf{v}_\theta(e^i) : \mathbb{R}_e^d \rightarrow \mathbb{R}^{d_1 \times d_r}$ ,  $\mathbf{u}_\theta(e^i) : \mathbb{R}_e^d \rightarrow \mathbb{R}^{d_2 \times d_r}$  map the embedding representation to form the low-rank perturbations. Both  $\mathbf{u}_\theta$  and  $\mathbf{v}_\theta$  are parametrized via a neural network.

### 3.2 INFERENCE AND LEARNING

Given  $y_1^{1:M}$ , we want to infer both the latent states,  $z_{1:T}^{1:M}$  and the dynamical embeddings,  $e^{1:M} = [e^1, \dots, e^M]$  as well as learn the parameters of the generative model,  $\Theta = \{\theta, \vartheta, \phi^1, \dots, \phi^M\}$ . Exact inference and learning requires computing the posterior,  $p_\Theta(z_{1:T}^{1:M}, e^{1:M} | y_{1:T}^{1:M})$ , and log marginal likelihood,  $\log p_\Theta(y_{1:T}^{1:M})$ , which are both intractable.

In this paper, we use a sequential variational autoencoder—an extension of variational autoencoders for state-space models—specifically, the Deep Kalman Filter (DKF) (Krishnan et al., 2015), to circumvent this issue. In order to learn the generative model, we maximize a lower-bound to the log marginal likelihood (commonly referred to as the ELBO). The ELBO for  $y_{1:T}^{1:M}$  is defined as follows (trial indices are omitted for ease of notation),

$$\begin{aligned} \mathcal{L}(y_{1:T}^{1:M}) = & \sum_{t,i} \mathbb{E}_{q_{\alpha,\beta}} [\log p_{\phi^i}(y_t^i | z_t^i)] \\ & - \mathbb{E}_{q_{\alpha\beta}} [\mathbb{D}_{\text{KL}}(q_\beta(z_t^i | \bar{y}_{1:T}^i, e^i) || p_{\theta,\vartheta}(z_t^i | z_{t-1}^i, e^i))] - \mathbb{D}_{\text{KL}}(q_\alpha(e^i | \bar{y}_{1:T}^i) || p(e^i)) \end{aligned} \quad (14)$$

where  $q_\alpha$  and  $q_\beta$  are encoders that approximate the posterior distributions over the dynamical embedding and latent state for dataset  $i$ , respectively, and the joint expectation factorizes as  $\mathbb{E}_{q_{\alpha,\beta}} \equiv \mathbb{E}_{q_\beta(z_t^i | \bar{y}_{1:T}^i, e^i) q_\alpha(e^i | \bar{y}_{1:T}^i)}$ . As described in Sec. 2, one of the challenges with integrating recordings in a common latent space is different dimensionalities (number of recorded neurons) as well as the dependence of neural activity on the shared latent space. We address this by training additional read-in networks  $\Omega_i : \mathbb{R}^{d_y} \rightarrow \mathbb{R}^{d_\theta}$  for each dataset that map  $y_t^i$  to an intermediate vector, which we denote by  $\bar{y}_t^i \in \mathbb{R}^{d_\theta}$ . This read-in network ensures that the latent states and dynamical-embeddings inferred from each dataset are aligned to live in the same space (Vermani et al., 2024b).

{4}------------------------------------------------

While there are many choices for parameterizing the encoders, we follow the parameterization in (Krishnan et al., 2015) for simplicity<sup>2</sup>, defined as follows,

$$\bar{y}_{b,1:T}^i = \Omega^i(y_{b,1:T}^i), \quad (15)$$

$$q_\alpha(e^i | \bar{y}_{b,1:T}^i) = \mathcal{N}(e_b^i | \text{agg}[\mu_\alpha(\bar{y}_{b,1:T}^i)], \text{agg}[\sigma_\alpha^2(\bar{y}_{b,1:T}^i)]), \quad (16)$$

$$q_\beta(z_{1:T}^i | \bar{y}_{1:T}^i, e_b^i) = \prod_{t=1}^T \mathcal{N}(z_t^i | \mu_\beta(\text{concat}[\bar{y}_{b,1:T}^i, e_b^i]), \sigma_\beta^2(\text{concat}[\bar{y}_{b,1:T}^i, e_b^i])), \quad (17)$$

where  $y_b^i$  denotes a randomly sampled mini-batch of trials  $b$  from dataset  $i$ ,  $\text{concat}$  is the concatenation operation, and  $\text{agg}$  is an aggregation operation. We aggregate the dynamical embedding over trials in a mini-batch that belong to the same dataset since we are interested in capturing inter-dataset, rather than intra-dataset variations, in the underlying dynamical systems. In practice, we parameterize  $\mu_\alpha(\cdot)$ ,  $\sigma_\alpha^2(\cdot)$  by a bidirectional recurrent neural network, and  $\mu_\beta(\cdot)$ ,  $\sigma_\beta^2(\cdot)$  by a regular recurrent network, and  $\text{agg}$  corresponds to a simple averaging function. We emphasize that  $\mu_\alpha$ ,  $\sigma_\alpha^2$ ,  $\mu_\beta$ , and  $\sigma_\beta^2$  are shared across all datasets (See Fig. 14 for details on inference).

### 3.3 PROOF OF CONCEPT

![Figure 3: Proof of Concept. A. Mean dynamical system corresponding to the slowest velocity recording learned by the proposed approach when trained with M = 20 datasets. B. Inferred embedding showing a limit cycle in the velocity space. C. Forecasting r^2 at (k = 50)-step for models trained with M = 2 or M = 20 datasets.](3ef2266eb61ae3929cdae1742c1f526e_img.jpg)

Figure 3 consists of three panels. Panel A shows a vector field representing the mean dynamical system learned by the model, with a central point and arrows indicating the flow. Panel B is a scatter plot of the inferred embedding, showing a clear limit cycle in the velocity space. Panel C is a bar chart comparing the prediction  $r^2$  for models trained with  $M=2$  and  $M=20$  datasets, showing that the model with  $M=20$  has a higher prediction  $r^2$ .

Figure 3: Proof of Concept. A. Mean dynamical system corresponding to the slowest velocity recording learned by the proposed approach when trained with M = 20 datasets. B. Inferred embedding showing a limit cycle in the velocity space. C. Forecasting r^2 at (k = 50)-step for models trained with M = 2 or M = 20 datasets.

Figure 3: **A.** Mean dynamical system corresponding to the slowest velocity recording learned by the proposed approach when trained with  $M = 20$  datasets. **B.** Samples from the inferred dynamical embedding for each dataset (see eq. 16). **C.** Forecasting  $r^2$  at  $(k = 50)$ -step for models trained with  $M = 2$  or  $M = 20$  datasets.

As a proof of concept, we revisit the motivating example presented in Section 2 as a means to validate the efficacy of our approach and investigate how it unifies dynamics across datasets. For both  $M = 2$  and  $M = 20$  datasets, we used an embedding dimensionality of 1 and allowed the network to make a rank-1 change to the dynamics parameters.

After training, we observed that the shared dynamics (when  $e = 0$ ) converged to a limit cycle with a slow velocity (Fig. 3A)—capturing the global topology that is shared across all datasets—and the model learned to modulate the velocity of the dynamics conditioned on the dynamical embedding which strongly correlated with the dataset specific velocity<sup>3</sup> (Fig. 3B). This demonstrated that the proposed approach is able to capture dataset-specific variability. Lastly, Fig. 3C demonstrates that the proposed approach is able to forecast well for both  $M = 2$  and  $M = 20$  datasets. We include further validation experiments when there is no model mismatch as well as the generalization of the trained model

to new data in Appendix B. We additionally include results on these recordings from multi-session CEBRA (Schneider et al., 2023) in Appendix B.

## 4 RELATED WORKS

**Multi-Dataset Training in Neuroscience.** Previous work has explored multi-dataset training for extracting latent representations in neuroscience, especially across datasets recorded during the same behavioral tasks. LFADS (Pandarinath et al., 2018), a variant of the seqVAE framework, used session-stitching with dataset-specific likelihood functions, but focused on single-animal recordings. Linderman et al. (2019) used a hierarchical Bayesian state-space model with switching linear dynamical systems, while Herrero-Vidal et al. (2021) developed a joint model with shared linear dynamics and dataset-specific likelihoods. In contrast to these approaches, we incorporate a more

<sup>2</sup>We evaluate alternative inference and learning formulations in Appendix D

<sup>3</sup>Note that we plot the absolute embedding samples since the likelihood function can introduce arbitrary invariance such as direction flipping, rotation, and so on.

{5}------------------------------------------------

expressive function to approximate the underlying family of dynamical systems which can disentangle variabilities across recordings. CEBRA (Schneider et al., 2023) and CS-VAE (Yi et al., 2023) have been developed for extracting latent representations by integrating multiple datasets. The multi-session training objective in CEBRA promotes invariant feature learning across datasets, while CS-VAE partitions the latent space to learn structured features from behavioral videos. In our framework, we jointly infer latent trajectories from multi-session recordings and learn a unified generative model that captures variations in dynamical systems from the inferred trajectories. Recently, there has been growing interest in using diverse neural recordings for training large-scale foundation models in neuroscience (Ye et al., 2023; Zhang et al., 2023; Caro et al., 2024; Azabou et al., 2024; Vermani et al., 2024a). These models leverage transformer-based architectures which lack recurrent hidden states and only incorporate temporal information indirectly via positional encoding. While our approach shares the same broad goal of pretraining a single generative model for rapid learning on downstream recordings, the focus of our work is on learning a family of dynamical systems underlying recordings.

**Recurrent Neural Network Models in Neuroscience** Integrative modeling of dynamical behaviors has also been explored in RNN models of neural systems (Yang et al., 2019; Driscoll et al., 2024). In Driscoll et al. (2024), the authors trained an RNN to perform multiple cognitive tasks and observed motifs corresponding to distinct dynamical behaviors. The broad idea of dynamical structure re-use is similar to our work but there are subtle differences—we are interested in capturing both topological and geometrical differences, and the “context” is learned from data. The motifs in Driscoll et al. (2024) corresponded to a distinct fixed point structure or topology with a pre-specified context input that could push the dynamics to task-relevant regions in the state space. The embedding analysis in Cotler et al. (2023), where a meta-model was trained to capture the activity of multiple trained RNNs is quite similar to our main idea since they observed similar dynamical properties in models that were close in the embedding space. Recent work on modeling motor adaptation (Pellegrino et al., 2023) by low-tensor rank learning in RNNs is broadly similar to our work since the authors adapt the weights in a low-rank RNN to capture variations in dynamics across trials. In contrast, we are interested in modeling dynamical variations across recording sessions and/or tasks.

Additional related works can be found in Appendix A.

## 5 EXPERIMENTS

We first validate the proposed method on synthetic data and then test our method on neural recordings from the primary motor and premotor cortex. We compare the proposed approach against the following baselines for all experiments.

We train a separate **Single Session** model using the seqVAE framework on each dataset. Given sufficient training data, this should result in the best performance, but will fail in trial-limited regimes. We consider a multi-session **Shared Dynamics** model with dataset-specific likelihoods (Pandarinarath et al., 2018; Herrero-Vidal et al., 2021). We also compare against a baseline where the embedding is provided as an additional input to the dynamics model (**Embedding-Input**), a similar formulation to CAVIA (Concat) (Zintgraf et al., 2019) and DYNAMO (Cotler et al., 2023). We also test the hypernetwork parametrization proposed in CoDA (Kirchmeyer et al., 2022), where the hypernetwork adapts all parameters as a linear function of the dynamical embedding (**Linear-Adapter**).

We include additional baselines for the motor cortex experiment; we evaluate single session **LFADS** (Pandarinarath et al., 2018) with the controller as an alternative generative model comparison. We also consider different methods for learning and inference. Specifically, we include single-session models as well as our proposed generative model trained using **Variational Sequential Monte Carlo** (VSMC) (Naesseth et al., 2018), and the **Deep Variational Bayes Filter** (DVBF) (Karl et al., 2016).

For all experiments, we split each of the  $M$  datasets into a training and test set and report reconstruction and forecasting metrics on the test set. To measure the generalization performance, we also report these metrics on held-out datasets. Further details on training and evaluation metrics can be found in Appendix G.

{6}------------------------------------------------

### 5.1 BIFURCATING SYSTEMS

In these experiments, we test whether our method could capture variations across multiple datasets, particularly in the presence of significant dynamical shifts, such as bifurcations commonly observed in real neural populations. We selected two parametric classes of dynamical systems for testing: i) a system undergoing a Hopf bifurcation and, ii) the unforced Duffing system. We include the results for training on datasets generated only from the Hopf system in Appendix C and we discuss the results from joint training on both systems here. We briefly outline the data generation process for the Duffing system (details of the data generation for the Hopf system can be found in Appendix E.2).

The latent trajectories for the Duffing system were generated from a family of stochastic differential equations,

$$\dot{z}_1 = z_2 + 5 dW_t, \quad \dot{z}_2 = a^i z_2 - z_1(b^i + cz_1^2) + 5 dW_t \quad (18)$$

with  $c = 0.1$ ,  $a, b \in \mathbb{R}$ , and  $dW_t$  denoting the Wiener process. In Fig. 4A, we visualize how the dynamical system changes as  $a$  and  $b$  vary. We chose  $M = 20$  pairs of  $(a^i, b^i)$  values (Fig 13),

![Figure 4: A. Example True Dynamics and Learned Dynamics. B. Generalization on Training Tasks. C. Mean reconstruction and forecasting r^2.](01da0d212fb571933f10f96556157745_img.jpg)

Figure 4 consists of three panels. Panel A shows a 2x2 grid of phase space plots. The left column is labeled 'Example True Dynamics' and the right column is labeled 'Learned Dynamics'. The rows are labeled 'varying a' (with values a=0.1 and a=-0.8) and 'varying b' (with values b=-0.1 and b=-1.4). Each plot shows trajectories in the (z1, z2) plane. Panel B is a bar chart titled 'Generalization on Training Tasks' showing 'Reconstruction r^2' for five methods: Our Method (purple), Linear Adapter (green), Single Session (grey), Shared Dynamics (orange), and Embedding Input (red). Panel C is a line plot titled 'Prediction r^2' showing performance over 'k-step' (1 to 50) for the same five methods.

Figure 4: A. Example True Dynamics and Learned Dynamics. B. Generalization on Training Tasks. C. Mean reconstruction and forecasting r^2.

Figure 4: A. (Left) True underlying dynamics from some example datasets used for pretraining as a function of parameters  $a$  and  $b$  and (Right) the embedding conditioned dynamics learnt by our model. B, C. Mean reconstruction and forecasting  $r^2$  of the observations for all datasets used for pretraining on test trials.

and generated latent trajectories of length  $T = 300$ . Observations were generated according to  $y_t^i \sim \mathcal{N}(C^i z_t^i, 0.01\mathbb{I})$  with the dimensionality of the observations varying between 30 and 100. In addition to these 20 Duffing system datasets, we included 11 datasets from the Hopf system (Appendix C), and used 128 trajectories from each of these 31 datasets for training all methods. We report performance on 64 test trajectories from each dataset. We used  $d_c = 2$  for all embedding-conditioned approaches and constrained the hypernetwork to make rank  $d_r = 1$  changes for our approach.

Our approach learned a good approximation to the ground-truth dynamics of the Duffing oscillator system, successfully disentangling different dynamical regimes (Fig. 4 B). Apart from learning the underlying topology of dynamics, it also better captured the geometrical properties compared to other embedding-conditioned baselines (Fig. 15). We observed similar results for datasets from the Hopf system—while our approach approximated the ground-truth system well, the Embedding-Input baseline displayed interference between dynamics and the Linear-Adapter learned a poor approximation to the ground-truth system (Fig. 16). Consequently, our approach outperformed other methods on forecasting observations with all methods having comparable reconstruction performance (Fig. 4B, C). Notably, apart from the  $d_c$ , we used the same architecture as when training on only the Hopf datasets, and did not observe any drop in performance for our approach, in contrast to baselines (Fig. 11C (Bottom), Fig. 4C).

Next, we tested the few-shot performance of all methods on new datasets, two generated from the Duffing oscillator system and one from the Hopf system, as a function of  $n_s$ , the number of trials used for learning the dataset specific read-in network,  $\Omega^i$  and likelihood. Our approach and the Linear-Adapter demonstrated comparable forecasting performance when using  $n_s = 1$  and  $n_s = 8$  training trajectories. However, with  $n_s = 16$  training trials, unlike other methods, our approach continued to improved and outperformed them (Table 1). This could be explained by looking at the

{7}------------------------------------------------

|  | $n_s = 1$ | $n_s = 8$ | $n_s = 16$ |
|-|-|-|-|
| Our Method | <b><math>0.69 \pm 0.072</math></b> | <b><math>0.78 \pm 0.051</math></b> | <b><math>0.87 \pm 0.037</math></b> |
| Linear-Adapter | <b><math>0.68 \pm 0.08</math></b> | <b><math>0.79 \pm 0.026</math></b> | $0.74 \pm 0.039$ |
| Single Session | $0.47 \pm 0.119$ | <b><math>0.79 \pm 0.014</math></b> | $0.79 \pm 0.047$ |
| Shared Dynamics | $-0.31 \pm 0.103$ | $-0.34 \pm 0.086$ | $-0.13 \pm 0.065$ |
| Embedding-Input | $0.59 \pm 0.084$ | <b><math>0.77 \pm 0.04</math></b> | $0.74 \pm 0.039$ |

Table 1: Few shot forecasting performance ( $k = 30$ -step) on 3 held-out datasets as a function of  $n_s$ , the number of trials used to learn dataset specific read-in network and likelihood. ( $\pm 1$  s.e.m)

inferred embedding on held-out datasets—as we increased the number of training trajectories, the model was able to consistently align to the “correct” embedding (Fig. 17).

### 5.2 MOTOR CORTEX RECORDINGS

Next, we tested the applicability of the proposed approach on neural data. We used single and multi-unit neural population recordings from the motor and premotor cortex during two behavioral tasks—the Centre-Out (CO) and Maze reaching tasks (Perich et al., 2018; Gallego et al., 2020; Churchland et al., 2012). In the CO task, subjects are trained to use a manipulandum to reach one of eight target locations on a screen. In the Maze task, subjects use a touch screen to reach a target location, while potentially avoiding obstacles. These recordings spanned different sessions, animals, and labs, and involved different behavioral modalities, while still having related behavioral components, making them a good testbed for evaluating various methods. For training, we used 40 sessions from the CO task, from subjects M and C, and 4 sessions from the Maze task from subjects Je and Ni. We set the dimensionality of latent dynamics to  $d_z = 30$ , and used an embedding dimensionality of  $d_e = 2$ , for all embedding-conditioned dynamics models. For our approach, we constrain the hypernetwork to make rank  $d_r = 6$  changes, although we verified that the performance was not sensitive to  $d_r$  (Fig 18). As a proxy for how well the various approaches learned the underlying dynamics, we report metrics on inferring the hand velocity using reconstructed and forecasted neural data from the models. Note that we align all recordings to the movement onset (details in Appendix G).

![Figure 5: Visualizing the embedding manifold. (Left) A scatter plot of the inferred embedding distribution (e_i) for Centre-Out and Maze tasks across subjects M, Ni, and Je. (Right) A 2x2 grid showing condition-averaged latent dynamics (3D plots) and position (2D plots) for a Maze session (Sub Ni) and a CO session (Sub C), along with R-squared values (0.73 and 0.82).](27b22513fc27a0ff5f230b062ad3112f_img.jpg)

Figure 5: Visualizing the embedding manifold. (Left) A scatter plot of the inferred embedding distribution (e\_i) for Centre-Out and Maze tasks across subjects M, Ni, and Je. (Right) A 2x2 grid showing condition-averaged latent dynamics (3D plots) and position (2D plots) for a Maze session (Sub Ni) and a CO session (Sub C), along with R-squared values (0.73 and 0.82).

Figure 5: Visualizing the embedding manifold. (Left) Each point corresponds to a sample from the inferred embedding distribution (see eq. 16) corresponding to each recording. (Right) The condition-averaged latent dynamics for a session from Maze (Sub Ni) (Top) and a CO Session (Bottom) generated by the model, along with the corresponding real and forecasted behavior.

The inferred dynamical embedding displayed distinct structures across behavioral tasks and subjects (Fig. 5, Left). While the CO task involves more stereotyped straight reaching behavior with the same stimulus conditions across datasets, the Maze task has more complex stimulus statistics which vary across sessions. The family of learned dynamics reflected this heterogeneity across recordings. We visualize these learned dynamical systems for two example sessions, one from each task, in Fig 5 (Right). Specifically, we used the trained encoders,  $q_\beta$  and  $q_\alpha$ , to estimate the latent state and embedding at the beginning of movement onset. We subsequently generate the latent dynamics from that state using  $f_{\theta, e^i}$  till the end of the movement onset. The condition-averaged principal components (PCs) of these generated latents are shown in the figure.

{8}------------------------------------------------

We observed that most of the approaches had adequate performance on reconstructing velocity from neural recordings, with our method and Linear-Adapter outperforming single session reconstruction performance on the CO task (Fig. 6A, top). Multi-Session CEBRA was not able to adequately capture the variability in the Maze sessions and had low reconstruction  $r^2$ . In terms of forecasting, the single-session model trained using the seqVAE framework had the best performance. Notably, our approach managed to balance learning both the CO and Maze tasks relative to other multi-session baselines, with all performing better on the CO task than the Maze (Fig. 6A, bottom). The generative model learned from CEBRA had poor forecasting performance which resulted in a negative  $r^2$  value (not plotted). Next, we tested if we can transfer these learned dynamics to new recordings as we varied  $n_s$  from 8 to 64 trials for learning the read-in network and likelihood. We used trials from 2 held-out sessions from Sub C and M, as well as 2 sessions from a new subject (Sub T) for evaluating all methods. We observed that our approach consistently performed well on both reconstruction and forecasting for held-out sessions from previously seen subjects, and reached good performance on sessions from Sub T as we increased the training trials (Fig. 6B, C ( $n_s = 32$ )). Moreover, our method outperformed all other baselines on forecasting, especially in very low-sample regimes, while having comparable reconstruction performance (Fig. 19).

![Figure 6: Performance and trajectories for hand velocity decoding. Panel A shows reconstruction and prediction r^2 for Maze and CO tasks across different methods. Panel B shows performance on held-out sessions and a new subject as a function of training samples. Panel C shows hand velocity trajectories for a held-out session and a new subject.](91be14371a97fb5ce9eeb29ae18d07c3_img.jpg)

Figure 6 consists of three panels: A, B, and C.

- Panel A:** Two bar charts comparing different methods. The top chart shows 'Reconstruction'  $r^2$  for 'Maze' (green) and 'CO' (orange) tasks. The bottom chart shows 'Prediction'  $r^2$  for the same tasks. Methods compared: Our Method, Linear Adapter, Single Session, Single Session Dyn, LFP+DN, and M5C-Cellm.
- Panel B:** Two line graphs showing 'Few Shot Performance' as a function of 'Training Samples ( $n_s$ )' (8, 16, 32, 64). The top graph shows reconstruction performance, and the bottom graph shows forecasting performance. Data is shown for 'Held-out session' (purple) and 'New subject' (blue).
- Panel C:** Four line plots showing 'Hand Velocity Trajectories' (400 ms after movement onset) for 'True' (black), 'Reconstructed' (red), and 'Predicted' (blue) data. The top two plots are for a 'Held-out Session Sub M', and the bottom two are for a 'Session from Sub T'. All plots are based on  $n_s = 32$  training samples.

Figure 6: Performance and trajectories for hand velocity decoding. Panel A shows reconstruction and prediction r^2 for Maze and CO tasks across different methods. Panel B shows performance on held-out sessions and a new subject as a function of training samples. Panel C shows hand velocity trajectories for a held-out session and a new subject.

Figure 6: **A.** (top)  $r^2$  for hand velocity decoding from reconstructed and (bottom) forecasted neural observations for Maze and Centre-Out sessions. **B.** Behavior reconstruction (top) and forecasting (bottom) performance on held-out sessions and sessions from a new subject as a function of the number of training samples. **C.** Hand velocity trajectories (400 ms after movement onset) predicted by our approach on 17 test trials from held-out session (top) and 13 test trials from a session on a new subject (bottom), after using  $n_s = 32$  trials for aligning to the pre-trained model.

Next, we evaluated the impact of the inference framework on effective learning and few-shot performance. We specifically tested single session models as well as our proposed generative model trained after performing inference using VSMC and DVBF (Details in Appendix D). In both cases, we observed that the inferred embedding distribution learned the underlying dynamical structure across datasets (Fig. 12A). Moreover, we were able to similarly exploit this learned structure for few-shot forecasting on novel recording sessions (Fig. 12B). We additionally investigated the effect of large-scale training for sample-efficient transfer on downstream tasks by only pretraining the model on 128 trials from 4 sessions spanning different tasks and subjects. Even in this case, the embedding distribution displayed clear clustering based on the task and subject. Moreover, the model performed comparably to the Single-Session model on reconstruction, while outperforming it on prediction for both tasks (Fig. 20 A, B). However, it demonstrated poor performance on new sessions given limited trials for learning the read-in and likelihood parameters (Fig. 20 C), underscoring the importance of large-scale training for generalizing to novel settings.

Finally, we probed the differences in the latent state evolution given the same initial condition while interpolating across the learned embedding. In order to do this, we chose an example session from the Maze and CO datasets and obtained their corresponding dynamical embedding from the model, shown as a solid blue and green circle in Fig. 7 (middle), respectively. A grid of points was sampled around each of these inferred embeddings (shown as shaded squares in Fig. 7 middle), and for each point we obtained the corresponding low-rank parameter changes to generate the latent trajectories. We observed that the embedding space learned a continuous representation of dynamics, which was reflected in similar predicted behaviors close to the original learned embedding (Fig 7). Interestingly,

{9}------------------------------------------------

![Figure 7: Interpolating across the embedding space. The figure shows a central plot of the embedding space with axes e1 and e2. A blue point represents the original embedding for a Maze (Sub Je) session, and a green point represents the original embedding for a CO (Sub C) session. Surrounding the plot are two 3x3 grids of small diagrams showing predicted behavior trajectories. The left grid is labeled 'Predicted behavior (Maze)' and the right grid is labeled 'Predicted behavior (CO)'. The trajectories show varying degrees of curvature and complexity, reflecting the interpolation between the two original embeddings.](4e0ade2f41b66d5602160da5cc978274_img.jpg)

Figure 7: Interpolating across the embedding space. The figure shows a central plot of the embedding space with axes e1 and e2. A blue point represents the original embedding for a Maze (Sub Je) session, and a green point represents the original embedding for a CO (Sub C) session. Surrounding the plot are two 3x3 grids of small diagrams showing predicted behavior trajectories. The left grid is labeled 'Predicted behavior (Maze)' and the right grid is labeled 'Predicted behavior (CO)'. The trajectories show varying degrees of curvature and complexity, reflecting the interpolation between the two original embeddings.

Figure 7: The predicted behavior for a Maze (Sub Je) session and CO (Sub C) session at 9 grid points around the original inferred embedding. The point closest to the original embedding is highlighted in blue and green respectively.

when we interpolated through the entire embedding space, the predicted behavior and corresponding dynamics continuously varied as well. Specifically, the predicted behavior and dynamics trajectories on the CO session demonstrated similarities over a large portion of the embedding space, with the trajectories shifting to more curved reaches further from the original embedding (Fig. 21). On the Maze task, the trajectories demonstrated more heterogeneity in responses, and decayed to a fixed point further away from the original embedding (Fig. 22).

## 6 DISCUSSION

We present a novel framework for jointly inferring and learning latent dynamics from heterogeneous neural recordings across sessions/subjects during related behavioral tasks. To the best of our knowledge, this is the first approach that facilitates learning a family of dynamical systems from heterogeneous recordings in a unified latent space, while providing a concise, interpretable manifold over dynamical systems. Our meta-learning approach mitigates the challenges of statistical inference from limited data, a common issue arising from the high flexibility of models used to approximate latent dynamics. Empirical evaluations demonstrate that the learned embedding manifold provides a useful inductive bias for learning from limited samples, with our proposed parametrization offering greater flexibility in capturing diverse dynamics while minimizing interference. We demonstrate that the few-shot performance of our proposed generative model is largely agnostic to the inference method. We observe that the generalization of our model depends on the amount of training data—when trained on smaller datasets, the model learns specialized solutions, whereas more data allows it to learn shared dynamical structures. This work enhances our capability to integrate, analyze, and interpret complex neural dynamics across diverse experimental conditions, broadening the scope of scientific inquiries possible in neuroscience.

## LIMITATIONS AND FUTURE WORK

Our current framework uses event aligned neural observations; in the future, it would be useful to incorporate task-related events, to broaden its applicability to complex, unstructured tasks. Further, the model’s generalization to novel settings depends on accurate embedding inference, a challenge noted in previous works that disentangle task inference and representation learning (Hummos et al., 2024). However, we observe consistent improvement in embedding inference with increase in the number of training samples from novel recordings. Our empirical observations demonstrate that using a hypernetwork improves the expressivity of the dynamical systems model relative to other parametrizations. It would be interesting to investigate the theoretical basis of this observation in the future. While our latent dynamics parametrization is expressive, it assumes shared structure across related tasks. Future work could extend the model to accommodate recordings without expected shared structures (for instance, by adding explicit modularity (Márton et al., 2021)). Investigating the performance of embedding-conditioned low-rank adaptation on RNN-based architectures presents another avenue for future research. Finally, the embedding manifold provides a map for interpolating across different dynamics. While we focus on rapid learning in this paper, our framework could have interesting applications for studying inter-subject variability, learning-induced changes in dynamics, or changes in dynamics across tasks in the future.

 Rest of paper (reference and Appendix) is removed.