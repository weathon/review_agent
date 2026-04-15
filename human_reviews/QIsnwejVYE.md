# Robust Latent Neural Operators for a Family of Systems with Sparse Observations

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 6, 5

## Abstract
Neural operator methods have achieved significant success in the efficient simulation and inverse problems of complex systems by learning a mapping between two infinite-dimensional Banach spaces. However, existing methods still exhibit room for optimization in terms of robustness and modeling accuracy. Specifically, existing methods are characterized by sensitivity to noise and a tendency to overlook the importance of sparse observations. Therefore, we propose a robust latent neural operator based on the variational autoencoder framework. In this method, an encoder based on recurrent neural networks effectively extracts sequential information and dynamical characteristics embedded in sparse observations. Subsequently, a neural operator in latent space and a decoder facilitate the modelling of the original system. Additionally, for certain higher-dimensional systems, opting for a lower-dimensional latent space can reduce task complexity while still maintaining satisfactory modeling performance. We conduct experiments across several representative systems, and the results validate that our method achieves superior modeling accuracy and enhanced robustness compared to the state of the art baseline approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a latent RNN-based neural operator designed to handle noise and sparse observations in solving partial differential equations (PDEs). The method is trained using a variational auto-encoder framework. While similar frameworks have been explored in literature on irregularly-sampled time series, this work proposes a novel encoder based on DeepONet to address the limitations in effectiveness and efficiency seen in existing approaches. The proposed method is validated on two 1D PDEs (the Diffusion-Reaction equation and the Kuramoto-Sivashinsky equation) and one 2D PDE (the Navier-Stokes equation).

### Strengths
This paper addresses a critical limitation of existing neural operators in handling noise and sparse data. The proposed method shows promising results in overcoming these challenges. The experimental setup and data preparation are clearly outlined, and the authors have also provided the implementation code in the supplementary materials.

### Weaknesses
The primary contributions of this paper are twofold: 1) the introduction of the latent RNN framework to address noise and sparse data in neural operators, and 2) the proposal of a new encoder based on DeepONet. However, several key issues remain to be addressed:

1. **The rationale behind the latent RNN framework in handling noise and sparse data is insufficiently explained**. I recommend that the authors conduct a more in-depth analysis of the differences between RNN-based methods and existing neural operators to better explain the performance gain of the proposed method. It would be helpful to provide additional insights to further justify the motivation behind the proposed method.

2. **While the DeepONet-based encoder introduces some modifications, its novelty appears limited**. The proposed framework (Equation 2) closely resembles that of [Rubanova et al., 2019] (Equations 5-7), with the primary distinction being the introduction of OPERATORsolve in place of ODESolve. This corresponds to the replacement of existing encoders (RNN-Decay and ODE-RNN) with OPERATOR-RNN. This DeepONet-based encoder is designed to enhance the effectiveness of RNN-Decay in handling irregularly sampled time series and to improve the computational efficiency of ODE-RNN. However, according to the ablation study, the performance improvement of the proposed encoder compared to RNN-Decay is not significant (Ab2 in Tables 2 and 3), especially considering the standard deviation. Additionally, there is no comparison of computational complexity with ODE-RNN to further substantiate the authors’ claims.

3. **The concept of neural operators utilizing latent space has been explored in previous works [Yin et al., 2022; Serrano et al., 2023]**. These studies employed an auto-decoder framework with implicit neural representations to learn the mapping between data space and latent space, as well as to model latent dynamics using Neural ODEs. While their primary focus is on spatially sparse data and temporal extrapolation, their methods appear applicable to the noise and temporally sparse scenarios considered in this paper. The contribution of this work could be further enhanced by comparing the proposed method with these approaches in both spatially sparse and temporally sparse settings.

I am willing to raise the score if the aforementioned issues are effectively addressed.

**Reference**

[Rubanova et al., 2019] "Latent ordinary differential equations for irregularly-sampled time series." NeurIPS 2019.

[Yin et al., 2022] "Continuous pde dynamics forecasting with implicit neural representations." ICLR 2022.

[Serrano et al., 2023] "Operator learning with neural fields: Tackling pdes on general geometries." NeurIPS 2023.

### Questions
1. The parenthetical and textual citations are incorrectly used throughout the paper. 

2. To provide a more comprehensive ablation study, consider replacing the OPERATOR-RNN encoder with ODE-RNN in an additional experiment. This configuration would be equivalent to Latent ODE [Rubanova et al., 2019]. While the authors have compared their model to Latent ODE in Table 2, the settings used in that comparison differ from those in the ablation study. By including this additional experiment, a direct comparison between OPERATOR-RNN, RNN-decay, and ODE-RNN can be conducted, offering a clearer understanding of their relative performance.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
In this paper, the authors focus on the challenges of robustness to noise and sparse observation in neural operators. To tackle these issues, they introduce a latent neural operator that builds upon a VAE framework, leveraging recurrent neural networks to extract information from sparse data. The authors employ a neural operator to infer dynamics in the latent space, followed by a decoder to reconstruct the original data. The extensive experiments conducted across various systems demonstrate the superiority of the proposed method compared to existing baselines.

### Strengths
1. The focus on robustness to noise and the capacity to handle sparse observations are critical considerations for neural operators, thus it is appreciated that this work focuses on these two important issues.
2. The proposed method, named RLNO, is somewhat new by extending the DeepONet within a VAE framework.

### Weaknesses
1. The inference for the posterior distribution of $z_0$ is somewhat unclear. Why do we need to model $ z_0 $ based on so many observations? Should it not be sufficient for $ z_0 $ to depend solely on $ s_0 $? If so, might it be possible to directly mitigate the issue of irregular sampling in the latent space using an operator solver?
2. The preservation of the functional properties of the proposed method needs demonstration. It is uncertain whether the overall framework retains its status as an operator after the incorporation of multiple non-local operators.
3. In Eq. (7), a more detailed description of $p_{\theta,\theta_{dec}}(s_0, s_1, \dots, s_T|z_0)$ would enhance comprehensibility. Additionally, I am confused by the use of $\theta$ and $\tilde{\theta}$, are they same?
4. The manuscript requires improvements in its written presentation. Some writing suggestions are given below.

### Questions
Several writing suggestions:
1. Line 129, if Fig.3 should be Fig. 1?
2. Line 236, repeated ``equation’’

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the Robust Latent Neural Operator (RLNO) that utilizes a recurrent neural network (RNN) encoder and the variational autoencoder (VAE) framework within a latent neural operator learning. The authors claim that the proposed model is more robust to the observation noise and learns the dynamics of a system more accurately.

### Strengths
This work introduces mainly two extensions of an existing work, Latent Neural ODE (LNODE) solver. First, it extends the LNODE solver to the operator learning framework. Second, the ODE-RNN encoder of LNODE is replaced with OPEARATOR-RNN in which the neural operator is used instead of an ODE solver. The idea is straightforward, and the proposed integration of each component looks natural and reasonable. The experiments show that the improvement of the proposed model is significant.

---

**After the discussion**, the contributions of the work were getting better recognizable; thus, I raised my score to 6 from 5.

### Weaknesses
The extensions rely on existing ideas (such as neural operator/DeepONet, latent space neural ODE, VAE/Evidence Lower Bound loss); thus, the manuscript needs more thorough revision for a clearer presentation of its technical novelty. For example, Algorithm 1 is almost identical to the work of [Rubanova et al. 2019]; thus, it would be worth highlighting the distinctive novelty.

The experiments were evaluated mostly for the "interpolation" task. The "extrapolation" evaluation should be further discussed as shown in the related work (e.g., LNODE).

### Questions
For OPERATORSolve, why are the two different neural operators (trained for different parameters $\theta$ and $\tilde{\theta}$) used? Does the proposed method train and use them independently for forward and backward evolution? If so, why?

The extrapolation evaluation should be further clarified. As described, the DR and KS experiments used the sparse observation data sampled from 100 steps of ground truth data and added noise to them. The predictions are evaluated and analyzed for 100 steps. What would the performance be if the prediction happens longer for, e.g., 200 steps and 400 steps? Alternatively, what if the sparse observation data is sampled only at the beginning of the ground truth data?

Minor questions should be clarified:
- L129: The text refers to Fig.3. Is this correct? Maybe Fig. 1?
- L194: Shouldn't it be "encoder" instead of "decoder"?
- L259: What is the RCDI method?
- L329: The text refers to Figure 3(c). Is this correct? Maybe Figure 3(b)?

Minor corrections are required:
- L57: typo "adetly"
- L99: missing space "structuresHao"
- L236: duplicated word "Equation equation 3"
- L309: typo "parse"

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
In this work, the authors introduce RLNO, a robust latent neural operator grounded in the VAE framework, to address the drawbacks of neural operator methods that lack robustness and accuracy due to their sensitivity to noise and inability to handle sparse data. RLNO comprises an RNN-based encoder to capture sequential and dynamical information from sparse observations, a latent space neural operator for modeling, and a decoder to reconstruct the original system. Experimental results across ODE and PDE systems verify RLNO's superiority in both accuracy and robustness.

### Strengths
The work presents a derivation of an evidence lower bound that transforms the optimization problem from maximizing observation probability to maximizing the bound, which provides a different approach to the optimization process. The study includes experimental evaluations across multiple test cases.

### Weaknesses
- The major limitation of this work lies in its insufficient justification for latent space dynamics inference, particularly since DeepONet already provides established methods for iterative dynamics inference. This design choice lacks clear comparative advantages over existing approaches.

- Though the encoder-decoder architecture is introduced to address sample complexity and sparse observations, the paper falls short in demonstrating how these components preserve DeepONet's fundamental functional properties. This gap raises concerns about the theoretical soundness of the proposed framework.

- The current manuscript structure and presentation hinder a clear understanding of the methodology and contributions.

### Questions
please see above.

### Soundness
2

### Presentation
2

### Contribution
2
