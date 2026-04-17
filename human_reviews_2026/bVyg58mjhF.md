# Robust Latent Neural Operators through Augmented Sparse Observation Encoding

- Decision: Reject
- Scores: 8, 4, 2, 6

## Abstract
Neural operator methods have achieved significant success in the efficient simulation and inverse problems of complex systems by learning a mapping between two infinite-dimensional Banach spaces. However, existing methods still exhibit room for optimization in terms of robustness and modeling accuracy. Specifically, existing methods are characterized by sensitivity to noise and a tendency to overlook the importance of multiple sparse observations in new domains. Therefore, we propose a robust latent neural operator based on the variational autoencoder framework. In this method, an encoder utilizing recurrent neural networks effectively captures sequential patterns and dynamical features from domain-specific sparse observations. Subsequently, a neural operator in latent space, followed by a decoder, enables the effective modeling of the original system. Additionally, for certain higher-dimensional complex systems, opting for a lower-dimensional latent space can reduce task complexity while still maintaining satisfactory modeling performance. We evaluate our approach on multiple representative systems, and experimental results demonstrate that it achieves superior modeling accuracy and enhanced robustness compared to state-of-the-art baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors propose a model called RLNO with VAE framework for operator learning. The proposed model has a RNN-based encoder, neura operator in the latent space, and decoder. This framework enables us to deal with noisy and sparse input data with low computational costs.

### Strengths
The proposed model is noise robust, computationally efficient, and can deal with temporal time interval inputs. They empirically show the performance of the proposed model with various situations.

### Weaknesses
The presentation should be improved. For example, please define $b_k$ and $c_k$ in Eq. (2) and what is $x_1$ and $x_2$ in Eq. (10)? Also, in line 289, equation  (Eq. equation 8) shoud be equation (8).

### Questions
- In Section 3.2, the authors insist that an advantage of the proposed model over the ODE-RNN method is the computational cost. Comparing the computational time and the MSE of the proposed model and the ODE-RNN model would be interesting.
- The authors insist that the proposed model accepts time series with nonconstant time intervals. Does the performance changes between the constant and nonconstant time interval cases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes the Robust Latent Neural Operator (RLNO), a variational autoencoder–like latent neural operator framework. The authors introduce an encoder, termed the OPERATOR-RNN, which takes as input a short window of possibly irregularly sampled time series data and outputs a Gaussian posterior over a finite-dimensional latent vector  $𝑧_0$. A latent DeepONet is then used as the evolution map in the latent space, generating a trajectory that is decoded back to the observed state space. The training objective maximizes an ELBO loss over $𝑧_0$. The authors perform ablations  are consider where the encoder is substituted with baselines, and vary the encoder window and the latent vector size. The authors consider a series of experiments on simple PDEs and compare against baselines with favorable results.

### Strengths
- The encoder architecture is a reasonable engineering upgrade to ODE/RNN/Decay approaches as it targets irregular sampling specifically.
- The ablations are useful as they show how choices affect the prediction capabilities of the model. Also, they consider noisy signals.
- The methods seems to be more accurate than the competition in the chosen metric.
- Moreover, the encoder architecture represents a reasonable engineering refinement over existing ODE-, RNN-, or Decay-based approaches, as it is specifically designed to handle irregularly sampled data. However, methods for tackling non-equidistant sampling have already been explored in the literature, including graph-based and recurrent–convolutional hybrids such as the GNN-tCNN and LSTM-tCNN architectures, which also operate on irregularly spaced observations.

### Weaknesses
- The main novelty of the paper is the encoder engineering improvement. Different latent flow approaches such that Learning Effective Dynamics (see “Multiscale simulations of complex systems by learning their effective dynamics”), Latent ODEs (Learning the intrinsic dynamics of spatio-temporal processes through Latent Dynamics Networks), and generative diffusion models (see Generative learning for forecasting the dynamics of high-dimensional complex systems) which the authors do not compare against. What the authors propose is in a way a subset of these methods.
- DeepONet is only effective when the outputs of the neural operator lie in a linear span of trunk features, which is not the case for complex physical systems. I would suggest the author to consider the approach described in Non linear Manifold Decoders for Operator Learning.
- The neural operator terminology is a bit loose and misused because the DeepONet, as used here is a map between finite dimensional vectors, not functions spaces.
- I believe that the sign for the 2D PDE is flipped, but perhaps this is a wrong interpretation of the derivation.

### Questions
- How is $p(s_i | g(z_i))$ defined?
- Can you consider non-GRF inputs, such as piece-wise, to test robustness beyond GRFs?
- How does your method compare to the baselines described above?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a method for learning operators between function spaces focusing on robustness to noise and ability to adapt to sparsely sampled observations. The method is named RLNO (Robust Latent Neural Operator), is designed to overcome the  degradation of that standard neural operator methods (e.g. DeepONet, FNO) when inputs are irregularly sampled or corrupted by noise.
To that end, RLNO introduces a variational autoencoder (VAE) structure around the neural operator: (1) an RNN-based encoder captures temporal and dynamical patterns from sequential or irregularly spaced observations, (2) a neural operator in latent space models the mapping between functional inputs and outputs, and (3) a decoder reconstructs the target functions in the original space.
By using a low-dimensional latent space, RLNO can tackle high-dimensional dynamical systems. Experiments on several benchmark PDE and dynamical datasets show improved accuracy and stability under noise compared to DeepONet, FNO, and related baselines.

### Strengths
- Aim: Paper tackles important practical problem, the robustness of NO methods to spareness of samples and noise of observations, and proposes a promising approach.
- Novelty: While each component (RNN-based temporal encoding, VAE-style probabilistic latent representation, and NO mappings) have been individually studied, RLNO’s novelty lies in integrating these three elements.
- Experiments: Baselines are appropriate and the performed empirical study shows the how RLNO overcomes the degradation of the competitors

### Weaknesses
First, unfortunately, I find that the paper is **not well written**. The presentation of the work is lacking on important aspects: 
- Overly strong claims:  Several claims need more nuance, and some are debatable in literature,  just to name two - line 053 _"superior computational efficiency without sacrificing accuracy"_, line 268 _"thereby demonstrating the necessity and superiority of our framework design"_.
- Flow, intuition and technical complexity: Paper overly focuses to high-level presentation, while technical aspects are often introduce in an abrupt manner.  This makes the content hard to parse and key contributions hard to identify.  
- Notation: For my personal taste, notations are not ideal. Further, there is inconsistencies in Eq (1) line 145 and Algorithms 2 and 3 lines 720 and 748 of the Appendix

Second, there is lack of theoretical guarantees.. claims are made citing other papers and generalising the reasoning, but not formally backed. The only try to make theoretical claim is in Theorem 1 in Appendix A.3. The claim is sloppy/not formally correct (e.g. confusing typos in lines 762, 768, domain of f is not consistent with Eq. (S6) ) and the proof is not provided but as many other thingd in the paper just briefly hinged.

Finally, since the core contribution is methodology, I find that the empirical study of two PDE problems (one 1D and other  2D) is a bit underwhelming w.r.t. publication quality requirements.

### Questions
NA

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The work builds upon the development of neural operators, which has been a highly cited and used method for learning maps between measurements to full state estimates. Or more broadly mapping functions to functions.

The key idea in this paper is to do the neural operator mapping in latent space.  That is the basic innovation of the paper, and the authors show that his is a much more way to learn the operator than in the original measurement space.  This is consistent across many emerging real-world examples:  work in the latent space instead of the measurement space in order to improve performance.

### Strengths
This is a very solid and useful contribution to the field of neural operators.   A clear and important in the next step of neural operators as exploiting the latent space is clearly what should be done.  The results back this up.

### Weaknesses
Not many examples were presented, and some are not very convincing (equation 11 is linear PDE is it not?).  I think something like 2D Kolmogorov flow in the turbulent regime would be a much more convincing set of data than what they have.  So I found the examples not to the level of where they should be.

### Questions
It seems the "sparsity" has not been well characterized in how this works?  Do the authors have a metric for this?  It is important to establish when the sampling will actually work or not.

How long can the roll out in latent space go?  Most latent space long-time roll outs eventually diverge or break.  Is there any guarantee about a stable long-term roll out?

### Soundness
3

### Presentation
3

### Contribution
3
