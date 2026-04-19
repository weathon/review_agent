# Cross-Modal Alignment via Variational Copula Modelling

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 8, 6, 6

## Abstract
Various data modalities are common in real-world applications. In healthcare, for example, electronic health records, medical images, and clinical notes provide comprehensive information for diagnosis and treatment.
    Thus, it is essential to develop multimodal learning methods that aggregate information from multiple modalities to generate meaningful representations for downstream tasks.
    The key challenge here is how to appropriately align the representations of the respective modalities and fuse them into a joint distribution.
    Existing methods mainly focus on fusing the representations via concatenation or the Kronecker product, which oversimplifies the interaction structure between modalities, prompting the need to model more complex interactions.
    Moreover, the notion of joint distribution of the latent representation that incorporates higher-order interactions between modalities is also underexplored.
    Copula is a powerful statistical structure in modelling the interactions between variables, as it bridges the joint distribution and marginal distributions of multiple variables.
    In this paper, we propose a novel copula modelling-driven multimodal learning framework, which focuses on learning the joint distribution of various modalities to capture the complex interaction among them.
    The key idea is interpreting the copula model as a tool to align the marginal distributions of the modalities efficiently. 
    By assuming a Gaussian mixture distribution for each modality and a copula model on the joint distribution, our model can also generate accurate representations for missing modalities.
    Extensive experiments on public MIMIC datasets demonstrate the superior performance of our model over other competitors.
    Ablation studies also validate the effectiveness of the copula alignment strategy and the robustness of our model over different choices of the copula family. 
    Code is anonymously available at https://anonymous.4open.science/r/CM2-C1FD/README.md.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
In this paper, they propose a copula-driven multimodal learning framework that captures complex interactions by learning the joint distribution of multiple modalities. By aligning marginal distributions with a copula model and assuming a Gaussian mixture distribution for each modality, their approach effectively generates accurate representations for missing modalities. Extensive experiments on public MIMIC datasets demonstrate the model's superior performance over competitors. Ablation studies validate the effectiveness of the copula alignment strategy and the model's robustness across different copula families.

### Strengths
There are three major contributions. 

First, they propose a copula-driven multimodal learning framework that captures complex interactions by learning the joint distribution of multiple modalities. 

Second, they approximate   marginal distributions by assuming a Gaussian mixture distribution for each modality, generating accurate representations for missing modalities.

Third, extensive experiments on public MIMIC datasets demonstrate the model's superior performance over competitors.

### Weaknesses
There are several major weaknesses in the paper:

\begin{itemize}
    \item \textbf{First}, if the goal is to integrate multiple modalities to improve prediction, accurately modeling the joint distribution of these modalities may not be necessary.
    
    \item \textbf{Second}, it is unclear how a Gaussian mixture distribution for each modality generates accurate representations for missing modalities, especially since the authors have not modeled the missing data mechanism.
    
    \item \textbf{Third}, there are numerous typos and missing details throughout the paper that require correction. For instance, what is \( c_M \) on line 160? What does \( q(\theta) \) represent on line 224? Additionally, the connection between \( f_\theta(x) \) and the copula distribution is not clarified. These omissions make it difficult for readers to understand the technical aspects of the paper.
\end{itemize}

### Questions
\begin{itemize}
    \item \textbf{First}, if the goal is to integrate multiple modalities to improve prediction, accurately modeling the joint distribution of these modalities may not be necessary.
    
    \item \textbf{Second}, it is unclear how a Gaussian mixture distribution for each modality generates accurate representations for missing modalities, especially since the authors have not modeled the missing data mechanism.
    
    \item \textbf{Third}, there are numerous typos and missing details throughout the paper that require correction. For instance, what is \( c_M \) on line 160? What does \( q(\theta) \) represent on line 224? Additionally, the connection between \( f_\theta(x) \) and the copula distribution is not clarified. These omissions make it difficult for readers to understand the technical aspects of the paper.
\end{itemize}

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a multimodal framework by aligning the modality embedding with copula following a VAE-like structure. Overall I think this paper is a solid paper. The study is comprehensive with detailed equations and experiment setup. The provided code is clean and includes all details.

### Strengths
This paper has several strengths:
1: The equations and theorems are clear and easy to follow. 
2: The ablation study is extensive. 
3: The provided code is clean and wel-organized. Although I haven't run it, it is easy to follow and results seem reproducible.

### Weaknesses
I have several questions after reading the experiments:
1: As part of the claimed contribution in this paper is to tackle the missingness, I think a common scenario in healthcare should be discussed - when certain variables or chunks of time series are missing. How can copula deal with the partially observed EHR tabular data, or time series? 
2: I think the baselines are not comprehensive. There are many multimodal works in the past year, but there were only 5 baselines and 4 of them were published more than two years ago. In particular, I think the lack of multimodal LLM and/or how copula can be applied in multimodal LLM would be very important.

### Questions
The authors did not evaluate the choice of backbone encoders, which I think should be a key lever in the experiment results. Can authors share more insights on the candidate backbone encoders and how much they differed in performance?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposed a copula-based model to model interactions among modalities. Gaussian mixture model was firstly employed to learn the marginal distribution of each component. After that, the parameters of the copula-based model, and of the classifier are jointly optimized via variation inference. Experiments are conducted on MIMIC dataset.

### Strengths
This paper is well organized and written. The idea of applying Copula model to model interactions is interesting.

### Weaknesses
My main concerns are about the Copula model. First, according to theorem 1, once the marginal distribution of each component F_i(z) is provided (we first assume we can learn it well), there exits a copula C to recover the joint distribution with C(F1(z1),F2(z2),...,Fm(zm)). That means, we should parameterize q(z1,...,zm) as C(F1(z1),...,Fm(zm)) or C(z1,...,zm)\pi_{m=1}^M Fm(zm). But in your variational family, the form of q(z) seems inconsistent with the above forms. Second, why do we assume GMM to fit the marginal distribution? Have you performed goodness of test to show that latent z are mixture of Gaussian? Empirically, how do we choose K in the GMM? 

The experimental result is not convincing enough, since the proposed method does not significantly outperform others. Specifically, the CI of these methods are very overlapped, making it hard to demonstrate the improvement. 

Moreover, although this idea is interesting, this paper seems a simple application of the copula model. From my perspective, the novelty does not reach the acceptance bar of ICLR.

### Questions
Where is the definition of \theta? From my current understanding, it refers to the parameter of classifiers. But according to line 249, it also includes parameters in the embedding (encoder). If that is the case, then the encoded latent variables will change after each iteration, then is that mean we should apply GMM to learn \mu and \Sigma in each iteration?

### Soundness
3

### Presentation
3

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
This paper addresses the missing modality problem in multimodal machine learning for healthcare using the statistical copula model. The copula model is powerful in modeling the interactions between random variables by directly describing the dependency structures between them. This paper assumes a Gaussian mixture distribution for the marginals (each modality) and uses the copula for the joint distribution. Given the probabilistic nature of the model, when a modality is missing, sampling can be performed. Therefore, the problem of missing modality and modeling of cross-modal interaction are jointly tackled. Empirical evaluations show promising performance in comparison with existing approaches.

### Strengths
- The paper is well-written and easy to follow. I find it quite pleasant to read the paper.
- The proposed method is technically sound. Using a copula to capture the joint distribution is a simple yet effective way of modeling the cross-modal interactions.
- Empirical evaluation shows good performance.

### Weaknesses
- The theoretical analysis is not in-depth. Section 4.5 presents an existing theorem to justify the uniqueness of the joint distribution, but its implication to the multimodal learning problem is not analyzed clearly and in-depth.
- The marginals are assumed to be mixtures of Gaussian. This assumption seems to lack justification and no alternatives to GMM is discussed.

### Questions
- How to interpret Fig. 2? Why is the Gumbel copula more focused on the positive dependence between the modalities while the Gaussian copula has less weight on modeling tail dependences?
- In Table 5, the effect of different copula families does not affect the performance for the partially matched dataset, what are the reasons behind?
- In Table 5, for a matched subset, copula families make quite a difference in the performance. How should the users choose the correct copula family to use for their own dataset?

### Soundness
3

### Presentation
4

### Contribution
3
