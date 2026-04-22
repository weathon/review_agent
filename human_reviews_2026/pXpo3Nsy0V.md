# Learning Treatment Representations for Downstream Instrumental Variable Regression

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
Traditional instrumental variable (IV) estimators face a fundamental constraint: they can only accommodate as many endogenous treatment variables as available instruments. This limitation becomes particularly challenging in settings where the treatment is presented in a high-dimensional and unstructured manner (e.g. descriptions of patient treatment pathways in a hospital). In such settings, researchers typically resort to applying unsupervised dimension reduction techniques to learn a low-dimensional treatment representation prior to implementing IV regression analysis. We show that such methods can suffer from substantial omitted treatment bias, violating exclusion restriction principle, due to implicit regularization in the representation learning step. We propose a novel approach to construct treatment representations by explicitly incorporating instrumental variables during the representation learning process. Our approach provides a framework for handling high-dimensional endogenous variables with limited instruments. We demonstrate both theoretically and empirically that fitting IV models on these instrument-guided representations ensures identification of directions that optimize outcome prediction. Our experiments show that our proposed methodology improves upon the conventional two-stage approaches that perform dimension reduction without incorporating instrument information.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the inability of IV models to handle high-dimensional treatments when instruments are scarce. The authors identify that existing two-step approaches that apply unsupervised dimension reduction (e.g., PCA) before IV regression violate the exclusion restriction and introduce omitted variable bias. To resolve this, they propose Instrument-Guided Representation Learning (IGRL), which incorporates instrumental variables directly into the representation learning process to ensure learned features capture only Z-driven variation in X. The approach provides both theoretical guarantees that the learned representations identify outcome-prediction-optimizing directions and empirical evidence of improvements over conventional two-stage methods.

### Strengths
- Learning representations of high-dimensional treatments is a difficult but relevant problem in many disciplines, e.g., when the treatment is a text
- The proposed method comes with theoretical improvement guarantees and promising empirical results
- The idea seems novel and rigorously executed

### Weaknesses
- The current version of the method does not allow for incorporating observed covariates, which could strengthen improvements in practice. Is this a straightforward extension or would this require more elaborate theory?
- As is common for IV methods, the proposed method depends on assumptions that may be difficult to verify in practice
- The specific setting (instruments + high-dimensional treatment) seems somewhat niche. A few clear motivating examples might strengthen the story of the paper.

### Questions
- I have one general question regarding the representation learning approach: Usually, we lose identifiability of the causal target quantity when using a treatment representation due to violation of the consistency assumption (no hidden variability within the treatment). Could you elaborate on how exactly your method addresses this problem? The results seem reasonable as they only provide improvement guarantees and no identifiable treatment effect.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method for representation learning of high-dimensional treatments that allows for a valid downstream instrumental variable treatment effect estimation setting. To do so, the paper proposes to incorporate IVs in the representation learning step. The procedure can be interpreted as a regularization of the unsupervised representation learning step to eliminate spurious backdoor paths. The paper compares the presented method with common statistical dimension reduction techniques, such as PCA, and ML methods, such as VAEs.

### Strengths
- The problem setting is interesting and novel to the best of my knowledge

### Weaknesses
- Line 67: The paper states that causal representation learning is uncommon in causal inference, but common in causal discovery. This is not true.
The paper thus lacks a sufficient discussion of related work on representation learning in causal inference tasks. 
- The method requires very strong untestable assumptions, limiting its applicability in practice.
- The paper neither contains a discussion on the method, its limitations, or the results, nor a conclusion, leaving it an unfinished work. Potentially, these important parts of a research paper are neglected because of the page limit. However, this cannot be seen as an excuse.
- Mathematical statements are neither proven in the main paper nor are proofs referenced.
- The paper is lacking a proper motivation with precise use case examples. The problem statement seems a bit far-fetched and unrealistic. For example, what exactly are the available instruments in line 39? In my opinion, it is rather rare in practice to have suitable instruments
- Many references are incomplete or not stated correctly. This is not in line with proper scientific work.
- Many language/grammar errors. Sentences are incomplete, making it difficult for the reader to follow the line of thought.

### Questions
- Line 95: The data does not include any observed confounders. Why is this reasonable? If we assume all confounders are unobserved, we introduce a lot of  uncertainty in the modelling
- Why exactly do naïve representation learning approaches introduce omitted variable bias? This needs to be explained further.
- Assumption 2.1/ structural equation for X: This assumption is very strong. Why would such an encoding exist? How can we validate it? 
- Experiments: Why is the proposed method not compared against other treatment representation learning methods? Even if they are not designed for IV 
analysis, the comparison is of high value to assess the usefulness of the proposed method. A comparison with PCA is insufficient.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses a fundamental limitation in instrumental variable (IV) regression when dealing with high-dimensional treatments: the violation of the exclusion restriction due to unsupervised dimensionality reduction. The authors propose Instrument-Guided Representation Learning (IGRL), a novel framework that integrates instruments directly into the representation learning process to ensure the latent treatment representation captures all instrument-driven variation. Through both linear (LIRR) and non-linear (IRAE) implementations, supported by theoretical guarantees and extensive experiments, the method demonstrates superior performance over conventional approaches in identifying outcome-improving interventions, effectively mitigating the omitted variable bias that plagues standard two-stage procedures.

### Strengths
The authors pinpoint how standard unsupervised dimensionality reduction of high dimensional treatments can violate the exclusion restriction by discarding instrument driven variation, a problem they term omitted treatment bias. The proposed solution of instrument guided representation learning represents a creative fusion of causal inference and representation learning, directly addressing this limitation in prior two stage approaches.

The authors develop a complete framework with specialized methods for both linear and nonlinear settings, supported by strong theoretical guarantees on the identifiability of valid intervention directions.  The writing is remarkably clear, with logical progression from linear to nonlinear cases.

### Weaknesses
The theoretical foundation of this paper relies on a set of strong structural assumptions that may be difficult to satisfy in real world applications. A key example is the core assumption of joint independence. This assumption requires the instrument, the confounder representation, and the orthogonal components to be fully independent. This is particularly challenging to guarantee with high dimensional and complex data. Furthermore, prerequisites such as the invertibility of the encoding and decoding functions and specific linear relationships among latent variables significantly limit the direct applicability of the theoretical model to practical scenarios. While these assumptions provide a necessary foundation for theoretical derivation, the paper does not sufficiently discuss the robustness of the estimates when these conditions are violated. This omission creates uncertainty for the method's practical deployment.

The paper exhibits notable shortcomings in reproducibility. This is particularly problematic for the non linear IRAE model. This model involves tuning regularization weights for up to six hyperparameters. The strategy for setting these critical parameters and their sensitivity to model performance are not elaborated. I encourage author provide anonymous github link.

### Questions
Please see my concerns in weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper considers an instrument variable setting with a potentially higher dimensional confounded treatment. The paper argues that naively adapted methods for such a setting, adopting dimensionality reduction on the treatment first and then performing IV regression, may introduce omitted variable bias. To address this challenge, the authors propose a framework to incorporate the instrument information during the dimensionality reduction/representation learning of the treatment to ensure improvement guarantees on soft interventions in this setting. The authors derive a theory and apply their method to both linear and non-linear setups, providing experimental results on (semi-)synthetic datasets.

### Strengths
- The IV setting with potentially high dimensional confounded treatments is an important and underexplored research direction in causal inference research.
- The motivation showing that omitted variable bias can also come from dimensionality reduction/representation learning of treatments (and not only of cofounders like most previous work showed) is an interesting and important finding
- Their method, ensuring that the IV information is maintained in the treatment representation, is a nice and intuitive idea and the theoretical analysis in linear and non-linear setting helps to motivate this choice.

### Weaknesses
The clarity of the paper could be improved and implications of different parts of the method could be mentioned.
- Part of the main motivation is that IV estimators “can only accommodate as many endogenous treatment variables as available instruments”. I think while in general it makes sense that effect estimation with high dimensional treatments is challenging, this statement should be explained more and shown in more detail.
- I think the related work is a bit limited to some specific works and could be extended to better show the contribution of their method with respect to existing works, in particular around representation learning for high dimensional confounders and OMV bias (e.g., https://www.pnas.org/doi/abs/10.1073/pnas.2427298122, https://arxiv.org/abs/2311.11321, https://arxiv.org/abs/2502.04274 ) , highdimensional treatments (e.g., https://arxiv.org/abs/2009.14061, https://arxiv.org/abs/2106.01939, https://arxiv.org/abs/2301.12292 ) , and representation learning for IVs (e.g., https://arxiv.org/abs/1612.09596 ).
- Assumptions 2.1 and 2.2, i.e. invertibility of encodings and existence and full rank of A, are specific to their novel setting and method. Thus, the implications of these assumptions should be discussed in more detail. How realistic are these in practice and how would minor violations of these assumptions affect the framework (theoretically/some intuition and maybe even provide some ablation study on simulated data)?
- The motivation of choosing soft interventions only and not deterministic ones is not fully clear to me. From a practical perspective, I think this would assume that during inference time we first can sample from the observational policy which might limit the practicality in real-world applications?

Presentation could be improved, especially in the experiments section
- The baselines should be introduced and described properly, also there is currently a mix between linear and non-linear setting (baselines are already referenced first and then described later).
- In the tables, it would be better to display their method in the last rows and not in the middle to better show their method.
- Table 3 over sample size could also be a plot to better compare the methods with sample size as x axis
- Intuition on why the different IRAE[x] variants perform differently and and some discussion on in which general settings the respective regularization terms are helpful and maybe some guideline when to use which variant could be helpful.
- real world applicability not clear, no motivation for real world application or any experiments to give some intuition provided (validation not possible but some insights could be shown in real world data)
- No discussion/conclusion section is provided. I think a short summary and mentioning of potential limitations would strengthen the paper.
- Presentation and fontsize of the plots needs to be improved clearly. Also, Figure 3 is hard to interpret and understand in general.

Overall, I think when my main concerns regarding unclarities are answered, I think this is a nice and interesting paper from the content side and I am open to increase my score. However, I think especially regarding the presentation (Figures, providing discussion, etc.), substantive changes are required for publication.

### Questions
- How realistic are Assumptions 2.1 and 2.2 in practice and how would minor violations of these assumptions affect the framework (theoretically/some intuition and maybe even provide some ablation study on simulated data)?
- Could the authors clarify why in ll. 1.66 they consider soft interventions and not deterministic ones and how they define them?
- Could the authors elaborate more on when it is useful/necessary to learn representations of V and when learning only D is sufficient?

### Soundness
3

### Presentation
1

### Contribution
3
