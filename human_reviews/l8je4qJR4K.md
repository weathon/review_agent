# Domain Generalization via Content Factors Isolation: A Two-level Latent Variable Modeling Approach

- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
The purpose of domain generalization is to develop models that exhibit a higher degree of generality, meaning they perform better when evaluated on data coming from previously unseen distributions. Models obtained via traditional methods often cannot distinguish between label-specific and domain-related features in the latent space. To confront this difficulty, we propose formulating a novel data generation process using a latent variable model and postulating a partition of the latent space into content and style parts while allowing for statistical dependency to exist between them. In this model, the distribution of content factors associated with observations belonging to the same class depends on only the label corresponding to that class. In contrast, the distribution of style factors has an additional dependency on the domain variable. We derive constraints that suffice to recover the collection of content factors block-wise and the collection of style factors component-wise while guaranteeing the isolation of content factors. This allows us to produce a stable predictor solely relying on the latent content factors. Building upon these theoretical insights, we propose a practical and efficient algorithm for determining the latent variables under the variational auto-encoder framework. Our simulations with dependent latent variables produce results consistent with our theory, and real-world experiments show that our method outperforms the competitors.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes to learn a two-level, hierarchical latent space with each layer partitioned into the content and style group. The content group controls the data-invariant features, while the style group controls the data style factors. Such a model aims to address the difficulty in domain generalization that label-specific and domain-related features are not well distinguished.

### Strengths
1. The paper is in general well-written and well-presented with the illustration figures.
2. The idea is motivated to address the practical issue in an active research field.
3. The author conducts various experiment settings, including toy data synthesis and ablation studies.

### Weaknesses
1. I don't have much experience in this particular research field, so based on my understanding, the main purpose of the paper is to learn a well-defined and smooth latent space that can distinguish the domain features and style features; therefore, the model can perform well when the underlying distribution shift happens. The two-level latent space seems to be related to the hierarchical VAEs, where multi-layer latent variables are used to learn different levels of data features. So, how does such a two-level latent space compare or connect to the hierarchical VAEs?

2. I understand learning a separated latent space for different data features can be beneficial to learning a smoother model manifold. But how does this model improve the performance of DG? The author mentioned that "The key to achieving DG based on our model is recovering the distribution of content factors and isolating them from style factors." An additional and detailed explanation would be good.

### Questions
Please see the Weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper aims to recover the label-specific content factors and isolate them from the style ones utilizing a two-level latent space, and consequently learn a stable predictor applicable to all domains. Specifically, they propose a novel data generation process and then exploit a VAE-based framework to achieve the latent variable identifiability based on some assumptions. Theoretical analysis and various experimental results demonstrate the effectiveness of this method.

### Strengths
1.This work proposes a two-level latent space that allows for better identifiablility of latent content factors from raw data.

2.This work designs a practical VAE-based framework and also provides sufficient theoretical analysis.

3.Extensive experiments that conducted on both synthetic and real-world datasets show the validity of the proposed framework.

### Weaknesses
1.It is common to decouple the raw data into domain-invariant and domain-specific parts in feature disentanglement methods and some causality-based methods. Though this work proposes a two-level latent space to assist the isolation of latent variables, the novelty is still limited. 

2.The proposed framework requires the use of domain variables, which are not accessible in some cases, limiting the application of this method.

3.To better show the superiority of this work, it is necessary to compare the experimental results of the proposed method with that of the similar works, such as LaCIM, iMSDA, and the works mentioned in Sec 3.1. Besides, the baselines compared in tables are somewhat outdated.

4.The authors are suggested to evaluate the algorithm on Domainnet dataset to verify its effectiveness on large-scale datasets.

### Questions
See Weakness.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a novel approach for domain generalization by introducing a two-level latent variable model.
The key idea is to partition the latent space into invariant content factors and variant style factors across domains. Specifically, the high-level latent space consists of content and style variables.
The content variables capture label-related information while the style variables represent domain-specific information.
To achieve identifiability, the model introduces a middle-level latent space with the same partition structure.
The middle-level content factors are derived from the high-level content factors via label-specific functions.
Similarly, the middle-level style factors are obtained by applying component-wise monotonic functions to the high-level style factors, which depend on the label, domain, and variable index.
The observation is then generated by applying a mixing function on the middle-level latent factors, which is shared across domains.
The key theoretical contribution is providing sufficient conditions to achieve identifiability and isolation of the content factors from the style factors. This relies on assuming the style factors follow an exponential family distribution conditioned on label and domain, plus a domain variability assumption.
Under these assumptions, the authors prove the content factors can be block-identified and style factors can be linearly identified.
Based on the theoretical results, the paper proposes a practical learning algorithm using a VAE framework.
The VAE encoder estimates the posterior of the latent factors.
Normalizing flows are used to transform between the high-level and middle-level latent variables.
An invariant classifier is trained solely on the recovered content factors.
Experiments on synthetic and real-world image datasets demonstrate the approach can effectively identify the latent factors and that training on just the content factors improves domain generalization performance.

### Strengths
- This paper made contributions: 1) A novel identifiable latent variable model with content/style partition 2) Sufficient conditions for identifiability and isolation of content factors 3) A practical learning algorithm based on VAEs and normalizing flows 4) Strong empirical performance on domain generalization tasks. The proposed approach offers a promising way to learn invariant representations for generalizable models. This paper proposes a novel two-level latent variable model with content/style partitioning to achieve domain generalization. This framework allows dependence between factors while still enabling identifiability. This work introduces sufficient conditions for identifiability and isolation of content factors based on exponential family priors and domain variability assumptions, combining VAEs and normalizing flows in a new way to estimate latent factors for domain generalization.
- Provides thorough theoretical analysis and identifiability guarantees for the proposed model.
- Learning invariant representations is an important open problem for building generalizable ML models. Methodology could be applied to other domain generalization areas beyond image classification.

### Weaknesses
- The method relies on specific assumptions about the latent variable distributions which may not hold universally. For example, the exponential family prior and domain variability assumptions.
- The theoretical analysis requires infinite data and may not provide guarantees for small sample sizes. More analysis of finite sample behavior would be useful.
- The model structure imposes some limitations, like only allowing dependence between factors through the label, but more complex relationships may exist in real datasets.
- While state-of-the-art results are shown, the gains are incremental. More significant jumps in performance may be needed to drive adoption.

### Questions
Please see above.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes two-level latent variables for improving domain generalization. The key idea is to employ other $z_c$ and $z_s$ to  recover the distribution of content factors and isolating them from style factors.

### Strengths
- The idea is innovative and interesting. 

- The experimental results are convincing.

### Weaknesses
- The notions used in this paper are confusing, making it hard to read. The authors can simplify the notions in formal setup because it seems that the paper only uses a few of them. Moreover, in Eq. (1), what is the meaning of the distribution inside $\phi(.,.)$.

- The generative process or data generation model also confuses me. It is not clear why $f_y(\hat{z_c})$ can return $z_{c_1}, z_{c_2}, z_{c_3}$. Also, the same question is for the style branch. What are $\hat{z_{s_1}}, \hat{z_{s_2}}, \hat{z_{s_3}}$? Are they the styles of the domains? 

- Eq. (7) is also hard to interpret to me. As far as I understand, in the first level you have a single variable $\hat{z_s}$ for the style and using the map $f_{e,y,i}$ to transform it to $p(z_s \mid y,e)$. However, why do you need the index $i$ here?

### Questions
Please answer my questions in the weakness session. Moreover, how do you design $f_y$ and $f_{e,y,i}$ using distinct flow-based architecture to incorporate the information of e, y, i? I am happy to increase my score if the authors can resolve my unclear points.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
