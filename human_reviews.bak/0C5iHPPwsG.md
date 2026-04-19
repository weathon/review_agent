# Autoencoder-Based General-Purpose Representation Learning for Entity Embedding

- Decision: Reject
- Scores: 3, 5, 8, 6

## Abstract
Recent advances in representation learning have successfully leveraged the underlying domain-specific structure of data across various fields. However, representing diverse and complex entities stored in tabular format within a latent space remains challenging.
In this paper, we introduce DeepCAE, a novel method for calculating the regularization term for multi-layer contractive autoencoders (CAEs). Additionally, we formalize a general-purpose entity embedding framework and use it to empirically show that DeepCAE outperforms all other tested autoencoder variants in both reconstruction performance and downstream prediction performance. Notably, when compared to a stacked CAE across 13 datasets, DeepCAE achieves a 34% improvement in reconstruction error.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This paper proposes a method called DeepCAE to calculate the regularization term for multi-layer contractive autoencoders and utilizes DeepCAE to power a general-purpose entity embedding framework. The experimental results show that DeepCAE outperformed the baselines.

### Strengths
The authors conducted sufficient experiments to validate the effectiveness of their proposed method. They also provide code and data to reproduce the results.

### Weaknesses
I think the prominent issue is the writing. 1) It **contains lots of tangential content** which is unnecessary to elaborate on in this paper. For example, in Section 2, I don't see the point of discussing PCA, variational autoencoders, and transformers, as they are not directly part of or foundational to your methodology. 2) The writing **lacks substantial details and is not self-contained**. This work is largely based on the previous CAE work by Rifai et al. (2021b), but the introduction to that work is incomplete, making it difficult to understand the proposed method and its details. Instead, the authors frequently refer readers to the original work. This problem also exists in the experiment, which did not clearly present the experiment settings. 3) **Lacking of logical coherence**. Some important arguments lack evidential support and are not clearly described.

Please see questions below for exact points.

### Questions
1. The argument "*Consequently, these representation methods not only are limited, but also fall short or are inapplicable ...*" in line 47-48, do you have evidence to support this claim, such as references or experimental results? Also, what do "*these representation methods*" refer to?
2. In the following paragraph starting at line 50, how are the two contributions related? The authors might consider clarifying that the framework is based on their proposed method.
3. In section 2, what is the purpose of elaborating on PCA, variational autoencoders, and transformers? If they only serve as baselines, a brief introduction in the experimental setup might suffice. Instead, the focus should be on elaborating CAE in this section on its principle and structure. For examples, the steps to encode the input and decode it, the loss function.
4. Eq. (1) lacks detailed description, such as the meaning of its symbols.
5. In line 92-93, "*Thanks to their ability to produce stable and robust embeddings, CAE were proven to be superior to Denoising Autoencoders (DAE)*", the real reason behind *produce stable and robust embeddings* and *superior performance* is missing. The authors should make this argument more well-founded.
6. In the first paragraph of section 4.1, the sentence "*we analyze related work and find that, to the best of our knowledge, all use stacking Wu et al. (2019); Aamir et al. (2021); Wang et al. (2020), including Rifai et al. (2011b), who originally proposed the CAE.*" is poorly written. Additionally, what kind of "stacking" was used? The authors should elaborate on how previous works implemented this.
7. In line 223, how is the conclusion "$O(d_x \times d_h^2)$ to $O(d_x \times d_h)$" derived? I cannot deduce this from Eq. (3) alone. And "*and $d_h$ is the hidden embedding space.*", do you mean $d_h$ is the dimension of the hidden embedding space?
8. The explanation of how Eq.(4) is obtained from Eq. (3) is unclear. The entire inference process from Eq. (3) to Eq. (8) lacks coherence.
9. In line 274, "*such as text and time-series*" what kind of information is the "text" and "time-series" exactly? Could you provide examples?
10. In Figure 1, how exactly do you concatenate the text encoding, TS encoding, and tabular data together?
11. In section 5, what are the experimental settings, such as the number of encoder layers and feature dimensions? Additionally, the dataset statistics should be included in the main paper rather than the appendix, as they are important.
12. How is the Mean Squared Error (MSE) computed? Is this metric conventionally used in previous work?
13. In the first paragraph of section 5.1.2, "*We trained XGBOOST (Chen & Guestrin, 2016) predictors ...*", what is the XGBOOST model, and what exactly is the downstream task?
14. In section 7, line 522-524, "*Furthermore, we argue that the augmentative capabilities of more complex architectures like Transformers and CNNs are not necessarily useful in the production of a compact representation of an entity.*", do you have evidence to support this argument?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes DeepCAE for learning general-purpose entity embeddings. Although DeepCAE extends from the contractive autoencoder, the authors provide a more effective design in calculating the multi-layered regularization term. Extensive experiments across 13 datasets demonstrate state-of-the-art performance of DeepCAE on reconstruction and ‌downstream‌ prediction tasks.

### Strengths
The extension of CAE for multi-layer setting is simple yet effective. 

The ‌authors conduct‌ experiments across 13 datasets and ‌cover‌ various types of entities.

The results demonstrate state-of-the-art performance on both reconstruction and ‌downstream‌ prediction.

### Weaknesses
The main paper should be self-contained. The authors may overly refer to the ‌original‌ CAE paper.

The motivation of DeepCAE and CAE is not clearly introduced. It is ‌confusing‌ for me why ‌they are designed for tabular data, and how ‌they are connected‌.

In the experimental results, the strengths of DeepCAE are not significant compared with the standard AE.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
1

### Summary
This paper introduces DEEPCAE, a versatile entity embedding framework based on autoencoders. By extending contractive autoencoders (CAE) to a multi-layer structure and preserving the original regularization design, DEEPCAE enhances both reconstruction accuracy and downstream prediction performance for complex entity embeddings. In tests across 13 datasets, DEEPCAE consistently outperformed other autoencoder variants in both reconstruction error and predictive tasks, achieving a 34% reduction in reconstruction error compared to a stacked CAE. This framework offers an efficient, scalable solution for general-purpose entity embeddings across diverse domains, ultimately reducing time spent on feature engineering and boosting model accuracy.

### Strengths
This paper introduces DEEPCAE, a multi-layer contractive autoencoder designed for general-purpose entity embedding. By extending the contractive autoencoder framework to multiple layers while preserving regularization, DEEPCAE overcomes limitations seen in stacked CAEs, opening new possibilities for autoencoders with high-dimensional data. The study is thorough, with DEEPCAE evaluated across 13 diverse datasets, showing consistently strong results in both reconstruction and downstream tasks that highlight its effectiveness. The paper is well-organized, with clear derivations and detailed appendices on model architecture and hyperparameters to ensure reproducibility. Overall, DEEPCAE offers an efficient, versatile solution for embedding across domains, reducing feature engineering time and adding practical value for cross-application embeddings in industrial settings.

### Weaknesses
DEEPCAE demonstrates impressive results with contractive regularization, but it may not have explored other well-established regularization techniques, like dropout or data augmentation, which are effective in preventing overfitting. It would be beneficial for the authors to consider incorporating these strategies into the DEEPCAE framework. Doing so could enhance the model's robustness, and evaluating their impact on performance in future experiments would provide valuable insights into improving its effectiveness.

### Questions
In the existing regularization strategies, have other methods been considered, such as dropout or data augmentation? These techniques have been proven effective in preventing overfitting.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work the authors extend the Contractive AutoEncoder (CAE) framework for the calculation of the Jacobian 
of the entire encoder in the contractive loss from single-layer to multi-layer settings ( DeepCAE ).


Empirically over tabular benchmarks, the authors show DeepCAE can be leveraged in a general purpose embedding 
framework where embeddings are feed to XGBoost to obtain gains in reconstruction performance and comparable/slightly better performance 
downstream prediction (classification/regression) performance as compared with various AutoEncoders and Transformer baselines 
( though not when compared with KernalPCA from a downstream performance perspective ).  
They additionally show the noteable reconstruction performance of DeepCAEs compared with Stacked CAEs .

### Strengths
The authors show how the CAE framework ( an AE with an additional objective component that is the Frobenius norm of the model with respect to the input ) can be extended to a multi-layer setting in a way that is advantageous when compared with prior extensions to CAEs which worked via stacking.  

They do an extensive empirical analysis to discuss reconstruction and downstream accuracy benefits of the method while discussing costs of the method (scaling as the method is cubic with respect to layer size ) and its downstream limitations compared to KernalPCA.

The setting ( encoding tabular data with multi-modal data types  ) is important and they show how its not handled readily or generally by Transformer type architectures.

### Weaknesses
1) Time comparisons and particularly error bars needed everywhere ( KernelPCA/StandardAE/DeepCAE ).  For your benchmarks are you all running multiple seeds per each?  

The main question ( which the authors have pointed out openly in the paper and for future work ) is if the cost of DeepCAEs is worth the effort?   They’ve shown reconstruction is slightly better, but for tasks its pretty comparable to AE and KernalPCA does better (still an interesting finding ). How important is reconstruction loss really for this setup?  What is the time complexity of KernalPCA and outside of reconstruction loss being subpar are there other reasons to not use it?

2) Are there stronger baselines to compare against both encoding wise and classifier wise (ie, XGBoost vs something else) against if what we care about is tabular performance using embeddings?  The former is the more important of the two and there is a NeurIPS workshop on fusing modalities for tabular data thats in its 3rd edition (https://table-representation-learning.github.io ) 

3) The general purpose embedding pipeline seems like the standard solution to re-using embeddings for downtsream tasks from vector databases and not particular to DeepCAE?  Is this the case?  If not, it could strengthen the paper to clarify how so if not.

4) It would be interesting to either show experiments on or discuss how DeepCAE does on just image or text data as well to compare its reconstruction and downstream task performance there.  Is there anything in particular that makes this approach specific to tabular data with multi-modal data?  If the method gives performance boosts when encoding image/timeseries/text, it would greatly strengthen the results of the paper and would make incorporating the method.

5) While the background on CAE was very much needed, the sections on VAE and Transformers were probably lesser so ( or could have been pushed into the appendix ) especially since you show effectively they are not nearly as effective.  This space could/should be used for looking more at KernalPCA vs Standard AE vs DeepCAE costs/tradeoffs and potentially other modalities ( point 4)

6) Did you all do experiments against the original CAE?  The paragraph starting at 280 made it seem like you would ( and in the final section you do with StackedCAE), but then in the experiments Convolutional AE is used instead which I wasn't expecting.  I'm assuming this is shown in past work when Stacked CAEs are introduced but having a sense of that as well would be good since its much cheaper computationally than both Stacked and Deep CAEs.

### Questions
Q: Did you all empirically assess how much time this k x d^3 adds time-wise compared with just a standard AE?  Its an offline computation, but getting a sense of what that tradeoff comes out to time wise for your datasets would be interesting ( ie, is it that big of a hit in the end since the datasets are all below 45k instances each ).  How does KernalPCA perform?  

Q: line 138 should probably cite the 2017 Transformers paper and not the 2023 arxiv one

Q: Is there a reason in particular for using tanh activations in your extension of CAE?  I get it allows for the decomposition shown, but are there other activations or ablations which could have been performed ?

### Soundness
4

### Presentation
4

### Contribution
2
