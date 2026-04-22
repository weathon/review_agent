# Permutation-Invariant Hierarchical Representation Learning for Reinforcement-Guided Feature Transformation

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Feature transformation aims to refine tabular feature spaces by mathematically transforming existing features into more predictive representations. Recent advances leverage generative intelligence to encode transformation knowledge into continuous embedding spaces, facilitating the exploration of superior feature transformation sequences. However, such methods face three critical limitations: 1) Neglecting hierarchical relationships between low-level features, mathematical operations and the resulting high-level feature abstractions, causing incomplete representations of the transformation process; 2) Incorrectly encoding transformation sequences as order-sensitive, introducing unnecessary biases into the learned continuous embedding space; 3) Relying on gradient-based search methods under the assumption of embedding space convexity, making these methods susceptible to being trapped in local optima. To address these limitations, we propose a novel framework consisting of two key components. First, we introduce a permutation-invariant hierarchical modeling module that explicitly captures hierarchical interactions from low-level features and operations to high-level feature abstractions. Within this module, an self-attention pooling mechanism ensures permutation invariance of the learned embedding space, aligning generated feature abstractions directly with predictive performance. Second, we develop a policy-guided multi-objective search strategy using reinforcement learning (RL) to effectively explore the embedding space. We select locally optimal search seeds from empirical data based on model performance, then simultaneously optimize predictive accuracy and minimize transformation sequence length starting from these seeds. Finally, extensive experiments are conducted to evaluate the effectiveness, efficiency and robustness of our framework. Our code and data are publicly accessible https://anonymous.4open.science/r/PHER-32A6.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a novel tabular feature transformation method aimed at addressing three limitations of existing works. First, two features at different hierarchies are designed to learn token-level and concept-level representations, thereby addressing the first limitation of ignoring the hierarchical relationships among the original features. Second, a specific self-attention pooling mechanism is adopted to learn permutation-invariant features, thus overcoming the second limitation of neglecting order invariance in the generated features. Thirdly, Proximal Policy Optimization is used to avoid the drawbacks of traditional gradient-based optimization methods. Experiments are conducted on various datasets in comparison with a rich set of baseline methods.

### Strengths
(1) The motivations and descriptions of the existing limitations are well introduced and clearly explained, especially the examples presented in Figure 1. 

(2) The quantitative results in Table 1 and the ablation studies in Figure 3 show that the proposed method is promising, and each design module contributes effectively.

### Weaknesses
(W1) The major weakness of this paper is the writing and organization. For some reason, the authors break the descriptions of the pipeline of the proposed methods into tiny fragments scattered throughout the methods section, which makes it difficult to follow and understand both the overall pipeline and the details of the proposed methods. Instead of writing several pieces of "why something matters" and "why use something," please write the entire problem statement and the proposed method into concise and coherent paragraphs using consistent notations and terms. 

For example, (1) The term “aggregation layer” used in Figure 2 and line 227 is never formally defined, but it appears to be a mean pooling layer as described in line 201, which takes readers extra efforts to link them. (2) The important token embedding notation E is not mentioned in the summarized introduction of the problem statement, while other key terms such as Γ and G are mentioned. (3) The relationship between E and G is unclear: while the authors provide many vague plain-language explanations, a clear and formal equation for their relationship is not found. Unfortunately, the closest one seems to contain typos in its definition: In line 229, the authors write “Thereafter, G′ is input into φ_con to obtain the global embedding G′ = φ_con(Mean(E)).” But it looks like the first G′ should be G instead, given the text before this line? Additionally, the “suggested” definition here G = Mean(E) seems inconsistent with the examples provided in line 206, where G is the mean over a set of E. 

In summary, similar readability issues make it difficult to understand and evaluate the proposed method, and a simple step-by-step explanation corresponding to Figure 2, using formal languages and equations, is needed.

(W2) While the performance of the features transformed by the proposed method is evaluated on downstream tasks, no direct evaluation was conducted on the embeddings themselves. For example, although the authors claim that “This structure guarantees that any permutation of generated concepts yields identical embeddings” in line 96, no direct evidence supports this claim for the dataset used in this paper’s experiments.

### Questions
(Q1) The authors are encouraged to respond to and explain my concerns listed in the Weaknesses section. Additional experimental results and demonstrations would also be greatly appreciated. 

(Q2) It appears that the URL to the paper’s code is broken. I received a “The requested file is not found” error for all .py files in the code folder.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper solves the problem of automated feature transformation using hierarchical modeling and policy-guided feature transformation. The proposed framework first maps the feature transformation sequence into the continuous embedding space and then employs the policy-guided multi-objective search to find the optimal feature transformation sequence in the continuous embedding space. Experimental results reveal the effectiveness of the propose method.

### Strengths
1. The paper is well-written and easy to follow.

2. The proposed hierarchical encoder-decoder models could generate better global embeddings.

3. The reward model in the policy-guided search contains the length of feature transformation sequence, which is helpful to find more compact transformation sequence and improve computation efficiency.

4. The extensive experiments demonstrate the superior performance of the proposed method.

### Weaknesses
1. The novelty is limited. The overall framework that first maps the feature transformation sequence into the continuous embedding space and then perform optimization in the continuous space is not new and is prosed in many existing methods such as DIFER Zhu et al. (2022b) and MOAT Wang et al. (2023). Thus, the innovation is incremental.

2. The proposed method is complicated and the involve huge computation. For example, in the hierarchical modeling stage, token-level encoder/decoder, concept-level encoder/decoder need to be trained. In the policy-guided multi-objective search stage, PPO agent needs to be trained. Thus, the computation complexity should be analyzed in detail.

3. Beside the performance comparison, the computational efficiency comparison should be performed, especially on large-scale datasets. 

4. It is not clear why the proposed self-attention pooling mechanism can ensure permutation invariance. The authors should further explain the permutation invariance in detail and demonstrate permutation invariance with some case studies.

5. It is suggested to analyze the interpretability of the generated features.

### Questions
see Weakness.

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
4

### Summary
This work proposes PHER to automatically transform tabular features. The core ideas are permutation-invariant hierarchical modeling and policy-guided multi-objective search. The former models both low-level nad high-level features interactions, using a self-attention pooling mechanism, while the later uses reinforcement learning to explore non-convex embedding space.

### Strengths
- Automatic feature engineering is an interesting topic.
- The empirical result is strong and shows the superior performance of PHER.
- This work provides source code for reproducibility.

### Weaknesses
- The RL approach might be computationally expensive, expecially on large datasets and high dimenstional datasets.
- Automatic feature engineering is an old topic. The RL approach of automatic feature engineering has been studied in 2017 [1].

[1] Nargesian, Fatemeh, et al. "Learning feature engineering for classification." Ijcai. Vol. 17. 2017.

### Questions
- For table 1, could you provide a supervised learning baseline without any feature engineering?
- Can a model trained on one dataset (e.g. Higgs Boson) be applied on another dataset (e.g. Amazon Employe)?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PHER, a unifying framework for automated feature transformation on tabular data, where the authors try to tackle the three main challenges of recent generative methods using the three stages, specifically by modeling hierarchical features, introducing permutation-invariant feature sets, and using a non-gradient approach (PPO) for final search.

I'll adjust my rating based on the author's response.

### Strengths
1. Well-written paper, with visually appealing figures.
2. The proposed system (using a permutation-invariant mechanism on the set of generated concepts) is technically sound.
3. Strong empirical results in quantitative performance.

### Weaknesses
1. The proposed system seems to have a complex design. The workflow requires 1) initial data collection using reinforcement learning; 2) a 3-stage procedure for training the hierarchical model; and 3) a final PPO search. This could introduce a barrier to adoption.
2. Fig. 5 shows that using high-quality seeds is much better than using random seeds. The paper does not fully explore how PHER would perform if the initial data collection phase yielded a weaker set of records. We do not know what the lower bound of quality is. Also, this shows a dependence on the random seed, which renders the proposed solution less robust.

### Questions
1. The mean pooling layer is illustrated with an example: $G_1 = \text{Mean} (E_{f_1}, E_{plus}, E_{f_7})$ for the transformation $f_1 + f_7$, which seems to be at risk of losing the crucial syntax structure of the transformation. For instance, how this simple mean pooling can distinguish between $f_1 + f_7$ and $f_7 \times f_1$ (the operator token is just one part of the average)? Is this a potential information bottleneck here?
2. How dependent is the method on the quality / diversity of the initial set of transformation records (which is used to train the hierarchical model and provide the top-$k$ for the search)? An ablation study on this would be helpful.
3. The PPO agent introduces several hyperparameters. Fig. 7 shows an optimal $\lambda$ of $0.9$, which means it prioritizes performance over the stated goal of minimizing transformation length. However, from Fig. 6, PHER does still produce shorter sequences. How does this justify the emphasis on the multi-objective balance? Also, given the complexity of the system, tuning these hyperparameters could be challenging in practice.


Also, I am not able to see the content of the files in the link provided, except for the `README.md`. I can only see the file structure of the repo. I cannot determine how complex the implementation is from the code.

### Soundness
2

### Presentation
4

### Contribution
3
