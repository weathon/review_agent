# Low-Dimension-to-High-Dimension Generalization and Its Implications for Length Generalization

- Decision: Reject
- Scores: 3, 6, 6, 5

## Abstract
Low-Dimension-to-High-Dimension (LDHD) generalization is a special case of Out-of-Distribution (OOD) generalization, where the training data are restricted to a low-dimensional subspace of the high-dimensional testing space. Assuming that each instance is generated from a latent variable and the dimension of the latent variable reflects the problem scale, the inherent scaling challenge in length generalization can be captured by the LDHD generalization in the latent space. We theoretically demonstrate that LDHD generalization is generally unattainable without exploiting prior knowledge to provide appropriate inductive bias. Specifically, we explore LDHD generalization in Boolean functions. We verify that different architectures trained with (S)GD converge to \emph{min-degree interpolators w.r.t. different linearly independent sets}. LDHD generalization is achievable if and only if the target function coincides with this inductive bias. Applying the insights from LDHD generalization to length generalization, we explain the effectiveness of CoT as changing the structure latent space to enable better LDHD generalization. We also propose a principle for position embedding design to handle both the inherent LDHD generalization and the nuisances such as the data format. Following the principle, we propose a novel position embedding called RPE-Square that remedies the RPE for dealing with the data format nuisance.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper is about a notion similar to domain adaption, but tailored to scenarios found in large language models, that the authors named low-dimension-to-high-dimension (LDHD) generalization. The authors give generic impossibility results in the style of the No-Free-Lunch theorem, and also specific impossibility results for several models, drawing attention to the issue that LDHD cannot happen without inductive bias.

### Strengths
This is an important problem. The authors made commendable efforts in understanding LDHD for Boolean functions and connections to the implicit regularization line of research to decipher what kind of inductive may or may not be helpful for LDHD.

### Weaknesses
The main issue for me is with the presentation and the writing. There is a big conceptual gap between the introduction, especially the two examples given involving addition, with the rest of the paper. After reading examples 1 and 2, I still don't have sense of the relevance toward LDHD that they are intending to convey. Due to this, I find the introduction to be difficult to understand.

It is not clear how the notion of LDHD relates to previous works on domain adaptation e.g., [1]. The only novelty in Definition 1 seems to be that the two domains are nested and both are included inside a larger ambient space. This scenario can be captured by previous notions of OOD generalization.

Given that Boolean functions play an important role, there should be more discussions about what is degree, what is the intuition behind degree profiles.

Assumption 2 has been refuted in [2]. While this doesn't invalidate results from Sec 4.2, I feel that this should be discussed.

[1] David, Shai Ben, et al. "Impossibility theorems for domain adaptation." Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics. JMLR Workshop and Conference Proceedings, 2010.

[2] Li, Zhiyuan, Yuping Luo, and Kaifeng Lyu. "Towards Resolving the Implicit Bias of Gradient Descent for Matrix Factorization: Greedy Low-Rank Learning." International Conference on Learning Representations.

### Questions
More specifically, I think the authors should address these following points to make the introduction more understandable:

L052: I don't understand Example 1. How does this example illustrate LDHD?
L062: What does relative position embedding have to do with the main thesis of this paper? I find this remark to be distracting.

L076: What is a "concept"? How does this show up in Example 1 and 2? Neither example 1 nor 2 have a label. Also, the notation phi is completely new. If the examples are meant to motivate Assumption 1, it should be made clear.

L088: Why is Sigma a field now? In example 1 and 2, Sigma is the set of tuple of digits.

L096: How is the No-Free Lunch theorem relevant here? The No-Free Lunch Theorem says that there's no learning algorithm is can be optimal for all problem instances. I don't see the relevance to LDHD.

L102: What is "minimal degree-profile bias"?
L107: What are "independent sets"? Do you mean "linearly independent"?

### Soundness
4

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper frames the length generalization problem in tasks like addition as a low-dimensional to high-dimensional (LDHD) generalization problem. With this motivation, the paper studies LDHD generalization in two theoretical setups on Boolean functions, with Boolean input sequences of length $N$. Here, the data dimensionality is controlled by the number of final positions fixed to 1—that is, sequences coming from a low-dimensional space of size  $N_0 < N$ have the last $N - N_0$ positions set to 1. 

In the first setup, the authors consider a random feature model *with projection* (RFMP), where the inputs are first projected into a subspace defined by matrix $V$ before being passed through an RFM. Extending the result in Abbe et al. (2023), they show gradient descent (GD) converges to a minimum degree interpolator, now with respect to a basis that depends on the choice of $V$. They use this result to argue that if the ground truth Boolean function depends on the higher dimensions of the input, the min-norm interpolator of the low-dimensional data will fail to capture it. 

The second setup considers variations of linear attention models, simplified to focus on the role of positional embedding. Here, they show that different positional embedding changes the inductive bias of GD by changing the bases with respect to which the min degree interpolator is chosen. Motivated by these findings, they suggest adjustments to the relative positional encoding that helps with length generalization for addition.

### Strengths
The paper's perspective on length generalization is interesting. The authors look at different theoretical models to show how solutions to the LDHD problem can change as the models are refined. Although the empirical examples are limited, they have brief results on the empirical implications of their theoretical analysis.

### Weaknesses
I found the paper a bit hard to parse and making connections between different parts of the results was challenging to me. I believe if the authors use some of the remaining space within the page limit to discuss the high-level/intuitive connections of their results in more detail or discuss proof sketches, it would improve the paper's clarity. As it stands, I find it difficult to fully grasp the results.

I will list some of the points I had trouble following as questions below. It’s possible that I missed some aspects of the paper. Thus, I'm open to reconsidering my score if needed.

### Questions
1.  **Thm 1 & Cor 1**: 
    - Can you explain why the min-degree interpolator cannot capture functions with $I(f)\not\subset [N_0]$?  
    - Does Cor. hold for any projection $V$? Then, 1) what's the purpose of adding the projection to the RFM? and 2) how can designing $V$ (line 287) enable LDHD generalization if the target function has a dependence on higher dimensions?
2.  **Sec 4.2**: I don't understand the intuition behind the PLAA model:
    - on notations: 1) i didn't find $\mathcal{U}_N$ in notations. Is it referring to upper triangular matrices? 2) Is $x$ a sequence of $\{\pm1\}$? Then what does it mean "value of $x_i$ is identical to itself" (line 307)?
    - What is the motivation for defining PLAA this way? One thing that confuses me is that if the input is a sequence of all 1s, then PLAA always outputs zero by definition.
    - In Defn 6, the only difference to PLAA, is adding an upper triangular mask on top of the weights. while in PLAA, the learnable parameters are already constrained to be upper triangular. Where is the APE here?

3.  Can you also explain your proposed RPE-squared positional encoding? How do your previous results explain the improvement in the performance? Do you characterize how this PE changes the inductive bias which leads to generalization or is this just a heuristic suggestion?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
By studying length generalization through the lens of generalization from lower to higher dimensional subspaces, the paper provides results that explain why this length generalization is hard to achieve with modern architectures.

More specifically, the paper tackles the problem of length generalization in modern deep learning architectures. A representative example is the addition of numbers that are larger than any numbers that have appeared in the training data. The paper's main approach is to frame this kind of generalization as a generalization ability to embeddings that come from a subspace of larger dimension than the dimension of the training data and is termed Low-Dimension-to-High-Dimension (LDHD) generalization.

Theorem 1 establishes a no free lunch theorem stating that, without introducing additional assumptions (e.g., by having priors on the function that is being learned by incorporating some kind of inductive bias), it's not possible to generalize to a higher dimensional subspace. Intuitively, this is because the additional dimensions appearing in the test set are orthogonal to those in the training set and therefore it's by construction impossible to infer anything about them from the training data. The paper then looks at different models and studies their structural biases.

Theorem 2 establishes that, in the limit, the random feature model with projection converges to a minimum degree interpolator with respect to an independent set defined by the projection. This  implies that LDHD generalization is only achievable for a restricted set of functions that depend on the subspace of the training set. A similar result is shown for a certain class of decoder only transformers. An additional result is provided for transformers with absolute positional encodings, where it is shown that if the rank of the attention matrix required for the target function exceeds the dimension of the subspace that the training data live in, then LDHD generalization is not possible. Finally, a result is given for generalized relational positional encodings where again we are shown that in the limit, training converges to a minimum degree interpolator with respect to the independent sets induced by the positional encodings. These results all reinforce the idea that generalizing to arbitrary length sequences with some standard ML models is going to be challenging.
These ideas are used to argue that Chain of Thought is potentially effective in length generalization partly because it doesn't increase the dimensionality of the space for test data.

Finally, some positional encodings are proposed to alleviate the challenges that arise from using an URF encoding.

### Strengths
- This subspace-centric perspective on length generalization is interesting.
- The results on decoder-only transformers build nicely on previous work and provide a nice connection to the rank of the attention.
- rpe squared encodings seem to work for URF.

### Weaknesses
I find confusing the way certain sections of this paper are written which makes the overall story and contribution hard to tease out:

- What is the motivation behind the random model with projections? The result about the Fourier basis has been established in previous work. The paper claims around line 262 that not all RFMP models converge to the min-degree interpolator. However, theorem 2 esentially establishes that if we do a change of basis for the data then the min-degree interpolator is achieved on the projected fourier basis. 
I don't see why theorem 2 is necessary to state corollary 1. Doesn't corollary 1 follow from the theorem in Abbe et al's paper?
I think it's unclear what exactly theorem 2 contributes to the discussion since it follows straightforwardly from the original result.

- Since chain of thought appears to improve results in those kinds of tasks, it seems important to understand how it fits into the LDHD generalization perspective. The paper provides a short discussion suggesting that CoT avoids increasing the dimensionality of the hidden space in which the data lie on and says that it 'extends the field' that is used by allowing for a symbol with undetermined values. It is not explained why this could be helpful (though it is suggested in the paper) to learn the correct target concept nor what quantitative predictions one can extract from this framing of CoT.  Why isn't CoT enough to deal with the generalization issue?

- I don't see how the proposed positional encodings connect to the contributions of this paper and the analysis that precedes them (line 431 claims that's the case). The subspace-centric view as I understand it is introduced so that the generalization problem is characterized in the hidden space. On the other hand, the paper seems to suggest that URF format makes generalization harder, or no? How does that connect with the comments about CoT being able to deal with the dimensionality of the hidden space? Why is CoT not enough and why are rpe-squared encodings needed? Do they somehow address the rank deficiency that is discussed in Corollary 2?

- The paper states in line 471: The design of RPE-Square illustrates how the principle can lead to position embeddings for better length generalization. What principle?  It addressed the URF formatting issue, but that seems orthogonal to the previous discussion.

Overall, I find the contribution of the paper potentially interesting, but it is quite hard to disentangle what the actual takeaways are from the paper. This is exacerbated by the writing being confusing and/or unclear in certain places that I have indicated. Until some of those things are cleared up I cannot recommend acceptance but I am obviously open to changing my score after the rebuttal.

### Questions
See above

### Soundness
3

### Presentation
2

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
This paper introduces Low-Dimension-to-High-Dimension (LDHD) generalization, a framework addressing the challenge of generalizing from low-dimensional training data to high-dimensional testing data, which is relevant in Out-of-Distribution (OOD) generalization scenarios. LDHD generalization captures the scaling issues in reasoning tasks, where models trained on small-scale data must generalize to larger, more complex cases. The authors theoretically demonstrate that LDHD generalization is unattainable without incorporating inductive biases, formalized through a version of the “No-Free-Lunch Theorem.”

The paper explores LDHD generalization specifically in Boolean functions and presents findings that different architectures under Stochastic Gradient Descent (SGD) can achieve LDHD generalization if the model’s inductive bias aligns with the target function. It introduces Random Feature Models with Projection (RFMP) and position-only linear attention with Advice (PLAA) models and examines how their biases support LDHD generalization.

In practical applications, the authors propose the RPE-Square position embedding as an enhancement of the standard Relative Position Embedding (RPE). RPE-Square is designed to improve length generalization in cases with unaligned data formats. Additionally, the paper provides insight into Chain-of-Thought (CoT) reasoning, suggesting that it extends the latent space without increasing dimensionality, thus supporting LDHD generalization.

The paper contributes a new perspective on generalization challenges in reasoning tasks. It emphasizes the role of inductive biases and offers practical methods for achieving length generalization in complex models.

### Strengths
Broader Empirical Validation:
While the theoretical analysis and specific experiments (e.g., RPE-Square’s impact) are well-supported. However, the paper would benefit from a broader empirical evaluation across different models and datasets, especially complex reasoning tasks beyond Boolean functions. The current examples, while illustrative, may not fully capture the diversity of real-world data distributions. Including results from larger-scale datasets, such as multi-step arithmetic tasks or complex symbolic logic problems, would more effectively validate LDHD generalization and highlight its practical applicability.
Additionally, comparisons with alternative approaches to length generalization (such as recent Transformer variants or other positional encoding methods) would provide a more comprehensive view of where LDHD generalization stands in practice.

Explicit Justification of Assumptions:
Certain assumptions, such as those in the matrix factorization regularization conjecture (Assumption 2), are crucial to the theoretical claims but have not been fully explored. While empirical results partially support these assumptions, more discussion of their generalizability and limitations would be helpful. For example, analyzing situations where Assumption 2 may not hold could offer insights into potential failure cases, enhancing the robustness of the claims.

Increased Clarity in Technical Sections:
Some theoretical sections, particularly regarding the degree-profile bias and proofs for the PLAA models, are dense and may be challenging for readers less familiar with these concepts. Adding more intuitive explanations alongside formal definitions could make these sections more accessible. Diagrams illustrating key ideas, such as the shift from low-dimensional to high-dimensional spaces or the impact of different inductive biases, would also aid comprehension.
Furthermore, clarifying how specific model components contribute to LDHD generalization in practical tasks (beyond theoretical guarantees) could make the theoretical claims more relatable and actionable for practitioners.

Deeper Analysis of LDHD Limitations:
The LDHD generalization framework is theoretically strong, but discussing limitations or boundary conditions more explicitly would be beneficial. For instance, scenarios where LDHD generalization may fail—even with the proposed inductive biases—or limitations in scaling to very high dimensions could be explored. This would provide a balanced perspective on the framework’s applicability.

More Contextualization within Prior Work:
While the paper acknowledges related literature, it could provide a deeper comparison with recent works on length generalization (e.g., Ahuja & Mansouri 2024, Zhou et al. 2024) and other inductive bias approaches in Transformer architectures. Explaining how LDHD generalization and RPE-Square offer improvements or distinct advantages over these approaches would strengthen the contextual foundation. Additionally, discussing whether existing models like Recursive Neural Networks (RNNs) or Graph Neural Networks (GNNs) could benefit from LDHD generalization principles would broaden the relevance of the contributions.

Position Embedding Design Evaluation:
The RPE-Square embedding demonstrates promise, but a more thorough ablation study evaluating its impact under various configurations (e.g., different distance metrics or special token arrangements) would provide a more detailed understanding of its robustness. Testing RPE-Square in domains beyond the addition task, such as symbolic reasoning or multi-step logical tasks, could further establish its versatility and performance consistency across data formats.

In summary, enhancing empirical validation, refining theoretical assumptions, and providing further clarity and contextualization would elevate the strength and reach of this paper’s contributions. These additions would also help bridge the gap between theoretical insights and practical applicability in diverse model architectures and reasoning tasks.

### Weaknesses
The paper introduces a valuable perspective with LDHD generalization; however, it could be strengthened by addressing certain weaknesses in both its theoretical and experimental approaches.

Limited Empirical Scope:
While the theoretical contributions are strong, the experimental validation is limited in scope. The paper primarily demonstrates LDHD generalization through RPE-Square on specific cases, such as addition tasks. Testing a broader range of tasks, such as multi-step logical reasoning, symbolic manipulation, or mathematical operations, would provide a more convincing demonstration of the applicability of LDHD generalization. Expanding beyond Boolean functions to real-world data sets in NLP or other reasoning tasks would further validate the practicality of the approach.

Additionally, comparing the performance of RPE-Square with other recent advances in length generalization, such as transformer variants designed for reasoning (e.g., Kazemnejad et al., 2024) or models trained on complex arithmetic tasks (e.g., Lee et al., 2023), would highlight RPE-Square’s strengths and limitations.

Assumption Dependence and Generality:
Certain key assumptions in the theoretical framework, like those in the minimal-nuclear-norm-solution conjecture (Assumption 2), may not hold universally, potentially limiting the framework’s generality. The assumptions play a significant role in supporting the theoretical findings, so the paper could benefit from a more detailed examination of their applicability and limitations. For instance, providing empirical tests or theoretical discussions on where these assumptions might fail would offer a more balanced understanding of LDHD generalization’s robustness.

Exploring whether alternative assumptions or regularization methods could yield similar results would help generalize the framework. This would make the theoretical claims more robust and adaptable to a wider range of models.

Theoretical Accessibility and Clarity:
The paper’s dense theoretical sections, especially those regarding the degree-profile bias and proofs for the PLAA model, may be challenging for readers without a strong background in this area. Adding intuitive explanations or visual aids alongside the formal proofs could significantly improve readability and accessibility. A step-by-step breakdown of key concepts, particularly for the degree profile and min-degree interpolator, would make these sections easier to understand.

Diagrams illustrating key ideas, such as low-to-high dimensional shifts or how different inductive biases support LDHD generalization, would also help clarify the framework’s mechanisms.

Deeper Analysis of LDHD Generalization’s Practical Implications:
The paper would benefit from a more explicit discussion of the practical limitations of LDHD generalization. While LDHD is theoretically promising, acknowledging potential challenges—such as cases where the high-dimensional space differs fundamentally from training distributions—would strengthen the work. Highlighting when LDHD generalization may fail, even with well-designed inductive biases, would provide a balanced perspective.
Furthermore, discussing the implications of the LDHD framework for models beyond transformers, such as Graph Neural Networks (GNNs) or Recursive Neural Networks (RNNs), would broaden its relevance and demonstrate the approach's flexibility.

Limited Contextualization within Recent Length Generalization Research:
Although the paper addresses related work, it could better contextualize LDHD generalization relative to recent advances. Comparing LDHD with other generalization frameworks, especially recent developments in length generalization (e.g., Ahuja & Mansouri, 2024; Zhou et al., 2024), would clarify its unique contributions and limitations.
Providing insights into whether the LDHD framework could be adapted for methods beyond transformers, such as RNNs or CNN-based architectures, would also enrich the discussion. Additionally, examining whether other position encoding methods, such as rotary embeddings, would benefit from an LDHD perspective could reveal additional practical insights.

Broader Evaluation of RPE-Square Design Choices:

While RPE-Square demonstrates improvements in the addition task, a more comprehensive evaluation of its design would be valuable. The current analysis could be strengthened by testing RPE-Square under various conditions, such as different distance metrics, transformations, or arrangements of special tokens. This would provide a deeper understanding of its robustness across task variations.

Extending RPE-Square’s evaluation to additional domains, such as symbolic logic, planning, or other reasoning-intensive tasks, would demonstrate its versatility and generalizability. Additionally, an ablation study to understand the specific impact of each component (e.g., relative vs. absolute position aspects) would clarify the primary drivers of its performance gains.

The paper would benefit from broader empirical validation, expanded theoretical clarity, and deeper contextualization. These enhancements would make LDHD generalization’s theoretical and practical implications more robust, accessible, and impactful, better aligning the work with its stated goals.

### Questions
Broader Task Validation for LDHD Generalization:
Question: Have you considered validating LDHD generalization on a wider set of tasks beyond Boolean functions and addition problems? For instance, tasks involving more complex arithmetic, symbolic reasoning, or planning tasks?
Suggestion: Expanding the evaluation to include a broader range of reasoning-intensive tasks would demonstrate the generality and practical applicability of LDHD generalization. Additionally, comparing your findings with those of other approaches to length generalization, such as Kazemnejad et al. (2024) and Lee et al. (2023), could add value by highlighting their relative strengths and limitations.

Justification and Generality of Assumptions:
Question: Can you provide more insights into the practical applicability of the minimal nuclear norm solution conjecture (Assumption 2) and other key assumptions? Are there scenarios in which these assumptions may not hold, and how might that affect LDHD generalization?
Suggestion: It would be helpful to clarify the scope of these assumptions, perhaps by discussing or empirically validating their generality across different architectures. Exploring whether alternative assumptions or regularization techniques could produce similar results would provide a more robust foundation for your theoretical contributions.

Improving Theoretical Accessibility:
Question: Would you consider adding intuitive explanations or diagrams alongside the formal theoretical sections to enhance accessibility, especially for concepts like the degree-profile bias and min-degree interpolators?
Suggestion: While the formal proofs are well-structured, additional explanations or illustrations could help a broader audience appreciate the theoretical underpinnings of LDHD generalization. Visual aids representing the concept of low-to-high dimensional shifts, for example, might clarify these ideas.

Implications of LDHD Generalization for Other Architectures:
Question: Do you envision LDHD generalization principles applying to architectures beyond transformers, such as Graph Neural Networks (GNNs) or Recursive Neural Networks (RNNs)? If so, could you provide insights on how LDHD generalization might be adapted to these structures?
Suggestion: Discussing the potential applicability of LDHD generalization principles to other architectures could broaden the relevance of your contributions. Exploring cases where GNNs or RNNs might also benefit from LDHD-inspired inductive biases would make the work more versatile and impactful.



Broader Analysis of RPE-Square Design Choices:
Question: Could you provide more details on the design decisions for RPE-Square, such as the choice of distance metrics or the arrangement of special tokens? Have you tested alternative configurations? If so, how do they affect performance?
Suggestion: Analyzing the impact of design choices within RPE-Square, possibly through an ablation study, would provide a deeper understanding of what makes this embedding effective. Additionally, extending RPE-Square evaluations to other domains (e.g., symbolic logic or planning tasks) would clarify its generalizability.
Explicit Discussion on Potential Limitations:
Question: Are there specific conditions or model architectures where LDHD generalization might fail, even with well-designed inductive biases? Could you outline any known limitations or failure modes?
Suggestion: A section explicitly discussing possible limitations or boundary conditions for LDHD generalization would provide a more balanced perspective on the approach’s applicability. This could include insights into cases where training and testing distributions diverge significantly or where high-dimensional testing distributions have features not represented in low-dimensional training data.
Relation to Recent Work in Length Generalization:
Question: How does LDHD generalization relate to recent studies on length generalization, such as Ahuja & Mansouri (2024) and Zhou et al. (2024)? Could you provide a deeper comparison of LDHD positions relative to these approaches?
Suggestion: Discussing how LDHD generalization contrasts with or complements other length generalization strategies would provide a clearer contextual foundation. This would also help highlight your framework’s unique contributions, especially regarding scaling challenges and inductive bias requirements.

Addressing these questions and suggestions would clarify the framework’s applicability, limitations, and comparative advantages, helping to deepen the impact and reach of the paper’s contributions.

### Soundness
2

### Presentation
2

### Contribution
3
