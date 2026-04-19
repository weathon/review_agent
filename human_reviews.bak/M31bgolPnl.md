# Adaptive dense reward:Understanding the Gap Between Action and Reward Space in Alignment

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 3

## Abstract
Reinforcement Learning from Human Feedback (RLHF) has proven highly effective in aligning Large Language Models (LLMs) with human preferences. However, the original RLHF typically optimizes under an overall reward, which can lead to a suboptimal learning process. This limitation stems from RLHF's lack of awareness regarding which specific tokens should be reinforced or suppressed. Moreover, conflicts in supervision can arise, for instance, when a chosen response includes erroneous tokens, while a rejected response contains accurate elements. To rectify these shortcomings, increasing dense reward methods, such as step-wise and token-wise RLHF, have been proposed. However, these existing methods are limited to specific tasks (like mathematics). In this paper, we propose the "Adaptive Message-wise RLHF" method, which robustly applies to various tasks. By defining pivot tokens as key indicators, our approach adaptively identifies essential information and converts sequence-level supervision into fine-grained, subsequence-level supervision. This aligns the density of rewards and action spaces more closely with the information density of the input. Experiments demonstrate that our method can be integrated into various training methods, significantly mitigating hallucinations and catastrophic forgetting problems, while outperforming other methods on multiple evaluation metrics. Our method improves the success rate on adversarial samples by 10\% compared to the sample-wise approach, and achieves a 1.3\% improvement on evaluation benchmarks such as MMLU, GSM8K, HumanEval, etc.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces Adaptive Message-wise RLHF, a method enhancing reinforcement learning in language models by dynamically adjusting reward signals at the subsequence level.
The method is introduced to improve alignment accuracy and mitigate issues like hallucinations and forgetting across diverse tasks.
Experiments are conducted on multiple datasets.

### Strengths
The method is evaluated on a variety of benchmarks.

### Weaknesses
in lines 272 - 274, the authors claim "Bradley terry model can be represented as an MDP", this is incorrect.

the paper needs spellcheck for e.g., typos or missing spaces. For a few examples: line 76, 292, 298, equation 13. 

missing references when citing existing papers. The authors should not assume their readers to find references to the terms/concepts needed in their paper.

Several papers have been cited multiple times: step-dpo, token-level dpo, Uesato et al. , Lightman et al.,  and helpsteer.

Figure 2-4 are not referred to in the main text. I failed to understand the messages of those figures.

Some notations are redundant: e.g., y_w, y_l, y_c, y_r

The paper's quality can be largely improved by improving the presentation.

Citing HER in the section of related work is a bit wild. There are many other related works to cite in the field of better credit assignments in RL.

### Questions
Why are some experiments truncated? Can the authors explain their efforts that could guarantee the comparisons are fair to all algorithms? 

How many repeated runs do the authors conduct in concluding the results? Are the improvements statistically significant?

How do the authors determine the values of delta? 

Is the b learnable? How do the authors implement this learnable parameter to stabilize the learning process? --- Learning with a learnable parameter sounds like shooting a moving target, and would be challenging intuitively. 

Equation 11 and equation 16 present different possible values for M.

In Table 1, could the authors also compare with other methods like step-dpo, token-dpo?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper addresses the RLHF (Reinforcement Learning with Human Feedback) task in Large Language Models (LLMs). The authors identify the limitations of the reward model, particularly its inaccuracies when responses contain erroneous tokens. They introduce a method called adaptive message-wise RLHF, which detects essential information and employs a mask matrix to enhance training. Experimental results in certain scenarios demonstrate the proposed method's effectiveness.

### Strengths
1.	The authors present an approach to overcoming limitations in RLHF.

2.	The method demonstrates efficiency in specific scenarios.

### Weaknesses
1.	Limited Contribution: The main contribution, calculating the mask value for each instance in Eq. 11, seems insufficiently significant.

2.	Questionable Effectiveness:

    a.	The preference dataset might contain biased data, potentially causing spurious correlations between the reward and outcome, which the proposed method might exacerbate.

    b.	Eq. 11 may not accurately identify the essential parts of the response, possibly disrupting training.

    c.	Eqs. 12 and 13 might overlook parts of the responses, leading to a loss of fluency and coherence.

3.	Poor Writing Quality: The paper's writing is subpar. For instance, Section 2 includes excessive, unnecessary related work.

4.	Inconsistent Notations: For instance, $y$ represents the whole response in Eq. 6 but seems to denote a token in Eq. 11. Similar confusion exists between $y_w$ in Eq. 6 and $y_{win}$ in Eq. 8.

5.	Unimpressive Experimental Results: The results lack significance. Only Qwen2-7b-instruct was used for evaluation, and more models need testing.

6.	Typos and Errors: There are typographical errors, such as $\pi\theta$

### Questions
1.	What does the title imply? Could you explain the concept of the gap between action and reward space alignment?

2.	How is the threshold b in Eq. 11 determined? Since it's adaptive, how can it be optimized in real-world applications?

3.	Have you experimented with alternative model architectures?

4.	What are the additional computational costs associated with this method?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents idea of assigning scalar rewards to subsequences of LLM-generated text. The main idea behind their methodology is to find a mask over the text, a mask to hide negative aspects of a winning answer and positive aspects of a losing answer in a preference learning setting. Then use the masked sequences for doing Reinforcement Learning from Human Feedback (RLHF) with (1) standard reward modeling based RLHF, (2) implicit RLHF with direct preference optimization (DPO) and (3) rejection sampling variant of RLHF. Token-level masks are generated by using a reference value  $b$ and a neutral region parameter $\delta$, if the token's reward is more than $b + \delta$ or is less than $b-\delta$, accordingly +1 or -1 mask is generated, otherwise is set to zero. Since the token level rewards are different for different sentences, mask gets adapted per sentence. The paper then applies masked variants of RLHF algorithms for tasks like language generation, reasoning, code, etc. The results demonstrate incremental improvements on various benchmarks.

### Strengths
The issues behind the sentence-level reward assignment and token-level reward assignment in RLHF are pointed out aptly by the paper: sentence-level being too coarse while token-level being prone to high variance. Thus, the paper's investigation on finding appropriate message-level techniques for reward assignment is highly relevant.

The strength of the technique is in its simplicity. Idea of using masks for inducing better reward assignments to token-level RLHF in order to reduce the variance is neat. Being simple, the technique finds easy pluggable application to different RLHF paradigms. 

The experimental setup tests the idea on multiple benchmarks, and the proposed algorithm shows improvements across most of them. I am unsure about the significance of these improvements as they come off rather incremental, except for the Human Eval dataset.

### Weaknesses
Lack of depth in the proposed algorithm: Although simple, the technique used in the paper lacks depth. Schmitt trigger approach used in the paper is not referenced properly and hence very hard to assign any significance to the particular idea. Masking with neutral region is a straightforward rule and does not need any control theory unless the particular ablations are shown on other masking techniques.

Statistical significance of the results: The plots in the paper do not use multiple random seeds, this makes it hard to gauge the effectiveness of the technique. Further, the table 1 of the main results does not show any mean and standard-deviation analysis of the values, and the statistical superiority of the results cannot be confirmed.

The obscurity in the writing of the paper: 
- Lines 46-48: Not clear what is meant by sample level supervision, and why it leads to any inconsistency between rewards and action spaces.
- In multiple places, new sentence starts just after a period.
- ORM and PRM terms are used, again, without any introduction for these terms. Also, the whole introduction section seems loosely assembled without making any logical flow. I had to understand the paper by going through entire text and then seeing things in the context of paragraphs appearing later in the paper.
- What is the purpose behind figure 2? What do the colors used on the text block mean? Also, what is the conclusion of the plot on the right? Why does the plot have two dotted yellow boxes?
- Figure 3 requires additional explanations -- clarifying why not mask both the positive and the negative aspects of any answer with +1/-1. The paper currently proposes only to mask the negative aspect in a positive sentence and positive aspect of a negative sentence, but then in equation 16, proposes a tertiary masking instead of binary.
- Figure 4, I am unsure how to read this figure to understand Schmitt triggers. 
- What is difference between equation 6 and equation 8? Aren't both showing the same thing unless $y_{win}$ is defined separately?

### Questions
Questions: 

- What are the authors referring to by "understanding gap between action and reward space in alignment" in the title? 

- "Human Eval et al." do you mean "Human Eval etc."?

Additional suggestions:
- A thorough pass through the introduction section would be highly beneficial for this paper.
- Section 2.3 on contextual dueling bandits is never used in any methodological development in the paper and thus can be removed from the main paper.

- Use of multiple seeds for the experiments would be highly recommended as it will give statistical strength to the results demonstrated. 

- Also, figures 6 and 7 would benefit by labeling the axes properly. The plots have very small legends, do not have any information on x or y axes about what is to be understood.

- Please add control theory related references in the related section and a discussion on importance of Schmitt triggers or ablations on masking techniques which show the one used in the paper is superior. Also, I am not convinced about author's claims of using control theory in this paper as it is not clearly supported in theory, experiments or references.

- Please rewrite the equations in the preliminary section, the distribution over which expectation is taken are not properly formatted. 


Overall, I find that the results of this paper are its main strength. The writing of this paper needs significant changes especially the introduction needs to be more lucid. Also, other aspects of formatting can be paid attention to. I am convinced to reject the current draft of this idea.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents "Adaptive Message-wise RLHF," a new method to improve Reinforcement Learning from Human Feedback (RLHF) by focusing on specific parts of a sequence rather than an overall reward. Traditional RLHF can be inefficient because it doesn’t recognize which tokens should be emphasized or downplayed. The new method identifies key “pivot tokens” to adaptively target important information, breaking supervision down to a fine-grained, subsequence level. This approach works well across different training methods, like PPO, DPO, and rejection sampling, and adapts to a variety of tasks. Experiments show it reduces issues like hallucinations and forgetting, achieving a 10% improvement on adversarial samples and 1.6% on benchmarks like MMLU and GSM8K. Future plans include refining subsequence handling and exploring control theory to strengthen this framework.

### Strengths
- The proposed method is conceptually simple, intuitive, and easy to implement.

- The idea of introducing a Schmitt trigger to ensure the stability of the reward signal is innovative.

- The model’s performance was thoroughly validated through large-scale experiments, considering various benchmarks, evaluation metrics, and training strategies.

### Weaknesses
- **Attention to writing quality is needed:**
  - There are numerous instances where spacing is incorrect.
    - For example, in the caption for Figure 1, there appear to be missing spaces after periods or commas.
    - In Section 4.1, there is a missing space after the paragraph title as well.

- **The paper could benefit from separate sections for literature review and problem definition.** Currently, both are included in the introduction section, resulting in insufficient review and imprecise problem definitions.
  - This study aims to align the action space with the reward space by putting selective attention on useful tokens. Accordingly, I recommend the authors provide a formal and explicit definition of alignment and derive how token selection could improve this alignment.
  - Regarding the formal definition of "alignment," reviewing the following literature may provide some useful insights:
    - Wang, T., & Isola, P. (2020, November). *Understanding contrastive representation learning through alignment and uniformity on the hypersphere*. In International conference on machine learning (pp. 9929-9939). PMLR.
    - Wang, C., Yu, Y., Ma, W., Zhang, M., Chen, C., Liu, Y., & Ma, S. (2022, August). *Towards representation alignment and uniformity in collaborative filtering*. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining (pp. 1816-1825).

- **Consistent errors in subscript formatting were observed in equations.**
  - For example, in equations (4), (6), (7), (8), and (9) on page 4, subscripts are consistently misformatted. Additionally, a thorough review for typos in other parts of the manuscript is recommended before submission.

- **The theoretical foundation appears weak.** The strengths of the proposed masking-based method compared to sequence-level methods (e.g., learning efficiency, nuanced and coherent generation) require rigorous explanation and justification.
  - The completeness of the paper would be enhanced if mathematical proofs were provided to demonstrate how, compared to sequence-level methods, introducing Eq. (16) reduces the variance of the reward signal, thereby preventing reward over-optimization.
  - For example, let’s say $\mathbf{r} \in \mathbb{R}$ is a sequence-level reward while $r_{t} \in \mathbb{R}$ is a token-level reward at time step $t$. Given $\mathbf{r} = \frac{1}{T}\sum_{t=1}^{T}r_{t}$, showing $\text{Var}(\mathbf{r}) = \mathbb{E}[\mathbf{r} - \mathbb{E}[\mathbf{r}]]^{2} > \mathbb{E}\_{\substack{ (s, a) \sim \pi\_{\theta} \\\ r=R(s, a) \\\ \hat{r}=r\cdot M(s,a) } }[\hat{r}\_{t} - \mathbb{E}[\hat{r}_{t}] ]^{2}$ would be one possible approach.

- **Visual content preparation is lacking.**
  - For example, Figure 2 on page 2 has significant wasted space. Adjusting the size of the figure titled "Dense Reward Comparison" so that the top and bottom lines align with those of the prompt box on the left, and increasing the font size in the legend, would help eliminate unnecessary white space.
  - Additionally, the text in the legends of Figures 5, 6, and 7 is very small, making it difficult to read. Increasing the font size will enhance readability.

### Questions
- On line 157, page 3, the statement "$\textit{``This approach not only enhances learning efficiency but also potentially leads to more coherent and contextually appropriate language generation."}$" is made. What theoretical or empirical evidence supports this claim?

- On page 5, it is stated that "$\textit{``token-level optimization enables more nuanced preference capture and potentially more coherent outputs"}$." What theoretical or empirical evidence supports this claim?

- On line 300, page 6, "$\textit{dynamically updates the threshold}$" is mentioned. On line 313, this threshold appears to refer to the baseline value $b$. However, there is no explanation in the text on how this $b$ is adaptively updated. Could you clarify the update process?

- Why were the PRM800K and Helpsteer datasets selected? Could you explain the reason why only those datasets were used despite there exist well-known benchmark datasets for RLHF studies, e.g., Anthrophic-3H dataset.

### Soundness
2

### Presentation
1

### Contribution
2
