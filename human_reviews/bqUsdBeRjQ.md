# Personalized Language Modeling from Personalized Human Feedback

- Decision: Reject
- Scores: 3, 6, 3, 6

## Abstract
Personalized large language models (LLMs)  are designed to tailor responses to individual user preferences. While Reinforcement Learning from Human Feedback (RLHF) is a commonly used framework for aligning LLMs with human preferences, vanilla RLHF assumes that all human preferences share the same distribution, preventing fine-tuned LLMs from generating personalized content when user preferences are diverse. In this work, we propose Personalized-RLHF (P-RLHF), an efficient framework that utilizes a lightweight user model to capture individual user preferences and jointly learns the user model and the personalized LLM from human feedback. P-RLHF exhibits the following three characteristics: (1) It enables an LLM to generate personalized content and scale efficiently with growing number of users. (2) It handles both explicit user preferences described as textual input and implicit user preferences encoded in the feedback data. (3) It eliminates the need for users to fully articulate their preferences, which are normally needed for prompting LLMs to generate personalized content yet are often impractical to obtain in real-world scenarios. Our experimental results show that personalized LLMs trained using P-RLHF generate responses that are more closely aligned with individual user preferences, outperforming vanilla, non-personalized RLHF and prompting-based personalization approaches across different tasks.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces a framework called Personalized Reinforcement Learning from Human Feedback (P-RLHF) aimed at personalizing large language models (LLMs) based on individual user preferences. The authors propose a lightweight user model that jointly learns user preferences (both explicit and implicit) and the LLM. The key claims are that this framework scales efficiently with increasing users, handles diverse feedback, and reduces the need for explicit user prompts.

### Strengths
- The paper addresses an important area of research—personalizing LLMs based on user feedback.
- The proposal to use both explicit and implicit feedback for personalization is valuable and well-motivated.
- The experiments cover several datasets, which helps validate the general applicability of the approach.

### Weaknesses
- The paper does not sufficiently address how implicit feedback is effectively used or how it handles contradictory user preferences.
- The scalability claims, while central to the paper, are not backed up by detailed computational analysis or resource cost comparisons with other approaches.
- There is limited discussion on how this approach performs in real-world dynamic environments where user preferences might shift over time.
- The experimental results, while positive, are not sufficiently compelling to clearly demonstrate the advantage of P-RLHF over simpler personalization methods.

### Questions
1. How does the model ensure that implicit feedback does not conflict with explicit user preferences, and how does it resolve such conflicts?
2. Can the authors provide more detailed information on the computational efficiency of P-RLHF? How does it compare in terms of time and memory consumption to other RLHF approaches?
3. How does the framework handle dynamic user preferences that may change over time, and how quickly can it adapt?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Personalized Reinforcement Learning from Human Feedback (P-RLHF), a novel framework designed to enhance the personalization capabilities of large language models (LLMs) by integrating both explicit and implicit user models. By leveraging user-specific information and feedback, P-RLHF effectively captures diverse and individualized user preferences, addressing the limitations of traditional RLHF methods that assume uniform user preferences.
The authors validate the efficacy of the P-RLHF framework through a series of comprehensive experiments, ranging from controlled synthetic settings to large-scale real-world datasets. These experiments demonstrate that P-RLHF not only successfully captures and adheres to individual user preferences but also maintains high performance and scalability in diverse application scenarios. The clear and logical presentation of the framework, coupled with intuitive response generation results, underscores the practical value and potential impact of personalized LLMs in real-world applications.
Overall, this paper makes substantial contributions to the field of personalized LLM by presenting a well-structured framework and a novel optimization method that together advances the state-of-the-art in personalized language model training.

### Strengths
1. The paper presents the P-RLHF framework, which integrates both explicit and implicit user models to facilitate personalized learning from user feedback, thereby introducing a novel approach to the research on personalized LLMs. The introduction of P-DPO represents an innovative optimization method that effectively balances personalization and generalization for both known and unknown users. This approach demonstrates excellent scalability while addressing the personalized needs of multiple users, marking a significant advancement over traditional RLHF methods.
2. The overall structure of the paper is exceptionally clear, methodically introducing the background, identifying the problem, proposing solutions, and validating the experiments related to personalized RLHF. The logical progression facilitates easy comprehension for readers. Each experimental section is meticulously designed with clearly defined objectives, enabling readers to effortlessly track the innovative aspects of the methodology and their corresponding outcomes.
3. The response generation results provided in the paper offer a clear and intuitive demonstration of how different user preferences are handled in a personalized manner. The personalized LLMs showcased in the study hold significant practical value, and the paper successfully achieves personalized content generation on the utilized datasets. This effectively highlights the framework’s capability to cater to diverse user preferences in real-world applications.

### Weaknesses
1. Zero-Length Responses in Experiment One: The occurrence of zero-length responses, which the authors justify as an expected outcome via mathematical proofs, raises questions about whether an LLM should indeed produce responses with no content. This result appears to stem from a polarized experimental design and methodology, which may not reflect practical application scenarios or user expectations. A more nuanced experimental setup could better balance realistic use cases with the need for distinct preference modeling.
2. Figure 1 Lacks Identification of Key Modules: While Figure 1 presents the overall framework, it does not clearly indicate critical components introduced by the authors, such as P-DPO. And there is no explanation of Figure 1 in the main text. This omission complicates the interpretation of the framework diagram, hindering the reader’s ability to intuitively grasp the structure and contributions of the proposed approach.
3. Insufficient Ablation Study: The ablation experiments, which appear only in the appendix, focus solely on parameter variations rather than examining which specific design choices contribute to the improved performance over baseline models. For example, it is unclear whether the personalized capabilities remain robust when only the implicit model is used without the explicit component. The paper lacks a thorough investigation into such aspects, leaving questions about the impact of each module unanswered.
4. Binary Nature of Evaluation Metrics: Both the preference modeling and experimental evaluation adopt a highly binary approach. In modeling preferences, options are presented as direct opposites (e.g., long vs. short responses), which likely contributes to the appearance of zero-length responses. Furthermore, evaluation is limited to win-rates relative to baseline models, which does not provide a clear view of the actual performance gains. Since the goal of personalization is to improve user satisfaction, more nuanced analyses, rather than a strictly binary approach to the generated content, are necessary to substantiate the effectiveness of the approach.
5. Lack of Consideration for Contextual Shifts in User Preferences: User preferences can be context-dependent, particularly when involving multidimensional combinations of preferences. In specific scenarios, user preferences may shift according to immediate needs or context. The reliance on a static dataset overlooks these potential dynamics, and the paper does not address how such variations might impact model performance.

### Questions
1. An explanation is requested for the occurrence of zero-length responses presented in Section 5.2 and Figure 3. Logically, even for users who prefer concise replies, it is expected that responses maintain a reasonable lower bound on length rather than resulting in either no content or excessively long responses. The current approach appears overly polarized and does not account for practical application scenarios where balanced response lengths are desired.
2. It is recommended that Figure 1, which outlines the proposed module design, be incorporated directly into the main body of the text. This integration would enhance the comprehensibility of the framework by ensuring that the figure is closely tied to the corresponding narrative, thereby making the overall framework easier to understand for readers.
3. I suggest the author add more nuanced analyses, rather than a strictly binary approach to the generated content, which are necessary to substantiate the effectiveness of the approach.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper addresses the important challenge of generalizing large language models (LLMs) to capture personalized user preferences. To achieve this, the authors develop several models that incorporate users' implicit preferences, along with a new learning objective aimed at optimizing the reference LLMs for personalization. They compare the proposed method, P-DPO, with the standard DPO in both synthetic and human evaluation settings. In both cases, P-DPO shows advantages. The paper is well-written, with a clearly motivated problem statement. The discussion on the limitations of existing RLHF methods is thorough and insightful.

However, the reviewer has the following concerns about this work.
1. The contribution appears limited. The primary innovation, the new learning objective P-DPO introduced in Section 4.4, facilitates personalization by incorporating user embeddings into the implicit reward model of DPO. However, this approach is not particularly novel, as the integration of user embeddings for personalization is extensively discussed in the recommendation literature [1].
2. The evaluation lacks comprehensiveness. The authors only consider DPO as a baseline and do not include recent work on personalized LLMs, such as [2] and [3]. Additional relevant studies can be found at https://github.com/HqWu-HITCS/Awesome-Personalized-LLM. Furthermore, the experiments overlook the recommendation task, which is a crucial application for personalized LLMs. In the human evaluation, the sample size is limited to at most 20 users and 76 samples, which may lead to biased results and unreliable conclusions.
3. The advantages of integrating both users' explicit and implicit embeddings are not well supported. The authors are suggested to include an ablation study to validate the importance of the learned implicit user embeddings.
4. The motivation for including the user-identifier-agnostic loss term in P-DPO is not clear to the reviewer. The advantage of this design is also not evident.

[1] SSE-PT: Sequential Recommendation Via Personalized Transformer
[2] Factual and Tailored Recommendation Endorsements using Language Models and Reinforcement Learning
[3] Personalized Large Language Models

### Strengths
1. The issues related to DPO in personalized modeling are thoroughly discussed.
2. The presentation is easy to follow and well-structured.
3. In both synthetic and human evaluations, P-DPO significantly outperforms DPO.

### Weaknesses
For a detailed discussion of the reviewers' concerns, please refer to the summary.

### Questions
1. Why include the user-identifier-agnostic loss term in P-DPO?

### Soundness
2

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
3

### Summary
The paper introduces Personalized-RLHF (P-RLHF), an innovative framework designed to enhance personalized content generation in large language models (LLMs) by leveraging a lightweight user model. Its key contributions include:
Scalability: P-RLHF allows LLMs to efficiently generate personalized content while accommodating a growing user base.
Preference Handling: The framework effectively processes both explicit user preferences, provided as textual input, and implicit preferences derived from feedback data.
Reduced User Burden: It alleviates the requirement for users to fully articulate their preferences, addressing the challenges of obtaining detailed user input in practical applications.

Overall, this paper is easy to follow, but there are issues such as a lack of innovation, no released code, and an excess of symbols. However, overall, this is a ready piece of work.

### Strengths
* The structure of this article is well-organized and clearly articulated, making it easy for readers to follow the flow of ideas and concepts presented throughout the text. Each section is logically arranged, allowing for a seamless understanding of the material.

* The section on RLHF is detailed, with a clear progression from general RLHF to the specifically designed personalized DPO. This work considers different user preferences from multiple angles and granularities. It clearly explains the differences between P-RLHF and Vanilla DPO to clarify the motivation.

* The work uses the TL;DR dataset, which has Reddit posts, summaries, and worker IDs. It conducts an experiment where most workers (70%) prefer longer responses and a smaller group (30%) prefers shorter ones. The top 10 workers with the most annotations were included for training. After the experiment, some workers liked shorter responses while others liked longer ones. Overall, this shows a strong effort and good understanding of data analysis, which is impressive.

* There is a solid exploration of how personalized LLMs can adapt to different user preferences through P-DPO fine-tuning. The study found that the personalized model can generate longer or shorter responses based on users' implicit preferences, demonstrating its flexibility and adaptability. The experiment also compared the performance of P-DPO using generic user embeddings with traditional vanilla DPO, showing that the personalized model excels in understanding and responding to user needs, validating its effectiveness in practical applications.

### Weaknesses
* It seems that this article does not have released code. I'm not sure if I just couldn't find it, but without released code, reproducibility cannot be guaranteed.

* Like other RL works, the notation in this paper is too numerous and complex. Although the structure and presentation of the article are good, it somewhat hinders the readers' understanding. Perhaps a table to organize the notations could be helpful.

* There are concerns regarding the effectiveness of not training an additional reward model and relying solely on the user model.

### Questions
* Does this work have released code and dataset?
* There are works have been done on personalized user preferences using RLHF. How does this one differ from them, and is there a comparison used as a baseline?

### Soundness
3

### Presentation
3

### Contribution
3
