# UOEP: User-Oriented Exploration Policy for Enhancing Long-Term User Experiences in Recommender Systems

- Decision: Reject
- Scores: 3, 8, 5

## Abstract
Reinforcement learning (RL) has gained traction for enhancing user long-term  experiences in recommender systems by effectively exploring users' interests. However, modern recommender system exhibit distinct user behavioral patterns among tens of millions of items, which increases the difficulty of exploration. For example, user behaviors with different activity levels require varying intensity of exploration, while previous studies often overlook this aspect and apply a uniform exploration strategy to all users, which ultimately hurts user experiences in the long run. To address these challenges, we propose User-Oriented Exploration Policy (UOEP), a novel approach facilitating fine-grained exploration among user groups. We first construct a distributional critic which allows policy optimization under varying quantile levels of cumulative reward feedbacks from users, representing user groups with varying activity levels. Guided by this critic, we devise a population of distinct actors aimed at effective and fine-grained exploration within its respective user group. To simultaneously enhance diversity and stability during the exploration process, we further introduce a population-level diversity regularization term and a supervision module. 
Experimental results on public recommendation datasets demonstrate that our approach outperforms all other baselines in terms of long-term performance, validating its user-oriented exploration effectiveness. Meanwhile, further analyses reveal our approach's additional benefits of improved performance for low-activity users as well as increased fairness among users.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the user exploration problem in the recommender system. Different from previous works that adopt a uniform exploration policy for all users, this paper proposes a user-oriented exploration policy to learn different exploration strategies for different types of users. Specifically, this paper applies the risk-averse distributional reinforcement learning to maximize $CVaR_{\alpha}$. Moreover, the authors divide users into different groups based on the quantile score of expected returns and utilize population-based reinforcement learning to learn separate agents to optimize $CVaR_{\alpha}$ with different quantile scores $\alpha$. Experiments are conducted on the recommender simulator based on three real-world datasets.

### Strengths
1.	User exploration in the recommender system is an important problem.
2.	The structure of this paper is well-organized and easy to follow.
3.	The authors evaluate the proposed method using a recommender simulator based on three real-world datasets, which is comprehensive.

### Weaknesses
1.	To design a user-oriented exploration policy for different types of users, the authors divide users into different groups by the $CVaR_{\alpha}$ with $\alpha \in [0.2, 0.4, 0.6, 0.8, 1.0]$. According to Eq. (1), there is a nested relation between these five user groups. For example, the user group with $\alpha = 0.4$ contains the users in the user group with $\alpha = 0.2$. This definition is problematic and will result in a redundancy in policy optimization for different user groups.
2.	The motivation of this paper is to design a separate user-oriented exploration policy for different user groups. However, to my understanding, there is no explicit exploration strategy design for different user groups, and only the optimization objective $CVaR_{\alpha}$ varies for different user groups, which does not necessarily promote user exploration for different groups.
3.	The used evaluation metrics (total reward and Depth) do not validate the effectiveness of exploration. Other exploration-related evaluation metrics such as diversity and coverage are necessary to demonstrate the exploration performance.

### Questions
See the Weaknesses for the questions.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposed a user-oriented exploration policy approach to facilitate fine-grained exploration with respect to user different activity levels. Specifically, it consists of a distributional critic that allows optimization at different quantiles; and a population of actors optimizing towards different return distributions. With several regularization losses to control diversity and stability, it demonstrates the superior performance with the proposal approaches by comparing to several baselines on public datasets.

### Strengths
1. The paper studies an important problem in recommendation system, optimizing user experience with respect to different activity level.
2. The paper is well motivated, and demonstrates to be a superior approach with several baselines and datasets.
3. The paper is clearly written and easy to follow.

### Weaknesses
The proposed approach is similar to an ensemble approach in inference. in the real world, such policy might encounter much more expensive serving cost with millions and even billions of action space, which might prevent itself from its adoption.

Also listed several questions down below.

### Questions
1. How does different quantile correspond to different exploration strengths?
2. Usually, activity levels are defined by the total volume of user engagement (clicks), instead of ctr. So it's possible that users have very few impressions, but high ctr. In that case, these users are still referred to as low-activity users. How does that affect the results?
3. In section 4.4, the paper only reported the superior performance for low-activity users only. However, it would also be good to report that for high-activity users as well.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors proposed the UOEP framework, which is an RL based recommendation system that can customize the exploration intensity for different user activity levels. Specifically, the authors define the activity level of users based on the return distribution under different quantiles and the framework learns multiple actors where each actor corresponds to a specific user group with a predefined level of activity. The authors conduct extensive offline analysis based on 3 public datasets KuaiRand1K, ML1M and RL4RS.

### Strengths
Strength

- The idea of providing different exploration intensity for different user cohorts is very practical and intuitively making sense.

- Proposed algorithms outperform the baselines in various offline analyses. Source code is provided and datasets are public, which make the results easier to reproduce.

- The paper is in general well written

### Weaknesses
I will combine both of my concerns and questions with this paper in this section.

1. Although the user argues "user behaviors with different activity levels require varying intensity of exploration, while previous studies often overlook this aspect and apply a uniform exploration strategy to all users", this is not true.

Exploration for different user activity levels (especially designing exploration strategies for new/cold-start/low-engagement users) are very common projects for industrial recommenders with a lot of existing strategies. In the domain of active learning, there are also a lot of previous works that proposed similar ideas to conduct user-side active learning based on criteria like activity level, popularity, prediction uncertainty etc. These existing works make the core technical contribution of this paper become more incremental.

2. In this paper, "the framework essentially learns multiple actors where each actor predicts for a specific user group with a predefined level of activity", this essentially leads to an increase of effective model size(multiple-actors instead of single actor). How much of the gain comes from a larger model size and how much is coming from a more effective exploration strategy?


3. In the introduction session, the quantile of CTR was used to illustrate the user's activity level. Is this reasonable? For example, in an extreme case, a new user with 1 impression and 1 click will lead to a 100% CTR but the system still knows little about him and needs more intensive exploration. Shouldn't metrics like total number of clicks etc be more suitable in this case?

### Questions
Please refer to the section above

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
