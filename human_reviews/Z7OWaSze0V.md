# Unifying User Preferences and Critic Opinions: A Multi-View Cross-Domain Item-sharing Recommender System

- Decision: Reject
- Scores: 3, 5, 5, 5

## Abstract
Traditional cross-domain recommender systems often assume user overlap and similar user behavior across domains. However, these presumptions may not always hold true in real-world situations. In this paper, we explore an less explored but practical scenario: cross-domain recommendation with distinct user groups, sharing only item-specific data. Specifically, we consider user and critic review scenarios. Critic reviews, typically from professional media outlets, provide expert and objective perspectives, while user reviews offer personalized insights based on individual experiences. The challenge lies in leveraging critic expertise to enhance personalized user recommendations without sharing user data. To tackle this, we propose a Multi-View Cross-domain Item-sharing Recommendation (MCIR) framework that synergizes user preferences with critic opinions. We develop separate embedding networks for users and critics. The user-rating network leverage a variational autoencoder to capture user scoring embeddings, while the user-review network use pretrained text embeddings to obtain user commentary embeddings. In contrast, critic network utilize multi-task learning to derive insights from critic ratings and reviews. Further, we use Graph Convolutional Network layers to gather neighborhood information from the user-critic-item graph, and implement an attentive integration mechanism and cross-view contrastive learning mechanism to align embeddings across different views. Real-world dataset experiments validate the effectiveness of the proposed MCIR framework, demonstrating its superiority over many state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors propose a Multi-View Cross-domain Item-sharing Recommendation (MCIR) framework that synergizes user preferences with critic opinions. The proposed MCIR achieves state-of-the-art performance on several real-world datasets.

### Strengths
- Utilizing critic review information for knowledge sharing across domains is interesting. 

- The proposed MCIR achieves state-of-the-art performance on several real-world datasets.

### Weaknesses
- There are many incorrect formula details in the article. For example, the authors obtain covariance $\Sigma$ via $W_T’h_T + b_T’ = diag(\Sigma_t)$. However, $\Sigma$ should be positive values. How to guarantee the output of $W_T’h_T + b_T’$ could be always positive? 

- Some model details are missing. For example, the authors adopt $g(\cdot)$ as the activation function in Eq.(1). However, what kind of activation function do you use in the experiments? ReLU or Sigmoid? Moreover, the authors adopt the dropout layer in the Eq.(1). How about the dropout ratio for this layer? 

- Some formulas are completely wrong. For example, the authors obtain user distribution as $p(u_i|R_i) \sim N(\mu_i, \Sigma_i)$. However, the reparameterization process in the paper is $u_i = \mu_i+\epsilon \Sigma_i$. It is completely wrong. The correct answer is $u_i = \mu_i+\epsilon \sqrt{\Sigma_i}$.

- The methodology is hard to follow with too many notations. I strongly encourage the authors to provide the pseudo algorithm table.

- Some important baselines are missing, e.g., PTUPCDR.

- The authors emphasize that critic reviews are much more valuable than common reviews. However, how to define whether the reviews are critic or not? 

- The dataset statistics are missing key information, e.g., number of overlapped users.

- Can the proposed method handle different ratios of overlapped users (e.g., only 10% users are overlapped across domains)?

======
Update: 

I acknowledge that I have read the authors response. Although the idea of this paper is interesting, it still needs major revision on technically details (e.g., formula and symbol corrections) to make it more precise.

### Questions
Please refer to the Weakness above.

### Soundness
2 fair

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors introduced a multi-view cross-domain item-sharing recommendation algorithm to involve critic comment from users. They involved many techniques (e.g., GCN and contrastive learning etc) to obtain the user item aligned embeddings. Various baseline methods and datasets were investigated in the experiments. Results show the advantages of the proposed.

### Strengths
S1. critic comment is a good idea to enhance recommendation in general

S2. sufficient baselines were selected in experiments

S3. Comprehensive experiments were conducted

### Weaknesses
W1. hard to follow the methodology

W2. unclear definitions

W3. critic comment is the one of the main contribution but it's not clear about how to define and how to detect critic comments

### Questions
Q1. It's hard to follow the methodology sections. Figure 2 doesn't help make it clearer. Instead it makes it more complicated to understand without knowing the meaning of letters and captions. I hope reading through text would help my understanding. However I still don't know why many steps are necessary and why so much components and techniques are required. For example, for definitions, what user-rating network, user-comment network, and critic embedding network. It seems that the final goal is to obtain user and item embeddings aligned in the same latent space. But by nature critic comment is hard to define and detect (please refer to the following question). It seems that contrastive loss and GCN are also included, but by checking Figure 2 are the left and right boxes decoupled? Do they have relationship? And what's their relationship?

Q2. “Experts Write More Complex and Detached Reviews, While Amateurs More Accessible and Emotional Ones” How to know which review is written by an expert or an amateur? Or how to quantify the critic metric for a comment. The statement is also related to Section "Critic Embedding Network" where it mentioned critic rating prediction task. It seems to be based on the inner product of v_j and w_l^c. What does the latent critic-rating vector w_j^c come from?

==========================
I acknowledge that I have read the authors response. I appreciate the authors efforts. But it didn't address my concerns. I would keep my original rating.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors proposed to explore a less explored scenario: cross-domain recommendation with distinct user groups, sharing only item-specific data. Towards this end, they proposed a multi-view cross-domain item-sharing recommendation framework that leverages user reviews, critic comments, item summaries, user ratings, and critic ratings. They collected a dataset with three domains, namely Game, Movie, and Music from Metacritic and compared with multiple baselines.

### Strengths
1. The cross-domain recommendation problem with distinct users across domains is an interesting yet overlooked problem. The authors took a look at this problem and proposed a complicated model that uses multiple types of data to solve it.
2. The proposed model showed good performance under the authors' setting and outperformed all baselines. Modeling-wise, this provides some insight into what is worth trying and effective in similar problems and can inspire more innovative solutions.

### Weaknesses
1. First of all, the authors did not do a good job of clearly formulating the problem they want to solve. I felt confused after I went over the paper for the first two times. When I first read it, I thought they were trying to do cross-domain recommendations when different domains shared items but not users, which is counter-intuitive. Then I realized it's critic that is shared by different domains. I recommend the authors state this very explicitly and use some space to formulate the problem using some formulas. Besides, the phrase "item-sharing" in the title is quite misleading.
2. In experiments, did the authors use data from two datasets for training and then predict the rating for the 3rd dataset? How does the evaluation of "Cross-domain" recommendation work in this paper? I did not quite understand after reading Section 4.1.
3. Besides the experiment section, this paper also needs more clarity in its description of the proposed model can be further improved to get better readability. Questions related to this can be found in the next section.

### Questions
1. Shouldn't the $y_{lj}$ in the left up corner of Figure 2 be $y^c_{lj}$?
2. In the user embedding network, why is $R_i$ used to construct $u_i$ instead of the other way around, using $u_i$ to generate $R_i$? The authors may have their rationale for designing the model in this way. But they failed to explain it clearly to the readers.
3. The explanation of the "attentive integrated mechanism" seems to be over-complicated to me. It actually follows the standard design of the attention mechanism with $v_j$ as the query vector and $w^c_l$ as the key and value vector. The introduction of $L_j$ and $w^c_0 = v_j$ does not seem to be necessary and only added complexity.
4. In the same section, the first sentence says "Given that only the items are shared between the critic and user domains". I believe the word "domain" does not mean the cross-domain studied in this paper is trying to cross the "user" and "critic" domains.  Am I right?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper builds a cross domain recommender system that leverages ratings and reviews from users belonging to different groups, and item descriptions. In particular, the different groups share no common users while makes the problem of information transfer across them more difficult. The paper builds an elaborate system with multiple components to overcome this challenge and obtains SOTA results.

### Strengths
1. The problem is well motivated 
2. The paper is written clearly
3. The results beat SOTA by a good margin on the given datasets

### Weaknesses
The centra weakness is the lack of a detailed study of its individual components. For a system, that has as many components as this one, it is crucial to measure the relative importance of each. While the paper has some study ablation studies I do not think they are exhaustive (please see questions below). For instance,

1. How useful is the attentive module? For instance what if all $v^a_j = 0$?
2. How useful is the graph network $\left(\eta_3 = \eta_4 = 0\right)$  ?
3. Why not also use a VAE in the critic embedding network for training $u_l^c$. Conversely why not use the (analogous) first term in equation of 5 in equation 3, and just not use the VAE at all? Could you please elaborate.


It is difficult to assess the impact of the proposed method (in terms of scope, generalizability, etc.) without understanding the value of its components.

### Questions
1. What are $\eta_n,\eta_s, \eta_c$? From the context I seem to gather that $\eta_n = \eta_4$?

2. What does ablation study C1 mean? Does that mean neither critic comments or user comments or item summary text are used? 

3. Similarly, what does C2 mean?

4. Does C3 mean making $\mathcal L_{Multi} = 0, v^a_j = 0$ and excising all the critic-item edges from the graph?

5. Does C4 mean, $\eta_4 =0 $?

6. How does the proposed method compare with the best prior methods (say BitGCF, EMCDR, SSCDR) in terms of training time, number of parameters, and inference cost?

7. [Minor] In equation 7, one could catenate and then project? Do you think that could lead to substantial gains?  

8. [Minor] Is there a reason the decoder network is chosen to be as simple as eq. 8? Needless to say, simple is good. But wondering if there are other motivations.

9. [Broad] It seems there are existing datasets used by prior baselines (ML-NF dataset). Is there a reason for not choosing it over Metacritic (or for that matter, using both)?

For all the ablation questions, please answer in terms of what happens to equation 14, wherever possible.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair
