# Fairness for the People, by the People: Minority Collective Action

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Machine learning models often preserve biases present in training data, leading to unfair treatment of certain minority groups. Despite an array of existing firm-side bias mitigation techniques, they typically incur utility costs and require organizational buy-in. Recognizing that many such models rely on user-contributed data, we introduce a framework for minority Algorithmic Collective Action, where a coordinated minority group strategically relabels its own data to enhance fairness, without altering the firm’s training process. We propose three practical, model-agnostic strategies to approximate ideal relabeling and validate them on five real-world datasets. Our findings show that with limited participation, a collective can substantially reduce unfairness with a small impact on overall accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a user-side fairness approach where minority users collectively relabel their data (i.e., Algorithmic Collective Action, ACA) to improve fairness without the decision-maker's intervention. It introduces three "model-agnostic" relabeling methods and shows they significantly reduce unfairness with minimal accuracy loss. The work also analyzes theoretical limits when only minority members act collectively. Experiments show the effectiveness and limits of the method.

### Strengths
1. Clarity is great. The paper is generally well-written and easy to understand.
2. The discussions on both how ACA can help with fairness and the limitations of ACA are extensive.
3. The connections to counterfactual fairness make sense to me.
3. The experiments can prove what the authors claim.

### Weaknesses
My main concerns lie in the settings:

1. Though I am familiar with the ACA proposed in [Hardt et al., 2023], I am not sure whether it is practical for the minority group collaborating to enhance the fairness of decision-making. In your design, at least all minority group members should share information and then select some members to flip their labels. Could you outline some examples in reality?

2. I am a bit confused about section 3 of approximating the counterfactual label. The minority group members seem to need to have access to lots of information. For RB-prob, they need to know their scores of being admitted; for RB-label, they need to know their neighbors' labels; for RB-dist, it might be more practical to only know the distances.

- Minor weaknesses on related works:

In fact, the setting formulated is very similar to strategic classification (SC) / performative prediction (PP) where the population move their features in response to the deployed models. Thus, it is worthwhile discussing the literature on these topics and the fairness regarding these settings. In my perspective, you are focusing on a setting where the objective of strategic agents is to enhance the fairness, which can be novel. However, I still feel it necessary to review some prominent works on SC/PP. Examples might be [1,2,3], where [1] and [2] are works proposing these frameworks, and [3] is a recent work on fairness in these settings.

[1] Hardt, Moritz, et al. "Strategic classification." Proceedings of the 2016 ACM conference on innovations in theoretical computer science. 2016.

[2] Perdomo, Juan, et al. "Performative prediction." International Conference on Machine Learning. PMLR, 2020.

[3] Jin, Kun, et al. "Addressing polarization and unfairness in performative prediction." arXiv preprint arXiv:2406.16756 (2024).

### Questions
See weaknesses 1, 2.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a user-driven approach (shift the focus from the firm side to the user side) to improve fairness without needing any cooperation from companies. The idea is that a coordinated group of minority users can work together to relabel their data, assigning the label they would likely receive if they were in the majority group, thereby making the model fairer when trained on such data.

### Strengths
- This paper frames the issue of fairness from a fresh and thought-provoking perspective. Rather than relying on traditional firm-side interventions, it shifts the focus to the user side, exploring how individuals from minority groups can collectively influence fairness without the company’s involvement. This user-driven framing feels both novel and insightful.

- The proposed strategy draws on the concept of counterfactual fairness, asking what the predicted outcome would be if an individual’s sensitive attribute were changed to that of another group. This integration of counterfactual reasoning into a user-led framework is innovative. To make this idea practical, the authors propose three ways to approximate these counterfactual labels, which makes the whole framework much more practical.

### Weaknesses
- The paper defines the minority and majority groups based on a binary sensitive attribute (for example, gender). However, in many real-world cases, bias may arise from other factors such as age, or even from interactions among multiple attributes (e.g., age + race + gender). Since the proposed method heavily relies on the definition of the minority group, it would be important to discuss how such groups should be identified or defined in practice.
- The three relabeling methods appear quite heuristic and may not accurately reflect the true counterfactual distribution $P(y \mid x_{A←0})$. For example, under the KNN-based approaches, if the feature space is high-dimensional, how can a simple Euclidean distance is able to capture meaningful counterfactual similarity.
- The model needs to be retrained after relabeling, which introduces additional computational costs, especially in streaming or continuously updated systems where retraining occurs in each round. In many firm-side applications, models are deployed offline and are not frequently updated. Moreover, when the minority group is very small, flipping only a few labels might have little practical impact. This suggests that the effectiveness of ACA in practice depends heavily on the platform’s architecture and feedback cycle, and this point should also be discussed.
- Trust among individuals could also be a major challenge. Coordinating collective action requires mutual confidence and shared motivation, which may not always exist in decentralized or large-scale user communities.
- My another concern is related about the method's assumption. Can these assumptions be reasonably held or implemented in practice?

### Questions
- In practice, how should the collective determine which users should relabel their data? What criteria or heuristics can ensure that the selected users meaningfully contribute to improving fairness rather than introducing additional noise?
- What does the size of the minority group need to be for the relabeling strategy to actually have a noticeable effect on fairness? Is there any rule of thumb or evidence for the minimum participation needed?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a user-side approach to fairness, where minority users coordinate to relabel their own data to reduce bias in models trained by firms, without requiring access to the training pipeline. The authors formalize this as Algorithmic Collective Action (ACA), link it to counterfactual fairness, and propose three simple, model-agnostic methods (RB-prob, RB-label, RB-dist) to approximate counterfactual relabeling. Experiments on tabular, image, and text datasets show clear fairness gains with minimal accuracy loss. The paper also analyzes theoretical limits of minority-only ACA and provides insight into how representation learning can improve fairness outcomes.

### Strengths
- Overall the paper is well-written and easy to follow.
- Shifts fairness work from firms to users is an interesting idea. The proposed algorithms are intuitive, easy to implement, and computationally efficient, making them appealing for real-world use by non-institutional actors.
- The experiment covers multiple datasets with analyses of Pareto frontiers, label-flip efficiency, and sensitivity to limited majority information.
- Analysis is good to have for understanding when and why minority-only action can or can’t achieve perfect fairness.

### Weaknesses
1. Comparisons are limited to random flipping and a few firm-side methods. I would suggest incorporating more recent pre-processing baselines (e.g., those fair subsampling or reweighting methods from existing packages[1,2]).
2. The concept of users manually relabeling their own data (e.g., resumes, engagement labels) is theoretically interesting but may not always reflect plausible or scalable user behavior. Discussion of realistic incentives, potential harms, or game-theoretic stability would enhance the social grounding.
3. I don't quite get why the waterbird dataset is being used to evaluate fairness. Can the authors offer some explanation for that?

[1] Han, Xiaotian, et al. "FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods." ICLR 2024  
[2] Bellamy, Rachel KE, et al. "AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias." IBM Journal of Research and Development 63.4/5 (2019): 4-1.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the idea of using algorithmic collective action (a framework proposed by Hardt et al. ICML 2023) to approach the problem of group/demographic unfairness in ML models. The paper aptly argues that fairness interventions in research and in practice have been viewed primarily from the firm side, but they pose a different question: what if users from the "minority" group use their “collective action” power to input data to the ML algorithm in a way that improves group fairness metrics (like statistical parity or equalized odds) even if the ML algorithm/firm doesn’t do any interventions itself (which it does not have the incentive to do due to accuracy loss)? The paper shows through numerical experiments on a number of datasets that this specific form of data pre-processing (attained through label flipping) can be effective in making algorithms more fair.

### Strengths
+ The underlying idea holds merit, and is a nice, complementary approach to how fairness has been viewed primarily from firm-side. 
+ The experiments also support that with the right “label flipping”, we can improve fairness.

### Weaknesses
While the idea is interesting, I believe there are a number of technical, practical/logical issues with it, as detailed below, as well as some lack of clarity on contributions.

- One of the main issues with ignoring the firm-side perspective is that the firm, after all, is after predictive accuracy. If the firm notices that some users are sending data to it that is decreasing its accuracy on the “majority” group (which label flipping by the "minority" group will likely do), then the firm will catch on. In other words, yes, this method does not require firm side participation, but the firm can undo the effects if it is decreasing its utility (which it likely will). This has a parallel with the idea of search engine optimization (SEO); in absence of good guardrails, it doesn’t take a lot to game search engines (a small “collective” could alter the algorithm’s recommendation sorting/output to their benefit); search engines know this well, so they have methods in place to counteract SEO. In a similar vein, firms could counteract this proposed type of collective action. 

- Another main source of concern with this proposed approach is in the assumption that users cannot manipulate their features, but they can change their labels. First, why would the users not be able to change their features? If they are applying for a job, they could 100% lie on their resumes about their qualifications. Second, why would the algorithm take self-reported, flipped labels, and accept them, without verifying them? This is also an odd and hard to justify choice. It is also worth noting that there is an ever-growing literature on “strategic machine learning”: users can manipulate the data they send to algorithmic systems, in order to improve the response they get. It is the standard in this literature to assume that the only thing users can alter when reporting to the mechanism is their dataset, and there is plenty of support around why this is the only natural assumption to make. Any changes to the true label/qualification state should come from genuine improvement (not flipped label reports). In the application areas mentioned and the type of datasets used, there is a distinction between the notion of true qualification and labels assigned by an algorithm. Users can only change their true qualifications; algorithms/firms decide what label to assign to each data point, even for training; i.e., labels are not self-reported. With this view, it is hard to reconcile the assumptions (in e.g. lines 138-139, and elsewhere) that users can change their labels (and can’t change their features) with practical settings.  

- On a technical note, the model seems to assume (line 95) that data for the majority and minority group is drawn i.i.d. . But then what is really distinguishing a majority group from a minority group? Also, to be clear, it would be appropriate to distinguish between “disadvantaged” and “advantaged” groups – this is the terminology that would refer to one group having lower qualification rates than others, and is the groups we want to protect through fairness interventions. The term “majority” vs. “minority” refers to data size on groups (this is not what we measure fairness against). The two notions may align (for instance, when using race as the protected attribute in the Adult dataset), but not always (for instance, when using race as the protected attribute in the COMPAS dataset). Also, on a deeper issue, if there is no relation between the feature distributions of the majority/minority group (where I’m using this paper’s terminology) and their label distributions, then why would the signal erasure strategy be relevant?

- In light of the above, the contribution of the paper seems limited. The paper, at top of Section 3, mentions rightly acknowledges that “the theory of signal erasure has been studied before”. On a related note, the claim of “we present the first practical algorithm for signal erasure” was unclear to me. The original ACA paper and its follow ups do present algorithms and establish their effectiveness, so it is not clear what was wrong or impractical about the existing algorithms?

- I am also unclear on the benchmarking done against Kamiran and Calders paper (in appendix E1). What version of CND is being used for baselines? Quoting from the Kamiran and Calders  paper: “We can now reduce the discrimination in the dataset by either promoting objects in CP from class− to + or by demoting objects in CD from class + to− (− represents the class “not +)”. In order to maintain the balance between the two classes, CND will always do both a promotion and a demotion at the same time.” Given that the way labels are flipped is not only on the minority group, comparing the methods across minority group flips does not seem immediately comparable.

### Questions
These question are further detailed in the weakness section. Short versions are included below as a summary. 

1. How can one account for firm-side undoing of the proposed ACA effort? It seems quite reasonable that firms would try to counteract these effects. Would the proposed ACA be robust to that? 

2. Why is it reasonable to assume users can report flipped labels, and that the algorithm will take them as is? What stops users from misreporting features? There is a growing line of literature on strategic machine learning which makes (and motivates) contradictory assumptions to these, with the latter seeming more relevant to/reflective of practical scenarios. 

3. Why is the data assumed to be i.i.d? Will this not preclude the relevance of single erasure strategies? The notions of majority vs. minority when it comes to fairness may need to be revisited or better justified. 

4. Why are the algorithms/methods proposed by prior work not considered practical? What is fundamentally different with the methods proposed here?

5. How is the paper benchmarked against works like Kamiran and Calders? Also, it seems both works (as well as many other pre-processing methods) include ideas around label flipping. Why is it that this label flipping is more effective than the label flipping achievable firm-side (like by Kamiran and Calders-type works)?

### Soundness
3

### Presentation
3

### Contribution
2
