# GeoArena: An Open Platform for Benchmarking Large Vision-language Models on WorldWide Image Geolocalization

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2

## Abstract
Image geolocalization aims to predict the geographic location of images captured anywhere on Earth, but its global nature presents significant challenges. Current evaluation methodologies suffer from two major limitations. First, static datasets: advanced approaches often rely on large vision-language models (LVLMs) to predict image locations, yet these models are frequently pretrained on the test datasets, compromising the accuracy of evaluating a model's actual geolocalization capability. Second, existing metrics primarily rely on exact geographic coordinates to assess predictions, which not only neglects the reasoning process but also raises privacy concerns when user-level location data is required. To address these issues, we propose GeoArena, a first open platform for evaluating LVLMs on worldwide image geolocalization tasks, offering true in-the-wild and user-preference-based benchmarking. GeoArena enables users to upload in-the-wild images for a more diverse evaluation corpus, and it leverages pairwise human judgments to determine which model output better aligns with human expectations. Our platform has been deployed online for three months, during which we collected over thousands voting records. Based on this data, we conduct a detailed analysis and establish a leaderboard of different LVLMs on the image geolocalization task.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a benchmarking platform, called GeoArena, for evaluating LVLMs on global image geolocalization tasks. This platform was deployed online for three months, and collected thousands of voting records for the subsequent ranking and detailed analysis. Notably, GeoArena stands from a new perspective, which differs from the existing benchmarks, using the user-upload images and user-preference-based evaluation to address the issues of data leakage and privacy.

### Strengths
- This paper is well-written and easy to follow
- The perspective presented in this work is novel and interesting
- The detailed analyses are insightful and valuable

### Weaknesses
Although I personally find the perspective of this paper quite compelling, I would like to raise a few concerns and look forward to the authors’ responses:
- The design of user-uploaded images is motivated by the potential data leakage in pretraining datasets. However, how do you ensure that a user-uploaded image was actually taken by the user themselves, rather than selected from online sources? Is there any verification or filtering mechanism in place to prevent such cases?
- The paper states that GeoArena collected thousands of voting records over three months, but does not specify the number of participating users. Could the authors provide more detailed statistics about the voting data, particularly the number of unique users? If a single user contributed the majority of the uploaded images and corresponding votes, this could introduce significant bias into the collected data
- This benchmark is built on user preference and does not include prediction accuracy. I still have concerns about voting (the sole evaluation method), because the task of geolocalization has a ground truth. If we neglect this important metric, what is the significance of performing geolocalization —merely to describe the scene? I would like to hear the authors’ discussion on this point, as it is directly related to the motivation of the paper
- In Table 3, the ranking results show that GPT‑4.1 mini outperforms GPT‑4.1, which appears inconsistent with the official API documentation describing GPT‑4.1 as more capable. Could the authors provide an explanation for this observation?

### Questions
Given some remaining concerns, I am currently giving a borderline accept score. I would like to see the authors’ rebuttal before deciding whether to raise my score.

### Soundness
3

### Presentation
3

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
The paper presents GeoArena, a platform to perform blind pairwise comparisons between two VLMs randomly sorted from a pool of VLMs. The platform has been online for months, and has collected over a thousand votes. These votes, including the input images, prompts, and voting data, will be made publicly available

### Strengths
The idea is new. I am not aware of any similar existing platform where models can be evaluated solely on their geolocalization capabilities.
The data might be useful for certain tasks, although not clear for which tasks, and it is not clear if it can be released.

### Weaknesses
1) The platform does not measure geolocalization accuracy, but measures how likely a human is to choose a VLM's (potentially wrong) answer related to geolocalization. It is unclear how this can be useful. While evaluating VLMs on geolocalization accuracy is useful (there are many papers on this subject), and evaluating VLMs on human helpfulness is useful as well (which is what LMArena does), it is not clear how combining the two things can be useful, because it provides a mixed signal which might be uncorrelated to its geolocalization capability. Table 5 suggests that users prefer longer responses, with a coefficient β of 0.526, but there is no indication that higher scores (or longer responses) actually correspond to better geolocalization capabilities. To summarize, I don't see what conclusions I can draw by seeing the leaderboard: a better model in the leaderboard will be better at making the user think that it found the right place? How is this useful to the research community?

2) GeoArena-1K is composed of images uploaded by users and their preference response. Can this data be publicly released? Have the users been asked for consent? I could not find a link to GeoArena in the paper hence I couldn't try it myself to see if there is a button to consent to the release of my preference data and uploaded images. If not, that data can't be released.

3) The prompt is provided by the user, meaning that the user could easily ask anything about an image. Is there anything that ensures that the platform is constrained to geolocalization, or can the platform be used for any kind of image-text "battle" between two VLMs?
Line 049 says that a limitation of GPS-based evaluation is that existing evaluations only consider the spatial distance between the final prediction and the ground truth, ignoring the model’s reasoning process. Why is that a limitation? In most cases the user would only care about having an accurate geolocalization

### Questions
Questions included in weaknesses above

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
3

### Summary
This paper proposes a dynamic, user-preference-based benchmarking platform for image geolocalization. The platform allows users to upload their own images, provide feedback, and contribute to the evaluation of vision-language models (VLMs) on geolocation tasks.

### Strengths
1. The paper is clearly written and well-structured, making it easy to follow.

2. The image geolocation task is an interesting and practically relevant problem. Leveraging pretrained VLMs for this purpose is a natural and meaningful direction, and benchmarking diverse VLMs on this task is important.

3. The proposed platform mitigates data leakage issues in VLM evaluation by enabling users to contribute their own evaluation data, thus creating a more dynamic and realistic benchmark.

4. The authors have implemented and deployed the platform, and the system and models are open-sourced, which enhances reproducibility and potential community impact.

### Weaknesses
1. Engineering-focused contribution.

The paper’s main contribution appears to be the development of a platform for data collection and do the evaluation based on the collected data. As such, the work leans more toward an engineering or system contribution than an academic one.

2. Granularity issues.

The user-contributed data may vary in granularity. For example, some users may specify only the country, while others may provide exact coordinates or city names. Without a clear mechanism to handle this variation, the benchmark may not produce consistent or fine-grained evaluations across samples.

3. Relevance and data quality concerns.

While user participation helps prevent data leakage, it introduces potential quality and relevance issues. Some users may submit irrelevant images, low-quality inputs, or noisy feedback, which can make the rankings unreliable or manipulable by unintentional or malicious contributions.

4. Safety and privacy risks.

Allowing open user contributions raises safety and privacy concerns. For instance, attackers might upload unsafe or inappropriate content to manipulate model rankings or compromise platform integrity. The paper would benefit from a clearer discussion of content moderation and safety mechanisms.

5. Volume and diversity dependence.

The benchmark’s effectiveness depends heavily on user participation and data diversity. If only a small or geographically concentrated group of users contributes, the resulting rankings could be biased or unrepresentative of global performance.


I appreciate the authors’ efforts to design a more practical and dynamic benchmarking platform for VLMs. However, several practical challenges, including data quality, safety, and the scientific novelty of the contribution, remain insufficiently addressed. Clarifying how these issues are mitigated would significantly strengthen the paper’s impact and credibility.

### Questions
Please see above.

### Soundness
2

### Presentation
3

### Contribution
2
