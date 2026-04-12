## Human Reviewer 1

### Summary
In this paper the authors present a new dataset of modern AAA games for the purpose of world model training, which they call the Open-World Video Game Dataset (OGameData). Then then present their model, GameGen-X, a diffusion transformer for generating and controlling game video. GameGen-X is similar to other video generation models with the addition of InstructNet, which modifies the latents of GameGen-X for controllability. The authors present comparisons with a number of open-source video models. Finding that they generally produce more game-like video and may be better at control, according to some metrics.

### Strengths
The primary strength of the paper without question is the authors' new dataset. There is no dataset even close to this in terms of quality or size, it's a really exciting potential addition to this research area. This is primarily a strength in terms of originality, quality, and significance. I say primarily since the authors do not include access to the dataset at the review stage, though they do not have some metrics. 

The authors' GameGen-X and InstructNet are also strengths, but I have concerns with them limiting them as strengths, as I'll get to below.

### Weaknesses
The paper is relatively free of weaknesses in terms of originality, thanks in large part due to OGameData. However, the authors' work has some weaknesses in terms of the quality, clarity, and significance. This primarily comes down to (1) the authors' stated motivations and how this aligns with their work, (2) the way the authors overview their system, and (3) the experiments

### Motivations and the Dataset

The authors motivate in two primary ways: (1) imagining this as a prototyping or early development tool for open world game developers and (2) imagining this as leading to future interactive experiences with greater user control. These are fine as motivations, but the authors' choice of processing the dataset runs counter to them, somewhat. Specifically, the authors have broken apart their video clips into distinct scenes, meaning their model or other models trained on this dataset will not observe scene transitions. This is somewhat of an oddity for either of the authors' stated motivations and there's no justification for this choice given in the paper. Similarly, the authors do not actually have control input for any of the collected game data. This is again a bit of an oddity given the authors' stated purposes. I would guess that the authors collected this data from some sort of web scrape of gameplay video rather than collecting the video themselves through playing games such that they could capture actual control inputs. However, this isn't specified in the paper. This is a potential concern, especially if the authors did scrape an online repository of videos when such scraping went against the terms and service of the site in question. This should be clarified. 

### System Overview

Simply, the authors do not describe any of their system implementation in sufficient detail for replication. The authors state that code will be made available but do not make such code available for review. As such, there's no detail on the system architecture in terms of parameters or hyper parameters. The authors also not disclose the computation required to train their model or the training split used from their dataset. All of this would be required (potentially in appendices or in an external code repo) to ensure that the work is replicable. 

### Experiments

I have a number of concerns with the current setup of the experiments. The authors only compare against open video models, which are not attempting the same task and are not trained on the same dataset. As such, it's unclear the extent to which this is just that the training dataset for GameGen-X is more similar to the test dataset. While the authors do specify that the experiments are over test data, given the distribution of games in the dataset, its highly likely that GameGen-X had already trained on the same game that the test data used in the experiments came from. As such, this seems much closer to testing on the training set. 

The authors also have several metrics that require human expert raters, but who these experts were or what information they had is not specified. Further, the authors say they only use a single-blind setup, which may suggest the experts knew who they were. As such, there's a clear risk of bias here in terms of the experts feeling social pressure to more positively rate the more game-like videos if they knew that was the goal of this research. Clarity around the methodology and whether the authors have ethics approval would be necessary for readers to trust any of the human participant-based results. Relatedly, the authors state that the SR metric is "evaluated by both human experts and PLLaVA". However, the authors only present a single number. As such, it's not clear how the human expert and PLLaVA evaluations were combined. This throws doubt upon the SR metrics. 

The authors repeatedly bold values from their own work, indicating it is the best, when there is equivalent work from prior models. This may mislead readers in terms of understanding when the performance. 

The ablation study is helpful, as it demonstrates the value of OGameData, and several of the authors' components. However, since the authors do not train any other models on their dataset outside of these ablations, it's difficult to determine the exact value of the different components of their work. 

Overall, I'd say that the experiments are currently the largest weakness of this paper.

### Questions
1. Where did the dataset come from?
2. Why splice up the data by scene?
3. What was the methodology with the human expert evaluators? 
4. Did the authors receive ethics approval?
5. The authors indicate the dataset is split nearly evening between first and third person video, but primarily show results for third person video, why is this?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes GameGen-X, a diffusion based model for open-world game generation. Specifically, this paper proposes two detailed crafted datasets: OGameData-Gen and OGameData-Ins. OGameData-Gen is used to pre-train the diffusion model to understand and generate continuous open-world game-style videos, where OGameData-Ins is used to instruct tune the model to understand special inputs (e.g., keyboard inputs) to better control the continuation of the game generation based on some input frames. The dataset is well-curated to have 1M videos, with multiple filtering metrics and human-in-loop filtering to maintain the high quality. Then, this paper trains a video diffusion model with two-stage training on the two datasets for open-world game generation. Specifically, an instruct net is designed to take in different special inputs. Empirically, on their provided evaluation dataset, GameGen-X achieves superior performance than other state-of-the-art video diffusion models (e.g., kling).

### Strengths
1. A large well-curated dataset for open-world video games. The curation of the dataset contains filtering on different aspects (e.g., semantic alignment, motion), which results in a high-quality large-scale dataset. 
2. The idea to build a video diffusion model for open-world video games is essentially interesting, and the results and demo videos are impressive. Besides, quantitatively, the proposed approach also achieves better performance than other SoTA diffusion models. 
3. Detailed ablation studies demonstrate the effectiveness of the proposed component (i.e., two-stage training strategy and the design of the instructnet).
4. This paper is well-written and easy to follow.

### Weaknesses
1. One main concern is the proposed GameGen-X is specially fine-tuned/designed for open-world video games, while other diffusion models compared (e.g., kling) are trained for a general text-to-video generation, which makes the comparison somehow unfair to other models. 
2. The qualitative examples in the website demo for game generation (e.g., under generation comparison) don't seem to look much better than other models (e.g., cogvideoX).

### Questions
Please refer to weakness.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
In this paper, the authors introduce a diffusion transformer model aimed at generating and controlling video game sequences in challenging 3D open-domain game worlds. The authors also present the gameplay dataset they collected to train the model, OGameData. The dataset has 1 million video clips from across 150 videos, annotated with text descriptions using GPT4o. Both the model and the dataset have 2 components. One for text-to-video generation (OGameData-GEN and the pretrained foundation model) and one for instruction tuning (OGameData-INS and InstructNet).

### Strengths
- The work is original in the sense that is the first main contribution to the field in terms of interactive video game generation in large scale, complex, open worlds

- It is great to see such examples of tackling complex research environments at scale, with potential direct benefits to the game development process. 

- The author(s) introduce a complex system, both in terms of the dataset it required for training (including a resource intensive collection and curation process), as well as in terms of the pretrained foundation model and the interactive control network, allowing users to control the output via either text or mouse and keyboard inputs

### Weaknesses
- There is one strong concern I have regarding the data collection process for the OGameData dataset. My score highly depends on evidence that data collection will pass the ethics review and there is evidence provided on the consent given by the humans that produced the data. There should be understanding and agreement for it to be used for research purposes and open sourced. Please elaborate on how the data for OGameData has been collected? In Appendix B.1. you mention selecting online video websites as one of the primary sources. It would be good to know:
   - The exact sources of the video data
   - Any agreements or permissions obtained from video creators and game studios
   - The ethical review process they followed, if any
   - How you plan to address potential copyright or licensing issues

- It is unclear why all UI elements have been removed from the dataset, it would be great to gain further clarity on that from the author(s). In a lot of open-world gameplay , the player relies on UI element understanding, such as health levels, navigation information via mini maps, affordance of actions to take, inventory etc. 
   - How does this decision impact the model's ability to generate realistic gameplay experiences?
   - Do you plan to incorporate UI elements in future iterations of the model?

- Please correct me if I missed this, but the main body of the paper does not clearly indicate that all the data and the generation is within the constraints of a single agent. What is the model’s ability to model other dynamic environment elements (NPCs, other players, moving vehicles etc.)? It would be good to:
   - Explicitly state whether the model is limited to single-agent scenarios
   - If so, discuss the implications of this limitation on the model's applicability
   - If not, provide details on how the model handles multiple dynamic elements in the environment

- The paper is dense, so it took a while to disambiguate if the main body of the paper provides sufficient detail for capturing the core contributions of the paper or if a lot of essential details were included in the appendix.

### Questions
Clarification Questions:

-	Is it the correct understanding that the OGameData-GEN dataset comprises of data from 150 video games, whilst the OGameData-INS dataset contains only a subset of 5 game titles? Without checking Appendix B for clarification, it is difficult for the reader to grasp these details from the main body of the paper.
-	For video clip compression (Section 3.2) it would be good to add more details about the size of the latent representation z, as well as the resolution of the video clips used in training.
-	How were the spatial and temporal downsampling factors determined (s_t, s_h, s_w)?
-	In section 3.2, under unified video generation and continuation, you mention incorporating bucket training, classifier-free diffusion guidance and rectified flow for better generalization performance – did you run any ablation studies to understand better the impact of introducing these 3 components?
-	What are the values x for context length that you considered for video continuation?
-	In the InstructNet design, what were the considerations for choosing N (the number of InstructNet blocks)? Did you experiment with different values?
-	For the multi-modal experts introduced in Section 3.3, what are the sizes considered for the instruction embeddings and keyboard input embeddings (f_I and f_O)?
-	Under Interactive control, you mention the incorporation of video prompts V_p enhances the model’s ability to generate motion-consistent frames – did you conduct any experiments or ablations to measure the observed improvement?
-	Is there a mention on the computational resources required to store and stream the data for training, as well as for training the foundation model and InstructNet? It would be a useful proxy for people planning to reproduce the work.
-	Similarly, is there any information presented on the inference times of GameGen-X?
-	In evaluating the control ability you mention using both human experts and PLLaVa. What is the ratio between the 2 evaluation modalities?
-	On qualitative results for generation, apart from the discussion on diversity, would it be possible to elaborate on the length and consistency of the videos generated by GameGen-X? From the demo videos included, most are under <8-10 seconds.
-	In the ablation studies (Tables 4 and 5), there seem to be no DD and IQ metrics – what is the reason for that?

Minor comments/Suggestions:

-	In Section 2.2, it would be good to specify the human experts’ level of familiarity with the titles and elaborate on how the GPT-4o text annotations were checked for quality and accuracy.
-	In Section 3.3, you introduce the c condition under the Interactive Control subsection, but it is mentioned beforehand in Multi-modal experts. It would be clearer to the reader to introduce the structure of c under the Multi-modal experts’ subsection, where it appears for the first time.
-	For readability, it would be good to illustrate z, the latent variable in Figure 4.
-	It would be useful to include a more detailed explanation on the choice of baselines in the experiments Section. For example, Mira is not included under the results for control ability, is it because it does not have support for it? It would be good to clarify that.
-	It would be good to link to Appendix D (Discussion) when mentioning remaining challenges in the conclusion.
-	I know this appeared after the submission deadline, but it would be worth adding to the related work section as a referece: https://www.decart.ai/articles/oasis-interactive-ai-video-game-model

### Soundness
3

### Presentation
1

### Contribution
3

### Rating
8

### Confidence
5

---

## Human Reviewer 4

### Summary
This work focuses on generating high-quality, controllable open-world game videos that feature game engine traits. It emphasizes interactive controllability to simulate gameplay effectively. Notably, the authors collected a large-scale Open-World Video Game Dataset (OGameData), which consists of over one million diverse gameplay video clips from more than 150 games, along with informative captions generated by GPT-4o. Methodologically, they introduce a diffusion transformer model as the foundation model and a specially designed network called InstructNet for interactive control. The model is trained on the large-scale OGameData dataset using a two-stage process involving pre-training of the foundation model and instruction tuning for InstructNet.

### Strengths
1. This work collects a substantial number of open-world game videos from over 150 games, ultimately constructing more than 1,000,000 text-video pairs with highly detailed annotations. Its scale and diversity of annotations make it stand out, and the release of this dataset is expected to advance the field of game video generation.
2. It produces high-quality, more general realistic game video content. Previous works on game video generation often focused on specific game types, primarily 2D games or limited early 3D games. This work offers a more diverse and high-definition range of scene types for game video generation.

### Weaknesses
1. This work attempts to address the interactive control of open-world game video generation for gameplay simulation. However, to fully tackle the interactive issue, the generation speed needs to be considered,  as interactive experiences demand stringent timing requirements, which poses significant challenges. For instance, Google’s [1] achieves real-time rendering, even making it a viable game engine. While this work focuses on higher-resolution video generation, exploring the relationship between speed and performance would be beneficial, along with providing data on rendering time and speed. 

[1] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024.

2. The paper claims to simulate game engine features like diverse events, yet the examples provided offer quite limited dynamic event simulation, primarily addressing environmental changes like weather and lighting. There remains a gap to true gameplay simulation, such as incorporating NPC interactions or triggering more game-like special events.

### Questions
1.  Please provide data on the time required to generate a video segment at different resolutions or for different types of content. A section to analyze the trade-offs between generation quality and speed would be better.

2.  The training details of InstructNet lack specificity regarding the acquisition of video data corresponding to keyboard bindings. It would be beneficial to include more comprehensive information on the data collection process and the training methodology employed.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4