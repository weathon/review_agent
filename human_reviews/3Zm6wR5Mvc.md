# LangNav: Language as a Perceptual Representation for Navigation

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 8, 3, 5, 3

## Abstract
We explore the use of language as a perceptual representation for vision-and-language navigation. Our approach uses off-the-shelf vision systems (for image captioning and object detection) to convert an  agent's egocentric panoramic view at each time step  into natural language descriptions. We then finetune a pretrained language model to select an action, based on the current view and the  trajectory history, that would best fulfill the navigation instructions. In contrast to the standard setup which adapts a pretrained language model to work directly with continuous visual features from pretrained vision models, our approach instead uses (discrete) language  as the perceptual representation. We explore two use cases of our language-based navigation ours approach on the R2R vision-and-language navigation benchmark: generating synthetic trajectories from a prompted large language model (GPT-4) with which to finetune a smaller  language model; and sim-to-real transfer where we transfer a policy learned on a simulated environment (ALFRED) to a real-world environment (R2R). Our approach is found to improve upon strong baselines that rely on visual features in settings where only a few gold trajectories (10-100) are available, demonstrating the potential of using language as a perceptual representation for learning navigation agents.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The objective of this article is to leverage language as a perceptual representation for vision-and-language navigation. Their system uses off-the-shelf vision systems (for image captioning and object detection) to convert an agent's egocentric panoramic view at each time step into natural language descriptions.

### Strengths
While it has become a common practice nowadays, leveraging the remarkable understanding and reasoning capabilities of large language models as opposed to vision models, makes using language as an intermediate representation is the optimal choice. The system efficiently achieves competitive results while utilizing minimal resources. The article effectively presents its ideas, provides sufficient experimental evidence to support its viewpoint, and offers a detailed implementation statement.

### Weaknesses
While it is indeed a valid approach to utilize language as a perceptual representation for vision-and-language navigation, there remains a significant domain gap in accurately converting vision into language. This discrepancy can result in an imprecise representation of language, subsequently impacting understanding and reasoning processes. The author should delve into more comprehensive discussions and conduct experiments to explore the influence of various vision models and propose potential solutions for rectifying inaccuracies when the vision model fails.

### Questions
Could you please provide a comparison between the latest COT technology on GPT-4 in terms of language representation (for instance, Graph of Thoughts: Solving Elaborate Problems with Large Language Models, not just NavGPT) and the fine-tuned LangNav to validate the effectiveness of fine-tuning LLaMA with synthetic data? It is important to highlight the advantages of such an approach, as the allocation of resources without clear benefits would be deemed unnecessary.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces LangNav, which explores using large language models to address vision-language navigation (VLN). LangNav applies off-the-shelf vision systems to create textual descriptions for the agent's observations, formulates prompts containing observations and decisions in navigation to query GPT-4 for synthesizing new trajectories in text form, and fine-tunes LLaMA on those trajectories to learn VLN. The paper shows that using only a few (#10/100) gold samples, GPT-4 can generate high-quality data that boosts the LLaMA performance in downstream navigation tasks. Moreover, the experiment of transferring LangNav pre-trained in synthetic ALFRED environments to photo-realistic MP3D environments demonstrates that language can be used to bridge the visual domain gap.

### Strengths
- This paper studies an emerging and valuable problem of applying LLMs to language-guided visual navigation to enjoy LLMs' generalization power and bridge the visual domain gap. The research is well-motivated.
- The proposed 2-phrase synthetic trajectory generation method is novel and very efficient.
- As a researcher in this topic, I am very impressed and excited to see that small LMs like LLaMA-7B can be tuned to solve VLN tasks, which is, in fact, very difficult.
    - Without complicated prompt engineering and with an effective discretization of the agent's observations (action options), LLaMA2-7B is tuned to outperform NavGPT on R2R-VLN, which the latter uses the very large and powerful GPT-4.
    - The modification on Imitation Learning helps the policy to learn to recover from deviations from the optimal path, demonstrating the potential flexibility in supervising LMs to adapt downstream tasks.
- Experiments on transferring LangNav pre-trained in ALFRED environments to MP3D environments justify the argument of using language to bridge the visual domain gap, suggesting the potential of building a generalizable agent to address visual navigation in distinct environments using different forms of instructions.

### Weaknesses
- The technical focus of this paper, including synthesizing trajectories in text form using GPT-4 with a few "gold samples" and tuning LLaMA for text-based navigation, is somewhat misaligned with the core argument of "using language as perceptual representation".
    - Essentially, "using language as perceptual representation" introduced in this paper is nothing but using image captions and object text labels to replace visual inputs, which is a very common practice in many V&L research and particularly the same as the important baseline NavGPT (Zhou et al., 2023). 
    - The success in synthesizing trajectories with GPT-4 and tuning LLaMA should be attributed more to the very powerful pre-trained large models themselves. This success does not justify the benefit of using language but is more like a compromise of LLMs only take textual inputs. Moreover, I might be able to create an R2R textgame without any actual images (like Jericho [1]) to sample trajectories.

- I do agree that using language is promising in bridging domain gaps (e.g., different forms of instructions, 2D-3D semantics, and the Sim-Real visual gap presented in this paper), but it is unclear how far this method (of using language alone) can go.
    - As mentioned in this paper, the proposed method is limited by the off-the-shelf visual captioning and detection models. The captions could be inaccurate, not comprehensive, or ambiguous (e.g., the eye chart vs. picture example). And the ability of LLMs to reason very detailed descriptions is unclear.
    - There is an emerging trend of developing visual-LLMs (VLMs) that should be able to preserve visual clues better, which potentially could reduce the value of LangNav in future research and application. It is not very clear to me how language-only for VLN can be further extended.
    - As aware by the authors, there are many recent works on automatically synthesizing 3D environments that can be used for embodied AI training (e.g., Text2Room [2], Pathdreamer [3,4], ProcTHOR [5], etc.), and VLN data generation pipeline (e.g., MARVAL [6], AutoHM3D [7], ScaleVLN [8], etc.). The cost of building and annotating these embodied data has been significantly reduced. In ScaleVLN, the fine-tuned agent achieves 81% SR compared to LangNav 45.6% SR. Hence, I do think the performance potential of the proposed LangNav needs more justification.

- The experiments comparing LangNav to RecBERT and DUET are not strictly fair. 
    - LangNav not only uses the 10/100 gold samples in fine-tuning but also the 10,000 synthetic trajectories. Those 10,000 data should not be neglected. From Table 1, LLaMA2-7B (LangNav without the 10,000 data) performs very similarly to RecBERT and DUET.
    - A more rigorous way is to create the same amount of "fake" data (using some automatic methods) to train RecBERT and DUET, although I am aware that such fake visual-text data can hardly be created at the same cost, and it is hard to define "fairness" between "language-only" and "language-visual" data.

- I believe one of the most important reasons for introducing LLMs to embodied AI tasks is to leverage LLMs' exceptional language understanding, task planning, tool-using, explicit reasoning with commonsense knowledge, communication, etc., abilities. 
    - However, it is unclear in this paper whether LangNav shows any of the above abilities (e.g., using commonsense to navigate).
    - It is likely that after tunning LLaMA for VLN, the model lost these abilities, and LLaMA is simply applied as a good initialization for learning the downstream R2R task.

- The claim "Sim-to-Real" for AI2THOR-ALFRED to MP3D-R2R is very misleading. R2R is still simulated, except that it uses photo-realistic images. "Sim-to-Real" usually refers to deployment to a physical robot in the real world.

[1] Interactive Fiction Games: A Colossal Adventure. Hausknecht et al., AAAI 2020.

[2] Text2room: Extracting textured 3D meshes from 2D text-to-image models. Höllein et al., ICCV 2023.

[3] Pathdreamer: A World Model for Indoor Navigation. Koh et al., ICCV 2021.

[4] Simple and Effective Synthesis of Indoor 3D Scenes. Koh et al., AAAI 2023.

[5] ProcTHOR: Large-Scale Embodied AI Using Procedural Generation. Deitke et al., NeurIPS 2022.

[6] A New Path: Scaling Vision-and-Language Navigation with Synthetic Instructions and Imitation Learning. Kamath et al., CVPR 2023.

[7] Learning from unlabeled 3D environments for vision-and-language navigation. Chen et al., ECCV 2022.

[8] Scaling Data Generation in Vision-and-Language Navigation. Wang et al., ICCV 2023.

### Questions
Questions without a star (*) are not essential to my evaluation of this paper, but I still hope the authors can kindly and patiently respond to them. Please also address my concerns mentioned in Weaknesses.

1. How important are the image captions and the object labels in text descriptions?
    - How do the trajectory generation and VLN results change without any of them?
    - Does history $H$ have both the image captions and object labels? Or just image captions? This is not very clear from page 3.

2. (*) There are two important conclusions that lack convincing evidence. 
    - (*) I was very impressed by the example in Figure 3. I tried to draw out the described structure, and it is indeed very spatially consistent. However, there is no numerical analysis to support the three claims of the quality of all generated trajectories (Strong Prior, Spatial Consistency, and Descriptive). Is this a general observation? I wonder if the authors can provide any quantitative results.
    - (*) I am also not fully convinced by the visualizations in Figure 4, which seems like two randomly picked examples. It is hard to say whether "eye chart" and "picture" are closer in visual feature space or text space. I wonder if the authors can give more comprehensive statistical analyses on the success/failure cases of LangNav vs. RecBERT.

3. (*) Why different LMs, LLaMA2 and LLaMA1 are used for R2R and ALFRED-R2R experiments, respectively?

4. (*) Table 2. Why do the gains are marginal when going from 2,000 to 10,000 trajectories? Is it due to the lack of diversity in generated trajectories?
    - (*) Why not use more different scenes and gold instructions to generate trajectories? How might it affect the quality of generated trajectories? Will it improve the fine-tuned LangNav?
    - (*) I am still not fully convinced by this "extremely few gold samples" setting. Why is this necessary since we have much more R2R data and the data is not too costly to generate? How does this setting generalize to other practical navigation problems?

5. The paper claims that NavGPT uses ground-truth distance, so it has an advantage and is not strictly comparable. But, using distance of viewpoints is very reasonable in R2R because the task is defined in a discretized space (DUET also uses this information). Even in continuous environments, a waypoint predictor can be applied to return the exact distance of navigable positions [9,10,11].
    - What would be the results of training LangNav with distance information?

6. (*) Are the RecBERT and DUET in Table 1 and Table 3 pre-trained on R2R, e.g., with tasks such as masked language modeling and single action prediction? This is unclear in the paper, but it is very important in the discussion of the few-shot learning and visual domain gap.

Others:
- Figure 1. Use the same color for image C (orange) and "C" in the caption (pink).
- Typo in S5 Limitations. "table 2" should be "table 4".
- Table 4. A "dagger" sign is missing in the caption of re-implementation. Also, I believe RecBERT hasn't been pre-trained on the val_unseen set. What does this re-implementation exactly do?
- In-text citations have inconsistent formats (e.g., author names inside/outside brackets).
- Please add an ethical statement and reproducibility statement at the end of the paper.

[9] Waypoint models for instruction-guided navigation in continuous environments. Krantz et al., CVPR 2021.

[10] Sim-to-real transfer for vision-and-language navigation. Anderson et al., PMLR 2020.

[11] Bridging the gap between learning in discrete and continuous environments for vision-and-language navigation. Hong et al., CVPR 2022.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes using language as the perceptual representation of the navigation agent environments in the domain of vision-and-language navigation. Such an approach would allow using LMs (language models) to model the navigation task as a pure language task. The visual scenes are expressed as text descriptions from vision models (such as a captioning model). Two empirical studies are shown -- first shows efficient synthetic data generation using just language by using prompt engineering on a LLMs (such as GPT-4), second shows that language enables more efficient sim-to-real transfer compared to vision models.

### Strengths
* The Case Study 2 is very interesting: it shows that when transferring from a simulated environment (AI2THOR) to real environment (R2R), using a model that relies on vision input generalizes worse than a model that relies solely on text input. This is also intuitive in that while actual pixel-level details can differ, the environments at semantic level should have similarities and the sim-to-real transfer should work better with a modality that learns semantic representation of the environment.

### Weaknesses
* The Case Study 1 is not very informative. The LangNav not only benefits from synthetic instructions but also benefits from the number of samples used for finetuning. While baselines are finetuned on only 10/100 real-world trajs, the LangNav model is finetuned on 10/100 real-world + 10,000 synthetic trajs. For a fair comparison and to truly understand the quality of synthetic instructions, we need a study something like:

  * A: trained on 10k real-world trajs
  * B: trained on 10k synthetic trajs
  * C: trained on 5k (sampled) real-world + 5k (sampled) synthetic trajs

  In the above setup, comparing A vs B will indicate the quality of synthetic trajs. Comparing A vs C will indicate if synthetic trajs can 
  complement a smaller number of real-world trajs to achieve the same quality.

* Given that no new models or methods are developed, the key contributions of this paper are relatively light.

Overall, I think this is a good workshop paper to motivate future work in this direction, but in itself, does not meet the bar as a conference paper.

### Questions
Some of the points below are more like suggestions for future improvement, rather than immediate action items:

* In 4.2, it is worth reiterating the differences between RecBERT and LangNav models.

* It would be more interesting if there can be a more end-to-end learning setup with vision-to-text module part of the training process.

* In 3.3, a strategy is described to make the model robust to deviations from logged (imitation) policy. A simple and more principled strategy would use something similar to RLHF -- use trained model policy pi_{t} to infer on trajectories from training set, assign rewards (simple goal based as an example) for the inferred trajectories, keep the ones with high reward (some threshold) and train pi_{t+1} on union of existing training dataset and those with high reward.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces LangNav, a novel approach to vision-based robotic navigation guided by natural language descriptions. LangNav uses off-the-shelf visual models to convert it into language descriptions and leverages language models to make navigation decisions. The proposed approach formulates the robotic navigation problem to leverage the capabilities of a language model for decision making. Actions are formulated to be selected from a finite set of “observations”. The paper proposes efficient synthetic dataset generation and transfer from ALFRED to R2R environments that have different 3D visual appearances. 
While LangNav shows promise in improving navigation performance, further research and empirical experiments are needed to address these concerns and validate its practicality in complex and diverse robotic environments.

### Strengths
1. The paper demonstrates how few-shot examples can be used to generate synthetic data and train a language-based navigation policy, thereby highlighting “language itself as a perceptual representation space”.
1. The authors demonstrate that a small LM like Llama-7B  outperforms a vision-based agent that is finetuned on the same seed data. 
1. The paper demonstrates “language as a domain-invariant representation” by training agent on ALFRED and transferring it to R2R environments.

### Weaknesses
1. Incomplete Visual/Spatial Information: The image captioning and object detection modules may not provide all the necessary visual and spatial information needed to successfully complete the navigation task. Since the dataset contains trajectories that were constructed based on the  vision systems to obtain the captions/objects, such a scenario will never be encountered in the training/test set. But this raises concerns about the reliance of LangNav for accurate and relevant visual captioning/detection in real-world scenarios. 
1. Applicability to Real-World Robots: Just like ImageNet, which organizes images according to the WordNet hierarchy but does not necessarily contain a wide range of real-world objects, it seems the scope of task and navigation instructions that were used to train are those where the captioning model provides task relevant descriptions. This raises concerns about the applicability of the LangNav approach to real-world robotic systems, where the environment and tasks can be much more complex and varied than what the training data might represent.
1. Simulation-to-Simulation Transfer: It seems inappropriate to label the R2R environment, based on Matterport scans, as "real-world." There is a lack of empirical experiments or evidence to support the effectiveness of transferring policies learned in simulated environments to genuinely real-world scenarios. This raises questions about the robustness and practicality of the proposed approach in real-world robotics applications.

### Questions
1. Episode Length and Performance: Are there empirical experiments or findings related to the performance of the LangNav system concerning the length of the navigation episode? Does the model's performance vary with longer or shorter episodes, and if so, how?
1. Expressivity of Observations and Actions: The expressivity of observations and actions seems to influence the size and structure of the prompt, so what role does it play in the overall task performance? 
1. Prompt History and Reasoning: Can you provide more details on how updating the history in the prompt affects the reasoning in the language model? Are there specific mechanisms or techniques used for incorporating historical information, and how does this impact the model's performance?
1. Biases in Synthetic Data Generation: Given that "LangNav underperforms baselines in data-rich regimes," can you elaborate on any potential biases in the synthetic data generation process? Are there challenges or limitations in scaling up the generation of synthetic data for training the LangNav system, and how might these biases be addressed?
1. Minor clarifications/suggestions:
- real-world environment (R2R)? Is it really real-world? → a more apt claim could be simulators with perceptually different representations or simulation to simulation transfer.


- In Section 4.1.1, “a real demonstration instruction and trajectory” is unclear. Does this mean rolling out the episode in the chosen simulated environment? 


- In Section 7, “real-word” -> “real-world”

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper tackles the task of Vision-and-Language Navigation (VLN) in the scarce data setting: defined to be small datasets with at most 100 instructions coupled with ground-truth trajectories. They propose to do this by decomposing VLN into visual perception and language-based decision-making, using language as the intermediate perceptual representation. Visual perception can be handled by existing vision-language models (VLMs), while a language-based decision-making policy can be fine-tuned from powerful language models (LMs). They design the representation as structured text prompts describing observations and trajectory history, which are extracted by VLMs. To train the LM policy, they synthesize a dataset by prompting Large Language Models (LLMs) to generate instructions and trajectories given the original small dataset given as an example. They conduct experiments to evaluate the performance of an LM policy fine-tuned on their synthesized data, and to evaluate how well an LM policy trained in simulation generalizes to real-world settings.

### Strengths
1. Experiments provide extensive comparisons across recent VLN baselines spanning a broad range of design choices: e.g. end-to-end learned systems, systems with explicit topological memory, systems using LLMs zero-shot for reasoning.

### Weaknesses
1. Evaluations do not directly address the main hypothesis. LangNav phrases its main focus as the hypothesis that a VLN system should use language as its perceptual representation (Sec. 1, para. 3). However, the paper’s evaluations focus on how language can help in low-data regimes rather than on the inherent merits of a language-based representation (e.g. an analysis of how well the representation captures and conveys information salient to decision-making, how easily and accurately the representation can be built with the proposed perception system etc.). I would suggest that the authors should clarify in the title and Secs. 1, 3 that their focus is on using language for VLN specifically in low-data situations.

2. Assumption of low-data setting for VLN in home environments seems contrived. Given rich and widely available datasets for VLN [1-4] and VLN-CE [5] tasks in household environments, the assumption on low data availability for VLN seems artificial. To better motivate this setting, the proposed system could also be tested on a wider variety of environments lacking extensive datasets (e.g. office environments, supermarkets, or industrial environments like workshops and warehouses). Since the inputs are purely language, it should not require excessive effort to hand-engineer a small number of trajectories based on real-world environments. These tests also serve to validate that LLMs can synthesise useful data in more exotic environments: while existing object-goal navigation works indicate LLMs certainly have learnt rich ‘common-sense’ about household environments, their capabilities in other environments remain less tested. I consider the need for strong motivation and evaluation of the low-data setting to be essential, especially given that the experiments indicate LangNav’s performance when trained on the entire dataset is still far from SOTA (Table 4).

3. Limited analysis provided on quality of proposed LLM-based data generation. Tables 1, 2 indicate that the LangNav agent trained on synthetic data converges to notably lower SR than an agent trained on the full R2R (Table 4), suggesting that LangNav’s generated synthetic data is poorer in some aspects compared to R2R. The authors should provide some insight into why this is the case, and more broadly explore the limitations of synthetic data generation using LLMs’ ‘common-sense’ as compared to data from hand-collected datasets like R2R, RxR etc.

4. Limited novelty or originality in using language for sim-to-real transfer. Similar observations and discussions on the effectiveness of using text or language intermediate representations to transfer a policy across environments have already been made in [6,7].

5. Limited exploration of visual perception and its coupling with the LM policy. LangNav largely assumes that existing VLMs enable VLN to be decomposed into visual perception and language-based decision-making, and thus consider only the decision-making aspect. However, failure modes such as the VLMs’ failure to identify features needed for LM policy to reason (Sec. 5) suggest that the coupling between the two components needs to be more deeply considered for robust VLN. E.g. Can VLMs be prompted also with navigation instructions or trajectory history to focus their attention on salient details, or can VLMs make use of some feedback from the LM policy?

[1] Yuankai Qi, Qi Wu, Peter Anderson, Xin Wang, William Yang Wang, Chunhua Shen, and Anton van den Hengel. Reverie: Remote embodied visual referring expression in real indoor environments. In CVPR, pages 9982–9991, 2020. 

[2] Fengda Zhu, Xiwen Liang, Yi Zhu, Qizhi Yu, Xiaojun Chang, and Xiaodan Liang. Soon: Scenario oriented object navigation with graph-based exploration. In CVPR, pages 12689–12699, 2021.

[3] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sunderhauf, Ian Reid, Stephen ¨ Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In CVPR, pages 3674–3683, 2018.

[4] Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, and Jason Baldridge. Room-across-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4392–4412, 2020.

[5] Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision-and-language navigation in continuous environments. In European Conference on Computer Vision, 2020.

[6] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cote, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. ALFWorld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations.

[7] Narasimhan, K., Barzilay, R., and Jaakkola, T. (2018). Grounding language for transfer in deep reinforcement learning. JAIR, 63(1):849–874.

### Questions
1. What are “gold” trajectories? 

2. Given that there seems to be minimal constraints or priors placed on the trajectories/scenarios the LLM can generate (aside from several trajectory examples), how realistic are the outputs? Are the trajectories clearly in the desired household environments, and do they exhibit the expected spatial layout of the house?

3. Is the random action approach (Sec. 3.3) used for training the LangNav agent, and if it is, how is it used? This replanning approach is possible given a map of the environment. However, given that most of the data LangNav agents are trained on are trajectories in an environment hallucinated by an LLM, and which we have no map for, how is the random action approach used?

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor
