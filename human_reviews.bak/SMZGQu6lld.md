# LLM-Prop: Predicting Physical And Electronic Properties of Crystalline Solids From Their Text Descriptions

- Decision: Reject
- Scores: 5, 3, 3

## Abstract
The prediction of crystal properties plays a crucial role in the crystal design process. Current methods for predicting crystal properties focus on modeling crystal structures using graph neural networks (GNNs). Although GNNs are powerful, accurately modeling the complex interactions between atoms and molecules within a crystal remains a challenge. Surprisingly, predicting crystal properties from crystal text descriptions is understudied, despite the rich information and expressiveness that text data offer. One of the main reasons is the lack of publicly available data for this task. In this paper, we develop and make public a benchmark dataset (TextEdge) that contains text descriptions of crystal structures with their properties. We then propose LLM-Prop, a method that leverages the general-purpose learning capabilities of large language models (LLMs) to predict the physical and electronic properties of crystals from their text descriptions. LLM-Prop outperforms the current state-of-the-art GNN-based crystal property predictor by about 4% on predicting band gap, 3% on classifying whether the band gap is direct or indirect, and 66% on predicting unit cell volume. LLM-Prop also outperforms a finetuned MatBERT, a domain-specific pre-trained BERT model, despite having 3 times fewer parameters. Our empirical results may highlight the current inability of GNNs to capture information pertaining to space group symmetry and Wyckoff sites for accurate crystal property prediction.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a novel LLM-based method for the prediction of crystal properties. The authors collect a dataset including crystal text descriptions with their properties, and use a T5-based finetune network to achieve SOTA performance on their benchmark. Also, the authors perform their model on two property prediction tasks with ablation.

### Strengths
(1). The authors conducted enough experiments to show the efficiency and capabilities of their model. The experiment is solid and convincing.
(2). This paper is well-written and easy to follow. The author’s motivation for this work is clear.
(3) The problem with crystal is important. And focusing on language and LLM is an important view for this problem.

### Weaknesses
1. Some details are not clear. How large is the pre-trained T5 model you used? The T5-small, T5-base, or T5-large? Specify this is important for the comparison of your efficiency. 
2. The authors mentioned there’s some related work that also used finetuned LLMs for crystal representation. These works collect crystal text descriptions based on Robocrystallographer too. However, the authors did not compare their dataset with theirs about the text content and text quality. So, the experiment is not solid.

### Questions
(1). You have mentioned there’s some related work that also used finetuned LLMs for crystal representation. They collect crystal text descriptions based on Robocrystallographer too. Could you compare your dataset with theirs about the text content and text quality? Showing the advantage of your dataset is important for your contribution to data collection.

(2). Section 5.1 “The possible reason for this improvement might be that LLMProp can easily access the most important information for volume prediction, e.g. space group information, from the text descriptions compared to GNNs.” Could you please make some more ablation about your text information to prove your statement? Also, this is important to measure the importance of different types of information.

(3). Could you please make a task description to show the importance of the two tasks (Band gap and volume) in the experiments? Many other properties are predicted in related works such as energy per atom and bulk modulus. Why did you choose these two tasks?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies the physical and electronic property prediction problem and proposes to use large language models to predict the properties of crystals from the text descriptions. The paper provides an input processing strategy to prepare the input text of the language model and adapt T5 for predictive tasks.

### Strengths
1. The studied problem is interesting and AI for science is also an important research area.
2. The paper is well-organized and easy to follow.

### Weaknesses
1. Although the task of crystal property prediction is an interesting problem for AI for science, the innovation of methodology is not surprising. This paper only uses the language model to make the prediction which is less novelty.
2. There are several recent works focusing on the text-rich graph where each node has text descriptions by jointly leveraging the GNN and LLM. These works are also related to the problem studied in this paper and should be discussed and compared.
3. The specific challenge in this problem is not clarified.

### Questions
1. What is the specific challenge in the problem of crystal property prediction?

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper investigates the task of predicting crystal properties with LLM decoder-only model. The model is trained on the constructed benchmark dataset (TextEdge) which contains crystal structure, description, properties such as band gap. The experimental result shows that the proposed method LLM-Prop is slightly better than previous method MatBERT and ALIGNN.

### Strengths
- The paper extends the previous crystal representations task based on text into the property prediction task based on text.
- The benchmark dataset TextEdge should be helpful in the predicting crystal properties domain.

### Weaknesses
- My top concern is the technical depth of the paper. For the method, this paper is an implementation for predicting crystal properties task with decoder part of T5. The discussion of choice of T5 is weak and Input Processing is also trivial. 
- For the Data collection in Sec 3.1, I did not see the challenging part and discussion about the process of data collection including quality control. I am not sure about the quality of data at all.

### Questions
N/A

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
