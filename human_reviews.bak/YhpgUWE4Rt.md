# Recurrent Neural Cellular Automata with Self-Attention for Multi-agent System

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3, 5

## Abstract
Many-agent systems, such as epidemic spread, rumor propagation through crowd, prey-predator model, and forest fire, exhibit complex global dynamics originated from local, relatively simple, and often stochastic interactions between agents. Despite significant advancements in predictive modeling through deep learning, such interactions among many agents have rarely explored as a specific domain for predictive modeling. We present Recurrent Attention-based Neural Cellular Automata (RA-NCA), to effectively discover the local stochastic interaction by associating the temporal information between neighboring agents in a permutation-invariant manner. RA-NCA exhibits the superior generalizability across various agent configurations (i.e., spatial distribution of agents), data efficiency and robustness in extremely data-limited scenarios even with the presence of stochastic interactions, and scalability through spatial dimension-independent prediction. We compare and evaluate RA-NCA with other NCA networks and scene prediction networks in the three synthetic multi-agent systems with thousands of agents, such as forest fire, host-pathogen, and stock market models.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a novel neural network model called Recurrent Attention-based Neural Cellular Automata (RA-NCA) for predicting the dynamics of many-agent systems with local and stochastic interactions, which has not been explored in existing work. The empirical results on the three synthetic cases show that the proposed method presents superior data efficiency and scalability.

### Strengths
- It addresses a challenging and underexplored problem of learning the hidden interaction rules in many-agent systems without prior knowledge or assumptions.
- It introduces a recurrent cellular attention module that combines LSTM and cellular self-attention to capture the temporal and permutation-invariant information of neighboring agents.
- It demonstrates the superior performance of RA-NCA over other NCA networks and video prediction networks in terms of data efficiency, robustness, and scalability across three synthetic datasets with different levels of stochasticity.
- The experimental results are conducted thoroughly and convincingly.

### Weaknesses
- It lacks an investigation on the scale of considered neighbors. Specifically, the authors consider only the case of Moore's neighborhood.
- It only evaluates RA-NCA on synthetic datasets, which may not reflect the complexity and diversity of real-world many-agent systems.
- It does not compare RA-NCA with other state-of-the-art methods for multi-agent interaction learning, e.g., networked agent learning derived from distributed optimization [1], which leverages a similar philosophy to resolve large-scale agent learning. Nonetheless, I do not regard it as a reason to reject this paper, and I hope the authors give further comparison in this paper, experimentally or conceptually.
- Some key concepts lack clarity, such as what is agent interaction? I suggest the authors give a brief introduction in the "Related work and Background".



[1] Zhang, K., Yang, Z., Liu, H., Zhang, T., & Basar, T. (2018, July). Fully decentralized multi-agent reinforcement learning with networked agents. In International Conference on Machine Learning (pp. 5872-5881). PMLR.

### Questions
1. Section 2: "However, recurrent neural networks based approaches have been ....", I didn't get the point of this claim.
2. Section 2: "However, these methods are not explicitly designed to preserve spatial structures in latent space ...". The question is that why can NCA preserve spatial structures in latent space?
3. Section 3: "Recently, graph-based, attention-based ,...". Please add references to existing work.
4. How does the NCA make the global interactions keep consistent with local interactions in Moore's neighborhood?
5. What is $c_{(i,j)}$ in equation (4)?
6. What is the value of $(t_{n_{pred}} - t_{n_{obs}})$ used for experiments?
7. Have the authors ever tried other training scales? expect for $64\times64$.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a new model for many-agent system prediction/simulation such as epidemic spread, rumor propagation through crowd,
prey-predator model, and forest fire. Unlike the previous CNN-based model which suffers from spatial dependency, the proposed new model 
 is based on self-attention which is permutation invariant and thus overcomes this issue. In addition, the LSTM is introduced to endow the model with memory, which is essential to leverage historical information to do prediction. The experiment section proves the effectiveness of the method and shows that the model outperforms other baselines even in extremely data-limited scenarios.

### Strengths
1. The paper is well-written and easy to understand
2. The motivation for model designing makes sense
3. The result shows it outperforms baselines in this field and other video prediction baselines

### Weaknesses
1. The method is trained and evaluated on in-house image datasets collected with stale simulators (released 20 years ago). As I am not an expert in the related field, I wonder if there are some new and common benchmarks for evaluating the method like real-world forest fire data, which should be unstructured and sophisticated. 
2. I can not find any potential for applying the method to important problems. The method represents the many-agent system with a structured semantic image and converts the problem to an image prediction problem on toy datasets.
3. The method is a simple combination of two existing and well-studied techniques. Thus technical novelty is limited as well.

### Questions
N/A

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces Recurrent Attention-based Neural Cellular Automata (RA-NCA), combining the attention mechanic with the recurrent LSTM neural network for state prediction in cellular automata (CA). In contrast to other CA approaches using CNNs as their feature projector, the proposed architecture  is presented as superior regarding training and sample efficiency, mainly due to the attention’s permutation invariance. RA-NCA is also shown to scale well in the transfer to larger CA systems. For the evaluation, three CA settings are being tested (forest fire, host-pathogen spread and a stock-marked model), comparing RA-NCA to a LSTM+CNN model and an ablation with only the attention mechanism. RA-NCA was also evaluated with respect to sample efficiency and compared to recent Scene/Video-Prediction Networks on three levels of stochastic transition for the forest fire domain.

### Strengths
The paper is concisely written and understandable. All key-concepts are explained and cleanly formalized. The domains are well described in the Appendix. The evaluation on three domains covers diverse aspects with regards to density and the amount of training data. The focus point of the attention invariance is well explained and nicely visualized in Fig. 2 and 3. Nevertheless, I would like to point in the direction of Tang and Ha (2021), which is quite close in terms of insights, but applied for the field of RL. A discussion on the key difference to this work, perhaps with focus on the (Moore) neighborhood, would certainly benefit the quality of this work. Regarding the proposed architecture, both key contributions of showcasing the data-efficiency and the model-scalability pose an interesting topic of research, but I would have liked to see some more focus on the latter.

Y. Tang and D. Ha, "The Sensory Neuron as a Transformer: Permutation-Invariant Neural Networks for Reinforcement Learning", in Advances in Neural Information Processing Systems, 2021.

### Weaknesses
While the paper contains some interesting foundations, it lacks focus on the field of CAs by mixing in notions of multi-agent systems and video prediction networks. From my view of (MA)RL and Game Theory, the mention of CA as multi-agent systems seems quite unintuitive, since I understand CAs more as conditional transition (probabilities) between cells rather than multiple agents interacting with each other. Apart from the notation and the section on Multi-Agent interaction, I also see no distinction being made between classical neural CAs and each cell being an agent.

Also, the evaluation could be improved. Evaluating only against two variation baselines (CNN+LSTM, only Self-Attention) seems to be on the weaker side. Convolutional-recurrent Networks are being steadily phased out of image recognition in favor of e.g., Vision-Transformers, in part due to sample efficiency of training convolutional kernels. The claim “The three networks consist of the similar level of trainable parameters.” on page 7 is similarly questionable, since CNN+LSTM and RA-NCA almost double the parameter count of Attention-CA. Since taking the hidden-state information out of the Attention-CA baseline seems to be impactful, comparing to Attention+RNN would perhaps be more insightful. 
I would have also liked to see a comparison to the obvious state-of-the-art architecture Attention-Transformer NCA  (Tesfaldet et al., 2022), that is cited but not used in the comparison between Transformer and LSTM. Even if this comparison is intended as part of a pre-study on the importance of the respective architecture components, there is still a significantly overlap to the section on video prediction networks.

The inclusion of the comparison to video prediction networks seems to overall weaken the focus of the paper without providing novel insights. Since I am not too familiar with video prediction networks I cannot speak to the quality of the chosen models as baselines. However, since the CA-video is reduced to minimal color diversity that simply represent very few states (binary even, as it is evaluated / trained), apart from the concept of "predicting batches of pixels from previous batches of pixels", I see not much of a connection to the field of NCAs. Furthermore, CNN+LSTM (and simple Attention) as Model Representatives are already covered in the NCA study and are already evaluated on the same settings as Fig. 6 and Fig. 7. The space dedicated to both the multi-agent interaction and video prediction networks feels disconnected from the core-idea and could have been used to round out the otherwise very interesting component-study in the main-field of NCAs for a more focused and in-depth paper. Also, the discussion on original cell-states on page 14 in the Appendix could perhaps deserve a mention in the main paper?

I would also recommend to more fairly mention the advantages and limitations (e.g., parameter disparity of the Attention-CA) of the other baselines and expand on the discussion. Comparable (or better) performance of the CNN+LSTM in Fig.5 (forest fire) up until data-amount of around 10% could be more discussed, as could
the good results on the other two domains (that are only found in Table 5 in the Appendix). I would also like to see the std-div. in the main paper tables (similar to the ones in the Appendix) and some more discussion on variance, since e.g. Fig.6 (the zoomed out bar-plots) shows how close performance is if you consider the variance.

In summary, I would recommend to shift the focus away from multi-agent interactions and video-prediction networks to focus and elaborate more on the two core-insights of this paper (Low-Data Efficiency and Scalability) to improve focus and balance of the paper. The scaling aspect in particular could offer an interesting insight into the degrees of scalability. It would be encouraged to show a direct comparison to the Transformer-NCA and make the Attention-CA baseline more comparable in terms of parameters to gain a better overview on the current state of the field of NCAs in
general.

### Questions
Could you please specify on the $\bigodot$ operator in Eq. 4?
Why was 32 chosen as the encoding dimension?
Is there a reason why the Transformer Architecture is not applicable in this setting? Would you consider using the Attention-Transformer NCA from Tesfaldet et al., 2022 as a more competitive baseline?
Could you please motivate/elaborate why you are considering NCA-cells as interacting agents. Apart from the summation of the neighbor’s hidden state there is not “interaction” in the classical multi-agent sense, that I would have understood here?

Minor Comments:
p3: “Recently, graph-based, attention- based, variational autoencoder-based NCA networks are also proposed.” could use citations, as could the “classical” ML-LSTM literature (e.g. Schmidthuber et al) that you are building on.

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a new Recurrent Attention-based Neural Cellular Automata (RA-NCA) in modeling complex systems involving many agents with local, often stochastic interactions.  RA-NCA's innovation lies in a recurrent cellular attention module that combines long short-term memory (LSTM) with cellular self-attention. By evaluating on three simulated multi-agent datasets, RA-NCA shows three good properties which are 1.) Robustness in the presence of stochastic interactions; 2.) Data Efficiency which requires less training data due to the permutation invariance inductive bias; and 3.) Scalability, as it can be trained on small systems and successfully applied to significantly larger systems without the need for re-training, and without a decrease in performance.

### Strengths
1. The paper proposes a new model that extends existing NCA methods to capture long-term local and stochastic interactions among agents.
2. The method is technically sound and the writing is in general easy to follow.
3. The experiment results compared with selected baselines show the superior of the proposed method.

### Weaknesses
1. My major question is the comparison with existing baselines. I understand the paper targets developing a new NCA method to address long-term local interaction among agents and the stochastic property. However, for the specific task it is dealing with, there are much more methods to compare with in literature on multi-agent dynamical system modeling. Examples include discrete GNN-based methods [1][2], and continuous GNN-based methods [3][4] where the key idea is similar to capture the influence from neighbors and past timestamps to make predictions in the future. Also there are some reinforcement learning literature that can address the same task. I believe at least the authors should have a thorough discussion about these directions.


[1] Alvaro Sanchez-Gonzalez et.al.  Learning to simulate complex physics with graph networks.


[2] Peter W. Battaglia et.al. Interaction Networks for Learning about Objects,Relations and Physics.


[3] Zijie Huang et.al. Learning continuous system dynamics from irregularly-sampled partial observations.


[4] Chengxi Zang et.al. Neural Dynamics on Complex Networks.

### Questions
1.  many-agent --> multi-agent?
2. In Figure 4, can you also plot the visualization from baselines as comparisons?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
