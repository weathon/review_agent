## Human Reviewer 1

### Summary
This paper proposes an auxiliary loss function for generative recommendation models. Specifically, it introduces a masked-token prediction loss where semantic ID tokens are replaced by mask tokens and the model is tasked with predicting the original semantic IDs. To further improve performance, the authors introduce (1) entropy-based masking, prioritizing tokens with higher entropy according to an auxiliary model, and (2) curriculum learning, using three phases with different masking strategies and ratios. Experiments are conducted on three public datasets.

### Strengths
1. Timely study on training objectives for generative recommendation.
2. Experiments are conducted on three public datasets.

### Weaknesses
1. The motivation and critique of the next-item prediction objective are unconvincing. The authors make two main arguments:
    1. "Predicting the next single item is not understanding the path that led there". Modern recommender models commonly use self-attentive Transformers, which explicitly attend to all previous tokens when predicting the next one. Mechanistically, this design allows the model to reference the full sequence structure. The disagreement between me and the author(s) may stem from different definitions of "understanding", which is a subjective term. I suggest using more precise and objective language to support this claim.
    2. "Next-item prediction intuitively prioritizes local transitions over global understanding". This is not intuitive to me. Attention enables access to all previous tokens, and is not designed to only focus on recent tokens. Without empirical evidence or references, the claim seems speculative.
2. Limited novelty. Masked-item (or token) prediction has been extensively studied, e.g., BERT4Rec. The authors argue their method differs because (1) they use a decoder instead of bidirectional encoders, and (2) they mask semantic tokens rather than item IDs. However:
    1. Decoder-style masked token prediction has been explored in language modeling (e.g., T5). I am not suggesting transferring objectives from other domains is inherently non-novel, but the contribution would be stronger with clear insights explaining why the proposed objective improves over next-item prediction, ideally supported by targeted analyses rather than only benchmark results.
    2. Masking tokens instead of items is a natural extension as generative recommender systems operate at the token level.
3. Model formulation is unclear. The token probability computation in Eqn. (5) and (6) is not well-defined. Typically, logits are produced by projecting the last hidden states at a specific position onto a token embedding table. As this work focuses on masked-token prediction, the choice of hidden-state position (mask position vs. the latest position) matters, and without clarity it is difficult to evaluate the method.
4. The RPG baseline shows substantially worse performance than what was reported in the original paper. While the authors state that the results were reproduced "using both the code and parameters from the authors (of RPG)", additional information is needed to support this claim. Simply reporting lower numbers without further explanation is not compelling. Providing more detail about the reproduction process, similar to how TIGER documented their reproduction of P5 in the appendix, would help readers evaluate the validity of the results. It would also be useful to know whether the authors communicated with the RPG authors to resolve discrepancies. I briefly checked the public issues in the RPG repository, where a reproduction discussion (issue #3) ultimately suggests that the reported numbers are reproducible.
5. The comparison against baselines may not be entirely fair. The proposed method leverages an auxiliary model to compute entropy scores, which introduces additional parameters. It is therefore difficult to disentangle whether the observed improvements stem from the proposed objective itself or from the benefit of this auxiliary model (or more trainable model parameters).
6. The pilot experiments appear problematic. The "full" and "truncated" settings seem to use different test sets; therefore, ranking metrics are not directly comparable.
7. Code is not available during the review phase.

### Questions
1. Why does next-item prediction fail to "understand the path", given the presence of self-attention over all tokens?
2. What evidence supports the claim that next-item prediction encourages local over global patterns?
3. Regarding weakness point 3: how are logits computed for masked tokens, using the corresponding masked position's hidden state, or always the final position?
4. Could the authors provide more information on reproducing the RPG baseline?

### Soundness
1

### Presentation
2

### Contribution
1

### Rating
2

### Confidence
5

---

## Human Reviewer 2

### Summary
This work targets the limitation of auto-regressive generative recommendation models, which primarily focus on next-item prediction and insufficiently exploit user history for preference understanding. The authors introduce Masked History Learning (MHL), a training paradigm that jointly performs next-item generation and masked reconstruction of past interactions. The framework incorporates entropy-guided masking to emphasize informative historical positions and a curriculum mechanism to gradually transition from history recovery to forward prediction. Experiments on multiple Amazon benchmarks demonstrate consistent improvements over state-of-the-art generative recommenders, indicating that explicitly leveraging masked historical signals enhances preference modeling in generative sequential recommendation.

### Strengths
1. Originality:
The paper addresses a weakness of auto-regressive generative recommenders by incorporating masked-history learning. While masked modeling is not new, its application to semantic-ID generation in recommendation represents a reasonable and relevant adaptation.
2. Quality:
The experiments are extensive and include strong baselines (e.g., RPG, TIGER, HSTU), with ablations conducted across masking strategies, curriculum phases, and semantic token settings. The methodology is technically sound.
3. Clarity:
The motivation, method framework, masking strategies, and scheduler are clearly explained and supported by diagrams. The appendices provide necessary implementation details.
4. Significance:
The results show consistent improvements, indicating that masked history learning provides value for capturing longer-range intent in generative recommendation.

### Weaknesses
1. The approach is largely an application and combination of existing concepts (masked modeling + curriculum learning) in a new context; conceptual innovation is incremental.
2. There is resemblance to BERT-style masked training and “auxiliary history prediction” paradigms in existing sequential recommendation works (e.g., BERT4Rec). A clearer distinction is needed.
3. Assumptions about user intent:
The narrative emphasizes long-term intent inference, but empirical proof is indirect. Would benefit from deeper analysis (e.g., long-sequence subsets, shift behavior settings).

### Questions
1. The paper primarily presents the framework of Masked History Learning(MHL); however, a more intuitive illustration is required to elucidate how MHL specifically identifies key historical information through the application of masks.
2. With respect to the interaction between masked history and auto-regressive modeling, how do masked history learning and future prediction objectives avoid conflicting with the training objectives?
3. Entropy calculation relies on the model's own prediction (Formula 10). In the early stage of training when the model has not yet converged, the estimation of entropy may not be accurate. How can the effectiveness of entropy-guided masks be guaranteed in the early stage of training?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
2

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper proposes Masked History Learning for generative recommendation. The method adds a masked history reconstruction task to a decoder only next item generator so the model learns both to predict the next item and to recover masked items in the user history. Mask selection uses predictive entropy to target uncertain positions. Training follows a curriculum with random masking first, then entropy guided masking, and finally fine tuning without masking. Experiments on three Amazon Reviews 2014 categories show gains over item ID and semantic ID baselines on Recall at K and NDCG at K, with ablations and sensitivity studies to support the design. A small study with title tokens suggests some transfer beyond discretised semantic IDs.

### Strengths
* The method is simple to add to existing semantic ID generators. 
* Results improve across three product domains and outperform TIGER, HSTU, and RPG. 
* The paper gives helpful implementation details and includes ablations on masking strategy and curriculum choice, along with sensitivity to codebook size and masking ratio. 
* The small text token study suggests the idea is not tied only to discretised semantic IDs.

### Weaknesses
* Masked item reconstruction in sequential recommendation is not new and appears in BERT4Rec and S3 Rec [1][2]. 
* Denoising reconstruction with encoder decoder training is common in BART and MASS [3][4]. 
* Generative recommendation with semantic identifiers was already introduced and scaled by TIGER and HSTU [5][6]. 
* The paper mainly repackages these parts for a decoder only generator with an entropy rule and a curriculum schedule. Evidence that the model captures user intent is indirect. 
* The main results rely on Amazon 2014 with leave one out splits and graph constrained beam search at inference, yet there is no decoding ablation to separate gains from the training signal versus the search procedure. 
* The paper does not report training overhead for entropy estimation and for the three phase schedule. The loss weights are fixed equal without a study of the trade off. 
* Results depend on a specific tokenization pipeline which raises sensitivity questions.


[1] Sun Fei Liu Jun Wu Jian Pei Changhua Lin Xiao Ou Wenwu and Jiang Peng. BERT4Rec Sequential Recommendation with Bidirectional Encoder Representations from Transformer. CIKM 2019.
[2] Zhou Kun Wang Hui Zhao Wayne Xin Zhu Yutao Wang Sirui Zhang Fuzheng Wang Zhongyuan and Wen Ji Rong. S3 Rec Self Supervised Learning for Sequential Recommendation with Mutual Information Maximization. CIKM 2020.
[3] Lewis Mike Liu Yinhan Goyal Naman Ghazvininejad Marjan Mohamed Abdelrahman Levy Omer Stoyanov Veselin and Zettlemoyer Luke. BART Denoising Sequence to Sequence Pretraining for Natural Language Generation Translation and Comprehension. ACL 2020.
[4] Song Kaitao Tan Xu Qin Tao Lu Jianfeng and Liu Tie Yan. MASS Masked Sequence to Sequence Pretraining for Language Generation. ICML 2019.
[5] Rajput Shashank Mehta Nikhil Singh Anima Keshavan Raghunandan Hulikal Vu Trung Heldt Lukasz Hong Lichan Tay Yi Tran Vinh Q. Samost Jonah Kula Maciej Chi Ed H. and Sathiamoorthy Maheswaran. Recommender Systems with Generative Retrieval TIGER. NeurIPS 2023.
[6] Zhai Jiaqi Liao Lucy Liu Xing Wang Yueming Li Rui Cao Xuan Gao Leon Gong Zhaojie Gu Fangda He Jiayuan Lu Yinghai and Shi Yu. Actions Speak Louder than Words Trillion Parameter Sequential Transducers for Generative Recommendations HSTU. ICML 2024.

### Questions
1) Could you report results with greedy decoding and without the graph constraint, then add the graph and beam search on top of each training variant? This would separate gains from the auxiliary reconstruction and masking from gains due to inference time search. 
2) How do results change when you sweep the weight ratio between the two losses? A simple decay of the reconstruction weight during training may change the balance between history fitting and next item prediction. Please provide a sweep table. 
3) Can you add targeted tests for long range behaviour? For example perturb early items, measure horizon k accuracy for k greater than one, or group users by coarse intent labels and report per group gains. The current pilot truncation is a helpful start but it can be expanded. 
4) Please give a focused ablation of the curriculum itself. Report random masking only, entropy guided masking only, random to entropy to inference as in the paper, and random to inference without entropy to show the specific gain of the middle phase. Also compare fixed mask ratio versus the adaptive ratio used in the second phase. 
5) What is the overhead of entropy guided selection and the full curriculum? Please include wall clock and tokens per second during training, and inference latency with and without the search graph. 
6) How sensitive are the entropy scores and the gains to the tokenizer choice? Please test different OPQ settings, a learned tokenizer, and a run without PCA, and relate these to the codebook size study.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper proposes a novel masked training framework for generative recommendation models.  The proposed method uses an entropy-guided masking policy that intelligently targets the most informative item. Extensive experiments are performed on three public datasets, where MHL is typically the best-performing method.

### Strengths
1. The authors enhance the prediction of the next item by reconstructing the auxiliary target of the masked items in the historical path.

2. The authors conducted extensive experiments on multiple public datasets (such as the Amazon series), and the results showed that the proposed model significantly outperformed the baseline model in indicators such as Hit@K and NDCG@K.

### Weaknesses
1. There is a lack of visualizations or case studies to support claims. Further analysis is needed to determine what the masked subtokens are and what characteristics they possess.

2. The authors do not provide a discussion of the model's operational efficiency or its application in large-scale industrial scenarios.

3. The paper lacks a theoretical explanation or analysis of the proposed Mask mechanism, and also lacks complexity or convergence analysis.

4. The author lacks a section explaining the use of LLM in the paper.

### Questions
1. Although the ablation experiment of masking ratio is given in Table 4, the parameter only ranges from 0.1 to 0.25, and there is no result greater than 0.25.

2. MHL claims to be able to identify users' long-term interests. Are there any quantitative results to show?

3. How do I obtain the semantic ID of an item? What VQ strategy should I use, k-means or VQVAE? How to set the hyperparameters? What are the corresponding ablation experiments and the rationale of the hyperparameter selection?

4. What is the additional cost of the method? How much is the additional cost compared to pure semantic ID-based GR?

5. Are there more experimental results to show how MHL compares with the latest semantic id-based baselines and benchmarks?

6. The dimensions of each symbol seem to require explicit explanation.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
5