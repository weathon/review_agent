# Multimodal Aligned Semantic Knowledge for Unpaired Image-text Matching

- Decision: Accept (Oral)
- Scores: 6, 6, 8

## Abstract
While existing approaches address unpaired image-text matching by constructing cross-modal aligned knowledge, they often fail to identify semantically corresponding visual representations for Out-of-Distribution (OOD) words. Moreover, the distributional variance of visual representations associated with different words varies significantly, which negatively impacts matching accuracy. To address these issues, we propose a novel method namely Multimodal Aligned Semantic Knowledge (MASK), which leverages word embeddings as bridges to associate words with their corresponding prototypes, thereby enabling semantic knowledge alignment between the image and text modalities. For OOD words, the representative prototypes are constructed by leveraging the semantic relationships encoded in word embeddings. Beyond that, we introduce a prototype consistency contrastive loss to structurally regularize the feature space, effectively mitigating the adverse effects of variance. Experimental results on the Flickr30K and MSCOCO datasets demonstrate that MASK achieves superior performance in unpaired matching.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses an underexplored issue in cross-modal matching tasks: image-text matching without pairing. The authors propose a new method called MASK, whose core is to construct a cross-modal aligned semantic knowledge base. The model uses word embeddings as a bridge to match words with their prototypes, and by introducing prototype consistency contrastive learning loss to regulate the feature space, it overcomes the shortcomings of existing methods in handling out-of-distribution words and variance of visual representation distribution. Experiments on the Flickr30K and MSCOCO datasets show that MASK achieves leading performance in the unmatched matching task and can effectively serve as a re-ranking module to enhance the performance of existing pre-trained multimodal models.

### Strengths
1.	The framework proposed in this paper is intuitive and effective. It utilizes the semantic relationships of word embeddings to construct prototypes for out-of-distribution (OOD) vocabulary and a prototype consistency contrastive learning loss, clearly pointing out the key weaknesses of existing methods and having a clear motivation.
2.	This paper introduces external pre-trained word vectors as an auxiliary supervisory signal to establish an equivariant mapping that preserves the relationship between regional representations and word embeddings. This enables regional representations to effectively capture the semantic relationships between words.
3.	The experimental results on Flickr30K and MSCOCO show that MASK outperforms existing models and knowledge-based methods significantly in the image-text matching task, without paired images, proving its effectiveness.
4.	MASK can acquire a kind of logical semantic knowledge based on conceptual prototypes through structured learning. Uses MASK as an reordering enhancement plugin for CLIP and ALBEF models, demonstrating its effectiveness in prototype consistency contrastive learning and semantic relationship alignment.
5.	The appendix provides mathematical proofs for concepts such as peer-to-peer transformation mapping and the rationality of loss functions, enhancing the rigor and depth of the method. This is particularly valuable in applied research papers.

### Weaknesses
1.	The explanation of the OOD vocabulary processing mechanism is unclear. This is one of the core innovations of this paper. However, the description in Section 3.4 is overly mathematical and lacks an intuitive, step-by-step concrete example. For instance, when encountering a word that does not exist in the knowledge base, how can we use the Glove word vectors to find similar known words and weight them to synthesize their visual prototypes? The current explanation remains at the formula level, which is less readable and reduces the comprehensibility of the method.
2.	To construct the prototype of the OOD vocabulary, we need to sample m pairs of representations from the knowledge base. The paper does not explain at all how this sampling is carried out. Is it random sampling? Or is it Top-K sampling based on the semantic similarity with the OOD words? Different sampling strategies will greatly affect the quality of the final prototype, which is an important unspecified hyperparameter and must be clarified.
3.	The paper did not analyze the overlap between the vocabulary in the test set and the VG's 12,385 concepts. If the vast majority of nouns in the test set are already present in the knowledge base, then the challenge and performance improvement attributed to OOD words might be overestimated. It is necessary to clearly define the specific definition and proportion of OOD in the experiments.

### Questions
1.	Does the MASK method have good generalization ability on other types of datasets? For instance, how does the model perform on some datasets that contain more domain-specific words or more complex visual scenarios? Is there a plan to verify it on more diverse datasets?
2.	If certain words are not included in the pre-trained word vectors, or if the semantics of the words have changed in a specific domain, how will the MASK method handle this situation? Has there been any consideration of reducing reliance on the pre-trained word vectors or introducing other methods to obtain the semantic information of the words?

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
The paper proposes MASK (Multimodal Aligned Semantic Knowledge), a framework for unpaired image-text matching that leverages external semantic knowledge to bridge the gap between textual and visual modalities.
Instead of relying on paired image-text data, the method constructs visual prototypes and uses pretrained word embeddings to generate prototypes for out-of-distribution (OOD) words through a weighted combination of existing ones.
A prototype-consistency contrastive loss is further introduced to reduce intra-class variance and enhance alignment stability between textual and visual representations.
Extensive experiments on Flickr30k and MSCOCO demonstrate competitive retrieval performance and validate the benefit of incorporating external knowledge.
The approach can also serve as a lightweight re-ranking module to enhance existing vision-language models.

### Strengths
1. This paper proposes a novel cross-modal semantic alignment method, MASK, which constructs representative prototypes for OOD words by exploiting the intrinsic relationships amongword embeddings.

2. The paper provides solid comparisons with both model-based and knowledge-based baselines on Flickr30k and MSCOCO, together with ablation studies, sensitivity analysis, and visualization of learned representations.
3. The proposed method can serve as a lightweight plug-in re-ranking module, enhancing existing vision-language models such as CLIP or ALBEF without retraining them end-to-end.

4. This paper introduce a prototype consistency contrastive learning loss to structurally regularize the feature space, mitigating the adverse impact of distributional variance.

### Weaknesses
1. Over-strong assumption and circular reasoning in the OOD prototype proof
The theoretical analysis in Appendix B relies on a strong linear/isometric assumption that equates textual and visual prototype spaces. Moreover, the proof assumes the existence of a relation-preserving mapping 
f,which is exactly what the model is trained to learn—thus creating a circular argument. Additional experiments or a more rigorous justification are needed to support this claim.

2. High dependence on region proposals
The method heavily relies on the quality of the region proposals provided by upstream object detectors. The experiments show large performance gaps between different detectors (e.g., BUTD vs. DETR/DINO), indicating that the framework’s robustness and generalization depend strongly on detector performance.

3. Insufficient analysis of pretrained word embeddings
The approach uses pretrained word vectors to construct OOD prototypes, but the effect of different embeddings (e.g., GloVe, word2vec, fastText) is not explored.  Since the semantic geometry of word embeddings directly affects OOD prototype generation, additional ablation or sensitivity analysis would strengthen the paper.

4. Missing discussion on the parameter m for OOD prototype construction.
In Eq.(8), OOD prototypes are computed by weighting m known prototypes, but the paper does not explain how m is chosen or provide related experiments. The selection of m could significantly affect the semantic quality of constructed prototypes.
Including an analysis or sensitivity study on this parameter would help clarify its impact and strengthen the conclusion.

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose MASK to address the issue of unpaired ITM by focusing on OOD words. 
(1) They use relationship among word embeddings to construct prototypes of OOD words. 
(2) A new consistency contrastive loss is proposed to make compact prototypes. 
(3) Pretrained word embeddings are used for relation-preserving equivalent mapping. 
Abound experiments are performed to illustrate the effectiveness and efficiency of MASK.

### Strengths
[1] MASK surpasses MACK[NeurIPS 22] and MACK++[TPAMI 24], which are the 1st work for unpaired ITM. So, MASK is the new SoTA work. 

[2] MASK made several smart designs(contributions): (1) Information retention loss: a self-supervised noise-adding objective, in which, 
FRM module will ONLY be used in knowledge construction phase, but NOT in infer phase. (2) External Textual knowledge(=GloVe) is used, which can be considered as auxiliary GT info, especially in Eq.(9) with the help of MTM module. (3) The semantic relationship between(among) words are captured(used/exploited) for enhancing generalization, which is established in Eq.(9) and Eq.(12).

### Weaknesses
[1] All of the weaknesses, please see the following Questions Part.

### Questions
[1] In Table 7，it seems that the table title should NOT be “detectors” but “architecture”. Otherwise, it will be the same as the title of Table 9. 
※ p4 ln210: Lcl is defined as follows D: WHAT is D? Appendix D? 
※ p5 ln233: f should satisfy B: WHAT is B? B!=8? You mean Eq.(8)? 

[2] It will be better if the authors give an answer about Fig. 2 that 
WHETHER the (1) Pretrained F-RCNN and (2) Pretraind word vectors 
will BP (which means: requires_grad == True)? 
※ I guess BOTH of them do NOT need grad to BP. 
※ Is there MORE module that also NOT need BP? F-RCNN? 

[3] OOD word/vocab is relative, NOT absolute. Because Glove (pretrained word emb) is likely to know so called “OOD word”. So, “OOD” is relative to CLIP/ALBEF (pretrained VLM) or F30k/COCO (dataset). 
※ In addition, in Fig. 2, “cat” seems to be a quite common seen word, NOT OOD. 
“otter[≈≈freshwater carnivorous mammal]/manga[≈≈comic]” may be more suitable. 
So, a more SOUND example should be given, especially in REAL TEST VISUALIZATION. 
Please give us some REAL test examples about your MASK with REAL OOD word. 
e.g. in T2I retrieval, a text query with OOD word makes original CLIP model search the GT image on Top 3, but after your MASK rerank, the GT image gets up to Top 1. 
You can make VISUALIZATION about the Pos as well as Neg on I2T and T2I (F30k). 

[4] What about the Neg (Wrong) example visualization? Is there any OOD word? 
How to EXPLAIN it? WHY it makes wrong? What TYPE of word it is? verb/prep or n./adj.？
※ We know that, verb(v.), preposition(prep.) are harder than noun(n.), adjective(adj.). 
Is there any method to make v./prep. understanding more precise? 
※ How to make these MISTAKES to be alleviated? MORE effective methods? 
※ We AGGRE WITH your work, but we also wonder HOW it is WORKED indeed ! 

[5] Big Models (Large Language Models) are popular, and may have super high score now. 
We want to know how many scores your MASK method can improve? 
※ Just Eval, DO NOT need any finetune/train! WHY NOT Keep pace with the times! 
※ SigLIP v1 can get 570.84 score on f30k 1K Test. We have already tested! 
see: https://huggingface.co/google/siglip-so400m-patch14-384/tree/main
※ Since Big Model can get 570+ score, CAN Big Model+MASK get 600 score (full score)? 

[6] More powerful region encoder, may lead to more score! 
※ F-RCNN ResNet 152 is more powerful than F-RCNN ResNet101. 
※ More SOTA image/region encoder for More Ablation studies? Swin Transformers? 

[7] What if w’ in the second term of Eq.(9) change into w? 
※ Because we think w is GT, but w’ not. Maybe GT is better. 

[8] p5 ln261: we first sample m paired ...
※ How to sample? I don’t understand. In Appendix? 
※ I guess: using GloVe(=pre-trained word embeddings) to get Top m high score word.

### Soundness
3

### Presentation
3

### Contribution
3
