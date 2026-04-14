# MULAN: Multimodal Protein Language Model for Sequence and Structure Encoding

- Decision: Reject
- Scores: 5, 3, 5, 5

## Abstract
Most protein language models (PLMs), which produce high-quality protein representations, use only protein sequences during training.
However, the known protein structure is crucial in many protein property prediction tasks, so there is a growing interest in incorporating the knowledge about the protein structure into a PLM. Currently, structure-aware PLMs are trained from scratch or introduce a huge parameter overhead for the structure encoder. In this study, we propose MULAN, a MULtimodal PLM for both sequence and ANgle-based structure encoding. MULAN has a pre-trained sequence encoder and an introduced parameter-efficient Structure Adapter, which are then fused and trained together. According to the evaluation on 9 downstream tasks, MULAN models of various sizes show quality improvement compared to both sequence-only ESM2 and structure-aware SaProt as well as comparable performance to Ankh, ESM3, ProstT5, and other PLMs considered in the study. Importantly, unlike other models, MULAN offers a cheap increase in the structural awareness of the protein representations due to finetuning of existing PLMs instead of training from scratch. We perform a detailed analysis of the proposed model and demonstrate its awareness of the protein structure.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The authors introduced a new method called MULAN to enhance existing protein language models with angle-based structural information. By introducing a lightweight Structure Adapter module, they incorporated structural information efficiently. The authors evaluated their mothod on a wide range of tasks with comparison to relevant baselines. The experiments illustrate the effectiveness of MULAN. They also did ablation studies to highlight the structural awareness of MULAN embeddings.

### Strengths
The authors did many experiments to demonstrate the advantages of MULAN. They included many relevant works and made careful comparison among these methods. The details of experiments for pre-training and downstream fine-tuning are illustrated clearly, and the analysis of experimental results are comprehensive. Also they did many ablation studies to investigate the effectiveness of their method.

### Weaknesses
**1. Motivation** 

The authors mentioned that existing sequence-structure models tend to be large that may hinders their applicability in downstream tasks. Although MULAN only introduces a lightweight structural module, it still may not be easy to use depend on the size of backbone PLM, e.g. ESM-2 650M or 3B. The proposed method seems not to be advantageous from the prospective of the motivation.

**2. Lack of novelty.**

 There are many related works regarding protein sequence and structure co-modeling, as listed by the authors. The way of adding structural bias into per-residue embeddings has been introduced many times, such as ESM-3 and DeProt. The structural features are only torsion angles which are commonly used such as Foldseek. Also, the pre-training data derives from the Swiss-Prot database, which has been widely used and I suspect its diversity for both sequence and structure modality.

**3. About some claims**

Through extensive experiments, the authors stated that **MULAN is beneficial for different PLMs** and **The Structure Adapter is the key factor for improvement**. Although they did an ablation study showing that further pre-training ESM-2 on Swiss-Prot resulted in worse performance than incorporating MULAN, I would suspect whether it works for structure-based PLM, such Saprot. MULAN didn't get outstanding performance compared to other structure-based Models, and the features it used were very simple, even less than Foldseek which uses both torsion angles and spatial distances. Could the authors try pre-training Saprot without MULAN on their training set and see how the downstream performance would be?


**4. Paper writing.**

In part 2.1 MULAN Architecture, the authors introduced the structure features they used. However, I think they should include some basic introductions about what are torsion angles $\phi$ and $\psi$. Only including the name of angles is confusing and may result in misunderstanding about used features.

### Questions
Please see the questions proposed above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors of this study have identified an interesting gap in the growing PLM space, an approach to enrich PLM-based protein representations with 3D structural information. This is in contrast to existing models, such as ProstT5, ESM3, etc., that initially train on sequence and structural (and other modal) data. They do this in their model MULAN by incorporating a structural adapter module to pre-trained ESM2 models and fine-tuning using a subset SwissProt Alpha Fold predicted structures. They evaluate their protein representations in ten previously described residue- and protein-level tasks. Overall the performance of their model was comparable with existing sequence-only and structurally-aware models that they compared against.

### Strengths
The paper presents an interesting and direct approach to incorporate structure information to enhance pre-trained PLMs. The text is generally clear. The technical dimension is robust with the training of multiple models each with hyper-parameter optimization by grid search. The inclusion of an ablation experiment is important and good practice.

### Weaknesses
The paper builds on existing technology but is ultimately not novel. The paper itself cites multiple other models that incorporate structure in training or fine-tuning. The encoding of structure with amino acid torsion angles is not new (see https://academic.oup.com/bioinformatics/article/37/2/162/5892762 as one example). If the ultimate utility is in quicker training and inference, the authors should provide quantitative comparison and better motivate why this use-case is significant. Additionally, the performance of the model is mostly on par or slightly worse than existing models (Table 3), which the authors acknowledge but does diminish the significance. The differences in performance are also relatively small. 

The authors mention ProstT5 and ESM3 models in Section 6.2, which both include structural information in training. I think it is necessary to compare MULAN with these models to evaluate how well the their approach compares to these models. That way users can compare the performance and cost tradeoffs between the two approaches. 

Ultimately, the work is interesting and a nice extension of pLM work but the study is not novel in model design, training set curation or featurization, or evaluation tasks, with at best marginal performance improvement in a fraction of comparisons. It is therefore my recommendation to reject the manuscript.

### Questions
1) What fraction of residues are unmasked once the pLDDT threshold is applied?
2) Why is training limited to the Swiss-Prot fraction of Alpha Fold DB? There are millions of predicted structures available, which might boost performance of MULAN?
3) Can you explain some more about the tasks chosen for evaluation? Why these tasks? 
4) Why does MULAN perform worse with more parameters (Table 1)? [I found the presentation of this table with bold vs. not bold to be confusing.]
5) Have you considered that some of the downstream task proteins are present in the set that you use to Swiss-Prot set to fine-tune for MULAN? If there is overlap I think the fraction of proteins in each task present in the training set should be reported (could be in Table 5). It is also worth reporting if any of the CASP12, TS115, or CB513 structures are present in the training set. It could be that they are mutually exclusive, but this should be made explicit none the less.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents MULAN, an approach for enhancing protein language models (PLMs) with 3D structural awareness. Built on ESM-2, a sequence-only PLM, MULAN incorporates a specialized "Structure Adapter" module that processes protein dihedral angles. This allows MULAN to combine sequence and structure information efficiently, offering a substantial improvement in predictive performance across multiple downstream tasks, such as protein-protein interaction and molecular function prediction. MULAN’s structure adapter reduces training overhead by enabling the fine-tuning of pre-trained models without extensive structural pre-training. Experimental results show that MULAN consistently outperforms both ESM-2 and SaProt, a structure-aware PLM, indicating the effectiveness of the proposed structural integration.

### Strengths
1. MULAN features a straightforward design by incorporating a Structure Adapter module to effectively integrate protein sequence and structure information. This approach requires minimal changes to the base architecture and can directly leverage pre-trained ESM-2 checkpoints, significantly reducing training costs while achieving multimodal fusion.
2. MULAN is thoroughly evaluated across various downstream tasks, demonstrating consistent performance gains over baseline models. The ablation studies rigorously test critical components, such as the Structure Adapter and pLDDT masking, providing clear insights into the model’s effectiveness.

### Weaknesses
1. MULAN relies primarily on dihedral angle information for structural inputs, which, while useful, may not fully capture the complexity of protein structure. The integration of additional structural features, such as inter-residue distance matrices or atomic coordinates, or functionally relevant properties like solvent-accessible surface area (SASA), could potentially improve model performance on downstream tasks by providing a more comprehensive structural context.
2. The paper does not include comparisons with the latest structure-aware protein language models, such as ESM-3, which incorporate advanced structural representations. Comparing MULAN with these models would provide a clearer perspective on its relative advantages and limitations within the current landscape of structure-aware PLMs.

### Questions
1. Have the authors considered adding other structural features, like inter-residue distances, atomic coordinates, or functional information? Adding these could potentially improve the model’s structural understanding and task performance.
2. How were structural features combined with ESM’s sequence embeddings? Were any experiments conducted to compare these methods?
3. Has the model been tested for multi-task learning, handling tasks like stability prediction, function annotation, and interaction prediction simultaneously? Given the modular Structure Adapter, it would be interesting to see if MULAN can perform multiple tasks at once and how it compares to single-task models.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes MULAN, which fusing protein 3D structure information with 1D sequence information through a structure adapter. In MULAN, the protein 3D structure information is represented as angle vector of backbone and side-chain torsion angles. Then, the structure adapter employs the transformer layer to encode the structure information and fuse it with the 1D sequence information extracted from a pre-trained PLM (Protein Language Model). The structure adapter and the pre-trained PLM is finetune together. Experimental results are performed to evaluate the effectiveness of MULAN.

### Strengths
1.	As demonstrated in the paper, MULAN is computationally efficient. It only takes about 3 days for finetuning a large ESM2 with 650 M parameters.
2.	The experimental results on 10 downstream tasks demonstrate the effectiveness of MULAN, especially with relatively sample PLMs.

### Weaknesses
1.	The effectiveness of MULAN on large PLM is limited. To achieve best performance in the practical tasks, we usually need to employ the large PLM rather than small PLM. So, improving the effect of MULAN on large PLM is important.
2.	The overall framework of MULAN is quite similar to existing methods, such as ESM-GearNet and LM-GVP. The only difference may be representation of encoding of structure information. Previous employ geometric-aware graph neural networks to encode the protein 3D structure, while MULAN employs a transformer model. 
3.	More analysis should be conducted to analysis the advantages of employing the transformer model for encoding structure information rather than geometric-aware graph neural network.

### Questions
1.	The expression of line 480 is a little confusing. 
2.	In current framework of MULAN, the PLM models are full finetuned. What about applying PEFT (Parameter-Efficient Fine-Tuning) methods to PLMs?

### Soundness
3

### Presentation
3

### Contribution
3
