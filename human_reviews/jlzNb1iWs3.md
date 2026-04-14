# The OMG dataset: An Open MetaGenomic corpus for mixed-modality genomic language modeling

- Decision: Accept (Poster)
- Scores: 5, 8, 6, 5

## Abstract
Biological language model performance depends heavily on pretraining data quality, diversity, and size. While metagenomic datasets feature enormous biological diversity, their utilization as pretraining data has been limited due to challenges in data accessibility, quality filtering and deduplication. Here, we present the Open MetaGenomic (OMG) corpus, a genomic pretraining dataset totalling 3.1T base pairs and 3.3B protein coding sequences, obtained by combining two largest metagenomic dataset repositories (JGI's IMG and EMBL's MGnify). We first document the composition of the dataset and describe the quality filtering steps taken to remove poor quality data. We make the OMG corpus available as a mixed-modality genomic sequence dataset that represents multi-gene encoding genomic sequences with translated amino acids for protein coding sequences, and nucleic acids for intergenic sequences. We train the first mixed-modality genomic language model (gLM2) that leverages genomic context information to learn robust functional representations, as well as coevolutionary signals in protein-protein interfaces and genomic regulatory syntax. Furthermore, we show that deduplication in embedding space can be used to balance the corpus, demonstrating improved performance on downstream tasks. The OMG dataset is publicly hosted on the Hugging Face Hub at https://huggingface.co/datasets/tattabio/OMG and gLM2 is available at https://huggingface.co/tattabio/gLM2_650M.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors present the Open MetaGenomic (OMG) dataset, a substantial, mixed-modality genomic corpus designed to support biological language model training. OMG is sourced from two major repositories, JGI’s IMG and EMBL’s MGnify, encompassing 3.1 trillion base pairs and 3.3 billion protein-coding sequences. The authors address critical challenges in metagenomic data usage, including accessibility, quality filtering, and deduplication. They also introduce gLM2, the first mixed-modality genomic language model pretrained on OMG with 150M and 650M parameters, capable of leveraging both genomic and protein-coding information to capture complex functional and regulatory patterns. gLM2 demonstrates decent improvements in functional representation learning of both protein and NA and co-evolutionary information modeling, outperforming existing models on tasks involving genomic structure prediction, protein-protein interactions, and regulatory syntax learning.

### Strengths
1. **High-Quality, Diverse Metagenomic Dataset**: By aggregating two of the largest metagenomic repositories, the authors create a richly diverse dataset that addresses critical gaps in sequence variety and covers underrepresented taxa through rigorous filtering and preprocessing. This dataset is likely to become an essential training resource for future models, representing a major contribution to the field.

2. **Mixed-Modality Model**: The authors propose an innovative modeling approach that integrates both protein and nucleic acid sequences, enabling the model to process protein-coding and regulatory sequences together. This combined approach opens the door to unique and complex downstream applications.

3. **Co-Evolutionary Signals**: This work may be the first biomolecular language model to demonstrate coevolutionary signals across sequences using the Categorical Jacobian method, marking a novel and compelling contribution to biomolecular modeling.

### Weaknesses
Since I don't have much experience in metagenomic and its relevant data processing, I will leave these aspects to other reviewers. Here, I will focus more on gLM2 model and performance.

1. The model is evaluated on the DGEB benchmark. However, insufficient information is provided about the evaluation methodology and associated metrics. I recommend adding a section discussing the evaluation process and metrics in detail.

2. After checking the original DGEB paper, it appears that gLM2 can not match ProGen2 on amino acid (AA) tasks (Appendix J), which is not reported in the current manuscript. The authors should either include these results or justify their omission.

3. From a protein language model perspective, the evaluation needs to be more comprehensive. The authors should include evaluations on standard benchmarks such as FLIP or ProteinGym. 

4. For the coevolution signal extraction experiment with 2ONK, the comparison would benefit from including NT2 and ProGen2 as additional baselines.

5. While sections 4.4 and 4.5 present well-analyzed individual cases, for contact prediction in particular, more examples are needed to demonstrate that gLM2 consistently outperforms alternative methods. Might be interesting to consider datasets like [1][2] for RNA and  CAMEO for protein.


[1] https://github.com/jaswindersingh2/SPOT-RNA-1D
[2] https://pmc.ncbi.nlm.nih.gov/articles/PMC7297115/

### Questions
1. Regarding the training and modeling capabilities of gLM2:
   - Is gLM2 trained on contig sequences containing both nucleotide sequences (IGS) and protein sequences? If so, does this limit the model to only modeling interactions between bio-sequences from the same contig? Can gLM2 predict contacts for Protein hetero-multimer interactions for example (also RNA-Protein interactions and DNA-Protein interactions)?


2. For the 2ONK example:
     * Are chains A and C from the same contig?
     * If yes, is there IGS between them?
     * What would be the effect on coevolution signal extraction if IGS were included?
     * Why were only chains A and C analyzed when 2ONK has 4 chains?

3. How ESM2 FLOPs were estimated? Or if this information comes from existing literature (e.g., the original ESM2 paper).

4. Regarding model hyperparameter design:
   - What is the rationale behind the chosen head number, layer number, and embedding dimensions?
   - Were preliminary experiments conducted to support these choices?
   - Contemporary LLMs like LLaMA typically use equal numbers of heads and layers, with embedding dimensions set to 128 times the number of heads

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The Open Megagenomic Corpus (OMG) and gLM2 model are important and novel contributions to biologic sequence modeling. Authors make use of the largest repositories of metagenomic data to construct a new database for training that is larger than exiting, popular biologic LM training datasets. Using both coding sequences and intergenic sequences, authors show that their modeling strategy captures long range interactions of coding and non-coding genomic elements at multiple scales of biologic sequence interaction.

The paper should be accepted given the technical and strategic innovation database creation and the novelty in modeling strategy that performs on par with an existing state-of-the-art model and captures important biology across sequence modalities.

### Strengths
The work addresses the need for larger databases of sequence data that can be used to train language models for biologic applications by making use of rapidly growing repositories of metagenomic sequence data. Authors utilize knowledge and methods from recent insights in NLP to construct the large database used for training and develop their own genomic deduplication procedure that captures biologic reality. They also utilize deep domain knowledge about metagenomic sequencing to filter terabytes of raw data for database construction. The new curated database is made available to the community. The authors utilize a novel modeling approach, constituting of which includes multi-modal input in the form of amino acid and nucleotide data.

### Weaknesses
The language model trained does not explicitly include eukaryotic genomic sequences and it is unclear gLM2 is meant to be used to model eukaryotic sequences. Authors should be more explicit about the application for eukaryotic sequence modeling. Authors do mention contamination and likely inclusion of eukaryotic sequences in metagenomic samples but this in in the context of challenges and mitigation thereof with metagenomic sequence data.

Applications of the gLM2 model are not explored. It would be interesting to see performance of gLM2 in areas where more and diverse training data is of importance, such as remote homology, or where the multi-modal training strategy is helpful, such as contig-level prediction of taxonomy.

### Questions
Authors claim their modeling approach achieves "more efficient scaling" performance with flops on a set of benchmark tasks than a prior top performing model; however, they only show two training flops. When looking at Appendix J there are some tasks where scaling is observed and some where it does not. It would be a stronger claim if there were additional data points for gLM2. On a per-task basis, it appears that gLM2 has similar scaling to ESM2 even if at the aggregated score level the scaling looks slightly better (Figure 3).

Minor comments:
1. "start a new contig" should be "start a new element" (l.255, 266-267)
2. DGEB is not defined (l.333)
3. Figure 5 has two "n" that are mentioned but not present in the figure

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper reports a large, well-curated, database of genomic sequences (protein coding and nucleic acids, genomic and metagenomic) with utility as a genomic pretraining dataset for biologic modeling. The dataset is unique and the authors demonstrate its utility in training a novel mixed-modality genomic language model (gLM2) that leverages genomic context information to learn functional representations and coevolutionary signals in protein-protein interfaces and genomic regulatory syntax.  These are very different kinds of tasks and that the model performs well in the vignettes shown is very promising for both the OMG corpus and the gLM. The gLM is not well described or validated; the corpus itself (OMG) is the focus of the paper. I agree that OMG is a very useful resource for the community, particularly in combination with gLMs but the authors do not sufficiently validate their gLM aside from a few interesting but poorly described vignettes.

### Strengths
OMG is a novel resource that will be useful for pretraining models to address many different biological questions
the gLM is a novel way of representing genomic architecture and function simultaneously and could, as such be a very powerful resource

### Weaknesses
the gLM is poorly described and not validated, there are a few small vignettes describing it but these do not get at validation or generalizibility for the very interesting tasks described.  I have a hard time giving this paper the strongest recommendation because of these concerns.

### Questions
Is there a reason the gLM wasn't validated in some way? The examples seem very cherry picked, can you give us some sense of how you'd go about doing the validation for these very different biological questions (regulation, protein-protein interaction, etc). Do you think the gLM is really generalizable for all these different tasks? Where it struggles, why do you think it struggles?

### Soundness
4

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
3

### Summary
The authors present open metagenomic datasets combined from two separate sources and filtered to maintain high-quality data. They further present the data set's utility in a trained model.

### Strengths
The authors give an excellent introduction to the field of metagenomics. They adequately describe the difficulties of balancing and filtering biological data sets and how some issues are exacerbated for metagenomic specifically. They detail each of the steps required to filter the data. Finally they open the data sets to the public to use.

### Weaknesses
While the methods for data filtering are explained, their impact is not always clear.
How many elements are removed for the edge-element removal?  
For example, in the assembly filtering, a histogram of the number of invalid characters, why a threshold of 20% was chosen? What is the meaning of "(>3 CDS >7 ..."?
For the element length-based filtering section, again, what is the impact of why the specific threshold was chosen? Could you provide a supporting figure?
Essentially, This presents a novel data set whose importance is still unclear.

### Questions
The importance of the results is not explained; could you present this with other protein-to-protein-related works? 
Which conclusion can I reach with this data that I couldn't other wise?

### Soundness
2

### Presentation
2

### Contribution
2
