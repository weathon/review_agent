# Node Identifiers: Compact, Discrete Representations for Efficient Graph Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
We present a novel end-to-end framework that generates highly compact (typically 6-15 dimensions), discrete (int4 type), and interpretable node representations—termed node identifiers (node IDs)—to tackle inference challenges on large-scale graphs. By employing vector quantization, we compress continuous node embeddings from multiple layers of a Graph Neural Network (GNN) into discrete codes, applicable under both self-supervised and supervised learning paradigms. These node IDs capture high-level abstractions of graph data and offer interpretability that traditional GNN embeddings lack. Extensive experiments on 34 datasets, encompassing node classification, graph classification, link prediction, and attributed graph clustering tasks, demonstrate that the generated node IDs significantly enhance speed and memory efficiency while achieving competitive performance compared to current state-of-the-art methods. Our source code is available at https://github.com/LUOyk1999/NodeID.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an end-to-end framework NID, which represents nodes with highly compact, discrete vectors, to deal with the inference challenges on large-scale graphs. Specifically, vector quantization is adopted to resident embeddings in each layer of the GNN, and NID can be optimized in either self-supervised or supervised learning ways. Extensive experiments show the effectiveness of the proposed framework.

### Strengths
1.	The paper is well-written and solid, especially practical in the case of large-scale graphs.
2.	The overview of the proposed framework helps readers to understand the main pipeline of the work.
3.	The experiment part is sufficient to clarify the effectiveness and efficiency of NID.

### Weaknesses
1.	The interpretability of the discrete representations of NID is not discussed in detail in the paper. As the authors point out that the most existed real-valued embeddings often lack interpretability, there should be a detailed discussion.
2.	Some concepts in the paper are vague. Refer to questions for more details. 
3.	The introduction of codebook seems useless since the codeword is used in neither loss of VQ nor loss of tasks.

### Questions
1.	What does it mean by “share similar 1-hop structures” in line 71 in the description of Figure 1? Why node in case A&B, C&D have similar 1-hop structure? What is the definition of so-called similar 1-hop structures? It is vague.
2.	Let’s extend the first question, so if two nodes have exactly the same second ID code in the figure 1 (but the first ID code is different), does that mean they have “similar 2-hop structures”? If yes, how can they have similar 2-hop structures on the basis of totally different 1-hop neighbors? Which neighbors of the 1-hop neighbors should be considered? If no, what does the same second ID code mean in that case?
3.	The definition of the M is vague. What does M mean in the size of codebook L*M?
4.	In subsection 3.2, the paper says that the Node_ID of each node will be sent into the MLP network, but my question is are you sending the id like “2341” (eg. 4-layer encoder) as the node representation into the MLP network? Why don’t you send the corresponding codeword into the MLP network? I know that code vectors are not used for a reconstruction task, so I am here talking about the downstream tasks.
5.	What will happen if two different nodes have the exactly same Node_ID? Since the size of each codebook is certain, let’s take figure 1 (2-dimensional) as an example, the codebook size of each layer is 5, then you actually can have 25 (5^2) different discrete representations. So there must be some nodes sharing the same Node_ID (collision). 
6.	For the argument in box 4.1, as I say, why directly use IDs as representations instead of corresponding vectors in the codebook?
7.	What is the interpretation of your discrete representations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new framework that creates compact and interpretable node representations, called node IDs, using vector quantization to convert GNN embeddings into discrete codes. Experiments across 34 datasets show that these node IDs improve speed and memory efficiency while maintaining competitive performance in various graph tasks.

### Strengths
1. The paper introduces a new framework for generating highly compact, discrete node representations (node IDs) that effectively addresses the inference challenges encountered in large-scale graph applications.
2. The authors conduct comprehensive experiments across 34 diverse datasets and tasks, demonstrating the effectiveness of their method. 
3. The paper is well-structured and easy to read.

### Weaknesses
1. The quantization of node embeddings into discrete node IDs may result in the loss of important structural information, as nuanced variations in node characteristics could be oversimplified into a single codeword. 
2. The framework necessitates joint training of the GNN and vector quantization components, which raises concerns about increased time complexity. The authors should provide an analysis of this complexity, particularly focusing on the impact of joint training on computational efficiency and scalability with larger datasets.

### Questions
How does the joint training of the GNN and vector quantization components affect the overall computational efficiency, especially when scaling to larger datasets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a vector quantization for GNN representations created from different "depths" of the network at each node.  Its similar to other recently proposed works (I'm most reminded of VQGraph), but its not trained with a reconstruction loss.  Extensive experimentation is provided, but it seems like most baselines are just copied from tables in recent work.  In general the quantization seems to work effectively, performing at about the same level as the original GNN.

Note: Interestingly when a baseline is recomputed by the authors (e.g. GCN) it seems to differ from the result reported in the paper the majority of the results come from (Polynormer).  This raises a substantial red flag, as the author's method should only perform as well as the baseline it quantitizes.  If we instead assume the author's methods perform within epsilon of the corresponding baseline in Polynormer, they would not be superior methods.

### Strengths
+ very well written paper about a pressing topic (graph quantization)
+ extensive results help provide details about the method
+ I'm confident the method seems to work well (but perhaps doesn't actually win on baselines, see weaknesses)

### Weaknesses
- Egregious experimental result reuse raises a number of inconsistencies for the few methods the authors have recomputed.
- Its difficult to place the results in context with the related work (ie the Polynormer paper results).

### Questions
Nice paper.  However when I dove into the experimental results of the Polynormer paper it raised a lot of questions.

1. The core issue is that in this paper you show weak methods (e.g. GCN) performing well on datasets.  
Lets take Photos for example.  In Polynormer, GCN is the weakest method (92%).  In your paper, GCN is the strongest method (96%).  This could be because you did a better grid search (good!) but if you don't similarly do a great grid search for the other baselines then I don't trust them at all :)

It makes it very hard understand how your method compares at all to the baselines.  (As a VQ paper the goal is not to really "beat" anything -- just do more with less)

### Soundness
3

### Presentation
3

### Contribution
3
