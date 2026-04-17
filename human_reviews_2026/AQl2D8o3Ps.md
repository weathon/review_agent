# REVEAL: Advancing Relation-based Video Understanding for Video-Question-Answering

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Video Question-Answering (Video-QA) comprises the capturing of complex visual relation changes over time, remaining a challenge even for advanced Vision-Language Models (VLM), i.a., because of the need to represent the visual content to a reasonably sized input for those models. 
To address this problem, we propose RElation-based Video rEpresentAtion Learning (REVEAL), a framework designed to capture visual relation information by encoding it into structured, decomposed representations. 
Specifically, inspired by spatiotemporal scene graphs, we propose to encode video sequences as sets of relation triplets in the form of (\textit{subject-predicate-object}) over time via their language embeddings. 
To this end, we extract explicit relations from video captions and introduce a Many-to-Many Noise Contrastive Estimation loss (MM-NCE) 
together with a Q-Former architecture to align an unordered set of video-derived queries with corresponding text-based relation descriptions.
During inference, the resulting Q-former produces an efficient token representation that can serve as input to a VLM for  Video-QA.

We evaluate the proposed framework on five challenging benchmarks: NeXT-QA, Intent-QA, STAR, VLEP, and TVQA. It shows that the resulting query-based video representation is able to outperform global alignment-based CLS or patch token representations and achieves competitive results against state-of-the-art models, particularly on tasks requiring temporal reasoning and relation comprehension. The code and models will be publicly released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents REVEAL (RElation-based Video rEpresentAtion Learning), a framework for Video Question Answering (VideoQA) that models videos as temporal relation triplets (subject–predicate–object) derived from captions. A Q-Former encodes video frames into query embeddings aligned with textual relations through a new Many-to-Many Noise Contrastive Estimation (MM-NCE) loss, enabling fine-grained visual–text alignment. Experiments on five benchmarks (NeXT-QA, Intent-QA, STAR, VLEP, TVQA) demonstrate competitive performance. Key contributions include relation-based video encoding, the MM-NCE alignment loss, and extensive evaluation across multiple datasets.

### Strengths
The paper is original in reformulating VideoQA through relation-based video representation and introducing the MM-NCE loss for aligning unordered multimodal sets. It creatively integrates ideas from scene graphs and contrastive learning into a unified, scalable framework.

### Weaknesses
The method’s novelty is limited. Theoretical justification for MM-NCE and evidence of its alignment correctness are lacking. Experimental gains are modest, with missing comparisons to recent methods. Writing clarity, figure details, and reference formatting also need significant improvement.

### Questions
1. The paper has significant issues in terms of writing format (e.g., reference citation style) as well as overall logical flow and clarity of presentation. For instance, the textual details in Figures 1 and 2 are unclear and require careful revision. Substantial effort is needed to improve the overall readability and structure.

2. The main claimed contribution lies in introducing an additional pretraining stage to enhance performance on the VideoQA task. Regardless of the level of novelty, the proposed pretraining essentially serves as a form of fine-grained alignment. Has the paper compared this approach with other fine-grained alignment methods? Moreover, in the Many-to-Many Noise Contrastive Estimation module, how is the alignment correctness between q and r ensured? Is there any theoretical or qualitative analysis to support this design?

3. From the experimental results, the proposed method does not demonstrate clear advantages over prior work—particularly in Table 2. In addition, several tables lack comparisons with the latest state-of-the-art methods.

### Soundness
2

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
3

### Summary
The paper "REVEAL: Advancing Relation-Based Video Understanding for Video-Question-Answering" introduces a novel framework that significantly enhances VideoQA by modeling video content as a structured collection of temporal (subject-predicate-object) relation triplets, departing from traditional global video representations. A central technical innovation is the Many-to-Many Noise Contrastive Estimation (MM-NCE) loss, which elegantly aligns unordered and incomplete sets of visual queries with text-derived relation descriptions using Hungarian matching, thereby robustly learning from non-exhaustive, web-supervised data. This modular framework, comprising dual-pathway vision encoders, temporal encoders, a Relation Q-Former, and a text-based Relation Encoder, seamlessly integrates with Large Language Models (LLMs) via adapters. Comprehensive experimental evaluation across five challenging VideoQA benchmarks (NeXT-QA, Intent-QA, STAR, VLEP, TVQA) demonstrates REVEAL's competitive or superior performance, especially in tasks requiring deep temporal and relational reasoning, underscoring the efficacy and promise of its relation-centric paradigm for more interpretable and robust video understanding.

### Strengths
1.REVEAL introduces a unique and promising paradigm by explicitly modeling video content as sets of temporal (subject-predicate-object) relation triplets. This contrasts with traditional global or patch-token representations, offering a finer-grained, more interpretable approach to capture interactions and evolution within videos, crucial for complex temporal and relational reasoning in VideoQA.
2.The proposed MM-NCE loss is a key technical innovation. It cleverly addresses the challenges of unordered set alignment and incomplete annotations in video-language learning. By using Hungarian matching for optimal query-relation correspondence in contrastive learning, MM-NCE efficiently learns from non-exhaustive, web-supervised relation data, enabling the model to infer relevant relationships and learn more general representations without penalizing missing annotations.
3.REVEAL demonstrates its effectiveness through extensive evaluation across five challenging VideoQA benchmarks (NeXT-QA, Intent-QA, STAR, VLEP, TVQA). It achieves competitive, and in many cases superior, performance against state-of-the-art models, particularly on tasks requiring temporal reasoning and relation comprehension. This robust experimental validation reinforces the benefits of relation-based representations for complex video semantics.
4.The REVEAL framework boasts a modular design, with distinct visual, temporal, Q-Former, and relation encoders. This architecture offers high flexibility and scalability, allowing visual features to be transformed into efficient tokens that seamlessly integrate with existing LLMs via adapters. The use of Mistral-7B for web-supervised relation extraction further highlights its potential for efficient knowledge distillation. This design facilitates future advancements and broader applications.

### Weaknesses
1.The core claim that relation triplets offer superior, structured representations for VideoQA, especially when connecting with LLMs, needs stronger evidence. While Table 6.a shows improved performance over caption-based NCE, it doesn't adequately demonstrate why (subject-predicate-object) triplets are intrinsically better or more efficient than other sophisticated global video representations or longer, richer textual descriptions.
2.The paper claims that its design choice to "not penalize missing relationships during training" is validated by the non-exhaustive nature of extracted relations. However, the theoretical underpinnings of how MM-NCE effectively handles incomplete relation sets without introducing noise or bias, and how the model reliably infers relevant relationships from unannotated content, require deeper theoretical analysis and empirical validation beyond a mere assertion.
3.While MM-NCE is presented as a solution for aligning unordered and incomplete sets, its unique theoretical advantages and practical distinctions from other advanced multi-to-multi matching or contrastive learning techniques (e.g., graph-matching, attention-based, or Transformer-based alignment) are not thoroughly analyzed. The mechanism for handling asymmetric set sizes (J(k) < M and M < J(k)) also needs clearer explanation.
4. The use of Mistral-7B for automated relation triplet extraction lacks comprehensive evaluation regarding its robustness, accuracy, and potential biases (e.g., preference for certain verbs, errors in complex sentences, or propagating dataset stereotypes). The quality gap between these automatically generated relations and human annotations, particularly with ambiguous video captions, is a significant unaddressed weakness.

### Questions
1.How does the explicit relation triplet representation intrinsically outperform other sophisticated global video representations for LLM-based VideoQA, considering potential information loss from unstructured captions?
2.How does MM-NCE's "free inference" for unannotated relations ensure robustness and prevent learning inaccurate or hallucinatory relationships, beyond merely not penalizing missing ones?
3.What are the precise theoretical and empirical distinctions of MM-NCE from other advanced multi-to-multi matching or contrastive learning techniques, especially regarding asymmetric set sizes and one-to-one correspondence?
4.Could you provide quantitative evaluation metrics (e.g., F1 score) for the Mistral-7B relation extraction against a reference set, and how does its quality impact downstream VideoQA performance and robustness?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces REVEAL, a relation-based video representation learning framework for VideoQA. The method encodes videos as sets of (subject, predicate, object) relation triplets extracted from captions via LLMs, then aligns visual “relation queries” (from a Q-Former) with text-based relation embeddings using a new Many-to-Many Noise Contrastive Estimation (MM-NCE) loss and Hungarian matching. The model is pretrained on WebVid-2M and evaluated on five VideoQA datasets (STAR, NExT-QA, Intent-QA, TVQA, VLEP), showing competitive or superior performance compared to recent VLM-based approaches.

### Strengths
-	The experimental evaluation is comprehensive, covering multiple challenging VideoQA datasets.

-	Relation understanding is a meaningful direction for improving compositional and semantic understanding in video-language learning.

### Weaknesses
-	The method assumes that vision queries and text-based relation embeddings are already semantically aligned (by choosing positive pair with Hungarian matching and cosine similarity), relying on representations. The MM-NCE loss primarily refines this existing bias rather than learning a new alignment.

-	The representation learning objective focuses solely on static object-level relations, ignoring temporal and causal relationships between events or entities. It is unclear how REVEAL achieves strong performance on causal or temporal reasoning benchmarks (e.g., Intent-QA, NExT-QA) compared to methods like Vamos, which explicitly incorporate temporal reasoning using the same backbone.

### Questions
-	Can you please explain why the method relies on the bias of pretrained models at the beginning and then refines it through contrastive learning? What happens if the relation understanding in those pretrained models is incorrect?
-	Is an objective that focuses only on object–object relation modeling truly effective for video understanding? How does this approach help the model improve on temporal or causal questions?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes REVEAL, a framework that models videos as (subject-predicate-object) relation triplets. REVEAL extracts relations from captions via Mistral-7B, uses a Q-Former to generate video-derived visual queries, and introduces MM-NCE loss to align these queries with text-based relation embeddings. It adopts a Slow-Fast dual pathway and is evaluated on 5 benchmarks, at least competing with state-of-the-art models.

### Strengths
Overall, this paper is well-written, with clear motivation for the research problem and systematically designed experiments.

It effectively redefines video representation for VideoQA by modeling video content as (subject-predicate-object) relation triplets, moving beyond the limitations of traditional global alignment methods or closed-vocabulary scene graphs.

It proposes the MM-NCE loss to align unordered, incomplete sets of visual queries and text-based relation embeddings—successfully addressing the challenge of variable relation counts per video, a gap that prior losses (such as MIL-NCE) failed to cover.

Additionally, it adapts the Slow-Fast architecture and Q-Former for relation modeling (using the fast pathway for temporal aggregation and the slow pathway for fine-grained spatial details), forming a novel and practical combination.

### Weaknesses
1. The paper compares REVEAL to models like ViLA and VideoChat but omits post-2024 VLMs such as Qwen2.5-VL. This gaps makes its "competitive against SOTA" claims less convincing for current research.

2. The paper uses unique hyperparameters for each dataset, unlike baselines that often use unified settings. This could inflate REVEAL’s gains , as improvements might stem from tuning rather than its core design.

3. Given the paper currently only evaluates REVEAL on multiple-choice VideoQA benchmarks, its arguments would be more convincing if the authors considered involving more recent or challenging benchmarks, such as free-ended QA benchmarks like MSVD and MSRVTT.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
