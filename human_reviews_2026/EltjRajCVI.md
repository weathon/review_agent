# Emergent Chess Skill Acquisition in Large Language Models

- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
We investigate the emergent behaviors of rule comprehension, tactical execution, and strategic competence in transformer-based models trained on algebraic chess notation. To support structured reasoning, we introduce a disambiguation-aware tokenization scheme that explicitly encodes promotions, castling, checks, and mates, enabling fine-grained modeling of chess rules and dynamics.

Our analysis reveals phase transitions in capabilities: shallow models fewer than 15 layers exhibit high illegality rates, while deeper models 20 layers or more increasingly demonstrate reliable tactical and positional behaviors. Training dynamics show while rule comprehension emerges early, higher-order abilities follow a hierarchical developmental path that mirrors curriculum learning. These trends remain consistent across decoding strategies and training distributions. 

Our findings suggest that transformer models can acquire human-aligned planning abilities in symbolic domains. Chess provides a tractable benchmark for evaluating the staged emergence of hierarchical competence in language models. Our methodology, including vocabulary design, architectural scaling, and behavioral evaluation, has the potential to generalize to other structured domains such as programming, formal logic, and mathematical proof systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates chess skills in decoder-only transformer models trained from scratch on algebraic chess notation. The authors focus on the training dynamics and developmental trajectory of these skills, rather than final performance. They systematically vary model depth (5 to 25 layers) and the training data distribution (a balanced dataset vs. a white-win-only dataset). Using a custom, disambiguation-aware tokenization scheme, they analyze the emergence of three hierarchical levels of competence: rule comprehension, tactical execution, and strategic planning. The paper concludes that chess provides a valuable, interpretable benchmark for studying how structured, hierarchical reasoning emerges in language models.

### Strengths
The dynamics of skill acquisition rather than just end-state performance is interesting.

The study is well-designed varying variables: architectural depth and data distribution.

The evaluation is good, moving beyond simple win rates or Elo.

### Weaknesses
The current evaluation protocol appears to test the models as the White player. It would be beneficial to clarify if any experiments were conducted with the model playing as Black.

There seems to be a slight inconsistency in the evaluation methodology that I would appreciate clarification on. Rule comprehension is measured based on unconstrained generation, whereas the strategic evaluation uses prefix-constrained decoding to enforce legality. Could the authors explain the rationale for this dual approach? I wonder if this might decouple the model's strategic choices from its internal rule knowledge, potentially affecting the interpretation of the strategic metrics for shallower models that have not yet mastered legality.

The paper mentions that the training data was filtered to include games between 80 and 200 plies. Could the authors elaborate on the justification for this specific range?

The custom disambiguation-aware tokenization scheme is an interesting feature of the methodology. Could the authors explain why this hand-engineered approach was chosen over standard, data-driven subword tokenization methods like BPE?

### Questions
Please refer to the weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
Using chess as the research domain, the study examines how models acquire various chess skills from scratch. Lower-level skills, such as making legal moves, are learned early in training, whereas higher-level strategies, such as sacrificing pieces, are only acquired in the later stages.

### Strengths
Provides a detailed characterization of skill acquisition during the model’s training process.

### Weaknesses
1. **I am not an expert in explainable AI!**
2. I find the **article’s conclusion quite obvious: higher-level skills are learned later in training**. This is predictable and does not provide the reader with additional insights. I suggest the authors focus on discussing how the existing findings in the paper can inform better strategies for training models.

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies how language models acquire chess skills when trained on algebraic chess notation. By introducing a disambiguation-aware tokenization scheme and train models of varying depths (5-25 layers) on different datasets to study the emergence of capabilities. They observe clear developmental patterns: shallow models struggle with move legality, while deeper models develop tactical and positional understanding. Models trained on balanced game outcomes consistently outperform those trained only on white-win games.

### Strengths
- The paper is well-organized and clearly written. 
- The intuition of this paper is great.

### Weaknesses
- The largest model studied (25 layers, ~100M parameters) is relatively small by current standards. It's unclear if the observed patterns would hold at scales of billions of parameters. 
- The paper doesn't compare performance against purpose-built chess engines. This makes it difficult to assess overall performance compared to other methods.
- The paper lacks information about the computing resources needed for training.
- The paper lacks cast studies.

### Questions
Please refer to the "Weaknesses" section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies how language models acquire chess-playing abilities when trained on algebraic chess notation. The authors introduce a custom disambiguation-aware tokenization scheme and train models of varying depths on datasets. The paper reveals an approach similar to curriculum learning, with rule comprehension emerging early and higher-order abilities following later.

### Strengths
- The motivation of the paper is sound. 
- The paper is well-structured with clear method descriptions and results presentation.

### Weaknesses
- The paper is titled with "Large Language Models." However, the maximum size of the models trained in the paper is 100M parameters, which is relatively small.
- As mentioned in Section 5.3, evaluations used only 10 games per configuration, which may limit the robustness of the proposed method, especially for cases like sacrifices or complex tactics.
- There's no analysis of how the custom tokenization scheme impacts learning compared to other alternatives.

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
2
