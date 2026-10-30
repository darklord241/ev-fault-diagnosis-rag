## Paper
**An Evaluation Framework for Fault Diagnosis Using Technical Manuals
in Retrieval-Augmented Large Language Models**

Lukens et al., PHM Society, 2025

## What the paper does

- Uses a technical vehicle manual as the knowledge source.
- Contains 224 documented malfunctions across 22 vehicle systems.
- Creates 300 synthetic operator observations associated with known fault codes.
- Makes the dataset/code available through GitHub, providing a reproducible basis
  for experimentation.
- Compares multiple approaches:
  - LLM-only
  - different RAG configurations
  - retrieval-only
- Evaluates the approaches using fault recommendation metrics such as Hit Rate,
  Precision, Recall, F1, hallucination rate and latency.

## Important findings / takeaways

### 1. Reproducible dataset and process
The paper provides a concrete dataset and implementation through GitHub.
This is useful because the work is experimentally reproducible rather than
being only a conceptual RAG proposal.

### 2. Multiple approaches can be compared
The paper compares LLM-only, several RAG configurations and retrieval-only
approaches.

This gives a useful baseline from which our system could be extended rather
than starting with a single assumed architecture.

### 3. Retrieval vs LLM/RAG
The results show that an LLM is not automatically better than retrieval.
For structured fault recommendation, retrieval-only performed particularly
well.

This suggests that our system should not assume that every query requires
LLM reasoning.

A possible architecture to investigate is:

User query
    ↓
Retrieval
    ↓
Relevant technical evidence
    ├── Direct retrieval result
    └── LLM-based synthesis when required
              ↓
        Natural-language answer

The LLM could therefore be used primarily to synthesize/explain retrieved
evidence rather than inventing the underlying technical information.

This needs to be experimentally evaluated rather than assumed.

### 4. Evaluation-first methodology
A major takeaway is the evaluation-first approach.

Instead of building a large system first and evaluating it only at the end:

1. Build a basic prototype.
2. Define evaluation criteria.
3. Evaluate the prototype.
4. Identify failure cases.
5. Modify the retrieval/reasoning approach.
6. Evaluate again.

This iterative process can be adopted for our project.

### 5. Possible extensions for our project

The paper focuses on a ground vehicle and its technical manual. Possible
extensions for our project include:

- **Domain extension:** apply the approach to electric two-wheelers.
- **Multi-source knowledge:** combine different types of technical sources
  instead of relying on a single technical manual.
- **Source awareness / provenance:** identify the source type associated with
  retrieved information and preserve the source when generating the answer.
- **Evidence-aware answers:** distinguish between information from official
  documentation, technical literature and community-reported information.

These are proposed extensions for our project, not contributions established
by the paper itself.

## Relevance to our project

The paper is highly relevant because its central workflow is similar to our
proposed problem:

Technical documentation
        ↓
Retrieval / RAG
        ↓
Fault recommendation
        ↓
Evaluation

Our proposed extension is:

EV two-wheeler information
        ↓
Multi-source retrieval
        ↓
Evidence-aware fault troubleshooting
        ↓
LLM synthesis where useful
        ↓
Sources / provenance
        ↓
Evaluation

## Limitations relevant to us

- The study is based on a single vehicle/manual.
- The operator observations are synthetically generated.
- The system primarily focuses on fault recommendation rather than a complete
  real-world diagnostic interaction.
- The work does not address multiple source types such as manufacturer
  documentation, technical literature and community reports.

## Potential role in our research

**Base-paper candidate: HIGH**

The paper provides a strong methodological foundation for:
- technical-document retrieval
- RAG-based fault assistance
- comparison of retrieval and LLM approaches
- evaluation methodology

Our potential contribution would need to go beyond simply changing the vehicle
domain.
