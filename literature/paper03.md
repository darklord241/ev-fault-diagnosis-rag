# Paper 03

## Paper

**Large Language Model Based Intelligent Fault Information Retrieval System
for New Energy Vehicles**

Zhang et al., 2025
Applied Sciences, 15(7), 4034

## What the paper does

- Proposes an intelligent fault-information retrieval system for New Energy
  Vehicles (NEVs).
- Combines LLMs, Knowledge Graphs (KGs) and Retrieval-Augmented Generation.
- Constructs a knowledge graph containing 122 NEV fault categories.
- Represents relationships between vehicle models, fault categories,
  fault phenomena, causes and solutions.
- Uses ChatGLM3-6B for knowledge extraction/reasoning and LoRA fine-tuning.
- Uses BERT-based fault classification and compares it with LSTM and
  traditional TF-IDF classifiers.
- Performs named entity extraction for vehicle/fault-related entities.
- Creates a terminology/paraphrase mapping to handle different expressions
  of the same NEV concepts.
- Provides fault cause analysis and solution recommendations through an
  interactive retrieval system.

## Important takeaways

### 1. NEV + LLM + RAG + Knowledge Graph already exists

This paper demonstrates that LLM/RAG-based fault-information systems have
already been applied to New Energy Vehicles.

Therefore, our research gap cannot simply be:
"apply RAG/LLMs to EV fault diagnosis."

### 2. Structured knowledge is important

The paper uses a knowledge graph and fault-tree structure to preserve
relationships between faults, symptoms, causes and solutions.

This suggests that pure vector retrieval may not always be sufficient for
complex fault reasoning.

### 3. Terminology variation is a retrieval problem

Different users/documents may describe the same vehicle, component or fault
using different terminology.

A terminology/paraphrase layer can improve recognition and retrieval.

### 4. Multiple NLP stages can be useful

The system separates:
- fault classification
- entity extraction
- knowledge retrieval
- LLM reasoning

Rather than treating the entire problem as a single LLM prompt.

### 5. Evaluation is performed at multiple levels

The paper evaluates classification and entity extraction using precision,
recall and F1, and also performs system-level functional testing.

## Relevance to our project

Very high.

The paper directly overlaps with our intended domain and technical approach:
NEV fault diagnosis + LLM + knowledge graph + RAG.

However, our proposed system may focus specifically on electric two-wheelers
and on source-aware, evidence-grounded troubleshooting.

## Potential extensions for our project

- Focus specifically on electric two-wheelers rather than broad NEVs.
- Combine multiple source types:
  - official/OEM documentation
  - technical literature
  - service information
  - clearly labelled community reports
- Preserve source provenance for every retrieved piece of evidence.
- Distinguish source reliability rather than treating all retrieved information
  equally.
- Evaluate end-to-end fault troubleshooting and answer grounding, not only
  classification/entity extraction.
- Compare retrieval-only, retrieval + LLM synthesis, and potentially
  knowledge-graph-assisted approaches.

## Limitations / observations

- Broad NEV domain rather than electric two-wheelers specifically.
- The system is strongly centered around a structured fault knowledge graph.
- Much of the reported evaluation concerns NLP components and system
  functionality rather than real-world technician diagnosis.
- Source provenance across heterogeneous evidence types is not the central
  focus.

## Role in our research

**Very strong related-work paper and possible architectural base, but not a
paper we can simply extend by changing NEVs to electric scooters.**

It forces us to define a more specific contribution around source-aware,
evidence-grounded and/or multi-source troubleshooting.
