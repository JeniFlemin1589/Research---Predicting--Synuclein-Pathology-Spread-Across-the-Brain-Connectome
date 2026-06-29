# RIP-GNN Methodology Diagram

This diagram maps precisely to the 6 primary steps of the RIP-GNN experimental pipeline. You can use this Mermaid.js chart directly in Markdown, Notion, or export it to an SVG/PNG for your IEEE MERCon paper.

```mermaid
flowchart TD
    %% Styling
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000
    classDef process fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef branch fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    classDef fusion fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef output fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000

    %% 1. Input Module
    subgraph Step1 [1. INPUT MODULE]
        direction LR
        IN1["α-Synuclein Pathology Data<br>(Henderson et al.)"]:::input
        IN2["Structural Connectome<br>(116 ABA Regions)"]:::input
    end

    %% 2. Preprocessing
    subgraph Step2 [2. DATA PREPROCESSING]
        direction TB
        DP1["Kinetic Traces<br>(T=1, 3, 6 months)"]:::process
        DP2["Temporal Aggregation<br>(pathology_z, delta_burden)"]:::process
    end

    %% 3. Graph Construction
    subgraph Step3 [3. GRAPH CONSTRUCTION]
        direction TB
        GC1["Node Features<br>(Prior vulnerability)"]:::process
        GC2["Weighted Edges<br>(Ipsi, Contra, Cross-Hemi)"]:::process
    end

    %% Connect Inputs to Preprocessing/Graph
    IN1 --> DP1
    DP1 --> DP2
    IN2 --> GC1
    IN2 --> GC2

    %% 4. Architecture
    subgraph Step4 [4. DUAL-BRANCH ARCHITECTURE]
        direction LR
        
        subgraph TempBranch [Temporal Branch]
            direction TB
            TB1["GRU Model<br>(Sequence Modeling)"]:::branch
            TB2["Temporal Embedding<br>(Kinetic features)"]:::branch
            TB1 --> TB2
        end
        
        subgraph SpatBranch [Spatial Branch]
            direction TB
            SB1["GATv2Conv Layers<br>(Graph Attention Messaging)"]:::branch
            SB2["Spatial Embedding<br>(Topology features)"]:::branch
            SB1 --> SB2
        end
    end

    %% Connect Preprocessing/Graph to Architecture
    DP2 --> TempBranch
    GC1 & GC2 --> SpatBranch

    %% 5. Fusion
    subgraph Step5 [5. FEATURE FUSION]
        direction TB
        FU1["Feature Concatenation<br>(Spatial + Temporal)"]:::fusion
        FU2["2-Layer MLP<br>(ReLU, Dropout)"]:::fusion
        FU1 --> FU2
    end

    %% Connect Branches to Fusion
    TB2 --> FU1
    SB2 --> FU1

    %% 6. Prediction
    subgraph Step6 [6. PREDICTION & VALIDATION]
        direction TB
        PR1["Sigmoid Activation"]:::output
        PR2["Regional Infiltration Probability (RIP)<br>(Per-region heatmap)"]:::output
        PR1 --> PR2
    end

    %% Connect Fusion to Prediction
    FU2 --> PR1

    %% Validation Feedback loop
    PR2 -.-> |"BCEWithLogits Loss<br>+ AdamW"| Step4
```
