flowchart LR
  %% ---------------- Sections ----------------
  subgraph RQ["Request"]
    A([Start]):::terminal --> B[Normalize goal and init budgets]:::state
  end

  subgraph ST["State Store"]
    SS[(AgentState)]:::datastore
  end

  subgraph LOOP["Tool Agent Loop"]
    B --> C[Save initial state]:::state --> SS
    SS --> D{Continue loop}:::decision

    D -- Yes --> E[[LLM Policy\nDecide next Action]]:::llm
    E --> F[Execute Action]:::exec
    F --> G{Action type}:::decision
  end

  subgraph ACT["Action Handlers"]
    G -- CALL_TOOL --> H[ToolRuntime call]:::tool
    H --> I[Append Observation\nTOOL_RESULT]:::state --> SS
    I --> J[Cursor plus]:::state --> SS --> D

    G -- WRITE_NOTE --> K[Append Observation\nNOTE]:::state --> SS
    K --> L[Cursor plus]:::state --> SS --> D

    G -- SYNTHESIZE --> M[[LLM Synthesizer\nGenerate finalAnswer]]:::llm
    M --> N[Set DONE and save answer]:::state --> SS --> Z([End]):::terminal

    G -- STOP --> O[Stop message]:::exec
    O --> P[Set DONE and save answer]:::state --> SS --> Z
  end

  subgraph EXIT["Exit Path"]
    D -- No --> X[[LLM Synthesizer\nGenerate finalAnswer]]:::llm
    X --> Y[Set DONE and save answer]:::state --> SS --> Z
  end

  %% ---------------- Styling ----------------
  classDef llm fill:#e8f0ff,stroke:#3b82f6,stroke-width:2px,color:#0b1b3a;
  classDef tool fill:#f1f5f9,stroke:#94a3b8,stroke-width:1.5px,color:#0f172a;
  classDef state fill:#ecfdf5,stroke:#22c55e,stroke-width:1.5px,color:#064e3b;
  classDef decision fill:#fff7ed,stroke:#f97316,stroke-width:1.5px,color:#7c2d12;
  classDef terminal fill:#111827,stroke:#111827,color:#ffffff,stroke-width:2px;
  classDef exec fill:#faf5ff,stroke:#a855f7,stroke-width:1.5px,color:#3b0764;
  classDef datastore fill:#ffffff,stroke:#0ea5e9,stroke-width:2px,color:#0c4a6e;

  %% Make main action flow slightly thicker
  linkStyle 6 stroke:#3b82f6,stroke-width:2.5px;
  linkStyle 7 stroke:#3b82f6,stroke-width:2.5px;
  linkStyle 8 stroke:#3b82f6,stroke-width:2.5px;
  linkStyle 15 stroke:#3b82f6,stroke-width:2.5px;
  linkStyle 16 stroke:#3b82f6,stroke-width:2.5px;
