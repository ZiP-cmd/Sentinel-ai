
# Sentinel AI — System Architecture

Sentinel AI is an educational fraud detection and investigation
safety demonstration. Its model, dashboard, API, and audit
demonstrations are not a production financial decision system.

## Architecture Diagram

```mermaid
flowchart TD
    A["Historical creditcard.csv dataset"]
    B["Chronological 60/20/20 split"]
    C["StandardScaler + weighted Logistic Regression"]
    D[("Local saved model artifact")]

    A --> B
    B -->|"Training partition"| C
    C --> D

    D --> E["Prediction service"]
    E --> F["Fraud score + review decision"]

    G["Streamlit dashboard"] --> E
    H["FastAPI /predict endpoint"] --> E

    B -->|"Validation partition"| I["Model evaluation and alert budgets"]
    D --> I
    I --> G

    F --> J["Evidence validation"]
    J --> K["Structured candidate claims"]
    K --> L["Claim validator"]

    J --> L
    L --> M["Safe response builder"]
    M --> N["Evidence-grounded investigator summary"]

    L --> O["Investigation audit demonstration"]
    O --> P["In-memory audit record"]
    O --> Q["Investigator-facing view"]

    R["Demo session + current role lookup"] --> S{"Authorized audit role?"}
    P --> S
    S -->|"Auditor / Compliance"| T["Audit record access"]
    S -->|"Investigator / Unverified"| U["Access denied"]

    classDef data fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef model fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef safety fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef restricted fill:#fce7f3,stroke:#be185d,color:#831843

    class A,B data
    class C,D,E,F,I model
    class J,K,L,M,N,O,Q safety
    class P,R,S,T,U restricted
```

## Major Components

### Machine Learning
- `src/save_model.py`: trains the selected model
- `src/prediction_service.py`: loads and scores transactions
- `src/model_evaluation.py`: evaluates model predictions
- `src/alert_budget.py`: evaluates fixed investigation capacities

### Interfaces
- `src/dashboard.py`: Streamlit model and safety demonstrations
- `src/api.py`: FastAPI prediction and health endpoints

### Evidence-Grounded Investigation
- `src/investigation_evidence.py`: validates prediction evidence
- `src/claim_validator.py`: checks structured claims
- `src/safe_response_builder.py`: reconstructs safe summaries
- `src/investigation_audit.py`: creates separate in-memory
  investigator views and audit records

### Demonstration Security
- `src/audit_sessions.py`: demo sessions and current-role lookup
- `src/audit_access.py`: role-based audit record access

## Important Design Boundaries

The FastAPI service provides model predictions; it does not
expose the complete investigation and audit workflow.

The Streamlit dashboard demonstrates prediction, evaluation,
and selected investigation-safety scenarios.

The audit workflow is implemented and tested as a separate
in-memory demonstration rather than a persistent audit service.

The claim validator checks known structured fields against
trusted evidence. It is not a general-purpose natural-language
fact-checking system.

Model alerts require human interpretation and do not establish
that fraud occurred.
```
