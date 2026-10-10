# Data Flow Diagram (Level 1)

```mermaid
flowchart LR
    User([User])
    Meeting([Meeting App])
    
    subgraph OpenMeet System
        Input[Audio & Screen Capture]
        NLP[Speech & Text Processing]
        AI[Context Engine & LLM]
        UI[Overlay Renderer]
    end
    
    LocalData[(Local Notes / Context)]
    
    Meeting -- Audio Stream --> Input
    Meeting -- Visuals --> Input
    User -- Audio Stream --> Input
    User -- Commands/Clicks --> UI
    
    Input -- Raw Audio --> NLP
    Input -- Screenshots --> NLP
    
    NLP -- Transcripts --> AI
    NLP -- OCR Text --> AI
    
    LocalData -- Grounding Context --> AI
    AI -- Generated Suggestions --> UI
    
    UI -- HUD Display --> User
```
