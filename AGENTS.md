# Maya1 TTS FastAPI - Agent Guidelines

## Build/Test Commands
- **Start server**: `./server.sh start` (runs on port 8880)
- **Stop server**: `./server.sh stop`
- **Server status**: `./server.sh status`
- **Run tests**: `pytest` (no test files currently exist)
- **Install dependencies**: `pip install -r requirements.txt`
- **Direct run**: `python -m uvicorn maya1.api_v2:app --host 0.0.0.0 --port 8000`

## Code Style Guidelines

### Imports & Structure
- Use absolute imports: `from maya1.module import Class`
- Standard library imports first, then third-party, then local imports
- Group imports by type with blank lines between groups

### Type Hints
- Use type hints for all function signatures and class attributes
- Import from `typing` for Optional, Literal, Union types
- Use Pydantic models for API request/response validation

### Naming Conventions
- Classes: `PascalCase` (e.g., `Maya1Model`, `CreateSpeechRequest`)
- Functions/variables: `snake_case` (e.g., `generate_speech`, `model_path`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TEMPERATURE`, `AUDIO_SAMPLE_RATE`)
- Private methods: prefix with underscore (`_generate_complete_audio`)

### Error Handling
- Use custom exception classes from `maya1.openai_api`
- Raise `InvalidRequestError` for client errors (400)
- Raise `APIError` for server errors (500)
- Always include descriptive error messages

### API Design
- Follow OpenAI API specification exactly
- Use Pydantic for request/response models
- Return StreamingResponse for audio files
- Include proper HTTP status codes and error handling

### Constants
- Store all magic numbers in `maya1.constants`
- Use descriptive names with prefixes (DEFAULT_, AUDIO_, SNAC_)
- Add inline comments explaining token IDs and model parameters