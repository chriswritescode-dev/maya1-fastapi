# Agent Development Guidelines

## Build/Run/Test
- **Run server**: `uvicorn maya1.api_v2:app --host 0.0.0.0 --port 8000 --reload`
- **Run tests**: `pytest` (all tests) or `pytest path/to/test_file.py::test_function` (single test)
- **Type check**: `pyright` (uses pyrightconfig.json)

## Code Style
- **No comments**: Code should be self-documenting; avoid cluttering with comments
- **DRY principles**: Avoid code duplication; extract reusable logic
- **Keep it simple**: Use minimal code to achieve the goal
- **Type hints**: Use basic typing (Optional, List, Dict) but avoid complex types
- **Imports**: Group as stdlib, third-party, local; use absolute imports from `maya1.*`
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Error handling**: Use try/finally for cleanup, log errors before raising HTTPException
- **Async**: Use `async def` for I/O operations, `await` for async calls
- **Memory**: Delete large objects immediately after use, call `torch.cuda.empty_cache()` strategically

## Architecture
- **OpenAI-only**: Codebase serves `/v1/audio/speech` endpoint (OpenAI-compatible)
- **No streaming**: Removed streaming pipeline; only batch processing remains
- **Batch processing**: SNAC decoder batches requests of same length for efficiency
- **Pipeline flow**: API → Pipeline.generate_speech() → SNACDecoder.decode_single_async() → batch processing
