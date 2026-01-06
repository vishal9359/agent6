# C++ Execution-Flow Visualization Agent

A static-analysis based agent that analyzes C++ projects and generates Mermaid flowcharts per function showing logical execution scenarios.

## Features

- **Static Analysis First**: Uses libclang for AST parsing and CFG analysis
- **Bottom-Up Callee Discovery**: Transitive function call tracking with caching
- **Scenario Flow Model (SFM)**: Structured JSON output as single source of truth
- **Mermaid Flowcharts**: Execution flow visualization (not call graphs)
- **Immutable Cache**: Functions analyzed once and reused

## Architecture

The agent implements a strict multi-stage pipeline:

1. **Stage 1 - Static Code Understanding**: Parse C++ AST using libclang
2. **Stage 2 - Scenario Flow Model (SFM)**: Generate structured JSON per function
3. **Stage 3 - Flowchart Generation**: Create Mermaid diagrams from SFM

## Installation

### Prerequisites

- Python 3.8+
- libclang (system library)
- LLVM (optional, for better C++ parsing)

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Configure libclang

Set the `CLANG_LIB_PATH` environment variable if needed:

```bash
# Linux
export CLANG_LIB_PATH=/usr/lib/llvm-18/lib/libclang.so

# Windows
set CLANG_LIB_PATH=C:\Program Files\LLVM\bin\libclang.dll
```

## Usage

### Basic Usage

```bash
python agent.py /path/to/cpp/project
```

### Advanced Options

```bash
python agent.py /path/to/cpp/project \
    -o output.json \
    --flowcharts-dir flowcharts \
    --compile-args -std=c++17 -I./include \
    --no-llm
```

### Options

- `root_dir`: Root directory of C++ project (required)
- `-o, --output`: Output JSON file path (default: `sfm_output.json`)
- `--flowcharts-dir`: Directory for Mermaid flowchart files (default: `flowcharts`)
- `--compile-args`: Compilation arguments (default: `-std=c++17`)
- `--no-llm`: Disable LLM for description/flowchart generation (use static analysis only)

## Output

### SFM JSON Format

Each function generates a Scenario Flow Model (SFM) JSON:

```json
{
  "function_name": "simple_name",
  "qualified_name": "namespace::class::function",
  "file_name": "relative/path/to/file.cpp",
  "module_name": "derived_module_name",
  "line_start": 10,
  "column_start": 5,
  "line_end": 120,
  "column_end": 1,
  "description": "Function description",
  "flowchart": "flowchart TD\n...",
  "callees": ["callee1", "callee2"]
}
```

### Flowchart Files

Each function's flowchart is saved as a separate `.mmd` file in the flowcharts directory.

## How It Works

### Bottom-Up Analysis

1. Discover all callees transitively
2. Analyze callees first (depth-first)
3. Then analyze the function itself
4. Cache results to avoid recomputation

### Callee Discovery

- Direct function calls are extracted from AST
- Transitive callees are discovered recursively
- Results are cached and reused
- Cycles are detected and prevented

### Static Analysis

The agent uses libclang to:
- Parse C++ AST
- Extract control flow (if/else, loops, switches)
- Resolve function symbols
- Track function calls
- Build execution flow models

## Limitations

- Only analyzes project source files (skips system headers)
- Does not handle templates with full precision
- Indirect function calls (via function pointers) may not be fully resolved
- Complex macros may not be expanded

## License

MIT

