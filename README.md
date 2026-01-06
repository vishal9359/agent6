# C++ Execution-Flow Visualization Agent

A static-analysis based agent that analyzes C++ projects and generates Mermaid flowcharts per function showing logical execution scenarios.

## Features

- **Static Analysis First**: Uses libclang for AST parsing and CFG analysis
- **Bottom-Up Callee Discovery**: Transitive function call tracking with caching
- **Scenario Flow Model (SFM)**: Structured JSON output as single source of truth
- **Mermaid Flowcharts**: Execution flow visualization (not call graphs)
- **Immutable Cache**: Functions analyzed once and reused
- **LLM-Powered Enhancement**: Optional module for generating rich descriptions and flowcharts

## Architecture

The system consists of two separate modules:

### Module 1: Static Analysis Agent (`agent.py`)

Implements a strict multi-stage pipeline:

1. **Stage 1 - Static Code Understanding**: Parse C++ AST using libclang
2. **Stage 2 - Scenario Flow Model (SFM)**: Generate structured JSON per function
3. **Stage 3 - Basic Flowchart Generation**: Create initial Mermaid diagrams

### Module 2: LLM-Powered Enhancement (`description_and_flowchart_agent.py`)

Consumes `sfm_output.json` and enhances it:

1. **Bottom-Up Processing**: Processes functions after all callees are processed
2. **Description Generation**: Uses LLM to generate human-readable descriptions with callee context
3. **Flowchart Generation**: Creates Mermaid flowcharts from final descriptions

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

### Two-Stage Workflow

#### Stage 1: Static Analysis

Generate initial SFM JSON from C++ code:

```bash
python agent.py /path/to/cpp/project
```

#### Stage 2: LLM Enhancement (Optional)

Enhance with LLM-powered descriptions and flowcharts:

```bash
python description_and_flowchart_agent.py sfm_output.json
```

### Static Analysis Agent Options

```bash
python agent.py /path/to/cpp/project \
    -o sfm_output.json \
    --flowcharts-dir flowcharts \
    --compile-args -std=c++17 -I./include \
    --no-llm
```

**Options:**
- `root_dir`: Root directory of C++ project (required)
- `-o, --output`: Output JSON file path (default: `sfm_output.json`)
- `--flowcharts-dir`: Directory for Mermaid flowchart files (default: `flowcharts`)
- `--compile-args`: Compilation arguments (default: `-std=c++17`)
- `--no-llm`: Disable LLM for description/flowchart generation (use static analysis only)

### LLM Enhancement Agent Options

```bash
python description_and_flowchart_agent.py sfm_output.json \
    -o flowcharts \
    --final-output sfm_final_output.json \
    --provider ollama \
    --model qwen3 \
    --temperature 0.2
```

**Options:**
- `sfm_json`: Path to sfm_output.json (required)
- `-o, --output-dir`: Output directory for flowcharts (default: `flowcharts`)
- `--final-output`: Output path for final SFM JSON (default: `sfm_final_output.json`)
- `--provider`: LLM provider: `ollama` or `openai` (default: `ollama`)
- `--model`: Model name (default: `qwen3` for ollama, `gpt-4` for openai)
- `--temperature`: LLM temperature (default: `0.2`)
- `--base-url`: LLM API base URL (for Ollama)
- `--api-key`: API key (for OpenAI, or set `OPENAI_API_KEY` env var)

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

Each function's flowchart is saved as a separate `.mmd` file in the flowcharts directory:

```
flowcharts/
 ├── func1.mmd
 ├── func2.mmd
 ├── ClassA_methodX.mmd
 └── ...
```

## LLM Configuration

The enhancement module supports multiple LLM providers:

### Ollama (Default)

```bash
# Set base URL if needed (default: http://localhost:11434)
export OLLAMA_BASE_URL=http://localhost:11434

python description_and_flowchart_agent.py sfm_output.json \
    --provider ollama \
    --model qwen3
```

### OpenAI

```bash
# Set API key
export OPENAI_API_KEY=your-api-key

python description_and_flowchart_agent.py sfm_output.json \
    --provider openai \
    --model gpt-4 \
    --temperature 0.2
```

## How It Works

### Static Analysis Module (`agent.py`)

#### Bottom-Up Analysis

1. Discover all callees transitively
2. Analyze callees first (depth-first)
3. Then analyze the function itself
4. Cache results to avoid recomputation

#### Callee Discovery

- Direct function calls are extracted from AST
- Transitive callees are discovered recursively
- Results are cached and reused
- Cycles are detected and prevented

#### Static Analysis

The agent uses libclang to:
- Parse C++ AST
- Extract control flow (if/else, loops, switches)
- Resolve function symbols
- Track function calls
- Build execution flow models

### LLM Enhancement Module (`description_and_flowchart_agent.py`)

#### Dependency-Aware Processing

1. **Build Dependency Graph**: Construct call graph from SFM JSON
2. **Topological Sort**: Determine bottom-up processing order
3. **Leaf Nodes First**: Process functions with no callees first
4. **Parent Functions**: Process only after all callees are complete

#### Description Generation

- Uses LLM to generate human-readable descriptions
- Incorporates callee descriptions as context
- Summarizes callee behavior in parent context
- Focuses on execution intent, not syntax

#### Flowchart Generation

- Generates Mermaid flowcharts from final descriptions
- Reflects logical execution steps
- Shows conditional paths and high-level operations
- Does NOT create function-call graphs

## Limitations

- Only analyzes project source files (skips system headers)
- Does not handle templates with full precision
- Indirect function calls (via function pointers) may not be fully resolved
- Complex macros may not be expanded

## License

MIT

