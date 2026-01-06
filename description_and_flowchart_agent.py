#!/usr/bin/env python3
"""
LLM-Powered Description and Flowchart Generation Agent

This module consumes sfm_output.json and generates:
1. Final human-readable descriptions using LLM
2. Mermaid flowcharts derived from final descriptions

⚠️ This module does NOT parse C++ code or AST.
It operates exclusively on sfm_output.json.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

# Optional LLM imports
try:
    from langchain.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("[WARN] LLM dependencies not found. Install: pip install langchain langchain-ollama")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class LLMConfig:
    """Configurable LLM provider and settings."""
    
    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen3",
        temperature: float = 0.2,
        max_tokens: int = 2000,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize LLM configuration.
        
        Args:
            provider: LLM provider ("ollama" or "openai")
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            base_url: Base URL for API (for Ollama)
            api_key: API key (for OpenAI)
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.api_key = api_key
    
    def validate(self) -> bool:
        """Validate configuration."""
        if self.provider == "ollama":
            if not LLM_AVAILABLE:
                logger.error("Ollama provider requires langchain-ollama")
                return False
            if not self.base_url:
                # Try to get from environment or use default
                try:
                    from agent5.config import SETTINGS
                    self.base_url = SETTINGS.ollama_base_url
                except ImportError:
                    self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return True
        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                logger.error("OpenAI provider requires openai package")
                return False
            if not self.api_key:
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    logger.error("OpenAI requires OPENAI_API_KEY environment variable")
                    return False
            return True
        else:
            logger.error(f"Unknown provider: {self.provider}")
            return False


class LLMGenerator:
    """LLM-based description and flowchart generator."""
    
    def __init__(self, config: LLMConfig):
        """Initialize LLM generator with configuration."""
        self.config = config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client based on configuration."""
        if self.config.provider == "ollama":
            if not LLM_AVAILABLE:
                raise RuntimeError("Ollama not available")
            self.client = ChatOllama(
                model=self.config.model,
                temperature=self.config.temperature,
                base_url=self.config.base_url,
            )
        elif self.config.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI not available")
            self.client = OpenAI(
                api_key=self.config.api_key,
            )
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def generate_description(
        self,
        function_name: str,
        qualified_name: str,
        file_name: str,
        callee_descriptions: list[dict[str, str]],
    ) -> str:
        """
        Generate final human-readable description using LLM.
        
        Args:
            function_name: Simple function name
            qualified_name: Fully qualified name (namespace::class::function)
            file_name: Source file path
            callee_descriptions: List of {name, description} for all callees
        
        Returns:
            Generated description string
        """
        # Build callee context
        callee_context = ""
        if callee_descriptions:
            callee_parts = []
            for callee in callee_descriptions:
                callee_parts.append(f"- {callee['name']}: {callee['description']}")
            callee_context = "\n".join(callee_parts)
        else:
            callee_context = "None (leaf function)"
        
        prompt = f"""You are a C++ code documentation expert. Generate a clear, concise, human-readable description of what this function does and how it executes.

Function: {function_name}
Qualified Name: {qualified_name}
File: {file_name}

Callees (functions called by this function):
{callee_context}

IMPORTANT RULES:
1. Do NOT repeat callee descriptions verbatim
2. Summarize callee behavior in the context of this parent function
3. Focus on execution intent and high-level logic flow, not syntax details
4. Avoid mentioning implementation details like variable names unless essential
5. Describe the control flow, conditions, and decision points
6. Explain the purpose and outcome of the function
7. Be concise but comprehensive (2-4 sentences)

Generate the description:"""

        try:
            if self.config.provider == "ollama":
                messages = [HumanMessage(content=prompt)]
                response = self.client.invoke(messages)
                description = getattr(response, "content", str(response))
            elif self.config.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are a C++ code documentation expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                description = response.choices[0].message.content
            
            return description.strip()
        except Exception as e:
            logger.error(f"Failed to generate description for {qualified_name}: {e}")
            return f"Function {function_name} in {file_name}. Callees: {len(callee_descriptions)}"
    
    def generate_flowchart(self, description: str, qualified_name: str) -> str:
        """
        Generate Mermaid flowchart from description.
        
        Args:
            description: Final function description
            qualified_name: Fully qualified function name
        
        Returns:
            Valid Mermaid flowchart string
        """
        prompt = f"""Generate a valid Mermaid flowchart that visualizes the execution flow described below.

Function: {qualified_name}

Description:
{description}

CRITICAL REQUIREMENTS:
1. Start with "flowchart TD"
2. Use simple node IDs (A, B, C, D, etc.) - NO spaces, special chars, or multi-word IDs
3. Every node must be connected - no orphaned nodes
4. All conditional branches must have proper Yes/No labels
5. ALL paths must eventually connect to an End node
6. Use proper Mermaid syntax:
   - Start: Start([Start])
   - End: End([End])
   - Action: NodeID[Action Description]
   - Decision: NodeID{{Decision?}}
   - Arrows with labels: NodeID -->|Label| NextNodeID
7. The flowchart MUST reflect the logical execution steps described
8. Do NOT create function-call graphs
9. Do NOT include AST or code-level details
10. Focus on high-level execution flow

Example valid syntax:
flowchart TD
    A([Start]) --> B[Initial Action]
    B --> C{{Condition?}}
    C -->|Yes| D[Action if True]
    C -->|No| E[Action if False]
    D --> F[Final Action]
    E --> F
    F --> G([End])

Generate ONLY valid Mermaid flowchart code. No markdown, no explanations, no code blocks:"""

        try:
            if self.config.provider == "ollama":
                messages = [HumanMessage(content=prompt)]
                response = self.client.invoke(messages)
                flowchart = getattr(response, "content", str(response))
            elif self.config.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at generating Mermaid flowcharts."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                flowchart = response.choices[0].message.content
            
            return self._clean_flowchart(flowchart)
        except Exception as e:
            logger.error(f"Failed to generate flowchart for {qualified_name}: {e}")
            return self._generate_fallback_flowchart()
    
    def _clean_flowchart(self, flowchart: str) -> str:
        """Clean and validate Mermaid flowchart syntax."""
        # Remove markdown code blocks
        flowchart = flowchart.strip()
        if flowchart.startswith("```"):
            lines = flowchart.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            flowchart = "\n".join(lines)
        
        flowchart = flowchart.strip()
        
        # Ensure it starts with flowchart TD
        if not flowchart.startswith("flowchart") and not flowchart.startswith("graph"):
            flowchart = "flowchart TD\n" + flowchart
        elif flowchart.startswith("graph"):
            flowchart = flowchart.replace("graph TD", "flowchart TD", 1)
            flowchart = flowchart.replace("graph LR", "flowchart LR", 1)
        
        # Basic validation: ensure Start and End nodes exist
        if "Start" not in flowchart and "([Start])" not in flowchart:
            # Add Start node if missing
            lines = flowchart.split("\n")
            if len(lines) > 1:
                first_node = lines[1].strip().split()[0] if len(lines) > 1 else "A"
                lines.insert(1, f"    Start([Start]) --> {first_node}")
                flowchart = "\n".join(lines)
        
        if "End" not in flowchart and "([End])" not in flowchart:
            # Add End node if missing
            lines = flowchart.split("\n")
            lines.append("    End([End])")
            flowchart = "\n".join(lines)
        
        return flowchart
    
    def _generate_fallback_flowchart(self) -> str:
        """Generate a simple fallback flowchart."""
        return """flowchart TD
    Start([Start]) --> Process[Process Function]
    Process --> End([End])"""


class DescriptionAndFlowchartAgent:
    """
    Main agent for generating descriptions and flowcharts from SFM JSON.
    
    Processes functions bottom-up, using callee descriptions to inform
    parent function descriptions.
    """
    
    def __init__(
        self,
        sfm_json_path: str | Path,
        output_dir: str | Path = "flowcharts",
        llm_config: Optional[LLMConfig] = None,
    ):
        """
        Initialize the agent.
        
        Args:
            sfm_json_path: Path to sfm_output.json
            output_dir: Directory for flowchart output
            llm_config: LLM configuration (default: Ollama with qwen3)
        """
        self.sfm_json_path = Path(sfm_json_path)
        self.output_dir = Path(output_dir)
        self.llm_config = llm_config or LLMConfig()
        
        if not self.llm_config.validate():
            raise ValueError("Invalid LLM configuration")
        
        self.llm = LLMGenerator(self.llm_config)
        
        # Data structures
        self.functions: dict[str, dict[str, Any]] = {}  # qualified_name -> SFM dict
        self.dependency_graph: dict[str, set[str]] = defaultdict(set)  # function -> callees
        self.reverse_dependency: dict[str, set[str]] = defaultdict(set)  # function -> callers
        self.processed: set[str] = set()  # Already processed functions
        
        # Load SFM JSON
        self._load_sfm_json()
        
        # Build dependency graph
        self._build_dependency_graph()
    
    def _load_sfm_json(self) -> None:
        """Load SFM JSON file."""
        if not self.sfm_json_path.exists():
            raise FileNotFoundError(f"SFM JSON not found: {self.sfm_json_path}")
        
        logger.info(f"Loading SFM JSON from: {self.sfm_json_path}")
        with open(self.sfm_json_path, encoding="utf-8") as f:
            sfm_list = json.load(f)
        
        # Index by qualified_name
        for sfm in sfm_list:
            qualified_name = sfm.get("qualified_name", "")
            if qualified_name:
                self.functions[qualified_name] = sfm
        
        logger.info(f"Loaded {len(self.functions)} functions")
    
    def _build_dependency_graph(self) -> None:
        """Build dependency graph from callees."""
        for qualified_name, sfm in self.functions.items():
            callees = sfm.get("callees", [])
            # Callees might be list of strings (qualified names) or list of dicts
            callee_names = []
            for callee in callees:
                if isinstance(callee, str):
                    callee_names.append(callee)
                elif isinstance(callee, dict):
                    # Could be {"uid": "..."} or {"name": "..."}
                    name = callee.get("qualified_name") or callee.get("name") or callee.get("uid")
                    if name:
                        callee_names.append(name)
            
            self.dependency_graph[qualified_name] = set(callee_names)
            
            # Build reverse dependency (callers)
            for callee in callee_names:
                self.reverse_dependency[callee].add(qualified_name)
        
        logger.info(f"Built dependency graph: {len(self.dependency_graph)} functions with dependencies")
    
    def _get_processing_order(self) -> list[str]:
        """
        Determine bottom-up processing order using topological sort.
        Leaf nodes (no callees) are processed first.
        """
        # Count in-degrees (number of unprocessed dependencies)
        in_degree: dict[str, int] = {}
        for func in self.functions:
            # In-degree = number of callees that are also in our function set
            callees = self.dependency_graph.get(func, set())
            # Count only callees that exist in our function set
            in_degree[func] = sum(1 for c in callees if c in self.functions)
        
        # Topological sort
        queue = deque()
        result = []
        
        # Start with leaf nodes (in_degree = 0)
        for func, degree in in_degree.items():
            if degree == 0:
                queue.append(func)
        
        while queue:
            func = queue.popleft()
            result.append(func)
            
            # Update callers (functions that depend on this one)
            for caller in self.reverse_dependency.get(func, set()):
                if caller in in_degree:
                    in_degree[caller] -= 1
                    if in_degree[caller] == 0:
                        queue.append(caller)
        
        # Handle any remaining functions (cycles or isolated)
        remaining = set(self.functions.keys()) - set(result)
        if remaining:
            logger.warning(f"{len(remaining)} functions not in dependency order (cycles?): {remaining}")
            result.extend(remaining)
        
        logger.info(f"Processing order determined: {len(result)} functions")
        return result
    
    def _get_callee_descriptions(self, qualified_name: str) -> list[dict[str, str]]:
        """Get descriptions of all callees (must be already processed)."""
        callees = self.dependency_graph.get(qualified_name, set())
        callee_descriptions = []
        
        for callee in callees:
            if callee in self.functions:
                sfm = self.functions[callee]
                description = sfm.get("description", "")
                if description:
                    callee_descriptions.append({
                        "name": callee,
                        "description": description,
                    })
                else:
                    logger.warning(f"Callee {callee} has no description (not processed yet?)")
        
        return callee_descriptions
    
    def process_function(self, qualified_name: str) -> None:
        """
        Process a single function:
        1. Generate description using LLM (with callee context)
        2. Generate flowchart from description
        3. Update in-memory SFM
        """
        if qualified_name not in self.functions:
            logger.warning(f"Function {qualified_name} not found in functions")
            return
        
        if qualified_name in self.processed:
            logger.debug(f"Skipping already processed: {qualified_name}")
            return
        
        sfm = self.functions[qualified_name]
        
        # Check dependencies
        callees = self.dependency_graph.get(qualified_name, set())
        unprocessed_callees = [c for c in callees if c in self.functions and c not in self.processed]
        
        if unprocessed_callees:
            logger.warning(
                f"Skipping {qualified_name} - unprocessed callees: {unprocessed_callees}"
            )
            return
        
        logger.info(f"Processing: {qualified_name}")
        
        # Get callee descriptions
        callee_descriptions = self._get_callee_descriptions(qualified_name)
        
        # Generate description
        logger.info(f"  Generating description (with {len(callee_descriptions)} callees)...")
        description = self.llm.generate_description(
            function_name=sfm.get("function_name", qualified_name),
            qualified_name=qualified_name,
            file_name=sfm.get("file_name", ""),
            callee_descriptions=callee_descriptions,
        )
        
        # Store description
        sfm["description"] = description
        logger.info(f"  Description generated ({len(description)} chars)")
        
        # Generate flowchart from description
        logger.info(f"  Generating flowchart...")
        flowchart = self.llm.generate_flowchart(description, qualified_name)
        
        # Store flowchart
        sfm["flowchart"] = flowchart
        logger.info(f"  Flowchart generated ({len(flowchart)} chars)")
        
        # Mark as processed
        self.processed.add(qualified_name)
        logger.info(f"  ✓ Completed: {qualified_name}")
    
    def process_all(self) -> None:
        """Process all functions in bottom-up order."""
        processing_order = self._get_processing_order()
        
        logger.info(f"Processing {len(processing_order)} functions...")
        
        for i, qualified_name in enumerate(processing_order, 1):
            logger.info(f"[{i}/{len(processing_order)}] Processing: {qualified_name}")
            try:
                self.process_function(qualified_name)
            except Exception as e:
                logger.error(f"Failed to process {qualified_name}: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"Completed processing {len(self.processed)} functions")
    
    def export_flowcharts(self) -> Path:
        """
        Export all flowcharts to output directory.
        Creates directory if it doesn't exist.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for qualified_name, sfm in self.functions.items():
            flowchart = sfm.get("flowchart", "")
            if flowchart:
                # Create safe filename
                safe_name = self._safe_filename(qualified_name)
                flowchart_path = self.output_dir / f"{safe_name}.mmd"
                
                with open(flowchart_path, "w", encoding="utf-8") as f:
                    f.write(f"# {qualified_name}\n")
                    f.write(f"# File: {sfm.get('file_name', '')}\n")
                    f.write(f"# Description: {sfm.get('description', '')[:100]}...\n\n")
                    f.write(flowchart)
                
                count += 1
        
        logger.info(f"Exported {count} flowcharts to: {self.output_dir}")
        return self.output_dir
    
    def _safe_filename(self, name: str) -> str:
        """Convert function name to safe filename."""
        import re
        # Replace namespace separators and special chars
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe = safe.replace("::", "_")
        safe = safe.replace(" ", "_")
        # Limit length
        if len(safe) > 100:
            safe = safe[:100]
        return safe
    
    def export_final_sfm(self, output_path: str | Path) -> Path:
        """
        Export final SFM JSON with descriptions and flowcharts.
        
        Args:
            output_path: Path to output JSON file (default: sfm_final_output.json)
        
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sfm_list = list(self.functions.values())
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sfm_list, f, indent=2)
        
        logger.info(f"Exported final SFM to: {output_path}")
        return output_path


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LLM-Powered Description and Flowchart Generation Agent"
    )
    parser.add_argument(
        "sfm_json",
        type=str,
        help="Path to sfm_output.json"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="flowcharts",
        help="Output directory for flowcharts (default: flowcharts)"
    )
    parser.add_argument(
        "--final-output",
        type=str,
        default="sfm_final_output.json",
        help="Output path for final SFM JSON (default: sfm_final_output.json)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM provider (default: ollama)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (default: qwen3 for ollama, gpt-4 for openai)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature (default: 0.2)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="LLM API base URL (for Ollama)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (for OpenAI, or set OPENAI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Determine model
    if not args.model:
        args.model = "qwen3" if args.provider == "ollama" else "gpt-4"
    
    # Create LLM config
    llm_config = LLMConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    
    # Create agent
    try:
        agent = DescriptionAndFlowchartAgent(
            sfm_json_path=args.sfm_json,
            output_dir=args.output_dir,
            llm_config=llm_config,
        )
        
        # Process all functions
        agent.process_all()
        
        # Export results
        agent.export_flowcharts()
        agent.export_final_sfm(args.final_output)
        
        logger.info("✓ All done!")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

