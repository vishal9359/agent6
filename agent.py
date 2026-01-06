#!/usr/bin/env python3
"""
C++ Execution-Flow Visualization Agent

A static-analysis based agent that analyzes C++ projects and generates
Mermaid flowcharts per function showing logical execution scenarios.

Architecture:
- Stage 1: Static Code Understanding (libclang AST parsing)
- Stage 2: Scenario Flow Model (SFM) generation
- Stage 3: Flowchart generation from SFM

Key Features:
- Bottom-up callee discovery with transitive tracking
- Immutable, reusable function analysis cache
- SFM JSON as single source of truth
- Depth-first deterministic traversal
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

try:
    from clang import cindex
except ImportError:
    raise ImportError(
        "clang package not found. Install with: pip install clang"
    )

# Optional LLM for description/flowchart generation (can be disabled)
try:
    from langchain.messages import HumanMessage
    from langchain_ollama import ChatOllama
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("[WARN] LLM dependencies not found. Description/flowchart generation will use fallback methods.")

# Configure libclang library path
CLANG_LIB_PATH = os.getenv("CLANG_LIB_PATH")
if CLANG_LIB_PATH:
    try:
        cindex.Config.set_library_file(CLANG_LIB_PATH)
    except Exception as e:
        print(f"[WARN] Failed to set libclang path {CLANG_LIB_PATH}: {e}")

# Try common library paths
if not CLANG_LIB_PATH:
    for alt_path in [
        "/usr/lib/llvm-18/lib/libclang.so",
        "/usr/lib/llvm-19/lib/libclang.so",
        "/usr/lib/llvm-17/lib/libclang.so",
        "/usr/lib/x86_64-linux-gnu/libclang.so.1",
        "C:/Program Files/LLVM/bin/libclang.dll",
    ]:
        try:
            cindex.Config.set_library_file(alt_path)
            print(f"[INFO] Using libclang at: {alt_path}")
            break
        except Exception:
            continue

SUPPORTED_EXT = (".c", ".cpp", ".cc", ".cxx")


class FunctionSFM:
    """Scenario Flow Model for a single function."""
    
    def __init__(
        self,
        function_name: str,
        qualified_name: str,
        file_name: str,
        module_name: str,
        line_start: int,
        column_start: int,
        line_end: int,
        column_end: int,
    ):
        self.function_name = function_name
        self.qualified_name = qualified_name
        self.file_name = file_name
        self.module_name = module_name
        self.line_start = line_start
        self.column_start = column_start
        self.line_end = line_end
        self.column_end = column_end
        self.description = ""
        self.flowchart = ""
        self.callees: list[str] = []  # List of qualified names
        self._is_finalized = False
    
    def finalize(self):
        """Mark SFM as finalized (immutable)."""
        self._is_finalized = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to SFM JSON format."""
        return {
            "function_name": self.function_name,
            "qualified_name": self.qualified_name,
            "file_name": self.file_name,
            "module_name": self.module_name,
            "line_start": self.line_start,
            "column_start": self.column_start,
            "line_end": self.line_end,
            "column_end": self.column_end,
            "description": self.description,
            "flowchart": self.flowchart,
            "callees": self.callees,
        }
    
    def is_finalized(self) -> bool:
        """Check if SFM is finalized."""
        return self._is_finalized


class ExecutionFlowAgent:
    """
    Main agent for C++ execution-flow visualization.
    
    Implements:
    - Static AST parsing with libclang
    - Bottom-up callee discovery
    - SFM generation and caching
    - Flowchart generation
    """
    
    def __init__(
        self,
        root_dir: str | Path,
        compile_args: list[str] | None = None,
        use_llm: bool = True,
    ):
        self.root_dir = Path(root_dir).resolve()
        self.compile_args = compile_args or ["-std=c++17"]
        self.use_llm = use_llm and LLM_AVAILABLE
        
        # Core data structures
        self.function_index: dict[str, FunctionSFM] = {}  # qualified_name -> SFM
        self.call_graph: dict[str, set[str]] = defaultdict(set)  # caller -> set of callees (qualified names)
        self.visited_functions: set[str] = set()  # Track visited during traversal
        self.call_stack: list[str] = []  # Prevent cycles
        
        # AST parsing structures
        self.index = cindex.Index.create()
        self.uid_to_qualified: dict[str, str] = {}  # UID -> qualified_name
        self.qualified_to_uid: dict[str, str] = {}  # qualified_name -> UID
        
        # LLM setup (optional)
        self.llm = None
        if self.use_llm:
            try:
                # Try to load settings if available
                try:
                    from agent5.config import SETTINGS
                    base_url = SETTINGS.ollama_base_url
                except ImportError:
                    base_url = "http://localhost:11434"
                
                self.llm = ChatOllama(model="qwen3", temperature=0.1, base_url=base_url)
            except Exception as e:
                print(f"[WARN] Failed to initialize LLM: {e}")
                self.use_llm = False
    
    def is_project_file(self, file_path: str) -> bool:
        """Check if file is a project source file (not system/external)."""
        path = Path(file_path)
        try:
            # Resolve symlinks and get absolute path
            resolved = path.resolve()
            # Check if file is within project root
            try:
                resolved.relative_to(self.root_dir)
                return True
            except ValueError:
                return False
        except Exception:
            return False
    
    def is_cpp_file(self, file_path: str) -> bool:
        """Check if file is a C++ source file."""
        return file_path.endswith(SUPPORTED_EXT)
    
    def get_module_name(self, file_path: str) -> str:
        """Get module name from file path relative to root directory."""
        rel = self.root_dir / Path(file_path)
        if not rel.exists():
            rel = Path(file_path)
        
        try:
            rel_path = rel.relative_to(self.root_dir)
        except ValueError:
            rel_path = Path(file_path)
        
        no_ext = rel_path.with_suffix("")
        return ".".join(str(no_ext).split(os.sep))
    
    def get_qualified_name(self, cursor: cindex.Cursor) -> str:
        """Get fully qualified function name including namespace/class."""
        name = cursor.spelling or "<anonymous>"
        
        # Build qualified name by traversing semantic parents
        parts = [name]
        parent = cursor.semantic_parent
        
        while parent:
            if parent.kind in (
                cindex.CursorKind.NAMESPACE,
                cindex.CursorKind.CLASS_DECL,
                cindex.CursorKind.STRUCT_DECL,
            ):
                parent_name = parent.spelling
                if parent_name:
                    parts.insert(0, parent_name)
            elif parent.kind in (
                cindex.CursorKind.CXX_METHOD,
                cindex.CursorKind.FUNCTION_DECL,
            ):
                # Stop at enclosing function (don't include it in name)
                break
            
            parent = parent.semantic_parent
        
        return "::".join(parts)
    
    def node_uid(self, cursor: cindex.Cursor) -> str:
        """Generate stable unique identifier for functions."""
        loc = cursor.location
        name = cursor.spelling or "<anonymous>"
        file_name = loc.file.name if loc.file else "<unknown>"
        return f"{name}:{file_name}:{loc.line}:{loc.column}"
    
    def parse_file(self, file_path: str) -> None:
        """Parse a single C++ file and extract function declarations."""
        if not self.is_cpp_file(file_path):
            return
        
        if not self.is_project_file(file_path):
            return  # Skip non-project files
        
        module_name = self.get_module_name(file_path)
        
        try:
            tu = self.index.parse(
                file_path,
                args=self.compile_args,
                options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
            
            def collect_functions(cursor: cindex.Cursor) -> None:
                """Collect all function declarations in the file."""
                # Skip if cursor is from a different file
                if cursor.location.file and cursor.location.file.name != file_path:
                    return
                
                # Check if it's a function/method
                if cursor.kind in (
                    cindex.CursorKind.FUNCTION_DECL,
                    cindex.CursorKind.CXX_METHOD,
                    cindex.CursorKind.CONSTRUCTOR,
                    cindex.CursorKind.DESTRUCTOR,
                ):
                    if cursor.spelling:
                        qualified_name = self.get_qualified_name(cursor)
                        uid = self.node_uid(cursor)
                        
                        # Map UID to qualified name
                        self.uid_to_qualified[uid] = qualified_name
                        self.qualified_to_uid[qualified_name] = uid
                        
                        # Create SFM if not exists
                        if qualified_name not in self.function_index:
                            extent = cursor.extent
                            sfm = FunctionSFM(
                                function_name=cursor.spelling,
                                qualified_name=qualified_name,
                                file_name=str(Path(file_path).relative_to(self.root_dir)),
                                module_name=module_name,
                                line_start=extent.start.line,
                                column_start=extent.start.column,
                                line_end=extent.end.line,
                                column_end=extent.end.column,
                            )
                            self.function_index[qualified_name] = sfm
                
                # Recurse into children
                for child in cursor.get_children():
                    collect_functions(child)
            
            collect_functions(tu.cursor)
            
        except Exception as e:
            print(f"[WARN] Failed to parse {file_path}: {e}")
    
    def extract_callee_name(self, cursor: cindex.Cursor) -> Optional[str]:
        """Extract callee function name from a call expression."""
        # Method 1: Use referenced cursor (most reliable)
        ref = cursor.referenced
        if ref and ref.kind in (
            cindex.CursorKind.FUNCTION_DECL,
            cindex.CursorKind.CXX_METHOD,
            cindex.CursorKind.CONSTRUCTOR,
            cindex.CursorKind.DESTRUCTOR,
        ):
            return self.get_qualified_name(ref)
        
        # Method 2: Extract from call expression structure
        children = list(cursor.get_children())
        if not children:
            return None
        
        func_cursor = children[0]
        
        # For member function calls: obj.method()
        if func_cursor.kind == cindex.CursorKind.MEMBER_REF_EXPR:
            for child in func_cursor.get_children():
                if child.kind == cindex.CursorKind.FIELD_IDENTIFIER:
                    method_name = child.spelling
                    # Try to find qualified name by resolving class
                    parent = func_cursor.semantic_parent
                    while parent:
                        if parent.kind in (cindex.CursorKind.CLASS_DECL, cindex.CursorKind.STRUCT_DECL):
                            class_name = parent.spelling
                            if class_name:
                                return f"{class_name}::{method_name}"
                        parent = parent.semantic_parent
                    return method_name
        
        # For direct function calls
        if func_cursor.spelling:
            return func_cursor.spelling
        
        # Try to get from display name
        try:
            display_name = cursor.displayname
            if display_name and "(" in display_name:
                return display_name.split("(")[0].strip()
        except Exception:
            pass
        
        return None
    
    def extract_call_edges(self, file_path: str) -> None:
        """Extract call relationships from a file."""
        if not self.is_cpp_file(file_path):
            return
        
        if not self.is_project_file(file_path):
            return
        
        try:
            tu = self.index.parse(
                file_path,
                args=self.compile_args,
                options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
            
            current_function: Optional[str] = None  # qualified_name
            
            def visit_calls(cursor: cindex.Cursor) -> None:
                nonlocal current_function
                
                # Skip if cursor is from a different file
                if cursor.location.file and cursor.location.file.name != file_path:
                    return
                
                # Update current function context
                if cursor.kind in (
                    cindex.CursorKind.FUNCTION_DECL,
                    cindex.CursorKind.CXX_METHOD,
                    cindex.CursorKind.CONSTRUCTOR,
                    cindex.CursorKind.DESTRUCTOR,
                ):
                    if cursor.spelling:
                        qualified_name = self.get_qualified_name(cursor)
                        if qualified_name in self.function_index:
                            current_function = qualified_name
                
                # Extract call expressions
                if cursor.kind == cindex.CursorKind.CALL_EXPR:
                    if current_function:
                        callee_qualified = self.extract_callee_name(cursor)
                        if callee_qualified:
                            # Check if callee is in our function index
                            if callee_qualified in self.function_index:
                                self.call_graph[current_function].add(callee_qualified)
                            else:
                                # Try simple name match
                                for qname in self.function_index:
                                    if qname.endswith(f"::{callee_qualified}") or qname == callee_qualified:
                                        self.call_graph[current_function].add(qname)
                                        break
                
                # Recurse into children
                for child in cursor.get_children():
                    old_function = current_function
                    visit_calls(child)
                    # Restore function context (functions can be nested in C++)
                    if old_function and cursor.kind not in (
                        cindex.CursorKind.FUNCTION_DECL,
                        cindex.CursorKind.CXX_METHOD,
                    ):
                        current_function = old_function
            
            visit_calls(tu.cursor)
            
        except Exception as e:
            print(f"[WARN] Failed to extract calls from {file_path}: {e}")
    
    def discover_callees_transitively(self, qualified_name: str) -> set[str]:
        """
        Discover all callees transitively for a function.
        Returns set of all transitive callees (qualified names).
        """
        all_callees: set[str] = set()
        
        # Direct callees
        direct_callees = self.call_graph.get(qualified_name, set())
        all_callees.update(direct_callees)
        
        # Transitive callees (recursive)
        for callee in direct_callees:
            if callee != qualified_name:  # Avoid self-loops
                transitive = self.discover_callees_transitively(callee)
                all_callees.update(transitive)
        
        return all_callees
    
    def build_sfm_from_ast(self, sfm: FunctionSFM, cursor: cindex.Cursor) -> None:
        """
        Build SFM description and flowchart from AST analysis (static analysis only).
        This is Stage 1: Static Code Understanding.
        """
        # Read function code
        file_path = self.root_dir / sfm.file_name
        if not file_path.exists():
            file_path = Path(sfm.file_name)
        
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                function_lines = lines[sfm.line_start - 1:sfm.line_end]
                function_code = "".join(function_lines)
        except Exception:
            function_code = ""
        
        # Analyze control flow and structure from AST
        control_flow_info = self._analyze_control_flow(cursor)
        
        # Generate description from AST analysis
        if self.use_llm and self.llm:
            sfm.description = self._generate_description_llm(function_code, control_flow_info, sfm.qualified_name)
        else:
            sfm.description = self._generate_description_static(control_flow_info, sfm.qualified_name)
        
        # Generate flowchart from AST analysis
        if self.use_llm and self.llm:
            sfm.flowchart = self._generate_flowchart_llm(function_code, control_flow_info, sfm.qualified_name, sfm.description)
        else:
            sfm.flowchart = self._generate_flowchart_static(control_flow_info, sfm.qualified_name)
    
    def _analyze_control_flow(self, cursor: cindex.Cursor) -> dict[str, Any]:
        """Analyze control flow structures from AST."""
        info = {
            "if_statements": [],
            "loops": [],
            "switch_statements": [],
            "returns": [],
            "function_calls": [],
            "variables": set(),
        }
        
        def analyze_node(c: cindex.Cursor) -> None:
            if c.kind == cindex.CursorKind.IF_STMT:
                info["if_statements"].append({
                    "line": c.location.line if c.location else 0,
                    "spelling": c.spelling or "",
                })
            elif c.kind == cindex.CursorKind.WHILE_STMT:
                info["loops"].append({
                    "type": "while",
                    "line": c.location.line if c.location else 0,
                })
            elif c.kind == cindex.CursorKind.FOR_STMT:
                info["loops"].append({
                    "type": "for",
                    "line": c.location.line if c.location else 0,
                })
            elif c.kind == cindex.CursorKind.DO_STMT:
                info["loops"].append({
                    "type": "do-while",
                    "line": c.location.line if c.location else 0,
                })
            elif c.kind == cindex.CursorKind.SWITCH_STMT:
                info["switch_statements"].append({
                    "line": c.location.line if c.location else 0,
                })
            elif c.kind == cindex.CursorKind.RETURN_STMT:
                info["returns"].append({
                    "line": c.location.line if c.location else 0,
                })
            elif c.kind == cindex.CursorKind.CALL_EXPR:
                callee = self.extract_callee_name(c)
                if callee:
                    info["function_calls"].append({
                        "name": callee,
                        "line": c.location.line if c.location else 0,
                    })
            elif c.kind == cindex.CursorKind.DECL_REF_EXPR:
                if c.spelling:
                    info["variables"].add(c.spelling)
            
            for child in c.get_children():
                analyze_node(child)
        
        analyze_node(cursor)
        info["variables"] = list(info["variables"])
        return info
    
    def _generate_description_static(self, control_flow_info: dict[str, Any], function_name: str) -> str:
        """Generate description using static analysis only."""
        parts = [f"Function: {function_name}"]
        
        if control_flow_info["if_statements"]:
            parts.append(f"Contains {len(control_flow_info['if_statements'])} conditional statement(s)")
        
        if control_flow_info["loops"]:
            loop_types = [l["type"] for l in control_flow_info["loops"]]
            parts.append(f"Contains {len(control_flow_info['loops'])} loop(s): {', '.join(loop_types)}")
        
        if control_flow_info["switch_statements"]:
            parts.append(f"Contains {len(control_flow_info['switch_statements'])} switch statement(s)")
        
        if control_flow_info["function_calls"]:
            call_names = [c["name"] for c in control_flow_info["function_calls"][:5]]
            parts.append(f"Calls {len(control_flow_info['function_calls'])} function(s): {', '.join(call_names)}")
        
        if control_flow_info["returns"]:
            parts.append(f"Has {len(control_flow_info['returns'])} return statement(s)")
        
        return ". ".join(parts) + "."
    
    def _generate_description_llm(self, function_code: str, control_flow_info: dict[str, Any], function_name: str) -> str:
        """Generate description using LLM (optional)."""
        if not self.use_llm or not self.llm:
            return self._generate_description_static(control_flow_info, function_name)
        
        prompt = f"""Analyze this C++ function and provide a clear, detailed description of its execution flow.

Function: {function_name}

Control Flow Analysis:
- If statements: {len(control_flow_info['if_statements'])}
- Loops: {len(control_flow_info['loops'])}
- Switch statements: {len(control_flow_info['switch_statements'])}
- Return statements: {len(control_flow_info['returns'])}
- Function calls: {len(control_flow_info['function_calls'])}

Function Code:
{function_code[:2000]}

Provide a comprehensive description of what this function does and how it executes."""
        
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return getattr(response, "content", str(response))
        except Exception as e:
            print(f"[WARN] LLM description generation failed: {e}")
            return self._generate_description_static(control_flow_info, function_name)
    
    def _generate_flowchart_static(self, control_flow_info: dict[str, Any], function_name: str) -> str:
        """Generate Mermaid flowchart using static analysis only."""
        lines = ["flowchart TD"]
        lines.append("    Start([Start])")
        
        node_id = "A"
        
        # Add conditional branches
        for i, if_stmt in enumerate(control_flow_info["if_statements"][:5]):  # Limit to 5
            cond_node = f"{node_id}{{Condition {i+1}?}}"
            lines.append(f"    {cond_node}")
            node_id = chr(ord(node_id) + 1)
            
            yes_node = f"{node_id}[Then]"
            no_node = chr(ord(node_id) + 1) + "[Else]"
            lines.append(f"    {cond_node} -->|Yes| {yes_node}")
            lines.append(f"    {cond_node} -->|No| {no_node}")
            node_id = chr(ord(no_node[0]) + 1)
        
        # Add loops
        for loop in control_flow_info["loops"][:3]:  # Limit to 3
            loop_node = f"{node_id}{{{loop['type'].title()} Loop}}"
            lines.append(f"    {loop_node}")
            node_id = chr(ord(node_id) + 1)
            
            body_node = f"{node_id}[Loop Body]"
            lines.append(f"    {loop_node} --> {body_node}")
            lines.append(f"    {body_node} --> {loop_node}")
            node_id = chr(ord(node_id) + 1)
        
        # Add function calls
        for call in control_flow_info["function_calls"][:5]:  # Limit to 5
            call_node = f"{node_id}[Call {call['name']}]"
            lines.append(f"    {call_node}")
            node_id = chr(ord(node_id) + 1)
        
        lines.append("    End([End])")
        
        # Connect last node to End
        if node_id > "A":
            prev_node = chr(ord(node_id) - 1)
            lines.append(f"    {prev_node} --> End")
        else:
            lines.append("    Start --> End")
        
        return "\n".join(lines)
    
    def _generate_flowchart_llm(self, function_code: str, control_flow_info: dict[str, Any], function_name: str, description: str) -> str:
        """Generate Mermaid flowchart using LLM (optional)."""
        if not self.use_llm or not self.llm:
            return self._generate_flowchart_static(control_flow_info, function_name)
        
        prompt = f"""Generate a valid Mermaid flowchart for this C++ function.

Function: {function_name}
Description: {description[:500]}

Control Flow:
- If statements: {len(control_flow_info['if_statements'])}
- Loops: {len(control_flow_info['loops'])}
- Function calls: {len(control_flow_info['function_calls'])}

Function Code:
{function_code[:2000]}

Generate ONLY valid Mermaid flowchart code starting with "flowchart TD".
Use simple node IDs (A, B, C, etc.).
Ensure all paths connect to End node.
No markdown, no explanations."""
        
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            flowchart = getattr(response, "content", str(response))
            return self._clean_flowchart(flowchart)
        except Exception as e:
            print(f"[WARN] LLM flowchart generation failed: {e}")
            return self._generate_flowchart_static(control_flow_info, function_name)
    
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
        
        return flowchart
    
    def analyze_function_bottom_up(self, qualified_name: str) -> None:
        """
        Analyze a function using bottom-up approach:
        1. First analyze all callees transitively
        2. Then analyze the function itself
        3. Cache results to avoid recomputation
        """
        # Check if already finalized (immutable cache)
        if qualified_name in self.function_index:
            sfm = self.function_index[qualified_name]
            if sfm.is_finalized():
                print(f"[CACHE] Reusing finalized SFM for: {qualified_name}")
                return  # Reuse existing analysis
        
        # Check for cycles
        if qualified_name in self.call_stack:
            print(f"[WARN] Cycle detected involving {qualified_name}, skipping recursion")
            return
        
        # Add to call stack
        self.call_stack.append(qualified_name)
        
        try:
            # Step 1: Discover all transitive callees
            callees = self.discover_callees_transitively(qualified_name)
            
            # Step 2: Analyze callees first (bottom-up)
            for callee in callees:
                if callee != qualified_name:  # Avoid self-loops
                    if callee not in self.visited_functions:
                        self.analyze_function_bottom_up(callee)
            
            # Step 3: Analyze this function
            if qualified_name not in self.visited_functions:
                self._analyze_function(qualified_name)
                self.visited_functions.add(qualified_name)
            
            # Step 4: Update callees list in SFM
            if qualified_name in self.function_index:
                sfm = self.function_index[qualified_name]
                # Include all transitive callees
                sfm.callees = sorted(list(callees))
                
                # Finalize SFM (make it immutable)
                sfm.finalize()
                print(f"[DONE] Finalized SFM for: {qualified_name} with {len(callees)} callees")
        
        finally:
            # Remove from call stack
            if self.call_stack and self.call_stack[-1] == qualified_name:
                self.call_stack.pop()
    
    def _analyze_function(self, qualified_name: str) -> None:
        """Analyze a single function (internal method)."""
        if qualified_name not in self.function_index:
            print(f"[WARN] Function {qualified_name} not found in index")
            return
        
        sfm = self.function_index[qualified_name]
        
        # Find the AST cursor for this function
        uid = self.qualified_to_uid.get(qualified_name)
        if not uid:
            print(f"[WARN] No UID found for {qualified_name}")
            return
        
        # Re-parse file to get cursor
        file_path = self.root_dir / sfm.file_name
        if not file_path.exists():
            file_path = Path(sfm.file_name)
        
        try:
            tu = self.index.parse(
                str(file_path),
                args=self.compile_args,
                options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
            
            def find_function_cursor(cursor: cindex.Cursor) -> Optional[cindex.Cursor]:
                if cursor.location.file and cursor.location.file.name != str(file_path):
                    return None
                
                if cursor.kind in (
                    cindex.CursorKind.FUNCTION_DECL,
                    cindex.CursorKind.CXX_METHOD,
                    cindex.CursorKind.CONSTRUCTOR,
                    cindex.CursorKind.DESTRUCTOR,
                ):
                    if cursor.spelling and self.get_qualified_name(cursor) == qualified_name:
                        if cursor.location.line == sfm.line_start:
                            return cursor
                
                for child in cursor.get_children():
                    result = find_function_cursor(child)
                    if result:
                        return result
                
                return None
            
            func_cursor = find_function_cursor(tu.cursor)
            if func_cursor:
                print(f"[INFO] Building SFM for: {qualified_name}")
                self.build_sfm_from_ast(sfm, func_cursor)
            else:
                print(f"[WARN] Could not find AST cursor for {qualified_name}")
        
        except Exception as e:
            print(f"[WARN] Failed to analyze {qualified_name}: {e}")
    
    def run(self) -> dict[str, FunctionSFM]:
        """
        Main execution pipeline:
        1. Parse all C++ files and collect functions
        2. Extract call relationships
        3. Analyze functions bottom-up
        4. Return finalized SFM dictionary
        """
        print(f"[INFO] Starting analysis of: {self.root_dir}")
        
        # Stage 1: Collect all functions
        print("[INFO] Stage 1: Collecting function declarations...")
        for root, _, files in os.walk(self.root_dir):
            for f in files:
                if self.is_cpp_file(f):
                    file_path = os.path.join(root, f)
                    self.parse_file(file_path)
        
        print(f"[INFO] Found {len(self.function_index)} functions")
        
        # Stage 2: Extract call relationships
        print("[INFO] Stage 2: Extracting call relationships...")
        for root, _, files in os.walk(self.root_dir):
            for f in files:
                if self.is_cpp_file(f):
                    file_path = os.path.join(root, f)
                    self.extract_call_edges(file_path)
        
        total_calls = sum(len(callees) for callees in self.call_graph.values())
        print(f"[INFO] Found {total_calls} call relationships")
        
        # Stage 3: Analyze all functions bottom-up
        print("[INFO] Stage 3: Analyzing functions (bottom-up)...")
        for qualified_name in list(self.function_index.keys()):
            if qualified_name not in self.visited_functions:
                self.analyze_function_bottom_up(qualified_name)
        
        print(f"[INFO] Completed analysis of {len(self.visited_functions)} functions")
        
        return self.function_index
    
    def export_sfm_json(self, output_path: str | Path) -> Path:
        """Export all SFMs to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sfm_list = [sfm.to_dict() for sfm in self.function_index.values()]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sfm_list, f, indent=2)
        
        print(f"[INFO] Exported {len(sfm_list)} SFMs to: {output_path}")
        return output_path
    
    def export_flowcharts(self, output_dir: str | Path) -> Path:
        """Export Mermaid flowcharts to separate files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for qualified_name, sfm in self.function_index.items():
            if sfm.flowchart:
                # Create safe filename from qualified name
                safe_name = re.sub(r'[<>:"/\\|?*]', '_', qualified_name)
                flowchart_path = output_dir / f"{safe_name}.mmd"
                
                with open(flowchart_path, "w", encoding="utf-8") as f:
                    f.write(f"# {sfm.qualified_name}\n")
                    f.write(f"# File: {sfm.file_name}\n\n")
                    f.write(sfm.flowchart)
                
                count += 1
        
        print(f"[INFO] Exported {count} flowcharts to: {output_dir}")
        return output_dir


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="C++ Execution-Flow Visualization Agent"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory of C++ project"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="sfm_output.json",
        help="Output JSON file path (default: sfm_output.json)"
    )
    parser.add_argument(
        "--flowcharts-dir",
        type=str,
        default="flowcharts",
        help="Directory for Mermaid flowchart files (default: flowcharts)"
    )
    parser.add_argument(
        "--compile-args",
        nargs="*",
        default=["-std=c++17"],
        help="Compilation arguments (default: -std=c++17)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM for description/flowchart generation (use static analysis only)"
    )
    
    args = parser.parse_args()
    
    # Create agent
    agent = ExecutionFlowAgent(
        root_dir=args.root_dir,
        compile_args=args.compile_args,
        use_llm=not args.no_llm,
    )
    
    # Run analysis
    try:
        agent.run()
        
        # Export results
        agent.export_sfm_json(args.output)
        agent.export_flowcharts(args.flowcharts_dir)
        
        print("[INFO] Analysis complete!")
        
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

