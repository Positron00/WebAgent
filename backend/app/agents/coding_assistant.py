"""
Coding Assistant Agent for the WebAgent backend.

This agent generates Python code for data visualization and analysis based on research findings.
"""
from typing import Dict, List, Any
import logging
from datetime import datetime
import json
import os

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage

from app.models.task import WorkflowState
from app.services.llm import get_llm_without_tags
from app.agents.base_agent import BaseAgent
from app.core.loadEnvYAML import get_agents_config
from app.core.config import settings

logger = logging.getLogger(__name__)

# Prompt for the Coding Assistant Agent
CODING_ASSISTANT_PROMPT = """You are a Coding Assistant Agent specializing in Python code generation for data visualization and analysis.

USER QUERY: {query}
RESEARCH FINDINGS: {research_findings}
DATA INSIGHTS: {data_insights}

Your task:
1. Generate Python code to visualize the key insights from the research
2. Create appropriate visualizations (charts, graphs, plots) that best represent the data
3. Use libraries such as matplotlib, seaborn, pandas, and numpy
4. Include clear comments explaining each section of the code
5. Make sure the code is clean, efficient, and properly structured
6. Format the code nicely with markdown code blocks

ALLOWED LIBRARIES: {allowed_libraries}

Your code should:
- Be self-contained and executable
- Have appropriate error handling
- Use sensible default data when specific data is not available (IMPORTANT: Generate sample data if input files don't exist)
- Create visually appealing and informative charts
- Include a title and labels for all visualizations

IMPORTANT: When handling missing files or data:
1. Use try-except blocks to gracefully catch errors
2. Provide users with clear instructions on how to input their own data
3. Generate sample data as a fallback with a note explaining it's sample data
4. Never assume external files exist without checking first

IMPORTANT: Focus on code that directly supports the insights from the research. Make sure your code is clean, well-documented, and follows best practices.

Your response should have the following format:

EXPLANATION:
A brief explanation of what your visualizations will show and why you chose these specific visualizations.

CODE:
```python
# Your complete Python code here
```

IMPORTANT: The code must be self-contained, properly formatted with good documentation, and fully executable. Never truncate the code.
"""

class CodingAssistantAgent(BaseAgent):
    """
    Coding Assistant Agent that generates code for data visualization.
    """
    
    def __init__(self):
        """Initialize the Coding Assistant Agent."""
        super().__init__(name="coding_assistant")
        
        # Get configuration for allowed modules
        agents_config = get_agents_config()
        self.allowed_libraries = ", ".join(agents_config.coding.allowed_modules)
        self.timeout_seconds = agents_config.coding.timeout_seconds
        
        # Initialize OpenAI client - we'll use GPT-4o mini with code interpreter
        self.openai_api_key = self.get_openai_api_key()
        self.openai_model = "gpt-4o-mini"
        
        if self.openai_api_key:
            self.openai_llm = ChatOpenAI(
                temperature=0.7, 
                model=self.openai_model,
                api_key=self.openai_api_key,
                max_tokens=5000
            )
            logger.info(f"Initialized OpenAI {self.openai_model} for Coding Assistant with max_tokens={self.openai_llm.max_tokens}")
        else:
            # Fallback to the default LLM if OpenAI key not found
            logger.warning("OpenAI API key not found. Using fallback LLM")
            self.openai_llm = None
            self.llm = get_llm_without_tags()
    
    def get_openai_api_key(self):
        """Get OpenAI API key from environment variables or .env.local file."""
        # Check if key is already in environment
        if "OPENAI_API_KEY" in os.environ:
            return os.environ["OPENAI_API_KEY"]
        
        # Check settings from app config
        if hasattr(settings, "OPENAI_API_KEY") and settings.OPENAI_API_KEY:
            return settings.OPENAI_API_KEY
        
        # Try loading from .env.local file
        try:
            env_path = os.path.join(os.getcwd(), ".env.local")
            if os.path.exists(env_path):
                with open(env_path, "r") as f:
                    for line in f:
                        if line.strip().startswith("OPENAI_API_KEY="):
                            key = line.strip().split("=", 1)[1].strip()
                            # Remove quotes if present
                            if (key.startswith('"') and key.endswith('"')) or \
                               (key.startswith("'") and key.endswith("'")):
                                key = key[1:-1]
                            return key
        except Exception as e:
            logger.error(f"Error reading .env.local file: {str(e)}")
        
        return None

    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Run the Coding Assistant Agent on the current workflow state.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state with code generation results
        """
        try:
            logger.info(f"Coding Assistant Agent processing query: {state.query}")
            
            # Get the research findings and data insights
            senior_research_output = state.reports.get("senior_research", {})
            research_findings = senior_research_output.get("synthesis", "No research findings available.")
            
            data_analysis = state.reports.get("data_analysis", {})
            data_insights = data_analysis.get("analysis", "No data insights available.")
            
            logger.debug("Generating visualization code based on research findings and data insights")
            
            # Build the prompt string manually
            prompt = CODING_ASSISTANT_PROMPT.format(
                query=state.query,
                research_findings=research_findings,
                data_insights=data_insights,
                allowed_libraries=self.allowed_libraries
            )
            
            # Use the most reliable method to get results
            result = await self._generate_with_retry(prompt)
            logger.debug(f"Raw result from _generate_with_retry: {result[:500]}...")
            
            # Parse the result as JSON if possible
            try:
                # Try parsing the JSON output
                if isinstance(result, str):
                    # Clean up the result to handle common JSON formatting issues
                    cleaned_result = result.strip()
                    
                    # Remove markdown code block markers if they enclose the entire result
                    if cleaned_result.startswith("```") and cleaned_result.endswith("```"):
                        cleaned_result = cleaned_result[3:-3].strip()
                    
                    # Remove "json" language marker if present
                    if cleaned_result.startswith("```json") and cleaned_result.endswith("```"):
                        cleaned_result = cleaned_result[7:-3].strip()
                    
                    # Try to parse as JSON if it looks like JSON
                    if (cleaned_result.startswith("{") and cleaned_result.endswith("}")) or \
                       (cleaned_result.startswith("[") and cleaned_result.endswith("]")):
                        try:
                            code_data = json.loads(cleaned_result)
                        except json.JSONDecodeError as e:
                            # Handle different JSON parsing errors
                            if "Invalid control character" in str(e):
                                logger.warning("Invalid control character in JSON. Attempting to sanitize...")
                                # First approach: replace control characters with spaces
                                import re
                                sanitized_json = re.sub(r'[\x00-\x1F\x7F]', ' ', cleaned_result)
                                try:
                                    code_data = json.loads(sanitized_json)
                                    logger.info("Successfully parsed JSON after sanitizing control characters")
                                except:
                                    # Second approach: manually escape control characters
                                    import re
                                    # Replace common control sequences with proper escapes
                                    sanitized_json = cleaned_result.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                                    # Also fix double quotes inside strings
                                    fixed_json = re.sub(r'("(?:[^"\\]|\\.)*)"([^"]*)"', r'\1\\"\2\\\"', sanitized_json)
                                    try:
                                        code_data = json.loads(fixed_json)
                                        logger.info("Successfully parsed JSON after escaping control characters")
                                    except:
                                        raise # Let it fall through to general error handling
                        except json.JSONDecodeError as e:
                            if "Extra data" in str(e):
                                import re
                                json_match = re.search(r'(\{.*\})', cleaned_result, re.DOTALL)
                                if json_match:
                                    logger.warning("Extra data error encountered. Using extracted JSON substring.")
                                    code_data = json.loads(json_match.group(1))
                                else:
                                    raise
                            else:
                                import re
                                fixed_json = re.sub(r'("(?:\\.|[^"\\])*")|\n', lambda m: m.group(1) or "\\n", cleaned_result)
                                try:
                                    code_data = json.loads(fixed_json)
                                except:
                                    raise  # Let it fall through to the extraction code
                    else:
                        # Try to extract JSON from text
                        import re
                        json_match = re.search(r'(\{.*\})', cleaned_result, re.DOTALL)
                        if json_match:
                            try:
                                code_data = json.loads(json_match.group(1))
                            except:
                                raise  # Let it fall through to the extraction code
                        else:
                            # Look for partial JSON structure with a code key
                            code_snippet_match = re.search(r'"code":\s*"(.*?)(?:"|$)', cleaned_result, re.DOTALL)
                            if code_snippet_match:
                                code_snippet = code_snippet_match.group(1)
                                # If we found a code snippet, create a simple data structure with it
                                code_data = {
                                    "explanation": "Generated code based on your query",
                                    "visualizations": [],
                                    "code": code_snippet
                                }
                                logger.warning("Found partial code snippet in malformed JSON")
                            else:
                                raise ValueError("No JSON structure found")
                else:
                    # Result is already structured data
                    code_data = result
                
            except Exception as json_error:
                logger.error(f"Error parsing coding output as JSON: {str(json_error)}")
                # Extract code blocks if present, otherwise use the raw output
                import re
                
                # First, try to find Python code blocks with triple backticks
                code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', result, re.DOTALL)
                
                if code_blocks:
                    # Use the longest code block
                    code = max(code_blocks, key=len).strip()
                    logger.warning("Error parsing JSON, using extracted code block from markdown")
                else:
                    # No code blocks with triple backticks found
                    
                    # Look for code snippet with "code": " pattern that often appears in malformed JSON
                    code_snippet_match = re.search(r'"code":\s*"(.*?)(?:"|$)', result, re.DOTALL)
                    if code_snippet_match:
                        code = code_snippet_match.group(1)
                        logger.warning("Extracted code snippet from malformed JSON using 'code' key pattern")
                    else:
                        # Next, try to find code blocks that might have different formatting
                        # Look for indented blocks that look like Python code
                        lines = result.split('\n')
                        code_lines = []
                        current_block = []
                        in_block = False
                        
                        for line in lines:
                            # Check for python code indicators
                            if any(marker in line.lower() for marker in ['import ', 'def ', 'class ', 'from ', '    ']):
                                if not in_block:
                                    in_block = True
                                current_block.append(line)
                            elif in_block and line.strip() == '':
                                # Empty line within a block is fine
                                current_block.append(line)
                            elif in_block:
                                # End of block
                                if len(current_block) > 2:  # Only consider blocks with at least 3 lines
                                    code_lines.extend(current_block)
                                    code_lines.append('')  # Add separator
                                current_block = []
                                in_block = False
                        
                        # Add the last block if it exists
                        if in_block and len(current_block) > 2:
                            code_lines.extend(current_block)
                        
                        if code_lines:
                            code = '\n'.join(code_lines)
                            logger.warning("Error parsing JSON, using extracted code lines")
                        else:
                            # If all else fails, look for Python file patterns
                            code_patterns = [
                                r'(import\s+[a-zA-Z0-9_]+.*?)(?=\n\n|\Z)',
                                r'(from\s+[a-zA-Z0-9_\.]+\s+import.*?)(?=\n\n|\Z)',
                                r'(def\s+[a-zA-Z0-9_]+\s*\(.*?)(?=\n\n|\Z)',
                                r'(class\s+[a-zA-Z0-9_]+.*?)(?=\n\n|\Z)'
                            ]
                            
                            matched_blocks = []
                            for pattern in code_patterns:
                                matches = re.findall(pattern, result, re.DOTALL)
                                matched_blocks.extend(matches)
                            
                            if matched_blocks:
                                code = '\n\n'.join(matched_blocks)
                                logger.warning("Error parsing JSON, using regex-extracted code patterns")
                            else:
                                # Last resort - look for anything that resembles Python code
                                python_keywords = ['def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'with', 'async', 'await']
                                candidate_lines = []
                                
                                for line in lines:
                                    line_stripped = line.strip()
                                    if line_stripped and any(keyword in line_stripped.split() for keyword in python_keywords):
                                        candidate_lines.append(line)
                                
                                if len(candidate_lines) > 3:  # Only if we have at least a few lines
                                    code = '\n'.join(candidate_lines)
                                    logger.warning("Error parsing JSON, using lines containing Python keywords")
                                else:
                                    # Absolute last resort
                                    code = "# Could not extract meaningful code from the response\n# Please try again with a more specific prompt"
                                    logger.warning("Could not extract any meaningful code from the response")
                
                # Create a basic structure with the extracted code
                code_data = {
                    "explanation": "Generated code based on your query",
                    "visualizations": [],
                    "code": code
                }
            
            # Ensure that the extracted code is not truncated
            if not code_data.get("code") or len(code_data.get("code").strip()) < 50:
                logger.warning("Extracted code appears to be truncated. Attempting fallback extraction from full result.")
                if "```python" in result:
                    extracted_code = result.split("```python", 1)[1].strip()
                    if "```" in extracted_code:
                        extracted_code = extracted_code.split("```", 1)[0].strip()
                    code_data["code"] = f"```python\n{extracted_code}\n```"
                    logger.info("Fallback extraction succeeded using split on '```python'.")
                elif any(marker in result for marker in ["import ", "def ", "class "]):
                    # Extract code directly if it looks like Python but doesn't have backticks
                    logger.info("Extracted code directly from result that contains Python keywords.")
                    code_data["code"] = f"```python\n{result}\n```"
                else:
                    logger.warning("No python code block marker found in result; using full result as code.")
                    code_data["code"] = result

            # Create the coding assistant report
            coding_report = {
                "status": "completed",
                "code": code_data.get("code", ""),  # Ensure we don't store the raw result if parsing succeeded
                "explanation": code_data.get("explanation", "Generated code based on your query"),
                "visualizations": code_data.get("visualizations", []),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add the report to the state
            state.update_with_agent_output("coding_assistant", coding_report)
            
            logger.info("Coding Assistant Agent completed code generation successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in Coding Assistant Agent: {str(e)}", exc_info=True)
            state.mark_error(f"Coding Assistant Agent error: {str(e)}")
            return state

    async def _generate_with_retry(self, prompt, max_retries=3, base_delay=5):
        """
        Generate code with retry logic for rate limits.
        
        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum number of retries (default 3)
            base_delay: Base delay for exponential backoff (default 5 seconds)
            
        Returns:
            The generated text
        """
        import time
        import asyncio
        import re
        
        # Prioritize OpenAI methods if available
        if self.openai_llm:
            methods = [
                self._try_openai_code_interpreter,
                self._try_openai_completion,
                self._try_langchain_invoke,  # As fallback
                self._try_fallback
            ]
        else:
            # Original methods if OpenAI not available
            methods = [
                self._try_direct_api,
                self._try_langchain_agenerate,
                self._try_langchain_invoke,
                self._try_fallback
            ]
        
        request_count = 0  # Track number of LLM requests made
        
        # Go through each method with retries
        for attempt, method in enumerate(methods):
            for retry in range(max_retries):
                if request_count > 0:
                    logger.info("Waiting 2 seconds between LLM requests to avoid rate limits.")
                    await asyncio.sleep(2)  # OpenAI has higher rate limits, so we can use shorter delays
                try:
                    logger.debug(f"Trying method {method.__name__}, attempt {retry+1}/{max_retries}")
                    result = await method(prompt)
                    request_count += 1
                    return result
                except Exception as e:
                    request_count += 1
                    error_msg = str(e)
                    logger.warning(f"Error in {method.__name__}: {error_msg}")
                    
                    # Check if it's a rate limit error
                    if "429" in error_msg or "Too Many Requests" in error_msg:
                        retry_after = 5  # Default
                        match = re.search(r'Retry-After[\'\"]:\s*[\'\"](\d+)[\'\"]', error_msg)
                        if match:
                            retry_after = int(match.group(1))
                        
                        wait_time = retry_after * (2 ** retry)
                        logger.info(f"Rate limited. Waiting {wait_time}s before retry {retry+1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                    else:
                        break
        
        # If all methods fail, return a basic response
        logger.error("All code generation methods failed")
        return """{"explanation": "Error generating code", "code": "# Error occurred during code generation\\n# Please try again later"}"""
    
    async def _try_openai_code_interpreter(self, prompt):
        """Use OpenAI with code interpreter tool enabled."""
        try:
            import openai
            
            # Configure the client
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # Create the messages
            messages = [
                {"role": "system", "content": "You are a specialized Python coding assistant. Generate clean, well-documented code using data visualization libraries like matplotlib, seaborn, pandas, and numpy."},
                {"role": "user", "content": prompt}
            ]
            
            try:
                # Try first with code interpreter as a tool type directly
                response = client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0.7,
                    tools=[{"type": "code_interpreter"}],
                    tool_choice="auto",
                    max_tokens=5000
                )
            except Exception as e:
                if "Missing required parameter: 'tools[0].function'" in str(e):
                    logger.warning("Trying alternative format for code interpreter tool")
                    # Alternative format with a function definition
                    response = client.chat.completions.create(
                        model=self.openai_model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=5000
                    )
                else:
                    raise
            
            # Extract the response content
            result = response.choices[0].message.content
            
            # Log the raw response for debugging
            logger.debug(f"Raw OpenAI response: {response}")
            logger.debug(f"Content from response: {result}")
            
            # Also check for tool outputs
            tool_outputs = []
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls is not None:
                for tool_call in response.choices[0].message.tool_calls:
                    if hasattr(tool_call, 'type') and tool_call.type == 'code_interpreter' and hasattr(tool_call, 'code_interpreter'):
                        if hasattr(tool_call.code_interpreter, 'input'):
                            tool_outputs.append(f"```python\n{tool_call.code_interpreter.input}\n```")
                        if hasattr(tool_call.code_interpreter, 'outputs'):
                            for output in tool_call.code_interpreter.outputs:
                                if hasattr(output, 'type') and output.type == 'image':
                                    tool_outputs.append(f"# Image output available but not displayed in text")
                                elif hasattr(output, 'text'):
                                    tool_outputs.append(f"# Output:\n# {output.text.replace('\n', '\n# ')}")
            
            # Combine the text response with any tool outputs
            if tool_outputs:
                result += "\n\n" + "\n\n".join(tool_outputs)
            
            # Extract explanation and code from structured format
            import re
            
            # Log the raw output for debugging
            logger.debug(f"Raw output from LLM: {result[:500]}...")
            
            # Extract code blocks
            code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', result, re.DOTALL)
            
            # Extract explanation
            explanation_match = re.search(r'EXPLANATION:(.*?)(?:CODE:|```|$)', result, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "Generated code based on your query"
            
            # If code blocks found, use the longest one
            if code_blocks:
                code = max(code_blocks, key=len).strip()
                logger.debug(f"Extracted code block with length: {len(code)}")
            else:
                # If no code blocks with backticks found, look for anything that might be code
                code_patterns = ['import ', 'def ', 'class ', 'if __name__']
                code_lines = []
                in_code_section = False
                
                for line in result.split('\n'):
                    if any(pattern in line for pattern in code_patterns):
                        in_code_section = True
                    
                    if in_code_section:
                        code_lines.append(line)
                
                code = '\n'.join(code_lines) if code_lines else result
                logger.debug(f"No code blocks found, extracted {len(code_lines)} lines with code-like patterns")
            
            # Create a JSON object with the extracted code and explanation
            formatted_result = {
                "explanation": explanation,
                "visualizations": [],
                "code": f"```python\n{code}\n```"
            }
            
            # Convert to JSON string
            import json
            result = json.dumps(formatted_result)
            logger.debug(f"Reformatted response as JSON: {result[:100]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI code interpreter error: {str(e)}")
            raise
    
    async def _try_openai_completion(self, prompt):
        """Use OpenAI chat completion without code interpreter."""
        try:
            # Use langchain's wrapper for OpenAI
            messages = [HumanMessage(content=prompt)]
            response = await self.openai_llm.agenerate([messages])
            return response.generations[0][0].text
        except Exception as e:
            logger.error(f"OpenAI completion error: {str(e)}")
            raise
    
    async def _try_direct_api(self, prompt):
        """Try using direct API call."""
        import requests
        import os
        from app.core.config import settings
        
        together_api_key = settings.TOGETHER_API_KEY
        
        url = "https://api.together.xyz/v1/completions"
        headers = {
            "Authorization": f"Bearer {together_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "prompt": prompt,
            "max_tokens": 2500,
            "temperature": 0.7,
        }
        
        # Use synchronous request in async function
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: requests.post(url, headers=headers, json=data)
        )
        
        response_data = response.json()
        
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["text"]
        else:
            raise ValueError(f"Unexpected API response: {response_data}")
    
    async def _try_langchain_agenerate(self, prompt):
        """Try using LangChain agenerate."""
        # Extract content from message to avoid JSON serialization issues
        response = await self.llm.agenerate([prompt], tags=[])
        return response.generations[0][0].text
    
    async def _try_langchain_invoke(self, prompt):
        """Try using LangChain invoke."""
        result = await self.llm.invoke(prompt)
        return result
    
    async def _try_fallback(self, prompt):
        """Try using OpenAI API as fallback."""
        from langchain_openai import ChatOpenAI
        from langchain.schema.messages import HumanMessage
        
        try:
            openai_llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=2500)
            messages = [HumanMessage(content=prompt)]
            response = await openai_llm.agenerate([messages])
            return response.generations[0][0].text
        except Exception as e:
            logger.error(f"OpenAI fallback failed: {str(e)}")
            # Last resort fallback
            return """{"explanation": "All code generation methods failed", "code": "# Error generating code\\n# Please try again later"}"""


def get_coding_assistant_agent():
    """
    Get an instance of the Coding Assistant Agent.
    
    Returns:
        A callable function that takes a WorkflowState and calls the agent's run method.
        This makes it compatible with LangGraph which expects nodes to be callable.
    """
    agent = CodingAssistantAgent()
    
    # Return a callable function instead of the agent directly
    async def coding_assistant_function(state: WorkflowState) -> WorkflowState:
        """Callable wrapper for the coding assistant agent"""
        return await agent.run(state)
    
    return coding_assistant_function 