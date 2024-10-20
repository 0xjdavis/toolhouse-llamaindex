from toolhouse import Toolhouse
import json
import dotenv

dotenv.load_dotenv()

def ToolhouseLlamaIndex(th: Toolhouse):
    api_key = th.api_key
    data = th.get_tools(bundle=th.bundle)
    
    # Start building the class string
    class_str = """
from toolhouse import Toolhouse
from toolhouse.models import RunToolsRequest
from typing import Optional
from toolhouse.services.tools import Tools
from toolhouse.models.RunToolsRequest import RunToolsRequest
from llama_index.core.tools.tool_spec.base import BaseToolSpec
import json
class ToolhouseToolsSpec(BaseToolSpec):
"""
    
    # Iterate through each function in the JSON data
    
    function_names = []
    for item in data:
        if item['type'] == 'function':
            func = item['function']
            func_name = func['name']
            function_names.append(func_name)
            description = func['description']
            parameters = func['parameters']['properties']
            
            # Build the function signature
            args = []
            for param_name, param_info in parameters.items():
                param_type = param_info['type']
                if param_type == 'string':
                    param_type = 'str'
                if param_type == 'number':
                    param_type = 'int'
                if param_name not in item.get('required', []):
                    param_type = f"Optional[{param_type}]"
                args.append(f"{param_name}: {param_type}")
            
            arg_str = ", ".join(args)
            
            # Build the function docstring
            docstring = f'    """{description}\n    Returns: {func_name} results\n\n    Args:\n'
            for param_name, param_info in parameters.items():
                param_desc = param_info['description']
                docstring += f"        {param_name}: {param_desc}\n"
            docstring += '    """'
            
            # Build the function body
            function_body = "        arguments = {\n"
            for param_name in parameters.keys():
                function_body += f'            "{param_name}": {param_name},\n'
            function_body = function_body.rstrip(',\n') + "\n        }\n"
            
            function_body += f"""
        tool = {{
            "id": f"id_{func_name}",
            "function": {{
                "name": "{func_name}",
                "arguments": json.dumps(arguments)
            }},
            "type": "function"
        }}
        
        th_tools = Tools("{api_key}")
        th_tools.set_base_url("https://api.toolhouse.ai/v1")
        run_tools_request = RunToolsRequest(tool, "openai", {json.dumps(th.metadata)}, "{th.bundle}")
        run_response = th_tools.run_tools(run_tools_request)
        return run_response.content
"""
            
            # Add the function to the class string
            class_str += f"""
    def {func_name}(self, {arg_str}):
    {docstring}
{function_body}
"""

    class_str += f"""
    spec_functions = {json.dumps(function_names)}
"""
    
    # Execute the class string in the global namespace
    exec(class_str, globals())
    if 'ToolhouseToolsSpec' in globals():
      return globals().get('ToolhouseToolsSpec')
