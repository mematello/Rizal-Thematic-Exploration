import re
import json
import os
import sys

def parse_character_data():
    frontend_ts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'lib', 'characterData.ts')
    backend_data_dir = os.path.join(os.path.dirname(__file__), '..', 'app', 'data')
    json_path = os.path.join(backend_data_dir, 'character_aliases.json')
    
    if not os.path.exists(backend_data_dir):
        os.makedirs(backend_data_dir)
        
    try:
        with open(frontend_ts_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract the array content
        match = re.search(r'export const CHARACTERS:\s*Character\[\]\s*=\s*\[(.*?)\];', content, re.DOTALL)
        if not match:
            print("Failed to find CHARACTERS array in characterData.ts")
            sys.exit(1)
            
        array_content = match.group(1)
        
        # We need to parse JS objects into JSON
        # Find all objects using bracket matching or regex
        objects = []
        for obj_match in re.finditer(r'\{[^{}]+\}', array_content):
            obj_str = obj_match.group(0)
            
            # Very basic string manipulation to make it valid JSON
            # 1. Quote unquoted keys: name: -> "name":
            obj_str = re.sub(r'(\w+)\s*:', r'"\1":', obj_str)
            
            try:
                # Use JSON to parse string
                obj = json.loads(obj_str)
                objects.append(obj)
            except json.JSONDecodeError as e:
                # If parsing fails, fall back to evaluation if absolutely needed,
                # but JS objects might have trailing commas or differing quotes.
                # Let's clean it up more safely: Replace single quotes with double quotes
                # assuming there's no complex nested single quotes inside arrays.
                obj_str = obj_str.replace("'", '"')
                # Remove trailing commas before closing braces/brackets
                obj_str = re.sub(r',\s*\}', '}', obj_str)
                obj_str = re.sub(r',\s*\]', ']', obj_str)
                try:
                    obj = json.loads(obj_str)
                    objects.append(obj)
                except Exception as e2:
                    print(f"Failed to parse object: {obj_str} - error: {e2}")
                    
        # Write to JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(objects, f, indent=4)
            
        print(f"Successfully extracted {len(objects)} characters to {json_path}")
        
    except Exception as e:
        print(f"Error parsing character data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parse_character_data()
