
import json

def convert_jsonl_to_md(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write("# Terminal History Log\n\n")
            for line in f_in:
                try:
                    entry = json.loads(line)
                    f_out.write(f"## Entry\n")
                    f_out.write(f"- **Timestamp:** {entry.get('timestamp', 'N/A')}\n")
                    f_out.write(f"- **Intent:** {entry.get('intent', 'N/A')}\n")
                    f_out.write(f"- **Code Input:**\n```python\n{entry.get('input', '')}\n```\n")
                    f_out.write(f"- **Code Output:**\n```text\n{entry.get('output', '')}\n```\n\n")
                except json.JSONDecodeError:
                    continue
    print(f"Successfully updated {output_file} with correct fields.")

if __name__ == "__main__":
    convert_jsonl_to_md('.agent_data/terminal_history.jsonl', '.agent_data/terminal_history.md')
