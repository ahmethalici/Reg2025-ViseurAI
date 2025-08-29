import json
import re
import sys

def insert_newlines(text: str) -> str:
    """
    Rule-based newline insertion for predicted_report text:
      1) After each semicolon (;), insert '\n  ' (newline + two spaces)
      2) Before numbered list items '2.', '3.', ... (but NOT '1.'), insert '\n  '
      3) Before ') with', insert '\n  '
      4) Before 'Note', insert '\n\n'
      5) Before any dash-list item '- ', insert '\n  '
    """
    text = re.sub(r';\s*', ';\n  ', text)
    text = re.sub(r'(?<!\n)(?P<num>(?!1\.)\d+\.\s)', r'\n  \g<num>', text)
    text = re.sub(r'\)\s+with', ')\n  with', text)
    text = re.sub(r'(?<!\n\n)(?P<n>Note)', r'\n\n\g<n>', text)
    text = re.sub(r'(?<!\n)(- )', r'\n  \1', text)
    return text

def process_json(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data:
        pred = entry.get('predicted_report')
        if isinstance(pred, str):
            entry['predicted_report'] = insert_newlines(pred)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 fix_newlines.py <input.json> <output.json>")
        sys.exit(1)
    inp, out = sys.argv[1], sys.argv[2]
    process_json(inp, out)
    print(f"Processed JSON written to: {out}")