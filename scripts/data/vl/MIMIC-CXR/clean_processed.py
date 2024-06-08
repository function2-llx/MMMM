import json

from mmmm.data.defs import PROCESSED_VL_DATA_ROOT

def main():
    for split in ['test', 'train']:
        with open(PROCESSED_VL_DATA_ROOT / 'MIMIC-CXR' / f'{split}-processed.json', 'r') as f:
            data = json.load(f)
        output = []
        for item in data:
            valid = True
            try:
                if not item['processed_report'].split('Impression:')[1].strip():
                    valid = False
                if not item['processed_report'].split('Findings:')[1].split('Impression:')[0].strip():
                    valid = False
            except:
                valid = False
            if valid:
                output.append(item)
        print(f'{split}: {len(data)} -> {len(output)}')
        with open(PROCESSED_VL_DATA_ROOT / 'MIMIC-CXR' / f'{split}-cleaned.json', 'w') as f:
            json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()