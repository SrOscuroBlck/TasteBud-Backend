import json
import sys

sys.path.insert(0, ".")
from config.database import get_session
from models.ingestion import MenuUpload

upload_id = sys.argv[1] if len(sys.argv) > 1 else "d2f6c785-d826-4a4f-8044-08ac88af97d2"
session = next(get_session())
upload = session.get(MenuUpload, upload_id)

if not upload:
    print(f"Upload {upload_id} not found")
    sys.exit(1)

print(f"STATUS: {upload.status}")
print(f"ERROR: {upload.error_message}")
print()

if upload.extracted_text:
    print("EXTRACTED TEXT (first 2000 chars):")
    print(upload.extracted_text[:2000])
    print("...")
    print(f"TOTAL TEXT LENGTH: {len(upload.extracted_text)}")
else:
    print("NO EXTRACTED TEXT")

print()

if upload.parsed_data:
    data = upload.parsed_data
    if isinstance(data, dict):
        print(f"PARSED DATA KEYS: {list(data.keys())}")
        data_str = json.dumps(data, ensure_ascii=False, indent=2)
        print(f"PARSED DATA (first 3000 chars):")
        print(data_str[:3000])
    else:
        print(f"PARSED DATA TYPE: {type(data)}")
        print(str(data)[:2000])
else:
    print("NO PARSED DATA")
