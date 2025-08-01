import os
import gdown

os.makedirs("Model", exist_ok=True)

# رابط المجلد من google drive
file_id = "1I6KDnvG0fntWs-ikO0ciKew5FcHmAvyj"
url = f"https://drive.google.com/uc?id={file_id}"

# file name
output = "Model/full_integrated_model.pkl"

# load file
gdown.download(url, output, quiet=False)
