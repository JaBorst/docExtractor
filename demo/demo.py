
# Add code to sys.path
import sys
sys.path.append('../src')

# Display
from IPython.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
#%% md
# ## 1. Load default model
#%%
# Select GPU ID
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#%%
import torch
from docextractorfork.models import load_model_from_path
from docextractorfork.utils import coerce_to_path_and_check_exist
from docextractorfork.utils.path import MODELS_PATH
from docextractorfork.utils.constant import MODEL_FILE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TAG = 'default'
model_path = coerce_to_path_and_check_exist(MODELS_PATH / TAG / MODEL_FILE)
model, (img_size, restricted_labels, normalize) = load_model_from_path(model_path, device=device, attributes_to_return=['train_resolution', 'restricted_labels', 'normalize'])
_ = model.eval()
#%% md
# ## 2. Load and pre-process an input image 
#%%
from PIL import Image
import numpy as np
from docextractorfork.utils.image import resize

img = Image.open('/home/jb/.projects/buchkind/data/corpus01/img/Buttstaedts Erzaehlungen fuer die Kinderwelt_1900_IJB.jpg')

# Resize 
img = resize(img, img_size)
print(f'image size is: {img.size}')

# Normalize and convert to Tensor
inp = np.array(img, dtype=np.float32) / 255
if normalize:
    inp = ((inp - inp.mean(axis=(0, 1))) / (inp.std(axis=(0, 1)) + 10**-7))
inp = torch.from_numpy(inp.transpose(2, 0, 1)).float().to(device)
#%% md
# ## 3. Predict segmentation maps and show results
#%%
from docextractorfork.utils.constant import LABEL_TO_COLOR_MAPPING
from docextractorfork.utils.image import LabeledArray2Image

# compute prediction
pred = model(inp.reshape(1, *inp.shape))[0].max(0)[1].cpu().numpy()

# Retrieve good color mapping and transform to image
restricted_colors = [LABEL_TO_COLOR_MAPPING[l] for l in restricted_labels]
label_idx_color_mapping = {restricted_labels.index(l) + 1: c for l, c in zip(restricted_labels, restricted_colors)}
pred_img = LabeledArray2Image.convert(pred, label_idx_color_mapping)

# Blend predictions with original image
mask = Image.fromarray((np.array(pred_img) == (0, 0, 0)).all(axis=-1).astype(np.uint8) * 127 + 128)
blend_img = Image.composite(img, pred_img, mask)
from matplotlib import pyplot as plt
plt.imshow(blend_img); plt.show()