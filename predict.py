import json
from torchvision import models

from functions import (get_predict_args, load_checkpoint, predict,
                       set_device)
from utility_functions import process_image

# get our command line arguments as 'arguments'
arguments = get_predict_args()

# separate arguments into variables
input_path = arguments.input_path
checkpoint_path = arguments.checkpoint
top_k = arguments.top_k
cat_file = arguments.cat_file
gpu = arguments.gpu
arch = arguments.arch

# set torch.device cuda/cpu
device = set_device(gpu)

# open the json file with the flower classes
with open(cat_file, 'r') as f:
    cat_to_name = json.load(f)

# get the specified model
model = getattr(models, arch)(pretrained=True)
    
# load our checkpoint
print('Loading checkpoint...\n')
model = load_checkpoint(model, device, checkpoint_path)
print('Checkpoint loaded!\n')

# process the image
print('Processing image...\n')
proc_image = process_image(input_path)

# run the prediction
print('Running prediction...\n')
probs, classes = predict(proc_image, model, top_k, device)

# get the flower names
flowers = [cat_to_name[i] for i in classes]

# print the top_k number of flowers and probabilities
print('\nThe top ' + str(top_k) + ' classes and probabilities are: ')
for i in range(top_k):
    idx = i + 1
    print(str(idx) + '. ' + str(flowers[i]) + ' || ' + str(probs[i]*100) + '%')

# print the top return
print('\nThe model is ' + str(probs[0]*100) + '% sure this is a ' + flowers[0] + '.')