# Get pretrained weights
import torch

#checkpoint = torch.hub.load_state_dict_from_url(
#            url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
#            map_location='cpu',
#            check_hash=True)



checkpoint = torch.load('./detr-r50-panoptic-00ce5173.pth', map_location='cpu')

# Remove class weights
del checkpoint["model"]["detr.class_embed.weight"]
del checkpoint["model"]["detr.class_embed.bias"]

# Save
torch.save(checkpoint,
           'detr-r50_panoptic_no-class-head.pth')
