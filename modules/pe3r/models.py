import os
import sys
sys.path.append(os.path.abspath('./modules/ultralytics'))

from transformers import AutoTokenizer, AutoModel, AutoProcessor, SamModel
from modules.mast3r.model import AsymmetricMASt3R

# from modules.sam2.build_sam import build_sam2_video_predictor
from modules.mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from modules.mobilesamv2 import sam_model_registry

from sam2.sam2_video_predictor import SAM2VideoPredictor

class Models:
    def __init__(self, device):
        # -- mast3r --
        # MAST3R_CKP = './checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
        MAST3R_CKP = 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
        self.mast3r = AsymmetricMASt3R.from_pretrained(MAST3R_CKP).to(device)

        # -- sam2 --
        # SAM2_CKP = "./checkpoints/sam2.1_hiera_large.pt"
        # SAM2_CKP = 'hujiecpp/sam2-1-hiera-large'
        # SAM2_CONFIG = "./configs/sam2.1/sam2.1_hiera_l.yaml"
        # self.sam2 = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CKP, device=device, apply_postprocessing=False)
        # self.sam2.eval()
        self.sam2 = SAM2VideoPredictor.from_pretrained('facebook/sam2.1-hiera-large', device=device)

        # -- mobilesamv2 & sam1 --
        # SAM1_ENCODER_CKP = './checkpoints/sam_vit_h.pt'
        # SAM1_ENCODER_CKP = 'facebook/sam-vit-huge/model.safetensors'
        SAM1_DECODER_CKP = './checkpoints/Prompt_guided_Mask_Decoder.pt'
        self.mobilesamv2 = sam_model_registry['sam_vit_h'](None)
        # image_encoder=sam_model_registry['sam_vit_h_encoder'](SAM1_ENCODER_CKP)
        sam1 = SamModel.from_pretrained('facebook/sam-vit-huge')
        image_encoder = sam1.vision_encoder

        prompt_encoder, mask_decoder = sam_model_registry['prompt_guided_decoder'](SAM1_DECODER_CKP)
        self.mobilesamv2.prompt_encoder = prompt_encoder
        self.mobilesamv2.mask_decoder = mask_decoder
        self.mobilesamv2.image_encoder=image_encoder
        self.mobilesamv2.to(device=device)
        self.mobilesamv2.eval()

        # -- yolov8 --
        YOLO8_CKP='./checkpoints/ObjectAwareModel.pt'
        self.yolov8 = ObjectAwareModel(YOLO8_CKP)

        # -- siglip --
        self.siglip = AutoModel.from_pretrained("google/siglip-large-patch16-256", device_map=device)
        self.siglip_tokenizer = AutoTokenizer.from_pretrained("google/siglip-large-patch16-256")
        self.siglip_processor = AutoProcessor.from_pretrained("google/siglip-large-patch16-256")