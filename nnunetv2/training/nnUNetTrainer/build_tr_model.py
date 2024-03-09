import sys
from .model.WaveU3S import WaveU3S

def build_tr_model(model_name, in_ch, out_ch):
    print(f"Received model name: {model_name}")  # Debug print
    if model_name == 'WaveU3S':
        print('Loading model WaveU3S!')
        return WaveU3S(in_channels=1, out_channels=14, img_size=(96, 192, 192), feature_size=16, num_heads=4, norm_name='batch', depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True).cuda()
    else:
        print(f"Model name '{model_name}' not found!")  # Debug print
    
        raise RuntimeError('Given model name not implemented!')
        
        