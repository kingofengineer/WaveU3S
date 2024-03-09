from model.Unetr_plus_plus import UNETR_PP

def build_model(model_name, in_ch, out_ch):
    if model_name == 'UNETR_PP':
        print('Loading model UNETR_PP!')
        return UNETR_PP(in_channels=1, out_channels=14, img_size=(96, 192, 192), feature_size=16, num_heads=4, norm_name='batch', depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True).cuda()
    else:
        raise RuntimeError('Given model name not implemented!')