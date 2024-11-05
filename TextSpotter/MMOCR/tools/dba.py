from mmocr.apis.inferencers import MMOCRInferencer

def infer(inputs, det="DBNet", det_weights=None, rec=None, rec_weights=None, kie=None, kie_weights=None, device=None, out_dir='results/', batch_size=1, show=False, print_result=False, save_pred=False, save_vis=False):
    # Set up the initial arguments
    init_args = {
        'det': det,
        'det_weights': det_weights,
        'rec': rec,
        'rec_weights': rec_weights,
        'kie': kie,
        'kie_weights': kie_weights,
        'device': device
    }

    # Set up the call arguments
    call_args = {
        'inputs': inputs,
        'out_dir': out_dir,
        'batch_size': batch_size,
        'show': show,
        'print_result': print_result,
        'save_pred': save_pred,
        'save_vis': save_vis
    }

    # Create an instance of MMOCRInferencer and call it
    ocr = MMOCRInferencer(**init_args)
    ocr(**call_args)
