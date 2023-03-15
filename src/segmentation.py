import onnxruntime

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def inference(raw_scan, modelpath='default.onnx'):
    """ Runs trained model on raw scan, yielding a segmented result.

    The raw scan provided is expected to be skull-stripped (no bone regions).
    The returned scan is segmented, with each voxel being one of 5 classes:

    Class   Description
    -----   -----------
      0     Background
      1     Something
      2     Something
      3     Something
      4     Something


    Parameters
    ----------

    raw_scan:
        Raw scan to be segmented. Expected to be skull stripped.

    modelpath:
        Path to model to be run (has to be in *.onnx format).

    Returns
    -------

    nibabel.Nifti1Image:
        Segmented version of raw scan.
    """

    ort_session = onnxruntime.InferenceSession(modelpath)

    batch_size = 200 # TODO: Verify this is correct
    x = None # TODO: Replace with proper input to model
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    return
