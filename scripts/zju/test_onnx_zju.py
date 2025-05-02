import numpy as np
import onnx
from onnxruntime import InferenceSession

if __name__ == '__main__':
    # func = np.random.random
    func = np.ones
    proprio = func((1, 50)).astype(np.float32)
    prop_his = func((1, 10, 50)).astype(np.float32)
    depth_his = func((1, 2, 58, 87)).astype(np.float32)
    hidden_obs_gru = func((1, 1, 64)).astype(np.float32)
    hidden_recon = func((2, 1, 256)).astype(np.float32)

    model = onnx.load('onnx/policy.onnx')
    onnx.checker.check_model(model)

    session = InferenceSession(model.SerializeToString(), providers=[
        'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

    outputs = session.run(
        input_feed={'proprio': proprio,
                    'prop_his': prop_his,
                    'depth_his': depth_his,
                    'hidden_obs_gru_in': hidden_obs_gru,
                    'hidden_recon_in': hidden_recon,
                    },
        output_names=['action', 'recon_rough', 'recon_refine', 'hidden_obs_gru_out', 'hidden_recon_out', 'est_mu']
    )

    proprio.tofile('trt/proprio.bin')
    prop_his.tofile('trt/prop_his.bin')
    depth_his.tofile('trt/depth_his.bin')
    hidden_obs_gru.tofile('trt/hidden_obs_gru.bin')
    hidden_recon.tofile('trt/hidden_recon.bin')

    for n, o in zip(['action', 'recon_rough', 'recon_refine', 'hidden_obs_gru_out', 'hidden_recon_out', 'est_mu'], outputs):
        print(n, o)
