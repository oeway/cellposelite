import onnxruntime as ort
import numpy as np
import torch

def export_onnx(net, file_path):
    # Note: the original cellpose model won't work directly
    # because it uses `F.avg_pool2d` to do global pooling
    # To fix that, you can convert the kernel size to fix number
    # See here: https://github.com/pytorch/pytorch/issues/34780#issuecomment-876969861
    X = torch.rand(1, 2, 224, 224)
    y, style = net(X)
    torch.onnx.export(
        net,
        X,
        file_path,
        input_names=["image"],
        output_names=["flow", "style"],
        dynamic_axes={
            "image": [0, 2, 3]
        },  # the batch, width and height of the image can be changed
        verbose=False,
        opset_version=11,
    )
    ort_session = ort.InferenceSession(file_path)
    exported_results = ort_session.run(None, {"image": X.numpy().astype(np.float32)})
    assert np.allclose(y.detach().numpy(), exported_results[0])
    print(f"ONNX File {file_path} exported successfully")
