import numpy as np

def export_as_torchscript(model, model_path):
  example = np.random.randint(0, 255, (1, 2, 256, 256))
  example = example.to(model.device)
  traced_model = torch.jit.trace(model.net, torch.from_numpy(example))
  traced_model.save(model_path)
