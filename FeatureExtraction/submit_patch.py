import azureml
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config(path="config.json")
cluster_name = "gpu-cluster1"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print(f"{cluster_name} bulundu.")
except ComputeTargetException:
    raise ValueError(f"{cluster_name} bulunamadı. Azure portalden kontrol et.")

base_image = None
for n in ["AzureML-ACPT-pytorch-2.2-cuda12.1",
          "AzureML-ACPT-pytorch-2.3-cuda12.1",
          "AzureML-PyTorch-2.2-GPU"]:
    try:
        _cur = Environment.get(ws, name=n)
        base_image = _cur.docker.base_image
        print(f"Kurated GPU ortam bulundu: {n}")
        break
    except Exception:
        pass

if not base_image:
    base_image = "mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:latest"

env = Environment(name="uni2h-feature-extraction-env-gpu")
env.docker.base_image = base_image
env.python.user_managed_dependencies = False

cd = CondaDependencies()
# libopenslide + python binding
cd.add_channel("conda-forge")
cd.add_conda_package("openslide")
cd.add_conda_package("openslide-python")

# AML runtime ve diğer pip paketleri
for p in [
    "azureml-core",
    "azureml-defaults",
    "azure-storage-blob",
    "timm==0.9.*",
    "numpy<2",
    "pandas==2.0.*",
    "h5py==3.10.*",
    "huggingface_hub==0.20.*",
    "scikit-image==0.21.*",
    "opencv-python-headless==4.8.*",
    "tqdm",
]:
    cd.add_pip_package(p)

env.python.conda_dependencies = cd

src = ScriptRunConfig(
    source_directory=".",                    # yerel klasör, workspace storage'a upload edilir
    script="patch_selection_debug.py",       # debug log'lu sürüm
    compute_target=compute_target,
    environment=env,
)

# ENV vars
src.run_config.environment_variables = {
    "HF_TOKEN": "...",
    "HUGGINGFACE_HUB_TOKEN": "...",
    "DEBUG": "1",
    "INFER_BATCH": "512",
    "USE_AMP": "1",
}

experiment = Experiment(workspace=ws, name="patch-uni2h")
run = experiment.submit(src)
print("Job gönderildi. Run ID:", run.id)
run.wait_for_completion(show_output=True)
