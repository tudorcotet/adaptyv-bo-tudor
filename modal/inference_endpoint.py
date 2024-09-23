"""An example on how to prepare a modal endpoint with https://modal.com/docs/reference/modal.Image#pip_install_from_pyproject.

- For larger models https://modal.com/docs/guide/lifecycle-functions#enter
"""

import modal

# following the flash atn example https://modal.com/docs/guide/cuda#for-more-complex-setups-use-an-officially-supported-cuda-image
cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .pip_install_from_pyproject("../pyproject.toml")
    .run_commands("mkdir -p pkg/src")
    .workdir("pkg")
    .copy_local_dir("../src/", "./src/")
    .copy_local_file("../pyproject.toml", ".")
    .run_commands("pip install -e .")
)
app = modal.App()

if not modal.is_local():
    from my_package.example import DatasetStats, DatasetWrapper


# set to 2 seconds by default
@app.cls(cpu=1, image=image, container_idle_timeout=2)
class Model:
    """Dummy model, showing how to create an endpoit which will perform a load once at container startup."""

    @modal.enter()
    def run_this_on_container_startup(self):
        """
        Load the model once at container startup.

        In a real-world scenario, this would involve loading a large model from disk.
        We we will simply always answer with the expected value of the dataset, so this is our "model".
        """
        self.y_mu = DatasetWrapper.mk_random({}, 100).compute_stats((0,))[1].mu

    @modal.method()
    def predict(self, x):
        """Always answer with the expected value of the dataset."""
        _ = x
        return self.y_mu


@app.local_entrypoint()
def main():
    """Call the app with a random input to demonstrate the endpoint."""
    import torch as pt

    ret = Model().predict.remote(pt.randn((5,)))
    print(f"{ret=}")
