import pytest


@pytest.mark.parametrize("distributed_executor_backend", ["mp"])
def test_ilama_lora_tp2(distributed_executor_backend, ilama_lora_files):
    pass
