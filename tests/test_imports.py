from lam_pinn.config import load_adapt_config, load_train_config
from lam_pinn.models.serial_net import SerialNetwork


def test_imports_and_model_instantiation():
    model = SerialNetwork(num_clusters=3, hidden_dim=64, dropout_p=0.2, gate_init=0.5)
    assert model.num_clusters == 3
    assert len(model.gates) == 3
    assert callable(load_train_config)
    assert callable(load_adapt_config)
