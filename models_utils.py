import torch
import models
import data_utils

def load_model(model_code, device, eegnet_params, lstm_hidden_dim, model_path=None, feat_code=0):
    if model_code == 0:
        model = models.EEGNetLSTM(eegnet_params, lstm_hidden_dim).to(device)
    elif model_code == 1:
        model = models.New_EEGNetLSTM(eegnet_params, lstm_hidden_dim).to(device)
    elif model_code == 2:
        model = models.New_EEGNetLSTM_AGGObsr(eegnet_params, lstm_hidden_dim).to(device)
    else:
        raise ValueError(f"Código de modelo {model_code} no soportado.")
    if feat_code == 2:
        model = models.New_LSTM_AGGObsr(eegnet_params, lstm_hidden_dim).to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Modelo cargado desde {model_path}")
    model.eval()
    return model


def get_dataloader(model_code): # to-do: move to data_utils...
    if model_code == 1:
        features_fun = data_utils.get_features_from_dict_bins
        dataloader_fun = data_utils.New_AggressiveBehaviorDatasetBin
    elif model_code == 2:
        features_fun = data_utils.get_features_from_dict_bins
        dataloader_fun = data_utils.New_AggressiveBehaviorDatasetBin_AGGObserved
    else:
        print('Not supported yet.')
        model_fun, features_fun, dataloader_fun = None, None, None
    return features_fun, dataloader_fun


def test_model(model, test_dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            if len(batch) == 3:  # Modelo con aggObs
                batch_features, batch_aggObs, batch_labels = batch
                batch_features = batch_features.to(device)
                batch_aggObs = batch_aggObs.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_features, batch_aggObs)
            else:  # Modelo sin aggObs
                batch_features, batch_labels = batch
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_features)

            probs = torch.sigmoid(logits).squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    return all_probs, all_labels


