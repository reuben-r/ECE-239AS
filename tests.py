from pathlib import Path

import torch
import numpy as np


def check_singleheadattention(model, checkpoint_file, device="cpu"):

    model.to(device)
    model.eval()
    data_file = "./test_cases.npz"
    key = "singleheadattention"

    ckpt = torch.load(checkpoint_file, map_location=device)

    with torch.no_grad():

        model.key.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.attention.head_0.key.weight"]
        )
        model.query.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_0.query.weight"
            ]
        )
        model.value.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_0.value.weight"
            ]
        )
        model.causal_mask_uniquename.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_0.causal_mask"
            ]
        )

    data = np.load(data_file)
    old_input = torch.from_numpy(data[key + "_input"])
    old_output = torch.from_numpy(data[key + "_output"])
    old_input = old_input.to(device)
    old_output = old_output.to(device)
    check_output = model(old_input)
    assert torch.allclose(check_output, old_output, atol=1e-5), "TEST CASE FAILED"

    return "TEST CASE PASSED!!!"


def check_multiheadattention(model, checkpoint_file, device="cpu"):

    model.to(device)
    model.eval()
    data_file = "./test_cases.npz"
    key = "multiheadattention"

    ckpt = torch.load(checkpoint_file, map_location=device)

    with torch.no_grad():
        model.head_0.key.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.attention.head_0.key.weight"]
        )
        model.head_0.query.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_0.query.weight"
            ]
        )
        model.head_0.value.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_0.value.weight"
            ]
        )
        model.head_0.causal_mask_uniquename.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_0.causal_mask"
            ]
        )

        model.head_1.key.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.attention.head_1.key.weight"]
        )
        model.head_1.query.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_1.query.weight"
            ]
        )
        model.head_1.value.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_1.value.weight"
            ]
        )
        model.head_1.causal_mask_uniquename.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_1.causal_mask"
            ]
        )

        model.head_2.key.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.attention.head_2.key.weight"]
        )
        model.head_2.query.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_2.query.weight"
            ]
        )
        model.head_2.value.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_2.value.weight"
            ]
        )
        model.head_2.causal_mask_uniquename.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_2.causal_mask"
            ]
        )

        model.head_3.key.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.attention.head_3.key.weight"]
        )
        model.head_3.query.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_3.query.weight"
            ]
        )
        model.head_3.value.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_3.value.weight"
            ]
        )
        model.head_3.causal_mask_uniquename.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_3.causal_mask"
            ]
        )

        model.out.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.attention.out.weight"]
        )

        model.out.bias.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.attention.out.bias"]
        )

    data = np.load(data_file)
    old_input = torch.from_numpy(data[key + "_input"])
    old_output = torch.from_numpy(data[key + "_output"])
    old_input = old_input.to(device)
    old_output = old_output.to(device)
    check_output = model(old_input)
    assert torch.allclose(check_output, old_output, atol=1e-5), "TEST CASE FAILED"

    return "TEST CASE PASSED!!!"


def check_feedforward(model, checkpoint_file, device="cpu"):

    model.to(device)
    model.eval()
    data_file = "./test_cases.npz"
    key = "feedforward"

    ckpt = torch.load(checkpoint_file, map_location=device)

    with torch.no_grad():

        model.fc1.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.feedforward.fc1.weight"]
        )
        model.fc2.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.feedforward.fc2.weight"]
        )
        model.fc1.bias.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.feedforward.fc1.bias"]
        )
        model.fc2.bias.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.feedforward.fc2.bias"]
        )

    data = np.load(data_file)
    old_input = torch.from_numpy(data[key + "_input"])
    old_output = torch.from_numpy(data[key + "_output"])
    old_input = old_input.to(device)
    old_output = old_output.to(device)
    check_output = model(old_input)

    assert torch.allclose(check_output, old_output, atol=1e-5), "TEST CASE FAILED"

    return "TEST CASE PASSED!!!"


def check_layernorm(model, checkpoint_file, device="cpu"):

    model.to(device)
    model.eval()
    data_file = "./test_cases.npz"
    key = "layernorm"

    ckpt = torch.load(checkpoint_file, map_location=device)

    with torch.no_grad():

        model.gamma.copy_(ckpt["model_state_dict"]["transformer_layers.0.norm1.gamma"])

        model.beta.copy_(ckpt["model_state_dict"]["transformer_layers.0.norm1.gamma"])

    data = np.load(data_file)
    old_input = torch.from_numpy(data[key + "_input"])
    old_output = torch.from_numpy(data[key + "_output"])
    old_input = old_input.to(device)
    old_output = old_output.to(device)
    check_output = model(old_input)
    assert torch.allclose(check_output, old_output, atol=1e-5), "TEST CASE FAILED"

    return "TEST CASE PASSED!!!"


def check_miniGPT(model, checkpoint_file, device="cpu"):

    model.to(device)
    model.eval()
    data_file = "./test_cases.npz"
    key = "minigpt"

    ckpt = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    data = np.load(data_file)
    old_input = torch.from_numpy(data[key + "_input"])
    old_output = torch.from_numpy(data[key + "_output"])
    old_input = old_input.to(device)
    old_output = old_output.to(device)
    check_output = model(old_input)
    assert torch.allclose(check_output, old_output, atol=1e-5), "TEST CASE FAILED"

    return "TEST CASE PASSED!!!"


def check_transformer(model, checkpoint_file, device="cpu"):

    model.to(device)
    model.eval()
    data_file = "./test_cases.npz"
    key = "transformer"

    ckpt = torch.load(checkpoint_file, map_location="cpu")

    with torch.no_grad():
        model.norm1.gamma.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.norm1.gamma"]
        )
        model.norm1.beta.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.norm1.gamma"]
        )
        model.norm2.gamma.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.norm2.gamma"]
        )
        model.norm2.beta.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.norm2.gamma"]
        )

        model.feedforward.fc1.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.feedforward.fc1.weight"]
        )
        model.feedforward.fc2.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.feedforward.fc2.weight"]
        )
        model.feedforward.fc1.bias.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.feedforward.fc1.bias"]
        )
        model.feedforward.fc2.bias.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.feedforward.fc2.bias"]
        )

        model.attention.head_0.key.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.attention.head_0.key.weight"]
        )
        model.attention.head_0.query.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_0.query.weight"
            ]
        )
        model.attention.head_0.value.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_0.value.weight"
            ]
        )
        model.attention.head_0.causal_mask_uniquename.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_0.causal_mask"
            ]
        )

        model.attention.head_1.key.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.attention.head_1.key.weight"]
        )
        model.attention.head_1.query.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_1.query.weight"
            ]
        )
        model.attention.head_1.value.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_1.value.weight"
            ]
        )
        model.attention.head_1.causal_mask_uniquename.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_1.causal_mask"
            ]
        )

        model.attention.head_2.key.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.attention.head_2.key.weight"]
        )
        model.attention.head_2.query.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_2.query.weight"
            ]
        )
        model.attention.head_2.value.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_2.value.weight"
            ]
        )
        model.attention.head_2.causal_mask_uniquename.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_2.causal_mask"
            ]
        )

        model.attention.head_3.key.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.attention.head_3.key.weight"]
        )
        model.attention.head_3.query.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_3.query.weight"
            ]
        )
        model.attention.head_3.value.weight.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_3.value.weight"
            ]
        )
        model.attention.head_3.causal_mask_uniquename.copy_(
            ckpt["model_state_dict"][
                "transformer_layers.0.attention.head_3.causal_mask"
            ]
        )

        model.attention.out.weight.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.attention.out.weight"]
        )

        model.attention.out.bias.copy_(
            ckpt["model_state_dict"]["transformer_layers.0.attention.out.bias"]
        )

    data = np.load(data_file)
    old_input = torch.from_numpy(data[key + "_input"])
    old_output = torch.from_numpy(data[key + "_output"])
    old_input = old_input.to(device)
    old_output = old_output.to(device)
    check_output = model(old_input)
    assert torch.allclose(check_output, old_output, atol=1e-5), "TEST CASE FAILED"

    return "TEST CASE PASSED!!!"


def check_bigram(model, checkpoint_file, device="cpu"):

    model.to(device)
    model.eval()
    data_file = "./test_cases.npz"
    # data_file = "./bigram_test_cases.npz"
    key = "bigram"

    ckpt = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    data = np.load(data_file)
    old_input = torch.from_numpy(data[key + "_input"])
    old_output = torch.from_numpy(data[key + "_output"])
    old_input = old_input.to(device)
    old_output = old_output.to(device)
    check_output = model(old_input)
    assert torch.allclose(check_output, old_output, atol=1e-5), "TEST CASE FAILED"

    return "TEST CASE PASSED!!!"
