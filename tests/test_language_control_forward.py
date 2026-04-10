import torch

from smart.language_control import LanguageControlHead, PromptScenarioDataset, TextPromptEncoder


def test_text_prompt_encoder_is_deterministic():
    encoder = TextPromptEncoder(hidden_dim=16, vocab_size=128, max_tokens=8)
    first = encoder(["Instruct <ego> to yield before turning right."])
    second = encoder(["Instruct <ego> to yield before turning right."])
    assert torch.equal(first.token_ids, second.token_ids)
    assert first.embedding.shape == (1, 16)


def test_language_control_head_forward_shapes():
    torch.manual_seed(7)
    head = LanguageControlHead(hidden_dim=16, token_size=32, vocab_size=256, max_prompt_tokens=12)
    prompts = [
        "Instruct <ego> to decelerate before turning right.",
        "Have <a1> cut in from the right lane.",
    ]
    scene_latent = torch.randn(2, 16)
    agent_tokens = torch.randn(3, 4, 16)
    agent_batch = torch.tensor([0, 1, 1], dtype=torch.long)

    output = head(
        prompts=prompts,
        scene_latent=scene_latent,
        agent_tokens=agent_tokens,
        agent_batch=agent_batch,
    )

    assert output["token_logits"].shape == (3, 4, 32)
    assert output["policy_query"].shape == (3, 16)
    assert output["conditioned_agent_tokens"].shape == (3, 4, 16)
    assert output["scene_text_loss"].ndim == 0


def test_prompt_scenario_dataset_reads_jsonl(tmp_path):
    annotation = tmp_path / "prompts.jsonl"
    annotation.write_text(
        '{"sid": "scene_001", "prompts": ["Make <ego> yield.", "Have <a1> go straight."]}\n',
        encoding="utf-8",
    )
    dataset = PromptScenarioDataset([str(annotation)])
    assert len(dataset) == 2
    item = dataset[0]
    assert item["scenario_id"] == "scene_001"
    assert "prompt" in item
