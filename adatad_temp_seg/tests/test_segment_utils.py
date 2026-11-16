import numpy as np
import pytest
from segment_utils import (
    SegmentationType,
    actionable_predictions,
    adjust_segments_to_scene_changes,
    build_predictions,
    cluster_representatives,
    flatten_captions,
    frame_range,
    generate_intervals,
    normalize_arguments,
    pair_text,
    parse_arguments,
    parse_json_object,
    prune_arguments,
    select_proposals,
    subject_object_pairs,
)


def proposal(start, end, score):
    return {"segment": [start, end], "score": score}


def segments(*bounds):
    return [{"segment": [start, end], "score": 0.0} for start, end in bounds]


def test_select_proposals_keeps_best_non_overlapping_above_threshold():
    proposals = [
        {**proposal(0, 10, 0.9), "label": "cut"},
        proposal(5, 15, 0.95),
        proposal(15, 20, 0.5),
        proposal(30, 40, 0.001),
    ]
    assert select_proposals(proposals) == [proposal(5, 15, 0.95), proposal(15, 20, 0.5)]


def test_select_proposals_returns_at_most_twenty_best():
    proposals = [proposal(i, i + 1, i / 100) for i in range(1, 31)]
    selected = select_proposals(proposals)
    assert [p["score"] for p in selected] == [i / 100 for i in range(30, 10, -1)]


def test_scene_changes_split_and_proposals_refine():
    assert adjust_segments_to_scene_changes([proposal(3, 25, 0.9)], [10.0], 30.0) == segments(
        (0.0, 3), (3, 10.0), (10.0, 25), (25, 30.0)
    )


def test_proposal_boundaries_too_close_to_a_cut_are_ignored():
    result = adjust_segments_to_scene_changes([proposal(9.5, 29, 0.9)], [10.0], 30.0)
    assert result == segments((0.0, 10.0), (10.0, 30.0))


def test_without_cuts_or_proposals_the_whole_video_is_one_segment():
    assert adjust_segments_to_scene_changes([], [], 12.5, min_duration=1) == segments((0.0, 12.5))


def test_generate_intervals():
    assert generate_intervals(12) == [
        {"segment": [0.0, 5.0], "score": 1},
        {"segment": [5.0, 10.0], "score": 1},
        {"segment": [10.0, 12], "score": 1},
    ]
    assert generate_intervals(0) == []


@pytest.mark.parametrize("fps", [23.976, 25, 29.97, 30, 50, 59.94, 60])
def test_frame_range_maps_rounded_cut_times_back_to_their_frames(fps):
    for frame in range(5000):
        cut = round(frame / fps, 2)
        assert frame_range([cut, cut + 10], fps, 10**6)[0] == frame


def test_one_frame_segments_between_close_cuts_are_kept():
    assert frame_range([3.34, 3.37], 29.97, 450) == (100, 101)
    assert frame_range([1.0, 1.03], 30, 450) == (30, 31)


def test_frame_range_is_clamped_to_the_video():
    assert frame_range([-1, 100], 30, 50) == (0, 50)


def test_flatten_captions():
    assert flatten_captions(["man", "dog"], [["a", "b"], "c"]) == ["a", "b", "c"]
    assert flatten_captions(["man"], [[["x", "y"], "z"]]) == ["x", "y", "z"]
    assert flatten_captions(["man"], [["a"], ["b"]]) == ["a"]
    assert flatten_captions([], []) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"argument0": "man"}', {"argument0": "man"}),
        ('```json\n{"argument0": "man"}\n```', {"argument0": "man"}),
        ('Sure! {"argument1": "door"} Hope this helps.', {"argument1": "door"}),
        ('{"argument0": "a"} {"argument0": "b"}', {"argument0": "a"}),
        ('use {braces} then {"argument0": "a"}', {"argument0": "a"}),
        ('[{"argument0": "a"}]', {"argument0": "a"}),
        ("no JSON here", None),
    ],
)
def test_parse_json_object(text, expected):
    assert parse_json_object(text) == expected


def test_normalize_arguments_coerces_llm_values():
    raw = {"argument0": " man ", "argument1": None, "argument2": ["onion", None, "board"]}
    assert normalize_arguments(raw) == {
        "argument0": "man",
        "argument1": "none",
        "argument2": "onion, board",
    }
    assert normalize_arguments({"argument0": 3, "argument1": "None", "extra": "x"}) == {
        "argument0": "3",
        "argument1": "none",
        "argument2": "none",
    }
    for empty in ["", "null", "N/A", []]:
        assert normalize_arguments({"argument0": empty})["argument0"] == "none"


def test_parse_arguments_without_json_is_all_none():
    assert parse_arguments("I cannot tell.") == dict.fromkeys(
        ["argument0", "argument1", "argument2"], "none"
    )


def test_prune_arguments():
    assert prune_arguments({"argument0": "man", "argument1": "none", "argument2": "knife"}) == {
        "argument0": "man",
        "argument1": "knife",
    }
    assert prune_arguments({"argument0": "none", "argument1": "door", "argument2": "none"}) == {
        "argument1": "door"
    }


def test_an_omitted_argument_counts_as_none():
    # The prompt asks for argument1 even when it is absent; leaving it out means the same.
    reply = '{"argument0": "man", "argument2": "knife"}'
    assert prune_arguments(parse_arguments(reply)) == {"argument0": "man", "argument1": "knife"}


def test_actionable_predictions_skip_none_actions():
    arguments = [
        {"argument0": "man", "argument1": "onion", "argument2": "none"},
        {"argument0": "none", "argument1": "none", "argument2": "none"},
        {"argument0": "woman", "argument1": "none", "argument2": "cup"},
    ]
    assert actionable_predictions(["cut", "None", "pour"], arguments) == [
        {
            "original_idx": 0,
            "action": "cut",
            "arguments": {"argument0": "man", "argument1": "onion"},
        },
        {
            "original_idx": 2,
            "action": "pour",
            "arguments": {"argument0": "woman", "argument1": "cup"},
        },
    ]


def test_pair_text_puts_the_subject_first():
    assert pair_text("cut", {"argument1": "onion", "argument0": "man"}) == "man cut onion"
    assert pair_text("open", {"argument1": "door"}) == "open door"


def test_clusters_become_predictions_with_their_share():
    labels = np.array([0, 0, 1, 0, -1])
    embeddings = np.array([[0, 0], [1, 1], [9, 9], [3, 3], [5, 5]], dtype=float)
    clusters = cluster_representatives(labels, embeddings)
    assert clusters == {0: ([0, 1, 3], 1), 1: ([2], 2)}

    predictions = [
        {"original_idx": i, "action": action, "arguments": arguments}
        for i, (action, arguments) in enumerate(
            [
                ("cut", {"argument0": "man", "argument1": "onion"}),
                ("cut", {"argument0": "chef", "argument1": "onion"}),
                ("walk", {"argument0": "dog"}),
                ("cut", {"argument0": "man", "argument1": "leek"}),
                ("open", {}),
            ]
        )
    ]
    captions = ["c0", "c1", "c2", "c3", "c4"]
    result = build_predictions(clusters, predictions, captions)
    assert list(result) == ["valid_parsed_actions_2", "valid_parsed_actions_3"]
    assert result["valid_parsed_actions_2"] == [
        {"pattern": {"action": "walk", "argument0": "dog"}, "probability": 0.25, "caption": "c2"}
    ]
    assert result["valid_parsed_actions_3"] == [
        {
            "pattern": {"action": "cut", "argument0": "chef", "argument1": "onion"},
            "probability": 0.75,
            "caption": "c1",
        }
    ]


def test_subject_object_pairs_follow_prediction_order():
    predictions = {
        "valid_parsed_actions_2": [
            {"pattern": {"action": "walk", "argument0": "dog"}, "probability": 0.5, "caption": None}
        ],
        "valid_parsed_actions_3": [
            {
                "pattern": {"action": "cut", "argument0": "man", "argument1": "onion"},
                "probability": 0.5,
                "caption": "a man cuts an onion",
            }
        ],
    }
    assert subject_object_pairs(predictions) == [
        {"argument_names": ["dog"], "action": "walk", "caption": ""},
        {"argument_names": ["man", "onion"], "action": "cut", "caption": "a man cuts an onion"},
    ]
    assert subject_object_pairs({}) == []


def test_segmentation_types():
    assert [t.value for t in SegmentationType] == [0, 1, 2]
    with pytest.raises(ValueError):
        SegmentationType(3)
