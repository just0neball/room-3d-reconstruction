"""Maps the fine-grained stage names emitted by Person A's scripts onto the four
user-facing stages the progress bar shows (extracting frames, estimating poses,
training, exporting)."""

# Names come from send_progress(...) calls in pipeline/*.py
STAGE_GROUPS = {
    "queued": "queued",
    "starting": "starting",
    "extracting_frames": "extract",
    "filtering_frames": "extract",
    "estimating_poses": "poses",
    "estimating_poses_vggt": "poses",
    "preparing_training_data": "poses",  # COLMAP undistortion + dataset prep
    "training_gaussians": "train",
    "exporting": "export",
    "packaging": "export",
    "uploading_results": "export",
    "done": "done",
}

GROUP_LABELS = {
    "queued": "Queued",
    "starting": "Starting worker",
    "extract": "Extracting frames",
    "poses": "Estimating camera poses",
    "train": "Training Gaussian splat",
    "export": "Exporting scene",
    "done": "Done",
}

# Order for a stepper UI
PIPELINE_GROUPS = ["extract", "poses", "train", "export"]


def group_for(stage: str) -> str:
    return STAGE_GROUPS.get(stage, stage)


def label_for(stage: str) -> str:
    group = group_for(stage)
    return GROUP_LABELS.get(group, stage.replace("_", " ").capitalize())
